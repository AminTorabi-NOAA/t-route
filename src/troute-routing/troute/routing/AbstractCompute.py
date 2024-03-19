from collections import defaultdict
from itertools import chain
from functools import partial
from tlz import concat
from joblib import delayed, Parallel
from datetime import datetime, timedelta
import time
import pandas as pd
import numpy as np
import copy
import multiprocessing
import troute.nhd_network as nhd_network
from troute.routing.fast_reach.mc_reach import compute_network_structured
import troute.routing.diffusive_utils_v02 as diff_utils
from troute.routing.fast_reach import diffusive

from abc import ABC, abstractmethod

# Here with a recursive solution, deepest layer of dictionary (which will be subnet_sets) from the
# networks_with_sibnetworks_ordered_jit can be extracted and put it in the self._ordered_subn_dict
def find_deepest_dicts(network):
    deepest_dicts = {}
    max_depth = -1

    def recurse(current_dict, depth):
        nonlocal deepest_dicts, max_depth
        if isinstance(current_dict, dict):
            if not any(isinstance(v, dict) for v in current_dict.values()):
                if depth > max_depth:
                    deepest_dicts = current_dict.copy()
                    max_depth = depth
                elif depth == max_depth:
                    deepest_dicts.update(current_dict) 
            else:
                for v in current_dict.values():
                    if isinstance(v, dict):
                        recurse(v, depth + 1)

    recurse(network, 0)
    return deepest_dicts
    
def _build_reach_type_list(reach_list, wbodies_segs):

    reach_type_list = [1 if (set(reaches) & wbodies_segs) else 0 for reaches in reach_list]

    return list(zip(reach_list, reach_type_list))

def _prep_da_dataframes(usgs_df,lastobs_df,param_df_sub_idx,exclude_segments=None):
    """
    Produce, based on the segments in the param_df_sub_idx (which is a subset
    representing a subnetwork of the larger collection of all segments),
    a subset of the relevant usgs gage observation time series
    and the relevant last-valid gage observation from any
    prior model execution.
    
    exclude_segments (list): segments to exclude from param_df_sub when searching for gages
                            This catches and excludes offnetwork upstreams segments from being
                            realized as locations for DA substitution. Else, by-subnetwork
                            parallel executions fail.

    Cases to consider:
    USGS_DF, LAST_OBS
    Yes, Yes: Analysis and Assimilation; Last_Obs used to fill gaps in the front of the time series
    No, Yes: Forecasting mode;
    Yes, No; Cold-start case;
    No, No: Open-Loop;

    For both cases where USGS_DF is present, there is a sub-case where the length of the observed
    time series is as long as the simulation.

    """
    
    subnet_segs = param_df_sub_idx
    # segments in the subnetwork ONLY, no offnetwork upstreams included
    if exclude_segments:
        subnet_segs = param_df_sub_idx.difference(set(exclude_segments))
    
    # NOTE: Uncomment to easily test no observations...
    # usgs_df = pd.DataFrame()
    if not usgs_df.empty and not lastobs_df.empty:
        # index values for last obs are not correct, but line up correctly with usgs values. Switched
        lastobs_segs = (lastobs_df.index.
                        intersection(subnet_segs).
                        to_list()
                    )
        lastobs_df_sub = lastobs_df.loc[lastobs_segs]
        usgs_segs = (usgs_df.index.
                    intersection(subnet_segs).
                    reindex(lastobs_segs)[0].
                    to_list()
                    )
        da_positions_list_byseg = param_df_sub_idx.get_indexer(usgs_segs)
        usgs_df_sub = usgs_df.loc[usgs_segs]
    elif usgs_df.empty and not lastobs_df.empty:
        lastobs_segs = (lastobs_df.index.
                        intersection(subnet_segs).
                        to_list()
                    )
        lastobs_df_sub = lastobs_df.loc[lastobs_segs]
        # Create a completely empty list of gages -- the .shape[1] attribute
        # will be == 0, and that will trigger a reference to the lastobs.
        # in the compute kernel below.
        usgs_df_sub = pd.DataFrame(index=lastobs_df_sub.index,columns=[])
        usgs_segs = lastobs_segs
        da_positions_list_byseg = param_df_sub_idx.get_indexer(lastobs_segs)
    elif not usgs_df.empty and lastobs_df.empty:
        usgs_segs = list(usgs_df.index.intersection(subnet_segs))
        da_positions_list_byseg = param_df_sub_idx.get_indexer(usgs_segs)
        usgs_df_sub = usgs_df.loc[usgs_segs]
        lastobs_df_sub = pd.DataFrame(index=usgs_df_sub.index,columns=["discharge","time","model_discharge"])
    else:
        usgs_df_sub = pd.DataFrame()
        lastobs_df_sub = pd.DataFrame()
        da_positions_list_byseg = []

    return usgs_df_sub, lastobs_df_sub, da_positions_list_byseg

def _prep_da_positions_byreach(reach_list, gage_index):
    """
    produce a list of indexes of the reach_list identifying reaches with gages
    and a corresponding list of indexes of the gage_list of the gages in
    the order they are found in the reach_list.
    """
    reach_key = []
    reach_gage = []
    for i, r in enumerate(reach_list):
        for s in r:
            if s in gage_index:
                reach_key.append(i)
                reach_gage.append(s)
    gage_reach_i = gage_index.get_indexer(reach_gage)

    return reach_key, gage_reach_i

def _prep_reservoir_da_dataframes( reservoir_usgs_df,
                                reservoir_usgs_param_df,
                                reservoir_usace_df,
                                reservoir_usace_param_df,
                                reservoir_rfc_df,
                                reservoir_rfc_param_df,
                                waterbody_types_df_sub,
                                t0, 
                                from_files,
                                exclude_segments=None):
    '''
    Helper function to build reservoir DA data arrays for routing computations

    Arguments
    ---------
    reservoir_usgs_df        (DataFrame): gage flow observations at USGS-type reservoirs
    reservoir_usgs_param_df  (DataFrame): USGS reservoir DA state parameters
    reservoir_usace_df       (DataFrame): gage flow observations at USACE-type reservoirs
    reservoir_usace_param_df (DataFrame): USACE reservoir DA state parameters
    reservoir_rfc_df         (DataFrame): gage flow observations and forecasts at RFC-type reservoirs
    reservoir_rfc_param_df   (DataFrame): RFC reservoir DA state parameters
    waterbody_types_df_sub   (DataFrame): type-codes for waterbodies in sub domain
    t0                        (datetime): model initialization time

    Returns
    -------
    * there are many returns, because we are passing explicit arrays to mc_reach cython code
    reservoir_usgs_df_sub                 (DataFrame): gage flow observations for USGS-type reservoirs in sub domain
    reservoir_usgs_df_time                  (ndarray): time in seconds from model initialization time
    reservoir_usgs_update_time              (ndarray): update time (sec) to search for new observation at USGS reservoirs
    reservoir_usgs_prev_persisted_flow      (ndarray): previously persisted outflow rates at USGS reservoirs
    reservoir_usgs_persistence_update_time  (ndarray): update time (sec) of persisted value at USGS reservoirs
    reservoir_usgs_persistence_index        (ndarray): index denoting elapsed persistence epochs at USGS reservoirs
    reservoir_usace_df_sub                (DataFrame): gage flow observations for USACE-type reservoirs in sub domain
    reservoir_usace_df_time                 (ndarray): time in seconds from model initialization time
    reservoir_usace_update_time             (ndarray): update time (sec) to search for new observation at USACE reservoirs
    reservoir_usace_prev_persisted_flow     (ndarray): previously persisted outflow rates at USACE reservoirs
    reservoir_usace_persistence_update_time (ndarray): update time (sec) of persisted value at USACE reservoirs
    reservoir_usace_persistence_index       (ndarray): index denoting elapsed persistence epochs at USACE reservoirs

    '''
    if not reservoir_usgs_df.empty:
        usgs_wbodies_sub      = waterbody_types_df_sub[
                                    waterbody_types_df_sub['reservoir_type']==2
                                ].index
        if exclude_segments:
            usgs_wbodies_sub = list(set(usgs_wbodies_sub).difference(set(exclude_segments)))
        reservoir_usgs_df_sub = reservoir_usgs_df.loc[usgs_wbodies_sub]
        reservoir_usgs_df_time = []
        for timestamp in reservoir_usgs_df.columns:
            reservoir_usgs_df_time.append((timestamp - t0).total_seconds())
        reservoir_usgs_df_time = np.array(reservoir_usgs_df_time)
        reservoir_usgs_update_time = reservoir_usgs_param_df['update_time'].loc[usgs_wbodies_sub].to_numpy()
        reservoir_usgs_prev_persisted_flow = reservoir_usgs_param_df['prev_persisted_outflow'].loc[usgs_wbodies_sub].to_numpy()
        reservoir_usgs_persistence_update_time = reservoir_usgs_param_df['persistence_update_time'].loc[usgs_wbodies_sub].to_numpy()
        reservoir_usgs_persistence_index = reservoir_usgs_param_df['persistence_index'].loc[usgs_wbodies_sub].to_numpy()
    else:
        reservoir_usgs_df_sub = pd.DataFrame()
        reservoir_usgs_df_time = pd.DataFrame().to_numpy().reshape(0,)
        reservoir_usgs_update_time = pd.DataFrame().to_numpy().reshape(0,)
        reservoir_usgs_prev_persisted_flow = pd.DataFrame().to_numpy().reshape(0,)
        reservoir_usgs_persistence_update_time = pd.DataFrame().to_numpy().reshape(0,)
        reservoir_usgs_persistence_index = pd.DataFrame().to_numpy().reshape(0,)
        if not waterbody_types_df_sub.empty:
            waterbody_types_df_sub.loc[waterbody_types_df_sub['reservoir_type'] == 2] = 1

    # select USACE reservoir DA data waterbodies in sub-domain
    if not reservoir_usace_df.empty:
        usace_wbodies_sub      = waterbody_types_df_sub[
                                    waterbody_types_df_sub['reservoir_type']==3
                                ].index
        if exclude_segments:
            usace_wbodies_sub = list(set(usace_wbodies_sub).difference(set(exclude_segments)))
        reservoir_usace_df_sub = reservoir_usace_df.loc[usace_wbodies_sub]
        reservoir_usace_df_time = []
        for timestamp in reservoir_usace_df.columns:
            reservoir_usace_df_time.append((timestamp - t0).total_seconds())
        reservoir_usace_df_time = np.array(reservoir_usace_df_time)
        reservoir_usace_update_time = reservoir_usace_param_df['update_time'].loc[usace_wbodies_sub].to_numpy()
        reservoir_usace_prev_persisted_flow = reservoir_usace_param_df['prev_persisted_outflow'].loc[usace_wbodies_sub].to_numpy()
        reservoir_usace_persistence_update_time = reservoir_usace_param_df['persistence_update_time'].loc[usace_wbodies_sub].to_numpy()
        reservoir_usace_persistence_index = reservoir_usace_param_df['persistence_index'].loc[usace_wbodies_sub].to_numpy()
    else: 
        reservoir_usace_df_sub = pd.DataFrame()
        reservoir_usace_df_time = pd.DataFrame().to_numpy().reshape(0,)
        reservoir_usace_update_time = pd.DataFrame().to_numpy().reshape(0,)
        reservoir_usace_prev_persisted_flow = pd.DataFrame().to_numpy().reshape(0,)
        reservoir_usace_persistence_update_time = pd.DataFrame().to_numpy().reshape(0,)
        reservoir_usace_persistence_index = pd.DataFrame().to_numpy().reshape(0,)
        if not waterbody_types_df_sub.empty:
            waterbody_types_df_sub.loc[waterbody_types_df_sub['reservoir_type'] == 3] = 1
    
    # RFC reservoirs
    if not reservoir_rfc_df.empty:
        rfc_wbodies_sub = waterbody_types_df_sub[
            waterbody_types_df_sub['reservoir_type']==4
            ].index
        if exclude_segments:
            rfc_wbodies_sub = list(set(rfc_wbodies_sub).difference(set(exclude_segments)))
        reservoir_rfc_df_sub = reservoir_rfc_df.loc[rfc_wbodies_sub]
        reservoir_rfc_totalCounts = reservoir_rfc_param_df['totalCounts'].loc[rfc_wbodies_sub].to_numpy()
        reservoir_rfc_file = reservoir_rfc_param_df['file'].loc[rfc_wbodies_sub].to_list()
        reservoir_rfc_use_forecast = reservoir_rfc_param_df['use_rfc'].loc[rfc_wbodies_sub].to_numpy()
        reservoir_rfc_timeseries_idx = reservoir_rfc_param_df['timeseries_idx'].loc[rfc_wbodies_sub].to_numpy()
        reservoir_rfc_update_time = reservoir_rfc_param_df['update_time'].loc[rfc_wbodies_sub].to_numpy()
        reservoir_rfc_da_timestep = reservoir_rfc_param_df['da_timestep'].loc[rfc_wbodies_sub].to_numpy()
        reservoir_rfc_persist_days = reservoir_rfc_param_df['rfc_persist_days'].loc[rfc_wbodies_sub].to_numpy()
    else:
        reservoir_rfc_df_sub = pd.DataFrame()
        reservoir_rfc_totalCounts = pd.DataFrame().to_numpy().reshape(0,)
        reservoir_rfc_file = []
        reservoir_rfc_use_forecast = pd.DataFrame().to_numpy().reshape(0,)
        reservoir_rfc_timeseries_idx = pd.DataFrame().to_numpy().reshape(0,)
        reservoir_rfc_update_time = pd.DataFrame().to_numpy().reshape(0,)
        reservoir_rfc_da_timestep = pd.DataFrame().to_numpy().reshape(0,)
        reservoir_rfc_persist_days = pd.DataFrame().to_numpy().reshape(0,)
        if not from_files:
            if not waterbody_types_df_sub.empty:
                waterbody_types_df_sub.loc[waterbody_types_df_sub['reservoir_type'] == 4] = 1

    return (
        reservoir_usgs_df_sub, reservoir_usgs_df_time, reservoir_usgs_update_time, reservoir_usgs_prev_persisted_flow, reservoir_usgs_persistence_update_time, reservoir_usgs_persistence_index,
        reservoir_usace_df_sub, reservoir_usace_df_time, reservoir_usace_update_time, reservoir_usace_prev_persisted_flow, reservoir_usace_persistence_update_time, reservoir_usace_persistence_index,
        reservoir_rfc_df_sub, reservoir_rfc_totalCounts, reservoir_rfc_file, reservoir_rfc_use_forecast, reservoir_rfc_timeseries_idx, reservoir_rfc_update_time, reservoir_rfc_da_timestep, reservoir_rfc_persist_days,
        waterbody_types_df_sub
        )
    

# -----------------------------------------------------------------------------
# Abstract Compute Class:
# Define all slots and pass function definitions to child classes
# -----------------------------------------------------------------------------
class AbstractCompute(ABC):
    """
    This just defines all of the slots that are be used by child classes.
    These need to be defined in a parent class so each child class can be
    combined into a single DataAssimilation object without getting a 
    'multiple-inheritance' error.
    """
    __slots__ = ["_reaches_ordered_bytw","_results",]
    
    def __init__(self, connections, rconn, wbody_conn, reaches_bytw, compute_func_name, parallel_compute_method, 
                 subnetwork_target_size, cpu_pool, t0, dt, nts, qts_subdivisions, independent_networks, param_df, 
                 q0, qlats, usgs_df, lastobs_df, reservoir_usgs_df, reservoir_usgs_param_df, reservoir_usace_df, 
                 reservoir_usace_param_df, reservoir_rfc_df, reservoir_rfc_param_df, da_parameter_dict, 
                 assume_short_ts, return_courant, waterbodies_df,data_assimilation_parameters,waterbody_types_df,
                 waterbody_type_specified,subnetwork_list,flowveldepth_interorder = {}, from_files = True,):
        """
        Run subnetworking pre-processing, then computing.
        """
        
        self._wbody_conn = wbody_conn
        self._reaches_bytw = reaches_bytw
        # self._compute_func_name = compute_func_name
        self._parallel_compute_method = parallel_compute_method
        # self._cpu_pool = cpu_pool
        self._t0 = t0
        self._dt = dt
        self._nts = nts
        self._qts_subdivisions = qts_subdivisions
        self._param_df = param_df
        self._q0 = q0
        self._qlats = qlats
        self._from_files = from_files
        self._reservoir_usgs_df = reservoir_usgs_df
        self._reservoir_usgs_param_df = reservoir_usgs_param_df
        self._reservoir_usace_df = reservoir_usace_df
        self._reservoir_usace_param_df = reservoir_usace_param_df
        self._reservoir_rfc_df = reservoir_rfc_df
        self._reservoir_rfc_param_df = reservoir_rfc_param_df
        self._da_parameter_dict = da_parameter_dict
        self._assume_short_ts = assume_short_ts
        self._return_courant = return_courant
        self._data_assimilation_parameters = data_assimilation_parameters
        self._waterbody_types_df = waterbody_types_df
        self._waterbody_type_specified = waterbody_type_specified
        self._usgs_df = usgs_df
        self._lastobs_df = lastobs_df
        self._waterbodies_df = waterbodies_df
        self._independent_networks = independent_networks
        self._subnetwork_list = subnetwork_list
        self._subnetwork_target_size = subnetwork_target_size
        self._connections = connections
        self._rconn = rconn
        self._reaches_ordered_bytw = {}
        self._results = []
        
        
        _compute_func_map = defaultdict(compute_network_structured,{"V02-structured": compute_network_structured,})
        self.compute_func = _compute_func_map[compute_func_name]
        self.da_decay_coefficient = self._da_parameter_dict.get("da_decay_coefficient", 0)
        self._param_df["dt"] = self._dt
        self._param_df = self._param_df.astype("float32")
        self._cpu_pool = multiprocessing.cpu_count()  # Number of CPUs available
        
    @property
    def get_output(self):
        return self._results
    

    @abstractmethod
    def _subset_domain(self,):
        pass
    
    @abstractmethod
    def _route(self,):
        pass


# -----------------------------------------------------------------------------
# Compute class definitions:
#   1. serial
#   2. by_network
#   3. by_subnetwork_jit
#   4. by_subnetwork_jit_clustered: 
# -----------------------------------------------------------------------------
class serial(AbstractCompute):
    def __init__(self, connections, rconn, wbody_conn, reaches_bytw, compute_func_name, parallel_compute_method, 
                 subnetwork_target_size, cpu_pool, t0, dt, nts, qts_subdivisions, independent_networks, param_df, 
                 q0, qlats, usgs_df, lastobs_df, reservoir_usgs_df, reservoir_usgs_param_df, reservoir_usace_df, 
                 reservoir_usace_param_df, reservoir_rfc_df, reservoir_rfc_param_df, da_parameter_dict, 
                 assume_short_ts, return_courant, waterbodies_df,data_assimilation_parameters,waterbody_types_df,
                 waterbody_type_specified,subnetwork_list,flowveldepth_interorder = {}, from_files = True,):
        
        super().__init__(connections, rconn, wbody_conn, reaches_bytw, compute_func_name, parallel_compute_method, 
                 subnetwork_target_size, cpu_pool, t0, dt, nts, qts_subdivisions, independent_networks, param_df, 
                 q0, qlats, usgs_df, lastobs_df, reservoir_usgs_df, reservoir_usgs_param_df, reservoir_usace_df, 
                 reservoir_usace_param_df, reservoir_rfc_df, reservoir_rfc_param_df, da_parameter_dict, 
                 assume_short_ts, return_courant, waterbodies_df,data_assimilation_parameters,waterbody_types_df,
                 waterbody_type_specified,subnetwork_list,flowveldepth_interorder = {}, from_files = True,)
        
       
                
    
    def _subset_domain(self,):
        #TODO Define subsetting method for serial
        self._reaches_ordered_bytw = {}
    
    def _route(self,):
        #TODO Define routing compute method for serial
        self._results = []
    
    
class by_network(serial):
    def __init__(self):
        """
        By Network compute class.
        
        #NOTE I think this can be a subclass of serial. It
        # just needs to route networks in parallel. -shorvath.
        """
        super().__init__()
    
    def _subset_domain(self,):
        #TODO Define subsetting method for by-network
        self._reaches_ordered_bytw = {}
    
    def _route(self,):
        #TODO Define routing compute method for by-network
        self._results = []
    
    
class by_subnetwork_jit(by_network):
    def __init__(self):
        """
        By Network JIT compute class.
        
        #NOTE I think this can be a subclass of by_network. It
        # just needs a couple extra steps to handle 'order',
        # e.g., 'flowveldepth_interorder'. -shorvath.
        """
        super().__init__()
        
    def _subset_domain(self,):
        #TODO Define subsetting method for by-network-jit
        self._reaches_ordered_bytw = {}
    
    def _route(self,):
        #TODO Define routing compute method for by-network-jit
        self._results = []
    
    
class by_subnetwork_jit_clustered(AbstractCompute):
    def __init__(self, connections, rconn, wbody_conn, reaches_bytw, compute_func_name, parallel_compute_method, 
                 subnetwork_target_size, cpu_pool, t0, dt, nts, qts_subdivisions, independent_networks, param_df, 
                 q0, qlats, usgs_df, lastobs_df, reservoir_usgs_df, reservoir_usgs_param_df, reservoir_usace_df, 
                 reservoir_usace_param_df, reservoir_rfc_df, reservoir_rfc_param_df, da_parameter_dict, 
                 assume_short_ts, return_courant, waterbodies_df,data_assimilation_parameters,waterbody_types_df,
                 waterbody_type_specified,subnetwork_list,flowveldepth_interorder = {}, from_files = True,):
        """
        By Network JIT Clustered compute class.
        
        #NOTE I think this can be a subclass of by_subnetwork_jit. It
        # just needs a couple extra steps to cluster subnetworks. -shorvath.
        """
        
        super().__init__(connections, rconn, wbody_conn, reaches_bytw, compute_func_name, parallel_compute_method, 
                 subnetwork_target_size, cpu_pool, t0, dt, nts, qts_subdivisions, independent_networks, param_df, 
                 q0, qlats, usgs_df, lastobs_df, reservoir_usgs_df, reservoir_usgs_param_df, reservoir_usace_df, 
                 reservoir_usace_param_df, reservoir_rfc_df, reservoir_rfc_param_df, da_parameter_dict, 
                 assume_short_ts, return_courant, waterbodies_df,data_assimilation_parameters,waterbody_types_df,
                 waterbody_type_specified,subnetwork_list,flowveldepth_interorder = {}, from_files = True,)
         # Here we find the networks_with_subnetworks_order_jit
        if not self._subnetwork_list[0] or not self._subnetwork_list[1]:
            self.networks_with_subnetworks_ordered_jit = nhd_network.build_subnetworks(
                    self._connections, self._rconn, self._subnetwork_target_size)
        else:
            self._ordered_subn_dict, self.reaches_ordered_bysubntw_clustered = copy.deepcopy(self._subnetwork_list)
        
        # we call the recursive function here to get the self._reaches_ordered_bytw
        self._ordered_subn_dict = find_deepest_dicts(self.networks_with_subnetworks_ordered_jit)
        
        self.execute_all()

    def _clustered_subntw(self,):
        
        self.reaches_ordered_bysubntw_clustered = {"segs": [], "upstreams": {}, "tw": [], "subn_reach_list": []}    
        self.reaches_ordered_bysubntw = defaultdict()

        for subn_tw, subnet in self._ordered_subn_dict.items():
            conn_subn = {k: self._connections[k] for k in subnet if k in self._connections}
            rconn_subn = {k: self._rconn[k] for k in subnet if k in self._rconn}

            if not self._waterbodies_df.empty and not self._usgs_df.empty:
                path_func = partial(nhd_network.split_at_gages_waterbodies_and_junctions,
                                    set(self._usgs_df.index.values),
                                    set(self._waterbodies_df.index.values),
                                    rconn_subn)

            elif self._waterbodies_df.empty and not self._usgs_df.empty:
                path_func = partial(nhd_network.split_at_gages_and_junctions,
                                    set(self._usgs_df.index.values),
                                    rconn_subn)

            elif not self._waterbodies_df.empty and self._usgs_df.empty:
                path_func = partial(nhd_network.split_at_waterbodies_and_junctions,
                                    set(self._waterbodies_df.index.values),
                                    rconn_subn)

            else:
                path_func = partial(nhd_network.split_at_junction, rconn_subn)

            self.reaches_ordered_bysubntw[subn_tw] = nhd_network.dfs_decomposition(rconn_subn, path_func)
            self.segs = list(chain.from_iterable(self.reaches_ordered_bysubntw[subn_tw]))
            
            self.reaches_ordered_bysubntw_clustered["segs"].extend(self.segs)
            self.reaches_ordered_bysubntw_clustered["tw"].append(subn_tw)
            self.reaches_ordered_bysubntw_clustered["subn_reach_list"].extend(self.reaches_ordered_bysubntw[subn_tw])
            self.reaches_ordered_bysubntw_clustered["upstreams"].update(self._independent_networks[subn_tw])    
            self._subnetwork_list = [self._ordered_subn_dict, self.reaches_ordered_bysubntw_clustered]
            self._subnetwork_list = copy.deepcopy(self._subnetwork_list)
    def _prepare_reservoir(self,):
        
        self.results_subn = defaultdict(list)
        self.flowveldepth_interorder = {}
        segs = self.reaches_ordered_bysubntw_clustered["segs"]
        self.offnetwork_upstreams = set()
        segs_set = set(segs)
        for seg in segs:
            for us in self._rconn[seg]:
                if us not in segs_set:
                    self.offnetwork_upstreams.add(us)
        
        segs.extend(self.offnetwork_upstreams)
        
        self.common_segs = list(self._param_df.index.intersection(segs))
        self.wbodies_segs = set(segs).symmetric_difference(self.common_segs)
        
        #Declare empty dataframe
        self.waterbody_types_df_sub = pd.DataFrame()

        if not self._waterbodies_df.empty:
            self.lake_segs = list(self._waterbodies_df.index.intersection(segs))
            self.waterbodies_df_sub = self._waterbodies_df.loc[self.lake_segs,
                ["LkArea", "LkMxE", "OrificeA", "OrificeC", "OrificeE",
                "WeirC","WeirE","WeirL","ifd","qd0", "h0"]]
            
            #If reservoir types other than Level Pool are active
            if not self._waterbody_types_df.empty:
                self.waterbody_types_df_sub = self._waterbody_types_df.loc[self.lake_segs,["reservoir_type"]]
        else:
            self.lake_segs = []
            self.waterbodies_df_sub = pd.DataFrame()   

        self.param_df_sub = self._param_df.loc[self.common_segs,["dt", "bw", "tw", "twcc", "dx", "n", "ncc", "cs", "s0", "alt"],
        ].sort_index()
        
        param_df_sub_super = self.param_df_sub.reindex(
            self.param_df_sub.index.tolist() + self.lake_segs
        ).sort_index()
        
        
        for us_subn_tw in self.offnetwork_upstreams:
            subn_tw_sortposition = param_df_sub_super.index.get_loc(
                us_subn_tw
            )
            self.flowveldepth_interorder[us_subn_tw][
                "position_index"
            ] = subn_tw_sortposition

        self.subn_reach_list = self.reaches_ordered_bysubntw_clustered["subn_reach_list"]
        self.upstreams = self.reaches_ordered_bysubntw_clustered["upstreams"]


        self.subn_reach_list_with_type = _build_reach_type_list(self.subn_reach_list, self.wbodies_segs)
        
        self.qlat_sub = self._qlats.loc[self.param_df_sub.index]
        self.q0_sub = self._q0.loc[self.param_df_sub.index]
                            
        self.param_df_sub = self.param_df_sub.reindex(self.param_df_sub.index.tolist() + self.lake_segs).sort_index()
        
        self.usgs_df_sub, self.lastobs_df_sub, self.da_positions_list_byseg = _prep_da_dataframes(self._usgs_df, self._lastobs_df, self.param_df_sub.index, self.offnetwork_upstreams)
        self.da_positions_list_byreach, self.da_positions_list_bygage = _prep_da_positions_byreach(self.subn_reach_list, self.lastobs_df_sub.index)

        self.qlat_sub = self.qlat_sub.reindex(self.param_df_sub.index)
        self.q0_sub = self.q0_sub.reindex(self.param_df_sub.index)
        # prepare reservoir DA data
        (self.reservoir_usgs_df_sub, 
        self.reservoir_usgs_df_time,
        self.reservoir_usgs_update_time,
        self.reservoir_usgs_prev_persisted_flow,
        self.reservoir_usgs_persistence_update_time,
        self.reservoir_usgs_persistence_index,
        self.reservoir_usace_df_sub, 
        self.reservoir_usace_df_time,
        self.reservoir_usace_update_time,
        self.reservoir_usace_prev_persisted_flow,
        self.reservoir_usace_persistence_update_time,
        self.reservoir_usace_persistence_index,
        self.reservoir_rfc_df_sub, 
        self.reservoir_rfc_totalCounts, 
        self.reservoir_rfc_file, 
        self.reservoir_rfc_use_forecast, 
        self.reservoir_rfc_timeseries_idx, 
        self.reservoir_rfc_update_time, 
        self.reservoir_rfc_da_timestep, 
        self.reservoir_rfc_persist_days,
        self.waterbody_types_df_sub,
        ) = _prep_reservoir_da_dataframes(self._reservoir_usgs_df, self._reservoir_usgs_param_df, self._reservoir_usace_df, self._reservoir_usace_param_df,
                                        self._reservoir_rfc_df, self._reservoir_rfc_param_df, self.waterbody_types_df_sub, self._t0, self._from_files,self.offnetwork_upstreams)        
        
    def _subset_domain(self,):
        #TODO Define subsetting method for by-network-jit-clustered
        self._reaches_ordered_bytw = {}
    
    def _route(self,):
        
        #TODO Define routing compute method for by-network-jit-clustered
        jobs = []
        with Parallel(n_jobs=self._cpu_pool, backend="loky") as parallel:

            jobs.append(
                delayed(self.compute_func)(
                    self._nts,
                    self._dt,
                    self._qts_subdivisions,
                    self.subn_reach_list_with_type,
                    self.upstreams,
                    self.param_df_sub.index.values,
                    self.param_df_sub.columns.values,
                    self.param_df_sub.values,
                    self.q0_sub.values.astype("float32"),
                    self.qlat_sub.values.astype("float32"),
                    self.lake_segs, 
                    self.waterbodies_df_sub.values,
                    self._data_assimilation_parameters,
                    self.waterbody_types_df_sub.values.astype("int32"),
                    self._waterbody_type_specified,
                    self._t0.strftime('%Y-%m-%d_%H:%M:%S'),
                    self.usgs_df_sub.values.astype("float32"),
                    # flowveldepth_interorder,  # obtain keys and values from this dataset
                    np.array(self.da_positions_list_byseg, dtype="int32"),
                    np.array(self.da_positions_list_byreach, dtype="int32"),
                    np.array(self.da_positions_list_bygage, dtype="int32"),
                    self.lastobs_df_sub.get(
                        "lastobs_discharge",
                        pd.Series(index=self.lastobs_df_sub.index, name="Null", dtype="float32"),
                    ).values.astype("float32"),
                    self.lastobs_df_sub.get(
                        "time_since_lastobs",
                        pd.Series(index=self.lastobs_df_sub.index, name="Null", dtype="float32"),
                    ).values.astype("float32"),
                    self.da_decay_coefficient,
                    # USGS Hybrid Reservoir DA data
                    self.reservoir_usgs_df_sub.values.astype("float32"),
                    self.reservoir_usgs_df_sub.index.values.astype("int32"),
                    self.reservoir_usgs_df_time.astype('float32'),
                    self.reservoir_usgs_update_time.astype('float32'),
                    self.reservoir_usgs_prev_persisted_flow.astype('float32'),
                    self.reservoir_usgs_persistence_update_time.astype('float32'),
                    self.reservoir_usgs_persistence_index.astype('float32'),
                    # USACE Hybrid Reservoir DA data
                    self.reservoir_usace_df_sub.values.astype("float32"),
                    self.reservoir_usace_df_sub.index.values.astype("int32"),
                    self.reservoir_usace_df_time.astype('float32'),
                    self.reservoir_usace_update_time.astype("float32"),
                    self.reservoir_usace_prev_persisted_flow.astype("float32"),
                    self.reservoir_usace_persistence_update_time.astype("float32"),
                    self.reservoir_usace_persistence_index.astype("float32"),
                    # RFC Reservoir DA data
                    self.reservoir_rfc_df_sub.values.astype("float32"),
                    self.reservoir_rfc_df_sub.index.values.astype("int32"),
                    self.reservoir_rfc_totalCounts.astype("int32"),
                    self.reservoir_rfc_file,
                    self.reservoir_rfc_use_forecast.astype("int32"),
                    self.reservoir_rfc_timeseries_idx.astype("int32"),
                    self.reservoir_rfc_update_time.astype("float32"),
                    self.reservoir_rfc_da_timestep.astype("int32"),
                    self.reservoir_rfc_persist_days.astype("int32"),
                    {
                        us: fvd
                        for us, fvd in self.flowveldepth_interorder.items()
                        if us in self.offnetwork_upstreams
                    },
                    self._assume_short_ts,
                    self._return_courant,
                    from_files = self._from_files,
                )
            )
            self.results_subn = parallel(jobs)

    def output(self):   
        # self.flowveldepth_interorder = {}
        
        # for subn_tw in self.reaches_ordered_bysubntw_clustered["tw"]:
        #     # TODO: This index step is necessary because we sort the segment index
        #     # TODO: I think there are a number of ways we could remove the sorting step
        #     #       -- the binary search could be replaced with an index based on the known topology
        #     self.flowveldepth_interorder[subn_tw] = {}
        #     
        #     subn_tw_sortposition = (self.results_subn[0].tolist().index(subn_tw))

        #     self.flowveldepth_interorder[subn_tw]["results"] = self.results_subn[1][subn_tw_sortposition]
        #     # what will it take to get just the tw FVD values into an array to pass to the next loop?
        #     # There will be an empty array initialized at the top of the loop, then re-populated here.
        #     # we don't have to bother with populating it after the last group

        
        self._results = self.results_subn 
        
    def execute_all(self,):
        self._clustered_subntw()
        self._prepare_reservoir()
        self._subset_domain()
        self._route()
        self.output()
