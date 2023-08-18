"""Basic Model Interface implementation for reservoirs."""

import numpy as np
from bmipy import Bmi
from pathlib import Path

# Here is the model we want to run
from model_DAforcing import DAforcing_model

class bmi_DAforcing(Bmi):

    def __init__(self):
        """Create a Bmi DA forcing model that is ready for initialization."""
        super(bmi_DAforcing, self).__init__()
        self._model = None
        self._values = {}
        #self._var_units = {}
        self._var_loc = "node"
        self._var_grid_id = 0
        #self._grids = {}
        #self._grid_type = {}

        self._start_time = 0.0
        self._end_time = np.finfo("d").max
        self._time_units = "s"

    #----------------------------------------------
    # Required, static attributes of the model
    #----------------------------------------------
    _att_map = {
        'model_name':         'DA forcing for Next Generation NWM',
        'version':            '',
        'author_name':        '',
        'grid_type':          'scalar', 
        'time_step_size':      1,       
        #'time_step_type':     'donno', #unused  
        #'step_method':        'none',  #unused
        #'time_units':         '1 hour' #NJF Have to drop the 1 for NGEN to recognize the unit
        'time_units':         'seconds' }

    #---------------------------------------------
    # Input variable names (CSDMS standard names)
    #---------------------------------------------
    _input_var_names = [
        # waterbody static variables
        #'lake_surface__elevation',
        #'LkArea',
        #'WeirE',
        #'WeirC',
        #'WeirL',
        #'dam_length',
        #'OrificeE',
        #'OrificeC',
        #'OrificeA',
        #'LkMxE',
        #'waterbody_id',
        #'ifd',
        #'upstream_ids',
        #'res_type',
        #'da_idx',
        #'time_step',
        #'rfc_forecast_persist_seconds',
        #'synthetic_flag',
        # dynamic forcing/DA variables        
        #'lake_water~incoming__volume_flow_rate',
        # RFC DA inputs
        #rfc_timeseries_offset_hours,
        #rfc_gage_id,
        #rfc_timeseries_folder,
        'lake_number',   
        ]

    #---------------------------------------------
    # Output variable names (CSDMS standard names)
    #---------------------------------------------
    _output_var_names = [
                         #'lake_water~outgoing__volume_flow_rate',
                         #'lake_surface__elevation',
                        # RFC DA ouputs
                        'waterbody_rfc__use_flag', #'use_RFC', 
                        'waterbody_rfc__observed_volume_flow_rate', #'rfc_timeseries_discharges', 
                        'waterbody_rfc__timeseries_index', #'rfc_timeseries_idx', 
                        'waterbody_rfc__timeseries_update_time', #'rfc_timeseries_update_time', 
                        'waterbody_rfc__da_time_step', #'rfc_da_time_step', 
                        'waterbody_rfc__total_count', #'rfc_total_counts',
                        'waterbody_rfc__file_of_observed_volume_flow_rate', #'rfc_timeseries_file',
                        ]

    #------------------------------------------------------
    # Create a Python dictionary that maps CSDMS Standard
    # Names to the model's internal variable names.
    #------------------------------------------------------
    #TODO update all these...
    _var_name_units_map = {
        #'channel_exit_water_x-section__volume_flow_rate':['streamflow_cms','m3 s-1'],
        #'channel_water_flow__speed':['streamflow_ms','m s-1'],
        #'channel_water__mean_depth':['streamflow_m','m'],
        #'lake_water~incoming__volume_flow_rate':['waterbody_cms','m3 s-1'],
        #'lake_water~outgoing__volume_flow_rate':['waterbody_cms','m3 s-1'],
        #'lake_surface__elevation':['waterbody_m','m'],
        #--------------   Dynamic inputs --------------------------------
        #'land_surface_water_source__volume_flow_rate':['streamflow_cms','m3 s-1'],
        #'coastal_boundary__depth':['depth_m', 'm'],
        #'usgs_gage_observation__volume_flow_rate':['streamflow_cms','m3 s-1'],
        #'reservoir_usgs_gage_observation__volume_flow_rate':['streamflow_cms','m3 s-1'],
        #'reservoir_usace_gage_observation__volume_flow_rate':['streamflow_cms','m3 s-1'],
        #'rfc_gage_observation__volume_flow_rate':['streamflow_cms','m3 s-1'],
        #'lastobs__volume_flow_rate':['streamflow_cms','m3 s-1']
        # TODO: RFC unit map
        'waterbody__type_number': ['',''],
        'waterbody__lake_number': ['','string'],  #'lake_number':['',''],
        'waterbody_rfc__use_flag': ['use_RFC','boolean'],     #'use_RFC':['',''], 
        'waterbody_rfc__observed_volume_flow_rate': ['streamflow_cms in timeseries', 'm3 s-1'], # 'rfc_timeseries_discharges':['streamflow_cms','m3 s-1'], 
        'waterbody_rfc__timeseries_index': [' ',' '], #'rfc_timeseries_idx':['time_step_count',''], 
        'waterbody_rfc__timeseries_update_time': ['time', 'sec'],      #'rfc_timeseries_update_time':['time','s'], 
        'waterbody_rfc__da_time_step': ['', 'sec'],  #'rfc_da_time_step':['time_step','s'], 
        'waterbody_rfc__total_count': ['','int'],   #'rfc_total_counts':['count',''],
        'waterbody_rfc__file_of_observed_volume_flow_rate': ['',''],  #'rfc_timeseries_file':['',''],
        'usace_timeslice_discharge':['streamflow_cms','m3 s-1'],
        'usace_timeslice_stationId':['',''],
        'usace_timeslice_time':['time',''],
    }

    #------------------------------------------------------
    # A list of static attributes. Not all these need to be used.
    #------------------------------------------------------
    _static_attributes_list = []


    #------------------------------------------------------------
    #------------------------------------------------------------
    # BMI: Model Control Functions
    #------------------------------------------------------------ 
    #------------------------------------------------------------

    #-------------------------------------------------------------------
    def initialize(self, bmi_cfg_file=None):
        
        # -------------- Read in the BMI configuration -------------------------#
        if bmi_cfg_file:
            bmi_cfg_file = Path(bmi_cfg_file)

        # ------------- Initialize t-route model ------------------------------#
        self._model = DAforcing_model(bmi_cfg_file) 

        # ----- Create some lookup tabels from the long variable names --------#
        self._var_name_map_long_first = {long_name:self._var_name_units_map[long_name][0] for \
                                         long_name in self._var_name_units_map.keys()}
        self._var_name_map_short_first = {self._var_name_units_map[long_name][0]:long_name for \
                                          long_name in self._var_name_units_map.keys()}
        self._var_units_map = {long_name:self._var_name_units_map[long_name][1] for \
                                          long_name in self._var_name_units_map.keys()}
        

        # -------------- Initalize all the variables --------------------------# 
        # -------------- so that they'll be picked up with the get functions --#
        self._values['waterbody__lake_number'] = np.zeros(1, dtype='<U19')
        self._values['waterbody__type_number'] = np.zeros(1, dtype=int)
        '''
        self._values['use_RFC'] = np.zeros(1, dtype=bool) 
        self._values['rfc_timeseries_discharges'] = np.zeros(289) 
        self._values['rfc_timeseries_idx'] = np.zeros(1) 
        self._values['rfc_timeseries_update_time'] = np.zeros(1) 
        self._values['rfc_da_time_step'] = np.zeros(1) 
        self._values['rfc_total_counts'] = np.zeros(1)
        self._values['rfc_timeseries_file'] = np.zeros(1,  dtype='<U19')
        self._values['usace_timeslice_discharge'] = np.zeros(5192)
        self._values['usace_timeslice_stationId'] = np.zeros(5192)
        self._values['usace_timeslice_time'] = np.zeros(5192)
        '''
        '''
        #FIXME Do this better..., load size of variables from config file??
        self._values['lake_surface__elevation'] = np.zeros(1)
        self._values['LkArea'] = np.zeros(1)
        self._values['WeirE'] = np.zeros(1)
        self._values['WeirC'] = np.zeros(1)
        self._values['WeirL'] = np.zeros(1)
        self._values['dam_length'] = np.zeros(1)
        self._values['OrificeE'] = np.zeros(1)
        self._values['OrificeC'] = np.zeros(1)
        self._values['OrificeA'] = np.zeros(1)
        self._values['LkMxE'] = np.zeros(1)
        self._values['waterbody_id'] = np.zeros(1)
        self._values['ifd'] = np.zeros(1)
        self._values['upstream_ids'] = np.zeros(1, dtype=int)
        self._values['reservoir_type'] = np.zeros(1)
        self._values['lake_water~incoming__volume_flow_rate'] = np.zeros(12)
        self._values['lake_water~outgoing__volume_flow_rate'] = np.zeros(1)


        #TODO: how will we know the size of these arrays?
        self._values['gage_observations'] = np.zeros(120)
        self._values['gage_time'] = np.zeros(120)

        self._values['da_idx'] = np.zeros(1, dtype=int)
        self._values['time_step'] = np.zeros(1)
        self._values['rfc_forecast_persist_seconds'] = np.zeros(1)
        self._values['synthetic_flag'] = np.zeros(289)
        
        #RFC DA        
        self._values['rfc_timeseries_offset_hours'] = np.zeros(1)
        self._values['rfc_forecast_persist_days'] = np.zeros(1)
        '''
        '''
        for var_name in self._input_var_names + self._output_var_names:
            # ---------- Temporarily set to 3 values ------------------#
            # ---------- so just set to zero for now ------------------#
            self._values[var_name] = np.zeros(3)
        '''


    def update(self):
        """Advance model by one time step."""
        if self._model._time==0.0:
            self._model.preprocess_static_vars(self._values) 

        self._model.run(self._values)

    def update_until(self, until):
        """Update model until a particular time.
        Parameters
        ----------
        until : int
            Time to run model until in seconds.
        """
        n_steps = int(until/self._model._time_step)

        for _ in range(int(n_steps)):
            self.update()

    def finalize(self):
        """Finalize model."""

        self._model = None
    
    def update_frac(self, time_frac):
        """Update model by a fraction of a time step.
        Parameters
        ----------
        time_frac : float
            Fraction fo a time step.
        """
        time_step = self.get_time_step()
        self._model.time_step = time_frac * time_step
        self.update()
        self._model.time_step = time_step

    def get_var_type(self, var_name):
        """Data type of variable.
        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        Returns
        -------
        str
            Data type.
        """
        return str(self.get_value_ptr(var_name).dtype)

    def get_var_units(self, var_name):
        """Get units of variable.
        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        Returns
        -------
        str
            Variable units.
        """
        return self._var_units[var_name]

    def get_var_nbytes(self, var_name):
        """Get units of variable.
        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        Returns
        -------
        int
            Size of data array in bytes.
        """
        return self.get_value_ptr(var_name).nbytes

    def get_var_itemsize(self, name):
        return np.dtype(self.get_var_type(name)).itemsize

    def get_var_location(self, name):
        return self._var_loc[name]

    def get_var_grid(self, var_name):
        """Grid id for a variable.
        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        Returns
        -------
        int
            Grid id.
        """
        for grid_id, var_name_list in self._grids.items():
            if var_name in var_name_list:
                return grid_id

    def get_grid_rank(self, grid_id):
        """Rank of grid.
        Parameters
        ----------
        grid_id : int
            Identifier of a grid.
        Returns
        -------
        int
            Rank of grid.
        """
        return len(self._model.shape)

    def get_grid_size(self, grid_id):
        """Size of grid.
        Parameters
        ----------
        grid_id : int
            Identifier of a grid.
        Returns
        -------
        int
            Size of grid.
        """
        return int(np.prod(self._model.shape))

    def get_value_ptr(self, var_name):
        """Reference to values.
        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        Returns
        -------
        array_like
            Value array.
        """
        return self._values[var_name]

    def get_value(self, var_name):
        """Copy of values.
        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        Returns
        -------
        output_df : pd.DataFrame
            Copy of values.
        """
        output_df = self.get_value_ptr(var_name)
        return output_df

    def get_value_at_indices(self, var_name, dest, indices):
        """Get values at particular indices.
        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        dest : ndarray
            A numpy array into which to place the values.
        indices : array_like
            Array of indices.
        Returns
        -------
        array_like
            Values at indices.
        """
        dest[:] = self.get_value_ptr(var_name).take(indices)
        return dest

    def set_value(self, var_name, src):
        """
        Set model values
        
        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        src : array_like
            Array of new values.
        """
        val = self.get_value_ptr(var_name)
        val[:] = src.reshape(val.shape)
        
        #self._values[var_name] = src

    def set_value_at_indices(self, name, inds, src):
        """Set model values at particular indices.
        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        src : array_like
            Array of new values.
        indices : array_like
            Array of indices.
        """
        val = self.get_value_ptr(name)
        val.flat[inds] = src

    def get_component_name(self):
        """Name of the component."""
        return self._name

    def get_input_item_count(self):
        """Get names of input variables."""
        return len(self._input_var_names)

    def get_output_item_count(self):
        """Get names of output variables."""
        return len(self._output_var_names)

    def get_input_var_names(self):
        """Get names of input variables."""
        return self._input_var_names

    def get_output_var_names(self):
        """Get names of output variables."""
        return self._output_var_names

    def get_grid_shape(self, grid_id, shape):
        """Number of rows and columns of uniform rectilinear grid."""
        var_name = self._grids[grid_id][0]
        shape[:] = self.get_value_ptr(var_name).shape
        return shape

    def get_grid_spacing(self, grid_id, spacing):
        """Spacing of rows and columns of uniform rectilinear grid."""
        spacing[:] = self._model.spacing
        return spacing

    def get_grid_origin(self, grid_id, origin):
        """Origin of uniform rectilinear grid."""
        origin[:] = self._model.origin
        return origin

    def get_grid_type(self, grid_id):
        """Type of grid."""
        return self._grid_type[grid_id]

    def get_start_time(self):
        """Start time of model."""
        return self._start_time

    def get_end_time(self):
        """End time of model."""
        return self._end_time

    def get_current_time(self):
        return self._model._time

    def get_time_step(self):
        return self._model._time_step

    def get_time_units(self):
        return self._time_units

    def get_grid_edge_count(self, grid):
        raise NotImplementedError("get_grid_edge_count")

    def get_grid_edge_nodes(self, grid, edge_nodes):
        raise NotImplementedError("get_grid_edge_nodes")

    def get_grid_face_count(self, grid):
        raise NotImplementedError("get_grid_face_count")

    def get_grid_face_nodes(self, grid, face_nodes):
        raise NotImplementedError("get_grid_face_nodes")

    def get_grid_node_count(self, grid):
        """Number of grid nodes.
        Parameters
        ----------
        grid : int
            Identifier of a grid.
        Returns
        -------
        int
            Size of grid.
        """
        return self.get_grid_size(grid)

    def get_grid_nodes_per_face(self, grid, nodes_per_face):
        raise NotImplementedError("get_grid_nodes_per_face")

    def get_grid_face_edges(self, grid, face_edges):
        raise NotImplementedError("get_grid_face_edges")

    def get_grid_x(self, grid, x):
        raise NotImplementedError("get_grid_x")

    def get_grid_y(self, grid, y):
        raise NotImplementedError("get_grid_y")

    def get_grid_z(self, grid, z):
        raise NotImplementedError("get_grid_z")
        
    def _parse_config(self, cfg):
        cfg_list = [cfg.get('flag'),cfg.get('file')]
        return cfg_list