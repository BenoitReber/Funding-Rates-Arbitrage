import numpy as np
import time

from core.utils.reserve_shared_memory import reserve_shared_memory

#from mpi4py.util.sync import *
import mpi4py.util.sync as mpi4py_sync
from mpi4py.util.sync import Semaphore

class TimeSerie:
    """
    Base class for all time series in the system.
    
    This class provides the foundation for storing and managing time series data
    with shared memory support for multi-process environments. It handles the basic
    properties of a time series including timestamps, data values, and metadata.
    
    Time series are implemented as circular buffers using numpy arrays, allowing
    efficient updates and memory usage for streaming data applications.
    """
    def __init__(self, data: np.ndarray,
                       timestamps: np.ndarray,
                       duration: int,
                       source: str,
                       symbol: str,
                       serie_type: str) -> None:
        """
        Initialize a TimeSerie object with data arrays and metadata.
        
        Parameters
        ----------
        data : np.ndarray
            Array containing the time series values. For OHLCV data, this would be a 2D array
            with each row containing [open, high, low, close, volume] values.
        timestamps : np.ndarray
            Array of timestamps corresponding to each data point.
            For OHLCV data, the timestamp t contains data for interval [t, t+duration).
        duration : int
            Time interval in milliseconds between consecutive data points.
        source : str
            Data source identifier (e.g., "binance", "local").
        symbol : str
            Trading pair symbol (e.g., "BTC/USDT").
        serie_type : str
            Type of time series (e.g., "OHLCV", "fundingRate").
        """
        # Note: for OHLCV data, the value associated with timestamp t contains data
        # acquired between t and t+duration. As a consequence, the latest value stored
        # is expected to be updated multiple times until we reach time t+duration.
        self.timestamps = timestamps
        self.data = data

        self.source = source
        self.symbol = symbol
        self.serie_type = serie_type
        self.duration = duration # time in ms between two values

        self.dependencies = []

        #self.metadata = metadata  # {source: "binance", pair: "BTC/USDT", type: "OHLCV"}
    
    @staticmethod 
    def create(rank, comm, data_shape, duration, cache, source="", symbol="", serie_type=""):
        """
        Create a new TimeSerie with shared memory arrays.
        
        This factory method allocates shared memory for the time series data and timestamps,
        allowing multiple processes to access the same data without copying.
        
        Parameters
        ----------
        rank : int
            Process rank in the MPI communicator.
        comm : MPI.Comm
            MPI communicator object.
        data_shape : tuple
            Shape of the data array to create (tuple of integers).
        duration : int
            Time interval in milliseconds between consecutive data points.
        cache : bool
            Whether to use cache for shared memory.
        source : str, optional
            Data source identifier.
        symbol : str, optional
            Trading pair symbol.
        serie_type : str, optional
            Type of time series.
            
        Returns
        -------
        TimeSerie
            A new TimeSerie instance with shared memory arrays.
        """
        data = reserve_shared_memory(data_shape, rank, comm, cache = cache, init = np.nan)
        timestamps = reserve_shared_memory(1, rank, comm, cache = cache, init = -1)
        ts = TimeSerie( data, timestamps, duration, source, symbol, serie_type )
        return ts
    
    def get_metadata(self):
        """
        Get the metadata of this time series.
        
        Returns
        -------
        tuple
            A tuple containing (source, symbol, serie_type, duration).
        """
        return (self.source, self.symbol, self.serie_type, self.duration)
    
    def reset_metadata(self, source, symbol, serie_type, duration):
        """
        Reset the metadata of this time series.
        
        Parameters
        ----------
        source : str
            New data source identifier.
        symbol : str
            New trading pair symbol.
        serie_type : str
            New type of time series.
        duration : int
            New time interval in milliseconds.
        ----
        """
        self.source = source
        self.symbol = symbol
        self.serie_type = serie_type
        self.duration = duration
    
    
class BuiltSerie(TimeSerie):
    """
    Time series computed locally from other time series.
    
    This class represents time series that are derived from other time series
    through a transformation function. It maintains dependencies on parent series
    and updates its values when the parent series are updated.
    
    The computation follows a directed acyclic graph (DAG) pattern where each
    BuiltSerie depends on one or more parent series (which can be either
    ThirdPartySerie or other BuiltSerie instances).
    """
    def __init__(self, data: np.ndarray,
                       timestamps: np.ndarray,
                       duration: int,
                       transition_function,
                       dependencies,
                       symbol: str,
                       serie_type: str):
        """
        Initialize a BuiltSerie with data arrays, dependencies, and a transition function.
        
        Parameters
        ----------
        data : np.ndarray
            Array containing the time series values.
        timestamps : np.ndarray
            Array of timestamps corresponding to each data point.
        duration : int
            Time interval in milliseconds between consecutive data points.
        transition_function : callable
            Function that computes the next value based on dependencies and current data.
            Should accept two parameters: list of dependency data arrays and current data array.
        dependencies : list
            List of TimeSerie objects that this series depends on.
        symbol : str
            Trading pair symbol (e.g., "BTC/USDT").
        serie_type : str
            Type of time series (e.g., "SMA", "RSI").
        """
        super().__init__(data, timestamps, duration, "local", symbol, serie_type)

        self.dependencies = dependencies # [TimeSerie]
        self.transition = transition_function # the function used to compute the value at the next time step

        self.symbol = symbol
        self.serie_type = serie_type

    @staticmethod 
    def new(rank, comm, data_shape, duration, cache, transition_function, dependencies, symbol, serie_type):
        """
        Create a new BuiltSerie with shared memory arrays.
        
        Parameters
        ----------
        rank : int
            Process rank in the MPI communicator.
        comm : MPI.Comm
            MPI communicator object.
        data_shape : tuple
            Shape of the data array to create.
        duration : int
            Time interval in milliseconds between consecutive data points.
        cache : bool
            Whether to use cache for shared memory.
        transition_function : callable
            Function that computes the next value based on dependencies and current data.
        dependencies : list
            List of TimeSerie objects that this series depends on.
        symbol : str
            Trading pair symbol.
        serie_type : str
            Type of time series.
            
        Returns
        -------
        BuiltSerie
            A new BuiltSerie instance with shared memory arrays.
        """
        tmp = TimeSerie.create( rank, comm, data_shape, duration, cache )
        ts = BuiltSerie( tmp.data, tmp.timestamps, duration, transition_function, dependencies, symbol, serie_type)
        return ts

    def update(self, next_timestamp=None):
        """
        Update this time series based on its dependencies.
        
        This method checks if all dependencies have up-to-date data for the next timestamp.
        If so, it computes the next value using the transition function and updates the
        time series data and timestamp arrays.
        
        The update follows these steps:
        1. Determine the next timestamp to compute
        2. Check if all parent time series have data available for this timestamp
        3. If all data is available, compute the next value and update the arrays
        4. If any data is missing, do nothing (will try again later)
        
        Returns
        -------
        None
        """
        miss_data = False

        if next_timestamp is None: 
            if self.timestamps[-1] > 0:
                next_timestamp = self.timestamps[-1] + self.duration
            else:
                next_timestamp = int( (self.dependencies[0].timestamps[-1]) / self.duration ) * self.duration
        
        # check if the parents timeseries are up to date
        for parent in self.dependencies:
            last_desired = int( next_timestamp / parent.duration ) * parent.duration
            if parent.timestamps[-1] < last_desired:
                miss_data = True
                break
            elif parent.timestamps[-1] > last_desired:
                miss_data = True
                self.update(parent.timestamps[-1])
                break

        if not miss_data:
            new_values = self._compute_next()
            self.data = np.roll(self.data, -1, axis=0)
            self.data[-1] = new_values
            self.timestamps = np.roll(self.timestamps, -1, axis=0)
            self.timestamps[-1] = next_timestamp #self.timestamps[-2] + self.duration

    def _compute_next(self):
        """
        Compute the next value using the transition function.
        
        This method applies the transition function to the dependency data arrays
        and the current data array to compute the next value in the time series.
        
        Returns
        -------
        array-like
            The computed value for the next time step.
        """
        return self.transition(
            [dep.data for dep in self.dependencies],
            self.data
        )
        

class ThirdPartySerie(TimeSerie):
    """
    Time series that integrates data from external sources.
    
    This class represents time series whose data comes from external sources
    (like cryptocurrency exchanges) rather than being computed locally.
    It includes synchronization mechanisms (semaphores) to notify other processes
    when new data is available.
    
    ThirdPartySerie instances are typically updated by data acquisition components
    that fetch data from external APIs or data feeds.
    """
    def __init__(self, data: np.ndarray, 
                       timestamps: np.ndarray, 
                       duration: int, 
                       source: str, 
                       symbol: str, 
                       serie_type: str,
                       semaphore: mpi4py_sync.Semaphore):
        """
        Initialize a ThirdPartySerie with data arrays, metadata, and a semaphore.
        
        Parameters
        ----------
        data : np.ndarray
            Array containing the time series values.
        timestamps : np.ndarray
            Array of timestamps corresponding to each data point.
        duration : int
            Time interval in milliseconds between consecutive data points.
        source : str
            External data source identifier (e.g., "binance", "bybit").
        symbol : str
            Trading pair symbol (e.g., "BTC/USDT").
        serie_type : str
            Type of time series (e.g., "OHLCV", "fundingRate").
        semaphore : mpi4py_sync.Semaphore
            Semaphore used for inter-process synchronization.
        """
        super().__init__(data, timestamps, duration, source, symbol, serie_type)

        # the semaphore used to notify of updates to series acquired from third-party sources 
        self.semaphore = semaphore

    @staticmethod 
    def new(rank, comm, data_shape, duration, cache, source, symbol, serie_type):
        """
        Create a new ThirdPartySerie with shared memory arrays.
        
        Parameters
        ----------
        rank : int
            Process rank in the MPI communicator.
        comm : MPI.Comm
            MPI communicator object.
        data_shape : tuple
            Shape of the data array to create.
        duration : int
            Time interval in milliseconds between consecutive data points.
        cache : bool
            Whether to use cache for shared memory.
        source : str
            External data source identifier.
        symbol : str
            Trading pair symbol.
        serie_type : str
            Type of time series.
            
        Returns
        -------
        ThirdPartySerie
            A new ThirdPartySerie instance with shared memory arrays and a semaphore.
        """
        # Create a TimeSerie instance with the given parameters
        tmp = TimeSerie.create(rank, comm, data_shape, duration, cache, source, symbol, serie_type)
        semaphore = Semaphore(value = 1, comm=comm, bounded=False)
        # Wrap it into a ThirdPartySerie
        ts = ThirdPartySerie(tmp.data, tmp.timestamps, tmp.duration, tmp.source, tmp.symbol, tmp.serie_type, semaphore)
        return ts

    def update(self, new_values, new_timestamp):
        """
        Update this time series with new values from an external source.
        
        This method is called by data acquisition components when new data is available.
        It updates the circular buffer by rolling the arrays and setting the new values.
        
        Parameters
        ----------
        new_values : array-like
            New data values to add to the time series.
        new_timestamp : int
            Timestamp corresponding to the new values, in milliseconds.
            
        Returns
        -------
        None
        """
        self.data = np.roll(self.data, -1, axis=0)
        self.data[-1] = new_values
        self.timestamps = np.roll(self.timestamps, -1, axis=0)
        self.timestamps[-1] = new_timestamp