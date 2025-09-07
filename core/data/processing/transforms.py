from talib import abstract
import numpy as np

# Module for time series data transformations
# This module provides functions for technical analysis and OHLCV data resampling

def ta_lib(x, y, function_name, **kwargs):
    """
    Apply a TA-Lib technical analysis function to OHLCV data.
    
    This function serves as a wrapper around TA-Lib's abstract API, allowing
    various technical indicators to be calculated on OHLCV (Open, High, Low, Close, Volume)
    time series data. It extracts the required price components from the input array
    and returns the most recent calculated value.
    
    Parameters
    ----------
    x : list
        List containing a numpy array with OHLCV data. The array is expected to have
        shape (n, 5) where n is the number of time periods and columns represent
        [open, high, low, close, volume].
    y : any
        Placeholder parameter for compatibility with the transition function interface,
        not used in this function.
    function_name : str
        Name of the TA-Lib function to apply (e.g., 'SMA', 'RSI', 'MACD').
    **kwargs : dict
        Additional parameters to pass to the TA-Lib function (e.g., timeperiod).
    
    Returns
    -------
    float or numpy.ndarray
        The most recent value of the calculated technical indicator.
    
    Notes
    -----
    - This function is designed to be used as a transition function in the BuiltSerie class
    - Only returns the last calculated value, suitable for incremental updates
    - Requires the TA-Lib library to be installed
    """
    # Create a TA-Lib function object based on the provided function name
    function = abstract.Function(function_name)
    
    # Extract OHLCV components from the input array
    # The input is expected to be a list with a numpy array at index 0
    # Each column in the array represents a different price component
    inputs = {
        'open': (x[0])[:,0],    # First column: Open prices
        'high': (x[0])[:,1],    # Second column: High prices
        'low': (x[0])[:,2],     # Third column: Low prices
        'close': (x[0])[:,3],   # Fourth column: Close prices
        'volume': (x[0])[:,4]   # Fifth column: Volume data
    }
    
    # Apply the TA-Lib function to the input data with any additional parameters
    tmp = np.array(function(inputs, **kwargs))
    
    # Return only the most recent calculated value
    # This is suitable for incremental updates in a time series
    return tmp[-1]

def ohlcv_from_ohlcv(x: list,  # List[np.ndarray],
                     y: np.ndarray,
                     duration_in: int,
                     duration_out: int) -> np.ndarray:
    """
    Resample OHLCV data from a shorter timeframe to a longer timeframe.
    
    This function converts OHLCV (Open, High, Low, Close, Volume) data from a shorter
    time duration to a longer one by aggregating multiple periods. For example, converting
    1-minute candles to 5-minute candles. The conversion follows standard OHLCV aggregation rules:
    - Open: First value of the period
    - High: Maximum value in the period
    - Low: Minimum value in the period
    - Close: Last value of the period
    - Volume: Sum of all values in the period
    
    Parameters
    ----------
    x : list
        List containing a single numpy array with OHLCV data at the input duration.
        The array is expected to have shape (n, 5) where columns represent
        [open, high, low, close, volume].
    y : np.ndarray
        Placeholder parameter for compatibility with the transition function interface,
        not used in this function.
    duration_in : int
        The duration (in time units) of the input data points.
    duration_out : int
        The desired output duration (in the same time units).
        Must be a multiple of duration_in.
    
    Returns
    -------
    np.ndarray
        A single OHLCV data point at the target duration, as a numpy array with shape (5,).
    
    Raises
    ------
    Exception
        If duration_out is not a multiple of duration_in, as this would make
        proper resampling impossible.
    
    Notes
    -----
    - This function is designed to be used as a transition function in the BuiltSerie class
    - Only the most recent output period is calculated, suitable for incremental updates
    - Assumes that enough input data points are available to form a complete output period
    """
    # Validate that the output duration is a multiple of the input duration
    # This is necessary for proper resampling without partial periods
    if not duration_out % duration_in == 0:
        # TODO: Replace with a specific error type and message
        # The current implementation has an incomplete error handling
        raise ValueError(f"Output duration ({duration_out}) must be a multiple of input duration ({duration_in})")
    # Extract the input time series data from the list
    ts_in = x[0]  # Input OHLCV data array
    
    # Calculate how many input periods make up one output period
    ratio = int(duration_out / duration_in)
    
    # Create the output OHLCV data point following standard aggregation rules:
    ts_out = np.array(
        [ ts_in[-ratio,0],             # Open: take the first open price in the period
          np.max(ts_in[(-ratio):,1]),  # High: take the maximum high price in the period
          np.min(ts_in[(-ratio):,2]),  # Low: take the minimum low price in the period
          ts_in[-1,3],                 # Close: take the last close price in the period
          np.sum(ts_in[(-ratio):,4])   # Volume: sum all volume values in the period
        ]
    )
    
    return ts_out