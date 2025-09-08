import duckdb
import pandas as pd

from core.data.models.time_series import *
from core.data.streaming.DataRegistry import *

from core.utils.table_names import recover_metadata

# Module for loading time series data from DuckDB database
# This module provides functionality to retrieve time series data with various
# filtering options including time ranges and result limits

def load_series(table_name: str, conn: duckdb.DuckDBPyConnection,
                   start: int = None, end: int = None, limit: int = None) -> TimeSerie:
    """
    Load a time series from DuckDB database with flexible filtering options.
    
    This function retrieves time series data from a DuckDB database table and constructs
    a TimeSerie object. It supports filtering by time range and limiting the number of
    results. The function handles various combinations of start, end, and limit parameters
    to provide intuitive query behavior.
    
    Parameters
    ----------
    table_name : str
        Name of the database table containing the time series data.
        The table name is expected to encode metadata about the series.
    conn : duckdb.DuckDBPyConnection
        Active connection to the DuckDB database.
    start : int, optional
        Start timestamp in milliseconds (inclusive). If None, retrieves from the earliest available data.
    end : int, optional
        End timestamp in milliseconds (inclusive). If None, retrieves up to the latest available data.
    limit : int, optional
        Maximum number of data points to return. If None, returns all data points in the specified range.
        When provided with start/end parameters, the function validates that the limit is reasonable.
    
    Returns
    -------
    TimeSerie
        A TimeSerie instance containing the loaded data and metadata.
    
    Raises
    ------
    ValueError
        If the series is not found in the database, no data is found in the specified range,
        or if the requested limit is less than the available data points in the range.
    
    Notes
    -----
    The function handles different combinations of parameters in specific ways:
    - With only limit: Returns the last 'limit' values
    - With start and limit: Returns up to 'limit' values from start
    - With end and limit: Returns up to 'limit' last values before end
    - With start, end, and limit: Validates that the range contains at most 'limit' points
    
    Future improvements could include an option to provide an existing TimeSerie object
    to populate with the loaded data.
    """
    # Extract metadata from the table name
    # The table name encodes information about the data source, symbol, type, and duration
    metadata = recover_metadata(table_name)
    source, symbol, serie_type, duration = metadata

    # Build the base SQL query with time range filters if provided
    # This constructs a query that selects all columns and applies timestamp filters
    query = f"SELECT * FROM {table_name}"
    conditions = []
    if start is not None:
        conditions.append(f"timestamp >= {start}")  # Inclusive start boundary
    if end is not None:
        conditions.append(f"timestamp <= {end}")    # Inclusive end boundary
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY timestamp"  # Ensure chronological order

    # Handle different limit cases based on which time boundaries are provided
    # This implements different query strategies for various parameter combinations
    if limit is not None:
        if start is None and end is None:
            # Case 1: Only limit provided - Get the most recent 'limit' values
            # First sort descending to get latest values, then re-sort to chronological order
            query = f"SELECT * FROM ({query}) ORDER BY timestamp DESC LIMIT {limit}"
            query = f"SELECT * FROM ({query}) ORDER BY timestamp"
        elif end is None:
            # Case 2: Start and limit provided - Get up to 'limit' values from start
            # Simple LIMIT clause is sufficient as the data is already filtered by start time
            query += f" LIMIT {limit}"
        elif start is None:
            # Case 3: End and limit provided - Get up to 'limit' last values before end
            # Similar to Case 1 but with end time filter already applied
            query = f"SELECT * FROM ({query}) ORDER BY timestamp DESC LIMIT {limit}"
            query = f"SELECT * FROM ({query}) ORDER BY timestamp"
        else:
            # Case 4: Start, end, and limit all provided
            # Validate that the requested range doesn't contain more points than the limit
            count_query = f"SELECT COUNT(*) as count FROM {table_name} WHERE timestamp >= {start} AND timestamp <= {end}"
            count = conn.execute(count_query).fetchdf()['count'].iloc[0]
            if count > limit:
                # Error if the range contains more points than the limit
                # This prevents unexpected truncation of data within a specified range
                raise ValueError(f"Available data points ({count}) in the specified time range exceeds requested limit {limit}")
            query += f" LIMIT {limit}"

    # Execute the constructed query and handle potential errors
    try:
        # Execute query and fetch results as a pandas DataFrame
        df = conn.execute(query).fetchdf()
    except duckdb.CatalogException:
        # Handle case where the table doesn't exist in the database
        raise ValueError(f"Series {metadata} not found in database")

    # Check if any data was returned
    if df.empty:
        # No data found within the specified filters
        raise ValueError("No data found for the specified time range")

    # Convert pandas DataFrame to numpy arrays for TimeSerie construction
    # Extract timestamps and data columns separately
    timestamps = df['timestamp'].to_numpy()  # Extract timestamp column as numpy array
    data_cols = [c for c in df.columns if c != 'timestamp']  # Get all non-timestamp columns
    data = df[data_cols].to_numpy()  # Extract data values as numpy array

    # Create and return a TimeSerie object with the loaded data and metadata
    # This constructs a TimeSerie instance that can be used by other components
    return TimeSerie(
        data=data,            # Numpy array of data values
        timestamps=timestamps, # Numpy array of timestamps
        duration=duration,     # Time interval between data points
        source=source,         # Data source identifier
        symbol=symbol,         # Trading pair symbol
        serie_type=serie_type  # Type of time series
    )

