import duckdb
import pandas as pd

from core.data.models.time_series import *
from core.data.streaming.DataRegistry import *

from core.utils.table_names import generate_table_name

# Module for persisting time series data to DuckDB database
# This module provides functionality to save TimeSerie objects to database tables
# with options for column naming and update strategies

def save_series(series: TimeSerie, conn: duckdb.DuckDBPyConnection, 
                column_names: list = None, replace: bool = False):
    """
    Save a time series to DuckDB database with customizable column names and update strategy.
    
    This function persists a TimeSerie object to a DuckDB database table. The table name
    is automatically generated from the series metadata (source, symbol, serie_type, duration).
    Only valid data points (timestamps > -infinity) are saved. The function supports both
    appending new data and replacing existing data based on matching timestamps.
    
    Parameters
    ----------
    series : TimeSerie
        The time series object to save. Contains data values, timestamps, and metadata.
    conn : duckdb.DuckDBPyConnection
        Active connection to the DuckDB database.
    column_names : list, optional
        Custom names for the data columns. If None, default names ('data_0', 'data_1', etc.)
        will be used. Must match the number of data columns in the series.
    replace : bool, default=False
        Update strategy:
        - If True: Replace existing data points with matching timestamps
        - If False: Skip existing data points (INSERT OR IGNORE)
    
    Raises
    ------
    ValueError
        If the length of column_names doesn't match the number of data columns.
    
    Notes
    -----
    - The table is created if it doesn't exist, with timestamp as the primary key
    - Only valid data points (timestamps > -infinity) are saved
    - For multi-dimensional data (e.g., OHLCV), each dimension becomes a separate column
    - The function handles both 1D and 2D data arrays appropriately
    """
    # Extract metadata from the series and generate a table name
    # The table name encodes information about the data source, symbol, type, and duration
    metadata = series.get_metadata()
    table_name = generate_table_name(*metadata)

    # Filter out invalid timestamps (those set to -infinity)
    # This ensures we only save valid data points from the circular buffer
    indices = np.argwhere(series.timestamps > -np.inf)[:,0]  # Get indices of valid timestamps
    timestamps = series.timestamps[indices]                  # Extract valid timestamps
    
    # Prepare DataFrame with timestamps as the first column
    # Using pandas DataFrame as an intermediate representation before saving to database
    df = pd.DataFrame({
        'timestamp': timestamps.flatten()  # Ensure timestamps are flattened to 1D array
    })

    # Extract and prepare data values for all valid timestamps
    # Handle both 1D and 2D data arrays appropriately
    data = series.data[indices]  # Get data for valid timestamps
    
    # Ensure data is 2D even for 1D time series (reshape single column data)
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)  # Convert to Nx1 array for consistent processing
    
    # Determine column names for data values
    num_columns = data.shape[1]  # Number of data dimensions (e.g., 5 for OHLCV)
    cols = column_names or [f'data_{i}' for i in range(num_columns)]  # Use provided names or defaults
    
    # Validate that provided column names match the data dimensions
    if len(cols) != num_columns:
        raise ValueError(f"Expected {num_columns} columns, got {len(cols)} names")
    
    # Add each data column to the DataFrame
    for i, col in enumerate(cols):
        df[col] = data[:, i]  # Add each dimension as a separate column

    # Create the database table if it doesn't exist
    # Define schema with timestamp as primary key and data columns as floats
    columns = ['timestamp BIGINT PRIMARY KEY'] + [f'"{col}" FLOAT' for col in cols]
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            {', '.join(columns)}
        )
    """)  # SQL to create table with appropriate schema

    # Insert data into the database with appropriate update strategy
    if replace:
        # Replace strategy: Delete existing rows with matching timestamps before inserting
        conn.execute(f"DELETE FROM {table_name} WHERE timestamp IN (SELECT timestamp FROM df)")
    
    # Insert all data rows, ignoring duplicates if replace=False
    # The OR IGNORE clause prevents primary key conflicts when append mode is used
    conn.execute(f"INSERT OR IGNORE INTO {table_name} SELECT * FROM df")