from uuid import uuid4
from typing import Dict, List, Optional, Union
import json

from ccxt import exchanges as ccxt_exchanges_list
from core.data.acquisition.CcxtDataSource import CCXT_adapter

from core.data.models.time_series import *
from core.data.streaming.series_manager import *

from core.utils.dates_conversion import *

from core.utils.table_names import *

import time

import mpi4py

import duckdb
from core.data.persistence.db_save import save_series
from core.data.persistence.db_load import load_series

# Module for managing time series data registry
# This module provides a central registry for time series data with functionality for
# creating, registering, loading, saving, and managing time series with their dependencies

class DataRegistry:
    """
    Central registry for managing time series data with metadata and dependencies.
    
    This class provides a comprehensive system for managing time series data, including:
    - Registration and lookup by UUID or metadata
    - Creation of built series with dependencies
    - Creation of third-party series from external sources
    - Loading and saving series from/to DuckDB database
    - Managing series dependencies through SeriesManager
    - Subscription to CCXT data sources
    
    The registry maintains two primary indexes:
    1. UUID-based index (_registry): Maps UUIDs to TimeSerie objects
    2. Metadata-based index (_metadata_index): Maps (source, symbol, type, duration) tuples to UUIDs
    
    Attributes
    ----------
    _registry : Dict[str, TimeSerie]
        Dictionary mapping UUIDs to TimeSerie objects
    _metadata_index : Dict[tuple, str]
        Dictionary mapping metadata tuples to UUIDs
    series_manager : SeriesManager
        Manager for handling dependencies between time series
    rank : int
        MPI rank for parallel processing
    comm : mpi4py.MPI.Intracomm
        MPI communicator for parallel processing
    """
    def __init__(self, comm: mpi4py.MPI.Intracomm, rank: int, timestamp=None):
        """
        Initialize a new DataRegistry.
        
        Parameters
        ----------
        comm : mpi4py.MPI.Intracomm
            MPI communicator for parallel processing
        rank : int
            MPI rank for parallel processing
        timestamp : int, optional
            Initial timestamp in milliseconds. If None, defaults to 0 in SeriesManager
            
        Notes
        -----
        - Creates empty registry and metadata index
        - Initializes a SeriesManager for dependency management
        - Stores MPI rank and communicator for parallel operations
        """
        # Track all series by UUID and metadata
        self._registry: Dict[str, TimeSerie] = {}
        self._metadata_index: Dict[tuple, str] = {}  # (source, symbol, type, duration) -> UUID
        
        # Initialize the series manager with the provided timestamp
        self.series_manager = SeriesManager(timestamp=timestamp)

        # Store MPI information for parallel processing
        self.rank = rank
        self.comm = comm

    def register(
        self,
        series: TimeSerie,
    ) -> str:
        """
        Register a time series in the registry and dependency manager.
        
        This method adds a time series to the registry, indexes it by metadata,
        and registers it with the series manager for dependency tracking.
        
        Parameters
        ----------
        series : TimeSerie
            The time series to register
            
        Returns
        -------
        str
            UUID of the registered series
            
        Raises
        ------
        ValueError
            If a series with the same metadata is already registered
            
        Notes
        -----
        - Generates a unique UUID for the series
        - Extracts metadata from the series for indexing
        - Registers the series with the dependency manager
        """
        # Generate unique ID
        serie_id = str(uuid4())
        self._registry[serie_id] = series

        # Extract metadata from the series
        source, pair, serie_type, duration = series.get_metadata()
        
        # Index by metadata (e.g., ("binance", "BTC/USDT", "OHLCV"))
        if all([source, pair, serie_type, duration]):
            metadata_key = (source, pair, serie_type, duration)
            if metadata_key in self._metadata_index:
                raise ValueError(f"Series {metadata_key} already registered")
            self._metadata_index[metadata_key] = serie_id
        
        # Update series manager
        self.series_manager.register(series)
        
        return serie_id

    @staticmethod
    def save_series_metadata(exchanges_dict: Dict[str, List[Dict]], filepath: str):
        """
        Save third party series metadata to JSON file
        
        This static method writes a dictionary of series metadata to a JSON file
        with pretty formatting.
        
        Parameters
        ----------
        exchanges_dict : Dict[str, List[Dict]]
            Dictionary mapping exchange names to lists of series metadata
        filepath : str
            Path to save the JSON metadata file
            
        Example format:
        {
            "binance": [
                {"symbol": "BTC/USDT", "serie_type": "OHLCV", "duration": 60000, "cache": 1000},
                {"symbol": "ETH/USDT", "serie_type": "fundingRate", "duration": 3600000}
            ],
            "hyperliquid": [
                {"symbol": "BTC/USD", "serie_type": "OHLCV", "duration": 60000}
            ]
        }
        
        Notes
        -----
        - Uses indent=4 for pretty formatting of the JSON file
        - Overwrites the file if it already exists
        """
        with open(filepath, 'w') as f:
            json.dump(exchanges_dict, f, indent=4)

    @staticmethod
    def load_series_metadata(filepath: str) -> Dict[str, List[Dict]]:
        """Load exchanges metadata from JSON file
        
        This static method reads a JSON file containing series metadata and
        returns it as a dictionary after validating its format.
        
        Parameters
        ----------
        filepath : str
            Path to the JSON metadata file
            
        Returns
        -------
        Dict[str, List[Dict]]
            Dictionary mapping exchange names to lists of series metadata
            
        Raises
        ------
        FileNotFoundError
            If the specified file does not exist
        ValueError
            If the file contains invalid JSON or has incorrect format
            
        Notes
        -----
        - Validates that each exchange maps to a list
        - Validates that each series has the required keys: symbol, serie_type, duration
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                # Validate format
                for exchange, series_list in data.items():
                    if not isinstance(series_list, list):
                        raise ValueError(f"Invalid format for exchange {exchange}: expected list")
                    for series in series_list:
                        required = {"symbol", "serie_type", "duration"}
                        if not all(key in series for key in required):
                            raise ValueError(f"Missing required keys in series metadata: {required}")
                return data
        except FileNotFoundError:
            raise FileNotFoundError(f"Metadata file not found: {filepath}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in file: {filepath}")
    
    def create_built_series(
        self,
        duration: Union[int, str],
        transition_function,
        dependencies: List[Union[str, tuple]],
        data_shape: int,
        cache: int,
        symbol: str,
        serie_type: str
    ) -> str:
        """
        Create and register a new BuiltSerie with dependencies.
        
        A BuiltSerie is a time series that is derived from other time series
        using a transition function. This method creates such a series,
        resolves its dependencies, and registers it in the registry.
        
        Parameters
        ----------
        duration : Union[int, str]
            Time interval between data points in milliseconds or as a string timeframe
        transition_function : callable
            Function that transforms dependency data into this series' data
        dependencies : List[Union[str, tuple]]
            List of dependency identifiers (UUIDs or metadata tuples)
        data_shape : int
            Number of data columns in the series
        cache : int
            Number of data points to cache in memory
        symbol : str
            Symbol/pair identifier (e.g., "BTC/USDT")
        serie_type : str
            Type of the series (e.g., "OHLCV", "fundingRate")
            
        Returns
        -------
        str
            UUID of the registered series
            
        Notes
        -----
        - Converts string timeframes to milliseconds
        - Resolves dependencies to actual TimeSerie objects
        - Creates a new BuiltSerie and registers it
        """
        # Convert string timeframe to milliseconds if needed
        if isinstance(duration, str):
            duration = timeframe_to_ms(duration)
        
        # Resolve dependencies to TimeSerie objects
        dep_series = [self.get_series(dep) for dep in dependencies]

        # Create the new BuiltSerie
        series = BuiltSerie.new(
            self.rank, 
            self.comm, 
            data_shape, 
            duration, 
            int(cache), 
            transition_function, 
            dep_series, 
            symbol, 
            serie_type
        )
        
        # Register and return the UUID
        return self.register(series)
    
    def create_third_party_series(
            self,
            source, 
            symbol, 
            serie_type, 
            duration: Union[int, str],
            cache: int = 1000,
            data_shape = None
    ) -> str:
        """
        Create and register a new ThirdPartySerie.
        
        A ThirdPartySerie is a time series that receives data from external sources
        like exchanges. This method creates such a series and registers it in the registry.
        
        Parameters
        ----------
        source : str
            Data source identifier (e.g., "binance", "hyperliquid")
        symbol : str
            Symbol/pair identifier (e.g., "BTC/USDT")
        serie_type : str
            Type of the series (e.g., "OHLCV", "fundingRate")
        duration : Union[int, str]
            Time interval between data points in milliseconds or as a string timeframe
        cache : int, optional
            Number of data points to cache in memory, default is 1000
        data_shape : int, optional
            Number of data columns in the series, if None, inferred from serie_type
            
        Returns
        -------
        str
            UUID of the registered series
            
        Raises
        ------
        Exception
            If data_shape cannot be inferred from serie_type
            
        Notes
        -----
        - Converts string timeframe to milliseconds
        - Infers data_shape from serie_type if not provided
        - Creates a new ThirdPartySerie and registers it
        """
        # Convert string timeframe to milliseconds if needed
        if type(duration) == str:
            duration = timeframe_to_ms(duration)
        
        # Infer data_shape from serie_type if not provided
        if not data_shape:
            try:
                correspondances = {
                    "fundingRate" : 1,
                    "tmpfundingRate" : 1,
                    "OHLCV" : 5,  # Open, High, Low, Close, Volume
                }
                data_shape = correspondances[serie_type]
            except KeyError:
                # The serie_type is not registered in the correspondances dictionary
                raise ValueError(f"Cannot infer data_shape for serie_type '{serie_type}'. Please provide data_shape explicitly.")
        
        # Create the new ThirdPartySerie
        series = ThirdPartySerie.new(
            self.rank, 
            self.comm, 
            data_shape, 
            duration, 
            int(cache), 
            source, 
            symbol, 
            serie_type
        )

        # Register and return the UUID
        return self.register(series)

    def create_third_party_from_file(self, filepath: str) -> List[str]:
        """
        Create third party series from a JSON metadata file.
        
        This method reads series metadata from a JSON file and creates
        corresponding ThirdPartySerie objects in the registry.
        
        Parameters
        ----------
        filepath : str
            Path to the JSON metadata file
            
        Returns
        -------
        List[str]
            List of UUIDs for the created series
            
        Notes
        -----
        - The metadata file should contain a dictionary mapping exchanges to lists of series info
        - Each series info should include "symbol", "serie_type", "duration", and optionally "cache" and "data_shape"
        - Uses the load_series_metadata static method to read the file
        """
        # Load metadata from file
        exchanges_metadata = self.load_series_metadata(filepath)
        
        # Store created series UUIDs
        created_series = []
        
        # Create series for each exchange and its metadata
        for exchange, series_list in exchanges_metadata.items():
            for series_meta in series_list:
                # Extract required parameters
                symbol = series_meta["symbol"]
                serie_type = series_meta["serie_type"]
                duration = series_meta["duration"]
                
                # Extract optional parameters with defaults
                cache = series_meta.get("cache", 1000)
                data_shape = series_meta.get("data_shape", None)
                
                # Create the series
                series_id = self.create_third_party_series(
                    source=exchange,
                    symbol=symbol,
                    serie_type=serie_type,
                    duration=duration,
                    cache=cache,
                    data_shape=data_shape
                )
                
                created_series.append(series_id)
        
        return created_series

    def save_ccxt_subscriptions(self, file_path: str) -> None:
        """
        Save metadata for all CCXT-based time series to a JSON file.
        
        This method extracts metadata for all time series from CCXT exchanges
        and saves it to a JSON file in a format suitable for later recreation.
        
        Parameters
        ----------
        file_path : str
            Path to save the JSON metadata file
            
        Notes
        -----
        - Only includes series from CCXT exchanges
        - Only includes series of types "OHLCV", "fundingRate", or "tmpfundingRate"
        - Includes the current cache size for each series
        - Uses the save_series_metadata static method to write the file
        """
        # Initialize dictionary to store metadata by exchange
        dico = {}
        
        # Iterate through all registered series
        for metadata, uuid in self._metadata_index.items():
            source, pair, serie_type, duration = metadata
            
            # Filter for CCXT exchanges and supported series types
            if source in ccxt_exchanges_list and serie_type in ["OHLCV", "fundingRate", "tmpfundingRate"]:
                # Create metadata entry for this series
                tmp = { 
                    "symbol": pair,
                    "serie_type": serie_type,
                    "duration": duration,
                    "cache": self._registry[uuid].timestamps.shape[0],  # Current cache size
                }
                
                # Add to the dictionary, creating a new list if needed
                if source not in dico.keys():
                    dico[source] = [tmp]
                else:
                    dico[source].append(tmp)
        
        # Save the collected metadata to file
        self.save_series_metadata(dico, file_path)

    def subscribe_to_ccxt_adapter(self, start_time=None, conn=None, verbose=False) -> CCXT_adapter:
        """
        Create a CCXT adapter and subscribe all relevant series to it.
        
        This method creates a new CCXT adapter and subscribes all registered series
        from CCXT exchanges to it for data acquisition.
        
        Parameters
        ----------
        start_time : int, optional
            Start time for data acquisition in milliseconds
        conn : duckdb.DuckDBPyConnection, optional
            DuckDB connection for database operations
        verbose : bool, optional
            Whether to enable verbose logging, default is False
            
        Returns
        -------
        CCXT_adapter
            The created and configured CCXT adapter
            
        Notes
        -----
        - Only subscribes series from CCXT exchanges
        - Only subscribes series of types "OHLCV", "fundingRate", or "tmpfundingRate"
        """
        # Create a new CCXT adapter
        ccxt_adapter = CCXT_adapter(start_time=start_time, conn=conn, verbose=verbose)
        
        # Subscribe all relevant series to the adapter
        for metadata, uuid in self._metadata_index.items():
            source, pair, serie_type, duration = metadata
            
            # Filter for CCXT exchanges and supported series types
            if source in ccxt_exchanges_list and serie_type in ["OHLCV", "fundingRate", "tmpfundingRate"]:
                series = self._registry[uuid]
                ccxt_adapter.subscribe(series)
                
        return ccxt_adapter
    
    def get_series(
        self, 
        identifier: Union[str, tuple]  # UUID or metadata tuple
    ) -> TimeSerie:
        """Retrieve series by UUID or metadata.
        
        This method retrieves a time series from the registry using either its UUID
        or a metadata tuple (source, symbol, type, duration).
        
        Parameters
        ----------
        identifier : Union[str, tuple]
            Either a UUID string or a metadata tuple (source, symbol, type, duration)
            
        Returns
        -------
        TimeSerie
            The requested time series
            
        Raises
        ------
        KeyError
            If no series is found with the given identifier
            
        Notes
        -----
        - String identifiers are treated as UUIDs for direct lookup
        - Tuple identifiers are treated as metadata for indexed lookup
        """
        if isinstance(identifier, tuple):
            if identifier not in self._metadata_index:
                raise KeyError(f"No series found with metadata {identifier}")
            identifier = self._metadata_index[identifier]
        
        if identifier not in self._registry:
            raise KeyError(f"No series found with UUID {identifier}")
            
        return self._registry[identifier]

    def find_series(
        self, 
        source: Optional[str] = None,
        pair: Optional[str] = None,
        type: Optional[str] = None,
        duration: Optional[Union[int, str]] = None,
    ) -> List[TimeSerie]:
        """
        Query series by metadata attributes.
        
        This method searches for time series matching the specified metadata criteria.
        Any parameter set to None acts as a wildcard that matches any value.
        
        Parameters
        ----------
        source : str, optional
            Data source identifier (e.g., "binance")
        pair : str, optional
            Symbol/pair identifier (e.g., "BTC/USDT")
        type : str, optional
            Type of the series (e.g., "OHLCV", "fundingRate")
        duration : Union[int, str], optional
            Time interval between data points in milliseconds or as a string timeframe
            
        Returns
        -------
        List[TimeSerie]
            List of time series matching the criteria
            
        Notes
        -----
        - Converts string timeframe to milliseconds if needed
        - Parameters set to None act as wildcards
        - Returns an empty list if no matches are found
        """
        # Convert string timeframe to milliseconds if needed
        if isinstance(duration, str):
            duration = timeframe_to_ms(duration)

        # Search for matching series
        results = []
        for (s, p, t, d), serie_id in self._metadata_index.items():
            # Check if all specified criteria match
            if (source is None or s == source) and \
               (pair is None or p == pair) and \
               (type is None or t == type) and \
               (duration is None or d == duration):
                results.append(self._registry[serie_id])
                
        return results

    def save_to_db(self, identifier: Union[str, tuple], conn: duckdb.DuckDBPyConnection, 
                   column_names: list = None, replace: bool = False) -> None:
        """
        Save a registered time series to DuckDB database.
        
        This method saves a single time series identified by its UUID or metadata tuple
        to the database using the save_series function.
        
        Parameters
        ----------
        identifier : Union[str, tuple]
            Either a UUID string or a metadata tuple (source, symbol, type, duration)
        conn : duckdb.DuckDBPyConnection
            DuckDB connection for database operations
        column_names : list, optional
            Optional list of column names for data columns
        replace : bool, optional
            Whether to replace existing data if the table already exists, default is False
            
        Raises
        ------
        KeyError
            If no series is found with the given identifier
            
        Notes
        -----
        - Uses the get_series method to retrieve the series
        - Uses the save_series function from core.data.persistence.db_save
        - If replace=False and the table exists, will append data instead of replacing
        """
        series = self.get_series(identifier)
        save_series(series, conn, column_names, replace)

    def save_all_to_db(self, conn: duckdb.DuckDBPyConnection, replace: bool = False) -> None:
        """
        Save all registered time series to the DuckDB database.
        
        This method iterates through all registered time series and saves each one
        to the database using the save_to_db method.
        
        Parameters
        ----------
        conn : duckdb.DuckDBPyConnection
            DuckDB connection for database operations
        replace : bool, optional
            Whether to replace existing data if tables already exist, default is False
            
        Notes
        -----
        - Calls save_to_db for each series in the registry
        - If any individual save fails, the others will still be attempted
        """
        for serie_id in self._registry:
            self.save_to_db(serie_id, conn, replace=replace)

    def load_series_from_db(self, 
                            source: str, 
                            symbol: str, 
                            serie_type: str, 
                            duration: Union[int, str],
                            conn: duckdb.DuckDBPyConnection, 
                            cache: int = 1000,
                            start: int = None, 
                            end: int = None,
                            limit: int = None) -> str:
        """
        Load a time series from DuckDB database and register it.
        
        This method loads time series data from a DuckDB database, creates a new
        ThirdPartySerie with that data, and registers it in the registry.
        
        Parameters
        ----------
        source : str
            Data source identifier (e.g., "binance")
        symbol : str
            Symbol/pair identifier (e.g., "BTC/USDT")
        serie_type : str
            Type of the series (e.g., "OHLCV", "fundingRate")
        duration : Union[int, str]
            Time interval between data points in milliseconds or as a string timeframe
        conn : duckdb.DuckDBPyConnection
            DuckDB connection for database operations
        cache : int, optional
            Number of data points to cache in memory, default is 1000
        start : int, optional
            Start timestamp for data loading (inclusive)
        end : int, optional
            End timestamp for data loading (inclusive)
        limit : int, optional
            Maximum number of data points to load
            
        Returns
        -------
        str
            UUID of the registered series
            
        Raises
        ------
        ValueError
            If the source is not supported for database loading
            
        Notes
        -----
        - Converts string timeframe to milliseconds if needed
        - Currently only supports CCXT exchange sources
        - Creates a new ThirdPartySerie and populates it with loaded data
        """
        # Convert string timeframe to milliseconds if needed
        if isinstance(duration, str):
            duration = timeframe_to_ms(duration)
        
        # Create metadata tuple for lookup
        metadata = (source, symbol, serie_type, duration)
        
        # Check if source is supported
        if source not in ccxt_exchanges_list:
            raise ValueError(f"Source {source} not supported yet for loading from database.")
            
        # Load the series data from the database
        loaded_series = load_series(
            generate_table_name(source, symbol, serie_type, duration),
            conn, start, end, limit
        )

        # Create a new third-party series
        self.create_third_party_series(
            source, 
            symbol, 
            serie_type, 
            duration,
            cache=cache,
        )

        # Get the newly created series
        serie_id = self._metadata_index[metadata]
        existing_series = self._registry[serie_id]

        # Calculate how many data points were loaded
        loaded_cache = loaded_series.timestamps.shape[0]

        # Copy the loaded data into the new series
        existing_series.timestamps[(-loaded_cache):] = loaded_series.timestamps.reshape(-1, 1)[:]
        existing_series.data[(-loaded_cache):] = loaded_series.data[:]

        return serie_id
    
    def load_all_from_db(self, conn: duckdb.DuckDBPyConnection, start: int = None, 
                     end: int = None, limit: int = None) -> tuple:
        """Try to load all series present in the registry from the database.
        
        This method attempts to load data for all registered series from the database
        and update them with the loaded data. It tracks and reports successes and failures.
        
        Parameters
        ----------
        conn : duckdb.DuckDBPyConnection
            DuckDB connection for database operations
        start : int, optional
            Start timestamp for data loading (inclusive)
        end : int, optional
            End timestamp for data loading (inclusive)
        limit : int, optional
            Maximum number of data points to load
            
        Returns
        -------
        tuple
            A tuple containing (success_list, not_found_list) where:
            - success_list: List of metadata tuples for successfully loaded series
            - not_found_list: List of metadata tuples for series not found in the database
            
        Notes
        -----
        - Uses the generate_table_name function to create table names from metadata
        - Uses the load_series function from core.data.persistence.db_load
        - Updates existing series in place with loaded data
        - Prints lists of successfully loaded series and series not found in the database
        - Continues loading remaining series even if some fail
        """
        success_list = []
        not_found_list = []
        
        # Try to load each series by its metadata
        for metadata in self._metadata_index.keys():
            source, symbol, serie_type, duration = metadata
            table_name = generate_table_name(source, symbol, serie_type, duration)

            serie_id = self._metadata_index[metadata]
            existing_series = self._registry[serie_id]
            cache = existing_series.timestamps.shape[0]
            
            try:
                loaded_series = load_series(table_name, conn, start, end, limit=cache)
                # Update existing series with loaded data
                loaded_cache = loaded_series.timestamps.shape[0]

                existing_series.timestamps[(-loaded_cache):] = loaded_series.timestamps.reshape(-1,1)[:]
                existing_series.data[(-loaded_cache):] = loaded_series.data[:]

                success_list.append(metadata)
            except (ValueError, duckdb.CatalogException):
                not_found_list.append(metadata)
        
        # Print results
        if success_list:
            print("\nSuccessfully loaded series:")
            for metadata in success_list:
                source, symbol, serie_type, duration = metadata
                print(f"- {source} {symbol} {serie_type} {duration}")
        
        if not_found_list:
            print("\nSeries not found in database:")
            for metadata in not_found_list:
                source, symbol, serie_type, duration = metadata
                print(f"- {source} {symbol} {serie_type} {duration}")
                
        return (success_list, not_found_list)