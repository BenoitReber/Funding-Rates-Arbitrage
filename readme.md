# Funding Rates Arbitrage Project

The aim of this porject is to develop an arbitrage system for cryptocurrency perpetual futures funding rates across multiple crypto-exchanges, leveraging real-time data acquisition/processing and multi-process capabilities. The idea is to identify and capitalize on fleeting arbitrage opportunities when spot price and perpetual futures price diverge.

**Project Status**: This project is temporarily paused while I focus on other priorities.

## State of the Project

### Accomplishments:
- **Data Acquisition**: Real-time data collection from multiple exchanges using CCXT.
- **Local Computation of Time Series**: Processing and transformation of raw market data into actionable time series.
- **Persistence**: Storage of time series data using DuckDB for querying.
- **Multi-process Capabilities**: Utilizing MPI4Py and shared memory for parallel data handling and computation.

### Remaining Work:
- **Execution of Strategies**: Implement automated trading strategies for identified arbitrage opportunities.
- **Backtesting Module**: Create a comprehensive backtesting framework to evaluate strategy performance on historical data.
- **Risk Management**: Integrate robust risk management protocols to control exposure and potential losses.
- **Alerting System**: Develop a real-time alerting system for critical events and opportunities.
- **User Interface**: Develop an intuitive user interface for monitoring, configuration, and strategy management.

## Project Structure
```
core/
├── data/
│   ├── acquisition/        # Real-time data collection from exchanges
│   │   ├── CcxtDataSource.py   # CCXT-based exchange data adapter with async support
│   │   └── simulated.py        # Simulated data source for testing
│   ├── models/             # Core data structures and time series models
│   │   └── time_series.py      # TimeSerie, BuiltSerie, ThirdPartySerie classes
│   ├── persistence/        # Database operations for data storage/retrieval
│   │   ├── db_load.py          # Load time series from DuckDB with filtering
│   │   └── db_save.py          # Save time series to DuckDB with update strategies
│   ├── processing/         # Data transformation and technical analysis
│   │   └── transforms.py       # TA-Lib integration and OHLCV resampling
│   └── streaming/          # Real-time data management and registry
│       ├── DataRegistry.py     # Central registry for time series management
│       └── series_manager.py   # Dependency management between series
│
├── ui              # not yet available
│
├── trading_engine  # not yet available
│
└── utils/                  # Utility functions and helpers
    ├── dates_conversion.py     # Time format conversions (ms ↔ timeframes)
    ├── reserve_shared_memory.py# MPI shared memory allocation
    └── table_names.py          # Database table naming conventions
```

## Technologies and Packages

### Core Dependencies

**CCXT (`ccxt.pro`)**
- **Purpose**: Real-time cryptocurrency exchange data acquisition
- **Usage**: Fetches OHLCV data, funding rates, and other market data from multiple exchanges
- **Features**: Asynchronous WebSocket connections, multi-exchange support

**DuckDB (`duckdb`)**
- **Purpose**: Database for time series storage
- **Usage**: Persisting and querying large volumes of market data
- **Features**: Fast analytical queries, efficient storage, SQL interface

**MPI4Py (`mpi4py`)**
- **Purpose**: Multi-process parallel computing and shared memory management
- **Usage**: Inter-process communication, shared memory arrays, synchronization
- **Features**: Distributed computing, process coordination, memory sharing

**TA-Lib (`talib`)**
- **Purpose**: Technical analysis indicators and financial functions
- **Usage**: Computing technical indicators (SMA, RSI, MACD, etc.) on OHLCV data
- **Features**: 150+ technical indicators, optimized C implementations

**Python Standard Library**
- **asyncio**: Asynchronous programming for real-time data collection
- **threading**: Multi-threading support for concurrent operations
- **time/datetime**: Time handling and conversions
- **json**: Configuration and metadata serialization