import time

import asyncio
import datetime

import ccxt.pro as ccxt

from core.data.streaming.DataRegistry import *
from core.data.models.time_series import *
from core.utils.dates_conversion import *
from core.data.persistence.db_save import *

import threading

#from ccxt import ccxt.base.errors.BadRequest as BadRequest

from ccxt.base.errors import BadRequest as BadRequest

class CCXT_adapter():
    """
    Adapter for the CCXT library to collect cryptocurrency market data.
    
    This class provides an interface to fetch various types of market data (OHLCV, funding rates)
    from cryptocurrency exchanges using the CCXT library. It manages asynchronous data collection,
    handles multiple time series with different durations, and provides mechanisms for data
    synchronization between coroutines and processes.
    
    The adapter follows a publisher-subscriber pattern where time series can subscribe
    to receive updates for specific exchange/symbol/data type combinations.
    """
    def __init__(self, start_time=None, conn=None, verbose=False):
        """
        Initialize the CCXT adapter with specified parameters.
        
        Parameters
        ----------
        start_time : int, optional
            Starting timestamp in milliseconds for data collection.
            If None, defaults to -1 (collect from earliest available data).
        conn : object, optional
            Database connection object for data persistence.
        verbose : bool, default=False
            Whether to print detailed logs during operation.
        """
        self.exchanges = {} # "binance" -> ccxt exchange object
        self._by_duration = {} # timeframe(ms) -> list of ThirdPartySerie objects
        self.tmp = {} # timeframe(ms) -> cache np.array
        self.timestamps = {} # timeframe(ms) -> list[int] last timestamp acquired
        self.conditions = {} # timeframe(ms) -> asyncio condition
        self.semaphores = {} # timeframe(ms) ->  mpi4py semaphore

        self.conn = conn

        self.verbose = verbose

        self.start_time = start_time if start_time else -1

    def subscribe(self, series: ThirdPartySerie):
        """
        Subscribe a time series to receive data updates from a specific exchange/symbol.
        
        This method registers a ThirdPartySerie object to receive updates for its
        associated exchange, symbol, and data type. It initializes necessary data structures
        for tracking timestamps, temporary storage, and synchronization mechanisms.
        
        Parameters
        ----------
        series : ThirdPartySerie
            The time series object to subscribe for data updates.
        """
        exchange, symbol, serie_type, duration= series.get_metadata()
        # create the exchange object if necessary
        if exchange not in self.exchanges.keys():
            self.exchanges[exchange] = getattr(ccxt, exchange)()

        if duration not in self._by_duration.keys():
            self._by_duration[duration] = [series]
        else:
            self._by_duration[duration].append(series)

        new_tmp = np.zeros(shape = series.data.shape[1:]) - 2**30
        if duration not in self.tmp.keys():
            self.tmp[duration] = [ new_tmp ]
        else:
            self.tmp[duration].append(new_tmp)
        
        tmp_start_time = int( self.start_time / (duration) ) * (duration)
        last_timestamp = series.timestamps[-1] if (len(series.timestamps) > 0 and series.timestamps[-1] >= -1) else -1
        oldest_timestamp = series.timestamps[0] if (len(series.timestamps) > 0 and series.timestamps[-1] >= -1) else -1
        #if oldest_timestamp < tmp_start_time:
        #    adapted_start_time = tmp_start_time
        if last_timestamp != -1:
            adapted_start_time = last_timestamp + duration
        elif last_timestamp == -1:
            adapted_start_time = tmp_start_time
        #else:
        #    adapted_start_time = last_timestamp
        if duration not in self.timestamps.keys():
            self.timestamps[duration] = [ adapted_start_time ]
        else:
            self.timestamps[duration].append( adapted_start_time )

        if duration not in self.semaphores.keys():
            self.semaphores[duration] = [series.semaphore]
        else:
            self.semaphores[duration].append( series.semaphore )
        
        if duration not in self.conditions.keys():
            self.conditions[duration] = asyncio.Condition()

    def subscribe_for_each(self, list_series:list):
        """
        Subscribe multiple time series at once.
        
        Parameters
        ----------
        list_series : list
            List of ThirdPartySerie objects to subscribe.
            
        Returns
        -------
        list
            List of subscription results.
        """
        tmp = [ self.subscribe(series) for series in list_series]
        return tmp
    
    async def collector(self, exchange, symbol, serie_type, duration, i):
        """
        Asynchronous collector for a specific data series.
        
        This coroutine continuously collects data for a specific exchange/symbol/type
        combination at regular intervals defined by the duration parameter. It handles
        different data types (OHLCV, fundingRate, tmpfundingRate) with appropriate
        timing and processing logic.
        
        Parameters
        ----------
        exchange : ccxt.Exchange
            CCXT exchange object to fetch data from.
        symbol : str
            Trading pair symbol (e.g., 'BTC/USDT').
        serie_type : str
            Type of data to collect ('OHLCV', 'fundingRate', or 'tmpfundingRate').
        duration : int
            Time interval in milliseconds between data points.
        i : int
            Index of this series in the duration group arrays.
        """
        #duration_in_seconds = exchange.parse_timeframe(timeframe)
        #duration = duration_in_seconds * 1000
        timeframe = ms_to_timeframe(duration)
        condition = self.conditions[duration]
        tmp = self.tmp[duration]
        timestamps = self.timestamps[duration]

        print( "Start collecting : ", exchange, symbol, serie_type, timeframe)

        while True:
            async with condition:
                ##try:
                # if we already have the data there is nothing to do
                if np.all(np.array(tmp[i]) > -2**30):
                    condition.notify_all() # I'm not sure that this line is usefull
                    continue

                if serie_type == "OHLCV":
                    # OHLCV with timestamp t contains data for the interval [t,t+duration)
                    desired = int(timestamps[i]) if int(timestamps[i]) > 0 else ( int(time.time() * 1000 / duration - 1) * duration )
                elif serie_type == "fundingRate":
                    # fundingRate with timestamp t concerns transactions made at time t (end of a funding interval)
                    desired = int(timestamps[i]) if int(timestamps[i]) > 0 else ( int(time.time() * 1000 / duration) * duration )
                elif serie_type == "tmpfundingRate":
                    # the funding rate for the funding period if it was closing now
                    #desired = int(timestamps[i]) if ( abs( int(timestamps[i]) - int(time.time() * 1000) ) < 50000 ) else ( int(time.time() * 1000 / duration) * duration )
                    desired = int(timestamps[i]) if ( abs( int(timestamps[i]) - int(time.time() * 1000) ) < duration ) else ( int(time.time() * 1000 / duration) * duration )
                else:
                    print("pb cible inconnue")

                if self.verbose:
                    print(f"Wait to download {datetime.datetime.fromtimestamp(desired/1000)} for {exchange}, {symbol}, {serie_type}, {timeframe}")

                if serie_type == "OHLCV":
                    await condition.wait_for( lambda : (time.time() * 1000 >= desired + duration) )
                elif serie_type == "fundingRate":
                    await condition.wait_for( lambda : (time.time() * 1000 >= desired) )
                elif serie_type == "tmpfundingRate":
                    await condition.wait_for( lambda : (time.time() * 1000 >= desired) )
                else:
                    print("pb cible inconnue")
                

                if serie_type == "OHLCV":
                    try:
                        new_data = await exchange.fetch_ohlcv(symbol, timeframe, since = desired, limit = 1)
                    except BadRequest:
                        print(f"Bad request for {exchange}, {symbol}, {serie_type}, {timeframe}")
                        new_data = [ [desired, 42, 42, 42, 42, 42] ]
                elif serie_type == "fundingRate":
                    try:
                        new_data = await exchange.fetchFundingRateHistory(symbol, since = desired, limit = 1)
                        new_data = [ [ (d["timestamp"]//duration)* (duration), d["fundingRate"]] for d in new_data ]
                    except BadRequest:
                        print(f"Bad request for {exchange}, {symbol}, {serie_type}, {timeframe}")
                        new_data = [ [desired, 42] ]
                elif serie_type == "tmpfundingRate":
                    new_data = await exchange.fetchFundingRates([symbol])
                    new_data = [ [ desired, new_data[symbol]["fundingRate"] ] ]
                    #if np.abs( int(time.time() * 1000) - desired )  > 5000:
                    #    new_data[0][1] = np.nan
                else:
                    print("pb cible inconnue") 
                
                if len(new_data) >= 2:
                    print( "problème trop de candles ", symbol, " :",len(new_data))
                    continue

                elif len(new_data) == 0:
                    continue
                    
                elif len(new_data) ==1:
                    #print(symbol, "new candle to process")
                    if self.verbose:
                        if serie_type == "tmpfundingRate":
                            print(datetime.datetime.fromtimestamp(int(time.time()/60)*60), exchange, symbol, serie_type,
                                datetime.datetime.fromtimestamp( int(new_data[-1][0]/1000) ) )
                        else:
                            print(datetime.datetime.fromtimestamp(int(time.time()/60)*60), exchange, symbol, serie_type,
                                datetime.datetime.fromtimestamp(int(new_data[-1][0]/1000 /60)*60) )
                        
                # check if we have the desired data and put it into the cache
                if new_data[-1][0] == desired:
                    tmp[i] = new_data[-1][1:]
                    timestamps[i] = new_data[-1][0]

                ##except Exception as e:
                #    print(f"{type(e).__name__}: {(str(e))}")
            
                #await asyncio.sleep(1)#1) # wait for at least one second before a new attempt

    async def data_updater(self, duration):
        """
        Asynchronous updater for time series of a specific duration.
        
        This coroutine waits for all collectors of a specific duration to gather their data,
        then updates all the corresponding time series and notifies waiting processes.
        It also handles persistence if a database connection is available.
        
        Parameters
        ----------
        duration : int
            Time interval in milliseconds for the group of time series to update.
        """
        # wait for all the new data to be collected in a temporary local variable
        # shared between coroutines but not between processes
        # then add the data to the shared memory 
        # we need to handle the case where we don't achieve to retrieve some data 
        # we could either add a time condition since last change (this would require all streams to be
        # of the same timeframe or it would be complicated) or we could in the collector process
        # check if some data is still waiting in the tmp
        # the idea if some data is missing would be to send the order to the trading engine to close 
        # all positions it opened
        semaphores = self.semaphores[duration]
        timestamps = self.timestamps[duration]
        condition = self.conditions[duration]
        tmp = self.tmp[duration]

        while True:
            while True:
                # wait unitl all the time series are acquired for the desired duration
                async with condition:
                    if np.all( np.array([ np.array(d).min() for d in tmp]) > -2**30):
                        break
                    await asyncio.sleep(0.5)
                    condition.notify_all()

            if True:#self.verbose:
                print(70*"_")
                print( datetime.datetime.fromtimestamp(timestamps[0]/1000), ": successful acquisition of duration ", duration)

            # update the shared memory space and reset the buffer of the feeder
            for i, series in enumerate( self._by_duration[duration] ):
                series.update( tmp[i], timestamps[i] )
                if self.verbose:
                    print(series.get_metadata())
                #print(f"Last values for {series.get_metadata()}")
                #print(f"Timestamps: \n{series.timestamps[-3:]}")
                #print(f"Values: \n{series.data[-3:]}")
                #print(tmp)
                #print(timestamps)

                next_desired = timestamps[i] + duration
                #print(next_desired)
                #tmp[i][:] = - 2**30
                tmp[i] =  np.array( [ - 2**30 for i in range(len(tmp[i])) ] )
                timestamps[i] = next_desired
            
            if self.conn is not None:
                self.persistence(self._by_duration[duration])

            if self.verbose:
                print(70*"_")

            # Notify the uopdate to the other processes
            for sema in semaphores:
                sema.release()
        #except Exception as e:
        #    print(f"{type(e).__name__}: {(str(e))}")
        return None

    def get_routines(self):
        """
        Generate all required asynchronous routines for data collection.
        
        This method creates collector coroutines for each subscribed time series
        and data_updater coroutines for each unique duration.
        
        Returns
        -------
        list
            List of coroutines (collector and data_updater) to be gathered.
        """
        collectors = []
        for duration in self._by_duration.keys():
            for i, series in enumerate( self._by_duration[duration] ):
                exchange, symbol, serie_type, duration = series.get_metadata()

                collectors.append( 
                    self.collector(
                        self.exchanges[exchange], 
                        symbol, 
                        serie_type, 
                        duration, 
                        i
                    )
                )

            collectors.append( self.data_updater(duration) )
        
        return collectors

    def persistence(self, list_series):
        """
        Save time series data to the database.
        
        Parameters
        ----------
        list_series : list
            List of ThirdPartySerie objects to save.
        """
        for series in list_series:
            save_series(series, self.conn, replace=False)
    
    def run(self):
        """
        Start the data collection process.
        
        This method synchronizes all timestamps to start from the oldest timestamp,
        creates all necessary coroutines, and launches them in a separate thread
        using asyncio.gather. The thread runs as a daemon to allow the main program
        to exit without waiting for it.
        """
        for dur in self.timestamps.keys():
            oldest_start = min( self.timestamps[dur] )
            for i in range( len(self.timestamps[dur]) ):
                self.timestamps[dur][i] = oldest_start

        routines = self.get_routines()

        async def feeder():
            await asyncio.gather(*routines)

            for ex in self.exchanges.keys():
                await self.exchanges[ex].close()

        feeder_thread = threading.Thread(target= (lambda : asyncio.run(feeder())), daemon=True)

        feeder_thread.start()