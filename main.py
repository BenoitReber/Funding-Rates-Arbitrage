# launch using: mpiexec -n 2 python main.py
# not working anymore: mpiexec -n 2 python -m mpi4py.futures main.py

import numpy as np
import pandas as pd

import duckdb

from mpi4py import MPI

from core.data.streaming.DataRegistry import *


#### parameters ####


if __name__ == '__main__':

    ###############################################################
    ##########           Common initialization           ##########
    ###############################################################

    ### MPI Initialization ###
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    ### DataRegistry Initialization ###
    registry = DataRegistry(comm, rank)

    ### Initialize time series acquired from third parties ###
    # open the subscription file and reserve a shared memory space with the appropriate size
    # depending on the desired sliding frame size
    registry.create_third_party_from_file("sub.json")

    ### Initialize time series computed locally from other data ###
    def ma_transition(deps, prev_data):
        ohlcv = deps[0]
        return np.mean(ohlcv.data[-2:], axis=0)

    ma_id = registry.create_built_series(
        duration = 60000,
        transition_function=ma_transition,
        dependencies=[("binance", "BTC/USDT", "OHLCV",60000)],  # Reference by metadata
        data_shape=5,  # Output shape
        cache = 2,
        symbol = "MA2BTC/USDT",
        serie_type = "OHLCV"
    )
    ma_serie = registry.get_series(ma_id)

    def max_transition(deps, prev_data):
        tmp = np.array( [ serie_data[-1] for serie_data in deps ] )
        return np.argmax(tmp, axis=0)

    max_id = registry.create_built_series(
        duration = 60000,
        transition_function = max_transition,
        dependencies=[  ("binance", "BTC/USDT", "OHLCV",60000),
                        ('local', 'MA2BTC/USDT', 'OHLCV', 60000)],  # Reference by metadata
        data_shape=5,  # Output shape
        cache = 2,
        symbol = "MAX2BTC/USDT",
        serie_type = "max"
    )
    max_serie = registry.get_series(max_id)



    ###############################################################
    ##########             Different processes           ##########
    ###############################################################

    # process in charge of:
    #   - loading data from the database when the program starts
    #   - acquiring OHLCV data in real time and putting it in shared memory
    #   - updating the persistence database for aquired ThirdPartySerie
    if rank == 0:
        print(60*"#")

        comm.barrier()

        print("Starting data acquisition process")

        ### Database Initialization ###
        conn = duckdb.connect('test.db')

        # load data from the database if available
        registry.load_all_from_db(conn)

        ccxt_feeder = registry.subscribe_to_ccxt_adapter( start_time= int( time.time() * 1000 / (8*60000*60) - 3) * (8*60000*60) - 60 * 60000,
                                    conn = conn,
                                    verbose = False
                                )

        comm.Barrier()

        print(60*"#")

        ccxt_feeder.run()

        import time
        while True:
            time.sleep(0.5)

        comm.Barrier()

    # process in charge of the computation of BuiltSerie objects
    # note that the computed time series are not saved in the database
    elif rank == 1:
        print(60*"#")

        comm.barrier()

        print("Starting BuiltSeries computation process")

        # wait for the data acquisition process to finish its initialization
        comm.Barrier()

        print(60*"#")

        import time
        while True:

            time.sleep(0.5)
            registry.series_manager.update_all(verbose=False)

        comm.Barrier()

    # process in charge of the UI
    elif rank == 2:

        # wait for the other processes to finish their initialization
        comm.Barrier()

        pass

    # process in charge of making trades
    elif rank == 2:

        # wait for the other processes to finish their initialization
        comm.Barrier()

        pass


comm.Disconnect()