import numpy as np
from mpi4py import MPI
from mpi4py.util import dtlib

# may be modified to take as argument the shape of the array we want to store
# this would allow us to use the same function for time series of tensors
# one should keep the case where we provide an int (like n in the current
# version)
# if cache = 1, shouldn't we choose shape (n) instead of (1,n) or is it 
# more usefull to keep things like that to future processing ?
def reserve_shared_memory(n, rank, comm, cache = 1, init = np.nan) -> np.ndarray:
    """
    Reserves a shared memory space for inter-process communication using MPI.

    Parameters
    ----------
    n : int
        The size of the array to be stored (number of elements).
    rank : int
        The rank of the current MPI process.
    comm : MPI.Comm
        The MPI communicator.
    cache : int, optional
        The number of rows/cache lines for the shared array (default is 1).
    init : float, optional
        The initial value to fill the shared array with (default is np.nan).

    Returns
    -------
    np.ndarray
        A NumPy array backed by the shared memory space.
    """
    # Define the MPI datatype and corresponding NumPy dtype
    datatype = MPI.DOUBLE # MPI.FLOAT is float32 (sufficient)
    np_dtype = dtlib.to_numpy_dtype(datatype)
    itemsize = datatype.Get_size()

    # Reserve a memory space shared between the processes
    # Only rank 0 allocates the memory, others attach to it
    win_size = n * cache * itemsize if rank == 0 else 0
    win = MPI.Win.Allocate_shared(win_size, itemsize, comm=comm)

    # Query the shared memory buffer and assert its item size
    buf, itemsize = win.Shared_query(0)
    assert itemsize == MPI.DOUBLE.Get_size() # MPI.FLOAT.Get_size()
    # Create a NumPy array from the shared buffer
    buf = np.array(buf, dtype='B', copy=False)

    # Reshape the buffer into the desired array shape
    data = np.ndarray(buffer=buf, dtype=np_dtype, shape=(cache,n))

    # Set all values in the shared array to the desired initial value
    data[:] = init

    return data