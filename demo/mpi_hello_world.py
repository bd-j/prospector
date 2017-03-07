# This is a short script to test the MPI implementation with the pattern used
# by Prospector.  However, try/except blocks are minimized to enable more
# useful error messages, and the code is simple enough to *only* test the MPI
# implementation.

# Invoke with:
# mpirun -np 4 python mpi_hello_world.py

import numpy as np
import sys
from mpi4py import MPI

def speak(i):
    print("I am core {} with task {}".format(pool.rank, i))
    return i, pool.rank

from emcee.utils import MPIPool
pool = MPIPool(debug=False, loadbalance=True)
if not pool.is_master():
    # Wait for instructions from the master process.
    pool.wait()
    sys.exit(0)


if __name__ == "__main__":
    
    result = pool.map(speak, np.arange(10))
    print(result)
    pool.close()
