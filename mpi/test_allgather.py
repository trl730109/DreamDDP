from mpi4py import MPI
import numpy as np
import time

comm = MPI.COMM_WORLD
num_workers = comm.size
rank = comm.rank

nnzs = np.zeros(num_workers, dtype=np.int32)
send_buf = np.array([rank], dtype=np.int32)
comm.Allgather(send_buf, nnzs)

print('[rank:%d] nnzs: %s' %(rank, nnzs))
