#!/usr/bin/env python
from mpi4py import MPI
import numpy as np
import time

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()
sizes = 16000

print ("MPI comm_size {}".format(comm_size)) 

#define a float16 mpi datatype
mpi_float16 = MPI.BYTE.Create_contiguous(2).Commit()
MPI._typedict['e'] = mpi_float16
def sum_f16_cb(buffer_a, buffer_b, t):
    assert t == mpi_float16
    array_a = np.frombuffer(buffer_a, dtype='float16')
    array_b = np.frombuffer(buffer_b, dtype='float16')
    array_b += array_a
#create new OP
mpi_sum_f16 = MPI.Op.Create(sum_f16_cb, commute=True)

def perform_test(dtype,op):
    start = time.time()
    data = np.array([comm_rank] * sizes,dtype=dtype)
    all_sum = np.empty_like(data)
    comm.Allreduce(data, all_sum, op=op) 

    x=np.array([data] * sizes,dtype=dtype)
    all_all_sum = np.empty_like(x)
    comm.Allreduce(x, all_all_sum, op=op) #MPI.SUM)
    end = time.time()

    return end-start

#float16 test
f16 = perform_test(np.float16,mpi_sum_f16)
#float32 test
f32 = perform_test(np.float32,MPI.SUM)
#float64 test
f64 = perform_test(np.float64,MPI.SUM)

if comm_rank == 0:
    print ("f32/f16 {} ".format(f32/f16))
    print ("f64/f16 {} ".format(f64/f16))
    print ("f64/f32 {} ".format(f64/f32))
