from mpi4py import MPI
import numpy as np
import time
import threading

comm = MPI.COMM_WORLD
#comm.Set_errhandler(MPI.ERRORS_RETURN)

mpi_float16 = MPI.BYTE.Create_contiguous(2).Commit()
MPI._typedict['e'] = mpi_float16

w = 1000
h = 1000
num_elements_of_tensor = w*h
shape = (w, h)
num_workers = comm.size
rank = comm.rank
values = {}
indexes = {}
nnzs = {}
for i in range(num_workers):
    values[i] = np.zeros(num_elements_of_tensor, dtype=np.float32)
    indexes[i] = np.zeros(num_elements_of_tensor, dtype=np.int32)
    nnzs[i] = np.array([0])


threshold = 0.99
storage = {}
storage['nnzs_1d'] = np.zeros(num_workers, dtype=np.int32)
storage['values_1d'] = np.zeros(num_elements_of_tensor * num_workers, dtype=np.float32)
storage['indexes_1d'] = np.zeros(num_elements_of_tensor * num_workers, dtype=np.int32)
storage['displ_1d'] =  np.zeros(num_workers, dtype=np.int32)
storage['result'] =  np.zeros(num_elements_of_tensor, dtype=np.float32)

def mpi_errhandler(pcomm, perr, *args):
    #create_errhandler
    print('mpi_errhandler')


def sparse_allreduce_native(sparse_tensor, result):
    tensor = sparse_tensor
    index = np.flatnonzero(tensor).astype(np.int32)
    
    nnz = len(index)
    values[rank][0:nnz] = tensor[index]
    indexes[rank] = index
    nnzs[rank] = nnz

    # Tell others how many nnzs
    for i in range(num_workers):
        #nnzs[i] = comm.bcast(nnzs[i], root=i)
        comm.Bcast([nnzs[i], 1, MPI.INT], root=i)
    comm.Barrier()
    #Aprint('[rank:%d] nnzs: %s' %(rank, nnzs))

    #print('[rank:%d] values: %s' %(rank, values))


    for i in range(num_workers):
        nnz = nnzs[i][0]
        #values[i][0:nnz] = comm.bcast(values[i][0:nnz], root=i)
        #indexes[i][0:nnz] = comm.bcast(indexes[i][0:nnz], root=i)
        comm.Bcast([values[i], nnz, MPI.FLOAT], root=i)
        comm.Bcast([indexes[i], nnz, MPI.INT], root=i)
    comm.Barrier()

    for i in range(num_workers):
        nnz = nnzs[i][0]
        index = indexes[i][0:nnz]
        result[index] += values[i][0:nnz]

    return result


def sparse_allreduce(sparse_tensor, storage):
    tensor = sparse_tensor.astype(np.float32)
    index = np.flatnonzero(tensor).astype(np.int32)
    nnzs_1d = storage['nnzs_1d']
    values_1d = storage['values_1d'].astype(np.float32)
    indexes_1d = storage['indexes_1d']
    displ_1d = storage['displ_1d']
    result = storage['result']
    result.fill(0)
    
    nnz = len(index)

    # Tell others how many nnzs
    comm.Allgather(np.array(nnz, dtype=np.int32), nnzs_1d[:num_workers])

    #print('[rank:%d] nnzs_1d: %s' %(rank, nnzs_1d))

    for i, nnz in enumerate(nnzs_1d[0:-1]):
        displ_1d[i+1] = displ_1d[i] + nnz
    #print('[rank:%d] displ_1d: %s' %(rank, displ_1d))
    #print('[rank:%d] tensor: %s' %(rank, tensor[index]))
    #print('[rank:%d] index: %s' %(rank, index))
    comm.Allgatherv(tensor[index], [values_1d, nnzs_1d[:num_workers], displ_1d[:num_workers], MPI.FLOAT])

    #comm.Allgatherv(tensor[index], [values_1d, nnzs_1d, displ_1d, mpi_float16])
    #print('[rank:%d] allgather values: %s' %(rank, values_1d))
    comm.Allgatherv(index, [indexes_1d, nnzs_1d[:num_workers], displ_1d[:num_workers], MPI.INT])
    comm.Barrier()
    #print('[rank:%d] allgather index: %s' %(rank, indexes_1d))

    for i in range(num_workers):
        nnz = nnzs_1d[i]
        displ = displ_1d[i]
        index = indexes_1d[displ:displ+nnz]
        result[index] += values_1d[displ:displ+nnz]

    return result



def dense_allreduce(tensor, result):
    op = MPI.SUM
    comm.Allreduce(tensor, result, op)
    comm.Barrier()
    return result


def _check(a, ra):
    ret = np.array_equal(a, ra)
    dist = np.linalg.norm(a-ra)
    print('[rank: %d] error: %s' % (rank, dist))

    #print('[rank: %d] Equal: %s' % (rank, ret))
    #print('[rank:%d] sar_result: %s' %(rank, a))
    #print('[rank:%d] dense_result: %s' %(rank, ra))
    return ret

def _empty(d):
    for k in d:
        d[k].fill(0)

def benchmark():
    global comm
    global rank
    global num_workers
    eh = comm.Create_errhandler(mpi_errhandler)
    comm.Set_errhandler(eh)
    raw = np.random.rand(*shape).astype(np.float32)
    tensor = raw.flatten()
    tensor[tensor<threshold] = 0.0
    nnz = np.count_nonzero(tensor)
    sparsity = 1-float(nnz)/tensor.size
    print('[rank: %d] Sparsity: %f' % (rank, sparsity)) 
    
    sar_result = np.zeros_like(tensor)
    dense_result = np.zeros_like(tensor)
    # Warmup
    sar_result = sparse_allreduce(tensor, storage).reshape(shape)
    dense_result = dense_allreduce(tensor, dense_result).reshape(shape)
    _check(sar_result, dense_result)
    iter = 1000
    t = time.time()
    tmprank = rank
    for i in range(iter):
        #if i == 8 and tmprank == 1:
        #    s = 10/0
        try:
            sar_result = sparse_allreduce(tensor, storage).reshape(shape)
            #time.sleep(1)
        except Exception as e:
            print('[rank: %d] Error: %s' % (rank, e))
            comm = comm.EXT_shrink()
            rank = comm.rank
            num_workers = comm.size
    #comm.EXT_failure_ack()
    #group = comm.EXT_failure_get_acked()
    #print('[rank: %d] %d failed' % (rank, group.Get_size()))
    rank = comm.rank
    num_workers = comm.size
    print('[rank: %d] Sparse time used: %f' % (rank, (time.time()-t)/iter))
    t = time.time()
    for i in range(iter):
        dense_result = dense_allreduce(tensor, dense_result)
    print('[rank: %d] Dense time used: %f' % (rank, (time.time()-t)/iter))
    _check(sar_result.reshape(shape), dense_result.reshape(shape))
    print('[rank: %d] Finished', rank)


#benchmark()

def wait():
    while True:
        print('[rank: %d] I am waiting ' % rank)
        time.sleep(1)

#benchmark()
if __name__ == '__main__':
    benchmark()

#print('[rank:%d] values: %s' %(rank, tensor))
#print('[rank:%d] sar_result: %s' %(rank, sar_result))
#print('[rank:%d] dense_result: %s' %(rank, dense_result))
