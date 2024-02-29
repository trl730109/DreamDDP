import numpy as np

def read_times_from_nccl_log(logfile, mode='allreduce', start=0, end=128*1024*1024, original=False, bw=False):
    print('fn: ', logfile)
    f = open(logfile)
    sizes = []
    times = []
    size_comms = {}
    for line in f.readlines():
        if original and line[0:2] != '--':
            items = ' '.join(line.split()).split(' ')
            if (len(items) == 11 or len(items) == 12) and items[0] != '#':
                try:
                    size = int(items[0])
                except:
                    continue
                if size == 8:
                    continue
                if (size >= start and size <= end):
                    if size not in size_comms:
                        size_comms[size] = [] 
                    try:
                        if mode == 'allreduce':
                            t = float(items[4])/(1000*1000)
                            if bw:
                                t = float(items[6])*8
                        else:
                            t = float(items[3])/(1000*1000)
                            if bw:
                                t = float(items[6])*8
                        size_comms[size].append(t)
                        sizes.append(size)
                        times.append(t)
                    except:
                        continue
        elif line[0:2] == '--':
            items = ' '.join(line.split()).split(' ')
            size = int(items[0][2:])
            t = float(items[1])/(1000*1000)
            times.append(t)
            if size not in size_comms:
                size_comms[size] = [] 
            size_comms[size].append(t)
            
    f.close()
    sizes = list(size_comms.keys())
    sizes.sort()
    #comms = [np.mean(size_comms[s]) for s in sizes]
    comms = [np.max(size_comms[s]) for s in sizes]
    comms = []
    for s in sizes:
        a = np.array(size_comms[s])
        a.sort()
        comms.append(np.mean(a))
    errors = [np.std(size_comms[s]) for s in sizes]
    #print('sizes: ', sizes)
    #print('comms: ', comms)
    #print('errors: ', errors)
    return np.array(sizes), np.array(comms), np.array(errors)

