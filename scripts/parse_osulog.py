from __future__ import print_function
import os
import numpy as np

ROOTPATH='/media/sf_Shared_Data/tmp/icdcs2019/mgdlogs/alllogs'
PARSE_DIR='logs/allreduce-comp-baseline-gwarmup-dc1-modelmgd/resnet50-n128-bs256-lr0.0100-ns16-sg2.50'

host_start=101
host_end=190


def read_logs(filename):
    with open(filename, 'r') as f:
        times = []
        for line in f.readlines():
            if line.find('average forward and backward time') > 0:
                t = float(line.split(':')[-1].split(',')[0])
                times.append(t)
        if len(times) == 0:
            times.append(0.0)
        return np.mean(times), np.std(times), np.max(times)


def parse():
    avaiables = []
    file_to_host_map = {}
    allfiles = []
    for host in range(host_start, host_end):
        hostname = '192.168.122.%d' % host
        path = os.path.join(ROOTPATH, hostname)
        path = os.path.join(path, PARSE_DIR)
        #print('path: ', path)
        if os.path.isdir(path):
            avaiables.append(path)

            for f in os.listdir(path):
                fullfilename = os.path.join(path,f)
                if os.path.isfile(fullfilename) and f.endswith('.log'):
                    allfiles.append(fullfilename)
                    file_to_host_map[fullfilename] = hostname
    hostset = set()
    allhosts = set()
    for f in allfiles:
        mean, std, maximun = read_logs(f)
        allhosts.add(file_to_host_map[f])
        if maximun > 1.0:
            hostset.add(file_to_host_map[f])
            print('maximun > 1.0, f: ', file_to_host_map[f], mean, std, maximun)
    print('total hosts: ', len(allhosts))
    print('hosts: ', len(hostset))
    
    



if __name__ == '__main__':
    parse()
