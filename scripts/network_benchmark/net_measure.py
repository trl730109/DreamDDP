#!/usr/bin/python

import os
import random
import time
import argparse
parser = argparse.ArgumentParser(description="iperf")
parser.add_argument('--src', type=str)
args = parser.parse_args()
source=args.src


cmd_header='iperf3 -c '
#addr_prefix='192.168.1.'
#servers_arr=range(172, 183)
addr_prefix='host'
servers_arr=range(1, 10)

#for server in servers_arr:
max_time = 1000
t=0
while t<max_time:
    i = random.randint(0, len(servers_arr)-1)
    server = servers_arr[i]
    cmd= cmd_header
    addr = '%s%d' % (addr_prefix, server)
    cmd+=addr
    cmd+=" >> "
    cmd+=source
    cmd+="_"
    cmd+=addr
    os.system(cmd)
    time.sleep(5)
    t+=5

