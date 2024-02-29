#!/bin/bash
#cmd='iperf3 -s &'
#cmd="cd /home/t716/shshi/share/repos/p2p-dl/scripts/network_benchmark; python net_measure.py"
cmd='cd /home/t716/shshi/share/repos/p2p-dl/scripts/network_benchmark; ./kill.sh'
#cmd='sudo apt-get install iperf3'
echo $cmd
remotehosts=( "1" "2" "3" "4" "5" "6" "7" "8" "9" "10" )
for number in "${remotehosts[@]}"
do
    #host=mgd$number
    host=host$number
    echo $host

    #cmd="cd /home/t716/shshi/share/repos/p2p-dl/scripts/network_benchmark; python net_measure.py"
    #cmd="$cmd --src $host &"
    echo $cmd
    ssh $host $cmd &
done 
