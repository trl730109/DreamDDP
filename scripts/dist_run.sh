#!/bin/bash
#remotehosts=( "gpu10" "gpu11" "gpu12" "gpu13" "gpu14" "gpu15" "gpu16" "gpu17" "gpu18" "gpu19")
#remotehosts=( "gpu10" "gpu11" "gpu12" "gpu13" )
#numofgpu=( 4 4 4 4 )
remotehosts=( "gpu18" "gpu19" )
numofgpu=( 2 2 )
#directory=/home/comp/csshshi/repositories/p2p-dl
directory=/home/comp/qiangwang/p2p-dl
baseport=5922
batchsize=32
echo $directory
#for host in ${remotehosts[@]}
i=0
rank=0
for host in "${remotehosts[@]}"
do
    ngpu="${numofgpu[$i]}"
    #echo $ngpu
    j=0
    while [ $j -lt $ngpu ]
    do
        if (( $j % 2 == 0 ))
        then
            role=active
        else
            role=passive
        fi
        port=$(expr $baseport + $j)
        args="CUDA_VISIBLE_DEVICES=$j python client.py --port $port --batch-size $batchsize --rank $rank --role $role > /dev/null 2>&1"
        j=$(expr $j + 1)
        rank=$(expr $rank + 1)
        echo $args
        cmd="cd $directory; $args" 
        ssh $host $cmd &
    done
    i=$(expr $i + 1)
done
