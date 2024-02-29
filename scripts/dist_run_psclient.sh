#!/bin/bash
#directory=/home/t716/shshi/share/repos/p2p-dl
directory=/home/comp/csshshi/repositories/p2p-dl
#directory=/home/shshi/repos/p2p-dl
#directory=/home/comp/qiangwang/p2p-dl
echo $directory
max=14
min=14
baseaddr=14
#max=1
#min=1
#baseaddr=1
rank=0
ngpu="${ngpu:-4}"
sparsity="${sparsity:-0.95}"
nworkers=$(( $ngpu * ( $max - $min + 1 ) ))
baseport=5922
# For cifar10 + resnet20
#dnn=resnet20

#dnn=mnistnet
#dataset=mnist
#batchsize=512

#dnn=resnet20
##dnn=vgg16
#dataset=cifar10
#datadir=./data
#batchsize=32
#baselr="${baselr:-0.1}"
#lr=$baselr 

#dnn=resnet50
dnn=alexnet
dataset=imagenet
datadir=/home/comp/csshshi/data/imagenet/ILSVRC2012_dataset
#datadir=/home/shshi/data/imagenet/ILSVRC2012_dataset
baselr=0.01
lr=0.01
batchsize=256


# For ImageNet + resnet50
#batchsize=64
#baselr=0.01
#lr=0.01
i=0
k=0
for ((number=$min;number <= $max;number++))
#for number in "${remotehosts[@]}"
do
    #ngpu="${numofgpu[$i]}"
    #host=host$number
    host=gpu$number
    #host=csr$number
    echo $host
    j=0
    while [ $j -lt $ngpu ]
    do
        if (( $k % 2 == 0 ))
        then
            role=active
            bs=$(expr $batchsize \* 1)
        else
            role=passive
            bs=$(expr $batchsize \* 1)
        fi
        #bindaddr=$(expr $baseaddr + $i)
        bindaddr=$number
        port=$(expr $baseport + $j)
        gpuid=$(expr $j \% 4)
        #gpuid=$(expr $j \% 2)
        #args="CUDA_VISIBLE_DEVICES=$gpuid python psclient.py --ngpu 1 --bind 192.168.0.$bindaddr --port $port --batch-size $bs --rank $rank --role active --dataset $dataset --dnn $dnn --nworkers $nworkers --lr $lr --data-dir $datadir --sparsity $sparsity > /dev/null 2>&1"
        args="CUDA_VISIBLE_DEVICES=$gpuid python client.py --ngpu 1 --bind 192.168.0.$bindaddr --port $port --batch-size $bs --rank $rank --role active --dataset $dataset --dnn $dnn --nworkers $nworkers --lr $lr --data-dir $datadir --sparsity $sparsity > /dev/null 2>&1"
        j=$(expr $j + 1)
        k=$(expr $k + 1)
        rank=$(expr $rank + 1)
        cmd="cd $directory; $args" 
        #cmd="echo 'cd $directory' >> /home/mgd/bin/restart.sh; echo '$args' >> ${directory}/scripts/restart.sh
	    #cmd="cd $directory; rm *.log"
        echo $host
        echo $cmd
        #eval $cmd &
        ssh $host $cmd &
        sleep 0.2
    done
    i=$(expr $i + 1)
done
