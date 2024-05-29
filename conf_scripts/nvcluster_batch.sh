#nworkers=16 dnn=lstm density=1 ./horovod_mpi.sh
#nworkers=16 dnn=lstm compressor=topk density=0.004 ./horovod_mpi.sh
#nworkers=16 dnn=lstm compressor=topk density=0.004 ./robust_mpi.sh

#nworkers=16 dnn=lstman4 compressor=topk density=0.001 ./horovod_mpi.sh
#nworkers=16 dnn=lstman4 density=1 ./horovod_mpi.sh
#nworkers=16 dnn=lstman4 compressor=topk density=0.001 ./robust_mpi.sh


#nworkers=16 dnn=resnet20 compressor=topk density=0.01 ./horovod_mpi.sh
#nworkers=16 dnn=resnet20 compressor=topk density=1 ./horovod_mpi.sh
#nworkers=16 dnn=resnet20 compressor=topk density=0.01 ./robust_mpi.sh

#nworkers=16 dnn=vgg16 compressor=topk density=0.01 batch_size=32 ./horovod_mpi.sh
#nworkers=16 dnn=vgg16 compressor=topk density=1 batch_size=32 ./horovod_mpi.sh
#nworkers=16 dnn=vgg16 compressor=topk density=0.01 batch_size=32 ./robust_mpi.sh

#nworkers=16 dnn=resnet50 compressor=topk density=0.001 batch_size=32 ./horovod_mpi.sh
#nworkers=16 dnn=alexnet compressor=topk density=0.001 batch_size=128 lr=0.04 ./horovod_mpi.sh
#nworkers=16 dnn=alexnet compressor=topk density=0.001 batch_size=128 lr=0.05 ./horovod_mpi.sh
#dnns=( "resnet50" "googlenet" "vgg16i" "alexnet" "inceptionv4" )
#dnns=( "resnet50" "resnet152" "inceptionv4" "densenet161" "densenet201" )
#dnns=( "resnet152" "inceptionv4" )
#dnns=( "resnet20" "vgg16" "resnet110" )
#dnns=( "lstm" )
dnns=( "lstmwt2" )
#thresholds=( "524288000" "0" ) #"16777216" "8388608" "1028576" "8192" "1024" "0" )
thresholds=( "524288000" ) #"16777216" "8388608" "1028576" "8192" "1024" "0" )
#thresholds=( "524288000" ) #"16777216" "8388608" "1028576" "8192" "1024" "0" )
#ns=( "16" "8" "4" "2" )
ns=( "32" )
#ns=( "8" )
#density=1.0
density=0.001
momentum_correction=1
#compressors=( "dgcsampling" ) #"randomkec" )
#compressors=( "none" ) #"randomkec" )
compressors=( "eftopkdecay" "eftopk" ) #"eftopkdd" )
#compressors=( "eftopkdecay" )
#batch_size=64
#batch_size=128
batch_size=4
lrs=( "20" )
#lrs=( "0.1" "0.2" "0.4" "0.566" "0.8" "1.6" "3.2" )
for dnn in "${dnns[@]}"
do
    for thres in "${thresholds[@]}"
    do
        for nworkers in "${ns[@]}"
        do
            for compressor in "${compressors[@]}"
            do
                for lr in "${lrs[@]}"
                do
                    lr=$lr batch_size=$batch_size dnn=$dnn density=$density threshold=$thres nworkers=$nworkers compressor=$compressor momentum_correction=$momentum_correction ./horovod_mpi_nvcluster.sh
                done
            done
        done
    done
done
