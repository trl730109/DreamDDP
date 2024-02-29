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
#dnns=( "fcn5net" "lenet" "resnet20" "vgg16" "lstm" "lstman4" )
#dnns=( "lstman4" )
#dnns=( "resnet50" "googlenet" "vgg16i" "alexnet" "inceptionv4" )
#dnns=( "resnet152" "inceptionv4" )
dnns=( "resnet50" )
#dnns=( "densenet161" "densenet201" )
#dnns=( "densenet201" )
#thresholds=( "524288000" "0" ) #"16777216" "8388608" "1028576" "8192" "1024" "0" )
#thresholds=( "0" ) #"16777216" "8388608" "1028576" "8192" "1024" "0" )
thresholds=( "524288000" ) #"16777216" "8388608" "1028576" "8192" "1024" "0" )
#ns=( "16" "8" "4" "2" )
#ns=( "16" )
ns=( "4" )
density=1.0
#compressors=( "dgcsampling" ) #"randomkec" )
compressors=( "none" ) #"randomkec" )
#batch_size=64
batch_size=128
max_epochs=90
for dnn in "${dnns[@]}"
do
    for thres in "${thresholds[@]}"
    do
        for nworkers in "${ns[@]}"
        do
            for compressor in "${compressors[@]}"
            do
                lr=0.1 batch_size=$batch_size max_epochs=$max_epochs dnn=$dnn density=$density threshold=$thres nworkers=$nworkers compressor=$compressor ./horovod_mpi_scigpu.sh
            	#lr=0.6 batch_size=$batch_size  dnn=$dnn density=$density max_epochs=40 threshold=$thres nworkers=$nworkers compressor=$compressor ./horovod_mpi.sh
                #dnn=$dnn density=$density threshold=$thres nworkers=$nworkers compressor=$compressor ./horovod_mpi.sh
            	#lr=1.024 batch_size=$batch_size  dnn=$dnn density=$density max_epochs=40 threshold=$thres nworkers=$nworkers compressor=$compressor ./horovod_mpi.sh
            done
        done
    done
done
