#dnns=( "resnet50" "googlenet" "vgg16i" "alexnet" "inceptionv4" )
#dnns=( "resnet50" "resnet152" "inceptionv4" "densenet161" "densenet201" )
#dnns=( "resnet152" "inceptionv4" )
#dnns=( "densenet161" "densenet201" )
dnns=( "resnet20" "vgg16" "resnet110" )
#dnns=( "alexnetbn" ) #"resnet110" "vgg16" )
#thresholds=( "524288000" "0" ) #"16777216" "8388608" "1028576" "8192" "1024" "0" )
#thresholds=( "0" ) #"16777216" "8388608" "1028576" "8192" "1024" "0" )
thresholds=( "524288000" ) #"16777216" "8388608" "1028576" "8192" "1024" "0" )
#ns=( "16" "8" "4" "2" )
ns=( "4" )
#ns=( "8" )
#density=1.0
#density=0.001
momentum_correction=1
NUM_RUNS=1
#compressors=( "dgcsampling" ) #"randomkec" )
#compressors=( "none" ) #"randomkec" )
#compressors=( "none" ) # "eftopkdecay" "eftopk" "eftopkdd" )
compressors=( "none" ) #"eftopk" "eftopkdecay" "eftopkdd" )
#compressors=( "eftopkdecay" )
#compressors=( "gaussian" )
#compressors=( "eftopkddr4" ) #"eftopkddr2" "eftopkddr3" )
#compressors=( "dgcsamplingdd" )
#compressors=( "dgcsampling" )
#lr=0.8
#lr=1.2
#lrs=( "6.4" )
#lrs=( "8.0" )
#lrs=( "9.6" )
#lrs=( "12.8" "11.2" "9.6" "8.0" "6.4" "3.2" "2.4" )
#lrs=( "3.2" "2.4" )
#lrs=( "0.0001" "0.001" "0.01" "0.1" "1.0" )
#lrs=( "0.01" "0.32" "0.1" "3.2" )
#lrs=( "1.2" "1.4" ) #"0.32" "0.1" "3.2" )
lrs=( "0.001" "0.01" "0.1" "1.0" ) 
batches=( "256" "128" "64" "32" )
densities=( "0.001" "0.005" "0.01" "0.05" )
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
                    for bs in "${batches[@]}"
                    do
                        for iter in `seq 1 ${NUM_RUNS}`
                        do
                            if [ "$compressor" = "none" ]; then 
                                batch_size=$bs lr=$lr dnn=$dnn density=1.0 threshold=$thres nworkers=$nworkers compressor=$compressor momentum_correction=0 ./horovod_mpi_scigpu.sh
                            else
                                for density in "${densities[@]}"
                                do
                                    batch_size=$bs lr=$lr dnn=$dnn density=$density threshold=$thres nworkers=$nworkers compressor=$compressor momentum_correction=0 ./horovod_mpi_scigpu.sh
                                    batch_size=$bs lr=$lr dnn=$dnn density=$density threshold=$thres nworkers=$nworkers compressor=$compressor momentum_correction=1 ./horovod_mpi_scigpu.sh
                                done
                            fi
                        done
                    done
                done
            done
        done
    done
done
