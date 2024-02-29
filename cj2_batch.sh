#dnns=( "resnet50" "googlenet" "vgg16i" "alexnet" "inceptionv4" )
#dnns=( "resnet50" "resnet152" "inceptionv4" "densenet161" "densenet201" )
#dnns=( "resnet152" "inceptionv4" )
#dnns=( "densenet161" "densenet201" )
dnns=( "lstm" )
#dnns=( "resnet20" "resnet110" "vgg16" )
#thresholds=( "524288000" "0" ) #"16777216" "8388608" "1028576" "8192" "1024" "0" )
#thresholds=( "0" ) #"16777216" "8388608" "1028576" "8192" "1024" "0" )
thresholds=( "524288000" ) #"16777216" "8388608" "1028576" "8192" "1024" "0" )
#ns=( "16" "8" "4" "2" )
#ns=( "32" )
ns=( "16" )
#density=1.0
density=0.001
momentum_correction=1
#compressors=( "dgcsampling" ) #"randomkec" )
#compressors=( "none" ) #"randomkec" )
#compressors=( "eftopkdecay" "eftopk" "eftopkdd" )
compressors=( "eftopkdecay" "eftopk" )
#compressors=( "eftopk" )
#compressors=( "eftopkddr4" ) #"eftopkddr2" "eftopkddr3" )
#compressors=( "dgcsamplingdd" )
#compressors=( "dgcsampling" )
#lr=0.8
#lr=1.2
#lrs=( "6.4" )
#lrs=( "8.0" )
#lrs=( "9.6" )
#lrs=( "12.8" "11.2" "9.6" "8.0" "6.4" "3.2" "2.4" )
#lrs=( "30" "22" "20" "10" "5" "4" "2" "1" ) 
lrs=( "20" )
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
                    lr=$lr dnn=$dnn density=$density threshold=$thres nworkers=$nworkers compressor=$compressor momentum_correction=$momentum_correction ./horovod_mpi_cj2.sh
                done
            done
        done
    done
done
