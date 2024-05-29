compressors=( "gtopk" "topk" )
densities=( "0.002" ) #"0.002" "0.005" "0.01" "0.05" "0.1" ) 
dnns=( "resnet20" ) #"vgg16" )
batch_size=4
for density in "${densities[@]}"
do
    for dnn in "${dnns[@]}"
    do
        for compressor in "${compressors[@]}"
        do
            nworkers=32 dnn=$dnn compressor=$compressor density=$density batch_size=$batch_size ./robust_mpi.sh
        done
    done
done


