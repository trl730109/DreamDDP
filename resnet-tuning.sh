gpuids="${gpuids:-0}"
nstreams="${nstreams:-1}"
nworkers=32
#threshold=524288000
#thresholds=( "524288" "1048576" "2097152" "8388608" "16777216" "33554432" "67108864" "134217728" "268435456" )
thresholds=( "524288000" )
#thresholds=( "67108864" "16777216" "8388608" "1028576" "8192" )
for threshold in "${thresholds[@]}"
do
    #nstreams=$nstreams gpuids=$gpuids max_epochs=1 batch_size=64 lr=1.2 dnn=resnet50 density=1 threshold=$threshold nworkers=$nworkers compressor=none momentum_correction=0 ./horovod_mpi_cj.sh
    #nstreams=$nstreams gpuids=$gpuids max_epochs=1 batch_size=32 lr=1.2 dnn=resnet152 density=1 threshold=$threshold nworkers=$nworkers compressor=none momentum_correction=0 ./horovod_mpi_cj.sh
    nstreams=$nstreams gpuids=$gpuids max_epochs=1 batch_size=32 lr=1.2 dnn=inceptionv4 density=1 threshold=$threshold nworkers=$nworkers compressor=none momentum_correction=0 ./horovod_mpi_cj.sh
    #nstreams=$nstreams gpuids=$gpuids max_epochs=1 batch_size=32 lr=1.2 dnn=densenet201 density=1 threshold=$threshold nworkers=$nworkers compressor=none momentum_correction=0 ./horovod_mpi_cj.sh
    #nstreams=$nstreams gpuids=$gpuids max_epochs=1 batch_size=32 lr=1.2 dnn=densenet161 density=1 threshold=$threshold nworkers=$nworkers compressor=none momentum_correction=0 ./horovod_mpi_cj.sh
    #nstreams=$nstreams gpuids=$gpuids max_epochs=1 batch_size=64 lr=1.2 dnn=vgg16i density=1 threshold=$threshold nworkers=$nworkers compressor=none momentum_correction=0 ./horovod_mpi_cj.sh
done
