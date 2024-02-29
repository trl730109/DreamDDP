dnn=resnet20
#dnn=$dnn compressor=gtopk nworkers=32 lr=0.8 ./robust_mpi.sh
#dnn=$dnn compressor=gtopk nworkers=32 lr=0.1 ./robust_mpi.sh
#dnn=$dnn nworkers=32 lr=0.8 ./horovod_mpi.sh
#dnn=$dnn nworkers=32 lr=0.1 ./horovod_mpi.sh

dnn=vgg16
#dnn=$dnn compressor=gtopk nworkers=32 lr=0.8 ./robust_mpi.sh
dnn=$dnn compressor=gtopk nworkers=32 lr=0.1 ./robust_mpi.sh
#dnn=$dnn nworkers=32 lr=0.8 ./horovod_mpi.sh
#dnn=$dnn nworkers=32 lr=0.1 ./horovod_mpi.sh
