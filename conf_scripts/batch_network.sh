dnn=resnet20
dnn=$dnn compressor=gtopk ./robust_mpi.sh
dnn=$dnn compressor=sigmathres ./robust_mpi.sh
#dnn=$dnn compressor=centralzero ./robust_mpi.sh

dnn=vgg16
dnn=$dnn compressor=gtopk ./robust_mpi.sh
dnn=$dnn compressor=sigmathres ./robust_mpi.sh
#dnn=$dnn compressor=centralzero ./robust_mpi.sh
