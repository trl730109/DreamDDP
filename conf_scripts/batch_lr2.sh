#dnn=resnet20 lr=0.1 density=0.001 ./robust_mpi.sh
#dnn=resnet20 lr=0.01   density=0.001 ./robust_mpi.sh
#dnn=resnet20 lr=0.0035 density=0.001 ./robust_mpi.sh
#dnn=vgg16 lr=0.1    ./horovod_mpi.sh
#dnn=vgg16 lr=0.01   ./horovod_mpi.sh
#dnn=vgg16 lr=0.001  ./horovod_mpi.sh
#dnn=vgg16 lr=0.1    density=0.001 ./robust_mpi.sh
#dnn=vgg16 lr=0.01   density=0.001 ./robust_mpi.sh
#dnn=vgg16 lr=0.001  density=0.001 ./robust_mpi.sh
#dnn=vgg16 lr=0.1    density=0.0001 ./robust_mpi.sh
#dnn=vgg16 lr=0.01   density=0.0001 ./robust_mpi.sh
#dnn=vgg16 lr=0.001  density=0.0001 ./robust_mpi.sh
#dnn=vgg16 lr=0.1    density=0.00001 ./robust_mpi.sh
#dnn=vgg16 lr=0.01   density=0.00001 ./robust_mpi.sh
#dnn=vgg16 lr=0.001  density=0.00001 ./robust_mpi.sh
#dnn=vgg16 lr=0.0001 density=0.00001 ./robust_mpi.sh
#dnn=lstm lr=1.0    ./horovod_mpi.sh
#dnn=lstm lr=1.0    density=0.005 ./robust_mpi.sh
#dnn=lstm lr=1.0    density=0.001 ./robust_mpi.sh
dnn=lstm lr=0.375  density=0.001 ./robust_mpi.sh
dnn=lstm lr=1.0    density=0.0005 ./robust_mpi.sh
dnn=lstm lr=0.375  density=0.0005 ./robust_mpi.sh
dnn=lstm lr=0.127  density=0.0005 ./robust_mpi.sh
