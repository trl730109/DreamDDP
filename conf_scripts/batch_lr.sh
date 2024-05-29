#dnn=resnet20 lr=0.1    ./horovod_mpi.sh
#dnn=resnet20 lr=0.1 density=0.001 ./robust_mpi.sh
#dnn=resnet20 lr=0.1 density=0.00025 ./robust_mpi.sh
#dnn=resnet20 lr=0.1 density=0.0001 ./robust_mpi.sh
#dnn=resnet20 lr=0.1 density=0.00005 ./robust_mpi.sh
#dnn=resnet20 lr=0.1 density=0.00001 ./robust_mpi.sh
#
#dnn=resnet20 lr=0.0125  density=0.00025 ./robust_mpi.sh
#dnn=resnet20 lr=0.00316 density=0.0001 ./robust_mpi.sh
#dnn=resnet20 lr=0.00112 density=0.00005 ./robust_mpi.sh
#dnn=resnet20 lr=0.0001  density=0.00001 ./robust_mpi.sh
#
#dnn=vgg16 lr=0.1    ./horovod_mpi.sh
#dnn=vgg16 lr=0.1 density=0.001 ./robust_mpi.sh
#dnn=vgg16 lr=0.1 density=0.00025 ./robust_mpi.sh
#dnn=vgg16 lr=0.1 density=0.0001 ./robust_mpi.sh
#dnn=vgg16 lr=0.1 density=0.00005 ./robust_mpi.sh
#dnn=vgg16 lr=0.1 density=0.00001 ./robust_mpi.sh
#
#dnn=vgg16 lr=0.0125  density=0.00025 ./robust_mpi.sh
#dnn=vgg16 lr=0.00316 density=0.0001 ./robust_mpi.sh
#dnn=vgg16 lr=0.00112 density=0.00005 ./robust_mpi.sh
#dnn=vgg16 lr=0.0001  density=0.00001 ./robust_mpi.sh

dnn=lstm lr=20 density=0.001 ./robust_mpi.sh
dnn=lstman4 lr=0.0003 density=0.001 ./robust_mpi.sh
dnn=lstm lr=20 ./horovod_mpi.sh
dnn=lstman4 lr=0.0003 ./horovod_mpi.sh
#dnn=lstm lr=1 density=0.00025 ./robust_mpi.sh
#dnn=lstm lr=1 density=0.0001 ./robust_mpi.sh
#dnn=lstm lr=1 density=0.00005 ./robust_mpi.sh
#dnn=lstm lr=1 density=0.00001 ./robust_mpi.sh
#dnn=lstm lr=5 ./horovod_mpi.sh
#dnn=lstm lr=1 density=0.001 ./robust_mpi.sh

#dnn=lstman4 lr=0.0002 ./horovod_mpi.sh
#dnn=lstman4 lr=0.0003 ./horovod_mpi.sh
#dnn=lstman4 lr=0.0003 density=0.001 ./robust_mpi.sh
#dnn=lstman4 lr=0.0002 density=0.00025 ./robust_mpi.sh
#dnn=lstman4 lr=0.0002 density=0.0001 ./robust_mpi.sh
#dnn=lstman4 lr=0.0002 density=0.00005 ./robust_mpi.sh
#dnn=lstman4 lr=0.0002 density=0.00001 ./robust_mpi.sh

#dnn=lstman4 lr=0.0003 density=0.00025 ./robust_mpi.sh
#dnn=lstman4 lr=0.0003 density=0.0001 ./robust_mpi.sh
#dnn=lstman4 lr=0.0003 density=0.00005 ./robust_mpi.sh
#dnn=lstman4 lr=0.0003 density=0.00001 ./robust_mpi.sh
