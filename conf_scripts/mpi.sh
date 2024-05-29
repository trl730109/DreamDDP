nworkers=4
lr=0.1
#lr=0.01
batch_size=32
dnn=resnet20
#dnn=vgg16
#dnn=alexnet
#dnn=resnet50
dataset=cifar10
#dataset=imagenet
#max_epochs=180
max_epochs=180
nstepsupdate=100
data_dir=./data
#data_dir=/home/comp/csshshi/data/imagenet/ILSVRC2012_dataset
#data_dir=/home/comp/csshshi/data/imagenet/imagenet_hdf5
#data_dir=/home/shshi/data/imagenet/imagenet_hdf5
mpirun -np $nworkers -hostfile cluster$nworkers -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    -x NCCL_P2P_DISABLE=1 \
    python -m mpi4py dl_trainer.py --dnn $dnn --dataset $dataset --max-epochs $max_epochs --batch-size $batch_size --nworkers $nworkers --data-dir $data_dir --lr $lr --nsteps-update $nstepsupdate --compression
