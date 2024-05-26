export HOROVOD_WITH_MPI=1
# Uncomment the following line if you want to use GLOO
# export HOROVOD_WITH_GLOO=1

# Default settings
dnn="${dnn:-resnet20}"
source exp_configs/$dnn.conf
nworkers="${nworkers:-4}"
density="${density:-0.01}"
threshold="${threshold:-524288000}"
compressor="${compressor:-topk}"
momentum_correction="${momentum_correction:-0}"
nwpernode=4
nstepsupdate=1
overlap_scalar=2
strategy='average'
nsteps_localsgd=10
PY=~/miniconda3/envs/DDP/bin/python3
MPIPATH=~/miniconda3/envs/DDP/
GRADSPATH=./logs/tzc

# Optimizers to run
optimizers=('SGD' 'Adam' 'AdamW')

# Loop through each optimizer
for optimizer_name in "${optimizers[@]}"
do
    echo "Running training with $optimizer_name optimizer..."
    horovodrun -np $nworkers -H localhost:4 $PY horovod_trainer.py \
        --optimizer_name $optimizer_name \
        --nsteps_localsgd $nsteps_localsgd \
        --strategy $strategy \
        --overlap_scalar $overlap_scalar \
        --dnn $dnn \
        --dataset $dataset \
        --max-epochs $max_epochs \
        --batch-size $batch_size \
        --nworkers $nworkers \
        --data-dir $data_dir \
        --lr $lr \
        --nsteps-update $nstepsupdate \
        --nwpernode $nwpernode \
        --density $density \
        --compressor $compressor \
        --threshold $threshold \
        --saved-dir $GRADSPATH \
        --momentum-correction $momentum_correction
done
