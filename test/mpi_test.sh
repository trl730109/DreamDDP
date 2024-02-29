MPIPATH=/home/shaohuais/.local/openmpi-4.0.0
nworkers="${nworkers:-16}"
$MPIPATH/bin/mpirun --oversubscribe --prefix $MPIPATH -np $nworkers -hostfile ../cluster$nworkers  -bind-to none -map-by slot \
    -mca pml ob1 -mca btl ^openib \
    -mca btl_tcp_if_include eth0 \
    ~/.local/osu-benchmarks-5.6.2/libexec/osu-micro-benchmarks/mpi/collective/osu_gatherv
    #-mca pml ob1 -mca btl openib \
    #-mca btl_openib_allow_ib 1\
