#ns=( "4" "8" "16" "32" )
ns=( "8" )
#rdmas=( "0" "1" )
rdmas=( "0" )
for rdma in "${rdmas[@]}"
do
    for nworkers in "${ns[@]}"
    do
        gpuid=0,1,2,3 rdma=$rdma iter=100 nworkers=$nworkers ./nccl_mpi.sh > logs/infocom21/nccl-rdma${rdma}-nworkers${nworkers}-v3.log 2>& 1
    done
done
