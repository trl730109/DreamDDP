nnodes="${nnodes:-2}"
logRoot=logs/v2_alphabeta
mkdir -p $logRoot
nnodes=$nnodes ./osu_mpi.sh 1>${logRoot}/n${nnodes}.log 2>&1
