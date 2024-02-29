nnodes=( "2" "3" "4" "5" "6" "7" "8" )
for nnode in "${nnodes[@]}"
do
    nnodes=$nnode ./single_osu.sh
done
