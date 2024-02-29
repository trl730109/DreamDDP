#nnodes=( "1" "2" "3" "4" "5" "6" "7" "8" )
nnodes=( "8" )
#njobs=( "1" "2" "3" "4" "5" "6" "7" "8" )
njobs=( "7" ) #"2" "3" "4" "5" "6" "7" "8" )
for nnode in "${nnodes[@]}"
do
    for njob in "${njobs[@]}"
    do
        nnodes=$nnode job_num=$njob ./multiple_osu.sh
    done
done
