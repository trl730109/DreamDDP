#date >> logs/multi-jobs.log
#gpuids=0 ./resnet-tuning.sh &
#wait
#date >> logs/multi-jobs.log
#echo "One job ends">> logs/multi-jobs.log
#
#date >> logs/multi-jobs.log
#gpuids=0 ./resnet-tuning.sh &
#gpuids=1 ./resnet-tuning.sh &
#wait
#date >> logs/multi-jobs.log
#echo "Two jobs end">> logs/multi-jobs.log
#
#date >> logs/multi-jobs.log
#gpuids=0 ./resnet-tuning.sh &
#gpuids=1 ./resnet-tuning.sh &
#gpuids=2 ./resnet-tuning.sh &
#wait
#date >> logs/multi-jobs.log
#echo "Three jobs end">> logs/multi-jobs.log
#
#date >> logs/multi-jobs.log
#gpuids=0 ./resnet-tuning.sh &
#gpuids=1 ./resnet-tuning.sh &
#gpuids=2 ./resnet-tuning.sh &
#gpuids=3 ./resnet-tuning.sh &
#wait
#date >> logs/multi-jobs.log
#echo "Four jobs end">> logs/multi-jobs.log

for nstreams in `seq 1 4`
do
    nstreams=$nstreams gpuids=0,1,2,3 ./resnet-tuning.sh
done
