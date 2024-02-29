#directory=/home/comp/csshshi/repositories/p2p-dl
directory=/home/esetstore/repos/p2p
max=16
#for host in "${remotehosts[@]}"
for ((number=1;number <= $max;number++))
do
    #host=host$number
    #host=csr$number
    host=gpu$number
    ssh $host $directory/scripts/killp2p.sh &
done
