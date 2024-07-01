#directory=/home/comp/csshshi/repositories/p2p-dl
# directory=/home/esetstore/repos/p2p
directory=/home/yinyiming/DDP-Train-error-grad
max=16
#for host in "${remotehosts[@]}"
# for ((number=1;number <= $max;number++))
# do
#     #host=host$number
#     #host=csr$number
#     host=gpu$number
#     ssh $host $directory/fault_exps/killp2p.sh &
# done


directory=/home/yinyiming/DDP-Train-error-grad
ssh gpu3 "bash $directory/fault_exps/killp2p.sh" &

