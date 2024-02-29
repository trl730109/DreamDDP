ROOT=/home/t716/shshi/
max=10
for i in `seq 1 $max`
do
    host=host$i
    echo $host 
    rsync -u -avz -r ./p2p $host:$ROOT
done
