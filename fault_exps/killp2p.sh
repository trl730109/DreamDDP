port=${port:-23456}
kill -9 `ps aux|grep "master_port=${port} fault_dist_trainer.py" | awk '{print $2}'`

kill -9 `ps aux|grep "master_port=22222 fault_dist_trainer.py" | awk '{print $2}'`


kill -9 `ps aux|grep "master_port=23456 fault_dist_trainer.py" | awk '{print $2}'`

kill -9 `ps aux|grep "fault_dist_trainer.py" | awk '{print $2}'`



