port=${port:-23456}
kill -9 `ps aux|grep "master_port=${port} fault_dist_trainer.py" | awk '{print $2}'`



