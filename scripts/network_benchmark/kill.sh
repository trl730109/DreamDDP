kill -9 `ps aux|grep net_measure.py |awk '{print $2}'`
kill -9 `ps aux|grep iperf3 |awk '{print $2}'`
