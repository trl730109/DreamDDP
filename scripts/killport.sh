kill -9 `ps aux|grep 'python client.py' | grep 'port 5922' | awk '{print $2}'`
