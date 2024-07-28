port=${port:-12222}
kill -9 `ps aux|grep "master_port=${port} debug_hf_model.py" | awk '{print $2}'`
kill -9 `ps aux|grep "debug_hf_model.py" | awk '{print $2}'`
