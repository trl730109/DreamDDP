# PY="${PY:-/home/comp/20481896/anaconda3/envs/py36/bin/python}"
PY="${PY:-/mnt/raid/tangzhenheng/anaconda3/envs/fusionai/bin/python}"

cluster_name=${cluster_name:-localhost}
dataset=${dataset:-cifar10}
dnn=${dnn:-gpt2}
model_dir=${model_dir:-gpt2}

case "$cluster_name" in
    "A6000")
        case "$dataset" in
            "cifar10" | "cifar100" | "mnist") data_dir="/workspace/tzc/DreamDDP/cifar" ;;
            "wikitext2") data_dir="/workspace/wikitext2" ;;
            "openwebtext") data_dir="/workspace/encoded_openwebtext" ;;
        esac
        case "$dnn" in
            "gpt2") model_dir="/workspace/models/gpt2" ;;
            "bert-base-uncased") model_dir="/data2/share/zhtang/bert-base-uncased" ;;
            "llama2-124M") model_dir="/workspace/models/Llama-2-7b-hf" ;;
            "Qwen2.5-1.5B") model_dir="/workspace/tzc/Qwen/Qwen2.5-1.5B" ;;
            "Qwen2.5-7B") model_dir="/workspace/tzc/Qwen/Qwen2.5-7B-Instruct" ;;
            
        esac
        ;;
    *)
        echo "Unknown cluster name: $cluster_name"
        exit 1
        ;;
esac












