# PY="${PY:-/home/comp/20481896/anaconda3/envs/py36/bin/python}"
# PY="${PY:-/mnt/raid/tangzhenheng/anaconda3/envs/fusionai/bin/python}"
PY="${PY:-/mnt/sdb/tangzhenheng/miniconda3/envs/DDP_Train/bin/python}"

cluster_name=${cluster_name:-localhost}
dataset=${dataset:-cifar10}
dnn=${dnn:-gpt2}
model_dir=${model_dir:-gpt2}

case "$cluster_name" in
    "localhost")
        case "$dataset" in
            "Tiny-ImageNet-200") data_dir="/datasets/tiny-imagenet-200" ;;
            "cifar10") data_dir="/datasets/cifar10" ;;
            "cifar100") data_dir="/datasets/cifar100" ;;
            "fmnist") data_dir="/datasets/fmnist" ;;
            "SVHN") data_dir="/datasets/SVHN" ;;
            "mnist") data_dir="/datasets" ;;
        esac
        ;;
    "gpuhome")
        case "$dataset" in
            "cifar10") data_dir="/home/comp/amelieczhou/datasets/cifar10" ;;
        esac
        ;;
    "GZ4090")
        case "$dataset" in
            "cifar10") data_dir="/data2/share/cifar10" ;;
            "cifar100") data_dir="/data2/share/zhtang/cifar100" ;;
            # "wikitext2") data_dir="/mnt/raid/tangzichen/wikitext2" ;;
            "wikitext2") data_dir="/data2/share/zhtang/wikitext2" ;;
            "tatsu-lab/alpaca") data_dir="/data2/share/zhtang/tatsu-lab/alpaca" ;;
            *) echo "No dataset matched" ;;
        esac
        case "$dnn" in
            "gpt2") model_dir="/data2/share/zhtang/gpt2" ;;
            # "gpt2") model_dir="/mnt/raid/tangzichen/gpt2" ;;
            # "bert-base-uncased") model_dir="/mnt/raid/tangzichen/bert-base-uncased" ;;
            "bert-base-uncased") model_dir="/data2/share/zhtang/bert-base-uncased" ;;
            "llama2-7B") model_dir="/data2/share/llama/Llama-2-7b-hf" ;;
            "llama2-124M") model_dir="/data2/share/zhtang/llama-2-7b-hf" ;;
            *) echo "No DNN matched" ;;
        esac
        ;;
    "GZA6000")
        case "$dataset" in
            "cifar10") data_dir="/data2/share/cifar10" ;;
            "cifar100") data_dir="/data2/share/zhtang/cifar100" ;;
            # "wikitext2") data_dir="/mnt/raid/tangzichen/wikitext2" ;;
            "wikitext2") data_dir="/data2/share/zhtang/wikitext2" ;;
            "tatsu-lab/alpaca") data_dir="/data2/share/zhtang/tatsu-lab/alpaca" ;;
            *) echo "No dataset matched" ;;
        esac
        case "$dnn" in
            # "gpt2") model_dir="/mnt/raid/tangzichen/gpt2" ;;
            # "gpt2") model_dir="/data2/share/zhtang/gpt2" ;;
            "gpt2") model_dir="/data2/share/zhtang/newgpt2/gpt2" ;;
            "bert-base-uncased") model_dir="/mnt/raid/tangzichen/bert-base-uncased" ;;
            "llama2-7B") model_dir="/data2/share/llama/Llama-2-7b-hf" ;;
            "llama2-124M") model_dir="/data2/share/zhtang/llama-2-7b-hf" ;;
            *) echo "No DNN matched" ;;
        esac
        ;;
    "A6000")
        case "$dataset" in
            "cifar10") data_dir="/data2/share/cifar10" ;;
            "cifar100") data_dir="/data2/share/zhtang/cifar100" ;;
            # "wikitext2") data_dir="/mnt/raid/tangzichen/wikitext2" ;;
            "wikitext2") data_dir="/workspace/wikitext2" ;;
            "openwebtext") data_dir="/workspace/encoded_openwebtext" ;;
            "alpaca") data_dir="/workspace/alpaca" ;;
            # "openwebtext") data_dir="/workspace/openwebtext" ;;
            *) echo "No dataset matched" ;;
        esac
        case "$dnn" in
            "gpt2") model_dir="/workspace/models/gpt2" ;;
            # "gpt2") model_dir="/mnt/raid/tangzichen/gpt2" ;;
            # "bert-base-uncased") model_dir="/mnt/raid/tangzichen/bert-base-uncased" ;;
            "bert-base-uncased") model_dir="/data2/share/zhtang/bert-base-uncased" ;;
            "llama2-7B") model_dir="/workspace/models/Llama-2-7b-hf" ;;
            "llama2-124M") model_dir="/workspace/models/Llama-2-7b-hf" ;;
            *) echo "No DNN matched" ;;
        esac
        ;;
    "scigpu")
        case "$dataset" in
            "ILSVRC2012-100"|"ILSVRC2012") data_dir="/home/datasets/imagenet/ILSVRC2012_dataset" ;;
            "Tiny-ImageNet-200") data_dir="/home/comp/20481896/datasets/tiny-imagenet-200" ;;
            "gld23k") data_dir="~/datasets/landmarks" ;;
            "cifar10") data_dir="~/datasets/cifar10" ;;
            "SVHN") data_dir="~/datasets/SVHN" ;;
            "cifar100") data_dir="~/datasets/cifar100" ;;
            "fmnist") data_dir="~/datasets/fmnist" ;;
            "femnist") data_dir="/home/comp/20481896/datasets/fed_emnist" ;;
            "femnist-digit") data_dir="/home/comp/20481896/datasets/femnist" ;;
            "mnist") data_dir="~/datasets" ;;
            "ptb") data_dir="/home/comp/20481896/datasets/PennTreeBank" ;;
            "shakespeare") data_dir="/home/comp/20481896/datasets/shakespeare" ;;
        esac
        ;;
    "esetstore")
        case "$dataset" in
            "ILSVRC2012-100"|"ILSVRC2012") data_dir="/home/esetstore/dataset/ILSVRC2012_dataset" ;;
            "Tiny-ImageNet-200") data_dir="/home/esetstore/dataset/tiny-imagenet-200" ;;
            "gld23k") data_dir="/home/esetstore/dataset/landmarks" ;;
            "cifar10") data_dir="/home/esetstore/dataset/cifar10" ;;
            "cifar100") data_dir="/home/esetstore/dataset/cifar100" ;;
            "fmnist") data_dir="/home/esetstore/dataset/fmnist" ;;
            "femnist") data_dir="/home/zhtang/zhtang_data/datasets" ;;
            "mnist") data_dir="/home/esetstore/dataset" ;;
            "SVHN") data_dir="/home/esetstore/dataset/SVHN" ;;
            "ptb") data_dir="/home/esetstore/repos/p2p/data/PennTreeBank" ;;
            "shakespeare") data_dir="/home/esetstore/dataset/shakespeare" ;;
        esac
        ;;
    *)
        echo "Unknown cluster name: $cluster_name"
        exit 1
        ;;
esac












