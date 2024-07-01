# PY="${PY:-/home/comp/20481896/anaconda3/envs/py36/bin/python}"
PY="${PY:-/home/yinyiming/miniconda3/envs/ddp/bin/python3}"


cluster_name=${cluster_name:-localhost}
dataset=${dataset:-cifar10}

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











