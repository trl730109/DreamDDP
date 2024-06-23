PY="${PY:-~/miniconda3/envs/DDP/bin/python3}"


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
            "cifar10") datadir="/home/comp/amelieczhou/datasets/cifar10" ;;
        esac
        ;;
    *)
        echo "Unknown cluster name: $cluster_name"
        exit 1
        ;;
esac











