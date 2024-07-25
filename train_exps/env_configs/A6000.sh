PY="${PY:-/workspace/pretrain/miniconda3/envs/pretrain/bin/python3}"


cluster_name=${cluster_name:-localhost}
dataset=${dataset:-cifar10}
#echo "cluster_name: $cluster_name"
case "$cluster_name" in
    "localhost")
        case "$dataset" in
            "Tiny-ImageNet-200") data_dir="/datasets/tiny-imagenet-200" ;;
            "cifar10") data_dir="/home/yinyiming/datasets/cifar10" ;;
            "cifar100") data_dir="/datasets/cifar100" ;;
            "fmnist") data_dir="/datasets/fmnist" ;;
            "SVHN") data_dir="/datasets/SVHN" ;;
            "mnist") data_dir="/datasets" ;;
        esac
        ;;
    "shenzhen")
        case "$dataset" in
            # "Tiny-ImageNet-200") data_dir="/datasets/tiny-imagenet-200" ;;
            # "cifar10") data_dir="/home/yinyiming/datasets/cifar10" ;;
            "cifar10") data_dir="/home/yinyiming/datasets/cifar10" ;;
            "cifar100") data_dir="/home/yinyiming/datasets/cifar100" ;;
            # "fmnist") data_dir="/datasets/fmnist" ;;
            # "SVHN") data_dir="/datasets/SVHN" ;;
            # "mnist") data_dir="/datasets" ;;
        esac
        ;;
    "A6000")
        case "$dataset" in
            # "Tiny-ImageNet-200") data_dir="/datasets/tiny-imagenet-200" ;;
            # "cifar10") data_dir="/home/yinyiming/datasets/cifar10" ;;
            "cifar10") data_dir="/home/yinyiming/datasets/cifar10" ;;
            "cifar100") data_dir="/home/yinyiming/datasets/cifar100" ;;
            "wikitext2") data_dir="workspace/wikitext2" ;;
            # "SVHN") data_dir="/datasets/SVHN" ;;
            # "mnist") data_dir="/datasets" ;;
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











