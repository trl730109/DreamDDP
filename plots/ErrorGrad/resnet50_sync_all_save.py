# Resnet-50 4-worker
# Sgd with sync all detect base
build_run("hpml-hkbu/DDP-Train/4-worker", CIFAR10_RES18,
{"": ""}, "Resnet-50")

build_run("hpml-hkbu/DDP-Train/y6eys79k", CIFAR10_RES18,
{"": ""}, "sgd_with_sync_all-noiTrue-tfix-resnet50-lora-nw4-SGD-LG20-lr0.1-bs128-nstd0.01-SyncP5-cifar100")
build_run("hpml-hkbu/DDP-Train/tm6fivg0", CIFAR10_RES18,
{"": ""}, "sgd_with_sync_all-noiTrue-tfix-resnet50-lora-nw4-SGD-LG20-lr0.1-bs128-nstd0.01-SyncP10-cifar100")
build_run("hpml-hkbu/DDP-Train/abj5w7vq", CIFAR10_RES18,
{"": ""}, "sgd_with_sync_all-noiTrue-tfix-resnet50-lora-nw4-SGD-LG20-lr0.1-bs128-nstd0.01-SyncP50-cifar100")

# Fix
build_run("hpml-hkbu/DDP-Train/0wqzkelw", CIFAR10_RES18,
{"": ""}, "sgd_with_sync_all-noiTrue-tfix-resnet50-lora-nw4-SGD-LG20-lr0.1-bs128-nstd0.01-SyncP5-cifar100")
build_run("hpml-hkbu/DDP-Train/wstpw6cv", CIFAR10_RES18,
{"": ""}, "sgd_with_sync_all-noiTrue-tfix-resnet50-lora-nw4-SGD-LG20-lr0.1-bs128-nstd0.01-SyncP10-cifar100")
build_run("hpml-hkbu/DDP-Train/n72z7ily", CIFAR10_RES18,
{"": ""}, "sgd_with_sync_all-noiTrue-tfix-resnet50-lora-nw4-SGD-LG20-lr0.1-bs128-nstd0.01-SyncP50-cifar100")
