
    #Res18
    # 0.1
    build_run("hpml-hkbu/DDP-Train/4okbu1kk", CIFAR10_RES18,
    {"": ""}, "sgd-noiTrue-tburst-resnet18-nw8-SGD-LG20-lr0.1-bs128-0.1burst")

    # 1.0
    build_run("hpml-hkbu/DDP-Train/egqsg78n", CIFAR10_RES18,
    {"": ""}, "sgd-noiTrue-tburst-resnet18-nw8-SGD-LG20-lr0.1-bs128-1.0burst")

    # fix
    # 0.1 
    build_run("hpml-hkbu/DDP-Train/zppeqg6s", CIFAR10_RES18,
    {"": ""}, "sgd_with_sync-noiTrue-tburst-resnet18-nw8-SGD-LG20-lr0.1-bs128-nstd0.0001-SyncP5-fix-0.1burst")
    build_run("hpml-hkbu/DDP-Train/gqlq6yjt", CIFAR10_RES18,
    {"": ""}, "sgd_with_sync-noiTrue-tburst-resnet18-nw8-SGD-LG20-lr0.1-bs128-nstd0.0001-SyncP50-fix-0.1burst")
    build_run("hpml-hkbu/DDP-Train/vt91s6yg", CIFAR10_RES18,
    {"": ""}, "sgd_with_sync-noiTrue-tburst-resnet18-nw8-SGD-LG20-lr0.1-bs128-nstd0.0001-SyncP10-fix-0.1burst")

    # 1.0
    build_run("hpml-hkbu/DDP-Train/q9sa9h4j", CIFAR10_RES18,
    {"": ""}, "sgd_with_sync-noiTrue-tburst-resnet18-nw8-SGD-LG20-lr0.1-bs128-nstd0.0001-SyncP50-fix-1.0burst")
    build_run("hpml-hkbu/DDP-Train/5m3ieefc", CIFAR10_RES18,
    {"": ""}, "sgd_with_sync-noiTrue-tburst-resnet18-nw8-SGD-LG20-lr0.1-bs128-nstd0.0001-SyncP10-fix-1.0burst")
    build_run("hpml-hkbu/DDP-Train/djzds6cs", CIFAR10_RES18,
    {"": ""}, "sgd_with_sync-noiTrue-tburst-resnet18-nw8-SGD-LG20-lr0.1-bs128-nstd0.0001-SyncP5-fix-1.0burst")

    # sync
    # 0.1
    build_run("hpml-hkbu/DDP-Train/j6dudqku", CIFAR10_RES18,
    {"": ""}, "sgd_with_sync-noiTrue-tburst-resnet18-nw8-SGD-LG20-lr0.1-bs128-nstd0.0001-SyncP10-sync-0.1burst")
    # 1.0
    build_run("hpml-hkbu/DDP-Train/wmd0n193", CIFAR10_RES18,
    {"": ""}, "sgd_with_sync-noiTrue-tburst-resnet18-nw8-SGD-LG20-lr0.1-bs128-nstd0.0001-SyncP10-sync-1.0burst")
