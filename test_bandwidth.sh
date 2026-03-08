#!/bin/bash
# 集群节点间带宽测试脚本
# Usage:
#   1. 在本地运行（会 SSH 到各节点）:
#      bash test_bandwidth.sh              # 使用默认 hosts
#      bash test_bandwidth.sh server        # 仅在当前机启动 iperf3 服务端
#      bash test_bandwidth.sh client <ip>   # 从当前机测到 <ip>
#
#   2. 全自动测试（推荐）:
#      bash test_bandwidth.sh auto          # SSH 到各节点，自动测所有节点对

hosts=('10.244.3.188' '10.244.4.109')
ports=(22 22)
IPERF_PORT=5201
DURATION=5

# 检查 iperf3 是否安装
check_iperf3() {
    if ! command -v iperf3 &> /dev/null; then
        echo "iperf3 未安装，尝试安装..."
        sudo apt-get update -qq && sudo apt-get install -y iperf3 2>/dev/null || {
            echo "请手动安装: sudo apt-get install iperf3"
            exit 1
        }
    fi
}

# 在远程节点启动 iperf3 服务端
start_server_remote() {
    local host=$1
    local port=${2:-22}
    echo ">>> 在 $host 启动 iperf3 服务端 (端口 $IPERF_PORT)..."
    ssh -p $port -o ConnectTimeout=5 $host "pkill iperf3 2>/dev/null; nohup iperf3 -s -D -p $IPERF_PORT > /dev/null 2>&1; sleep 1; pgrep iperf3 && echo 'Server started' || echo 'Failed to start'"
}

# 从 client_host 测到 server_host
test_bw() {
    local client_host=$1
    local client_port=$2
    local server_host=$3
    local server_port=$4
    echo ""
    echo "========== $client_host -> $server_host =========="
    ssh -p $client_port -o ConnectTimeout=5 $client_host "iperf3 -c $server_host -p $IPERF_PORT -t $DURATION -f m" 2>/dev/null || echo "测试失败"
}

# 全自动模式：测所有节点对
run_auto() {
    check_iperf3
    total=${#hosts[@]}
    
    echo "集群节点: ${hosts[*]}"
    echo "测试时长: ${DURATION}s, iperf 端口: $IPERF_PORT"
    echo ""
    
    for ((i=0; i<total; i++)); do
        # 在节点 i 上启动服务端
        start_server_remote ${hosts[$i]} ${ports[$i]}
        sleep 2
        
        # 从其他节点测到节点 i
        for ((j=0; j<total; j++)); do
            [ $i -eq $j ] && continue
            test_bw ${hosts[$j]} ${ports[$j]} ${hosts[$i]} ${ports[$i]}
        done
        
        # 停止节点 i 上的服务端
        ssh -p ${ports[$i]} ${hosts[$i]} "pkill iperf3 2>/dev/null" 2>/dev/null
    done
    
    echo ""
    echo "========== 测试完成 =========="
}

# 本地 server 模式（在当前机启动 iperf3 -s）
run_server() {
    check_iperf3
    pkill iperf3 2>/dev/null
    echo "启动 iperf3 服务端，端口 $IPERF_PORT..."
    iperf3 -s -p $IPERF_PORT
}

# 本地 client 模式（从当前机测到目标 IP）
run_client() {
    local target=${1:-127.0.0.1}
    check_iperf3
    echo "测试 本机 -> $target (${DURATION}s)..."
    iperf3 -c $target -p $IPERF_PORT -t $DURATION -f m
}

# Main
case "${1:-auto}" in
    auto)   run_auto ;;
    server) run_server ;;
    client) run_client "$2" ;;
    *)
        echo "Usage: $0 {auto|server|client [target_ip]}"
        echo "  auto   - 自动 SSH 到各节点测试所有节点对 (默认)"
        echo "  server - 在当前机启动 iperf3 服务端"
        echo "  client <ip> - 从当前机测到目标 IP"
        exit 1
        ;;
esac
