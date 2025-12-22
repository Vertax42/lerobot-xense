#!/bin/bash

# ARX5 CAN接口配置脚本
# 功能：配置 can1, can3 接口为 1000000 波特率并启动

# set -e  # 遇到错误时退出 - 注释掉，让脚本继续执行所有接口配置

echo "=== ARX5 CAN接口配置脚本 ==="
echo "配置接口: can1, can3"
echo "波特率: 1000000 bps"
echo "================================"

# 停止占用CAN接口的进程
echo "停止占用CAN接口的进程..."
echo "检查并停止slcand进程..."

# 检查并停止slcand进程
if pgrep slcand >/dev/null 2>&1; then
    echo "发现slcand进程，正在停止..."
    sudo pkill slcand
    sleep 1
    
    # 验证进程是否已停止
    if pgrep slcand >/dev/null 2>&1; then
        echo "警告: slcand进程仍在运行，尝试强制停止..."
        sudo pkill -9 slcand
        sleep 1
    fi
    
    if ! pgrep slcand >/dev/null 2>&1; then
        echo "✅ slcand进程已成功停止"
    else
        echo "✗ 无法停止slcand进程，请手动检查"
    fi
else
    echo "✅ 未发现slcand进程"
fi

# 检查并停止arx_can相关进程
echo "检查并停止arx_can相关进程..."
if pgrep -f "arx_can.*\.sh" >/dev/null 2>&1; then
    echo "发现arx_can脚本进程，正在停止..."
    sudo pkill -f "arx_can.*\.sh"
    sleep 1
    
    if ! pgrep -f "arx_can.*\.sh" >/dev/null 2>&1; then
        echo "✅ arx_can脚本进程已成功停止"
    else
        echo "✗ 无法停止arx_can脚本进程，请手动检查"
    fi
else
    echo "✅ 未发现arx_can脚本进程"
fi

echo ""

# 使用SLCAN方法配置CAN接口
echo "使用SLCAN方法配置CAN接口..."

# 函数：使用SLCAN方法配置单个CAN接口
configure_slcan_interface() {
    local device=$1
    local interface=$2
    
    echo "配置 $interface 接口 (设备: $device)..."
    
    # 检查设备是否存在
    if [ ! -e "$device" ]; then
        echo "  ✗ 设备 $device 不存在，跳过"
        return 1
    fi
    
    # 使用slcand命令创建CAN接口
    echo "  使用slcand创建 $interface 接口..."
    if sudo slcand -o -f -s8 "$device" "$interface" 2>/dev/null; then
        echo "  ✅ slcand创建 $interface 成功"
    else
        echo "  ✗ slcand创建 $interface 失败"
        return 1
    fi
    
    # 等待接口创建
    sleep 1
    
    # 启动接口
    echo "  启动 $interface..."
    if sudo ifconfig "$interface" up 2>/dev/null; then
        echo "  ✅ $interface 启动成功"
    else
        echo "  ✗ $interface 启动失败"
        return 1
    fi
    
    # 验证配置
    if ip link show "$interface" >/dev/null 2>&1; then
        local current_state=$(ip -details link show "$interface" | grep -o "state [A-Z-]*" | cut -d' ' -f2)
        echo "  ✅ $interface 配置验证成功 (状态: $current_state, 波特率: 1000000 bps via SLCAN)"
        return 0
    else
        echo "  ✗ $interface 配置验证失败"
        return 1
    fi
}

# 配置所有CAN接口
devices=("/dev/arxcan1" "/dev/arxcan3")
interfaces=("can1" "can3")
success_count=0
total_count=${#interfaces[@]}

echo "开始配置 $total_count 个CAN接口..."
for i in "${!interfaces[@]}"; do
    echo ""
    if configure_slcan_interface "${devices[$i]}" "${interfaces[$i]}"; then
        ((success_count++))
    fi
done

echo "================================"
echo "配置完成!"
echo "成功配置: $success_count/$total_count 个接口"

if [ $success_count -eq $total_count ]; then
    echo "✅ 所有CAN接口配置成功"
    echo ""
    echo "当前CAN接口状态:"
    for interface in "${interfaces[@]}"; do
        if ip link show "$interface" >/dev/null 2>&1; then
            bitrate=$(ip -details link show "$interface" | grep -o "bitrate [0-9]*" | cut -d' ' -f2)
            state=$(ip -details link show "$interface" | grep -o "state [A-Z-]*" | cut -d' ' -f2)
            echo "  $interface: 波特率=$bitrate, 状态=$state"
        fi
    done
    exit 0
else
    echo "✗ 部分CAN接口配置失败"
    echo "请检查接口是否存在或是否有其他进程占用"
    exit 1
fi
