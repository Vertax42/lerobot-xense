#!/bin/bash

# bash scripts for ARX5 CAN interface configuration
# function: configure can1, can3 interface to 1000000 baud rate and start

echo "=== ARX5 CAN interface configuration script ==="
echo "Interface: can1, can3"
echo "Baud rate: 1000000 bps"
echo "================================================"

# stop the processes that occupy the CAN interface
echo "Stopping the processes that occupy the CAN interface..."
echo "Checking and stopping slcand process..."

# check and stop slcand process
if pgrep slcand >/dev/null 2>&1; then
    echo "slcand process found, stopping..."
    sudo pkill slcand
    sleep 1
    
    # verify if the process has been stopped
    if pgrep slcand >/dev/null 2>&1; then
        echo "Warning: slcand process is still running, trying to force stop..."
        sudo pkill -9 slcand
        sleep 1
    fi
    
    if ! pgrep slcand >/dev/null 2>&1; then
        echo "✓ slcand process has been successfully stopped"
    else
        echo "✗ cannot stop slcand process, please check manually"
    fi
else
    echo "✓ slcand process not found"
fi

# check and stop arx_can related processes
echo "Checking and stopping arx_can related processes..."
if pgrep -f "arx_can.*\.sh" >/dev/null 2>&1; then
    echo "arx_can script process found, stopping..."
    sudo pkill -f "arx_can.*\.sh"
    sleep 1
    
    if ! pgrep -f "arx_can.*\.sh" >/dev/null 2>&1; then
        echo "✓ arx_can script process has been successfully stopped"
    else
        echo "✗ cannot stop arx_can script process, please check manually"
    fi
else
    echo "✓ arx_can script process not found"
fi

echo ""

# use SLCAN method to configure CAN interface
echo "Using SLCAN method to configure CAN interface..."

# function: use SLCAN method to configure single CAN interface
configure_slcan_interface() {
    local device=$1
    local interface=$2
    
    echo "Configuring $interface interface (device: $device)..."
    
    # check if the device exists
    if [ ! -e "$device" ]; then
        echo "  ✗ device $device does not exist, skipping"
        return 1
    fi
    
    # use slcand command to create CAN interface
    echo "  creating $interface interface using slcand..."
    if sudo slcand -o -f -s8 "$device" "$interface" 2>/dev/null; then
        echo "  ✓ slcand created $interface successfully"
    else
        echo "  ✗ slcand created $interface failed"
        return 1
    fi
    
    # wait for interface creation
    sleep 1
    
    # start interface
    echo "  starting $interface..."
    if sudo ifconfig "$interface" up 2>/dev/null; then
        echo "  ✓ $interface started successfully"
    else
        echo "  ✗ $interface started failed"
        return 1
    fi
    
    # verify configuration
    if ip link show "$interface" >/dev/null 2>&1; then
        local current_state=$(ip -details link show "$interface" | grep -o "state [A-Z-]*" | cut -d' ' -f2)
        echo "  ✓ $interface configuration verified successfully (state: $current_state, baud rate: 1000000 bps via SLCAN)"
        return 0
    else
        echo "  ✗ $interface configuration verified failed"
        return 1
    fi
}

# configure all CAN interfaces
devices=("/dev/arxcan1" "/dev/arxcan3")
interfaces=("can1" "can3")
success_count=0
total_count=${#interfaces[@]}

echo "Starting to configure $total_count CAN interfaces..."
for i in "${!interfaces[@]}"; do
    echo ""
    if configure_slcan_interface "${devices[$i]}" "${interfaces[$i]}"; then
        ((success_count++))
    fi
done

echo "================================"
echo "Configuration completed!"
echo "Successfully configured: $success_count/$total_count interfaces"

if [ $success_count -eq $total_count ]; then
    echo "✓ All CAN interfaces configured successfully"
    echo ""
    echo "Current CAN interface status:"
    for interface in "${interfaces[@]}"; do
        if ip link show "$interface" >/dev/null 2>&1; then
            bitrate=$(ip -details link show "$interface" | grep -o "bitrate [0-9]*" | cut -d' ' -f2)
            state=$(ip -details link show "$interface" | grep -o "state [A-Z-]*" | cut -d' ' -f2)
            echo "  $interface: baud rate=$bitrate, state=$state"
        fi
    done
    exit 0
else
    echo "✗ some CAN interfaces configuration failed"
    echo "Please check if the interface exists or if there are other processes occupying it"
    exit 1
fi
