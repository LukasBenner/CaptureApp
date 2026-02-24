#!/usr/bin/env bash
set -e 

CONTAINER="multireaderdev" 
IMAGE="scirocco2017/multireaderdevcontainer:1" 

if [ "$EUID" -ne 0 ]; then
  echo "This script has to be executed as root or with sudo"
  exit 1
fi

echo "Setting trigger mode..."
v4l2-ctl -c trigger_mode=1
v4l2-ctl -c gain=0

# Prüfen, ob Container läuft
if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
    echo "Container is already running → connecting..."
    docker exec -it "$CONTAINER" /bin/bash
    exit
fi

echo "Container is not running → Starting..."
docker run -it \
  --name multireaderdev \
  --device /dev/video0 \
  --device /dev/i2c-7 \
  --device /dev/gpiochip2 \
  --device /dev/spidev0.0 \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -v /multireader-trigger:/multireader-trigger \
  -v /home/aesculap/DetectionDemo:/workspace/DetectionDemo \
  -v /home/aesculap/.ssh:/root/.ssh \
  -v /data/lens:/data/lens \
  -v /sys/class/hwmon/hwmon1/temp1_input:/sys/class/hwmon/hwmon1/temp1_input \
  -e GST_DEBUG=2 \
  -v /tmp/argus_socket:/tmp/argus_socket \
  -e XDG_RUNTIME_DIR=/tmp \
  -v /tmp:/tmp \
  -e DISPLAY \
  -e GST_INSTALL_DIR="/home/GX100_Application/gst-plugins" \
  -e GST_PLUGIN_PATH="/home/GX100_Application/gst-plugins" \
  -e PATH="$PATH:$HOME/.pub-cache/bin" \
  -e PATH="/opt/flutter/bin:/opt/flutter/bin/cache/dart-sdk/bin:${PATH}" \
  --workdir /workspace/DetectionDemo \
  "$IMAGE"
