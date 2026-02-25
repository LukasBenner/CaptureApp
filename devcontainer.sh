#!/usr/bin/env bash
set -e 

CONTAINER="captureapp"
SERVICE="captureapp"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.yml"

if [ "$EUID" -ne 0 ]; then
  echo "This script has to be executed as root or with sudo"
  exit 1
fi

echo "Setting trigger mode..."
v4l2-ctl -c trigger_mode=1
v4l2-ctl -c gain=0

if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
  echo "Container is already running → connecting..."
  docker compose -f "$COMPOSE_FILE" exec "$SERVICE" /bin/bash
  exit
fi

echo "Container is not running → Starting..."
docker compose -f "$COMPOSE_FILE" up -d
docker compose -f "$COMPOSE_FILE" exec "$SERVICE" /bin/bash
