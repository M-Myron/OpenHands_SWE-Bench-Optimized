#!/usr/bin/env bash
set -euo pipefail

echo "[INFO] Setting up Docker-in-Docker inside this container..."

# 1. Install Docker if missing
if ! command -v docker >/dev/null 2>&1; then
  echo "[INFO] Docker not found; installing via apt..."
  apt-get update
  # You can switch to docker-ce if you prefer; docker.io is enough to debug
  apt-get install -y docker.io
else
  echo "[INFO] Docker client already present at: $(command -v docker)"
fi

# 2. Start dockerd (Docker daemon) inside this container
mkdir -p /var/lib/docker

if pgrep -x dockerd >/dev/null 2>&1; then
  echo "[INFO] dockerd already running."
else
  echo "[INFO] Starting dockerd..."
  dockerd --host=unix:///var/run/docker.sock --storage-driver=vfs >/var/log/dockerd.log 2>&1 &
fi

# 3. Wait for dockerd to become ready
echo "[INFO] Waiting for Docker daemon to become ready..."
for i in $(seq 1 30); do
  if docker info >/dev/null 2>&1; then
    echo "[INFO] Docker daemon is up (docker info succeeded)."
    break
  fi
  echo "[INFO] Still waiting for dockerd... (${i}/30)"
  sleep 1
done

if ! docker info >/dev/null 2>&1; then
  echo "[ERROR] Docker daemon failed to start. Dumping log:"
  cat /var/log/dockerd.log || true
  exit 1
fi

echo "[INFO] Docker-in-Docker setup complete."
echo "[INFO] Running command: $*"
exec "$@"
