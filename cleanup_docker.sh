#!/bin/bash
set -e

echo "=== Docker Full Cleanup Script ==="

# 1. Prune all Docker resources (images, containers, volumes, build cache)
echo "[1/5] Pruning all Docker resources..."
docker system prune -a --volumes -f 2>/dev/null || true
docker builder prune -a -f 2>/dev/null || true

# 2. Stop Docker
echo "[2/5] Stopping Docker..."
sudo systemctl stop docker docker.socket

# 3. Unmount overlay2 filesystems, then remove all data in Docker root directory
echo "[3/5] Unmounting overlay2 and removing /datadisk/docker/ contents..."
sudo umount /datadisk/docker/overlay2/*/merged 2>/dev/null || true
sudo rm -rf /datadisk/docker/*

# 4. Restart Docker
echo "[4/5] Starting Docker..."
sudo systemctl start docker

# 5. Fix permissions for current user
echo "[5/5] Fixing Docker socket permissions..."
sudo usermod -aG docker "$(whoami)"
sudo chmod 666 /var/run/docker.sock

echo ""
echo "=== Done ==="
docker system df
sudo du -sh /datadisk/docker/
