#!/bin/bash
set -e

echo "=== Docker Full Cleanup Script ==="

# Detect Docker root dir dynamically (default /var/lib/docker)
DOCKER_ROOT="$(docker info --format '{{.DockerRootDir}}' 2>/dev/null || echo /var/lib/docker)"
echo "Docker root: $DOCKER_ROOT"

# 1. Force-remove ALL containers (running + stopped) and ALL images
echo "[1/5] Removing all containers and images..."
docker ps -aq | xargs -r docker rm -f 2>/dev/null || true
docker images -aq | xargs -r docker rmi -f 2>/dev/null || true
docker system prune -a --volumes -f 2>/dev/null || true
docker builder prune -a -f 2>/dev/null || true

# 2. Stop Docker
echo "[2/5] Stopping Docker..."
sudo systemctl stop docker docker.socket

# 3. Unmount overlay2 filesystems, then remove all data in Docker root directory
echo "[3/5] Unmounting overlay2 and removing $DOCKER_ROOT contents..."
sudo umount "$DOCKER_ROOT"/overlay2/*/merged 2>/dev/null || true
sudo rm -rf "${DOCKER_ROOT:?}"/*

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
sudo du -sh "$DOCKER_ROOT"
