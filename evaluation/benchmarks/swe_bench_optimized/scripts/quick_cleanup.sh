#!/usr/bin/env bash
# Quick cleanup - kills all orphaned docker cleanup processes
# Usage: ./quick_cleanup.sh

echo "Killing orphaned docker cleanup processes..."
pkill -f "docker container prune" 2>/dev/null || true
pkill -f "docker image prune" 2>/dev/null || true
pkill -f "docker builder prune" 2>/dev/null || true

# Verify
REMAINING=$(pgrep -f "docker.*prune" 2>/dev/null || true)
if [ -z "$REMAINING" ]; then
    echo "✓ All orphaned processes cleaned up!"
else
    echo "⚠ Some processes still running, trying force kill..."
    pkill -9 -f "docker.*prune" 2>/dev/null || true
    sleep 1
    STILL_REMAINING=$(pgrep -f "docker.*prune" 2>/dev/null || true)
    if [ -z "$STILL_REMAINING" ]; then
        echo "✓ Force kill successful!"
    else
        echo "✗ Some processes still running. Run cleanup_orphaned_processes.sh for detailed cleanup."
    fi
fi
