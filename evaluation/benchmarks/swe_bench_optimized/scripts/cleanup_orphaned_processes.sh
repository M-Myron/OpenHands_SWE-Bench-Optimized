#!/usr/bin/env bash
# Script to clean up orphaned Docker cleanup processes left behind when run_infer.sh is killed
# These are background processes that run "docker container prune", "docker image prune", etc.

set -e

echo "=============================================================="
echo "Cleaning Up Orphaned Docker Cleanup Processes"
echo "=============================================================="
echo ""

# Function to find and kill processes
cleanup_processes() {
    local pattern="$1"
    local description="$2"
    
    # Find PIDs matching the pattern
    PIDS=$(pgrep -f "$pattern" 2>/dev/null || true)
    
    if [ -n "$PIDS" ]; then
        echo "Found $description:"
        ps -f -p $PIDS 2>/dev/null || true
        echo ""
        echo "Killing processes..."
        kill $PIDS 2>/dev/null || true
        sleep 1
        
        # Force kill if still running
        REMAINING=$(pgrep -f "$pattern" 2>/dev/null || true)
        if [ -n "$REMAINING" ]; then
            echo "Force killing remaining processes..."
            kill -9 $REMAINING 2>/dev/null || true
        fi
        echo "✓ Cleaned up $description"
    else
        echo "No $description found"
    fi
    echo ""
}

# Clean up Docker cleanup loop processes
echo "1. Checking for Docker cleanup loop processes..."
echo "--------------------------------------------------------------"
cleanup_processes "docker container prune" "Docker cleanup loop processes"

# Clean up any sleep processes from the cleanup loop
echo "2. Checking for orphaned sleep processes (from cleanup loops)..."
echo "--------------------------------------------------------------"
# Be more specific to avoid killing unrelated sleep processes
cleanup_processes "sleep 1800.*docker.*prune" "orphaned sleep processes from Docker cleanup"

# Clean up any stuck docker prune commands
echo "3. Checking for stuck docker prune commands..."
echo "--------------------------------------------------------------"
cleanup_processes "docker.*prune -" "stuck docker prune commands"

# Clean up bash processes running the cleanup loop
echo "4. Checking for bash cleanup loop processes..."
echo "--------------------------------------------------------------"
# Look for bash processes with the specific cleanup loop pattern
BASH_CLEANUP_PIDS=$(ps aux | grep -E "bash.*docker.*prune|while true.*docker" | grep -v grep | awk '{print $2}' || true)
if [ -n "$BASH_CLEANUP_PIDS" ]; then
    echo "Found bash cleanup processes: $BASH_CLEANUP_PIDS"
    kill $BASH_CLEANUP_PIDS 2>/dev/null || true
    sleep 1
    # Force kill if needed
    REMAINING=$(ps aux | grep -E "bash.*docker.*prune|while true.*docker" | grep -v grep | awk '{print $2}' || true)
    if [ -n "$REMAINING" ]; then
        kill -9 $REMAINING 2>/dev/null || true
    fi
    echo "✓ Cleaned up bash cleanup processes"
else
    echo "No bash cleanup processes found"
fi
echo ""

# Show remaining docker-related background processes (for verification)
echo "=============================================================="
echo "Remaining Docker-related processes:"
echo "=============================================================="
ps aux | grep -E "docker.*(prune|container|image)" | grep -v grep || echo "None found"
echo ""

# Show process tree for any remaining cleanup-related processes
echo "=============================================================="
echo "Process tree check:"
echo "=============================================================="
REMAINING_CLEANUP=$(pgrep -f "docker.*prune" 2>/dev/null || true)
if [ -n "$REMAINING_CLEANUP" ]; then
    echo "Warning: Some cleanup processes still running:"
    pstree -p $REMAINING_CLEANUP 2>/dev/null || ps -f -p $REMAINING_CLEANUP
else
    echo "✓ No cleanup processes found in process tree"
fi
echo ""

echo "=============================================================="
echo "Cleanup Complete!"
echo "=============================================================="
echo ""
echo "Summary:"
echo "  - Killed orphaned Docker cleanup loop processes"
echo "  - Killed associated sleep and prune commands"
echo "  - Cleaned up bash wrapper processes"
echo ""
echo "You can now safely run new evaluations."
echo "=============================================================="
