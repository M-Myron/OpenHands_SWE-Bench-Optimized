# Orphaned Process Cleanup

## Problem

When you interrupt `run_infer.sh` or `rollout_swegym.sh` (using Ctrl+C or kill), the background Docker cleanup processes continue running indefinitely. These processes:

- Run every 30 minutes executing `docker prune` commands
- Consume system resources
- Can interfere with subsequent evaluation runs
- Create confusion about what's actually running

## Solution

### Automatic Cleanup (Improved Scripts)

The scripts have been updated to use `trap` to automatically kill background processes when interrupted:

```bash
# This now happens automatically in run_infer.sh and rollout_swegym.sh
trap "kill $CLEANUP_PID 2>/dev/null || true" EXIT INT TERM
```

**This works for:**
- Normal script completion
- Ctrl+C (SIGINT)
- `kill` command (SIGTERM)
- Script errors (EXIT)

### Manual Cleanup (For Existing Orphans)

If you already have orphaned processes from previous runs, use the cleanup script:

```bash
cd /home/v-murongma/code/OpenHands_SWE-Bench-Optimized/evaluation/benchmarks/swe_bench_optimized/scripts

# Run the cleanup script
./cleanup_orphaned_processes.sh
```

The script will:
1. ✓ Find and kill Docker cleanup loop processes
2. ✓ Kill orphaned sleep processes (from the 1800s sleep in the loop)
3. ✓ Kill stuck docker prune commands
4. ✓ Kill bash processes running the cleanup loops
5. ✓ Show you remaining processes for verification

## Usage Examples

### Check for Orphaned Processes

```bash
# Check for docker prune processes
pgrep -f "docker.*prune" || echo "None found"

# Check for cleanup loop processes  
ps aux | grep -E "docker container prune|docker image prune" | grep -v grep
```

### Clean Up Orphaned Processes

```bash
# Simple cleanup
./cleanup_orphaned_processes.sh

# Verify cleanup worked
pgrep -f "docker.*prune" || echo "All cleaned up!"
```

### Before Starting New Evaluation

```bash
# 1. Clean up any orphans from previous runs
./cleanup_orphaned_processes.sh

# 2. Run your evaluation (now with automatic cleanup on interrupt)
./run_infer.sh ...
```

## How It Works

### The Background Cleanup Loop (in run_infer.sh)

```bash
(
  while true; do
    sleep 1800  # 30 minutes
    docker container prune -f
    docker image prune -a -f --filter "until=30m"
    docker builder prune -f --filter "until=30m"
  done
) &
CLEANUP_PID=$!
```

### The Trap (Now Added)

```bash
trap "kill $CLEANUP_PID 2>/dev/null || true; echo 'Cleanup process stopped'" EXIT INT TERM
```

This ensures the background process is killed when:
- Script finishes normally (EXIT)
- User presses Ctrl+C (INT)
- Script is killed (TERM)

### The Cleanup Script

Finds processes by pattern matching:
- `docker container prune`
- `docker image prune`
- `docker builder prune`
- Associated `sleep 1800` commands
- Parent bash processes running the loops

## Troubleshooting

### Processes Still Running After Cleanup

```bash
# Force kill all docker prune processes
pkill -9 -f "docker.*prune"

# Check again
pgrep -f "docker.*prune" || echo "All gone!"
```

### Verify No Orphans Before Long Run

```bash
# Before starting a long evaluation run
./cleanup_orphaned_processes.sh

# Verify
ps aux | grep docker | grep prune | grep -v grep
# Should show nothing

# Now safe to start
./run_infer.sh ...
```

### Multiple Cleanup Processes Found

This is normal if you've killed the script multiple times. Each interruption leaves one orphaned process group. The cleanup script handles multiple orphans automatically.

## Prevention

**New runs won't create orphans** because the scripts now use `trap` to clean up automatically. But if you have old orphans from before this fix, run the cleanup script once.

## Files Modified

1. **`run_infer.sh`** - Added trap to kill cleanup process on exit/interrupt
2. **`rollout_swegym.sh`** - Added trap to kill cleanup process on exit/interrupt  
3. **`cleanup_orphaned_processes.sh`** - New script to clean existing orphans

## Quick Reference

```bash
# Find orphans
pgrep -f "docker.*prune"

# Clean orphans
./cleanup_orphaned_processes.sh

# Verify clean
ps aux | grep "docker.*prune" | grep -v grep || echo "Clean!"
```
