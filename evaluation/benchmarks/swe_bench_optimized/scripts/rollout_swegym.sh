#!/bin/bash

# NOTE: this script is for rolling out the SWE-Gym dataset for **TRAINING**
# For more information, please refer to
# 1. the Github Repo: https://github.com/SWE-Gym/SWE-Gym
# 2. the paper: https://arxiv.org/abs/2412.21139

MODEL=$1  # eg your llm config name in config.toml (eg: "llm.claude-3-5-sonnet-20241022-t05")
EXP_NAME=$2 # "train-t05"
N_WORKERS=${3:-64}
N_RUNS=${4:-1}

export EXP_NAME=$EXP_NAME
# use 2x resources for rollout since some codebases are pretty resource-intensive
export DEFAULT_RUNTIME_RESOURCE_FACTOR=2
export ITERATIVE_EVAL_MODE=false
# Skip instances that reach maximum retries instead of crashing the entire evaluation
# Failed instances will be logged to maximum_retries_exceeded.jsonl
export EVAL_SKIP_MAXIMUM_RETRIES_EXCEEDED=true

echo "=============================================================="
echo "SWE-Gym Rollout Configuration"
echo "=============================================================="
echo "MODEL: $MODEL"
echo "EXP_NAME: $EXP_NAME"
echo "N_WORKERS: $N_WORKERS"
echo "N_RUNS: $N_RUNS"
echo ""

# Pre-flight Docker check
echo "=== Pre-flight Docker Check ==="
DOCKER_PS_TIME=$(date +%s.%N)
docker ps > /dev/null 2>&1
DOCKER_PS_ELAPSED=$(awk "BEGIN {printf \"%.2f\", $(date +%s.%N) - $DOCKER_PS_TIME}")
echo "Docker daemon response time: ${DOCKER_PS_ELAPSED}s"

if (( $(echo "$DOCKER_PS_ELAPSED > 2.0" | bc -l) )); then
    echo "⚠️  WARNING: Docker daemon is slow to respond (>${DOCKER_PS_ELAPSED}s)"
    echo "   This may cause issues during evaluation"
fi

INITIAL_CONTAINERS=$(docker ps -q | wc -l)
INITIAL_IMAGES=$(docker images -q | wc -l)
echo "Initial state: $INITIAL_CONTAINERS containers, $INITIAL_IMAGES images"

# Check blob loading configuration if enabled
if [ -n "$BLOB_MOUNT_DIR" ]; then
    echo ""
    echo "=== Blob Image Loading Configuration ==="
    echo "BLOB_MOUNT_DIR: $BLOB_MOUNT_DIR"
    echo "BLOB_IMAGES_SUBDIR: ${BLOB_IMAGES_SUBDIR:-images}"
    echo "BLOB_LOAD_TIMEOUT: ${BLOB_LOAD_TIMEOUT:-1200}s"
    
    if [ -d "$BLOB_MOUNT_DIR" ]; then
        echo "✅ Blob mount directory exists"
        IMAGES_DIR="$BLOB_MOUNT_DIR/${BLOB_IMAGES_SUBDIR:-images}"
        if [ -d "$IMAGES_DIR" ]; then
            IMAGE_COUNT=$(find "$IMAGES_DIR" -name "*.tar.gz" 2>/dev/null | wc -l)
            echo "✅ Found $IMAGE_COUNT prebuilt images in blob storage"
        else
            echo "⚠️  WARNING: Images directory not found: $IMAGES_DIR"
        fi
    else
        echo "❌ ERROR: Blob mount directory not found: $BLOB_MOUNT_DIR"
    fi
fi

echo ""
echo "Docker disk usage:"
docker system df
echo "=============================================================="
echo ""

DATASET="SWE-Gym/SWE-Gym"  # change this to the "/SWE-Gym-Lite" if you want to rollout the lite subset
SPLIT="train"

if [ -z "$ALLHANDS_API_KEY" ]; then
    echo "ALLHANDS_API_KEY is not set. Will rollout and evaluate locally using Docker. WARNING: A large value of N_WORKERS will result in a large number of Docker containers being spun up and may crash your machine."
    export RUNTIME=docker
else
    echo "ALLHANDS_API_KEY is set. Continuing rollout and evaluation with remote runtime..."
    export RUNTIME=remote
    export SANDBOX_REMOTE_RUNTIME_API_URL="https://runtime.eval.all-hands.dev"
    export EVAL_DOCKER_IMAGE_PREFIX="us-central1-docker.pkg.dev/evaluation-092424/swe-bench-images"
fi

EVAL_LIMIT=3000
MAX_ITER=100


# ===== Run inference =====
source "evaluation/utils/version_control.sh"
get_openhands_version

echo "OPENHANDS_VERSION: $OPENHANDS_VERSION"
echo "MODEL_CONFIG: $MODEL_CONFIG"
echo "DATASET: $DATASET"
echo "SPLIT: $SPLIT"

# Default to NOT use Hint
export USE_INSTANCE_IMAGE=true
export USE_HINT_TEXT=false
export RUN_WITH_BROWSING=false
echo "USE_HINT_TEXT: $USE_HINT_TEXT"
EVAL_NOTE="$OPENHANDS_VERSION-no-hint-$EXP_NAME"

function run_eval() {
  local eval_note=$1
  local n_runs=$2
  
  # Start monitoring in background
  MONITOR_LOG="evaluation/evaluation_outputs/docker_monitor_$(date +%Y%m%d_%H%M%S).log"
  mkdir -p "$(dirname "$MONITOR_LOG")"
  echo "Starting Docker monitoring - logs: $MONITOR_LOG"
  
  (
    while true; do
      {
        echo "=== Docker Monitor - $(date '+%Y-%m-%d %H:%M:%S') ==="
        
        # Count docker load processes
        LOAD_COUNT=$(ps aux | grep -E "docker load|gzip.*docker" | grep -v grep | wc -l)
        echo "Active 'docker load' processes: $LOAD_COUNT"
        
        if [ $LOAD_COUNT -gt 0 ]; then
          echo "Process details:"
          ps aux | grep -E "docker load|gzip.*docker" | grep -v grep | awk '{printf "  PID: %-7s CPU: %-5s MEM: %-5s TIME: %-10s\n", $2, $3"%", $4"%", $10}' | head -10
          if [ $LOAD_COUNT -gt 10 ]; then
            echo "  ... and $((LOAD_COUNT - 10)) more"
          fi
        fi
        
        # Docker stats
        CONTAINERS=$(docker ps -q 2>/dev/null | wc -l)
        IMAGES=$(docker images -q 2>/dev/null | wc -l)
        echo "Docker: $CONTAINERS containers, $IMAGES images"
        
        # System resources
        echo "CPU load: $(uptime | awk -F'load average:' '{print $2}')"
        echo "Memory: $(free -h | grep Mem | awk '{print $3 "/" $2 " (" $3/$2*100 "%)"}')"
        
        # Docker disk usage
        echo "Docker disk usage:"
        docker system df --format "  {{.Type}}: {{.Size}} ({{.Reclaimable}} reclaimable)" 2>/dev/null || echo "  Unable to get disk usage"
        
        echo "---"
      } >> "$MONITOR_LOG" 2>&1
      sleep 30  # Update every 30 seconds
    done
  ) &
  MONITOR_PID=$!
  echo "Docker monitor started with PID: $MONITOR_PID"
  
  # Set up trap to kill monitor process on script exit or interruption
  trap "kill $MONITOR_PID 2>/dev/null || true; echo 'Docker monitor stopped'" EXIT INT TERM
  
  # Start periodic Docker cleanup in background (only for local docker runtime)
  if [ "$RUNTIME" = "docker" ]; then
    (
    #   while true; do
    #     sleep 1800  # Every 30 minutes
    #     echo "### Running periodic Docker cleanup during evaluation... ###"
    #     # Remove stopped containers
    #     docker ps -q --filter "name=openhands-runtime-" --filter "status=exited" | xargs -r docker rm 2>/dev/null || true
    #     # Remove SWE-bench evaluation images
    #     docker images --format "{{.Repository}}:{{.Tag}}" | grep -E "^(us-central1-docker\.pkg\.dev/evaluation-092424/swe-bench-images|docker\.io/xingyaoww/|mmr1115/openhands-runtime)" | xargs -r docker rmi -f 2>/dev/null || true
    #     # Prune dangling images and build cache
    #     docker image prune -f 2>/dev/null || true
    #     docker builder prune -f --filter "until=30m" 2>/dev/null || true
    #     # Show current usage
    #     echo "Current Docker usage:"
    #     docker system df
    #     echo "Running containers: $(docker ps -q | wc -l)"
    #     echo "Total images: $(docker images -q | wc -l)"
    #   done
    while true; do
        sleep 1800  # Every 30 minutes (more frequent)
        echo "Running periodic Docker cleanup..."
        # Remove stopped containers
        docker container prune -f 2>/dev/null || true
        # Remove ALL unused images older than 30 minutes (much more aggressive)
        docker image prune -a -f --filter "until=30m" 2>/dev/null || true
        # Also remove build cache to free up space
        docker builder prune -f --filter "until=30m" 2>/dev/null || true
        # Show current usage
        echo "Current Docker usage:"
        docker system df
        echo "Running containers: $(docker ps -q | wc -l)"
        echo "Total images: $(docker images -q | wc -l)"
        done
    ) &
    CLEANUP_PID=$!
    
    # Set up trap to kill cleanup process on script exit or interruption
    trap "kill $CLEANUP_PID 2>/dev/null || true; echo 'Cleanup process stopped'" EXIT INT TERM
  fi
  
  COMMAND="poetry run python evaluation/benchmarks/swe_bench_optimized/run_infer.py \
    --agent-cls CodeActAgent \
    --llm-config $MODEL \
    --max-iterations $MAX_ITER \
    --eval-num-workers $N_WORKERS \
    --eval-note $eval_note \
    --dataset $DATASET \
    --split $SPLIT \
    --n-runs $n_runs"

  if [ -n "$EVAL_LIMIT" ]; then
    echo "EVAL_LIMIT: $EVAL_LIMIT"
    COMMAND="$COMMAND --eval-n-limit $EVAL_LIMIT"
  fi

  # Run the command
  eval $COMMAND
  
  # Kill the cleanup background process (trap will also handle this)
  if [ "$RUNTIME" = "docker" ]; then
    kill $CLEANUP_PID 2>/dev/null || true
    trap - EXIT INT TERM  # Remove trap after cleanup
  fi
  
  # Kill the monitor background process
  kill $MONITOR_PID 2>/dev/null || true
  echo "Docker monitoring stopped. Logs saved to: $MONITOR_LOG"
}

# ============================================================
# Run inference for all runs (optimized to reuse Docker images)
# ============================================================
echo "### Running inference with N_RUNS=$N_RUNS (processing each instance $N_RUNS times before moving to next) ###"
unset SANDBOX_ENV_GITHUB_TOKEN # prevent the agent from using the github token to push

while true; do
    echo "### Running inference... ###"
    INFER_OUTPUT=$(run_eval "$EVAL_NOTE" "$N_RUNS")
    INFER_STATUS=$?
    echo "INFER_STATUS: $INFER_STATUS"

    echo "### Cleaning up remote runtime... ###"
    # ./evaluation/utils/scripts/cleanup_remote_runtime.sh

    # Also cleanup local docker containers and images if running locally
    if [ "$RUNTIME" = "docker" ]; then
        echo "### Cleaning up local docker containers... ###"
        docker ps -q --filter "name=openhands-runtime-" | xargs -r docker stop
        docker ps -aq --filter "name=openhands-runtime-" | xargs -r docker rm
        
        echo "### Cleaning up SWE-bench related docker images... ###"
        docker images --format "{{.Repository}}:{{.Tag}}" | grep -E "^(us-central1-docker\.pkg\.dev/evaluation-092424/swe-bench-images|docker\.io/xingyaoww/|mmr1115/openhands-runtime)" | xargs -r docker rmi -f 2>/dev/null || true
        
        echo "### Pruning dangling images and build cache... ###"
        docker image prune -f 2>/dev/null || true
        docker builder prune -f --filter "until=1h" 2>/dev/null || true
        
        echo "### Docker cleanup completed. Current usage: ###"
        docker system df
    fi

    if [ $INFER_STATUS -eq 0 ]; then
        echo "### Inference completed successfully for all runs. ###"
        break
    else
        echo "### Inference failed with exit code $INFER_STATUS. Retrying... ###"
    fi
done

# ============================================================
# Evaluate all runs
# ============================================================
echo "### Starting evaluation for all $N_RUNS runs ###"

# Extract output files from the inference output
# The Python script creates separate directories for each run (e.g., ...-run_1/, ...-run_2/)
OUTPUT_FILES=()
for run_idx in $(seq 1 $N_RUNS); do
    if [ $N_RUNS -gt 1 ]; then
        OUTPUT_FILE=$(echo "$INFER_OUTPUT" | grep -o "### OUTPUT FILE FOR RUN $run_idx:.* ###" | sed "s/### OUTPUT FILE FOR RUN $run_idx: \(.*\) ###/\1/")
    else
        OUTPUT_FILE=$(echo "$INFER_OUTPUT" | grep -o '### OUTPUT FILE:.* ###' | sed 's/### OUTPUT FILE: \(.*\) ###/\1/')
    fi
    
    if [ -z "$OUTPUT_FILE" ]; then
        echo "Warning: Could not extract OUTPUT_FILE for run $run_idx from inference output"
        # Fallback: try to find the output file based on naming pattern
        BASE_OUTPUT_DIR="evaluation/evaluation_outputs/outputs"
        if [ $N_RUNS -gt 1 ]; then
            PATTERN="${EVAL_NOTE}-run_${run_idx}"
        else
            PATTERN="${EVAL_NOTE}"
        fi
        OUTPUT_FILE=$(find "$BASE_OUTPUT_DIR" -name "output.jsonl" -path "*${PATTERN}*" | head -n 1)
    fi
    
    if [ -n "$OUTPUT_FILE" ] && [ -f "$OUTPUT_FILE" ]; then
        echo "Found OUTPUT_FILE for run $run_idx: $OUTPUT_FILE"
        OUTPUT_FILES+=("$OUTPUT_FILE")
    else
        echo "Error: OUTPUT_FILE not found for run $run_idx"
    fi
done

# Evaluate each run
for run_idx in $(seq 1 $N_RUNS); do
    OUTPUT_FILE="${OUTPUT_FILES[$((run_idx-1))]}"
    
    if [ -z "$OUTPUT_FILE" ]; then
        echo "### Skipping evaluation for run $run_idx (output file not found) ###"
        continue
    fi
    
    echo "### Evaluating run $run_idx/$N_RUNS on $OUTPUT_FILE ... ###"
    
    while true; do
        COMMAND="poetry run python evaluation/benchmarks/swe_bench/eval_infer.py \
        --eval-num-workers $((N_WORKERS * 2)) \
        --input-file $OUTPUT_FILE \
        --dataset $DATASET \
        --split $SPLIT"

        if [ -n "$EVAL_LIMIT" ]; then
            echo "EVAL_LIMIT: $EVAL_LIMIT"
            COMMAND="$COMMAND --eval-n-limit $EVAL_LIMIT"
        fi
        echo "Running command: $COMMAND"
        
        eval $COMMAND
        EVAL_STATUS=$?
        
        if [ $EVAL_STATUS -eq 0 ]; then
            echo "### Evaluation completed successfully for run $run_idx. ###"
            break
        else
            echo "### Evaluation failed with exit code $EVAL_STATUS. Retrying... ###"
        fi

        ./evaluation/utils/scripts/cleanup_remote_runtime.sh

        if [ "$RUNTIME" = "docker" ]; then
            echo "### Cleaning up local docker containers... ###"
            docker ps -q --filter "name=openhands-runtime-" | xargs -r docker stop
            docker ps -aq --filter "name=openhands-runtime-" | xargs -r docker rm
            
            echo "### Cleaning up SWE-bench related docker images... ###"
            docker images --format "{{.Repository}}:{{.Tag}}" | grep -E "^(us-central1-docker\.pkg\.dev/evaluation-092424/swe-bench-images|docker\.io/xingyaoww/|mmr1115/openhands-runtime)" | xargs -r docker rmi -f 2>/dev/null || true
            
            echo "### Pruning dangling images and build cache... ###"
            docker image prune -f 2>/dev/null || true
            docker builder prune -f --filter "until=1h" 2>/dev/null || true
            
            echo "### Docker cleanup completed. Current usage: ###"
            docker system df
        fi
    done

    # Update the output with evaluation results
    echo "### Updating the output with evaluation results for run $run_idx... ###"
    poetry run python evaluation/benchmarks/swe_bench/scripts/eval/update_output_with_eval.py $OUTPUT_FILE

    echo "### Combining the final completions for run $run_idx... ###"
    poetry run python evaluation/benchmarks/swe_bench/scripts/eval/combine_final_completions.py $OUTPUT_FILE

    echo "### DONE for run $run_idx! ###"
    echo "You can find the final output at $(dirname $OUTPUT_FILE)/$FINAL_OUTPUT_FILE"
done

echo ""
echo "=============================================================="
echo "ALL $N_RUNS RUNS COMPLETED!"
echo "=============================================================="

# Show monitoring summary if available
if [ -f "$MONITOR_LOG" ]; then
    echo ""
    echo "=== Docker Monitoring Summary ==="
    echo "Full monitoring log: $MONITOR_LOG"
    
    # Peak docker load processes
    PEAK_LOADS=$(grep "Active 'docker load' processes:" "$MONITOR_LOG" | awk '{print $5}' | sort -rn | head -1)
    echo "Peak concurrent docker load processes: $PEAK_LOADS"
    
    # Peak containers
    PEAK_CONTAINERS=$(grep "Docker:" "$MONITOR_LOG" | awk '{print $2}' | sort -rn | head -1)
    echo "Peak running containers: $PEAK_CONTAINERS"
    
    # Peak images
    PEAK_IMAGES=$(grep "Docker:" "$MONITOR_LOG" | awk '{print $4}' | sort -rn | head -1)
    echo "Peak Docker images: $PEAK_IMAGES"
    
    echo ""
    echo "Last 5 monitoring snapshots:"
    grep -A 10 "=== Docker Monitor" "$MONITOR_LOG" | tail -55
fi
