#!/usr/bin/env bash
set -eo pipefail

# Auto-configure for SWE-Gym
export DATASET=${1:-"SWE-Gym/SWE-Gym"}
export SPLIT=${2:-"train"}
export NUM_WORKERS=${3:-4}
export EVAL_LIMIT=${4:-500}

# Set appropriate image prefix based on runtime
# if [ -n "$ALLHANDS_API_KEY" ]; then
#     export EVAL_DOCKER_IMAGE_PREFIX="${EVAL_DOCKER_IMAGE_PREFIX:-us-central1-docker.pkg.dev/evaluation-092424/swe-bench-images}"
#     export OH_RUNTIME_RUNTIME_IMAGE_REPO="${OH_RUNTIME_RUNTIME_IMAGE_REPO:-us-central1-docker.pkg.dev/evaluation-092424/swe-bench-images}"
# else
#     export EVAL_DOCKER_IMAGE_PREFIX="${EVAL_DOCKER_IMAGE_PREFIX:-docker.io/xingyaoww/}"
#     export OH_RUNTIME_RUNTIME_IMAGE_REPO="${OH_RUNTIME_RUNTIME_IMAGE_REPO:-ghcr.io/openhands/runtime}"
# fi

export EVAL_DOCKER_IMAGE_PREFIX="${EVAL_DOCKER_IMAGE_PREFIX:-docker.io/xingyaoww/}"
export OH_RUNTIME_RUNTIME_IMAGE_REPO="${OH_RUNTIME_RUNTIME_IMAGE_REPO:-docker.io/mmr1115/openhands-runtime}"

echo "=========================================="
echo "SWE-Gym Image Pre-build Script"
echo "=========================================="
echo "Dataset: $DATASET"
echo "Split: $SPLIT"
echo "Eval Docker Image Prefix: $EVAL_DOCKER_IMAGE_PREFIX"
echo "Runtime Image Repo: $OH_RUNTIME_RUNTIME_IMAGE_REPO"
echo "=========================================="

./evaluation/benchmarks/swe_bench/scripts/prebuild_images.sh \
    "$DATASET" "$SPLIT" "$NUM_WORKERS" "$EVAL_LIMIT"
