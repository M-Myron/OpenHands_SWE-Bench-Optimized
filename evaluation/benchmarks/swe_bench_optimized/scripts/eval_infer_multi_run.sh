#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# eval_infer_multi_run.sh
#
# Wrapper around eval_infer_multi_run.py.
# Evaluates multiple runs of the same task/model by reusing a single Docker
# container per instance, which avoids repeated image pulls/loads.
#
# Usage:
#   bash evaluation/benchmarks/swe_bench_optimized/scripts/eval_infer_multi_run.sh \
#       <base_dir>          \   # dir containing ...-run_1, ...-run_2, ... subdirs
#       <dataset_name>      \   # e.g.  "SWE-Gym/SWE-Gym"   or  "princeton-nlp/SWE-bench_Verified"
#       <split>             \   # e.g.  "train"  or  "test"
#       <num_workers>           # number of parallel instance workers (default: 4)
#
# Examples:
#   bash evaluation/benchmarks/swe_bench_optimized/scripts/eval_infer_multi_run.sh \
#       evaluation/evaluation_outputs/outputs/SWE-Gym__SWE-Gym-train/CodeActAgent/Qwen3-Coder-480B_maxiter_100_N_v0.61.0-no-hint \
#       "SWE-Gym/SWE-Gym" "train" 4
# ---------------------------------------------------------------------------
set -eo pipefail

BASE_DIR=$1
DATASET_NAME=${2:-"princeton-nlp/SWE-bench_Verified"}
SPLIT=${3:-"test"}
NUM_WORKERS=${4:-4}

if [ -z "$BASE_DIR" ]; then
    echo "Usage: $0 <base_dir> [dataset_name] [split] [num_workers]"
    echo ""
    echo "  base_dir       Directory that contains run subdirectories (*-run_1, *-run_2, ...)"
    echo "  dataset_name   HuggingFace dataset name  (default: princeton-nlp/SWE-bench_Verified)"
    echo "  split          Dataset split              (default: test)"
    echo "  num_workers    Parallel instance workers  (default: 4)"
    exit 1
fi

if [ ! -d "$BASE_DIR" ]; then
    echo "Error: BASE_DIR does not exist or is not a directory: $BASE_DIR"
    exit 1
fi

# Resolve to absolute path
BASE_DIR=$(realpath "$BASE_DIR")

echo "=============================================================="
echo "  Multi-Run SWE-bench Evaluation (Docker image reuse)"
echo "=============================================================="
echo "  BASE_DIR     : $BASE_DIR"
echo "  DATASET_NAME : $DATASET_NAME"
echo "  SPLIT        : $SPLIT"
echo "  NUM_WORKERS  : $NUM_WORKERS"
echo "=============================================================="
echo ""

# Determine the repo root (where pyproject.toml lives) so that
# `poetry run` can find the project regardless of CWD.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR/../../../../..")"
REPO_ROOT="$(realpath "$REPO_ROOT")"

echo "Running from repo root: $REPO_ROOT"
cd "$REPO_ROOT"

poetry run python evaluation/benchmarks/swe_bench_optimized/eval_infer_multi_run.py \
    --base-dir        "$BASE_DIR"      \
    --dataset         "$DATASET_NAME"  \
    --split           "$SPLIT"         \
    --eval-num-workers "$NUM_WORKERS"

exit_code=$?

echo ""
echo "=============================================================="
if [ $exit_code -eq 0 ]; then
    echo "  Multi-run evaluation completed successfully."
else
    echo "  Multi-run evaluation exited with code $exit_code."
fi
echo "=============================================================="

exit $exit_code
