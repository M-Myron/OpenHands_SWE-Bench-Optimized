#!/usr/bin/env bash
# Run trajectory quality labeling on a small set of instances.
#
# Usage:
#   bash run_label_trajectory.sh                              # first 5 instances, with prompt logging
#   bash /home/v-murongma/code/OpenHands_SWE-Bench-Optimized/evaluation/evaluation_analysis/oracle_guided_analysis/run_label_trajectory.sh 10                           # first 10 instances
#   bash run_label_trajectory.sh 0 getmoto__moto-4787 getmoto__moto-4895  # specific instances
#
# Environment variables (optional):
#   OUTPUT_JSONL    — path to output.jsonl (default: CodeActAgent GLM-5 run_1)
#   PREPROCESS_DIR  — path to preprocessed fact graphs (default: swegym_v6_phase1)
#   LLM_BASE_URL    — LLM endpoint (default: http://127.0.0.1:8000/v1)
#   LLM_MODEL       — model name (default: zai-org/GLM-5-FP8)
#   SAVE_PROMPTS    — 1 to save prompts, 0 to skip (default: 1)
#   STEPS_PER_BATCH — steps per LLM call (default: 30)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# ── Defaults ──
OUTPUT_JSONL="${OUTPUT_JSONL:-$REPO_ROOT/evaluation/evaluation_outputs/outputs/SWE-Gym__SWE-Gym-train/CodeActAgent/GLM-5-FP8_maxiter_100_N_v0.61.0-no-hint-train-glm5_fp8_t05-run_1/output.jsonl}"
PREPROCESS_DIR="${PREPROCESS_DIR:-$REPO_ROOT/evaluation/evaluation_outputs/outputs/SWE-Gym__SWE-Gym-train/preprocess/swegym_v6_phase1}"
LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:8000/v1}"
LLM_MODEL="${LLM_MODEL:-zai-org/GLM-5-FP8}"
SAVE_PROMPTS="${SAVE_PROMPTS:-1}"
STEPS_PER_BATCH="${STEPS_PER_BATCH:-50}"

# ── Parse args ──
MAX_INSTANCES="${1:-5}"
shift 2>/dev/null || true
INSTANCE_IDS=("$@")

# Build results dir name alongside the output.jsonl
OUTPUT_DIR="$(dirname "$OUTPUT_JSONL")"
RESULTS_DIR="$OUTPUT_DIR/labeled_trajectory_facts"

echo "============================================================"
echo "Trajectory Quality Labeling"
echo "============================================================"
echo "  Output JSONL:     $OUTPUT_JSONL"
echo "  Preprocess dir:   $PREPROCESS_DIR"
echo "  Results dir:      $RESULTS_DIR"
echo "  LLM endpoint:     $LLM_BASE_URL"
echo "  LLM model:        $LLM_MODEL"
echo "  Steps per batch:  $STEPS_PER_BATCH"
echo "  Save prompts:     $SAVE_PROMPTS"

if [[ ${#INSTANCE_IDS[@]} -gt 0 ]]; then
    echo "  Instance IDs:     ${INSTANCE_IDS[*]}"
    MAX_INSTANCES=0  # ignore max when specific IDs given
else
    echo "  Max instances:    $MAX_INSTANCES"
fi
echo "============================================================"
echo ""

# ── Health check ──
if ! curl -fsS "$LLM_BASE_URL/models" >/dev/null 2>&1; then
    echo "[ERROR] LLM endpoint not reachable: $LLM_BASE_URL"
    echo "        Start the model server or set LLM_BASE_URL."
    exit 1
fi
echo "[OK] LLM endpoint reachable"

# ── Build command ──
CMD=(
    python3 "$SCRIPT_DIR/label_trajectory.py"
    --output-jsonl "$OUTPUT_JSONL"
    --preprocess-dir "$PREPROCESS_DIR"
    --results-dir "$RESULTS_DIR"
    --llm-base-url "$LLM_BASE_URL"
    --llm-model "$LLM_MODEL"
    --steps-per-batch "$STEPS_PER_BATCH"
)

if [[ "$SAVE_PROMPTS" == "1" ]]; then
    CMD+=(--save-prompts)
fi

if [[ ${#INSTANCE_IDS[@]} -gt 0 ]]; then
    CMD+=(--instance-ids "${INSTANCE_IDS[@]}")
elif [[ "$MAX_INSTANCES" -gt 0 ]]; then
    CMD+=(--max-instances "$MAX_INSTANCES")
fi

mkdir -p "$RESULTS_DIR"

echo ""
echo "[CMD] ${CMD[*]}"
echo ""

"${CMD[@]}" 2>&1 | tee "$RESULTS_DIR/run.log"

echo ""
echo "[DONE] Results saved to: $RESULTS_DIR"
echo "  Labels:   $RESULTS_DIR/labels/"
if [[ "$SAVE_PROMPTS" == "1" ]]; then
    echo "  Prompts:  $RESULTS_DIR/prompts/"
fi
echo "  Summary:  $RESULTS_DIR/summary.jsonl"
echo "  Run log:  $RESULTS_DIR/run.log"
