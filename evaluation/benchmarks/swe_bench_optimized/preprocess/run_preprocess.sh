#!/usr/bin/env bash
# run_preprocess.sh — Filter + LLM-reduce fact graphs in one go.
#
# Usage:
#   # Full pipeline (filter + reduce):
#   bash evaluation/benchmarks/swe_bench_optimized/preprocess/run_preprocess.sh
#
#   # Filter only (no LLM):
#   bash evaluation/benchmarks/swe_bench_optimized/preprocess/run_preprocess.sh --filter-only
#
#   # Reduce only (filter must exist):
#   bash evaluation/benchmarks/swe_bench_optimized/preprocess/run_preprocess.sh --reduce-only
#
#   # Dry-run reduce (no LLM calls):
#   bash evaluation/benchmarks/swe_bench_optimized/preprocess/run_preprocess.sh --dry-run
#
#   # Process specific instances:
#   bash evaluation/benchmarks/swe_bench_optimized/preprocess/run_preprocess.sh --instance-ids conan-io__conan-10408
#
# Environment variables (all optional, sensible defaults provided):
#   PREPROCESS_INPUT_DIR   — Input swegym_v6 directory
#   PREPROCESS_FILTER_JSON — Output path for filter JSON
#   PREPROCESS_OUTPUT_DIR  — Output directory for reduced graphs
#   PREPROCESS_PERCENTILE  — Percentile threshold (default: 95)
#   PREPROCESS_API_BASE    — LLM API base URL (default: http://localhost:8000/v1)
#   PREPROCESS_MODEL       — Model name (default: glm-5)
#   PREPROCESS_API_KEY     — API key (default: EMPTY)
#   PREPROCESS_DATASET     — HuggingFace dataset (default: SWE-Gym/SWE-Gym)
#   PREPROCESS_SPLIT       — Dataset split (default: train)
#   PREPROCESS_MAX_WORKERS — Parallel workers for reduction (default: 4)

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT"

# ── Defaults ─────────────────────────────────────────────────────────────
_BASE="evaluation/evaluation_outputs/outputs/SWE-Gym__SWE-Gym-train/preprocess"

INPUT_DIR="${PREPROCESS_INPUT_DIR:-${_BASE}/swegym_v6}"
FILTER_JSON="${PREPROCESS_FILTER_JSON:-${_BASE}/swegym_v6_filter.json}"
OUTPUT_DIR="${PREPROCESS_OUTPUT_DIR:-${_BASE}/swegym_v6_reduced}"
PHASE1_DIR="${PREPROCESS_PHASE1_DIR:-${_BASE}/swegym_v6_phase1}"
PERCENTILE="${PREPROCESS_PERCENTILE:-95}"
API_BASE="${PREPROCESS_API_BASE:-http://localhost:8000/v1}"
MODEL="${PREPROCESS_MODEL:-glm-5}"
API_KEY="${PREPROCESS_API_KEY:-EMPTY}"
DATASET="${PREPROCESS_DATASET:-SWE-Gym/SWE-Gym}"
SPLIT="${PREPROCESS_SPLIT:-train}"
MAX_WORKERS="${PREPROCESS_MAX_WORKERS:-4}"

# ── Parse flags ──────────────────────────────────────────────────────────
RUN_FILTER=1
RUN_REDUCE=1
DRY_RUN=""
DEBUG_LOG="--debug-log-dir ${OUTPUT_DIR}/_debug_logs"
PHASES="1,2"
# INSTANCE_IDS="getmoto__moto-4895"
INSTANCE_IDS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --filter-only)  RUN_REDUCE=0; shift ;;
    --reduce-only)  RUN_FILTER=0; shift ;;
    --dry-run)      DRY_RUN="--dry-run"; shift ;;
    --debug-log)    DEBUG_LOG="--debug-log-dir ${OUTPUT_DIR}/_debug_logs"; shift ;;
    --phases)       shift; PHASES="$1"; shift ;;
    --instance-ids) shift; INSTANCE_IDS="$*"; break ;;
    *)              echo "Unknown flag: $1"; exit 1 ;;
  esac
done

echo "========================================================"
echo "  FACT GRAPH PREPROCESSING"
echo "========================================================"
echo "  Input dir:    $INPUT_DIR"
echo "  Filter JSON:  $FILTER_JSON"
echo "  Output dir:   $OUTPUT_DIR"
echo "  Phase1 dir:   $PHASE1_DIR"
echo "  Percentile:   $PERCENTILE"
echo "  API base:     $API_BASE"
echo "  Model:        $MODEL"
echo "  Max workers:  $MAX_WORKERS"
echo "  Filter step:  $([ $RUN_FILTER -eq 1 ] && echo YES || echo SKIP)"
echo "  Reduce step:  $([ $RUN_REDUCE -eq 1 ] && echo YES || echo SKIP)"
echo "  Phases:       $PHASES"
[ -n "$DRY_RUN" ] && echo "  Mode:         DRY RUN (no LLM calls)"
[ -n "$DEBUG_LOG" ] && echo "  Debug log:    ${OUTPUT_DIR}/_debug_logs"
[ -n "$INSTANCE_IDS" ] && echo "  Instance IDs: $INSTANCE_IDS"
echo "========================================================"

# ── Step 1: Filter ───────────────────────────────────────────────────────
if [ $RUN_FILTER -eq 1 ]; then
  echo ""
  echo "### Step 1: Filtering fact graphs (p${PERCENTILE}) ###"
  python -m evaluation.benchmarks.swe_bench_optimized.preprocess.filter_fact_graphs \
    --preprocess-dir "$INPUT_DIR" \
    --output "$FILTER_JSON" \
    --percentile "$PERCENTILE"
  echo ""
fi

# ── Step 2: Reduce ───────────────────────────────────────────────────────
if [ $RUN_REDUCE -eq 1 ]; then
  if [ ! -f "$FILTER_JSON" ]; then
    echo "ERROR: Filter JSON not found: $FILTER_JSON"
    echo "Run with --filter-only first, or without --reduce-only."
    exit 1
  fi

  echo ""
  echo "### Step 2: LLM-assisted graph reduction ###"

  INSTANCE_ARG=""
  if [ -n "$INSTANCE_IDS" ]; then
    INSTANCE_ARG="--instance-ids $INSTANCE_IDS"
  fi

  python -m evaluation.benchmarks.swe_bench_optimized.preprocess.reduce_fact_graphs \
    --preprocess-dir "$INPUT_DIR" \
    --filter-json "$FILTER_JSON" \
    --output-dir "$OUTPUT_DIR" \
    --api-base "$API_BASE" \
    --model "$MODEL" \
    --api-key "$API_KEY" \
    --dataset "$DATASET" \
    --split "$SPLIT" \
    --max-workers "$MAX_WORKERS" \
    --phases "$PHASES" \
    --phase1-dir "$PHASE1_DIR" \
    $DRY_RUN \
    $DEBUG_LOG \
    $INSTANCE_ARG
  echo ""
fi

# ── Done ─────────────────────────────────────────────────────────────────
echo "========================================================"
echo "  DONE"
echo "========================================================"
echo ""
echo "To use in inference, set:"
echo "  export ORACLE_GRAPH_FILTER_JSON=$FILTER_JSON"
echo "  export ORACLE_PREPROCESS_DIR=$OUTPUT_DIR"
echo ""
echo "Example:"
echo "  ORACLE_GRAPH_FILTER_JSON=$FILTER_JSON \\"
echo "  ORACLE_PREPROCESS_DIR=$OUTPUT_DIR \\"
echo "  bash evaluation/benchmarks/swe_bench_optimized/scripts/run_oracle_guided_infer_instance_major.sh \\"
echo "    llm.eval_glm5_fp8_t0 HEAD OracleGuidedCodeActAgent 100 100 32"
