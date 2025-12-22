#!/usr/bin/env bash
# Script to replay and refine SWE-bench trajectories
# Usage: ./replay_and_refine.sh [OPTIONS]

set -eo pipefail

# Default values
MODEL_CONFIG="llm"
AGENT_CLASS="CodeActAgent"
MAX_ITERATIONS=50
EVAL_NOTE="replay_refine"
BATCH_MODE=false

# Parse command line arguments
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Replay and refine SWE-bench trajectories based on analysis results.

Required (one of):
    -t, --trajectory PATH       Path to single trajectory JSON file
    -d, --trajectory-dir DIR    Directory containing trajectory files (for batch mode)

Required:
    -r, --refinement PATH       Path to refinement input JSON file
    -o, --output-dir DIR        Output directory for refined trajectories

Optional:
    -m, --model-config NAME     LLM config name (default: llm)
    -a, --agent-class NAME      Agent class name (default: CodeActAgent)
    -i, --max-iterations NUM    Maximum iterations (default: 50)
    -n, --eval-note TEXT        Evaluation note (default: replay_refine)
    -b, --batch                 Enable batch processing mode
    -h, --help                  Show this help message

Examples:
    # Single refinement
    $0 -t output/trajectories/django__django-12345.json \\
       -r refinement.json \\
       -o refined_output

    # Batch refinement
    $0 -d output/trajectories \\
       -r refinement_batch.json \\
       -o refined_output \\
       --batch

    # With custom settings
    $0 -t output/trajectories/django__django-12345.json \\
       -r refinement.json \\
       -o refined_output \\
       -m claude \\
       -a CodeActAgent \\
       -i 100
EOF
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--trajectory)
            TRAJECTORY_PATH="$2"
            shift 2
            ;;
        -d|--trajectory-dir)
            TRAJECTORY_DIR="$2"
            BATCH_MODE=true
            shift 2
            ;;
        -r|--refinement)
            REFINEMENT_INPUT="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -m|--model-config)
            MODEL_CONFIG="$2"
            shift 2
            ;;
        -a|--agent-class)
            AGENT_CLASS="$2"
            shift 2
            ;;
        -i|--max-iterations)
            MAX_ITERATIONS="$2"
            shift 2
            ;;
        -n|--eval-note)
            EVAL_NOTE="$2"
            shift 2
            ;;
        -b|--batch)
            BATCH_MODE=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate required arguments
if [ -z "$REFINEMENT_INPUT" ]; then
    echo "Error: Refinement input file is required (-r/--refinement)"
    exit 1
fi

if [ -z "$OUTPUT_DIR" ]; then
    echo "Error: Output directory is required (-o/--output-dir)"
    exit 1
fi

if [ "$BATCH_MODE" = true ]; then
    if [ -z "$TRAJECTORY_DIR" ]; then
        echo "Error: Trajectory directory is required for batch mode (-d/--trajectory-dir)"
        exit 1
    fi
else
    if [ -z "$TRAJECTORY_PATH" ]; then
        echo "Error: Trajectory path is required (-t/--trajectory)"
        exit 1
    fi
fi

# Display configuration
echo "======================================"
echo "Replay and Refine Configuration"
echo "======================================"
echo "Model Config: $MODEL_CONFIG"
echo "Agent Class: $AGENT_CLASS"
echo "Max Iterations: $MAX_ITERATIONS"
echo "Eval Note: $EVAL_NOTE"
echo "Batch Mode: $BATCH_MODE"
echo "Refinement Input: $REFINEMENT_INPUT"
echo "Output Directory: $OUTPUT_DIR"

if [ "$BATCH_MODE" = true ]; then
    echo "Trajectory Directory: $TRAJECTORY_DIR"
else
    echo "Trajectory Path: $TRAJECTORY_PATH"
fi
echo "======================================"
echo ""

# Build command
COMMAND="poetry run python evaluation/benchmarks/swe_bench/replay_and_refine.py \
    --refinement-input \"$REFINEMENT_INPUT\" \
    --output-dir \"$OUTPUT_DIR\" \
    --model-config \"$MODEL_CONFIG\" \
    --agent-class \"$AGENT_CLASS\" \
    --max-iterations $MAX_ITERATIONS \
    --eval-note \"$EVAL_NOTE\""

if [ "$BATCH_MODE" = true ]; then
    COMMAND="$COMMAND --trajectory-dir \"$TRAJECTORY_DIR\" --batch"
else
    COMMAND="$COMMAND --trajectory-path \"$TRAJECTORY_PATH\""
fi

# Execute command
echo "Executing: $COMMAND"
echo ""
eval $COMMAND

echo ""
echo "======================================"
echo "Refinement Complete!"
echo "======================================"
echo "Output saved to: $OUTPUT_DIR"
