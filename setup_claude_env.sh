#!/usr/bin/env bash
# Quick setup script to load environment and start Claude proxy
# Usage: source setup_claude_env.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if .env file exists
if [ ! -f "$SCRIPT_DIR/.env" ]; then
    echo "Error: .env file not found!"
    echo "Please create .env file with your credentials:"
    echo ""
    echo "  cp .env.example .env"
    echo "  # Then edit .env with your actual API key"
    echo ""
    return 1 2>/dev/null || exit 1
fi

# Load environment variables
echo "Loading environment variables from .env..."
set -a  # automatically export all variables
source "$SCRIPT_DIR/.env"
set +a

# Verify required variables are set
if [ -z "$CLAUDE_API_KEY" ]; then
    echo "Error: CLAUDE_API_KEY is not set in .env"
    return 1 2>/dev/null || exit 1
fi

if [ -z "$CLAUDE_RESOURCE" ]; then
    echo "Warning: CLAUDE_RESOURCE is not set, using default: lucia-claude-eastus2"
    export CLAUDE_RESOURCE="lucia-claude-eastus2"
fi

echo "✓ Environment variables loaded successfully!"
echo "  CLAUDE_RESOURCE: $CLAUDE_RESOURCE"
echo "  CLAUDE_API_KEY: ${CLAUDE_API_KEY:0:20}... (hidden)"
echo ""
echo "To start the proxy server, run:"
echo "  ./start_claude_proxy.sh"
echo ""
echo "Or to run evaluation directly:"
echo "  ./evaluation/benchmarks/swe_bench_optimized/scripts/run_infer.sh ..."
