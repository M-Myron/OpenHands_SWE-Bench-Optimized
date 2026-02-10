#!/usr/bin/env bash
# Start the Azure AI Foundry Claude Proxy Server
# This proxy allows LiteLLM (OpenHands) to access Azure AI Foundry Claude models

set -e

PORT=${1:-8001}

echo "=============================================================="
echo "Starting Azure AI Foundry Claude Proxy Server"
echo "=============================================================="
echo ""

# Check if anthropic package is installed
if ! python -c "import anthropic" 2>/dev/null; then
    echo "Error: anthropic package not found"
    echo "Installing required packages..."
    pip install anthropic flask
fi

# Check if CLAUDE_API_KEY is set
if [ -z "$CLAUDE_API_KEY" ]; then
    echo "Error: CLAUDE_API_KEY environment variable is not set"
    echo "Please set it before running this script:"
    echo "  export CLAUDE_API_KEY='your-api-key-here'"
    exit 1
fi

# Set default resource if not provided
export CLAUDE_RESOURCE="${CLAUDE_RESOURCE:-lucia-claude-eastus2}"

echo "Starting proxy on port $PORT..."
echo "Resource: $CLAUDE_RESOURCE"
echo ""

# Start the proxy
python azure_foundry_proxy.py --port $PORT

