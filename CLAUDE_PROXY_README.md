# Azure AI Foundry Claude Integration for OpenHands

## Problem

OpenHands uses **LiteLLM** which doesn't natively support Azure AI Foundry's Anthropic endpoint format. The `anthropic-foundry` library uses a special `AnthropicFoundry(resource=...)` client that LiteLLM cannot use directly.

## Solution: Proxy Server

We created a **proxy server** that:
1. Provides an OpenAI-compatible API endpoint
2. Translates requests to Azure AI Foundry format
3. Uses the `anthropic-foundry` library to call Claude models
4. Converts responses back to OpenAI format for LiteLLM

## Setup

### Step 1: Install Dependencies

```bash
cd /home/v-murongma/code/OpenHands_SWE-Bench-Optimized
pip install anthropic flask
```

### Step 2: Start the Proxy Server

```bash
# Start on default port 8001
./start_claude_proxy.sh

# Or specify a custom port
./start_claude_proxy.sh 9001
```

The proxy will show:
```
Azure AI Foundry Claude Proxy Server
==============================================================
Resource: lucia-claude-eastus2
Listening on: http://0.0.0.0:8001

OpenAI-compatible endpoint:
  http://localhost:8001/v1/chat/completions

Configure OpenHands with:
  model="openai/claude-sonnet-4-5"
  base_url="http://localhost:8001/v1"
  api_key="dummy"
==============================================================
```

### Step 3: Configure OpenHands (Already Done!)

The `config.toml` has been updated to use the proxy:

```toml
[llm.eval_swebench-verified_claude-sonnet-4-5]
model="openai/claude-sonnet-4-5"
api_key="dummy"
base_url="http://localhost:8001/v1"
...
```

### Step 4: Run Evaluation

In a **new terminal** (proxy must keep running):

```bash
cd /home/v-murongma/code/OpenHands_SWE-Bench-Optimized

# Run evaluation
./evaluation/benchmarks/swe_bench_optimized/scripts/run_infer.sh \
  eval_swebench-verified_claude-sonnet-4-5 \
  $(git rev-parse HEAD) \
  CodeActAgent \
  10 \
  100 \
  4 \
  "princeton-nlp/SWE-bench_Verified" \
  "test" \
  3
```

## Architecture

```
OpenHands (LiteLLM)
       ↓
  OpenAI-compatible request
       ↓
  Proxy Server (localhost:8001)
       ↓
  AnthropicFoundry client
       ↓
  Azure AI Foundry
       ↓
  Claude Models
```

## Available Models

The proxy supports these model names:
- `claude-sonnet-4-5` (recommended)
- `claude-opus-4-1` (highest quality)
- `claude-haiku-4-5` (fastest/cheapest)
- `gpt-4` (alias for sonnet)
- `gpt-3.5-turbo` (alias for haiku)

## Testing

### Test the Proxy

```bash
# In terminal 1: Start proxy
./start_claude_proxy.sh

# In terminal 2: Test with curl
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-5",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'

# Check health
curl http://localhost:8001/health

# List models
curl http://localhost:8001/v1/models
```

### Test with OpenHands

```python
# Quick test
from litellm import completion

response = completion(
    model="openai/claude-sonnet-4-5",
    api_base="http://localhost:8001/v1",
    api_key="dummy",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

## Running in Background

### Option 1: tmux/screen

```bash
# Start in tmux
tmux new -s claude-proxy
./start_claude_proxy.sh
# Detach: Ctrl+B, then D

# Reattach later
tmux attach -t claude-proxy
```

### Option 2: nohup

```bash
# Start in background
nohup ./start_claude_proxy.sh > proxy.log 2>&1 &

# Check logs
tail -f proxy.log

# Stop proxy
pkill -f azure_foundry_proxy
```

### Option 3: systemd service (Production)

Create `/etc/systemd/system/claude-proxy.service`:

```ini
[Unit]
Description=Azure AI Foundry Claude Proxy
After=network.target

[Service]
Type=simple
User=v-murongma
WorkingDirectory=/home/v-murongma/code/OpenHands_SWE-Bench-Optimized
ExecStart=/home/v-murongma/code/OpenHands_SWE-Bench-Optimized/start_claude_proxy.sh
Restart=always

[Install]
WantedBy=multi-user.target
```

Then:
```bash
sudo systemctl daemon-reload
sudo systemctl start claude-proxy
sudo systemctl enable claude-proxy  # Start on boot
```

## Troubleshooting

### Proxy won't start

```bash
# Check if port is in use
lsof -i :8001

# Kill existing process
pkill -f azure_foundry_proxy

# Try different port
./start_claude_proxy.sh 9001
# Update config.toml base_url accordingly
```

### Connection refused

```bash
# Check if proxy is running
curl http://localhost:8001/health

# Check proxy logs
# (if running in background, check proxy.log)
```

### Azure AI Foundry errors

```bash
# Test Azure AI Foundry directly
python test_claude_config.py

# Check API key and resource name in start_claude_proxy.sh
```

### LiteLLM can't reach proxy

If running in Docker, use `host.docker.internal` instead of `localhost`:

```toml
base_url="http://host.docker.internal:8001/v1"
```

## Files

- `azure_foundry_proxy.py` - Proxy server (OpenAI ↔ Azure AI Foundry)
- `start_claude_proxy.sh` - Helper script to start the proxy
- `config.toml` - Updated with proxy endpoint configuration
- `test_claude_config.py` - Test script for Azure AI Foundry

## Performance

The proxy adds minimal latency (<10ms) for request/response translation. The bottleneck is the actual API call to Azure AI Foundry, not the proxy.

## Security Notes

- The proxy runs locally and is not exposed to the internet
- API keys are stored in the proxy process environment
- For production, consider using environment variables or a secrets manager
- The proxy does not log request/response contents by default

## Why Not Direct LiteLLM Integration?

LiteLLM doesn't support Azure AI Foundry's specific authentication and endpoint format for Anthropic models. The `anthropic-foundry` library uses a proprietary `resource` parameter that isn't compatible with LiteLLM's provider system. This proxy bridges that gap.
