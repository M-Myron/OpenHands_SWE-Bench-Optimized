# Claude Configuration for OpenHands SWE-bench Evaluation

## Problem
Azure AI Foundry hosts Claude models, but LiteLLM (used by OpenHands) needs the correct configuration to access them.

## Solution Steps

### Step 1: Test the Endpoint
Run the test script to identify the correct LiteLLM configuration:

```bash
cd /home/v-murongma/code/OpenHands_SWE-Bench-Optimized
python test_claude_config.py
```

This will:
1. Verify your Anthropic Foundry setup works
2. Test different LiteLLM configurations
3. Show you the exact config that works

### Step 2: Update config.toml
Based on the test results, update your `config.toml` with the working configuration.

**IMPORTANT**: Do not put actual API keys in `config.toml` if you plan to commit it to git!

### Current Configuration (May Need Adjustment)
```toml
[llm.eval_swebench-verified_claude-sonnet-4-5]
model="anthropic/claude-sonnet-4-5"
api_key="your-api-key-from-env-or-vault"
base_url="https://lucia-claude-eastus2.services.ai.azure.com/models/chat/completions"
num_retries=15
retry_min_wait=15
retry_max_wait=120
retry_multiplier=2
timeout=600
max_output_tokens=16384
temperature=0.0
```

### Alternative Configurations to Try

**Option A: OpenAI-Compatible Mode**
```toml
[llm.eval_swebench-verified_claude-sonnet-4-5]
model="openai/claude-sonnet-4-5"  # Use openai/ prefix
api_key="dummy"  # Proxy doesn't validate this
base_url="http://localhost:8001/v1"  # Use proxy
```

**Option B: Custom LLM Provider**
```toml
[llm.eval_swebench-verified_claude-sonnet-4-5]
model="claude-sonnet-4-5"  # No prefix
api_key="your-key-here"
base_url="https://your-resource.services.ai.azure.com/models"
custom_llm_provider="anthropic"
```

**Option C: Environment Variables + Simple Model Name**
```bash
# Set environment variables
export CLAUDE_API_KEY="your-api-key-here"
export CLAUDE_RESOURCE="your-resource-name"
```

```toml
[llm.eval_swebench-verified_claude-sonnet-4-5]
model="openai/claude-sonnet-4-5"
api_key="dummy"
base_url="http://localhost:8001/v1"
```

### Step 3: Run SWE-bench Evaluation

Once you have the working configuration:

```bash
# Source environment if needed
source .env.claude  # Optional

# Run evaluation
cd /home/v-murongma/code/OpenHands_SWE-Bench-Optimized/evaluation/benchmarks/swe_bench_optimized/scripts

# Single run test
./run_infer.sh \
  eval_swebench-verified_claude-sonnet-4-5 \
  $(git rev-parse HEAD) \
  CodeActAgent \
  1 \
  100 \
  4 \
  "princeton-nlp/SWE-bench_Verified" \
  "test" \
  1

# Multiple runs (optimized with Docker image reuse!)
./run_infer.sh \
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

### Step 4: Debugging

If you still get errors, check:

1. **Endpoint URL**: Run `test_claude_direct.py` to test raw HTTP requests
2. **LiteLLM Version**: Ensure you have the latest version
   ```bash
   pip install --upgrade litellm
   ```
3. **Azure AI API Version**: The endpoint might need specific API version headers
4. **Model Availability**: Verify the model is deployed in your resource

### Common Error Messages

**"Resource not found" (404)**
- Wrong base_url endpoint
- Model not deployed in your Azure resource
- Incorrect API path (/models vs /v1/messages vs /chat/completions)

**"LLM Provider NOT provided"**
- Missing or incorrect model prefix
- Need to add `custom_llm_provider` field

**Authentication errors**
- Check API key is correct
- Verify key has access to the resource

## Next Steps

1. Run `python test_claude_config.py` to find working config
2. Update `config.toml` based on results
3. Test with a small evaluation run
4. Once working, run full multi-run evaluation with optimization!

## Contact/Support

If issues persist, check:
- LiteLLM Azure AI docs: https://docs.litellm.ai/docs/providers/azure_ai
- Anthropic Foundry docs
- Azure AI Foundry model deployment status
