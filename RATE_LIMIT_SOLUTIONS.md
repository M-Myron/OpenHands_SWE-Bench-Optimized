# Azure AI Foundry Rate Limit Solutions

## Understanding the Rate Limit

**Error:** `Rate limit of 1000000 per 60s exceeded for UserByModelByMinuteUncachedInputTokens`

- **Limit Type:** Input tokens per minute (1M tokens/60 seconds)
- **Scope:** Per Azure subscription/account, NOT per IP address
- **Model:** Applies to your specific Azure AI Foundry resource

## Solutions Implemented

### 1. ✅ Automatic Retry with Exponential Backoff (DONE)

The proxy now includes automatic retry logic:

```python
# Configurable via environment variables:
PROXY_MAX_RETRIES=5              # Max retry attempts
PROXY_INITIAL_RETRY_DELAY=5.0    # Initial wait time (seconds)
PROXY_MAX_RETRY_DELAY=120.0      # Maximum wait time (seconds)
```

**How it works:**
- Automatically detects rate limit errors
- Waits 60 seconds (as specified by Azure) before retrying
- Exponential backoff for subsequent retries
- Returns proper 429 error after max retries exceeded

**Restart your proxy to use this:**
```bash
# Kill existing proxy
pkill -f azure_foundry_proxy

# Restart with retry support
python code/OpenHands_SWE-Bench-Optimized/azure_foundry_proxy.py --port 8001
```

---

## Additional Solutions (Choose Based on Your Needs)

### 2. Reduce Concurrent Workers

Your `run_infer.sh` is processing multiple instances in parallel. Reduce workers:

```bash
# In your run command, reduce NUM_WORKERS
# Current: NUM_WORKERS=8 (or higher)
# Try:     NUM_WORKERS=2 or NUM_WORKERS=1

export NUM_WORKERS=2
bash evaluation/benchmarks/swe_bench_optimized/scripts/run_infer.sh \
  MODEL_CONFIG COMMIT_HASH AGENT EVAL_LIMIT MAX_ITER 2 ...
```

### 3. Enable Prompt Caching (Highly Recommended!)

Azure counts **uncached** tokens. Enable prompt caching to reduce token usage:

**In OpenHands config (`config.toml`):**
```toml
[llm]
# Add caching parameters
cache_prompts = true
cache_ttl = 3600  # Cache for 1 hour
```

**In the proxy (modify API call):**
```python
# In azure_foundry_proxy.py, add system message caching
api_params = {
    "model": claude_model,
    "max_tokens": max_tokens,
    "temperature": temperature,
    "messages": anthropic_messages,
    "system": [{
        "type": "text",
        "text": system_message,
        "cache_control": {"type": "ephemeral"}  # Enable caching
    }]
}
```

### 4. Request Higher Rate Limits from Azure

Contact Azure support to increase your quota:

1. Go to [Azure Portal](https://portal.azure.com)
2. Navigate to: **Azure AI Foundry** → Your Resource → **Quotas**
3. Request increase for:
   - `UserByModelByMinuteUncachedInputTokens`: Request 5M or 10M tokens/min
   - Consider upgrading to higher-tier subscription

### 5. Use Multiple Azure Resources (Load Balancing)

Create multiple Azure AI Foundry resources and rotate between them:

**Environment setup:**
```bash
export CLAUDE_RESOURCES="lucia-claude-eastus2,lucia-claude-westus,lucia-claude-northeu"
export CLAUDE_API_KEYS="key1,key2,key3"
```

**Proxy modification (round-robin):**
```python
# Add to azure_foundry_proxy.py
import itertools

RESOURCES = os.getenv("CLAUDE_RESOURCES", "lucia-claude-eastus2").split(",")
API_KEYS = os.getenv("CLAUDE_API_KEYS", CLAUDE_API_KEY).split(",")

resource_pool = itertools.cycle(zip(RESOURCES, API_KEYS))

# In chat_completions():
resource, api_key = next(resource_pool)
client = AnthropicFoundry(api_key=api_key, resource=resource)
```

### 6. Add Local Rate Limiting (Proactive)

Prevent hitting Azure limits by limiting locally:

```python
# Add to azure_foundry_proxy.py
from collections import deque
from threading import Lock

class TokenBucket:
    def __init__(self, rate=900000, per=60):  # 900K tokens per 60s (buffer)
        self.rate = rate
        self.per = per
        self.tokens = deque()
        self.lock = Lock()
    
    def consume(self, tokens):
        with self.lock:
            now = time.time()
            # Remove old tokens outside the window
            while self.tokens and self.tokens[0][0] < now - self.per:
                self.tokens.popleft()
            
            # Check if we can consume
            total = sum(t[1] for t in self.tokens)
            if total + tokens > self.rate:
                # Calculate wait time
                oldest = self.tokens[0][0] if self.tokens else now
                wait = (oldest + self.per) - now
                return False, wait
            
            self.tokens.append((now, tokens))
            return True, 0

bucket = TokenBucket()

# In chat_completions(), before API call:
estimated_tokens = len(str(messages)) * 4  # Rough estimate
allowed, wait_time = bucket.consume(estimated_tokens)
if not allowed:
    time.sleep(wait_time)
```

---

## Recommended Strategy

**For immediate relief:**
1. ✅ Use the updated proxy with retry logic (already done)
2. Reduce `NUM_WORKERS` to 2-4
3. Enable prompt caching

**For long-term:**
1. Request higher Azure quotas
2. Implement multiple Azure resources with load balancing
3. Add local token bucket rate limiting

---

## Monitoring Rate Limit Usage

Add monitoring to see how close you are to limits:

```python
# In azure_foundry_proxy.py
import threading
from collections import defaultdict

class RateLimitMonitor:
    def __init__(self):
        self.usage = defaultdict(int)
        self.lock = threading.Lock()
        self.start_time = time.time()
    
    def record(self, input_tokens):
        with self.lock:
            minute = int(time.time() / 60)
            self.usage[minute] += input_tokens
    
    def get_current_minute_usage(self):
        minute = int(time.time() / 60)
        return self.usage[minute]

monitor = RateLimitMonitor()

# After each successful API call:
monitor.record(openai_response['usage']['prompt_tokens'])
current_usage = monitor.get_current_minute_usage()
print(f"[Monitor] Current minute usage: {current_usage:,} / 1,000,000 tokens")
if current_usage > 800000:
    print(f"[Monitor] WARNING: Approaching rate limit!")
```

---

## Testing the Fix

1. Restart the proxy:
```bash
pkill -f azure_foundry_proxy
python code/OpenHands_SWE-Bench-Optimized/azure_foundry_proxy.py --port 8001
```

2. Test with custom retry settings:
```bash
PROXY_MAX_RETRIES=3 PROXY_INITIAL_RETRY_DELAY=60.0 \
python code/OpenHands_SWE-Bench-Optimized/azure_foundry_proxy.py --port 8001
```

3. Monitor logs for retry messages:
```
[Proxy] Rate limit hit (attempt 1/5)
[Proxy] Waiting 60 seconds before retry...
```

Now the proxy will automatically handle rate limits gracefully! 🎉
