#!/usr/bin/env python3
"""
Azure AI Foundry Claude Proxy Server

This proxy server provides an OpenAI-compatible API endpoint that forwards requests
to Azure AI Foundry's Claude models using the anthropic-foundry library.

This allows LiteLLM (used by OpenHands) to access Azure AI Foundry Claude models
through a standard OpenAI-compatible interface.

Usage:
    python azure_foundry_proxy.py --port 8001

Then configure OpenHands to use:
    model="openai/claude-sonnet-4-5"
    base_url="http://localhost:8001/v1"
"""

import argparse
import json
import os
import time
from typing import Optional
from functools import wraps

from anthropic import AnthropicFoundry, RateLimitError
from flask import Flask, request, jsonify, Response
import uuid

app = Flask(__name__)

# Azure AI Foundry configuration
CLAUDE_RESOURCE = os.getenv("CLAUDE_RESOURCE", "lucia-claude-eastus2")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")

if not CLAUDE_API_KEY:
    raise ValueError("CLAUDE_API_KEY environment variable must be set")

# Initialize Anthropic Foundry client
client = AnthropicFoundry(api_key=CLAUDE_API_KEY, resource=CLAUDE_RESOURCE)

# Model mapping (OpenAI-style names to Claude model names)
MODEL_MAPPING = {
    "claude-sonnet-4-5": "claude-sonnet-4-5",
    "claude-opus-4-1": "claude-opus-4-1",
    "claude-haiku-4-5": "claude-haiku-4-5",
    "gpt-4": "claude-sonnet-4-5",  # Alias
    "gpt-3.5-turbo": "claude-haiku-4-5",  # Alias
}

# Rate limiting configuration
MAX_RETRIES = int(os.getenv("PROXY_MAX_RETRIES", "5"))
INITIAL_RETRY_DELAY = float(os.getenv("PROXY_INITIAL_RETRY_DELAY", "5.0"))
MAX_RETRY_DELAY = float(os.getenv("PROXY_MAX_RETRY_DELAY", "120.0"))


def retry_with_backoff(func):
    """Decorator to retry API calls with exponential backoff on rate limit errors."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        retry_count = 0
        delay = INITIAL_RETRY_DELAY
        
        while retry_count <= MAX_RETRIES:
            try:
                return func(*args, **kwargs)
            except RateLimitError as e:
                retry_count += 1
                if retry_count > MAX_RETRIES:
                    print(f"[Proxy] Max retries ({MAX_RETRIES}) reached. Giving up.")
                    raise
                
                # Extract wait time from error message if available
                error_msg = str(e)
                wait_time = delay
                if "60 seconds" in error_msg.lower():
                    wait_time = 60
                elif "please wait" in error_msg.lower():
                    # Try to extract number from message
                    import re
                    match = re.search(r'wait (\d+) seconds', error_msg.lower())
                    if match:
                        wait_time = int(match.group(1))
                
                wait_time = min(wait_time, MAX_RETRY_DELAY)
                
                print(f"[Proxy] Rate limit hit (attempt {retry_count}/{MAX_RETRIES})")
                print(f"[Proxy] Waiting {wait_time} seconds before retry...")
                print(f"[Proxy] Error: {error_msg}")
                
                time.sleep(wait_time)
                
                # Exponential backoff for next attempt
                delay = min(delay * 2, MAX_RETRY_DELAY)
        
        return func(*args, **kwargs)
    
    return wrapper


def convert_openai_to_anthropic_messages(openai_messages):
    """Convert OpenAI message format to Anthropic format."""
    anthropic_messages = []
    
    for msg in openai_messages:
        role = msg["role"]
        content = msg["content"]
        
        # Skip system messages - Anthropic handles them differently
        if role == "system":
            continue
        
        # Convert assistant to assistant, user to user
        anthropic_messages.append({
            "role": role if role in ["user", "assistant"] else "user",
            "content": content
        })
    
    return anthropic_messages


def convert_anthropic_to_openai_response(anthropic_response, model_name):
    """Convert Anthropic response to OpenAI format."""
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": anthropic_response.content[0].text
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": anthropic_response.usage.input_tokens,
            "completion_tokens": anthropic_response.usage.output_tokens,
            "total_tokens": anthropic_response.usage.input_tokens + anthropic_response.usage.output_tokens
        }
    }


@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """OpenAI-compatible chat completions endpoint."""
    start_time = time.time()
    
    try:
        data = request.json
        parse_time = time.time()
        
        # Extract parameters from the request
        # These come from your config.toml via LiteLLM
        model = data.get("model", "claude-sonnet-4-5")
        messages = data.get("messages", [])
        max_tokens = data.get("max_tokens") or data.get("max_completion_tokens") or 4096
        temperature = data.get("temperature", 0.0)  # Default to 0.0 for deterministic results
        stream = data.get("stream", False)
        
        # Additional OpenAI parameters that might be passed
        top_p = data.get("top_p")
        stop = data.get("stop")
        
        # Map model name
        claude_model = MODEL_MAPPING.get(model, model)
        
        # Convert messages
        anthropic_messages = convert_openai_to_anthropic_messages(messages)
        
        if not anthropic_messages:
            return jsonify({"error": "No valid messages provided"}), 400
        
        conversion_time = time.time()
        
        print(f"\n{'='*70}")
        print(f"[Proxy] Request #{uuid.uuid4().hex[:6]}")
        print(f"  Model: {claude_model}")
        print(f"  Messages: {len(anthropic_messages)}")
        print(f"  Max Tokens: {max_tokens}")
        print(f"  Temperature: {temperature}")
        print(f"{'='*70}")
        
        # Build Anthropic API call parameters
        api_params = {
            "model": claude_model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": anthropic_messages,
        }
        
        # Add optional parameters if provided
        if top_p is not None:
            api_params["top_p"] = top_p
        if stop is not None:
            # Anthropic supports stop_sequences
            api_params["stop_sequences"] = stop if isinstance(stop, list) else [stop]
        
        # Call Anthropic API with retry logic
        @retry_with_backoff
        def make_api_call():
            return client.messages.create(**api_params)
        
        api_call_start = time.time()
        response = make_api_call()
        api_call_time = time.time() - api_call_start
        
        # Convert response
        openai_response = convert_anthropic_to_openai_response(response, model)
        
        total_time = time.time() - start_time
        proxy_overhead = total_time - api_call_time
        
        print(f"\n{'='*70}")
        print(f"[Proxy] Response Complete")
        print(f"  Input Tokens: {openai_response['usage']['prompt_tokens']}")
        print(f"  Output Tokens: {openai_response['usage']['completion_tokens']}")
        print(f"  Total Tokens: {openai_response['usage']['total_tokens']}")
        print(f"  ---")
        print(f"  API Call Time: {api_call_time:.3f}s")
        print(f"  Proxy Overhead: {proxy_overhead*1000:.1f}ms")
        print(f"  Total Time: {total_time:.3f}s")
        print(f"{'='*70}\n")
        
        return jsonify(openai_response)
    
    except RateLimitError as e:
        print(f"[Proxy] Rate Limit Error (after retries): {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": {
                "message": f"Rate limit exceeded after {MAX_RETRIES} retries: {str(e)}",
                "type": "rate_limit_error",
                "code": "rate_limit_exceeded"
            }
        }), 429
    
    except Exception as e:
        print(f"[Proxy] Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": {
                "message": str(e),
                "type": "proxy_error"
            }
        }), 500


@app.route('/v1/models', methods=['GET'])
def list_models():
    """List available models."""
    return jsonify({
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "azure-ai-foundry"
            }
            for model_id in MODEL_MAPPING.keys()
        ]
    })


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "resource": CLAUDE_RESOURCE})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Azure AI Foundry Claude Proxy Server')
    parser.add_argument('--port', type=int, default=8001, help='Port to run the proxy server on')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    args = parser.parse_args()
    
    print("=" * 70)
    print("Azure AI Foundry Claude Proxy Server")
    print("=" * 70)
    print(f"Resource: {CLAUDE_RESOURCE}")
    print(f"Listening on: http://{args.host}:{args.port}")
    print("")
    print("Rate Limiting Configuration:")
    print(f"  Max Retries: {MAX_RETRIES}")
    print(f"  Initial Retry Delay: {INITIAL_RETRY_DELAY}s")
    print(f"  Max Retry Delay: {MAX_RETRY_DELAY}s")
    print("")
    print("OpenAI-compatible endpoint:")
    print(f"  http://localhost:{args.port}/v1/chat/completions")
    print("")
    print("Configure OpenHands with:")
    print(f'  model="openai/claude-sonnet-4-5"')
    print(f'  base_url="http://localhost:{args.port}/v1"')
    print(f'  api_key="dummy"')
    print("")
    print("Environment Variables:")
    print("  PROXY_MAX_RETRIES - Maximum retry attempts (default: 5)")
    print("  PROXY_INITIAL_RETRY_DELAY - Initial retry delay in seconds (default: 5.0)")
    print("  PROXY_MAX_RETRY_DELAY - Maximum retry delay in seconds (default: 120.0)")
    print("=" * 70)
    
    app.run(host=args.host, port=args.port, debug=False)
