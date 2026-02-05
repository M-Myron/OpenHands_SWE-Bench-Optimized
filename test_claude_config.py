#!/usr/bin/env python3
"""
Test Claude via Azure AI Foundry and identify the correct LiteLLM configuration.

This script:
1. Tests using the anthropic-foundry library (your working code)
2. Tests using LiteLLM with various configurations
3. Identifies the correct config for OpenHands

Run: 
    export CLAUDE_API_KEY='your-api-key-here'
    export CLAUDE_RESOURCE='your-resource-name'  # optional, defaults to lucia-claude-eastus2
    python test_claude_config.py
"""

import os
import sys

# Get credentials from environment
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
CLAUDE_RESOURCE = os.getenv("CLAUDE_RESOURCE", "lucia-claude-eastus2")

if not CLAUDE_API_KEY:
    print("Error: CLAUDE_API_KEY environment variable is not set")
    print("Please set it before running:")
    print("  export CLAUDE_API_KEY='your-api-key-here'")
    sys.exit(1)

print("=" * 70)
print("Step 1: Testing with Anthropic Foundry (Your Working Code)")
print("=" * 70)

try:
    from anthropic import AnthropicFoundry
    
    client = AnthropicFoundry(
        api_key=CLAUDE_API_KEY,
        resource=CLAUDE_RESOURCE
    )
    
    message = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=100,
        messages=[
            {"role": "user", "content": "Say 'Anthropic Foundry works!' if you can read this."}
        ]
    )
    
    print("✓ Anthropic Foundry library works!")
    print(f"Response: {message.content[0].text}")
    print(f"Model: {message.model}")
    
    # Inspect the actual API endpoint used
    print(f"\nClient base URL: {client._client.base_url if hasattr(client, '_client') else 'N/A'}")
    
except Exception as e:
    print(f"✗ Anthropic Foundry test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("Step 2: Testing with LiteLLM - Various Configurations")
print("=" * 70)

try:
    import litellm
    from litellm import completion
    
    # Configuration options to try
    configs = [
        {
            "name": "anthropic/ with Azure base_url",
            "model": "anthropic/claude-sonnet-4-5",
            "api_key": CLAUDE_API_KEY,
            "api_base": f"https://{CLAUDE_RESOURCE}.services.ai.azure.com/models",
        },
        {
            "name": "claude-sonnet-4-5 with custom_llm_provider",
            "model": "claude-sonnet-4-5",
            "api_key": CLAUDE_API_KEY,
            "api_base": f"https://{CLAUDE_RESOURCE}.services.ai.azure.com/models",
            "custom_llm_provider": "anthropic",
        },
        {
            "name": "openai/ prefix with Azure endpoint (OpenAI-compatible)",
            "model": "openai/claude-sonnet-4-5",
            "api_key": CLAUDE_API_KEY,
            "api_base": f"https://{CLAUDE_RESOURCE}.services.ai.azure.com/models",
        },
    ]
    
    for config in configs:
        print(f"\nTrying: {config['name']}")
        print("-" * 70)
        
        try:
            # Set environment variables for Anthropic if needed
            os.environ["ANTHROPIC_API_KEY"] = config["api_key"]
            
            response = completion(
                model=config["model"],
                messages=[
                    {"role": "user", "content": "Say 'LiteLLM works!' if you can read this."}
                ],
                api_key=config.get("api_key"),
                api_base=config.get("api_base"),
                custom_llm_provider=config.get("custom_llm_provider"),
                max_tokens=100,
            )
            
            print(f"✓ SUCCESS!")
            print(f"Response: {response.choices[0].message.content}")
            print(f"Model: {response.model}")
            
            print(f"\n{'='*70}")
            print(f"✓ WORKING LITELLM CONFIG:")
            print(f"  model: {config['model']}")
            print(f"  api_base: {config.get('api_base')}")
            if config.get('custom_llm_provider'):
                print(f"  custom_llm_provider: {config['custom_llm_provider']}")
            print(f"{'='*70}")
            
            # Show how to use in config.toml
            print(f"\nFor config.toml, use:")
            print(f'  model="{config["model"]}"')
            print(f'  api_key="YOUR_API_KEY"')
            print(f'  base_url="{config.get("api_base")}"')
            if config.get('custom_llm_provider'):
                print(f'  custom_llm_provider="{config["custom_llm_provider"]}"')
            
            break  # Stop after first success
            
        except Exception as e:
            print(f"✗ Failed: {e}")
    
except ImportError:
    print("✗ LiteLLM not installed. Install with: pip install litellm")
except Exception as e:
    print(f"✗ LiteLLM test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("Testing Complete")
print("=" * 70)
