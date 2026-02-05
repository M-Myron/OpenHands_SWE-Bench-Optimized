#!/usr/bin/env python3
"""
Test script to verify Claude API configuration works with LiteLLM.
This script tests the Claude models configured for SWE-bench evaluation.

Usage:
    python test_claude_litellm.py
    
    # Or test specific model:
    python test_claude_litellm.py --model claude-sonnet-4-5
"""

import os
import sys
from pathlib import Path

# Add the OpenHands directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from openhands.core.config import LLMConfig, load_from_toml


def test_claude_with_litellm(model_config_name: str = "eval_swebench-verified_claude-sonnet-4-5"):
    """Test Claude API using LiteLLM (as OpenHands does)."""
    
    print("=" * 70)
    print(f"Testing Claude with LiteLLM - Config: {model_config_name}")
    print("=" * 70)
    
    # Load configuration from config.toml
    config_file = Path(__file__).parent / "config.toml"
    if not config_file.exists():
        print(f"Error: config.toml not found at {config_file}")
        return False
    
    # Load the LLM config
    configs = load_from_toml(config_file)
    
    # Get the specific LLM config
    llm_config = None
    if hasattr(configs, 'llms') and model_config_name in configs.llms:
        llm_config = configs.llms[model_config_name]
    else:
        print(f"Error: Config '{model_config_name}' not found in config.toml")
        print(f"Available configs: {list(configs.llms.keys()) if hasattr(configs, 'llms') else []}")
        return False
    
    print(f"\nConfiguration loaded:")
    print(f"  Model: {llm_config.model}")
    print(f"  Base URL: {llm_config.base_url}")
    print(f"  Temperature: {llm_config.temperature}")
    print(f"  Max tokens: {llm_config.max_output_tokens}")
    print(f"  Timeout: {llm_config.timeout}")
    print()
    
    try:
        # Import LiteLLM
        import litellm
        from litellm import completion
        
        # Configure LiteLLM with retry settings
        litellm.num_retries = llm_config.num_retries
        litellm.request_timeout = llm_config.timeout
        
        # For Azure Anthropic, set environment variables
        os.environ["AZURE_API_KEY"] = llm_config.api_key
        os.environ["AZURE_API_BASE"] = llm_config.base_url
        
        print("Sending test request to Claude API via LiteLLM...")
        print("-" * 70)
        
        # Make a test completion request
        response = completion(
            model=llm_config.model,
            messages=[
                {
                    "role": "user",
                    "content": "Hello! Please respond with a short greeting to confirm the API is working."
                }
            ],
            api_key=llm_config.api_key,
            api_base=llm_config.base_url,
            temperature=llm_config.temperature,
            max_tokens=llm_config.max_output_tokens,
        )
        
        print("✓ API Request Successful!")
        print()
        print(f"Model: {response.model}")
        print(f"Response: {response.choices[0].message.content}")
        print()
        print("Token usage:")
        if hasattr(response, 'usage'):
            print(f"  Input tokens: {response.usage.prompt_tokens}")
            print(f"  Output tokens: {response.usage.completion_tokens}")
            print(f"  Total tokens: {response.usage.total_tokens}")
        
        print()
        print("=" * 70)
        print("✓ Claude API configuration is working correctly!")
        print("=" * 70)
        return True
        
    except Exception as e:
        print(f"✗ Error testing Claude API: {e}")
        print()
        print("Troubleshooting tips:")
        print("1. Verify your API key is correct")
        print("2. Check that the Azure resource endpoint is accessible")
        print("3. Ensure the model name matches your Azure deployment")
        print("4. Check if litellm is installed: pip install litellm")
        print()
        import traceback
        traceback.print_exc()
        return False


def test_all_claude_configs():
    """Test all Claude configurations."""
    configs_to_test = [
        "eval_swebench-verified_claude-sonnet-4-5",
        "eval_swebench-verified_claude-opus-4-1", 
        "eval_swebench-verified_claude-haiku-4-5",
    ]
    
    results = {}
    for config_name in configs_to_test:
        print("\n")
        success = test_claude_with_litellm(config_name)
        results[config_name] = success
    
    print("\n")
    print("=" * 70)
    print("Summary:")
    print("=" * 70)
    for config_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {config_name}")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Claude API configuration for SWE-bench")
    parser.add_argument(
        "--model",
        type=str,
        choices=["sonnet", "opus", "haiku", "all"],
        default="sonnet",
        help="Which Claude model to test (default: sonnet)",
    )
    
    args = parser.parse_args()
    
    if args.model == "all":
        test_all_claude_configs()
    else:
        model_map = {
            "sonnet": "eval_swebench-verified_claude-sonnet-4-5",
            "opus": "eval_swebench-verified_claude-opus-4-1",
            "haiku": "eval_swebench-verified_claude-haiku-4-5",
        }
        test_claude_with_litellm(model_map[args.model])
