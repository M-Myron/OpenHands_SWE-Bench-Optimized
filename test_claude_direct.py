#!/usr/bin/env python3
"""
Quick test to verify Claude API works via Azure AI Foundry.
This tests the direct HTTP endpoint to understand the correct URL format.

Usage:
    export CLAUDE_API_KEY='your-api-key-here'
    export CLAUDE_RESOURCE='your-resource-name'  # optional
    python test_claude_direct.py
"""

import requests
import json
import os
import sys

# Get credentials from environment
API_KEY = os.getenv("CLAUDE_API_KEY")
RESOURCE = os.getenv("CLAUDE_RESOURCE", "lucia-claude-eastus2")

if not API_KEY:
    print("Error: CLAUDE_API_KEY environment variable is not set")
    print("Please set it before running:")
    print("  export CLAUDE_API_KEY='your-api-key-here'")
    sys.exit(1)

# Try different endpoint patterns
endpoints_to_try = [
    f"https://{RESOURCE}.services.ai.azure.com/models/chat/completions",
    f"https://{RESOURCE}.services.ai.azure.com/v1/messages",
    f"https://{RESOURCE}.services.ai.azure.com/models/v1/messages", 
    f"https://{RESOURCE}.services.ai.azure.com/models/claude-sonnet-4-5/chat/completions",
]

headers = {
    "api-key": API_KEY,
    "Content-Type": "application/json",
}

# Anthropic-style payload
anthropic_payload = {
    "model": "claude-sonnet-4-5",
    "max_tokens": 100,
    "messages": [
        {"role": "user", "content": "Hello! Say 'test successful' if you can read this."}
    ]
}

# OpenAI-style payload (alternative)
openai_payload = {
    "model": "claude-sonnet-4-5",
    "messages": [
        {"role": "user", "content": "Hello! Say 'test successful' if you can read this."}
    ],
    "max_tokens": 100,
}

print("=" * 70)
print("Testing Azure AI Foundry Claude Endpoints")
print("=" * 70)

for endpoint in endpoints_to_try:
    print(f"\nTrying endpoint: {endpoint}")
    print("-" * 70)
    
    # Try Anthropic-style first
    try:
        response = requests.post(
            endpoint,
            headers=headers,
            json=anthropic_payload,
            timeout=30
        )
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("✓ SUCCESS with Anthropic-style payload!")
            print(f"Response: {response.json()}")
            print(f"\n{'='*70}")
            print(f"✓ WORKING ENDPOINT: {endpoint}")
            print(f"{'='*70}")
            break
        else:
            print(f"Response: {response.text[:500]}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Try OpenAI-style
    try:
        response = requests.post(
            endpoint,
            headers=headers,
            json=openai_payload,
            timeout=30
        )
        print(f"Status Code (OpenAI-style): {response.status_code}")
        
        if response.status_code == 200:
            print("✓ SUCCESS with OpenAI-style payload!")
            print(f"Response: {response.json()}")
            print(f"\n{'='*70}")
            print(f"✓ WORKING ENDPOINT: {endpoint}")
            print(f"{'='*70}")
            break
        else:
            print(f"Response: {response.text[:500]}")
    except Exception as e:
        print(f"Error: {e}")

print("\n" + "=" * 70)
print("Test complete")
print("=" * 70)
