# OpenHands Agent Message Flow & Prompt Organization Report

## Executive Summary

This report explains how OpenHands organizes chat history into prompts during agent execution and how this relates to the training data conversion process.

---

## Part 1: Agent Runtime Message Organization

### Overview

During agent execution, OpenHands **does NOT** send a single concatenated text prompt. Instead, it maintains a **structured conversation history** as a list of `Message` objects, which are then serialized to the LLM's expected format.

### Message Structure

Each `Message` object contains:

```python
class Message(BaseModel):
    role: Literal["user", "system", "assistant", "tool"]
    content: Sequence[TextContent | ImageContent]  # List of content blocks
    tool_calls: list[MessageToolCall] | None
    tool_call_id: str | None
    name: str | None
    # ... other fields
```

### The Conversation Flow Pipeline

#### 1. **Event Processing → Message Conversion**
   - Location: `openhands/memory/conversation_memory.py`
   - Class: `ConversationMemory.process_events()`
   
   The agent's actions and observations are converted to messages:
   
   ```python
   # Events (internal representation)
   [SystemMessageAction, MessageAction, CmdRunAction, CmdOutputObservation, ...]
   
   # ↓ Converted to ↓
   
   # Messages (LLM format)
   [
       Message(role="system", content=[TextContent(...)]),
       Message(role="user", content=[TextContent(...)]),
       Message(role="assistant", content=[TextContent(...)], tool_calls=[...]),
       Message(role="tool", content=[TextContent(...)], tool_call_id="..."),
       ...
   ]
   ```

#### 2. **Message List Serialization**
   - Location: `openhands/llm/llm.py`
   - Method: `format_messages_for_llm()`
   
   Messages are serialized to dictionaries:
   
   ```python
   # Message objects
   messages = [Message(...), Message(...), ...]
   
   # ↓ Serialized to ↓
   
   # Dictionary format for LLM API
   [
       {"role": "system", "content": "..."},
       {"role": "user", "content": "..."},
       {"role": "assistant", "content": "...", "tool_calls": [...]},
       {"role": "tool", "content": "...", "tool_call_id": "..."},
   ]
   ```

#### 3. **Two Serialization Modes**

The `Message.to_chat_dict()` method uses different serializers:

**A. List Serializer (Default for Function Calling/Vision/Caching)**
```python
{
    "role": "assistant",
    "content": [
        {"type": "text", "text": "I'll read the file"},
        {"type": "image_url", "image_url": {"url": "..."}}  # for vision
    ],
    "tool_calls": [{"id": "...", "type": "function", "function": {...}}]
}
```

**B. String Serializer (For Models Without Advanced Features)**
```python
{
    "role": "assistant",
    "content": "I'll read the file"  # Single string
}
```

#### 4. **Function Calling Conversion**
   - Location: `openhands/llm/fn_call_converter.py`
   
   **For models WITHOUT native function calling support**, OpenHands converts:
   
   ```python
   # FROM: Structured function calls
   {
       "role": "assistant",
       "tool_calls": [{"type": "function", "function": {"name": "execute_bash", ...}}]
   }
   
   # TO: XML-like text format
   {
       "role": "assistant",
       "content": """
       <function=execute_bash>
       <parameter=command>ls -la</parameter>
       </function>
       """
   }
   ```
   
   And back after the model responds.

#### 5. **Final LLM Invocation**
   - Location: `openhands/llm/llm.py`
   - Method: `completion()`
   
   ```python
   # The messages list is sent to the LLM provider
   response = litellm.completion(
       model="gpt-4",
       messages=[...],  # List of message dicts
       tools=[...],     # Tool definitions
       ...
   )
   ```

### Key Insight: The LLM Provider Handles Chat Template

**Important:** OpenHands does NOT apply a chat template itself. Instead:

1. **OpenHands** sends structured messages: `[{"role": "user", "content": "..."}, ...]`
2. **The LLM provider** (via LiteLLM/API) applies the chat template internally
3. **The tokenizer's `apply_chat_template()`** is what converts messages to the final token sequence

Example of what happens inside the LLM provider:
```python
# What OpenHands sends
messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Hello"},
]

# What the tokenizer does (internally at provider)
tokenizer.apply_chat_template(messages) 
# → "<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
```

---

## Part 2: Training Data Conversion Process

### Why Content is List Format in Raw Completions

When you examine `output.with_completions.jsonl.gz`, the messages have content in **list format**:

```json
{
    "role": "assistant",
    "content": [
        {"type": "text", "text": "I'll help you fix this issue"}
    ]
}
```

**Reason:** This is the **serialized output from `Message.to_chat_dict()`** when:
- Function calling is enabled (most common for CodeActAgent)
- Vision is enabled
- Prompt caching is enabled

This is the **exact format** that was sent to the LLM API during inference.

### The Conversion Problem

For fine-tuning, you need to convert this back to the format that `tokenizer.apply_chat_template()` expects:

```python
# Raw completion format (won't work with tokenizer)
{"role": "assistant", "content": [{"type": "text", "text": "..."}]}

# ↓ Convert to ↓

# Training format (works with tokenizer)
{"role": "assistant", "content": "..."}
```

### Your Notebook's Solution

The `normalize_message_content()` function extracts text from the list structure:

```python
def normalize_message_content(messages: list[dict]) -> list[dict]:
    normalized = []
    for msg in messages:
        content = msg.get('content', '')
        
        if isinstance(content, list):
            # Extract text from [{"type": "text", "text": "..."}]
            text_parts = [
                item.get('text', '') 
                for item in content 
                if isinstance(item, dict) and item.get('type') == 'text'
            ]
            msg['content'] = '\n'.join(text_parts)  # Convert to string
        
        normalized.append(msg)
    return normalized
```

This produces messages that can be tokenized:

```python
tokenizer.apply_chat_template(normalized_messages)
# ✓ Works! Converts to proper token sequence
```

---

## Part 3: Complete Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ INFERENCE TIME (Agent Execution)                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Events (Actions + Observations)                            │
│    ↓                                                        │
│  ConversationMemory.process_events()                        │
│    ↓                                                        │
│  List[Message] objects (internal)                           │
│    ↓                                                        │
│  LLM.format_messages_for_llm()                              │
│    ↓                                                        │
│  List[dict] with content as list                            │
│  [{"role": "...", "content": [{"type": "text", ...}]}]      │
│    ↓                                                        │
│  (Optional) convert_fncall_messages_to_non_fncall_messages  │
│    ↓                                                        │
│  litellm.completion(messages=[...])                         │
│    ↓                                                        │
│  LLM Provider applies chat template internally              │
│    ↓                                                        │
│  Response with tool_calls or text                           │
│    ↓                                                        │
│  Logged to output.with_completions.jsonl.gz                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ TRAINING TIME (Data Conversion)                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Load output.with_completions.jsonl.gz                      │
│    ↓                                                        │
│  Extract messages (still in list format)                    │
│    ↓                                                        │
│  convert_from_multiple_tool_calls_to_single_tool_call_messages│
│    ↓                                                        │
│  convert_fncall_messages_to_non_fncall_messages             │
│  (Converts tool_calls to XML text format)                   │
│    ↓                                                        │
│  normalize_message_content()                                │
│  (Converts content from list to string)                     │
│    ↓                                                        │
│  Tokenize with apply_chat_template()                        │
│    ↓                                                        │
│  Filter by token length (< 128k)                            │
│    ↓                                                        │
│  Save as training JSONL                                     │
│  [{"messages": [{"role": "...", "content": "..."}]}]        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Part 4: Answering Your Question

> "In a interaction round during the trajectory, openhands should organize all history chat into a single prompt and send to the LLM right?"

**Answer:** **Not exactly.** Here's the nuance:

1. **OpenHands sends a LIST of structured messages**, not a single concatenated prompt string
2. **The LLM API/tokenizer** is responsible for converting this list into the final prompt string using chat templates
3. **Each interaction round** sends the FULL conversation history as a list:

```python
# Round 1
messages = [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "Fix the bug"},
]

# Round 2 (after assistant responds and observes)
messages = [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "Fix the bug"},
    {"role": "assistant", "content": "...", "tool_calls": [...]},
    {"role": "tool", "content": "command output", "tool_call_id": "..."},
]

# Round 3 (continues building up)
messages = [... all previous + new messages ...]
```

4. **The final output** in `output.with_completions.jsonl.gz` contains:
   - `messages`: The ENTIRE conversation history as a list
   - Each message has content in list format (for function calling compatibility)
   - This is the exact format sent to the LLM during the LAST turn

---

## Part 5: Standard Format Summary

### For LLM APIs (OpenHands sends this):
```python
[
    {"role": "system", "content": [{"type": "text", "text": "..."}]},
    {"role": "user", "content": [{"type": "text", "text": "..."}]},
    {"role": "assistant", "content": [...], "tool_calls": [...]},
    {"role": "tool", "content": [...], "tool_call_id": "...", "name": "..."}
]
```

### For Training (Tokenizer expects this):
```python
[
    {"role": "system", "content": "..."},  # String content
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
]
```

### Conversion Required:
- **List content → String content**
- **Structured tool_calls → Text representation** (for non-function-calling models)
- **Apply chat template** to get final token IDs

---

## Key Takeaways

1. **OpenHands maintains structured message lists**, not monolithic prompt strings
2. **The LLM provider applies chat templates**, not OpenHands
3. **Training data needs normalization** from list format to string format
4. **Function calling is converted** to/from text format for non-supporting models
5. **Each turn sends the full history**, accumulated from all previous turns
6. **The tokenizer's `apply_chat_template()`** is the final step that creates the actual prompt tokens

This architecture allows OpenHands to work with multiple LLM providers while maintaining a consistent internal representation.
