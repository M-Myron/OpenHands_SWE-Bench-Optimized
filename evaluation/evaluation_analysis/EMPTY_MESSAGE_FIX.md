# Judge Evaluator - Empty Message Fix

## Problem Solved ✅

Assistant messages in trajectories were appearing as `[Empty message]` because modern LLMs use function/tool calling where the `content` field is often `null` when making tool calls.

## Solution

Integrated OpenHands' **function call converter** (discovered from TrajAttriBot) to convert function-calling messages into human-readable text format.

## What Changed

### 1. Updated `judge_evaluator.py`

- **Added imports**:
  ```python
  from openhands.llm.fn_call_converter import (
      convert_fncall_messages_to_non_fncall_messages,
      convert_from_multiple_tool_calls_to_single_tool_call_messages,
  )
  ```

- **Added `PromptBuilder.convert_fncall_messages()` method**:
  Converts messages with `tool_calls` to plain text format

- **Updated `build_prompt()` method**:
  Automatically converts messages before formatting if tools are provided

- **Updated `evaluate_instance()` and `run_evaluation()`**:
  Passes `tools` parameter through the evaluation pipeline

- **Data loading already includes tools**:
  The `load_data()` method already extracts tools from raw completions

### 2. Updated `extract_prompts.ipynb`

Added comprehensive demonstration showing:
- The root cause of empty messages
- How to manually convert messages
- Before/after comparison
- Complete explanation with examples

## How It Works

### Before Conversion
```json
{
  "role": "assistant",
  "content": null,  // Empty!
  "tool_calls": [{
    "function": {
      "name": "run_in_terminal",
      "arguments": "{\"command\": \"cat file.py\"}"
    }
  }]
}
```

### After Conversion
```json
{
  "role": "assistant",
  "content": "<function=run_in_terminal>\n<parameter=command>cat file.py</parameter>\n</function>",
  // tool_calls removed
}
```

## Usage

The fix is automatic! Just run the evaluator as normal:

```bash
python judge_evaluator.py \
    --input-files path/to/output.with_completions.jsonl.gz \
    --output-dir ./results \
    --model gpt-4.1 \
    --limit 10
```

The evaluator will:
1. Load messages and tools from the data
2. Automatically convert function-calling messages to text
3. Build prompts with readable assistant actions
4. Send to the LLM judge with full context

## Testing

Use `extract_prompts.ipynb` to:
1. Load any instance by ID
2. See the conversion in action
3. Compare original vs converted messages
4. Examine the full prompt sent to the judge

## Credits

Solution discovered by examining the TrajAttriBot codebase (`test.ipynb`), which uses the same OpenHands function call converter for trajectory analysis.

## Technical Details

The conversion happens in two steps:

1. **Split multiple tool calls**: `convert_from_multiple_tool_calls_to_single_tool_call_messages()`
   - Handles messages with multiple tool calls
   - Splits into separate messages

2. **Convert to text format**: `convert_fncall_messages_to_non_fncall_messages()`
   - Transforms tool calls into XML-style text format
   - Preserves all information in readable form
   - Removes tool_calls field

This approach:
- ✅ Preserves all information
- ✅ Makes trajectories human-readable
- ✅ Provides better context for LLM judge
- ✅ Handles edge cases (multiple calls, nested content, etc.)
