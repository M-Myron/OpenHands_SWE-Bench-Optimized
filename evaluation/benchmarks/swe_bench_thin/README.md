# Thin Docker Runtime for OpenHands SWE-bench Evaluation

## Overview

ThinDockerRuntime is a lightweight alternative to the full DockerRuntime that runs a pure-Python HTTP server (`thin_executor.py`) directly inside SWE-bench Docker containers. This eliminates the 8-15 minute runtime image build step by skipping the openhands runtime layer entirely.

## Architecture

```
Host (OpenHands)                    Container (SWE-bench image)
┌─────────────────────┐             ┌──────────────────────────┐
│ ThinDockerRuntime    │   HTTP      │ thin_executor.py         │
│ (thin_docker_       │ ◄─────────► │ (pure-Python HTTP server)│
│  runtime.py)        │  localhost   │                          │
│                     │  :random_port│ Uses container's testbed │
│ Sets RUNTIME=       │             │ Python + conda env       │
│ thin_docker         │             │                          │
└─────────────────────┘             └──────────────────────────┘
```

### Key Files
- **Runtime**: `openhands/runtime/impl/thin_docker/thin_docker_runtime.py`
- **Executor** (runs inside container): `openhands/runtime/impl/thin_docker/thin_executor.py`
- **Evaluation scripts**: `evaluation/benchmarks/swe_bench_thin/scripts/`
- **Inference runner**: `evaluation/benchmarks/swe_bench_thin/run_infer.py`

## Available Scripts

| Script | Purpose |
|--------|---------|
| `scripts/run_infer.sh` | Basic inference |
| `scripts/run_infer_instance_major.sh` | Instance-major inference (recommended) |
| `scripts/run_infer_and_eval.sh` | Combined inference + evaluation |
| `scripts/eval_infer.sh` | Evaluation only |
| `scripts/run_oracle_guided_infer_instance_major.sh` | Oracle-guided V1 |
| `scripts/run_oracle_guided_v2_infer_instance_major.sh` | Oracle-guided V2 |

All shell scripts set `export RUNTIME=thin_docker` and call existing `swe_bench_optimized/*.py` scripts. No separate Python files are needed.

## Usage

```bash
# Basic thin inference
bash evaluation/benchmarks/swe_bench_thin/scripts/run_infer.sh \
    llm.eval HEAD CodeActAgent 500 100 4

# Instance-major (recommended for large runs)
bash evaluation/benchmarks/swe_bench_thin/scripts/run_infer_instance_major.sh \
    llm.eval HEAD CodeActAgent 500 100 8 princeton-nlp/SWE-bench_Verified test

# Inference + evaluation
bash evaluation/benchmarks/swe_bench_thin/scripts/run_infer_and_eval.sh \
    llm.eval HEAD CodeActAgent 500 100 8 princeton-nlp/SWE-bench_Verified test

# Evaluation only
bash evaluation/benchmarks/swe_bench_thin/scripts/eval_infer.sh \
    path/to/output.jsonl "" princeton-nlp/SWE-bench_Verified test
```

Output defaults to `evaluation/evaluation_outputs/outputs_thin/`.

## Fixes Applied (April 2026)

### Crash Fixes
1. **Abstract method stubs** on `ThinDockerRuntime` — added `get_mcp_config()` and `call_tool_mcp()`
2. **FileWriteObservation handler** in `conversation_memory.py` — was crashing with `ValueError: Unknown observation type`
3. **`capture_output` / `text=True` removed** — testbed Python in SWE-bench images can be **Python 3.6** (e.g. `django__django__3.1`); `subprocess.run(capture_output=True, text=True)` is 3.7+. Switched to `Popen(..., stdout=PIPE, stderr=PIPE).communicate()` + manual decode.
4. **Directory view missing `impl_source: 'oh_aci'`** — caused `FileEditObservation.__str__()` to take the `prev_exist=False` branch and hit an assertion (`old_content should be empty if the file is new`). Added the extra so OH_ACI short-circuit fires.

### Output Format Alignment (thin executor → match regular runtime)

| Operation | What was wrong | Fix |
|-----------|---------------|-----|
| `run` (bash) | Status lines appeared twice | Removed suffix from content; `to_agent_observation()` adds them |
| `run` (bash) | Garbage text `<(pwd)"; echo "` from shell echoing | Removed `-i` flag from `Popen(['bash', '--norc', '--noprofile'])` (non-interactive) |
| `run` (bash) | Leading whitespace stripped | `cmd_output.strip()` → `cmd_output.rstrip()` |
| `edit view` (file) | No line numbers, `view_range` ignored | Added `cat -n` format with numbered lines, proper range |
| `edit view` (file) | No truncation for huge files | Added `_maybe_truncate()` with 16000-char limit (matches openhands-aci) |
| `edit view` (file) | Missing range validation | start_line OOR → error, end_line < start_line → error, end_line > num_lines → "NOTE: We only show up to N" |
| `edit view` (dir) | Missing hidden-files tail message | Now runs the exact two `find` commands aci runs and appends `"N hidden files/directories... ls -la"` |
| `edit create` | Returned `FileWriteObservation`, allowed overwrite | Returns `FileEditObservation`, rejects existing files |
| `edit str_replace` | Showed `+/-` diff lines | Shows numbered snippet (±4 lines) + review footer |
| `edit insert` | Only `"The file has been edited."` | Shows numbered snippet around insertion + review footer |
| `write` | `"File written successfully to..."` | Empty string `""` (matching regular runtime) |
| **routing**: OH_ACI `read` | `str_replace_editor view` creates `FileReadAction(impl_source='oh_aci', view_range=...)` — thin was routing ALL reads to `handle_file_read` (no line numbers, no view_range) | `route_action` read branch now checks `impl_source == 'oh_aci' or view_range is not None` → routes to `handle_file_edit(command='view', view_range=...)` |

### Remaining Known Differences (ranked by result impact)

| Gap | Frequency | Estimated impact on pass@1 | Severity |
|---|---|---|---|
| `str_replace` whitespace fallback uses `.strip()` instead of aci's `_match_and_strip_indent` | Medium | Wastes 1-2 iterations on edits with wrong indent → costs iter budget on hard instances | **Medium** (~1-3% abs) |
| Linter hints missing after edits | Every edit | Agent occasionally misses its own syntax errors for one extra iteration | **Low-Medium** |
| `CmdOutputMetadata` prefix/suffix exact format | Every bash cmd | **Unverified** — needs diffing against `action_execution_server.py` | **Unknown** |
| `undo_edit` — not implemented (returns error) | Rare | Agent adapts; most agents overwrite instead | Low |
| Binary file detection (`binaryornot`) | Rare | Thin would `UnicodeDecodeError` → error obs | Low |
| Encoding auto-detection — UTF-8 only | Rare | Non-utf8 fixtures would error | Low |
| Markdown conversion for `.pdf/.docx` | ~Never in SWE-bench | Zero | Zero |
| File permissions not preserved | N/A | Minimal | Zero |
| `browse` / `IPython` / `MCP` | N/A | Return errors (intentional) | Zero |

**Aggregate estimated drop vs. original runtime: ~1-5% absolute pass@1**, dominated by the str_replace indent fallback. All other items are edge cases.

### Next verification steps (for continuation)
1. **Diff `CmdOutputMetadata.prefix/suffix` format** between thin `BashSession.execute` and `openhands/runtime/action_execution_server.py` bash handling. Agents trained on exact suffix strings (`[The command completed with exit code N]`) may misparse if different.
2. **Port aci's `_match_and_strip_indent`** into `thin_executor.handle_file_edit` str_replace to cut the 1-3% iter-waste loss.
3. **Side-by-side trajectory diff** using `evaluation/evaluation_analysis/inspect_response.ipynb`: point cells 7/9 at matching instance under `outputs/` vs `outputs_thin/`.

### Alternative: install openhands_aci in container
For guaranteed bit-exact parity, create a tiny venv at `/tmp/thin_py/` with Python 3.10+ and `openhands_aci` at container startup, then have `handle_file_edit` delegate to `openhands_aci.editor.editor.OHEditor`. This eliminates all re-implementation drift. Not currently implemented — the stdlib re-implementation is "good enough" for SWE-bench eval.

## How It Works

1. `ThinDockerRuntime` starts a raw SWE-bench Docker container
2. Copies `thin_executor.py` into the container at `/tmp/`
3. Starts it with the container's testbed Python
4. All actions (bash, file read/write/edit) go through HTTP to thin executor
5. Docker cleanup is skipped (`_WORKER_RUNTIME == 'docker'` check in instance_major)

## Verifying Output Parity

To compare thin vs regular runtime outputs, use the notebook at `evaluation/evaluation_analysis/inspect_response.ipynb`. Point cell 7 at both:
- `outputs_thin/.../llm_completions/<instance>/` (thin)
- `outputs/.../llm_completions/<instance>/` (regular)

And compare the tool observation content. The observation text seen by the LLM should now be identical.
