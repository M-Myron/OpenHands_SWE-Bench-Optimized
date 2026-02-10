# SWE-Bench Trajectory Judge Evaluator

A comprehensive evaluation system for analyzing agent trajectories on SWE-Bench tasks using an LLM judge. This system implements a sophisticated failure taxonomy and intent-based correctness evaluation.

## Overview

The evaluator analyzes agent performance by comparing:
- **Golden fix** (ground truth patch)
- **Golden tests** (ground truth test cases)
- **Agent trajectory** (reasoning steps, tool usage, exploration)
- **Agent patch** (agent's proposed fix)
- **Test results** (agent's test execution outputs)

Unlike simple pass/fail metrics, this judge evaluates:
- **Intent alignment**: Does the agent understand the bug?
- **Patch correctness**: Does the fix match golden semantics?
- **Partial fixes**: Core intent covered but edge cases missed?
- **Accidental passes**: Tests pass but semantics are wrong?
- **Process quality**: RCA, exploration, validation rigor

## Features

- ✅ **Robust LLM calling** with exponential backoff and retry logic
- ✅ **Checkpoint/resume** support for long-running evaluations
- ✅ **Structured JSON output** with comprehensive failure taxonomy
- ✅ **Multi-dimensional quality scores** (0-4 scale)
- ✅ **Detailed failure analysis** (external vs internal, inferability)
- ✅ **Result analysis tools** for aggregation and visualization

## Installation

```bash
# Ensure you have the required dependencies
pip install openai pandas tqdm
```

## Usage

### 1. Run Evaluation

Basic usage:

```bash
python judge_evaluator.py \
  --input-files /path/to/output.with_completions.jsonl.gz \
  --output-dir ./evaluation_results \
  --model gpt-4o
```

With options:

```bash
python judge_evaluator.py \
  --input-files /path/to/file1.jsonl.gz /path/to/file2.jsonl.gz \
  --output-dir ./evaluation_results \
  --model gpt-4o \
  --limit 100 \
  --max-retries 5 \
  --checkpoint-interval 10
```

**Arguments:**
- `--input-files`: One or more gzipped JSONL files with trajectory data
- `--output-dir`: Directory to save results (created if doesn't exist)
- `--model`: LLM model to use (default: `gpt-4o`)
- `--limit`: Limit evaluation to N instances (for testing)
- `--max-retries`: Max retry attempts for failed LLM calls (default: 5)
- `--checkpoint-interval`: Save checkpoint every N instances (default: 10)

### 2. Analyze Results

Print overview and statistics:

```bash
python analyze_results.py \
  --results ./evaluation_results/evaluation_results.jsonl
```

Export to CSV for further analysis:

```bash
python analyze_results.py \
  --results ./evaluation_results/evaluation_results.jsonl \
  --export-csv ./analysis.csv
```

Get detailed report for specific instance:

```bash
python analyze_results.py \
  --results ./evaluation_results/evaluation_results.jsonl \
  --instance django__django-12184
```

## Output Format

### Files Generated

1. **`evaluation_results.jsonl`**: Full results with judge evaluations
2. **`checkpoint.jsonl`**: Incremental checkpoint (for resume)
3. **`evaluation_summary.json`**: Aggregated statistics

### JSON Schema

Each evaluation produces:

```json
{
  "instance_id": "django__django-12184",
  "resolved": true,
  "judge_evaluation": {
    "outcome": {
      "status": "partial_fix|true_pass|accidental_pass|wrong_fix|inconclusive",
      "final_test_state": "all_green|some_fail|not_run|unknown"
    },
    "intent": {
      "requirements": ["requirement 1", "requirement 2"],
      "edge_cases": ["edge case 1"],
      "non_functional_constraints": ["performance", "backwards compat"]
    },
    "patch_alignment": {
      "alignment_score": 3,
      "missing_requirements": ["edge case X"],
      "extra_behavior_changes": [],
      "accidental_pass_risk": "low|medium|high|unknown",
      "notes": "explanation"
    },
    "primary_failure": {
      "class": "external|internal|none",
      "inferability": "inferable_from_issue|inferable_from_golden|non_inferable|na",
      "reason_code": "INT_RCA_PARTIAL",
      "stage": "rca|design|implementation|..."
    },
    "secondary_failures": [...],
    "quality_scores": {
      "spec_alignment": 3,
      "repo_exploration": 2,
      "root_cause_quality": 3,
      "patch_correctness": 2,
      "validation_rigor": 2,
      "iteration_efficiency": 3
    },
    "evidence": {
      "golden_quotes": ["quote from golden"],
      "agent_quotes": ["quote from agent"],
      "commands_run": ["command 1"],
      "diff_comparison_notes": ["note 1"]
    },
    "narrative": {
      "one_paragraph_diagnosis": "summary of what went wrong",
      "counterfactual_fix": "what should have been done"
    }
  },
  "metadata": {
    "elapsed": 12.5,
    "tokens": 8432,
    "model": "gpt-4o"
  },
  "timestamp": "2026-02-03T10:30:00"
}
```

## Failure Taxonomy

### External Failures (Task/Dataset Issues)

- `EXT_INCOMPLETE_SPEC`: Issue description lacks key requirements
- `EXT_AMBIGUOUS_INTENT`: Multiple valid interpretations
- `EXT_NONINFERABLE_CONSTRAINT`: Golden enforces constraints not in spec
- `EXT_GOLDEN_EDGECASE_SURPRISE`: Unexpected edge case in golden tests
- `EXT_DATASET_ARTIFACT`: Flaky tests, wrong base commit, etc.

### Internal Failures (Agent Issues)

**Understanding Phase:**
- `INT_PARSE_MISREAD`: Misunderstood issue constraints
- `INT_SEARCH_INSUFFICIENT`: Failed to explore relevant code
- `INT_TESTS_NOT_INSPECTED`: Didn't check existing tests

**Root Cause Analysis:**
- `INT_RCA_WRONG`: Incorrect causal explanation
- `INT_RCA_PARTIAL`: Right direction, incomplete understanding

**Design/Implementation:**
- `INT_FIX_LOCATION_WRONG`: Patched wrong module/layer
- `INT_FIX_STRATEGY_WRONG_LEVEL`: Too invasive or too shallow
- `INT_IMPL_LOGIC_BUG`: Wrong logic/conditions
- `INT_EDGECASE_MISSED`: Missed discoverable edge cases
- `INT_API_MISUSE`: Incorrect API usage
- `INT_PERF_REGRESSION`: Performance degradation

**Validation:**
- `INT_VALIDATION_WEAK`: Insufficient testing
- `INT_TEST_OUTPUT_IGNORED`: Ignored failure messages

**Process:**
- `INT_TOOL_MISUSE`: Incorrect tool usage
- `INT_LOOPING`: Repeated actions without progress
- `INT_THRASHING`: Large rewrites without converging
- `INT_PREMATURE_STOP`: Stopped with known issues
- `INT_EVIDENCE_GAP`: Claims not supported by evidence

## Quality Scores (0-4)

Each dimension scored 0-4:

- **spec_alignment**: How well agent understood the issue
- **repo_exploration**: Thoroughness of code exploration
- **root_cause_quality**: Accuracy of bug diagnosis
- **patch_correctness**: Semantic correctness of fix
- **validation_rigor**: Testing and verification quality
- **iteration_efficiency**: Convergence without thrashing

**Score interpretation:**
- 0: Completely off track
- 1: Minimal progress, major issues
- 2: Partial success, significant gaps
- 3: Good, minor issues
- 4: Excellent

## Resuming Interrupted Runs

The evaluator automatically saves checkpoints. If interrupted:

```bash
# Just re-run the same command
python judge_evaluator.py \
  --input-files /path/to/output.with_completions.jsonl.gz \
  --output-dir ./evaluation_results \
  --model gpt-4o
```

Already-processed instances are skipped automatically.

## Example Workflow

```bash
# Step 1: Run evaluation (start small for testing)
python judge_evaluator.py \
  --input-files evaluation/evaluation_outputs/outputs/.../output.with_completions.jsonl.gz \
  --output-dir ./judge_results \
  --model gpt-4o \
  --limit 10

# Step 2: Check progress
cat ./judge_results/evaluation_summary.json

# Step 3: Analyze results
python analyze_results.py \
  --results ./judge_results/evaluation_results.jsonl

# Step 4: Export for further analysis
python analyze_results.py \
  --results ./judge_results/evaluation_results.jsonl \
  --export-csv ./judge_results/analysis.csv

# Step 5: Investigate specific failures
python analyze_results.py \
  --results ./judge_results/evaluation_results.jsonl \
  --instance django__django-12184
```

## Configuration

Edit `judge_evaluator.py` to change:
- Azure OpenAI endpoint/API key
- Retry timing (base_wait_time, max_wait_time)
- Model temperature (currently 0.0 for consistency)

## Tips

1. **Start small**: Use `--limit 10` to test configuration
2. **Monitor tokens**: Check `evaluation_summary.json` for token usage
3. **Check errors**: Failed LLM calls are logged and tracked
4. **Resume safely**: Checkpoints enable safe resumption
5. **Analyze patterns**: Use CSV export for pandas/excel analysis

## Interpreting Results

### High-Quality Outcomes
- `outcome.status = "true_pass"` + `alignment_score >= 3`
- All quality scores >= 3
- `failure_class = "none"`

### Concerning Patterns
- `accidental_pass_risk = "high"` → Tests pass but semantics wrong
- `outcome = "partial_fix"` → Core intent covered, edges missed
- Many `INT_EDGECASE_MISSED` → Agent needs better edge case reasoning
- Many `EXT_INCOMPLETE_SPEC` → Dataset issue or spec quality

### Resolved vs Unresolved
- Compare quality scores between resolved/unresolved instances
- Identify if failures are more external or internal
- Check if resolved instances have different failure patterns

## Troubleshooting

**LLM rate limits:**
- Increase `--max-retries` and wait times in code
- Reduce parallelism (currently sequential)

**Out of memory:**
- Process smaller batches with `--limit`
- Truncate very long trajectories (see `format_trajectory`)

**Invalid JSON from LLM:**
- Check `validation_errors` in output
- May need to adjust prompt or use more capable model

**Checkpoint corruption:**
- Delete `checkpoint.jsonl` and restart
- Results are appended, so partial progress is saved

## Advanced Usage

### Custom Analysis

```python
from analyze_results import ResultsAnalyzer

analyzer = ResultsAnalyzer('./judge_results/evaluation_results.jsonl')

# Access DataFrame
df = analyzer.df

# Filter high-risk accidental passes
high_risk = df[df['accidental_pass_risk'] == 'high']

# Get instance details
details = analyzer.get_instance_details('django__django-12184')
```

### Batch Processing Multiple Runs

```bash
for file in evaluation/evaluation_outputs/outputs/**/output.with_completions.jsonl.gz; do
  output_dir="judge_results/$(basename $(dirname $file))"
  python judge_evaluator.py \
    --input-files "$file" \
    --output-dir "$output_dir" \
    --model gpt-4o
done
```

## Architecture

```
judge_evaluator.py
├── EvaluationConfig: Configuration dataclass
├── LLMJudgeClient: Azure OpenAI client with retry logic
├── PromptBuilder: Constructs judge prompts from instance data
├── ResultParser: Extracts and validates JSON from LLM responses
└── TrajectoryEvaluator: Main orchestration class

analyze_results.py
└── ResultsAnalyzer: Analysis and reporting utilities
```

## Citation

If you use this evaluation system, please cite the SWE-Bench paper and this evaluation framework.

## License

Follows the same license as OpenHands/SWE-Bench.
