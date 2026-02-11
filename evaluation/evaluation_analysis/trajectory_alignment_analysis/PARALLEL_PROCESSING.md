# Parallel Processing in Judge Evaluator

## Overview

The judge evaluator now supports **parallel processing** to significantly speed up evaluation of large datasets. This feature uses Python's `ThreadPoolExecutor` to evaluate multiple instances concurrently.

## Key Features

### 1. Configurable Parallelism
- Control the number of parallel workers with `--max-workers` flag
- Default: 4 workers (providing ~4x speedup for I/O-bound LLM API calls)
- Use `--max-workers 1` for single-threaded mode (useful for debugging)

### 2. Thread-Safe Operations
- **Checkpoint saving**: Uses thread locks to prevent race conditions
- **Progress tracking**: Atomic updates to results list
- **Error handling**: Each worker handles errors independently

### 3. Maintained Features
- **Checkpointing**: Still works correctly - checkpoints are written atomically
- **Resume capability**: Can still resume from checkpoint if interrupted
- **Result ordering**: Final results are sorted by instance_id for consistency
- **Debug mode**: Prompt saving works correctly in parallel mode

## Usage

### Basic Parallel Execution (4 workers)
```bash
python judge_evaluator.py \
  --input-files data.jsonl.gz \
  --output-dir results/ \
  --max-workers 4
```

### High Parallelism (8 workers)
```bash
python judge_evaluator.py \
  --input-files data.jsonl.gz \
  --output-dir results/ \
  --max-workers 8
```

### Single-Threaded (for debugging)
```bash
python judge_evaluator.py \
  --input-files data.jsonl.gz \
  --output-dir results/ \
  --max-workers 1
```

## Performance Considerations

### Optimal Worker Count

The optimal number of workers depends on several factors:

1. **API Rate Limits**: If your API has rate limits, too many workers may hit the limit
2. **Network Latency**: More workers help when network latency is high
3. **CPU/Memory**: Each worker consumes some resources
4. **LLM Response Time**: Longer response times benefit more from parallelism

**Recommendation**: Start with 4 workers and adjust based on:
- Monitor API rate limit errors
- Check CPU/memory usage
- Measure actual speedup achieved

### Expected Speedup

For I/O-bound LLM API calls:
- **1 worker**: Baseline (sequential processing)
- **4 workers**: ~3-4x speedup
- **8 workers**: ~6-8x speedup
- **16 workers**: Diminishing returns, may hit rate limits

### When to Use Single-Threaded Mode

Use `--max-workers 1` when:
- Debugging specific instances
- API has strict rate limits
- You need deterministic logging output
- Memory is constrained

## Implementation Details

### Thread Safety

The evaluator uses:
- `threading.Lock()` for checkpoint file writes
- Thread-local LLM clients (each thread has its own OpenAI client)
- Atomic list operations for result collection

### Error Handling

Each worker handles errors independently:
- Failed instances are logged with error details
- Other workers continue processing
- All results (successful and failed) are saved to checkpoint

### Progress Tracking

- Uses `tqdm` progress bar that updates from all workers
- Checkpoint logs appear at regular intervals
- Final results show total instances processed

### Result Consistency

- Results are collected in completion order (not submission order)
- Final output file is sorted by `instance_id` for consistency
- Checkpoint file maintains insertion order

## Example Run

```bash
#!/bin/bash

# Run evaluation with 4 parallel workers
python judge_evaluator.py \
  --input-files output.with_completions.jsonl.gz \
  --output-dir judge_results_$(date +%Y%m%d_%H%M%S) \
  --model gpt-4.1 \
  --max-workers 4 \
  --checkpoint-interval 10 \
  --save-prompts

# Monitor progress
tail -f judge_results_*/evaluation_results.jsonl | wc -l
```

## Troubleshooting

### Issue: API Rate Limit Errors

**Solution**: Reduce `--max-workers` to 2 or 1

### Issue: Memory Usage Too High

**Solution**: 
- Reduce `--max-workers`
- Process data in batches using `--limit`

### Issue: Inconsistent Results

**Solution**: This shouldn't happen, but if you see issues:
- Check checkpoint file for corruption
- Verify thread-safe operations are working
- Use `--max-workers 1` to verify results

### Issue: Deadlock or Hanging

**Solution**:
- Check for network issues
- Verify API endpoint is responding
- Try single-threaded mode to isolate the problem

## Performance Metrics

Track these metrics to optimize:

```python
# From evaluation_summary.json
{
  "total_evaluated": 100,
  "llm_stats": {
    "total_calls": 100,
    "failed_calls": 2,
    "total_tokens": 500000
  }
}
```

Calculate:
- **Throughput**: instances/minute
- **Success rate**: (total - failed) / total
- **Cost**: total_tokens × token_price

## Future Enhancements

Potential improvements:
- [ ] Async/await support for even better concurrency
- [ ] Dynamic worker pool sizing based on API latency
- [ ] Built-in rate limiting to prevent API errors
- [ ] Distributed processing across multiple machines
- [ ] Real-time metrics dashboard

## Summary

✅ **Parallel processing is enabled by default (4 workers)**  
✅ **Thread-safe checkpointing ensures reliability**  
✅ **Typical speedup: 3-4x for I/O-bound LLM calls**  
✅ **Fully compatible with existing features (checkpointing, debug mode, etc.)**  

Use `--max-workers` to tune performance for your specific use case!
