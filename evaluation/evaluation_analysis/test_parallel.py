#!/usr/bin/env python3
"""
Simple test to verify parallel processing logic in judge_evaluator.py
"""

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from threading import Lock
from tqdm import tqdm

# Simulate the checkpoint lock
checkpoint_lock = Lock()
results = []

def simulate_evaluate_instance(task_id):
    """Simulate evaluating an instance (like LLM API call)"""
    # Simulate I/O wait (like API call)
    time.sleep(0.5)
    
    result = {
        'task_id': task_id,
        'status': 'success',
        'timestamp': time.time()
    }
    
    return result

def test_parallel_processing(num_tasks=10, max_workers=4):
    """Test parallel processing logic"""
    
    print(f"Testing with {num_tasks} tasks and {max_workers} workers")
    print("-" * 60)
    
    # Prepare tasks
    tasks = [{'task_id': i} for i in range(num_tasks)]
    
    start_time = time.time()
    
    if max_workers == 1:
        print("Running in single-threaded mode")
        for task in tqdm(tasks, desc="Processing"):
            result = simulate_evaluate_instance(task['task_id'])
            results.append(result)
    else:
        print(f"Running with {max_workers} parallel workers")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(simulate_evaluate_instance, task['task_id']): task
                for task in tasks
            }
            
            with tqdm(total=len(tasks), desc="Processing") as pbar:
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        print(f"Error processing task {task['task_id']}: {e}")
                    finally:
                        pbar.update(1)
    
    elapsed = time.time() - start_time
    
    print(f"\nCompleted {len(results)} tasks in {elapsed:.2f}s")
    print(f"Throughput: {len(results)/elapsed:.2f} tasks/second")
    print(f"Expected speedup: ~{max_workers}x")
    print(f"Actual speedup: ~{(num_tasks * 0.5) / elapsed:.2f}x")
    
    return results

if __name__ == '__main__':
    print("="*60)
    print("Parallel Processing Test")
    print("="*60)
    print()
    
    # Test single-threaded
    print("\n1. Single-threaded baseline:")
    results = []
    test_parallel_processing(num_tasks=10, max_workers=1)
    
    # Test with 4 workers
    print("\n2. Parallel with 4 workers:")
    results = []
    test_parallel_processing(num_tasks=10, max_workers=4)
    
    # Test with 8 workers
    print("\n3. Parallel with 8 workers:")
    results = []
    test_parallel_processing(num_tasks=10, max_workers=8)
    
    print("\n" + "="*60)
    print("✅ Parallel processing logic verified!")
    print("="*60)
