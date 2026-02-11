"""
SWE-Bench Trajectory Judge Evaluator

This script evaluates agent trajectories against golden patches using an LLM judge.
It implements a comprehensive failure taxonomy and intent-based correctness evaluation.

Features:
- Parallel processing with configurable worker threads (default: 4)
- Automatic function call message conversion for readable trajectories
- Checkpoint-based resumption for interrupted runs
- Thread-safe result saving and progress tracking
- Optional debug mode to save prompts for inspection
- Comprehensive error handling and retry logic

Performance:
- Use --max-workers to control parallelism (4 workers = ~4x speedup for I/O-bound LLM calls)
- Thread-safe checkpointing ensures no data loss even with parallel execution
- Results are sorted by instance_id in final output for consistency
"""

import gzip
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import pandas as pd
from tqdm import tqdm
from openai import AzureOpenAI
import argparse
import logging
from datetime import datetime
import random
import copy
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Add OpenHands to path to use the function call converter
openhands_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(openhands_path))

from openhands.llm.fn_call_converter import (
    FunctionCallConversionError,
    convert_fncall_messages_to_non_fncall_messages,
    convert_from_multiple_tool_calls_to_single_tool_call_messages,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for the evaluation run"""
    input_files: List[str]
    output_dir: str
    model_name: str = "gpt-4o"
    max_retries: int = 5
    base_wait_time: float = 2.0
    max_wait_time: float = 60.0
    checkpoint_interval: int = 10
    azure_endpoint: str = "http://52.151.57.21:9999"
    api_key: str = "194c2e2102f4aa951440be25c2cc777a"
    api_version: str = "2024-09-01-preview"
    save_prompts: bool = False  # Set to True to save prompts for debugging
    max_workers: int = 4  # Number of parallel workers for evaluation


class LLMJudgeClient:
    """Azure OpenAI client with retry logic and rate limiting"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.client = AzureOpenAI(
            azure_endpoint=config.azure_endpoint,
            api_key=config.api_key,
            api_version=config.api_version
        )
        self.total_calls = 0
        self.total_tokens = 0
        self.failed_calls = 0
    
    def call_with_retry(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.0
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Call the LLM with exponential backoff retry logic.
        
        Returns:
            (response_content, metadata) where metadata includes tokens, time, etc.
        """
        wait_time = self.config.base_wait_time
        
        for attempt in range(self.config.max_retries):
            try:
                start_time = time.perf_counter()
                
                response = self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=messages,
                    temperature=temperature,
                )
                
                elapsed = time.perf_counter() - start_time
                self.total_calls += 1
                
                # Extract response content
                content = response.choices[0].message.content if response.choices else None
                
                # Extract token usage
                tokens = None
                if hasattr(response, 'usage') and response.usage:
                    tokens = response.usage.total_tokens
                    self.total_tokens += tokens
                
                metadata = {
                    'elapsed': elapsed,
                    'tokens': tokens,
                    'attempt': attempt + 1,
                    'model': self.config.model_name
                }
                
                logger.info(f"LLM call successful (attempt {attempt + 1}): {elapsed:.2f}s, {tokens} tokens")
                return content, metadata
                
            except Exception as e:
                self.failed_calls += 1
                logger.warning(f"LLM call failed (attempt {attempt + 1}/{self.config.max_retries}): {e}")
                
                if attempt < self.config.max_retries - 1:
                    # Add jitter to avoid thundering herd
                    jitter = random.uniform(0, wait_time * 0.1)
                    sleep_time = min(wait_time + jitter, self.config.max_wait_time)
                    logger.info(f"Waiting {sleep_time:.2f}s before retry...")
                    time.sleep(sleep_time)
                    wait_time *= 2  # Exponential backoff
                else:
                    logger.error(f"Max retries exceeded for LLM call")
                    return None, {
                        'error': str(e),
                        'attempt': attempt + 1,
                        'model': self.config.model_name
                    }
        
        return None, {'error': 'Max retries exceeded'}


class PromptBuilder:
    """Builds the judge prompt from instance data"""
    
    SYSTEM_PROMPT = """You are an impartial, evidence-driven evaluator for software engineering agent trajectories. You receive: issue text, golden fix (diff), golden tests (diff or test code), agent trajectory (steps + tool logs + reasoning), agent patch (diff), and any test outputs.

Your task is to:

1. infer the **bug** and the **intended behavior** using *all artifacts*, with **golden fix + golden tests as the primary ground truth for intent**;
2. evaluate whether the agent's behavior and patch matches that intent;
3. diagnose failures with a structured taxonomy; and
4. detect subtle cases where test results are misleading: partial fixes, accidental passes, overfitting.

**Hard rules:**

* Use only evidence from provided artifacts. Do not assume repo context beyond what is included.
* Every non-trivial claim must cite evidence by quoting snippets or referencing step numbers/file paths/hunks.
* Prefer intent inferred from golden fix/tests over the agent's self-reported intent.
* If agent passes tests but diverges semantically from golden intent, label it as **accidental or misaligned**, not as success.
* If agent fails tests but captures the core intent, label it as **partial success** with missing edge cases, and identify which aspects are missing and whether they were inferable from golden artifacts or only from issue text.
* If intent cannot be inferred from golden fix/tests, set outcome to `inconclusive` and explain precisely what is missing.
* JSON must be strictly parseable. If you output invalid JSON, rewrite JSON only.

Output must include:

1. A human-readable analysis report
2. A single valid JSON object exactly matching the schema below (no extra keys; no text after JSON)"""
    
    REPORT_STRUCTURE = """
### A) Analysis Report format (must follow exactly)

Use headings in this order:

1. **Outcome Classification**
2. **Inferred Intended Behavior (from Golden Fix/Tests)**
3. **Agent Patch vs Golden Intent**
4. **Test Results Reliability Assessment**
5. **Primary Failure Determination**
6. **Secondary Contributing Factors**
7. **Actionable Recommendations**

### B) JSON Schema (must match exactly; no extra keys)

```json
{
  "trajectory_id": "string_or_null",
  "outcome": {
    "status": "true_pass|partial_fix|accidental_pass|wrong_fix|inconclusive",
    "final_test_state": "all_green|some_fail|not_run|unknown"
  },
  "intent": {
    "requirements": ["string"],
    "edge_cases": ["string"],
    "non_functional_constraints": ["string"]
  },
  "patch_alignment": {
    "alignment_score": 0,
    "missing_requirements": ["string"],
    "extra_behavior_changes": ["string"],
    "accidental_pass_risk": "low|medium|high|unknown",
    "notes": "string"
  },
  "primary_failure": {
    "class": "external|internal|none",
    "inferability": "inferable_from_issue|inferable_from_golden|non_inferable|na",
    "reason_code": "EXT_INCOMPLETE_SPEC|EXT_AMBIGUOUS_INTENT|EXT_NONINFERABLE_CONSTRAINT|EXT_GOLDEN_EDGECASE_SURPRISE|EXT_DATASET_ARTIFACT|INT_PARSE_MISREAD|INT_SEARCH_INSUFFICIENT|INT_TESTS_NOT_INSPECTED|INT_RCA_WRONG|INT_RCA_PARTIAL|INT_FIX_LOCATION_WRONG|INT_FIX_STRATEGY_WRONG_LEVEL|INT_IMPL_LOGIC_BUG|INT_EDGECASE_MISSED|INT_API_MISUSE|INT_PERF_REGRESSION|INT_VALIDATION_WEAK|INT_TEST_OUTPUT_IGNORED|INT_TOOL_MISUSE|INT_LOOPING|INT_THRASHING|INT_PREMATURE_STOP|INT_EVIDENCE_GAP|NONE",
    "stage": "ingestion|repro|exploration|rca|design|implementation|validation|iteration|finalization"
  },
  "secondary_failures": [
    {
      "class": "external|internal",
      "inferability": "inferable_from_issue|inferable_from_golden|non_inferable|na",
      "reason_code": "EXT_INCOMPLETE_SPEC|EXT_AMBIGUOUS_INTENT|EXT_NONINFERABLE_CONSTRAINT|EXT_GOLDEN_EDGECASE_SURPRISE|EXT_DATASET_ARTIFACT|INT_PARSE_MISREAD|INT_SEARCH_INSUFFICIENT|INT_TESTS_NOT_INSPECTED|INT_RCA_WRONG|INT_RCA_PARTIAL|INT_FIX_LOCATION_WRONG|INT_FIX_STRATEGY_WRONG_LEVEL|INT_IMPL_LOGIC_BUG|INT_EDGECASE_MISSED|INT_API_MISUSE|INT_PERF_REGRESSION|INT_VALIDATION_WEAK|INT_TEST_OUTPUT_IGNORED|INT_TOOL_MISUSE|INT_LOOPING|INT_THRASHING|INT_PREMATURE_STOP|INT_EVIDENCE_GAP"
    }
  ],
  "quality_scores": {
    "spec_alignment": 0,
    "repo_exploration": 0,
    "root_cause_quality": 0,
    "patch_correctness": 0,
    "validation_rigor": 0,
    "iteration_efficiency": 0
  },
  "evidence": {
    "golden_quotes": ["string"],
    "agent_quotes": ["string"],
    "commands_run": ["string"],
    "diff_comparison_notes": ["string"]
  },
  "narrative": {
    "one_paragraph_diagnosis": "string",
    "counterfactual_fix": "string"
  }
}
```

**Alignment score guidance (0–4):**

* 0: unrelated to intent
* 1: touches symptom only
* 2: addresses core but misses major requirements
* 3: matches intent, minor gaps
* 4: matches intent fully
"""
    
    @staticmethod
    def convert_fncall_messages(messages: List[Dict], tools: List[Dict]) -> Optional[List[Dict]]:
        """
        Convert function-calling messages to plain text format.
        
        This transforms messages with tool_calls into human-readable text that shows
        what the assistant is doing, solving the "empty message" problem.
        
        Args:
            messages: List of message dictionaries
            tools: List of tool definitions
            
        Returns:
            Converted messages, or None if conversion fails
        """
        # Make a deep copy to avoid modifying original
        message_copy = copy.deepcopy(messages)
        
        # Handle None content (required by converter)
        for message in message_copy:
            if message.get('content') is None:
                message['content'] = ''
        
        try:
            # First, handle multiple tool calls in one message
            single_call_messages = convert_from_multiple_tool_calls_to_single_tool_call_messages(
                message_copy, ignore_final_tool_result=True
            )
            
            # Then convert function calls to plain text
            converted = convert_fncall_messages_to_non_fncall_messages(
                single_call_messages, tools, add_in_context_learning_example=False
            )
            
            return converted
        except FunctionCallConversionError as e:
            logger.warning(f"Failed to convert function calling messages: {e}")
            return None
    
    @staticmethod
    def normalize_message_content(messages: List[Dict]) -> List[Dict]:
        """Convert content from list format to string format for tokenizer compatibility."""
        normalized = []
        for msg in messages:
            msg_copy = msg.copy()
            content = msg_copy.get('content', '')
            
            # If content is a list (like [{'type': 'text', 'text': '...'}])
            if isinstance(content, list):
                # Extract text from all items and concatenate
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        text_parts.append(item.get('text', ''))
                msg_copy['content'] = '\n'.join(text_parts)
            elif content is None:
                msg_copy['content'] = ''
            
            normalized.append(msg_copy)
        return normalized
    
    @staticmethod
    def format_trajectory(messages: List[Dict]) -> str:
        """Format agent trajectory for the prompt"""
        trajectory_lines = []
        for i, msg in enumerate(messages):
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            
            trajectory_lines.append(f"\n### Step {i} [{role.upper()}]")
            trajectory_lines.append("-" * 80)
            
            if content:
                # Handle list content (defensive coding - should already be normalized)
                if isinstance(content, list):
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict) and item.get('type') == 'text':
                            text_parts.append(item.get('text', ''))
                    content = '\n'.join(text_parts) if text_parts else str(content)
                
                # Truncate very long content
                if len(content) > 10000 and role != "assistant":
                    content = content[:10000] + "\n... [truncated] ..."
                trajectory_lines.append(content)
            else:
                trajectory_lines.append("[Empty message]")
        
        return "\n".join(trajectory_lines)
    
    @classmethod
    def build_prompt(
        cls,
        instance_id: str,
        instance: Dict[str, Any],
        messages: List[Dict],
        git_patch: str,
        resolved: bool,
        tools: Optional[List[Dict]] = None
    ) -> str:
        """Build the complete judge prompt"""
        
        # Try to convert function calling messages to text format first
        # This solves the "empty message" problem
        if tools:
            converted_messages = cls.convert_fncall_messages(messages, tools)
            if converted_messages:
                logger.info(f"Converted {len(messages)} messages with function calls to text format")
                messages = converted_messages
        
        # Normalize messages
        normalized_messages = cls.normalize_message_content(messages)
        trajectory = cls.format_trajectory(normalized_messages)
        
        # Extract instance fields
        problem_statement = instance.get('problem_statement', 'N/A')
        golden_patch = instance.get('patch', 'N/A')
        golden_test_patch = instance.get('test_patch', 'N/A')
        repo = instance.get('repo', 'N/A')
        base_commit = instance.get('base_commit', 'N/A')
        
        # Build the user prompt
        user_prompt = f"""
## Inputs Begin

### [INSTANCE_METADATA]

- Instance ID: {instance_id}
- Repository: {repo}
- Base Commit: {base_commit}
- Resolved (by evaluator): {resolved}

### [ISSUE]

{problem_statement}

### [GOLDEN_FIX_DIFF]

{golden_patch}

### [GOLDEN_TESTS_DIFF_OR_CODE]

{golden_test_patch}

### [AGENT_TRAJECTORY]

{trajectory}

### [AGENT_PATCH_DIFF]

{git_patch if git_patch else "No patch generated"}

### [AGENT_TEST_OUTPUT]

(Test outputs are embedded in the trajectory above)

## Inputs End

{cls.REPORT_STRUCTURE}

Please provide your analysis following the required format.
"""
        
        return user_prompt


class ResultParser:
    """Parses and validates LLM judge outputs"""
    
    @staticmethod
    def extract_json_from_response(response: str) -> Optional[Dict]:
        """Extract JSON from response that may contain markdown or other text"""
        # Try to find JSON in code blocks first
        import re
        
        # Look for ```json ... ``` blocks
        json_block_pattern = r'```json\s*(.*?)\s*```'
        matches = re.findall(json_block_pattern, response, re.DOTALL)
        
        if matches:
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
        
        # Try to find raw JSON object
        # Look for outermost { ... }
        stack = []
        start_idx = None
        
        for i, char in enumerate(response):
            if char == '{':
                if not stack:
                    start_idx = i
                stack.append(char)
            elif char == '}':
                if stack:
                    stack.pop()
                    if not stack and start_idx is not None:
                        # Found complete JSON object
                        json_str = response[start_idx:i+1]
                        try:
                            return json.loads(json_str)
                        except json.JSONDecodeError:
                            # Continue searching
                            start_idx = None
        
        return None
    
    @staticmethod
    def validate_json_schema(data: Dict) -> Tuple[bool, List[str]]:
        """Validate the JSON output against expected schema"""
        errors = []
        
        # Required top-level keys
        required_keys = [
            'trajectory_id', 'outcome', 'intent', 'patch_alignment',
            'primary_failure', 'secondary_failures', 'quality_scores',
            'evidence', 'narrative'
        ]
        
        for key in required_keys:
            if key not in data:
                errors.append(f"Missing required key: {key}")
        
        # Validate outcome
        if 'outcome' in data:
            if 'status' not in data['outcome']:
                errors.append("Missing outcome.status")
            elif data['outcome']['status'] not in [
                'true_pass', 'partial_fix', 'accidental_pass', 'wrong_fix', 'inconclusive'
            ]:
                errors.append(f"Invalid outcome.status: {data['outcome']['status']}")
        
        # Validate quality scores are 0-4
        if 'quality_scores' in data:
            for key, value in data['quality_scores'].items():
                if not isinstance(value, (int, float)) or value < 0 or value > 4:
                    errors.append(f"quality_scores.{key} must be 0-4, got {value}")
        
        return len(errors) == 0, errors


class TrajectoryEvaluator:
    """Main evaluator that orchestrates the evaluation process"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.llm_client = LLMJudgeClient(config)
        self.results = []
        self.checkpoint_path = Path(config.output_dir) / "checkpoint.jsonl"
        self.output_path = Path(config.output_dir) / "evaluation_results.jsonl"
        self.summary_path = Path(config.output_dir) / "evaluation_summary.json"
        
        # Thread-safe lock for writing checkpoints
        self.checkpoint_lock = threading.Lock()
        
        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Create prompts directory if debug mode is enabled
        if config.save_prompts:
            self.prompts_dir = Path(config.output_dir) / "prompts"
            self.prompts_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Debug mode: Prompts will be saved to {self.prompts_dir}")
        else:
            self.prompts_dir = None
        
        # Load checkpoint if exists
        self.processed_ids = self._load_checkpoint()
    
    def _load_checkpoint(self) -> set:
        """Load already processed instance IDs from checkpoint"""
        if self.checkpoint_path.exists():
            processed = set()
            with open(self.checkpoint_path, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        processed.add(data.get('instance_id'))
                    except json.JSONDecodeError:
                        continue
            logger.info(f"Loaded checkpoint: {len(processed)} instances already processed")
            return processed
        return set()
    
    def _save_checkpoint(self, result: Dict):
        """Append result to checkpoint file (thread-safe)"""
        with self.checkpoint_lock:
            with open(self.checkpoint_path, 'a') as f:
                f.write(json.dumps(result) + '\n')
    
    def _save_prompt(self, instance_id: str, system_prompt: str, user_prompt: str):
        """Save prompt to file for debugging"""
        if not self.prompts_dir:
            return
        
        # Create safe filename from instance_id
        safe_filename = instance_id.replace('/', '_').replace('\\', '_')
        prompt_file = self.prompts_dir / f"{safe_filename}.txt"
        
        try:
            with open(prompt_file, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("SYSTEM PROMPT\n")
                f.write("=" * 80 + "\n")
                f.write(system_prompt)
                f.write("\n\n")
                f.write("=" * 80 + "\n")
                f.write("USER PROMPT\n")
                f.write("=" * 80 + "\n")
                f.write(user_prompt)
            logger.debug(f"Saved prompt for {instance_id} to {prompt_file}")
        except Exception as e:
            logger.warning(f"Failed to save prompt for {instance_id}: {e}")
    
    def load_data(self) -> pd.DataFrame:
        """Load data from gzipped JSONL files"""
        data = []
        
        for file_path in self.config.input_files:
            logger.info(f"Loading {file_path}")
            
            with gzip.open(file_path, 'rb') as f:
                for line in tqdm(f, desc=f'Processing {Path(file_path).name}'):
                    raw_data = json.loads(line)
                    data.append({
                        'instance_id': raw_data['instance_id'],
                        'instance': raw_data['instance'],
                        'resolved': raw_data['report']['resolved'],
                        'messages': raw_data['raw_completions']['messages']
                        if raw_data['raw_completions'] is not None
                        else None,
                        'git_patch': raw_data['test_result'].get('git_patch', ''),
                        'tools': raw_data['raw_completions']['tools']
                        if raw_data['raw_completions'] is not None
                        and 'tools' in raw_data['raw_completions']
                        else None,
                    })
        
        df = pd.DataFrame(data)
        logger.info(f"Loaded {len(df)} total instances")
        
        # Filter out instances with no messages
        df = df[~df['messages'].isna()]
        logger.info(f"{len(df)} instances have messages")
        
        return df
    
    def evaluate_instance(
        self,
        instance_id: str,
        instance: Dict,
        messages: List[Dict],
        git_patch: str,
        resolved: bool,
        tools: Optional[List[Dict]] = None
    ) -> Dict:
        """Evaluate a single instance"""
        
        # Build prompt (with tools for function call conversion)
        user_prompt = PromptBuilder.build_prompt(
            instance_id, instance, messages, git_patch, resolved, tools=tools
        )
        
        # Save prompt for debugging if enabled
        if self.config.save_prompts:
            self._save_prompt(instance_id, PromptBuilder.SYSTEM_PROMPT, user_prompt)
        
        messages_for_llm = [
            {"role": "system", "content": PromptBuilder.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        
        # Call LLM
        response_content, metadata = self.llm_client.call_with_retry(messages_for_llm)
        
        if response_content is None:
            logger.error(f"Failed to get response for {instance_id}")
            return {
                'instance_id': instance_id,
                'error': 'LLM call failed',
                'metadata': metadata
            }
        
        # Parse response
        judge_json = ResultParser.extract_json_from_response(response_content)
        
        if judge_json is None:
            logger.error(f"Failed to extract JSON from response for {instance_id}")
            return {
                'instance_id': instance_id,
                'error': 'JSON extraction failed',
                'raw_response': response_content[:1000],
                'metadata': metadata
            }
        
        # Validate schema
        is_valid, errors = ResultParser.validate_json_schema(judge_json)
        
        if not is_valid:
            logger.warning(f"Schema validation failed for {instance_id}: {errors}")
        
        # Build result
        result = {
            'instance_id': instance_id,
            'resolved': resolved,
            'judge_evaluation': judge_json,
            'raw_response': response_content,
            'metadata': metadata,
            'validation_errors': errors if not is_valid else [],
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def run_evaluation(self, limit: Optional[int] = None):
        """Run evaluation on all instances with parallel processing"""
        
        # Load data
        df = self.load_data()
        
        if limit:
            df = df.head(limit)
            logger.info(f"Limited to {limit} instances for evaluation")
        
        # Filter out already processed
        df = df[~df['instance_id'].isin(self.processed_ids)]
        logger.info(f"{len(df)} instances remaining to process")
        
        if len(df) == 0:
            logger.info("No instances to process!")
            return
        
        # Prepare tasks
        tasks = []
        for idx, row in df.iterrows():
            tasks.append({
                'instance_id': row['instance_id'],
                'instance': row['instance'],
                'messages': row['messages'],
                'git_patch': row['git_patch'],
                'resolved': row['resolved'],
                'tools': row.get('tools', None)
            })
        
        # Use single-threaded mode if max_workers is 1, else use parallel
        if self.config.max_workers == 1:
            logger.info("Running in single-threaded mode")
            for task in tqdm(tasks, desc="Evaluating instances"):
                result = self.evaluate_instance(**task)
                self.results.append(result)
                self._save_checkpoint(result)
        else:
            logger.info(f"Running with {self.config.max_workers} parallel workers")
            
            # Use ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                # Submit all tasks
                future_to_task = {
                    executor.submit(self.evaluate_instance, **task): task
                    for task in tasks
                }
                
                # Process completed futures with progress bar
                with tqdm(total=len(tasks), desc="Evaluating instances") as pbar:
                    for future in as_completed(future_to_task):
                        task = future_to_task[future]
                        try:
                            result = future.result()
                            self.results.append(result)
                            self._save_checkpoint(result)
                            
                            # Log progress at checkpoints
                            if len(self.results) % self.config.checkpoint_interval == 0:
                                logger.info(f"Checkpoint: {len(self.results)} instances evaluated")
                            
                        except Exception as e:
                            logger.error(f"Error evaluating {task['instance_id']}: {e}")
                            # Save error result
                            error_result = {
                                'instance_id': task['instance_id'],
                                'error': str(e),
                                'timestamp': datetime.now().isoformat()
                            }
                            self.results.append(error_result)
                            self._save_checkpoint(error_result)
                        finally:
                            pbar.update(1)
        
        logger.info(f"\nEvaluation complete: {len(self.results)} instances")
        
        # Save final results
        self._save_final_results()
        self._generate_summary()
    
    def _save_final_results(self):
        """Save all results to final output file (sorted by instance_id)"""
        # Sort results by instance_id for consistency
        sorted_results = sorted(self.results, key=lambda x: x.get('instance_id', ''))
        
        with open(self.output_path, 'w') as f:
            for result in sorted_results:
                f.write(json.dumps(result) + '\n')
        logger.info(f"Saved {len(sorted_results)} results to {self.output_path}")
    
    def _generate_summary(self):
        """Generate summary statistics"""
        summary = {
            'total_evaluated': len(self.results),
            'llm_stats': {
                'total_calls': self.llm_client.total_calls,
                'failed_calls': self.llm_client.failed_calls,
                'total_tokens': self.llm_client.total_tokens,
            },
            'outcome_distribution': {},
            'failure_class_distribution': {},
            'avg_alignment_score': 0,
            'errors': []
        }
        
        # Compute distributions
        outcomes = []
        alignment_scores = []
        failure_classes = []
        
        for result in self.results:
            if 'error' in result:
                summary['errors'].append({
                    'instance_id': result['instance_id'],
                    'error': result['error']
                })
                continue
            
            judge_eval = result.get('judge_evaluation', {})
            
            outcome = judge_eval.get('outcome', {}).get('status')
            if outcome:
                outcomes.append(outcome)
            
            alignment = judge_eval.get('patch_alignment', {}).get('alignment_score')
            if alignment is not None:
                alignment_scores.append(alignment)
            
            primary_failure = judge_eval.get('primary_failure', {}).get('class')
            if primary_failure:
                failure_classes.append(primary_failure)
        
        # Distributions
        from collections import Counter
        summary['outcome_distribution'] = dict(Counter(outcomes))
        summary['failure_class_distribution'] = dict(Counter(failure_classes))
        summary['avg_alignment_score'] = sum(alignment_scores) / len(alignment_scores) if alignment_scores else 0
        
        # Save summary
        with open(self.summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved summary to {self.summary_path}")
        logger.info(f"\nSummary:")
        logger.info(f"  Total evaluated: {summary['total_evaluated']}")
        logger.info(f"  Outcome distribution: {summary['outcome_distribution']}")
        logger.info(f"  Avg alignment score: {summary['avg_alignment_score']:.2f}")
        logger.info(f"  LLM calls: {summary['llm_stats']['total_calls']}")
        logger.info(f"  LLM tokens: {summary['llm_stats']['total_tokens']}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate SWE-Bench trajectories with LLM judge')
    parser.add_argument(
        '--input-files',
        nargs='+',
        required=True,
        help='Input JSONL.gz files to evaluate'
    )
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Directory to save evaluation results'
    )
    parser.add_argument(
        '--model',
        default='gpt-4.1',
        help='Model to use for judging (default: gpt-4.1)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of instances to evaluate (for testing)'
    )
    parser.add_argument(
        '--max-retries',
        type=int,
        default=5,
        help='Maximum retries for LLM calls'
    )
    parser.add_argument(
        '--checkpoint-interval',
        type=int,
        default=10,
        help='Save checkpoint every N instances'
    )
    parser.add_argument(
        '--save-prompts',
        action='store_true',
        help='Save prompts to files for debugging (saved in output_dir/prompts/)'
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=4,
        help='Number of parallel workers (default: 4, use 1 for single-threaded)'
    )
    
    args = parser.parse_args()
    
    # Create config
    config = EvaluationConfig(
        input_files=args.input_files,
        output_dir=args.output_dir,
        model_name=args.model,
        max_retries=args.max_retries,
        checkpoint_interval=args.checkpoint_interval,
        save_prompts=args.save_prompts,
        max_workers=args.max_workers
    )
    
    # Run evaluation
    evaluator = TrajectoryEvaluator(config)
    evaluator.run_evaluation(limit=args.limit)


if __name__ == '__main__':
    main()
