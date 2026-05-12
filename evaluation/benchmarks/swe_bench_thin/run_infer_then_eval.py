"""Per-instance interleaved infer + eval for SWE-bench (ThinDockerRuntime).

This driver runs inference for one instance and then immediately evaluates it
*in the same worker*, reusing the just-pulled Docker image. Compared to the
two-phase pipeline (`run_infer` then `eval_infer`), this:

  - Avoids the eval cold-start pull burst (each image pull happens once and
    is reused right away, dramatically reducing containerd concurrent-pull
    races on shared base layers).
  - Skips re-pulling images: by the time eval runs, the image is local and
    `_ensure_image_available` short-circuits via `images.get`.
  - Produces the SAME two output files as the legacy pipeline:
      <out>/output.jsonl                  # inference rows (one per instance)
      <out>/output.swebench_eval.jsonl    # eval rows  (one per instance)
    so existing downstream tooling keeps working.

This is a NEW entrypoint and intentionally does NOT modify any existing code.
It simply imports `process_instance` from the existing infer + eval modules
and wires them together.

Usage (typically via the matching shell wrapper):

  poetry run python evaluation/benchmarks/swe_bench_thin/run_infer_then_eval.py \\
      --agent-cls CodeActAgent \\
      --llm-config llm.eval_glm5_fp8_t0 \\
      --max-iterations 100 \\
      --eval-num-workers 4 \\
      --eval-note v0.61.0-thin-no-hint \\
      --eval-output-dir evaluation/evaluation_outputs/outputs_thin \\
      --dataset princeton-nlp/SWE-bench_Verified \\
      --split test \\
      --mode swe \\
      [--eval-n-limit 500]
"""

from __future__ import annotations

import copy
import errno
import fcntl
import json
import os
from functools import partial
from typing import Any

import pandas as pd
from datasets import load_dataset

import openhands.agenthub  # noqa: F401  (registers agents)
from evaluation.benchmarks.swe_bench_thin.eval_infer import (
    ConditionalImports,
    process_git_patch,
)
from evaluation.benchmarks.swe_bench_thin.eval_infer import (
    process_instance as eval_process_instance,
)
from evaluation.benchmarks.swe_bench_thin import run_infer as _thin_run_infer
from evaluation.benchmarks.swe_bench_thin.run_infer import (
    filter_dataset,
    process_instance as infer_process_instance,
    set_dataset_type,
)
from evaluation.utils.shared import (
    EvalMetadata,
    EvalOutput,
    check_maximum_retries_exceeded,
    make_metadata,
    prepare_dataset,
    run_evaluation,
)
from openhands.core.config import (
    get_agent_config_arg,
    get_evaluation_parser,
    get_llm_config_arg,
)
from openhands.core.config.condenser_config import NoOpCondenserConfig
from openhands.core.config.utils import get_condenser_config_arg
from openhands.core.logger import openhands_logger as logger


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _build_conditional_imports(dataset_name: str) -> ConditionalImports:
    """Mirror the import-time switch in `eval_infer.py`'s __main__."""
    if 'SWE-Gym' in dataset_name:
        from swegym.harness.grading import get_eval_report
        from swegym.harness.run_evaluation import (
            APPLY_PATCH_FAIL,
            APPLY_PATCH_PASS,
        )
    else:
        from swebench.harness.grading import get_eval_report
        from swebench.harness.run_evaluation import (
            APPLY_PATCH_FAIL,
            APPLY_PATCH_PASS,
        )
    return ConditionalImports(
        get_eval_report=get_eval_report,
        APPLY_PATCH_FAIL=APPLY_PATCH_FAIL,
        APPLY_PATCH_PASS=APPLY_PATCH_PASS,
    )


def _make_test_spec(dataset_name: str, instance_dict: dict[str, Any]):
    """Build the swebench/swegym TestSpec for a single instance dict."""
    if 'SWE-Gym' in dataset_name:
        from swegym.harness.test_spec import make_test_spec
    else:
        from swebench.harness.test_spec.test_spec import make_test_spec
    return make_test_spec(instance_dict)


def _atomic_append_jsonl(path: str, record: dict[str, Any]) -> None:
    """Append a JSON line under an exclusive file lock so multiple workers
    can write concurrently without interleaving."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    line = json.dumps(record, default=str) + '\n'
    # Open in append mode; flock the same fd before writing.
    with open(path, 'a', encoding='utf-8') as f:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            f.write(line)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError as e:
                if e.errno != errno.EINVAL:
                    raise
        finally:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass


def _already_evaluated_ids(eval_output_path: str) -> set[str]:
    if not os.path.exists(eval_output_path):
        return set()
    done: set[str] = set()
    with open(eval_output_path, 'r', encoding='utf-8') as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rec = json.loads(ln)
            except Exception:
                continue
            iid = rec.get('instance_id')
            if iid:
                done.add(iid)
    return done


def _print_eval_report(eval_output_file: str) -> None:
    """Print the same final summary as the legacy eval_infer.py:
    counts of resolved / failed_apply_patch / error_eval / empty_generation
    over all rows in `eval_output_file`."""
    records: list[dict[str, Any]] = []
    with open(eval_output_file, 'r', encoding='utf-8') as f:
        for ln_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(
                    f'Skipping corrupted JSON on line {ln_no} in {eval_output_file}: {e}'
                )
    if not records:
        logger.warning('[InferThenEval] No eval rows to summarize.')
        return

    df = pd.DataFrame(records)
    fields = ['resolved', 'failed_apply_patch', 'error_eval', 'empty_generation']

    def _get(row, field):
        try:
            return row['test_result']['report'][field]
        except Exception:
            return 0

    total = len(df)
    logger.info('==================== Eval Report ====================')
    for field in fields:
        try:
            count = int(df.apply(_get, args=(field,), axis=1).sum())
        except Exception as e:
            logger.warning(f'Could not aggregate field {field}: {e}')
            continue
        pct = (count / total) if total else 0.0
        logger.info(f'# {field}: {count} / {total}. ({pct:.2%})')
    logger.info('=====================================================')


def _already_inferred_patches(infer_output_path: str) -> dict[str, str]:
    """Return {instance_id: model_patch} for every instance that already has
    an inference row in `output.jsonl`. Empty patches are kept (eval will then
    fail gracefully with APPLY_PATCH_FAIL)."""
    if not os.path.exists(infer_output_path):
        return {}
    out: dict[str, str] = {}
    with open(infer_output_path, 'r', encoding='utf-8') as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rec = json.loads(ln)
            except Exception:
                continue
            iid = rec.get('instance_id')
            if not iid:
                continue
            patch = ''
            tr = rec.get('test_result') or {}
            if isinstance(tr, dict):
                patch = tr.get('git_patch', '') or ''
            out[iid] = patch
    return out


def eval_only_for_instance(
    instance_id: str,
    *,
    model_patch: str,
    metadata: EvalMetadata,
    dataset_name: str,
    eval_output_file: str,
    eval_log_dir: str,
    instance_dict: dict[str, Any],
) -> None:
    """Run the eval phase ONLY (no inference) for a single instance and append
    the result to `eval_output_file`. Used by the resume catch-up pass.
    Designed to be picklable for multiprocessing."""
    try:
        test_spec = _make_test_spec(dataset_name, instance_dict)
        conditional_imports = _build_conditional_imports(dataset_name)

        eval_row = pd.Series(
            {
                **instance_dict,
                'instance_id': instance_id,
                'model_patch': process_git_patch(model_patch),
                'test_spec': test_spec,
                'instance': instance_dict,
                'test_result': {},
            }
        )
        eval_metadata = copy.deepcopy(metadata)
        eval_output: EvalOutput = eval_process_instance(
            instance=eval_row,
            metadata=eval_metadata,
            reset_logger=True,
            log_dir=eval_log_dir,
            runtime_failure_count=0,
            conditional_imports=conditional_imports,
        )
        try:
            rec = eval_output.model_dump()
        except AttributeError:
            rec = eval_output.dict()
        _atomic_append_jsonl(eval_output_file, rec)
        logger.info(
            f'[InferThenEval][catch-up] {instance_id}: eval complete -> {eval_output_file}'
        )
    except Exception as e:
        logger.exception(
            f'[InferThenEval][catch-up] {instance_id}: eval FAILED: {e}. '
            'Will be picked up on next resume.'
        )


def _run_eval_catch_up(
    *,
    infer_output_file: str,
    eval_output_file: str,
    eval_log_dir: str,
    metadata: EvalMetadata,
    dataset_name: str,
    instance_id_to_dict: dict[str, dict[str, Any]],
    num_workers: int,
) -> None:
    """For instances where infer is done but eval is missing, run eval-only.

    This makes the resume behavior correct: previously, `prepare_dataset`
    would skip already-inferred instances entirely, leaving any
    infer-done/eval-missing rows orphaned forever.
    """
    inferred = _already_inferred_patches(infer_output_file)
    if not inferred:
        return
    evaluated = _already_evaluated_ids(eval_output_file)
    missing = [iid for iid in inferred if iid not in evaluated]
    # Only consider instances that exist in the current dataset filter.
    missing = [iid for iid in missing if iid in instance_id_to_dict]
    if not missing:
        logger.info(
            '[InferThenEval][catch-up] All inferred instances already have eval rows.'
        )
        return

    logger.info(
        f'[InferThenEval][catch-up] {len(missing)} instance(s) have infer but '
        f'no eval row; running eval-only with {num_workers} workers.'
    )

    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed

    fn = partial(
        eval_only_for_instance,
        metadata=metadata,
        dataset_name=dataset_name,
        eval_output_file=eval_output_file,
        eval_log_dir=eval_log_dir,
    )

    # Use spawn to avoid fork-with-threads warnings/deadlocks.
    ctx = mp.get_context('spawn')
    with ProcessPoolExecutor(
        max_workers=max(1, int(num_workers)), mp_context=ctx
    ) as ex:
        futures = []
        for iid in missing:
            futures.append(
                ex.submit(
                    fn,
                    iid,
                    model_patch=inferred[iid],
                    instance_dict=instance_id_to_dict[iid],
                )
            )
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                logger.exception(f'[InferThenEval][catch-up] worker failed: {e}')


# --------------------------------------------------------------------------- #
# Combined per-instance step
# --------------------------------------------------------------------------- #


def combined_process_instance(
    instance: pd.Series,
    metadata: EvalMetadata,
    reset_logger: bool = True,
    runtime_failure_count: int = 0,
    *,
    dataset_name: str,
    eval_output_file: str,
    eval_log_dir: str,
    instance_id_to_dict: dict[str, dict[str, Any]],
    skip_eval_if_done: bool = True,
) -> EvalOutput:
    """Top-level (picklable) per-instance step: run inference, then immediately
    evaluate the produced patch against the now-cached Docker image.

    Inference output is returned (so the harness writes it to output.jsonl
    exactly as today). Eval output is appended to ``eval_output_file`` under
    an exclusive file lock. Eval failures are caught & logged; they do NOT
    crash inference.
    """
    # Phase 1: inference (image gets pulled & cached here)
    infer_output: EvalOutput = infer_process_instance(
        instance=instance,
        metadata=metadata,
        reset_logger=reset_logger,
        runtime_failure_count=runtime_failure_count,
    )

    instance_id = instance.instance_id

    # Phase 2: evaluation (image is local, no pull needed)
    try:
        if skip_eval_if_done and instance_id in _already_evaluated_ids(
            eval_output_file
        ):
            logger.info(
                f'[InferThenEval] {instance_id}: eval already present; skipping.'
            )
            return infer_output

        git_patch = ''
        try:
            git_patch = (infer_output.test_result or {}).get('git_patch', '') or ''
        except Exception:
            git_patch = ''

        model_patch = process_git_patch(git_patch)

        inst_dict = instance_id_to_dict.get(instance_id)
        if inst_dict is None:
            logger.warning(
                f'[InferThenEval] {instance_id}: not found in dataset map; '
                'skipping inline eval.'
            )
            return infer_output

        test_spec = _make_test_spec(dataset_name, inst_dict)
        conditional_imports = _build_conditional_imports(dataset_name)

        eval_row = pd.Series(
            {
                **inst_dict,
                'instance_id': instance_id,
                'model_patch': model_patch,
                'test_spec': test_spec,
                'instance': inst_dict,
                'test_result': {},
            }
        )

        eval_metadata = copy.deepcopy(metadata)
        eval_output: EvalOutput = eval_process_instance(
            instance=eval_row,
            metadata=eval_metadata,
            reset_logger=reset_logger,
            log_dir=eval_log_dir,
            runtime_failure_count=0,
            conditional_imports=conditional_imports,
        )

        try:
            rec = eval_output.model_dump()  # pydantic v2
        except AttributeError:
            rec = eval_output.dict()  # pydantic v1
        _atomic_append_jsonl(eval_output_file, rec)
        logger.info(
            f'[InferThenEval] {instance_id}: eval complete -> {eval_output_file}'
        )
    except Exception as e:
        logger.exception(
            f'[InferThenEval] {instance_id}: inline eval FAILED: {e}. '
            'Inference result is preserved; you can re-run eval_infer later.'
        )

    return infer_output


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def _main() -> None:
    parser = get_evaluation_parser()
    parser.add_argument(
        '--dataset',
        type=str,
        default='princeton-nlp/SWE-bench_Verified',
    )
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument(
        '--mode',
        type=str,
        default='swe',
        choices=['swe', 'swt', 'swt-ci'],
    )
    args, _ = parser.parse_known_args()

    # ---- Load dataset (mirror swe_bench_thin/run_infer.py) ---- #
    dataset = load_dataset(args.dataset, split=args.split)
    set_dataset_type(args.dataset)
    swe_bench_tests = filter_dataset(dataset.to_pandas(), 'instance_id')
    logger.info(
        f'Loaded dataset {args.dataset} split={args.split}: '
        f'{len(swe_bench_tests)} tasks'
    )

    if _thin_run_infer.DATASET_TYPE == 'SWE-Gym':
        with open(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                '..',
                'swe_bench_optimized',
                'split',
                'swegym_verified_instances.json',
            ),
            'r',
        ) as f:
            swegym_verified_instances = json.load(f)
            swe_bench_tests = swe_bench_tests[
                swe_bench_tests['instance_id'].isin(swegym_verified_instances)
            ]
        logger.info(
            f'{len(swe_bench_tests)} tasks left after SWE-Gym verified filter'
        )

    # ---- LLM / agent / condenser config (same as run_infer.py) ---- #
    llm_config = None
    if args.llm_config:
        llm_config = get_llm_config_arg(args.llm_config, args.config_file)
        llm_config.log_completions = True
        llm_config.modify_params = False
    if llm_config is None:
        raise ValueError(f'Could not find LLM config: --llm_config {args.llm_config}')

    condenser_name = os.environ.get('EVAL_CONDENSER')
    if condenser_name:
        condenser_config = get_condenser_config_arg(condenser_name, args.config_file)
        if condenser_config is None:
            raise ValueError(
                f'Could not find Condenser config: EVAL_CONDENSER={condenser_name}'
            )
    else:
        condenser_config = NoOpCondenserConfig()

    agent_config = None
    if args.agent_config:
        agent_config = get_agent_config_arg(args.agent_config, args.config_file)

    details = {'mode': args.mode}
    dataset_description = (
        args.dataset.replace('/', '__') + '-' + args.split.replace('/', '__')
    )

    metadata = make_metadata(
        llm_config,
        dataset_description,
        args.agent_cls,
        args.max_iterations,
        args.eval_note,
        args.eval_output_dir,
        details=details,
        agent_config=agent_config,
        condenser_config=condenser_config,
    )

    output_file = os.path.join(metadata.eval_output_dir, 'output.jsonl')
    eval_output_file = output_file.replace('.jsonl', '.swebench_eval.jsonl')
    eval_log_dir = output_file.replace('.jsonl', '.swebench_eval.logs')
    os.makedirs(eval_log_dir, exist_ok=True)

    print(f'### OUTPUT FILE: {output_file} ###')
    print(f'### EVAL OUTPUT FILE: {eval_output_file} ###')

    # Build instance_id -> raw dataset dict map (before prepare_dataset filters).
    # Needed both by the inline eval and by the resume catch-up pass.
    full_df = swe_bench_tests
    instance_id_to_dict: dict[str, dict[str, Any]] = {
        row['instance_id']: row.to_dict() for _, row in full_df.iterrows()
    }

    # ---- Resume catch-up: run eval for instances where infer is done but
    # eval is missing. Without this, prepare_dataset would skip those
    # instances entirely on resume (since the infer row already exists),
    # and they would never be evaluated.
    _run_eval_catch_up(
        infer_output_file=output_file,
        eval_output_file=eval_output_file,
        eval_log_dir=eval_log_dir,
        metadata=metadata,
        dataset_name=args.dataset,
        instance_id_to_dict=instance_id_to_dict,
        num_workers=args.eval_num_workers,
    )

    instances = prepare_dataset(swe_bench_tests, output_file, args.eval_n_limit)

    # Stringify list-typed fields just like run_infer.py
    if len(instances) > 0 and not isinstance(
        instances['PASS_TO_PASS'][instances['PASS_TO_PASS'].index[0]], str
    ):
        for col in ['PASS_TO_PASS', 'FAIL_TO_PASS']:
            instances[col] = instances[col].apply(lambda x: str(x))

    combined_fn = partial(
        combined_process_instance,
        dataset_name=args.dataset,
        eval_output_file=eval_output_file,
        eval_log_dir=eval_log_dir,
        instance_id_to_dict=instance_id_to_dict,
    )

    run_evaluation(
        instances,
        metadata,
        output_file,
        args.eval_num_workers,
        combined_fn,
        timeout_seconds=8 * 60 * 60,
        max_retries=5,
    )

    check_maximum_retries_exceeded(metadata.eval_output_dir)

    # Final summary of inline eval results.
    if os.path.exists(eval_output_file):
        n_done = sum(1 for _ in open(eval_output_file, 'r', encoding='utf-8'))
        logger.info(
            f'[InferThenEval] Inline eval rows written: {n_done} -> {eval_output_file}'
        )
        _print_eval_report(eval_output_file)
    else:
        logger.warning(
            f'[InferThenEval] No inline eval rows produced at {eval_output_file}'
        )


if __name__ == '__main__':
    _main()
