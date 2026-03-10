#!/usr/bin/env python3
"""Optimized multi-run evaluation script for SWE-bench / SWE-Gym.

Key optimization: For each instance, ALL N runs are evaluated inside a SINGLE Docker
container, eliminating repeated image pulls/loads between runs. After all runs for an
instance finish the container is stopped, the runtime-created container is removed by
runtime.close(), and the instance-specific Docker image is deleted to reclaim disk space.

Ordering:
    Old (eval_multirun.sh):  run_1 ∀ instances → run_2 ∀ instances → …
    New (this script):       instance_1 × run_1…run_N → instance_2 × run_1…run_N → …

Parallelism:
    --eval-num-workers N  →  N instances processed concurrently, each in its own thread.

Output:
    Each run keeps its existing output directory. Evaluated results are written to
    <run_dir>/output.swebench_eval.jsonl  (appended as soon as a result is ready).
    A status file  <base_dir>/multirun_eval_status.json  records which instances have
    been fully evaluated; re-running the script safely skips already-done instances.
    Per-run log files are written to <base_dir>/multirun_eval_logs/:
        instance_<id>_run_1.log, instance_<id>_run_2.log, …
    Container startup logs appear in run_1's file; teardown logs in the last run's file.

Usage:
    poetry run python evaluation/benchmarks/swe_bench_optimized/eval_infer_multi_run.py \\
        --base-dir  <path_that_contains_run_1_run_2_..._dirs> \\
        --dataset   "SWE-Gym/SWE-Gym" \\
        --split     "train" \\
        --eval-num-workers 4
"""

import argparse
import copy
import json
import logging
import os
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable

import pandas as pd
from tqdm import tqdm

from evaluation.benchmarks.swe_bench.resource.mapping import (
    get_instance_resource_factor,
)
from evaluation.utils.shared import (
    EvalMetadata,
    EvalOutput,
    get_default_sandbox_config_for_eval,
    get_openhands_config_for_eval,
)
from openhands.core.config import LLMConfig, OpenHandsConfig
from openhands.core.logger import openhands_logger as logger
from openhands.core.main import create_runtime
from openhands.events.action import CmdRunAction
from openhands.events.observation import CmdOutputObservation
from openhands.utils.async_utils import call_async_from_sync

# ──────────────────────────────────────────────────────────────────────────────
# Docker image helpers  (mirrors eval_infer.py / run_infer.py convention)
# ──────────────────────────────────────────────────────────────────────────────

DOCKER_IMAGE_PREFIX = os.environ.get('EVAL_DOCKER_IMAGE_PREFIX', 'docker.io/xingyaoww/')
logger.info(f'Using docker image prefix (SWE-Gym): {DOCKER_IMAGE_PREFIX}')

# Official SWE-bench base images live under docker.io/swebench/
SWEBENCH_DOCKER_IMAGE_PREFIX = os.environ.get(
    'SWEBENCH_DOCKER_IMAGE_PREFIX', 'docker.io/swebench/'
)


def get_instance_docker_image(instance_id: str, is_swegym: bool = True) -> str:
    """Return the docker image tag for a given instance_id.

    SWE-Gym naming  (is_swegym=True):
        docker.io/xingyaoww/sweb.eval.x86_64.astropy_s_astropy-12907

    SWE-bench naming (is_swegym=False):
        docker.io/swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest

    Mirrors the logic in evaluation/benchmarks/swe_bench_optimized/run_infer.py.
    """
    if is_swegym:
        image_name = 'sweb.eval.x86_64.' + instance_id
        image_name = image_name.replace('__', '_s_')  # github slash
        image_name = image_name.replace('/', '_s_')
        image_name = image_name.lower()
        return (DOCKER_IMAGE_PREFIX.rstrip('/') + '/' + image_name).lower()
    else:
        # Official SWE-bench image: sweb.eval.x86_64.<repo>_1776_<name>:latest
        # instance_id format: "astropy__astropy-12907"
        repo, name = instance_id.split('__', 1)
        image_name = f'sweb.eval.x86_64.{repo}_1776_{name}:latest'.lower()
        return (SWEBENCH_DOCKER_IMAGE_PREFIX.rstrip('/') + '/' + image_name).lower()


# ──────────────────────────────────────────────────────────────────────────────
# Patch utilities  (copied verbatim from eval_infer.py)
# ──────────────────────────────────────────────────────────────────────────────


def process_git_patch(patch: str) -> str:
    if not isinstance(patch, str):
        return ''
    if not patch.strip():
        return ''
    patch = patch.replace('\r\n', '\n')
    lines = patch.split('\n')
    for i, line in enumerate(lines):
        if line.startswith('diff --git'):
            patch = '\n'.join(lines[i:])
            break
    patch = patch.rstrip() + '\n'
    return patch


# ──────────────────────────────────────────────────────────────────────────────
# Conditional-import container (SWE-bench vs SWE-Gym)
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class ConditionalImports:
    get_eval_report: Callable
    APPLY_PATCH_FAIL: str
    APPLY_PATCH_PASS: str


# ──────────────────────────────────────────────────────────────────────────────
# Status tracking  (thread-safe)
# ──────────────────────────────────────────────────────────────────────────────


class EvalStatus:
    """Persistent, thread-safe record of per-instance evaluation progress."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._lock = threading.Lock()
        if os.path.exists(path):
            with open(path) as f:
                self._data: dict = json.load(f)
        else:
            self._data = {}

    # ── public helpers ────────────────────────────────────────────────────────

    def is_all_runs_done(self, instance_id: str) -> bool:
        with self._lock:
            return self._data.get(instance_id, {}).get('all_runs_done', False)

    def mark_run_done(
        self,
        instance_id: str,
        run_num: int,
        total_runs: int,
        image_name: str,
    ) -> bool:
        """Mark one run as complete. Returns True iff ALL runs are now done."""
        with self._lock:
            entry = self._data.setdefault(
                instance_id,
                {
                    'completed_runs': [],
                    'all_runs_done': False,
                    'image_name': image_name,
                    'docker_cleaned': False,
                },
            )
            if run_num not in entry['completed_runs']:
                entry['completed_runs'].append(run_num)
            all_done = len(entry['completed_runs']) >= total_runs
            if all_done:
                entry['all_runs_done'] = True
            self._flush()
            return all_done

    def mark_docker_cleaned(self, instance_id: str) -> None:
        with self._lock:
            if instance_id in self._data:
                self._data[instance_id]['docker_cleaned'] = True
                self._flush()

    def get_completed_runs(self, instance_id: str) -> list[int]:
        with self._lock:
            return list(self._data.get(instance_id, {}).get('completed_runs', []))

    # ── internal ─────────────────────────────────────────────────────────────

    def _flush(self) -> None:
        tmp = self._path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(self._data, f, indent=2)
        os.replace(tmp, self._path)


# ──────────────────────────────────────────────────────────────────────────────
# Docker cleanup
# ──────────────────────────────────────────────────────────────────────────────


def cleanup_instance_docker(
    instance_id: str,
    image_name: str,
    ilogger: logging.Logger | None = None,
) -> None:
    """Remove the Docker image (and any leftover containers) for the instance.

    runtime.close() already removes the container it spawned; we only need to
    remove the image so disk space is reclaimed.
    """
    _log = ilogger or logger
    try:
        # Safety: remove any containers still using the image (shouldn't normally exist)
        result = subprocess.run(
            ['docker', 'ps', '-aq', '--filter', f'ancestor={image_name}'],
            capture_output=True,
            text=True,
            timeout=30,
        )
        for cid in result.stdout.strip().split('\n'):
            cid = cid.strip()
            if cid:
                subprocess.run(
                    ['docker', 'rm', '-f', cid],
                    capture_output=True,
                    timeout=30,
                )
                _log.info(f'Removed leftover container {cid}')

        # Remove the image
        rm = subprocess.run(
            ['docker', 'rmi', '-f', image_name],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if rm.returncode == 0:
            _log.info(f'Removed Docker image: {image_name}')
        else:
            _log.warning(
                f'docker rmi returned {rm.returncode}: {rm.stderr.strip()}'
            )
    except Exception as exc:
        _log.warning(f'Docker cleanup failed: {exc}')


# ──────────────────────────────────────────────────────────────────────────────
# Per-run evaluation logic (runs inside an already-started runtime)
# ──────────────────────────────────────────────────────────────────────────────


class _ThreadRoutingHandler(logging.Handler):
    """A single handler attached to openhands_logger that routes each log record
    to the file handler registered for the emitting thread.

    This solves the parallel-worker logging problem cleanly:
    - Added ONCE to the global logger at startup (no repeated add/remove)
    - Each worker thread registers its per-run FileHandler before doing work
      and unregisters it afterwards
    - Records from unregistered threads (e.g. the main thread) are dropped
      here because they already go to openhands_global.log via the plain
      FileHandler that is also attached to the global logger
    - Thread safety: the routing dict is protected by a lock; emit() only
      holds the lock for the dict lookup, not during actual file I/O

    Key property: record.thread is set by the logging framework to the integer
    thread ID of the thread that called the logger, so it is reliable even when
    call_async_from_sync runs a coroutine synchronously on the calling thread.
    """

    def __init__(self) -> None:
        super().__init__()
        self._map_lock = threading.Lock()
        self._handlers: dict[int, logging.Handler] = {}  # thread_id -> FileHandler

    def register(self, handler: logging.Handler) -> None:
        """Register *handler* for the calling thread."""
        with self._map_lock:
            self._handlers[threading.get_ident()] = handler

    def unregister(self) -> None:
        """Unregister the handler for the calling thread."""
        with self._map_lock:
            self._handlers.pop(threading.get_ident(), None)

    def emit(self, record: logging.LogRecord) -> None:
        with self._map_lock:
            handler = self._handlers.get(int(record.thread))
        if handler is not None:
            handler.handle(record)


# Singleton router – installed on the global logger once in main().
_THREAD_ROUTER: _ThreadRoutingHandler | None = None


class _RunLogContext:
    """Context manager that registers a fresh FileHandler for this log_file with
    _THREAD_ROUTER so that all openhands_logger records emitted from this thread
    (including runtime_build.py, docker_runtime.py, …) are written to the run's
    log file for the duration of the with-block.

    The FileHandler is ONLY used for routing global openhands_logger records; it
    is separate from the named per-run logger's own persistent handler, so
    closing it here does not affect further writes via ilogger.
    """

    _FMT = logging.Formatter(
        '%(asctime)s - %(name)s:%(levelname)s - %(message)s'
    )

    def __init__(self, log_file: str) -> None:
        self._log_file = log_file
        self._fh: logging.FileHandler | None = None

    def __enter__(self) -> None:
        fh = logging.FileHandler(self._log_file, mode='a')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(self._FMT)
        self._fh = fh
        if _THREAD_ROUTER is not None:
            _THREAD_ROUTER.register(fh)

    def __exit__(self, *_) -> None:
        if _THREAD_ROUTER is not None:
            _THREAD_ROUTER.unregister()
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def _make_run_logger(
    instance_id: str, run_num: int, log_dir: str
) -> tuple[logging.Logger, str]:
    """Create an isolated per-run logger and return (logger, log_file_path).

    Produces one log file per run::

        <log_dir>/instance_<instance_id>_run_<N>.log

    The named logger keeps its own persistent FileHandler open for the entire
    instance lifetime.  Callers that also want openhands internal messages
    (runtime_build, docker_runtime, …) should wrap their work in
    ``_RunLogContext(log_file)`` which installs a SEPARATE routing handler via
    _THREAD_ROUTER without touching the named logger's handler.
    """
    run_label = f'run_{run_num}'
    log_name = f'eval_multirun.{instance_id}.{run_label}'
    run_logger = logging.getLogger(log_name)
    log_file = os.path.join(log_dir, f'instance_{instance_id}_{run_label}.log')

    # Avoid duplicate handlers if somehow called twice.
    if run_logger.handlers:
        return run_logger, log_file

    run_logger.setLevel(logging.DEBUG)
    run_logger.propagate = False  # do NOT forward to the root/global logger

    os.makedirs(log_dir, exist_ok=True)

    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s:%(levelname)s - %(message)s')
    )
    run_logger.addHandler(fh)

    # Also mirror WARNING+ to the console so the main process sees errors
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(
        logging.Formatter(
            f'[{instance_id}][{run_label}] %(asctime)s - %(levelname)s - %(message)s'
        )
    )
    run_logger.addHandler(ch)

    return run_logger, log_file


def _eval_one_patch_in_runtime(
    runtime,
    instance_id: str,
    model_patch: str,
    test_spec,
    ci: ConditionalImports,
    is_swegym: bool,
    run_label: str,
    ilogger: logging.Logger,
) -> dict:
    """Apply *model_patch* inside *runtime*, run the eval script and return test_result."""
    test_result: dict = {
        'report': {
            'empty_generation': False,
            'resolved': False,
            'failed_apply_patch': False,
            'error_eval': False,
            'test_timeout': False,
        }
    }

    if not model_patch.strip():
        test_result['report']['empty_generation'] = True
        return test_result

    # ── upload patch + eval script ─────────────────────────────────────────
    with tempfile.TemporaryDirectory() as tmp:
        patch_path = os.path.join(tmp, 'patch.diff')
        with open(patch_path, 'w') as f:
            f.write(model_patch)
        runtime.copy_to(patch_path, '/tmp')

        eval_sh_path = os.path.join(tmp, 'eval.sh')
        with open(eval_sh_path, 'w') as f:
            f.write(test_spec.eval_script)
        runtime.copy_to(eval_sh_path, '/tmp')

    # chmod +x
    action = CmdRunAction(command='chmod +x /tmp/eval.sh')
    action.set_hard_timeout(600)
    obs = runtime.run_action(action)
    if not (isinstance(obs, CmdOutputObservation) and obs.exit_code == 0):
        ilogger.error(f'[{run_label}] chmod failed: {obs}')
        test_result['report']['error_eval'] = True
        return test_result

    # ── apply patch ────────────────────────────────────────────────────────
    apply_cmd = (
        'cd /testbed && '
        "(git apply -v /tmp/patch.diff && echo 'APPLY_PATCH_PASS' || "
        "(echo 'Failed to apply patch with git apply, trying with patch command...' && "
        "(patch --batch --fuzz=5 -p1 -i /tmp/patch.diff && echo 'APPLY_PATCH_PASS' || "
        "echo 'APPLY_PATCH_FAIL')))"
    )
    action = CmdRunAction(command=apply_cmd)
    action.set_hard_timeout(600)
    obs = runtime.run_action(action)
    apply_patch_output = obs.content if isinstance(obs, CmdOutputObservation) else ''
    test_result['apply_patch_output'] = apply_patch_output

    if 'APPLY_PATCH_FAIL' in apply_patch_output:
        ilogger.info(
            f'[{run_label}] {ci.APPLY_PATCH_FAIL}:\n{apply_patch_output}'
        )
        test_result['report']['failed_apply_patch'] = True
        return test_result

    if 'APPLY_PATCH_PASS' not in apply_patch_output:
        ilogger.error(
            f'[{run_label}] Unexpected apply output:\n{apply_patch_output}'
        )
        test_result['report']['error_eval'] = True
        return test_result

    ilogger.info(
        f'[{run_label}] {ci.APPLY_PATCH_PASS}:\n{apply_patch_output}'
    )

    # ── run eval script in background ─────────────────────────────────────
    log_file = '/tmp/eval_output.log'
    action = CmdRunAction(command=f'/tmp/eval.sh > {log_file} 2>&1 & echo $!')
    action.set_hard_timeout(300)
    obs = runtime.run_action(action)

    if not (isinstance(obs, CmdOutputObservation) and obs.exit_code == 0):
        ilogger.error(
            f'[{run_label}] Failed to start eval: {obs.content}'
        )
        test_result['report']['error_eval'] = True
        return test_result

    pid = obs.content.split()[-1].strip()
    ilogger.info(f'[{run_label}] Eval started PID={pid}')

    # ── poll for completion ────────────────────────────────────────────────
    start_t = time.time()
    timeout_s = 1800  # 30 min
    while True:
        elapsed = time.time() - start_t
        if elapsed > timeout_s:
            ilogger.warning(
                f'[{run_label}] Eval timed out after {timeout_s}s'
            )
            test_result['report']['test_timeout'] = True
            break
        chk = CmdRunAction(command=f'ps -p {pid} > /dev/null; echo $?')
        chk.set_hard_timeout(300)
        chk_obs = runtime.run_action(chk)
        if (
            isinstance(chk_obs, CmdOutputObservation)
            and chk_obs.content.split()[-1].strip() == '1'
        ):
            ilogger.info(f'[{run_label}] Eval done in {elapsed:.0f}s')
            break
        ilogger.info(f'[{run_label}] Waiting … {elapsed:.0f}s elapsed')
        time.sleep(30)

    # ── read log ───────────────────────────────────────────────────────────
    cat = CmdRunAction(command=f'cat {log_file}')
    cat.set_hard_timeout(300)
    cat_obs = runtime.run_action(cat)
    if not (isinstance(cat_obs, CmdOutputObservation) and cat_obs.exit_code == 0):
        ilogger.error(f'[{run_label}] Could not read eval log')
        test_result['report']['error_eval'] = True
        return test_result

    test_output = cat_obs.content
    test_result['test_output'] = test_output

    # ── grade ──────────────────────────────────────────────────────────────
    with tempfile.TemporaryDirectory() as tmp:
        grade_log_dir = os.path.join(tmp, 'logs', instance_id.lower())
        os.makedirs(grade_log_dir, exist_ok=True)
        test_output_path = os.path.join(grade_log_dir, 'test_output.txt')
        with open(test_output_path, 'w') as f:
            if is_swegym:
                f.write(apply_patch_output + '\n')
                f.write(f'{ci.APPLY_PATCH_PASS} (pred)\n')
            f.write(test_output)
        try:
            if is_swegym:
                extra = {'log_path': test_output_path}
            else:
                extra = {'test_log_path': test_output_path}
            ilogger.info(f'[{instance_id}] Grading answer...')
            _report = ci.get_eval_report(
                test_spec=test_spec,
                prediction={'model_patch': model_patch, 'instance_id': instance_id},
                include_tests_status=True,
                **extra,
            )
            report = _report[instance_id]
            ilogger.info(
                f'[{instance_id}] report: {report}\n'
                f'Result for {instance_id}: resolved: {report["resolved"]}'
            )
            test_result['report']['resolved'] = report['resolved']
            if 'tests_status' in report:
                test_result['report']['tests_status'] = report['tests_status']
        except Exception as exc:
            ilogger.error(f'[{instance_id}] Error grading: {exc}')
            test_result['report']['error_eval'] = True

    return test_result


# ──────────────────────────────────────────────────────────────────────────────
# Process one instance across ALL runs (main per-instance worker)
# ──────────────────────────────────────────────────────────────────────────────


def _get_runtime_config(
    instance_id: str, dataset_name: str, is_swegym: bool = True
) -> OpenHandsConfig:
    """Build an OpenHandsConfig for the eval runtime of *instance_id*."""
    image = get_instance_docker_image(instance_id, is_swegym=is_swegym)
    sandbox_config = get_default_sandbox_config_for_eval()
    sandbox_config.base_container_image = image
    sandbox_config.remote_runtime_resource_factor = get_instance_resource_factor(
        dataset_name=dataset_name,
        instance_id=instance_id,
    )
    sandbox_config.platform = 'linux/amd64'
    config = get_openhands_config_for_eval(
        runtime=os.environ.get('RUNTIME', 'docker'),
        sandbox_config=sandbox_config,
    )
    return config


def process_instance_all_runs(
    instance_id: str,
    # Each entry: (run_num, model_patch, output_file, metadata_for_run)
    run_entries: list[tuple[int, str, str, EvalMetadata]],
    test_spec,
    ci: ConditionalImports,
    status: EvalStatus,
    log_base_dir: str,
    is_swegym: bool,
    dataset_name: str,
    output_write_lock: threading.Lock,
) -> None:
    """Evaluate every run of *instance_id* inside a single Docker container."""
    image_name = get_instance_docker_image(instance_id, is_swegym=is_swegym)
    total_runs = len(run_entries)

    # One log file per run. Container startup (create_runtime / connect) are
    # wrapped in run_1's _RunLogContext so docker-related internal log messages
    # from the global openhands_logger are routed to run_1's file via
    # _THREAD_ROUTER instead of only going to openhands_global.log.
    first_run_num = run_entries[0][0]
    first_run_logger, first_log_file = _make_run_logger(
        instance_id, first_run_num, log_base_dir
    )
    first_run_logger.info(
        f'Starting {total_runs} run(s) in single container. image={image_name}\n'
        f'Hint: run "tail -f {first_log_file}" to see live logs in a separate shell'
    )

    config = _get_runtime_config(instance_id, dataset_name, is_swegym=is_swegym)

    with _RunLogContext(first_log_file):
        try:
            runtime = create_runtime(config)
            call_async_from_sync(runtime.connect)
            first_run_logger.info('Runtime connected.')
        except Exception as exc:
            first_run_logger.error(f'Failed to create/connect runtime: {exc}')
            # Mark all runs as errored so callers don't hang
            for run_num, model_patch, output_file, metadata in run_entries:
                _append_result(
                    output_file=output_file,
                    eval_output=EvalOutput(
                        instance_id=instance_id,
                        test_result={
                            'report': {
                                'empty_generation': False,
                                'resolved': False,
                                'failed_apply_patch': False,
                                'error_eval': True,
                                'test_timeout': False,
                            }
                        },
                        metadata=metadata,
                    ),
                    write_lock=output_write_lock,
                )
                status.mark_run_done(instance_id, run_num, total_runs, image_name)
            return

    try:
        for idx, (run_num, model_patch, output_file, metadata) in enumerate(
            run_entries
        ):
            run_label = f'run_{run_num}'
            ilogger, log_file = _make_run_logger(instance_id, run_num, log_base_dir)

            with _RunLogContext(log_file):
                ilogger.info(
                    f'Starting evaluation.\n'
                    f'Hint: run "tail -f {log_file}" to see live logs in a separate shell'
                )

                # Reset repository state before each run (except the very first).
                #
                # Why not recreate the container?
                #   All runs evaluate patches against the *same* instance, so eval.sh
                #   is identical and pip installs are idempotent. Recreating the
                #   container would negate the whole optimisation.
                #
                # Reset strategy (ordered by scope):
                #   1. git reset --hard        – restore tracked files to HEAD
                #   2. git clean -ffdx         – remove untracked/ignored files
                #                                (-ff also recurses into submodules,
                #                                 -x removes .pyc/__pycache__ etc.)
                #   3. rm /tmp/{patch,eval}*   – clear previous run's tmp artefacts
                #
                # This does NOT undo pip/conda installs made by eval.sh, but those
                # are deterministic for a fixed instance so cross-run contamination
                # is negligible.
                if idx > 0:
                    ilogger.info('Resetting /testbed to clean state')
                    reset_cmd = (
                        'cd /testbed'
                        ' && git reset --hard'
                        ' && git clean -ffdx'          # -ff: submodules; -x: ignored
                        ' && rm -f /tmp/patch.diff /tmp/eval.sh /tmp/eval_output.log'
                    )
                    reset_action = CmdRunAction(command=reset_cmd)
                    reset_action.set_hard_timeout(300)
                    reset_obs = runtime.run_action(reset_action)
                    if not (
                        isinstance(reset_obs, CmdOutputObservation)
                        and reset_obs.exit_code == 0
                    ):
                        ilogger.warning(
                            f'reset returned non-zero '
                            f'(exit={getattr(reset_obs, "exit_code", "?")}): {reset_obs}'
                        )

                test_result = _eval_one_patch_in_runtime(
                    runtime=runtime,
                    instance_id=instance_id,
                    model_patch=model_patch,
                    test_spec=test_spec,
                    ci=ci,
                    is_swegym=is_swegym,
                    run_label=run_label,
                    ilogger=ilogger,
                )

                eval_output = EvalOutput(
                    instance_id=instance_id,
                    test_result=test_result,
                    metadata=copy.deepcopy(metadata),
                )

                # Append result to this run's output file immediately
                _append_result(output_file, eval_output, output_write_lock)
                ilogger.info(f'Result appended → {output_file}')

                # Update status
                all_done = status.mark_run_done(
                    instance_id, run_num, total_runs, image_name
                )
                if all_done:
                    ilogger.info(f'All {total_runs} runs complete.')

    finally:
        last_run_num = run_entries[-1][0]
        teardown_logger, teardown_log = _make_run_logger(
            instance_id, last_run_num, log_base_dir
        )
        with _RunLogContext(teardown_log):
            runtime.close()
            teardown_logger.info('Runtime closed.')
            cleanup_instance_docker(instance_id, image_name, teardown_logger)
            status.mark_docker_cleaned(instance_id)
            teardown_logger.info('Docker cleanup done.')


# ──────────────────────────────────────────────────────────────────────────────
# Output writing helper
# ──────────────────────────────────────────────────────────────────────────────


def _append_result(
    output_file: str,
    eval_output: EvalOutput,
    write_lock: threading.Lock,
) -> None:
    """Thread-safely append *eval_output* as a JSON line to *output_file*."""
    with write_lock:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'a') as f:
            # Use Pydantic's own JSON serializer so SecretStr and other
            # special field types are handled correctly.
            f.write(eval_output.model_dump_json() + '\n')


# ──────────────────────────────────────────────────────────────────────────────
# Already-evaluated instance detection
# ──────────────────────────────────────────────────────────────────────────────


def _load_already_evaluated(output_file: str) -> set[str]:
    """Return the set of instance_ids already present in *output_file*."""
    done: set[str] = set()
    if not os.path.exists(output_file):
        return done
    with open(output_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if 'instance_id' in obj:
                    done.add(obj['instance_id'])
            except json.JSONDecodeError:
                pass
    return done


# ──────────────────────────────────────────────────────────────────────────────
# Summary printing
# ──────────────────────────────────────────────────────────────────────────────


def _print_run_summary(run_output_file: str, run_label: str) -> None:
    """Print resolved/failed counts for a single run's evaluation file."""
    if not os.path.exists(run_output_file):
        logger.info(f'[Summary][{run_label}] output file not found: {run_output_file}')
        return
    predictions = pd.read_json(run_output_file, lines=True)
    if predictions.empty:
        logger.info(f'[Summary][{run_label}] no results.')
        return
    fields = ['resolved', 'failed_apply_patch', 'error_eval', 'empty_generation']
    for field in fields:
        try:
            count = predictions.apply(
                lambda row: row['test_result']['report'].get(field, False), axis=1
            ).sum()
            logger.info(
                f'[Summary][{run_label}] {field}: {count}/{len(predictions)} '
                f'({count / len(predictions):.2%})'
            )
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Multi-run SWE-bench eval with Docker image reuse across runs'
    )
    parser.add_argument(
        '--base-dir',
        required=True,
        help='Directory that contains the run subdirectories (e.g. …-run_1, …-run_2, …)',
    )
    parser.add_argument(
        '--dataset',
        default='princeton-nlp/SWE-bench_Verified',
        help='HuggingFace dataset name (e.g. SWE-Gym/SWE-Gym)',
    )
    parser.add_argument(
        '--split',
        default='test',
        help='Dataset split (e.g. train, test)',
    )
    parser.add_argument(
        '--eval-num-workers',
        type=int,
        default=4,
        help='Number of instances to evaluate in parallel',
    )
    parser.add_argument(
        '--eval-n-limit',
        type=int,
        default=None,
        help='Limit the number of instances to evaluate (for debugging)',
    )
    parser.add_argument(
        '--run-pattern',
        default='*run_*',
        help='Glob pattern to identify run subdirectories inside --base-dir',
    )
    args = parser.parse_args()

    base_dir = os.path.abspath(args.base_dir)
    if not os.path.isdir(base_dir):
        raise SystemExit(f'--base-dir does not exist: {base_dir}')

    is_swegym = 'SWE-Gym' in args.dataset

    # ── conditional imports ───────────────────────────────────────────────
    if is_swegym:
        from swegym.harness.grading import get_eval_report
        from swegym.harness.run_evaluation import APPLY_PATCH_FAIL, APPLY_PATCH_PASS
        from swegym.harness.test_spec import SWEbenchInstance, make_test_spec
        from swegym.harness.utils import load_swebench_dataset
    else:
        from swebench.harness.grading import get_eval_report
        from swebench.harness.run_evaluation import APPLY_PATCH_FAIL, APPLY_PATCH_PASS
        from swebench.harness.test_spec.test_spec import SWEbenchInstance, make_test_spec
        from swebench.harness.utils import load_swebench_dataset

    ci = ConditionalImports(
        get_eval_report=get_eval_report,
        APPLY_PATCH_FAIL=APPLY_PATCH_FAIL,
        APPLY_PATCH_PASS=APPLY_PATCH_PASS,
    )

    # ── discover run directories ──────────────────────────────────────────
    import glob

    run_dirs_raw = sorted(
        glob.glob(os.path.join(base_dir, args.run_pattern))
    )
    run_dirs = [d for d in run_dirs_raw if os.path.isdir(d)]

    if not run_dirs:
        raise SystemExit(
            f'No run directories found in {base_dir} matching pattern "{args.run_pattern}"'
        )

    logger.info(f'Found {len(run_dirs)} run directories:')
    for rd in run_dirs:
        logger.info(f'  {rd}')

    # ── build run_num → (output_file, eval_output_file, metadata) map ────
    # run_num is 1-based; we derive it from the sorted order of directories.
    RunInfo = dict  # {run_num, run_dir, output_file, eval_output_file, metadata}
    run_infos: list[RunInfo] = []
    for idx, run_dir in enumerate(run_dirs):
        run_num = idx + 1
        output_file = os.path.join(run_dir, 'output.jsonl')
        eval_output_file = os.path.join(run_dir, 'output.swebench_eval.jsonl')
        if not os.path.isfile(output_file):
            logger.warning(
                f'[run_{run_num}] output.jsonl not found in {run_dir}, skipping.'
            )
            continue
        # Try to load metadata
        metadata: EvalMetadata | None = None
        metadata_path = os.path.join(run_dir, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path) as f:
                metadata = EvalMetadata.model_validate_json(f.read())
        else:
            metadata = EvalMetadata(
                agent_class='dummy_agent',
                llm_config=LLMConfig(model='dummy_model'),
                max_iterations=1,
                eval_output_dir=run_dir,
                start_time=time.strftime('%Y-%m-%d %H:%M:%S'),
                git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'])
                .decode()
                .strip(),
                dataset=args.dataset,
                details={},
            )
        run_infos.append(
            {
                'run_num': run_num,
                'run_dir': run_dir,
                'output_file': output_file,
                'eval_output_file': eval_output_file,
                'metadata': metadata,
            }
        )

    if not run_infos:
        raise SystemExit('No valid run directories with output.jsonl found.')

    total_run_count = len(run_infos)
    logger.info(f'Processing {total_run_count} valid run(s).')

    # ── load full SWE-bench/SWE-Gym dataset once ──────────────────────────
    logger.info(f'Loading dataset {args.dataset} split={args.split} …')
    full_dataset: list[SWEbenchInstance] = load_swebench_dataset(
        args.dataset, args.split
    )
    instance_id_to_dataset = {inst['instance_id']: inst for inst in full_dataset}
    logger.info(f'Dataset loaded: {len(instance_id_to_dataset)} instances.')

    # ── load predictions from every run, group by instance_id ────────────
    # instance_id  →  list of (run_num, model_patch, eval_output_file, metadata)
    from collections import defaultdict

    instance_run_map: dict[
        str, list[tuple[int, str, str, EvalMetadata]]
    ] = defaultdict(list)

    required_fields = ['instance_id', 'model_patch', 'test_result']

    for ri in run_infos:
        run_num = ri['run_num']
        output_file = ri['output_file']
        eval_output_file = ri['eval_output_file']
        metadata = ri['metadata']

        # Which instance_ids already evaluated for THIS run?
        already_done = _load_already_evaluated(eval_output_file)

        logger.info(
            f'[run_{run_num}] Loading {output_file} '
            f'({len(already_done)} already evaluated)'
        )
        with open(output_file) as f:
            for line in tqdm(f, desc=f'run_{run_num}'):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                iid = obj.get('instance_id')
                if iid is None:
                    continue
                if iid not in instance_id_to_dataset:
                    logger.warning(
                        f'[run_{run_num}] instance_id {iid!r} not found in dataset, skipping.'
                    )
                    continue
                if iid in already_done:
                    logger.debug(f'[run_{run_num}] {iid} already evaluated, skipping.')
                    continue

                # Extract model_patch
                model_patch = obj.get('model_patch', '')
                if not model_patch and 'test_result' in obj:
                    model_patch = obj['test_result'].get('git_patch', '')
                model_patch = process_git_patch(model_patch) if model_patch else ''

                instance_run_map[iid].append(
                    (run_num, model_patch, eval_output_file, metadata)
                )

    all_instance_ids = sorted(instance_run_map.keys())
    logger.info(
        f'Total unique instances across all runs to evaluate: {len(all_instance_ids)}'
    )

    if args.eval_n_limit is not None:
        all_instance_ids = all_instance_ids[: args.eval_n_limit]
        logger.info(f'Limited to {len(all_instance_ids)} instances (--eval-n-limit).')

    # ── build test specs once (shared across runs for the same instance) ──
    logger.info('Building test specs …')
    instance_id_to_test_spec: dict = {}
    for iid in all_instance_ids:
        inst_data = instance_id_to_dataset[iid]
        instance_id_to_test_spec[iid] = make_test_spec(inst_data)
    logger.info('Test specs ready.')

    # ── status file ────────────────────────────────────────────────────────
    status_file_path = os.path.join(base_dir, 'multirun_eval_status.json')
    status = EvalStatus(status_file_path)
    logger.info(f'Status file: {status_file_path}')

    # ── log directory ──────────────────────────────────────────────────────
    log_base_dir = os.path.join(base_dir, 'multirun_eval_logs')
    os.makedirs(log_base_dir, exist_ok=True)

    # Set up the global openhands_logger:
    #   1. A plain FileHandler writes ALL internal messages to openhands_global.log
    #      (useful for debugging / messages from the main thread).
    #   2. _THREAD_ROUTER routes per-thread records to the correct per-run file.
    #      Worker threads register their FileHandler via _RunLogContext before
    #      calling create_runtime / connect, so docker-related messages
    #      (runtime_build.py, docker_runtime.py, …) land in the right run log.
    global _THREAD_ROUTER
    global_log_file = os.path.join(log_base_dir, 'openhands_global.log')
    _global_fh = logging.FileHandler(global_log_file, mode='a')
    _global_fh.setLevel(logging.DEBUG)
    _global_fh.setFormatter(
        logging.Formatter(
            '%(asctime)s - %(name)s:%(levelname)s: %(filename)s:%(lineno)d - %(message)s'
        )
    )
    logger.addHandler(_global_fh)
    _THREAD_ROUTER = _ThreadRoutingHandler()
    _THREAD_ROUTER.setLevel(logging.DEBUG)
    logger.addHandler(_THREAD_ROUTER)
    logger.info(f'Global openhands log: {global_log_file}')

    # ── shared write lock (instances running in parallel share output files) ──
    output_write_lock = threading.Lock()

    # ── filter out already-fully-done instances ────────────────────────────
    instances_to_process = [
        iid
        for iid in all_instance_ids
        if not status.is_all_runs_done(iid)
    ]
    logger.info(
        f'{len(instances_to_process)} instances still need evaluation '
        f'({len(all_instance_ids) - len(instances_to_process)} already done).'
    )

    if not instances_to_process:
        logger.info('Nothing to do!')
    else:
        # ── run evaluation with thread pool ───────────────────────────────
        logger.info(
            f'Starting evaluation with {args.eval_num_workers} parallel worker(s)…'
        )
        with ThreadPoolExecutor(max_workers=args.eval_num_workers) as executor:
            futures = {
                executor.submit(
                    process_instance_all_runs,
                    instance_id=iid,
                    run_entries=instance_run_map[iid],
                    test_spec=instance_id_to_test_spec[iid],
                    ci=ci,
                    status=status,
                    log_base_dir=log_base_dir,
                    is_swegym=is_swegym,
                    dataset_name=args.dataset,
                    output_write_lock=output_write_lock,
                ): iid
                for iid in instances_to_process
            }

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc='Instances',
            ):
                iid = futures[future]
                try:
                    future.result()
                    logger.info(f'[{iid}] Worker finished successfully.')
                except Exception as exc:
                    logger.error(f'[{iid}] Worker raised exception: {exc}', exc_info=True)

    # ── print per-run summaries ────────────────────────────────────────────
    logger.info('=' * 60)
    logger.info('Per-run evaluation summaries:')
    logger.info('=' * 60)
    for ri in run_infos:
        _print_run_summary(ri['eval_output_file'], f"run_{ri['run_num']}")

    logger.info('=' * 60)
    logger.info('Multi-run evaluation complete.')
    logger.info('=' * 60)


if __name__ == '__main__':
    main()
