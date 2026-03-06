# Workspace/Testbed Import Mismatch — Investigation & Fix Report

## TL;DR

SWE-bench evaluation agents edit files in `/workspace/$REPO__$VERSION/` but the conda testbed environment's editable install still points to `/testbed/`. This means test runs import the **original, unmodified** code from `/testbed` instead of the agent's edits in `/workspace`, causing confusion and false failures.

**Fix**: Run `pip install -e . --no-deps` from `/workspace/$REPO__$VERSION/` during `initialize_runtime()` to redirect the editable install target.

---

## 1. Root Cause

### Container Layout (Inside SWE-bench Docker Image)

```
/testbed/                          ← Original repo source (read-only intent)
/opt/miniconda3/envs/testbed/      ← Conda env with editable install
    lib/python3.x/site-packages/
        django.egg-link            ← Points to /testbed  ← THE PROBLEM
/workspace/$REPO__$VERSION/        ← Agent's working copy (cp from /testbed)
```

### Boot Sequence (`instance_swe_entry.sh`)

1. `cp -r /testbed /workspace/$WORKSPACE_NAME` — creates agent's working copy
2. `conda activate testbed` — activates the conda env
3. Agent edits files under `/workspace/$REPO__$VERSION/`
4. Tests run: `python -c "import django"` → resolves via `.egg-link` → **`/testbed/django/`** ❌

### Why Tests Fail to See Agent's Edits

The conda env's `.egg-link` (or `__editable__*.pth`) file was written by `pip install -e /testbed` when the Docker image was built. Copying `/testbed` to `/workspace` does **not** update this pointer.

---

## 2. Empirical Verification

Verified live inside a running container (`django__django__3.1` instance):

```bash
# BEFORE fix
python -c "import django; print(django.__file__)"
# → /testbed/django/__init__.py   ← wrong

# AFTER running: pip install -e . --no-deps  (from /workspace/django__django__3.1/)
python -c "import django; print(django.__file__)"
# → /workspace/django__django__3.1/django/__init__.py  ← correct

# Confirmed: echo "# test" >> django/__init__.py was visible after reinstall ✓
```

---

## 3. Trajectory Evidence (Bug Confirmed in Real Runs)

Two real trajectories analyzed, both showing the same failure pattern:

### Trajectory 1: `gpt-4.1` run on `django__django-12663` (DecimalField TypeError fix)
- **Step 13**: Agent applies fix to `/workspace/django__django__3.2/django/db/models/fields/__init__.py`
- **Step 31** (test output): `Testing against Django installed in '/testbed/django'` ← imports /testbed
- **Step 33**: Agent re-applies the same fix directly to `/testbed/django/...` to work around the issue
- **Result**: Agent wasted ~20 steps and discovered the problem empirically

### Trajectory 2: Same instance, different run — same pattern confirmed

### Trajectory 3: `Qwen3-Coder-30B` on `django__django-13023` (debug-run_2, WITH fix applied)
- **Step 32** (test output): `Testing against Django installed in '/workspace/django__django__3.2/django'` ✓
- Agent applied fix once, tests passed — **no `/testbed` confusion**
- This confirms the fix works correctly

---

## 4. The Fix

### What Was Changed

Added a `pip install -e . --no-deps` step inside `initialize_runtime()`, immediately after the `which python` assertion that confirms the testbed conda env is active.

### Files Modified

| File | Status |
|------|--------|
| `evaluation/benchmarks/swe_bench/run_infer.py` | ✅ Fixed (lines 435–449) |
| `evaluation/benchmarks/swe_bench_optimized/run_infer.py` | ✅ Fixed (same block) |

### The Fix Block (identical in both files)

```python
# After this assertion:
assert_and_raise(
    obs.exit_code == 0 and 'testbed' in obs.content,
    f'Expected to find python interpreter from testbed, but got: {str(obs)}',
)

# --- NEW BLOCK ---
# Re-install the package in editable mode from /workspace so that agent's edits
# are immediately reflected when tests import the package. Without this, the
# testbed conda env's editable install still points to /testbed (the original copy),
# making the agent's changes invisible to the test runner.
action = CmdRunAction(
    command=f'cd /workspace/{workspace_dir_name} && pip install -e . --no-deps -q 2>&1 || true'
)
action.set_hard_timeout(300)
logger.info(action, extra={'msg_type': 'ACTION'})
obs = runtime.run_action(action)
logger.info(obs, extra={'msg_type': 'OBSERVATION'})
logger.info(
    f'Re-installed package from /workspace/{workspace_dir_name} in editable mode '
    f'so agent edits are reflected in test imports (exit_code={obs.exit_code})'
)
```

### Design Decisions

- **`--no-deps`**: Avoids re-installing all dependencies (fast, ~2–5 sec)
- **`-q`**: Quiet output to keep logs clean
- **`2>&1 || true`**: Suppresses errors for repos with no `setup.py`/`setup.cfg` (prevents abort)
- **Guard condition**: Only runs under `if DATASET_TYPE != 'Multimodal' and DATASET_TYPE != 'SWE-bench-Live':` — those datasets don't use the testbed conda pattern

---

## 5. Known Limitations & Follow-up Work

### 5.1 Debug Artifact In `swe_bench/run_infer.py`

There is a hardcoded debug override in the non-iterative eval path (around line ~870):

```python
# REMOVE BEFORE PRODUCTION USE:
instances = prepare_dataset(
    swe_bench_tests,
    output_file,
    eval_n_limit=1,
    eval_ids=['django__django-13023'],
)
```

This overrides `--eval-n-limit` and always runs only `django__django-13023`. **Must be removed** before running real evaluation sweeps.

### 5.2 Repos Without `setup.py` / `pyproject.toml`

The `|| true` guard handles this gracefully (no abort), but the editable install will silently fail for such repos. These repos likely don't have the testbed/workspace split problem anyway (no Python package to install), but worth verifying for edge cases.

### 5.3 `swe_bench_optimized/run_infer.py` — Verify Parity

Both files are believed to have the fix applied, but should be double-checked to ensure they are in sync, especially if either file receives future updates.

### 5.4 The Fix Has Not Been Extended to `SWE-bench-Live` or `SWE-Gym`

- **SWE-bench-Live**: Uses `instance_swe_entry_live.sh` and is excluded by the guard condition. Its setup may differ — investigate if similar import issues exist there.
- **SWE-Gym**: Uses OpenHands-built images (not official SWE-bench images), may have a different editable install layout.

---

## 6. How to Verify the Fix in New Trajectories

When inspecting a new trajectory's LLM completion JSON, look for this in the test output (step where agent runs the test suite):

```
# GOOD (fix working):
Testing against Django installed in '/workspace/django__django__3.2/django'

# BAD (fix not working or bypassed):
Testing against Django installed in '/testbed/django'
```

In the notebook [evaluation/evaluation_analysis/inspect_response.ipynb](evaluation/evaluation_analysis/inspect_response.ipynb), the `extract_tool_trajectory()` function can be used to scan through steps and look for this pattern in `OBSERVATION` outputs.

---

## 7. Key File Locations

```
evaluation/benchmarks/swe_bench/
├── run_infer.py                          ← Main evaluator (fix at lines 435–449)
├── scripts/setup/
│   ├── instance_swe_entry.sh             ← cp /testbed → /workspace, conda activate
│   ├── instance_swe_entry_live.sh        ← SWE-bench-Live variant
│   └── instance_swe_entry_rebench.sh     ← SWE-rebench variant
└── prompts/
    ├── swe_gpt4.j2                       ← 8-step workflow for GPT-4 models
    ├── swe_default.j2                    ← 6-phase workflow for other models
    └── swt.j2                            ← Test-writing template

evaluation/benchmarks/swe_bench_optimized/
└── run_infer.py                          ← Optimized variant (fix also applied)

evaluation/evaluation_analysis/
└── inspect_response.ipynb               ← Notebook for trajectory inspection
```

---

## 8. Session Continuation Checklist

For the next session working on this topic:

- [ ] **Remove debug artifact** in `swe_bench/run_infer.py` (~line 870): the hardcoded `eval_ids=['django__django-13023']` override
- [ ] **Run a production sweep** with the fix to measure resolution rate improvement vs. baseline (pre-fix trajectories show the `/testbed` confusion)
- [ ] **Investigate SWE-bench-Live** to check if the same import issue exists in `instance_swe_entry_live.sh`
- [ ] **Investigate SWE-Gym** image layout to determine if fix is needed there
- [ ] **Check `swe_bench_optimized/run_infer.py`** is in sync with `swe_bench/run_infer.py` for the fix block
- [ ] **Baseline comparison**: Compare solve rates of runs before and after the fix on the same benchmark subset (e.g., `django__django-12663` class of issues)
