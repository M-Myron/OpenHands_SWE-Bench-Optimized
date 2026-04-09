import json
import gzip
import re
from collections import Counter, defaultdict
from tqdm import tqdm

# ── Load full dataset ──
FILE_PATH = '/home/v-murongma/code/OpenHands_SWE-Bench-Optimized/evaluation/evaluation_outputs/outputs/SWE-Gym__SWE-Gym-train/CodeActAgent/Qwen3-Coder-480B-A35B-Instruct_maxiter_100_N_v0.61.0-no-hint-train-qwen3_coder_480b_a35b_instruct-t05/Qwen3-Coder-480B-A35B-Instruct_maxiter_100_N_v0.61.0-no-hint-train-qwen3_coder_480b_a35b_instruct-t05-run_1/output.with_completions.jsonl.gz'

raw_data = []
with gzip.open(FILE_PATH, 'rt', encoding='utf-8') as f:
    for line in tqdm(f, desc="Loading"):
        raw_data.append(json.loads(line))

data = [d for d in raw_data if d.get('instance') is not None]
print(f"Total valid instances: {len(data)}")


def compute_noise_metrics(d):
    """Compute heuristic noise metrics for one instance."""
    inst = d['instance']
    ps = inst['problem_statement']
    patch = inst['patch']
    test_patch = inst['test_patch']

    # ── Category A: Issue Description Noise ──
    ps_len = len(ps)
    patch_len = len(patch)

    patch_files = [l for l in patch.split('\n') if l.startswith('diff --git')]
    num_patch_files = len(patch_files)

    patch_added = sum(1 for l in patch.split('\n') if l.startswith('+') and not l.startswith('+++'))
    patch_removed = sum(1 for l in patch.split('\n') if l.startswith('-') and not l.startswith('---'))

    # ── Category B: Patch-to-Test Misalignment ──
    test_files = [l for l in test_patch.split('\n') if l.startswith('diff --git')]
    num_test_files = len(test_files)

    # Count hunks in patch
    num_patch_hunks = len(re.findall(r'^@@', patch, re.MULTILINE))
    # Count test functions added
    test_added_lines = [l for l in test_patch.split('\n') if l.startswith('+') and not l.startswith('+++')]
    num_test_functions = sum(1 for l in test_added_lines if re.search(r'def test_', l))
    test_added = len(test_added_lines)

    # ── Category C: Test-as-Feedback Noise ──
    # Count string literal assertions in test patch
    string_assertions = 0
    total_assertions = 0
    for line in test_added_lines:
        line_s = line.strip()
        is_assertion = any(kw in line_s for kw in [
            'assert', 'should.equal', 'should.contain', 'assertEqual',
            'assertRaises', 'assertIn', 'match(', 'raises(',
            'should.have.key', '.equals(', 'assert_equal',
        ])
        if is_assertion:
            total_assertions += 1
            # Check if assertion contains a string literal
            if re.search(r'["\'].*["\']', line_s):
                string_assertions += 1

    string_assertion_ratio = string_assertions / max(total_assertions, 1) if total_assertions > 0 else 0.0

    # Check for new function/class definitions in patch referenced in tests
    new_defs_in_patch = set()
    for line in patch.split('\n'):
        if line.startswith('+') and not line.startswith('+++'):
            m = re.search(r'def (\w+)', line)
            if m:
                new_defs_in_patch.add(m.group(1))
            m = re.search(r'class (\w+)', line)
            if m:
                new_defs_in_patch.add(m.group(1))

    new_symbols_in_tests = 0
    for sym in new_defs_in_patch:
        if sym in test_patch:
            new_symbols_in_tests += 1

    return {
        'instance_id': d['instance_id'],
        'repo': inst['repo'],
        'ps_len': ps_len,
        'patch_len': patch_len,
        'num_patch_files': num_patch_files,
        'patch_added': patch_added,
        'patch_removed': patch_removed,
        'num_patch_hunks': num_patch_hunks,
        'num_test_files': num_test_files,
        'num_test_functions': num_test_functions,
        'test_added': test_added,
        'total_assertions': total_assertions,
        'string_assertions': string_assertions,
        'string_assertion_ratio': string_assertion_ratio,
        'new_defs_in_patch': len(new_defs_in_patch),
        'new_symbols_in_tests': new_symbols_in_tests,
        # Flags
        'flag_A_underspec': ps_len < 100,
        'flag_B_hunk_test_ratio': (num_patch_hunks / max(num_test_functions, 1)) > 5 if num_test_functions > 0 else num_patch_hunks > 5,
        'flag_C_string_assertions': string_assertion_ratio > 0.5 if total_assertions > 0 else False,
        'flag_C_symbol_coupling': new_symbols_in_tests > 0,
    }


# ── Compute metrics for all instances ──
metrics = []
for d in tqdm(data, desc="Computing metrics"):
    metrics.append(compute_noise_metrics(d))

print(f"Computed metrics for {len(metrics)} instances")


# ── Noise Distribution ──
print("\n" + "="*80)
print("NOISE DISTRIBUTION ACROSS FULL SWE-GYM TRAIN SET")
print("="*80)

N = len(metrics)

flags = [
    ('flag_A_underspec', 'Category A: Under-specified issue (<100 chars)'),
    ('flag_B_hunk_test_ratio', 'Category B: High hunk-to-test ratio (>3)'),
    ('flag_C_string_assertions', 'Category C: String literal assertion ratio > 50%'),
    ('flag_C_symbol_coupling', 'Category C: New symbol coupling in tests'),
]

print(f"\nTotal instances: {N}\n")
for flag_key, flag_desc in flags:
    count = sum(1 for m in metrics if m[flag_key])
    print(f"  {flag_desc}: {count}/{N} ({100*count/N:.1f}%)")

# Any noise
any_a = sum(1 for m in metrics if m['flag_A_underspec'])
any_b = sum(1 for m in metrics if m['flag_B_hunk_test_ratio'])
any_c = sum(1 for m in metrics if m['flag_C_string_assertions'] or m['flag_C_symbol_coupling'])
any_noise = sum(1 for m in metrics if m['flag_A_underspec'] or m['flag_B_hunk_test_ratio'] or m['flag_C_string_assertions'] or m['flag_C_symbol_coupling'])

print(f"\n  Any Category A noise: {any_a}/{N} ({100*any_a/N:.1f}%)")
print(f"  Any Category B noise: {any_b}/{N} ({100*any_b/N:.1f}%)")
print(f"  Any Category C noise: {any_c}/{N} ({100*any_c/N:.1f}%)")
print(f"  Any noise flag: {any_noise}/{N} ({100*any_noise/N:.1f}%)")


# ── Per-repo distribution ──
print("\n" + "="*80)
print("PER-REPO NOISE DISTRIBUTION")
print("="*80)

repo_metrics = defaultdict(list)
for m in metrics:
    repo_metrics[m['repo']].append(m)

print(f"\n{'Repo':<40} {'N':>5} {'A-noise':>8} {'B-noise':>8} {'C-noise':>8} {'Any':>8}")
print("-"*80)
for repo in sorted(repo_metrics.keys(), key=lambda r: len(repo_metrics[r]), reverse=True):
    ms = repo_metrics[repo]
    n = len(ms)
    a = sum(1 for m in ms if m['flag_A_underspec'])
    b = sum(1 for m in ms if m['flag_B_hunk_test_ratio'])
    c = sum(1 for m in ms if m['flag_C_string_assertions'] or m['flag_C_symbol_coupling'])
    any_ = sum(1 for m in ms if m['flag_A_underspec'] or m['flag_B_hunk_test_ratio'] or m['flag_C_string_assertions'] or m['flag_C_symbol_coupling'])
    print(f"{repo:<40} {n:>5} {a:>5}({100*a/n:>4.0f}%) {b:>5}({100*b/n:>4.0f}%) {c:>5}({100*c/n:>4.0f}%) {any_:>5}({100*any_/n:>4.0f}%)")


# ── Co-occurrence matrix ──
print("\n" + "="*80)
print("NOISE CO-OCCURRENCE")
print("="*80)

co_AB = sum(1 for m in metrics if m['flag_A_underspec'] and m['flag_B_hunk_test_ratio'])
co_AC = sum(1 for m in metrics if m['flag_A_underspec'] and (m['flag_C_string_assertions'] or m['flag_C_symbol_coupling']))
co_BC = sum(1 for m in metrics if m['flag_B_hunk_test_ratio'] and (m['flag_C_string_assertions'] or m['flag_C_symbol_coupling']))
co_ABC = sum(1 for m in metrics if m['flag_A_underspec'] and m['flag_B_hunk_test_ratio'] and (m['flag_C_string_assertions'] or m['flag_C_symbol_coupling']))

print(f"\n  A ∩ B: {co_AB}/{N} ({100*co_AB/N:.1f}%)")
print(f"  A ∩ C: {co_AC}/{N} ({100*co_AC/N:.1f}%)")
print(f"  B ∩ C: {co_BC}/{N} ({100*co_BC/N:.1f}%)")
print(f"  A ∩ B ∩ C: {co_ABC}/{N} ({100*co_ABC/N:.1f}%)")


# ── Save full metrics ──
with open('noise_metrics_full.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"\nFull metrics saved to noise_metrics_full.json")
