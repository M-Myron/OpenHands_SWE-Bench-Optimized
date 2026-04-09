# Noise Distribution: SWE-bench Verified

Heuristic noise analysis on the SWE-bench Verified test split (N = 500), using the taxonomy defined in [noise_taxonomy_swegym.md](noise_taxonomy_swegym.md).

**Data source:** `princeton-nlp__SWE-bench_Verified-test / Qwen3-Coder-30B-A3B-Instruct_maxiter_100_N_v0.61.0-no-hint-run_1 / output.with_completions.jsonl.gz`

---

## Overall Distribution (N = 500)

| Noise Signal | Count | Percentage |
|-------------|-------|------------|
| **Category A: Issue Description Noise** | | |
| Under-specified issue (< 100 chars) | 0 | 0.0% |
| **Category B: P→T Noise** | | |
| High hunk-to-test ratio (> 5) | 28 | 5.6% |
| **Category C: T→F Noise** | | |
| String literal assertion ratio > 50% | 112 | 22.4% |
| New symbol coupling in tests | 33 | 6.6% |
| Any Category C flag | 138 | 27.6% |
| **Overall** | | |
| Any noise flag | 155 | 31.0% |

## Co-occurrence

| Combination | Count | Percentage |
|------------|-------|------------|
| A ∩ B | 0 | 0.0% |
| A ∩ C | 0 | 0.0% |
| B ∩ C | 11 | 2.2% |
| A ∩ B ∩ C | 0 | 0.0% |

## Per-Repository Distribution

| Repository | N | A-noise | B-noise | C-noise | Any |
|-----------|---|---------|---------|---------|-----|
| django/django | 231 | 0% | 4% | 27% | 30% |
| sympy/sympy | 75 | 0% | 9% | 33% | 39% |
| sphinx-doc/sphinx | 44 | 0% | 9% | 41% | 45% |
| matplotlib/matplotlib | 34 | 0% | 3% | 15% | 18% |
| scikit-learn/scikit-learn | 32 | 0% | 6% | 9% | 16% |
| astropy/astropy | 22 | 0% | 0% | 32% | 32% |
| pydata/xarray | 22 | 0% | 9% | 27% | 27% |
| pytest-dev/pytest | 19 | 0% | 5% | 26% | 32% |
| pylint-dev/pylint | 10 | 0% | 10% | 30% | 30% |
| psf/requests | 8 | 0% | 0% | 38% | 38% |
| mwaskom/seaborn | 2 | 0% | 0% | 0% | 0% |
| pallets/flask | 1 | 0% | 0% | 0% | 0% |

## Per-Difficulty Distribution

| Difficulty | N | A-noise | B-noise | C-noise | Any |
|-----------|---|---------|---------|---------|-----|
| < 15 min fix | 194 | 0% | 1% | 27% | 27% |
| 15 min – 1 hour | 261 | 0% | 7% | 27% | 31% |
| 1–4 hours | 42 | 0% | 17% | 31% | 43% |
| > 4 hours | 3 | 0% | 67% | 67% | 67% |

## Noise Flag vs. Resolve Rate (Qwen3-Coder-480B, 59.2% overall)

Cross-referencing noise flags with actual agent resolve outcomes from `report.json` (296 resolved / 500 total).

### Resolve Rate by Noise Flag

| Flag | Flagged (resolve rate) | Unflagged (resolve rate) | Delta |
|------|----------------------|-------------------------|-------|
| B: Hunk/test ratio > 5 | 12/28 (42.9%) | 284/468 (60.7%) | **-17.8pp** |
| C: String assertion > 50% | 58/110 (52.7%) | 238/386 (61.7%) | **-8.9pp** |
| C: Symbol coupling | 20/32 (62.5%) | 276/464 (59.5%) | +3.0pp |
| Any C | 75/136 (55.1%) | 221/360 (61.4%) | **-6.2pp** |
| Any noise flag | 82/153 (53.6%) | 214/343 (62.4%) | **-8.8pp** |

### Noise Prevalence: Resolved vs. Unresolved

| Flag | Resolved (N=296) | Unresolved (N=200) |
|------|------------------|-------------------|
| B: Hunk/test > 5 | 4.1% | 8.0% |
| C: String assert > 50% | 19.6% | 26.0% |
| C: Symbol coupling | 6.8% | 6.0% |
| Any C | 25.3% | 30.5% |
| Any noise | 27.7% | 35.5% |

### Per-Repo: Resolve Rate (Noisy vs. Clean)

| Repository | N | Noisy | Clean | RR-noisy | RR-clean | Delta |
|-----------|---|-------|-------|----------|----------|-------|
| django/django | 230 | 69 | 161 | 71.0% | 70.2% | +0.8pp |
| sympy/sympy | 74 | 29 | 45 | 48.3% | 66.7% | **-18.4pp** |
| sphinx-doc/sphinx | 44 | 20 | 24 | 0.0% | 0.0% | +0.0pp |
| matplotlib/matplotlib | 34 | 6 | 28 | 50.0% | 57.1% | -7.1pp |
| scikit-learn/scikit-learn | 31 | 5 | 26 | 80.0% | 84.6% | -4.6pp |
| pydata/xarray | 22 | 6 | 16 | 66.7% | 75.0% | -8.3pp |
| pytest-dev/pytest | 19 | 6 | 13 | 66.7% | 92.3% | **-25.6pp** |
| pylint-dev/pylint | 10 | 3 | 7 | 0.0% | 28.6% | -28.6pp |

**Key findings:**

- Instances flagged with **any noise** have a **8.8pp lower resolve rate** (53.6% vs. 62.4%).
- **Category B** (patch-to-test misalignment) shows the strongest signal: **-17.8pp** resolve rate drop for flagged instances. This makes sense — when the hunk-to-test ratio is high, the agent's patch may address the right issue but miss untested aspects that still cause test failures.
- **String literal assertions** (C1) correlate with a **-8.9pp** drop, consistent with the hypothesis that implementation-specific test assertions penalize correct alternative implementations.
- **Symbol coupling** (C3) shows no correlation (+3.0pp), suggesting that when agents solve a problem, they often end up using similar function names as the golden patch.
- Unresolved instances have higher noise prevalence across all categories (35.5% vs. 27.7% any noise).
- Per-repo, the strongest effects are in `sympy` (-18.4pp) and `pytest` (-25.6pp). `django` shows no difference, likely because its large sample dilutes the effect and its noise is predominantly C-type which has a weaker individual signal.

---

## Comparison: SWE-bench Verified vs. SWE-Gym Train

| Metric | SWE-Gym Train (N=2,092) | SWE-bench Verified (N=500) |
|--------|------------------------|---------------------------|
| Any noise flag | 48.9% | 31.0% |
| A: Under-specified | 1.4% | 0.0% |
| B: Hunk-to-test > 5 | 21.5% | 5.6% |
| C: Any C flag | 35.4% | 27.6% |
| C: String assertions > 50% | 24.2% | 22.4% |
| C: Symbol coupling | 17.0% | 6.6% |

**Key observations:**

- SWE-bench Verified is substantially cleaner than SWE-Gym Train (31.0% vs. 48.9% any noise), consistent with its human-verified curation.
- Category A noise is absent — all Verified issues have detailed problem statements (> 100 chars).
- Category B drops significantly (5.6% vs. 21.5%), indicating better patch-to-test alignment in Verified instances.
- Category C remains the dominant noise source (27.6%), primarily driven by string literal assertions (22.4%). This is expected since test implementation patterns are not part of the verification criteria.
- Harder instances have more noise: 43% for 1–4 hour problems vs. 27% for < 15 min, likely because complex patches have more hunks and more opportunity for implementation-specific testing.
- `sphinx-doc/sphinx` (45%) and `sympy/sympy` (39%) are the noisiest repos; `scikit-learn` (16%) and `matplotlib` (18%) are the cleanest.

---

*Generated using heuristic detectors from `compute_noise_distribution.py`. Full per-instance metrics saved in `noise_metrics_swebench_verified.json`.*
