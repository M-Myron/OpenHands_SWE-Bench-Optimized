# Noise Taxonomy in SWE-Bench / SWE-Gym Training Data

## 1. Problem Statement

SWE-Bench and its derivatives (SWE-Gym) are widely used benchmarks for evaluating and training LLM-based coding agents on real-world software engineering tasks. Each instance consists of a **triple**: (Issue Description, Golden Code Fix Patch, Golden Test Patch). In the context of agent training, these three components serve distinct roles:

1. **Issue Description** — the natural **input** to the agent. This is what the agent sees and must reason about.
2. **Golden Code Fix Patch** — serves as the **oracle reference** for guided training methods (e.g., reinforcement learning reward shaping, SFT data generation via oracle-guided trajectories, or trajectory filtering).
3. **Golden Test Patch** — serves as the natural **feedback signal**. Tests are used to evaluate agent-generated patches in both SFT filtering (accept/reject trajectories) and RL (reward function).

However, these three components are sourced directly from real GitHub pull requests, where they were never designed to form a coherent training signal. The issue was written by a user, the patch was written by a developer (possibly addressing more than what the issue asked), and the tests were written to validate the developer's specific implementation. The relationships between these components embed severe noise that systematically affects both **evaluation reliability** and **training signal quality**.

We formalize three categories of misalignment noise across these components and propose detection methods for each.

---

## 2. Noise Taxonomy

### Category A: Issue Description Noise (I→P/T Noise)

**Definition:** The issue description, as the sole input to the agent, is misaligned with the golden code fix patch and/or the golden test patch. The misalignment can take the form of the issue being too narrow (patch does more), too vague (patch cannot be derived), or actively misleading (issue suggests one approach, patch does another).

Since both the golden patch and golden test patch are downstream of the issue — the patch is supposed to *solve* the issue and the tests are supposed to *validate* the solution — noise in the issue description propagates to both. A competent human developer (or agent), reading only the issue description, would reasonably produce a patch that addresses the stated problem but would **miss** some of the changes present in the golden patch, or would be **confused** by the issue itself.

#### Subcategories:

**A1. Multi-Issue Bundle (Scope Expansion)**
The golden patch addresses multiple issues or concerns simultaneously, but the issue description only describes one. The patch includes opportunistic fixes, refactors, or related improvements beyond the stated scope.

*Example from data:* `getmoto__moto-4956` — Issue describes incorrect `describe_listeners` response format, but the golden patch modifies 12 files across `ec2`, `elb`, `elbv2`, `iam`, and internal packages (66,245 chars of patch for a 3,368-char issue).

*Example:* `getmoto__moto-5012` — Issue requests cancellation reasons for `write_transactions`, but the patch rewrites the entire DynamoDB exception hierarchy (56,621-char patch for a 1,027-char issue).

**A2. Implicit Requirements**
The golden patch implements additional behavior that is implied but not stated in the issue. This includes edge cases, input validation, error handling, or conformance to an external specification not mentioned in the issue.

*Example:* `getmoto__moto-4847` — Issue reports missing `DomainValidationOptions` on certificate describe, but the patch also changes `Serial` from int to str, adds `RenewalEligibility`, `Options`, and other fields.

*Example:* `getmoto__moto-7514` — Issue asks for gzip input support and CSV output for S3 Select, but the patch also adds bz2 compression support, error handling for unknown keys, and CSV output serialization details.

**A3. Under-specified Issue**
The issue description is extremely terse or merely a title with no elaboration, making it impossible to derive the full code change from the description alone.

*Example:* `pydantic__pydantic-5322` — Entire problem statement is "Handle frozen and extra configs for dataclasses\n\n" (49 chars). The golden patch modifies 4 files with 58 lines changed. Patch/issue ratio: 107.5x.

*Example:* `iterative__dvc-8251` — Problem statement is "`labels`/`type`/`desc` support in `dvc.yaml` file." (52 chars). 3 files changed.

*Example:* `iterative__dvc-2153` — Problem statement is a Discord link (119 chars).

**A4. Ambiguous or Misleading Issue**
The issue description is confusing to an agent because it is ambiguous, phrased as a question rather than a clear request, or provides a suggested fix approach that diverges significantly from the actual golden patch. These issues actively mislead the agent rather than merely being incomplete.

Sub-patterns:

- **Ambiguous intent**: The issue describes a symptom but multiple equally valid fixes could address it. The golden patch picks one interpretation, but the issue text does not disambiguate.

  *Example:* `python__mypy-16717` — Issue title is "Starting from version 1.0.0, a union of Generators is (sometimes) treated as if it has no ReturnType." The issue reports a regression but does not specify whether the fix should be in type inference, union handling, or generator return type resolution. The golden patch changes `get_generator_return_type` to treat `Iterator[X]` as shorthand for `Generator[X, Any, None]` — one of several plausible interpretations.

- **Question-form issues**: The issue is phrased as a question ("Is this expected behavior?", "Why does X happen?") rather than an actionable fix request. An agent may struggle to determine whether to fix, document, or investigate.

  *Example:* `python__mypy-15490` — Issue title is "Should `__class_getitem__` be considered part of the interface of Protocol classes?" This reads as a design question, not a bug report. The golden patch adds a hardcoded `EXCLUDED_PROTOCOL_ATTRIBUTES` frozenset — a specific design decision that cannot be inferred from the question alone.

  *Example:* `dask__dask-9087` — Issue title is "How do I initialize processes in `dask`'s multi-process scheduler?" This reads as a usage question, but the golden patch adds an `initializer` parameter to the multiprocessing scheduler.

- **Suggested-fix divergence**: The issue author provides a code suggestion or approach, but the golden patch implements a substantially different solution. An agent following the issue's suggestion would produce a patch that diverges from golden and likely fails the golden tests.

  *Example:* `getmoto__moto-5137` — Issue author provides a complete pytest suite with their own pagination implementation attempt (adapting from #4951). The golden patch uses an entirely different approach: the `@paginate(pagination_model=PAGINATION_MODEL)` decorator pattern instead of the user's manual pagination implementation.

  *Example:* `conan-io__conan-14727` — Issue author links to their own commit (`d554a6ad84`) implementing a `--profile` shortcut. The golden patch takes a different approach, restructuring `get_profiles_from_args` and modifying the CLI argument parser differently than the user's implementation.

  *Example:* `getmoto__moto-6006` — Issue asks about DynamoDB error deserialization compatibility with Java SDK and suggests fixing the error message format. The golden patch instead fixes a completely unrelated Glue model's `as_dict` version parameter — a total mismatch between the suggested fix domain and the actual fix.

#### Detection Methods:

| Method | Type | Description |
|--------|------|-------------|
| **LLM Entailment Check** | LLM-assisted | Prompt an LLM with the issue description and ask it to enumerate the expected changes. Then compare the enumerated changes against the actual patch diff. Changes in the patch not covered by the LLM's enumeration indicate I→P noise. |
| **Issue Completeness Score** | LLM-assisted | Prompt an LLM to rate the issue description on a scale of 1-5 for: (a) specificity of the desired behavior change, (b) whether reproduction steps are provided, (c) whether the scope of the fix is bounded. Instances scoring ≤ 2 are flagged. |
| **Issue Clarity Classifier** | LLM-assisted | Classify the issue as: `clear_request`, `ambiguous`, `question_form`, `has_divergent_suggestion`, or `under_specified`. Flag non-`clear_request` instances as A4 noise. |

---

### Category B: Patch-to-Test Misalignment (P→T Noise)

**Definition:** The golden test patch does not fully cover the core functionality changes introduced by the golden code fix patch. Some implemented behavioral changes are untested.

This matters because when tests are used as a **reward signal** for training, an agent's patch that correctly implements part of the fix may receive no positive signal if the tested subset doesn't overlap with what the agent changed, or may receive full credit while missing critical changes that happen to be untested.

#### Subcategories:

**B1. Partial Test Coverage of Core Changes**
The patch introduces multiple functional changes, but the test patch only validates a subset. Non-trivial behavioral changes remain untested.

*Example:* `getmoto__moto-5940` — Patch adds `update_table`, changes type annotations, modifies table versioning and update time logic. Test only adds 2 assertion lines checking `VersionId` inside Table dict and `UpdateTime` key existence — does not test `update_table` at all.

**B2. Untested Side Effects**
The patch changes behavior in multiple code paths, but tests only exercise the primary path.

*Example:* `getmoto__moto-6726` — Patch modifies `core/responses.py`, `core/utils.py`, and `s3/responses.py` to add gzip request decompression. Tests only cover the core response path, not S3-specific decompression behavior.

#### Detection Methods:

| Method | Type | Description |
|--------|------|-------------|
| **Patch-Test File Overlap** | Heuristic | Compare the set of source files modified by the patch against the set of modules imported or tested by the test patch. Low overlap suggests under-testing. |
| **Hunks-to-Tests Ratio** | Heuristic | Count the number of distinct code hunks in the patch vs. the number of distinct test cases in the test patch. A high hunk-to-test ratio (> 5) suggests partial coverage. |
| **LLM Coverage Analysis** | LLM-assisted | Provide the LLM with the code patch and ask it to list all behavioral changes. Then provide the test patch and ask which behavioral changes are exercised. Report untested changes. |

---

### Category C: Test-as-Feedback Signal Noise (T→F Noise)

**Definition:** The golden test patch contains assertions or test designs that are specific to the **implementation pattern** of the golden fix rather than the **functional intent** of the issue. When used as feedback for agent-generated patches, these tests produce **false negatives** (reject correct alternative implementations) or **false positives** (accept incorrect patches that happen to match surface patterns).

This is the most insidious noise category because it directly corrupts the training signal: agents may learn to mimic implementation patterns rather than solve problems.

#### Subcategories:

**C1. Hard-Coded Error Messages**
Tests assert exact error message strings that are implementation-specific. An alternative correct implementation with a different (but equally valid) error message would fail.

*Example:* Tests checking `err["Message"].should.equal("A]listener already exists on this port for the given ...load balancer ...")` — any alternative wording fails the test.

*Example:* `getmoto__moto-5012` — Test checks `.should.equal("ValidationException")` for a specific error code string.

**C2. Assertion on Internal Types or Structures**
Tests assert on the type of assertion (e.g., `assertRaises` vs. `assertWarns`), or on internal data structure shapes that are implementation decisions rather than externally observable behavior.

**C3. New Function/Method Name Coupling**
Tests call newly introduced functions or methods by their exact name as defined in the golden patch. An agent that solves the problem with a different decomposition (different function name, different module structure) would fail even if functionally correct.

*Example:* `iterative__dvc-1848` — Test imports `test_no_recursive_spawn` by name and asserts specific daemon behavior tied to the implementation's spawn-guard mechanism.

**C4. Performance / Timing Assertions**
Tests include assertions on execution time, memory usage, or ordering that are performance-dependent rather than correctness-dependent.

*Example:* `dask__dask-6779` — Tests include `test_array_store_final_order` and `test_terminal_node_backtrack` which assert on task graph ordering — a performance optimization detail rather than functional correctness.

**C5. Structural Pattern Matching**
Tests check structural properties of the output that could be satisfied by the golden implementation but not by functionally equivalent alternatives: specific field ordering, specific key names for internal representations, or specific intermediate computation results.

*Example:* `getmoto__moto-5177` — Tests check that `GrantId` and `KeyId` match specific generated values, coupling to internal ID generation.

#### Detection Methods:

| Method | Type | Description |
|--------|------|-------------|
| **String Literal Assertion Scan** | Heuristic | Parse the test patch for assertions containing string literals (error messages, format strings). Count the proportion of assertions that check exact strings vs. behavioral outcomes. |
| **New Symbol Reference Detection** | Heuristic | Extract new function/method/class names introduced in the golden code patch. Check if the test patch references these names directly. High coupling suggests C3 noise. |
| **LLM Functional vs. Implementation Classification** | LLM-assisted | For each test assertion in the test patch, prompt an LLM to classify it as: (a) testing functional intent (would pass for any correct implementation), or (b) testing implementation pattern (tied to specific coding choices). Report the fraction of implementation-coupled assertions. |
| **Alternative Implementation Test** | Execution-based | Generate an alternative patch using an LLM that solves the issue differently, then run the golden tests against it. If the alternative is functionally correct but fails the tests, this confirms T→F noise. |

---

## 3. Noise Distribution Estimation Plan

### Phase 1: Heuristic Pre-screening (All instances)

Compute the following for every instance in SWE-Gym (N ≈ 2,092 valid instances):

| Metric | Noise Category | Threshold |
|--------|---------------|-----------|

| `len(issue)` | A3 (Under-specified) | < 100 chars → flag |
| `issue_has_question_form` | A4 (Ambiguous) | contains "?" in title or first sentence → flag |
| `issue_has_code_suggestion` | A4 (Divergent suggestion) | contains code block + patch diverges → flag |
| `num_patch_hunks / num_test_cases` | B (P→T) | > 5 → flag |
| `string_literal_assertion_ratio` | C (T→F) | > 0.5 → flag |
| `new_symbol_test_coupling` | C3 (Name coupling) | any → flag |

### Phase 2: LLM-Assisted Deep Annotation (Stratified sample)

Stratified sample of ~200 instances (100 flagged by heuristics + 100 random unflagged) for LLM-assisted annotation:

1. **I→P Entailment Check**: For each instance, prompt an LLM to predict the expected scope of changes from the issue. Compare to actual patch. Label: `{full_alignment, partial_alignment, weak_alignment}`.

2. **P→T Coverage Check**: For each instance, prompt an LLM to enumerate behavioral changes in the patch and assess which are tested. Label: `{fully_tested, partially_tested, minimally_tested}`.

3. **T→F Implementation Coupling Check**: For each test assertion, classify as functional vs. implementation-specific. Compute the ratio. Label: `{low_coupling (<20%), medium (20-50%), high (>50%)}`.

### Phase 3: Distribution Estimation

Using the heuristic flags calibrated by LLM annotations, estimate population-level noise prevalence:

- **Per-category prevalence**: What fraction of instances exhibit each noise type?
- **Co-occurrence**: How often do multiple noise types co-occur in the same instance?
- **Per-repo distribution**: Do noise patterns vary by repository?
- **Severity distribution**: Within each category, what is the severity distribution?

### Full Dataset Distribution (N = 2,092 valid instances)

| Noise Signal | Count | Percentage |
|-------------|-------|------------|
| **Category A: Issue Description Noise (I→P/T)** | | |
| Under-specified issue (< 100 chars) | 30 | 1.4% |
| Any Category A flag (heuristic) | 30 | 1.4% |
| **Category B: P→T Noise** | | |
| High hunk-to-test ratio (> 5) | 449 | 21.5% |
| **Category C: T→F Noise** | | |
| String literal assertion ratio > 50% | 506 | 24.2% |
| New symbol coupling in tests | 355 | 17.0% |
| Any Category C flag | 741 | 35.4% |
| **Overall** | | |
| Any noise flag | 1,023 | 48.9% |

#### Co-occurrence

| Combination | Count | Percentage |
|------------|-------|------------|
| A ∩ B | 6 | 0.3% |
| A ∩ C | 8 | 0.4% |
| B ∩ C | 185 | 8.8% |
| A ∩ B ∩ C | 2 | 0.1% |

#### Per-Repository Distribution

| Repository | N | A-noise | B-noise | C-noise | Any |
|-----------|---|---------|---------|---------|-----|
| pandas-dev/pandas | 568 | 0% | 17% | 14% | 26% |
| Project-MONAI/MONAI | 370 | 1% | 42% | 31% | 60% |
| getmoto/moto | 326 | 0% | 15% | 76% | 78% |
| python/mypy | 252 | 0% | 22% | 13% | 29% |
| iterative/dvc | 180 | 3% | 18% | 56% | 64% |
| dask/dask | 116 | 0% | 12% | 30% | 40% |
| modin-project/modin | 82 | 18% | 26% | 23% | 49% |
| conan-io/conan | 71 | 1% | 10% | 83% | 83% |
| facebookresearch/hydra | 59 | 3% | 17% | 19% | 36% |
| pydantic/pydantic | 44 | 2% | 7% | 55% | 55% |
| bokeh/bokeh | 24 | 0% | 25% | 58% | 71% |

**Key observation:** Noise prevalence is strongly repo-dependent. `conan-io/conan` (83% any noise) and `getmoto/moto` (80%) are particularly noisy, driven primarily by C-noise (implementation-specific test assertions, 83% and 76% respectively). `pandas-dev/pandas` (40%) and `python/mypy` (40%) are the cleanest. Note that the current Category A heuristic only captures under-specified issues (< 100 chars); the A1–A4 subcategories (scope expansion, implicit requirements, ambiguous/misleading issues) require LLM-assisted detection and are expected to substantially increase A-noise prevalence.

---

## 4. Proposed Curation Strategies

Once noise is detected and quantified, potential curation approaches include:

1. **Issue Augmentation**: Use LLM to expand under-specified issues with details inferred from the patch, creating a more complete issue description for training.

2. **Patch Scoping**: Decompose multi-issue patches into the minimal subset of changes that address the stated issue. Remove orthogonal refactors and opportunistic fixes.

3. **Test Decontamination**: Rewrite implementation-specific test assertions to be implementation-agnostic. Replace hard-coded string checks with semantic checks. Replace function-name-coupled tests with behavioral probes.

4. **Instance Filtering**: Remove instances with severe noise (e.g., ratio > 50x, issue < 50 chars) from the training set.

5. **Noise-Aware Training**: Instead of binary pass/fail reward from tests, use a weighted reward that discounts implementation-specific test assertions.

---

## 5. Concrete Examples Summary Table

| Sample | Instance ID | Noise Types | Evidence |
|--------|-------------|-------------|----------|
| 6 | `getmoto__moto-5012` | A1, A2 | 1,027-char issue → 56,621-char patch (55x), 3 files, exception hierarchy rewrite |
| 7 | `getmoto__moto-4956` | A1 | 3,368-char issue → 66,245-char patch (20x), 12 files, 69,550-char test patch |
| 8 | `getmoto__moto-5177` | A1, A3 | 148-char issue → 7,407-char patch (50x), 2 files, 173 lines added |
| 30 | `pydantic__pydantic-5322` | A3 | 49-char issue → 5,267-char patch (108x), 4 files |
| 22 | `iterative__dvc-8251` | A3 | 52-char issue, 3 files |
| 21 | `iterative__dvc-2153` | A3 | 119-char issue (Discord link only, no description) |
| — | `python__mypy-15490` | A4 (question) | "Should `__class_getitem__` be considered part of Protocol interface?" — design question, not bug |
| — | `dask__dask-9087` | A4 (question) | "How do I initialize processes?" — usage question, patch adds `initializer` param |
| — | `getmoto__moto-5137` | A4 (divergent) | User provides full pagination implementation; golden uses `@paginate` decorator |
| — | `getmoto__moto-6006` | A4 (divergent) | Issue about DynamoDB Java SDK errors, patch fixes unrelated Glue model bug |
| — | `conan-io__conan-14727` | A4 (divergent) | User links own commit with approach; golden restructures differently |
| 3 | `getmoto__moto-4847` | A2, C1 | Issue about missing DomainValidationOptions, patch also fixes Serial type, adds RenewalEligibility |
| 1 | `getmoto__moto-7514` | A2, C3 | Issue asks for gzip+CSV, patch also adds bz2, error handling; tests couple to `test_bzipped_json` |
| 2 | `getmoto__moto-5940` | B1 | Patch adds `update_table` + versioning changes; test only checks 2 assertions |
| 24 | `dask__dask-6779` | C4 | Tests assert task graph ordering — performance detail |
| 45 | `Project-MONAI__MONAI-2492` | A1, A2 | 683-char issue → 15,383-char patch (22x), 5 files |
| 49 | `Project-MONAI__MONAI-1070` | A1 | 592-char issue → 7,542-char patch (13x), 3 files |

---

## Appendix: Sampled Instance Indices

50 instances sampled with `random.seed(42)` from 2,092 valid instances in the SWE-Gym train split (`output.with_completions.jsonl.gz`, Qwen3-Coder run 1). Full sample data saved in `sampled_50_instances.json`.
