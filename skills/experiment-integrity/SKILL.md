# Experiment Integrity and Code Reuse Skill

## Purpose

Use this skill whenever reviewing, modifying, debugging, testing, rerunning, or publishing an experimental pipeline.

The goals are to:

* reuse existing trusted code
* prevent old implementations from being used again
* prevent stale outputs from being silently reused
* preserve reproducibility and provenance
* require validation before formal execution
* document every code change
* commit and push every completed modification to GitHub

---

## 1. Repository Preflight

Before changing or running code, inspect the repository state:

```bash
git rev-parse --show-toplevel
git branch --show-current
git rev-parse HEAD
git status --short
git remote -v
git log -n 10 --oneline
git diff
git diff --cached
```

When remote access is available:

```bash
git fetch --all --prune
git status -sb
```

Confirm:

* repository root
* current branch and commit
* working-tree state
* remote tracking state
* whether relevant fixes exist only locally
* whether the executing code matches GitHub

Do not assume local code, remote code, imported code, and the code that generated existing outputs are identical.

A formal run should not silently use a dirty working tree.

---

## 2. Read Previous Debug Records

Before modifying code:

1. Read relevant entries in `DEBUG_CHANGELOG.md`.
2. Identify previously fixed bugs related to the task.
3. Search the repository for the old buggy patterns.
4. Check whether any formal, legacy, debug, ablation, notebook, provider, or launcher path still contains them.
5. Check whether outputs created by old implementations can still be resumed.

A previous fix is incomplete if another reachable code path still contains the bug.

---

## 3. Inspect Before Implementing

Before writing new code:

1. Identify the actual formal entrypoint.
2. Trace the execution path:

```text
launcher
→ entrypoint
→ imported pipeline
→ generator
→ transformation
→ evaluator
→ metric
→ aggregation
```

3. Search for existing implementations.
4. Prefer extending or importing trusted code.
5. Do not create duplicate implementations without necessity.

Important behavior should normally have one authoritative implementation:

* data generation
* hashing
* seed generation
* pairing validation
* provenance validation
* resume validation
* metrics
* aggregation

If a new implementation is necessary, document why existing code cannot be reused.

---

## 4. Stale-Code Prevention

Search all relevant paths before applying a fix.

Example:

```bash
rg -n "function_name|class_name|config_field"
rg -n "resume|completed|skip existing"
rg -n "canonical_sha256|pairing_hash"
rg -n "old_bug_pattern|new_fixed_pattern"
```

A fix applied to one file is not enough when another executable file contains copied logic.

Legacy code may remain only when it is:

* clearly labeled
* unreachable from formal pipelines
* excluded from formal aggregation
* documented in `DEBUG_CHANGELOG.md`

---

## 5. Provenance Requirements

Each formal sample or run should record applicable fields:

```text
run_id
sample_id
source identifier
input SHA
output SHA
sample seed
configuration SHA
model ID and revision
scheduler and dtype
pipeline version
entrypoint path and SHA
Git branch and commit
working-tree status
creation time
```

Provenance must be sufficient to answer:

* Which source produced this output?
* Which code version was used?
* Which configuration was used?
* Can this sample be independently reproduced?
* Is this output compatible with the current pipeline?

---

## 6. Canonical Metadata Hashing

Never hash:

* `str(dict)`
* unordered dictionaries
* uncontrolled CSV strings
* inconsistent path, integer, float, boolean, or null representations

Use one shared canonicalization function before hashing.

Requirements:

* dictionary keys are sorted
* expected field types are normalized
* paths use one representation
* NaN and Inf are rejected
* JSON serialization is deterministic

The same implementation must be used:

* before metadata is written
* after metadata is reloaded
* during resume validation
* during aggregation

Required invariant:

```text
hash before serialization
==
hash after serialization and reload
```

A mismatch must stop the pipeline.

---

## 7. Randomness and Shared State

Formal experiments should use explicit deterministic per-sample seeds.

Example:

```python
sample_seed = base_seed + run_id
```

Each sample should be independent of:

* processing order
* previous samples
* batch history
* resume position
* hidden global RNG state

Review variables created outside sample loops:

```python
shared_input = create_input()

for sample in samples:
    process(sample, shared_input)
```

Confirm that reuse is intentional. Otherwise create or derive the state separately for each sample.

---

## 8. Resume Safety

Never resume based only on:

* run ID
* file existence
* CSV row existence
* directory existence
* completed flag

Before skipping an existing sample, verify applicable fields:

```text
source SHA
source metadata SHA
sample seed
configuration SHA
model revision
code commit
entrypoint SHA
pipeline version
output SHA
```

If any required value differs:

```text
do not skip
do not append to the old run
do not mix old and new results
```

Recompute the sample or use a new output directory.

---

## 9. Output Management

Formal reruns after a code, data, or protocol change should use a new immutable directory:

```text
outputs/<experiment>/<dataset>/<timestamp>/
```

Separate outputs into categories such as:

```text
formal/
smoke/
debug/
ablation/
legacy/
invalid/
```

Outputs invalidated by a bug must be:

* marked or moved to `invalid/` or `legacy/`
* accompanied by an invalidation reason
* excluded from resume
* excluded from aggregation
* excluded from formal reports

Preserve useful logs and provenance as audit evidence.

---

## 10. Fail-Closed Validation

Formal execution must stop when:

* required provenance is missing
* input or output SHA differs
* configuration hashes differ
* metadata round-trip hashes differ
* duplicate inputs appear unexpectedly
* model revision is unknown
* sample count differs
* required files are missing
* unexpected files are included
* NaN or Inf appears
* resume compatibility cannot be proven
* code revision differs from the recorded version

Do not weaken a correctly functioning fail-closed gate merely to continue an experiment.

---

## 11. Required Tests

Before a full formal run, execute applicable tests.

### Unit or regression tests

Test:

* deterministic seed behavior
* canonical metadata round trip
* file and configuration hashing
* duplicate-input detection
* resume mismatch rejection
* stale-output rejection
* historical bug regression

### Smoke test

Use at least two samples and verify:

* sample seeds differ
* independent sample hashes differ
* metadata hashes survive reload
* configuration hashes match
* output hashes exist
* no NaN or Inf occurs

### Negative test

Modify an important field such as:

* seed
* source SHA
* config SHA
* code commit
* output SHA

The pipeline must stop.

---

## 12. Mandatory `DEBUG_CHANGELOG.md` Update

Every code modification must update `DEBUG_CHANGELOG.md` in the same commit.

Use this format:

```markdown
## YYYY-MM-DD — Change title

### Problem
What failed and why it matters.

### Root cause
The exact implementation, stale-code, provenance, or resume issue.

### Affected files
- `path/to/file.py:function`

### Affected outputs
Invalid, legacy, incomplete, or contaminated outputs.

### Fix
The implemented change.

### Reused code
Existing trusted helpers reused.

### Historical bug coverage
Previous entries reviewed, old patterns searched, and stale copies handled.

### Regression prevention
Assertions, fail-closed gates, tests, and validation.

### Validation
Commands and test results.

### Git provenance
- Repository:
- Branch:
- Commit:
- Remote branch:
- Push status:
- Entry point:
- Formal output eligibility:
```

A code change without a corresponding `DEBUG_CHANGELOG.md` entry is incomplete.

---

## 13. Commit and Push

After validation:

```bash
git status --short
git diff --check
git diff

git add <relevant files only>

git diff --cached --stat
git diff --cached

git commit -m "<clear change description>"
git push
```

Verify the remote publication:

```bash
git rev-parse HEAD
git status -sb
git log -1 --oneline
git ls-remote origin <branch>
```

If push fails:

* report the exact error
* keep the local commit
* do not claim GitHub was updated
* classify the work as locally committed but not published

Never say “fixed” when the change exists only in the working tree.

Never say “pushed” unless the remote commit has been verified.

---

## 14. Completion Gate

A modification is complete only when:

```text
repository and remote inspected
+
DEBUG_CHANGELOG.md reviewed
+
existing code searched
+
authoritative implementation identified
+
reachable stale copies handled
+
fix implemented
+
tests passed
+
provenance and resume gates passed
+
affected outputs separated
+
DEBUG_CHANGELOG.md updated
+
commit created
+
commit pushed
+
remote commit verified
```

---

## 15. Final Report

After modifying code, report:

### Changed

* files and functions
* authoritative implementation
* existing code reused

### Historical coverage

* relevant `DEBUG_CHANGELOG.md` entries
* old patterns searched
* remaining legacy paths

### Validation

* unit tests
* smoke tests
* negative tests
* provenance and resume checks

### Outputs

* valid
* invalid
* legacy
* requiring recomputation

### GitHub

* repository
* branch
* commit SHA
* remote branch
* push status
* remote verification
* `DEBUG_CHANGELOG.md` entry location
