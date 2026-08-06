---
name: resolve-transcribeit-issue
description: Implement one tracked TranscribeIt TI-NNN issue end to end across Rust code, regression tests, provider behavior, documentation, examples, benchmarks, and the canonical issue registry. Use when the user asks to proceed with, fix, resolve, continue, restructure, or take the next TranscribeIt issue. Do not use for a review-only request unless implementation is also requested.
---

# Resolve a TranscribeIt issue

## Establish the contract

1. Inspect the branch, `git status`, and overlapping diffs. Preserve unrelated work.
2. Read the canonical entry in `docs/issues/README.md` and `docs/issues/TI-NNN.md`, then verify it against current code. Root `TODO.md` is a compatibility scratchpad only.
3. Trace affected CLI parsing, provider construction, prepared-media paths, storage/cleanup, normalization, manifest output, docs, tasks, and benchmark surfaces.
4. State boundaries that depend on an external provider, account, region, model revision, feature flag, native library, or live object store.

## Implement

- Add a focused regression that fails for the defect or missing contract whenever practical.
- Fix the shared abstraction instead of one caller when multiple providers or output paths share the rule.
- Cover negative, limit, malformed-response, concurrency, cleanup, and compatibility cases appropriate to the issue.
- Keep production modules near 300 lines. Do not grow a ratcheted oversized module past its reviewed ceiling; extract parsing, policy, persistence, tests, or orchestration first.
- Preserve bounded response handling, protocol restrictions, atomic/private persistence, cancellation, staged-resource cleanup, typed retryability, timestamp invariants, and secret redaction.
- Update public docs and runnable examples in the same change. Keep model names, parameters, defaults, limits, environment variables, capabilities, and errors identical to implementation.

Do not broaden the issue into unrelated cleanup. Add a separate `TI-NNN` candidate for a newly confirmed independent defect.

## Validate

1. Run focused tests while iterating.
2. For Rust changes, finish with `$check-transcribeit`'s default gate.
3. Add all-feature validation for Sherpa, feature, native bootstrap, or all-feature CI work.
4. Add live-provider probes only when the issue's outcome depends on current remote behavior. Reuse configured keys without exposing them and record exact model/request compatibility.
5. Validate affected docs, tasks, workflows, benchmark artifacts, and skills with their focused validators.

## Close the work

Only after required gates pass:

- update the canonical `TI-NNN` page with implementation, contract boundary, and exact validation evidence;
- regenerate and check the registry with `bun run scripts/issues_registry.ts generate` and `bun run scripts/issues_registry.ts check`;
- update root `TODO.md` compatibility notes when present;
- run `git diff --check` and review the complete issue-scoped diff;
- report files changed, evidence categories, and remaining external or deployment risks;
- do not claim a commit, push, deployment, or live-provider result that did not occur;
- propose no more than three prioritized, bounded next development goals.
