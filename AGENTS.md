# TranscribeIt repository guidance

## Scope and source of truth

- TranscribeIt is a Rust CLI for local, hosted, and OpenAI-compatible speech-to-text providers.
- Treat `src/`, tests, `README.md`, `docs/`, `benchmarks/`, `Taskfile.yaml`, and CI as one product surface. Provider names, defaults, limits, capabilities, environment variables, output schemas, and failure behavior must stay consistent across them.
- Use `docs/issues/README.md` as the canonical tracked engineering registry and `docs/issues/TI-NNN.md` as the source of truth for each finding. Root `TODO.md` and `BENCHMARKS.local.md` are compatibility scratchpads, not public contracts.
- Inspect `git status` and relevant overlapping diffs before editing. Preserve all user-owned and unrelated changes.
- Work in English. Do not commit, push, tag, publish, merge, close remote work, or alter releases unless the user explicitly asks.

## Repository skills

- Use `$check-transcribeit` for focused or complete repository validation.
- Use `$resolve-transcribeit-issue` when implementing a tracked issue or backlog item end to end.
- Use `$benchmark-transcribeit` for provider, model, quality, latency, or compatibility benchmarks.
- After changing repository skills, run `bun run scripts/validate_skills.ts`, which uses Bun's native YAML parser.
- After changing issue pages or metadata, run `bun run scripts/issues_registry.ts generate` and then `bun run scripts/issues_registry.ts check`.

## Branching and dependencies

- Work on `develop` or short-lived feature branches. Do not work directly on `main`.
- Dependency update PRs must target `develop`. Renovate is the only dependency updater for this repository.
- Before processing dependency updates, inspect open PRs and stale branches. Do not merge dependency changes unless formatting, tests, strict Clippy, and the security audit pass.
- Keep only `develop` and `main` after merged temporary branches are fully processed.
- Avoid new dependencies when the standard library or an existing dependency is sufficient.

## Rust design and security

- Keep production modules focused and normally at or below 300 physical lines. New production modules must not exceed 400 lines.
- Existing oversized modules are ratcheted at reviewed ceilings. Run `python3 -B scripts/check_module_size.py` after Rust changes; do not add or widen an exception merely to pass the gate. Split responsibilities first.
- Prefer small modules, typed errors and state transitions, explicit ownership, bounded concurrency, and atomic persistence over large coordinators or implicit side effects.
- Keep blocking filesystem, archive, media-probe, and subprocess work from monopolizing async runtime threads. Preserve cancellation and cleanup boundaries around uploaded or staged resources.
- Treat media containers, playlists, paths, URLs, provider responses, SSE frames, archives, cache files, manifests, and environment configuration as untrusted input. Enforce size and protocol limits before materialization.
- Keep credentials, signed URLs, transcript contents, request identifiers, and local paths out of logs, errors, tracked benchmark artifacts, and generated help.

## Provider credentials and runtime checks

- Existing configured API keys may be reused for provider tests and benchmarks without per-run confirmation.
- Never print, copy into tracked files, or otherwise expose API key values. Prefer provider environment variables or owner-readable key files over argv secrets.
- If a required key is missing, ask the user to obtain or provide it through the ignored environment-file workflow.
- Distinguish model capability from provider-family capability. Record unsupported formats and compatibility failures as results; do not silently substitute a different model or format.
- Live provider checks may incur cost and prove only the tested account, region, model revision, request shape, and fixture. Keep that boundary explicit.

## Documentation and benchmarks

- When behavior changes, update the nearest CLI, provider, architecture, troubleshooting, README, benchmark, and example surfaces that describe it.
- Keep documented commands runnable. Use provider-specific environment variables and avoid examples that expose response bodies containing transcript or request data.
- Keep historical observations dated and separate from current runtime guarantees.
- Publish benchmark results only through the tracked protocol in `benchmarks/README.md`. Record commit/dirty state, machine and tool versions, exact commands, fixture hashes, run classification, retry/warm-state details, and sanitized outputs.
- Never treat one fixture, a generated summary, or provider self-reported metadata as transcript-quality ground truth.
- When evaluating a downloadable model, use an explicit evaluation-owned directory when possible and record its pre-run disk usage. After artifact hashes and benchmark evidence are captured, remove only the model files and directories created by that evaluation, verify the reclaimed space, and record the cleanup outcome. Apply the same cleanup after failed or cancelled runs. Never remove a pre-existing shared cache or use a broad path, glob, or unresolved variable as the cleanup target; if an artifact must be retained, record its owner, reason, and expiry.

## Validation

- During implementation, run focused tests and validators for the behavior changed, including negative, boundary, cleanup, and concurrency cases where appropriate.
- Every Rust change finishes with `./scripts/check.sh`, which includes formatting, repository-policy and hook tests, the module-size ratchet, skill and issue-registry validation, all-target tests, and exact strict Clippy.
- Run focused feature-boundary validation if optional Cargo features are introduced again.
- Run `cargo test --doc` after public Rust API documentation changes.
- Run `cargo audit --deny warnings` after dependency or security-sensitive changes.
- Run `actionlint .github/workflows/*.yml` after workflow changes when `actionlint` is available.
- For docs, tasks, scripts, benchmarks, or skills without Rust changes, run only the relevant validators. Do not describe an unavailable tool or skipped live-provider check as passing.

## Completion and review

- Reconcile implementation, regression coverage, documentation, benchmark evidence, and the canonical `TI-NNN` issue page before marking work complete; update local compatibility ledgers when present.
- Distinguish focused, default, all-feature, live-provider, and deployed evidence. Do not claim one as proof of another.
- In review, prioritize correctness, security, cleanup, bounded resource use, provider-contract parity, documentation drift, and missing regressions over style-only comments.
- After completing a goal, propose no more than three prioritized, bounded next development goals.
