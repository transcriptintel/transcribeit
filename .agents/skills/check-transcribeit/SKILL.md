---
name: check-transcribeit
description: Validate TranscribeIt Rust changes, repository policy, documentation, workflows, skills, benchmarks, optional Sherpa features, and live provider smoke checks. Use when the user asks to check, test, lint, verify, audit, diagnose a build, or confirm that TranscribeIt work is ready. Do not modify failures unless the user also asks for fixes.
---

# Check TranscribeIt

Run from the repository root. Inspect the worktree first and preserve unrelated changes.

## Select the gate

- For a diagnosis-only request, reproduce and explain the failure without editing.
- During Rust implementation, run focused tests for the changed behavior. Finish every Rust change with `./scripts/check.sh`.
- Add the all-feature gate when Sherpa, feature boundaries, native-library setup, or all-feature CI changes.
- For docs, Taskfile, scripts, workflows, skills, or benchmark artifacts without Rust changes, run only the relevant validators below.
- Treat live provider requests as a separate evidence class. Reuse configured keys, but never print secrets, signed URLs, response bodies, transcript text, or request identifiers.

## Default Rust gate

Run:

```bash
./scripts/check.sh
```

The script is authoritative and must include:

1. `cargo fmt --all -- --check`
2. Repository-policy and Codex-hook unit tests under `scripts/tests` and `.codex/hooks/tests`
3. `python3 -B scripts/check_module_size.py`
4. `bun run scripts/validate_skills.ts`
5. `bun run scripts/issues_registry.ts check`
6. `cargo test --all-targets`
7. `cargo clippy --all-targets -- -D warnings`

Stop on failure and report the failing command and actionable output.

## Additional gates

- All features: `cargo test --all-targets --all-features` and `cargo clippy --all-targets --all-features -- -D warnings`.
- Public Rust docs: `cargo test --doc`.
- Dependencies or security-sensitive code: `cargo audit --deny warnings`.
- GitHub workflows: `actionlint .github/workflows/*.yml` when available.
- Shell scripts: `bash -n scripts/*.sh` and `shellcheck scripts/*.sh` when available.
- Taskfile: `task --list-all` plus a focused non-secret runtime probe when the task changed.
- Repository skills: `bun run scripts/validate_skills.ts`.
- Issue pages: regenerate with `bun run scripts/issues_registry.ts generate`, then run the matching `check`.
- Tracked benchmark JSON: run the sanitizer and schema/result assertions documented in `benchmarks/README.md`.
- Documentation links: validate local links across `README.md`, `docs/`, and `benchmarks/`.

## Runtime dependencies

FFmpeg-dependent tests are required, not optional. Confirm `ffmpeg` and `ffprobe` are installed rather than interpreting a missing binary as a pass. For Sherpa-enabled validation, bootstrap the pinned native libraries into an explicit temporary root and set the emitted absolute library path.

## Report

Report each relevant gate as pass, fail, skipped, or not applicable. Separate focused, default, all-feature, live-provider, and deployed evidence. Include test counts when available, and never describe a skipped provider or native-library check as passed.
