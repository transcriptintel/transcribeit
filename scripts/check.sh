#!/usr/bin/env bash
set -euo pipefail

cargo fmt --all -- --check
python3 -B -m unittest discover -s scripts/tests -p 'test_*.py' -v
python3 -B -m unittest discover -s .codex/hooks/tests -p 'test_*.py' -v
bun test scripts/tests/*.test.ts
python3 -B scripts/check_module_size.py
bun run scripts/validate_skills.ts
bun run scripts/issues_registry.ts check
bun run scripts/corpus.ts check
bun run scripts/benchmark_harness.ts validate benchmarks/matrices/hosted-smoke.yaml
cargo test --all-targets
cargo clippy --all-targets -- -D warnings
