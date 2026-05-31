#!/usr/bin/env bash
set -euo pipefail

cargo fmt --all -- --check
cargo test --all-targets
cargo clippy --all-targets -- -D warnings

