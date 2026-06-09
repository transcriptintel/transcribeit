# Repository Instructions

Always answer in English.

## Branching

- Work on `develop` or short-lived feature branches.
- Do not work directly on `main`.
- Dependency update PRs must target `develop`.
- Keep the repository branch set clean: only `develop` and `main` should remain after processing merged branches.

## Dependency Updates

- Renovate is the only dependency updater for this repository.
- Before processing dependency updates, check for open PRs and stale branches.
- Do not merge dependency updates unless formatting, tests, and Clippy pass.

## Rust Checks

Run the shared repository check before handing off Rust changes:

```bash
./scripts/check.sh
```

The required Clippy command is:

```bash
cargo clippy --all-targets -- -D warnings
```

