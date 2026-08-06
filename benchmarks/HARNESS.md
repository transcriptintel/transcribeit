# Benchmark harness

The maintained Bun harness turns a reviewed YAML matrix into resumable,
transcript-free benchmark evidence. It uses the pinned representative corpus and
the production `transcribeit run` CLI; it does not define a second provider
implementation.

## Safety and execution policy

Matrix files declare one of three policies:

- `manual`: an operator starts the run after reviewing provider scope and cost;
- `scheduled`: an externally approved schedule may start the run;
- `local_ci`: local-only entries may run in CI and may enforce generous
  tolerances.

Any matrix containing a hosted entry requires `--allow-hosted` at runtime.
Validation rejects hosted entries in `local_ci`, and failure-enforced tolerances
outside `local_ci`. CI validates the tracked matrix and runs the deterministic
fake-provider tests, but never contacts a hosted provider.

Credentials are named only through `required_env`; values stay in the process
environment. Matrix `args` cannot override credentials, provider/model, input,
output, retry, or timeout options. Child stdout and stderr are captured only long
enough to classify the result and are never written to run state. Provider
metadata is allowlisted to capability and quality fields; transcript text,
segments, words, request IDs, signed URLs, error bodies, and local paths are not
published.

Attempt transcript and manifest files are hashed and deleted by default. Use
`--keep-attempt-outputs` only for a deliberate ignored debugging run, then remove
that exact run directory after review. The harness does not download models. A
new downloadable-model evaluation must still use an evaluation-owned directory,
record artifact revisions/hashes and disk use, and remove that exact directory
after evidence capture as required by [the benchmark protocol](README.md).

## Matrix schema

The tracked example is
[`hosted-smoke.yaml`](matrices/hosted-smoke.yaml). Every behavior-affecting field
is explicit:

- `fixture_ids` selects stable IDs from `corpus/v1/manifest.yaml`;
- `concurrency`, `retries`, and `timeout_seconds` control execution;
- each entry declares provider, model classification, `local` or `hosted`,
  `cold` or `warm`, repetitions, required environment names, and safe extra args;
- tolerances declare enforcement, relative and absolute latency margins, and a
  minimum success rate.

`cache_state` records the operational state established by the evaluator. The
harness does not pretend to flush an OS/provider cache or warm an external
server. Use separate entry IDs for cold and warm observations and establish that
state before each run.

## Commands

Validate and inspect the exact secret-free plan without running providers:

```bash
bun run scripts/benchmark_harness.ts validate benchmarks/matrices/hosted-smoke.yaml
bun run scripts/benchmark_harness.ts plan benchmarks/matrices/hosted-smoke.yaml
```

Run a reviewed hosted matrix. The run directory must be a named child of
`output/benchmarks/`:

```bash
bun run scripts/benchmark_harness.ts run \
  --matrix benchmarks/matrices/hosted-smoke.yaml \
  --run-dir output/benchmarks/hosted-smoke-YYYYMMDD \
  --binary target/release/transcribeit \
  --allow-hosted
```

`state.json` is written atomically after every status transition. Re-running the
same command skips passed, failed, and unconfigured attempts; interrupted
`running` attempts return to the queue. Failures are retried only with the
explicit `--retry-failures` option. A changed matrix, binary hash, or output
retention policy is rejected instead of being mixed into existing state.

Publish and validate an allowlisted result:

```bash
bun run scripts/benchmark_harness.ts publish \
  --state output/benchmarks/hosted-smoke-YYYYMMDD/state.json \
  --output output/benchmarks/hosted-smoke-YYYYMMDD/result.sanitized.json

bun run scripts/benchmark_harness.ts validate-result \
  output/benchmarks/hosted-smoke-YYYYMMDD/result.sanitized.json
```

Manually inspect the sanitized file before adding it to `benchmarks/results/`.
A clean commit becomes `clean_commit_benchmark`; a dirty worktree remains a
`dirty_worktree_reference` and must not be promoted as a regression baseline.

Compare two results from the same matrix:

```bash
bun run scripts/benchmark_harness.ts compare \
  --baseline benchmarks/results/baseline.sanitized.json \
  --candidate output/benchmarks/candidate/result.sanitized.json
```

Comparisons require the same matrix ID and exact matrix-definition SHA-256.
Latency violates policy only when both the relative and absolute margins are
exceeded. Hosted and manual/scheduled matrices are always report-only. A nonzero
comparison exit is possible only for a `local_ci` matrix whose explicit
`enforcement: fail` tolerance is violated.
