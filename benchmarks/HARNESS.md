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

Both `local` and `apple-speech` are local execution providers. Local whisper.cpp
entries receive the matrix `model` through `--model`; Apple Speech uses its
system-managed transcriber and receives no `--model` or `--remote-model` option.
An Apple Speech entry must include an explicit, non-`auto` `--language` argument.

Credentials are named only through `required_env`. Each child receives only
allowlisted system variables and that entry's required environment values; other
ambient provider credentials are withheld. A private, empty `.env` in the run
directory prevents `dotenvy` from walking up to the repository `.env`. Matrix
`args` are limited to a reviewed grammar and cannot override credentials,
provider/model, input, output, retry, or timeout options. Child stdout is
discarded, while stderr is drained into a bounded in-memory head/tail buffer for
classification and Darwin RSS parsing; neither stream is written to run state.
Provider metadata is allowlisted to capability and quality fields; transcript
text, segments, words, request IDs, signed URLs, error bodies, and local paths are
not published.

Attempt transcript and manifest files are read only long enough to collect
allowlisted metrics and hashes, then deleted by default before publication. When
`reference_scoring` is enabled, normalization, WER, term recall, timing alignment,
and explicit unsupported-capability states are computed in memory. Overlapping
multi-speaker references are not serialized into ordinary WER or boundary MAE;
both metrics report `unsupported_overlapping_reference`. Non-overlapping boundary
MAE requires at least 50% exact-word timestamp alignment. Transcript, segment,
and word text are not written to run state or a published result. Use
`--keep-attempt-outputs` only for a deliberate ignored debugging run, then remove
that exact run directory after review.

The harness does not download or delete models. A downloadable model must use a
new evaluation-owned directory rather than a shared cache. Record its initial
size, pinned revision/hash/bytes, and disk state; after run evidence is captured,
including after a failed or cancelled run, remove that exact model directory and
verify its absence. Publication of a matrix with an `evaluation_download`
artifact requires the cleanup record described below. A macOS-managed Apple
Speech asset is recorded as shared system state and must not be deleted or
claimed as evaluation-owned.

## Matrix schema

The tracked example is
[`hosted-smoke.yaml`](matrices/hosted-smoke.yaml). Every behavior-affecting field
is explicit:

- `fixture_ids` selects stable IDs from `corpus/v1/manifest.yaml`;
- `attempt_order` is either `entry-fixture-repetition` or
  `fixture-repetition-entry`; the latter
  deterministically interleaves providers for each fixture and repetition;
- `concurrency`, `retries`, and `timeout_seconds` control execution;
  `timeout_seconds` is passed as the provider request timeout and enforced as a
  hard child-process wall;
- `measurements.reference_scoring` enables transcript-free scoring against the
  reviewed corpus, while `measurements.peak_rss` captures
  `peak_rss_bytes` through Darwin `/usr/bin/time -l`;
- each entry declares provider, model classification, `local` or `hosted`,
  `cold` or `warm`, repetitions, required environment names, safe extra args, and
  an optional artifact lifecycle and pinned identity;
- tolerances declare enforcement, relative and absolute latency margins, and a
  minimum success rate.

`cache_state` records the operational state established by the evaluator. The
harness does not pretend to flush an OS/provider cache or warm an external
server. Use separate entry IDs for cold and warm observations and establish that
state before each run.

The tracked [`ti-014-apple-large-v3.yaml`](matrices/ti-014-apple-large-v3.yaml)
matrix is the local comparison example. It runs Apple Speech and pinned local
whisper.cpp large-v3 sequentially, but alternates the two entries for every
fixture repetition. Its three repetitions across four fixtures produce 24
attempts. The Apple asset and the hashed local model are both classified as
operationally warm; the harness does not clear macOS assets or OS filesystem
caches to manufacture a cold result.

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

For TI-014, first materialize and verify the corpus, build the release binary,
and place large-v3 in a new, narrowly named evaluation directory. Keep that model
directory separate from the resumable run directory so it can be removed before
publication:

```bash
bun run scripts/corpus.ts verify
cargo build --release --locked

MODEL_CACHE_DIR=output/benchmarks/ti-014-large-v3-evaluation \
  target/release/transcribeit download-model \
  --model-size large-v3 \
  --output-dir output/benchmarks/ti-014-large-v3-evaluation

MODEL_CACHE_DIR=output/benchmarks/ti-014-large-v3-evaluation \
  bun run scripts/benchmark_harness.ts run \
  --matrix benchmarks/matrices/ti-014-apple-large-v3.yaml \
  --run-dir output/benchmarks/ti-014-run-YYYYMMDD \
  --binary target/release/transcribeit
```

Before downloading, verify that the exact model directory does not exist and
record its initial byte count and free space in an ignored operator log. Free
space is deliberately not part of the sanitized result allowlist. Before
deleting the model, verify its downloaded byte count and SHA-256 against the
matrix. After evidence capture, remove only that exact directory, verify that it
is absent, and write an owner-readable cleanup record. The following fail-fast
procedure requires the target to be the canonical, non-symlink direct child that
was reserved for this evaluation:

```bash
set -euo pipefail
ti014_benchmark_root="$PWD/output/benchmarks"
ti014_model_root="$PWD/output/benchmarks/ti-014-large-v3-evaluation"
test -d "$ti014_benchmark_root"
test ! -L "$ti014_benchmark_root"
test "$(realpath "$ti014_benchmark_root")" = "$ti014_benchmark_root"
test "$(dirname "$ti014_model_root")" = "$ti014_benchmark_root"
test -d "$ti014_model_root"
test ! -L "$ti014_model_root"
test "$(realpath "$ti014_model_root")" = "$ti014_model_root"
find "$ti014_model_root" -depth -delete
test ! -e "$ti014_model_root"
```

Then record the observed cleanup outcome:

```json
{
  "schema_version": "transcribeit.downloaded-model-cleanup.v1",
  "matrix_sha256": "<matrix_sha256 from state.json>",
  "evaluation_root_sha256": "<evaluation_root_sha256 from state.json>",
  "artifact_sha256s": [
    "64d182b440b98d5203c4f9bd541544d84c605196c4f7b845dfa11fb23594d1e2"
  ],
  "recorded_at_utc": "<UTC timestamp after cleanup>",
  "evaluation_root_preexisting": false,
  "initial_bytes": 0,
  "downloaded_bytes": 3095033483,
  "reclaimed_bytes": 3095033483,
  "cleanup_completed": true,
  "evaluation_root_absent_after_cleanup": true
}
```

The matrix and root hashes come from this run's `state.json`; artifact hashes must
exactly match the sorted evaluation-download artifacts in the matrix. The cleanup
timestamp must be no earlier than the final run-state update and no later than
publication. Numeric values must be observed rather than copied from this
example. Never use a broad path, glob, unresolved variable, shared cache, or the
repository root as the cleanup target.

`state.json` is written atomically after every status transition. An owner-only
run lock excludes concurrent writers for the full invocation; a well-formed lock
whose process has exited is recovered only when state proves that no attempt is
still marked `running`. A live or invalid lock, unreadable state, or stale lock
paired with a `running` attempt requires explicit operator intervention and is
never stolen. Re-running the same command skips passed, failed, and unconfigured
attempts; interrupted active attempts return to the queue. On POSIX, timeout or
cancellation kills the detached process group, and cancellation clears the
active attempt and removes its output before the harness exits. Downloaded-model
cleanup remains the operator's responsibility after cancellation. Failures are
retried only with the explicit `--retry-failures` option. A changed matrix,
attempt plan, binary hash, evaluation-root identity, repository HEAD/worktree
fingerprint, captured CPU/OS/tool environment, or output retention policy is
rejected instead of being mixed into existing state.

Publish and validate an allowlisted result:

```bash
bun run scripts/benchmark_harness.ts publish \
  --state output/benchmarks/hosted-smoke-YYYYMMDD/state.json \
  --output output/benchmarks/hosted-smoke-YYYYMMDD/result.sanitized.json

bun run scripts/benchmark_harness.ts validate-result \
  output/benchmarks/hosted-smoke-YYYYMMDD/result.sanitized.json
```

For a matrix containing an `evaluation_download` artifact, publication requires
the completed cleanup record:

```bash
bun run scripts/benchmark_harness.ts publish \
  --state output/benchmarks/ti-014-run-YYYYMMDD/state.json \
  --cleanup-record output/benchmarks/ti-014-run-YYYYMMDD/model-cleanup.json \
  --cleanup-root output/benchmarks/ti-014-large-v3-evaluation \
  --output output/benchmarks/ti-014-run-YYYYMMDD/result.sanitized.json

bun run scripts/benchmark_harness.ts validate-result \
  output/benchmarks/ti-014-run-YYYYMMDD/result.sanitized.json
```

Publication records that normal attempt outputs were already removed. A run made
with `--keep-attempt-outputs` cannot be published as having completed that cleanup;
remove its exact ignored run directory after deliberate debugging instead.

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
