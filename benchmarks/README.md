# Reproducible benchmark records

Tracked benchmark records contain the environment, exact redacted command,
fixture identity, result metrics, and sanitized raw provider/manifest fields needed
to compare a later run. Source media and unsanitized transcripts remain outside
Git.

Quality benchmarks use the versioned representative corpus in
[`corpus/`](./corpus/README.md). Run `bun run scripts/corpus.ts fetch` to
materialize its ignored media and reviewed references, then
`bun run scripts/corpus.ts verify` before a benchmark. The tracked manifest pins
fixture identity, rights, coverage, and hashes; `corpus/v1/scoring.yaml` defines
the separate accuracy, terms, timing, speaker, metadata, latency, and failure
measurements.

Use the maintained [Bun/YAML benchmark harness](HARNESS.md) for new matrices.
It validates deterministic fixture selection, records local/hosted and warm/cold
state, supports deterministic provider interleaving, resumes atomically, preserves
individual failures, deletes attempt output before publication by default, and
publishes an allowlisted result schema. Reviewed-reference scoring is computed in
memory without retaining transcript text. Darwin runs can also record peak
process RSS. Hosted execution always requires an explicit `--allow-hosted` opt-in
and remains report-only.

Apple Speech is a local execution provider but does not receive a model option;
its matrix entry must declare an explicit locale. Local whisper.cpp receives its
model through `--model`. The tracked TI-014 matrix demonstrates the paired,
sequentially interleaved form of this comparison.

Before a run, capture the environment and fixture identity:

```bash
./scripts/benchmark-metadata.sh path/to/fixture > output/benchmark-environment.json
```

Run the exact command recorded with the result. Preserve the complete local output
under the ignored `output/` directory, then sanitize a manifest before considering
it for `benchmarks/results/`:

```bash
jq -f benchmarks/sanitize-manifest.jq \
  output/<run>/<fixture>.manifest.json \
  > output/<run>/<fixture>.sanitized.json
```

Review sanitized JSON manually before tracking it. In particular, remove source
text, speaker names, local paths, signed URLs, provider request identifiers, and
any error body that could contain submitted data. A reproducible result must state
whether the worktree was dirty; only clean-commit runs should become regression
baselines. Dirty-worktree measurements can remain compatibility/reference records.

For every downloadable model evaluation, prefer a new evaluation-owned directory
instead of a shared cache. Record whether that exact directory existed before the
run, its initial and downloaded byte counts, model revision and hashes, and free
disk space. Keep free-space observations in an ignored operator log because they
are not part of the sanitized result allowlist. After the run succeeds, fails, or
is cancelled, remove only artifacts created in that directory, verify that the
directory is absent, and record bytes reclaimed. Never clean a pre-existing
cache, a broad path or glob, or a target derived from an unresolved variable. A
deliberately retained artifact needs a private exact path plus a documented
owner, reason, and expiry.

When a matrix entry declares `artifact.lifecycle: evaluation_download`, the
harness refuses publication without a validated
`transcribeit.downloaded-model-cleanup.v1` record supplied through both
`publish --cleanup-record` and `publish --cleanup-root`. The latter must name the
now-absent, run-bound direct child of `output/benchmarks/`. See
[the harness cleanup protocol](HARNESS.md) for the exact schema and command.
`system_managed` Apple Speech assets remain owned and shared by macOS; never
delete them as benchmark cleanup.

Each tracked record must include:

- producing commit and dirty/clean state;
- CPU, logical-core count, RAM, OS/kernel, Rust, FFmpeg, and architecture;
- exact command with secrets represented only by environment-variable names;
- fixture duration, byte size, codec/container, and SHA-256;
- provider/model or local model artifact identity;
- warm/cold state, wall time, processing time, RTF, retries, and output shape;
- hashes of untracked transcript/raw artifacts when the artifacts themselves
  cannot be tracked safely;
- downloaded-model cleanup status, exact evaluation scope, and bytes reclaimed,
  or `not_applicable` for providers that did not download local artifacts. For
  `apple-speech`, record whether macOS requested a shared locale-asset install
  and whether AVAudioFile used the original file or required the WAV fallback;
  do not delete or claim ownership of a system-managed asset. The harness removes
  transcript-bearing attempt directories, while the operator separately removes
  only the evaluation-owned model directory. The shared public corpus remains in
  place; remove the ignored run directory only after its sanitized result has
  been copied and validated.

For a matrix record, add a structural assertion alongside manual sanitization
review. For example, the TI-007 representative-corpus reference is checked with:

```bash
jq -e '
  .schema_version == "transcribeit.benchmark-result.v1" and
  .sanitization.secrets_removed == true and
  .sanitization.transcript_text_removed == true and
  (.providers | length) == 10 and
  ([.providers[].attempts] | add) == 100 and
  ([.providers[].successful] | add) == 99 and
  ([.providers[].failed] | add) == 1 and
  .downloaded_model_cleanup.evaluation_root_absent_after_cleanup == true
' benchmarks/results/2026-08-06-ti-007-representative-corpus.sanitized.json
```

The narrower TI-009 live-account smoke record uses a dedicated schema because it
intentionally retains only status, latency, classifications, and sanitized error
categories:

```bash
jq -e '
  .schema_version == "transcribeit.live-provider-smoke.v1" and
  .producing_commit.worktree_dirty == false and
  (.providers | length) == 6 and
  ([.providers[] | select(.configured == true)] | length) == 6 and
  ([.providers[] | select(.status == "passed")] | length) == 6 and
  ([.providers[] | select(.status == "failed")] | length) == 0 and
  ([.providers[] | select(.status == "skipped")] | length) == 0 and
  .failure_semantics_probe.observed_exit == "nonzero" and
  .temporary_local_artifact_cleanup.evaluation_root_absent_after_cleanup == true and
  .sanitization.response_bodies_discarded == true and
  .sanitization.transcript_text_removed == true
' benchmarks/results/2026-08-06-ti-009-live-provider-smoke.sanitized.json
```

The TI-014 private-fixture Apple/local comparison is exploratory rather than a
baseline and uses a transcript-free schema:

```bash
jq -e '
  .schema_version == "transcribeit.exploratory-comparison.v1" and
  .classification == "dirty_worktree_exploratory_private_fixture" and
  .baseline_eligible == false and
  (.providers | length) == 2 and
  ([.providers[].wall_seconds | length] | add) == 4 and
  .comparison.accuracy_conclusion == "unsupported_without_reviewed_reference" and
  .downloaded_model_cleanup.cleanup_completed == true and
  .downloaded_model_cleanup.evaluation_root_absent_after_cleanup == true and
  .sanitization.transcript_text_removed == true and
  .sanitization.local_paths_removed == true and
  .sanitization.raw_transcripts_and_manifests_deleted == true
' benchmarks/results/2026-08-17-ti-014-apple-vs-large-v3-exploratory.sanitized.json
```
