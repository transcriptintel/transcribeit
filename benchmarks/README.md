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
disk space. After the run succeeds, fails, or is cancelled, remove only artifacts
created in that directory, verify that the directory is absent, and record bytes
reclaimed. Never clean a pre-existing cache, a broad path or glob, or a target
derived from an unresolved variable. A deliberately retained artifact needs a
private exact path plus a documented owner, reason, and expiry.

Each tracked record must include:

- producing commit and dirty/clean state;
- CPU, logical-core count, RAM, OS/kernel, Rust, FFmpeg, and architecture;
- exact command with secrets represented only by environment-variable names;
- fixture duration, byte size, codec/container, and SHA-256;
- provider/model or local model artifact identity;
- warm/cold state, wall time, processing time, RTF, retries, and output shape;
- hashes of untracked transcript/raw artifacts when the artifacts themselves
  cannot be tracked safely.
- downloaded-model cleanup status, exact evaluation scope, and bytes reclaimed,
  or `not_applicable` for providers that did not download local artifacts.

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
