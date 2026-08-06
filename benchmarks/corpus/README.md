# Representative transcription corpus

`v1/manifest.yaml` is the immutable definition of the TranscribeIt quality
corpus. It pins every upstream object by URL, revision-bearing path, byte count,
and SHA-256. Media and expanded reference annotations are materialized under the
ignored `samples/corpus/v1/` directory; they are not committed.

Materialize and verify the corpus:

```bash
bun run scripts/corpus.ts fetch
bun run scripts/corpus.ts verify
```

`fetch` validates every downloaded source before extracting only the declared
member. By default it removes the source archives and metadata downloads after
the output hashes pass. Use `--keep-downloads` only for a deliberate local cache;
the cache remains ignored under `samples/corpus/.downloads/`.

Validate the tracked definitions without downloading media:

```bash
bun run scripts/corpus.ts check
```

## Coverage

The four stable fixtures cover clean US English, a deterministic 10 dB noisy
derivative, clean Mandarin-L1 English, and a 40-minute four-person AMI design
meeting. The AMI fixture includes manual per-speaker word and segment XML, so
timestamp, speaker-attributed accuracy, overlap, and diarization can be scored.
The short fixtures include reviewed source transcripts and non-medical terms;
the manifest records whether each capability has reference data.

`v1/scoring.yaml` keeps word accuracy, term handling, timing, speakers, metadata,
latency, and failures separate. A missing timing or speaker capability is
reported as unsupported, not converted into an artificially favorable zero
error. TI-008 can implement the runner against this contract without changing
the corpus identity.

## Rights and attribution

- FLEURS `en_us` is licensed CC BY 4.0 by Google. The clean utterance is copied
  unchanged; the noisy fixture is identified as a deterministic modification.
- SpeechOcean762 is licensed CC BY 4.0. The selected utterance is attributed to
  Junbo Zhang and contributors and remains linked to OpenSLR resource 101.
- The AMI Meeting Corpus and public manual annotations are licensed CC BY 4.0
  by the AMI Consortium.

Keep these attributions, source links, license identifiers, and modification
notices with any redistributed fixture. The corpus intentionally excludes the
private interview used for earlier exploratory benchmarks.
