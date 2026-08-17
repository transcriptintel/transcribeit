import { chmod, mkdir, rm, symlink, unlink } from "node:fs/promises";
import { join } from "node:path";

import { repositoryRoot } from "../benchmark_harness/config";
import { resolveDirectBenchmarkChild } from "../benchmark_harness/paths";
import { buildAttempts, resetAttemptOutcome, runMatrix } from "../benchmark_harness/runner";
import { sha256, writeJsonAtomic } from "../benchmark_harness/state";
import type { AttemptRecord, BenchmarkMatrix, FixtureIdentity } from "../benchmark_harness/types";
import type { CorpusManifest } from "../corpus";

const repository = repositoryRoot();
const fakeBinary = join(repository, "scripts/tests/fixtures/fake_transcribeit.ts");
const fixturePath = "scripts/tests/fixtures/benchmark-input.txt";

describe("benchmark runner", () => {
  test("resets every outcome field without changing attempt identity", () => {
    const fixture = fixtureIdentity();
    const attempt = buildAttempts(baseMatrix(), new Map([[fixture.id, fixture]]))[0];
    const identity = identityFields(attempt);
    seedStaleOutcome(attempt);

    resetAttemptOutcome(attempt);

    expect(identityFields(attempt)).toEqual(identity);
    expect(outcomeFields(attempt)).toEqual(emptyOutcome());
  });

  test("clears failed-attempt evidence on retry and accepts Rust quality values", async () => {
    await chmod(fakeBinary, 0o755);
    const matrix = baseMatrix("fail-once");
    matrix.entries[0].required_env = ["FAKE_FAIL_ONCE_PATH"];
    const corpus = await fixtureCorpus();
    const runDirectory = uniqueRunDirectory("retry");
    const absoluteRunDirectory = join(repository, runDirectory);
    const marker = join(absoluteRunDirectory, "fail-once.marker");
    const previousMarker = process.env.FAKE_FAIL_ONCE_PATH;
    process.env.FAKE_FAIL_ONCE_PATH = marker;
    try {
      const first = await runMatrix(matrix, corpus, runOptions(runDirectory, false));
      expect(first.attempts[0]).toMatchObject({ status: "failed", error_category: "rate_limit" });
      seedStaleOutcome(first.attempts[0]);
      await writeJsonAtomic(join(absoluteRunDirectory, "state.json"), first);

      const retried = await runMatrix(matrix, corpus, runOptions(runDirectory, true));
      const attempt = retried.attempts[0];
      expect(attempt.status).toBe("passed");
      expect(attempt.error_category).toBeNull();
      expect(attempt.quality).toMatchObject({ timing_source: "model_native", speaker_source: "unknown" });
      expect(attempt.output_sha256).not.toBe("a".repeat(64));
      expect(attempt.manifest_sha256).not.toBe("b".repeat(64));
      expect(attempt.apple_speech).toBeNull();
      expect(attempt.reference_metrics).toBeNull();
      expect(attempt.remote_cleanup).toBe("not_applicable");
      expect(await Bun.file(join(absoluteRunDirectory, "attempts", attempt.key)).exists()).toBe(false);
    } finally {
      restoreEnvironment("FAKE_FAIL_ONCE_PATH", previousMarker);
      await rm(absoluteRunDirectory, { recursive: true, force: true });
    }
  });

  test("removes interrupted output and clears evidence before unconfigured reclassification", async () => {
    await chmod(fakeBinary, 0o755);
    const requiredEnvironment = `FAKE_REQUIRED_${process.pid}_${Math.random().toString(16).slice(2)}`;
    const previousRequired = process.env[requiredEnvironment];
    process.env[requiredEnvironment] = "configured";
    const matrix = baseMatrix("fail");
    matrix.entries[0].required_env = [requiredEnvironment];
    const corpus = await fixtureCorpus();
    const runDirectory = uniqueRunDirectory("unconfigured");
    const absoluteRunDirectory = join(repository, runDirectory);
    try {
      const first = await runMatrix(matrix, corpus, runOptions(runDirectory, false));
      const attempt = first.attempts[0];
      expect(attempt.status).toBe("failed");
      seedStaleOutcome(attempt);
      const staleOutput = join(absoluteRunDirectory, "attempts", attempt.key);
      await mkdir(staleOutput, { mode: 0o700 });
      await Bun.write(join(staleOutput, "stale.txt"), "stale output\n");
      await writeJsonAtomic(join(absoluteRunDirectory, "state.json"), first);
      delete process.env[requiredEnvironment];

      const reclassified = await runMatrix(matrix, corpus, runOptions(runDirectory, true));
      const skipped = reclassified.attempts[0];
      expect(skipped.status).toBe("skipped");
      expect(skipped.error_category).toBe("unconfigured");
      expect(skipped.started_at_utc).not.toBeNull();
      expect(skipped.completed_at_utc).toBe(skipped.started_at_utc);
      for (const field of clearedMeasurementFields) expect(skipped[field]).toBeNull();
      expect(await Bun.file(staleOutput).exists()).toBe(false);
    } finally {
      restoreEnvironment(requiredEnvironment, previousRequired);
      await rm(absoluteRunDirectory, { recursive: true, force: true });
    }
  });

  test("enforces the wall timeout across the POSIX process group", async () => {
    if (process.platform === "win32") return;
    await chmod(fakeBinary, 0o755);
    const matrix = baseMatrix("timeout");
    matrix.timeout_seconds = 1;
    matrix.measurements.peak_rss = true;
    matrix.entries[0].required_env = ["FAKE_GRANDCHILD_MARKER_PATH", "FAKE_GRANDCHILD_DELAY_MS"];
    const corpus = await fixtureCorpus();
    const runDirectory = uniqueRunDirectory("timeout");
    const absoluteRunDirectory = join(repository, runDirectory);
    const grandchildMarker = join(absoluteRunDirectory, "grandchild-survived.marker");
    const previousMarker = process.env.FAKE_GRANDCHILD_MARKER_PATH;
    const previousDelay = process.env.FAKE_GRANDCHILD_DELAY_MS;
    process.env.FAKE_GRANDCHILD_MARKER_PATH = grandchildMarker;
    process.env.FAKE_GRANDCHILD_DELAY_MS = "1800";
    try {
      const state = await runMatrix(matrix, corpus, runOptions(runDirectory, false));
      const attempt = state.attempts[0];
      expect(attempt.status).toBe("failed");
      expect(attempt.error_category).toBe("timeout");
      expect(attempt.wall_ms).toBeGreaterThanOrEqual(750);
      // Loaded runners may take several seconds to reap the killed process group;
      // keep the bound well below the fake child's natural 30-second runtime.
      expect(attempt.wall_ms).toBeLessThan(15_000);
      expect(await Bun.file(join(absoluteRunDirectory, "attempts", attempt.key)).exists()).toBe(false);
      await Bun.sleep(1_300);
      expect(await Bun.file(grandchildMarker).exists()).toBe(false);
    } finally {
      restoreEnvironment("FAKE_GRANDCHILD_MARKER_PATH", previousMarker);
      restoreEnvironment("FAKE_GRANDCHILD_DELAY_MS", previousDelay);
      await rm(absoluteRunDirectory, { recursive: true, force: true });
    }
  });

  test("rejects a symlink run child without touching its external target", async () => {
    const matrix = baseMatrix();
    const corpus = await fixtureCorpus();
    const runDirectory = uniqueRunDirectory("symlink");
    const absoluteRunDirectory = join(repository, runDirectory);
    const suffix = `${process.pid}-${Math.random().toString(16).slice(2)}`;
    const externalDirectory = join(repository, `output/benchmark-runner-external-${suffix}`);
    const sentinel = join(externalDirectory, "sentinel.txt");
    await mkdir(externalDirectory, { recursive: true, mode: 0o700 });
    await Bun.write(sentinel, "must survive\n");
    await symlink(externalDirectory, absoluteRunDirectory, "dir");
    try {
      await expect(runMatrix(matrix, corpus, runOptions(runDirectory, false))).rejects.toThrow("symbolic link");
      expect(await Bun.file(sentinel).text()).toBe("must survive\n");
    } finally {
      await unlink(absoluteRunDirectory).catch(() => undefined);
      await rm(externalDirectory, { recursive: true, force: true });
    }
  });

  test("binds evaluation-download resumes to an existing MODEL_CACHE_DIR identity", async () => {
    await chmod(fakeBinary, 0o755);
    const matrix = baseMatrix();
    matrix.entries[0].artifact = {
      lifecycle: "evaluation_download",
      revision: "test-revision",
      sha256: "c".repeat(64),
      bytes: 1,
    };
    const corpus = await fixtureCorpus();
    const runDirectory = uniqueRunDirectory("evaluation-root");
    const absoluteRunDirectory = join(repository, runDirectory);
    const suffix = `${process.pid}-${Math.random().toString(16).slice(2)}`;
    const evaluationRoot = join(repository, `output/benchmarks/test-model-cache-${suffix}`);
    const otherRoot = join(repository, `output/benchmarks/test-model-cache-other-${suffix}`);
    const previous = process.env.MODEL_CACHE_DIR;
    await mkdir(evaluationRoot, { mode: 0o700 });
    await mkdir(otherRoot, { mode: 0o700 });
    try {
      delete process.env.MODEL_CACHE_DIR;
      await expect(runMatrix(matrix, corpus, runOptions(runDirectory, false))).rejects.toThrow("require MODEL_CACHE_DIR");

      process.env.MODEL_CACHE_DIR = evaluationRoot;
      const state = await runMatrix(matrix, corpus, runOptions(runDirectory, false));
      const expectedIdentity = await resolveDirectBenchmarkChild(evaluationRoot, "test evaluation root");
      expect(state.evaluation_root_sha256).toBe(expectedIdentity.identitySha256);
      expect(state.evaluation_root_sha256).toMatch(/^[a-f0-9]{64}$/);

      process.env.MODEL_CACHE_DIR = otherRoot;
      await expect(runMatrix(matrix, corpus, runOptions(runDirectory, false))).rejects.toThrow(
        "different evaluation model root identity",
      );
    } finally {
      restoreEnvironment("MODEL_CACHE_DIR", previous);
      await rm(absoluteRunDirectory, { recursive: true, force: true });
      await rm(evaluationRoot, { recursive: true, force: true });
      await rm(otherRoot, { recursive: true, force: true });
    }
  });
});

const clearedMeasurementFields = [
  "wall_ms",
  "processing_ms",
  "real_time_factor",
  "peak_rss_bytes",
  "output_sha256",
  "manifest_sha256",
  "capabilities",
  "quality",
  "output_shape",
  "preprocessing",
  "apple_speech",
  "reference_metrics",
  "remote_cleanup",
] as const satisfies ReadonlyArray<keyof AttemptRecord>;

function baseMatrix(model = "pass"): BenchmarkMatrix {
  return {
    schema_version: 1,
    matrix_id: "fake-runner-matrix",
    description: "Deterministic fake runner matrix",
    execution_policy: "local_ci",
    attempt_order: "entry-fixture-repetition",
    concurrency: 1,
    retries: 0,
    timeout_seconds: 10,
    fixture_ids: ["fake-fixture"],
    entries: [
      {
        id: "fake-entry",
        provider: "local",
        model,
        execution: "local",
        cache_state: "cold",
        repetitions: 1,
        required_env: [],
        args: [],
      },
    ],
    measurements: { reference_scoring: false, peak_rss: false },
    tolerances: {
      enforcement: "fail",
      max_relative_latency_regression_percent: 25,
      max_absolute_latency_regression_ms: 100,
      min_success_rate: 0.5,
    },
  };
}

function runOptions(runDirectory: string, retryFailures: boolean) {
  return {
    runDirectory,
    binary: fakeBinary,
    allowHosted: false,
    retryFailures,
    keepAttemptOutputs: false,
  };
}

function uniqueRunDirectory(label: string): string {
  return `output/benchmarks/test-runner-${label}-${process.pid}-${Math.random().toString(16).slice(2)}`;
}

function fixtureIdentity(): FixtureIdentity {
  return {
    id: "fake-fixture",
    path: fixturePath,
    duration_seconds: 1,
    bytes: 1,
    sha256: "0".repeat(64),
  };
}

async function fixtureCorpus(): Promise<CorpusManifest> {
  const bytes = new Uint8Array(await Bun.file(join(repository, fixturePath)).arrayBuffer());
  return {
    schema_version: 1,
    corpus_id: "fake-runner-corpus",
    materialization_root: "scripts/tests/fixtures",
    sources: {},
    fixtures: [
      {
        id: "fake-fixture",
        audio: {
          path: fixturePath,
          duration_seconds: 1,
          bytes: bytes.byteLength,
          sha256: sha256(bytes),
          container: "txt",
          codec: "fake",
          sample_rate_hz: 1,
          channels: 1,
          materialize: {},
        },
        reference: {},
      },
    ],
  };
}

function seedStaleOutcome(attempt: AttemptRecord): void {
  Object.assign(attempt, {
    status: "failed",
    started_at_utc: "2026-01-01T00:00:00.000Z",
    completed_at_utc: "2026-01-01T00:01:00.000Z",
    wall_ms: 60_000,
    processing_ms: 59_000,
    real_time_factor: 60,
    peak_rss_bytes: 123_456,
    error_category: "stale-error",
    output_sha256: "a".repeat(64),
    manifest_sha256: "b".repeat(64),
    capabilities: { stale: true },
    quality: {
      timing_source: "synthetic",
      timing_reliable: false,
      timestamps_clamped: true,
      speaker_source: "none",
      warning_count: 99,
    },
    output_shape: {
      segments: 99,
      characters: 99,
      last_end_ms: 99,
      zero_duration_segments: 99,
      reversed_segments: 99,
      word_timestamps: true,
      speaker_labels: true,
    },
    preprocessing: "wav_fallback",
    apple_speech: {
      resolved_locale: "stale",
      apple_intelligence_available: true,
      asset_install_requested: true,
      asset_managed_by: "macos",
      on_device: true,
    },
    reference_metrics: {
      status: "unavailable_reference",
      word_accuracy: { status: "unavailable_reference" },
      domain_terms: { status: "unavailable_reference" },
      timing: { status: "unavailable_reference" },
      speakers: { status: "unavailable_reference" },
      word_timestamps: { status: "unavailable_reference" },
    },
    remote_cleanup: "failed",
  });
}

function identityFields(attempt: AttemptRecord): object {
  return {
    key: attempt.key,
    entry_id: attempt.entry_id,
    provider: attempt.provider,
    model: attempt.model,
    execution: attempt.execution,
    cache_state: attempt.cache_state,
    fixture: structuredClone(attempt.fixture),
    repetition: attempt.repetition,
  };
}

function outcomeFields(attempt: AttemptRecord): object {
  const identity = new Set(Object.keys(identityFields(attempt)));
  return Object.fromEntries(Object.entries(attempt).filter(([key]) => !identity.has(key)));
}

function emptyOutcome(): object {
  return {
    status: "pending",
    started_at_utc: null,
    completed_at_utc: null,
    wall_ms: null,
    processing_ms: null,
    real_time_factor: null,
    peak_rss_bytes: null,
    error_category: null,
    output_sha256: null,
    manifest_sha256: null,
    capabilities: null,
    quality: null,
    output_shape: null,
    preprocessing: null,
    apple_speech: null,
    reference_metrics: null,
    remote_cleanup: null,
  };
}

function restoreEnvironment(name: string, previous: string | undefined): void {
  if (previous === undefined) delete process.env[name];
  else process.env[name] = previous;
}
