import { chmod, mkdir, rm } from "node:fs/promises";
import { join } from "node:path";

import { loadMatrix, repositoryRoot, validateMatrix } from "../benchmark_harness/config";
import { compareResults, publishResult, validatePublishedResult } from "../benchmark_harness/result";
import { runMatrix } from "../benchmark_harness/runner";
import { sha256 } from "../benchmark_harness/state";
import type { BenchmarkMatrix, PublishedResult } from "../benchmark_harness/types";
import type { CorpusManifest } from "../corpus";

const repository = repositoryRoot();
const fakeBinary = join(repository, "scripts/tests/fixtures/fake_transcribeit.ts");
const fixturePath = "scripts/tests/fixtures/benchmark-input.txt";

describe("benchmark harness", () => {
  test("validates the tracked hosted matrix and plans explicit policy", async () => {
    const { matrix } = await loadMatrix("benchmarks/matrices/hosted-smoke.yaml");
    expect(matrix.execution_policy).toBe("manual");
    expect(matrix.concurrency).toBe(1);
    expect(matrix.retries).toBe(0);
    expect(matrix.entries).toHaveLength(6);
    expect(matrix.entries.every((entry) => entry.execution === "hosted")).toBe(true);
  });

  test("rejects credential flags, tight tolerances, and hosted CI execution", () => {
    const matrix = baseMatrix();
    matrix.execution_policy = "local_ci";
    matrix.entries[0].execution = "hosted";
    matrix.entries[0].provider = "openai";
    matrix.entries[0].args = ["--api-key", "unsafe"];
    matrix.tolerances.max_relative_latency_regression_percent = 5;
    Object.assign(matrix, { unexpected_field: true });
    const result = validateMatrix(matrix, corpusWithFixture({ bytes: 26, sha256: "0".repeat(64) }));
    expect(result.errors.some((error) => error.includes("credential"))).toBe(true);
    expect(result.errors.some((error) => error.includes("local_ci"))).toBe(true);
    expect(result.errors.some((error) => error.includes("25"))).toBe(true);
    expect(result.errors.some((error) => error.includes("unsupported fields"))).toBe(true);
  });

  test("runs, resumes, preserves failures, sanitizes, and compares", async () => {
    await chmod(fakeBinary, 0o755);
    const bytes = new Uint8Array(await Bun.file(join(repository, fixturePath)).arrayBuffer());
    const matrix = baseMatrix();
    matrix.entries.push({ ...matrix.entries[0], id: "fake-failure", model: "fail" });
    const corpus = corpusWithFixture({ bytes: bytes.byteLength, sha256: sha256(bytes) });
    const suffix = `${process.pid}-${Math.random().toString(16).slice(2)}`;
    const runDirectory = `output/benchmarks/test-harness-${suffix}`;
    const absoluteRunDirectory = join(repository, runDirectory);
    const counterPath = join(absoluteRunDirectory, "invocations.txt");
    await mkdir(absoluteRunDirectory, { recursive: true, mode: 0o700 });
    process.env.FAKE_COUNTER_PATH = counterPath;
    try {
      const options = {
        runDirectory,
        binary: fakeBinary,
        allowHosted: false,
        retryFailures: false,
        keepAttemptOutputs: false,
      };
      const first = await runMatrix(matrix, corpus, options);
      expect(first.attempts.map((attempt) => attempt.status)).toEqual(["passed", "failed"]);
      expect(first.attempts[1].error_category).toBe("rate_limit");
      expect(first.attempts[0].capabilities?.segments).toBe(true);
      expect(first.attempts[0].quality?.warning_count).toBe(1);
      expect(await Bun.file(counterPath).text()).toBe("1\n1\n");
      expect(await Bun.file(join(absoluteRunDirectory, "attempts", first.attempts[0].key)).exists()).toBe(false);

      const resumed = await runMatrix(matrix, corpus, options);
      expect(resumed.attempts.map((attempt) => attempt.status)).toEqual(["passed", "failed"]);
      expect(await Bun.file(counterPath).text()).toBe("1\n1\n");

      const outputPath = join(absoluteRunDirectory, "published.json");
      const published = await publishResult(join(absoluteRunDirectory, "state.json"), outputPath);
      expect(published.summary).toEqual({ attempts: 2, passed: 1, failed: 1, skipped: 0 });
      const serialized = await Bun.file(outputPath).text();
      expect(serialized).not.toContain("synthetic transcript");
      expect(serialized).not.toContain("must-not-be-published");
      expect(serialized).not.toContain("example.invalid");
      expect(validatePublishedResult(published)).toEqual([]);
      const unsafe = structuredClone(published) as PublishedResult & { transcript_text?: string };
      unsafe.transcript_text = "must be rejected";
      expect(validatePublishedResult(unsafe).some((error) => error.includes("unsupported fields"))).toBe(true);
      const unsafeEnvironment = structuredClone(published) as PublishedResult;
      Object.assign(unsafeEnvironment.environment.machine, { secret: "must be rejected" });
      expect(validatePublishedResult(unsafeEnvironment).some((error) => error.includes("environment.machine"))).toBe(true);

      const candidate = structuredClone(published);
      candidate.attempts[0].wall_ms = Number(published.attempts[0].wall_ms) + 1000;
      candidate.matrix.tolerances.enforcement = "fail";
      candidate.matrix.tolerances.max_relative_latency_regression_percent = 25;
      candidate.matrix.tolerances.max_absolute_latency_regression_ms = 100;
      expect(compareResults(published, candidate).exit_failure).toBe(true);
      for (const entry of candidate.matrix.entries) entry.execution = "hosted";
      for (const attempt of candidate.attempts) attempt.execution = "hosted";
      expect(compareResults(published, candidate).exit_failure).toBe(false);
      candidate.matrix_sha256 = "f".repeat(64);
      expect(() => compareResults(published, candidate)).toThrow("different matrix definitions");
    } finally {
      delete process.env.FAKE_COUNTER_PATH;
      await rm(absoluteRunDirectory, { recursive: true, force: true });
    }
  });

  test("requires explicit hosted execution opt-in", async () => {
    const bytes = new Uint8Array(await Bun.file(join(repository, fixturePath)).arrayBuffer());
    const matrix = baseMatrix();
    matrix.execution_policy = "manual";
    matrix.tolerances.enforcement = "report_only";
    matrix.entries[0].provider = "openai";
    matrix.entries[0].execution = "hosted";
    const runDirectory = `output/benchmarks/test-hosted-${process.pid}-${Math.random().toString(16).slice(2)}`;
    try {
      await expect(
        runMatrix(matrix, corpusWithFixture({ bytes: bytes.byteLength, sha256: sha256(bytes) }), {
          runDirectory,
          binary: fakeBinary,
          allowHosted: false,
          retryFailures: false,
          keepAttemptOutputs: false,
        }),
      ).rejects.toThrow("--allow-hosted");
    } finally {
      await rm(join(repository, runDirectory), { recursive: true, force: true });
    }
  });
});

function baseMatrix(): BenchmarkMatrix {
  return {
    schema_version: 1,
    matrix_id: "fake-local-matrix",
    description: "Deterministic fake local matrix",
    execution_policy: "local_ci",
    concurrency: 2,
    retries: 0,
    timeout_seconds: 30,
    fixture_ids: ["fake-fixture"],
    entries: [
      {
        id: "fake-success",
        provider: "local",
        model: "pass",
        execution: "local",
        cache_state: "cold",
        repetitions: 1,
        required_env: [],
        args: [],
      },
    ],
    tolerances: {
      enforcement: "fail",
      max_relative_latency_regression_percent: 25,
      max_absolute_latency_regression_ms: 100,
      min_success_rate: 0.5,
    },
  };
}

function corpusWithFixture(identity: { bytes: number; sha256: string }): CorpusManifest {
  return {
    schema_version: 1,
    corpus_id: "fake-corpus",
    materialization_root: "scripts/tests/fixtures",
    sources: {},
    fixtures: [
      {
        id: "fake-fixture",
        audio: {
          path: fixturePath,
          bytes: identity.bytes,
          sha256: identity.sha256,
          duration_seconds: 1,
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
