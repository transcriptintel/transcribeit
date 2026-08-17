import { chmod, mkdir, rm, symlink, unlink } from "node:fs/promises";
import { join } from "node:path";

import { loadMatrix, repositoryRoot, resolveFixtures, validateMatrix } from "../benchmark_harness/config";
import { compareResults, publishResult, validatePublishedResult } from "../benchmark_harness/result";
import { buildAttempts, buildCommand, parsePeakRss, runMatrix } from "../benchmark_harness/runner";
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

  test("validates and interleaves the Apple Speech versus large-v3 matrix", async () => {
    const { matrix, corpus } = await loadMatrix("benchmarks/matrices/ti-014-apple-large-v3.yaml");
    expect(matrix.attempt_order).toBe("fixture-repetition-entry");
    expect(matrix.measurements).toEqual({ reference_scoring: true, peak_rss: true });
    expect(matrix.entries).toHaveLength(2);
    const apple = matrix.entries[0];
    expect(apple.provider).toBe("apple-speech");
    expect(apple.execution).toBe("local");
    const fixture = resolveFixtures(matrix, corpus).values().next().value!;
    const command = buildCommand("transcribeit", matrix, apple, fixture, "attempt-output");
    expect(command).not.toContain("--model");
    expect(command).not.toContain("--remote-model");
    expect(command).toContain("en-US");

    const attempts = buildAttempts(matrix, resolveFixtures(matrix, corpus));
    expect(attempts).toHaveLength(24);
    expect(attempts.slice(0, 4).map((attempt) => `${attempt.provider}/${attempt.repetition}`))
      .toEqual(["apple-speech/1", "local/1", "apple-speech/2", "local/2"]);
  });

  test("requires Apple Speech to be local with an explicit language", () => {
    const matrix = baseMatrix();
    matrix.entries[0].provider = "apple-speech";
    matrix.entries[0].model = "speech-transcriber";
    matrix.entries[0].execution = "hosted";
    const invalid = validateMatrix(matrix, corpusWithFixture({ bytes: 26, sha256: "0".repeat(64) }));
    expect(invalid.errors.some((error) => error.includes("requires execution local"))).toBe(true);
    expect(invalid.errors.some((error) => error.includes("explicit non-auto --language"))).toBe(true);
  });

  test("parses Darwin peak RSS without retaining time output", () => {
    expect(parsePeakRss("  32194560  maximum resident set size\n")).toBe(32_194_560);
    expect(parsePeakRss("real 1.0\n")).toBeNull();
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
    matrix.entries[0].required_env = ["FAKE_COUNTER_PATH"];
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
      expect(published.cleanup).toEqual({
        attempt_outputs_removed: true,
        evaluation_root_sha256: null,
        downloaded_model: null,
        apple_speech_asset: null,
      });
      expect(published.attempts[0]).toMatchObject({
        processing_ms: null,
        peak_rss_bytes: null,
        output_shape: {
          segments: 0,
          characters: 0,
          last_end_ms: null,
          zero_duration_segments: 0,
          reversed_segments: 0,
          word_timestamps: false,
          speaker_labels: false,
        },
        preprocessing: "canonical_wav",
        apple_speech: null,
        reference_metrics: null,
      });
      const serialized = await Bun.file(outputPath).text();
      expect(serialized).not.toContain("synthetic transcript");
      expect(serialized).not.toContain("must-not-be-published");
      expect(serialized).not.toContain("example.invalid");
      expect(validatePublishedResult(published)).toEqual([]);
      const invalidRecordedAt = structuredClone(published) as PublishedResult;
      invalidRecordedAt.recorded_at_utc = "not-a-timestamp";
      expect(validatePublishedResult(invalidRecordedAt).some((error) => error.includes("recorded_at_utc"))).toBe(true);
      const incompleteSanitization = structuredClone(published) as PublishedResult;
      incompleteSanitization.sanitization.request_ids_removed = false as never;
      expect(validatePublishedResult(incompleteSanitization).some((error) => error.includes("sanitization flags"))).toBe(true);
      const unsafe = structuredClone(published) as PublishedResult & { transcript_text?: string };
      unsafe.transcript_text = "must be rejected";
      expect(validatePublishedResult(unsafe).some((error) => error.includes("unsupported fields"))).toBe(true);
      const unsafeEnvironment = structuredClone(published) as PublishedResult;
      unsafeEnvironment.environment.machine.cpu = { transcript_text: "must be rejected" } as never;
      expect(validatePublishedResult(unsafeEnvironment).some((error) => error.includes("must not contain nested data"))).toBe(true);
      const unsafeShape = structuredClone(published) as PublishedResult;
      Object.assign(unsafeShape.attempts[0].output_shape!, { transcript_text: "must be rejected" });
      expect(validatePublishedResult(unsafeShape).some((error) => error.includes("output_shape"))).toBe(true);
      const unsafeScoring = structuredClone(published) as PublishedResult;
      Object.assign(unsafeScoring.attempts[0], {
        reference_metrics: {
          status: "unavailable_reference",
          word_accuracy: { status: "unavailable_reference", reference_text: "must be rejected" },
          domain_terms: { status: "unavailable_reference" },
          timing: { status: "unavailable_reference" },
          speakers: { status: "unavailable_reference" },
          word_timestamps: { status: "unavailable_reference" },
        },
      });
      expect(validatePublishedResult(unsafeScoring).some((error) => error.includes("word_accuracy"))).toBe(true);
      for (const forbidden of [
        "/home/example/private/file.json", "/root/private/file.json",
        "C:\\Users\\example\\private.txt",
        `sk-${"proj-"}${"a".repeat(22)}`,
        `AIza${"A".repeat(32)}`,
        `AKIA${"A".repeat(16)}`,
      ]) {
        const unsafeString = structuredClone(published) as PublishedResult;
        unsafeString.environment.machine.cpu = forbidden;
        expect(validatePublishedResult(unsafeString).some((error) => error.includes("forbidden path or credential"))).toBe(true);
      }
      const emptyAttempts = structuredClone(published) as PublishedResult;
      emptyAttempts.attempts = [];
      emptyAttempts.summary = { attempts: 0, passed: 0, failed: 0, skipped: 0 };
      expect(validatePublishedResult(emptyAttempts).some((error) => error.includes("missing attempt key"))).toBe(true);
      const missingAttempt = structuredClone(published) as PublishedResult;
      missingAttempt.attempts.pop();
      missingAttempt.summary = { attempts: 1, passed: 1, failed: 0, skipped: 0 };
      expect(validatePublishedResult(missingAttempt).some((error) => error.includes("missing attempt key"))).toBe(true);
      const duplicateAttempt = structuredClone(published) as PublishedResult;
      duplicateAttempt.attempts.push(structuredClone(duplicateAttempt.attempts[0]));
      duplicateAttempt.summary = { attempts: 3, passed: 2, failed: 1, skipped: 0 };
      expect(validatePublishedResult(duplicateAttempt).some((error) => error.includes("duplicate attempt key"))).toBe(true);
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
      const attemptsPath = join(absoluteRunDirectory, "attempts");
      await Bun.write(join(attemptsPath, "leftover.txt"), "must block publication\n");
      await expect(publishResult(join(absoluteRunDirectory, "state.json"), outputPath)).rejects.toThrow("not been removed");
      await rm(join(attemptsPath, "leftover.txt"), { force: true });
      await rm(attemptsPath, { recursive: true, force: true });
      await symlink(absoluteRunDirectory, attemptsPath, "dir");
      await expect(publishResult(join(absoluteRunDirectory, "state.json"), outputPath)).rejects.toThrow("symbolic link");
      await unlink(attemptsPath);

      const retainedState = await Bun.file(join(absoluteRunDirectory, "state.json")).json();
      retainedState.producing_commit.worktree_fingerprint_sha256 = retainedState.producing_commit.worktree_fingerprint_sha256.startsWith("a")
        ? "b".repeat(64)
        : "a".repeat(64);
      await Bun.write(join(absoluteRunDirectory, "state.json"), `${JSON.stringify(retainedState)}\n`);
      await expect(publishResult(join(absoluteRunDirectory, "state.json"), outputPath)).rejects.toThrow("worktree changed");
      retainedState.producing_commit = published.producing_commit;
      retainedState.keep_attempt_outputs = true;
      await Bun.write(join(absoluteRunDirectory, "state.json"), `${JSON.stringify(retainedState)}\n`);
      await expect(publishResult(join(absoluteRunDirectory, "state.json"), outputPath)).rejects.toThrow("retained attempt outputs");
    } finally {
      delete process.env.FAKE_COUNTER_PATH;
      await rm(absoluteRunDirectory, { recursive: true, force: true });
    }
  });

  test("requires verified evaluation-download cleanup and records the macOS-owned Apple asset", async () => {
    await chmod(fakeBinary, 0o755);
    const bytes = new Uint8Array(await Bun.file(join(repository, fixturePath)).arrayBuffer());
    const matrix = baseMatrix();
    Object.assign(matrix.entries[0], { artifact: { lifecycle: "evaluation_download", revision: "test-revision", sha256: "a".repeat(64), bytes: 123 }, required_env: ["MODEL_CACHE_DIR"] });
    matrix.entries.push({
      ...matrix.entries[0],
      id: "fake-apple",
      provider: "apple-speech",
      model: "speech-transcriber",
      args: ["--language", "en-US"],
      artifact: { lifecycle: "system_managed" },
    });
    const corpus = corpusWithFixture({ bytes: bytes.byteLength, sha256: sha256(bytes) });
    const suffix = `${process.pid}-${Math.random().toString(16).slice(2)}`;
    const runDirectory = `output/benchmarks/test-cleanup-${suffix}`;
    const absoluteRunDirectory = join(repository, runDirectory);
    const evaluationRoot = join(repository, "output/benchmarks", `test-model-${suffix}`);
    const wrongRoot = join(repository, "output/benchmarks", `test-wrong-model-${suffix}`);
    const statePath = join(absoluteRunDirectory, "state.json");
    const outputPath = join(absoluteRunDirectory, "published.json");
    const cleanupPath = join(absoluteRunDirectory, "cleanup.json");
    const previousModelCache = process.env.MODEL_CACHE_DIR;
    try {
      await mkdir(evaluationRoot, { recursive: false, mode: 0o700 });
      process.env.MODEL_CACHE_DIR = evaluationRoot;
      const state = await runMatrix(matrix, corpus, {
        runDirectory,
        binary: fakeBinary,
        allowHosted: false,
        retryFailures: false,
        keepAttemptOutputs: false,
      });
      expect(state.attempts.every((attempt) => attempt.status === "passed")).toBe(true);
      expect(state.evaluation_root_sha256).toMatch(/^[a-f0-9]{64}$/);
      const cleanup = {
        schema_version: "transcribeit.downloaded-model-cleanup.v1" as const,
        matrix_sha256: state.matrix_sha256,
        evaluation_root_sha256: state.evaluation_root_sha256!,
        artifact_sha256s: ["a".repeat(64)],
        recorded_at_utc: new Date().toISOString(),
        evaluation_root_preexisting: false,
        initial_bytes: 0,
        downloaded_bytes: 123,
        reclaimed_bytes: 123,
        cleanup_completed: true,
        evaluation_root_absent_after_cleanup: true,
      };
      await Bun.write(cleanupPath, `${JSON.stringify(cleanup)}\n`);
      await expect(publishResult(statePath, outputPath)).rejects.toThrow("--cleanup-record and --cleanup-root");
      await expect(publishResult(statePath, outputPath, cleanupPath, evaluationRoot)).rejects.toThrow("still exists");
      await rm(evaluationRoot, { recursive: true });
      await expect(publishResult(statePath, outputPath, cleanupPath, wrongRoot)).rejects.toThrow("does not match the run state");
      await Bun.write(cleanupPath, `${JSON.stringify({ ...cleanup, recorded_at_utc: "2000-01-01T00:00:00.000Z" })}\n`);
      await expect(publishResult(statePath, outputPath, cleanupPath, evaluationRoot)).rejects.toThrow("predates this benchmark run");
      await Bun.write(cleanupPath, `${JSON.stringify({ ...cleanup, matrix_sha256: "b".repeat(64) })}\n`);
      await expect(publishResult(statePath, outputPath, cleanupPath, evaluationRoot)).rejects.toThrow("matrix_sha256");
      await Bun.write(cleanupPath, `${JSON.stringify({ ...cleanup, artifact_sha256s: ["b".repeat(64)] })}\n`);
      await expect(publishResult(statePath, outputPath, cleanupPath, evaluationRoot)).rejects.toThrow("artifact_sha256s");

      await Bun.write(cleanupPath, `${JSON.stringify({ ...cleanup, unexpected_path: "must be rejected" })}\n`);
      await expect(publishResult(statePath, outputPath, cleanupPath, evaluationRoot)).rejects.toThrow("unsupported fields");
      await Bun.write(cleanupPath, `${JSON.stringify({ ...cleanup, initial_bytes: 1 })}\n`);
      await expect(publishResult(statePath, outputPath, cleanupPath, evaluationRoot)).rejects.toThrow("initial_bytes must be zero");
      await Bun.write(cleanupPath, `${JSON.stringify({ ...cleanup, recorded_at_utc: "2999-01-01T00:00:00.000Z" })}\n`);
      await expect(publishResult(statePath, outputPath, cleanupPath, evaluationRoot)).rejects.toThrow("later than publication");
      await Bun.write(cleanupPath, `${JSON.stringify(cleanup)}\n`);
      const published = await publishResult(statePath, outputPath, cleanupPath, evaluationRoot);
      expect(published.cleanup).toEqual({
        attempt_outputs_removed: true,
        evaluation_root_sha256: state.evaluation_root_sha256,
        downloaded_model: cleanup,
        apple_speech_asset: {
          owner: "macos",
          cleanup_attempted: false,
          lifecycle: "system_managed_shared",
        },
      });
      expect(validatePublishedResult(published)).toEqual([]);

      const unsafeCleanup = structuredClone(published) as PublishedResult;
      Object.assign(unsafeCleanup.cleanup.downloaded_model!, { local_path: "must be rejected" });
      expect(validatePublishedResult(unsafeCleanup).some((error) => error.includes("cleanup.downloaded_model"))).toBe(true);
      const unsafeArtifact = structuredClone(published) as PublishedResult;
      Object.assign(unsafeArtifact.matrix.entries[0].artifact!, { source_url: "must be rejected" });
      expect(validatePublishedResult(unsafeArtifact).some((error) => error.includes("artifact"))).toBe(true);
    } finally {
      if (previousModelCache === undefined) delete process.env.MODEL_CACHE_DIR;
      else process.env.MODEL_CACHE_DIR = previousModelCache;
      await rm(evaluationRoot, { recursive: true, force: true });
      await rm(wrongRoot, { recursive: true, force: true });
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
    attempt_order: "entry-fixture-repetition",
    concurrency: 2,
    retries: 0,
    timeout_seconds: 30,
    fixture_ids: ["fake-fixture"],
    measurements: { reference_scoring: false, peak_rss: false },
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
