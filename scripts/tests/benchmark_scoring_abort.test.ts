import { chmod, rm } from "node:fs/promises";
import { join } from "node:path";

import { repositoryRoot } from "../benchmark_harness/config";
import { runMatrix } from "../benchmark_harness/runner";
import {
  ScoringIntegrityError,
  validatedScoringCorpus,
  verifyReviewedReferenceBytes,
} from "../benchmark_harness/scoring";
import { sha256 } from "../benchmark_harness/state";
import type { BenchmarkMatrix, RunState } from "../benchmark_harness/types";
import type { CorpusManifest } from "../corpus";

const repository = repositoryRoot();
const fakeBinary = join(repository, "scripts/tests/fixtures/fake_transcribeit.ts");
const fixturePath = "scripts/tests/fixtures/benchmark-input.txt";

describe("fatal benchmark scoring integrity", () => {
  test("propagates a typed reviewed-reference failure and aborts a peer process group", async () => {
    if (process.platform === "win32") return;
    await chmod(fakeBinary, 0o755);
    const suffix = `${process.pid}-${Math.random().toString(16).slice(2)}`;
    const runDirectory = `output/benchmarks/test-scoring-abort-${suffix}`;
    const absoluteRunDirectory = join(repository, runDirectory);
    const marker = join(absoluteRunDirectory, "peer-grandchild.marker");
    const previousMarker = process.env.FAKE_GRANDCHILD_MARKER_PATH;
    const previousDelay = process.env.FAKE_GRANDCHILD_DELAY_MS;
    process.env.FAKE_GRANDCHILD_MARKER_PATH = marker;
    process.env.FAKE_GRANDCHILD_DELAY_MS = "1200";
    const fatal = reviewedReferenceFailure();
    try {
      const error = await rejectedError(runMatrix(
        concurrentMatrix(),
        await fixtureCorpus(),
        runOptions(runDirectory),
        async () => { throw fatal; },
      ));

      expect(error).toBe(fatal);
      expect(error).toBeInstanceOf(ScoringIntegrityError);
      expect((error as ScoringIntegrityError).kind).toBe("reviewed_reference");
      expect(error.message).toBe("reviewed reference integrity verification failed");
      expect(error.message).not.toContain("sensitive reviewed words");
      await Bun.sleep(1_300);
      expect(await Bun.file(marker).exists()).toBe(false);
      const state = await Bun.file(join(absoluteRunDirectory, "state.json")).json() as RunState;
      expect(state.attempts.every((attempt) => attempt.error_category === null)).toBe(true);
      expect(JSON.stringify(state)).not.toContain("local_output_invalid");
      expect(JSON.stringify(state)).not.toContain("provider_error");
    } finally {
      restoreEnvironment("FAKE_GRANDCHILD_MARKER_PATH", previousMarker);
      restoreEnvironment("FAKE_GRANDCHILD_DELAY_MS", previousDelay);
      await rm(absoluteRunDirectory, { recursive: true, force: true });
    }
  });

  test("keeps ordinary malformed provider output attempt-local", async () => {
    await chmod(fakeBinary, 0o755);
    const runDirectory = `output/benchmarks/test-provider-output-${process.pid}-${Math.random().toString(16).slice(2)}`;
    const absoluteRunDirectory = join(repository, runDirectory);
    const matrix = baseMatrix();
    matrix.entries[0].model = "malformed-output";
    try {
      const state = await runMatrix(matrix, await fixtureCorpus(), runOptions(runDirectory));

      expect(state.attempts[0].status).toBe("failed");
      expect(state.attempts[0].error_category).toBe("local_output_invalid");
      expect(state.attempts[0].error_category).not.toBe("provider_error");
    } finally {
      await rm(absoluteRunDirectory, { recursive: true, force: true });
    }
  });

  test("uses a typed transcript-free error for malformed scoring corpus state", () => {
    let error: Error;
    try {
      validatedScoringCorpus({ schema_version: 1, fixtures: "invalid" });
      throw new Error("expected corpus validation to fail");
    } catch (caught) {
      if (!(caught instanceof Error)) throw caught;
      error = caught;
    }
    expect(error.name).toBe("ScoringIntegrityError");
    expect(error).toBeInstanceOf(ScoringIntegrityError);
    expect((error as ScoringIntegrityError).kind).toBe("corpus");
    expect(error.message).toBe("benchmark scoring corpus integrity verification failed");
    expect(error.message).not.toContain("/");
  });
});

function reviewedReferenceFailure(): ScoringIntegrityError {
  const bytes = new TextEncoder().encode("sensitive reviewed words\n");
  try {
    verifyReviewedReferenceBytes(bytes, bytes.byteLength, "0".repeat(64));
  } catch (error) {
    if (error instanceof ScoringIntegrityError) return error;
    throw new Error("expected a typed scoring integrity error");
  }
  throw new Error("expected reviewed-reference verification to fail");
}

function concurrentMatrix(): BenchmarkMatrix {
  const matrix = baseMatrix();
  matrix.concurrency = 2;
  matrix.measurements.reference_scoring = true;
  matrix.entries = [
    { ...matrix.entries[0], id: "scoring-entry" },
    {
      ...matrix.entries[0],
      id: "peer-entry",
      model: "timeout",
      required_env: ["FAKE_GRANDCHILD_MARKER_PATH", "FAKE_GRANDCHILD_DELAY_MS"],
    },
  ];
  return matrix;
}

function baseMatrix(): BenchmarkMatrix {
  return {
    schema_version: 1,
    matrix_id: "scoring-abort-test",
    description: "Synthetic scoring abort test",
    execution_policy: "local_ci",
    attempt_order: "entry-fixture-repetition",
    concurrency: 1,
    retries: 0,
    timeout_seconds: 10,
    fixture_ids: ["fake-fixture"],
    entries: [{
      id: "fake-entry",
      provider: "local",
      model: "pass",
      execution: "local",
      cache_state: "cold",
      repetitions: 1,
      required_env: [],
      args: [],
    }],
    measurements: { reference_scoring: false, peak_rss: false },
    tolerances: {
      enforcement: "fail",
      max_relative_latency_regression_percent: 25,
      max_absolute_latency_regression_ms: 100,
      min_success_rate: 1,
    },
  };
}

function runOptions(runDirectory: string) {
  return {
    runDirectory,
    binary: fakeBinary,
    allowHosted: false,
    retryFailures: false,
    keepAttemptOutputs: false,
  };
}

async function fixtureCorpus(): Promise<CorpusManifest> {
  const bytes = new Uint8Array(await Bun.file(join(repository, fixturePath)).arrayBuffer());
  return {
    schema_version: 1,
    corpus_id: "scoring-abort-test",
    materialization_root: "scripts/tests/fixtures",
    sources: {},
    fixtures: [{
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
    }],
  };
}

async function rejectedError(promise: Promise<unknown>): Promise<Error> {
  try {
    await promise;
  } catch (error) {
    if (error instanceof Error) return error;
    throw new Error("expected an Error rejection");
  }
  throw new Error("expected promise to reject");
}

function restoreEnvironment(name: string, previous: string | undefined): void {
  if (previous === undefined) delete process.env[name];
  else process.env[name] = previous;
}
