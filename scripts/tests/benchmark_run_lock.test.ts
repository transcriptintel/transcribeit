import { randomUUID } from "node:crypto";
import { chmod, rm, stat } from "node:fs/promises";
import { join } from "node:path";

import { repositoryRoot } from "../benchmark_harness/config";
import { runLockFileName } from "../benchmark_harness/run_lock";
import { resetAttemptOutcome, runMatrix } from "../benchmark_harness/runner";
import { sha256, writeJsonAtomic } from "../benchmark_harness/state";
import type { BenchmarkMatrix } from "../benchmark_harness/types";
import type { CorpusManifest } from "../corpus";

const repository = repositoryRoot();
const fakeBinary = join(repository, "scripts/tests/fixtures/fake_transcribeit.ts");
const fixturePath = "scripts/tests/fixtures/benchmark-input.txt";

describe("benchmark run ownership", () => {
  test("holds an owner-only lock and excludes a concurrent invocation", async () => {
    await chmod(fakeBinary, 0o755);
    const runDirectory = uniqueRunDirectory("concurrent");
    const absoluteRunDirectory = join(repository, runDirectory);
    const lockPath = join(absoluteRunDirectory, runLockFileName);
    const marker = join(absoluteRunDirectory, "grandchild.marker");
    const previousMarker = process.env.FAKE_GRANDCHILD_MARKER_PATH;
    const previousDelay = process.env.FAKE_GRANDCHILD_DELAY_MS;
    process.env.FAKE_GRANDCHILD_MARKER_PATH = marker;
    process.env.FAKE_GRANDCHILD_DELAY_MS = "1500";
    const matrix = baseMatrix("timeout");
    matrix.timeout_seconds = 1;
    matrix.entries[0].required_env = ["FAKE_GRANDCHILD_MARKER_PATH", "FAKE_GRANDCHILD_DELAY_MS"];
    try {
      const firstRun = runMatrix(matrix, await fixtureCorpus(), runOptions(runDirectory));
      await waitForFile(lockPath);
      expect((await stat(lockPath)).mode & 0o777).toBe(0o600);

      const error = await rejectedError(runMatrix(matrix, await fixtureCorpus(), runOptions(runDirectory)));
      expect(error.message).toBe("benchmark run is already active or lock recovery is in progress");
      expect(error.message).not.toContain(runDirectory);
      expect(error.message).not.toContain(absoluteRunDirectory);

      const completed = await firstRun;
      expect(completed.attempts[0]).toMatchObject({ status: "failed", error_category: "timeout" });
      expect(await Bun.file(lockPath).exists()).toBe(false);
    } finally {
      restoreEnvironment("FAKE_GRANDCHILD_MARKER_PATH", previousMarker);
      restoreEnvironment("FAKE_GRANDCHILD_DELAY_MS", previousDelay);
      await rm(absoluteRunDirectory, { recursive: true, force: true });
    }
  });

  test("rejects resume when the captured machine or tool environment drifts", async () => {
    await chmod(fakeBinary, 0o755);
    const runDirectory = uniqueRunDirectory("environment");
    const absoluteRunDirectory = join(repository, runDirectory);
    const statePath = join(absoluteRunDirectory, "state.json");
    try {
      const first = await runMatrix(baseMatrix(), await fixtureCorpus(), runOptions(runDirectory));
      first.environment.machine.cpu = `${first.environment.machine.cpu}-changed`;
      await writeJsonAtomic(statePath, first);

      const error = await rejectedError(runMatrix(baseMatrix(), await fixtureCorpus(), runOptions(runDirectory)));
      expect(error.message).toBe("existing run state uses a different captured environment");
      expect(error.message).not.toContain(runDirectory);
      expect(error.message).not.toContain(absoluteRunDirectory);
      expect(await Bun.file(join(absoluteRunDirectory, runLockFileName)).exists()).toBe(false);
    } finally {
      await rm(absoluteRunDirectory, { recursive: true, force: true });
    }
  });

  test("recovers a well-formed stale lock without stealing a live owner", async () => {
    await chmod(fakeBinary, 0o755);
    const runDirectory = uniqueRunDirectory("stale");
    const absoluteRunDirectory = join(repository, runDirectory);
    const lockPath = join(absoluteRunDirectory, runLockFileName);
    try {
      await runMatrix(baseMatrix(), await fixtureCorpus(), runOptions(runDirectory));
      await writeStaleLock(lockPath);

      const resumed = await runMatrix(baseMatrix(), await fixtureCorpus(), runOptions(runDirectory));
      expect(resumed.attempts[0].status).toBe("passed");
      expect(await Bun.file(lockPath).exists()).toBe(false);
      expect(await Bun.file(join(absoluteRunDirectory, ".benchmark-run.lock-recovery")).exists()).toBe(false);
    } finally {
      await rm(absoluteRunDirectory, { recursive: true, force: true });
    }
  });

  test("refuses stale-lock recovery while run state has an active attempt", async () => {
    await chmod(fakeBinary, 0o755);
    const runDirectory = uniqueRunDirectory("stale-running");
    const absoluteRunDirectory = join(repository, runDirectory);
    const lockPath = join(absoluteRunDirectory, runLockFileName);
    try {
      const state = await runMatrix(baseMatrix(), await fixtureCorpus(), runOptions(runDirectory));
      resetAttemptOutcome(state.attempts[0]);
      state.attempts[0].status = "running";
      state.attempts[0].started_at_utc = new Date().toISOString();
      await writeJsonAtomic(join(absoluteRunDirectory, "state.json"), state);
      await writeStaleLock(lockPath);

      const error = await rejectedError(runMatrix(baseMatrix(), await fixtureCorpus(), runOptions(runDirectory)));
      expect(error.message).toBe(
        "stale benchmark lock requires operator intervention because run state may still be active",
      );
      expect(error.message).not.toContain(runDirectory);
      expect(error.message).not.toContain(absoluteRunDirectory);
      expect(await Bun.file(lockPath).exists()).toBe(true);
    } finally {
      await rm(absoluteRunDirectory, { recursive: true, force: true });
    }
  });

  test("refuses stale-lock recovery when existing run state cannot be verified", async () => {
    await chmod(fakeBinary, 0o755);
    const runDirectory = uniqueRunDirectory("stale-invalid-state");
    const absoluteRunDirectory = join(repository, runDirectory);
    const statePath = join(absoluteRunDirectory, "state.json");
    const lockPath = join(absoluteRunDirectory, runLockFileName);
    try {
      await runMatrix(baseMatrix(), await fixtureCorpus(), runOptions(runDirectory));
      await Bun.write(statePath, "not valid JSON\n");
      await writeStaleLock(lockPath);

      const error = await rejectedError(runMatrix(baseMatrix(), await fixtureCorpus(), runOptions(runDirectory)));
      expect(error.message).toBe(
        "stale benchmark lock requires operator intervention because run state may still be active",
      );
      expect(error.message).not.toContain(runDirectory);
      expect(error.message).not.toContain(absoluteRunDirectory);
      expect(await Bun.file(lockPath).exists()).toBe(true);
    } finally {
      await rm(absoluteRunDirectory, { recursive: true, force: true });
    }
  });
});

function baseMatrix(model = "pass"): BenchmarkMatrix {
  return {
    schema_version: 1,
    matrix_id: "run-lock-test",
    description: "Run lock test",
    execution_policy: "local_ci",
    attempt_order: "entry-fixture-repetition",
    concurrency: 1,
    retries: 0,
    timeout_seconds: 10,
    fixture_ids: ["fake-fixture"],
    entries: [{
      id: "fake-entry",
      provider: "local",
      model,
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
    corpus_id: "run-lock-test",
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

async function waitForFile(path: string): Promise<void> {
  const deadline = Date.now() + 3_000;
  while (!(await Bun.file(path).exists())) {
    if (Date.now() >= deadline) throw new Error("timed out waiting for benchmark run lock");
    await Bun.sleep(10);
  }
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

async function writeStaleLock(path: string): Promise<void> {
  const exited = Bun.spawn([process.execPath, "-e", ""], { stdout: "ignore", stderr: "ignore" });
  await exited.exited;
  await Bun.write(path, `${JSON.stringify({
    schema_version: "transcribeit.benchmark-run-lock.v1",
    pid: exited.pid,
    owner_token: randomUUID(),
    created_at_utc: new Date().toISOString(),
  })}\n`);
  await chmod(path, 0o600);
}

function uniqueRunDirectory(label: string): string {
  return `output/benchmarks/test-run-lock-${label}-${process.pid}-${Math.random().toString(16).slice(2)}`;
}

function restoreEnvironment(name: string, previous: string | undefined): void {
  if (previous === undefined) delete process.env[name];
  else process.env[name] = previous;
}
