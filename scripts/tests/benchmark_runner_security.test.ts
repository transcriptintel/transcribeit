import { chmod, mkdir, rm, stat, unlink } from "node:fs/promises";
import { basename, join, relative } from "node:path";

import { repositoryRoot } from "../benchmark_harness/config";
import { buildAttempts, runMatrix } from "../benchmark_harness/runner";
import { sha256, writeJsonAtomic } from "../benchmark_harness/state";
import { SubprocessSupervisor } from "../benchmark_harness/subprocess";
import type { BenchmarkMatrix, FixtureIdentity } from "../benchmark_harness/types";
import type { CorpusManifest } from "../corpus";

const repository = repositoryRoot();
const fakeBinary = join(repository, "scripts/tests/fixtures/fake_transcribeit.ts");
const fixturePath = "scripts/tests/fixtures/benchmark-input.txt";

describe("benchmark runner isolation", () => {
  test("blocks parent dotenv and ambient provider values with a private empty sentinel", async () => {
    await chmod(fakeBinary, 0o755);
    const parentDotenv = join(repository, "output/benchmarks/.env");
    const parentExisted = await Bun.file(parentDotenv).exists();
    const previousSecret = process.env.FAKE_PARENT_DOTENV_SECRET;
    if (!parentExisted) await Bun.write(parentDotenv, "FAKE_PARENT_DOTENV_SECRET=from-parent-dotenv\n");
    process.env.FAKE_PARENT_DOTENV_SECRET = "from-ambient-environment";
    const runDirectory = uniqueRunDirectory("dotenv");
    const absoluteRunDirectory = join(repository, runDirectory);
    try {
      const state = await runMatrix(baseMatrix("dotenv-probe"), await fixtureCorpus(), runOptions(runDirectory));
      expect(state.attempts[0].status).toBe("passed");
      const sentinel = join(absoluteRunDirectory, ".env");
      expect(await Bun.file(sentinel).text()).toBe("");
      expect((await stat(sentinel)).mode & 0o777).toBe(0o600);
    } finally {
      restoreEnvironment("FAKE_PARENT_DOTENV_SECRET", previousSecret);
      if (!parentExisted) await unlink(parentDotenv).catch(() => undefined);
      await rm(absoluteRunDirectory, { recursive: true, force: true });
    }
  });

  test("drains verbose stderr without changing a successful attempt", async () => {
    await chmod(fakeBinary, 0o755);
    const runDirectory = uniqueRunDirectory("stderr");
    const absoluteRunDirectory = join(repository, runDirectory);
    try {
      const state = await runMatrix(baseMatrix("stderr-spam"), await fixtureCorpus(), runOptions(runDirectory));
      expect(state.attempts[0].status).toBe("passed");
    } finally {
      await rm(absoluteRunDirectory, { recursive: true, force: true });
    }
  });

  test("rejects selected fixtures with no eligible entry", () => {
    const matrix = baseMatrix();
    matrix.fixture_ids.push("orphan-fixture");
    matrix.entries[0].fixture_ids = ["fake-fixture"];
    const first = fixtureIdentity("fake-fixture");
    const orphan = fixtureIdentity("orphan-fixture");
    expect(() => buildAttempts(matrix, new Map([[first.id, first], [orphan.id, orphan]]))).toThrow(
      "orphan-fixture has no eligible entry",
    );
  });

  test("resumes an interrupted attempt but rejects static attempt-plan drift", async () => {
    await chmod(fakeBinary, 0o755);
    const matrix = baseMatrix();
    const corpus = await fixtureCorpus();
    const runDirectory = uniqueRunDirectory("resume-plan");
    const absoluteRunDirectory = join(repository, runDirectory);
    const statePath = join(absoluteRunDirectory, "state.json");
    try {
      const first = await runMatrix(matrix, corpus, runOptions(runDirectory));
      first.attempts[0].status = "running";
      first.attempts[0].output_sha256 = "a".repeat(64);
      await writeJsonAtomic(statePath, first);
      const resumed = await runMatrix(matrix, corpus, runOptions(runDirectory));
      expect(resumed.attempts[0].status).toBe("passed");
      expect(resumed.attempts[0].output_sha256).not.toBe("a".repeat(64));

      resumed.attempts[0].fixture.bytes += 1;
      await writeJsonAtomic(statePath, resumed);
      await expect(runMatrix(matrix, corpus, runOptions(runDirectory))).rejects.toThrow("does not match the current plan");
    } finally {
      await rm(absoluteRunDirectory, { recursive: true, force: true });
    }
  });

  test("rejects a resume whose repository provenance binding was altered", async () => {
    await chmod(fakeBinary, 0o755);
    const matrix = baseMatrix();
    const corpus = await fixtureCorpus();
    const runDirectory = uniqueRunDirectory("provenance");
    const absoluteRunDirectory = join(repository, runDirectory);
    const statePath = join(absoluteRunDirectory, "state.json");
    try {
      const first = await runMatrix(matrix, corpus, runOptions(runDirectory));
      first.producing_commit.worktree_fingerprint_sha256 = "f".repeat(64);
      await writeJsonAtomic(statePath, first);
      await expect(runMatrix(matrix, corpus, runOptions(runDirectory))).rejects.toThrow("worktree fingerprint");
    } finally {
      await rm(absoluteRunDirectory, { recursive: true, force: true });
    }
  });

  test("cancellation kills the active POSIX process group and its grandchild", async () => {
    if (process.platform === "win32") return;
    await chmod(fakeBinary, 0o755);
    const suffix = `${process.pid}-${Math.random().toString(16).slice(2)}`;
    const root = join(repository, `output/benchmarks/test-supervisor-cancel-${suffix}`);
    const outputDirectory = join(root, "attempt");
    const marker = join(root, "grandchild-survived.marker");
    await mkdir(root, { mode: 0o700 });
    const supervisor = new SubprocessSupervisor();
    try {
      const running = supervisor.run(
        [fakeBinary, "run", "--model", "timeout", "--output-dir", outputDirectory],
        {
          cwd: root,
          env: {
            PATH: process.env.PATH ?? "/usr/bin:/bin",
            FAKE_GRANDCHILD_MARKER_PATH: marker,
            FAKE_GRANDCHILD_DELAY_MS: "1200",
          },
          timeoutMs: 10_000,
        },
      );
      await Bun.sleep(150);
      supervisor.cancel("SIGINT");
      const result = await running;
      expect(result.interrupted).toBe(true);
      expect(result.aborted).toBe(false);
      expect(result.timedOut).toBe(false);
      await Bun.sleep(1_300);
      expect(await Bun.file(marker).exists()).toBe(false);
    } finally {
      supervisor.dispose();
      await rm(root, { recursive: true, force: true });
    }
  });

  test("rejects post-attempt fixture drift without disclosing its path", async () => {
    const suffix = `${process.pid}-${Math.random().toString(16).slice(2)}`;
    const runDirectory = uniqueRunDirectory("fixture-drift");
    const absoluteRunDirectory = join(repository, runDirectory);
    const fixture = join(repository, `output/benchmarks/test-runner-fixture-${suffix}.txt`);
    const binary = await createMutatingBinary(suffix);
    const previousTarget = process.env.FAKE_MUTATE_FIXTURE_PATH;
    await Bun.write(fixture, "stable fixture\n");
    process.env.FAKE_MUTATE_FIXTURE_PATH = fixture;
    const matrix = baseMatrix("mutate-fixture");
    matrix.entries[0].required_env = ["FAKE_MUTATE_FIXTURE_PATH"];
    try {
      const error = await rejectedError(
        runMatrix(matrix, await corpusForFixture(fixture), { ...runOptions(runDirectory), binary }),
      );
      expect(error.message).toBe("benchmark fixture identity changed during run");
      expect(error.message).not.toContain(basename(fixture));
      expect(error.message).not.toContain(fixture);
    } finally {
      restoreEnvironment("FAKE_MUTATE_FIXTURE_PATH", previousTarget);
      await rm(absoluteRunDirectory, { recursive: true, force: true });
      await rm(fixture, { force: true });
      await rm(binary, { force: true });
    }
  });

  test("aborts peer process groups and settles workers before returning binary drift", async () => {
    if (process.platform === "win32") return;
    const suffix = `${process.pid}-${Math.random().toString(16).slice(2)}`;
    const runDirectory = uniqueRunDirectory("worker-abort");
    const absoluteRunDirectory = join(repository, runDirectory);
    const binary = await createMutatingBinary(suffix);
    const marker = join(repository, `output/benchmarks/test-runner-grandchild-${suffix}.marker`);
    const previousBinaryTarget = process.env.FAKE_MUTATE_BINARY_PATH;
    const previousMarker = process.env.FAKE_GRANDCHILD_MARKER_PATH;
    const previousDelay = process.env.FAKE_GRANDCHILD_DELAY_MS;
    process.env.FAKE_MUTATE_BINARY_PATH = binary;
    process.env.FAKE_GRANDCHILD_MARKER_PATH = marker;
    process.env.FAKE_GRANDCHILD_DELAY_MS = "1200";
    const matrix = baseMatrix();
    matrix.concurrency = 2;
    matrix.entries = [
      {
        ...matrix.entries[0],
        id: "drift-entry",
        model: "mutate-binary",
        required_env: ["FAKE_MUTATE_BINARY_PATH"],
      },
      {
        ...matrix.entries[0],
        id: "slow-entry",
        model: "timeout",
        required_env: ["FAKE_GRANDCHILD_MARKER_PATH", "FAKE_GRANDCHILD_DELAY_MS"],
      },
    ];
    try {
      const error = await rejectedError(
        runMatrix(matrix, await fixtureCorpus(), { ...runOptions(runDirectory), binary }),
      );
      expect(error.message).toBe("benchmark binary identity changed during run");
      expect(error.message).not.toContain(basename(binary));
      expect(error.message).not.toContain(binary);
      await Bun.sleep(1_300);
      expect(await Bun.file(marker).exists()).toBe(false);
    } finally {
      restoreEnvironment("FAKE_MUTATE_BINARY_PATH", previousBinaryTarget);
      restoreEnvironment("FAKE_GRANDCHILD_MARKER_PATH", previousMarker);
      restoreEnvironment("FAKE_GRANDCHILD_DELAY_MS", previousDelay);
      await rm(absoluteRunDirectory, { recursive: true, force: true });
      await rm(binary, { force: true });
      await rm(marker, { force: true });
    }
  });
});

function baseMatrix(model = "pass"): BenchmarkMatrix {
  return {
    schema_version: 1,
    matrix_id: "runner-isolation",
    description: "Runner isolation test",
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

function uniqueRunDirectory(label: string): string {
  return `output/benchmarks/test-runner-security-${label}-${process.pid}-${Math.random().toString(16).slice(2)}`;
}

function fixtureIdentity(id: string): FixtureIdentity {
  return {
    id,
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
    corpus_id: "runner-isolation",
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

async function corpusForFixture(path: string): Promise<CorpusManifest> {
  const bytes = new Uint8Array(await Bun.file(path).arrayBuffer());
  return {
    schema_version: 1,
    corpus_id: "runner-drift",
    materialization_root: "output/benchmarks",
    sources: {},
    fixtures: [
      {
        id: "fake-fixture",
        audio: {
          path: relative(repository, path),
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

async function createMutatingBinary(suffix: string): Promise<string> {
  const path = join(repository, `output/benchmarks/test-runner-binary-${suffix}.ts`);
  const source = await Bun.file(fakeBinary).text();
  await Bun.write(
    path,
    `${source}\nif (model === "mutate-fixture" && process.env.FAKE_MUTATE_FIXTURE_PATH) {\n` +
      `  await Bun.write(process.env.FAKE_MUTATE_FIXTURE_PATH, "changed fixture\\n");\n` +
      `}\nif (model === "mutate-binary" && process.env.FAKE_MUTATE_BINARY_PATH) {\n` +
      `  await Bun.sleep(300);\n` +
      `  await Bun.write(process.env.FAKE_MUTATE_BINARY_PATH, "changed binary\\n");\n` +
      `}\n`,
  );
  await chmod(path, 0o700);
  return path;
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
