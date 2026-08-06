import { chmod, mkdir, readdir, rm, stat } from "node:fs/promises";
import { join, resolve, sep } from "node:path";

import { repositoryRoot, resolveFixtures } from "./config";
import { binaryIdentity, captureEnvironment } from "./environment";
import { loadRunState, matrixSha256, sha256, writeJsonAtomic } from "./state";
import type { AttemptRecord, BenchmarkMatrix, FixtureIdentity, MatrixEntry, RunState } from "./types";
import type { CorpusManifest } from "../corpus";

type RunOptions = {
  runDirectory: string;
  binary: string;
  allowHosted: boolean;
  retryFailures: boolean;
  keepAttemptOutputs: boolean;
};

const ownedOptions = new Set(["--provider", "--model", "--remote-model", "--input", "--output-dir", "--max-retries", "--request-timeout-secs"]);

export async function runMatrix(matrix: BenchmarkMatrix, corpus: CorpusManifest, options: RunOptions): Promise<RunState> {
  const repository = repositoryRoot();
  const runDirectory = safeRunDirectory(options.runDirectory);
  const statePath = join(runDirectory, "state.json");
  const attemptsRoot = join(runDirectory, "attempts");
  await mkdir(attemptsRoot, { recursive: true, mode: 0o700 });
  await chmod(runDirectory, 0o700);
  if (matrix.entries.some((entry) => entry.execution === "hosted") && !options.allowHosted) {
    throw new Error("matrix contains hosted entries; rerun with --allow-hosted after reviewing provider cost and scope");
  }
  const fixtures = resolveFixtures(matrix, corpus);
  await verifyFixtures(fixtures);
  const binary = await binaryIdentity(options.binary);
  const expectedHash = matrixSha256(matrix);
  let state: RunState;
  if (await Bun.file(statePath).exists()) {
    state = await loadRunState(statePath);
    if (state.matrix_sha256 !== expectedHash) throw new Error("existing run state belongs to a different matrix");
    if (state.binary_classification !== binary.classification) {
      throw new Error("existing run state uses a different binary classification");
    }
    if (state.binary_sha256 !== binary.sha256) throw new Error("existing run state uses a different binary hash");
    if (state.keep_attempt_outputs !== options.keepAttemptOutputs) {
      throw new Error("existing run state uses a different attempt-output retention policy");
    }
    for (const attempt of state.attempts) if (attempt.status === "running") attempt.status = "pending";
  } else {
    const commit = Bun.spawnSync(["git", "rev-parse", "HEAD"], { cwd: repository }).stdout.toString().trim();
    const dirty = Bun.spawnSync(["git", "status", "--porcelain"], { cwd: repository }).stdout.length > 0;
    state = {
      schema_version: "transcribeit.benchmark-run-state.v1",
      matrix,
      matrix_sha256: expectedHash,
      producing_commit: { hash: commit, worktree_dirty: dirty },
      started_at_utc: new Date().toISOString(),
      updated_at_utc: new Date().toISOString(),
      binary_classification: binary.classification,
      binary_sha256: binary.sha256,
      environment: captureEnvironment(),
      keep_attempt_outputs: options.keepAttemptOutputs,
      attempts: buildAttempts(matrix, fixtures),
    };
    await writeJsonAtomic(statePath, state);
  }

  let saveQueue = Promise.resolve();
  const save = (): Promise<void> => {
    state.updated_at_utc = new Date().toISOString();
    saveQueue = saveQueue.then(() => writeJsonAtomic(statePath, state));
    return saveQueue;
  };
  const queue = state.attempts.filter((attempt) => {
    if (attempt.status === "passed" || attempt.status === "skipped") return false;
    if (attempt.status === "failed" && !options.retryFailures) return false;
    return true;
  });
  let cursor = 0;
  const workers = Array.from({ length: Math.min(matrix.concurrency, queue.length) }, async () => {
    while (cursor < queue.length) {
      const attempt = queue[cursor++];
      const entry = matrix.entries.find((candidate) => candidate.id === attempt.entry_id);
      const fixture = fixtures.get(attempt.fixture.id);
      if (!entry || !fixture) throw new Error(`attempt ${attempt.key} no longer resolves to its matrix entry and fixture`);
      await executeAttempt(state, attempt, entry, fixture, attemptsRoot, options, save);
    }
  });
  await Promise.all(workers);
  await saveQueue;
  return state;
}

async function executeAttempt(
  state: RunState,
  attempt: AttemptRecord,
  entry: MatrixEntry,
  fixture: FixtureIdentity,
  attemptsRoot: string,
  options: RunOptions,
  save: () => Promise<void>,
): Promise<void> {
  const missing = entry.required_env.filter((name) => !process.env[name]);
  if (missing.length) {
    Object.assign(attempt, {
      status: "skipped",
      started_at_utc: new Date().toISOString(),
      completed_at_utc: new Date().toISOString(),
      error_category: "unconfigured",
    });
    await save();
    console.log(`${attempt.key}: skipped (unconfigured)`);
    return;
  }
  attempt.status = "running";
  attempt.started_at_utc = new Date().toISOString();
  attempt.completed_at_utc = null;
  attempt.error_category = null;
  await save();

  const outputDirectory = join(attemptsRoot, attempt.key);
  assertChildPath(outputDirectory, attemptsRoot);
  await rm(outputDirectory, { recursive: true, force: true });
  await mkdir(outputDirectory, { recursive: true, mode: 0o700 });
  const command = buildCommand(options.binary, state.matrix, entry, fixture, outputDirectory);
  console.log(`${attempt.key}: running`);
  const started = performance.now();
  const child = Bun.spawn(command, { cwd: repositoryRoot(), env: process.env, stdout: "pipe", stderr: "pipe" });
  const stdoutPromise = new Response(child.stdout).text();
  const stderrPromise = new Response(child.stderr).text();
  const exitCode = await child.exited;
  const wallMs = performance.now() - started;
  await stdoutPromise;
  const stderr = await stderrPromise;
  attempt.wall_ms = wallMs;
  attempt.real_time_factor = wallMs / 1000 / fixture.duration_seconds;
  attempt.completed_at_utc = new Date().toISOString();
  if (exitCode !== 0) {
    attempt.status = "failed";
    attempt.error_category = classifyError(stderr);
  } else {
    try {
      await collectSuccess(attempt, outputDirectory);
      attempt.status = "passed";
    } catch (error) {
      attempt.status = "failed";
      attempt.error_category = "local_output_invalid";
    }
  }
  if (!options.keepAttemptOutputs) await rm(outputDirectory, { recursive: true, force: true });
  await save();
  console.log(`${attempt.key}: ${attempt.status} (${Math.round(wallMs)} ms${attempt.error_category ? `, ${attempt.error_category}` : ""})`);
}

async function collectSuccess(attempt: AttemptRecord, outputDirectory: string): Promise<void> {
  const files = await readdir(outputDirectory);
  const manifestName = files.find((name) => name.endsWith(".manifest.json"));
  const outputName = files.find((name) => name.endsWith(".txt"));
  if (!manifestName || !outputName) throw new Error("expected text and manifest output");
  const manifestText = await Bun.file(join(outputDirectory, manifestName)).text();
  const output = await Bun.file(join(outputDirectory, outputName)).arrayBuffer();
  const manifest = JSON.parse(manifestText);
  attempt.output_sha256 = sha256(new Uint8Array(output));
  attempt.manifest_sha256 = sha256(manifestText);
  const capabilityNames = ["segments", "word_timestamps", "speaker_labels", "language_per_segment", "emotion_per_segment", "native_timestamps"];
  attempt.capabilities = Object.fromEntries(capabilityNames.map((name) => [name, manifest.capabilities?.[name] === true]));
  attempt.quality = {
    timing_source: safeEnum(manifest.quality?.timing_source, ["provider_native", "model_generated", "synthetic", "none"]),
    timing_reliable: typeof manifest.quality?.timing_reliable === "boolean" ? manifest.quality.timing_reliable : null,
    timestamps_clamped: typeof manifest.quality?.timestamps_clamped === "boolean" ? manifest.quality.timestamps_clamped : null,
    speaker_source: safeEnum(manifest.quality?.speaker_source, ["provider_native", "model_generated", "local_postprocess", "none"]),
    warning_count: Array.isArray(manifest.quality?.warnings) ? manifest.quality.warnings.length : 0,
  };
  attempt.remote_cleanup = cleanupClassification(attempt.provider, manifest);
}

export function buildCommand(binary: string, matrix: BenchmarkMatrix, entry: MatrixEntry, fixture: FixtureIdentity, outputDirectory: string): string[] {
  for (const arg of entry.args) if (ownedOptions.has(arg.split("=", 1)[0])) throw new Error(`entry ${entry.id} overrides harness-owned option`);
  const command = [binary, "run", "--provider", entry.provider];
  if (entry.provider === "local") command.push("--model", entry.model);
  else if (entry.provider !== "azure" && entry.provider !== "nvidia-riva") command.push("--remote-model", entry.model);
  command.push(
    "--max-retries",
    String(matrix.retries),
    "--request-timeout-secs",
    String(matrix.timeout_seconds),
    "--input",
    fixture.path,
    "--output-format",
    "text",
    "--output-dir",
    outputDirectory,
    ...entry.args,
  );
  return command;
}

export function commandTemplate(matrix: BenchmarkMatrix, entry: MatrixEntry): string {
  const model = entry.provider === "local"
    ? ` --model ${entry.model}`
    : entry.provider === "azure" || entry.provider === "nvidia-riva"
      ? ""
      : ` --remote-model ${entry.model}`;
  return `transcribeit run --provider ${entry.provider}${model} --max-retries ${matrix.retries} --request-timeout-secs ${matrix.timeout_seconds} --input $FIXTURE --output-format text --output-dir $ATTEMPT_DIR${entry.args.length ? ` ${entry.args.join(" ")}` : ""}`;
}

function buildAttempts(matrix: BenchmarkMatrix, fixtures: Map<string, FixtureIdentity>): AttemptRecord[] {
  const attempts: AttemptRecord[] = [];
  for (const entry of matrix.entries) {
    const ids = entry.fixture_ids ?? matrix.fixture_ids;
    for (const id of ids) {
      const fixture = fixtures.get(id);
      if (!fixture) throw new Error(`fixture ${id} is not available`);
      for (let repetition = 1; repetition <= entry.repetitions; repetition++) {
        attempts.push({
          key: `${entry.id}--${fixture.id}--${String(repetition).padStart(3, "0")}`,
          entry_id: entry.id,
          provider: entry.provider,
          model: entry.model,
          execution: entry.execution,
          cache_state: entry.cache_state,
          fixture: { id: fixture.id, duration_seconds: fixture.duration_seconds, bytes: fixture.bytes, sha256: fixture.sha256 },
          repetition,
          status: "pending",
          started_at_utc: null,
          completed_at_utc: null,
          wall_ms: null,
          real_time_factor: null,
          error_category: null,
          output_sha256: null,
          manifest_sha256: null,
          capabilities: null,
          quality: null,
          remote_cleanup: null,
        });
      }
    }
  }
  return attempts;
}

async function verifyFixtures(fixtures: Map<string, FixtureIdentity>): Promise<void> {
  for (const fixture of fixtures.values()) {
    const path = resolve(repositoryRoot(), fixture.path);
    const metadata = await stat(path).catch(() => undefined);
    if (!metadata || metadata.size !== fixture.bytes) throw new Error(`fixture ${fixture.id} is missing or has the wrong byte size; run corpus fetch/verify`);
    const digest = Bun.spawnSync(["shasum", "-a", "256", path]).stdout.toString().trim().split(/\s+/)[0];
    if (digest !== fixture.sha256) throw new Error(`fixture ${fixture.id} failed SHA-256 verification`);
  }
}

function safeRunDirectory(value: string): string {
  const repository = repositoryRoot();
  const allowed = join(repository, "output/benchmarks");
  const target = resolve(repository, value);
  if (target === allowed || !target.startsWith(`${allowed}${sep}`)) throw new Error("run directory must be a named child of output/benchmarks");
  return target;
}

function assertChildPath(target: string, root: string): void {
  const resolvedTarget = resolve(target);
  const resolvedRoot = resolve(root);
  if (!resolvedTarget.startsWith(`${resolvedRoot}${sep}`)) throw new Error("attempt output escaped its run directory");
}

function safeEnum(value: unknown, allowed: string[]): string | null {
  return typeof value === "string" && allowed.includes(value) ? value : null;
}

function cleanupClassification(provider: string, manifest: any): AttemptRecord["remote_cleanup"] {
  if (provider === "qwen-filetrans" || provider === "deepgram") {
    const cleanup = manifest?.provider_metadata?.data?.staging?.cleanup;
    if (!cleanup) return provider === "deepgram" ? "not_applicable" : null;
    if (cleanup.attempted === true && cleanup.deleted === true && cleanup.error == null) return "deleted";
    return cleanup.attempted === false ? "not_attempted" : "failed";
  }
  if (provider === "gemini") {
    const file = manifest?.provider_metadata?.data?.file;
    if (!file) return null;
    if (file.delete_attempted === true && file.deleted === true && file.delete_error == null) return "deleted";
    return file.delete_attempted === false ? "not_attempted" : "failed";
  }
  return "not_applicable";
}

export function classifyError(stderr: string): string {
  const value = stderr.toLowerCase();
  if (/\b(401|403|unauthorized|forbidden|invalid api key|authentication)\b/.test(value)) return "authentication";
  if (/\b429\b|rate.?limit|quota/.test(value)) return "rate_limit";
  if (/timed? out|timeout|deadline exceeded/.test(value)) return "timeout";
  if (/\b5\d\d\b|internal server|service unavailable/.test(value)) return "provider_5xx";
  if (/unsupported|not supported|unimplemented/.test(value)) return "unsupported";
  if (/malformed|failed to parse|invalid json|invalid response/.test(value)) return "malformed_response";
  if (/\b400\b|bad request|reject/.test(value)) return "rejected_request";
  if (/dns|connect|transport|network|certificate|tls|grpc status/.test(value)) return "transport";
  return "provider_error";
}
