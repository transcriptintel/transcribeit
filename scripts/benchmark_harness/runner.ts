import { join, resolve } from "node:path";

import { collectAttemptSuccess, parsePeakRss } from "./attempt_output";
import { expectedAttemptIdentities, type ExpectedAttemptIdentity } from "./attempt_plan";
import { childEnvironment } from "./child_environment";
import { repositoryRoot, resolveFixtures } from "./config";
import { binaryIdentity, captureEnvironment } from "./environment";
import { verifyAllRunInputs, verifyAttemptInputs, verifyInitialFixtures } from "./input_integrity";
import { captureRepositoryProvenance } from "./provenance";
import { reconcileAttemptPlan, sameEnvironment, sameProvenance } from "./resume_binding";
import {
  ensureDirectDirectory,
  evaluationRootBinding,
  prepareRunDirectory,
  removeDirectDirectory,
  type EvaluationRootBinding,
} from "./runner_paths";
import { loadRunState, matrixSha256, writeJsonAtomic } from "./state";
import { ScoringIntegrityError } from "./scoring";
import { SubprocessSupervisor } from "./subprocess";
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
type AttemptCollector = typeof collectAttemptSuccess;

export async function runMatrix(matrix: BenchmarkMatrix, corpus: CorpusManifest, options: RunOptions, collector: AttemptCollector = collectAttemptSuccess): Promise<RunState> {
  const { runDirectory, attemptsRoot, lock } = await prepareRunDirectory(options.runDirectory);
  let invocationFailed = false;
  try {
    return await runMatrixLocked(matrix, corpus, options, collector, runDirectory, attemptsRoot);
  } catch (error) {
    invocationFailed = true;
    throw error;
  } finally {
    try {
      await lock.release();
    } catch (error) {
      if (!invocationFailed) throw error;
    }
  }
}

async function runMatrixLocked(
  matrix: BenchmarkMatrix,
  corpus: CorpusManifest,
  options: RunOptions,
  collector: AttemptCollector,
  runDirectory: string,
  attemptsRoot: string,
): Promise<RunState> {
  const evaluationRoot = await evaluationRootBinding(matrix, runDirectory);
  const statePath = join(runDirectory, "state.json");
  if (matrix.entries.some((entry) => entry.execution === "hosted") && !options.allowHosted) {
    throw new Error("matrix contains hosted entries; rerun with --allow-hosted after reviewing provider cost and scope");
  }
  const fixtures = resolveFixtures(matrix, corpus);
  await verifyInitialFixtures(fixtures);
  const binary = await binaryIdentity(options.binary);
  const expectedHash = matrixSha256(matrix);
  const expectedAttempts = buildAttempts(matrix, fixtures);
  const provenance = await captureRepositoryProvenance();
  const environment = captureEnvironment();
  let state: RunState;
  if (await Bun.file(statePath).exists()) {
    state = await loadRunState(statePath);
    if (state.matrix_sha256 !== expectedHash) throw new Error("existing run state belongs to a different matrix");
    if (state.binary_classification !== binary.classification) {
      throw new Error("existing run state uses a different binary classification");
    }
    if (state.binary_sha256 !== binary.sha256) throw new Error("existing run state uses a different binary hash");
    if (state.evaluation_root_sha256 !== (evaluationRoot?.identitySha256 ?? null)) {
      throw new Error("existing run state uses a different evaluation model root identity");
    }
    if (!sameProvenance(state.producing_commit, provenance)) {
      throw new Error("existing run state uses a different repository HEAD or worktree fingerprint");
    }
    if (!sameEnvironment(state.environment, environment)) {
      throw new Error("existing run state uses a different captured environment");
    }
    if (state.keep_attempt_outputs !== options.keepAttemptOutputs) {
      throw new Error("existing run state uses a different attempt-output retention policy");
    }
    reconcileAttemptPlan(state.attempts, expectedAttempts);
    for (const attempt of state.attempts) if (attempt.status === "running") attempt.status = "pending";
  } else {
    state = {
      schema_version: "transcribeit.benchmark-run-state.v1",
      matrix,
      matrix_sha256: expectedHash,
      producing_commit: provenance,
      started_at_utc: new Date().toISOString(),
      updated_at_utc: new Date().toISOString(),
      binary_classification: binary.classification,
      binary_sha256: binary.sha256,
      evaluation_root_sha256: evaluationRoot?.identitySha256 ?? null,
      environment,
      keep_attempt_outputs: options.keepAttemptOutputs,
      attempts: expectedAttempts,
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
  const supervisor = new SubprocessSupervisor();
  try {
    let hasWorkerError = false;
    let firstWorkerError: unknown;
    const recordWorkerError = (error: unknown): void => {
      if (supervisor.interruptedSignal) return;
      if (!hasWorkerError) firstWorkerError = error;
      hasWorkerError = true;
      supervisor.abort();
    };
    const workers = Array.from({ length: Math.min(matrix.concurrency, queue.length) }, async () => {
      try {
        while (cursor < queue.length && !supervisor.interruptedSignal && !supervisor.internallyAborted) {
          const attempt = queue[cursor++];
          const entry = matrix.entries.find((candidate) => candidate.id === attempt.entry_id);
          const fixture = fixtures.get(attempt.fixture.id);
          if (!entry || !fixture) throw new Error("planned attempt no longer resolves to its matrix entry and fixture");
          await executeAttempt(
            state,
            attempt,
            entry,
            fixture,
            attemptsRoot,
            runDirectory,
            binary,
            evaluationRoot,
            options,
            supervisor,
            save,
            collector,
          );
        }
      } catch (error) {
        recordWorkerError(error);
        throw error;
      }
    });
    await Promise.allSettled(workers);
    let finalInputError: unknown;
    try {
      await verifyAllRunInputs(binary, fixtures);
    } catch (error) {
      finalInputError = error;
    }
    try {
      await saveQueue;
    } catch (error) {
      if (!supervisor.interruptedSignal && !hasWorkerError) {
        firstWorkerError = error;
        hasWorkerError = true;
      }
    }
    if (hasWorkerError) throw firstWorkerError;
    if (supervisor.interruptedSignal) throw new Error(`benchmark interrupted by ${supervisor.interruptedSignal}`);
    if (finalInputError) throw finalInputError;
    if (!sameProvenance(provenance, await captureRepositoryProvenance())) {
      throw new Error("repository HEAD or worktree changed during the benchmark run");
    }
    return state;
  } finally {
    supervisor.dispose();
  }
}

async function executeAttempt(
  state: RunState,
  attempt: AttemptRecord,
  entry: MatrixEntry,
  fixture: FixtureIdentity,
  attemptsRoot: string,
  runDirectory: string,
  binary: { path: string; sha256: string },
  evaluationRoot: EvaluationRootBinding,
  options: RunOptions,
  supervisor: SubprocessSupervisor,
  save: () => Promise<void>,
  collector: AttemptCollector,
): Promise<void> {
  resetAttemptOutcome(attempt);
  const outputDirectory = join(attemptsRoot, attempt.key);
  await removeDirectDirectory(outputDirectory, attemptsRoot, "attempt output directory");
  const missing = entry.required_env.filter((name) => !process.env[name]);
  if (missing.length) {
    const now = new Date().toISOString();
    Object.assign(attempt, {
      status: "skipped",
      started_at_utc: now,
      completed_at_utc: now,
      error_category: "unconfigured",
    });
    await save();
    await verifyAttemptInputs(binary, fixture);
    console.log(`${attempt.key}: skipped (unconfigured)`);
    return;
  }
  attempt.status = "running";
  attempt.started_at_utc = new Date().toISOString();
  await save();

  let wallMs = 0;
  let cancelled = false;
  let executionError: unknown;
  try {
    await ensureDirectDirectory(outputDirectory, attemptsRoot, "attempt output directory");
    const command = buildCommand(binary.path, state.matrix, entry, fixture, outputDirectory);
    console.log(`${attempt.key}: running`);
    const measuredCommand = state.matrix.measurements.peak_rss && process.platform === "darwin"
      ? ["/usr/bin/time", "-l", ...command]
      : command;
    await verifyAttemptInputs(binary, fixture);
    const result = await supervisor.run(measuredCommand, {
      cwd: runDirectory,
      env: childEnvironment(entry, evaluationRoot),
      timeoutMs: state.matrix.timeout_seconds * 1000,
    });
    wallMs = result.wallMs;
    if (result.interrupted || result.aborted) {
      cancelled = true;
      resetAttemptOutcome(attempt);
    } else {
      attempt.wall_ms = wallMs;
      attempt.peak_rss_bytes = state.matrix.measurements.peak_rss ? parsePeakRss(result.stderr) : null;
      attempt.real_time_factor = wallMs / 1000 / fixture.duration_seconds;
      attempt.completed_at_utc = new Date().toISOString();
    }
    if (result.interrupted || result.aborted) {
      // Leave the attempt resumable without requiring --retry-failures.
    } else if (result.timedOut) {
      attempt.status = "failed";
      attempt.error_category = "timeout";
    } else if (result.exitCode !== 0) {
      attempt.status = "failed";
      attempt.error_category = classifyError(result.stderr);
    } else {
      try {
        await collector(
          attempt,
          outputDirectory,
          result.stderr,
          state.matrix.measurements.reference_scoring,
        );
        attempt.status = "passed";
      } catch (error) {
        if (error instanceof ScoringIntegrityError) { resetAttemptOutcome(attempt); throw error; }
        attempt.status = "failed";
        attempt.error_category = "local_output_invalid";
      }
    }
  } catch (error) {
    executionError = error;
  } finally {
    if (cancelled || !options.keepAttemptOutputs) {
      try {
        await removeDirectDirectory(outputDirectory, attemptsRoot, "attempt output directory");
      } catch (error) {
        executionError ??= error;
      }
    }
    try {
      await verifyAttemptInputs(binary, fixture);
    } catch (error) {
      executionError ??= error;
    }
  }
  if (executionError) throw executionError;
  await save();
  console.log(`${attempt.key}: ${attempt.status} (${Math.round(wallMs)} ms${attempt.error_category ? `, ${attempt.error_category}` : ""})`);
}

export function buildCommand(binary: string, matrix: BenchmarkMatrix, entry: MatrixEntry, fixture: FixtureIdentity, outputDirectory: string): string[] {
  for (const arg of entry.args) if (ownedOptions.has(arg.split("=", 1)[0])) throw new Error(`entry ${entry.id} overrides harness-owned option`);
  const repository = repositoryRoot();
  const command = [resolve(repository, binary), "run", "--provider", entry.provider];
  if (entry.provider === "local") command.push("--model", entry.model);
  else if (!["apple-speech", "azure", "nvidia-riva"].includes(entry.provider)) command.push("--remote-model", entry.model);
  command.push(
    "--max-retries",
    String(matrix.retries),
    "--request-timeout-secs",
    String(matrix.timeout_seconds),
    "--input",
    resolve(repository, fixture.path),
    "--output-format",
    "text",
    "--output-dir",
    resolve(repository, outputDirectory),
    ...entry.args,
  );
  return command;
}

export function commandTemplate(matrix: BenchmarkMatrix, entry: MatrixEntry): string {
  const model = entry.provider === "local"
    ? ` --model ${entry.model}`
    : ["apple-speech", "azure", "nvidia-riva"].includes(entry.provider)
      ? ""
      : ` --remote-model ${entry.model}`;
  return `transcribeit run --provider ${entry.provider}${model} --max-retries ${matrix.retries} --request-timeout-secs ${matrix.timeout_seconds} --input $FIXTURE --output-format text --output-dir $ATTEMPT_DIR${entry.args.length ? ` ${entry.args.join(" ")}` : ""}`;
}

export function buildAttempts(matrix: BenchmarkMatrix, fixtures: Map<string, FixtureIdentity>): AttemptRecord[] {
  return expectedAttemptIdentities(matrix).map((identity) => {
    const entry = matrix.entries.find((candidate) => candidate.id === identity.entry_id);
    const fixture = fixtures.get(identity.fixture_id);
    if (!entry || !fixture) throw new Error(`planned attempt ${identity.key} does not resolve to an entry and fixture`);
    return newAttempt(identity, entry, fixture);
  });
}

function newAttempt(identity: ExpectedAttemptIdentity, entry: MatrixEntry, fixture: FixtureIdentity): AttemptRecord {
  return {
    key: identity.key,
    entry_id: entry.id,
    provider: entry.provider,
    model: entry.model,
    execution: entry.execution,
    cache_state: entry.cache_state,
    fixture: { id: fixture.id, duration_seconds: fixture.duration_seconds, bytes: fixture.bytes, sha256: fixture.sha256 },
    repetition: identity.repetition,
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

export function resetAttemptOutcome(attempt: AttemptRecord): void {
  Object.assign(attempt, {
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
  });
}

export { parsePeakRss } from "./attempt_output";

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
