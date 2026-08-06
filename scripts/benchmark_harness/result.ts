import { resolve } from "node:path";

import { commandTemplate } from "./runner";
import { isMap, loadRunState, matrixSha256, writeJsonAtomic } from "./state";
import type { AttemptRecord, BenchmarkMatrix, PublishedResult, RunState } from "./types";

export async function publishResult(statePath: string, outputPath: string): Promise<PublishedResult> {
  const state = await loadRunState(statePath);
  const incomplete = state.attempts.filter((attempt) => attempt.status === "pending" || attempt.status === "running");
  if (incomplete.length) throw new Error(`cannot publish: ${incomplete.length} attempts are incomplete`);
  const result: PublishedResult = {
    schema_version: "transcribeit.benchmark-harness-result.v1",
    matrix_sha256: state.matrix_sha256,
    recorded_at_utc: new Date().toISOString(),
    classification: state.producing_commit.worktree_dirty ? "dirty_worktree_reference" : "clean_commit_benchmark",
    producing_commit: state.producing_commit,
    runtime: { binary_classification: state.binary_classification, binary_sha256: state.binary_sha256 },
    environment: state.environment,
    matrix: {
      ...state.matrix,
      entries: state.matrix.entries.map((entry) => ({ ...entry, command_template: commandTemplate(state.matrix, entry) })),
    },
    summary: summarize(state.attempts),
    attempts: state.attempts.map(sanitizeAttempt),
    sanitization: {
      credential_values_removed: true,
      stdout_stderr_removed: true,
      transcript_text_removed: true,
      request_ids_removed: true,
      signed_urls_removed: true,
      local_absolute_paths_removed: true,
      provider_metadata_allowlisted: true,
    },
  };
  const errors = validatePublishedResult(result);
  if (errors.length) throw new Error(errors.join("\n"));
  await writeJsonAtomic(resolve(outputPath), result);
  return result;
}

function sanitizeAttempt(attempt: AttemptRecord): AttemptRecord {
  const capabilityNames = new Set(["segments", "word_timestamps", "speaker_labels", "language_per_segment", "emotion_per_segment", "native_timestamps"]);
  const errorCategories = new Set(["unconfigured", "authentication", "rate_limit", "timeout", "provider_5xx", "unsupported", "malformed_response", "rejected_request", "transport", "provider_error", "local_output_invalid"]);
  return {
    key: attempt.key,
    entry_id: attempt.entry_id,
    provider: attempt.provider,
    model: attempt.model,
    execution: attempt.execution,
    cache_state: attempt.cache_state,
    fixture: {
      id: attempt.fixture.id,
      duration_seconds: attempt.fixture.duration_seconds,
      bytes: attempt.fixture.bytes,
      sha256: attempt.fixture.sha256,
    },
    repetition: attempt.repetition,
    status: attempt.status,
    started_at_utc: attempt.started_at_utc,
    completed_at_utc: attempt.completed_at_utc,
    wall_ms: attempt.wall_ms,
    real_time_factor: attempt.real_time_factor,
    error_category: attempt.error_category && errorCategories.has(attempt.error_category) ? attempt.error_category : attempt.error_category ? "provider_error" : null,
    output_sha256: attempt.output_sha256,
    manifest_sha256: attempt.manifest_sha256,
    capabilities: attempt.capabilities
      ? Object.fromEntries(Object.entries(attempt.capabilities).filter(([name, value]) => capabilityNames.has(name) && typeof value === "boolean"))
      : null,
    quality: attempt.quality
      ? {
          timing_source: attempt.quality.timing_source,
          timing_reliable: attempt.quality.timing_reliable,
          timestamps_clamped: attempt.quality.timestamps_clamped,
          speaker_source: attempt.quality.speaker_source,
          warning_count: attempt.quality.warning_count,
        }
      : null,
    remote_cleanup: attempt.remote_cleanup,
  };
}

export function summarize(attempts: AttemptRecord[]): PublishedResult["summary"] {
  return {
    attempts: attempts.length,
    passed: attempts.filter((attempt) => attempt.status === "passed").length,
    failed: attempts.filter((attempt) => attempt.status === "failed").length,
    skipped: attempts.filter((attempt) => attempt.status === "skipped").length,
  };
}

export function validatePublishedResult(value: unknown): string[] {
  const errors: string[] = [];
  if (!isMap(value)) return ["result must be a mapping"];
  if (value.schema_version !== "transcribeit.benchmark-harness-result.v1") errors.push("unsupported result schema_version");
  if (typeof value.matrix_sha256 !== "string" || !/^[a-f0-9]{64}$/.test(value.matrix_sha256)) {
    errors.push("invalid matrix_sha256");
  }
  if (!['clean_commit_benchmark', 'dirty_worktree_reference'].includes(String(value.classification))) errors.push("invalid result classification");
  if (!isMap(value.producing_commit) || typeof value.producing_commit.hash !== "string" || typeof value.producing_commit.worktree_dirty !== "boolean") {
    errors.push("invalid producing_commit");
  }
  if (!isMap(value.runtime) || typeof value.runtime.binary_classification !== "string" || !/^[a-f0-9]{64}$/.test(String(value.runtime.binary_sha256))) {
    errors.push("invalid runtime binary identity");
  }
  if (!isMap(value.environment) || !isMap(value.environment.machine) || !isMap(value.environment.tools)) errors.push("invalid environment metadata");
  if (!isMap(value.matrix) || !Array.isArray(value.matrix.entries)) errors.push("result matrix is invalid");
  else {
    const normalizedMatrix = {
      ...value.matrix,
      entries: value.matrix.entries.map((entry) => {
        if (!isMap(entry)) return entry;
        const { command_template: _commandTemplate, ...definition } = entry;
        return definition;
      }),
    } as BenchmarkMatrix;
    if (typeof value.matrix_sha256 === "string" && matrixSha256(normalizedMatrix) !== value.matrix_sha256) {
      errors.push("matrix content does not match matrix_sha256");
    }
    const tolerances = isMap(value.matrix.tolerances) ? value.matrix.tolerances : undefined;
    if (tolerances?.enforcement === "fail" && value.matrix.execution_policy !== "local_ci") {
      errors.push("failure-enforced result tolerance requires local_ci policy");
    }
    if (value.matrix.execution_policy === "local_ci" && value.matrix.entries.some((entry) => isMap(entry) && entry.execution === "hosted")) {
      errors.push("local_ci result cannot contain hosted entries");
    }
  }
  if (!Array.isArray(value.attempts)) errors.push("result attempts must be a list");
  if (!isMap(value.summary)) errors.push("result summary is invalid");
  else if (Array.isArray(value.attempts)) {
    const attempts = value.attempts as unknown[];
    if (attempts.some((attempt) => isMap(attempt) && ['pending', 'running'].includes(String(attempt.status)))) {
      errors.push("published result cannot contain incomplete attempts");
    }
    const counts = {
      attempts: attempts.length,
      passed: attempts.filter((attempt) => isMap(attempt) && attempt.status === "passed").length,
      failed: attempts.filter((attempt) => isMap(attempt) && attempt.status === "failed").length,
      skipped: attempts.filter((attempt) => isMap(attempt) && attempt.status === "skipped").length,
    };
    for (const [name, count] of Object.entries(counts)) if (value.summary[name] !== count) errors.push(`summary.${name} does not match attempts`);
  }
  if (!isMap(value.sanitization) || value.sanitization.transcript_text_removed !== true || value.sanitization.stdout_stderr_removed !== true) {
    errors.push("result sanitization flags are incomplete");
  }
  const serialized = JSON.stringify(value);
  if (/\/Users\/|file:\/\/|Bearer\s|-----BEGIN|[?&](?:token|key|signature)=/i.test(serialized)) {
    errors.push("result contains a forbidden path or credential pattern");
  }
  validateAllowlist(value, errors);
  return errors;
}

function validateAllowlist(value: Record<string, unknown>, errors: string[]): void {
  rejectUnexpected(value, ["schema_version", "matrix_sha256", "recorded_at_utc", "classification", "producing_commit", "runtime", "environment", "matrix", "summary", "attempts", "sanitization"], "result", errors);
  if (isMap(value.producing_commit)) rejectUnexpected(value.producing_commit, ["hash", "worktree_dirty"], "producing_commit", errors);
  if (isMap(value.runtime)) rejectUnexpected(value.runtime, ["binary_classification", "binary_sha256"], "runtime", errors);
  if (isMap(value.environment)) {
    rejectUnexpected(value.environment, ["machine", "tools"], "environment", errors);
    if (isMap(value.environment.machine)) {
      rejectUnexpected(value.environment.machine, ["cpu", "logical_cores", "memory_bytes", "os", "kernel", "architecture"], "environment.machine", errors);
    }
    if (isMap(value.environment.tools)) rejectUnexpected(value.environment.tools, ["rustc", "ffmpeg"], "environment.tools", errors);
  }
  if (isMap(value.matrix)) {
    rejectUnexpected(value.matrix, ["schema_version", "matrix_id", "description", "execution_policy", "concurrency", "retries", "timeout_seconds", "fixture_ids", "entries", "tolerances"], "matrix", errors);
    if (isMap(value.matrix.tolerances)) {
      rejectUnexpected(value.matrix.tolerances, ["enforcement", "max_relative_latency_regression_percent", "max_absolute_latency_regression_ms", "min_success_rate"], "matrix.tolerances", errors);
    }
    if (Array.isArray(value.matrix.entries)) {
      for (const [index, entry] of value.matrix.entries.entries()) {
        if (isMap(entry)) {
          rejectUnexpected(entry, ["id", "provider", "model", "execution", "cache_state", "repetitions", "fixture_ids", "required_env", "args", "command_template"], `matrix.entries[${index}]`, errors);
          if (typeof entry.command_template === "string" && /--(?:api-key|api-key-file|azure-api-key|dashscope-api-key|gemini-api-key|nvidia-api-key|deepgram-api-key)\b/.test(entry.command_template)) {
            errors.push(`matrix.entries[${index}].command_template contains a credential option`);
          }
        }
      }
    }
  }
  if (Array.isArray(value.attempts)) {
    for (const [index, attempt] of value.attempts.entries()) {
      if (!isMap(attempt)) continue;
      rejectUnexpected(attempt, ["key", "entry_id", "provider", "model", "execution", "cache_state", "fixture", "repetition", "status", "started_at_utc", "completed_at_utc", "wall_ms", "real_time_factor", "error_category", "output_sha256", "manifest_sha256", "capabilities", "quality", "remote_cleanup"], `attempts[${index}]`, errors);
      if (isMap(attempt.fixture)) rejectUnexpected(attempt.fixture, ["id", "duration_seconds", "bytes", "sha256"], `attempts[${index}].fixture`, errors);
      if (isMap(attempt.capabilities)) rejectUnexpected(attempt.capabilities, ["segments", "word_timestamps", "speaker_labels", "language_per_segment", "emotion_per_segment", "native_timestamps"], `attempts[${index}].capabilities`, errors);
      if (isMap(attempt.quality)) rejectUnexpected(attempt.quality, ["timing_source", "timing_reliable", "timestamps_clamped", "speaker_source", "warning_count"], `attempts[${index}].quality`, errors);
    }
  }
  if (isMap(value.summary)) rejectUnexpected(value.summary, ["attempts", "passed", "failed", "skipped"], "summary", errors);
  if (isMap(value.sanitization)) {
    rejectUnexpected(value.sanitization, ["credential_values_removed", "stdout_stderr_removed", "transcript_text_removed", "request_ids_removed", "signed_urls_removed", "local_absolute_paths_removed", "provider_metadata_allowlisted"], "sanitization", errors);
  }
}

function rejectUnexpected(value: Record<string, unknown>, allowed: string[], label: string, errors: string[]): void {
  const unexpected = Object.keys(value).filter((key) => !allowed.includes(key));
  if (unexpected.length) errors.push(`${label} contains unsupported fields: ${unexpected.join(", ")}`);
}

export async function loadPublishedResult(path: string): Promise<PublishedResult> {
  const file = Bun.file(path);
  if (!(await file.exists())) throw new Error(`result not found: ${path}`);
  const value = await file.json();
  const errors = validatePublishedResult(value);
  if (errors.length) throw new Error(errors.join("\n"));
  return value as PublishedResult;
}

type ComparisonGroup = {
  key: string;
  execution: "local" | "hosted";
  baseline_median_wall_ms: number | null;
  candidate_median_wall_ms: number | null;
  relative_regression_percent: number | null;
  absolute_regression_ms: number | null;
  baseline_success_rate: number;
  candidate_success_rate: number;
  violations: string[];
};

export function compareResults(baseline: PublishedResult, candidate: PublishedResult): {
  enforcement: "report_only" | "fail";
  exit_failure: boolean;
  groups: ComparisonGroup[];
} {
  if (baseline.matrix.matrix_id !== candidate.matrix.matrix_id) throw new Error("cannot compare different matrix IDs");
  if (baseline.matrix_sha256 !== candidate.matrix_sha256) throw new Error("cannot compare different matrix definitions");
  const keys = new Set(candidate.attempts.map((attempt) => `${attempt.entry_id}/${attempt.fixture.id}`));
  const groups: ComparisonGroup[] = [];
  for (const key of [...keys].sort()) {
    const [entryId, fixtureId] = key.split("/");
    const baselineAttempts = baseline.attempts.filter((attempt) => attempt.entry_id === entryId && attempt.fixture.id === fixtureId);
    const candidateAttempts = candidate.attempts.filter((attempt) => attempt.entry_id === entryId && attempt.fixture.id === fixtureId);
    const baselineMedian = median(baselineAttempts.filter(passed).map((attempt) => Number(attempt.wall_ms)));
    const candidateMedian = median(candidateAttempts.filter(passed).map((attempt) => Number(attempt.wall_ms)));
    const baselineRate = successRate(baselineAttempts);
    const candidateRate = successRate(candidateAttempts);
    const absolute = baselineMedian === null || candidateMedian === null ? null : candidateMedian - baselineMedian;
    const relative = absolute === null || baselineMedian === 0 ? null : absolute / baselineMedian * 100;
    const tolerance = candidate.matrix.tolerances;
    const violations: string[] = [];
    if (candidateRate < tolerance.min_success_rate) violations.push("success_rate");
    if (
      absolute !== null && relative !== null &&
      absolute > tolerance.max_absolute_latency_regression_ms &&
      relative > tolerance.max_relative_latency_regression_percent
    ) violations.push("latency");
    groups.push({
      key,
      execution: candidateAttempts[0]?.execution ?? "local",
      baseline_median_wall_ms: baselineMedian,
      candidate_median_wall_ms: candidateMedian,
      relative_regression_percent: relative,
      absolute_regression_ms: absolute,
      baseline_success_rate: baselineRate,
      candidate_success_rate: candidateRate,
      violations,
    });
  }
  const enforcement = candidate.matrix.tolerances.enforcement;
  const exitFailure = enforcement === "fail" && groups.some((group) => group.execution === "local" && group.violations.length > 0);
  return { enforcement, exit_failure: exitFailure, groups };
}

function passed(attempt: AttemptRecord): boolean {
  return attempt.status === "passed" && typeof attempt.wall_ms === "number";
}

function successRate(attempts: AttemptRecord[]): number {
  return attempts.length ? attempts.filter((attempt) => attempt.status === "passed").length / attempts.length : 0;
}

function median(values: number[]): number | null {
  if (!values.length) return null;
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}
