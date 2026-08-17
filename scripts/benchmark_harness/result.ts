import { lstat, readdir, realpath } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";

import { hasAppleSpeech, hasEvaluationDownload, loadDownloadedModelCleanup } from "./cleanup";
import { resolveDirectBenchmarkChild } from "./paths";
import { captureRepositoryProvenance } from "./provenance";
import { loadPublishedResult, validatePublishedResult } from "./result_validation";
import { commandTemplate } from "./runner";
import { isMap, loadRunState, writeJsonAtomic } from "./state";
import type { AttemptRecord, BenchmarkMatrix, PublishedResult, RunState } from "./types";

export { loadPublishedResult, validatePublishedResult };

export async function publishResult(
  statePath: string,
  outputPath: string,
  cleanupRecordPath?: string,
  cleanupRootPath?: string,
): Promise<PublishedResult> {
  const state = await loadRunState(statePath);
  const publishedAtUtc = new Date().toISOString();
  const incomplete = state.attempts.filter((attempt) => attempt.status === "pending" || attempt.status === "running");
  if (incomplete.length) throw new Error(`cannot publish: ${incomplete.length} attempts are incomplete`);
  const currentProvenance = await captureRepositoryProvenance();
  if (
    currentProvenance.hash !== state.producing_commit.hash ||
    currentProvenance.worktree_dirty !== state.producing_commit.worktree_dirty ||
    currentProvenance.worktree_fingerprint_sha256 !== state.producing_commit.worktree_fingerprint_sha256
  ) {
    throw new Error("cannot publish: repository HEAD or worktree changed after the benchmark run");
  }
  if (state.keep_attempt_outputs !== false) {
    throw new Error("cannot publish: retained attempt outputs may contain transcripts; rerun without --keep-attempt-outputs");
  }
  const needsDownloadedCleanup = hasEvaluationDownload(state.matrix);
  if (needsDownloadedCleanup && (!cleanupRecordPath || !cleanupRootPath)) {
    throw new Error("cannot publish: evaluation-download artifacts require --cleanup-record and --cleanup-root evidence");
  }
  if (!needsDownloadedCleanup && (cleanupRecordPath || cleanupRootPath)) {
    throw new Error("cannot publish: cleanup evidence does not correspond to an evaluation-download artifact");
  }
  let downloadedCleanup: PublishedResult["cleanup"]["downloaded_model"] = null;
  if (needsDownloadedCleanup) {
    const cleanupRoot = await resolveDirectBenchmarkChild(cleanupRootPath!, "cleanup root");
    if (cleanupRoot.state !== "absent") throw new Error("cannot publish: cleanup root still exists");
    if (cleanupRoot.identitySha256 !== state.evaluation_root_sha256) {
      throw new Error("cannot publish: cleanup root does not match the run state");
    }
    downloadedCleanup = await loadDownloadedModelCleanup(cleanupRecordPath!, {
      matrix: state.matrix,
      matrixSha256: state.matrix_sha256,
      evaluationRootSha256: cleanupRoot.identitySha256,
      notBeforeUtc: state.updated_at_utc,
      notAfterUtc: publishedAtUtc,
    });
  }
  await assertAttemptOutputsRemoved(statePath);
  const result: PublishedResult = {
    schema_version: "transcribeit.benchmark-harness-result.v1",
    matrix_sha256: state.matrix_sha256,
    recorded_at_utc: publishedAtUtc,
    classification: state.producing_commit.worktree_dirty ? "dirty_worktree_reference" : "clean_commit_benchmark",
    producing_commit: {
      hash: state.producing_commit.hash,
      worktree_dirty: state.producing_commit.worktree_dirty,
      worktree_fingerprint_sha256: state.producing_commit.worktree_fingerprint_sha256,
    },
    runtime: { binary_classification: state.binary_classification, binary_sha256: state.binary_sha256 },
    environment: sanitizeEnvironment(state.environment),
    matrix: sanitizeMatrix(state.matrix),
    summary: summarize(state.attempts),
    attempts: state.attempts.map(sanitizeAttempt),
    cleanup: {
      attempt_outputs_removed: true,
      evaluation_root_sha256: state.evaluation_root_sha256,
      downloaded_model: downloadedCleanup,
      apple_speech_asset: hasAppleSpeech(state.matrix)
        ? { owner: "macos", cleanup_attempted: false, lifecycle: "system_managed_shared" }
        : null,
    },
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

async function assertAttemptOutputsRemoved(statePath: string): Promise<void> {
  const attemptsPath = join(dirname(resolve(statePath)), "attempts");
  const metadata = await lstat(attemptsPath).catch((error: unknown) => {
    if (isMissing(error)) return undefined;
    throw new Error("cannot publish: attempt-output directory could not be inspected safely");
  });
  if (!metadata) return;
  if (metadata.isSymbolicLink()) throw new Error("cannot publish: attempt-output directory must not be a symbolic link");
  if (!metadata.isDirectory()) throw new Error("cannot publish: attempt-output path is not a directory");
  if (await realpath(attemptsPath) !== attemptsPath) {
    throw new Error("cannot publish: attempt-output directory must use its canonical path");
  }
  if ((await readdir(attemptsPath)).length) throw new Error("cannot publish: attempt outputs have not been removed");
}

function isMissing(error: unknown): boolean {
  return typeof error === "object" && error !== null && "code" in error && error.code === "ENOENT";
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
    processing_ms: attempt.processing_ms,
    real_time_factor: attempt.real_time_factor,
    peak_rss_bytes: attempt.peak_rss_bytes,
    error_category: attempt.error_category && errorCategories.has(attempt.error_category)
      ? attempt.error_category
      : attempt.error_category ? "provider_error" : null,
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
    output_shape: attempt.output_shape
      ? {
          segments: attempt.output_shape.segments,
          characters: attempt.output_shape.characters,
          last_end_ms: attempt.output_shape.last_end_ms,
          zero_duration_segments: attempt.output_shape.zero_duration_segments,
          reversed_segments: attempt.output_shape.reversed_segments,
          word_timestamps: attempt.output_shape.word_timestamps,
          speaker_labels: attempt.output_shape.speaker_labels,
        }
      : null,
    preprocessing: attempt.preprocessing,
    apple_speech: attempt.apple_speech
      ? {
          resolved_locale: attempt.apple_speech.resolved_locale,
          apple_intelligence_available: attempt.apple_speech.apple_intelligence_available,
          asset_install_requested: attempt.apple_speech.asset_install_requested,
          asset_managed_by: attempt.apple_speech.asset_managed_by,
          on_device: attempt.apple_speech.on_device,
        }
      : null,
    reference_metrics: sanitizeReferenceMetrics(attempt.reference_metrics),
    remote_cleanup: attempt.remote_cleanup,
  };
}

function sanitizeEnvironment(environment: RunState["environment"]): RunState["environment"] {
  return {
    machine: {
      cpu: environment.machine.cpu,
      logical_cores: environment.machine.logical_cores,
      memory_bytes: environment.machine.memory_bytes,
      os: environment.machine.os,
      os_version: environment.machine.os_version,
      os_build: environment.machine.os_build,
      kernel: environment.machine.kernel,
      architecture: environment.machine.architecture,
    },
    tools: {
      rustc: environment.tools.rustc,
      ffmpeg: environment.tools.ffmpeg,
      swift: environment.tools.swift,
      bun: environment.tools.bun,
    },
  };
}

function sanitizeMatrix(matrix: BenchmarkMatrix): PublishedResult["matrix"] {
  const sanitized: Record<string, unknown> = {};
  const scalarFields = new Set([
    "schema_version", "matrix_id", "description", "execution_policy", "attempt_order", "concurrency", "retries",
    "timeout_seconds",
  ]);
  for (const [name, value] of Object.entries(matrix)) {
    if (scalarFields.has(name)) sanitized[name] = value;
    else if (name === "fixture_ids") sanitized[name] = [...matrix.fixture_ids];
    else if (name === "measurements") sanitized[name] = sanitizeMetric(value, ["reference_scoring", "peak_rss"]);
    else if (name === "tolerances") sanitized[name] = sanitizeMetric(value, [
      "enforcement", "max_relative_latency_regression_percent", "max_absolute_latency_regression_ms", "min_success_rate",
    ]);
    else if (name === "entries") sanitized[name] = matrix.entries.map((entry) => sanitizeMatrixEntry(matrix, entry));
  }
  return sanitized as PublishedResult["matrix"];
}

function sanitizeMatrixEntry(matrix: BenchmarkMatrix, entry: BenchmarkMatrix["entries"][number]): Record<string, unknown> {
  const sanitized: Record<string, unknown> = {};
  const scalarFields = new Set(["id", "provider", "model", "execution", "cache_state", "repetitions"]);
  for (const [name, value] of Object.entries(entry)) {
    if (scalarFields.has(name)) sanitized[name] = value;
    else if (["fixture_ids", "required_env", "args"].includes(name) && Array.isArray(value)) sanitized[name] = [...value];
    else if (name === "artifact" && isMap(value)) {
      sanitized[name] = sanitizeMetric(value, ["lifecycle", "revision", "sha256", "bytes"]);
    }
  }
  sanitized.command_template = commandTemplate(matrix, entry);
  return sanitized;
}

function sanitizeReferenceMetrics(metrics: AttemptRecord["reference_metrics"]): AttemptRecord["reference_metrics"] {
  if (!metrics) return null;
  return {
    status: metrics.status,
    word_accuracy: sanitizeMetric(metrics.word_accuracy, [
      "status", "reference_words", "hypothesis_words", "substitutions", "deletions", "insertions", "wer",
    ]) as typeof metrics.word_accuracy,
    domain_terms: sanitizeMetric(metrics.domain_terms, [
      "status", "expected_terms", "matched_terms", "term_recall",
    ]) as typeof metrics.domain_terms,
    timing: sanitizeMetric(metrics.timing, [
      "status", "start_boundary_mae_ms", "end_boundary_mae_ms", "timestamp_coverage", "scored_segments",
      "timing_origin", "timing_reliable",
    ]) as typeof metrics.timing,
    speakers: sanitizeMetric(metrics.speakers, ["status"]) as typeof metrics.speakers,
    word_timestamps: sanitizeMetric(metrics.word_timestamps, ["status"]) as typeof metrics.word_timestamps,
  };
}

function sanitizeMetric(value: unknown, allowed: string[]): Record<string, unknown> {
  if (!isMap(value)) return {};
  return Object.fromEntries(
    Object.entries(value).filter(
      ([name, child]) => allowed.includes(name) && (child === null || ["string", "number", "boolean"].includes(typeof child)),
    ),
  );
}

export function summarize(attempts: AttemptRecord[]): PublishedResult["summary"] {
  return {
    attempts: attempts.length,
    passed: attempts.filter((attempt) => attempt.status === "passed").length,
    failed: attempts.filter((attempt) => attempt.status === "failed").length,
    skipped: attempts.filter((attempt) => attempt.status === "skipped").length,
  };
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
