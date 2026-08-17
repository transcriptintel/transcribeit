import { isMap } from "./state";
import { rejectUnexpected } from "./result_validation_primitives";

export function validatePublishedShape(value: Record<string, unknown>, errors: string[]): void {
  rejectUnexpected(value, [
    "schema_version", "matrix_sha256", "recorded_at_utc", "classification", "producing_commit", "runtime",
    "environment", "matrix", "summary", "attempts", "cleanup", "sanitization",
  ], "result", errors);
  allowMap(value.producing_commit, ["hash", "worktree_dirty", "worktree_fingerprint_sha256"], "producing_commit", errors);
  allowMap(value.runtime, ["binary_classification", "binary_sha256"], "runtime", errors);
  validateEnvironmentShape(value.environment, errors);
  validateMatrixShape(value.matrix, errors);
  if (Array.isArray(value.attempts)) {
    for (const [index, attempt] of value.attempts.entries()) validateAttemptShape(attempt, index, errors);
  }
  allowMap(value.summary, ["attempts", "passed", "failed", "skipped"], "summary", errors);
  validateCleanupShape(value.cleanup, errors);
  allowMap(value.sanitization, [
    "credential_values_removed", "stdout_stderr_removed", "transcript_text_removed", "request_ids_removed",
    "signed_urls_removed", "local_absolute_paths_removed", "provider_metadata_allowlisted",
  ], "sanitization", errors);
}

function validateEnvironmentShape(value: unknown, errors: string[]): void {
  if (!allowMap(value, ["machine", "tools"], "environment", errors)) return;
  allowMap(value.machine, [
    "cpu", "logical_cores", "memory_bytes", "os", "os_version", "os_build", "kernel", "architecture",
  ], "environment.machine", errors);
  allowMap(value.tools, ["rustc", "ffmpeg", "swift", "bun"], "environment.tools", errors);
}

function validateMatrixShape(value: unknown, errors: string[]): void {
  if (!allowMap(value, [
    "schema_version", "matrix_id", "description", "execution_policy", "attempt_order", "concurrency", "retries",
    "timeout_seconds", "fixture_ids", "measurements", "entries", "tolerances",
  ], "matrix", errors)) return;
  allowMap(value.measurements, ["reference_scoring", "peak_rss"], "matrix.measurements", errors);
  allowMap(value.tolerances, [
    "enforcement", "max_relative_latency_regression_percent", "max_absolute_latency_regression_ms", "min_success_rate",
  ], "matrix.tolerances", errors);
  if (!Array.isArray(value.entries)) return;
  for (const [index, entry] of value.entries.entries()) {
    const label = `matrix.entries[${index}]`;
    if (!allowMap(entry, [
      "id", "provider", "model", "execution", "cache_state", "repetitions", "fixture_ids", "required_env",
      "args", "artifact", "command_template",
    ], label, errors)) continue;
    if (entry.artifact !== undefined) {
      allowMap(entry.artifact, ["lifecycle", "revision", "sha256", "bytes"], `${label}.artifact`, errors);
    }
  }
}

function validateAttemptShape(value: unknown, index: number, errors: string[]): void {
  const label = `attempts[${index}]`;
  if (!allowMap(value, [
    "key", "entry_id", "provider", "model", "execution", "cache_state", "fixture", "repetition", "status",
    "started_at_utc", "completed_at_utc", "wall_ms", "processing_ms", "real_time_factor", "peak_rss_bytes",
    "error_category", "output_sha256", "manifest_sha256", "capabilities", "quality", "output_shape",
    "preprocessing", "apple_speech", "reference_metrics", "remote_cleanup",
  ], label, errors)) return;
  allowMap(value.fixture, ["id", "duration_seconds", "bytes", "sha256"], `${label}.fixture`, errors);
  if (value.capabilities !== null) allowMap(value.capabilities, [
    "segments", "word_timestamps", "speaker_labels", "language_per_segment", "emotion_per_segment", "native_timestamps",
  ], `${label}.capabilities`, errors);
  if (value.quality !== null) allowMap(value.quality, [
    "timing_source", "timing_reliable", "timestamps_clamped", "speaker_source", "warning_count",
  ], `${label}.quality`, errors);
  if (value.output_shape !== null) allowMap(value.output_shape, [
    "segments", "characters", "last_end_ms", "zero_duration_segments", "reversed_segments",
    "word_timestamps", "speaker_labels",
  ], `${label}.output_shape`, errors);
  if (value.apple_speech !== null) allowMap(value.apple_speech, [
    "resolved_locale", "apple_intelligence_available", "asset_install_requested", "asset_managed_by", "on_device",
  ], `${label}.apple_speech`, errors);
  validateReferenceMetricsShape(value.reference_metrics, `${label}.reference_metrics`, errors);
}

function validateReferenceMetricsShape(value: unknown, label: string, errors: string[]): void {
  if (value === null || !allowMap(value, [
    "status", "word_accuracy", "domain_terms", "timing", "speakers", "word_timestamps",
  ], label, errors)) return;
  allowMap(value.word_accuracy, [
    "status", "reference_words", "hypothesis_words", "substitutions", "deletions", "insertions", "wer",
  ], `${label}.word_accuracy`, errors);
  allowMap(value.domain_terms, ["status", "expected_terms", "matched_terms", "term_recall"], `${label}.domain_terms`, errors);
  allowMap(value.timing, [
    "status", "start_boundary_mae_ms", "end_boundary_mae_ms", "timestamp_coverage", "scored_segments",
    "timing_origin", "timing_reliable",
  ], `${label}.timing`, errors);
  allowMap(value.speakers, ["status"], `${label}.speakers`, errors);
  allowMap(value.word_timestamps, ["status"], `${label}.word_timestamps`, errors);
}

function validateCleanupShape(value: unknown, errors: string[]): void {
  if (!allowMap(value, [
    "attempt_outputs_removed", "evaluation_root_sha256", "downloaded_model", "apple_speech_asset",
  ], "cleanup", errors)) return;
  if (value.downloaded_model !== null) allowMap(value.downloaded_model, [
    "schema_version", "matrix_sha256", "evaluation_root_sha256", "artifact_sha256s", "recorded_at_utc",
    "evaluation_root_preexisting", "initial_bytes", "downloaded_bytes", "reclaimed_bytes", "cleanup_completed",
    "evaluation_root_absent_after_cleanup",
  ], "cleanup.downloaded_model", errors);
  if (value.apple_speech_asset !== null) allowMap(
    value.apple_speech_asset,
    ["owner", "cleanup_attempted", "lifecycle"],
    "cleanup.apple_speech_asset",
    errors,
  );
}

function allowMap(value: unknown, allowed: readonly string[], label: string, errors: string[]): value is Record<string, unknown> {
  if (!isMap(value)) {
    errors.push(`${label} must be a mapping`);
    return false;
  }
  rejectUnexpected(value, allowed, label, errors);
  return true;
}
