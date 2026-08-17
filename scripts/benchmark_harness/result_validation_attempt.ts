import { isMap } from "./state";
import { providers, type BenchmarkMatrix } from "./types";
import {
  isFiniteRange,
  isIsoTimestamp,
  isNullableBoolean,
  isNullableFiniteRange,
  isNullableSafeIntegerRange,
  isSafeIntegerRange,
  safeIdPattern,
  safeLocalePattern,
  sha256Pattern,
} from "./result_validation_primitives";

const terminalStatuses = ["passed", "failed", "skipped"];
const errorCategories = [
  "unconfigured", "authentication", "rate_limit", "timeout", "provider_5xx", "unsupported", "malformed_response",
  "rejected_request", "transport", "provider_error", "local_output_invalid",
];
const timingSources = ["provider_native", "model_native", "model_generated", "synthetic", "unknown", "none"];
const speakerSources = ["provider_native", "model_generated", "local_postprocess", "unknown", "none"];
const preprocessingKinds = ["original_media", "wav_fallback", "canonical_wav"];
const remoteCleanupStatuses = ["deleted", "failed", "not_attempted", "not_applicable"];

export function validateAttemptValues(
  value: unknown,
  index: number,
  matrix: BenchmarkMatrix | undefined,
  recordedAtUtc: unknown,
  errors: string[],
): void {
  const label = `attempts[${index}]`;
  if (!isMap(value)) {
    errors.push(`${label} must be a mapping`);
    return;
  }
  validateIdentity(value, label, errors);
  validateFixture(value.fixture, label, errors);
  if (!terminalStatuses.includes(String(value.status))) errors.push(`${label}.status must be terminal`);
  validateLifecycle(value, label, recordedAtUtc, errors);
  validateNumbersAndHashes(value, label, errors);
  validateTimingConsistency(value, label, errors);
  validateStatusConsistency(value, label, errors);
  validateCapabilities(value.capabilities, label, errors);
  validateQuality(value.quality, label, errors);
  validateOutputShape(value.output_shape, label, errors);
  if (value.preprocessing !== null && !preprocessingKinds.includes(String(value.preprocessing))) {
    errors.push(`${label}.preprocessing has an invalid value`);
  }
  validateAppleSpeech(value, label, errors);
  validateReferenceMetrics(value.reference_metrics, label, errors);
  if (value.remote_cleanup !== null && !remoteCleanupStatuses.includes(String(value.remote_cleanup))) {
    errors.push(`${label}.remote_cleanup has an invalid value`);
  }
  validateSuccessfulEvidence(value, matrix, label, errors);
}

function validateIdentity(value: Record<string, unknown>, label: string, errors: string[]): void {
  for (const name of ["key", "entry_id", "model"] as const) {
    if (!boundedString(value[name], 512)) errors.push(`${label}.${name} must be a non-empty bounded string`);
  }
  if (typeof value.entry_id !== "string" || !safeIdPattern.test(value.entry_id)) errors.push(`${label}.entry_id is invalid`);
  if (typeof value.provider !== "string" || !(providers as readonly string[]).includes(value.provider)) {
    errors.push(`${label}.provider is unsupported`);
  }
  if (value.execution !== "local" && value.execution !== "hosted") errors.push(`${label}.execution is invalid`);
  if (value.cache_state !== "cold" && value.cache_state !== "warm") errors.push(`${label}.cache_state is invalid`);
  if (!isSafeIntegerRange(value.repetition, 1, 100)) errors.push(`${label}.repetition must be an integer from 1 to 100`);
}

function validateFixture(value: unknown, attemptLabel: string, errors: string[]): void {
  const label = `${attemptLabel}.fixture`;
  if (!isMap(value)) {
    errors.push(`${label} must be a mapping`);
    return;
  }
  if (typeof value.id !== "string" || !safeIdPattern.test(value.id)) errors.push(`${label}.id is invalid`);
  if (!isFiniteRange(value.duration_seconds, Number.MIN_VALUE)) errors.push(`${label}.duration_seconds must be positive and finite`);
  if (!isSafeIntegerRange(value.bytes, 1)) errors.push(`${label}.bytes must be a positive safe integer`);
  if (typeof value.sha256 !== "string" || !sha256Pattern.test(value.sha256)) errors.push(`${label}.sha256 must be lowercase SHA-256`);
}

function validateLifecycle(
  value: Record<string, unknown>,
  label: string,
  recordedAtUtc: unknown,
  errors: string[],
): void {
  if (!isIsoTimestamp(value.started_at_utc)) errors.push(`${label}.started_at_utc must be an ISO timestamp`);
  if (!isIsoTimestamp(value.completed_at_utc)) errors.push(`${label}.completed_at_utc must be an ISO timestamp`);
  if (isIsoTimestamp(value.started_at_utc) && isIsoTimestamp(value.completed_at_utc)) {
    if (value.completed_at_utc < value.started_at_utc) errors.push(`${label}.completed_at_utc predates started_at_utc`);
    if (isIsoTimestamp(recordedAtUtc) && value.completed_at_utc > recordedAtUtc) {
      errors.push(`${label}.completed_at_utc is later than result publication`);
    }
  }
}

function validateNumbersAndHashes(value: Record<string, unknown>, label: string, errors: string[]): void {
  for (const name of ["wall_ms", "processing_ms", "real_time_factor"] as const) {
    if (!isNullableFiniteRange(value[name]) || (typeof value[name] === "number" && value[name] > Number.MAX_SAFE_INTEGER)) {
      errors.push(`${label}.${name} must be null or a non-negative finite safe-range number`);
    }
  }
  if (!isNullableSafeIntegerRange(value.peak_rss_bytes, 1)) {
    errors.push(`${label}.peak_rss_bytes must be null or a positive safe integer`);
  }
  for (const name of ["output_sha256", "manifest_sha256"] as const) {
    if (value[name] !== null && (typeof value[name] !== "string" || !sha256Pattern.test(value[name]))) {
      errors.push(`${label}.${name} must be null or lowercase SHA-256`);
    }
  }
}

function validateTimingConsistency(value: Record<string, unknown>, label: string, errors: string[]): void {
  if (
    typeof value.wall_ms !== "number" || !Number.isFinite(value.wall_ms) ||
    typeof value.real_time_factor !== "number" || !Number.isFinite(value.real_time_factor) ||
    !isMap(value.fixture) || typeof value.fixture.duration_seconds !== "number" || value.fixture.duration_seconds <= 0
  ) return;
  const expected = value.wall_ms / 1_000 / value.fixture.duration_seconds;
  if (Math.abs(value.real_time_factor - expected) > Math.max(Number.EPSILON * 16, Math.abs(expected) * 1e-12)) {
    errors.push(`${label}.real_time_factor is inconsistent with wall_ms and fixture duration`);
  }
}

function validateStatusConsistency(value: Record<string, unknown>, label: string, errors: string[]): void {
  if (value.status === "passed") {
    if (value.error_category !== null) errors.push(`${label}.error_category must be null for a passed attempt`);
    if (value.wall_ms === null || value.real_time_factor === null) errors.push(`${label} passed timing evidence is incomplete`);
  } else if (value.status === "failed") {
    if (typeof value.error_category !== "string" || !errorCategories.includes(value.error_category) || value.error_category === "unconfigured") {
      errors.push(`${label}.error_category is invalid for a failed attempt`);
    }
    if (value.wall_ms === null || value.real_time_factor === null) errors.push(`${label} failed timing evidence is incomplete`);
  } else if (value.status === "skipped") {
    if (value.error_category !== "unconfigured") errors.push(`${label}.error_category must be unconfigured for a skipped attempt`);
    for (const name of [
      "wall_ms", "processing_ms", "real_time_factor", "peak_rss_bytes", "output_sha256", "manifest_sha256", "capabilities",
      "quality", "output_shape", "preprocessing", "apple_speech", "reference_metrics", "remote_cleanup",
    ]) if (value[name] !== null) errors.push(`${label}.${name} must be null for a skipped attempt`);
  }
}

function validateCapabilities(value: unknown, attemptLabel: string, errors: string[]): void {
  if (value === null) return;
  const label = `${attemptLabel}.capabilities`;
  if (!isMap(value)) {
    errors.push(`${label} must be a mapping or null`);
    return;
  }
  for (const name of [
    "segments", "word_timestamps", "speaker_labels", "language_per_segment", "emotion_per_segment", "native_timestamps",
  ]) if (typeof value[name] !== "boolean") errors.push(`${label}.${name} must be a boolean`);
}

function validateQuality(value: unknown, attemptLabel: string, errors: string[]): void {
  if (value === null) return;
  const label = `${attemptLabel}.quality`;
  if (!isMap(value)) {
    errors.push(`${label} must be a mapping or null`);
    return;
  }
  if (value.timing_source !== null && !timingSources.includes(String(value.timing_source))) errors.push(`${label}.timing_source is invalid`);
  if (!isNullableBoolean(value.timing_reliable)) errors.push(`${label}.timing_reliable must be boolean or null`);
  if (!isNullableBoolean(value.timestamps_clamped)) errors.push(`${label}.timestamps_clamped must be boolean or null`);
  if (value.speaker_source !== null && !speakerSources.includes(String(value.speaker_source))) errors.push(`${label}.speaker_source is invalid`);
  if (!isSafeIntegerRange(value.warning_count, 0)) errors.push(`${label}.warning_count must be a non-negative safe integer`);
}

function validateOutputShape(value: unknown, attemptLabel: string, errors: string[]): void {
  if (value === null) return;
  const label = `${attemptLabel}.output_shape`;
  if (!isMap(value)) {
    errors.push(`${label} must be a mapping or null`);
    return;
  }
  for (const name of ["segments", "characters", "zero_duration_segments", "reversed_segments"] as const) {
    if (!isSafeIntegerRange(value[name], 0)) errors.push(`${label}.${name} must be a non-negative safe integer`);
  }
  if (!isNullableSafeIntegerRange(value.last_end_ms, 0)) errors.push(`${label}.last_end_ms must be null or a non-negative safe integer`);
  for (const name of ["word_timestamps", "speaker_labels"] as const) {
    if (typeof value[name] !== "boolean") errors.push(`${label}.${name} must be a boolean`);
  }
  if (isSafeIntegerRange(value.segments, 0)) {
    if (isSafeIntegerRange(value.zero_duration_segments, 0) && value.zero_duration_segments > value.segments) {
      errors.push(`${label}.zero_duration_segments cannot exceed segments`);
    }
    if (isSafeIntegerRange(value.reversed_segments, 0) && value.reversed_segments > value.segments) {
      errors.push(`${label}.reversed_segments cannot exceed segments`);
    }
    if (
      isSafeIntegerRange(value.zero_duration_segments, 0) && isSafeIntegerRange(value.reversed_segments, 0) &&
      value.zero_duration_segments + value.reversed_segments > value.segments
    ) errors.push(`${label} invalid-duration counts cannot exceed segments`);
  }
}

function validateAppleSpeech(value: Record<string, unknown>, label: string, errors: string[]): void {
  const apple = value.apple_speech;
  if (apple === null) return;
  if (!isMap(apple)) {
    errors.push(`${label}.apple_speech must be a mapping or null`);
    return;
  }
  if (value.provider !== "apple-speech") errors.push(`${label}.apple_speech is allowed only for the Apple Speech provider`);
  if (apple.resolved_locale !== null && (typeof apple.resolved_locale !== "string" || !safeLocalePattern.test(apple.resolved_locale))) {
    errors.push(`${label}.apple_speech.resolved_locale is invalid`);
  }
  for (const name of ["apple_intelligence_available", "asset_install_requested", "on_device"] as const) {
    if (!isNullableBoolean(apple[name])) errors.push(`${label}.apple_speech.${name} must be boolean or null`);
  }
  if (apple.asset_managed_by !== null && apple.asset_managed_by !== "macos") {
    errors.push(`${label}.apple_speech.asset_managed_by is invalid`);
  }
}

function validateSuccessfulEvidence(
  value: Record<string, unknown>,
  matrix: BenchmarkMatrix | undefined,
  label: string,
  errors: string[],
): void {
  const scoringEnabled = matrix?.measurements?.reference_scoring === true;
  if (!scoringEnabled && value.reference_metrics !== null) {
    errors.push(`${label}.reference_metrics must be null when matrix reference_scoring is disabled`);
  }
  if (matrix?.measurements?.peak_rss === false && value.peak_rss_bytes !== null) {
    errors.push(`${label}.peak_rss_bytes must be null when peak_rss is disabled`);
  }
  if (value.status !== "passed") return;
  for (const name of ["output_sha256", "manifest_sha256"] as const) {
    if (typeof value[name] !== "string" || !sha256Pattern.test(value[name])) errors.push(`${label}.${name} is required for a passed attempt`);
  }
  for (const name of ["capabilities", "quality", "output_shape", "preprocessing"] as const) {
    if (value[name] === null) errors.push(`${label}.${name} is required for a passed attempt`);
  }
  if (value.provider === "apple-speech" && value.apple_speech === null) {
    errors.push(`${label}.apple_speech is required for a passed Apple Speech attempt`);
  }
  if (value.provider === "apple-speech" && isMap(value.apple_speech)) {
    validateSuccessfulAppleMetadata(value.apple_speech, label, errors);
  }
  if (value.provider !== "apple-speech" && value.apple_speech !== null) {
    errors.push(`${label}.apple_speech must be null for non-Apple providers`);
  }
  if (scoringEnabled !== (value.reference_metrics !== null)) {
    errors.push(`${label}.reference_metrics does not match matrix reference_scoring`);
  }
  if (isMap(value.capabilities) && isMap(value.output_shape)) {
    for (const name of ["word_timestamps", "speaker_labels"] as const) {
      if (value.capabilities[name] !== value.output_shape[name]) {
        errors.push(`${label}.output_shape.${name} does not match capabilities.${name}`);
      }
    }
  }
  const expectedPreprocessing = value.provider === "local"
    ? ["canonical_wav"]
    : value.provider === "apple-speech"
      ? ["original_media", "wav_fallback"]
      : ["original_media"];
  if (!expectedPreprocessing.includes(String(value.preprocessing))) {
    errors.push(`${label}.preprocessing does not match its provider`);
  }
}

function validateSuccessfulAppleMetadata(
  apple: Record<string, unknown>,
  attemptLabel: string,
  errors: string[],
): void {
  const label = `${attemptLabel}.apple_speech`;
  if (typeof apple.resolved_locale !== "string" || !safeLocalePattern.test(apple.resolved_locale)) {
    errors.push(`${label}.resolved_locale is required and must be safe for a passed Apple Speech attempt`);
  }
  if (apple.apple_intelligence_available !== true) {
    errors.push(`${label}.apple_intelligence_available must be true for a passed Apple Speech attempt`);
  }
  if (typeof apple.asset_install_requested !== "boolean") {
    errors.push(`${label}.asset_install_requested must be a boolean for a passed Apple Speech attempt`);
  }
  if (apple.asset_managed_by !== "macos") {
    errors.push(`${label}.asset_managed_by must be macos for a passed Apple Speech attempt`);
  }
  if (apple.on_device !== true) {
    errors.push(`${label}.on_device must be true for a passed Apple Speech attempt`);
  }
}

function validateReferenceMetrics(value: unknown, attemptLabel: string, errors: string[]): void {
  if (value === null) return;
  const label = `${attemptLabel}.reference_metrics`;
  if (!isMap(value)) {
    errors.push(`${label} must be a mapping or null`);
    return;
  }
  if (value.status === "unavailable_reference") {
    for (const name of ["word_accuracy", "domain_terms", "timing", "speakers", "word_timestamps"] as const) {
      statusOnly(value[name], "unavailable_reference", `${label}.${name}`, errors);
    }
    return;
  }
  if (value.status !== "scored") {
    errors.push(`${label}.status is invalid`);
    return;
  }
  validateWordAccuracy(value.word_accuracy, `${label}.word_accuracy`, errors);
  validateDomainTerms(value.domain_terms, `${label}.domain_terms`, errors);
  validateTiming(value.timing, `${label}.timing`, errors);
  for (const name of ["speakers", "word_timestamps"] as const) {
    statusOnly(value[name], ["unsupported_reference", "unsupported_provider", "available_not_scored"], `${label}.${name}`, errors);
  }
}

function validateWordAccuracy(value: unknown, label: string, errors: string[]): void {
  if (!isMap(value)) return errors.push(`${label} must be a mapping`), undefined;
  if (value.status === "unsupported_overlapping_reference") return statusOnly(value, value.status, label, errors);
  if (value.status !== "scored") return errors.push(`${label}.status is invalid`), undefined;
  for (const name of ["reference_words", "hypothesis_words", "substitutions", "deletions", "insertions"] as const) {
    if (!isSafeIntegerRange(value[name], name === "reference_words" ? 1 : 0)) errors.push(`${label}.${name} is invalid`);
  }
  if (!isFiniteRange(value.wer, 0)) errors.push(`${label}.wer must be non-negative and finite`);
  if (
    isSafeIntegerRange(value.reference_words, 1) && isSafeIntegerRange(value.hypothesis_words, 0) &&
    isSafeIntegerRange(value.substitutions, 0) && isSafeIntegerRange(value.deletions, 0) && isSafeIntegerRange(value.insertions, 0)
  ) {
    if (value.substitutions + value.deletions > value.reference_words) errors.push(`${label} edit counts exceed reference_words`);
    if (value.hypothesis_words !== value.reference_words - value.deletions + value.insertions) errors.push(`${label} word counts are inconsistent`);
    const expected = (value.substitutions + value.deletions + value.insertions) / value.reference_words;
    if (typeof value.wer === "number" && Math.abs(value.wer - expected) > Number.EPSILON * 8) errors.push(`${label}.wer is inconsistent with edit counts`);
  }
}

function validateDomainTerms(value: unknown, label: string, errors: string[]): void {
  if (!isMap(value)) return errors.push(`${label} must be a mapping`), undefined;
  if (value.status === "unsupported_reference") return statusOnly(value, value.status, label, errors);
  if (value.status !== "scored") return errors.push(`${label}.status is invalid`), undefined;
  if (!isSafeIntegerRange(value.expected_terms, 1)) errors.push(`${label}.expected_terms must be positive`);
  if (!isSafeIntegerRange(value.matched_terms, 0)) errors.push(`${label}.matched_terms must be non-negative`);
  if (!isFiniteRange(value.term_recall, 0, 1)) errors.push(`${label}.term_recall must be from 0 to 1`);
  if (isSafeIntegerRange(value.expected_terms, 1) && isSafeIntegerRange(value.matched_terms, 0)) {
    if (value.matched_terms > value.expected_terms) errors.push(`${label}.matched_terms cannot exceed expected_terms`);
    const expected = value.matched_terms / value.expected_terms;
    if (typeof value.term_recall === "number" && Math.abs(value.term_recall - expected) > Number.EPSILON * 8) {
      errors.push(`${label}.term_recall is inconsistent with term counts`);
    }
  }
}

function validateTiming(value: unknown, label: string, errors: string[]): void {
  if (!isMap(value)) return errors.push(`${label} must be a mapping`), undefined;
  const unavailable = [
    "unsupported_reference", "unsupported_provider", "unsupported_overlapping_reference", "insufficient_alignment",
  ];
  if (unavailable.includes(String(value.status))) return statusOnly(value, String(value.status), label, errors);
  if (value.status !== "scored") return errors.push(`${label}.status is invalid`), undefined;
  for (const name of ["start_boundary_mae_ms", "end_boundary_mae_ms"] as const) {
    if (!isFiniteRange(value[name], 0)) errors.push(`${label}.${name} must be non-negative and finite`);
  }
  if (!isFiniteRange(value.timestamp_coverage, 0.5, 1)) errors.push(`${label}.timestamp_coverage must be from 0.5 to 1`);
  if (!isSafeIntegerRange(value.scored_segments, 1)) errors.push(`${label}.scored_segments must be positive`);
  if (value.timing_origin !== null && !timingSources.includes(String(value.timing_origin))) errors.push(`${label}.timing_origin is invalid`);
  if (!isNullableBoolean(value.timing_reliable)) errors.push(`${label}.timing_reliable must be boolean or null`);
}

function statusOnly(value: unknown, allowed: string | string[], label: string, errors: string[]): void {
  const statuses = Array.isArray(allowed) ? allowed : [allowed];
  if (!isMap(value) || !statuses.includes(String(value.status)) || Object.keys(value).length !== 1) {
    errors.push(`${label} must contain only a valid status`);
  }
}

function boundedString(value: unknown, maximum: number): value is string {
  return typeof value === "string" && value.length > 0 && value.length <= maximum && !/[\u0000-\u001f\u007f]/.test(value);
}
