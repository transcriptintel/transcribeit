import { validatePublishedCleanup } from "./cleanup";
import { expectedAttemptIdentities } from "./attempt_plan";
import { validateAttemptValues } from "./result_validation_attempt";
import {
  normalizedPublishedMatrix,
  validateEnvironmentValues,
  validateIdentityAndRuntime,
  validateMatrixValues,
} from "./result_validation_matrix";
import {
  containsForbiddenString,
  isIsoTimestamp,
  isSafeIntegerRange,
  sha256Pattern,
} from "./result_validation_primitives";
import { validatePublishedShape } from "./result_validation_shape";
import { isMap, matrixSha256 } from "./state";
import type { BenchmarkMatrix, PublishedResult } from "./types";

export function validatePublishedResult(value: unknown): string[] {
  const errors: string[] = [];
  if (!isMap(value)) return ["result must be a mapping"];
  validatePublishedShape(value, errors);
  if (value.schema_version !== "transcribeit.benchmark-harness-result.v1") {
    errors.push("unsupported result schema_version");
  }
  if (!isIsoTimestamp(value.recorded_at_utc)) errors.push("recorded_at_utc must be an ISO timestamp");
  if (typeof value.matrix_sha256 !== "string" || !sha256Pattern.test(value.matrix_sha256)) {
    errors.push("invalid matrix_sha256");
  }
  if (value.classification !== "clean_commit_benchmark" && value.classification !== "dirty_worktree_reference") {
    errors.push("invalid result classification");
  }
  validateIdentityAndRuntime(value, errors);
  validateEnvironmentValues(value.environment, errors);
  const matrix = validateMatrixValues(value.matrix, errors);
  if (isMap(value.matrix) && typeof value.matrix_sha256 === "string") {
    const normalized = normalizedPublishedMatrix(value.matrix);
    if (matrixSha256(normalized) !== value.matrix_sha256) errors.push("matrix content does not match matrix_sha256");
  }
  validateAttemptsAndSummary(value, matrix, errors);
  validatePublishedCleanup(
    value.cleanup,
    value.matrix,
    value.matrix_sha256,
    latestAttemptCompletion(value.attempts),
    typeof value.recorded_at_utc === "string" ? value.recorded_at_utc : "",
    errors,
  );
  validateSanitization(value.sanitization, errors);
  if (containsForbiddenString(value)) errors.push("result contains a forbidden path or credential pattern");
  return errors;
}

function validateAttemptsAndSummary(
  value: Record<string, unknown>,
  matrix: BenchmarkMatrix | undefined,
  errors: string[],
): void {
  if (!Array.isArray(value.attempts)) {
    errors.push("result attempts must be a list");
  } else {
    if (!value.attempts.length) errors.push("published result must contain attempts");
    for (const [index, attempt] of value.attempts.entries()) {
      validateAttemptValues(attempt, index, matrix, value.recorded_at_utc, errors);
    }
    validateAttemptPlan(matrix, value.attempts, errors);
  }
  if (!isMap(value.summary)) {
    errors.push("result summary is invalid");
    return;
  }
  const summaryNames = ["attempts", "passed", "failed", "skipped"] as const;
  for (const name of summaryNames) {
    if (!isSafeIntegerRange(value.summary[name], 0)) errors.push(`summary.${name} must be a non-negative safe integer`);
  }
  if (!Array.isArray(value.attempts)) return;
  const counts = {
    attempts: value.attempts.length,
    passed: value.attempts.filter((attempt) => isMap(attempt) && attempt.status === "passed").length,
    failed: value.attempts.filter((attempt) => isMap(attempt) && attempt.status === "failed").length,
    skipped: value.attempts.filter((attempt) => isMap(attempt) && attempt.status === "skipped").length,
  };
  for (const [name, count] of Object.entries(counts)) {
    if (value.summary[name] !== count) errors.push(`summary.${name} does not match attempts`);
  }
  if (
    isSafeIntegerRange(value.summary.attempts, 0) &&
    isSafeIntegerRange(value.summary.passed, 0) &&
    isSafeIntegerRange(value.summary.failed, 0) &&
    isSafeIntegerRange(value.summary.skipped, 0) &&
    value.summary.passed + value.summary.failed + value.summary.skipped !== value.summary.attempts
  ) errors.push("summary terminal counts do not add up to summary.attempts");
}

function validateAttemptPlan(
  matrix: BenchmarkMatrix | undefined,
  attempts: unknown[],
  errors: string[],
): void {
  if (!matrix) return;
  let expectedList;
  try {
    expectedList = expectedAttemptIdentities(matrix);
  } catch (error) {
    errors.push(`published result attempt plan is invalid: ${error instanceof Error ? error.message : String(error)}`);
    return;
  }
  if (!expectedList.length) {
    errors.push("published result matrix produces no expected attempts");
    return;
  }
  const expected = new Map(expectedList.map((identity) => [identity.key, identity]));
  if (expected.size !== expectedList.length) errors.push("published result matrix produces duplicate attempt keys");
  const actual = new Map<string, Record<string, unknown>>();
  for (const [index, attempt] of attempts.entries()) {
    if (!isMap(attempt) || typeof attempt.key !== "string") {
      errors.push("published result contains an invalid attempt identity");
      continue;
    }
    if (actual.has(attempt.key)) {
      errors.push(`published result contains duplicate attempt key ${attempt.key}`);
      continue;
    }
    actual.set(attempt.key, attempt);
    const planned = expected.get(attempt.key);
    if (!planned) {
      errors.push(`published result contains unexpected attempt key ${attempt.key}`);
      continue;
    }
    if (expectedList[index]?.key !== attempt.key) errors.push(`published result attempt ${attempt.key} is out of matrix order`);
    const fixtureId = isMap(attempt.fixture) ? attempt.fixture.id : undefined;
    if (
      attempt.entry_id !== planned.entry_id || attempt.provider !== planned.provider || attempt.model !== planned.model ||
      attempt.execution !== planned.execution || attempt.cache_state !== planned.cache_state ||
      fixtureId !== planned.fixture_id || attempt.repetition !== planned.repetition
    ) errors.push(`published result attempt ${attempt.key} does not match its matrix identity`);
  }
  for (const key of expected.keys()) if (!actual.has(key)) errors.push(`published result is missing attempt key ${key}`);
}

function latestAttemptCompletion(value: unknown): string {
  if (!Array.isArray(value)) return "";
  return value.reduce((latest, attempt) => {
    const completed = isMap(attempt) && isIsoTimestamp(attempt.completed_at_utc) ? attempt.completed_at_utc : "";
    return completed > latest ? completed : latest;
  }, "");
}

function validateSanitization(value: unknown, errors: string[]): void {
  const fields = [
    "credential_values_removed", "stdout_stderr_removed", "transcript_text_removed", "request_ids_removed",
    "signed_urls_removed", "local_absolute_paths_removed", "provider_metadata_allowlisted",
  ];
  if (!isMap(value) || fields.some((name) => value[name] !== true)) {
    errors.push("result sanitization flags are incomplete");
  }
}

export async function loadPublishedResult(path: string): Promise<PublishedResult> {
  const file = Bun.file(path);
  if (!(await file.exists())) throw new Error(`result not found: ${path}`);
  const value = await file.json();
  const errors = validatePublishedResult(value);
  if (errors.length) throw new Error(errors.join("\n"));
  return value as PublishedResult;
}
