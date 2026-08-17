import { resolve } from "node:path";

import { isMap } from "./state";
import type { BenchmarkMatrix, DownloadedModelCleanup } from "./types";

type CleanupBindings = {
  matrix: BenchmarkMatrix;
  matrixSha256: string;
  evaluationRootSha256: string;
  notBeforeUtc: string;
  notAfterUtc: string;
};

export function hasEvaluationDownload(matrix: BenchmarkMatrix): boolean {
  return matrix.entries.some((entry) => entry.artifact?.lifecycle === "evaluation_download");
}

export function hasAppleSpeech(matrix: BenchmarkMatrix): boolean {
  return matrix.entries.some((entry) => entry.provider === "apple-speech");
}

export async function loadDownloadedModelCleanup(
  path: string,
  bindings: CleanupBindings,
): Promise<DownloadedModelCleanup> {
  const file = Bun.file(resolve(path));
  if (!(await file.exists())) throw new Error("cannot publish: cleanup record was not found");
  let value: unknown;
  try {
    value = await file.json();
  } catch {
    throw new Error("cannot publish: cleanup record is not valid JSON");
  }
  const errors = validateDownloadedModelCleanup(value, bindings, "cleanup record");
  if (errors.length) throw new Error(`cannot publish: ${errors.join("\n")}`);
  return value as DownloadedModelCleanup;
}

export function validatePublishedCleanup(
  cleanup: unknown,
  matrix: unknown,
  matrixSha256: unknown,
  notBeforeUtc: string,
  notAfterUtc: string,
  errors: string[],
): void {
  if (!isMap(cleanup)) {
    errors.push("result cleanup evidence is invalid");
    return;
  }
  if (cleanup.attempt_outputs_removed !== true) errors.push("cleanup.attempt_outputs_removed must be true");
  const entries = isMap(matrix) && Array.isArray(matrix.entries) ? matrix.entries.filter(isMap) : [];
  const evaluationEntries = entries.filter(
    (entry) => isMap(entry.artifact) && entry.artifact.lifecycle === "evaluation_download",
  );
  if (evaluationEntries.length) {
    const typedMatrix = matrix as unknown as BenchmarkMatrix;
    const rootHash = typeof cleanup.evaluation_root_sha256 === "string" ? cleanup.evaluation_root_sha256 : "";
    errors.push(...validateDownloadedModelCleanup(cleanup.downloaded_model, {
      matrix: typedMatrix,
      matrixSha256: typeof matrixSha256 === "string" ? matrixSha256 : "",
      evaluationRootSha256: rootHash,
      notBeforeUtc,
      notAfterUtc,
    }, "cleanup.downloaded_model"));
    if (!sha256Pattern.test(rootHash)) errors.push("cleanup.evaluation_root_sha256 must be lowercase SHA-256");
  } else {
    if (cleanup.downloaded_model !== null) {
      errors.push("cleanup.downloaded_model must be null without evaluation-download artifacts");
    }
    if (cleanup.evaluation_root_sha256 !== null) {
      errors.push("cleanup.evaluation_root_sha256 must be null without evaluation-download artifacts");
    }
  }
  const hasApple = entries.some((entry) => entry.provider === "apple-speech");
  if (hasApple) {
    if (
      !isMap(cleanup.apple_speech_asset) ||
      cleanup.apple_speech_asset.owner !== "macos" ||
      cleanup.apple_speech_asset.cleanup_attempted !== false ||
      cleanup.apple_speech_asset.lifecycle !== "system_managed_shared"
    ) {
      errors.push("cleanup.apple_speech_asset must record the shared macOS-owned lifecycle");
    }
  } else if (cleanup.apple_speech_asset !== null) {
    errors.push("cleanup.apple_speech_asset must be null without Apple Speech entries");
  }
}

const sha256Pattern = /^[a-f0-9]{64}$/;

function artifactSha256s(matrix: BenchmarkMatrix): string[] {
  return [...new Set(matrix.entries
    .filter((entry) => entry.artifact?.lifecycle === "evaluation_download")
    .map((entry) => entry.artifact?.sha256)
    .filter((value): value is string => typeof value === "string"))].sort();
}

function expectedDownloadedBytes(matrix: BenchmarkMatrix): number {
  const artifacts = new Map<string, number>();
  for (const entry of matrix.entries) {
    if (entry.artifact?.lifecycle !== "evaluation_download" || !entry.artifact.bytes) continue;
    artifacts.set(entry.artifact.sha256 ?? entry.id, entry.artifact.bytes);
  }
  return [...artifacts.values()].reduce((sum, bytes) => sum + bytes, 0);
}

function validateDownloadedModelCleanup(value: unknown, bindings: CleanupBindings, label: string): string[] {
  const errors: string[] = [];
  if (!isMap(value)) return [`${label} must be a mapping`];
  rejectUnexpected(value, [
    "schema_version", "matrix_sha256", "evaluation_root_sha256", "artifact_sha256s", "recorded_at_utc",
    "evaluation_root_preexisting", "initial_bytes", "downloaded_bytes", "reclaimed_bytes", "cleanup_completed",
    "evaluation_root_absent_after_cleanup",
  ], label, errors);
  if (value.schema_version !== "transcribeit.downloaded-model-cleanup.v1") errors.push(`${label} schema_version is invalid`);
  if (typeof value.matrix_sha256 !== "string" || !sha256Pattern.test(value.matrix_sha256)) {
    errors.push(`${label}.matrix_sha256 must be lowercase SHA-256`);
  }
  if (value.matrix_sha256 !== bindings.matrixSha256) errors.push(`${label}.matrix_sha256 does not match the run`);
  if (typeof value.evaluation_root_sha256 !== "string" || !sha256Pattern.test(value.evaluation_root_sha256)) {
    errors.push(`${label}.evaluation_root_sha256 must be lowercase SHA-256`);
  }
  if (value.evaluation_root_sha256 !== bindings.evaluationRootSha256) {
    errors.push(`${label}.evaluation_root_sha256 does not match the run`);
  }
  const expectedArtifacts = artifactSha256s(bindings.matrix);
  if (!expectedArtifacts.length) errors.push(`${label} has no declared evaluation artifact hashes`);
  if (
    !Array.isArray(value.artifact_sha256s) ||
    value.artifact_sha256s.some((hash) => typeof hash !== "string" || !sha256Pattern.test(hash)) ||
    JSON.stringify(value.artifact_sha256s) !== JSON.stringify(expectedArtifacts)
  ) {
    errors.push(`${label}.artifact_sha256s must exactly match the sorted evaluation artifacts`);
  }
  if (!isIsoTimestamp(value.recorded_at_utc)) {
    errors.push(`${label}.recorded_at_utc must be an ISO timestamp`);
  } else {
    const recordedAt = Date.parse(value.recorded_at_utc as string);
    if (!isIsoTimestamp(bindings.notBeforeUtc) || recordedAt < Date.parse(bindings.notBeforeUtc)) {
      errors.push(`${label}.recorded_at_utc predates this benchmark run`);
    }
    if (!isIsoTimestamp(bindings.notAfterUtc) || recordedAt > Date.parse(bindings.notAfterUtc)) {
      errors.push(`${label}.recorded_at_utc is later than publication`);
    }
  }
  if (value.evaluation_root_preexisting !== false) errors.push(`${label} must identify a non-preexisting evaluation root`);
  for (const name of ["initial_bytes", "downloaded_bytes", "reclaimed_bytes"] as const) {
    if (!Number.isSafeInteger(value[name]) || Number(value[name]) < 0) {
      errors.push(`${label}.${name} must be a non-negative safe integer`);
    }
  }
  if (value.evaluation_root_preexisting === false && value.initial_bytes !== 0) {
    errors.push(`${label}.initial_bytes must be zero for a non-preexisting evaluation root`);
  }
  if (value.cleanup_completed !== true) errors.push(`${label}.cleanup_completed must be true`);
  if (value.evaluation_root_absent_after_cleanup !== true) {
    errors.push(`${label}.evaluation_root_absent_after_cleanup must be true`);
  }
  const expectedBytes = expectedDownloadedBytes(bindings.matrix);
  if (typeof value.downloaded_bytes === "number" && value.downloaded_bytes < expectedBytes) {
    errors.push(`${label}.downloaded_bytes is smaller than the declared evaluation artifacts`);
  }
  if (
    typeof value.downloaded_bytes === "number" &&
    typeof value.reclaimed_bytes === "number" &&
    value.reclaimed_bytes < value.downloaded_bytes
  ) {
    errors.push(`${label}.reclaimed_bytes must cover downloaded_bytes`);
  }
  return errors;
}

function isIsoTimestamp(value: unknown): boolean {
  if (typeof value !== "string") return false;
  const timestamp = Date.parse(value);
  return Number.isFinite(timestamp) && new Date(timestamp).toISOString() === value;
}

function rejectUnexpected(value: Record<string, unknown>, allowed: string[], label: string, errors: string[]): void {
  const unexpected = Object.keys(value).filter((key) => !allowed.includes(key));
  if (unexpected.length) errors.push(`${label} contains unsupported fields: ${unexpected.join(", ")}`);
}
