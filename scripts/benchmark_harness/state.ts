import { randomUUID } from "node:crypto";
import { constants } from "node:fs";
import { lstat, mkdir, open, rename, unlink } from "node:fs/promises";
import { basename, dirname, join, resolve } from "node:path";

import type { BenchmarkMatrix, RunState } from "./types";

export function sha256(value: string | Uint8Array): string {
  const hasher = new Bun.CryptoHasher("sha256");
  hasher.update(value);
  return hasher.digest("hex");
}

export function matrixSha256(matrix: BenchmarkMatrix): string {
  return sha256(JSON.stringify(matrix));
}

export async function writeJsonAtomic(path: string, value: unknown): Promise<void> {
  const payload = `${JSON.stringify(value, null, 2)}\n`;
  const destination = resolve(path);
  const parent = dirname(destination);
  await mkdir(parent, { recursive: true, mode: 0o700 });

  const temporary = join(parent, `.${basename(destination)}.${process.pid}.${randomUUID()}.tmp`);
  const noFollow = process.platform !== "win32" && typeof constants.O_NOFOLLOW === "number" ? constants.O_NOFOLLOW : 0;
  const flags = constants.O_WRONLY | constants.O_CREAT | constants.O_EXCL | noFollow;
  let handle: Awaited<ReturnType<typeof open>> | undefined;
  let committed = false;
  try {
    handle = await open(temporary, flags, 0o600);
    await handle.chmod(0o600);
    const opened = await handle.stat();
    validateOwnedTemporaryFile(opened);
    await handle.writeFile(payload, "utf8");
    await handle.sync();
    await handle.close();
    handle = undefined;

    const named = await lstat(temporary);
    validateOwnedTemporaryFile(named, opened);
    await rename(temporary, destination);
    committed = true;
  } catch (error) {
    await handle?.close().catch(() => undefined);
    if (!committed) await unlink(temporary).catch(() => undefined);
    throw error;
  }
}

type FileMetadata = Awaited<ReturnType<Awaited<ReturnType<typeof open>>["stat"]>>;

function validateOwnedTemporaryFile(metadata: FileMetadata, opened?: FileMetadata): void {
  if (!metadata.isFile() || metadata.isSymbolicLink() || metadata.nlink !== 1) {
    throw new Error("atomic JSON temporary path is not an exclusively owned regular file");
  }
  if (typeof process.getuid === "function" && metadata.uid !== process.getuid()) {
    throw new Error("atomic JSON temporary file has an unexpected owner");
  }
  if (process.platform !== "win32" && (metadata.mode & 0o777) !== 0o600) {
    throw new Error("atomic JSON temporary file permissions are not owner-only");
  }
  if (opened && (metadata.dev !== opened.dev || metadata.ino !== opened.ino)) {
    throw new Error("atomic JSON temporary path changed before commit");
  }
}

export async function loadRunState(path: string): Promise<RunState> {
  const file = Bun.file(path);
  if (!(await file.exists())) throw new Error(`run state not found: ${path}`);
  const value = await file.json();
  const errors = validateRunState(value);
  if (errors.length) throw new Error(errors.join("\n"));
  return value as RunState;
}

export function validateRunState(value: unknown): string[] {
  const errors: string[] = [];
  if (!isMap(value)) return ["run state must be a mapping"];
  if (value.schema_version !== "transcribeit.benchmark-run-state.v1") errors.push("unsupported run state schema_version");
  if (typeof value.matrix_sha256 !== "string" || !/^[a-f0-9]{64}$/.test(value.matrix_sha256)) {
    errors.push("run state matrix_sha256 must be lowercase SHA-256");
  } else if (!isMap(value.matrix) || matrixSha256(value.matrix as BenchmarkMatrix) !== value.matrix_sha256) {
    errors.push("run state matrix content does not match matrix_sha256");
  }
  if (isMap(value.matrix)) {
    if (!['entry-fixture-repetition', 'fixture-repetition-entry'].includes(String(value.matrix.attempt_order))) {
      errors.push("run state matrix attempt_order is invalid");
    }
    if (
      !isMap(value.matrix.measurements) ||
      typeof value.matrix.measurements.reference_scoring !== "boolean" ||
      typeof value.matrix.measurements.peak_rss !== "boolean"
    ) {
      errors.push("run state matrix measurements are invalid");
    }
  }
  if (
    !isMap(value.producing_commit) ||
    typeof value.producing_commit.hash !== "string" ||
    typeof value.producing_commit.worktree_dirty !== "boolean" ||
    typeof value.producing_commit.worktree_fingerprint_sha256 !== "string" ||
    !/^[a-f0-9]{64}$/.test(value.producing_commit.worktree_fingerprint_sha256)
  ) {
    errors.push("run state producing_commit is invalid");
  }
  if (!isIsoTimestamp(value.started_at_utc) || !isIsoTimestamp(value.updated_at_utc)) {
    errors.push("run state lifecycle timestamps are invalid");
  } else if (Date.parse(value.updated_at_utc as string) < Date.parse(value.started_at_utc as string)) {
    errors.push("run state updated_at_utc predates started_at_utc");
  }
  if (!Array.isArray(value.attempts)) errors.push("run state attempts must be a list");
  if (typeof value.binary_sha256 !== "string" || !/^[a-f0-9]{64}$/.test(value.binary_sha256)) {
    errors.push("run state binary_sha256 must be lowercase SHA-256");
  }
  const hasEvaluationDownload = isMap(value.matrix) && Array.isArray(value.matrix.entries) && value.matrix.entries.some(
    (entry) => isMap(entry) && isMap(entry.artifact) && entry.artifact.lifecycle === "evaluation_download",
  );
  if (hasEvaluationDownload) {
    if (typeof value.evaluation_root_sha256 !== "string" || !/^[a-f0-9]{64}$/.test(value.evaluation_root_sha256)) {
      errors.push("run state evaluation_root_sha256 must bind the evaluation-download root");
    }
  } else if (value.evaluation_root_sha256 !== null) {
    errors.push("run state evaluation_root_sha256 must be null without evaluation-download artifacts");
  }
  if (!isMap(value.environment) || !isMap(value.environment.machine) || !isMap(value.environment.tools)) {
    errors.push("run state environment is invalid");
  }
  else {
    const keys = new Set<string>();
    for (const [index, attempt] of value.attempts.entries()) {
      if (!isMap(attempt)) {
        errors.push(`attempts[${index}] must be a mapping`);
        continue;
      }
      if (typeof attempt.key !== "string" || !attempt.key) errors.push(`attempts[${index}].key is invalid`);
      else if (keys.has(attempt.key)) errors.push(`attempts[${index}].key is duplicated`);
      else keys.add(attempt.key);
      if (!['pending', 'running', 'passed', 'failed', 'skipped'].includes(String(attempt.status))) {
        errors.push(`attempts[${index}].status is invalid`);
      }
    }
  }
  return errors;
}

export function isMap(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isIsoTimestamp(value: unknown): boolean {
  if (typeof value !== "string") return false;
  const timestamp = Date.parse(value);
  return Number.isFinite(timestamp) && new Date(timestamp).toISOString() === value;
}
