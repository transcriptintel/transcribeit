import { chmod, mkdir, rename } from "node:fs/promises";
import { dirname } from "node:path";

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
  await mkdir(dirname(path), { recursive: true, mode: 0o700 });
  const temporary = `${path}.tmp`;
  await Bun.write(temporary, `${JSON.stringify(value, null, 2)}\n`);
  await chmod(temporary, 0o600);
  await rename(temporary, path);
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
  if (!isMap(value.producing_commit) || typeof value.producing_commit.hash !== "string" || typeof value.producing_commit.worktree_dirty !== "boolean") {
    errors.push("run state producing_commit is invalid");
  }
  if (!Array.isArray(value.attempts)) errors.push("run state attempts must be a list");
  if (typeof value.binary_sha256 !== "string" || !/^[a-f0-9]{64}$/.test(value.binary_sha256)) {
    errors.push("run state binary_sha256 must be lowercase SHA-256");
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
