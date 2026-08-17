import { chmod, lstat, mkdir, open, realpath, rm } from "node:fs/promises";
import { join, resolve, sep } from "node:path";

import { hasEvaluationDownload } from "./cleanup";
import { ensureBenchmarkRoot, resolveDirectBenchmarkChild } from "./paths";
import { acquireRunLock, type RunLock } from "./run_lock";
import type { BenchmarkMatrix } from "./types";

export type EvaluationRootBinding = {
  absolutePath: string;
  identitySha256: string;
} | null;

export async function prepareRunDirectory(value: string): Promise<{
  runDirectory: string;
  attemptsRoot: string;
  lock: RunLock;
}> {
  await ensureBenchmarkRoot();
  const candidate = await resolveDirectBenchmarkChild(value, "run directory");
  if (candidate.state === "absent") await mkdir(candidate.absolutePath, { mode: 0o700 });
  const confirmed = await resolveDirectBenchmarkChild(candidate.absolutePath, "run directory");
  if (confirmed.state !== "directory") throw new Error("run directory could not be created safely");
  await chmod(confirmed.absolutePath, 0o700);
  const lock = await acquireRunLock(confirmed.absolutePath);
  try {
    await ensureEmptyDotEnv(confirmed.absolutePath);
    const attemptsRoot = join(confirmed.absolutePath, "attempts");
    await ensureDirectDirectory(attemptsRoot, confirmed.absolutePath, "attempts directory");
    return { runDirectory: confirmed.absolutePath, attemptsRoot, lock };
  } catch (error) {
    await lock.release().catch(() => undefined);
    throw error;
  }
}

export async function evaluationRootBinding(
  matrix: BenchmarkMatrix,
  runDirectory: string,
): Promise<EvaluationRootBinding> {
  if (!hasEvaluationDownload(matrix)) return null;
  const configured = process.env.MODEL_CACHE_DIR;
  if (!configured) throw new Error("evaluation-download matrices require MODEL_CACHE_DIR");
  const root = await resolveDirectBenchmarkChild(configured, "MODEL_CACHE_DIR");
  if (root.state !== "directory") {
    throw new Error("MODEL_CACHE_DIR must exist before an evaluation-download benchmark starts");
  }
  if (root.absolutePath === runDirectory) {
    throw new Error("MODEL_CACHE_DIR must be distinct from the benchmark run directory");
  }
  return { absolutePath: root.absolutePath, identitySha256: root.identitySha256 };
}

export async function ensureDirectDirectory(target: string, parent: string, label: string): Promise<void> {
  const state = await directDirectoryState(target, parent, label);
  if (state === "absent") await mkdir(resolve(target), { mode: 0o700 });
  if (await directDirectoryState(target, parent, label) !== "directory") {
    throw new Error(`${label} could not be created safely`);
  }
}

export async function removeDirectDirectory(target: string, parent: string, label: string): Promise<void> {
  if (await directDirectoryState(target, parent, label) === "absent") return;
  await rm(resolve(target), { recursive: true });
  if (await directDirectoryState(target, parent, label) !== "absent") {
    throw new Error(`${label} could not be removed safely`);
  }
}

async function ensureEmptyDotEnv(runDirectory: string): Promise<void> {
  const path = join(runDirectory, ".env");
  let metadata = await lstatIfExists(path, "run .env sentinel");
  if (!metadata) {
    try {
      const handle = await open(path, "wx", 0o600);
      await handle.close();
    } catch (error) {
      if (!isAlreadyExists(error)) throw new Error("run .env sentinel could not be created safely");
    }
    metadata = await lstatIfExists(path, "run .env sentinel");
  }
  if (!metadata || metadata.isSymbolicLink() || !metadata.isFile() || metadata.size !== 0) {
    throw new Error("run .env sentinel must be an empty non-symlink file");
  }
  if (await realpath(path) !== path) throw new Error("run .env sentinel must use its canonical path");
  await chmod(path, 0o600);
}

async function directDirectoryState(target: string, parent: string, label: string): Promise<"absent" | "directory"> {
  const resolvedParent = resolve(parent);
  const parentMetadata = await lstatIfExists(resolvedParent, `${label} parent`);
  if (!parentMetadata || parentMetadata.isSymbolicLink() || !parentMetadata.isDirectory()) {
    throw new Error(`${label} parent must be a non-symlink directory`);
  }
  if (await realpath(resolvedParent) !== resolvedParent) {
    throw new Error(`${label} parent must resolve to its canonical directory`);
  }
  const resolvedTarget = resolve(target);
  const prefix = `${resolvedParent}${sep}`;
  const childName = resolvedTarget.startsWith(prefix) ? resolvedTarget.slice(prefix.length) : "";
  if (!childName || childName.includes(sep)) throw new Error(`${label} must be a direct child of its parent`);
  const metadata = await lstatIfExists(resolvedTarget, label);
  if (!metadata) return "absent";
  if (metadata.isSymbolicLink()) throw new Error(`${label} must not be a symbolic link`);
  if (!metadata.isDirectory()) throw new Error(`${label} must be a directory when it exists`);
  if (await realpath(resolvedTarget) !== resolvedTarget) {
    throw new Error(`${label} must resolve to its canonical directory`);
  }
  return "directory";
}

async function lstatIfExists(path: string, label: string) {
  return lstat(path).catch((error: unknown) => {
    if (isMissing(error)) return undefined;
    throw new Error(`${label} could not be inspected safely`);
  });
}

function isMissing(error: unknown): boolean {
  return typeof error === "object" && error !== null && "code" in error && error.code === "ENOENT";
}

function isAlreadyExists(error: unknown): boolean {
  return typeof error === "object" && error !== null && "code" in error && error.code === "EEXIST";
}
