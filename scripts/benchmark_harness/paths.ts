import { lstat, mkdir, realpath } from "node:fs/promises";
import { dirname, relative, resolve, sep } from "node:path";

import { sha256 } from "./state";

export type DirectBenchmarkChild = {
  absolutePath: string;
  repositoryRelativePath: string;
  identitySha256: string;
  state: "absent" | "directory";
};

const repository = resolve(import.meta.dir, "../..");
const outputRoot = resolve(repository, "output");
const benchmarkRoot = resolve(repository, "output/benchmarks");

export async function ensureBenchmarkRoot(): Promise<void> {
  await requireRealDirectory(repository, "repository root", false);
  await requireRealDirectory(outputRoot, "output", true);
  await requireRealDirectory(benchmarkRoot, "output/benchmarks", true);
}

export async function resolveDirectBenchmarkChild(value: string, label: string): Promise<DirectBenchmarkChild> {
  if (!value.trim()) throw new Error(`${label} must be a named direct child of output/benchmarks`);
  const rootMetadata = await lstat(benchmarkRoot).catch(() => undefined);
  if (!rootMetadata?.isDirectory() || rootMetadata.isSymbolicLink()) {
    throw new Error("output/benchmarks must be a real directory");
  }
  if (await realpath(benchmarkRoot) !== benchmarkRoot) throw new Error("output/benchmarks must use its canonical path");
  const absolutePath = resolve(repository, value);
  if (dirname(absolutePath) !== benchmarkRoot) {
    throw new Error(`${label} must be a named direct child of output/benchmarks`);
  }
  const metadata = await lstat(absolutePath).catch((error: unknown) => {
    if (isMissing(error)) return undefined;
    throw new Error(`${label} could not be inspected safely`);
  });
  if (metadata?.isSymbolicLink()) throw new Error(`${label} must not be a symbolic link`);
  if (metadata && !metadata.isDirectory()) throw new Error(`${label} must be a directory when it exists`);
  if (metadata && await realpath(absolutePath) !== absolutePath) throw new Error(`${label} must use its canonical path`);
  const repositoryRelativePath = relative(repository, absolutePath).split(sep).join("/");
  return {
    absolutePath,
    repositoryRelativePath,
    identitySha256: sha256(`transcribeit.benchmark-child.v1\0${repositoryRelativePath}`),
    state: metadata ? "directory" : "absent",
  };
}

function isMissing(error: unknown): boolean {
  return typeof error === "object" && error !== null && "code" in error && error.code === "ENOENT";
}

async function requireRealDirectory(path: string, label: string, create: boolean): Promise<void> {
  let metadata = await lstat(path).catch((error: unknown) => {
    if (isMissing(error)) return undefined;
    throw new Error(`${label} could not be inspected safely`);
  });
  if (!metadata && create) {
    await mkdir(path, { mode: 0o700 }).catch((error: unknown) => {
      if (!isAlreadyExists(error)) throw new Error(`${label} could not be created safely`);
    });
    metadata = await lstat(path).catch(() => undefined);
  }
  if (!metadata?.isDirectory() || metadata.isSymbolicLink()) throw new Error(`${label} must be a real directory`);
  if (await realpath(path) !== path) throw new Error(`${label} must use its canonical path`);
}

function isAlreadyExists(error: unknown): boolean {
  return typeof error === "object" && error !== null && "code" in error && error.code === "EEXIST";
}
