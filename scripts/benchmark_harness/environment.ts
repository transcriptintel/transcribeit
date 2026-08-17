import { cpus, totalmem } from "node:os";
import { basename, relative, resolve } from "node:path";

import { repositoryRoot } from "./config";
import { sha256 } from "./state";
import type { RunState } from "./types";

export async function binaryIdentity(binary: string): Promise<{ path: string; classification: string; sha256: string }> {
  const repository = repositoryRoot();
  const path = resolve(repository, binary);
  const file = Bun.file(path);
  if (!(await file.exists())) throw new Error(`benchmark binary not found: ${binary}`);
  const relativePath = relative(repository, path);
  const classification = relativePath.startsWith("..") || relativePath.startsWith("/") ? basename(path) : relativePath;
  return { path, classification, sha256: sha256(new Uint8Array(await file.arrayBuffer())) };
}

export function captureEnvironment(): RunState["environment"] {
  const command = (args: string[]): string => Bun.spawnSync(args, { stdout: "pipe", stderr: "ignore" }).stdout.toString().trim();
  const firstLine = (value: string): string => value.split("\n", 1)[0];
  const processors = cpus();
  const memoryValue = totalmem();
  const macos = process.platform === "darwin";
  return {
    machine: {
      cpu: processors[0]?.model || command(["uname", "-m"]) || "unknown",
      logical_cores: processors.length || navigator.hardwareConcurrency,
      memory_bytes: Number.isFinite(memoryValue) && memoryValue > 0 ? memoryValue : null,
      os: process.platform,
      os_version: macos ? command(["sw_vers", "-productVersion"]) || null : null,
      os_build: macos ? command(["sw_vers", "-buildVersion"]) || null : null,
      kernel: command(["uname", "-r"]) || "unknown",
      architecture: process.arch,
    },
    tools: {
      rustc: firstLine(command(["rustc", "--version"])) || "unavailable",
      ffmpeg: firstLine(command(["ffmpeg", "-version"])) || "unavailable",
      swift: firstLine(command(["swift", "--version"])) || "unavailable",
      bun: Bun.version,
    },
  };
}
