import { stat } from "node:fs/promises";
import { resolve } from "node:path";

import { repositoryRoot } from "./config";
import type { FixtureIdentity } from "./types";

export type BenchmarkBinaryIdentity = {
  path: string;
  sha256: string;
};

export async function verifyInitialFixtures(fixtures: Map<string, FixtureIdentity>): Promise<void> {
  for (const fixture of fixtures.values()) {
    const path = resolve(repositoryRoot(), fixture.path);
    const metadata = await stat(path).catch(() => undefined);
    if (!metadata?.isFile() || metadata.size !== fixture.bytes) {
      throw new Error(`fixture ${fixture.id} is missing or has the wrong byte size; run corpus fetch/verify`);
    }
    const digest = await sha256File(path).catch(() => undefined);
    if (digest !== fixture.sha256) {
      throw new Error(`fixture ${fixture.id} failed SHA-256 verification`);
    }
  }
}

export async function verifyAttemptInputs(
  binary: BenchmarkBinaryIdentity,
  fixture: FixtureIdentity,
): Promise<void> {
  await verifyBinaryUnchanged(binary);
  await verifyFixtureUnchanged(fixture);
}

export async function verifyAllRunInputs(
  binary: BenchmarkBinaryIdentity,
  fixtures: Map<string, FixtureIdentity>,
): Promise<void> {
  await verifyBinaryUnchanged(binary);
  for (const fixture of fixtures.values()) await verifyFixtureUnchanged(fixture);
}

async function verifyBinaryUnchanged(expected: BenchmarkBinaryIdentity): Promise<void> {
  try {
    const before = await stat(expected.path);
    if (!before.isFile()) throw new Error("not a regular file");
    const digest = await sha256File(expected.path);
    const after = await stat(expected.path);
    if (!after.isFile() || before.size !== after.size || digest !== expected.sha256) {
      throw new Error("identity mismatch");
    }
  } catch {
    throw new Error("benchmark binary identity changed during run");
  }
}

async function verifyFixtureUnchanged(fixture: FixtureIdentity): Promise<void> {
  try {
    const path = resolve(repositoryRoot(), fixture.path);
    const before = await stat(path);
    if (!before.isFile() || before.size !== fixture.bytes) throw new Error("identity mismatch");
    const digest = await sha256File(path);
    const after = await stat(path);
    if (!after.isFile() || after.size !== fixture.bytes || before.size !== after.size || digest !== fixture.sha256) {
      throw new Error("identity mismatch");
    }
  } catch {
    throw new Error("benchmark fixture identity changed during run");
  }
}

async function sha256File(path: string): Promise<string> {
  const hasher = new Bun.CryptoHasher("sha256");
  const reader = Bun.file(path).stream().getReader();
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      hasher.update(value);
    }
  } finally {
    reader.releaseLock();
  }
  return hasher.digest("hex");
}
