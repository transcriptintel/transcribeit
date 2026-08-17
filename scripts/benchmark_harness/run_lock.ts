import { randomUUID } from "node:crypto";
import { constants } from "node:fs";
import { lstat, mkdir, open, realpath, rename, rmdir, unlink } from "node:fs/promises";
import { join, resolve } from "node:path";

import { validateRunState } from "./state";

type FileMetadata = Awaited<ReturnType<Awaited<ReturnType<typeof open>>["stat"]>>;

type LockIdentity = {
  dev: number | bigint;
  ino: number | bigint;
};

type LockMetadata = LockIdentity & {
  pid: number;
};

export type RunLock = {
  release(): Promise<void>;
};

export const runLockFileName = ".benchmark-run.lock";
const recoveryDirectoryName = ".benchmark-run.lock-recovery";
const lockSchema = "transcribeit.benchmark-run-lock.v1";
const maximumLockBytes = 4 * 1024;
const maximumStateBytes = 16 * 1024 * 1024;

export async function acquireRunLock(runDirectory: string): Promise<RunLock> {
  const directory = resolve(runDirectory);
  await validateRunDirectory(directory);
  const lockPath = join(directory, runLockFileName);
  const recoveryPath = join(directory, recoveryDirectoryName);
  if (await pathExists(recoveryPath)) throw activeRunError();

  try {
    return await createLock(lockPath);
  } catch (error) {
    if (!isAlreadyExists(error)) throw safeLockError("benchmark run lock could not be created safely");
  }

  const existing = await readLock(lockPath);
  if (!existing) return acquireRunLock(directory);
  if (processIsAlive(existing.pid)) throw activeRunError();
  return recoverStaleLock(directory, lockPath, recoveryPath, existing);
}

async function createLock(lockPath: string): Promise<RunLock> {
  const flags = constants.O_WRONLY | constants.O_CREAT | constants.O_EXCL | noFollowFlag();
  const handle = await open(lockPath, flags, 0o600);
  let acquired = false;
  try {
    await handle.chmod(0o600);
    const payload = `${JSON.stringify({
      schema_version: lockSchema,
      pid: process.pid,
      owner_token: randomUUID(),
      created_at_utc: new Date().toISOString(),
    })}\n`;
    await handle.writeFile(payload, "utf8");
    await handle.sync();
    const opened = await handle.stat();
    validateLockFile(opened);
    const named = await lstat(lockPath);
    validateLockFile(named, opened);
    acquired = true;
    let released = false;
    return {
      async release(): Promise<void> {
        if (released) return;
        released = true;
        let releaseError: unknown;
        try {
          const current = await lstat(lockPath);
          validateLockFile(current, opened);
          await unlink(lockPath);
        } catch {
          releaseError = safeLockError("benchmark run lock ownership changed before release");
        } finally {
          await handle.close().catch(() => undefined);
        }
        if (releaseError) throw releaseError;
      },
    };
  } finally {
    if (!acquired) {
      await handle.close().catch(() => undefined);
      await unlink(lockPath).catch(() => undefined);
    }
  }
}

async function recoverStaleLock(
  directory: string,
  lockPath: string,
  recoveryPath: string,
  expected: LockMetadata,
): Promise<RunLock> {
  try {
    await mkdir(recoveryPath, { mode: 0o700 });
  } catch (error) {
    if (isAlreadyExists(error)) throw activeRunError();
    throw safeLockError("benchmark run lock recovery could not start safely");
  }

  const quarantinePath = join(directory, `.benchmark-run.lock-stale-${randomUUID()}`);
  let quarantined = false;
  try {
    await validateRecoveryDirectory(recoveryPath);
    const current = await readLock(lockPath);
    if (!current) return await createLock(lockPath);
    if (!sameIdentity(current, expected) || processIsAlive(current.pid)) throw activeRunError();
    await assertStaleRecoveryAllowed(directory);
    await rename(lockPath, quarantinePath);
    quarantined = true;
    const moved = await lstat(quarantinePath);
    validateLockFile(moved);
    if (!sameIdentity(moved, expected)) {
      throw safeLockError("benchmark run lock changed during stale recovery");
    }
    await unlink(quarantinePath);
    quarantined = false;
    try {
      return await createLock(lockPath);
    } catch (error) {
      if (isAlreadyExists(error)) throw activeRunError();
      throw error;
    }
  } finally {
    if (quarantined) await unlink(quarantinePath).catch(() => undefined);
    await rmdir(recoveryPath).catch(() => undefined);
  }
}

async function readLock(lockPath: string): Promise<LockMetadata | null> {
  const named = await lstat(lockPath).catch((error: unknown) => {
    if (isMissing(error)) return null;
    throw safeLockError("benchmark run lock could not be inspected safely");
  });
  if (!named) return null;
  validateLockFile(named);
  if (named.size <= 0 || named.size > maximumLockBytes) throw invalidLockError();

  let handle: Awaited<ReturnType<typeof open>>;
  try {
    handle = await open(lockPath, constants.O_RDONLY | noFollowFlag());
  } catch {
    throw invalidLockError();
  }
  try {
    const opened = await handle.stat();
    validateLockFile(opened, named);
    if (opened.size <= 0 || opened.size > maximumLockBytes) throw invalidLockError();
    const payload = new Uint8Array(maximumLockBytes + 1);
    const { bytesRead } = await handle.read(payload, 0, payload.byteLength, 0);
    if (bytesRead <= 0 || bytesRead > maximumLockBytes) throw invalidLockError();
    const parsed = JSON.parse(new TextDecoder().decode(payload.subarray(0, bytesRead))) as unknown;
    if (!isValidLockPayload(parsed)) throw invalidLockError();
    return { dev: opened.dev, ino: opened.ino, pid: parsed.pid };
  } catch (error) {
    if (error instanceof Error && error.message.startsWith("benchmark run lock")) throw error;
    throw invalidLockError();
  } finally {
    await handle.close().catch(() => undefined);
  }
}

async function assertStaleRecoveryAllowed(directory: string): Promise<void> {
  const statePath = join(directory, "state.json");
  const named = await lstat(statePath).catch((error: unknown) => {
    if (isMissing(error)) return null;
    throw staleRecoveryBlockedError();
  });
  if (!named) return;

  let handle: Awaited<ReturnType<typeof open>> | undefined;
  try {
    validateStateFile(named);
    handle = await open(statePath, constants.O_RDONLY | noFollowFlag());
    const opened = await handle.stat();
    validateStateFile(opened, named);
    const payload = new Uint8Array(opened.size + 1);
    const { bytesRead } = await handle.read(payload, 0, payload.byteLength, 0);
    if (bytesRead !== opened.size) throw staleRecoveryBlockedError();
    const afterRead = await handle.stat();
    validateStateFile(afterRead, opened);
    if (afterRead.size !== opened.size) throw staleRecoveryBlockedError();
    const namedAfterRead = await lstat(statePath);
    validateStateFile(namedAfterRead, opened);
    if (namedAfterRead.size !== opened.size) throw staleRecoveryBlockedError();

    const value = JSON.parse(new TextDecoder().decode(payload.subarray(0, bytesRead))) as unknown;
    if (validateRunState(value).length > 0) throw staleRecoveryBlockedError();
    const attempts = (value as Record<string, unknown>).attempts as Array<Record<string, unknown>>;
    if (attempts.some((attempt) => attempt.status === "running")) throw staleRecoveryBlockedError();
  } catch {
    throw staleRecoveryBlockedError();
  } finally {
    await handle?.close().catch(() => undefined);
  }
}

function validateLockFile(metadata: FileMetadata, expected?: FileMetadata): void {
  if (!metadata.isFile() || metadata.isSymbolicLink() || metadata.nlink !== 1) throw invalidLockError();
  if (typeof process.getuid === "function" && metadata.uid !== process.getuid()) throw invalidLockError();
  if (process.platform !== "win32" && (metadata.mode & 0o777) !== 0o600) throw invalidLockError();
  if (expected && !sameIdentity(metadata, expected)) throw invalidLockError();
}

function validateStateFile(metadata: FileMetadata, expected?: FileMetadata): void {
  if (!metadata.isFile() || metadata.isSymbolicLink() || metadata.nlink !== 1) throw staleRecoveryBlockedError();
  if (metadata.size <= 0 || metadata.size > maximumStateBytes) throw staleRecoveryBlockedError();
  if (typeof process.getuid === "function" && metadata.uid !== process.getuid()) throw staleRecoveryBlockedError();
  if (process.platform !== "win32" && (metadata.mode & 0o777) !== 0o600) throw staleRecoveryBlockedError();
  if (expected && !sameIdentity(metadata, expected)) throw staleRecoveryBlockedError();
}

async function validateRunDirectory(directory: string): Promise<void> {
  const metadata = await lstat(directory).catch(() => undefined);
  if (!metadata?.isDirectory() || metadata.isSymbolicLink()) {
    throw safeLockError("benchmark run directory is not safe for locking");
  }
  if (await realpath(directory).catch(() => "") !== directory) {
    throw safeLockError("benchmark run directory is not safe for locking");
  }
  if (typeof process.getuid === "function" && metadata.uid !== process.getuid()) {
    throw safeLockError("benchmark run directory is not owned by the current user");
  }
  if (process.platform !== "win32" && (metadata.mode & 0o077) !== 0) {
    throw safeLockError("benchmark run directory permissions must be owner-only");
  }
}

async function validateRecoveryDirectory(path: string): Promise<void> {
  const metadata = await lstat(path);
  if (!metadata.isDirectory() || metadata.isSymbolicLink()) {
    throw safeLockError("benchmark run lock recovery is not safe");
  }
  if (await realpath(path).catch(() => "") !== path) {
    throw safeLockError("benchmark run lock recovery is not safe");
  }
  if (typeof process.getuid === "function" && metadata.uid !== process.getuid()) {
    throw safeLockError("benchmark run lock recovery is not owned by the current user");
  }
  if (process.platform !== "win32" && (metadata.mode & 0o077) !== 0) {
    throw safeLockError("benchmark run lock recovery permissions must be owner-only");
  }
}

function processIsAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    if (hasCode(error, "ESRCH")) return false;
    if (hasCode(error, "EPERM")) return true;
    throw safeLockError("benchmark run lock owner could not be checked safely");
  }
}

function isValidLockPayload(value: unknown): value is {
  schema_version: string;
  pid: number;
  owner_token: string;
  created_at_utc: string;
} {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return false;
  const lock = value as Record<string, unknown>;
  if (Object.keys(lock).sort().join(",") !== "created_at_utc,owner_token,pid,schema_version") return false;
  if (lock.schema_version !== lockSchema) return false;
  if (!Number.isSafeInteger(lock.pid) || (lock.pid as number) <= 0) return false;
  if (typeof lock.owner_token !== "string" || !/^[a-f0-9-]{36}$/.test(lock.owner_token)) return false;
  if (typeof lock.created_at_utc !== "string") return false;
  const timestamp = Date.parse(lock.created_at_utc);
  return Number.isFinite(timestamp) && new Date(timestamp).toISOString() === lock.created_at_utc;
}

function sameIdentity(left: LockIdentity, right: LockIdentity): boolean {
  return left.dev === right.dev && left.ino === right.ino;
}

function noFollowFlag(): number {
  return process.platform !== "win32" && typeof constants.O_NOFOLLOW === "number" ? constants.O_NOFOLLOW : 0;
}

async function pathExists(path: string): Promise<boolean> {
  return lstat(path).then(() => true).catch((error: unknown) => {
    if (isMissing(error)) return false;
    throw safeLockError("benchmark run lock recovery could not be inspected safely");
  });
}

function activeRunError(): Error {
  return safeLockError("benchmark run is already active or lock recovery is in progress");
}

function invalidLockError(): Error {
  return safeLockError("benchmark run lock is invalid; confirm no benchmark is active before removing it");
}

function staleRecoveryBlockedError(): Error {
  return safeLockError("stale benchmark lock requires operator intervention because run state may still be active");
}

function safeLockError(message: string): Error {
  return new Error(message);
}

function hasCode(error: unknown, code: string): boolean {
  return typeof error === "object" && error !== null && "code" in error && error.code === code;
}

function isAlreadyExists(error: unknown): boolean {
  return hasCode(error, "EEXIST");
}

function isMissing(error: unknown): boolean {
  return hasCode(error, "ENOENT");
}
