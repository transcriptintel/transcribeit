import { lstat, readlink } from "node:fs/promises";
import { resolve } from "node:path";

const repository = resolve(import.meta.dir, "../..");

export type RepositoryProvenance = {
  hash: string;
  worktree_dirty: boolean;
  worktree_fingerprint_sha256: string;
};

export async function captureRepositoryProvenance(): Promise<RepositoryProvenance> {
  const decoder = new TextDecoder();
  const hash = decoder.decode(git(["rev-parse", "HEAD"])).trim();
  if (!/^[a-f0-9]{40,64}$/.test(hash)) throw new Error("cannot resolve repository HEAD for benchmark provenance");
  const status = git(["status", "--porcelain=v1", "-z", "--untracked-files=all"]);
  const diff = git(["diff", "--binary", "--no-ext-diff", "HEAD", "--"]);
  const untracked = decoder.decode(git(["ls-files", "--others", "--exclude-standard", "-z"]))
    .split("\0")
    .filter(Boolean)
    .sort();
  const hasher = new Bun.CryptoHasher("sha256");
  hasher.update("transcribeit.repository-provenance.v1\0");
  hasher.update(hash);
  hasher.update("\0status\0");
  hasher.update(status);
  hasher.update("\0diff\0");
  hasher.update(diff);
  for (const path of untracked) {
    const absolute = resolve(repository, path);
    const metadata = await lstat(absolute);
    hasher.update("\0untracked\0");
    hasher.update(path);
    hasher.update("\0");
    if (metadata.isSymbolicLink()) hasher.update(await readlink(absolute));
    else if (metadata.isFile()) hasher.update(new Uint8Array(await Bun.file(absolute).arrayBuffer()));
    else hasher.update(`unsupported:${metadata.mode}`);
  }
  return {
    hash,
    worktree_dirty: status.byteLength > 0,
    worktree_fingerprint_sha256: hasher.digest("hex"),
  };
}

function git(args: string[]): Uint8Array {
  const child = Bun.spawnSync(["git", ...args], { cwd: repository, stdout: "pipe", stderr: "ignore" });
  if (child.exitCode !== 0) throw new Error("cannot inspect repository provenance");
  return new Uint8Array(child.stdout);
}
