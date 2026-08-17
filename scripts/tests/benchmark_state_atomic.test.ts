import { lstat, mkdir, mkdtemp, readdir, rm, stat, symlink } from "node:fs/promises";
import { tmpdir } from "node:os";
import { basename, join } from "node:path";

import { writeJsonAtomic } from "../benchmark_harness/state";

describe("atomic benchmark JSON persistence", () => {
  test("does not follow the legacy fixed temporary symlink", async () => {
    if (process.platform === "win32") return;
    const root = await mkdtemp(join(tmpdir(), "transcribeit-atomic-json-"));
    const destination = join(root, "state.json");
    const external = join(root, "external.json");
    const fixedTemporary = `${destination}.tmp`;
    try {
      await Bun.write(external, "external-content\n");
      await symlink(external, fixedTemporary);

      await writeJsonAtomic(destination, { safe: true });

      expect(await Bun.file(external).text()).toBe("external-content\n");
      expect((await lstat(fixedTemporary)).isSymbolicLink()).toBe(true);
      expect(await Bun.file(destination).json()).toEqual({ safe: true });
      expect(await generatedTemporaries(root, destination)).toEqual([]);
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });

  test("commits concurrent writers as complete owner-only regular files", async () => {
    const root = await mkdtemp(join(tmpdir(), "transcribeit-atomic-json-"));
    const destination = join(root, "result.json");
    const values = Array.from({ length: 32 }, (_, writer) => ({ writer, payload: "x".repeat(writer) }));
    try {
      await Promise.all(values.map((value) => writeJsonAtomic(destination, value)));

      const published = await Bun.file(destination).json();
      expect(values.some((value) => JSON.stringify(value) === JSON.stringify(published))).toBe(true);
      const metadata = await stat(destination);
      expect(metadata.isFile()).toBe(true);
      expect(metadata.mode & 0o777).toBe(0o600);
      expect(await generatedTemporaries(root, destination)).toEqual([]);
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });

  test("removes only its generated temporary file when rename fails", async () => {
    const root = await mkdtemp(join(tmpdir(), "transcribeit-atomic-json-"));
    const destination = join(root, "occupied");
    const unrelated = join(root, "unrelated.json");
    try {
      await mkdir(destination);
      await Bun.write(unrelated, "leave-me-alone\n");

      await expect(writeJsonAtomic(destination, { cannot: "replace-directory" })).rejects.toThrow();

      expect(await Bun.file(unrelated).text()).toBe("leave-me-alone\n");
      expect((await stat(destination)).isDirectory()).toBe(true);
      expect(await generatedTemporaries(root, destination)).toEqual([]);
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });
});

async function generatedTemporaries(parent: string, destination: string): Promise<string[]> {
  const prefix = `.${basename(destination)}.`;
  return (await readdir(parent)).filter((entry) => entry.startsWith(prefix) && entry.endsWith(".tmp"));
}
