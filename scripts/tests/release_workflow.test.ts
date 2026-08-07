import { join } from "node:path";

const repository = join(import.meta.dir, "../..");

describe("release workflow", () => {
  test("guards tags and publishes the supported archive matrix", async () => {
    const workflow = Bun.YAML.parse(
      await Bun.file(join(repository, ".github/workflows/release.yml")).text(),
    ) as any;
    expect(workflow.on.push.tags).toEqual(["v*"]);
    expect(workflow.permissions.contents).toBe("read");

    const guardSource = workflow.jobs.guard.steps
      .map((step: { run?: string }) => step.run ?? "")
      .join("\n");
    expect(guardSource).toContain("git merge-base --is-ancestor");
    expect(guardSource).toContain("cargo metadata --locked");
    expect(guardSource).toContain("tag_version");

    const variants = workflow.jobs.build.strategy.matrix.include;
    expect(variants.map((variant: { target: string }) => variant.target)).toEqual([
      "x86_64-unknown-linux-gnu",
      "aarch64-unknown-linux-gnu",
      "x86_64-apple-darwin",
      "aarch64-apple-darwin",
      "x86_64-pc-windows-msvc",
    ]);
    const buildSource = workflow.jobs.build.steps
      .map((step: { run?: string }) => step.run ?? "")
      .join("\n");
    expect(buildSource).toContain("cargo build --release --locked --target");
    expect(buildSource).toContain("README.md LICENSE");

    const release = workflow.jobs.release;
    expect(release.needs).toBe("build");
    expect(release.permissions.contents).toBe("write");
    expect(release.steps.some((step: { uses?: string }) => step.uses === "actions/download-artifact@v8")).toBe(true);
    expect(release.steps.some((step: { uses?: string }) => step.uses === "softprops/action-gh-release@v3")).toBe(true);
    expect(release.steps.some((step: { run?: string }) => step.run?.includes("sha256sum"))).toBe(true);
  });

  test("preflights release builds on the primary runner platforms", async () => {
    const workflow = Bun.YAML.parse(
      await Bun.file(join(repository, ".github/workflows/ci.yml")).text(),
    ) as any;
    expect(workflow.on.workflow_dispatch).toBeDefined();
    const job = workflow.jobs["release-build"];
    expect(job.strategy.matrix.include.map((variant: { os: string }) => variant.os)).toEqual([
      "ubuntu-latest",
      "macos-latest",
      "windows-latest",
    ]);
    const source = job.steps.map((step: { run?: string }) => step.run ?? "").join("\n");
    expect(source).toContain("cargo build --release --locked --target");
    expect(source).toContain("--version");
  });
});
