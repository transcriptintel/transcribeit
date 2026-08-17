import { join } from "node:path";

const repository = join(import.meta.dir, "../..");
const node24AuditCommit = "858dc40f52ca2b8570b7a997c1c4e35c6fc9a432";

describe("security workflow", () => {
  test("uses the Node 24 RustSec action with its required report permissions", async () => {
    const workflow = Bun.YAML.parse(
      await Bun.file(join(repository, ".github/workflows/security.yml")).text(),
    ) as any;

    expect(workflow.permissions.contents).toBe("read");
    expect(workflow.permissions.checks).toBe("write");
    expect(workflow.permissions.issues).toBe("write");

    const audit = workflow.jobs.rustsec.steps.find(
      (step: { uses?: string }) => step.uses?.startsWith("rustsec/audit-check@"),
    );
    expect(audit?.uses).toBe(`rustsec/audit-check@${node24AuditCommit}`);
    expect(audit?.with?.token).toBe("${{ secrets.GITHUB_TOKEN }}");
  });
});
