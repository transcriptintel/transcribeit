#!/usr/bin/env bun

import { existsSync } from "node:fs";
import { basename, join } from "node:path";

export type IssueStatus = "open" | "in-progress" | "resolved";

export type Issue = {
  id: string;
  title: string;
  priority: string;
  status: IssueStatus;
  area: string;
  resolved?: string;
  path: string;
  body: string;
};

type YamlMap = Record<string, unknown>;

const repository = join(import.meta.dir, "..");
const issuesDirectory = join(repository, "docs/issues");
const samplesDirectory = join(repository, "samples");
const registryPath = join(issuesDirectory, "README.md");
const issuePattern = /^TI-\d{3}$/;
const mediaFilenamePattern =
  /\b[\w@.+-]+\.(?:aac|aiff|flac|m4a|mka|mov|mp3|mp4|ogg|opus|wav|webm)\b/i;
const allowedFields = new Set(["id", "title", "priority", "status", "area", "resolved"]);

function asMap(value: unknown): YamlMap | undefined {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return undefined;
  return value as YamlMap;
}

function stringField(map: YamlMap, key: string): string | undefined {
  const value = map[key];
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

export function parseIssuePage(path: string, source: string): { issue?: Issue; errors: string[] } {
  const errors: string[] = [];
  const match = source.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n([\s\S]+)$/);
  if (!match) return { errors: [`${path}: expected YAML frontmatter and a non-empty body`] };

  let metadata: YamlMap | undefined;
  try {
    metadata = asMap(Bun.YAML.parse(match[1]));
  } catch (error) {
    errors.push(`${path}: invalid YAML frontmatter: ${String(error)}`);
  }
  if (!metadata) return { errors: [...errors, `${path}: frontmatter must be a mapping`] };

  const unexpected = Object.keys(metadata).filter((key) => !allowedFields.has(key));
  if (unexpected.length) errors.push(`${path}: unsupported fields: ${unexpected.join(", ")}`);

  const id = stringField(metadata, "id");
  const title = stringField(metadata, "title");
  const priority = stringField(metadata, "priority");
  const status = stringField(metadata, "status") as IssueStatus | undefined;
  const area = stringField(metadata, "area");
  const resolved = stringField(metadata, "resolved");
  const expectedId = basename(path, ".md");

  if (!id || !issuePattern.test(id)) errors.push(`${path}: id must match TI-NNN`);
  if (id && id !== expectedId) errors.push(`${path}: id ${id} must match filename ${expectedId}`);
  if (!title) errors.push(`${path}: title must be a non-empty string`);
  if (!priority || !/^P[0-3]$/.test(priority)) {
    errors.push(`${path}: priority must be P0, P1, P2, or P3`);
  }
  if (!status || !["open", "in-progress", "resolved"].includes(status)) {
    errors.push(`${path}: status must be open, in-progress, or resolved`);
  }
  if (!area) errors.push(`${path}: area must be a non-empty string`);
  if (status === "resolved" && (!resolved || !/^\d{4}-\d{2}-\d{2}$/.test(resolved))) {
    errors.push(`${path}: resolved issues require an ISO resolved date`);
  }
  if (status !== "resolved" && resolved) {
    errors.push(`${path}: unresolved issues cannot set resolved`);
  }

  const body = `${match[2].trimEnd()}\n`;
  if (mediaFilenamePattern.test(body)) {
    errors.push(
      `${path}: issue pages must not include media filenames; use a sanitized fixture id or hash`,
    );
  }
  if (id && title && !body.startsWith(`# ${id} — ${title}\n`)) {
    errors.push(`${path}: body heading must match id and title`);
  }
  if (!body.includes("\n## Required outcome and acceptance\n")) {
    errors.push(`${path}: body must contain '## Required outcome and acceptance'`);
  }
  if (status === "resolved" && !body.includes("\n## Outcome and validation\n")) {
    errors.push(`${path}: resolved body must contain '## Outcome and validation'`);
  }
  if (body.length < 200) errors.push(`${path}: issue record is too short to preserve evidence`);

  if (errors.length || !id || !title || !priority || !status || !area) return { errors };
  return { issue: { id, title, priority, status, area, resolved, path, body }, errors };
}

export function containsSampleFilename(source: string, filenames: Iterable<string>): boolean {
  const normalizedSource = source.toLocaleLowerCase("en-US");
  return Array.from(filenames).some((filename) =>
    normalizedSource.includes(filename.toLocaleLowerCase("en-US")),
  );
}

async function sampleFilenames(): Promise<Set<string>> {
  const filenames = new Set<string>();
  if (!existsSync(samplesDirectory)) return filenames;
  const glob = new Bun.Glob("**/*");
  for await (const path of glob.scan({ cwd: samplesDirectory, onlyFiles: true })) {
    filenames.add(basename(path));
  }
  return filenames;
}

function displayStatus(status: IssueStatus): string {
  return status === "in-progress" ? "In progress" : status[0].toUpperCase() + status.slice(1);
}

export function renderRegistry(issues: Issue[]): string {
  const active = issues.filter((issue) => issue.status !== "resolved").length;
  const rows = issues
    .map(
      (issue) =>
        `| [${issue.id}](./${issue.id}.md) | ${issue.priority} | ${displayStatus(issue.status)} | ${issue.area} | ${issue.title} |`,
    )
    .join("\n");
  return `# TranscribeIt issue registry

This is the canonical tracked registry for TranscribeIt engineering findings and
planned implementation work. Each \`TI-NNN\` page owns its status, priority,
acceptance criteria, outcome, and validation evidence. Regenerate this index with
\`bun run scripts/issues_registry.ts generate\` and verify it with
\`bun run scripts/issues_registry.ts check\`.

- Total findings: ${issues.length}
- Active findings: ${active}

| ID | Priority | Status | Area | Summary |
|---|---:|---|---|---|
${rows}

## Working agreement

1. Select the highest-priority applicable issue and set its frontmatter status to
   \`in-progress\` before implementation.
2. Confirm the live code still supports the finding and keep the required outcome
   and acceptance criteria current.
3. Align implementation, regression coverage, public documentation, examples,
   benchmark evidence, and provider/runtime boundaries.
4. Set an issue to \`resolved\` only after its required gates pass. Add the ISO
   resolution date plus concrete outcome and validation evidence.
5. Run \`bun run scripts/issues_registry.ts generate\` and then the matching
   \`check\`; never hand-edit this generated index.
6. Identify private fixtures only with sanitized ids, hashes, sizes, and durations.
   Never include a filename from \`samples/\` in an issue page.
`;
}

export async function loadIssues(): Promise<{ issues: Issue[]; errors: string[] }> {
  const glob = new Bun.Glob("TI-*.md");
  const paths: string[] = [];
  for await (const path of glob.scan({ cwd: issuesDirectory, onlyFiles: true })) paths.push(path);
  paths.sort();

  const issues: Issue[] = [];
  const errors: string[] = [];
  const ids = new Set<string>();
  const protectedFilenames = await sampleFilenames();
  for (const relativePath of paths) {
    const path = `docs/issues/${relativePath}`;
    const parsed = parseIssuePage(path, await Bun.file(join(issuesDirectory, relativePath)).text());
    errors.push(...parsed.errors);
    if (!parsed.issue) continue;
    if (containsSampleFilename(parsed.issue.body, protectedFilenames)) {
      errors.push(
        `${path}: issue pages must not include filenames found under samples/; use a sanitized fixture id or hash`,
      );
    }
    if (ids.has(parsed.issue.id)) errors.push(`${path}: duplicate id ${parsed.issue.id}`);
    ids.add(parsed.issue.id);
    issues.push(parsed.issue);
  }
  issues.sort((left, right) => left.id.localeCompare(right.id));
  if (!issues.length) errors.push("docs/issues: no TI-NNN pages found");
  if (issues.length) {
    const maximum = Number(issues.at(-1)?.id.slice(3));
    for (let index = 1; index <= maximum; index += 1) {
      const id = `TI-${String(index).padStart(3, "0")}`;
      if (!ids.has(id)) errors.push(`docs/issues: missing ${id}`);
    }
  }
  return { issues, errors };
}

async function run(command: string): Promise<void> {
  const { issues, errors } = await loadIssues();
  const registry = renderRegistry(issues);
  if (command === "generate") {
    if (errors.length) throw new Error(errors.join("\n"));
    await Bun.write(registryPath, registry);
    console.log(`Generated issue registry for ${issues.length} findings.`);
    return;
  }
  if (command !== "check") {
    throw new Error("usage: bun run scripts/issues_registry.ts <check|generate>");
  }
  const file = Bun.file(registryPath);
  if (!(await file.exists()) || (await file.text()) !== registry) {
    errors.push("docs/issues/README.md: generated content is stale; run the generate command");
  }
  if (errors.length) throw new Error(errors.join("\n"));
  console.log(`Issue registry validation passed: ${issues.length} findings.`);
}

if (import.meta.main) {
  run(Bun.argv[2] ?? "").catch((error) => {
    console.error(`issue registry: ${error instanceof Error ? error.message : String(error)}`);
    process.exit(1);
  });
}
