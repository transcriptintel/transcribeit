import { describe, expect, test } from "bun:test";

import {
  containsSampleFilename,
  parseIssuePage,
  renderRegistry,
  type Issue,
} from "../issues_registry";

const openPage = `---
id: "TI-001"
title: "Bound a test contract"
priority: "P2"
status: "open"
area: "Tests"
---
# TI-001 — Bound a test contract

## Required outcome and acceptance

Preserve enough concrete acceptance detail to make this issue independently
actionable, including the affected behavior, the regression boundary, and the
commands that must pass before the status can change.
`;

describe("issue registry", () => {
  test("parses a valid open issue", () => {
    const parsed = parseIssuePage("docs/issues/TI-001.md", openPage);
    expect(parsed.errors).toEqual([]);
    expect(parsed.issue?.status).toBe("open");
  });

  test("requires outcome evidence and a date for resolved issues", () => {
    const resolved = openPage.replace('status: "open"', 'status: "resolved"');
    const parsed = parseIssuePage("docs/issues/TI-001.md", resolved);
    expect(parsed.errors.some((error) => error.includes("ISO resolved date"))).toBeTrue();
    expect(parsed.errors.some((error) => error.includes("Outcome and validation"))).toBeTrue();
  });

  test("rejects media filenames without echoing the protected name", () => {
    const protectedName = "private-client-recording.m4a";
    const parsed = parseIssuePage(
      "docs/issues/TI-001.md",
      openPage.replace("affected behavior", `affected behavior in ${protectedName}`),
    );

    expect(
      parsed.errors.some((error) => error.includes("must not include media filenames")),
    ).toBeTrue();
    expect(parsed.errors.every((error) => !error.includes(protectedName))).toBeTrue();
  });

  test("detects any filename currently protected by the samples directory", () => {
    const protectedName = "private-client-reference.txt";
    expect(
      containsSampleFilename(`Evidence came from ${protectedName}.`, [protectedName]),
    ).toBeTrue();
    expect(
      containsSampleFilename("Evidence uses a sanitized fixture id.", [protectedName]),
    ).toBeFalse();
  });

  test("renders active counts and stable links", () => {
    const issues: Issue[] = [
      {
        id: "TI-001",
        title: "Bound a test contract",
        priority: "P2",
        status: "open",
        area: "Tests",
        path: "docs/issues/TI-001.md",
        body: "body",
      },
      {
        id: "TI-002",
        title: "Close a test contract",
        priority: "P1",
        status: "resolved",
        area: "Tests",
        resolved: "2026-08-06",
        path: "docs/issues/TI-002.md",
        body: "body",
      },
    ];
    const rendered = renderRegistry(issues);
    expect(rendered).toContain("Active findings: 1");
    expect(rendered).toContain("[TI-002](./TI-002.md)");
    expect(rendered).toContain("## Working agreement");
  });
});
