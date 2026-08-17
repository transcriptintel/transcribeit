#!/usr/bin/env bun

import { relative, resolve } from "node:path";

import { loadMatrix, repositoryRoot, resolveFixtures } from "./benchmark_harness/config";
import { commandTemplate, runMatrix } from "./benchmark_harness/runner";
import { compareResults, loadPublishedResult, publishResult, validatePublishedResult } from "./benchmark_harness/result";

type Options = Record<string, string | boolean>;

async function main(): Promise<void> {
  const command = Bun.argv[2];
  const { positional, options } = parseArgs(Bun.argv.slice(3));
  if (command === "validate") {
    const path = positional[0] ?? requiredOption(options, "matrix");
    const { matrix, corpus } = await loadMatrix(path);
    console.log(`Matrix validation passed: ${matrix.matrix_id}, ${matrix.entries.length} entries, ${resolveFixtures(matrix, corpus).size} fixtures.`);
    return;
  }
  if (command === "plan") {
    const path = positional[0] ?? requiredOption(options, "matrix");
    const { matrix, corpus } = await loadMatrix(path);
    const fixtures = resolveFixtures(matrix, corpus);
    const plan = {
      matrix_id: matrix.matrix_id,
      execution_policy: matrix.execution_policy,
      attempt_order: matrix.attempt_order,
      concurrency: matrix.concurrency,
      retries: matrix.retries,
      timeout_seconds: matrix.timeout_seconds,
      measurements: matrix.measurements,
      attempts: matrix.entries.reduce((sum, entry) => sum + (entry.fixture_ids ?? matrix.fixture_ids).length * entry.repetitions, 0),
      fixtures: [...fixtures.values()].map(({ path: _path, ...fixture }) => fixture),
      entries: matrix.entries.map((entry) => ({
        id: entry.id,
        execution: entry.execution,
        cache_state: entry.cache_state,
        repetitions: entry.repetitions,
        required_env: entry.required_env,
        artifact: entry.artifact ?? null,
        command_template: commandTemplate(matrix, entry),
      })),
      hosted_execution_requires: matrix.entries.some((entry) => entry.execution === "hosted") ? "--allow-hosted" : null,
    };
    console.log(JSON.stringify(plan, null, 2));
    return;
  }
  if (command === "run") {
    const matrixPath = requiredOption(options, "matrix");
    const runDirectory = requiredOption(options, "run-dir");
    const { matrix, corpus } = await loadMatrix(matrixPath);
    const state = await runMatrix(matrix, corpus, {
      runDirectory,
      binary: String(options.binary ?? "target/release/transcribeit"),
      allowHosted: options["allow-hosted"] === true,
      retryFailures: options["retry-failures"] === true,
      keepAttemptOutputs: options["keep-attempt-outputs"] === true,
    });
    const completed = state.attempts.filter((attempt) => !['pending', 'running'].includes(attempt.status)).length;
    console.log(`Run state saved: ${completed}/${state.attempts.length} attempts complete.`);
    return;
  }
  if (command === "publish") {
    const statePath = requiredOption(options, "state");
    const outputPath = requiredOption(options, "output");
    const result = await publishResult(
      statePath,
      outputPath,
      typeof options["cleanup-record"] === "string" ? options["cleanup-record"] : undefined,
      typeof options["cleanup-root"] === "string" ? options["cleanup-root"] : undefined,
    );
    console.log(`Published sanitized result: ${relative(repositoryRoot(), resolve(outputPath))} (${result.summary.passed} passed, ${result.summary.failed} failed, ${result.summary.skipped} skipped).`);
    return;
  }
  if (command === "validate-result") {
    const path = positional[0] ?? requiredOption(options, "result");
    const file = Bun.file(path);
    if (!(await file.exists())) throw new Error(`result not found: ${path}`);
    const errors = validatePublishedResult(await file.json());
    if (errors.length) throw new Error(errors.join("\n"));
    console.log(`Benchmark result validation passed: ${path}`);
    return;
  }
  if (command === "compare") {
    const baseline = await loadPublishedResult(requiredOption(options, "baseline"));
    const candidate = await loadPublishedResult(requiredOption(options, "candidate"));
    const comparison = compareResults(baseline, candidate);
    console.log(JSON.stringify(comparison, null, 2));
    if (comparison.exit_failure) process.exitCode = 1;
    return;
  }
  throw new Error("usage: benchmark_harness.ts <validate|plan|run|publish|validate-result|compare> [options]");
}

function parseArgs(args: string[]): { positional: string[]; options: Options } {
  const positional: string[] = [];
  const options: Options = {};
  for (let index = 0; index < args.length; index++) {
    const value = args[index];
    if (!value.startsWith("--")) {
      positional.push(value);
      continue;
    }
    const name = value.slice(2);
    if (["allow-hosted", "retry-failures", "keep-attempt-outputs"].includes(name)) options[name] = true;
    else {
      const next = args[++index];
      if (!next || next.startsWith("--")) throw new Error(`--${name} requires a value`);
      options[name] = next;
    }
  }
  return { positional, options };
}

function requiredOption(options: Options, name: string): string {
  const value = options[name];
  if (typeof value !== "string" || !value) throw new Error(`--${name} is required`);
  return value;
}

if (import.meta.main) {
  main().catch((error) => {
    console.error(`benchmark harness: ${error instanceof Error ? error.message : String(error)}`);
    process.exit(1);
  });
}
