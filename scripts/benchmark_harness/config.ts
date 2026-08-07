import { join, resolve, sep } from "node:path";

import { validateManifest, type CorpusManifest } from "../corpus";
import {
  providers,
  type BenchmarkMatrix,
  type FixtureIdentity,
  type MatrixEntry,
  type TolerancePolicy,
} from "./types";

type MapValue = Record<string, unknown>;

const repository = join(import.meta.dir, "../..");
const corpusPath = join(repository, "benchmarks/corpus/v1/manifest.yaml");
const idPattern = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;
const envPattern = /^[A-Z][A-Z0-9_]*$/;
const matrixFields = new Set(["schema_version", "matrix_id", "description", "execution_policy", "concurrency", "retries", "timeout_seconds", "fixture_ids", "tolerances", "entries"]);
const toleranceFields = new Set(["enforcement", "max_relative_latency_regression_percent", "max_absolute_latency_regression_ms", "min_success_rate"]);
const entryFields = new Set(["id", "provider", "model", "execution", "cache_state", "repetitions", "fixture_ids", "required_env", "args"]);
const forbiddenArgs = new Set([
  "--api-key",
  "--api-key-file",
  "--azure-api-key",
  "--dashscope-api-key",
  "--gemini-api-key",
  "--nvidia-api-key",
  "--deepgram-api-key",
  "--input",
  "-i",
  "--output-dir",
  "-o",
  "--provider",
  "-p",
  "--model",
  "-m",
  "--remote-model",
  "--max-retries",
  "--request-timeout-secs",
  "--base-url",
  "-b",
  "--qwen-api-base-url",
  "--gemini-api-base-url",
  "--deepgram-api-base-url",
  "--nvidia-riva-server",
  "--nvidia-riva-function-id",
]);

function asMap(value: unknown, label: string, errors: string[]): MapValue | undefined {
  if (typeof value === "object" && value !== null && !Array.isArray(value)) return value as MapValue;
  errors.push(`${label} must be a mapping`);
  return undefined;
}

function stringValue(value: unknown, label: string, errors: string[]): string {
  if (typeof value === "string" && value.trim()) return value.trim();
  errors.push(`${label} must be a non-empty string`);
  return "";
}

function stringList(value: unknown, label: string, errors: string[], allowEmpty = false): string[] {
  if (!Array.isArray(value) || (!allowEmpty && value.length === 0) || value.some((item) => typeof item !== "string" || !item.trim())) {
    errors.push(`${label} must be ${allowEmpty ? "a" : "a non-empty"} string list`);
    return [];
  }
  return value.map((item) => String(item).trim());
}

function integer(value: unknown, label: string, errors: string[], minimum: number, maximum: number): number {
  if (Number.isInteger(value) && Number(value) >= minimum && Number(value) <= maximum) return Number(value);
  errors.push(`${label} must be an integer from ${minimum} to ${maximum}`);
  return minimum;
}

function rejectUnexpected(value: MapValue, allowed: Set<string>, label: string, errors: string[]): void {
  const unexpected = Object.keys(value).filter((key) => !allowed.has(key));
  if (unexpected.length) errors.push(`${label} contains unsupported fields: ${unexpected.join(", ")}`);
}

export function validateMatrix(value: unknown, corpus: CorpusManifest): { matrix?: BenchmarkMatrix; errors: string[] } {
  const errors: string[] = [];
  const raw = asMap(value, "matrix", errors);
  if (!raw) return { errors };
  rejectUnexpected(raw, matrixFields, "matrix", errors);
  if (raw.schema_version !== 1) errors.push("schema_version must be 1");
  const matrixId = stringValue(raw.matrix_id, "matrix_id", errors);
  if (matrixId && !idPattern.test(matrixId)) errors.push("matrix_id must be lowercase kebab-case");
  const description = stringValue(raw.description, "description", errors);
  const executionPolicy = stringValue(raw.execution_policy, "execution_policy", errors);
  if (!['manual', 'scheduled', 'local_ci'].includes(executionPolicy)) {
    errors.push("execution_policy must be manual, scheduled, or local_ci");
  }
  const concurrency = integer(raw.concurrency, "concurrency", errors, 1, 8);
  const retries = integer(raw.retries, "retries", errors, 0, 10);
  const timeoutSeconds = integer(raw.timeout_seconds, "timeout_seconds", errors, 1, 86_400);
  const fixtureIds = stringList(raw.fixture_ids, "fixture_ids", errors);
  validateFixtureIds(fixtureIds, corpus, "fixture_ids", errors);

  const toleranceRaw = asMap(raw.tolerances, "tolerances", errors);
  if (toleranceRaw) rejectUnexpected(toleranceRaw, toleranceFields, "tolerances", errors);
  const tolerances: TolerancePolicy = {
    enforcement: stringValue(toleranceRaw?.enforcement, "tolerances.enforcement", errors) as TolerancePolicy["enforcement"],
    max_relative_latency_regression_percent: numberRange(
      toleranceRaw?.max_relative_latency_regression_percent,
      "tolerances.max_relative_latency_regression_percent",
      errors,
      25,
      10_000,
    ),
    max_absolute_latency_regression_ms: numberRange(
      toleranceRaw?.max_absolute_latency_regression_ms,
      "tolerances.max_absolute_latency_regression_ms",
      errors,
      100,
      86_400_000,
    ),
    min_success_rate: numberRange(toleranceRaw?.min_success_rate, "tolerances.min_success_rate", errors, 0, 1),
  };
  if (!['report_only', 'fail'].includes(tolerances.enforcement)) {
    errors.push("tolerances.enforcement must be report_only or fail");
  }

  const entries: MatrixEntry[] = [];
  if (!Array.isArray(raw.entries) || raw.entries.length === 0) errors.push("entries must be a non-empty list");
  else {
    const ids = new Set<string>();
    for (const [index, entryValue] of raw.entries.entries()) {
      const label = `entries[${index}]`;
      const entryRaw = asMap(entryValue, label, errors);
      if (!entryRaw) continue;
      rejectUnexpected(entryRaw, entryFields, label, errors);
      const id = stringValue(entryRaw.id, `${label}.id`, errors);
      if (id && !idPattern.test(id)) errors.push(`${label}.id must be lowercase kebab-case`);
      if (ids.has(id)) errors.push(`${label}.id is duplicated: ${id}`);
      ids.add(id);
      const provider = stringValue(entryRaw.provider, `${label}.provider`, errors);
      if (!(providers as readonly string[]).includes(provider)) errors.push(`${label}.provider is unsupported`);
      const model = stringValue(entryRaw.model, `${label}.model`, errors);
      if (model.startsWith("/") || model.startsWith("~") || model.split("/").includes("..")) {
        errors.push(`${label}.model must be an alias or repository-relative classification, not an absolute/parent path`);
      }
      const execution = stringValue(entryRaw.execution, `${label}.execution`, errors);
      if (!['local', 'hosted'].includes(execution)) errors.push(`${label}.execution must be local or hosted`);
      const cacheState = stringValue(entryRaw.cache_state, `${label}.cache_state`, errors);
      if (!['cold', 'warm'].includes(cacheState)) errors.push(`${label}.cache_state must be cold or warm`);
      const repetitions = integer(entryRaw.repetitions, `${label}.repetitions`, errors, 1, 100);
      const entryFixtures = entryRaw.fixture_ids === undefined
        ? undefined
        : stringList(entryRaw.fixture_ids, `${label}.fixture_ids`, errors);
      if (entryFixtures) validateFixtureIds(entryFixtures, corpus, `${label}.fixture_ids`, errors);
      const requiredEnv = stringList(entryRaw.required_env ?? [], `${label}.required_env`, errors, true);
      for (const name of requiredEnv) if (!envPattern.test(name)) errors.push(`${label}.required_env contains invalid name ${name}`);
      const args = stringList(entryRaw.args ?? [], `${label}.args`, errors, true);
      for (const arg of args) {
        const flag = arg.split("=", 1)[0];
        if (forbiddenArgs.has(flag)) errors.push(`${label}.args cannot override harness-owned or credential option ${flag}`);
        if (/key|token|secret|credential/i.test(flag)) errors.push(`${label}.args cannot contain credential-like option ${flag}`);
        if (arg.startsWith("/") || arg.startsWith("~") || arg.includes("://") || arg.includes("?")) {
          errors.push(`${label}.args cannot contain absolute paths or URLs; use reviewed environment configuration`);
        }
      }
      if (execution === "local" && provider !== "local") errors.push(`${label}: execution local requires provider local`);
      if (execution === "hosted" && provider === "local") errors.push(`${label}: provider local requires execution local`);
      entries.push({
        id,
        provider: provider as MatrixEntry["provider"],
        model,
        execution: execution as MatrixEntry["execution"],
        cache_state: cacheState as MatrixEntry["cache_state"],
        repetitions,
        fixture_ids: entryFixtures,
        required_env: requiredEnv,
        args,
      });
    }
  }
  if (executionPolicy === "local_ci" && entries.some((entry) => entry.execution === "hosted")) {
    errors.push("local_ci matrices cannot contain hosted entries");
  }
  if (tolerances.enforcement === "fail" && executionPolicy !== "local_ci") {
    errors.push("failure-enforced tolerances are allowed only for local_ci matrices");
  }
  if (entries.some((entry) => entry.execution === "hosted") && executionPolicy === "local_ci") {
    errors.push("hosted entries require manual or scheduled execution policy");
  }
  return errors.length
    ? { errors }
    : {
        matrix: {
          schema_version: 1,
          matrix_id: matrixId,
          description,
          execution_policy: executionPolicy as BenchmarkMatrix["execution_policy"],
          concurrency,
          retries,
          timeout_seconds: timeoutSeconds,
          fixture_ids: fixtureIds,
          entries,
          tolerances,
        },
        errors,
      };
}

function numberRange(value: unknown, label: string, errors: string[], minimum: number, maximum: number): number {
  if (typeof value === "number" && Number.isFinite(value) && value >= minimum && value <= maximum) return value;
  errors.push(`${label} must be a finite number from ${minimum} to ${maximum}`);
  return minimum;
}

function validateFixtureIds(ids: string[], corpus: CorpusManifest, label: string, errors: string[]): void {
  const known = new Set(corpus.fixtures.map((fixture) => fixture.id));
  if (new Set(ids).size !== ids.length) errors.push(`${label} contains duplicates`);
  for (const id of ids) if (!known.has(id)) errors.push(`${label} contains unknown fixture ${id}`);
}

export async function loadCorpus(): Promise<CorpusManifest> {
  const raw = Bun.YAML.parse(await Bun.file(corpusPath).text());
  const parsed = validateManifest(raw);
  if (!parsed.manifest || parsed.errors.length) throw new Error(parsed.errors.join("\n"));
  return parsed.manifest;
}

export async function loadMatrix(path: string): Promise<{ matrix: BenchmarkMatrix; corpus: CorpusManifest }> {
  const absolute = resolve(repository, path);
  if (absolute !== repository && !absolute.startsWith(`${repository}${sep}`)) throw new Error("matrix path must stay inside the repository");
  const file = Bun.file(absolute);
  if (!(await file.exists())) throw new Error(`matrix file not found: ${path}`);
  const corpus = await loadCorpus();
  let raw: unknown;
  try {
    raw = Bun.YAML.parse(await file.text());
  } catch (error) {
    throw new Error(`invalid matrix YAML: ${String(error)}`);
  }
  const parsed = validateMatrix(raw, corpus);
  if (!parsed.matrix || parsed.errors.length) throw new Error(parsed.errors.join("\n"));
  return { matrix: parsed.matrix, corpus };
}

export function resolveFixtures(matrix: BenchmarkMatrix, corpus: CorpusManifest): Map<string, FixtureIdentity> {
  const selected = new Set(matrix.fixture_ids);
  for (const entry of matrix.entries) for (const id of entry.fixture_ids ?? []) selected.add(id);
  return new Map(
    corpus.fixtures
      .filter((fixture) => selected.has(fixture.id))
      .map((fixture) => [
        fixture.id,
        {
          id: fixture.id,
          path: fixture.audio.path,
          duration_seconds: Number(fixture.audio.duration_seconds),
          bytes: fixture.audio.bytes,
          sha256: fixture.audio.sha256,
        },
      ]),
  );
}

export function repositoryRoot(): string {
  return repository;
}
