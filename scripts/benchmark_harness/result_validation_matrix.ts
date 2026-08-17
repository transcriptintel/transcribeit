import { providers, type BenchmarkMatrix } from "./types";
import { isMap } from "./state";
import {
  gitHashPattern,
  hasOnlyStrings,
  isFiniteRange,
  isNullableString,
  isSafeIntegerRange,
  safeEnvironmentNamePattern,
  safeIdPattern,
  safeRevisionPattern,
  sha256Pattern,
} from "./result_validation_primitives";

const executionPolicies = ["manual", "scheduled", "local_ci"];
const attemptOrders = ["entry-fixture-repetition", "fixture-repetition-entry"];
const executions = ["local", "hosted"];
const cacheStates = ["cold", "warm"];
const artifactLifecycles = ["evaluation_download", "system_managed", "provider_managed"];
const modelPattern = /^[A-Za-z0-9][A-Za-z0-9._/-]{0,255}$/;
const classificationPattern = /^[A-Za-z0-9._-]+(?:\/[A-Za-z0-9._-]+)*$/;
const reviewedLocalePattern = /^(?:auto|system|[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})*)$/;

export function validateIdentityAndRuntime(value: Record<string, unknown>, errors: string[]): void {
  if (!isMap(value.producing_commit)) {
    errors.push("invalid producing_commit");
  } else {
    const commit = value.producing_commit;
    if (typeof commit.hash !== "string" || !gitHashPattern.test(commit.hash)) errors.push("producing_commit.hash is invalid");
    if (typeof commit.worktree_dirty !== "boolean") errors.push("producing_commit.worktree_dirty must be a boolean");
    if (typeof commit.worktree_fingerprint_sha256 !== "string" || !sha256Pattern.test(commit.worktree_fingerprint_sha256)) {
      errors.push("producing_commit.worktree_fingerprint_sha256 must be lowercase SHA-256");
    }
    const expected = commit.worktree_dirty === true ? "dirty_worktree_reference" : "clean_commit_benchmark";
    if (value.classification !== expected) errors.push("result classification does not match producing_commit.worktree_dirty");
  }
  if (!isMap(value.runtime)) {
    errors.push("invalid runtime");
  } else {
    if (!safeRelativeClassification(value.runtime.binary_classification)) {
      errors.push("runtime.binary_classification must be a safe relative classification");
    }
    if (typeof value.runtime.binary_sha256 !== "string" || !sha256Pattern.test(value.runtime.binary_sha256)) {
      errors.push("runtime.binary_sha256 must be lowercase SHA-256");
    }
  }
}

export function validateEnvironmentValues(value: unknown, errors: string[]): void {
  if (!isMap(value) || !isMap(value.machine) || !isMap(value.tools)) {
    errors.push("invalid environment metadata");
    return;
  }
  const machine = value.machine;
  for (const name of ["cpu", "os", "kernel", "architecture"] as const) {
    if (isMap(machine[name]) || Array.isArray(machine[name])) errors.push(`environment.machine.${name} must not contain nested data`);
    if (!boundedString(machine[name], 256)) errors.push(`environment.machine.${name} must be a non-empty bounded string`);
  }
  if (!isSafeIntegerRange(machine.logical_cores, 1, 1_000_000)) {
    errors.push("environment.machine.logical_cores must be a positive safe integer");
  }
  if (machine.memory_bytes !== null && !isSafeIntegerRange(machine.memory_bytes, 1)) {
    errors.push("environment.machine.memory_bytes must be null or a positive safe integer");
  }
  for (const name of ["os_version", "os_build"] as const) {
    if (!isNullableString(machine[name]) || (typeof machine[name] === "string" && !boundedString(machine[name], 256))) {
      errors.push(`environment.machine.${name} must be null or a non-empty bounded string`);
    }
  }
  for (const name of ["rustc", "ffmpeg", "swift", "bun"] as const) {
    if (!boundedString(value.tools[name], 512)) errors.push(`environment.tools.${name} must be a non-empty bounded string`);
  }
}

export function validateMatrixValues(value: unknown, errors: string[]): BenchmarkMatrix | undefined {
  if (!isMap(value)) {
    errors.push("result matrix is invalid");
    return undefined;
  }
  if (value.schema_version !== 1) errors.push("matrix.schema_version must be 1");
  if (typeof value.matrix_id !== "string" || !safeIdPattern.test(value.matrix_id)) {
    errors.push("matrix.matrix_id must be lowercase kebab-case");
  }
  if (!boundedString(value.description, 1_024)) errors.push("matrix.description must be a non-empty bounded string");
  enumValue(value.execution_policy, executionPolicies, "matrix.execution_policy", errors);
  enumValue(value.attempt_order, attemptOrders, "matrix.attempt_order", errors);
  integerValue(value.concurrency, 1, 8, "matrix.concurrency", errors);
  integerValue(value.retries, 0, 10, "matrix.retries", errors);
  integerValue(value.timeout_seconds, 1, 86_400, "matrix.timeout_seconds", errors);
  validateIds(value.fixture_ids, "matrix.fixture_ids", errors, false);
  validateMeasurements(value.measurements, errors);
  validateTolerances(value.tolerances, errors);
  validateEntries(value.entries, value, errors);
  if (
    value.execution_policy === "local_ci" &&
    Array.isArray(value.entries) &&
    value.entries.some((entry) => isMap(entry) && entry.execution === "hosted")
  ) errors.push("local_ci result cannot contain hosted entries");
  if (isMap(value.tolerances) && value.tolerances.enforcement === "fail" && value.execution_policy !== "local_ci") {
    errors.push("failure-enforced result tolerance requires local_ci policy");
  }
  return value as unknown as BenchmarkMatrix;
}

export function normalizedPublishedMatrix(value: Record<string, unknown>): BenchmarkMatrix {
  return {
    ...value,
    entries: Array.isArray(value.entries)
      ? value.entries.map((entry) => {
          if (!isMap(entry)) return entry;
          const { command_template: _commandTemplate, ...definition } = entry;
          return definition;
        })
      : value.entries,
  } as unknown as BenchmarkMatrix;
}

function validateMeasurements(value: unknown, errors: string[]): void {
  if (!isMap(value)) {
    errors.push("matrix.measurements must be a mapping");
    return;
  }
  for (const name of ["reference_scoring", "peak_rss"] as const) {
    if (typeof value[name] !== "boolean") errors.push(`matrix.measurements.${name} must be a boolean`);
  }
}

function validateTolerances(value: unknown, errors: string[]): void {
  if (!isMap(value)) {
    errors.push("matrix.tolerances must be a mapping");
    return;
  }
  enumValue(value.enforcement, ["report_only", "fail"], "matrix.tolerances.enforcement", errors);
  finiteValue(value.max_relative_latency_regression_percent, 25, 10_000, "matrix.tolerances.max_relative_latency_regression_percent", errors);
  finiteValue(value.max_absolute_latency_regression_ms, 100, 86_400_000, "matrix.tolerances.max_absolute_latency_regression_ms", errors);
  finiteValue(value.min_success_rate, 0, 1, "matrix.tolerances.min_success_rate", errors);
}

function validateEntries(value: unknown, matrix: Record<string, unknown>, errors: string[]): void {
  if (!Array.isArray(value) || !value.length) {
    errors.push("matrix.entries must be a non-empty list");
    return;
  }
  const ids = new Set<string>();
  for (const [index, entry] of value.entries()) {
    const label = `matrix.entries[${index}]`;
    if (!isMap(entry)) {
      errors.push(`${label} must be a mapping`);
      continue;
    }
    if (typeof entry.id !== "string" || !safeIdPattern.test(entry.id)) errors.push(`${label}.id must be lowercase kebab-case`);
    else if (ids.has(entry.id)) errors.push(`${label}.id is duplicated`);
    else ids.add(entry.id);
    enumValue(entry.provider, providers as readonly string[], `${label}.provider`, errors);
    if (!safeModel(entry.model)) errors.push(`${label}.model must be a safe non-empty classification`);
    enumValue(entry.execution, executions, `${label}.execution`, errors);
    enumValue(entry.cache_state, cacheStates, `${label}.cache_state`, errors);
    integerValue(entry.repetitions, 1, 100, `${label}.repetitions`, errors);
    if (entry.fixture_ids !== undefined) validateIds(entry.fixture_ids, `${label}.fixture_ids`, errors, false);
    validateEnvironmentNames(entry.required_env, `${label}.required_env`, errors);
    const language = validateArgs(entry.args, entry.provider, `${label}.args`, errors);
    validateArtifact(entry.artifact, label, errors);
    if (!boundedString(entry.command_template, 4_096)) errors.push(`${label}.command_template must be a non-empty bounded string`);
    else if (entry.command_template !== expectedCommandTemplate(matrix, entry)) {
      errors.push(`${label}.command_template does not match the matrix entry`);
    }
    const local = entry.provider === "apple-speech" || entry.provider === "local";
    if ((local && entry.execution !== "local") || (!local && entry.execution !== "hosted")) {
      errors.push(`${label}.execution does not match its provider`);
    }
    if (entry.provider === "apple-speech" && (!language || language.toLowerCase() === "auto")) {
      errors.push(`${label} Apple Speech requires an explicit non-auto language`);
    }
    if (
      isMap(entry.artifact) && entry.artifact.lifecycle === "evaluation_download" &&
      (!Array.isArray(entry.required_env) || !entry.required_env.includes("MODEL_CACHE_DIR"))
    ) errors.push(`${label} evaluation_download requires MODEL_CACHE_DIR`);
  }
}

function validateArtifact(value: unknown, entryLabel: string, errors: string[]): void {
  if (value === undefined) return;
  const label = `${entryLabel}.artifact`;
  if (!isMap(value)) {
    errors.push(`${label} must be a mapping`);
    return;
  }
  enumValue(value.lifecycle, artifactLifecycles, `${label}.lifecycle`, errors);
  if (value.revision !== undefined && (typeof value.revision !== "string" || !safeRevisionPattern.test(value.revision))) {
    errors.push(`${label}.revision must be a safe pinned identifier`);
  }
  if (value.sha256 !== undefined && (typeof value.sha256 !== "string" || !sha256Pattern.test(value.sha256))) {
    errors.push(`${label}.sha256 must be lowercase SHA-256`);
  }
  if (value.bytes !== undefined && !isSafeIntegerRange(value.bytes, 1)) errors.push(`${label}.bytes must be a positive safe integer`);
  if (value.lifecycle === "evaluation_download" &&
    (typeof value.revision !== "string" || typeof value.sha256 !== "string" || !isSafeIntegerRange(value.bytes, 1))) {
    errors.push(`${label} evaluation_download requires revision, sha256, and bytes`);
  }
}

function validateIds(value: unknown, label: string, errors: string[], allowEmpty: boolean): void {
  if (!hasOnlyStrings(value, allowEmpty) || value.some((id) => !safeIdPattern.test(id))) {
    errors.push(`${label} must contain only lowercase kebab-case IDs`);
    return;
  }
  if (new Set(value).size !== value.length) errors.push(`${label} contains duplicates`);
}

function validateEnvironmentNames(value: unknown, label: string, errors: string[]): void {
  if (!hasOnlyStrings(value) || value.some((name) => !safeEnvironmentNamePattern.test(name))) {
    errors.push(`${label} must contain only environment variable names`);
    return;
  }
  if (new Set(value).size !== value.length) errors.push(`${label} contains duplicates`);
}

function validateArgs(value: unknown, provider: unknown, label: string, errors: string[]): string | undefined {
  if (!Array.isArray(value) || value.some((argument) => typeof argument !== "string" || /[\u0000-\u001f\u007f]/.test(argument))) {
    errors.push(`${label} must be a scalar string list without control characters`);
    return undefined;
  }
  const seen = new Set<string>();
  let language: string | undefined;
  for (let index = 0; index < value.length; index++) {
    const argument = value[index];
    if (argument === "--language" || argument.startsWith("--language=")) {
      if (seen.has("--language")) errors.push(`${label} cannot repeat --language`);
      seen.add("--language");
      const candidate = argument === "--language" ? value[++index] : argument.slice("--language=".length);
      if (typeof candidate !== "string" || !reviewedLocalePattern.test(candidate)) {
        errors.push(`${label} --language must use a safe locale, system, or auto value`);
      } else language = candidate;
      continue;
    }
    if (argument === "--gemini-autoclean") {
      if (seen.has(argument)) errors.push(`${label} cannot repeat ${argument}`);
      if (provider !== "gemini") errors.push(`${label} ${argument} is valid only for Gemini`);
      seen.add(argument);
      continue;
    }
    errors.push(`${label} contains unsupported reviewed option ${argument}`);
  }
  return language;
}

function expectedCommandTemplate(matrix: Record<string, unknown>, entry: Record<string, unknown>): string {
  const provider = String(entry.provider);
  const model = provider === "local"
    ? ` --model ${String(entry.model)}`
    : ["apple-speech", "azure", "nvidia-riva"].includes(provider)
      ? ""
      : ` --remote-model ${String(entry.model)}`;
  const args = Array.isArray(entry.args) && entry.args.length ? ` ${entry.args.join(" ")}` : "";
  return `transcribeit run --provider ${provider}${model} --max-retries ${String(matrix.retries)} --request-timeout-secs ${String(matrix.timeout_seconds)} --input $FIXTURE --output-format text --output-dir $ATTEMPT_DIR${args}`;
}

function safeModel(value: unknown): boolean {
  return boundedString(value, 256) && modelPattern.test(value) && !value.includes("//") &&
    !value.split("/").includes("..");
}

function safeRelativeClassification(value: unknown): boolean {
  return boundedString(value, 256) && classificationPattern.test(value) &&
    !value.split("/").includes("..") && !value.startsWith(".");
}

function boundedString(value: unknown, maximum: number): value is string {
  return typeof value === "string" && value.trim().length > 0 && value.length <= maximum && !/[\u0000-\u001f\u007f]/.test(value);
}

function enumValue(value: unknown, allowed: readonly string[], label: string, errors: string[]): void {
  if (typeof value !== "string" || !allowed.includes(value)) errors.push(`${label} has an invalid value`);
}

function integerValue(value: unknown, minimum: number, maximum: number, label: string, errors: string[]): void {
  if (!isSafeIntegerRange(value, minimum, maximum)) errors.push(`${label} must be an integer from ${minimum} to ${maximum}`);
}

function finiteValue(value: unknown, minimum: number, maximum: number, label: string, errors: string[]): void {
  if (!isFiniteRange(value, minimum, maximum)) errors.push(`${label} must be a finite number from ${minimum} to ${maximum}`);
}
