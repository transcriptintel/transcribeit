#!/usr/bin/env bun

import { mkdir, rename, rm, rmdir } from "node:fs/promises";
import { dirname, join, relative, resolve, sep } from "node:path";

type MapValue = Record<string, unknown>;

export type CorpusManifest = {
  schema_version: number;
  corpus_id: string;
  materialization_root: string;
  sources: Record<string, Source>;
  fixtures: Fixture[];
};

type Source = {
  url: string;
  sha256: string;
  bytes: number;
};

type Fixture = {
  id: string;
  audio: {
    path: string;
    sha256: string;
    bytes: number;
    materialize: MapValue;
  };
  reference: MapValue;
  [key: string]: unknown;
};

const repository = join(import.meta.dir, "..");
const manifestPath = join(repository, "benchmarks/corpus/v1/manifest.yaml");
const scoringPath = join(repository, "benchmarks/corpus/v1/scoring.yaml");
const downloadsRoot = join(repository, "samples/corpus/.downloads");
const sha256Pattern = /^[a-f0-9]{64}$/;
const fixtureIdPattern = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;
const requiredMetricGroups = [
  "word_accuracy",
  "domain_terms",
  "timing",
  "speakers",
  "metadata_completeness",
  "latency",
  "provider_failures",
];

function asMap(value: unknown, label: string, errors: string[]): MapValue | undefined {
  if (typeof value === "object" && value !== null && !Array.isArray(value)) return value as MapValue;
  errors.push(`${label} must be a mapping`);
  return undefined;
}

function nonEmptyString(value: unknown): value is string {
  return typeof value === "string" && value.trim().length > 0;
}

function normalizedWords(value: string): string[] {
  return value
    .normalize("NFKC")
    .toLowerCase()
    .replaceAll("’", "'")
    .match(/[\p{L}\p{N}]+(?:'[\p{L}\p{N}]+)*/gu) ?? [];
}

function containsTokens(text: string, term: string): boolean {
  const haystack = normalizedWords(text);
  const needle = normalizedWords(term);
  return needle.length > 0 && haystack.some((_, index) => needle.every((token, offset) => haystack[index + offset] === token));
}

function isSafeRepositoryPath(value: unknown, root: string): value is string {
  if (!nonEmptyString(value) || value.startsWith("/")) return false;
  const target = resolve(repository, value);
  const allowed = resolve(repository, root);
  return target === allowed || target.startsWith(`${allowed}${sep}`);
}

function isSafeArchiveMember(value: unknown): value is string {
  return (
    nonEmptyString(value) &&
    !value.startsWith("-") &&
    !value.startsWith("/") &&
    !/[?*[\]\\]/.test(value) &&
    !value.split("/").some((component) => component === "..")
  );
}

export function validateManifest(value: unknown): { manifest?: CorpusManifest; errors: string[] } {
  const errors: string[] = [];
  const manifest = asMap(value, "manifest", errors);
  if (!manifest) return { errors };
  if (manifest.schema_version !== 1) errors.push("schema_version must be 1");
  if (!nonEmptyString(manifest.corpus_id)) errors.push("corpus_id must be a non-empty string");
  if (manifest.materialization_root !== "samples/corpus/v1") {
    errors.push("materialization_root must be samples/corpus/v1");
  }

  const sources = asMap(manifest.sources, "sources", errors);
  if (sources) {
    for (const [id, raw] of Object.entries(sources)) {
      const source = asMap(raw, `sources.${id}`, errors);
      if (!source) continue;
      if (!nonEmptyString(source.url) || !source.url.startsWith("https://")) {
        errors.push(`sources.${id}.url must use HTTPS`);
      }
      if (!nonEmptyString(source.sha256) || !sha256Pattern.test(source.sha256)) {
        errors.push(`sources.${id}.sha256 must be lowercase SHA-256`);
      }
      if (!Number.isInteger(source.bytes) || (source.bytes as number) <= 0) {
        errors.push(`sources.${id}.bytes must be a positive integer`);
      }
    }
  }

  if (!Array.isArray(manifest.fixtures) || manifest.fixtures.length < 4) {
    errors.push("fixtures must contain at least four entries");
  } else {
    const ids = new Set<string>();
    const conditions = new Set<string>();
    const accents = new Set<string>();
    let hasLongRecording = false;
    let hasMultipleSpeakers = false;
    let hasTiming = false;
    let hasDiarization = false;
    let hasDomainTerms = false;
    for (const [index, raw] of manifest.fixtures.entries()) {
      const label = `fixtures[${index}]`;
      const fixture = asMap(raw, label, errors);
      if (!fixture) continue;
      if (!nonEmptyString(fixture.id) || !fixtureIdPattern.test(fixture.id)) {
        errors.push(`${label}.id must be a lowercase kebab-case stable ID`);
      } else if (ids.has(fixture.id)) {
        errors.push(`${label}.id is duplicated: ${fixture.id}`);
      } else ids.add(fixture.id);

      const audio = asMap(fixture.audio, `${label}.audio`, errors);
      if (audio) {
        if (!isSafeRepositoryPath(audio.path, "samples/corpus/v1")) {
          errors.push(`${label}.audio.path must stay under samples/corpus/v1`);
        }
        if (!nonEmptyString(audio.sha256) || !sha256Pattern.test(audio.sha256)) {
          errors.push(`${label}.audio.sha256 must be lowercase SHA-256`);
        }
        if (!Number.isInteger(audio.bytes) || (audio.bytes as number) <= 0) {
          errors.push(`${label}.audio.bytes must be a positive integer`);
        }
        if (typeof audio.duration_seconds !== "number" || audio.duration_seconds <= 0) {
          errors.push(`${label}.audio.duration_seconds must be positive`);
        } else if (audio.duration_seconds >= 1800) hasLongRecording = true;
        for (const field of ["container", "codec"] as const) {
          if (!nonEmptyString(audio[field])) errors.push(`${label}.audio.${field} must be non-empty`);
        }
        if (!Number.isInteger(audio.sample_rate_hz) || (audio.sample_rate_hz as number) <= 0) {
          errors.push(`${label}.audio.sample_rate_hz must be a positive integer`);
        }
        if (!Number.isInteger(audio.channels) || (audio.channels as number) <= 0) {
          errors.push(`${label}.audio.channels must be a positive integer`);
        }
        const materialize = asMap(audio.materialize, `${label}.audio.materialize`, errors);
        if (materialize && !["remote_file", "tar_member", "white_noise"].includes(String(materialize.kind))) {
          errors.push(`${label}.audio.materialize.kind is unsupported`);
        }
        if (materialize && ["remote_file", "tar_member"].includes(String(materialize.kind))) {
          if (!nonEmptyString(materialize.source) || !sources?.[materialize.source]) {
            errors.push(`${label}.audio.materialize.source must name a declared source`);
          }
        }
        if (materialize?.kind === "tar_member" && !isSafeArchiveMember(materialize.member)) {
          errors.push(`${label}.audio.materialize.member must be a safe relative archive member`);
        }
        if (
          materialize?.kind === "white_noise" &&
          (!nonEmptyString(materialize.source_fixture) || !Number.isInteger(materialize.seed) || typeof materialize.snr_db !== "number")
        ) {
          errors.push(`${label}.audio.materialize white_noise requires source_fixture, integer seed, and snr_db`);
        }
      }

      const rights = asMap(fixture.rights, `${label}.rights`, errors);
      if (rights && (!nonEmptyString(rights.license) || !nonEmptyString(rights.attribution))) {
        errors.push(`${label}.rights requires license and attribution`);
      }
      const reference = asMap(fixture.reference, `${label}.reference`, errors);
      if (reference && !["inline_reviewed", "archive_members", "none"].includes(String(reference.kind))) {
        errors.push(`${label}.reference.kind is unsupported`);
      }
      if (reference?.kind === "inline_reviewed") {
        if (!nonEmptyString(reference.text) || !isSafeRepositoryPath(reference.path, "samples/corpus/v1")) {
          errors.push(`${label}.reference inline_reviewed requires text and a safe path`);
        }
        if (!nonEmptyString(reference.sha256) || !sha256Pattern.test(reference.sha256)) {
          errors.push(`${label}.reference.sha256 must be lowercase SHA-256`);
        }
        if (!Number.isInteger(reference.bytes) || (reference.bytes as number) <= 0) {
          errors.push(`${label}.reference.bytes must be a positive integer`);
        }
        if (!nonEmptyString(reference.source) || !sources?.[reference.source]) {
          errors.push(`${label}.reference.source must name a declared provenance source`);
        }
        if (
          !nonEmptyString(reference.source_key) ||
          !Number.isInteger(reference.source_key_column) ||
          !Number.isInteger(reference.source_text_column) ||
          reference.reviewed !== true
        ) {
          errors.push(`${label}.reference inline_reviewed requires source key columns and reviewed: true`);
        }
      }
      if (reference?.kind === "archive_members") {
        if (!nonEmptyString(reference.source) || !sources?.[reference.source]) {
          errors.push(`${label}.reference.source must name a declared archive source`);
        }
        if (!Array.isArray(reference.files) || reference.files.length === 0) {
          errors.push(`${label}.reference.files must not be empty`);
        } else {
          for (const [fileIndex, rawFile] of reference.files.entries()) {
            const file = asMap(rawFile, `${label}.reference.files[${fileIndex}]`, errors);
            if (!file) continue;
            if (!isSafeArchiveMember(file.member) || !isSafeRepositoryPath(file.path, "samples/corpus/v1")) {
              errors.push(`${label}.reference.files[${fileIndex}] requires an archive member and safe path`);
            }
            if (!nonEmptyString(file.sha256) || !sha256Pattern.test(file.sha256)) {
              errors.push(`${label}.reference.files[${fileIndex}].sha256 must be lowercase SHA-256`);
            }
            if (!Number.isInteger(file.bytes) || (file.bytes as number) <= 0) {
              errors.push(`${label}.reference.files[${fileIndex}].bytes must be a positive integer`);
            }
          }
        }
        if (reference.reviewed !== true) errors.push(`${label}.reference archive_members requires reviewed: true`);
      }
      if (reference?.kind === "none" && !nonEmptyString(reference.classification)) {
        errors.push(`${label}.reference none requires an explicit classification`);
      }
      const coverage = asMap(fixture.coverage, `${label}.coverage`, errors);
      if (coverage && (!Array.isArray(coverage.conditions) || !Array.isArray(coverage.accents))) {
        errors.push(`${label}.coverage requires conditions and accents lists`);
      } else if (coverage) {
        for (const value of coverage.conditions as unknown[]) if (nonEmptyString(value)) conditions.add(value);
        for (const value of coverage.accents as unknown[]) if (nonEmptyString(value)) accents.add(value);
        if (typeof coverage.speaker_count === "number" && coverage.speaker_count > 1) hasMultipleSpeakers = true;
        if (coverage.timestamp_reference === true) hasTiming = true;
        if (coverage.diarization_reference === true) hasDiarization = true;
        if (Array.isArray(coverage.domain_terms) && coverage.domain_terms.length > 0) hasDomainTerms = true;
        if (reference?.kind === "inline_reviewed" && Array.isArray(coverage.domain_terms)) {
          for (const term of coverage.domain_terms) {
            if (!nonEmptyString(term) || !containsTokens(String(reference.text), term)) {
              errors.push(`${label}.coverage domain term is absent from the reviewed reference: ${String(term)}`);
            }
          }
        }
      }
    }
    if (!conditions.has("clean") || !conditions.has("noisy")) errors.push("corpus must cover clean and noisy audio");
    if (accents.size < 2) errors.push("corpus must document at least two accent groups");
    if (!hasLongRecording) errors.push("corpus must include a recording of at least 30 minutes");
    if (!hasMultipleSpeakers) errors.push("corpus must include a multiple-speaker fixture");
    if (!hasTiming) errors.push("corpus must include timestamp reference data");
    if (!hasDiarization) errors.push("corpus must include diarization reference data");
    if (!hasDomainTerms) errors.push("corpus must include non-medical domain terms");
  }

  if (errors.length) return { errors };
  return { manifest: manifest as CorpusManifest, errors };
}

export function validateScoring(value: unknown): string[] {
  const errors: string[] = [];
  const scoring = asMap(value, "scoring", errors);
  if (!scoring) return errors;
  if (scoring.schema_version !== 1) errors.push("scoring.schema_version must be 1");
  const metrics = asMap(scoring.metrics, "scoring.metrics", errors);
  if (metrics) {
    for (const group of requiredMetricGroups) {
      if (!asMap(metrics[group], `scoring.metrics.${group}`, errors)) continue;
    }
  }
  return errors;
}

async function sha256(path: string): Promise<string> {
  const reader = Bun.file(path).stream().getReader();
  const hasher = new Bun.CryptoHasher("sha256");
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    hasher.update(value);
  }
  return hasher.digest("hex");
}

async function assertFile(path: string, expectedHash: string, expectedBytes: number): Promise<void> {
  const file = Bun.file(path);
  if (!(await file.exists())) throw new Error(`missing ${relative(repository, path)}`);
  if (file.size !== expectedBytes) {
    throw new Error(`${relative(repository, path)} has ${file.size} bytes; expected ${expectedBytes}`);
  }
  const actualHash = await sha256(path);
  if (actualHash !== expectedHash) {
    throw new Error(`${relative(repository, path)} SHA-256 ${actualHash}; expected ${expectedHash}`);
  }
}

function corpusPath(path: unknown): string {
  if (!isSafeRepositoryPath(path, "samples/corpus/v1")) throw new Error(`unsafe corpus path: ${String(path)}`);
  return resolve(repository, path);
}

async function runBytes(command: string[]): Promise<Uint8Array> {
  const process = Bun.spawn(command, { stdout: "pipe", stderr: "pipe" });
  const [exitCode, stdout, stderr] = await Promise.all([
    process.exited,
    new Response(process.stdout).arrayBuffer(),
    new Response(process.stderr).text(),
  ]);
  if (exitCode !== 0) throw new Error(`${command[0]} failed (${exitCode}): ${stderr.trim()}`);
  return new Uint8Array(stdout);
}

async function writeAtomic(path: string, bytes: Uint8Array | string): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  const temporary = `${path}.tmp-${process.pid}`;
  try {
    await Bun.write(temporary, bytes);
    await rename(temporary, path);
  } catch (error) {
    await rm(temporary, { force: true });
    throw error;
  }
}

function wavDataChunk(bytes: Uint8Array): { offset: number; length: number } {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (new TextDecoder().decode(bytes.subarray(0, 4)) !== "RIFF" || new TextDecoder().decode(bytes.subarray(8, 12)) !== "WAVE") {
    throw new Error("white_noise source must be a RIFF/WAVE file");
  }
  let offset = 12;
  let isFloatMono = false;
  while (offset + 8 <= bytes.length) {
    const id = new TextDecoder().decode(bytes.subarray(offset, offset + 4));
    const length = view.getUint32(offset + 4, true);
    const payload = offset + 8;
    if (id === "fmt ") {
      isFloatMono =
        view.getUint16(payload, true) === 3 &&
        view.getUint16(payload + 2, true) === 1 &&
        view.getUint16(payload + 14, true) === 32;
    }
    if (id === "data") {
      if (!isFloatMono || length % 4 !== 0) throw new Error("white_noise source must be mono PCM f32le");
      return { offset: payload, length };
    }
    offset = payload + length + (length % 2);
  }
  throw new Error("WAV data chunk is missing");
}

export function createNoisyWav(source: Uint8Array, seed: number, snrDb: number): Uint8Array {
  const output = source.slice();
  const chunk = wavDataChunk(output);
  const view = new DataView(output.buffer, output.byteOffset, output.byteLength);
  const samples = chunk.length / 4;
  let signalPower = 0;
  const noise = new Float64Array(samples);
  let state = seed >>> 0;
  let noisePower = 0;
  for (let index = 0; index < samples; index += 1) {
    const sample = view.getFloat32(chunk.offset + index * 4, true);
    signalPower += sample * sample;
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    const value = ((state >>> 0) / 0xffffffff) * 2 - 1;
    noise[index] = value;
    noisePower += value * value;
  }
  const scale = Math.sqrt(signalPower / samples) / (10 ** (snrDb / 20) * Math.sqrt(noisePower / samples));
  for (let index = 0; index < samples; index += 1) {
    const sample = view.getFloat32(chunk.offset + index * 4, true);
    view.setFloat32(chunk.offset + index * 4, Math.max(-1, Math.min(1, sample + noise[index] * scale)), true);
  }
  return output;
}

async function loadDefinition(): Promise<CorpusManifest> {
  const parsed = validateManifest(Bun.YAML.parse(await Bun.file(manifestPath).text()));
  const scoringErrors = validateScoring(Bun.YAML.parse(await Bun.file(scoringPath).text()));
  const errors = [...parsed.errors, ...scoringErrors];
  if (errors.length || !parsed.manifest) throw new Error(errors.join("\n"));
  return parsed.manifest;
}

async function fetchSource(id: string, source: Source, touched: Set<string>): Promise<string> {
  await mkdir(downloadsRoot, { recursive: true });
  const path = join(downloadsRoot, source.sha256);
  const existing = Bun.file(path);
  if (await existing.exists()) {
    touched.add(path);
    await assertFile(path, source.sha256, source.bytes);
    return path;
  }
  const temporary = `${path}.partial-${process.pid}`;
  touched.add(temporary);
  const child = Bun.spawn(
    ["curl", "-fL", "--proto", "=https", "--proto-redir", "=https", "--retry", "3", source.url, "-o", temporary],
    {
      stdout: "inherit",
      stderr: "inherit",
    },
  );
  if ((await child.exited) !== 0) throw new Error(`download failed: ${id}`);
  await assertFile(temporary, source.sha256, source.bytes);
  await rename(temporary, path);
  touched.delete(temporary);
  touched.add(path);
  return path;
}

async function materialize(manifest: CorpusManifest, keepDownloads: boolean): Promise<void> {
  const touched = new Set<string>();
  const sourcePaths = new Map<string, string>();
  const source = async (id: string): Promise<string> => {
    const definition = manifest.sources[id];
    if (!definition) throw new Error(`unknown source ${id}`);
    const cached = sourcePaths.get(id);
    if (cached) return cached;
    const path = await fetchSource(id, definition, touched);
    sourcePaths.set(id, path);
    return path;
  };

  try {
    for (const fixture of manifest.fixtures) {
      const materialize = fixture.audio.materialize;
      const output = corpusPath(fixture.audio.path);
      if (materialize.kind === "remote_file") {
        await writeAtomic(output, await Bun.file(await source(String(materialize.source))).bytes());
      } else if (materialize.kind === "tar_member") {
        const archive = await source(String(materialize.source));
        await writeAtomic(output, await runBytes(["tar", "-xOzf", archive, "--", String(materialize.member)]));
      } else if (materialize.kind === "white_noise") {
        const parent = manifest.fixtures.find((candidate) => candidate.id === materialize.source_fixture);
        if (!parent) throw new Error(`unknown source fixture ${String(materialize.source_fixture)}`);
        const input = await Bun.file(corpusPath(parent.audio.path)).bytes();
        await writeAtomic(output, createNoisyWav(input, Number(materialize.seed), Number(materialize.snr_db)));
      }

      if (fixture.reference.kind === "inline_reviewed") {
        const provenance = await Bun.file(await source(String(fixture.reference.source))).text();
        const keyColumn = Number(fixture.reference.source_key_column);
        const textColumn = Number(fixture.reference.source_text_column);
        const row = provenance
          .split(/\r?\n/)
          .map((line) => line.split("\t"))
          .find((columns) => columns[keyColumn] === fixture.reference.source_key);
        if (!row || row[textColumn] !== fixture.reference.text) {
          throw new Error(`reference provenance mismatch for ${fixture.id}`);
        }
        await writeAtomic(corpusPath(fixture.reference.path), `${String(fixture.reference.text)}\n`);
      } else if (fixture.reference.kind === "archive_members") {
        const archive = await source(String(fixture.reference.source));
        for (const raw of fixture.reference.files as MapValue[]) {
          await writeAtomic(corpusPath(raw.path), await runBytes(["unzip", "-p", archive, String(raw.member)]));
        }
      }
    }
    await verify(manifest);
  } finally {
    if (!keepDownloads) {
      for (const path of touched) await rm(path, { force: true });
      await rmdir(downloadsRoot).catch(() => undefined);
    }
  }
}

async function verify(manifest: CorpusManifest): Promise<void> {
  for (const fixture of manifest.fixtures) {
    await assertFile(corpusPath(fixture.audio.path), fixture.audio.sha256, fixture.audio.bytes);
    if (fixture.reference.kind === "inline_reviewed") {
      await assertFile(corpusPath(fixture.reference.path), String(fixture.reference.sha256), Number(fixture.reference.bytes));
    } else if (fixture.reference.kind === "archive_members") {
      for (const raw of fixture.reference.files as MapValue[]) {
        await assertFile(corpusPath(raw.path), String(raw.sha256), Number(raw.bytes));
      }
    }
  }
  console.log(`Verified ${manifest.fixtures.length} corpus fixtures.`);
}

async function run(command: string, options: string[]): Promise<void> {
  const manifest = await loadDefinition();
  if (command === "check") {
    console.log(`Corpus definition validation passed: ${manifest.fixtures.length} fixtures.`);
    return;
  }
  if (command === "fetch") {
    await materialize(manifest, options.includes("--keep-downloads"));
    return;
  }
  if (command === "verify") {
    await verify(manifest);
    return;
  }
  throw new Error("usage: bun run scripts/corpus.ts <check|fetch|verify> [--keep-downloads]");
}

if (import.meta.main) {
  run(Bun.argv[2] ?? "", Bun.argv.slice(3)).catch((error) => {
    console.error(`corpus: ${error instanceof Error ? error.message : String(error)}`);
    process.exit(1);
  });
}
