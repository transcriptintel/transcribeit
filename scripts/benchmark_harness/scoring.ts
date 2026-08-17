import { resolve, sep } from "node:path";

import { validateManifest } from "../corpus";

type MapValue = Record<string, any>;
export type ScoringIntegrityKind = "corpus" | "reviewed_reference";

export class ScoringIntegrityError extends Error {
  override readonly name = "ScoringIntegrityError";
  constructor(readonly kind: ScoringIntegrityKind) {
    super(kind === "reviewed_reference"
      ? "reviewed reference integrity verification failed"
      : "benchmark scoring corpus integrity verification failed");
  }
}

export type ReviewedReferenceToken = { text: string; start?: number; end?: number };
export type ReviewedScoringReference = {
  tokens: ReviewedReferenceToken[];
  coverage: {
    domain_terms?: string[];
    timestamp_reference: boolean;
    diarization_reference: boolean;
    overlapping_speech: boolean;
  };
};
type Alignment = { substitutions: number; deletions: number; insertions: number; exactPairs: Array<[number, number]> };
type Unavailable = { status: "unavailable_reference" };
type CapabilityStatus =
  | Unavailable
  | { status: "unsupported_reference" | "unsupported_provider" | "available_not_scored" };
type TimingMetrics =
  | Unavailable
  | {
      status:
        | "unsupported_reference"
        | "unsupported_provider"
        | "unsupported_overlapping_reference"
        | "insufficient_alignment";
    }
  | {
      status: "scored";
      start_boundary_mae_ms: number; end_boundary_mae_ms: number; timestamp_coverage: number;
      scored_segments: number; timing_origin: string | null; timing_reliable: boolean | null;
    };

export type ReferenceMetrics = {
  status: "scored" | "unavailable_reference";
  word_accuracy:
    | Unavailable
    | { status: "unsupported_overlapping_reference" }
    | {
        status: "scored"; reference_words: number; hypothesis_words: number;
        substitutions: number; deletions: number; insertions: number; wer: number;
      };
  domain_terms:
    | Unavailable
    | { status: "unsupported_reference" }
    | { status: "scored"; expected_terms: number; matched_terms: number; term_recall: number };
  timing: TimingMetrics;
  speakers: CapabilityStatus;
  word_timestamps: CapabilityStatus;
};
const repository = resolve(import.meta.dir, "../..");
const corpusPath = resolve(repository, "benchmarks/corpus/v1/manifest.yaml");
let corpusPromise: Promise<MapValue> | undefined;
/** Score a successful attempt without returning reference text, transcript text, or paths. */
export async function scoreTranscript(
  fixtureId: string,
  transcriptText: string,
  manifest: Record<string, any>,
): Promise<ReferenceMetrics> {
  const fixture = await findFixture(fixtureId);
  if (!fixture || fixture.reference?.reviewed !== true) return unavailableMetrics();
  const reference = await referenceTokens(fixture);
  if (!reference?.length) return unavailableMetrics();
  return scoreReviewedTranscript({
    tokens: reference,
    coverage: {
      domain_terms: Array.isArray(fixture.coverage?.domain_terms)
        ? fixture.coverage.domain_terms.filter((term: unknown): term is string => typeof term === "string")
        : [],
      timestamp_reference: fixture.coverage?.timestamp_reference === true,
      diarization_reference: fixture.coverage?.diarization_reference === true,
      overlapping_speech: fixture.coverage?.overlapping_speech === true,
    },
  }, transcriptText, manifest);
}

/** Pure scoring boundary for reviewed or synthetic references. */
export function scoreReviewedTranscript(
  reviewed: ReviewedScoringReference,
  transcriptText: string,
  manifest: Record<string, any>,
): ReferenceMetrics {
  const reference = reviewed.tokens.flatMap((token) =>
    normalizedWords(token.text).map((text) => ({ text, start: token.start, end: token.end })),
  );
  if (!reference.length) return unavailableMetrics();
  const hypothesis = normalizedWords(transcriptText);
  const domainTerms = reviewed.coverage.domain_terms ?? [];
  const hasTimingReference = reviewed.coverage.timestamp_reference;
  const hasSpeakerReference = reviewed.coverage.diarization_reference;
  const hasOverlappingSpeakers = reviewed.coverage.overlapping_speech;
  const segments = Array.isArray(manifest?.transcript?.segments) ? manifest.transcript.segments : [];
  return {
    status: "scored",
    word_accuracy: hasOverlappingSpeakers
      ? { status: "unsupported_overlapping_reference" }
      : scoreWordAccuracy(reference, hypothesis),
    domain_terms: domainTerms.length ? scoreTerms(hypothesis, domainTerms) : { status: "unsupported_reference" },
    timing: hasOverlappingSpeakers
      ? { status: "unsupported_overlapping_reference" }
      : scoreTiming(reference, segments, manifest, hasTimingReference),
    speakers: capabilityStatus(hasSpeakerReference, hasSpeakerLabels(segments)),
    word_timestamps: capabilityStatus(hasTimingReference, hasWordTimestamps(segments)),
  };
}
function unavailableMetrics(): ReferenceMetrics {
  const unavailable = (): Unavailable => ({ status: "unavailable_reference" });
  return {
    status: "unavailable_reference",
    word_accuracy: unavailable(),
    domain_terms: unavailable(),
    timing: unavailable(),
    speakers: unavailable(),
    word_timestamps: unavailable(),
  };
}
async function findFixture(id: string): Promise<MapValue | undefined> {
  corpusPromise ??= loadScoringCorpus();
  const corpus = await corpusPromise;
  const fixtures = corpus.fixtures as MapValue[];
  const matches = fixtures.filter((fixture) => fixture.id === id);
  if (matches.length > 1) corpusIntegrityFailure();
  return matches[0];
}

async function loadScoringCorpus(): Promise<MapValue> {
  try {
    return validatedScoringCorpus(Bun.YAML.parse(await Bun.file(corpusPath).text()));
  } catch (error) {
    if (error instanceof ScoringIntegrityError) throw error;
    corpusIntegrityFailure();
  }
}
export function validatedScoringCorpus(value: unknown): Record<string, any> {
  const { manifest, errors } = validateManifest(value);
  if (!manifest || errors.length) corpusIntegrityFailure();
  return manifest as unknown as MapValue;
}

async function referenceTokens(fixture: MapValue): Promise<ReviewedReferenceToken[] | undefined> {
  if (fixture.reference?.kind === "inline_reviewed") {
    if (typeof fixture.reference.text !== "string") referenceIntegrityFailure();
    const canonical = new TextEncoder().encode(`${fixture.reference.text}\n`);
    verifyReviewedReferenceBytes(canonical, fixture.reference.bytes, fixture.reference.sha256);
    const materialized = await verifiedReferenceFile(fixture.reference);
    if (!bytesEqual(canonical, materialized)) referenceIntegrityFailure();
    return normalizedWords(fixture.reference.text).map((text) => ({ text }));
  }
  if (fixture.reference?.kind !== "archive_members" || !Array.isArray(fixture.reference.files)) referenceIntegrityFailure();
  const tokens: ReviewedReferenceToken[] = [];
  for (const file of fixture.reference.files) {
    const bytes = await verifiedReferenceFile(file);
    if (!String(file?.member).endsWith(".words.xml")) continue;
    const xml = new TextDecoder("iso-8859-1").decode(bytes);
    for (const match of xml.matchAll(/<w\s+([^>]*)>([\s\S]*?)<\/w>/g)) {
      const attrs = xmlAttributes(match[1]);
      if (attrs.punc === "true") continue;
      const start = Number(attrs.starttime);
      const end = Number(attrs.endtime);
      if (!Number.isFinite(start) || !Number.isFinite(end)) continue;
      for (const text of normalizedWords(decodeXml(match[2]))) tokens.push({ text, start, end });
    }
  }
  tokens.sort((left, right) => Number(left.start) - Number(right.start) || Number(left.end) - Number(right.end));
  return tokens;
}

async function verifiedReferenceFile(reference: MapValue): Promise<Uint8Array> {
  const path = safeReferencePath(reference.path);
  if (!path) referenceIntegrityFailure();
  const file = Bun.file(path);
  let bytes: Uint8Array;
  try {
    if (!(await file.exists())) referenceIntegrityFailure();
    bytes = await file.bytes();
  } catch {
    referenceIntegrityFailure();
  }
  verifyReviewedReferenceBytes(bytes, reference.bytes, reference.sha256);
  return bytes;
}

export function verifyReviewedReferenceBytes(bytes: Uint8Array, expectedBytes: unknown, expectedSha256: unknown): void {
  if (
    !Number.isSafeInteger(expectedBytes) ||
    Number(expectedBytes) < 0 ||
    typeof expectedSha256 !== "string" ||
    !/^[a-f0-9]{64}$/.test(expectedSha256)
  ) referenceIntegrityFailure();
  const hasher = new Bun.CryptoHasher("sha256");
  hasher.update(bytes);
  if (bytes.byteLength !== expectedBytes || hasher.digest("hex") !== expectedSha256) referenceIntegrityFailure();
}

function referenceIntegrityFailure(): never {
  throw new ScoringIntegrityError("reviewed_reference");
}
function corpusIntegrityFailure(): never {
  throw new ScoringIntegrityError("corpus");
}

function bytesEqual(left: Uint8Array, right: Uint8Array): boolean {
  return left.byteLength === right.byteLength && left.every((value, index) => value === right[index]);
}

function safeReferencePath(value: unknown): string | undefined {
  if (typeof value !== "string" || value.startsWith("/")) return undefined;
  const root = resolve(repository, "samples/corpus/v1/references");
  const path = resolve(repository, value);
  return path.startsWith(`${root}${sep}`) ? path : undefined;
}

function normalizedWords(text: string): string[] {
  return text
    .normalize("NFKC")
    .toLowerCase()
    .replaceAll("’", "'")
    .match(/[\p{L}\p{N}]+(?:'[\p{L}\p{N}]+)*/gu) ?? [];
}

function align(reference: string[], hypothesis: string[]): Alignment {
  const columns = hypothesis.length + 1;
  const directions = new Uint8Array((reference.length + 1) * columns);
  let previous = new Uint32Array(columns);
  let current = new Uint32Array(columns);
  for (let column = 1; column < columns; column++) {
    previous[column] = column;
    directions[column] = 3;
  }
  for (let row = 1; row <= reference.length; row++) {
    current[0] = row;
    directions[row * columns] = 2;
    for (let column = 1; column < columns; column++) {
      const equal = reference[row - 1] === hypothesis[column - 1];
      const diagonal = previous[column - 1] + (equal ? 0 : 1);
      const deletion = previous[column] + 1;
      const insertion = current[column - 1] + 1;
      if (diagonal <= deletion && diagonal <= insertion) {
        current[column] = diagonal;
        directions[row * columns + column] = equal ? 1 : 4;
      } else if (deletion <= insertion) {
        current[column] = deletion;
        directions[row * columns + column] = 2;
      } else {
        current[column] = insertion;
        directions[row * columns + column] = 3;
      }
    }
    [previous, current] = [current, previous];
  }
  let row = reference.length;
  let column = hypothesis.length;
  let substitutions = 0;
  let deletions = 0;
  let insertions = 0;
  const exactPairs: Array<[number, number]> = [];
  while (row || column) {
    const direction = directions[row * columns + column];
    if (direction === 1) {
      exactPairs.push([--row, --column]);
    } else if (direction === 4) {
      substitutions++;
      row--;
      column--;
    } else if (direction === 2) {
      deletions++;
      row--;
    } else {
      insertions++;
      column--;
    }
  }
  exactPairs.reverse();
  return { substitutions, deletions, insertions, exactPairs };
}

function scoreWordAccuracy(reference: ReviewedReferenceToken[], hypothesis: string[]): ReferenceMetrics["word_accuracy"] {
  const alignment = align(reference.map((token) => token.text), hypothesis);
  return {
    status: "scored",
    reference_words: reference.length,
    hypothesis_words: hypothesis.length,
    substitutions: alignment.substitutions,
    deletions: alignment.deletions,
    insertions: alignment.insertions,
    wer: (alignment.substitutions + alignment.deletions + alignment.insertions) / reference.length,
  };
}

function scoreTerms(hypothesis: string[], terms: string[]): ReferenceMetrics["domain_terms"] {
  let matched = 0;
  for (const term of terms) {
    const target = normalizedWords(term);
    const present =
      target.length > 0 &&
      hypothesis.some((_, index) => target.every((token, offset) => hypothesis[index + offset] === token));
    if (present) matched++;
  }
  return { status: "scored", expected_terms: terms.length, matched_terms: matched, term_recall: matched / terms.length };
}

function scoreTiming(
  reference: ReviewedReferenceToken[],
  rawSegments: MapValue[],
  manifest: MapValue,
  hasReference: boolean,
): TimingMetrics {
  if (!hasReference) return { status: "unsupported_reference" };
  const hypothesis: Array<ReviewedReferenceToken & { segment: number }> = [];
  for (const [segment, value] of rawSegments.entries()) {
    const start = Number(value?.start_secs);
    const end = Number(value?.end_secs);
    if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) continue;
    for (const text of normalizedWords(typeof value?.text === "string" ? value.text : "")) {
      hypothesis.push({ text, start, end, segment });
    }
  }
  if (!hypothesis.length) return { status: "unsupported_provider" };
  const alignment = align(
    reference.map((token) => token.text),
    hypothesis.map((token) => token.text),
  );
  const groups = new Map<number, number[]>();
  for (const [referenceIndex, hypothesisIndex] of alignment.exactPairs) {
    const indexes = groups.get(hypothesis[hypothesisIndex].segment) ?? [];
    indexes.push(referenceIndex);
    groups.set(hypothesis[hypothesisIndex].segment, indexes);
  }
  let startError = 0;
  let endError = 0;
  let scoredSegments = 0;
  const covered = new Set<number>();
  for (const [segment, indexes] of groups) {
    const candidate = hypothesis.find((token) => token.segment === segment)!;
    const starts = indexes.map((index) => reference[index].start).filter(Number.isFinite) as number[];
    const ends = indexes.map((index) => reference[index].end).filter(Number.isFinite) as number[];
    if (!starts.length || !ends.length) continue;
    startError += Math.abs(Number(candidate.start) - Math.min(...starts)) * 1000;
    endError += Math.abs(Number(candidate.end) - Math.max(...ends)) * 1000;
    scoredSegments++;
    indexes.filter((index) => Number.isFinite(reference[index].start) && Number.isFinite(reference[index].end))
      .forEach((index) => covered.add(index));
  }
  const timestampCoverage = covered.size / reference.length;
  if (!scoredSegments || timestampCoverage < 0.5) return { status: "insufficient_alignment" };
  return {
    status: "scored",
    start_boundary_mae_ms: startError / scoredSegments,
    end_boundary_mae_ms: endError / scoredSegments,
    timestamp_coverage: timestampCoverage,
    scored_segments: scoredSegments,
    timing_origin: typeof manifest?.quality?.timing_source === "string" ? manifest.quality.timing_source : null,
    timing_reliable: typeof manifest?.quality?.timing_reliable === "boolean" ? manifest.quality.timing_reliable : null,
  };
}

function capabilityStatus(hasReference: boolean, available: boolean): CapabilityStatus {
  if (!hasReference) return { status: "unsupported_reference" };
  return { status: available ? "available_not_scored" : "unsupported_provider" };
}

function hasSpeakerLabels(segments: MapValue[]): boolean {
  return segments.some((segment) => typeof segment?.speaker === "string" && segment.speaker.trim().length > 0);
}

function hasWordTimestamps(segments: MapValue[]): boolean {
  return segments.some(
    (segment) =>
      Array.isArray(segment?.words) &&
      segment.words.some((word: MapValue) => Number.isFinite(word?.start_secs) && Number.isFinite(word?.end_secs)),
  );
}

function xmlAttributes(text: string): Record<string, string> {
  return Object.fromEntries([...text.matchAll(/([\w:-]+)="([^"]*)"/g)].map((match) => [match[1], match[2]]));
}

function decodeXml(text: string): string {
  return text
    .replace(/&#x([0-9a-f]+);/gi, (_, value) => String.fromCodePoint(Number.parseInt(value, 16)))
    .replace(/&#([0-9]+);/g, (_, value) => String.fromCodePoint(Number(value)))
    .replaceAll("&apos;", "'")
    .replaceAll("&quot;", '"')
    .replaceAll("&lt;", "<")
    .replaceAll("&gt;", ">")
    .replaceAll("&amp;", "&");
}
