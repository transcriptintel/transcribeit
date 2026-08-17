import { readdir } from "node:fs/promises";
import { join } from "node:path";

import { scoreTranscript } from "./scoring";
import { sha256 } from "./state";
import type { AttemptRecord } from "./types";

export async function collectAttemptSuccess(
  attempt: AttemptRecord,
  outputDirectory: string,
  stderr: string,
  referenceScoring: boolean,
): Promise<void> {
  const files = await readdir(outputDirectory);
  const manifestName = files.find((name) => name.endsWith(".manifest.json"));
  const outputName = files.find((name) => name.endsWith(".txt"));
  if (!manifestName || !outputName) throw new Error("expected text and manifest output");
  const manifestText = await Bun.file(join(outputDirectory, manifestName)).text();
  const output = new Uint8Array(await Bun.file(join(outputDirectory, outputName)).arrayBuffer());
  const outputText = new TextDecoder().decode(output);
  const manifest = JSON.parse(manifestText);
  attempt.output_sha256 = sha256(output);
  attempt.manifest_sha256 = sha256(manifestText);
  const processingSeconds = finiteNumber(manifest.stats?.processing_time_secs);
  attempt.processing_ms = processingSeconds === null ? null : processingSeconds * 1000;
  const capabilityNames = [
    "segments",
    "word_timestamps",
    "speaker_labels",
    "language_per_segment",
    "emotion_per_segment",
    "native_timestamps",
  ];
  attempt.capabilities = Object.fromEntries(
    capabilityNames.map((name) => [name, manifest.capabilities?.[name] === true]),
  );
  attempt.quality = {
    timing_source: safeEnum(manifest.quality?.timing_source, [
      "provider_native", "model_native", "model_generated", "synthetic", "unknown", "none",
    ]),
    timing_reliable: typeof manifest.quality?.timing_reliable === "boolean" ? manifest.quality.timing_reliable : null,
    timestamps_clamped: typeof manifest.quality?.timestamps_clamped === "boolean"
      ? manifest.quality.timestamps_clamped
      : null,
    speaker_source: safeEnum(manifest.quality?.speaker_source, [
      "provider_native", "model_generated", "local_postprocess", "unknown", "none",
    ]),
    warning_count: Array.isArray(manifest.quality?.warnings) ? manifest.quality.warnings.length : 0,
  };
  const segments = Array.isArray(manifest.transcript?.segments) ? manifest.transcript.segments : [];
  const lastEnd = segments.reduce((maximum: number | null, segment: any) => {
    const value = finiteNumber(segment?.end_secs);
    return value === null ? maximum : Math.max(maximum ?? 0, value);
  }, null);
  attempt.output_shape = {
    segments: segments.length,
    characters: Number.isInteger(manifest.stats?.total_characters)
      ? manifest.stats.total_characters
      : segments.reduce(
          (sum: number, segment: any) => sum + (typeof segment?.text === "string" ? segment.text.length : 0),
          0,
        ),
    last_end_ms: lastEnd === null ? null : Math.round(lastEnd * 1000),
    zero_duration_segments: segments.filter((segment: any) => {
      const start = finiteNumber(segment?.start_secs);
      const end = finiteNumber(segment?.end_secs);
      return start !== null && end !== null && end === start;
    }).length,
    reversed_segments: segments.filter((segment: any) => {
      const start = finiteNumber(segment?.start_secs);
      const end = finiteNumber(segment?.end_secs);
      return start !== null && end !== null && end < start;
    }).length,
    word_timestamps: manifest.capabilities?.word_timestamps === true,
    speaker_labels: manifest.capabilities?.speaker_labels === true,
  };
  attempt.preprocessing = preprocessingClassification(attempt.provider, stderr);
  attempt.apple_speech = appleSpeechObservation(attempt.provider, manifest);
  attempt.reference_metrics = referenceScoring
    ? await scoreTranscript(attempt.fixture.id, outputText, manifest)
    : null;
  attempt.remote_cleanup = cleanupClassification(attempt.provider, manifest);
}

export function parsePeakRss(stderr: string): number | null {
  const match = stderr.match(/^\s*(\d+)\s+maximum resident set size\s*$/m);
  if (!match) return null;
  const value = Number(match[1]);
  return Number.isSafeInteger(value) && value > 0 ? value : null;
}

function safeEnum(value: unknown, allowed: string[]): string | null {
  return typeof value === "string" && allowed.includes(value) ? value : null;
}

function finiteNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function preprocessingClassification(provider: string, stderr: string): AttemptRecord["preprocessing"] {
  if (provider === "apple-speech") {
    return stderr.includes("AVAudioFile could not read the original media") ? "wav_fallback" : "original_media";
  }
  return provider === "local" ? "canonical_wav" : "original_media";
}

function appleSpeechObservation(provider: string, manifest: any): AttemptRecord["apple_speech"] {
  if (provider !== "apple-speech") return null;
  const response = manifest?.provider_metadata?.data?.response;
  const locale = typeof response?.locale === "string" && /^[A-Za-z0-9_-]{1,35}$/.test(response.locale)
    ? response.locale
    : null;
  return {
    resolved_locale: locale,
    apple_intelligence_available: typeof response?.apple_intelligence_available === "boolean"
      ? response.apple_intelligence_available
      : null,
    asset_install_requested: typeof response?.asset_install_requested === "boolean"
      ? response.asset_install_requested
      : null,
    asset_managed_by: response?.asset_managed_by === "macos" ? "macos" : null,
    on_device: typeof response?.on_device === "boolean" ? response.on_device : null,
  };
}

function cleanupClassification(provider: string, manifest: any): AttemptRecord["remote_cleanup"] {
  if (provider === "qwen-filetrans" || provider === "deepgram") {
    const cleanup = manifest?.provider_metadata?.data?.staging?.cleanup;
    if (!cleanup) return provider === "deepgram" ? "not_applicable" : null;
    if (cleanup.attempted === true && cleanup.deleted === true && cleanup.error == null) return "deleted";
    return cleanup.attempted === false ? "not_attempted" : "failed";
  }
  if (provider === "gemini") {
    const file = manifest?.provider_metadata?.data?.file;
    if (!file) return null;
    if (file.delete_attempted === true && file.deleted === true && file.delete_error == null) return "deleted";
    return file.delete_attempted === false ? "not_attempted" : "failed";
  }
  return "not_applicable";
}
