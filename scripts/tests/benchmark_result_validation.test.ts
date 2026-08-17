import { validatePublishedResult } from "../benchmark_harness/result";
import { matrixSha256 } from "../benchmark_harness/state";
import type { BenchmarkMatrix, PublishedResult } from "../benchmark_harness/types";

describe("published benchmark result validation", () => {
  test("accepts strict passed, failed, and skipped terminal records", () => {
    expect(validatePublishedResult(validResult())).toEqual([]);
    expect(validatePublishedResult(validAppleResult())).toEqual([]);
    expect(validatePublishedResult(validScoredResult())).toEqual([]);

    const failed = validResult();
    Object.assign(failed.attempts[0], {
      status: "failed",
      error_category: "timeout",
      output_sha256: null,
      manifest_sha256: null,
      capabilities: null,
      quality: null,
      output_shape: null,
      preprocessing: null,
      remote_cleanup: null,
    });
    failed.summary = { attempts: 1, passed: 0, failed: 1, skipped: 0 };
    expect(validatePublishedResult(failed)).toEqual([]);

    const skipped = validResult();
    Object.assign(skipped.attempts[0], {
      status: "skipped",
      wall_ms: null,
      processing_ms: null,
      real_time_factor: null,
      peak_rss_bytes: null,
      error_category: "unconfigured",
      output_sha256: null,
      manifest_sha256: null,
      capabilities: null,
      quality: null,
      output_shape: null,
      preprocessing: null,
      apple_speech: null,
      reference_metrics: null,
      remote_cleanup: null,
    });
    skipped.summary = { attempts: 1, passed: 0, failed: 0, skipped: 1 };
    expect(validatePublishedResult(skipped)).toEqual([]);
  });

  test("rejects bogus terminal status even when the summary is adjusted", () => {
    const result = validResult();
    result.attempts[0].status = "invented" as never;
    result.summary = { attempts: 1, passed: 0, failed: 0, skipped: 0 };

    expect(invalid(result, "status must be terminal")).toBe(true);
    expect(invalid(result, "terminal counts")).toBe(true);
  });

  test("rejects inconsistent classification, hashes, timestamps, and numeric ranges", () => {
    const dirty = validResult();
    dirty.producing_commit.worktree_dirty = true;
    expect(invalid(dirty, "classification does not match")).toBe(true);

    const hash = validResult();
    hash.attempts[0].output_sha256 = "A".repeat(64);
    expect(invalid(hash, "output_sha256")).toBe(true);

    const timestamp = validResult();
    timestamp.attempts[0].completed_at_utc = "2026-08-17T11:59:59.000Z";
    expect(invalid(timestamp, "predates started_at_utc")).toBe(true);

    const number = validResult();
    number.attempts[0].wall_ms = -1;
    expect(invalid(number, "wall_ms")).toBe(true);
  });

  test("rejects invalid matrix scalars, free-form args, models, and binary classifications", () => {
    const concurrency = validResult();
    concurrency.matrix.concurrency = "2" as never;
    rehash(concurrency);
    expect(invalid(concurrency, "matrix.concurrency")).toBe(true);

    const args = validResult();
    args.matrix.entries[0].args = ["private transcript text"];
    rehash(args);
    expect(invalid(args, "unsupported reviewed option")).toBe(true);

    const model = validResult();
    model.matrix.entries[0].model = "model containing private prose";
    model.attempts[0].model = model.matrix.entries[0].model;
    rehash(model);
    expect(invalid(model, "safe non-empty classification")).toBe(true);

    const binary = validResult();
    binary.runtime.binary_classification = "release binary containing private prose";
    expect(invalid(binary, "safe relative classification")).toBe(true);
  });

  test("rejects bad output, Apple, reference, and cleanup shapes", () => {
    const nestedText = validResult();
    nestedText.attempts[0].output_shape!.characters = "private transcript" as never;
    expect(invalid(nestedText, "output_shape.characters")).toBe(true);

    const apple = validAppleResult();
    apple.attempts[0].apple_speech!.on_device = "yes" as never;
    expect(invalid(apple, "apple_speech.on_device")).toBe(true);

    const reference = validScoredResult();
    reference.attempts[0].reference_metrics!.speakers.status = "scored" as never;
    expect(invalid(reference, "speakers must contain only a valid status")).toBe(true);

    const cleanup = validResult();
    cleanup.cleanup.attempt_outputs_removed = "yes" as never;
    expect(invalid(cleanup, "attempt_outputs_removed")).toBe(true);
  });

  test("requires complete Apple metadata for passed attempts while allowing nullable failure diagnostics", () => {
    const invalidAppleMetadata: Array<{
      field: keyof NonNullable<PublishedResult["attempts"][number]["apple_speech"]>;
      value: unknown;
      error: string;
    }> = [
      { field: "resolved_locale", value: null, error: "resolved_locale is required" },
      { field: "apple_intelligence_available", value: false, error: "apple_intelligence_available must be true" },
      { field: "asset_install_requested", value: null, error: "asset_install_requested must be a boolean" },
      { field: "asset_managed_by", value: null, error: "asset_managed_by must be macos" },
      { field: "on_device", value: false, error: "on_device must be true" },
    ];
    for (const { field, value, error } of invalidAppleMetadata) {
      const result = validAppleResult();
      Object.assign(result.attempts[0].apple_speech!, { [field]: value });
      expect(invalid(result, error)).toBe(true);
    }

    const failed = validAppleResult();
    Object.assign(failed.attempts[0], {
      status: "failed",
      error_category: "provider_error",
      apple_speech: {
        resolved_locale: null,
        apple_intelligence_available: null,
        asset_install_requested: null,
        asset_managed_by: null,
        on_device: null,
      },
    });
    failed.summary = { attempts: 1, passed: 0, failed: 1, skipped: 0 };
    expect(validatePublishedResult(failed)).toEqual([]);
  });

  test("rejects nested text and additional macOS or Unix absolute roots", () => {
    const nested = validResult();
    nested.environment.machine.cpu = { transcript: "private text" } as never;
    expect(invalid(nested, "must not contain nested data")).toBe(true);

    for (const path of ["/System/Library/private", "/bin/private", "/sbin/private", "/dev/private", "/run/private", "/nix/private"]) {
      const absolute = validResult();
      absolute.environment.machine.cpu = path;
      expect(invalid(absolute, "forbidden path or credential")).toBe(true);
    }
  });
});

function validResult(): PublishedResult {
  const matrix: BenchmarkMatrix = {
    schema_version: 1,
    matrix_id: "strict-local-test",
    description: "Strict local result validation fixture",
    execution_policy: "local_ci",
    attempt_order: "entry-fixture-repetition",
    concurrency: 1,
    retries: 0,
    timeout_seconds: 30,
    fixture_ids: ["sample-fixture"],
    measurements: { reference_scoring: false, peak_rss: false },
    entries: [{
      id: "local-test",
      provider: "local",
      model: "large-v3",
      execution: "local",
      cache_state: "warm",
      repetitions: 1,
      required_env: [],
      args: [],
    }],
    tolerances: {
      enforcement: "fail",
      max_relative_latency_regression_percent: 25,
      max_absolute_latency_regression_ms: 100,
      min_success_rate: 1,
    },
  };
  const publishedMatrix = {
    ...matrix,
    entries: matrix.entries.map((entry) => ({ ...entry, command_template: commandTemplate(matrix, entry) })),
  };
  return {
    schema_version: "transcribeit.benchmark-harness-result.v1",
    matrix_sha256: matrixSha256(matrix),
    recorded_at_utc: "2026-08-17T12:00:02.000Z",
    classification: "clean_commit_benchmark",
    producing_commit: {
      hash: "a".repeat(40),
      worktree_dirty: false,
      worktree_fingerprint_sha256: "b".repeat(64),
    },
    runtime: { binary_classification: "target/release/transcribeit", binary_sha256: "c".repeat(64) },
    environment: {
      machine: {
        cpu: "Apple M4 Max",
        logical_cores: 16,
        memory_bytes: 64_000_000_000,
        os: "darwin",
        os_version: "26.0",
        os_build: "25A354",
        kernel: "25.0.0",
        architecture: "arm64",
      },
      tools: {
        rustc: "rustc 1.97.0 (example 2026-08-01)",
        ffmpeg: "ffmpeg version 8.0 Copyright 2000-2026 the FFmpeg developers",
        swift: "Apple Swift version 6.2 (swiftlang-6.2.0) Target: arm64-apple-macosx26.0",
        bun: "1.3.14",
      },
    },
    matrix: publishedMatrix,
    summary: { attempts: 1, passed: 1, failed: 0, skipped: 0 },
    attempts: [{
      key: "local-test--sample-fixture--001",
      entry_id: "local-test",
      provider: "local",
      model: "large-v3",
      execution: "local",
      cache_state: "warm",
      fixture: { id: "sample-fixture", duration_seconds: 1, bytes: 1, sha256: "d".repeat(64) },
      repetition: 1,
      status: "passed",
      started_at_utc: "2026-08-17T12:00:00.000Z",
      completed_at_utc: "2026-08-17T12:00:01.000Z",
      wall_ms: 100,
      processing_ms: 90,
      real_time_factor: 0.1,
      peak_rss_bytes: null,
      error_category: null,
      output_sha256: "e".repeat(64),
      manifest_sha256: "f".repeat(64),
      capabilities: {
        segments: true,
        word_timestamps: false,
        speaker_labels: false,
        language_per_segment: false,
        emotion_per_segment: false,
        native_timestamps: true,
      },
      quality: {
        timing_source: "model_native",
        timing_reliable: true,
        timestamps_clamped: false,
        speaker_source: "none",
        warning_count: 0,
      },
      output_shape: {
        segments: 1,
        characters: 5,
        last_end_ms: 1_000,
        zero_duration_segments: 0,
        reversed_segments: 0,
        word_timestamps: false,
        speaker_labels: false,
      },
      preprocessing: "canonical_wav",
      apple_speech: null,
      reference_metrics: null,
      remote_cleanup: "not_applicable",
    }],
    cleanup: {
      attempt_outputs_removed: true,
      evaluation_root_sha256: null,
      downloaded_model: null,
      apple_speech_asset: null,
    },
    sanitization: {
      credential_values_removed: true,
      stdout_stderr_removed: true,
      transcript_text_removed: true,
      request_ids_removed: true,
      signed_urls_removed: true,
      local_absolute_paths_removed: true,
      provider_metadata_allowlisted: true,
    },
  };
}

function validAppleResult(): PublishedResult {
  const result = validResult();
  const entry = result.matrix.entries[0];
  Object.assign(entry, {
    id: "apple-test",
    provider: "apple-speech",
    model: "speech-transcriber",
    args: ["--language", "en-US"],
    artifact: { lifecycle: "system_managed" },
  });
  entry.command_template = commandTemplate(result.matrix as BenchmarkMatrix, entry);
  Object.assign(result.attempts[0], {
    key: "apple-test--sample-fixture--001",
    entry_id: "apple-test",
    provider: "apple-speech",
    model: "speech-transcriber",
    preprocessing: "original_media",
    apple_speech: {
      resolved_locale: "en-US",
      apple_intelligence_available: true,
      asset_install_requested: false,
      asset_managed_by: "macos",
      on_device: true,
    },
  });
  result.cleanup.apple_speech_asset = {
    owner: "macos",
    cleanup_attempted: false,
    lifecycle: "system_managed_shared",
  };
  rehash(result);
  return result;
}

function validScoredResult(): PublishedResult {
  const result = validResult();
  result.matrix.measurements.reference_scoring = true;
  result.attempts[0].reference_metrics = {
    status: "scored",
    word_accuracy: {
      status: "scored",
      reference_words: 1,
      hypothesis_words: 1,
      substitutions: 0,
      deletions: 0,
      insertions: 0,
      wer: 0,
    },
    domain_terms: { status: "scored", expected_terms: 1, matched_terms: 1, term_recall: 1 },
    timing: {
      status: "scored",
      start_boundary_mae_ms: 0,
      end_boundary_mae_ms: 0,
      timestamp_coverage: 1,
      scored_segments: 1,
      timing_origin: "model_native",
      timing_reliable: true,
    },
    speakers: { status: "unsupported_reference" },
    word_timestamps: { status: "unsupported_provider" },
  };
  rehash(result);
  return result;
}

function commandTemplate(matrix: BenchmarkMatrix, entry: BenchmarkMatrix["entries"][number]): string {
  const model = entry.provider === "local"
    ? ` --model ${entry.model}`
    : ["apple-speech", "azure", "nvidia-riva"].includes(entry.provider)
      ? ""
      : ` --remote-model ${entry.model}`;
  return `transcribeit run --provider ${entry.provider}${model} --max-retries ${matrix.retries} --request-timeout-secs ${matrix.timeout_seconds} --input $FIXTURE --output-format text --output-dir $ATTEMPT_DIR${entry.args.length ? ` ${entry.args.join(" ")}` : ""}`;
}

function rehash(result: PublishedResult): void {
  const normalized = {
    ...result.matrix,
    entries: result.matrix.entries.map(({ command_template: _template, ...entry }) => entry),
  } as BenchmarkMatrix;
  result.matrix_sha256 = matrixSha256(normalized);
}

function invalid(result: PublishedResult, fragment: string): boolean {
  return validatePublishedResult(result).some((error) => error.includes(fragment));
}
