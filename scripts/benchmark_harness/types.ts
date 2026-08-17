import type { ReferenceMetrics } from "./scoring";

export const providers = [
  "apple-speech",
  "local",
  "openai",
  "azure",
  "qwen-filetrans",
  "gemini",
  "nvidia-riva",
  "deepgram",
] as const;

export type Provider = (typeof providers)[number];
export type ExecutionClass = "local" | "hosted";
export type CacheState = "cold" | "warm";
export type ExecutionPolicy = "manual" | "scheduled" | "local_ci";
export type AttemptOrder = "entry-fixture-repetition" | "fixture-repetition-entry";
export type ToleranceEnforcement = "report_only" | "fail";
export type AttemptStatus = "pending" | "running" | "passed" | "failed" | "skipped";

export type ModelArtifact = {
  lifecycle: "evaluation_download" | "system_managed" | "provider_managed";
  revision?: string;
  sha256?: string;
  bytes?: number;
};

export type MatrixEntry = {
  id: string;
  provider: Provider;
  model: string;
  execution: ExecutionClass;
  cache_state: CacheState;
  repetitions: number;
  fixture_ids?: string[];
  required_env: string[];
  args: string[];
  artifact?: ModelArtifact;
};

export type MeasurementPolicy = {
  reference_scoring: boolean;
  peak_rss: boolean;
};

export type TolerancePolicy = {
  enforcement: ToleranceEnforcement;
  max_relative_latency_regression_percent: number;
  max_absolute_latency_regression_ms: number;
  min_success_rate: number;
};

export type BenchmarkMatrix = {
  schema_version: 1;
  matrix_id: string;
  description: string;
  execution_policy: ExecutionPolicy;
  attempt_order: AttemptOrder;
  concurrency: number;
  retries: number;
  timeout_seconds: number;
  fixture_ids: string[];
  entries: MatrixEntry[];
  measurements: MeasurementPolicy;
  tolerances: TolerancePolicy;
};

export type FixtureIdentity = {
  id: string;
  path: string;
  duration_seconds: number;
  bytes: number;
  sha256: string;
};

export type AttemptRecord = {
  key: string;
  entry_id: string;
  provider: Provider;
  model: string;
  execution: ExecutionClass;
  cache_state: CacheState;
  fixture: Omit<FixtureIdentity, "path">;
  repetition: number;
  status: AttemptStatus;
  started_at_utc: string | null;
  completed_at_utc: string | null;
  wall_ms: number | null;
  processing_ms: number | null;
  real_time_factor: number | null;
  peak_rss_bytes: number | null;
  error_category: string | null;
  output_sha256: string | null;
  manifest_sha256: string | null;
  capabilities: Record<string, boolean> | null;
  quality: {
    timing_source: string | null;
    timing_reliable: boolean | null;
    timestamps_clamped: boolean | null;
    speaker_source: string | null;
    warning_count: number;
  } | null;
  output_shape: {
    segments: number;
    characters: number;
    last_end_ms: number | null;
    zero_duration_segments: number;
    reversed_segments: number;
    word_timestamps: boolean;
    speaker_labels: boolean;
  } | null;
  preprocessing: "original_media" | "wav_fallback" | "canonical_wav" | null;
  apple_speech: {
    resolved_locale: string | null;
    apple_intelligence_available: boolean | null;
    asset_install_requested: boolean | null;
    asset_managed_by: "macos" | null;
    on_device: boolean | null;
  } | null;
  reference_metrics: ReferenceMetrics | null;
  remote_cleanup: "deleted" | "failed" | "not_attempted" | "not_applicable" | null;
};

export type DownloadedModelCleanup = {
  schema_version: "transcribeit.downloaded-model-cleanup.v1";
  matrix_sha256: string;
  evaluation_root_sha256: string;
  artifact_sha256s: string[];
  recorded_at_utc: string;
  evaluation_root_preexisting: boolean;
  initial_bytes: number;
  downloaded_bytes: number;
  reclaimed_bytes: number;
  cleanup_completed: boolean;
  evaluation_root_absent_after_cleanup: boolean;
};

export type RunState = {
  schema_version: "transcribeit.benchmark-run-state.v1";
  matrix: BenchmarkMatrix;
  matrix_sha256: string;
  producing_commit: { hash: string; worktree_dirty: boolean; worktree_fingerprint_sha256: string };
  started_at_utc: string;
  updated_at_utc: string;
  binary_classification: string;
  binary_sha256: string;
  evaluation_root_sha256: string | null;
  environment: {
    machine: {
      cpu: string;
      logical_cores: number;
      memory_bytes: number | null;
      os: string;
      os_version: string | null;
      os_build: string | null;
      kernel: string;
      architecture: string;
    };
    tools: { rustc: string; ffmpeg: string; swift: string; bun: string };
  };
  keep_attempt_outputs: boolean;
  attempts: AttemptRecord[];
};

export type PublishedResult = {
  schema_version: "transcribeit.benchmark-harness-result.v1";
  matrix_sha256: string;
  recorded_at_utc: string;
  classification: "clean_commit_benchmark" | "dirty_worktree_reference";
  producing_commit: RunState["producing_commit"];
  runtime: { binary_classification: string; binary_sha256: string };
  environment: RunState["environment"];
  matrix: Omit<BenchmarkMatrix, "entries"> & {
    entries: Array<MatrixEntry & { command_template: string }>;
  };
  summary: { attempts: number; passed: number; failed: number; skipped: number };
  attempts: AttemptRecord[];
  cleanup: {
    attempt_outputs_removed: boolean;
    evaluation_root_sha256: string | null;
    downloaded_model: DownloadedModelCleanup | null;
    apple_speech_asset: {
      owner: "macos";
      cleanup_attempted: false;
      lifecycle: "system_managed_shared";
    } | null;
  };
  sanitization: {
    credential_values_removed: true;
    stdout_stderr_removed: true;
    transcript_text_removed: true;
    request_ids_removed: true;
    signed_urls_removed: true;
    local_absolute_paths_removed: true;
    provider_metadata_allowlisted: true;
  };
};
