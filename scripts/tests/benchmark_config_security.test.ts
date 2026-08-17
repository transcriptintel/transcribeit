import { loadMatrix, validateMatrix } from "../benchmark_harness/config";
import type { BenchmarkMatrix } from "../benchmark_harness/types";
import type { CorpusManifest } from "../corpus";

describe("benchmark matrix security", () => {
  test("keeps both tracked matrices inside the reviewed argument grammar", async () => {
    const hosted = await loadMatrix("benchmarks/matrices/hosted-smoke.yaml");
    const local = await loadMatrix("benchmarks/matrices/ti-014-apple-large-v3.yaml");
    expect(hosted.matrix.entries.find((entry) => entry.provider === "gemini")?.args).toEqual(["--gemini-autoclean"]);
    expect(local.matrix.entries.every((entry) => entry.args[0] === "--language")).toBe(true);
  });

  test("rejects unknown, orphaned, free-text, control, and credential-like arguments", () => {
    const cases: Array<{ args: string[]; message: string }> = [
      { args: ["free text"], message: "unsupported reviewed option" },
      { args: ["--unknown"], message: "unsupported reviewed option" },
      { args: ["--language", "../../secret"], message: "safe locale" },
      { args: ["--language", "en-US\n--api-key"], message: "safe locale" },
      { args: ["--api-key=secret"], message: "credential" },
      { args: ["--language"], message: "safe locale" },
      { args: ["--gemini-autoclean=true"], message: "unsupported reviewed option" },
    ];
    for (const candidate of cases) {
      const matrix = baseMatrix();
      matrix.entries[0].args = candidate.args;
      const errors = validateMatrix(matrix, corpus()).errors;
      expect(errors.some((error) => error.includes(candidate.message))).toBe(true);
    }
  });

  test("requires a safe pinned revision and explicit evaluation cache environment", () => {
    const matrix = baseMatrix();
    matrix.entries[0].artifact = {
      lifecycle: "evaluation_download",
      revision: "../unsafe revision",
      sha256: "a".repeat(64),
      bytes: 1,
    };
    let errors = validateMatrix(matrix, corpus()).errors;
    expect(errors.some((error) => error.includes("safe pinned identifier"))).toBe(true);
    expect(errors.some((error) => error.includes("MODEL_CACHE_DIR"))).toBe(true);

    matrix.entries[0].artifact.revision = "5359861c739e955e79d9a303bcbc70fb988958b1";
    matrix.entries[0].required_env = ["MODEL_CACHE_DIR"];
    errors = validateMatrix(matrix, corpus()).errors;
    expect(errors).toEqual([]);
  });
});

function baseMatrix(): BenchmarkMatrix {
  return {
    schema_version: 1,
    matrix_id: "config-security",
    description: "Config security test",
    execution_policy: "local_ci",
    attempt_order: "entry-fixture-repetition",
    concurrency: 1,
    retries: 0,
    timeout_seconds: 10,
    fixture_ids: ["fixture"],
    entries: [
      {
        id: "entry",
        provider: "local",
        model: "test",
        execution: "local",
        cache_state: "cold",
        repetitions: 1,
        required_env: [],
        args: ["--language", "en-US"],
      },
    ],
    measurements: { reference_scoring: false, peak_rss: false },
    tolerances: {
      enforcement: "fail",
      max_relative_latency_regression_percent: 25,
      max_absolute_latency_regression_ms: 100,
      min_success_rate: 1,
    },
  };
}

function corpus(): CorpusManifest {
  return {
    schema_version: 1,
    corpus_id: "config-security",
    materialization_root: "samples/corpus/v1",
    sources: {},
    fixtures: [
      {
        id: "fixture",
        audio: {
          path: "samples/corpus/v1/fixture.wav",
          sha256: "b".repeat(64),
          bytes: 1,
          duration_seconds: 1,
          container: "wav",
          codec: "pcm",
          sample_rate_hz: 16_000,
          channels: 1,
          materialize: {},
        },
        reference: {},
      },
    ],
  };
}
