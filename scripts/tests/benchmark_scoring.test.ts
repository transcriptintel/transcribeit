import {
  scoreReviewedTranscript,
  scoreTranscript,
  type ReviewedScoringReference,
  verifyReviewedReferenceBytes,
} from "../benchmark_harness/scoring";

const timedReference: ReviewedScoringReference = {
  tokens: [
    { text: "alpha", start: 1, end: 1.2 },
    { text: "beta", start: 1.2, end: 1.5 },
  ],
  coverage: {
    domain_terms: ["alpha beta"],
    timestamp_reference: true,
    diarization_reference: true,
    overlapping_speech: false,
  },
};

describe("benchmark reference scoring", () => {
  test("normalizes English and reports WER counts and aggregate domain-term recall", async () => {
    const metrics = await scoreTranscript(
      "speechocean-mandarin-l1-computer-clean",
      "I want to buy an old COMPUTER!",
      {},
    );

    expect(metrics.status).toBe("scored");
    expect(metrics.word_accuracy).toEqual({
      status: "scored",
      reference_words: 7,
      hypothesis_words: 7,
      substitutions: 2,
      deletions: 0,
      insertions: 0,
      wer: 2 / 7,
    });
    expect(metrics.domain_terms).toEqual({
      status: "scored",
      expected_terms: 1,
      matched_terms: 1,
      term_recall: 1,
    });
    expect(metrics.timing).toEqual({ status: "unsupported_reference" });
    expect(metrics.speakers).toEqual({ status: "unsupported_reference" });
    expect(metrics.word_timestamps).toEqual({ status: "unsupported_reference" });
    expect(JSON.stringify(metrics)).not.toContain("computer");
  });

  test("marks provider capabilities unsupported when reviewed timing metadata is absent", () => {
    const metrics = scoreReviewedTranscript(timedReference, "alpha beta", { transcript: { segments: [] } });

    expect(metrics.timing).toEqual({ status: "unsupported_provider" });
    expect(metrics.speakers).toEqual({ status: "unsupported_provider" });
    expect(metrics.word_timestamps).toEqual({ status: "unsupported_provider" });
  });

  test("scores reviewed segment boundaries without claiming word timestamps or speakers", () => {
    const metrics = scoreReviewedTranscript(timedReference, "alpha beta", {
      capabilities: { word_timestamps: false, speaker_labels: false },
      quality: { timing_source: "model_native", timing_reliable: true },
      transcript: {
        segments: [{ text: "alpha beta", start_secs: 1, end_secs: 1.5 }],
      },
    });

    expect(metrics.timing.status).toBe("scored");
    if (metrics.timing.status !== "scored") throw new Error("expected scored timing");
    expect(metrics.timing.scored_segments).toBe(1);
    expect(metrics.timing.start_boundary_mae_ms).toBe(0);
    expect(metrics.timing.end_boundary_mae_ms).toBe(0);
    expect(metrics.timing.timestamp_coverage).toBeGreaterThan(0);
    expect(Number.isFinite(metrics.timing.start_boundary_mae_ms)).toBe(true);
    expect(Number.isFinite(metrics.timing.end_boundary_mae_ms)).toBe(true);
    expect(metrics.timing.timing_origin).toBe("model_native");
    expect(metrics.timing.timing_reliable).toBe(true);
    expect(metrics.speakers).toEqual({ status: "unsupported_provider" });
    expect(metrics.word_timestamps).toEqual({ status: "unsupported_provider" });
  });

  test("returns a transcript-free unavailable aggregate for unknown fixtures", async () => {
    const metrics = await scoreTranscript("fake-fixture", "must not be returned", {});

    expect(metrics.status).toBe("unavailable_reference");
    const statuses = Object.values(metrics).map((value) => (typeof value === "object" ? value.status : value));
    expect(new Set(statuses)).toEqual(new Set(["unavailable_reference"]));
    expect(JSON.stringify(metrics)).not.toContain("must not be returned");
  });

  test("rejects a reviewed-reference hash mismatch without exposing path or text", () => {
    const bytes = new TextEncoder().encode("private reference\n");
    expect(() => verifyReviewedReferenceBytes(bytes, bytes.byteLength, "0".repeat(64))).toThrow(
      "reviewed reference integrity verification failed",
    );
    try {
      verifyReviewedReferenceBytes(bytes, bytes.byteLength, "0".repeat(64));
    } catch (error) {
      expect(String(error)).not.toContain("private reference");
      expect(String(error)).not.toContain("/");
    }
  });

  test("does not serialize overlapping multi-speaker references into WER or timing MAE", () => {
    const overlapping: ReviewedScoringReference = {
      ...timedReference,
      coverage: { ...timedReference.coverage, overlapping_speech: true },
    };
    const metrics = scoreReviewedTranscript(overlapping, "alpha beta", {
      transcript: {
        segments: [{
          text: "alpha beta",
          start_secs: 1,
          end_secs: 1.5,
          speaker: "speaker-1",
          words: [{ start_secs: 1, end_secs: 1.2 }],
        }],
      },
    });

    expect(metrics.word_accuracy).toEqual({ status: "unsupported_overlapping_reference" });
    expect(metrics.timing).toEqual({ status: "unsupported_overlapping_reference" });
    expect(metrics.domain_terms).toMatchObject({ status: "scored", matched_terms: 1 });
    expect(metrics.speakers).toEqual({ status: "available_not_scored" });
    expect(metrics.word_timestamps).toEqual({ status: "available_not_scored" });
  });

  test("blocks serialized WER and timing for overlap even without diarization labels", () => {
    const overlapping: ReviewedScoringReference = {
      ...timedReference,
      coverage: {
        ...timedReference.coverage,
        diarization_reference: false,
        overlapping_speech: true,
      },
    };
    const metrics = scoreReviewedTranscript(overlapping, "alpha beta", {
      transcript: { segments: [{ text: "alpha beta", start_secs: 1, end_secs: 1.5 }] },
    });

    expect(metrics.word_accuracy).toEqual({ status: "unsupported_overlapping_reference" });
    expect(metrics.timing).toEqual({ status: "unsupported_overlapping_reference" });
    expect(metrics.speakers).toEqual({ status: "unsupported_reference" });
  });

  test("does not report timing MAE below fifty-percent timestamp alignment", () => {
    const sparse: ReviewedScoringReference = {
      tokens: [
        { text: "alpha", start: 1, end: 1.2 },
        { text: "beta", start: 1.2, end: 1.5 },
        { text: "gamma", start: 1.5, end: 1.8 },
      ],
      coverage: {
        timestamp_reference: true,
        diarization_reference: false,
        overlapping_speech: false,
      },
    };
    const metrics = scoreReviewedTranscript(sparse, "beta", {
      transcript: { segments: [{ text: "beta", start_secs: 1.2, end_secs: 1.5 }] },
    });

    expect(metrics.timing).toEqual({ status: "insufficient_alignment" });
  });
});
