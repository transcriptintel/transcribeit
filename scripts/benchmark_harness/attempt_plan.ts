import type { BenchmarkMatrix } from "./types";

export type ExpectedAttemptIdentity = {
  key: string;
  entry_id: string;
  provider: string;
  model: string;
  execution: string;
  cache_state: string;
  fixture_id: string;
  repetition: number;
};

export function expectedAttemptIdentities(matrix: BenchmarkMatrix): ExpectedAttemptIdentity[] {
  if (!Array.isArray(matrix.entries) || !matrix.entries.length || !Array.isArray(matrix.fixture_ids)) {
    throw new Error("matrix must define entries and fixtures before attempts can be planned");
  }
  for (const entry of matrix.entries) {
    if (!Number.isInteger(entry.repetitions) || entry.repetitions < 1) {
      throw new Error(`matrix entry ${entry.id} has invalid repetitions`);
    }
  }
  const fixtures = [...matrix.fixture_ids];
  for (const entry of matrix.entries) {
    for (const fixtureId of entry.fixture_ids ?? []) if (!fixtures.includes(fixtureId)) fixtures.push(fixtureId);
  }
  for (const fixtureId of fixtures) {
    if (!matrix.entries.some((entry) => (entry.fixture_ids ?? matrix.fixture_ids).includes(fixtureId))) {
      throw new Error(`matrix fixture ${fixtureId} has no eligible entry`);
    }
  }
  const identity = (entry: BenchmarkMatrix["entries"][number], fixtureId: string, repetition: number): ExpectedAttemptIdentity => ({
    key: `${entry.id}--${fixtureId}--${String(repetition).padStart(3, "0")}`,
    entry_id: entry.id,
    provider: entry.provider,
    model: entry.model,
    execution: entry.execution,
    cache_state: entry.cache_state,
    fixture_id: fixtureId,
    repetition,
  });
  if (matrix.attempt_order === "fixture-repetition-entry") {
    const identities: ExpectedAttemptIdentity[] = [];
    for (const fixtureId of fixtures) {
      const entries = matrix.entries.filter((entry) => (entry.fixture_ids ?? matrix.fixture_ids).includes(fixtureId));
      const repetitions = Math.max(...entries.map((entry) => entry.repetitions));
      for (let repetition = 1; repetition <= repetitions; repetition++) {
        for (const entry of entries) if (repetition <= entry.repetitions) identities.push(identity(entry, fixtureId, repetition));
      }
    }
    return identities;
  }
  const identities: ExpectedAttemptIdentity[] = [];
  for (const entry of matrix.entries) {
    for (const fixtureId of entry.fixture_ids ?? matrix.fixture_ids) {
      for (let repetition = 1; repetition <= entry.repetitions; repetition++) {
        identities.push(identity(entry, fixtureId, repetition));
      }
    }
  }
  return identities;
}
