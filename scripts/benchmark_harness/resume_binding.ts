import type { RepositoryProvenance } from "./provenance";
import type { AttemptRecord, RunState } from "./types";

export function reconcileAttemptPlan(stored: AttemptRecord[], expected: AttemptRecord[]): void {
  if (stored.length !== expected.length) throw new Error("existing run state has a different attempt plan length");
  for (let index = 0; index < expected.length; index++) {
    if (JSON.stringify(attemptIdentity(stored[index])) !== JSON.stringify(attemptIdentity(expected[index]))) {
      throw new Error(`existing run state attempt ${index} does not match the current plan`);
    }
  }
}

export function sameProvenance(left: RepositoryProvenance, right: RepositoryProvenance): boolean {
  return left.hash === right.hash &&
    left.worktree_dirty === right.worktree_dirty &&
    left.worktree_fingerprint_sha256 === right.worktree_fingerprint_sha256;
}

export function sameEnvironment(left: RunState["environment"], right: RunState["environment"]): boolean {
  return JSON.stringify(left) === JSON.stringify(right);
}

function attemptIdentity(attempt: AttemptRecord): object {
  return {
    key: attempt.key,
    entry_id: attempt.entry_id,
    provider: attempt.provider,
    model: attempt.model,
    execution: attempt.execution,
    cache_state: attempt.cache_state,
    fixture: attempt.fixture,
    repetition: attempt.repetition,
  };
}
