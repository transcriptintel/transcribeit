# TranscribeIt issue registry

This is the canonical tracked registry for TranscribeIt engineering findings and
planned implementation work. Each `TI-NNN` page owns its status, priority,
acceptance criteria, outcome, and validation evidence. Regenerate this index with
`bun run scripts/issues_registry.ts generate` and verify it with
`bun run scripts/issues_registry.ts check`.

- Total findings: 10
- Active findings: 3

| ID | Priority | Status | Area | Summary |
|---|---:|---|---|---|
| [TI-001](./TI-001.md) | P1 | Resolved | Security and correctness | Deep-review P1 correctness and security remediation |
| [TI-002](./TI-002.md) | P2 | Resolved | Provider reliability | P2 provider hardening, portability, and benchmark reproducibility |
| [TI-003](./TI-003.md) | P2 | Resolved | Architecture and governance | Output and command orchestration exceed maintainable boundaries |
| [TI-004](./TI-004.md) | P2 | Resolved | Local inference | Evaluate a native Rust ONNX path for Qwen3-ASR |
| [TI-005](./TI-005.md) | P3 | In progress | Local inference | Decide whether to package llama.cpp Qwen3-ASR support |
| [TI-006](./TI-006.md) | P2 | Resolved | Benchmark quality | Build a representative multi-fixture transcription corpus |
| [TI-007](./TI-007.md) | P2 | Resolved | Benchmark reproducibility | Rebaseline the hosted and local provider comparison |
| [TI-008](./TI-008.md) | P3 | Open | Benchmark tooling | Add an automation-friendly benchmark harness |
| [TI-009](./TI-009.md) | P2 | Open | Provider operations | Run and record the live provider smoke matrix |
| [TI-010](./TI-010.md) | P2 | Resolved | Local inference | Retire the unused Sherpa-ONNX integration |

## Working agreement

1. Select the highest-priority applicable issue and set its frontmatter status to
   `in-progress` before implementation.
2. Confirm the live code still supports the finding and keep the required outcome
   and acceptance criteria current.
3. Align implementation, regression coverage, public documentation, examples,
   benchmark evidence, and provider/runtime boundaries.
4. Set an issue to `resolved` only after its required gates pass. Add the ISO
   resolution date plus concrete outcome and validation evidence.
5. Run `bun run scripts/issues_registry.ts generate` and then the matching
   `check`; never hand-edit this generated index.
