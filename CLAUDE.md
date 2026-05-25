# Project instructions for AI coding agents

This file applies to any AI coding agent (Claude, Cursor, Windsurf,
etc.) working in this repository. Read it once per session and treat
it as a hard rule when its content is relevant.

## Optimization exploration log: `documentation/f2x-sha-todo.md`

The F_2 SHA-256 prover path (`protocol/src/f2_prove.rs`,
`test-uair/src/sha256_f2.rs`, the Metal-Blake3 commit pipeline,
and surrounding helpers) is under active iterative optimization.
[`documentation/f2x-sha-todo.md`](documentation/f2x-sha-todo.md)
is the canonical ledger of what's been tried, what shipped, what
was investigated and rejected, and what remains open.

**Rule:** any time you investigate, design, or even just consider
an optimization, refactor, or design alternative on the F_2 SHA-256
prover path — and decide *not* to implement it (or implement only
part of it) — **add or update an entry in
`documentation/f2x-sha-todo.md` before ending your turn**.

This rule applies to all four of:
- **Shipping** a change → add a "Shipped work" entry with the
  commit SHA, the rationale, and the measured impact.
- **Trying** something that didn't work → add to
  "Investigated, didn't help" with the hypothesis, the test, the
  result, and (if known) the real culprit.
- **Spotting** a known optimization you didn't pursue → add to
  "Identified but not implemented" with where the opportunity
  lives, the proposed approach, and the expected gain.
- **Forming** a hypothesis you didn't have time to test → add to
  "Open questions" with a concrete test plan.

The point of the doc is that the *next* agent (or the same agent
in three weeks) doesn't repeat dead ends or re-discover the same
hot paths. Skipping the doc update wastes future-you's time.

Format guidance lives in the "How to use this doc" section at the
bottom of `documentation/f2x-sha-todo.md`. Match the existing
entries' style (lead with the *what*, then *why*, then *result*
or *expected impact*).

When you update the doc, do it as a real edit (not a vague
mention in a commit message). One concrete paragraph beats a
hand-wavy reference every time.

## Scope of this rule

- **In scope:** anything touching `protocol/src/f2_prove.rs`,
  `protocol/src/lib.rs` (in F_2-relevant helpers like
  `absorb_public_columns`), `protocol/src/f2_native_ic.rs`,
  `protocol/benches/f2_sha256.rs`, `test-uair/src/sha256_f2.rs`,
  `zip-plus/src/metal_gpu/`, `zip-plus/src/pcs/phase_commit.rs`,
  `zip-plus/src/merkle.rs`, and the related plan / doc files
  under `protocol/src/f2_*_plan.md` and `documentation/sha-f2x-doc/`.
- **Out of scope:** unrelated paths (integer prover, ECDSA UAIR,
  general piop / poly utilities not specific to F_2). For those,
  use ordinary commit messages and code comments — no need to
  touch the F_2 todo doc.

If you're unsure whether your work is in scope, lean toward adding
an entry. A redundant entry is cheap; a missing one costs time
later.
