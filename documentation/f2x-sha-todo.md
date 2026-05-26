# SHA-256 F_2 prover — TODO and exploration log

A running ledger of optimization work on the SHA-256 F_2 prover path
(`protocol/src/f2_prove.rs`, `test-uair/src/sha256_f2.rs`, and the
Metal-Blake3 commit pipeline). Each entry captures what was tried,
the measurement that motivated or refuted it, and whether it shipped.

**Keep this current** — any time you investigate an optimization or
design idea and decide *not* to implement it, add an entry here (or
amend an existing one) so the next person doesn't redo the
investigation from scratch. See the "How to use this doc" section
at the bottom.

---

## Fast-described sketch todo's

- [ ] Review how we do PCS open, with lifts and so on. Suspect unnecessary lifting etc
- [ ] GF(128)
- [x] Are we doing multipoint eval? Now yes — see "Multipoint-eval
      phase wired into the F_2 prove path" under Shipped work.
- [ ] SHIFT vector proving — **row-level** shift virtualization (needed
      for the K-discharge work, Issue 1). The per-cell SHIFTR case
      already ships via `F2BitOpVirtualSpec`; see the "Per-cell SHIFTR
      virtualisation in-tree canary" entry under Shipped work below.
- [ ] Review approach to add mod 2^32 in documentation
- [ ] Hadamard product implementation


---

## Branch: Group 8 fully removed (this branch)

This branch (`f2-clean`, cherry-picked from `main-beta`) **does not
include** the Group 8 commits originally on `claude/gkr-virtual-cols`:
the (X^32) ideal swap for additions, the combined CSA-majority +
Binius-carry K column, and the `F2KVirtualSpec`-driven K-col
exclusion from commit + γ-batched open. Test-uair's witness generator
and `Sha256F2Ideal` are at the pre-Group-8 shape (27 binary cols,
`(X^32 - 1)` for additions, no `W_K_*` cols).

The protocol code (`protocol/src/f2_prove.rs`) and the bench
(`protocol/benches/f2_sha256.rs`) have been scrubbed of all K-virt
machinery: no `F2KVirtualSpec` / `F2KSource` struct definitions, no
`k_specs` parameter on any function, no K-tail layout assertions, no
K-aware witness slicing, no `sha_f2_k_virtuals()` helper or call
sites. The intermediate "stub the helper to neutralize the plumbing"
state (commit 7a01531) is now superseded by the deep removal in
commit 57af4e0.

Consequences:
- Issue 1 (trusted K-virtual MLE eval discharge) is structurally
  absent on this branch — there is no K machinery, sound or
  otherwise.
- The "exclude 7 combined-K cols from commit + γ-batched open"
  optimisation is gone (the witness has no K cols to exclude).
- Re-introducing K-virtuals on this branch would require restoring
  the struct definitions and the plumbing through the prove/commit/
  open/verify entry points — not a one-line flag flip.

Entries below referring to "K cols" / "combined-K" / "F2KVirtualSpec"
describe machinery that ran on `claude/gkr-virtual-cols` but is
**not present** on this branch.

---

## Shipped work (chronological, most recent first)

### Parallelise SHA F_2 witness gen post-loop builders (commit `<TBD>`)
- **What**: in `test-uair/src/sha256_f2.rs::generate_random_trace`,
  switch the per-row builder loops that run *after* the
  sequential compression rounds to `cfg_iter!` / `cfg_into_iter!`
  with a `PAR_MIN_LEN = 4096`-row split threshold:
  - 6 Σ / σ / SHR derived columns
    (`sigma0_vals`, `sigma1_vals`, `sig0_vals`, `sig1_vals`,
    `shr3_w_vals`, `shr10_w_vals`)
  - 3 Ch / Maj operands (`u_ef_vals`, `u_neg_e_g_vals`, `maj_vals`)
  - 1 packed-`PA_C` compensator builder (the largest single
    loop: 13 per-step κ-residue extractions per row across all
    `2^num_vars` rows)
  Plumbing: new `parallel` feature on `zinc-test-uair`
  (`rayon` optional dep + `zinc-utils/parallel` propagation);
  `zinc-protocol`'s `parallel` feature now also enables
  `zinc-test-uair/parallel`. The 4096-row min_len keeps small
  fixtures (`nvars = 9`, `n = 512`) on the single-thread path so
  rayon's task-spawn overhead doesn't dominate.
- **Why**: the step-2+3 chained-Binius rewrite traded the prior
  CSA-flattened impl's bounded-active-row CSA work for an
  unconditional `O(n)` packed-`PA_C` builder. At `nvars = 22` this
  alone caused a ~170 ms (+170%) regression in `WitnessGen`, the
  only stage that got slower in step 2+3. The post-loop builders
  are embarrassingly parallel — each row is independent — so the
  fix is mechanical: `cfg_into_iter!(0..n, MIN_LEN)`.
- **Measured impact** (Apple M-series, `--features
  parallel,simd,unchecked`, vs the `pre-step23` baseline saved
  before any of the alignment work):
  | nvars | Stage      | Pre step-2+3 | Post step-2+3 + parallel | Δ      |
  |-------|------------|--------------|--------------------------|--------|
  | 9     | WitnessGen | 76.2 µs      | 12.1 µs                  | −84%   |
  | 9     | Prove      | 2.83 ms      | 2.71 ms                  | −5%    |
  | 9     | Verify     | 2.58 ms      | 1.89 ms                  | −26%   |
  | 20    | WitnessGen | 25.7 ms      | 12.8 ms                  | −50%   |
  | 20    | Prove      | 888 ms       | 543 ms                   | −39%   |
  | 20    | Verify     | 148 ms       | 112 ms                   | −24%   |
  | 22    | WitnessGen | 102.7 ms     | 53.3 ms                  | −49%   |
  | 22    | Prove      | 6.96 s       | 4.09 s                   | −41%   |
  | 22    | Verify     | 585 ms       | 453 ms                   | −22%   |

  Net wall clock at `nvars = 22`: 7.65 s → 4.60 s, **−40%**.
- **Verification**: all 12 `f2_prove::tests` pass.
- **Out of scope (not pursued)**: the sequential compression
  loop (`for j in 16..rpc { ... }` plus the round-update body)
  is *not* parallelised — each iteration depends on prior rows'
  outputs (`w_vals[t - 16]`, `a_vals[k + 3]`, etc.), so the
  inter-row dependency chain rules out a straight `par_iter`.
  A per-compression `cfg_join!` across the 7 independent
  compressions would help (each compression block of 68 rows is
  independent up to the H_0 input), but at `nvars = 22` the
  sequential compression work is already dwarfed by the
  post-loop O(n) builders, so the payoff is small.

### Chained-Binius rewrite of C5–C11 with packed `PA_C` (steps 2+3 of 5, commit `<TBD>`)
- **What** — full structural alignment of the SHA-256 F_2 UAIR with the
  `sha-f2x-doc` spec at the AIR-shape level, modulo the still-deferred
  Hadamard discharge (step 5):
  1. `protocol/src/f2_native_ic.rs` — replaced the zero-stub at
     `prove_linear` and `prove_combined` (lines that used to push
     `DynamicPolynomialF::ZERO` / `F2RowExpr::zero()` for each bit-op
     spec) with proper per-cell `BitOp::Rot(c)` / `BitOp::ShiftR(c)`
     materialisation. For `prove_linear` (the MLE-first path the SHA
     F_2 UAIR takes), bit-op virtuals' MLE eval coefficients are a
     re-indexing of the source column's `up_evals` coefficients —
     `O(D)` per spec, no extra per-cell loop. For `prove_combined`,
     per-row bits flow through `v.rotate_left(c)` / `v >> c`. Unlocks
     UAIR-internal `BitOpSpec` end-to-end for any F_2 UAIR (the
     verifier-side machinery in `verifier.rs` / `prover.rs` was
     already in place; only the F_2-native IC fast path lagged).
  2. `test-uair/src/sha256_f2.rs` — full rewrite of C5–C11:
     - **Columns**: dropped the 6 CSA-majority columns (`W_M_W1`,
       `W_M_W2`, `W_M_T1_{1..4}`), the 7 per-add full-carry columns
       (`W_C_{W,T1,T2,A,E,FF_A,FF_E}`), and the 7 scalar compensator
       columns (`PA_C_{W,T1,T2,A,E,FF_A,FF_E}`). Added 6
       chained-Binius intermediate-sum columns (`W_W_S{1,2}`,
       `W_T1_S{1..4}`) per doc §3.7 Definition 1 and 1 packed
       compensator column `PA_C` (bits 0..12 hold the 13 per-step κ
       compensators per the doc's `binius-packing` allocation; bits
       13..31 are zero). Total column delta: 14 public + 27
       committed = 41 → 8 public + 20 committed = 28 (−13).
     - **Constraints**: replaced the 7 polynomial CSA-flattened
       `LinSum` checks with 13 per-step `BiniusAdd` LSB scalar
       constraints (doc §3.6 `BiniusAdd`): each is
       `(target + x + y + κ_k) ∈ (X)`, where κ_k is bit j_k of
       `PA_C`. Bit extraction via 12 internal `BitOpSpec`s
       (`SHR^j(PA_C)` for j ∈ {1..12}; κ_0 uses `PA_C` directly).
     - **Ideal**: added `Sha256F2Ideal::LsbX` for the principal
       ideal `(X)` (coefficient at position 0 = 0, positions 1..31
       unconstrained). `RotXw1` keeps C1–C4; `ShiftX32` from step 1
       is dormant but kept as a variant for the eventual Hadamard
       wiring.
     - **Witness gen**: dropped `binius_carry` / `csa_step`
       helpers; replaced CSA layers with chained `wrapping_add`
       calls materialising the partial sums. `PA_C` builder packs
       13 per-step LSB residues (= 0 on rows where the witness is
       consistent with the binary add, residue otherwise).
- **Why** — direct shape parity with the doc's column and constraint
  layout, with the per-step structure that step 5 (external Hadamard
  discharge) can plug into without further rewrites. The CSA-flattened
  LinSum was a closed-form trick that didn't generalise.
- **Status vs. soundness** — step 2+3 swaps one AIR-level-incomplete
  scheme for another; both rely on honest witnesses for bits 1..31
  of each addition. The 13 per-step LSB checks pin only bit 0; bits
  1..31 are pinned by the column-level Hadamard
  `(x + X·c) ⊙ (y + X·c) = c + X·c` in `F_shift`, which is still
  external/deferred per doc §3.4. AND-side Hadamards (C12–C14) are
  also deferred. **No new soundness regression**: the prior CSA-
  flattened LinSum was equally AIR-level-incomplete (the CSA
  majority columns and final carries were under-constrained by the
  algebraic identity).
- **Verification** — `cargo test -p zinc-protocol --features parallel
  --lib f2_prove::tests` (12 tests, all pass, incl.
  `prove_then_verify_sha256_f2_roundtrips`); `cargo test -p
  zinc-test-uair --lib sha256_f2` (4 tests, all pass). Benches build
  clean.
- **Out of scope (steps 4+5 still open)**:
  - Step 4 (the LSB check unfold of `BiniusAdd`) — *folded into this
    change*, since the AIR-level half of `BiniusAdd` is exactly the
    LSB check; the Hadamard half is step 5.
  - Step 5 — wire up the external Hadamard-product discharge
    (`(x + X·c) ⊙ (y + X·c) = c + X·c` in `F_shift`) for the 13
    addition-side and 3 AND-side relations. Will need:
    - the `W_β` packed carry-out column (deferred along with step 5
      — committing it without consumers wastes commit cost),
    - virtual carry word `c` reconstruction (bits 0..30 from
      `SHR^1` of XOR-of-(target,x,y); bit 31 = β_k from `W_β`),
    - a Hadamard-product PIOP. Mechanism is a separate design pass.
- **Caveat carried over** — `cargo build -p zinc-protocol` *without*
  `--features parallel` still fails in `f2_prove.rs:603` from
  `cfg_iter!` shape, pre-existing.

### C5–C11 modular-addition constraints moved into `F_shift = F_2[X]/(X^32)` (commit `<TBD>`)
- **What**: `test-uair/src/sha256_f2.rs`. Step 1 of aligning the F_2
  SHA-256 UAIR with `documentation/sha-f2x-doc/`.
  1. `Sha256F2Ideal` lifted from a unit struct to a payload-free
     enum with two variants: `RotXw1` (`(X^32 − 1)` — `F_rot`) and
     `ShiftX32` (`(X^32)` — `F_shift`). `IdealCheck::contains`
     dispatches on the variant; `ShiftX32` membership is "coefficients
     at positions 0..32 are all zero," matching the principal
     ideal `(X^32)`.
  2. The constraint impl now uses two ideal handles: `ideal_rot` for
     the rotation identities (C1–C4) and `ideal_shift` for the
     modular-addition LinSum constraints (C5–C11). C12–C18 are
     unaffected (boundary pins use `assert_zero`; AND Hadamards are
     external).
  3. The witness generator's `PA_C_*` compensator builders now use
     non-cyclic `(x << 1)` (`shl1`) instead of `rotl1` for the
     `X·m_j` / `X·c_*` contributions. Under the `F_shift` convention,
     `X·c` drops bit 31 of `c` via the quotient — the compensator
     residue must match. The constraint expression `mbs(w_c_*,
     x_scalar)` is unchanged: in `F_shift` the resulting polynomial's
     bit-32 coefficient is unconstrained, so the `IdealCheck` only
     pins bits 0..31.
- **Why**: the spec's Lemma `binius-vc` lives in `F_shift` for good
  reason — the `(X^32)` quotient kills the bit-31 carry-out
  automatically, so the per-step Binius Hadamard `(a + X·c) ⊙ (b +
  X·c) = c + X·c` is a *ring* equation in `F_shift`, not a relation
  that has to know about cyclic wrap. Under the prior `RotXw1`
  framing, bit-31 of each `m_j`/`c_*` wrapped to bit 0 via cyclic
  rotation, which (a) made `PA_C_*` non-zero on honest active rows
  whenever the final carry-out was 1, and (b) coupled bit-32 effects
  back into the LSB check — a structural divergence from the doc that
  this swap eliminates. Under `ShiftX32`, `PA_C_*` is zero on every
  active row as the doc specifies (the modular-add identity is exact
  in `F_shift`).
- **Verification**: all 12 tests in
  `cargo test -p zinc-protocol --features parallel --lib
  f2_prove::tests` pass, including
  `prove_then_verify_sha256_f2_roundtrips`. The four
  `zinc-test-uair` `sha256_f2` shape-level lib tests pass.
- **Status vs the broader spec alignment plan**: this is step 1 of 5
  from the gap analysis in conversation. Done: ideal split + matching
  witness compensator. Remaining:
  - **(2)** Rip out the CSA-flattened linear identity in favour of
    chained-Binius per-step `BiniusAdd` constraints. Replace the 6
    `W_M_*` and 7 `W_C_*` columns with the doc's 6 chained-Binius
    intermediate-sum columns (`W_W^{(1,2)}`, `W_{T_1}^{(1..4)}`) +
    1 packed `W_β` carry-out column.
  - **(3)** Collapse the 7 scalar `PA_C_*` columns into one packed
    bit-poly `PA_C` with the doc's bit allocation.
  - **(4)** Rewrite C5–C11 as 13 per-step `BiniusAdd` constraints
    (each = one LSB equation + one column-level Hadamard relation).
  - **(5)** Register the 13 addition-side Hadamards + 3 AND-side
    Hadamards with whatever Hadamard-product mechanism we adopt.
- **Caveat — pre-existing build state**: `cargo build -p zinc-protocol`
  *without* `--features parallel` does not compile on this branch
  (errors in `f2_prove.rs:603` from `cfg_iter!` expanding to
  `Iterator::reduce` with a rayon-shaped call). This is independent
  of the ideal swap and was on the baseline tree; flagged so the
  next agent doesn't blame this change.

### Multipoint-eval phase wired into the F_2 prove path (commit `<TBD>`)
- **What**: added a `MultipointEval<BinaryFieldGF192>` step between
  the F_2 sumcheck and the PCS open. The phase reduces the per-column
  MLE evaluation claims at the sumcheck point `r*` (carried in
  `F2Proof.column_evals_at_rstar`) to a single set of MLE eval
  claims at a fresh point `r_0`, batched across columns. The
  F_2 PCS open now happens at `r_0` instead of `r*`.
  Proof shape: `F2FullProof` gains `multipoint_eval:
  MultipointEvalProof<BinaryFieldGF192>` + `open_evals_at_r_0:
  Vec<BinaryFieldGF192>` (length = num primary + num virtual cols).
- **Why**: the multipoint-eval primitive is the standard mechanism
  for reducing row-shifted MLE eval claims to non-shifted ones at
  a related point — the prerequisite for sound row-shift discharge
  in the K-virtual cols (Issue 1 below) and for the row-shifted
  bit-op virtuals that `f2_native_ic.rs:735-740` currently punts on.
  The integer protocol already runs this (Step 5 in
  `protocol/src/prover.rs`); the F_2 path was the laggard.
- **Today this is degenerate overhead.** The F_2 sumcheck doesn't
  surface any row-shifted column evals yet (no Hadamard product
  support — see the inline comment in `f2_native_ic.rs:735-740`),
  so multipoint-eval is called with empty `down_evals` + `shifts`.
  It just rerandomises the open point and γ-batches up_evals. The
  payoff lands when Hadamard / K-discharge wires shifted-col evals
  into the sumcheck output — those flow through
  `MultipointEval::prove_as_subprotocol`'s `shifts` + `down_evals`
  params without further protocol changes.
- **Perf cost** (estimated, not measured): one extra α-projection
  per prove (~500 ms at nvars=22) because the recompute pattern
  duplicates work already done inside
  `prove_f2_uair_with_groups` (where the original `projected_trace`
  is consumed by the column-evals-at-r* materialisation). Plus one
  additional MLE-eval pass per col at `r_0` (~few hundred ms at
  nvars=22), one degree-2 sumcheck of `num_vars` rounds, and
  `num_total_cols × 24 B` extra proof bytes for `open_evals_at_r_0`.
  Bench will show the regression.
- **Followups**:
  - ~~Refactor to avoid the α-projection recompute~~ **shipped** —
    `prove_f2_uair_with_groups` now returns `projected_trace` as a
    third element; the multipoint-eval phase consumes it directly.
    The savings at small nvars are within bench noise (the inline
    α-recompute was cheaper than my initial estimate), but the
    refactor's cost (one per-col `Vec` clone in
    `column_evals_at_rstar`) scales with `cells × bytes` while the
    saved recompute scales with `cells × ops`, so the win grows
    with `2^N`. See subsequent commit.
  - **Tamper test for the new path**: existing
    `verify_f2_full_rejects_tampered_*` tests cover the sumcheck +
    open paths; add an analogue that tampers with
    `multipoint_eval` or `open_evals_at_r_0` and confirms the
    correct error variant fires.

### Per-cell SHIFTR virtualisation: in-tree canary (commit `<TBD>`)
- **What**: extended `prove_then_verify_sha256_f2_with_k_virtuals_roundtrips`
  in `protocol/src/f2_prove.rs` to also pass `F2BitOpVirtualSpec`s for
  `W_SHR3_W = SHR^3(W_W)` and `W_SHR10_W = SHR^10(W_W)`. Adjusted the
  expected `batch_size` to account for the two extra excluded cols.
- **Why**: the bench (`protocol/benches/f2_sha256.rs`) already declares
  these specs via `sha_f2_bit_op_virtuals()` and threads them through
  every prove/verify callsite (shipped in commit `fdd2d01`), so the
  commit-side savings are already realised at bench time. But no CI-
  runnable test exercised the full prove-then-verify roundtrip with
  non-empty `bit_op_specs` on the real SHA-256 F_2 UAIR — the two
  in-tree tests (`prove_then_verify_sha256_f2_roundtrips` and
  `..._with_k_virtuals_roundtrips`) both passed empty `bit_op_specs`.
  This entry closes that gap so a regression in the per-cell SHIFTR
  reconstruction (`apply_bit_op_u32` at f2_prove.rs:300 or the
  `verify_f2_open_with_virtuals` derivation loop at f2_prove.rs:2575-2700)
  fails a test, not just a bench.
- **Where**: `protocol/src/f2_prove.rs` (test `prove_then_verify_sha256_f2_with_k_virtuals_roundtrips`,
  lines ~4797-4836 after the edit).
- **Related future work — trace-builder JIT (not in this entry)**:
  `Sha256F2Uair::generate_random_trace` still materialises
  `W_SHR3_W` and `W_SHR10_W` row-by-row (lines 836-837 + the MLE push
  at 1052-1053), even though they're now excluded from the commit.
  Cutting that work needs the prover to JIT-materialise bit-op virtual
  MLEs inside `prove_f2_full_with_bit_ops` before
  `prove_f2_uair_with_groups` consumes the trace — currently the
  caller is expected to pre-materialise them at the declared
  `col_idx`. Estimated savings at nvars=22: ~30 MB allocation +
  ~5 ms CPU per prove (2 cols out of 41). Symmetric with the K-cols
  pattern, which also pre-materialises and accepts the cost.

### Reuse the 768 MB commit slab process-wide (commit `<TBD>`)
- **What**: cached the `commit_grouped` GPU-inline slab in a
  process-wide `static OnceLock<Mutex<Vec<u8>>>` in
  `zip-plus/src/pcs/phase_commit.rs`. The Mutex is held for the
  full GPU-branch duration. Slab grows on demand, never shrinks
  — for SHA-256 F_2 the size is constant per protocol, so the
  resize is a one-time first-call cost. No zero-init needed
  between commits since `scatter_matrix_into_gpu_slab` fully
  overwrites the active region.
- **Why**: each commit was allocating a fresh 768 MB
  `Vec<u8>` then writing through it (cold pages, cold cache).
  Caching reuses the same DRAM pages — they stay resident across
  commits and the second touch hits warm cache.
- **Result at nvars=22**:
  `Commit-AfterPrevProve` 772 ms → 566 ms (−206 ms, −27%).
  `Prove e2e` 2.17 s → **1.53 s** (−640 ms, −29.5%).
  Cumulative since CPU baseline: 2.96 s → 1.53 s (−48%).
  The expected-gain estimate (100–200 ms) substantially under-
  shot the actual gain — the cold-DRAM penalty on a fresh
  768 MB write-through was far larger than the alloc cost alone.

### Stream public-col absorb per-column (commit `c5b1c64`)
- **What**: replaced the 235 MB single-buffer absorb in
  `protocol/src/lib.rs::absorb_public_columns` with a per-col
  reusable 17 MB scratch. Blake3 `update` is associative, so the
  post-absorb transcript state is byte-identical.
- **Why**: the 235 MB alloc + cold-cache Blake3 hash cost ~150 ms
  inside the e2e prove loop (where UAIR + Open had just evicted L3),
  vs ~40 ms in a tight micro-bench loop.
- **Result at nvars=22**:
  `Commit-AfterPrevProve` 900 ms → 772 ms (−128 ms, −14%).
  `Prove e2e` 2.34 s → 2.17 s (−170 ms, −7%).

### Drop UAIR clones in `prove_f2_uair_with_groups` (commit `22f618a`)
- **What**: (a) skip `trace.binary_poly.iter().cloned().collect()`
  when `virtual_specs.is_empty()` via a `Cow<'_, [..]>`; (b) move
  `eq_table` and `weighted_col` into `F2EqColRound1FastPath` instead
  of cloning.
- **Why**: trace clone was ~560 MB / ~110 ms at nvars=22; eq+wcol
  clone was ~192 MB / ~75 ms. Both invisible to the UAIR-a..d
  kernel sub-benches because they happened in the "wrap-up" between
  sub-steps.
- **Result at nvars=22**: `UAIR-FULL` 1.22 s → 0.96 s (−260 ms,
  −21%). Surfaced via new `UAIR-FULL` micro bench.

### Pre-paired commit + prove entry points (commit `744f0bd`)
- **What**: new `commit_pre_paired_witness`,
  `commit_and_absorb_pre_paired_witness`,
  `prove_f2_full_pre_paired_with_bit_ops` that take a pre-paired
  witness slice. `ProverFixture` pre-pairs once in `setup_prover`.
- **Why**: pairing + spec translation + MLE clone shouldn't be on
  the per-iter hot path; conceptually it's witness gen.
- **Result at nvars=22**: `Commit-micro` 790 ms → 561 ms (−229 ms,
  −29%). e2e Prove −130 ms.

### Blake3 transcript: `update_rayon` for big absorbs (commit `744f0bd`)
- **What**: `Blake3Transcript::absorb_inner` routes inputs ≥256 KB
  through `Hasher::update_rayon`. Enabled via
  `features = ["rayon"]` on workspace `blake3` dep.
- **Why**: the 235 MB public-col absorb (now ~17 MB per-col after
  the streaming rewrite, still well above the threshold) parallelises
  4-8× on M-series.
- **Result**: rolled in with the pre-paired commit; pre/post can't
  be cleanly isolated. Estimated savings ~100 ms.

### Large-leaf Metal Blake3 kernel + GPU-inline-fused commit (commit `9b8f178`)
- **What**: extended `hash_kernels.metal` from bounded 64-chunk
  multi-chunk path to a streaming subtree-stack kernel (handles up
  to 16 GB / leaf). Lifted the matching 64 KB asserts in
  `merkle.rs` and `metal_gpu/mod.rs`. Dropped `commit_grouped`'s
  dispatch threshold to `num_leaves ≥ 256` (no per-leaf-bytes cap).
- **Why**: SHA-256 F_2's commit shape has `num_leaves=262144` at
  nvars=22 — plenty of GPU parallelism — but the kernel's hard 64 KB
  / leaf cap blocked GPU dispatch entirely.
- **Result at nvars=22**: e2e Prove 2.96 s → 2.40 s (−19%). At
  nvars=16: 32 → 25 ms (−21%). At nvars=9 the threshold
  (`num_leaves=32 < 256`) keeps CPU; no regression.

### Filter standalone benches via `pair_primary_witness_polys_pub` (commit `9b8f178`)
- **What**: bench helpers (`Commit-Pair / Encode / Fused / GPU /
  GPU-Inline` in Steps + Micro groups) now use
  `pair_primary_witness_polys_pub` instead of pairing the full
  35-col trace. They now measure the same shape the production
  commit produces.
- **Why**: standalone benches were overstating commit cost by
  ~2.6× by pairing public + virtuals.

### Combined-K virtual cols + (X^32) ideal (commits `d3c4636`, `7db1e68`, `2274425`)
- **What**: replaced Binius-style per-add carry-save tree (6 majorities
  + 7 carries) with a single combined-K column per addition, under
  the `(X^32)` ideal. New `F2KVirtualSpec` excludes the 7 K cols
  from commit + γ-batched open.
- **Why**: cuts witness col count 27 → 21 (commit `7db1e68`), then
  the K-virtualisation cuts paired batch_size 11 → 7 (commit
  `2274425`), shrinking PCS opened/values by ~40%.
- **Result at nvars=22**: opened/values 5.05 MB → 3.03 MB raw;
  total proof zstd-3 950 KB → 677 KB (−29%).
- **Soundness caveat**: K cols' MLE evals at r* are currently
  trusted (Issue 1 — see `protocol/src/f2_gkr_plan.md`). Local
  soundness for additions moves to the SHA-256 boundary check.
  Full sound discharge needs the "ShiftR-of-MLE" mechanism.

---

## Investigated, didn't help

### GPU warm-up dispatch (rejected)
- **Hypothesis**: the 320 ms gap between `Commit` (isolated, 577 ms)
  and `Commit-AfterPrevProve` (900 ms) was Metal dispatch
  cold-start after ~1.4 s of CPU-only UAIR + Open between commits.
- **Test**: added a tiny 32-byte no-op Blake3 dispatch at the
  start of `commit_pre_paired_witness` to wake the GPU before
  the real dispatch.
- **Result**: no change. `Commit-AfterPrevProve` 772 → 782 ms,
  `Commit` 557 → 563 ms — both moved by the no-op's ~5–10 ms cost
  with zero compensating speedup. **GPU dispatch warmth is NOT
  the bottleneck.**
- **Real culprit**: most likely the 768 MB slab allocation inside
  `commit_grouped` (`vec![0u8; num_leaves × leaf_bytes]` =
  `262144 × 3072` at nvars=22). Cold allocator + cold-cache
  encode work. See open item below.

---

## Identified but not implemented

### Hadamard discharge for the 13 + 3 column-level relations (step 5 of 5)
- **What**: an external `(x + X·c) ⊙ (y + X·c) = c + X·c` (in
  `F_shift`) check for each of the 13 chained-`BiniusAdd` binary
  steps (C5–C11), plus the 3 AND-flavoured relations
  (C12 = `W_E ⊙ W_E^{↓1}`, C13 = `(1 − W_E) ⊙ W_E^{↓2}`,
  C14 the `Maj` Hadamard from doc §3.5). Total 16 Hadamard
  relations.
- **Why it matters**: this is the last piece of AIR-level soundness
  for the chained-Binius construction. Without it, bits 1..31 of
  every modular addition (and every AND output) are honest-prover-
  only at the AIR layer. Steps 1–3 shipped the surrounding
  algebraic shape (ideal split, packed `PA_C`, 13 LSB checks via
  bit-op-virtual `SHR^j(PA_C)`); step 5 is the actual soundness
  load-bearer.
- **Inputs / outputs at the AIR layer**:
  - For each addition step, the Hadamard inputs are virtual:
    bits 0..30 of `c` = `SHR^1` of XOR-of-(target, x, y); bit 31
    of `c` = β_k, a committed bit (`W_β`, packed analogously to
    `PA_C`; not yet a primary column on this branch — would be
    added with step 5).
  - For each AND step, inputs are free linear combinations of
    `W_E`, `W_A` (and their shifts) per doc §3.5.
  - Output is the per-row Hadamard product, which the
    constraint side declares equals the right-hand side
    (`c + X·c` for adds; the committed AND-result column for
    AND/Maj).
- **What's needed**: a PIOP-layer Hadamard-product check on
  column-level commitments. Distinct from per-cell SHR/Rot
  (already shipped) and from row-shift virtuals — Hadamard is a
  per-bit product across two columns, not a re-indexing.
  Concrete protocol shape is a separate design pass; placeholder
  references include the doc's §3.4 ("mechanism left as a design
  parameter") and the Binius literature's Hadamard-via-grand-
  product approach.
- **Adjacent column work that lands with step 5**: commit `W_β`
  as a primary column (1 packed bit-poly_32, bits 0..12 used for
  the 13 β_k); add 12 internal `BitOpSpec`s for `SHR^j(W_β)`
  bit extraction (mirrors the `PA_C` setup). Until then, β
  values are unconstrained — the deferred Hadamard would be the
  thing that pins them.

### Sound discharge for K-virtual MLE evaluations at r* (Issue 1)
- **What**: currently the 7 K-virtual cols' MLE evaluations at
  `r*` are extracted from the sumcheck transcript and **trusted**
  — no derivation from the source recipe, no PCS-side
  reconstruction. The constraint system relies on the SHA-256
  boundary check (publicly-fixed final digest) to make this
  honest-prover-only design effectively sound.
- **What's needed**: a **row-shift** discharge mechanism for the
  cross-row shifts inside each K source recipe (e.g. `wit(cols::W_W, 16)`,
  `wit(cols::W_SIG0, 1)`). Note: MLE evaluation at a fixed point does
  *not* commute with row shifts, so the verifier **cannot** derive
  `MLE(SHIFT_row^k(v))(r*)` directly from `MLE(v)(r*)`. The mechanism
  has to relate the two via something other than evaluation-at-the-
  same-point — exact shape is left to the design owner; placeholder
  text deliberately vague here. **The per-cell `ShiftR^1` that wraps
  the K recipe is already handled** by `F2BitOpVirtualSpec` and
  `apply_bit_op_u32` (shipped, with the in-tree canary above); the
  K-discharge work just needs the row-shift piece on top.
- **Where**: `protocol/src/f2_prove.rs::verify_f2_full_with_bit_ops`
  would gain a "derive K col evals from source col evals + spec"
  step, analogous to how `F2VirtualBpSpec` (XOR) virtuals are
  derived today. The recipe is already captured in
  `F2KVirtualSpec` (sources + row shifts + final BitOp), waiting
  for the discharge mechanism to land.
- **Estimated effort**: depends on whether the shift-predicate
  machinery is already in `piop/src/shift_predicate.rs` and
  usable as-is. The math sketch lives in
  `protocol/src/f2_gkr_plan.md` rev 2.

### Open-FULL micro overcounts vs in-situ
- **Observation**: `Open-FULL` (520 ms) is higher than the
  in-situ `prove_f2_open` measurement (430 ms) at nvars=22.
- **Suspected cause**: Open-FULL uses `iter_batched` with a
  setup that re-builds `(hint, subclaim, transcript)` each iter
  by re-running commit + UAIR. Some allocation/cache state in
  setup leaks into the timed window.
- **Not blocking** but means the wrap-up accounting is slightly
  off in the other direction; don't read Open-FULL as the true
  open cost.
- **Fix idea**: maybe run setup once outside the batch and clone
  the inputs per-iter? Or accept the inflation and document it.

### Sumcheck data-clone elimination
- **Where**: after the two clones we eliminated in commit
  `22f618a`, there's still room. The `projected_trace` at
  `f2_prove.rs:940` builds a length-`num_total_cols` `Vec` of
  `DenseMultilinearExtension<BinaryFieldGF192>`, each ~96 MB at
  nvars=22 — totalling ~3.4 GB of intermediate allocations per
  prove.
- **Already-applied mitigation**: line 1025-1056 zero-copy
  reinterprets each col's `Vec<BinaryFieldGF192>` into
  `Vec<<BinaryFieldGF192 as Field>::Inner>` via `ManuallyDrop` +
  `from_raw_parts`. So at least the per-col-evaluation pass
  doesn't add another copy.
- **Possible next step**: the projection loop itself could
  process columns in chunks and discard the per-col vec as soon
  as it's been consumed by the sumcheck, avoiding the full
  ~3.4 GB live-at-once footprint.

### Update `Open-h-FinalAssembly` + `UAIR-e-WrapUp` per-step benches
- Currently `UAIR-FULL` and `Open-FULL` exist (commit `22f618a`).
  Their delta vs sub-step sums identifies wrap-up regions but
  doesn't decompose them further.
- A pair of `*-WrapUp` benches that time JUST the assembly /
  transcript-absorb / drop chunks (subtracting the kernels) would
  let us isolate each contributor without re-instrumenting the
  prover.
- Low priority — the residual wrap-up at nvars=22 is now ~40 ms
  for Open, ~270 ms for UAIR (post-clone-removal). Not currently
  the largest pain.

### Investigate Metal `MTLCommandQueue` reuse / `MTLCommandBuffer` pooling
- **Why**: the GPU dispatch warmup test (rejected above) shows the
  per-dispatch cost isn't variable — but each commit builds a fresh
  `CommandBuffer` and encoder. Maybe pooling these would shave a
  few ms per commit. Not validated.

### Parallelise Merkle leaf hashing within a leaf? (rejected — kept as note)
- Per-leaf is ~3 KB at SHA-256 F_2 shape (3 Blake3 chunks). Within-
  leaf `update_rayon` is pure overhead at that size. Cross-leaf
  parallelism via `cfg_iter!` already exists.
- Would only matter for protocols with much larger leaves
  (per-leaf > ~256 KB).

---

## Open questions / hypotheses to test

### Why is `Commit-AfterPrevProve` still ~47 ms slower than `Commit` (isolated)?
- After the slab-reuse fix the gap dropped from 215 ms to ~47 ms
  (566 vs 519 ms). The big suspect — the 768 MB slab alloc — was
  confirmed and fixed. The residual ~47 ms is probably the
  per-row encode rayon parallelism re-spawning worker thread
  state after the pool sat idle during the prior UAIR/Open, or
  cold L2 cache on the paired-witness MLE evaluations re-read.
- Low priority — at this point we're chasing noise-level variance.

### Does the bench fixture's `num_rows = 8` choice make sense for measurement?
- `setup_prover` hard-codes `num_rows = 8` independent of `nvars`,
  so the codeword/encoder dimensions grow disproportionately with
  `nvars` (`row_len = poly_size / 8 = 2^(nvars-3)`).
- This means SHA-256 F_2's commit has many thin leaves (~3 KB ×
  262k at nvars=22) — atypical for protocols with bigger
  cells/cols.
- Bench numbers reflect this specific choice. Consider whether
  a 2-row variant (matching the original protocol-doc shape) is
  also worth keeping for cross-protocol comparison.

### Can we get sound K-virtual MLE evaluation via the `f2_native_ic` lane?
- The F_2-native IC stage (`f2_native_ic::F2NativeIc::prove_linear`)
  already evaluates degree-1 constraint expressions in bit-pack
  form. The K column's source recipe (`ShiftR^1 of XOR of
  row-shifted sources`) is degree-1. Maybe there's a clean way
  to fold the K derivation into the IC stage's per-cell eval, so
  it lands on the sumcheck-extracted column claims naturally.

---

## How to use this doc

When you investigate any optimization, design alternative, or hypothesis
on the F_2 SHA-256 prover path — even if you don't change code —
record it here. Concretely:

1. **If you ship a change**: add a "Shipped work" entry with the
   commit SHA, what changed, why, and the measured impact.
2. **If you try something and it doesn't work**: add an
   "Investigated, didn't help" entry with the hypothesis, the
   test, the result, and (if known) the real culprit.
3. **If you identify an optimization but defer it**: add an
   "Identified but not implemented" entry with where the
   opportunity lives, the approach, and the expected gain.
4. **If you have a hypothesis but no time to test**: add it to
   "Open questions" with a concrete test plan.

This avoids the next agent (or you, in 3 weeks) repeating dead
ends or re-discovering the same hot-paths.
