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
- [ ] Are we doing multipoint eval?
- [ ] SHIFT vector proving — **row-level** shift virtualization (needed
      for the K-discharge work, Issue 1). The per-cell SHIFTR case
      already ships via `F2BitOpVirtualSpec`; see the "Per-cell SHIFTR
      virtualisation in-tree canary" entry under Shipped work below.
- [ ] Review approach to add mod 2^32 in documentation
- [ ] Hadamard product implementation


---

## Shipped work (chronological, most recent first)

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

### Sound discharge for K-virtual MLE evaluations at r* (Issue 1)
- **What**: currently the 7 K-virtual cols' MLE evaluations at
  `r*` are extracted from the sumcheck transcript and **trusted**
  — no derivation from the source recipe, no PCS-side
  reconstruction. The constraint system relies on the SHA-256
  boundary check (publicly-fixed final digest) to make this
  honest-prover-only design effectively sound.
- **What's needed**: a **row-shift** predicate gadget — a mechanism
  for the verifier to evaluate `MLE(SHIFT_row^k(v))(r*)` from
  `MLE(v)(r*)` for the cross-row shifts that appear inside each K
  source recipe (e.g. `wit(cols::W_W, 16)`, `wit(cols::W_SIG0, 1)`).
  Likely a shift-predicate over GF(2^192). **The per-cell `ShiftR^1`
  that wraps the K recipe is already handled** by
  `F2BitOpVirtualSpec` and `apply_bit_op_u32` (shipped, with the
  in-tree canary above); the K-discharge work just needs the row-
  shift piece on top.
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
