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

## Shipped work (chronological, most recent first)

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

### Reuse the 768 MB commit slab across commits
- **Why it matters**: this is now the dominant residual cost in
  `Commit-AfterPrevProve` vs isolated Commit (~215 ms remaining
  gap). Each commit allocates a fresh 768 MB slab (`vec![0u8; ...]`
  inside `phase_commit.rs::commit_grouped`'s GPU-inline branch),
  writes to it, then drops it at end of commit. In a tight loop
  the allocator returns the same pages and the cache stays warm;
  in e2e the allocator's been disturbed and the slab hits cold
  DRAM.
- **Where**: `zip-plus/src/pcs/phase_commit.rs` ~line 170, the
  `if go_gpu { let mut slab = vec![0u8; ...]; … }` branch.
- **Approach options**:
  - (a) Per-`ZipPlus` static thread-local scratch resized
        on-demand. Reset to zero each commit (cheap memset).
  - (b) Caller-owned `&mut Vec<u8>` parameter passed into
        `commit_grouped` / `commit_pre_paired_witness`. Bench
        fixture / production prover owns the buffer.
  - (c) `MetalContext`-owned scratch (since it already holds
        per-process scratch in `HashScratch`).
- **Expected gain**: 100–200 ms on e2e Prove at nvars=22; bigger
  at higher nvars.

### Sound discharge for K-virtual MLE evaluations at r* (Issue 1)
- **What**: currently the 7 K-virtual cols' MLE evaluations at
  `r*` are extracted from the sumcheck transcript and **trusted**
  — no derivation from the source recipe, no PCS-side
  reconstruction. The constraint system relies on the SHA-256
  boundary check (publicly-fixed final digest) to make this
  honest-prover-only design effectively sound.
- **What's needed**: a mechanism for the verifier to access
  `MLE(ShiftR^1(v))(r*)` from `MLE(v)(r*)` without a dedicated
  commitment — likely a shift-predicate gadget that lifts to
  GF(2^192).
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

### Why is `Commit-AfterPrevProve` still ~210 ms slower than `Commit` (isolated)?
- After the per-col absorb fix, the gap dropped from 320 ms to
  210 ms. The remaining gap is inside `commit_grouped` — the
  encode + scatter + GPU dispatch + Merkle build.
- The 768 MB slab alloc (see above) is the top suspect. **Test
  plan**: instrument the inside of `commit_grouped`'s GPU branch
  to time (slab alloc, encode-and-scatter, GPU dispatch, tree
  build) separately in both isolated and AfterPrevProve contexts.
- Secondary suspect: the per-row encode rayon parallelism may
  re-spawn worker thread state if the rayon pool went idle during
  the prior UAIR/Open.

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
