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

### Round-1 fast path for the Hadamard degree-3 zerocheck (Phase 1; working tree)
- **What**: `HadamardRound1FastPath` (`piop/src/lookup/hadamard.rs`) + a
  `prepare_hadamard_group_with_fast` that takes **packed operand columns**
  (`BinaryPoly<D>`, 3 per relation) and uses `with_round_1_fast` instead of
  materialising the 1536 full-width `GF128::Inner` bit-slices.
  `prove_f2_hadamard_phase` now builds packed columns (`build_operand_column`
  + the existing `build_adder_operand_columns`) rather than calling the
  (now-removed) `build_all_operand_slices`. Modelled on the booleanity
  degree-3 fast path, but over `GF(2^128)`: boundary points are
  `F::from_with_cfg(2/3)` (= `X`, `X+1`), **not** booleanity's `one+one`
  (which collapses to `0,1` in char 2 — that shortcut is correct only in
  booleanity's large-field integer pipeline).
- **Correctness**: `round_1_message` does NOT assume `M(0)=M(1)=0` — unlike
  booleanity's `v(v−1)` (structurally 0 for any bit), the Hadamard term
  `U·V−W` is non-zero on a corrupt row, so it computes all four `M(0..3)`
  to match the generic path (the negative tests `corrupt_w_is_rejected` /
  `adder_rejects_wrong_sum` fail otherwise). Gated by a new piop test
  `fast_path_matches_generic` asserting the fast- and generic-path
  `MultiDegreeSumcheckProof`s are **byte-identical**; 58 piop + 50 protocol
  lib tests green.
- **Result (Prove-Hadamard, M-series, `parallel simd unchecked`)**:

  | nvars | generic | fast path | Δ     |
  |-------|---------|-----------|-------|
  | 9     | 10.35 ms| 8.03 ms   | −22%  |
  | 16    | 650.7 ms| 622.6 ms  | −4%   |
  | 20    | 28.83 s | 18.83 s   | −35%  |

  Verify unchanged. **Partial win**: round-1 compute is similar (4 boundary
  points either way), so where the working set fits RAM (nvars≤16) the gain
  is small; the nvars=20 −35% is the ~2× peak-memory reduction (round-1's
  24 GB materialise removed, but `fold_with_r1` still emits ~12 GB of
  half-size slices). nvars≥20 stays memory-bound until **Phase 2** (slice
  dedup) lands — see "Identified but not implemented".

### Hadamard discharge cost measured — new `f2_sha256` "Hadamard" A/B group (working tree)
- **What**: added a `hadamard` criterion group to `protocol/benches/f2_sha256.rs`
  that A/Bs the *same* `prove_f2_full_with_*` / `verify_f2_full_with_*`
  family **with** the 16 SHA-256 Hadamard relations (3 ANDs + 13 adders,
  from `Sha256F2HadamardLayout::relations()`) vs **without** (`&[]` specs,
  what the rest of the bench measures). Both sides non-pre-paired so the
  A/B delta is purely the discharge. Also extended `f2_full_proof_parts`
  to count the previously-omitted Hadamard fields (`uair.hadamard_proof`'s
  sumcheck + `bit_slice_evals`, `hadamard_pair_evals`,
  `hadamard_adder_parents`) — before this the with/without proof sizes came
  out byte-identical because those fields weren't in the breakdown.
- **Result (M-series, `parallel simd unchecked`, no `metal_gpu`, sample_size 10)** —
  the discharge is **cheap to verify, catastrophic to prove, and worsens
  super-linearly**:

  | nvars | Prove no-had | Prove +16had | ×    | Verify no-had | Verify +16had | × |
  |-------|--------------|--------------|------|---------------|---------------|---|
  | 9     | 2.33 ms      | 10.35 ms     | 4.4× | 1.06 ms       | 1.41 ms       | 1.33× |
  | 16    | 47.0 ms      | 650.7 ms     | 13.8×| 4.67 ms       | 5.15 ms       | 1.10× |
  | 20    | 739.9 ms     | **28.83 s**  | 39×  | 42.9 ms       | 44.2 ms       | 1.03× |

  - **Verify**: negligible (+3% at nvars=20, shrinking with nvars). Not a concern.
  - **Proof size**: ~**constant +26 KB raw / +25 KB zstd**, independent of
    nvars (so 3.0% at nvars=9 → 0.4% at nvars=20). Dominated by
    `bit_slice_evals` (24.6 KB); the zerocheck `sumcheck` is only ~1 KB,
    `pair_evals` 144 B, `adder_parents` 624 B. Size is a non-issue.
  - **Prove**: the headline problem. The no-had prove scales ~linearly
    (47 → 740 ms = 15.7× for the 16× rows of nvars 16→20); the +had prove
    scales **super-linearly** (651 ms → 28.8 s = **44×** for the same 16×
    rows). At nvars=20, **28.1 s of the 28.8 s is discharge overhead.**
- **Root-cause hypothesis (unconfirmed — needs profiling)**: the super-
  linearity lives entirely in the discharge path, consistent with a
  **memory-bandwidth bound**. The Hadamard zerocheck materialises bit-slice
  MLEs for all 16 relations' operands (D=32 slices/column → 32× the per-
  column data); at nvars=20 the working set (~hundreds of MB per operand
  set) blows past cache, so wall-time grows faster than O(rows). The degree-3
  zerocheck round-1 (which touches the full (μ+5)-var hypercube, ≈50% of the
  sumcheck point-work) is the prime suspect for the constant factor.
- **Optimisation opportunities this surfaced** (see "Identified but not
  implemented"): (a) a `{0,1}`-slice round-1 fast path for the degree-3
  zerocheck — round 1 is the biggest round and the bit-slices are still
  0/1-valued there (AND/select instead of GF(2^128) mults); the main
  γ-sumcheck already has `F2EqColRound1FastPath`, this ports it to the
  Hadamard zerocheck. (b) batch the 16 per-relation zerochecks instead of
  running them as separate groups. (c) avoid full bit-slice materialisation
  / cache-block the expansion to fix the super-linear memory behaviour.

### `LEAF_GROUP_SIZE` 8 → 1 to shrink `open.opened/values` 8× (commit `<TBD>`, measurements pending)
- **What**: `protocol/src/f2_prove.rs::LEAF_GROUP_SIZE` flipped from
  `8` to `1`. With group size 1 each Merkle leaf hashes exactly one
  paired-storage column stack (`paired_batch × num_rows × 8 B`),
  so an opening ships one column of cells instead of eight. The
  lower layers (`zip-plus::pcs::phase_commit::commit_grouped`,
  `MerkleTree::new_from_column_groups`) are fully parameterised
  over `group_size`; the only invariant is power-of-two, which 1
  satisfies. No other code changes needed — every call site
  references the constant, not the literal `8`.
- **Why**: the `open.opened/values` region of the proof was
  ~4.34 MB at the bench's nvars=22 shape (`987 openings × 8
  LEAF_GROUP_SIZE × 9 paired_batch × 8 num_rows × 8 B/cell`).
  The `LEAF_GROUP_SIZE` factor exists purely to amortise Blake3
  setup during commit; the wire-side cost is a `LEAF_GROUP_SIZE×`
  bloat on values, partly offset by `log2(LEAF_GROUP_SIZE)`
  fewer Merkle siblings per opening.
- **Expected proof-size delta** (analytic, nvars=22, 987 openings):
  - `open.opened/values`: 4.34 MB → 555 KB (−3.8 MB).
  - `open.opened/merkle`: +3 siblings × 987 openings × 32 B
    (Blake3 hash) ≈ +95 KB.
  - **Net: ~−3.7 MB on the dominant proof region.**
- **Expected prove-time cost**: Blake3 leaf hashing loses 8× of
  its setup amortisation — each leaf hashes 72 B (one paired
  column stack) instead of 576 B, so per-byte throughput
  collapses on the Metal-Blake3 leaf kernel and the CPU fallback.
  Leaf count grows 8× (`codeword_len / 1` instead of
  `codeword_len / 8`), and the Merkle tree gains 3 internal
  levels.
- **Verification**: 12/12 F_2 lib tests pass
  (`cargo test -p zinc-protocol --lib f2_prove --release`),
  including `prove_then_verify_sha256_f2_roundtrips` and the
  full `prove_then_verify_f2_full_*` suite.
- **Measurements pending**: still need to bench
  `Zinc+ F_2 SHA-256/Commit*` and `Prove e2e` at nvars=22 to
  confirm the prover-time regression is acceptable for the
  ~3.7 MB proof-size win. If the commit-time hit exceeds the
  proof-size benefit, this entry should move to "Investigated,
  didn't help" and the constant should revert.
- **How to bench**: `cargo bench -p zinc-protocol --bench
  f2_sha256 --features parallel,simd,unchecked,metal_gpu --
  "Zinc\+ F_2 SHA-256/"` and compare against the previous
  baseline. Look at `Commit-Fused-GPU-Inline`, `Prove`, and
  the proof-size breakdown for `open.opened/{values,merkle}`.

### F_2 native IC: short-circuit `assert_zero` slots to `ZERO` in both builders (commit TBD)
- **What**: in `protocol/src/f2_native_ic.rs`, `F2NativeMleFirstBuilder::assert_zero`
  (used by `prove_linear`) and `F2NativeRowBuilder::assert_zero` (used
  by `prove_combined`) now push `DynamicPolynomialF::ZERO` /
  `F2RowExpr::zero()` instead of the actual constraint expression.
  The expression itself is dropped on the floor. Mirrors the
  `prove_hybrid` pattern in `piop/src/ideal_check.rs::prove_hybrid`
  but at the builder level — sound for the F_2 protocol's flow
  because the F_2 IC's `combined_mle_values` are used only for
  transcript absorption, not as inputs to the downstream γ-batched
  sumcheck (which operates on the trace columns, not on the IC's
  per-constraint polynomials).
- **Why**: for an honest prover, an `assert_zero` constraint
  evaluates to the zero polynomial cell-wise on every row, and the
  IC verifier's `verify_as_subprotocol` already filters zero-ideal
  slots out of `batched_ideal_check`. Computing the (degree-2,
  selector × residual) F[X] expression and absorbing its coefficients
  was pure busywork in `prove_linear` — one `DynamicPolynomialF`
  polynomial multiply + a 32-coefficient transcript absorb per
  zero-ideal slot, scaled by `num_constraints`. The integer PIOP
  builder explicitly notes the same optimization was reverted there
  for sumcheck-consistency reasons in the integer setting; the F_2
  protocol's structure makes it sound here (see "Why" above).
- **Measured (criterion `--quick`, nvars=22, full perf feature set)**:

  | Bench (Prove e2e) | Before | After  | Δ            |
  |-------------------|-------:|-------:|-------------:|
  | F_2 SHA-256       | 2.92 s | 2.22 s | **−24%** (high variance, [1.99 s, 3.14 s] band) |
  | F_2 Blake3        | 5.90 s | 5.12 s | **−13%**     |

  All 27 `zinc-test-uair` tests still pass; SHA + Blake3 e2e bench
  paths complete with successful `verify_f2_uair` /
  `verify_f2_open_with_virtuals`.
- **Constraint mix at play**: SHA-256 has ~3 zero-ideal pins out of
  ~17 constraints (~18%); Blake3 has 30 zero-ideal out of 96 (~31%).
  Surprisingly SHA shows the larger win despite the lower zero-ideal
  fraction — likely the per-slot polynomial product dominates more
  for SHA's shape (fewer total constraints, so each slot is a
  larger fraction of work).

### SIMD-batched α-projection (commit TBD) — 2.07× on random, tied on real SHA-256
- **What**: added two NEON-batched 4-cell α-projection kernels in
  [`poly/src/univariate/binary_gf128.rs`](../poly/src/univariate/binary_gf128.rs)
  plus a column-projection wrapper, wired into
  [`f2_prove.rs:877`](../protocol/src/f2_prove.rs#L877) via
  `project_column_with_powers`:
  - **`eval_f2_poly_d_at_with_powers_simd_x4`** (dense): four cells in
    lockstep, `D=32` unconditional iterations. Each iter does one
    16-byte `vld1q_u64` of α^i shared across four accumulators, plus
    four `vandq_u64(pw, mask_k)` + `veorq_u64` per accumulator. Scalar
    fallback for non-aarch64 keeps the same shape with regular u64
    XOR/AND.
  - **`eval_f2_poly_d_at_with_powers_simd_x4_sparse`** (union-skip):
    same body but iterates only positions present in
    `cells[0] | cells[1] | cells[2] | cells[3]` via `trailing_zeros`,
    paying loop-control overhead in exchange for skipping all-clear
    bit positions.

  Wired the **dense** variant into the prove path. Sparse stays
  exported for future use on UAIRs with low cell-popcount/high cell
  correlation.

- **Why now**: the GF(2^192) → GF(2^128) field swap (entry below)
  delivered the predicted 1.5–3.6× wins on every field-touching
  micro-bench *except* `UAIR-b-AlphaProject` — the dominant single
  phase at 1067 ms / nvars=22 (≈90% of UAIR). That kernel uses a
  branchy `trailing_zeros` walk where one fewer XOR per set bit
  (2 vs 3 u64 words for GF128 vs GF192) is invisible against
  loop-control overhead. SIMD batching is the natural next lever to
  amortise that overhead.

- **Measured (criterion, `--features parallel,simd,unchecked`,
  Apple M-series 8-core):**

  Random 65 536-cell column (`binary_gf_compare::project_col_65536`):
  | Kernel | Time | vs branchy |
  |---|---|---|
  | GF128 branchy | 883 µs | 1.00× |
  | GF128 branchless | 603 µs | 1.46× |
  | **GF128 SIMD-x4 dense** | **407 µs** | **2.17×** |
  | GF128 SIMD-x4 sparse | 473 µs | 1.87× |

  Real SHA-256 trace (`Micro/UAIR-b-AlphaProject/nvars=22`):
  | Kernel | Time | vs branchy |
  |---|---|---|
  | GF128 branchy (baseline) | 1067 ms | 1.00× |
  | GF128 SIMD-x4 dense | 1059–1085 ms | ~1.00× (tied within ±2% noise) |
  | GF128 SIMD-x4 sparse | 1105 ms | 0.97× (regressed) |

- **Why the random 2× win collapses to "tied" on real**: the prove
  path's per-cell cost in branchy mode is `popcount(cell) × ~5 ops`.
  On random 32-bit cells, avg popcount is 16, so branchy does ~80
  ops/cell — matching the binary_gf_compare ~12.5 ns/cell. On the
  actual SHA-256 trace, the same branchy kernel runs at ~6.2 ns/cell
  (172 M cells in 1067 ms), implying real-cell avg popcount is closer
  to ~8 — half of random. SIMD-x4 pays a fixed `D=32` cost per cell
  regardless, so it can't undercut a branchy kernel already running
  at 8-bit-per-cell speed.

- **Why sparse regressed**: my hypothesis was that with avg
  popcount~8 per cell, the union of 4 cells would stay well below
  `D=32`. Measurement disagreed: union saturates near `D` on the
  real trace (cells in the same column are mutually uncorrelated +
  the column carries ~all 32 bit positions across its rows), so
  sparse iterates ~`D` times anyway AND pays the loop-control
  overhead. Lesson: low **per-cell** popcount ≠ low **union**
  popcount when cells are independent.

- **Tests**: three correctness tests in `binary_gf128.rs::tests`:
  `simd_x4_matches_scalar_branchy_random` (64 random batches),
  `simd_x4_matches_scalar_branchy_edges` (zero / all-ones /
  single-bit-per-position), `simd_x4_sparse_matches_simd_x4`
  (64 random + 4 adversarial), `project_column_matches_scalar_per_cell`
  (column lengths covering every `len % 4` remainder).

- **Decision**: keep dense SIMD-x4 wired in. The on-prove-path delta
  is within noise but never negative across runs, and the random-
  workload `2×` win means new UAIRs with denser bit distributions
  (e.g. Blake3, Poseidon-over-F_2) get the speedup for free without
  re-engineering the prove path.

- **Open follow-ups**:
  - The α-projection lever appears exhausted at this UAIR shape. The
    next prove-e2e lever is not in this phase; candidates:
    Blake3/Metal commit pipeline, IC sumcheck round work, witness-gen
    in `test-uair/src/sha256_f2.rs`.
  - If a future UAIR has high cell correlation (e.g. lookup-heavy
    columns), `project_column_with_powers_sparse` may pay off. Kept
    exported for that case.

### Field swap GF(2^192) → GF(2^128) on the F_2 prover path (commit TBD)
- **What**: swapped the projecting field from `BinaryFieldGF192`
  (`Uint<3>` storage, FIPS 186-2 pentanomial, Toom-Cook 3-way clmul
  over 6 inner PMULL/PCLMUL muls, 192×192 F_2 matrix inversion for
  `AlphaPolyBasis`) to `BinaryFieldGF128` (`Uint<2>` storage, GHASH
  pentanomial, Karatsuba 2-way clmul over 3 inner muls, 128×128
  matrix inversion). Touched `poly/src/univariate/binary_gf128.rs`
  (promoted from benchmark-only to production with full
  `Field`/`PrimeField`/`InnerTransparentField` surface, `AlphaPolyBasis`,
  `lift_f2_*_to_gf128`/`eval_f2_*_at`/`lift_bp_to_f2_poly_1`/
  `lift_gf128_to_f2_poly_2`/`eval_bits_at`), deprecated
  `binary_gf192` (kept alive only for `binary_gf_compare` bench),
  and global type/module renames across `protocol/src/f2_prove.rs`,
  `protocol/benches/f2_sha256.rs`, `protocol/src/f2_native_ic.rs`,
  `piop/src/{ideal_check,sumcheck/multi_degree,projections}.rs`,
  `test-uair/src/sha256_f2.rs`,
  `poly/src/univariate/binary_f2_wide.rs`, and the
  `f2_prove_plan.md` / `f2_open_plan.md` / `f2-prove-optimizations.md`
  docs.
- **Why**: the field-size choice was the dominant lever the F_2
  prove path's hot loops have left. Algebraically the `BinaryF2Poly<W>`
  widths in the open proof scale linearly with the projecting-field
  bit width (each is `⌈(D + k·n − k)/64⌉` for some constant `k`),
  so cutting `n` from 192 to 128 shrinks every wide-poly type and
  every wide-poly multiplication kernel by the same factor.
- **Width rescale in `F2OpenProof<D>`** (the single algebraic change
  beyond mechanical s/192/128/):
    | Slot | GF192 | GF128 | Bound (D=32) |
    |---|---|---|---|
    | `q_i'` (lift target) | `BinaryF2Poly<3>` | `BinaryF2Poly<2>` | 128 b |
    | `b_g[i]` mid-product | `BinaryF2Poly<4>` | `BinaryF2Poly<3>` | D + n − 1 |
    | `b_vector`, `combined_row` entry | `BinaryF2Poly<7>` | `BinaryF2Poly<5>` | D + 2n − 1 |
    | `lifted_claim a'` | `BinaryF2Poly<10>` | `BinaryF2Poly<7>` | D + 3n − 2 |
  Const-generic call sites (`f2_poly_mul::<3,7,10>`,
  `f2_inner_product::<3,4,7>`, `encode_f2_lin_open::<7>`,
  `eval_f2_wide_poly_at::<10>`, `absorb_f2_poly_slice::<7|10, _>`)
  all rescaled by the same map.
- **Measured wins** (criterion, `--features parallel,simd,unchecked`,
  nvars=22, Apple M-series 8-core):

    | Bench | GF192 | GF128 | Speedup |
    |---|---|---|---|
    | `binary_gf/mul` | 3 ns | 2 ns | 1.5× |
    | `binary_gf/square` | 3 ns | 2 ns | 1.5× |
    | `binary_gf/inverse` | 2.53 µs | 1.51 µs | 1.68× |
    | `binary_gf/alpha_precompute` (D=32) | 277 ns | 178 ns | 1.56× |
    | `binary_gf/project_col_65536/branchless` | 685 µs | 484 µs | 1.42× |
    | `Micro/Open-a-AlphaBasis/nv=22` | 158.45 µs | 43.99 µs | **3.6×** |
    | `Micro/Open-b-LiftedEqTensor/nv=22` | 22.64 ms | 13.82 ms | **1.64×** |
    | `Micro/Open-d-CombinedRow/nv=22` | 101.33 ms | 76.20 ms | **1.33×** |
    | `Micro/UAIR-c-Sumcheck/nv=22` | 28.64 ms | 19.10 ms | **1.50×** |
    | `Micro/UAIR-b-AlphaProject/nv=22` | 1067 ms | 1065 ms | 1.00× ❌ |

  Proof bytes (per `proof_size_breakdown` table, nvars=22):
  `uair.alpha` 24 → 16, `uair.gamma` 24 → 16,
  `open.lifted_claim` 80 → 56, `open.b_vector` 448 → 320.
  Aggregate proof size shrinks ~150 KB at nvars=22 from these scalar
  components (open.combined_row dominates total bytes and is
  unaffected by the field swap — it's already F_2-poly-bound).

- **Why prove e2e barely moved** (Δ ≈ 1% on `Prove/nvars=22`,
  inside criterion's CI band): UAIR is dominated by
  `UAIR-b-AlphaProject` (1065 ms ≈ 90% of UAIR), which uses the
  **branchy** kernel `eval_f2_poly_d_at_with_powers` at
  [`f2_prove.rs:885`](../protocol/src/f2_prove.rs#L885). That
  kernel's per-set-bit work is dominated by `trailing_zeros` +
  `bits &= bits - 1` + memory load; saving one XOR per set bit (2
  vs 3 u64 words) is invisible. The branchless variant **does**
  benefit (`project_col_branchless 1.42×`) but a previous experiment
  showed it regresses ~8 ms on the actual SHA-256 trace (predictable
  bits favour branch prediction) — see comment at
  [`f2_prove.rs:868-876`](../protocol/src/f2_prove.rs#L868-L876).
  Net: the field-size lever is pulled, but the dominant cost is
  outside its reach. Next lever for prove latency would be a
  SIMD-batched α-projection that processes 2–4 cells per iteration to
  amortise the loop-control overhead, OR an alternative kernel shape
  that benefits from the 2/3 word reduction.
- **GF192 fate**: kept compiled, marked
  `#[deprecated(note = "F_2 prover uses BinaryFieldGF128; binary_gf192
  is kept only for the binary_gf_compare bench. ...")]` on
  `pub mod binary_gf192;` in `poly/src/univariate.rs`.
  `binary_gf_compare.rs` carries `#![allow(deprecated)]` so the
  comparison bench keeps running.
- **Out of scope** (deferred): soundness/security write-up updates in
  `f2_prove_plan.md` / `f2_open_plan.md` — per-challenge SZ error
  drops from `d/2^192` to `d/2^128`, still well above 100-bit total
  but a formal sum-of-errors derivation across all FS draws is owed.
  TODO already flagged at `binary_gf128.rs:31-56`. Itoh–Tsujii
  inverse remains naive Fermat (no production hot path hits it).

### Parallelise q1 lift in `build_lifted_eq_tensor` (commit `d8c14d0`, −86% LiftedEqTensor, −27% Prove e2e)
- **What**: `build_lifted_eq_tensor` (in `protocol/src/f2_prove.rs`)
  built the q1 lifted-eq tensor via a plain
  `q1_gf.iter().map(|g| basis.lift(g)).collect()`. At SHA-256 F_2
  nvars=22 with num_rows=8, q1 has 2^{22−3} = 524 288 entries, each
  one a 192×192 F_2 matrix-vec multiply (`AlphaPolyBasis::lift`,
  ~600 ops per call). All single-threaded.
- **Fix**: switched the q1 collect to
  `cfg_iter!(q1_gf).map(...).collect()`. The lift is
  data-independent across entries, so the parallelism is
  straight-line. q0 stayed sequential (it's only ≤ num_rows
  entries, ~8 — rayon spawn overhead dominates).
- **Measured (criterion `--save-baseline` / `--baseline`,
  nvars=22, `--features parallel,simd,unchecked,metal_gpu`)**:

  | Bench                          | Before    | After    | Δ      |
  |--------------------------------|-----------|----------|--------|
  | Open-b-LiftedEqTensor/nv=22    | 163.2 ms  | 22.8 ms  | **−86.1%** (7.1×) |
  | Prove e2e/nvars=22             | 3.11 s    | 2.27 s   | **−27.3%** |

  Both p < 0.01. The 7.1× lift speedup matches the parallelism
  factor on M-series (8 cores). Prove e2e baseline of 3.11s was
  on the high end of run-to-run variance; conservative bound on
  the change is −18.5% (lower CI endpoint).
- **Bonus**: `verify_f2_open_with_virtuals` also calls
  `build_lifted_eq_tensor` (line 2237). Verifier gets the same
  parallel speedup for free.
- **Lesson**: scan for `.iter().map().collect()` patterns on
  large data where the per-element work is non-trivial — they're
  free parallelism wins. The audit missed this; finding it took
  a per-Open-micro bench breakdown to see that LiftedEqTensor
  was eating 161 ms out of ~745 ms Open.

### Commit slab: 64 MB capacity floor after GPU dispatch (commit `7550055`, RSS hygiene)
- **What**: `zip-plus/src/pcs/phase_commit.rs::commit_grouped`'s
  GPU branch caches a process-wide `Vec<u8>` (`COMMIT_SLAB_SCRATCH`)
  that grows on demand but was never shrunk. At SHA-256 F_2
  nvars=22 the slab reaches ~768 MB and stays there for the life
  of the process, even after smaller-nvars commits. Added
  `slab.shrink_to(64 * 1024 * 1024)` after the GPU dispatch +
  Merkle tree build, before releasing the Mutex.
- **Behaviour**:
  - `slab.shrink_to(min_cap)` sets `capacity = max(len, min_cap)`.
    `slab.len() == required` post-resize, so capacity becomes
    `max(required, 64 MB)` after the shrink.
  - `required > 64 MB` (e.g., nvars=22): no-op, no realloc.
    Steady-state same-nvars hot loops pay nothing.
  - `required ≤ 64 MB` (e.g., nvars=16 with our commit shape):
    capacity shrinks to 64 MB, releasing the historical peak.
- **Limitation**: this only releases memory when the *current*
  commit is small. The "ran nvars=22 once, now idle" case still
  holds the ~768 MB. Releasing across idle periods would need an
  explicit release API (a `pub fn release_commit_slab_scratch()`
  on `ZipPlus` that callers invoke between batches).
- **Measured (criterion `--save-baseline` / `--baseline`, nvars=22,
  `--features parallel,simd,unchecked,metal_gpu`)**:

  | Bench                          | Before   | After    | Δ      | verdict |
  |--------------------------------|----------|----------|--------|---------|
  | Commit-Fused-GPU-Inline/nv=22  | 583.4 ms | 585.1 ms | +0.30% | no change (p=0.57) |
  | Prove/nvars=22                 | 2.54 s   | 2.47 s   | −2.74% | no change (p=0.58) |

  Initial Prove A/B run reported a +13% regression with p=0.02
  but a `[+3%, +24%]` confidence interval; the re-run gave
  −2.74% with p=0.58. Prove at nvars=22 has high run-to-run
  variance (10 samples × ~2.5 s each, with thermal/scheduler
  effects across 25–40 s of bench wall time). The Commit micro
  is the direct test of the change and shows no change in either
  run — confirming the shrink is no-op at nvars=22 as expected.
- **Verification**: 13/13 F_2 lib tests pass.
- **Lesson recorded**: at this nvars Prove takes long enough to
  swing 10%+ between criterion sample sets. For changes whose
  direct test (here: Commit micro) shows no change, trust that
  signal over a noisy higher-level bench. See also the GPU
  threadgroup-cap revert/re-ship saga, where the same
  "statistically significant" verdict turned out to be noise on
  a too-small baseline.

### PMULL-accelerated `f2_poly_mul` (commit `613de5d`, −21.5% Prove e2e, −92.4% coherence check)
- **What**: `poly/src/univariate/binary_f2_wide.rs::f2_poly_mul<W_A, W_B, W_OUT>`
  was a bit-by-bit schoolbook: for each set bit of `a`, XOR a
  word-shifted `b` into the accumulator (`xor_shifted`). Replaced
  with a word-level schoolbook using
  `binary_gf128::clmul_64x64` (PMULL on aarch64, PCLMUL on x86_64,
  scalar fallback otherwise): for each `(ai, bi)` pair, one 64×64
  carryless multiply produces a 128-bit partial product whose two
  halves XOR into `acc[ai+bi]` and `acc[ai+bi+1]`.
- **Why this matters**: the audit had categorised this as a
  verify-only optimisation ("coherence-check `<3,7,10>` mults at
  row_len scale"). In practice `f2_poly_mul` is used heavily by
  the **prover's** `prove_f2_open` too — multiple shapes
  (`<1,3,4>` / `<3,4,7>` / `<3,7,10>`) inside the encoding-
  consistency assembly + the per-column `b_vector` / `combined_row`
  builds. All shapes get the same ~5–10× speedup.
- **Measured (criterion `--save-baseline` / `--baseline`, nvars=22,
  `--features parallel,simd,unchecked,metal_gpu`)**:

  | Bench                          | Before  | After    | Δ      |
  |--------------------------------|---------|----------|--------|
  | Prove e2e                      | 3.17 s  | 2.49 s   | **−21.5%** |
  | VerifyOpen-d-Coherence/nv=22   | 46.3 ms | 3.53 ms  | **−92.4%** (13×) |

  Both p < 0.01.
- **Implementation detail**: made `binary_gf128::clmul_64x64`
  `pub(crate)` so `binary_f2_wide` could reuse the existing
  hardware-dispatched helper (avoids duplicating the aarch64/x86_64
  cfg dance). Added `a_word == 0` / `b_word == 0` early-skip in
  each inner iteration so sparse high-order words don't pay PMULL
  cost.
- **Lesson**: the audit's "verify-only" claim was wrong because it
  didn't trace the function's call sites. A grep for
  `f2_poly_mul` in `protocol/src/` would have surfaced 11+ call
  sites in `prove_f2_open` / `verify_f2_open_with_virtuals`. When
  an audit recommends a function-level change, verify the
  function's actual call graph before sizing the impact.
- **Verification**: 93/93 zinc-poly lib tests pass under
  `--features simd`; 13/13 F_2 protocol lib tests pass.

### Metal Blake3 dispatch: lift `min(256)` threadgroup cap (commit `e4d6a25`, re-shipped after realistic-workload re-test)
- **What**: `zip-plus/src/metal_gpu/mod.rs:201-214`. Removed the
  hard-coded `min(256)` clamp on `threads_per_threadgroup`; the
  pipeline's own `max_total_threads_per_threadgroup()` now wins
  (already accounts for kernel resource use — per-thread private
  memory is ~864 B worst case from the 24-deep subtree-stack).
  Originally landed as `31c422b`, reverted in `58d2124` after a
  +9% A/B regression at Prove/nvars=16. After commit `6e40edd`
  scaled `NUM_COMPRESSIONS` with `num_vars`, a fresh A/B on the
  realistic shape reverses the verdict.
- **Measured (criterion `--save-baseline` / `--baseline`, realistic
  workload, `--features parallel,simd,unchecked,metal_gpu`)**:

  | Bench (nvars)               | Before (256 cap) | After (cap removed) | Δ      | criterion verdict |
  |-----------------------------|------------------|---------------------|--------|-------------------|
  | Prove/16                    | 47.4 ms          | 46.6 ms             | −2.24% | improved (p<0.01) |
  | Prove/20                    | 675 ms           | 677 ms              | +0.28% | no change (p=0.61) |
  | Prove/22                    | 3.45 s           | 3.49 s              | +1.15% | no change (p=0.75) |
  | Commit-Fused-GPU-Inline/22  | 584 ms           | 587 ms              | +0.50% | no change (p=0.38) |

  Net: neutral-to-positive everywhere on the realistic shape; the
  +9% regression at nvars=16 from the original A/B was noise on a
  ~30 ms baseline (a 3 ms swing is "statistically significant" at
  p<0.05 but practically within thermal/scheduler noise at that
  scale). The realistic-shape `Prove/nvars=16` baseline is ~47 ms,
  pushing the noise floor below the effect size.
- **Lesson**: the original revert decision was correct given the
  data available — but criterion's statistical significance test
  doesn't distinguish "real effect" from "thermal noise at a small
  baseline." When the baseline is small enough that a 3 ms swing
  is the entire reported delta, treat any verdict with skepticism
  and re-test under more loaded conditions.

### F_2 native IC: bounds-check elimination in `prove_linear` XOR-fold (commit `54a564b`, −10.1% Prove e2e)
- **What**: in `protocol/src/f2_native_ic.rs::prove_linear`, the two
  hot loops (up_evals at ~line 686, down_evals at ~line 714) read
  `coeffs[d] += &eq_table[row]` per set bit. Each access carried a
  bounds check that LLVM couldn't eliminate (the loop control
  structure doesn't make the bounds provable from rustc's vantage).
  Replaced with `get_unchecked_mut(d)` / `get_unchecked(row)` after
  proving the indices safe:
  - `d < D`: `bp_to_u64::<D>(cell)` masks bits above position `D−1`
    (`BinaryPoly<D>` invariant — all constructors mask; `Mul` is
    `unimplemented!()` so multiplication can't break it).
    `debug_assert!(D == 64 || bits >> D == 0)` guards the invariant.
  - `row < eq_table.len()` and `row < col.evaluations.len()` by the
    outer `for row in 0..usable_rows` bound and the trace shape.
  - In the shifts loop: `i = j − s` with `j ≥ s` and
    `j < s + usable_rows`, so `i < usable_rows`.

  Also hoisted `eq_val = eq_table.get_unchecked(row)` out of the
  inner `while bits != 0` loop and added a `bits == 0 → continue`
  short-circuit so the eq_table load is skipped on zero cells.
- **Why**: the audit's #4 had been dismissed at the artificial
  480-row workload as "1.1% Prove ceiling, not worth it." After the
  realistic-shape bench fix (commit `6e40edd`), the real ceiling
  was 9.6% Prove. The agent's specific proposal (loop inversion +
  bit-d masks) was still wrong (more iterations, big memory cost,
  no efficient NEON masked XOR-reduce — see the entry below in
  "Investigated, didn't help"). But the target was real, and the
  XOR-fold had per-iteration overhead LLVM wasn't eliding.
- **Measured (criterion `--save-baseline` / `--baseline`, nvars=22,
  `--features parallel,simd,unchecked,metal_gpu`)**:

  | Bench                          | Before   | After    | Δ      |
  |--------------------------------|----------|----------|--------|
  | UAIR-a-F2NativeIC (= `prove_linear`) | 446.0 ms | 359.2 ms | **−19.4%** |
  | UAIR-FULL                      | 1.045 s  | 951.6 ms | **−8.9%**  |
  | Prove e2e                      | 3.90 s   | 3.51 s   | **−10.1%** |

  All three criterion-verdict "Performance has improved" with
  p < 0.01. 13/13 F_2 lib tests still pass.
- **Bounds-eliding scope is local**: only the inner XOR-fold's two
  loads. The cell load, the bit-iter primitives (`trailing_zeros`,
  `bits &= bits − 1`), and the GF(2^192) XOR itself are unchanged.
  This kind of micro-tweak shouldn't normally yield 19% — the
  size of the win is a signal that LLVM was paying a real
  per-iteration cost for the bounds checks at the inner-loop's
  unrolled-shape; possibly the bounds check defeated some
  vectorisation or register allocation.

### Scale chained-compression count with `num_vars` (commit `6e40edd`)
- **What**: `NUM_COMPRESSIONS: usize = 7` was a hardcoded constant
  set for the minimum `num_vars = 9`. At every larger `num_vars`,
  the trace still produced only 7 compressions (480 active rows)
  and zero-padded the rest. Replaced with
  `pub const fn num_compressions(num_vars) -> usize`, computed as
  `((1 << num_vars) − 4) / ROWS_PER_COMP` — the largest `N` such
  that `N · 68 + 4 ≤ 2^num_vars`. The witness generator now uses
  `let big_n = cols::num_compressions(num_vars)`.
- **Why this matters for past results**: every prior benchmark in
  this branch (and presumably in `claude/gkr-virtual-cols`, which
  has the same hardcoded constant) was measuring a 480-active-row
  workload at every nvars — 99.99% zero-padded at nvars=22. All
  optimization rankings and impact estimates from the audit were
  derived on that artificial shape. The realistic numbers shift
  the picture substantially.
- **Realistic-workload measurements** (criterion `--features
  parallel,simd,unchecked,metal_gpu`, post-fix):

  | nvars | Old Prove (480 active) | New Prove (realistic) | Active rows |
  |-------|------------------------|-----------------------|-------------|
  | 16    | 30.62 ms               | 47.18 ms (+54%)       | ~65 K       |
  | 20    | 437.7 ms               | 672.1 ms (+54%)       | ~1.1 M      |
  | 22    | 2.251 s                | 4.372 s (+94%)        | ~4.19 M     |

  At nvars=22 component breakdown:
  | Region                      | Time     | % of Prove |
  |-----------------------------|----------|------------|
  | UAIR-FULL                   | 1.004 s  | 23.0%      |
  | └ UAIR-a-F2NativeIC (= `prove_linear`) | 420.4 ms | 9.6% |
  | Commit (micro)              | 768.5 ms | 17.6%      |
  | └ Commit-Fused-GPU-Inline   | 581.6 ms | 13.3%      |
  | Open + misc                 | ~2.60 s  | ~59%       |

  UAIR is now the largest single phase; Open dominates the bucket
  after. `prove_linear` is **9.6% of Prove** — up from 1.1% on the
  artificial shape.
- **Implications for previously-recorded audit verdicts**:
  - The "Investigated, didn't help" entry for the audit's #4
    (`F2NativeIc::prove_linear` loop inversion) was decided on the
    artificial 1.1% number. The target is now 9.6% e2e; the
    specific approach the audit proposed is still wrong (see that
    entry), but `prove_linear` is now a meaningful optimization
    target and needs a different approach. Re-investigation
    pending.
  - The "skip zero rows" alternative I floated alongside #4 is
    **dead** — at realistic shape every row is active.
  - The GPU-cap A/B (commit `31c422b` reverted in `58d2124`)
    used the artificial shape — the +9% regression at nvars=16
    may or may not hold at realistic shape, but the revert was
    correct given the data we had.
  - The "1.5× commit/open regression vs `claude/gkr-virtual-cols`"
    earlier estimate was also on the artificial shape. The
    realistic comparison requires patching the same scaling fix
    onto `claude/gkr-virtual-cols` and re-benching — not done in
    this session.
- **Verification**: 13/13 F_2 protocol lib tests pass at nvars=9
  (where `num_compressions(9) = 7`, unchanged); 19/19 test-uair
  lib tests pass. Compile clean under
  `--features parallel,simd,unchecked,metal_gpu`.

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

### Precompute the `γ'^k·σ^b` weight table in the Hadamard zerocheck comb (rejected)
- **Hypothesis**: the degree-3 Hadamard comb does
  `gpow * sigma_powers[b] * (u·v − w)` = 3 GF(2^128)-mults/term in rounds
  2..n. Folding the two batch challenges into one precomputed
  `weights[k·D + b] = γ'^k·σ^b` (constant across hypercube points) drops it
  to 2 mults/term — expected ~10–17% on Prove-Hadamard.
- **Implementation**: added a flat `weights` field to
  `HadamardRound1FastPath` (replacing `gamma_powers`/`sigma_powers`) and
  built it once in `prepare_hadamard_group_with_fast`, used in both
  `round_1_message` and the `comb_fn`. Value-preserving (the
  `fast_path_matches_generic` byte-identical-proof test still passed; all
  59 piop + 50 protocol lib tests green).
- **Measurement (rigorous Criterion A/B, nvars=16, same machine state via
  `git stash` + `--save-baseline`/`--baseline`)**: change
  **[+1.5% +6.9% +15.4%] (p=0.03) — regressed.** Reverted.
- **Why it didn't help**: rounds 2..n are **memory-bandwidth bound**, not
  arithmetic-bound. The generic multi-degree sumcheck rebuilds a
  1537-element value array per hypercube point (the eq MLE + 1536 slices);
  that traffic dominates. Trading a cheap CLMUL field-mult for a load from
  a larger 512-entry `weights` table (vs the tiny, cache-resident 16-entry
  `gamma_powers` + 32-entry `sigma_powers`) is net-neutral-to-negative.
  **Lesson / culprit for the discharge cost**: the real bottleneck is the
  generic per-point value-array machinery over 1536 polynomials — only a
  *specialised* Hadamard sumcheck that folds slices in place (avoiding that
  rebuild) or a smaller polynomial count would move Prove-Hadamard; comb
  micro-opts won't. (Phase-1 round-1 fast path already bypasses the rebuild
  for round 1, which is why it — not this — was the win.)

### Fast u64 scatter in `scatter_matrix_into_gpu_slab` for F2PackU64 cells (rejected)
- **Hypothesis (from optimization survey)**: the per-cell write in
  `zip-plus/src/merkle.rs::scatter_matrix_into_gpu_slab` goes through
  `cell.write_transcription_bytes_exact(dst)`, which for
  `BinaryU64Poly<D>` expands to
  `dst.copy_from_slice(&value.to_le_bytes())`. Replacing it with
  `ptr::write_unaligned::<u64>(dst.cast(), cell.pack_u64().to_le())`
  removes the slice creation and the `[u8;8]` intermediate.
  Estimated impact: 2–5% Commit wall time (revised down from the
  audit's 5–15%).
- **Implementation**: added a `FastScatterCell` trait in
  `zip-plus/src/merkle.rs` with a default body using
  `write_transcription_bytes_exact`, plus an override for
  `BinaryU64Poly<D>` using direct unaligned u64 write. Threaded
  through `ZipTypes::Cw` as a bound; default-body impls for the
  other Cw types (`BinaryRefPoly<D>`, `DensePolynomial<R, D>`,
  `i128`, `Int<K>`). Bound propagation also required adding
  `FastScatterCell` to the generic `CwR` bound in
  `protocol/benches/e2e.rs:220-227`.
- **Measurement (criterion `--save-baseline` / `--baseline`,
  nvars=22, `--features parallel,simd,unchecked,metal_gpu`)**:

  | Bench                              | Before   | After    | Δ      | criterion verdict |
  |------------------------------------|----------|----------|--------|-------------------|
  | Commit-Fused-GPU-Inline/nvars=22   | 584.19 ms | 586.67 ms | +0.42% | no change (p=0.19) |

  Reverted. LLVM was already optimising the
  `to_le_bytes`+`copy_from_slice` pattern down to equivalent
  instructions; the architectural cost (new trait + impls in
  4 files + bound propagation through `ZipTypes::Cw`) wasn't
  justified by zero measured benefit.
- **Lesson**: when the optimization is "remove indirection LLVM
  should already see through," prefer measurement before
  architectural changes. The audit's 5–15% estimate was
  speculative — LLVM's IR-level optimisation of small fixed-size
  byte copies is mature on both x86_64 and aarch64.

### `F2NativeIc::prove_linear` loop inversion + bit-d masks (rejected — wrong approach; superseded by `54a564b`)
- **Status note**: the *target* (`prove_linear`'s XOR-fold) turned
  out to be a real optimization opportunity at the realistic
  workload — see the "F_2 native IC: bounds-check elimination in
  `prove_linear` XOR-fold" entry under Shipped work (`54a564b`,
  −10.1% Prove e2e). The agent's specific *approach* (loop
  inversion + precomputed bit-d masks) is still rejected for the
  reasons below; bounds-check elimination on the existing
  sparse-iter shape gave the win instead.
- **Hypothesis (from optimization survey)**: the per-bit
  `coeffs[d] += &eq_table[row]` accumulation in
  `protocol/src/f2_native_ic.rs:686-705` (and the analogous shifts
  loop `:715-742`) could be rewritten with `d` as the outer loop
  and `i` as the inner, using a precomputed `bit_d_mask: Vec<u64>`
  per column. Claimed impact: 5-10% UAIR/IC.
- **Why the claim doesn't hold up**:
  1. The current code already short-circuits on `bits == 0` via
     `while bits != 0 { ... }` — zero cells contribute nothing.
     For random data (~50% set bits per cell), the inner loop
     runs ~16 times per cell.
  2. The proposed inversion iterates ALL `(d, i)` pairs:
     `32 × usable_rows` per column. Strictly MORE iterations than
     the current sparse pattern (~16 × usable_rows for random
     data; far less when cells are zero).
  3. Per-column bit-d masks are large at scale: at nvars=22
     they'd be `usable_rows / 8 × D = 4M/8 × 32 = 16 MB per col`,
     ~450 MB total across 28 cols. Cache-hostile.
  4. The proposed SIMD vectorisation needs an efficient
     "masked XOR-reduce" primitive; NEON (Apple Silicon) lacks a
     direct one. AVX-512's `vpternlogq` could help on x86 but
     doesn't apply to the target platform.
- **Measurement (criterion at nvars=22, --features parallel,simd,
  unchecked,metal_gpu)**:

  | Micro                          | Time     | % of UAIR-FULL | % of Prove e2e |
  |--------------------------------|----------|----------------|----------------|
  | UAIR-a-F2NativeIC (`prove_linear`) | 24.5 ms | 5.8%           | ~1.1%          |
  | UAIR-FULL                      | 424.5 ms | 100%           | ~19%           |
  | Prove (for context, nvars=22)  | 2.25 s   | —              | 100%           |

  Even a *complete* elimination of `prove_linear` time would only
  yield ~1.1% Prove e2e — far below the 5-10% the survey
  estimated. A realistic rewrite that saves, say, 50% of
  `prove_linear`'s time is ~0.5% e2e: too small to justify the
  added complexity.
- **Adjacent opportunity (not pursued either, same code)**: at
  `nvars=22` only ~476 of 4M rows (= 7 compressions × 68 rows
  per comp) carry non-zero cells; the remaining ~99.99% hit the
  `while bits != 0` short-circuit but still pay loop-control +
  cell load + `bp_to_u64`. Plumbing an "active row hint" through
  the Uair trait so `prove_linear` iterates only `0..active_rows`
  would skip ~117 M empty iterations across all cols. Upper bound
  on savings: still ≤ `prove_linear`'s 24 ms, so ≤ 1% Prove e2e.
  Tracked here but not implemented in this session — the payoff
  is small and the abstraction change touches `Uair`/`UairSignature`.

### Metal Blake3 dispatch: lifting the `min(256)` threadgroup cap (rejected on artificial trace — re-shipped on realistic shape; see `e4d6a25` under Shipped work)
- **Hypothesis**: the hard-coded `min(256)` cap on
  `threads_per_threadgroup` in
  `zip-plus/src/metal_gpu/mod.rs:201-214` was under-occupying Apple
  Silicon GPUs at SHA-256 F_2 nvars≥16 where `num_cols` is in the
  10⁴–10⁵ range. Lifting the cap so the pipeline's own
  `max_total_threads_per_threadgroup()` wins would (per the
  optimization-survey agent's estimate) give 10–30% Blake3 throughput
  improvement.
- **Test**: removed the `.min(256)` clamp; the dispatch now uses
  `(max_threads / exec_width) * exec_width` directly. Both
  `gpu_blake3_matches_cpu_*` equivalence tests and 13/13 F_2 lib
  tests still passed.
- **Measurement (A/B with criterion `--save-baseline` /
  `--baseline`,** target `Zinc+ F_2 SHA-256/Prove/nvars=(16|20|22)`,
  10 samples + warmup, `--features parallel,simd,unchecked,metal_gpu`):

  | nvars | Before (cap=256) | After (cap removed) | Δ      | criterion verdict |
  |-------|------------------|---------------------|--------|-------------------|
  | 16    | 30.62 ms         | 33.12 ms            | +9.34% | regressed         |
  | 20    | 437.7 ms         | 434.3 ms            | −0.76% | within noise      |
  | 22    | 2.251 s          | 2.245 s             | −0.24% | no change         |

  Net: clear regression at the smallest GPU-active nvars, neutral
  at larger nvars.
- **Why it didn't help**: the assumption that the cap was the
  binding constraint was wrong. At nvars=16 the commit produces only
  ~4096 leaves (= 4096 work-items). With threadgroup_size = max
  (1024 on M-series), that's 4 threadgroups total — too few to
  occupy all GPU cores. The original 256-thread cap produced 16
  threadgroups at nvars=16, balancing per-threadgroup occupancy
  against having enough threadgroups for the dispatcher to spread
  across cores. At nvars=20+ both threadgroup sizes produce enough
  threadgroups to saturate the GPU, so the cap is invisible. The
  256 was empirically well-tuned.
- **Lesson**: don't lift kernel-dispatch caps without measuring at
  the smallest expected workload size — Apple's
  `max_total_threads_per_threadgroup` reports a *per-threadgroup
  ceiling*, not the optimal threadgroup size for a given total work
  count. The two interact via cores-per-GPU.
- **If revisited**: a size-dependent rule could potentially do
  better than either fixed choice — e.g. clamp so
  `num_threadgroups ≥ 2 × core_count`, then use the largest
  threadgroup that satisfies that. Not pursued in this session.

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

### Small-value / multi-round-skip prover for the Hadamard zerocheck (the real Binius64-style lever; scoped, NOT built)
- **Source**: Dao–DeStefano–Bagad–Domb–Thaler, "Speeding Up Sum-Check
  Proving" (cs.nyu.edu/~zd2131/papers/26-587.pdf, Mar 2026), §3 (tower-field
  small-value arithmetic), §5 (small-value prover + cost), §6 (eq-poly
  optimisation). Same regime Binius64's AND reduction uses (univariate skip,
  binius.xyz/blueprint/backend/ands). §2.1 explicitly names GF(2^128)/GF(2)
  (Binius) as a target.
- **Idea**: our Phase-1 round-1 fast path is the `v=1` case of the
  small-value prover — round 1 runs on packed 0/1 bits, then `fold_with_r1`
  materialises half-size F-slices and the generic sumcheck takes rounds
  2..n. Extending to `v>1` keeps the first `v` rounds on packed bits and
  defers materialisation, which (a) shrinks the peak F-slice memory
  geometrically (`1536 · 2^(μ-v) · 16 B`: nvars=20 v=1→12 GB, v=2→6 GB,
  v=5→0.75 GB — i.e. it's what actually clears the nvars≥20 swap), and
  (b) moves `v` rounds off the memory-bound generic value-array rebuild.
- **Mechanism (Dao §5.1)**: don't bind `X_1..X_v` one round at a time
  (each binding makes the slices large via the random challenge). Instead
  compute the `v`-variate **prefix polynomial**
  `q(X_1..X_v) = Σ_{x'} comb(X_1..X_v, x')` over the grid `{0,1,X,X+1}^v`
  (the GF(2^128) boundary points `F::from_with_cfg(0,1,2,3)`) from the 0/1
  base evals, then read round `i`'s message off `q` and interpolate
  `q(r_1..r_{i-1}, X_i, ·)` for the bound prefix. After `v` rounds, fold to
  the `2^(μ-v)`-size F-slices.
- **Cost / payoff**: speedup `Θ((d²κ)^{1/δ})`, `δ=log₂(d+1)=2` for our
  `d=3`, so ≈ `3√κ` where `κ = cost(bb)/cost(ss)`. Their measured 10.9×
  (Spartan) is over **256-bit prime fields** (`bb`≈40-100 cyc). We're over
  **GF(2^128)** where `bb` (CLMUL) is ~5-15 cyc → `κ` is small → realistic
  **~2-5×**, mostly a **memory** win at nvars≥19. The §6 eq-optimisation
  (we have the `eq_r` factor) stacks on top. A *naïve* prefix does ~1.33×
  the arithmetic of generic rounds 1+2, so it **regresses small nvars** —
  must be **size-gated** (`v=1` for production nvars=9; `v>1` only when the
  memory benefit dominates). The efficient multiproduct (Dao Procedure 1,
  `O(d log d)` vs `O(d²)` bb) is what keeps `v>1` from regressing; needed
  for a real win.
- **Why it's a real build, not an increment**: the multi-degree sumcheck's
  `Round1FastPath` hook (`piop/src/sumcheck/multi_degree.rs`) skips **one**
  round only (`round_1_message` + `fold_with_r1`). Multi-round skip needs a
  framework change (loop the fast path for `v` rounds, or a "prove from
  round v+1" entry) **plus** the prefix-polynomial machinery (Procedures
  1+2) **plus** the eq-opt — a multi-component effort touching shared piop
  code. Verifier/proof are **unchanged** (same protocol), so the existing
  `fast_path_matches_generic` byte-identity test extends to gate it — the
  one big de-risk. (NB this is the verifier-preserving small-value variant,
  NOT Binius64's univariate skip, which changes the proof/degree.)
- **Recommended first increment**: a unit test that computes the `v=2`
  prefix `q` and asserts its derived round-1+round-2 messages equal a
  generic 2-round run's `group_messages` (validates the math before any
  framework surgery), then the framework `v`-round-skip hook, then size-gate
  + A/B at nvars=16/20.

### Round-1 fast path for the Hadamard degree-3 zerocheck (Phase 1 — ✅ SHIPPED; see Shipped-work entry for results)
- **Where**: `piop/src/lookup/hadamard.rs` (`prepare_hadamard_group` builds
  the group via `MultiDegreeSumcheckGroup::new(3, …)` — no fast path) and
  `protocol/src/f2_hadamard.rs::prove_f2_hadamard_phase` (calls
  `build_all_operand_slices`, which materialises every operand bit-slice).
- **Root cause being fixed**: the discharge zerocheck materialises **1536
  bit-slices** (16 relations × 3 operands × 32 bits) as full-width
  `GF128::Inner` (= `Uint<2>`, 16 B) MLEs → 24 GB at nvars=20 (the
  super-linear prove blowup; see the "Hadamard discharge cost" shipped
  entry). Round 1 also touches the full (μ)-var hypercube.
- **Fix (proposed approach)**: port the booleanity degree-3 round-1 fast
  path (`piop/src/lookup/booleanity.rs::BooleanityRound1FastPath`) to the
  Hadamard group. The fast path reads the **packed operand columns**
  (`DenseMultilinearExtension<BinaryPoly<D>>`, 4 B/row — the `op_mask` that
  `build_operand_slices` computes internally, and `build_adder_operand_columns`
  already produces) instead of materialising 16-B F-slices; round 1 does
  boolean arithmetic and `fold_with_r1` emits half-size folded F-slices for
  rounds 2..n. The Hadamard round poly factors as
  `M(t) = eq(t,r₀)·Σ_{b'} E_other(b')·G(t,b')` with
  `G(t,b') = Σ_k γ'^k Σ_b σ^b (U_b(t)·V_b(t) + W_b(t))` (char-2), and
  `G = 0` on the honest hypercube ⇒ `M(0)=M(1)=0`, so only `M(2),M(3)` need
  computing (mirrors booleanity's `p1(1)=0`). Each operand bit folds by the
  `{0,1}² → {0,t,1−t,1}` lookup.
- **CRITICAL field subtlety**: the booleanity fast path computes its
  boundary points as `one+one` / `one+one+one` — correct only for the
  **large prime field** of the integer/Q[X] pipeline (where booleanity
  lives). Over `GF(2^128)` (char 2) those collapse: `2 = 0`, `3 = 1`. The
  multi-degree sumcheck instead uses `F::from_with_cfg(k)`, which under the
  bit-pattern convention is the field element `X` for `k=2`, `X+1` for `k=3`
  (distinct) — see `prover.rs:170-196` and `F2EqColRound1FastPath`
  (`f2_prove.rs:639`, the degree-2 GF128 template). The Hadamard fast path
  MUST use `F::from_with_cfg(2/3)` and `eq(t,r₀)=(1−t)(1−r₀)+t·r₀`, not the
  booleanity integer shortcut.
- **Correctness gate**: a test that runs the same zerocheck through the
  generic `new(3,…)` group and the `with_round_1_fast` group and asserts an
  **identical** `MultiDegreeSumcheckProof` (the `Round1FastPath` contract:
  `round_1_message` must equal what a faithful generic round-1 would emit).
- **Expected gain**: ~2× peak memory (round-1 24 GB materialise removed;
  `fold_with_r1` still emits ~12 GB of half-size slices at nvars=20) + the
  round-1 boolean-arithmetic win. Does NOT fully kill the nvars≥20 blowup —
  that needs **Phase 2** below.

### Phase 2 — cut the Hadamard slice count via `(col,Δ)` dedup
- **Where**: same path. The 16 relations' operands reference a small set of
  underlying columns with overlap (e.g. `W_E` feeds C12/C13/C14; the adder
  chains reuse `W_W`, `W_T1`, …), but `build_all_operand_slices` materialises
  slices **per operand occurrence**, not per distinct `(col,Δ)`.
- **Fix**: restructure the comb to operate on per-`(col,Δ)` slices with the
  XOR/complement structure encoded in the comb coefficients, so each distinct
  pair's slices are built once. Reduces memory + compute proportionally to the
  dedup factor; pairs with Phase 1 to make nvars≥20 viable.

### Fast u64 scatter in `scatter_matrix_into_gpu_slab` (rejected on second-pass measurement)
- **Status**: implemented in a temporary branch state via a new
  `FastScatterCell` trait (default body using
  `write_transcription_bytes_exact`, override for `BinaryU64Poly<D>`
  using `ptr::write_unaligned::<u64>(dst, self.pack_u64().to_le())`),
  added as a bound on `ZipTypes::Cw`. Compiled clean, 13/13 F_2 lib
  tests passed, but criterion A/B showed **no measurable change**
  — reverted. See entry under "Investigated, didn't help" below.

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

### ψ_α-projected sumcheck term for Hadamard discharge — proves convolution, not coefficient-wise (design pass on step 5)
- **What (proposed)**: leave the 16 Hadamard relations unprocessed
  in the IC (already the state on this branch — C12–C14 are
  witness-only, the adds pin only the LSB via the `(X)` ideal), then
  *after* the ψ_α projection add one extra batched-sumcheck term
  `∑_x (ψ(v)·ψ(u) − ψ(w))·eq(x, r) = 0`, reusing the IC point `r`
  (already in `GF(2^128)^μ`, so the proposer's "ψ(r)" is a no-op).
  Attractive because it reuses the already-projected trace and the
  existing degree-2 sumcheck + lift-and-project open with near-zero
  extra commitment — would be the cheapest possible discharge.
- **Why it does not work as written**: ψ_α is an `F_2`-algebra
  homomorphism `F_2[X] → GF(2^128)`, so `ψ(v)·ψ(u) = ψ(v·u)` with
  `·` the **polynomial / convolution** product in `F_2[X]`. The
  SHA Hadamards (AND, Maj, and the Binius adder identity
  `(â + X·ĉ) ⊙ (b̂ + X·ĉ) = ĉ + X·ĉ`) are **coefficient-wise** (`⊙`),
  *not* convolution. So even for an honest prover,
  `ψ(v)·ψ(u) − ψ(w) ≠ 0` when `w = v ⊙ u`. Counterexample:
  `v = u = 1 + X` gives `ψ(v)·ψ(u) = (1+α)² = 1 + α²` (char 2),
  but `ψ(v ⊙ u) = ψ(1 + X) = 1 + α`, and `1 + α² ≠ 1 + α` for
  `α ∉ F_2`. The term proves the wrong relation (per-row
  convolution). The Binius carry identity in particular *requires*
  per-coefficient idempotency `c_i² = c_i`, which convolution (`ĉ²`)
  destroys — so convolution can never stand in for the adder ⊙.
- **Corrected direction**: the sumcheck *shape* is right; the fix is
  to NOT collapse the bit-axis with ψ_α for the Hadamard columns.
  Expand the `D = 32` coefficient positions into 5 extra sumcheck
  variables, so `V, U, W` become MLEs over `{0,1}^{μ+5}` and
  `W(x,b) = V(x,b)·U(x,b)` is a genuine elementwise product; then
  `∑_{x,b} (V·U − W)·eq((x,b),(r,ρ)) = 0` is sound + complete (the
  standard Binius AND zerocheck). Cost: the open must expose
  **bit-level MLE evaluations** at a `(μ+5)`-point — a multilinear /
  eq functional of each cell's bits, not the α-power functional ψ_α —
  same lift-and-project machinery, different weights; and the
  Hadamard sumcheck has arity `μ+5`, so it likely runs as a
  *separate* sumcheck (groups in `MultiDegreeSumcheck` share
  `num_vars`) rather than folded into the μ-var eq·col group. The
  row-axis of its eq can still reuse the IC's `r`, so the "reuse r"
  instinct survives.
- **Correction-polynomial variant (also rejected)**: compute
  `f = ∑_x (u_x·v_x − w_x)·eq(x,r)` in the X-domain during the IC (a
  degree-≤62 poly in X, cheap to send), then check
  `∑_x (ψ(u)ψ(v) − ψ(w))·eq(x,r) = ψ(f)` in the projected sumcheck.
  Fails: the LHS *is* `f★(α)` for `f★ = ∑_x (u_x·v_x − w_x)·eq(x,r)`
  the committed columns' true convolution discrepancy, so the check
  only forces `f = f★` (SZ over α) — `f` is pinned to whatever the
  committed `w` produces, making the term a tautology that constrains
  `w` not at all. Deeper reason: `ψ(u)ψ(v)` only ever exposes the 63
  convolution coefficients `c_n = ∑_{j+k=n} u_j v_k`; the AND diagonal
  `∑_i u_i v_i` is provably not a function of them (it lives on the
  main diagonal the product polynomial doesn't carry), so no X-domain
  `f` — itself just another convolution object — can recover it. The
  coefficient index must be a live sumcheck variable. (If `f` is read
  as the coefficient-wise `∑_x (u_x⊙v_x − w_x)·eq`, the IC cannot even
  form it — ⊙ is not a ring op in F_2[X] — and it collapses to the
  original broken `=0`.)
- **General impossibility (covers the "check f's F_2-coefficients
  are 0" variant and every X-domain variant)**: everything the
  ψ/IC view exposes about a Hadamard pair is a function of the
  *convolution* `u·v` (and of `w`). Convolution does not determine
  the Hadamard. Witness: `u = 1+X, v = 1+X³` vs `u' = 1+X², v' =
  1+X+X²` have the **same** product `u·v = u'·v' = 1+X+X³+X⁴`, but
  **different** coefficient-wise products `u⊙v = 1` vs `u'⊙v' =
  1+X²`. Put `w := 1`: the honest instance `(u,v,w)` (here `w=u⊙v`)
  and the false instance `(u',v',w)` (here `w≠u'⊙v'`) produce
  *identical* `ψ(u)ψ(v)`, `ψ(w)`, and `f = ∑(u·v−w)·eq` — every
  coefficient, including the F_2 parts. No predicate on those can
  accept one and reject the other, so no such check is sound.
  Separately, "f's F_2-coefficients = 0" also breaks **completeness**:
  the honest `f★` is the (nonzero) cross-term discrepancy and its
  F_2-components are generically nonzero, so it rejects honest
  provers. Bottom line: the coefficient index must be a live
  sumcheck variable; convolution-domain tricks cannot recover the
  diagonal.
- **Status**: semantics confirmed = coefficient-wise / bitwise AND
  (case 3). **Chosen design = per-coefficient-slice zerocheck,
  booleanity-style** (full plan: `protocol/src/f2_hadamard_plan.md`).
  Run a μ-var degree-3 zerocheck
  `∑_x eq(x,r)·∑_k∑_b (γ')^k σ^b (U_{k,b}·V_{k,b} − W_{k,b}) = 0` over
  the bit-slice MLEs, reusing the booleanity infrastructure
  (`piop/src/lookup/booleanity.rs`: `build_shifted_bit_slice_mles`,
  `build_virtual_booleanity_mles`, `finalize_booleanity_*`,
  `verify_bit_decomposition_consistency`) with the self-product
  `v(v−1)` swapped for the cross-product `U·V − W`. The 32 slice evals
  per column ride **one** column opening via the recombination check
  `∑_b a^b v_b(r*) = parent_eval` — no per-slice openings.
  - **F_2-specific subtlety**: the recombination element must be fresh
    *after* the bit-slice evals (SZ over a degree-31 poly in `a` pins
    all slices). The integer path reuses its early Step-3 projection
    (`prover.rs:520`) and is sound only because its bit-slices also feed
    the whole CPR; our Hadamard bit-slices are touched only by this
    check, so they need their own fresh binding. Fix: either run the
    Hadamard sumcheck before α (reuse α, Wiring R) or sample a fresh `a`
    + a second `ψ_a` open (Wiring F). Both reuse the existing open.
  - **Alternative (documented, not chosen)**: bit-axis expansion — a
    (μ+5)-var zerocheck folding the bit axis, discharged by an
    eq-over-bits open. Smaller proof (one bit-MLE eval/col vs D
    slice-evals/col) but needs a new sumcheck arity + a new per-cell
    open contraction. Revisit if proof size dominates.
  - **Prereqs**: `W_β` carry column (absent on this branch; needed for
    the 13 adders, not the 3 ANDs); missing `W_E ↓1,↓2` / `W_A ↓1,↓2`
    shifted-bit-slice specs; row-shift discharge for shifted operands
    (shared with Issue 1 below).
  - Supersedes the bare "add one ψ-projected term" idea for AND/adder.
  - **Progress (Wiring R chosen)**: A0 landed — the piop cross-product
    zerocheck at `piop/src/lookup/hadamard.rs`
    (`prepare/finalize_hadamard_{group,prover,verifier}` +
    `HadamardTriple`), mirroring `booleanity.rs` with the self-product
    `v(v−1)` swapped for `U·V − W`. Degree-3 single group, γ'/σ-batched.
    Unit test passes (honest `W=U⊙V` accepted; flipped `W` ⇒ non-zero
    claimed sum, rejected by the `claimed_sum==0` gate) and is
    clippy-clean. Committed `a110d32`.
  - **Progress: A1 core landed** (`ade7ab9`) —
    `protocol/src/f2_hadamard.rs`: `prove_f2_hadamard_phase` /
    `verify_f2_hadamard_phase` + `F2HadamardSpec`. Builds Δ=0 bit-slices
    for the distinct referenced columns via `compute_bit_slices_flat`
    (NOT `build_shifted_bit_slice_mles` — that asserts shift ≠ 0;
    `uair/src/lib.rs:259`), runs the degree-3 group, and exposes
    per-slice evals tied to the committed columns by
    `verify_bit_decomposition_consistency`. Tests over real
    `BinaryPoly<D>` columns: honest round-trip incl. the
    `Σ_b α^b v_b(r*_H)` recombination at a test α, corrupt-W rejected,
    empty-specs no-op. Requires the `parallel` feature (the F_2 path's
    round-1 fast path uses rayon `reduce`).
  - **A1 wiring DONE (working tree, NOT committed)**: the Hadamard
    zerocheck phase is threaded into `prove_f2_uair_with_groups` /
    `verify_f2_uair_with_groups` *before* α (Wiring R), reusing
    `ic_state.evaluation_point` as `r`. Added `F2Proof.hadamard_proof:
    Option<F2HadamardProof>`, a `hadamard_specs` param on both group
    fns (existing callers pass `&[]` → no-op), and two
    `F2VerifyError` variants (`MissingHadamardProof`, `Hadamard`). New
    e2e test `hadamard_phase_roundtrips_in_flow` (3-col `HadF2Uair`,
    trace with `W = U⊙V`, driven through the real group fns via
    `F2Types<D>`): honest round-trip accepts, a flipped `W` bit is
    rejected with `F2VerifyError::Hadamard`. **All 37 protocol lib
    tests pass** (`--features parallel`).
    - **Not committed**: `f2_prove.rs` carries ~600 lines of
      pre-existing uncommitted GF128-refactor WIP from before this
      session; the ~80 Hadamard lines are interleaved and can't be
      isolated without interactive staging. Commit needs the owner to
      either land the WIP or accept a bundled commit. (A0 `a110d32`
      and A1-core `ade7ab9` were committable because they were new /
      clean files.)
  - **A3/A4 discharge wired in-flow (trusted; working tree)**: the
    prover computes each Hadamard column's α-eval at `r*_H`
    (`alpha_parent_evals`, committed `390c846`), absorbs them, and ships
    them in `F2Proof.hadamard_parent_evals`; the verifier recombines
    `Σ_b α^b·v_b == parent_eval` (`verify_bit_decomposition_consistency`)
    after α (Wiring R, so α is fresh w.r.t. the bit-slice evals). The
    in-flow round-trip test now exercises the recombination; all 37
    protocol lib tests pass. The `f2_prove.rs` threading is still
    uncommitted (entangled with the GF128 WIP — option 1: lands after
    the WIP).
  - **SOUND DISCHARGE SHIPPED (Approach A — SUPERSEDED by Approach B
    below; kept for the soundness rationale)**: the trusted `parent_evals`
    are now PCS-bound. Full-pipeline change in `protocol/src/f2_prove.rs`:
    - **`F2FullProof`** gains four fields (all `None`/empty without
      Hadamard): `hadamard_multipoint_eval: Option<MultipointEvalProof>`,
      `hadamard_evals_at_rstar_h: Vec<Gf>` (every projected column's eval
      at `r*_H` — the second mp's `up_evals`),
      `hadamard_open_evals_at_r0h: Vec<Gf>` (evals at `r_0^H`),
      `hadamard_open: Option<F2OpenProof<D>>`.
    - **New entry points** `prove_f2_full_with_hadamard` /
      `verify_f2_full_with_hadamard`. The existing
      `prove/verify_f2_full_with_bit_ops` (+ pre-paired) signatures are
      **unchanged** — all ~15 bench call sites untouched. The post-commit
      body was factored into private `prove_f2_full_impl` /
      `verify_f2_full_impl(.., hadamard_specs)`; the old entries call them
      with `&[]` (this also de-duplicated the prove/pre-paired bodies that
      were copy-pasted before).
    - **The discharge**: after the main mp+open at `r*`/`r_0`, when
      `hadamard_specs` is non-empty, a **second** `MultipointEval` collapses
      *every* projected column's eval at `r*_H` to a fresh `r_0^H`, then a
      second `prove_f2_open` opens the **same contiguous witness slice** at
      `r_0^H` (collapsing all columns — not just the Hadamard subset —
      reuses the witness-slice open verbatim and dodges subset-column
      mapping; the parent-evals are then a subset of the bound
      `evals_at_rstar_h`). Verifier mirrors and adds the **binding check**
      `hadamard_evals_at_rstar_h[distinct] == uair.hadamard_parent_evals`,
      which upgrades the kept-as-is in-flow recombination to sound.
    - **Soundness-critical detail (caught during impl)**: the second mp's
      `up_evals` (`hadamard_evals_at_rstar_h`) **must be absorbed into the
      transcript before the second mp samples its γ** — exactly as
      `prove_f2_uair_with_groups` absorbs `column_evals_at_rstar` before the
      main mp. Without it a prover seeing γ first could fit fake per-column
      `r*_H` evals to the γ-batched sum while keeping the witness slice
      honest, defeating the binding; with it, SZ over the fresh γ pins
      every up-eval. Added to both prover and verifier.
    - **Soundness chain**: second open binds `open_evals_at_r0h[witness]`
      to the commitment; mp `verify_subclaim` ties `open_evals_at_r0h`
      (all cols, public via local recompute / virtual via derive / witness
      via open) back to `evals_at_rstar_h`; the up-eval absorb + SZ pins
      each `evals_at_rstar_h[j]`; the binding check then ties the trusted
      `parent_evals` (hence the bit-slices, via the recombination + the
      zerocheck) to the committed columns. Works for both witness and
      public distinct columns.
    - **Tests**: new e2e `prove_then_verify_f2_full_with_hadamard_roundtrips`
      (HadF2Uair, 3-col `W=U⊙V`, driven through the `_with_hadamard`
      entries): honest round-trip accepts (and asserts the binding equality
      holds by construction), and the reachable tampers are rejected —
      flipped `W` → zerocheck, swapped parent-eval → recombination, dropped
      second open → `MissingHadamardDischarge`, tampered `r*_H` eval →
      second mp. **38 protocol lib tests pass** (`--features parallel`);
      `f2_prove.rs` clippy-clean. The `HadamardParentEvalBindingMismatch`
      variant is defence-in-depth (only a malicious prover shipping
      fake bit-slice MLEs decoupled from the commit can reach it — the
      recombination + zerocheck over-determine the slices, so no single
      tamper of an honest proof reaches it without tripping an earlier
      gate).
    - **Helper extracted**: `recompute_public_col_evals_at::<D>` — the
      public-col α-project + MLE-eval recompute, now shared by all three
      verifier points (`r*`, `r_0`, `r_0^H`).
  - **SOUND DISCHARGE REWORKED → Approach B (two-point multipoint-eval;
    working tree on `f2-clean`)**: Approach A was the user-identified
    "wrong turn" — it opened the witness slice **twice** (a separate
    `r*_H` mp+open) and leaned on a `parent_eval == evals_at_rstar_h`
    binding check. Approach B folds the Hadamard claims into the **main**
    mp instead, exactly as requested: each AND pair's claim
    `MLE[v](r*_H)` enters the *single* multipoint-eval as a **pointed
    shift** and comes out as a claim on `MLE[v]` at the shared `r_0`,
    unifying every `MLE[v]` evaluation across points under **one** open.
    - **New piop primitive** (`piop/src/multipoint_eval.rs`, additive —
      the integer mp is untouched): `PointedShiftClaim { point, shift,
      source_col }` + `prove/verify_as_subprotocol_with_pointed_shifts`
      + `verify_subclaim_pointed`. A pointed shift carries its **own**
      point (here `r*_H`) and a `shift` Δ; `shift = 0` ⇒ the shift
      predicate is `eq` ⇒ a plain point-claim, so Δ=0 AND pairs fold as
      point claims and Δ≠0 pairs (row-shifted operands) fold via the
      shift predicate at `r*_H` — **closing the AND row-shift soundness**
      (ledger Issue 1) for the AND relations in the same pass. Unit test
      `two_point_pointed_shifts_roundtrip` passes (17 piop mp tests green).
    - **`F2FullProof` slimmed**: the four `hadamard_*` fields
      (`hadamard_multipoint_eval`, `hadamard_evals_at_rstar_h`,
      `hadamard_open_evals_at_r0h`, `hadamard_open`) are **removed**. The
      existing `multipoint_eval` + `open` + `open_evals_at_r_0` now carry
      the folded `r*_H` claims; the prover feeds `uair.hadamard_pair_evals`
      as the pointed shifts' `down_evals`, the verifier rebuilds the same
      `PointedShiftClaim`s from `subclaim.hadamard_pairs`/`hadamard_rstar`
      and checks them with `verify_subclaim_pointed(open_evals_at_r_0,
      pointed_shift_sources)`. Seven now-dead `F2FullVerifyError` variants
      removed (`MissingHadamardDischarge`, `Hadamard{EvalsAtRstarH,OpenEvals}
      LengthMismatch`, `Hadamard{Public,Virtual}ColumnEvalMismatchAtR0H`,
      `HadamardParentEval{CountMismatch,BindingMismatch}`).
    - **Why sound without the binding check**: the AND pair-evals
      (`hadamard_pair_evals`) are still the recombination's inputs (corrupt
      → `HadamardRecombination`) AND now the mp's down-terms folded into the
      single `r_0` open — SZ over the mp's γ pins them to the committed
      columns directly, so the separate A-style binding equality is no
      longer needed. Adders stay trusted (their computed-β operands are
      virtual/non-column, so they can't fold as pointed shifts).
    - **Tests**: `prove_then_verify_f2_full_with_hadamard_roundtrips`
      rewritten for B (tampers: flipped `W` → zerocheck, swapped
      parent-eval → recombination, tampered `open_evals_at_r_0` → the
      two-point mp subclaim check); `_with_operand_hadamards_roundtrips`'s
      honest path now exercises the Δ≠0 pointed shifts end-to-end. All 19
      protocol `f2_prove` tests + the SHA-256 e2e tests pass
      (`--features parallel`); `multipoint_eval.rs`/`f2_prove.rs`
      clippy-clean (the only `arithmetic_side_effects` hits are pre-existing
      in the `poly` GF128/GF192 code, out of scope).
  - **Identified, not done (optimisations on the sound discharge)**:
    (a) ~~the second mp+open opens the witness slice a **second** time~~
    **DONE via Approach B above** — folded into one two-point mp + one
    open. (b) The
    `prove_f2_full_impl` / `verify_f2_full_impl` Hadamard blocks duplicate
    the main mp+open block (~60 lines each) — a `*_mp_and_open_at_point`
    helper would DRY it but threads ~13 params; left inline for review
    clarity. (c) `{0,1}`-slice round-1 fast path for the degree-3 zerocheck.
  - **OPERAND MACHINERY SHIPPED (Phase B — row-shift / XOR / complement,
    working tree on `f2-clean`)**: `F2HadamardSpec` generalised from three
    plain column indices to three **operands**. An operand is the bitwise
    XOR of `(col, ↓Δ row-shift)` terms, optionally bitwise-complemented
    (`1−`). This covers the SHA AND relations C12–C14 (Appendix A):
    `W_E ⊙ W_E^↓1`, `(1−W_E) ⊙ W_E^↓2`, and the Maj combo
    `(W_A + W_A^↓2) ⊙ (W_A^↓1 + W_A^↓2) = (W_MAJ + W_A^↓2)`.
    - **`+`/`1−` are F_2[X] addition (bitwise XOR), NOT F-linear sums** —
      so operand bit-slices are built by XOR-ing source bits at the bit
      level (`build_operand_slices`, a `{0,1}`-valued F MLE), *not* via the
      integer prover's `build_virtual_booleanity_mles` (which does
      F-addition and would give `2` for `1+1`). **Correction to plan §5.1**,
      which suggested reusing `build_virtual_booleanity_mles` — wrong for
      AND operands. The zerocheck itself
      (`piop/src/lookup/hadamard.rs`) is operand-agnostic: it just needs `D`
      slices per operand, laid out `U_0,V_0,W_0,U_1,…`.
    - **Discharge** (`f2_hadamard.rs`): because ψ_α is an F_2-algebra hom,
      `ψ_α(A ⊕ B) = ψ_α(A) + ψ_α(B)`, so each operand's parent eval at
      `r*_H` derives *linearly* from the per-`(col, Δ)` evals plus the
      complement constant `Σ_{b<D} α^b`. The prover ships one trusted eval
      per distinct `(col, Δ)` **pair** (`F2Proof.hadamard_pair_evals`,
      renamed from `hadamard_parent_evals`); the verifier derives operand
      parents (`derive_operand_parents`) and recombines via
      `verify_bit_decomposition_consistency`. `F2VerifierSubclaim`'s
      `hadamard_distinct: Vec<usize>` became `hadamard_pairs: Vec<(usize,
      usize)>`.
    - **Soundness**: the binding check (`verify_f2_full_impl`) ties each
      `Δ = 0` pair eval to the bound `hadamard_evals_at_rstar_h[col]` (the
      second-mp discharge), so `Δ = 0`-only operands — including the
      complemented `(1−col)` since the `Σα^b` const is deterministic — are
      **fully sound**. `Δ ≠ 0` pair evals are prover-supplied / **trusted**
      (MLE eval doesn't commute with row shifts; the row-shift discharge is
      shared with Issue 1, honest-prover-first per plan §6). So among
      C12–C14, the complement-only sub-expressions are sound and the
      shifted ones are honest-prover.
    - **Tests**: `f2_hadamard.rs` unit tests (plain / row-shift / complement
      / Maj-combo round-trips + corrupt-W rejection) and a full-flow e2e
      `prove_then_verify_f2_full_with_operand_hadamards_roundtrips`
      (5-col `HadOpF2Uair`, three relations one per operand kind, through
      `prove/verify_f2_full_with_hadamard`): honest accept + flipped-W
      reject. **43 protocol lib tests pass**; clippy-clean on the changed
      files.
  - **Still open after this**: the **X· bit-shift** in operands (a bit-index
    reindex `b ↦ b+1`, drop bit D-1, zero bit 0 — needed for the adder
    operands `x + X·c`; its parent derivation is `α·(ψ_α(c) − α^{D-1}·c_{D-1})`,
    not yet implemented), the **`W_β` carry column** for the 13 adder
    relations, then **register the 16 SHA relations on `sha256_f2`**
    (C12–C14 first — they need no `W_β` and their columns are already
    committed witness cols, so the operand machinery above can drive them;
    then the 13 adders). And the **sound row-shift discharge** (Issue 1) to
    close the `Δ ≠ 0` trust gap.
  - **C12/C13/C14 WIRED & TESTED on real `sha256_f2`**
    (`sha256_f2_and_hadamards_roundtrips`, working tree): the **three SHA
    AND relations** discharge e2e — 3 of the 16. **Row-shift direction
    (resolved):** the codebase `↓Δ` is `row i → col[i+Δ]` (operand
    `row_shift`, `build_shifted_bit_slice_mles`, `ShiftSpec`
    `uair/src/lib.rs:44`). The SHA fills are written `i−Δ`
    (`u_ef[t]=e[t]&e[t−1]`, `u_neg_e_g[t]=(¬e[t])&e[t−2]`,
    `maj[t]=Maj(a[t],a[t−1],a[t−2])`, `sha256_f2.rs:963`), so **Appendix A
    is `i−Δ` and was re-expressed `i+Δ` by shifting the result column up**:
    C12 `u:W_E^↓1,v:W_E,w:W_UEF^↓1`; C13 `u:¬(W_E^↓2),v:W_E,w:W_UNEG_E_G^↓2`;
    C14 `u:W_A^↓2⊕W_A, v:W_A^↓1⊕W_A, w:W_MAJ^↓2⊕W_A` (Binius
    `(x⊕z)(y⊕z)=Maj⊕z`). **Boundary subtlety RESOLVED**: C13's complement
    (`¬(W_E^↓2)`=all-ones at the zero-padded tail) and C14's combo
    (`W_A^↓2⊕W_A`=`W_A` at the tail) only diverge at rows `t≥n−2`, which lie
    in the **zero slack** (`generate_random_trace` zero-inits, fills only
    active rows), so the un-shifted term `W_E[t]`/`W_A[t]`=0 there and the
    products vanish → honest sum 0. `(W_E,0)`/`(W_A,0)` bound; `Δ≠0` pairs
    trusted (row-shift gap). Lesson logged: **always check the honest
    zerocheck sum is 0 to confirm a registration's boundary handling.**
  - **ADDER OPERAND MACHINERY SHIPPED (`F2AdderSpec`, working tree)**: the
    13 adder relations C5–C11 are `(x + X·c) ⊙ (y + X·c) = c + X·c` (Binius
    carry identity). `F2AdderSpec { t, x, y: F2OperandTerm }` +
    `build_adder_operand_columns` build `U,V,W` per row from the three
    committed columns, with the carry `c = SHR¹(t⊕x⊕y)` on bits `0..D−2` and
    `c[D−1] = β = Maj(x_{D−1}, y_{D−1}, (t⊕x⊕y)_{D−1})` **computed** (the
    overflow carry — so **no committed `W_β` is needed for completeness**;
    `W_β`/X·-bit-shift were the plan's route but computing `β` is simpler).
    Because `c` is *defined* from `t⊕x⊕y`, the zerocheck genuinely verifies
    `t = x+y` (the carry recurrence), not a tautology — confirmed by the
    synthetic tests `adder_round_trips` (honest accept) + `adder_rejects_wrong_sum`.
    Threaded through prove/verify (`adder_specs` param on the phase, the
    group fns, `_full_impl`, `_full_with_hadamard`); `F2Proof` gained
    `hadamard_adder_parents` (the operands' **trusted** parents at `r*_H` —
    the carry `β` + bit-shifts break the pair-eval algebra, so adders are
    honest-prover, no Δ=0 binding).
  - **ROW-SELECTIVITY SOLVED — ALL 13 ADDERS WIRED e2e (3+13 = 16/16 SHA
    relations done)** (`sha256_f2_all_adders_with_selectors_roundtrips`).
    The adders are **row-selective** — `target = x + y` holds only on each
    chain/anchor's active rows; the trace zeroes the S-columns (targets)
    off-chain while the inputs stay non-zero, and the IC's LSB
    `assert_in_ideal(target+x+y+κ)` still holds there only via the `κ` (PA_C)
    compensator. A uniform zerocheck would reject (documented by
    `sha256_f2_adders_need_row_selector`, which still asserts the *un*masked
    rejection). **Fix = `F2AdderSpec::active_rows`** (per-row mask): the
    builder zeroes `U,V,W` off-active so the per-row term is `0⊙0−0 = 0`.
    Since adder parents are trusted (honest-prover), the zeroing rides the
    same trust — **no verifier-side selector / zerocheck change needed**
    (simpler than the indicator-MLE route). The masks are a public
    structural property of the SHA layout — per 68-row block
    (`start = i·ROWS_PER_COMP`): C5a/b/c `[start, start+52)`;
    C6a–e/C8/C9 `[start, start+64)` (anchored where the unshifted S-column
    operand lives — even C6e/C8 whose target is row-shifted); C7
    `[start+3, start+67)` (its target `W_T2` is unshifted); C10/C11
    `[start+64, start+68)` (digest feed-forward). The 13 i+Δ specs + masks
    are in the test. **Lesson**: anchor each adder's mask at the row where
    its *unshifted* operand (the S-column) is materialised, NOT at the
    target's storage row (the C6e off-by-shift that the bisect caught).

### Production integration — the 16 relations derive from a layout builder
- **Shipped**: `protocol::f2_hadamard::Sha256F2HadamardLayout` (struct of the
  ~20 SHA column indices + `ROWS_PER_COMP`/`ROUNDS_PER_COMP`/`num_compressions`
  + `num_vars`) with `and_specs()` / `adder_specs()` / `relations()`. The
  relation **topology** (the i−Δ→i+Δ re-expression, the operand wirings, the
  active-row mask ranges derived from `rows_per_comp`/`rounds_per_comp`)
  lives here once, in the SHA-specific `f2_hadamard` module; the caller fills
  the column indices from the SHA UAIR's `cols` (a thin glue) so there's **no
  reverse `test-uair → protocol` dependency** and no duplicated indices.
- **Why**: the 16 relations + masks were previously hand-written, duplicated
  across the AND-only and adder-only e2e tests. Now there's a single source
  of truth; `sha256_f2_full_hadamard_roundtrips` discharges **all 16** via
  one `prove/verify_f2_full_with_hadamard` call built from
  `layout.relations()`, and the focused AND/adder tests + the
  row-selector-rejection test all consume the builder (masks cleared for the
  latter). 50 protocol lib tests pass; clippy-clean.

### `f2_sha256` bench was stale — fixed (and a nvars=9 baseline)
- **Found while running the bench** (`RUSTFLAGS="-C target-cpu=native"
  cargo bench --features "parallel simd unchecked" --bench f2_sha256 --
  "Steps"`): the bench didn't compile (5 `prove_f2_uair_with_groups` call
  sites still used the pre-`hadamard_specs` signature — never updated since
  `13c169e`, because benches aren't in the default `cargo build`/`test`),
  and the `2-VerifyOpen` micro-bench **panicked** (`EvalConsistency`): it
  verified the open at `r*`, but the full proof opens at the multipoint-eval
  output point `r_0`. Fixed both: added the `&[]` hadamard/adder args, and
  made `2-VerifyOpen` replicate the mp phase (mp `verify_as_subprotocol` →
  absorb `open_evals_at_r_0` → subclaim at `r_0`) before timing the open.
- **Sibling benches `f2_sha256_rs.rs` + `f2_blake3.rs` had the *identical*
  pair of drifts** (surfaced later, building with `--all-targets`): the same
  5 stale `prove_f2_uair_with_groups` call sites (missing `&[]` hadamard +
  `&[]` adder args since `13c169e`) **and** the same `2-VerifyOpen` panic
  (opened at `r*`, proof opens at `r_0`). Fixed identically — the mp-phase
  replication is generic (all three pass `&[]` XOR-virtual-bp specs, so
  `open_evals_at_r_0` is exactly the primary col evals); added the two
  imports (`F2VerifierSubclaim`, `multipoint_eval::MultipointEval`) the
  siblings lacked. All three `2-VerifyOpen` micro-benches now run clean
  (smoke-tested at nvars=9). These three benches aren't in the default build,
  so any signature change to `prove_f2_uair_with_groups` / `F2FullProof`
  must remember to touch all three.
- **Baseline (nvars=9, 7-compression / 512-row fixture, M-series CPU,
  `parallel simd unchecked`, no `metal_gpu`)** — prove: 1-Commit ≈ 272 µs,
  2-UAIR ≈ 953 µs, 3-Open ≈ 660 µs (≈ 1.89 ms total); verify:
  1-VerifyUAIR ≈ 113 µs, 2-VerifyOpen ≈ 702 µs (≈ 0.82 ms total). The
  Hadamard/adder code is a **no-op on this path** (the bench registers no
  relations), so these reflect the unchanged baseline. NB Criterion's
  `change:` deltas are vs a stale saved baseline (and the Commit
  "regression" is CPU-vs-`metal_gpu`-GPU) — read absolute numbers, not the
  deltas. The `e2e` group (full prove/verify, nvars sweep to 22) and `micro`
  group were not run here.

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

### `prove_f2_open` per-column loop is over the full witness slice (20 cols), not the paired-commit set (9)
- **Where**: `protocol/src/f2_prove.rs:1945-1991` (the
  `per_col_results` loop) and `:2020-2069` (the
  `per_col_combined` loop). Both iterate `num_cols =
  trace_binary_cols.len()` which the caller passes as
  `&trace.binary_poly[num_pub_bin..]` = 20 cols at
  `prove_f2_full*` call sites (`f2_prove.rs:2684`,
  `f2_prove.rs:2796`).
- **Observation**: those 20 cols include the 12 bit-op virtuals
  (`SHR^j(PA_C)` for j ∈ 1..12). Their MLE eval claims at r* are
  ALREADY independent witness claims because the open's job is to
  bind them via γ-batching + Schwartz-Zippel discharge — except the
  γ_g for each virtual col can be folded into the γ for its source
  via the per-cell BitOp algebra (XOR/Rot/SHR all commute with the
  linear γ-fold). At least for the SHIFTR case (which is all 12
  virtuals here), the virtual's contribution to `b_g`, `a_g'`, and
  `combined_row_g` can be derived from the source's
  contribution by applying the same BitOp to the *cells before
  inner-producting with q1*. In other words: open does NOT need to
  re-lift+inner-product the virtual cell stream — it can derive
  the virtual's per-row partial from the source's per-row partial
  after the cell-level `apply_bit_op_u32`.
- **Expected impact**: drop the per-col loop's outer iteration count
  from 20 → 8 (1 primary witness per source for each of the 5
  non-virtual witness cols plus 3 source cols that have virtuals).
  Estimated 30-50% of `prove_f2_open` wall time given the per-col
  cost dominates (each col is a full `O(num_rows × row_len)`
  cell-lift + multiply pass). Verify-side mirror needs the same
  shape change. **Effort: M**, with a careful soundness pass on
  what "fold γ into the source via BitOp" means at the wide-poly
  product layer (currently the lift is `bp_to_f2_poly_1` then
  multiply by γ as a `BinaryF2Poly<3>` — the BitOp's
  bit-permutation needs to land *before* the lift, not after).

### Multipoint-eval as degenerate γ-rerandomization (down_evals=[])
- **Where**: `protocol/src/f2_prove.rs:2628-2638` and
  `f2_prove.rs:2758-2768` — both call sites pass `down_evals=&[]`
  and `shifts=&[]`.
- **What it's doing today**: with empty shifts/downs, the
  `MultipointEval::prove_as_subprotocol`
  (`piop/src/multipoint_eval.rs:159-180`) reduces to a degree-2
  sumcheck of `eq(b, r*) · Σ_j γ_j · trace_j(b)`. That's
  essentially a γ-rerandomization of the per-column claims from
  r* to a fresh r_0, with the full sumcheck cost (`num_vars`
  rounds of degree-2 messages over GF(2^192)). The combined
  evaluation at r_0 is then sent through the open path.
- **Why it isn't free overhead**: the rerandomization moves the
  open's `r*` to a new `r_0` so the prover can't tailor the open
  to the IC sumcheck's exact point. Without it, the open's
  Schwartz-Zippel batching is restricted to the IC point — fine
  for the current single-open structure but blocks any future
  multi-point lookup batching.
- **Possible optimization**: when `shifts.is_empty()`, the
  multipoint-eval phase could fall back to a leaner γ-fold-only
  protocol that skips the eq-table build and the sumcheck (just
  draws γ and the verifier checks the same fold via a single
  GF(2^192) batch identity). The current shape pays the full
  sumcheck cost for what is effectively a free rerandomization.
- **Expected impact**: small but nonzero — at nvars=20 the inner
  sumcheck is ~25 rounds × O(2^num_vars / 2^round) over
  GF(2^192). The fold-only fallback would save the per-round
  message bytes and the eq-table eval. Estimated 2-4% Prove e2e.
  **Effort: M** (a new sub-protocol shape + verifier mirror).
  Lower priority than the per-col open loop above.

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

### Generalise `F2BitOpVirtualSpec` to a "multi-source XOR with per-source per-cell BitOp" spec (chained-Binius cols)
- **What**: introduce a new virtual-spec type whose evaluation rule
  is `target[i] = XOR_k op_k(source_k[i])` for a list of
  `(source_col, BitOp)` pairs, generalising the current per-cell
  `F2BitOpVirtualSpec` (which is the single-source case). Then
  declare the 6 chained-Binius intermediate cols (`W_W_S1`,
  `W_W_S2`, `W_T1_S1..S4`, in `test-uair/src/sha256_f2.rs:1068-1073`,
  protocol indices `cols::W_W_S1` etc) as virtuals under this spec.
- **Why it works on this branch (vs the K-virt machinery on
  `claude/gkr-virtual-cols`)**: every chained-Binius intermediate
  `s_k` is bit-by-bit determined by its predecessor + XOR
  contributions + a single LSB carry bit pulled from `PA_C`, **but
  the constraint `(s_k + ...)[0] = 0 (mod X^32 − 1)` already locks
  `s_k` bit 0**. The remaining bits 1..31 of `s_k` are NOT yet
  AIR-pinned (that's the Hadamard step 5 work in the entry above).
  Concretely: the LHS of C5a in `test-uair/src/sha256_f2.rs:500-503`
  forces `W_W_S1 ≡ W_W + W_SIG0^{↓1} + κ_{W,1} (mod X^32)`, but
  `mod X^32 − 1` is also `mod X^32` only at bit 0 — bits 1..31 are
  free in the current AIR. So **declaring `W_W_S1` virtual today is
  unsound at the AIR layer** until Hadamard discharge (step 5) lands.
- **Why this is different from the K-virt Issue 1 hole**: this spec
  is purely per-cell (no row shifts) — it derives `target[i]` from
  the *same row* of source cols. MLE evaluation at a fixed `r*`
  commutes with per-cell BitOp + XOR, so the verifier can recompute
  `MLE(target)(r*) = XOR_k op_k(MLE(source_k)(r*))` with no
  row-shift discharge needed. Soundness for the eval-at-r* step is
  free. The blocker is only the AIR-layer bit-1..31 pinning, which
  is the same Hadamard work the K-virt rollback also waits on.
- **Expected impact if Hadamard lands**: drops `paired_batch` from 9
  → 6 (excluding 6 cols from commit halves the leaf-bytes input to
  Blake3, and shrinks paired storage by ~1/3). At nvars=20 the
  per-leaf-byte work and the GPU slab size shrink ~33%, plausibly
  buying back ~10-15% of the Commit phase. The open's per-column
  loop in `prove_f2_open` still runs over the full witness slice
  (20 cols), so this doesn't directly speed open; that would need a
  separate "open-virtuals" change.
- **Effort**: **L** as long as Hadamard discharge is the gate. The
  spec type itself is a small addition; the BlockerN is the
  unrelated Hadamard step 5 work. Calling it out so future-me
  doesn't try the spec change in isolation and produce an unsound
  prover. **Do not attempt without Hadamard.**

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
