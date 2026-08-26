# Limber's Zinc+ row, rerun with the folded main-beta pipeline

**Date:** 2026-08-26 · **Machine:** MacBook, Apple M4 (10 cores, 16 GB) —
same box and methodology as `docs/limber-2026-1635-reproduction.md`
(2026-08-21, on `bench/limber-repro-main`): single-threaded, `simd
unchecked`, `-C target-cpu=native`, medians of 5, prover time excludes
witness generation (~0.1 s), proof sizes raw + zstd.

## 1. Correction to the 2026-08-21 reconstruction: Limber pinned main-beta

The Aug-21 doc inferred (paper only, no artifact) that Limber ran the
public `main` branch. Limber's repo README has since (2026-08-24, limber-impl
commit `0de4412`) published the recipe:

> `NethermindEth/zinc-plus` at commit `7eadc16`, `crypto-primitives`
> pinned to `2cf39db8`, `RAYON_NUM_THREADS=1 RUSTFLAGS="-C
> target-cpu=native" cargo bench --bench e2e --features "parallel simd
> unchecked iprs-rate-1-8"`.

`7eadc16` ("Updated README", 2026-05-18) is **not on `main`** — it is on
**`main-beta`**, where it was the branch tip from May 18 to Jun 4. Today's
`main-beta` tip (`91cf8aa`) is only two first-parent commits ahead:
`48fb482` (bit-op virtual-col perf, irrelevant to an int-only UAIR) and
the `runtime-const-field` merge (value-sized `Fp` as the integer-path
field) — the change that accounts for most of the unfolded speedup from
their ~2.06 s to our 0.94–0.97 s.

So: **Limber did use the main-beta branch, at a snapshot ~6 weeks behind
its tip at their paper date.** The branch attribution in §2 of the Aug-21
doc is wrong; its measurements and takeaways stand (its `main`@`5c7694f`
reconstruction matched their numbers because both that commit and
`7eadc16` predate the value-sized-Fp integer path).

## 2. Limber did NOT use the folded pipeline (and could not have)

At `7eadc16`, `prove_folded_4x` / `IntFoldedZincTypes4x` already exist,
but the int 4× fold was hard-wired to 64-bit quarters
(`v = q_0 + 2^64 q_1 + 2^128 q_2 + 2^192 q_3`, `split_int_column_4x`
takes `words[0..3]` + `v >> 192`), targeting 256-bit ECDSA cells. On a
2048-bit cell, a `Q=2` quartering truncates `q_3` and the multipoint-eval
recombination fails — folded proofs of their statement were impossible
without generalizing the radix. Their quoted 2.06 s / 514 ms / 1.2 MB
matches the unfolded reconstruction to ~10 % (Aug-21 doc §3). Verdict:
**their numbers are the unfolded path.**

## 3. What this branch adds: radix-generalized 4× int fold

Three sites generalized so the quarter width derives from
`INT_QUARTER_LIMBS = Q` (quarter = `Q−1` words, radix `2^(64·(Q−1))`;
`Q = 2` reproduces the original 2^64 behavior bit-for-bit — the existing
`test_e2e_sha_ecdsa_folded_4x_round_trip` and folding unit tests pass
unchanged):

1. `zip-plus/src/pcs/folding.rs::split_int_column_4x` — witness split.
2. `protocol/src/lib.rs::compute_int_fold_4x_lifted_evals` — per-quarter
   lifted evals (u64 fast path kept for `Q = 2`).
3. `protocol/src/verifier.rs` (`verify_folded_4x_inner`) — the
   `c0 + R·c1 + R²·c2 + R³·c3` recombination weights.

Bench (`protocol/benches/limber_multiswap.rs`) additions:

- `RsaFolded4xZincTypes`: `Int<34>` cells quartered at radix `2^512`
  into `Int<9>` (512-bit magnitude + sign-headroom limb); quarter
  `Cw = Int<14>`, `CombR = Int<18>`.
- `RsaModMulWide16Uair`: 16 int columns = 4 modmuls/row, so nvars = 11
  (2048 rows) carries the same 8192 constraint slots while the folded
  row length `4·2048 = 8192` still fits the F65537 NTT cap at rate 1/8
  (the wide8/nvars=12 shape folds to 16384-length rows → codeword
  131072 > 65536, unsettable at rate 1/8).
- `FOLD=1` / `WIDE16=1` env knobs.

**Overflow validation:** the full folded wide16 nvars=11 run was first
executed as a CHECKED build (no `unchecked`; every add/mul guarded) —
prove 5.37 s, verify 0.45 s, proof verifies. No overflow with the
`Int<14>`/`Int<18>` sizing, so the unchecked timings below are safe (cf.
the Aug-21 doc's confirmed silent i64-Cw overflow — this is the guard
against a repeat).

## 4. Results (all same statement: 8192 modmul slots ⊇ 6,209 constraints)

Run: `[FOLD=1] [WIDE16=1|WIDE8=1] NVARS=… REPS=5 RUSTFLAGS="-C
target-cpu=native" cargo bench --features "simd unchecked iprs-rate-1-8"
--bench limber_multiswap`

| Run | Shape | Prove | Verify | Proof raw | Proof zstd |
|---|---|---|---|---|---|
| Limber's quoted Zinc+ row (M4 Pro) | ? | 2.06 s | 514 ms | ? | 1.2 MB |
| Limber-Hyrax (their headline, M4 Pro) | — | 1.32 s | 39 ms | — | 170 KB |
| unfolded, 8×4096 (Aug-21 shape, rerun) | flat | 0.972 s | 123 ms | 1.70 MB | 1.28 MB |
| unfolded, 16×2048 (same shape as folded) | flat | 1.285 s | 94.7 ms | 1.26 MB | 0.96 MB |
| **folded 4×, 16×2048, radix 2^512** | 4n rows | **0.852 s** | **57.8 ms** | 1.42 MB | **686 KiB** |

(The 8×4096 rerun vs Aug-21's 0.94 s / 119 ms / 1.31 MB: within 3 % —
machine parity holds.)

**Folded main-beta vs what Limber quoted:** prove **2.4× faster**,
verify **8.9× faster**, proof **1.75× smaller** (686 KiB vs 1.2 MB).

**Folded vs best unfolded main-beta (0.94 s / 119 ms / 1.31 MB):** prove
1.10× faster, verify 2.1× faster, proof 1.9× smaller. The fold pays off
mostly on the verifier (re-encoding a `CombR = Int<18>` row instead of
`Int<44>`) and proof size (narrower combined row + narrower opened
columns); the prover side is dominated by the unchanged ideal
check/sumcheck, so its gain is modest.

**Versus Limber-Spartan itself (1.32 s / 39 ms / 170 KB single-threaded):**
folded main-beta Zinc+ proves 1.55× faster, verifies 1.5× slower, proof
4× larger. (With `parallel`, unfolded main-beta already reached 0.185 s
prove — the folded parallel point was not measured here.)

## 5. Caveats

- Statement is the same honest-width mock as the Aug-21 doc: real
  `a·b = c + u·N` witnesses, unwired, no range checks on `c`, `u` —
  matching the scope of Limber's own footnoted mock.
- Both the standard and folded main-beta paths use the **fixed
  secp256k1 projecting prime** (`fixed_prime.rs`, `prover.rs` step 1) —
  the SHA+ECDSA demo shortcut, not the FS-sampled random prime; not
  generally sound as-is, but uniform across every row of the table.
- Rate 1/8 with 100 column openings (~100-bit) throughout, like
  Limber's chosen feature set.
- The 4× fold is the shipped maximum; for 2048-bit cells a deeper fold
  (e.g. 32× to 64-bit quarters) is the natural next step and should
  shrink the combined row / opened columns further, at the cost of
  32n-length rows (needs a wider-domain NTT or rectangular geometry).

## 6. Artifacts

Branch `bench/limber-repro-beta-folded` (off `bench/limber-repro-beta`
= `main-beta`@`91cf8aa` + harness), worktree
`.claude/worktrees/limber-repro-beta`.
