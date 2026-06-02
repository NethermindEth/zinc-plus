# Handoff — F_2 SHA-256 Hadamard discharge: performance state & the next lever

*Written 2026-06 after a long optimization + investigation session. Start here,
then go to the canonical ledger `documentation/f2x-sha-todo.md` and the design
doc `documentation/f2-hadamard-univariate-skip-design.md` for detail.*

> **▶ ACTIVE PLAN (2026-06-01): the "next lever" below is now being built as a
> direct port of Binius64's oblong univariate zerocheck — see
> `documentation/f2-hadamard-oblong-port-plan.md` (reference repo `~/binius64`).
> Phases A+B + the GF(2⁸) speed lever + the Phase-C `ψ_z` tie are DONE (29 tests).
> A **standalone oblong AND zerocheck works over `GF(2¹²⁸)`** (`poly`:
> `binary_subspace.rs` + `oblong_and.rs`); the prover-side **GF(2⁸) byte-lookup
> NTT** (`binary_gf8.rs` + `oblong_and_gf8.rs`, embedding verified over all 65536
> pairs) makes the round message **2.56× faster** (99.6 vs 255 ns/word, nvars=16);
> **Phase C** (`protocol/src/f2_oblong_hadamard.rs`) wires the discharge with the
> **`ψ_z` recombination tie** (reuses the `ψ_α` machinery), is **Fiat-Shamir**, and
> now **batches all 16 SHA relations** (3 ANDs + 13 adders) into one zerocheck.
> **Headline A/B** (`f2_sha256` bench): the **GF(2⁸)-accelerated oblong discharge
> is 5–11× faster than the fused bit-slice one** and the **win grows with size** —
> nvars=16 **5.3×** (395→74.7 ms), nvars=20 **11.2×** (13.2 s→1.17 s). Two
> compounding levers: the `Gf8Scheme` (byte-lookup NTT, naive→~2×) and
> **parallelism** (the oblong prover was single-threaded vs the parallel fused
> baseline). Continue from the port plan's "Progress" + §5. **Remaining**: (1) the
> **eq-split** (task #6, next speed step — eq-weighting is still per-word GF128);
> (2) the **multipoint-eval binding** in `f2_prove` (task #7 pt 2, production
> integration, open at `γ`/Approach B; doesn't change discharge prove cost);
> (3) sound adder carry binding (Issue 1); then the **e2e `Prove` A/B**.**

## TL;DR (the one thing to know)

The Hadamard discharge is **~92% of the nvars=16 prove** and is **memory-
bandwidth-bound**: it materialises **1536 GF(2¹²⁸) bit-slices (≈1.6 GB)** and
streams them every sumcheck round. Evidence: Prove-Hadamard nvars=16 is **1085 ms
single-thread → 464 ms on 10 cores = only 2.34× scaling** (a compute-bound job on
the M4's ~6 P-equivalents should scale ~6×). The performance gap to Binius64 is
**this `×D=32` bit-slice data volume**, not slow arithmetic. **The next lever is
the word-level AND reduction (~32× less data); byte-identical sumcheck
optimisations are exhausted — they cannot reduce the slice volume.**

## Current production state (what's shipped & live)

- **Discharge path**: `protocol/src/f2_hadamard.rs::prove_f2_hadamard_phase` →
  `zinc_piop::lookup::hadamard::prepare_hadamard_group_with_fast` (round-1 fast
  path, bit-packed) + a **fused coeff-form `RoundPolyEvaluator`** for rounds
  2..n, **size-gated at `num_vars >= FUSED_MIN_VARS = 14`** (piop
  `hadamard.rs:557`). Below 14 (incl. the **nvars=9 production default**) it
  uses the generic per-point path — unchanged, no regression.
- **Shipped this session** (byte-identical, gated by `fast_path_matches_generic`
  + `fused_matches_generic_large`):
  - `9505876` — fused rounds-≥2 evaluator (loop `(relation,bit)`-outer,
    point-inner; eliminates the generic per-point 1536-slice scatter-gather).
  - `7a27606` — coefficient-form round poly + Karatsuba (6 muls/term vs 14).
  - **Net: discharge 554 → 418 ms (−24.5%) at nvars=16; Prove-Hadamard 600 →
    464 ms (−22.7%).** This fixed the *access pattern*, NOT the data volume — so
    it does not close the Binius gap.

## Benchmarks (Apple M4, 10 cores = 4P+6E, `target-cpu=native`, `parallel simd unchecked`)

| metric | value |
|---|---|
| Prove-Hadamard nvars=16 (10 cores) | 464 ms |
| Prove-Hadamard nvars=16 (1 thread) | 1085 ms → **2.34× scaling (memory-bound)** |
| Prove-NoHadamard nvars=16 | 46 ms ⇒ **discharge ≈ 418 ms (90%)** |
| nvars=9 (production default) Prove-Hadamard | ~7.8 ms (generic, gated) |
| **Binius64 ref** | 111.82 ms proof / 1365 keccak-f on **64-core Graviton4** |

Hardware-normalised, we're roughly in the same per-core ballpark, but our **2.34×
scaling** means on equal HW we'd still be ~3–4× slower — because Binius's
word-level/GF(2) representation is ~32× less data ⇒ not memory-bound.

## What was tried and RULED OUT (do not repeat without reading why)

1. **`(col,Δ)` slice dedup** — ~2% ceiling (the 13 adders are distinct
   carry-computed operands; quadratic `U·V` blocks per-pair factoring). Ledger
   "Phase 2 dedup".
2. **γ'^k·σ^b weight precompute** — rejected, +6.9% (memory-bound, not arithmetic).
3. **skip=2 small-value prover (REMOVED, `885c64b`)** — 40–74× *slower*. Bad
   impl: `build_prefix_q_v2` called Procedure 1 per `(x'',relation,bit)`, each a
   `prepare_eval_aux` `batch_invert` (GF128 inversion) + Vec allocs.
4. **Byte-identical small-value prover, properly costed (CLOSED)** — per-term
   probe: **7.12 µs (prefix) vs 11.4 ns (standard) = ~624× worse** as-impl; even
   overhead-hoisted the `O(d^v)`-bb floor is ~2× worse at **d=3** (off-hypercube
   `bb` dominates Procedure 1; the boolean `ss` is a minority — the `3√κ`
   asymptotic needs high d). The 19× "inner-comb ceiling" (Phase-0 probe) was
   real but only the 4 boolean cells of the 16-cell v=2 grid.
5. **Univariate skip (Gruen/Binius, verifier-visible)** — would work but needs an
   additive NTT over binary fields + a GF(2⁸) tower subfield + a new
   verifier/soundness; the codebase has none of these (the `pntt/radix8` FFT is
   prime-field). It's the *sumcheck half* of the word-level rearchitecture below.

**Key meta-lesson:** every byte-identical path leaves the **1536-slice GF(2¹²⁸)
data volume** unchanged, so none can fix the memory-bound bottleneck. Only
changing the *representation* (fewer/smaller slices) can.

## The next lever — WORD-LEVEL AND reduction (the real fix)

**▶ See the concrete plan: [`f2-hadamard-oblong-port-plan.md`](f2-hadamard-oblong-port-plan.md).**
After reading the actual Binius64 source (`~/binius64`), the target is its
**oblong univariate zerocheck** AND reduction: keep operands as packed words
(`Vec<Word>`), handle the bit dimension with a univariate-skip round + additive
NTT, never materialise the 1536 GF(2¹²⁸) slices.

Two corrections to earlier notes in this repo (which conflated the *original*
Binius with **Binius64**):
- **The field is SHARED, not a tower.** Binius64 `B128` is the **GHASH GF(2¹²⁸)**
  (`X¹²⁸+X⁷+X²+X+1`, reduces with `0x87`) — *identical to our `BinaryFieldGF128`*.
  No tower-field rewrite. (The original Binius's heavy GF(2)/GF(2⁸) tower packing
  is a different system.) Binius64 uses a single `GF(2⁸)` subfield *only* for ≤3
  deterministic small-field challenges in the skip round.
- So adopting it is a **port over our existing field**, not a field
  re-architecture. The real risk is the **integration seam** (tying the oblong
  zerocheck's operand evals to our committed columns via ψ_α / a shift-reduction
  analogue), not the field or the zerocheck. See the port plan §4.

Represent the operands as **packed words** (≈48 word-columns) instead of **1536
bit-slices**, so the discharge streams ~32× less data ⇒ no longer memory-bound,
scales across cores, and the arithmetic shrinks too. It is **not** byte-identical —
it changes how `W = U ⊙ V` is argued (verifier-visible).

**What to scope first (design pass, before any code):**
- The mechanism to argue bitwise `W = U ⊙ V` over *packed* operands without the
  32 bit-slices — Binius's AND reduction routes through its "shift reduction" /
  oblong-multilinearisations (see `binius.xyz/blueprint/backend/ands`, building
  on Gruen [Gru24] + Hu et al. [Hu+25]). Figure out the analogue for our
  ψ_α / Wiring-R setting.
- How it slots into the existing flow: the discharge runs *before* ψ_α and feeds
  the recombination at α (GF(2¹²⁸)) — the word-level argument must still produce
  the per-operand evals the recombination ties to the column openings.
- Soundness + the field for the sumcheck (likely small-characteristic / GF(2⁸)
  tower — which also needs a tower-field arithmetic primitive the repo lacks).
- A data/cost estimate: target ~48 word-columns vs 1536 slices ⇒ aim to make the
  discharge bandwidth comparable to the base prove (~46 ms), i.e. ~10× on the
  discharge.

This is a multi-week, research-grade effort (new argument + verifier + soundness
+ tower-field arithmetic), but it is now the **clearly-correct** target,
evidenced by the 2.34× scaling, not a guess.

## Reproduce / pointers

```sh
# Discharge A/B at nvars=16 (multi-thread):
RUSTFLAGS="-C target-cpu=native" F2_ONLY=hadamard HAD_NVARS=16 \
  cargo bench -p zinc-protocol --features "parallel simd unchecked" \
  --bench f2_sha256 -- "Prove-Hadamard"
# Add RAYON_NUM_THREADS=1 to see the 2.34× scaling (memory-bound proof).
# Bench knobs (added this session): F2_ONLY=<e2e|steps|micro|hadamard> runs one
# criterion group (criterion runs every group body otherwise — the e2e sweep
# goes to 2^22 and swaps a 16 GB box); HAD_NVARS=9,16 scopes the Hadamard sweep.

# The two diagnostic probes (poly/benches/binary_gf_compare.rs):
RUSTFLAGS="-C target-cpu=native" cargo bench -p zinc-poly --features simd \
  --bench binary_gf_compare -- "discharge_comb"           # sv vs bb inner comb: ~19×
RUSTFLAGS="-C target-cpu=native" cargo bench -p zinc-poly --features simd \
  --bench binary_gf_compare -- "prefix_v2_vs_standard_term" # prefix term: ~624× worse
```

- **Canonical ledger**: `documentation/f2x-sha-todo.md` — see the "ROOT CAUSE"
  entry (memory-bound), the small-value/skip entries (closed), and the shipped
  fused-prover entry.
- **Design doc**: `documentation/f2-hadamard-univariate-skip-design.md` — §10–11
  (Phase-0/1 net analysis) + §1–6 (the verifier-visible rearchitecture scope).
- **Code**: discharge wiring `protocol/src/f2_hadamard.rs`; the sumcheck +
  fused evaluator `piop/src/lookup/hadamard.rs` (`HadamardRoundEvaluator`,
  `RoundPolyEvaluator` hook in `piop/src/sumcheck/prover.rs`).
- **Key commits (this session, newest first)**: `3d88e49` (root cause:
  memory-bound) · `a099df1` (byte-identical path closed, 624×/term) ·
  `1c5b60a` (net analysis ~neutral d=3) · `7c27ed9` (Phase-0 probe 19×) ·
  `f358309`/`9f77c0e`/`4c5a4d1` (design pass + univariate-skip gap) ·
  `7a27606` (coeff-form −24.5%) · `9505876` (fused evaluator −13%) ·
  `46c28e0` (bench knobs + dedup/skip findings) · `885c64b` (removed skip=2).
