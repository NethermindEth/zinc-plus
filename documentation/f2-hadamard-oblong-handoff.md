# Handoff — Oblong AND zerocheck for the F_2 SHA-256 Hadamard discharge

*Written 2026-06-02 after a long build session. This is the **START HERE** for
the oblong-discharge work. Companions: the plan
`documentation/f2-hadamard-oblong-port-plan.md` (phased roadmap + §4 the
integration seam, with full soundness), the ledger `documentation/f2x-sha-todo.md`
(shipped entries with measurements), and the original perf handoff
`documentation/f2-hadamard-handoff.md` (why this lever exists). Reference impl:
the local repo `~/binius64` (`crates/prover/src/and_reduction/`,
`crates/verifier/src/protocols/bitand.rs`).*

## TL;DR (the one thing to know)

We replaced the memory-bound **bit-slice AND discharge** (1536 GF(2¹²⁸) slices,
~92% of the nvars=16 prove) with Binius64's **word-packed oblong univariate
zerocheck**. As a **standalone discharge prover** it is now **5–14× faster than
the fused bit-slice discharge** and the win grows with size. It is **fully built,
tested, and benchmarked in isolation — but NOT yet wired into the real prover**, so
there is **no end-to-end prove measurement yet**. The single most valuable next
step is the **production integration** (task #7 pt 2 below): fold the discharge's
`ψ_z` evals into `f2_prove`'s multipoint-eval. That converts the standalone speedup
into a measurable e2e prove speedup (projected ~3.7× at nvars=16, unverified).

## Measured discharge A/B (standalone prover only)

`f2_sha256` bench, group `Zinc+ F_2 SHA-256 Hadamard`, arms `Discharge-Fused`
(ψ_α bit-slice, today's production) vs `Discharge-Oblong` (ψ_z, naive GF128) vs
`Discharge-Oblong-GF8` (ψ_z, GF(2⁸) byte-lookup NTT). Same SHA trace columns + the
same 16 relations. Apple M4, `target-cpu=native`, `parallel simd unchecked`,
sample_size 10. **Numbers drift with thermal state — these are a clean, cool run:**

| nvars | Fused | Oblong (GF128) | **Oblong-GF8** | **GF8 vs Fused** |
|---|---|---|---|---|
| 9  | 3.7 ms  | 2.6 ms  | ~1.7 ms  | ~2× (noisy; see caveats) |
| 16 | ~395 ms | ~102 ms | **~70 ms** | **~5.6×** |
| 20 | ~13 s   | ~1.6 s  | **~1.2 s** | **~11×** |

How we got there, compounding (newest last): word-packed representation (1.1–1.4×)
→ **GF(2⁸) swap** (`Gf8Scheme`, ~2×) → **parallelism** (the biggest single lever —
the prover was single-threaded vs the parallel fused baseline) → **eq-split**
(modest on aarch64; see the finding below).

**What the bench does NOT include**: the ψ_z tie, the PCS commit/open, the
multipoint-eval — it is purely `prove_oblong_and_batch_gf8` on the trace columns.
Also: it rebuilds `Gf8Scheme::new()` (the NTT byte-table) **every iteration** —
fixed cost, amortised away at nvars≥16 but a real fraction at nvars=9, so the
**nvars=9 number is setup-inflated and serial** (`with_min_len(2¹⁴)` keeps the 8K-word
nvars=9 workload single-threaded). Trust nvars=16/20.

## Code map (all new, ~2800 lines, no production prover path touched)

- **`poly/src/univariate/binary_subspace.rs`** — `BinarySubspace`, `lagrange_evals`,
  `extrapolate_over_subspace`, `evaluate_univariate` (ports of binius64
  `math/{binary_subspace,univariate}`).
- **`poly/src/univariate/oblong_and.rs`** — the protocol core: the `OblongScheme`
  trait (round message + fold weights + small challenges), `MonomialScheme`
  (naive GF128), the `OblongChannel`/`ReplayChannel` Fiat–Shamir abstraction,
  `prove/verify_oblong_and_channel`, the Phase-2 degree-≤3 sumcheck (`round_poly`,
  `fold_low` — **parallel**, `with_min_len(PAR_MIN_LEN=2¹⁴)`), `eq_indicator`,
  the explicit wrappers. **SKIPPED_VARS=5, WORD_BITS=32.**
- **`poly/src/univariate/binary_gf8.rs`** — `Gf8` (GF(2⁸) derived from GHASH:
  `θ`=relative norm, `m=minpoly(θ)`, `α↦θ` is a verified field hom), `embed`,
  and a **64KB `mul_table` + `gf8_mul_embed_tables()`** (fetch the mul/embed
  tables once per kernel — critical on aarch64).
- **`poly/src/univariate/oblong_and_gf8.rs`** — `Gf8Ntt` (the byte-lookup additive
  NTT), `gf8_round_message` (base) + `gf8_round_message_split` (eq-split),
  `Gf8Scheme` (the GF(2⁸) `OblongScheme` over `embed(H₈)`), `embed_subspaces`,
  `eq_indicator_gf8`. All parallel.
- **`protocol/src/f2_oblong_hadamard.rs`** — the discharge wiring:
  `prove/verify_oblong_and_relation` (one relation, Fiat–Shamir), the **batched**
  `prove/verify_oblong_and_batch[_gf8]` (all 16 relations), the **`ψ_z` tie**
  (`batched_tie_check`), `build_stacked_operands`, the `TranscriptChannel` adapter.
- **`protocol/benches/f2_sha256.rs`** — the three `Discharge-*` bench arms.

## Key architectural facts (so you don't rederive them)

1. **`ψ_z` is `ψ_α` with `L_i(z)` for `α^b`.** Both `F_2`-linearly collapse a
   word's `D` bits into one `GF(2¹²⁸)` scalar (monomial-at-α vs Lagrange-at-z). The
   oblong zerocheck outputs `a_eval = ψ_z(U)(γ)` by construction, so the tie
   **reuses the existing tested `ψ_α` machinery** (`pair_alpha_evals` +
   `derive_operand_parents` from `f2_hadamard.rs`) fed `base_lagrange_at(z)` and the
   row-point `γ`. Soundness is the same `(D−1)/|F|` bound (plan §4).
2. **The binding is a row-space open at `γ`, NOT a joint `(z,γ)` open.** `z` only
   picks the bit-recombination weights. The `ψ_z(col↓Δ)(γ)` pair-evals fold into the
   **main multipoint-eval** exactly like the current `ψ_α` pair-evals do ("Approach
   B", `f2_prove.rs:128-135`). This is the whole content of task #7 pt 2.
3. **Batching = stacking.** All 16 relations (3 ANDs + 13 adders) stack into ONE
   zerocheck: relation index = high row-vars, word index = low, padded to a
   power-of-two relation count with zero-operand relations. The batched tie is
   `a_eval = Σ_rel eq(rel; γ_rel)·ψ_z(U_rel)(γ_word)` (`γ` = `[γ_word (low num_vars),
   γ_rel (high log k_pad)]`).
4. **GF(2⁸) path uses the `embed(H₈)` subspace**, not the monomial one — the
   verifier must extrapolate `R₀` over `Gf8Scheme::full_subspace()` and use the
   embed-base Lagrange weights for the tie. Soundness rests on `embed` being a hom
   (verified over all 65536 pairs).
5. **aarch64 GF(2⁸) finding (important).** Without GFNI, `Gf8::mul` (log/antilog +
   `%255` + a per-op `OnceLock` fetch) costs ≈ a GF128 CLMUL. So the **eq-split only
   pays with the 64KB mul-table fetched once per kernel** (then 1.30× same-mul;
   ~1.06× vs the prior best). The same wall blocks SIMD packing (no native packed
   GF(2⁸) mul on aarch64).

## What's DONE (≈35 tests green across `poly` + `protocol`)

Phase A primitives · Phase B standalone prove/verify (accepts honest, rejects
corrupt) · Phase C `ψ_z` tie (plain/shift/complement/Maj round-trips + mis-wiring
rejected) · Fiat–Shamir · batched all-16 (ANDs + adders) · GF(2⁸) swap ·
**parallelization** · eq-split. Commits `6660bff` → `3057190` on branch `f2-clean`.

## NEXT STEPS, in priority order

### 1. ★ Production integration — fold `ψ_z` evals into `f2_prove`'s multipoint-eval (task #7 pt 2)
**This is the high-value next step** — it's what makes the 5–14× *real* (e2e
measurable). The discharge currently produces `a/b/c_eval` at `(z, γ)`; the tie
checks them against in-memory columns. The real protocol must instead:
- project the discharge columns with `project_column_with_powers(col,
  base_lagrange_at(z))` (= `ψ_z`, exactly what `pair_alpha_evals` already does),
- fold the `ψ_z(col↓Δ)(γ)` pair-evals into `f2_prove`'s **main multipoint-eval**
  (Δ=0 point claim, Δ≠0 shift predicate), bound by the single PCS open — mirroring
  the `ψ_α` Approach-B path at `f2_prove.rs:120-135,949-1010`.
- §4-(i) projection-point choice: keep `α` for the IC columns, project the
  discharge columns at `z` too (extra openings on the discharge columns only).
- Adders ship **trusted** ψ_z parents today (the carry isn't committed — ledger
  Issue 1); a sound carry binding (the row/bit-shift discharge) is the follow-up.
- **Gate**: `Prove-Hadamard` e2e bench with the oblong discharge ≈ `Prove-NoHadamard`
  + ~70 ms instead of + ~390 ms. Add a `Prove-Hadamard-Oblong` bench arm.

### 2. Gruen's eq-trick on Phase-2 (clean, portable speed + smaller proof)
My Phase-2 is the naive form: degree-3 round polys (folds `eq` as a 4th table) +
the full `eq_indicator(r)` materialised (2²⁰ GF128 ≈ 16 MB at nvars=16). Gruen
factors `eq` out: send the degree-2 `h_i(t)=Σ_rest (a·b−c)(t,rest)` (3 evals),
reconstruct `g_i = eq(t;r_i)·h_i` with a running `∏ eq(γ_j;r_j)` scalar. Drops the
`eq` table (fold 3 not 4, no `eq_indicator` build), 3 evals/round not 4, **smaller
proof**. Composes with the eq-split (the scalar just includes the deterministic
small `r_i`). The fused discharge already does this (`piop` `MultiDegreeSumcheck`)
— mirror it. Estimate ~70.7→~63 ms at nvars=16, grows at nvars=20. **Touch**: the
Phase-2 in `oblong_and.rs` (`round_poly`, the fold loop, `verify_oblong_and_channel`
round-consistency + closing) + the round-trip tests (proof shape changes).

### 3. Lower-value / hardware-specific
- **SIMD packing** (`Gf8` 16-lane) — **blocked on aarch64** (no GFNI ⇒ no native
  packed GF(2⁸) mul; the 64KB table doesn't vectorise via NEON `TBL`). An **x86/GFNI
  lever only** (binius's `PackedAESBinaryField16x8b`). Defer unless targeting x86.
- **Hoist `Gf8Scheme::new()`** out of the bench's per-iteration loop (and the fused
  arm's setup) for a clean, fair nvars=9 number.
- **Sound adder carry binding** (Issue 1) — needed before the adders are
  soundness-complete (currently honest-prover-trusted, same as the fused discharge).

## Open questions / caveats
- No e2e measurement until step 1 lands. Standalone discharge ≠ e2e prove.
- The eq-split is modest on aarch64; revisit on GFNI hardware.
- Bench numbers drift with thermal state — re-baseline cool, compare within one run.
- The `Discharge-Oblong-GF8/nvars=9` arm is setup-inflated + serial (see TL;DR).

## Reproduce
```sh
# Discharge A/B (all three arms), nvars 9/16/20:
RUSTFLAGS="-C target-cpu=native" F2_ONLY=hadamard \
  cargo bench -p zinc-protocol --features "parallel simd unchecked" \
  --bench f2_sha256 -- "Discharge"
# Scope to one size with HAD_NVARS=16 (or 9,16).

# Tests (serial + parallel paths):
cargo test -p zinc-poly --features "simd parallel" oblong
cargo test -p zinc-poly --features simd binary_gf8
cargo test -p zinc-protocol --features "parallel simd unchecked" --lib f2_oblong
```

## Commit trail (branch `f2-clean`, oldest first)
`6660bff` Phases A+B · `4f2d002` GF(2⁸) field+embed · `7da8e1c` GF(2⁸) NTT ·
`8677fae` Phase-C ψ_z tie · `7ed6318` Fiat–Shamir · `a50638a` batched ANDs ·
`06077b3` +adders (all 16) · `1a0b09f` discharge bench · `a6e4780` GF(2⁸) swap ·
`39ebc1d` **parallelize** · `9149b91` eq-split + 64KB mul-table (+ interleaved
`docs(...)` commits). All work is in the 7 files listed under "Code map".
