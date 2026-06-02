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
zerocheck**. As a **standalone discharge prover** it is **5–14× faster than the
fused bit-slice discharge** (win grows with size). It is now also **wired into the
e2e prove path (measurement-first)**: the `Prove-Hadamard-Oblong` bench arm measures
a **~5.6× faster Hadamard prove at nvars=16** (101 ms vs the fused 567 ms; discharge
overhead ~54 ms vs ~390–520 ms) — confirming the standalone win e2e and beating the
~3.7× projection. **The sound `ψ_z`→commitment binding is now DONE (NEXT STEP #1
below)**: rather than a second z-open, the **open was rewritten un-lifted** so its
bound bit-slice claim `a' = Σ_b c_b·X^b` yields *both* `ψ_α` (main) and `ψ_z`
(discharge) — the discharge folds into the **same** multipoint + open.
`prove/verify_f2_full_with_oblong_hadamard` round-trips (honest accept + tamper
rejects); the e2e oblong path is now a **verified** proof, not just a measurement.

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

**Post-Gruen** (Phase-2 eq-trick, task #2, now shipped — see below): a back-to-back
hot-machine A/B drops the GF8 arm further to **~58 ms** (nvars=16) / **~1.07 s**
(nvars=20), ~14–19% on top of the table above (the table is a separate cool run, so
don't subtract the cells directly). Full A/B + drift caveats in the ledger.

How we got there, compounding (newest last): word-packed representation (1.1–1.4×)
→ **GF(2⁸) swap** (`Gf8Scheme`, ~2×) → **parallelism** (the biggest single lever —
the prover was single-threaded vs the parallel fused baseline) → **eq-split**
(modest on aarch64; see the finding below) → **Phase-2 Gruen eq-trick** (degree-2
MLE-check, no eq-table fold; ~14–19% on the GF8 arm + smaller proof).

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
**parallelization** · eq-split · **Phase-2 Gruen eq-trick** (degree-2 MLE-check,
no eq-table fold; proof 2 vs 4 coeffs/round). Commits `6660bff` → `3057190` (+ the
Gruen eq-trick, working tree, pending commit) on branch `f2-clean`.

## NEXT STEPS, in priority order

### 1. ✅ DONE — Production integration: sound `ψ_z` binding in `f2_prove` (task #7 pt 2)
**Shipped** (ledger has the full entry; commits `91eebec` `4b5aca8` `317a83a` `aa05f21`).
The oblong discharge's `ψ_z` operand evals are now **bound to the commitment**.

**✅ Measurement-first increment (earlier).** Measured (Apple M4, nvars=16):
Prove-NoHadamard 46.5 ms, Prove-Hadamard (fused) ~567 ms, Prove-Hadamard-Oblong
~101 ms — the oblong discharge adds **~54 ms** (vs the fused's ~390–520 ms), **~5.6×
faster** Hadamard prove.

**✅ Sound binding — the mechanism that actually worked (un-lifted open, NOT a
second z-open).** The original "second multipoint + second z-open" plan was
**superseded**: instead of opening twice, the **open was rewritten to be un-lifted**
(`b840839`) so its bound per-column claim is the bit-slice poly `a' = Σ_b c_b·X^b`
(coefficients = the bit-slice MLE evals). That **one** bound object yields *both*
projections — `ψ_α(col)(r_0) = Σ_b c_b·α^b` (main, via the open's Check 2) **and**
`ψ_z(col)(r_0) = Σ_b c_b·L_b(z)` (discharge). So the discharge rides the **same**
open. Concretely (`prove/verify_f2_full_with_oblong_hadamard`):
- The discharge's `ψ_z(col↓Δ)(γ_word)` AND-pair claims fold into the **same** main
  multipoint-eval as pointed-shifts, over z-projections of **all witness primary
  cols** appended to the trace (claimed at `r*`), reduced to the single `r_0`.
- The open exposes its batch `γ`; the **ψ_z binding check**
  `ψ_z(a') == Σ_g γ_g·z_r0_evals[g]` ties the z-evals to `a'`; `oblong_tie_from_bound`
  recombines the now-bound AND pair-evals + trusted adder parents to the operand evals.
- *Why "all witness cols", not just the AND-referenced ones*: the open's γ-batch
  mixes every witness col, so binding `ψ_z` via `a'` needs the whole `z_r0_evals`
  vector (can't extract per-col `a'_g` from the batched `a'`).
- Adders ship **trusted** `ψ_z` parents (bit-level carry recurrence doesn't decompose
  into row-shift pair-evals — ledger Issue 1; = fused-discharge soundness parity).
- **Gate met**: `prove_then_verify_f2_full_with_oblong_hadamard_roundtrips` — honest
  accept + corrupt-W / tampered-ψ_z / tampered-pair-eval rejection; 60 protocol tests green.

**Follow-ups (not blockers)**: a `Verify-Hadamard-Oblong` bench arm + e2e prove/verify
A/B vs the fused discharge; a sound adder-carry binding (Issue 1).

### 2. ✅ DONE — Gruen's eq-trick on Phase-2 (degree-2 MLE-check, smaller proof)
**Shipped** (working tree; the ledger has the full entry + A/B). Replaced the naive
degree-3 Phase-2 (folded `eq` as a 4th table + the full 2ⁿ `eq_indicator(r)`, ≈16 MB
at nvars=16) with binius's eq-factored **MLE-check** (`quadratic_mle.rs` +
`mlecheck.rs`): the prover ships the degree-2 prime poly
`h(t)=Σ_rest (a·b−c)(t,rest)·eq_rest[rest]` truncated to `[c₁,c₂]` (the verifier
recovers `c₀ = claim − r_i·(c₁+c₂)`), maintains only the half-size `eq_rest`
(`sum_fold_low`, mul-free), folds **3 tables not 4**, and closes on `a·b−c == claim`
(the `eq_star` factor threads out round by round). Result: **~14–19% on the GF8 arm**
(the GF128 win is Phase-1-dominated, so smaller) + the **proof shrinks 4→2
coeffs/round**. One behavioural change: there's no per-round consistency rejection
anymore — a corrupt witness surfaces at `FinalCheck` (`OblongError::RoundConsistency`
removed; the corrupt-witness tests now assert `FinalCheck`). **Deferred micro-opt**:
binius's 2-eval prover (compute `h(1),h(∞)`, recover `h(0)` from the claim;
~6→4 muls/pair) needs a per-round field inverse + prover claim-tracking — skipped
for prover simplicity (no inverse, no claim thread).

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
`docs(...)` commits) · **Phase-2 Gruen eq-trick** (working tree, pending commit —
Phase-2 rewrite in `oblong_and.rs` + corrupt-witness test updates there and in
`f2_oblong_hadamard.rs`). All work is in the 7 files listed under "Code map".
