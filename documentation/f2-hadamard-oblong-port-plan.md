# Port plan — Binius64 oblong univariate zerocheck for the F_2 Hadamard discharge

*Goal: replace our memory-bound bit-slice AND discharge (1536 GF(2¹²⁸) slices,
2.34× scaling — see `f2-hadamard-handoff.md`) with Binius64's word-packed
**oblong univariate zerocheck**, so the AND check streams ~48 packed word-columns
instead of 1536 slices and the bit dimension is handled by a univariate-skip
round. Reference implementation: the local repo `~/binius64`,
`crates/prover/src/and_reduction/` + `crates/verifier/src/protocols/bitand.rs`.*

## Progress (2026-06-01) — Phases A + B + GF(2⁸) speed lever + Phase-C ψ_z tie DONE

**Landed** (working tree, 29 tests green — 23 `poly` + 6 `protocol`, no prover
path touched):
- `poly/src/univariate/binary_subspace.rs` — P2 (`BinarySubspace`) + P3
  (`lagrange_evals`, `extrapolate_over_subspace`, `evaluate_univariate`). Ports of
  binius64 `math/src/{binary_subspace,univariate}.rs` over `BinaryFieldGF128`.
- `poly/src/univariate/oblong_and.rs` — the additive-NTT word extension
  (`AdditiveNtt::{new,extend_word,fold_word_at}`, `base_lagrange_at`), the Phase-1
  `univariate_round_message`, **and the full standalone protocol**
  `prove_oblong_and` / `verify_oblong_and` (round message → fold at `z` → Phase-2
  eq-weighted degree-≤3 sumcheck → closing `a·b−c==eval`), `eq_indicator`,
  `AndCheckOutput`, `OblongError`. **Naive GF(2¹²⁸)**. `SKIPPED_VARS=5`, `WORD_BITS=32`.
- `poly/src/univariate/binary_gf8.rs` (P1) — `Gf8` derived from GHASH (`θ`=relative
  norm, `m=minpoly(θ)`, `α↦θ` ⇒ field hom), `embed: Gf8→GF128`, verified over all
  65536 pairs. `poly/src/univariate/oblong_and_gf8.rs` (P4) — `Gf8Ntt` byte-lookup
  additive NTT + `gf8_round_message` + `embed_subspaces`; the NTT + products run in
  GF(2⁸), cross-checked against the GF(2¹²⁸) computation over `embed(H₈)`.
- `protocol/src/f2_oblong_hadamard.rs` (Phase C) — `prove/verify_oblong_and_relation`
  for one AND relation + the **`ψ_z` recombination tie** (reuses `pair_alpha_evals`
  + `derive_operand_parents` with `base_lagrange_at(z)` weights and the row-point
  `γ`). Explicit challenges; in-memory tie (no PCS open yet).

**De-risked**:
- The **field is directly compatible** (§0 empirically confirmed; Open-Q #1
  resolved for the GF(2¹²⁸) path — our monomial/LSB-first basis with the standard
  `{X⁰…X⁵}` subspace is self-consistent; binius byte-matching only matters for the
  GF(2⁸) embedding + cross-checking binius vectors).
- The **whole protocol is correct over our GHASH field**: the Phase-1 cross-check
  (`round_message_matches_folded_sum_claim`) AND the full prove→verify round-trip
  (`full_round_trip_accepts_honest`, n=4/6/8) pass; corrupting a C bit is rejected
  at round 0; a tampered closing eval fails the final check. (Found+fixed one bug:
  `eq_indicator` was big-endian vs `fold_low`'s low-bit-first binding.)
- The **memory win is structural/real already**: 48 packed word-columns (post-fold
  48 GF128 MLEs) vs 1536 GF128 bit-slices = the ×D=32 data-volume cut.
- The **GF(2⁸) speed lever works**: single-relation Phase-1 round message at
  nvars=16 is **2.56× faster** in GF(2⁸) (99.6 vs 255 ns/word), *with the
  eq-weighting still per-word in GF(2¹²⁸)* — headroom remains (eq-split below).
- The **Phase-C `ψ_z` tie composes** (the architectural risk): the recombination
  derives the correct operand evals for plain / shifted / complemented / Maj-combo
  operands, and `tie_catches_wrong_operand_wiring` shows it rejects a mis-bound
  column — real soundness teeth, a drop-in reuse of the `ψ_α` machinery.

**Remaining (next sessions), in order:**
1. **Phase C finish** — (a) **Fiat-Shamir**: sample `z`/`γ` from a
   `Blake3Transcript` in `protocol` (drop the explicit args); (b) **PCS opening at
   `γ`**: the tie checks the derived evals against the *in-memory* columns; the
   real protocol opens the `ψ_z`-projected columns through the PCS at `(z, γ)`
   (the §4-(i)/(ii) projection-point choice — start with (i), `z` only on the
   discharge columns) + the `↓Δ` shift opening.
2. **eq-split (more speed)**: split the `n` row-vars into ≤3 **deterministic
   GF(2⁸)** skip challenges `{α,α²,α⁴}` (eq weighted in GF(2⁸), accumulated per
   `2^k`-word chunk) + the big-field remainder (embed + GF(2¹²⁸) eq once per
   chunk). Cuts the remaining per-word GF(2¹²⁸) mults by the chunk size — the part
   the 2.56× hasn't captured. Packing 16 `Gf8` lanes (binius `PackedAESBinaryField16x8b`)
   is a further SIMD lever. Parameterise the subspace so the GF(2⁸) path
   (`embed(H₈)`) and the naive path share one verifier; the prover is then a
   drop-in (same eval values).
3. **Phase D**: batch all 16 relations (3 ANDs + 13 adders) into one oblong
   zerocheck + A/B vs the current fused discharge at nvars=16/20.

## 0. The big de-risking facts (why this is a port, not a research project)

- **Same field.** Binius64 `B128 = BinaryField128bGhash` has modulus
  `X¹²⁸+X⁷+X²+X+1`, reduces with `0x87`, stored as `u128` — **identical to our
  `BinaryFieldGF128`** (`poly/.../binary_gf128.rs`: `REDUCTION_LOW_GF128 = 0x87`,
  same `f(X)`). The GHASH arithmetic is directly compatible; no
  field-isomorphism/basis rewrite. (Verify only the bit-endianness convention at
  the seam — both look LSB-first/monomial.)
- **It's over GF(2¹²⁸), not a tower.** My earlier "tower-field obstacle" was the
  *original* Binius; Binius64 uses B128 + a *single* 8-bit subfield (AES
  `GF(2⁸)`) only for ≤3 deterministic small-field challenges in the skip round.
- **The witness stays packed.** Operands are `Vec<Word>` (`PackedBinaryField128x1b`),
  the 64 bits are an "oblong" dimension — *never* materialised as separate
  GF(2¹²⁸) slices. This is the memory win.
- **Working reference code exists** for every piece. The math primitives are
  small and self-contained.

## 1. The protocol (adapted to our 32-bit words)

Binius64 proves `A·B − C = 0` for oblong multilinears `A,B,C` over variables
`(Z, X₀..Xₙ₋₁)`: `Z` = bit-index-within-word (their `SKIPPED_VARS = 6` for
64-bit words; **ours = 5 for 32-bit SHA words**), `X` = row/word-index vars.

- **Phase 1 — univariate-skip round** (`bitand.rs:47-68`): prover sends
  `R₀(Z) = Σ_{X∈{0,1}ⁿ} (A·B − C)(Z,X)·eq(X; r)`, a univariate of degree
  `≤ 2(|D|−1)` over a `BinarySubspace` of dim `SKIPPED_VARS+1`
  (**ours: dim 6 = 64 points; |D| = 32**). Since the AND holds, `R₀` is 0 on the
  base domain, so the prover sends only the `|D|` extension-domain evals
  (**ours: 32 values**). Verifier extrapolates `R₀(z)` at a sampled `z` via
  `extrapolate_over_subspace`.
- **Phase 2 — standard degree-2 sumcheck** over the `n` row vars (`bitand.rs:70-79`),
  reducing to `a_eval·b_eval − c_eval = eval`.
- **Output** (`AndCheckOutput`): `a_eval, b_eval, c_eval, z_challenge, eval_point`.
  These per-operand evals are then tied to the committed witness by the **shift
  reduction** (Binius) — in our prover that's the integration seam (§4).
- **Challenges**: first ≤3 word-vars use *deterministic* `GF(2⁸)` challenges
  (`[0x2,0x4,0x10]` in AES basis; tensor product must be `F₂`-independent), the
  rest are random `GF(2¹²⁸)`. The small-field challenges are what let the NTT
  lookup run in the 8-bit field.

The prover's hot kernel is `univariate_round_message_extension_domain`
(`sumcheck_round_messages.rs`): per word-triple it does `3×(D/8)` NTT-lookup
reads + adds into the round message — an additive NTT precomputed as an
`8×256×4` packed-field table (`ntt_lookup.rs`).

## 2. Prerequisites to port/build (with effort, from the repo's own structure)

| # | Piece | Source (binius64) | Effort | Notes |
|---|---|---|---|---|
| P1 | **`GF(2⁸)` subfield + embedding into our GHASH `GF(2¹²⁸)`** | `field/src/aes_field.rs` (`AESTowerField8b`, `x⁸+x⁴+x³+x+1`) | **Med** | The one new field type. Need an `F₂`-linear embed `GF(2⁸) ↪ GF(2¹²⁸)` (8\|128 so a subfield exists) + the deterministic skip challenges in our embedding. |
| P2 | **`BinarySubspace`** | `math/src/binary_subspace.rs` | **Low** | `F₂`-subspace by ordered basis; `get(i)=Σ basis[j]·bit(i,j)`. Self-contained. |
| P3 | **`lagrange_evals_scalars` + `extrapolate_over_subspace`** | `math/src/univariate.rs` | **Low** | O(n) barycentric over a subspace. Self-contained (needs only field ops). |
| P4 | **NTT lookup table** | `prover/src/and_reduction/ntt_lookup.rs` | **Med** | `8×256×4` packed table (~8 KB), additive-NTT-by-byte. Self-contained given a packed-bit field. |
| P5 | **Oblong univariate round-message kernel** | `prover/src/and_reduction/sumcheck_round_messages.rs` | **High** | The hot loop; entangled with `PackedExtension`, `eq_ind_partial_eval`, `FieldBuffer`, rayon. Port carefully. |
| P6 | **Oblong/packed witness layout** | our `BinaryPoly<32>` columns are *already* packed (32 bits/cell) | **Low** | Map operand → `[Word32]` over rows; `Z` indexes the 32 bits. |

P2–P3 are essentially copy-paste (generic over our `BinaryFieldGF128`). P1 and P4–P5
are the real new code, but all have a working template.

## 3. Mapping our 16 relations onto AND constraints

- Our `F2Operand` = "XOR of `(col, ↓Δ)` terms, optionally complemented" is exactly
  Binius's "XOR-sum of *shifted value indices*" (`A = w[i]<<s ⊕ w[j] ⊕ …`). The
  complement `1⊕(…)` is an extra constant term. **Direct correspondence.**
- The **3 AND relations** (C12–C14) map straight to `A·B = C`.
- The **13 adders** (C5–C11) are already in Binius carry-AND form
  (`(x⊕Xc)·(y⊕Xc) = c⊕Xc`) — i.e. AND constraints with shifted operands + the
  derived carry. They fit the same reduction (the `Xc` shift is a shifted value
  index; the carry `c` becomes an operand the shift reduction ties down).
- **Adapt the constants**: `SKIPPED_VARS = 5`, `|D| = ROWS_PER_HYPERCUBE_VERTEX =
  32`, univariate domain dim 6; NTT lookup over a 5-bit subspace, 4 bytes/word.

## 4. Integration — the `ψ_z` recombination (the seam, now well-scoped)

Our discharge runs **before ψ_α** (Wiring R): it currently produces
`bit_slice_evals` at `r*_H`, which the **recombination at α** ties to the column
openings (`derive_operand_parents` + `verify_bit_decomposition_consistency`).

**Key structural fact: the oblong challenge `z` plays the same role as `α`.** Both
are a single random `GF(2¹²⁸)` point that **F₂-linearly collapses a word's `D`
bits into one field scalar** — just in different bases:

- `ψ_α(W) = Σ_b W_b·α^b` — bits as *monomial coefficients*, evaluated at `α`.
- `ψ_z(W) = Σ_i W_i·L_i(z)` — bits as *values*, the oblong univariate (additive-NTT
  Lagrange `L_i` over the subspace) evaluated at `z`.

So the oblong zerocheck's output is exactly `a_eval = ψ_z(A)(r)` **by
construction** (your `base_lagrange_at`·`fold_word_at`). The tie is therefore the
**`ψ_z` recombination** — the direct analog of today's `ψ_α` recombination, with
`L_i(z)` replacing `α^b`. This is the chosen route (was "route B"). Route (A)
(porting Binius's shift reduction) remains the fallback if a column's `↓Δ`
row-shift can't be expressed as a clean `(z, r)` opening.

### Correctness
`ψ_z` is **F₂-linear** (bits in `{0,1}⊂GF(2)`, char 2):
`ψ_z(W⊕W') = Σ_i (W_i⊕W'_i)L_i(z) = ψ_z(W) + ψ_z(W')`. So it commutes with our
XOR-operand structure:
`ψ_z((col↓s ⊕ col_j ⊕ …) ⊕ cmpl) = Σ ψ_z(col↓s) ⊕ cmpl·Σ_i L_i(z)`.
The tie reduces to: derive each operand eval as the F₂-sum of its `(col,Δ)` terms'
`ψ_z`-projected column evals at `r` (+ the complement constant `Σ_i L_i(z)`), then
check it equals the zerocheck's `a/b/c_eval`. Identical shape to
`derive_operand_parents`, with `L_i(z)` for `α^b`. (`ψ_z` is not multiplicative —
fine; the AND is enforced by the zerocheck, the projection only carries the XOR
tie.)

### Soundness — same `≤(D−1)/|F|` as `ψ_α`
`ψ_z(W)` is the unique degree-`≤D−1` univariate interpolating the `D` bits over the
subspace, evaluated at `z`. For `W≠W'` the difference is `Σ_{i∈T} L_i(z)` over the
nonempty differing set — a **nonzero** poly in `z` (Lagrange basis is
`GF(2¹²⁸)`-independent) of degree `≤D−1`, so `Pr_z[ψ_z(W)=ψ_z(W')] ≤ (D−1)/2¹²⁸`
(= `31/2¹²⁸` for `D=32`) — **identical to the monomial `ψ_α`** (`Σ(W_b⊕W'_b)α^b`,
same degree/bound). Conditions, all met: (1) `{L_i}` independent ✓ (Lagrange over
`D` distinct subspace points); (2) `z` random and drawn **after** the column
commitments ✓ (it's the univariate-skip Fiat–Shamir challenge); (3) subspace dim
`≥ log₂ D = 5` ✓. The total discharge error is then
`deg(R₀)/|F| + (degree-2 sumcheck) + (D−1)/|F|`, all negligible at `|F|=2¹²⁸`.

### The one remaining decision (cost, not soundness): which projection point
Today's Wiring-R *reuses* `α` for both the main column projection and the
discharge tie. Since `z ≠ α` and the bases differ (subspace vs monomial), a column
used in the discharge **and** elsewhere (e.g. `W_E`) would be projected at two
points. The PCS opens whatever projected column it's handed, so this is purely an
opening-count question:
- **(i) Keep `α` (monomial) for the main open; project the discharge columns at
  `z` (subspace) too.** Smaller change; costs extra openings for the discharge
  columns (they already get a separate discharge opening today, so the delta is
  the basis + point, not a new opening class).
- **(ii) Re-base the whole projection to the subspace** so `z` is the single
  point everywhere — fully unified with Binius, no double-opening, but it touches
  the IC / multipoint-eval / open path beyond the discharge.

Start with **(i)** (localised to the discharge); consider **(ii)** later if the
double-open cost shows up in the A/B. Either is sound (the argument above is
independent of how many points a column is opened at).

## 5. Phased roadmap

- **Phase A — primitives (P1–P4).** Add a `GF(2⁸)` type + embedding into our
  GHASH field; port `BinarySubspace`, `lagrange_evals_scalars`,
  `extrapolate_over_subspace`, the NTT lookup. **Gate**: unit tests vs binius64's
  (same subspace ⇒ same Lagrange/extrapolate values; NTT lookup matches a naive
  additive NTT). *Mostly mechanical; the field embedding is the only subtlety.*
- **Phase B — standalone oblong zerocheck (P5).** Implement the prover+verifier
  for ONE toy AND `A·B=C` over our `GF(2¹²⁸)`, `D=32`. **Gate**: accepts honest,
  rejects corrupted `C`; the verifier's `a·b−c=eval` check passes; cross-check the
  round message vs a naive `Σ (A·B−C)·eq` reference. *This is the bulk of the new
  code, but `~/binius64` is a line-by-line template.*
- **Phase C — the integration seam (§4).** Wire the oblong zerocheck into
  `prove_f2_hadamard_phase` for ONE simple AND relation (e.g. C12) and tie its
  evals to the committed columns via the **`ψ_z` recombination** (§4; `L_i(z)` for
  `α^b`, soundness verified). **Gate**: full prove→verify round-trip for that
  relation (à la the existing `plain_and_round_trips`). *The
  architectural risk lives here.*
- **Phase D — all 16 relations + measure.** Wire the 3 ANDs + 13 adders (the
  shifted operands, complement, carry). **A/B** the new discharge vs the current
  fused one at nvars=16/20 (wall-clock + scaling). **Target**: discharge from
  memory-bound ~418 ms / 2.34× scaling down toward the base-prove regime
  (~tens of ms, scaling with cores) — the Binius ~123 ns/AND ballpark.

## 6. Effort & risk

- **Phase A**: ~1 week (field embedding + 4 self-contained math ports).
- **Phase B**: ~1–2 weeks (the oblong kernel + verifier; template available).
- **Phase C**: ~1–2 weeks. Soundness of the `ψ_z` tie is now **resolved** (§4);
  remaining risk is the `↓Δ` row-shift opening + the §4-(i)/(ii) projection-point
  choice — fall back to porting Binius's shift reduction (route A) if needed.
- **Phase D**: ~1 week (wiring + A/B).
- Total ~4–6 weeks, gated so we learn early (Phase C is the go/no-go for the
  architecture). The verifier changes (it's a new sub-protocol) — so this is NOT
  byte-identical; gate by accept/reject + soundness, not the existing
  byte-identity tests.

## 7. Open questions to resolve first

1. **Bit-endianness at the field seam** — confirm binius64's `B128` byte/bit
   order matches ours (both reduce with `0x87`, both `u128`; check
   `SerializeBytes`). If they differ it's a trivial fixed permutation.
2. **The `GF(2⁸) ↪ GF(2¹²⁸)` embedding** — pick the subfield generator + the
   deterministic skip challenges so the tensor-product `F₂`-independence holds in
   *our* GHASH field (binius uses AES-basis `[0x2,0x4,0x10]`; ours will differ).
3. **The integration tie (§4)** — ✅ **SOUNDNESS RESOLVED**: the `ψ_z`
   recombination (`L_i(z)` for `α^b`) is correct (F₂-linear ⇒ commutes with the
   XOR operands) and sound (`≤(D−1)/2¹²⁸`, identical to `ψ_α`); see §4. Remaining
   sub-questions for Phase C: (a) the projection-point decision §4-(i) vs (ii)
   (cost, not soundness) — start with (i); (b) the **`↓Δ` row-shift** in the
   opening — confirm `ψ_z(col↓Δ)(r)` is a clean shift of `ψ_z(col)(r)` in our PCS
   (it is for `Δ=0`; `Δ≠0` is the "trusted shifted-eval" gap shared with the
   current discharge, `f2_hadamard_plan.md` §6 / ledger Issue 1) — else fall back
   to route (A), Binius's shift reduction, which exists precisely to discharge
   these shifts soundly.
4. **Adders/carry** — confirm the carry operand `c` (derived, currently trusted
   per `f2_hadamard_plan.md` §6 Issue 1) ties down under the new reduction the
   same way (or better) than today.
