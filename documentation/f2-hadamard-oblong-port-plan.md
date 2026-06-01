# Port plan — Binius64 oblong univariate zerocheck for the F_2 Hadamard discharge

*Goal: replace our memory-bound bit-slice AND discharge (1536 GF(2¹²⁸) slices,
2.34× scaling — see `f2-hadamard-handoff.md`) with Binius64's word-packed
**oblong univariate zerocheck**, so the AND check streams ~48 packed word-columns
instead of 1536 slices and the bit dimension is handled by a univariate-skip
round. Reference implementation: the local repo `~/binius64`,
`crates/prover/src/and_reduction/` + `crates/verifier/src/protocols/bitand.rs`.*

## Progress (2026-06-01) — Phases A + B DONE: standalone oblong AND zerocheck works end-to-end (naive GF128)

**Landed** (working tree, `poly` crate, 14 tests green, full lib suite 135 passed,
clippy-clean, no prover path touched):
- `poly/src/univariate/binary_subspace.rs` — P2 (`BinarySubspace`) + P3
  (`lagrange_evals`, `extrapolate_over_subspace`, `evaluate_univariate`). Ports of
  binius64 `math/src/{binary_subspace,univariate}.rs` over `BinaryFieldGF128`.
- `poly/src/univariate/oblong_and.rs` — the additive-NTT word extension
  (`AdditiveNtt::{new,extend_word,fold_word_at}`, `base_lagrange_at`), the Phase-1
  `univariate_round_message`, **and the full standalone protocol**
  `prove_oblong_and` / `verify_oblong_and` (round message → fold at `z` → Phase-2
  eq-weighted degree-≤3 sumcheck → closing `a·b−c==eval`), `eq_indicator`,
  `AndCheckOutput`, `OblongError`. **Naive GF(2¹²⁸)**, no GF(2⁸) lookup yet.
  `SKIPPED_VARS=5`, `WORD_BITS=32`.

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

**Remaining (next sessions), in order:**
1. **GF(2⁸) speed (P1 + P4)**: a `GF(2⁸)` (AES) subfield + `F₂`-embedding into our
   GHASH `GF(2¹²⁸)`, and the `8×256×(WORD_BITS/16)` **byte-lookup NTT** so the NTT
   + per-point products run in the 8-bit field (the naive path does them in
   GF(2¹²⁸)). This is what makes the prover *fast*; needed for the Phase-D A/B.
   Pick our own deterministic skip challenges (binius's `[0x2,0x4,0x10]` are
   AES-basis). The standalone `verify_oblong_and` already accepts evals computed by
   a faster NTT (same values), so this is a drop-in prover-side swap.
2. **Fiat-Shamir**: replace the explicit `z`/`gammas` args with a `Blake3Transcript`
   (prover absorbs the round message, samples `z`; absorbs each round poly, samples
   `γ`). Mechanical; the math is settled.
3. **Phase C** integration seam (§4, route B): tie the `a/b/c_eval` at `(z, γ)` to
   the committed columns via the ψ_α recombination over the oblong point. The
   architectural risk lives here.
4. **Phase D**: all 16 relations (3 ANDs + 13 adders) + A/B at nvars=16/20.

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

## 4. Integration — the architectural seam (highest risk)

Our discharge runs **before ψ_α** (Wiring R): it currently produces
`bit_slice_evals` at `r*_H`, which the **recombination at α** ties to the column
openings (`derive_operand_parents` + `verify_bit_decomposition_consistency`).

The oblong zerocheck instead produces `a_eval/b_eval/c_eval` at `(z, r)`. These
must be tied to the committed columns. Two routes:

- **(A) Port the shift reduction too** (`prover/src/protocols/shift/`,
  `verifier/.../shift`). It's the natural partner: it takes the operand evals +
  the shifted-value-index structure and proves consistency with the committed
  words (the "monster" multilinear). Cleanest, but more code.
- **(B) Adapt our ψ_α recombination** to consume `(z, r)` evals: each operand
  eval is an `F₂`-linear (XOR) combination of its `(col,Δ)` terms' evals at
  `(z, r)`; reduce to per-`(col,Δ)` column evals at `(z, r)`, apply the `↓Δ`
  shift, and open via the existing PCS. This reuses our machinery but must handle
  the oblong `z`-coordinate in the opening (the column openings are over rows; the
  `z` (bit) coordinate needs an extra MLE-eval over the 5 bit-vars). Smaller, but
  the soundness of the adapted tie needs a proof.

**Recommendation: prototype (B)** (less new infra, reuses ψ_α/PCS), fall back to
(A) if the oblong-`z` opening doesn't compose. Either way this seam — not the
zerocheck itself — is the design risk, because it's where our architecture and
Binius's diverge.

Soundness note: the oblong zerocheck's error is `deg(R₀)/|F| + (sumcheck) +
(tie)`; with `|F|=2¹²⁸` and `deg(R₀) ≤ 2·31 = 62` the skip round is negligible;
the tie (§4) carries the real soundness obligation.

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
  evals to the committed columns via route (B). **Gate**: full prove→verify
  round-trip for that relation (à la the existing `plain_and_round_trips`). *The
  architectural risk lives here.*
- **Phase D — all 16 relations + measure.** Wire the 3 ANDs + 13 adders (the
  shifted operands, complement, carry). **A/B** the new discharge vs the current
  fused one at nvars=16/20 (wall-clock + scaling). **Target**: discharge from
  memory-bound ~418 ms / 2.34× scaling down toward the base-prove regime
  (~tens of ms, scaling with cores) — the Binius ~123 ns/AND ballpark.

## 6. Effort & risk

- **Phase A**: ~1 week (field embedding + 4 self-contained math ports).
- **Phase B**: ~1–2 weeks (the oblong kernel + verifier; template available).
- **Phase C**: ~1–2 weeks, **highest risk** (the ψ_α/shift tie + its soundness).
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
3. **The integration tie (§4)** — does route (B) (ψ_α recombination over the
   oblong `(z,r)` point) compose soundly, or do we need the shift reduction (A)?
   This is the one to settle on paper before Phase C.
4. **Adders/carry** — confirm the carry operand `c` (derived, currently trusted
   per `f2_hadamard_plan.md` §6 Issue 1) ties down under the new reduction the
   same way (or better) than today.
