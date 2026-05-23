# F_2 MLE-opening — lift-and-project plan

Continuation of [`f2_prove_plan.md`](f2_prove_plan.md). The current
F_2 prove path lands at MLE evaluation claims of the form

```
MLE[ψ_α(w_g)](r*) = a_g   for each committed column g,
```

with `r* ∈ GF(2^192)^μ` and `a_g ∈ GF(2^192)`. `ψ_α : F_2[X] →
GF(2^192)` is the ring homomorphism `X ↦ α` for a fixed transcript
challenge `α`; `w_g : {0,1}^μ → F_2[X]<D>` is the witness column;
`ψ_α(w_g) : {0,1}^μ → GF(2^192)` is its projection.

This document plans the **next step**: discharging those claims via a
"lift-and-project" reduction to a Zip+ open over `F_2[X]`. The
existing F_2 commit (`RaaF2Code`, `Cw = BinaryPoly<D>`) is reused
unchanged; only the open side is new.

## Setup

Write each MLE evaluation claim as a tensor inner product. Split the
`μ` indexing bits of the hypercube into two halves, `μ = μ_1 + μ_2`,
and reshape `w_g`'s evaluations into a `2^{μ_1} × 2^{μ_2}` matrix
`M_{w_g}` with entries in `F_2[X]<D>`. The eq-MLE factorises:

```
eq(b, r*) = q_1[b_1] · q_2[b_2]   for b = (b_1, b_2),
```

where `q_1 ∈ GF(2^192)^{2^{μ_1}}` and `q_2 ∈ GF(2^192)^{2^{μ_2}}` are
the two halves of the eq-tensor (built by `build_eq_x_r` on the two
halves of `r*` — exactly what `point_to_tensor` already does). The
claim becomes

```
(*)   q_1^T · ψ_α(M_{w_g}) · q_2 = a_g   in GF(2^192).
```

## The lift

`GF(2^192) = F_2[X] / ⟨X^{192} + X^7 + X^2 + X + 1⟩`. We lift
`q_1, q_2 ∈ GF(2^192)^*` to `q_1', q_2' ∈ F_2[X]<192>` so that
`ψ_α(q_i'[k]) = q_i[k]` for every entry. The witness matrix
`M_{w_g}` is already in `F_2[X]<D>`.

**Inverse-of-evaluation lift** (not bit-pattern). The naive
"interpret each `q_i[k] ∈ GF(2^192)` as its canonical degree-<192
representative in `F_2[X]`" is *not* the right lift for a
transcript-fresh α — applying `ψ_α` to the canonical representative
of `g` gives `(g_bits)(α) = Σ g_i · α^i`, which equals `g` only when
`α` is the field's quotient generator `X` itself.  For a
random α, that identity fails (and the verifier check
`ψ_α(a') = a` fails accordingly).

Correct construction: with overwhelming probability over a
transcript-fresh α, `{1, α, α^2, …, α^{191}}` is an F_2-basis of
`GF(2^192)`. The lift is the unique `c ∈ F_2^{192}` solving
`Σ_j c_j · α^j = q_i[k]`; we set `q_i'[k] := Σ_j c_j X^j ∈
F_2[X]<192>`. By construction `ψ_α(q_i'[k]) = q_i[k]`. The lift
table (a 192×192 F_2 matrix inverse) is precomputed once per α and
reused across all `q_i[k]`. See
[`AlphaPolyBasis`](../../poly/src/univariate/binary_gf192.rs) for
the concrete implementation; cost is `O(192³)` F_2 ops for the
inverse + `O(192²)` ops per entry.

Compute the **unreduced** product over `F_2[X]`:

```
(**)  a_g' := q_1'^T · M_{w_g} · q_2'   in F_2[X].
```

`a_g'` is a single F_2[X] polynomial of degree at most `(192 - 1) +
(D - 1) + (192 - 1) = D + 381`. For `D = 32` that's degree ≤ 413
(414 bits, ≈52 bytes serialised — vs 24 bytes for `a_g` in
GF(2^192)).

**Lift correctness.** `ψ_α` is a ring hom `F_2[X] → GF(2^192)`
sending `X ↦ α`. Applying it to (**) coefficient-by-coefficient
commutes through the products and sums:

```
ψ_α(a_g') = ψ_α(q_1'^T · M_{w_g} · q_2')
          = ψ_α(q_1')^T · ψ_α(M_{w_g}) · ψ_α(q_2')
          = q_1^T · ψ_α(M_{w_g}) · q_2
          = a_g.        (by (*))
```

So `(**) ∧ ψ_α(a_g') = a_g  ⇒  (*)`. The verifier discharge of (*) is
therefore: (i) receive `a_g'`, (ii) verify (**) via a Zip+ open over
`F_2[X]`, (iii) check `ψ_α(a_g') = a_g` (one polynomial evaluation
in GF(2^192), ~414 multiplications by `α`).

## Why this fits Zip+ unchanged in structure

The current integer-typed Zip+ proves exactly the same shape of
claim — `q_1^T · M_w · q_2 = a'` — except over a `Z[X]` lift rather
than an `F_2[X]` lift. Concretely:

| Aspect | Integer Zip+ | F_2[X] Zip+ |
|--------|--------------|-------------|
| Witness type (`Eval`) | `Int<N>` | `BinaryPoly<D>` |
| Codeword type (`Cw`) | `Int<K>` (widened) | `BinaryPoly<D>` (no widening — F_2 add is XOR) |
| Challenge type (`Chal`) | `Int<N>` | `BinaryPoly<192>` (= GF(2^192) representative) |
| Eval point (`Pt`) | `Int<N>` | `BinaryPoly<192>` |
| Combined-row type (`CombR`) | `Int<M>` (further widened) | `BinaryPoly<223>` (= D + 192 − 1) |
| Final claim type | `F_q` after `Z → F_q` projection | `GF(2^192)` after `F_2[X] → GF(2^192)` projection (`X = α`) |
| Verifier projection | `mod q` | `eval at α` |

The structural analogy is exact: in both cases the prover commits in
a narrow type, opens in a wider type, and the verifier reduces the
wider-type claim to a field claim via a transcript-fixed projection.
The integer pipeline uses `mod q`; we use `X ↦ α`.

**Key feasibility point — `F_2[X]`-linearity of `RaaF2Code`.** The
encoder is `Repeat → Permute_1 → Accumulate (XOR) → Permute_2 →
Accumulate (XOR)`. Each step is `F_2`-linear, and `F_2[X]`-scalar
multiplication commutes with `F_2` addition + permutation (because
multiplying every entry of a row by a fixed `F_2[X]` polynomial
distributes over XOR and permutation acts only on the *position* of
entries, not on their *values*). So the encoder is `F_2[X]`-linear:
`encode(Σ c_i · row_i)  =  Σ c_i · encode(row_i)` for `c_i ∈
F_2[X]`. This is the condition Zip+ proximity relies on. **No
re-commit, no re-encode** — the existing `commit_f2_trace` output is
already a valid `F_2[X]`-linear commitment.

## Polynomial-bound chain (concrete sizes)

For `D = 32`, `μ_1 = μ_2 = μ/2`, `num_rows = 2^{μ_1}`,
`row_len = 2^{μ_2}`:

| Quantity | Type | Bits | Bytes |
|----------|------|------|-------|
| Witness cell `M_{w_g}[i, j]` | `BinaryPoly<32>` | 32 | 4 |
| Codeword cell (committed) | `BinaryPoly<32>` | 32 | 4 |
| q_1', q_2' entry | `BinaryPoly<192>` | 192 | 24 |
| Single product `q_1'[i] · M[i,j]` | `BinaryF2Poly<4>` (≥223 bits) | 256 | 32 |
| Combined row entry `Σ_i q_1'[i] · M[i,j]` | `BinaryF2Poly<4>` | 256 | 32 |
| Final `a_g'` | `BinaryF2Poly<7>` (≥414 bits) | 448 | 56 |

The "rough" sizes use `BinaryF2Poly<W>`'s `64·W`-bit native packing.
A tighter `BinaryPoly<223>` / `BinaryPoly<414>` would save a few
bytes per opening but require introducing those exact widths to the
trait hierarchy. Recommended: use `BinaryF2Poly<W>` throughout, since
it already exists and the F_2[X] multiplication kernel
`f2_poly_mul<W_A, W_B, W_OUT>` is generic over W.

## Implementation pieces

### Piece 1 — F_2[X] inner-product kernels

The Zip+ `ZipTypes::EvalDotChal` / `CombDotChal` /
`ArrCombRDotChal` slots need F_2[X] flavours:

- **`EvalDotChal`**: `inner_product(eval: &BinaryPoly<D>, alphas:
  &[BinaryPoly<192>]) -> BinaryF2Poly<W_CombR>`. For our setting
  `degree_bound = 0`, so `alphas = [ONE]` and this collapses to the
  identity lift `BinaryPoly<D> ↪ BinaryF2Poly<W_CombR>`.
- **`CombDotChal`**: Same shape, used inside `prove_pre_open_f` for
  the wider type. Same collapse for `degree_bound = 0`.
- **`ArrCombRDotChal`**: `inner_product(row: &[BinaryF2Poly<W>],
  q: &[BinaryF2Poly<W'>]) -> BinaryF2Poly<W''>`. Just F_2[X]
  multiplications (`f2_poly_mul`) accumulated into XOR.

Naming proposal: `BinaryF2InnerProduct` (live in
`poly/src/univariate/binary_f2_wide.rs` next to `f2_poly_mul`), with
const-generic widths so each slot picks its `W_OUT`.

### Piece 2 — A new `ZipTypes` impl for the F_2[X] open

Add `BinPolyF2OpenZipTypes<D>` (next to `TestBinPolyF2ZipTypes` in
[`zip-plus/src/pcs/test_utils.rs`](../../zip-plus/src/pcs/test_utils.rs),
but in production code now — promote out of `#[cfg(test)]`):

```rust
type Eval     = BinaryPoly<D>;
type Cw       = BinaryPoly<D>;             // reuse the F_2 commit
type Chal     = BinaryF2Poly<3>;           // GF(2^192) representative
type Pt       = BinaryF2Poly<3>;
type CombR    = BinaryF2Poly<4>;           // up to D + 192 - 1 bits
type Comb     = BinaryF2Poly<4>;           // single-row combined entry
type EvalDotChal     = BinaryF2EvalDotChal<D>;
type CombDotChal     = BinaryF2CombDotChal<D>;
type ArrCombRDotChal = BinaryF2InnerProduct<4, 3, 7>;
```

`Fmod`, `PrimeTest`, `NUM_COLUMN_OPENINGS` retain their integer-side
roles (the verifier's transcript still hashes those bytes; we just
pick deterministic defaults).

### Piece 3 — `LinearCode<BinPolyF2OpenZipTypes>` for `RaaF2Code`

`RaaF2Code` is already `LinearCode<TestBinPolyF2ZipTypes<D>>`. For
the new ZipTypes it needs the same impl, just with the new
`CombR`/`Comb` flavours. The `encode` method is unchanged (still
emits `BinaryPoly<D>` codewords). `encode_wide` needs to operate on
`BinaryF2Poly<4>` rows (multiply each entry by `+ 1` polynomial
constants — i.e., copy + XOR-accumulate). `encode_f` is unused in
the F_2[X] path (no prime field involvement).

### Piece 4 — A new prove/verify entry point for the F_2[X] open

The bound `F: PrimeField + FromWithConfig<Zt::CombR> + ...` that
`ZipPlus::prove_f` requires expresses the verifier's final
projection target. For us, F = `BinaryFieldGF192`. But the integer
prove/verify path internally constructs the combined row in
`F`-typed form (`prove_pre_open_f` line ~437), which conflates
"the wider commit-domain type" with "the field". For F_2[X] we want
those to be **the same wide type** (`BinaryF2Poly<4>`), and the
projection to F happens only at the very end (`ψ_α(a_g')`).

Cleanest approach: a parallel module
[`zip-plus/src/pcs/phase_prove_f2.rs`](../../zip-plus/src/pcs/) that
mirrors `prove_f_inner` but operates entirely in `BinaryF2Poly<W>`
arithmetic for the combined row, the b-vector, and the final eval.
The verifier mirror lives in `phase_verify_f2.rs`. Both end with
`a_g' ∈ BinaryF2Poly<7>` returned to the caller, who then performs
the `ψ_α(a_g') = a_g` check.

### Piece 5 — Lift step in the protocol layer

In `protocol/src/f2_prove.rs`, add `prove_f2_open` and
`verify_f2_open` that consume the `F2VerifierSubclaim` from the
existing IC+sumcheck pipeline plus the commitments + hints, and run
the F_2[X] Zip+ open per column. Output: the `a_g'`-bundle
(plus the per-column Zip+ open proofs).

The lift itself is trivial: `BinaryFieldGF192` already stores its
data in `BinaryF2Poly<3>`-compatible form (a `Uint<3>` of 192 bits),
so each `q_i` entry can be reinterpreted as `BinaryF2Poly<3>` with no
arithmetic. A helper `BinaryFieldGF192 -> BinaryF2Poly<3>` is the
only new conversion needed.

### Piece 6 — Final verifier `ψ_α` check

`ψ_α(a_g')` evaluates a degree-≤413 polynomial in `BinaryF2Poly<7>`
at `α ∈ GF(2^192)`. Use the existing `eval_f2_poly_d_at` machinery
in [`poly/src/univariate/binary_gf192.rs`](../../poly/src/univariate/binary_gf192.rs);
it currently takes `BinaryPoly<D>` with `D ≤ 64`, so we'd add a
`BinaryF2Poly<W>` variant that walks the `64·W` bits the same way.

## Soundness sketch (not a proof)

Two soundness components combine:

1. **Zip+ over F_2[X]** is sound by the same argument as over Z[X]:
   the linear-code distance bounds the probability that a malformed
   `(M_w, a_g')` survives the column-opening proximity check. The
   probability is `(δ/2)^{NUM_COLUMN_OPENINGS}` where `δ` is the
   minimum distance of `RaaF2Code`; this carries over without
   change since the code is `F_2`-linear (and hence `F_2[X]`-linear,
   which is a strictly stronger linearity).
2. **`ψ_α` zero-test**: if `a_g' ≠ q_1'^T · M_w · q_2'` in `F_2[X]`,
   the F_2[X] open rejects (component 1). If `a_g' = q_1'^T · M_w ·
   q_2'` but `ψ_α(a_g') ≠ a_g`, the verifier's local check rejects.
   The only remaining adversarial strategy is to commit to a `M_w`
   that satisfies `ψ_α(M_w) ≠ ψ_α(w)` for the "real" witness — but
   `α` is a transcript challenge sampled *after* commit (Step 3 of
   the prove pipeline), so this strategy has probability `≤ 192 ·
   2^{-192}` per column (degree of `a_g'` over `|GF(2^192)|`).

## Risks and open questions

1. **Type-system gymnastics.** The `ZipTypes` trait fixes
   `EvalDotChal` / `CombDotChal` / `ArrCombRDotChal` as associated
   types with `Trait Bounds` involving `Zt::CombR` etc. Threading
   `BinaryF2Poly<W>` through with the right `W` per slot may need
   `feature(generic_const_exprs)` or hardcoded widths.
   *Mitigation*: hardcode `W = 4` for the combined row (covers up to
   `D ≤ 64`, leaving room) and `W = 7` for the final eval. These
   work without unstable features.
2. **Merkle-leaf transcription size.** Leaves are `Cw = BinaryPoly<D>`
   = `D/8` bytes (4 bytes for `D=32`). Unchanged from the existing
   F_2 commit. Column openings transcribe `num_rows · D/8` bytes
   per opened column, plus a Merkle path; consistent with the
   integer-side budget at `D = 32`.
3. **Combined-row transcription size.** Prover sends the `num_rows`
   entries of the b-vector (each `BinaryF2Poly<4>` = 32 bytes). For
   `num_vars = 16`, `num_rows = 256`, that's 8 KB per column open
   — bigger than the integer protocol's b-vector (`Int<32>` = 8
   bytes/entry → 2 KB). Mitigation: pack the tightest width
   (`ceil((D + 192) / 8)` = 28 bytes for `D=32`) instead of
   `64·4 = 32`; saves ~10%. Not a blocker.
4. **`ψ_α` cost.** Verifier evaluates a degree-≤413 polynomial in
   GF(2^192) — ~413 mults at ~24-byte each, microsecond-scale.
   Negligible.
5. **Quotient by the field reduction polynomial.** A subtle point:
   `q_i'` is the *unique* degree-<192 representative of a
   `GF(2^192)` element. The product `q_1'^T · M_w · q_2'` in
   `F_2[X]` is computed *without* reducing modulo `X^{192} + X^7 +
   X^2 + X + 1`. The reduction only happens implicitly when the
   verifier computes `ψ_α(a_g')`. This is **deliberate** — if we
   reduced earlier, we'd lose information needed to discharge the
   `F_2[X]`-linear Zip+ open. Worth a doc comment in the F_2[X]
   prove implementation.
6. **Batching across columns.** The Zip+ tree contains all binary
   columns as separate `cw_matrices`. The standard Zip+ batching
   (random-linear-combination per-poly) carries over: the verifier
   samples a per-column challenge `γ_g`, the prover sends a single
   combined `a' = Σ_g γ_g · a_g'`, and the lift check becomes
   `ψ_α(a') = Σ_g γ_g · a_g`. One `F_2[X]` open per
   commitment-tree, not per column.

## Implementation order

1. **F_2[X] arithmetic widening.** Add `BinaryF2Poly<W>`
   multiplication accessors at widths 4 and 7. Add `eval_f2_wide_at`
   (the `BinaryF2Poly<W>`-typed analogue of `eval_f2_poly_d_at`)
   in `binary_gf192.rs`. Unit tests: round-trip
   `ψ_α(lift(g)) = g` for `g ∈ GF(2^192)`.
2. **F_2[X] inner-product traits.** `BinaryF2InnerProduct<W_A,
   W_B, W_OUT>` and the `EvalDotChal` / `CombDotChal` flavours
   that produce `BinaryF2Poly<W_OUT>` outputs. Unit-tested
   independently against the integer-typed analogues.
3. **`BinPolyF2OpenZipTypes` + `LinearCode<...>` for `RaaF2Code`.**
   Standalone ZipTypes promotion. Smoke test: `commit_then_open`
   with a hardcoded witness and a hardcoded eval point.
4. **`phase_prove_f2.rs` / `phase_verify_f2.rs`.** Adapt
   `prove_f_inner` / `verify` to the all-`F_2[X]` flow. Column
   openings reuse the existing Merkle machinery (it's hash-typed,
   not field-typed). End-to-end test against a single committed
   column.
5. **Protocol-level wrapper `prove_f2_open` / `verify_f2_open`.**
   Plumb the F_2[X] open into the existing
   `commit_prove_verify_f2_roundtrip` test. Final result: a
   roundtrip that proves all of Steps 0, 2, 3, 4, 7 of the F_2
   pipeline, with the verifier's last act being the `ψ_α(a') = a`
   check.
6. **Batched open across columns.** Add the per-column
   `γ_g`-batching so multiple committed columns produce a single
   F_2[X] Zip+ open.

Pieces 1-2 are pure additions in `poly/`; piece 3 is a new ZipTypes
struct (no breaking changes); pieces 4-5 are new functions in
`zip-plus/` and `protocol/`; piece 6 is an optional optimisation.
None of this requires touching the existing F_2 commit code or the
landed IC + sumcheck pipeline.
