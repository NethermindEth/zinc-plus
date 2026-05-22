# F_2 proving path — implementation plan

UAIR variant: every column is typed in `F_2[X]<32>` (no integer / `Z[X]`
columns). Prover and verifier work entirely in characteristic 2 from
the very start.

## Current pipeline (Z-based, for reference)

The existing prover ([`protocol/src/prover.rs`](src/prover.rs)) is a
7-step type-state machine. Each step consumes the previous state and
produces the next:

| Step | What | Field domain |
|------|------|--------------|
| 1 | Prime projection `φ_q : Z[X] → F_q[X]` | `F_q[X]` after step |
| 2 | Ideal check (`prove_combined` / `prove_linear` / `prove_hybrid`) | `F_q[X]` |
| 3 | Evaluation projection `ψ_α : F_q[X] → F_q` at `X = α` | `F_q` after step |
| 4 | Combined CPR + lookup multi-degree sumcheck | `F_q` |
| 5 | Multipoint MLE evaluation | `F_q` |
| 6 | Lift + project to commit lanes | `F_q` |
| 7 | PCS open | (commit-domain types) |

`F_q` is a random prime drawn from the transcript (or the fixed
secp256k1 base prime under the `fixed-prime` branch).

## F_2 pipeline (target)

| Step | What | Field domain |
|------|------|--------------|
| 1 | (mute — already in `F_2[X]`) | `F_2[X]` |
| 2 | Ideal check over `F_2[X]` | `F_2[X]` |
| 3 | Evaluation projection `ψ_α` at `X = α ∈ GF(2^192)` | `GF(2^192)` after step |
| 4 | Combined CPR + lookup multi-degree sumcheck over `GF(2^192)` | `GF(2^192)` |
| 5+ | MLE evaluation claims & PCS open | — out of scope for this iteration |

User instruction:

> The proving is similar as before, except that now there's no
> sampling of a random prime. So the step where we reduce from Z[X] to
> F[X] is simply mute: we are already in F_2[X]. The ideal check part
> happens in the same way. Then, in the project to F step, we sample
> a random element alpha from a fixed field extension of size 2^192 of
> F_2, and replace X by alpha. This way, our constraints are now over
> the binary field F_{2^192}. Now we run the sumchecks and everything
> (the rest of the protocol) as usual, over F_{2^192}. After that, we
> are left with MLE evaluation claims of the columns projected onto
> F_{2^192}. I will explain how to prove those in a next iteration of
> our work. For now let's implement up to this point.

## Pieces landed

**Commit a3d0f9e — F_2-RAA commit primitive.** `RaaF2Code` (XOR
accumulate, no codeword widening). `Cw = BinaryPoly<32>` instead of
the widened `DensePolynomial<i64, 32>`. The Merkle commitment now
stores genuine F_2-codeword cells.

**Commit a3d0f9e — standalone F_2[X] linear combination.**
`BinaryF2Poly<W>` (bit-packed `F_2[X]<64·W>`) + `f2_poly_mul` (F_2
carryless schoolbook) + `f2_lin_comb` (`F_2[X]<32>` cells with
`F_2[X]<128>` coefficients → `F_2[X]<160>` combined-row entries).

**Commit 310ce0b — GF(2^192) field type + evaluation map.**
`BinaryFieldGF192` (NIST FIPS 186-2 reduction polynomial) with
Add/Sub/Mul/Inv/Pow and `a^{2^192} = a` verified by test. Plus
`eval_f2_poly_d32_at` / `eval_f2_u64_poly_at` /
`eval_f2_wide_poly_at` for the X = α substitution.

## Pieces remaining

### Step 2 — Ideal check over GF(2^192)[X]

The existing `IdealCheckProtocol::prove_{combined,linear,hybrid}`
([`piop/src/ideal_check.rs`](../piop/src/ideal_check.rs)) bind `F:
PrimeField` and operate on `F[X]`-typed polynomial witnesses (so
challenges and witnesses share the same coefficient field). The F_2
analogue follows the same shape:

- **Coefficient field**: `GF(2^192)`. Random challenges are drawn
  from `GF(2^192)`.
- **Polynomial ring for the IC**: `GF(2^192)[X]`. The constraint
  polynomial expressions, ideal generators, and IC intermediate
  polynomials all live here.
- **Lifting the trace**: every committed cell starts in `F_2[X]`
  (specifically `F_2[X]<32>`). `F_2 ⊂ GF(2^192)` is the trivial
  inclusion, so each `F_2[X]<32>` cell embeds canonically into
  `GF(2^192)[X]` — coefficient 0 stays `GF(2^192)::zero()`,
  coefficient 1 stays `GF(2^192)::one()`. No randomness is used in
  this lift; it's a representation change.
- **IC output point**: lives in `GF(2^192)^{num_vars}` (the MLE
  hypercube of the witnesses, but with GF(2^192)-valued challenges).

Steps 2 and 3 are *not* the same step and do not commute: step 2's
output is still in `GF(2^192)[X]` (polynomials in `X`), step 3 then
specializes `X = α` to land in `GF(2^192)`. This mirrors the original
protocol structure (`F_q[X] → F_q` is the same kind of `X → α`
substitution), just with `F_q` replaced by `GF(2^192)` and the prior
`Z[X] → F_q[X]` step muted because we never had `Z` to begin with.

The arithmetic primitives the IC needs in this regime:
- `GF(2^192) + GF(2^192) → GF(2^192)`: have it (XOR).
- `GF(2^192) × GF(2^192) → GF(2^192)`: have it (clmul + reduce mod
  pentanomial).
- `F_2[X]<32> · GF(2^192) → GF(2^192)[X]<32>`: scalar multiply a
  binary cell by a `GF(2^192)` coefficient — a 32-element vector of
  `GF(2^192)` values (zero where the bit is unset, the scalar where
  it's set). Cheap.
- `GF(2^192)[X] + GF(2^192)[X]` and `GF(2^192)[X] × GF(2^192)`: bag
  of `GF(2^192)` ops. Need a representation, probably a thin
  `DensePolynomial<GF(2^192), D>` — or `DynamicPolynomialF<...>`
  with the existing infrastructure generalised over `Field` rather
  than `PrimeField`.

Concretely: a sound port reuses the IC's combinatorial structure
(per-constraint folding, the sumcheck-style reduction over the row
axis) but swaps every `F::*` primitive for the `GF(2^192)` equivalent
and every `DynamicPolynomialF<F>` for a polynomial type over
`GF(2^192)`.

### Step 3 — Evaluation projection (X → α)

Sample `α ← Transcript::get_challenge() ∈ GF(2^192)` and substitute
`X = α` in every `GF(2^192)[X]`-valued object that survived step 2:

- Each trace cell (originally `F_2[X]<32>`, used by step 2 via the
  trivial `F_2 ⊂ GF(2^192)` coefficient lift) → a single
  `GF(2^192)` value. Already implemented per-cell as
  `eval_f2_poly_d32_at` / `eval_f2_u64_poly_at` /
  `eval_f2_wide_poly_at`; integration into the prover
  state-machine is the missing piece (a new "projected trace"
  type analogous to
  [`piop/src/projections.rs::ColumnMajorTrace<F>`](../piop/src/projections.rs),
  but with `F = GF(2^192)` and dense MLEs over `GF(2^192)`).
- The IC's emitted `GF(2^192)[X]` polynomials (constraint claims,
  evaluation-point coordinates, etc.) → `GF(2^192)` values via
  ordinary polynomial-at-a-point evaluation. The existing
  `EvaluatablePolynomial::evaluate_at_point` machinery should fit
  once the coefficient type is `GF(2^192)`.

After step 3, every value the verifier and prover are reasoning about
is a single `GF(2^192)` element — no polynomial-in-X anywhere.

### Step 4 — Sumcheck over GF(2^192)

The existing [`MultiDegreeSumcheck`](../piop/src/sumcheck.rs) binds
`F: PrimeField`. `BinaryFieldGF192` is a `Field` but not a
`PrimeField` (no integer modulus). Two options:

- **(A) Synthetic PrimeField impl.** Give `BinaryFieldGF192` a
  degenerate `PrimeField` impl with `Config = ()`, `Modulus = ()`,
  `modulus_minus_one_div_two = 0`, `make_cfg = ()`. Works as long as
  no call site reads `modulus()` or `modulus_minus_one_div_two()` for
  anything load-bearing. Quick — but mathematically dishonest and
  fragile to future code that does read those.

- **(B) Narrow the trait bound.** Audit which `PrimeField` methods
  the sumcheck (and IC) actually use; carve out a `ProtocolField`
  trait that both `PrimeField` and `BinaryFieldGF192` satisfy. Then
  change the relevant signatures from `F: PrimeField` to
  `F: ProtocolField`. Cleaner; touches ~36 files. Multi-day.

Recommendation: do (A) first to unblock end-to-end, with a
`#[cfg(debug_assertions)] panic!` in any unused-but-required method.
Convert to (B) when the F_2 path becomes load-bearing.

### Step 5+ — Out of scope for this iteration

The user said: "After that, we are left with MLE evaluation claims of
the columns projected onto F_{2^192}. I will explain how to prove
those in a next iteration of our work."

## Implementation order

1. **Decide (A) vs (B)** for the `PrimeField` trait-bound question.
   Everything downstream is shaped by this choice. (A) = synthetic
   `PrimeField` impl on `BinaryFieldGF192`; (B) = carve out a
   `ProtocolField` trait and narrow piop's bound. See "Step 4" above.

2. **`GF(2^192)[X]` polynomial type.** A `DensePolynomial<GF(2^192),
   D>` (or `DynamicPolynomialF<GF(2^192)>` if the dynamic-degree
   variant is needed for the IC). Should fall out of (A)/(B)
   automatically once `GF(2^192)` satisfies the necessary trait
   surface; if not, add a concrete impl block.

3. **F_2[X] → GF(2^192)[X] coefficient lift.** Trivial map: each
   `Boolean` coefficient becomes `GF(2^192)::zero()` or
   `GF(2^192)::one()`. Small helper `lift_f2_poly_to_gf192_poly`.

4. **Port `IdealCheckProtocol::prove_{combined,linear,hybrid}` to a
   `_f2` variant** that takes lifted-trace and `GF(2^192)`
   challenges. Mirrors the existing combinatorial structure;
   every `F`-typed primitive becomes `GF(2^192)` and every
   `DynamicPolynomialF<F>` becomes the corresponding
   `GF(2^192)[X]` type from step 2.

5. **Wire trace projection** (the eval-at-α already implemented
   per-cell) into a new prover `ProverEvalProjectedF2` type-state,
   parallel to `ProverEvalProjected`. Apply it to both the lifted
   trace cells and any `GF(2^192)[X]` objects the IC emitted.

6. **Hook the existing sumcheck to `GF(2^192)`** — assuming step 1
   chose (A) or (B) appropriately. The sumcheck's internal `F::*`
   primitives map onto `GF(2^192)` field ops.

7. **Define an `F_2 ZincTypes`** that uses `BinaryZt = ZipTypes` with
   `Cw = BinaryPoly<32>` and `RaaF2Code` as the binary lane. The
   arbitrary/int lanes are dead in the all-F_2 UAIR — either elide
   them from this UAIR's `UairSignature` or wire them to no-op
   `ZipTypes`. (The current trait shape requires three lanes; a
   small `ZincTypes`-trait tweak would let us drop them.)

8. **End-to-end test**: a tiny UAIR with all-F_2 columns and a
   single linear constraint over an all-zero trace. Prover produces
   a proof; verifier accepts up to the MLE eval claims. (Proving
   the MLE eval claims themselves is the user's next iteration.)
