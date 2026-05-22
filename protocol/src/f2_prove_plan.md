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

### Step 2 — Ideal check over F_2[X]

The existing `IdealCheckProtocol::prove_{combined,linear,hybrid}`
([`piop/src/ideal_check.rs`](../piop/src/ideal_check.rs)) bind `F:
PrimeField` and produce claims in `F`. For F_2[X] we need a parallel
implementation that operates directly on `F_2[X]<32>`-typed values.
The IC structure (per-row inner products, batch-of-sumcheck-like
folds) carries over, but every arithmetic primitive switches:

- Random challenges drawn from `GF(2^192)` (not a prime field).
- Per-row dot products evaluated in `F_2[X]` and then mapped to
  `GF(2^192)` after `α` is sampled.
- The IC output point lives in `GF(2^192)^{num_vars}` (not `F^...`).

The naive interpretation of "the ideal check part happens in the same
way" needs care: the random challenges in F_q ideal-check come from
the SAME prime field where the constraints live. In F_2[X] the
constraints live in F_2[X] but the random challenges have to come
from somewhere — most naturally `GF(2^192)` (matching where step 3
lands us). This means step 2 in the F_2 path is *effectively running
already in `GF(2^192)`*: the IC's polynomial witnesses are projected
to `GF(2^192)` via `ψ_α` on the fly, and the IC's claim is a
`GF(2^192)` claim.

Equivalently: step 2 and step 3 partially commute. We can sample `α`
early (right after the commitment), project everything to
`GF(2^192)`, and run the IC there. The user's description matches
this: "the ideal check happens in the same way" but operates over
`GF(2^192)`-projected witnesses.

### Step 3 — Evaluation projection

Sample `α ← Transcript::get_challenge() ∈ GF(2^192)`, then for every
trace cell `cell ∈ F_2[X]<32>` apply `eval_f2_poly_d32_at(&cell,
&α)`. Output: column-wise `DenseMultilinearExtension<GF(2^192)>`.

(Already implemented at the per-cell level. Integration into the
prover state-machine is the missing piece — needs a new "projected
trace" type analogous to
[`piop/src/projections.rs::ColumnMajorTrace`](../piop/src/projections.rs).)

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

1. Wire the trace projection into a new
   `ProverEvalProjectedF2` type-state, parallel to
   `ProverEvalProjected`.
2. Either (A) — implement degenerate `PrimeField` for
   `BinaryFieldGF192` — or (B) — narrow the trait bound to
   `ProtocolField`. Decision needed before step 3.
3. Port `IdealCheckProtocol::prove_*` to a parallel
   `prove_*_f2` variant operating in `F_2[X]` with `GF(2^192)`
   challenges.
4. Hook the existing sumcheck to `BinaryFieldGF192` (assuming step 2
   chose (A) or (B) appropriately).
5. Define an `F_2 ZincTypes` that uses `BinaryZt = ZipTypes` with
   `Cw = BinaryPoly<32>` and `RaaF2Code` as the binary lane. The
   arbitrary/int lanes are dead in the all-F_2 UAIR — either elide
   them entirely from this UAIR's `UairSignature` or wire to no-op
   `ZipTypes`. (The trait shape currently requires three lanes.)
6. End-to-end test: a tiny UAIR with all-F_2 columns, single linear
   constraint, all-zero trace → prover produces a proof; verifier
   accepts. Stop at "have an MLE eval claim".
