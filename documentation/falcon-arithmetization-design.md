# Design note: batched Falcon signature verification (Zinc+ UAIR)

**Status:** design + skeleton, on branch `falcon` (off `main-beta`). A UAIR that
verifies **2^10 Falcon (FN-DSA / FIPS 206) signatures at once**, under a
**shared public key `h`** with **per-signature messages**. Scope is the
**algebraic core**: `HashToPoint` (SHAKE-256) and `Decompress` are out of scope —
the hash point `c` is taken as public input and `s_2` as a witness. Modulus
handling is **Option B** (`Z[X]` + explicit mod-`q` quotient). The ring is
**negacyclic** `Z_q[X]/(X^n+1)`.

Skeleton: `test-uair/src/falcon.rs` (`FalconBatchUair`, `falcon_norm_comb_fn`).
Cross-refs: `uair/src/ideal/rotation.rs` (`RotationIdeal`, `remainder_is_zero`);
`piop/src/lookup/booleanity.rs` (the zerocheck group the norm adapts);
`piop/src/sumcheck/multi_degree.rs` (the shared multi-degree sumcheck);
`test-uair/src/ecdsa.rs` (algebraic-AIR template).

---

## 0. TL;DR

Parameters: `q = 12289`, `n = 512`, `W = 32` (limb degree), `L = n/W = 16` (limbs
per ring element), `(q−1)/2 = 6144`, `⌊β²⌋ = 34034726`. Batch size `2^10`.

**One signature per row.** The trace has `2^10` rows. Each ring element
(`h, c, s_1, s_2, u`, degree `< n`) is stored as `L = 16` **limb cells**, each a
degree-`<W` `arbitrary_poly` cell of 32 integer coefficients:
`P = Σ_{m<L} X^{W·m} · limb_m`. Coefficients are **centered** in `(−q/2, q/2]`.

| piece | constraint | how |
|---|---|---|
| **ring eq (per row = per sig)** | `s_1 + s_2·h − c − q·u ∈ (X^n+1)` | reconstruct each poly from its `L` limbs, one `assert_in_ideal` vs `RotationIdeal::<R,N>::new(−1)` — **one constraint verifies all 2^10 sigs** |
| **norm bound (per sig)** | `Σ_i s_1[i]² + Σ_i s_2[i]² + slack = ⌊β²⌋`, `slack ≥ 0` | a **booleanity-adapted zerocheck over the signature rows**, reading the coefficient-slices of the `s_1, s_2` limb cells — **no second representation** |

Why limbs of degree `< 32` rather than one `D = n` cell: it keeps cells at the
native, exercised width **and** makes the coefficients directly available to the
norm via the coefficient-slice machinery (the integer analogue of booleanity's
bit-slices). The same limb cells serve both constraints — there is **no second
representation and no consistency bridge**.

---

## 1. Layout

`2^10` rows (signature `j` per row). Columns (`arbitrary_poly`, each logical
polynomial = `L = 16` limb cells of degree `< 32`):

| columns | role | kind |
|---|---|---|
| `h[0..L]` | shared public key (identical on every row) | public |
| `c[0..L]` | per-signature hash point | public |
| `s_1[0..L]` | recomputed short-vector half (centered) | witness |
| `s_2[0..L]` | decompressed signature half | witness |
| `u[0..L]` | mod-`q`/negacyclic quotient (Option B) | witness |

Plus one `int` column `slack` (per-signature squared-norm slack, witness). Total
`5·L = 80` `arbitrary_poly` columns + 1 `int`. (Shared-key/per-message: `h` is
one public polynomial broadcast to every row; `c` varies per row.)

The full degree-`<n` polynomial is the limb reconstruction
`P(X) = Σ_{m<L} X^{W·m} · limb_m(X)`.

---

## 2. Ring equation — one per-row ideal check

Per signature (per row), Option B:

```
  s_1 + s_2·h − c − q·u  ∈  (X^n + 1).
```

`constrain_general` reconstructs `s_1, s_2, h, c, u` from their limbs (each
`Σ_m X^{W·m}·limb_m`, built with `mbs` by the constant scalars `X^{W·m}`, which
is why `Self::Scalar = DensePolynomial<R, N>` — it must hold degree `W·(L−1) =
480 < N`), forms the residual, and asserts membership in
`RotationIdeal::<R, N>::new(−1)` (the negacyclic ideal `(X^n − (−1))`).
`remainder_is_zero` (`uair/src/ideal/rotation.rs:98`) reduces the degree-`<2n`
residual modulo `X^n+1` in `O(n)`, root-agnostic, so `a = −1` works unchanged.

Because this is a per-row UAIR constraint, **one constraint discharges all 2^10
signatures' ring equations**. `u` is range-bounded (`|u[i]| ≲ 2²²`, the
negacyclically-reduced `s_2·h` over `q`), checked alongside the coefficient
ranges (§3). The whole thing is checked over the random soundness prime `q₀`
(the existing `φ_q`); `q₀ ≫ ⌊β²⌋` so the norm (§3) does not wrap.

Perf note: the `X^{W·m}` scalars are constants rebuilt inline per row in the
skeleton; hoist them once when wiring the prover.

---

## 3. Per-signature norm — a booleanity-adapted zerocheck

Each signature must *individually* satisfy `Σ_i s_1[i]² + Σ_i s_2[i]² ≤ ⌊β²⌋`
(not the aggregate). With one signature per row, this is a **zerocheck over the
signature rows**, directly adapting `piop/src/lookup/booleanity.rs`:

```
  Σ_j eq(r, j) · ( Σ_slices slice(j)²  +  slack(j)  −  ⌊β²⌋ )  =  0.
```

- `slice` ranges over the `2N` **coefficient-slices** of the `s_1, s_2` limb
  cells: for limb `m` and position `p`, `slice_{m,p}(j)` is the coefficient
  `32m+p` of signature `j`. These are extracted from the committed limb cells
  exactly as booleanity extracts bit-slices (`compute_bit_slices_flat`),
  generalized from `{0,1}` bits to integer coefficients — so **no second
  representation** is committed.
- `slack(j)` is the per-signature `int` witness. `⌊β²⌋` is public.
- The per-row `eq`-zerocheck enforces the bracket `= 0` for **every** signature
  independently (exactly how booleanity enforces `v²−v=0` per row), giving the
  per-signature bound. The per-row combiner is `falcon_norm_comb_fn` (degree 2).

This is a new sumcheck group structured like booleanity: a degree-2 group pushed
into the shared `MultiDegreeSumcheck`, sharing the final point `r*`; its endpoint
evaluations (the coefficient-slices and `slack` at `r*`) are bound to the
committed limb columns through the existing opening (the CPR `up_eval`/closing
mechanism booleanity already uses for `int_witness_bit_cols`). Claimed sum `0`.

`slack ≥ 0` and the **centered ranges** `s_1[i] ∈ (−q/2, q/2]` (and the `s_2`,
`u` bounds) are separate range checks — a bit-decomposition + booleanity pass on
the coefficients. These are the soundness-critical bound (a non-minimal `s_1[i]`
must not be able to fake a smaller norm) and are the bulk of the norm-side cost.

Computed over `q₀ ≫ ⌊β²⌋`, the field sum equals the integer norm (no wraparound,
guaranteed once the coefficient bit-widths bound `|s_i|`).

---

## 4. Soundness map

| ingredient | status |
|---|---|
| ring eq `s_1+s_2·h−c−q·u ∈ (X^n+1)` per sig | per-row ideal check; `RotationIdeal::<R,N>::new(−1)` (existing, `a=−1` supported); reconstructed from limbs |
| mod-`q` (`q=12289`) | Option B: explicit `q·u` quotient + range on `u` |
| per-sig `‖·‖² ≤ ⌊β²⌋` | booleanity-adapted zerocheck over rows + `slack ≥ 0` range; per-row `eq` ⇒ per-signature enforcement |
| centered `s_1[i] ∈ (−q/2,q/2]` | bit-decomposition + booleanity (soundness-critical) — to be wired |
| coefficient access for the norm | coefficient-slices of the limb cells (booleanity bit-slice machinery, generalized to integer coeffs) — **no second representation** |
| `c = HashToPoint`, `s_2 = Decompress` | **out of scope**: `c` public, `s_2` witness |

---

## 5. Relation to code / implementation plan

| need | hook |
|---|---|
| negacyclic reduction | `RotationIdeal::<R, N>::new(R::from(-1))` (`uair/src/ideal/rotation.rs`) |
| ring-eq UAIR (limb reconstruction + ideal check) | **done** in `test-uair/src/falcon.rs` (`FalconBatchUair`) |
| shared multi-degree sumcheck | `piop/src/sumcheck/multi_degree.rs` (`MultiDegreeSumcheckGroup::new`) |
| **norm zerocheck group** (the main remaining work) | clone `piop/src/lookup/booleanity.rs`: a degree-2 group whose combiner is `falcon_norm_comb_fn`, claimed sum 0, endpoint bound via the CPR `up_eval` mechanism (as `int_witness_bit_cols` does); wire into `protocol/src/{prover,verifier}.rs` step-4 (both unfolded and folded paths) |
| coefficient-slice extraction for `arbitrary_poly` cells | generalize `compute_bit_slices_flat` from `BinaryPoly` bits to integer coefficients |
| range / centering checks | booleanity on coefficient bit-slices + `(X−2)` value reconstruction |

**Phase 1 (this commit):** layout, the ring-eq UAIR, the norm combiner, the doc.
**Phase 2:** the norm zerocheck group + coefficient-slice extraction + range
checks + prover/verifier wiring + a real witness generator.

---

## 6. Open questions

1. **Limb count.** `L = N/W = 16` for Falcon-512. A "5 limbs" request is
   `n = 160` (reduced ring) — `L` is the single knob (`N = W·L`).
2. **Centering range checks at scale.** `2^10 · 2 · 512` coefficient ranges is
   the dominant cost; confirm bit-decomposition + booleanity is the route (vs a
   revived `Word` lookup), and how `u`'s `~22`-bit range is checked.
3. **Per-row scalar hoisting.** Build the `X^{W·m}` reconstruction scalars once,
   not per row.
4. **Padding.** `2^10` rows exactly = `2^10` sigs (no padding); if `num_vars`
   exceeds `SIGS_LOG`, gate the ring eq / norm with a row selector.
5. **Endpoint binding for the norm group.** Confirm the coefficient-slice
   evaluations at `r*` bind to the committed limb cells via the CPR `up_eval`
   path without extra opening rounds (the booleanity precedent).

---

## Phase-2b status & open bug (step-4 wiring + e2e)

**Done & green:** the norm zerocheck group (`piop/src/lookup/norm.rs`,
unit-tested), the `NormSpec` signature accessor, the valid witness generator,
and the step-4 norm-group wiring into the **unfolded** prover/verifier
(`CprProof.norm_value_evals` + serialization, `ProtocolError::Norm`, dynamic
group index). All 12 pre-existing protocol e2e tests still pass (no regression;
the new serialized field round-trips). The architecture is confirmed viable:
`prove()` takes `project_scalar` generically and the combined poly is
`DynamicPolynomialF`, so the scalar degree (`N=512`, for limb reconstruction)
is decoupled from the cell degree (`W=32`); only the `do_test` *helper* coupled
them, so the Falcon e2e uses a custom harness.

**Open bug** (`protocol` test `test_e2e_falcon_batch`, currently `#[ignore]`):
the end-to-end test compiles and the **prover succeeds**, but the **verifier
fails the ring-equation ideal check** — `ProtocolError::IdealCheck(NotInIdeal)`,
*not* `ProtocolError::Norm`, so the norm wiring is **not** implicated. The i64
witness test proves the residual reduces to exactly `0 mod (X^n+1)`, so the
defect is in the full protocol's handling of the **limb-reconstruction ring
constraint** — a regime no prior UAIR exercised: cell-poly X-degree `~2n`,
a `W=n` rotation ideal, and a degree-`n` scalar (`X^{32m}`). The prover only
*claims* the constraint value, so it doesn't notice; the verifier's membership
check rejects. **Next:** isolate whether it's (a) scalar projection /
`MulByScalar` of the high-degree `X^{32m}` scalars, (b) the combined-poly
product of two reconstructed operands, or (c) the `RotationIdeal<_, 512>`
reduction at this degree — e.g. by unit-testing the ideal check on a single
hand-built reconstructed residual. (A fallback worth weighing: whole-poly
`D=2n` cells with the one-line ideal check and **no** reconstruction, trading
the native-cell-size benefit of degree-`<32` limbs for a simpler constraint —
this is the design tension flagged when the limb layout was chosen.)

**Not yet wired:** the folded prover/verifier paths (`prove_folded`,
`prove_folded_4x_inner` and their verifiers) — they leave `norm_value_evals`
empty, so existing folded tests pass but a folded Falcon proof would skip the
norm group.

## How to use this doc

Ledger for the batched Falcon arithmetization on branch `falcon`. **Out of
scope** for `documentation/f2x-sha-todo.md` (that tracks the `F_2` SHA-256
prover path; this is a separate arithmetization). Append Phase-2 outcomes here
(lead with *what*, then *why*, then *result*).
