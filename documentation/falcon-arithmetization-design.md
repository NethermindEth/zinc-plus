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
| **ring eq (per row = per sig)** | `s_1 + s_3 − c − q·u ∈ (X^n+1)` | reconstruct each poly from its limbs, one `assert_in_ideal` vs `RotationIdeal::<R,N>::new(−1)`. **Linear** (degree 1): the `s_2·h` product is carried by the committed witness `s_3` (see below) — **one constraint verifies all 2^10 sigs** |
| **product witness** | `s_3 − s_2·h = 0` (zero ideal) | `s_3` = the *full unreduced* product `s_2·h`; degree-2 **`assert_zero`** so it is excluded from the effective degree (keeping the ring eq's effective degree at 1 → **MLE-first**), still bound by the step-4 sumcheck. See Phase-3 status. |
| **norm bound (per sig)** | `Σ_i s_1[i]² + Σ_i s_2[i]² + slack = ⌊β²⌋`, `slack ≥ 0` | a **booleanity-adapted zerocheck over the signature rows**, reading the coefficient-slices of the `s_1, s_2` limb cells — **no second representation** |

The product-witness split makes the UAIR **MLE-first eligible**
(`count_effective_max_degree = 1`), giving a **2.3–5.5× prover speedup** over the
Combined lane (Phase-3 status).

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
| `s_3[0..L3]` | full unreduced product `s_2·h` (`L3 = 2L = 32` limbs) | witness |

Plus one `int` column `slack` (per-signature squared-norm slack, witness). Total
`5·L + L3 = 112` `arbitrary_poly` columns + 1 `int`. (Shared-key/per-message:
`h` is one public polynomial broadcast to every row; `c` varies per row.) The
`s_3` column is the Phase-3 product-witness that makes the ring equation linear
(MLE-first eligible).

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

## Phase-2b status (step-4 wiring + e2e) — COMPLETE

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

**End-to-end test passes** (`protocol` test `test_e2e_falcon_batch`): a real
synthetic Falcon batch proves and verifies, exercising both the per-signature
negacyclic ring ideal-check (`s_1 + s_2·h − c − q·u ∈ (X^n+1)`, reconstructed
from degree-`<32` limbs) and the squared-norm zerocheck against committed
columns. The custom harness decouples scalar degree (`N`) from cell degree
(`32`) and projects the ideal to `RotationIdeal<F, N>`.

**Root cause of an earlier failure (fixed).** The first e2e run failed the
verifier's ring-equation ideal check (`IdealCheck(NotInIdeal)`), with the
combined polynomial collapsing to degree `2·63 = 126` instead of `~2n`.
Diagnosis (the `falcon_*` unit tests in `protocol/src/lib.rs` plus a
`max_degree`/cell-degree print): the reconstruction `Σ_m X^{W·m}·limb_m` was
collapsing to `arb[base] + X^W·Σ(other limbs)` because **every `X^{W·m}` scalar
was returning `X^W`'s projection**. The culprit is `piop::scalar_proj_cache`,
which keys projected-scalar lookups on **pointer identity** (`scalar as *const
S`) on the documented assumption that a UAIR passes each scalar by reference to
a stable local. `FalconBatchUair` built the `X^{W·m}` scalars in a *reused
stack local* inside the reconstruction loop, so all 15 aliased one address; the
cache stored `X^W`'s projection there and returned it for every `X^{W·m}`. Fix:
materialize the reconstruction scalars into a `Vec` (stable, distinct
addresses) — a 5-line change in `FalconBatchUair::constrain_general`. The
pointer-identity cache is a latent footgun for any UAIR that builds distinct
scalar values at the same address; worth a value-keyed fallback or a debug
assertion someday, but the per-UAIR fix is correct and zero-cost here.

**Not yet wired:** the folded prover/verifier paths (`prove_folded`,
`prove_folded_4x_inner` and their verifiers) — they leave `norm_value_evals`
empty, so existing folded tests pass but a folded Falcon proof would skip the
norm group.

**Benchmarks** (`protocol/benches/e2e.rs`, group `Zinc+ Falcon`): mirrors the
SHA / SHA+ECDSA e2e benches (Prove + Verify + proof size, plus a per-step
breakdown), reusing `BenchZincTypes` (i64 cells) and the secp256k1 field;
defaults to `2^{4,6}` signatures with `sample_size(10)` (the prove is heavier
than the linear UAIRs). Initial numbers (Apple Silicon, criterion `--quick`):
**2^4 = 16 sigs → Prove 321 ms, Verify 10.4 ms, proof 3.87 MB raw / 633 KiB
zstd-3**. The proof is *large* for the signature count: the per-row combined
polynomial is degree `~2n` (the limb reconstruction), so the multipoint-eval /
coefficient-MLE data scales with `2n`. This is concrete evidence for the
degree-`<32`-limb vs whole-poly-`D=2n` trade — the limbs keep cells small (and
sidestep the untested large-`D` commitment path) but inflate the constraint
X-degree and hence the proof. Shrinking it is future work (whole-poly cells, or
the coefficient-MLE Mechanism 1 of §4, or batching the reconstruction).

## Phase-3 status (MLE-first product-witness split) — COMPLETE

**What.** The ring equation was `s_1 + s_2·h − c − q·u ∈ (X^n+1)` — a single
degree-2 constraint (the `s_2·h` product of two trace polynomials). Degree 2
made `count_effective_max_degree::<FalconBatchUair>() = 2`, so the prover could
only ever take the **Combined** lane; the fast **MLE-first** lane (which
requires all non-zero-ideal constraints to be linear) never fired. We split the
quadratic out behind a committed **product witness** `s_3`:

- New witness column `s_3` = the **full, unreduced** product `s_2·h` over `Z[X]`
  (degree `< 2n−1`), stored in `L3 = 2L = 32` limb cells. The scalar degree bound
  rises to `2N = 1024` (the reconstruction needs `X^{W·(L3−1)} = X^{992}`).
- **Ring equation** becomes **linear** (degree 1): `s_1 + s_3 − c − q·u ∈
  (X^n+1)` — `s_3` is a trace value, not a product. `RotationIdeal` reduces the
  degree-`<2n` residual mod `X^n+1` exactly as before (the old `s_2·h` residual
  was already degree `<2n`, so this is not new territory).
- **Product constraint** `s_3 − s_2·h = 0` is a degree-2 **`assert_zero`** (zero
  ideal). Because `s_3` carries the *full unreduced* product, the difference is
  *identically* the zero polynomial over `Z[X]`, so the `assert_zero` is honest
  and complete (a *reduced* `s_3` would differ from `s_2·h` by a multiple of
  `X^n+1` and break completeness — hence the full product / 32 limbs).

**Why it's sound.** Zero-ideal constraints are excluded from
`count_effective_max_degree` (now `= 1` → MLE-first eligible) but are **still
bound to the witness**: the step-4 sumcheck runs at `count_max_degree + 2 = 4`
(`protocol/src/prover.rs` `count_max_degree`, `combined_poly_resolver` `+ 2`),
and `ConstraintFolder::assert_zero` folds the product into the RLC like any
other constraint (`combined_poly_resolver/folder.rs`). The ideal-check (step 2)
merely sends `ZERO` for the product's slot (its honest value); the *binding*
lives in the sumcheck. This is exactly the SHA-256 UAIR's pattern (degree-2
`assert_zero` pinning constraints that preserve MLE-first eligibility). The
degree-2 term is **witness×public** (`s_2` witness, `h` public), structurally
identical to SHA's `s_msg_init·(w_W − pa_m)`.

**Result — MLE-first not only fires, it's a large win.** `do_bench_e2e` now
emits a `Prove (MLE-first)` line (gated on `count_effective_max_degree ≤ 1`),
and `test_e2e_falcon_batch` exercises the lane (dispatch `(any_linear=true,
any_nonlinear=false) → step1_mle_first → prove_linear`) and verifies. Apple
Silicon, `--features parallel,simd,unchecked`, `sample_size(10)`:

| batch | Prove (Combined) | Prove (MLE-first) | speedup |
|---|---|---|---|
| `2^4` = 16 sigs | 107.2 ms | **45.8 ms** | **2.34×** |
| `2^6` = 64 sigs | 384.4 ms | **69.7 ms** | **5.51×** |
| `2^10` = 1024 sigs (headline) | 7.38 s | **1.31 s** | **5.65×** |

At the headline `2^10` batch the proof is 6.35 MB raw / 3.04 MB zstd-3. The win
**grows with batch size** (plateauing ≈ 5.6×) because the Combined lane builds the full
row-by-row combined polynomial for the degree-2 product (over `~2n`-coefficient
`F[X]` values) in the ideal check, whereas MLE-first evaluates the *linear* ring
equation via cheap column-MLE evaluations and forces the product slot to `ZERO`
— the product is then handled only in the sumcheck, which **both** lanes pay
equally (both run at degree 4). The 32 extra `s_3` columns add commit cost to
*both* lanes (apples-to-apples), and that cost is far outweighed by the
combined-poly savings. MLE-first (45.8 ms @ `2^4`) also beats the *previous*
single-constraint Combined path (≈ 69.7 ms with features, Phase-2b).

**Why we keep `s_3` as the full product (not reduced + quotient).** A reduced
`s_3` (16 limbs) would need an explicit negacyclic quotient `w` and the
constraint `s_2·h − s_3 − (X^n+1)·w = 0` — two extra witnesses (32 limbs total,
same as the full product's 32) *and* an `X^n`-degree scalar. The full product is
strictly simpler (one witness) for the same limb budget, so it wins.

**Cost / open.** `s_3` adds 32 `arbitrary_poly` witness columns (80 → 112) and
bumps the scalar degree to `2N`. The committed witness grows accordingly, so the
proof size is comparable to (slightly larger than) Phase-2b; the prover *time*
is the headline gain. Shrinking the proof (whole-poly cells, coefficient-MLE
Mechanism 1 of §4, or a smaller `s_3` encoding) remains future work, as does
wiring `s_3` through the **folded** paths (still not wired — see Phase-2b).

## Phase-4 status (folded paths + arb-lane degree fold) — IN PROGRESS

Goal: shrink the Falcon proof. Measured composition of an unfolded Falcon proof
(`nvars=6`, MLE-first): **`zip` (PCS) = 94.4%**, ideal_check 2.0%, resolver
1.1%, combined_sumcheck + multipoint_eval < 0.1%. For Falcon the binary lane is
empty and the int lane is 1 `slack` column, so the PCS bytes are *essentially
all arbitrary-poly*. **Folding the arb lane is the only lever that matters.**

### Done (committed, tested)

- **Norm zerocheck wired into the folded (1× + 4×) prover/verifier**
  (`c7ba13d`). The folded paths inline steps 1–6 and previously skipped the
  norm group, leaving `norm_value_evals` empty. Now mirrored from the unfolded
  `step4_sumcheck` / `step4_sumcheck_verify`, gated on `norm_spec()` (no-op for
  SHA/ECDSA; existing folded tests still pass).
- **`split_arb_column` / `split_arb_columns`** in `zip-plus/src/pcs/folding.rs`
  (`029966a`): degree-split a `DensePolynomial<C, D>` column into
  `DensePolynomial<C, HALF_D>` (`v = u + X^{D/2}·w`), the arb analogue of the
  binary `split_column`. Two rounds (D→D/2→D/4) give the **same 4-block layout**
  (c[0]=a0, c[1]=a2, c[2]=a1, c[3]=a3) as the binary 4× fold, so the verifier's
  `(1−γ₁)(1−γ₂)c[0] + γ₁(1−γ₂)c[2] + (1−γ₁)γ₂c[1] + γ₁γ₂c[3]` reassembly applies
  unchanged. Unit-tested (reconstruction + block layout).

### Three obstacles found (why this is its own path, not a shared-trait edit)

1. **Scalar-degree coupling.** `prove_folded_4x` bounds
   `U::Scalar = DensePolynomial<Int<INT_LIMBS>, D>` (scalar degree = cell degree
   `D=32`). Falcon's scalar is degree `2N=1024` (decoupled, holds `X^{992}`), so
   Falcon cannot even instantiate the bound. Fix: relax to a generic
   `project_scalar` closure as the unfolded `prove` already does.
2. **Arb/int width coupling (the killer).** The shared `IntFoldedZincTypes4x`
   trace is `UairTrace<Int<INT_LIMBS>, Int<INT_LIMBS>, D>` — arb-cell coeff width
   = int-cell width = `INT_LIMBS` — and `split_int_column_4x` requires
   `INT_LIMBS ≥ 4`. So folding arb through the shared trait forces `Int<4>`
   (256-bit) arb coeffs, though Falcon's are ≤30 bits. PCS column size ∝
   (coeff width)×(degree): unfolded `Int<1>×32 = 32`; shared-trait fold
   `Int<4>×8 = 32` — **the 4× degree win is exactly cancelled by the 4× width
   bloat.** Net zero. The arb fold MUST keep narrow `Int<1>` coeffs, which means
   decoupling arb width from the int lane — a dedicated path, not the shared
   `IntFoldedZincTypes4x`.
3. **MultiZip single-point coupling.** `MultiZip3::prove_f` opens all three
   lanes at one shared point. A folded arb opens at `r0_ext = r_0‖γ₁‖γ₂`
   (num_vars+2) while Falcon's unfolded int lane opens at `r_0` (num_vars) —
   different points, can't share one `MultiZip3`. **Resolved by** the existing
   **non-multi** step-7 branch, which already opens each lane at its own point
   (today: arb at `r_0`, binary/int at `r0_ext`). The focused arb fold simply
   *inverts* this: arb at `r0_ext` (split), int/binary at `r_0` (unsplit).

### Design — focused arb-degree fold (narrow coeffs, arb-only)

Fold **only** the arb lane along degree (binary empty, int tiny → left at
`r_0`). New `ArbFoldedZincTypes4x<D, QUARTER_D>` trait:
`ArbitraryZt::Eval = DensePolynomial<C, QUARTER_D>` with the UAIR's *native
narrow* coeff `C` (Falcon: `i64`/`Int<1>`); `BinaryZt`/`IntZt` unfolded at full
degree.

- **Prover** (new `prove_folded_arb_4x`, modelled on `prove_folded_4x_inner`):
  step 0 `split_arb_columns` twice (D→HALF_D→QUARTER_D), commit narrow
  quarter-degree arb; binary/int commit unsplit. Step 6: arb bar_u is the
  full-`D` degree poly already produced by `compute_lifted_evals_capped`
  (no change); sample γ₁,γ₂. Step 7 (non-multi): open **arb at `r0_ext`**, int
  at `r_0`, binary skipped.
- **Verifier**: change `arb_eval_f` from the plain `⟨a, coeffs⟩` inner product
  to the **4-block reassembly** — copy `bin_eval_f` (verifier.rs:2410) verbatim,
  swapping `ZtF::BinaryZt::Chal`→`ArbitraryZt::Chal` and the range to
  `arb_range`. int/binary stay unfolded (`open_eval = bar_u(α)` at `r_0`).
- **Types/test**: a Falcon `ArbFoldedZincTypes4x` instance with a quarter-degree
  (`QUARTER_D=8`) arb IPRS code; an e2e round-trip test + a folded bench line.

Expected: ~4× reduction of the arb PCS bytes (the 94% component) → roughly a 3–4×
smaller Falcon proof.

### COMPLETE — arb fold works end-to-end (`33cbb36`)

- **`split_arb_columns`** (`029966a`) — degree split, unit-tested.
- **`ArbFoldedZincTypes4x<D, QUARTER_D>`** trait + **`prove_folded_arb_4x`**
  (`6c03063`) + **`verify_folded_arb_4x`** + Falcon type instance
  (`TestArbFoldedZincTypes` + `ArbitraryPolyZipTypesIprsQuarter`, `Eval = Cw =
  DensePolynomial<i64, 8>`) + e2e round-trip test `test_e2e_falcon_arb_folded`
  (`33cbb36`).
- Cell type is **`i64`** (not `Int<INT_LIMBS>` as first drafted) — matches the
  existing `ArbitraryPolyZipTypesIprs` and avoids the unproven
  `DensePolynomial<Int<1>,_>` inner-product path.
- Prover: split arb twice (`D→HALF_D→QUARTER_D`), commit/open **arb at `r0_ext`**,
  **int at `r_0`** (unsplit), binary skipped (asserted empty) — per-lane points,
  separate Zips (no MultiZip). Verifier: arb via the **4-block** `eval_f`
  (`bin_eval_f` copy), int via plain `⟨a, coeffs⟩`. `setup`: `pp_arb` for `4n`,
  `pp_int`/`pp_bin` for `n`.

**Result — measured (`num_vars=3`, same trace):**

| | unfolded | arb-folded | reduction |
|---|---|---|---|
| arb PCS (`proof.zip`) | 3 130 752 B | **837 888 B** | **3.74×** |
| total proof | 3 283 510 B | **990 646 B** | **3.31×** |

The 4× degree fold of the arb lane (the 94% component), with narrow `i64`
coeffs preserved, shrinks the Falcon proof ~3.3×.

### ⚠ Pre-existing MLE-first verification bug (found here, NOT the arb fold)

While testing, found that Falcon's **MLE-first proof does not verify**. The
unfolded `test_e2e_falcon_batch` only runs `prove::<false>` (Combined); flipping
it to `prove::<true>` fails with `Resolver(WrongSumcheckSum)` — the *same* error
the arb-fold test hits with `MLE_FIRST=true`. So an MLE-first Falcon proof had
**never been end-to-end verified**: the Phase-3 bench *times* `prove::<true>`
but *verifies* a `prove::<false>` proof, so the **Phase-3 "2.3–5.5× MLE-first
speedup" is for proofs that currently fail verification.**

Root-cause hypothesis: in `prove_linear`, the degree-2 zero-ideal product
`s_3 − s_2·h` is forced to `ZERO` in the ideal-check value, but the CPR sumcheck
(`prepare_verifier`) reconstructs `expected_sum` from those ideal-check values
while the prover's `claimed_sum` folds the *actual* product. The MLE-first
ideal-check appears to evaluate the product as `s3_eval − s2_eval·h_eval`
(product-of-MLE-evals, ≠ 0 at a random point) rather than the MLE-of-product
(= 0 on the hypercube), so `claimed_sum ≠ expected_sum`. This is a piop-level
issue in the MLE-first lane (`evaluate_combined_polynomials` / the zero-ideal
slot handling), independent of the arb fold. **Needs separate investigation**;
the arb fold composes with whichever ideal-check lane verifies (Combined today).

## How to use this doc

Ledger for the batched Falcon arithmetization on branch `falcon`. **Out of
scope** for `documentation/f2x-sha-todo.md` (that tracks the `F_2` SHA-256
prover path; this is a separate arithmetization). Append phase outcomes here
(lead with *what*, then *why*, then *result*).
