# Booleanity-catalog reduction: soundness argument

## Purpose

The fold-first (V2) SumFold prover enforces booleanity `D*(D-1) = 0` for a
*catalog* of sources. The full catalog is 17 word columns x 32 bits + 3 virtual
Ch/Maj families x 32 bits = 640 sources. `ShaBooleanityCatalog`
(`piop/src/neutron_nova/projection_sha.rs`) offers two reduced tiers:

- **Tier 1** drops the direct per-bit checks of the five Ch/Maj auxiliary
  columns `Uef, UNegEg, Maj, Ch2Comp, MajComp` (480 sources).
- **Tier 2** additionally drops the four XOR-result columns
  `Sigma0, Sigma1, SmallSigma0, SmallSigma1` (352 sources).

Dropped columns stay committed and keep feeding the residuals, the virtual
recipes, and the openings; only their *own* booleanity zerocheck term is
removed. This document states what remains enforced and argues that the removed
checks were either implied by the survivors or benign. It is a review artifact,
not a machine-checked proof; Section 6 records the residual risk and the gate.

Notation: for a word column `C`, `C[b]` is its bit-`b` value at the row under
discussion, `C[z+s]` is the value `s` rows forward, and `C(2) = sum_b C[b]*2^b`
is the word's value under scalarization at the ideal-check point `X = 2`. All
arithmetic is in the prime field `F` (bn254 scalar field). "Binary" means the
field element lies in `{0, 1}`.

## 1. Hypotheses that survive any reduction

The following are enforced by the verifier regardless of catalog
(`verify_fold_first_sha_sumfold`, `verify_fold_first_linear_ideal_fold_mixed_hyrax`):

- **(H1) Core booleanity.** `A, E, W, MuPacked` and the four overflow columns
  `OvSigma0, OvSigma1, OvSmallSigma0, OvSmallSigma1` are binary at every bit and
  row. These columns are in the never-droppable set of every tier.
- **(H2) Virtual booleanity.** The three reconstructed virtual families are
  binary at every bit and row:
  - `ch1[b] = E[z+2] + E[z+1] - 2*Uef[z+2]  in {0,1}`
  - `ch2[b] = E[z+2] - E[z] + 2*UNegEg[z+2] + 2*Ch2Comp[z]  in {0,1}`
  - `maj[b] = A[z] + A[z+1] + A[z+2] - 2*Maj[z+2] - 2*MajComp[z]  in {0,1}`

  (`reconstruct_virtual_ch_maj_at_row_unchecked`, projection_sha.rs). The
  virtual families are always in the catalog; H1 and H2 are enforced together by
  the gamma-weighted univariate-skip zerocheck (`verify_skip_round`, failure =
  `SkipRoundZeroCheckFailed`).
- **(H3) Ideal membership.** The folded residual polynomials satisfy
  `verify_fresh_sha_ideal_polys`, i.e.
  - `R0, R1 in (X^32 - 1)` (cyclic XOR identity for the big-sigma rotations);
  - `R4, R5, R6, R9, R10 in (X - 1*2)`, i.e. `R_f(2) = 0` (the additive
    recurrences, evaluated as 32-bit integer identities with the `MuPacked`
    carry windows and the integer compensators absorbing overflow);
  - all other families (`R2, R3, R7, R8, R11..R17`) are the **zero ideal** and
    must vanish identically, coefficient by coefficient.

  The ideal map is `production_sha_nonzero_ideals`; `RotX2` is
  `RotationIdeal<F, 1>` so membership is exactly evaluation at `X = 2`
  (`uair/src/ideal/rotation.rs`, `remainder_is_zero` with `W = 1`).
- **(H4) UAIR boundary pins (imported).** `Ch2Comp` and `MajComp` are zero on
  every inner row and nonzero only on the final two boundary rows, pinned by the
  UAIR circuit constraints C17-C21 (`test-uair/src/sha_ecdsa.rs`). These pins
  live in the AIR statement, not in the projection residuals, and are assumed as
  hypotheses here.

Catalog reduction removes none of H1-H4. It removes only the *direct* booleanity
zerocheck terms of the listed columns. The tier identifier and source count are
absorbed into the transcript before any challenge is sampled
(`absorb_sha_booleanity_catalog`), so prover and verifier must agree on the
catalog or Fiat-Shamir diverges.

## 2. Residual read-map of the dropped columns

From the residual builder (`residual_polys_at_row_with_constants`,
projection_sha.rs:11550+) and the virtual recipes:

| Column | Enters residuals as | Enters virtual |
|---|---|---|
| `Uef` | `-(Uef+UNegEg)[z+3]` in **R5 and R6** (as the sum) | `ch1` |
| `UNegEg` | `-(Uef+UNegEg)[z+3]` in **R5 and R6** (as the sum) | `ch2` |
| `Maj` | `-Maj[z+3]` in **R5 only** | `maj` |
| `Ch2Comp` | none | `ch2` |
| `MajComp` | none | `maj` |
| `Sigma0` | `-Sigma0` in **R0** (X^32-1); `Sigma0[z+3]` in **R5** | none |
| `Sigma1` | `-Sigma1` in **R1** (X^32-1); `Sigma1[z+3]` in R5, R6 | none |
| `SmallSigma0` | `-SmallSigma0` in **R2** (zero); `[z+1]` in **R4** | none |
| `SmallSigma1` | `-SmallSigma1` in **R3** (zero); `[z+14]` in **R4** | none |

Two structural facts drive the whole argument:

- **(F1)** `Uef` and `UNegEg` appear in R5/R6 *only through their sum*
  `Uef + UNegEg`, with identical coefficient and shift. `Ch2Comp` and `MajComp`
  appear in *no* residual at all.
- **(F2)** Each XOR-result column enters exactly one ideal family that pins it
  bitwise (R0/R1 mod `X^32-1`, R2/R3 as zero) plus one additive family (R4/R5/R6)
  that reads it at word level (`X = 2`). The pinning family determines the
  column from `A`/`W` and the overflow; the additive family reads it as an
  integer with a shift.

## 3. Tier 1 lemma (Ch/Maj auxiliary columns)

**Claim.** Under H1-H4, dropping the direct booleanity of
`Uef, UNegEg, Maj, Ch2Comp, MajComp` does not admit any witness that both passes
verification and fails to yield a valid SHA execution under extraction.

The honest identities are the bit algebra `a + b - 2*(a AND b) = a XOR b` and
`a + b + c - 2*majority(a,b,c) = a XOR b XOR c`, so honestly all five columns are
binary and, in particular,
`Ch[b] = Uef[b] + UNegEg[b]` with disjoint support (the two AND-terms of the SHA
choose function cannot both fire), which is exactly the sum R5/R6 read.

### 3.1 What virtual booleanity alone admits

Fix the row and bit and hold the core bits `E[z+2], E[z+1], E[z]` (H1). The
constraint `ch1[b] in {0,1}` reads `2*Uef[z+2] = E[z+2] + E[z+1] - ch1[b]`. Over
the four core-bit settings, the admissible `Uef[z+2]` values are:

| `E[z+2] E[z+1]` | honest `Uef` (`ch1`) | other admissible `Uef` (`ch1`) |
|---|---|---|
| 0, 0 | 0  (ch1=0) | -1/2 (ch1=1) |
| 0, 1 | 0  (ch1=0) |  1/2 (ch1=0->1) |
| 1, 0 | 0  (ch1=0) |  1/2 (ch1=1) |
| 1, 1 | 1  (ch1=0) |  1/2 (ch1=1) |

So virtual booleanity admits exactly one non-honest alternative per bit, and it
is always the honest value shifted by `+-1/2` (a "half-integer"). The same holds
for `UNegEg` via `ch2` and for `Maj` via `maj`. Virtual booleanity does **not**
force the aux columns binary; it leaves the half-integer coset.

### 3.2 The admitted deviations split into benign and rejected

Write a deviation as `Uef' = Uef + u`, `UNegEg' = UNegEg + v`,
`Maj' = Maj + m`, `Ch2Comp' = Ch2Comp + c`, `MajComp' = MajComp + w`, each an
element of `{0, +-1/2}` per bit (3.1). Two cases:

**(a) Sum-preserving (benign).** Suppose the deviation keeps every virtual in
`{0,1}` (H2) *and* preserves the word-level reads R5/R6 depend on, i.e.
`(u + v)[z+3] = 0` and `m[z+3] = 0` at the relevant rows. Then by F1 the
residual polynomials R5, R6 (and every other family, since Ch2Comp/MajComp enter
none) are bit-for-bit identical to the honest ones. The folded target, the
openings, and the extractor output are unchanged. The canonical example: at a
bit where the honest `ch1 = ch2 = 1`, set `u = +1/2, v = -1/2`; then
`Uef + UNegEg` is preserved, `ch1: 1 -> 0` and `ch2: 1 -> 0` stay binary, and
nothing downstream moves. Because `Uef + UNegEg` is the actual "choose"
contribution the arithmetic consumes, a sum-preserving reassignment re-encodes
the *same* computation. These assignments are indistinguishable from honest and
harmless. (The Ch2Comp/MajComp inner-row zero pin H4 further shrinks this set at
the circuit level, since it forbids moving weight onto those columns off the
boundary.)

**(b) Sum-changing (rejected).** Suppose instead the deviation changes a read:
`(u+v)(2)` or `m(2)` is nonzero at some row (shift +3). Then R5(2) and/or R6(2)
shift by `-8 * (u+v)(2)` resp. `-8 * m(2)` (R5 only for `m`), a nonzero field
element. For verification to still pass (H3), this shift must be cancelled by
other columns R5/R6 read: `A, E, W, Sigma0/1, MuPacked`. But those columns are
themselves pinned -- `A, E` by their zero-family boundary/feed-forward residuals
(R7-R10) and the public pins, `W` by the message schedule R4 and message pin
R11, `MuPacked` by H1 and the high-bit pin R17, and any change to `A/E`
propagates into the always-checked virtual `maj`/`ch` and the (X^32-1) sigma
families. Hence a sum-changing deviation reduces to falsifying the original
recurrence soundness of the AIR under H1/H3/H4 -- the *same* obligation the full
catalog relies on. Crucially, `Uef`/`UNegEg`/`Maj` booleanity was never what made
the recurrence sound: R5/R6 `in (X-2)` read the preserved sums, and 3.1 shows
the only thing direct booleanity ever excluded was the benign half-integer coset
of case (a).

**Conclusion (Tier 1).** The dropped direct checks removed only the benign
sum-preserving half-integer malleability. Every deviation that changes what the
recurrences read remains rejected by H3 exactly as before. Tier 1 is sound
relative to the original AIR soundness.

## 4. Tier 2 lemma (XOR-result columns)

Tier 2 additionally drops `Sigma0, Sigma1, SmallSigma0, SmallSigma1`.

### 4.1 Pointwise determination

By F2 and H3, each XOR-result column is *determined* by always-checked data:

- **SmallSigma0/1.** R2, R3 are zero families, so coefficient-by-coefficient
  `SmallSigma0[b] = rotsum0[b] - 2*OvSmallSigma0[b]`, where
  `rotsum0[b] = (W.rot(25) + W.rot(14) + W.shr(3))[b]` is a sum of three `W`
  bits, hence in `{0,1,2,3}` by H1, and `OvSmallSigma0[b]` is binary by H1.
- **Sigma0/1.** R0, R1 `in (X^32 - 1)` reduce `A*rho_sig0` cyclically; since
  `Sigma0` and `OvSigma0` have degree `< 32`, their coefficients equal the
  reduced ones, giving `Sigma0[b] = (A*rho_sig0 mod X^32-1)[b] - 2*OvSigma0[b]`.

In every case the XOR-result bit is a fixed affine function of core-binary
inputs and the overflow, and it is binary **iff** the overflow equals the honest
carry `floor(rotsum/2)`. So the XOR-result's own booleanity was, under H1+H3,
equivalent to "the overflow column is the correct carry."

### 4.2 No benign malleability; deviations are rejected

Unlike the aux columns, an XOR-result column enters its additive family
*alone*, not inside a free sum (F2). Flipping an overflow bit
`Ov[b] -> Ov[b] +- 1` forces (to keep the pinning family in its ideal)
`Sigma[b] -> Sigma[b] -+ 2`, a non-binary value. That change moves the additive
family at word level: e.g. a flipped `OvSmallSigma0[b]` shifts `SmallSigma0(2)`
by `-+2*2^b`, and `SmallSigma0` enters R4 at shift +1, so `R4(2)` moves by
`-+4*2^b != 0`. There is no partner column to preserve the read (the only
partner, `Ov`, lives in the *bitwise* pinning family while `Sigma` is read at
`X = 2` -- the bit-level/word-level asymmetry blocks simultaneous cancellation).
Hence any wrong overflow breaks a downstream `(X - 2)` family unless compensated
by `A/E/W/MuPacked`, which again reduces to the original AIR soundness as in
3.2(b).

**Conclusion (Tier 2).** Given H1+H3, the four XOR-result columns are pointwise
determined; their booleanity is equivalent to overflow correctness, and any
incorrect overflow is caught by the word-level `(X - 2)` additive families with
no benign coset. Tier 2 is sound relative to (i) Tier 1 and (ii) the claim that
the `(X-2)` families detect every word-level perturbation the pins do not fix --
the same AIR-soundness reduction the full catalog already assumes.

## 5. Case-to-test cross-reference

| Lemma case | Class | Test |
|---|---|---|
| 3.1 half-integer admitted by virtual booleanity | analysis | `fold_first::tests::reduced_catalog_rejects_tampering_in_every_dropped_column` (Uef=1/2 with zero inputs -> ch1 non-binary -> rejected) |
| 3.2(b) aux boolean/non-binary tamper rejected | reject | `reduced_catalog_rejects_tampering_in_every_dropped_column` (per column, boolean + non-binary) |
| 4.2 sigma overflow tamper rejected | reject | same test, `Sigma0/1, SmallSigma0/1` cases |
| kept-column check still fires | reject | `reduced_catalog_still_rejects_kept_column_tampering` (E non-binary) |
| honest reduced-catalog completeness | accept | `fold_first_prove_and_verify_on_reduced_catalogs`; `production_sha::tests::fold_first_mixed_hyrax_roundtrips_all_booleanity_catalogs` |
| masked/reference Gram agree per tier | accept | `mask_fast_path_matches_reference_on_reduced_catalogs`, `..._with_non_binary_dropped_columns` |
| catalog mismatch diverges transcript | reject | `fold_first_mixed_hyrax_roundtrips_all_booleanity_catalogs` (prove Tier2 / verify Full) |

Note on 3.1: the pure sum-preserving benign move of 3.2(a) requires a
relation-satisfying witness where two virtuals are simultaneously 1 at the same
bit; the piop-layer tests start from the all-zero witness, where the reachable
half-integer perturbation lands in the *rejected* case (a lone `Uef = 1/2`
leaves `ch1 = -1`, non-binary). The benign coset is therefore argued in 3.2(a)
and exercised for completeness by the honest roundtrips; it is not separately
tamper-tested because, by construction, it produces a byte-identical proof.

## 6. Residual risk and gate

- **Tier 1** rests on the benign-malleability characterization (3.1-3.2), which
  is self-contained given H1-H4, plus the standard AIR-soundness reduction for
  sum-changing deviations. Confidence: high.
- **Tier 2** additionally rests on 4.2's claim that the `(X-2)` word-level
  families catch every incorrect overflow. This is the same obligation the full
  catalog already discharges (the full catalog also relies on R4/R5/R6 to bind
  the recurrence), but Tier 2 makes the XOR-result booleanity *depend* on it
  rather than enforcing it directly. Confidence: high but not machine-checked;
  the `X^32-1` reduction in 4.1 for `Sigma0/1` should be re-derived by a reviewer
  against `remainder_is_zero`.

**Decision (owner-approved, 2026-07-07).** The fold-first (V2) flow ships with
`ShaBooleanityCatalog::Tier2DropXorResults` as its default (352 sources): the
combined-sweep bench and the standalone V2 prove/verify use it unless
`SHA256_COMBINED_SWEEP_BOOLEANITY_TIER` overrides it. The residual risk in this
section is accepted for the V2 flow; the tamper matrix in Section 5 is the
standing empirical backing and must stay green. The `X^32-1` reduction in 4.1
remains flagged for an independent reviewer pass but does not block the V2
default.

The V1 flow keeps `ShaBooleanityCatalog::Full` (640 sources): it preserves V1's
canonical prepared-basis fast path and serves as the untouched A/B baseline.
Every `ShaBooleanityCatalog` variant still carries a `SOUNDNESS:` doc pointer to
this file; changing the V1 default is a separate change gated on the same
sign-off checklist.
