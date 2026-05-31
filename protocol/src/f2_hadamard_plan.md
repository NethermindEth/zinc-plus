# F_2 SHA-256 Hadamard discharge — implementation plan

Status: design accepted, not yet implemented. Goal: discharge SHA-256's
16 column-level Hadamard (coefficient-wise / bitwise-AND) relations
`W = U ⊙ V` on the all-`F_2` prover path.

**Chosen approach (v2): per-coefficient-slice zerocheck, reusing the
booleanity bit-slice machinery.** This supersedes both (a) the rejected
"add one ψ-projected sumcheck term" sketch (proves the *convolution*
product — unsound for AND; see the ledger entry in
`documentation/f2x-sha-todo.md` for the counterexample) and (b) the
bit-axis-expansion variant (correct, but needs a new `(μ+5)`-var
sumcheck arity and a new open contraction — kept as the alternative in
§3).

All line refs are against `f2-clean`.

---

## 1. Why a single projection can't do this (one-line recap)

`ψ_α` is a ring hom, so `ψ_α(U)ψ_α(V) = ψ_α(U·V)` (the **convolution**
product), which only exposes the 63 convolution coefficients — the AND
diagonal `Σ_b U_b V_b` is provably not a function of them. The bit
index must be a live variable. Full argument + collision counterexample
in the ledger.

---

## 2. The design: per-slice zerocheck (booleanity-style)

### 2.1 The check

For each coefficient slice `b ∈ {0..31}`, let `U_b(x) = bit_b(U[x])` be
the b-th bit-slice MLE over the `μ`-var row hypercube (`F_2`-valued).
The relation `W = U ⊙ V` is exactly `W_b(x) = U_b(x)·V_b(x)` for all
`(b, x)`. Batch all 32 slices and all 16 relations into one `μ`-var
**zerocheck**:

```
Σ_x  eq(x, r) · Σ_{k=0..15} Σ_{b=0..31} (γ')^k · σ^b · ( U_{k,b}(x)·V_{k,b}(x) − W_{k,b}(x) )  =  0
```

- `eq(x,r)` reuses the **IC point `r`** (`ic_state.evaluation_point`,
  `f2_native_ic.rs:670`) for the row axis. `γ'` batches the 16
  relations, `σ` batches the 32 slices (both drawn after commit).
- Per-variable degree: `eq` (1) × `U·V` (2) = **degree 3**.
- This is structurally the booleanity zerocheck
  (`Σ_k α^k v_k(v_k−1)·eq`, `piop/src/lookup/booleanity.rs:1-7`) with the
  self-product `v(v−1)` replaced by the cross-product `U·V − W` over two
  columns.

### 2.2 What we reuse from the booleanity path (`piop/src/lookup/booleanity.rs`)

This is the main reason to prefer this design — the bit-slice plumbing
already exists and is tested:

- **`build_shifted_bit_slice_mles::<F,D>`** (`booleanity.rs:43`) — builds
  the `F::Inner`-valued bit-slice MLEs for a column at a row-shift Δ
  (flat `spec*D + bit` layout, zero-padded past the tail — exactly the
  `↓Δ` row-shift our operands need). `ShiftedBitSliceSpec` is the spec.
- **`build_virtual_booleanity_mles::<F,D>`** (`booleanity.rs:185`) —
  builds `F_2`-linear combinations of bit-slices (`Σ_j coeff_j·v_j`),
  which is exactly how our virtual operands (`A+A^↓2`, `1−E`, `x+X·c`)
  are formed. `VirtualBoolSpec` / `VirtualBoolSource` are the specs.
- **`finalize_booleanity_prover` / `_verifier`** (`booleanity.rs:936,
  1028`) — extract the bit-slice evals at `r*` from the folded MLEs,
  absorb them, and re-derive on the verifier.
- **`verify_bit_decomposition_consistency`** (`booleanity.rs:1098`) — the
  recombination check `Σ_b a^b·v_b(r*) == parent_eval` that ties the
  sent bit-slice evals to a single per-column opening (see §2.3). This
  is the key trick that avoids per-slice openings.

The only genuinely new sumcheck logic is the cross-product `comb_fn`
(`U·V − W` instead of `v(v−1)`).

### 2.3 Pinning the bit-slice evals — the one F_2-specific subtlety

The verifier needs `U_{k,b}(r*)`, `V_{k,b}(r*)`, `W_{k,b}(r*)` to check
the zerocheck's final claim, and these must be tied to the committed
columns — otherwise a prover could run the zerocheck on fake bits. The
booleanity mechanism: the prover **sends** all `D` bit-slice evals per
column, and `verify_bit_decomposition_consistency` checks
`Σ_b a^b·v_b(r*) == parent_eval`, where `parent_eval` is the column's
`a`-projected MLE eval at `r*`, discharged by **one** ordinary column
opening. So no per-slice openings — the `D` slices ride one opening.

**Soundness needs `a` fresh after the bit-slice evals are absorbed.**
Then `Σ_b a^b·(v_b^{sent} − v_b^{true}) = 0` is a degree-≤31 polynomial
in `a` that vanishes only if every coefficient is zero (SZ over `a`),
pinning all `D` slice evals. The integer path reuses its early Step-3
projection `a` for this (`prover.rs:520`) and gets away with it **only
because** those bit-slices also feed the whole CPR constraint system,
which over-constrains them. Our Hadamard bit-slices are touched only by
the Hadamard zerocheck, so they need their own fresh binding. Two clean
wirings (both reuse the existing open; pick in Phase A):

- **Wiring F (fresh element, shared sumcheck).** Run the Hadamard
  zerocheck as a **second group in the existing `μ`-var sumcheck call**
  (groups share `num_vars`, `multi_degree.rs:313-318` — the eq·col group
  is degree 2, the Hadamard group degree 3; multi-degree groups in one
  call are exactly this case, cf. `multi_degree_two_groups`,
  `multi_degree.rs:590`). After the sumcheck, absorb the bit-slice evals,
  then sample a fresh `a`, and discharge the Hadamard columns'
  `a`-projected evals at the shared `r*` via a **second open instance**
  (the existing `prove_f2_open` re-run with element `a` instead of `α` —
  same code, different `AlphaPolyBasis(a)`). Recombination ties the
  slices.

- **Wiring R (reorder, reuse α).** Run the Hadamard zerocheck as a
  **separate `μ`-var sumcheck *before* α is sampled** (it only needs the
  bit-slice MLEs + the IC `r`, no projection). Absorb its bit-slice
  evals, *then* sample α — now α is fresh w.r.t. those evals and doubles
  as both the main projection and the recombination element. The
  Hadamard columns' α-evals at the Hadamard point `r*_H` fold into the
  multipoint-eval (which must then carry two points — the "proper"
  multipoint-eval the ledger wants for Issue 1 anyway), or a small
  second open at `r*_H`.

Wiring F maximizes code reuse (shared sumcheck call, open re-run);
Wiring R avoids a second projection element but needs the two-point
multipoint-eval. **CHOSEN: Wiring R.** Rationale: one projection
element (α serves both the main sumcheck and the recombination), no
second `ψ_a` open, and the two-point discharge is the "proper"
multipoint-eval we want for Issue 1 anyway. For Phase A (synthetic,
shift-free) the discharge can be a simple second `prove_f2_open` at the
Hadamard point `r*_H` under α (reusing the open at a different point),
deferring the full two-point multipoint-eval to Phase C/D.

---

## 3. Comparison with bit-axis expansion (the alternative)

Both are correct and do the same total sumcheck work; both must pin the
bit-level data with post-commit randomness. The difference is purely
engineering:

| | Per-slice (chosen) | Bit-axis expansion |
|--|--|--|
| Sumcheck | `μ`-var degree-3; sits as a 2nd group in the existing call | `μ+5`-var degree-3; separate call (new arity) |
| Bit data sent | `D·#cols` slice evals (~`32·24`) | `#cols` bit-MLE evals (~`24`) |
| Pin / discharge | recombination + one extra column open (existing open, reused) | eq-over-bits open at the sumcheck's `ρ*` (**new** per-cell contraction) |
| Code reuse | **High** — booleanity bit-slice infra + recombination + open all exist | Low — new contraction in the fold + Check 4 |
| Proof size | larger | smaller |

The per-slice approach trades proof size for reuse of existing, tested
code — the right call for a first sound implementation. Bit-axis is the
optimization to revisit if proof size dominates.

---

## 4. Pipeline order

Current: `commit → IC(r) → α → γ → sumcheck#1(μ, eq·col) → absorb col
evals@r* → multipoint-eval → open(ψ_α)`.

New (Wiring R): `commit → IC(r) → γ', σ → **Hadamard sumcheck#0 (μ-var
deg-3, bit-slice MLEs, eq over r) → absorb bit-slice evals@r*_H** → α
(now fresh) → γ → sumcheck#1 (μ-var eq·col) → absorb col evals@r* →
multipoint-eval (+ Hadamard cols' α-evals @r*_H) → open(ψ_α) →
recombination check (Σ_b α^b·v_b(r*_H) == parent_eval@r*_H)`. The
Hadamard sumcheck is its own `prove_as_subprotocol` call (it precedes α,
so it cannot share sumcheck#1); both reuse the IC point `r` for their
row-axis eq.

---

## 5. Components

### 5.1 Bit-slice + operand MLEs (`f2_prove.rs`, reuse `booleanity.rs`)
Register `ShiftedBitSliceSpec`s for every (column, Δ) the operands use
and `VirtualBoolSpec`s for the `F_2`-linear operand combinations, then
call `build_shifted_bit_slice_mles` / `build_virtual_booleanity_mles`.
The `X·c` bit-shift is a bit-index reindex (`b ↦ b+1`, drop bit 31, zero
bit 0); add it as a small variant of the shifted-slice builder or fold
it into the `VirtualBoolSpec` source indexing. Carry word `c`: bits
0..30 = `SHR¹(t+x+y)` (XOR of three slices, then bit-shift), bit 31 =
`W_β` (§5.4).

### 5.2 The degree-3 Hadamard group (`f2_prove.rs` → `multi_degree.rs`)
`MultiDegreeSumcheckGroup::new(3, poly, comb_fn)` added to the `vec![…]`
passed to the existing `prove_as_subprotocol`. `poly` = `[eq_r,
U_{0,0}, V_{0,0}, W_{0,0}, …]` (the bit-slice + operand MLEs); `comb_fn`
= `eq · Σ_k Σ_b (γ')^k σ^b (U·V + W)` (`−W = +W` in char 2), via Horner.
Degree 3 is supported (boundary points `0,1,X,X+1` in GF(2^128),
`prover.rs:194-236`). No round-1 fast path (the existing
`F2EqColRound1FastPath` is deg-2 eq·col only); a bit-slice fast path
exploiting `{0,1}`-valued slices is a future optimization.

### 5.3 Discharge (reuse `prove_f2_open` + `verify_bit_decomposition_consistency`)
Per Wiring F: a second `prove_f2_open` over the Hadamard columns with
the fresh element `a`, giving `parent_eval_a` per column; then
`verify_bit_decomposition_consistency(parent_evals, bit_slice_evals, a,
D)`. Merkle openings shared with the main open (same commitment, same
raw leaves).

### 5.4 `W_β` carry column (prerequisite for the 13 adder relations)
Absent on `f2-clean`. Add `cols::W_BETA` (1 packed `BinaryPoly<32>`,
bits 0..12 = the 13 `β_k = Maj(a[31],b[31],D[31])`,
`arithmetization.tex:317-322`), its generator (mirror the `PA_C` setup,
`sha256_f2.rs:990-1029`), the bits-13..31-zero public sweep, and
`SHR^{j}(W_β)` extraction. The 3 AND relations need no `W_β` → land
first.

### 5.5 Missing ShiftSpecs (`sha256_f2.rs:335-367`)
Add `W_E ↓1, ↓2` and `W_A ↓1, ↓2` (only `↓4` exist today) — needed for
C12/C13/C14. (Add as `ShiftedBitSliceSpec`s, not `ShiftSpec`s, since the
Hadamard path consumes bit-slices, not packed down-rows.)

### 5.6 Verifier (`verify_f2_uair_with_groups`, `f2_prove.rs:1183-1315`)
1. Hadamard group **claimed sum == 0** (zerocheck; `claimed_sums()`,
   `multi_degree.rs:135`). New `F2VerifyError::NonZeroHadamardClaim`.
2. Subclaim consistency: `eq(r*,r)·Σ_k Σ_b (γ')^k σ^b (U_{k,b}(r*)·
   V_{k,b}(r*) + W_{k,b}(r*)) == expected_evaluations()[hadamard_group]`,
   the operand evals rebuilt from the sent base bit-slice evals by the
   `F_2`-linear recipe (mirror `f2_prove.rs:1251-1275`).
3. `verify_bit_decomposition_consistency` ties bit-slice evals to the
   column openings (§5.3).

---

## 6. Dependencies & risks
1. **Row-shift discharge (shared with ledger "Issue 1").** Operands use
   `↓Δ` row-shifts; `build_shifted_bit_slice_mles` materializes the
   shifted slices for the *sumcheck*, but the verifier still needs the
   shifted columns' `parent_eval` at `r*`. MLE eval does not commute
   with row shifts. Options as in the ledger's Issue 1: (a) extend
   multipoint-eval / shift-predicate (sound, larger, shared fix); (b)
   honest-prover-first trusted absorb, matching today's posture, then
   close later. Recommend (b) for the first e2e, (a) jointly with
   Issue 1. (This affects discharge of shifted operands, not the
   sumcheck itself.)
2. **`W_β`** blocks the 13 adder relations; AND relations don't need it.
3. **Cost**: the degree-3 slice group's `comb_fn` is ~`32·16` triples;
   round 1 over `2^{μ-1}` slots. Comparable to bit-axis; the second
   `ψ_a` open of ~24 columns is the other new cost. Proof carries
   `D·#cols` slice evals.
4. **Soundness re-derivation** of the fresh-`a` recombination + the
   slice/relation batching, alongside the GF(2^128) field-swap TODO
   (`binary_gf128.rs:31-56`).

---

## 7. Phased implementation
- **Phase A — foundation in isolation (Wiring R).** Synthetic UAIR: one
  relation `W = U ⊙ V`, three primary committed columns, no shifts.
  Build (A0) the piop cross-product zerocheck (degree-3 group + finalize
  + recombination, mirroring `booleanity.rs`) with a unit test; then
  (A1) wire it into the F_2 prove flow *before* α, reusing
  `build_shifted_bit_slice_mles` at Δ=0; (A2) sample α; (A3) discharge
  the Hadamard cols' α-evals at `r*_H` via a second `prove_f2_open` at
  `r*_H`; (A4) recombination. Prove e2e.
- **Phase B — virtuals & shifts.** Add `VirtualBoolSpec` operands
  (`A+A^↓2`, `1−E`), the `X·c` bit-shift, and row-shifts via the §6
  route. Extend the synthetic UAIR to cover each.
- **Phase C — SHA wiring.** Add `W_β` (§5.4) + the missing
  `ShiftedBitSliceSpec`s; register the 16 relations (Appendix A); run
  e2e on `sha256_f2`. AND relations first, then the 13 adders.
- **Phase D — soundness & opt.** Row-shift discharge (joint Issue 1);
  the `{0,1}`-slice round-1 fast path.

---

## 8. Tests & benchmarks
- **piop**: degree-3 cross-product zerocheck — accept on
  `W=U⊙V`, reject on a flipped bit; `claimed_sum==0` gate.
- **protocol**: recombination round-trip (`verify_bit_decomposition_
  consistency` accept/reject); the second `ψ_a` open.
- **test-uair**: Phase-A/B synthetic UAIR e2e; `sha256_f2` e2e accept on
  a valid trace, reject on a flipped AND/carry bit. Assert the IC still
  routes through `prove_linear` (`effective_max_degree == 1`,
  `sha256_f2.rs:1177-1181`) — the Hadamard degree lives only in the new
  group.
- **bench** (`protocol/benches/f2_sha256.rs`): `Hadamard-Slices` and
  `Open-psi_a` groups; prove/verify deltas at nvars=22.

---

## 9. Ledger / doc updates (CLAUDE.md)
On implementation, move the Hadamard entries in
`documentation/f2x-sha-todo.md` to "Shipped work" with SHAs + measured
deltas, and log: the per-slice-vs-bit-axis decision (chosen: per-slice,
for reuse), Wiring F vs R, the `{0,1}`-slice fast path (identified, not
done), and the row-shift coupling to Issue 1.

---

## Appendix A — Hadamard relation inventory

Columns `cols::NAME (index)` from `sha256_f2.rs:202-278`
(`NUM_BIN_PUB=8`). `↓Δ` row-shift, `X·` bit-shift, `1−`/`+` are
`F_2`-linear.

### A.1 The 16 relations (`W = U ⊙ V`)
AND-flavoured (`arithmetization.tex` §3.4–3.5):

| # | U | V | W |
|---|---|---|---|
| C12 | `W_E (9)` | `W_E^↓1` | `W_UEF (17)` |
| C13 | `1 − W_E` | `W_E^↓2` | `W_UNEG_E_G (18)` |
| C14 | `W_A (8) + W_A^↓2` | `W_A^↓1 + W_A^↓2` | `W_MAJ (19) + W_A^↓2` |

Adder (`(x+X·c) ⊙ (y+X·c) = c+X·c`; `c[0..30]=SHR¹(t+x+y)`, `c[31]=β_k`
from `W_β` bit `j`):

| # | t | x | y | j |
|---|---|---|---|---|
| C5a | `W_W_S1 (22)` | `W_W (10)` | `W_SIG0^↓1 (13)` | 0 |
| C5b | `W_W_S2 (23)` | `W_W_S1 (22)` | `W_W^↓9 (10)` | 1 |
| C5c | `W_W^↓16 (10)` | `W_W_S2 (23)` | `W_SIG1^↓14 (14)` | 2 |
| C6a | `W_T1_S1 (24)` | `W_E (9)` | `W_SIGMA1^↓3 (12)` | 3 |
| C6b | `W_T1_S2 (25)` | `W_T1_S1 (24)` | `W_UEF^↓3 (17)` | 4 |
| C6c | `W_T1_S3 (26)` | `W_T1_S2 (25)` | `W_UNEG_E_G^↓3 (18)` | 5 |
| C6d | `W_T1_S4 (27)` | `W_T1_S3 (26)` | `PA_K^↓3 (2)` | 6 |
| C6e | `W_T1^↓3 (20)` | `W_T1_S4 (27)` | `W_W^↓3 (10)` | 7 |
| C7  | `W_T2 (21)` | `W_SIGMA0 (11)` | `W_MAJ (19)` | 8 |
| C8  | `W_A^↓4 (8)` | `W_T1^↓3 (20)` | `W_T2^↓3 (21)` | 9 |
| C9  | `W_E^↓4 (9)` | `W_A (8)` | `W_T1^↓3 (20)` | 10 |
| C10 | `W_A^↓4 (8)` | `W_A (8)` | `PA_A (0)` | 11 |
| C11 | `W_E^↓4 (9)` | `W_E (9)` | `PA_E (1)` | 12 |

(Hadamard operands `U=x+X·c`, `V=y+X·c`, `W=c+X·c`; `t` enters via `c`.)

### A.2 Base columns needing bit-slices
18 of 20 witness cols (all but `W_SHR3_W (15)`, `W_SHR10_W (16)`) +
`PA_A/PA_E/PA_K (0,1,2)` + new `W_β` ≈ **25 columns**.

### A.3 Status (`f2-clean`)
- `W_UEF/W_UNEG_E_G/W_MAJ` filled as correct bitwise ANDs
  (`sha256_f2.rs:963-977`). `PA_C (4)` is the LSB compensator κ — not
  the carry word; `W_β` must be added (§5.4).
- Booleanity is **structural** (`BinaryPoly<32> ⊂ F_2[X]`) — slices are
  `{0,1}` automatically; no booleanity gadget needed (and indeed the
  Hadamard check *reuses* the booleanity bit-slice machinery without its
  `v(v−1)` term).
