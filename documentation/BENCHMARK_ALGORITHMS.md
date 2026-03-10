# Zinc+ SHA-256 8× Folded Benchmark: Algorithmic Specification

> **Benchmarks covered:**
> ```
> cargo bench --bench steps_sha256_8x_folded -p zinc-snark --features "parallel simd asm qx-constraints no-f2x" -- "Prover"
> cargo bench --bench steps_sha256_8x_folded -p zinc-snark --features "parallel simd asm qx-constraints no-f2x" -- "Verifier"
> ```
>
> With `true-ideal` enabled (replaces trivial ideal on C7–C9 with the degree-one ideal $(X-2)$):
> ```
> cargo bench --bench steps_sha256_8x_folded -p zinc-snark --features "parallel simd asm qx-constraints no-f2x true-ideal" -- "Prover"
> cargo bench --bench steps_sha256_8x_folded -p zinc-snark --features "parallel simd asm qx-constraints no-f2x true-ideal" -- "Verifier"
> ```
>
> With `full-fold` enabled (32× folding: `BinaryPoly<32>` → `BinaryPoly<1>` in 5 rounds, instead of 4× folding to `BinaryPoly<8>` in 2 rounds):
> ```
> cargo bench --bench steps_sha256_8x_folded -p zinc-snark --features "parallel simd asm qx-constraints no-f2x true-ideal full-fold" -- "Prover"
> cargo bench --bench steps_sha256_8x_folded -p zinc-snark --features "parallel simd asm qx-constraints no-f2x true-ideal full-fold" -- "Verifier"
> ```
>
> With `boundary` enabled (adds a third Ideal Check and CPR group for boundary constraints):
> ```
> cargo bench --bench steps_sha256_8x_folded -p zinc-snark --features "parallel simd asm qx-constraints no-f2x boundary" -- "Prover"
> cargo bench --bench steps_sha256_8x_folded -p zinc-snark --features "parallel simd asm qx-constraints no-f2x boundary" -- "Verifier"
> ```
>
> **Note on `full-fold`:** The `full-fold` feature flag **changes the folding depth**
> and PCS configuration. Instead of 2 folding rounds (32→16→8), 5 rounds are
> applied (32→16→8→4→2→1), committing to `BinaryPoly<1>` columns. This
> increases `num_vars` by 3 (from 11 to 14), changes the PCS code from
> `PnttConfigF2_16R4B32<2>` (input 2048) to `PnttConfigF2_16R4B32<3>`
> (input 16384), and the ring-eval bridge produces $20 \times 1 = 20$
> field elements instead of $20 \times 8 = 160$. The extra folding
> rounds increase the folding proof size (from 80 to 200 field elements)
> but the PCS operates on single-bit columns, potentially reducing
> proximity proof overhead per opening.
>
> **Note on `no-f2x`:** The `no-f2x` feature flag **changes the UAIR types**
> used by the benchmark. The benchmark file contains `#[cfg(feature = "no-f2x")]`
> gates (lines 60–81) that switch the type aliases:
>
> | | Without `no-f2x` | With `no-f2x` | With `no-f2x` + `true-ideal` |
> |---|---|---|---|
> | **BenchBpUair** | `Sha256Uair` (16 constraints: C1–C4 cyclotomic + C5–C16 zero) | `Sha256UairBpNoF2x` (12 constraints: C5–C16, all zero) | same as `no-f2x` |
> | **BenchQxUair** | `Sha256UairQx` (3 constraints, all trivial) | `Sha256UairQxNoF2x` (7 constraints: C1–C4 cyclotomic + C7–C9 trivial) | `Sha256UairQxNoF2x` (7 constraints: C1–C4 cyclotomic + C7–C9 DegreeOne $(X-2)$) |
> | **BenchBdryUair** | — | With `boundary`: `Sha256UairBoundaryNoF2x` (13 constraints, degree 2, all zero ideal) | same |
> | **Trace columns** | 30 (27 bitpoly + 3 integer) | 38 (35 bitpoly + 3 integer); with `boundary`: 45 (42 + 3) | 44 (41 bitpoly + 3 integer); with `boundary`: 51 (48 + 3) |
> | **Witness generator** | `Sha256Uair::generate_witness()` | `generate_no_f2x_witness()` | `generate_no_f2x_witness()` (+ correction columns & corrected carries) |
>
> The core mathematical change: the 4 F₂[X] rotation constraints
> (C1–C4) are moved from the Bp UAIR to the Qx UAIR. In Qx, they are
> reformulated over Z[X] with explicit integer-carry quotient columns
> μ = μ\_lo + 2·μ\_hi (8 new bitpoly columns at indices 27–34).
> Since this document describes the benchmark **with** `no-f2x` enabled,
> all parameters, constraint counts, and proof sizes reflect the no-f2x variant.
>
> **Note on `true-ideal`:** The `true-ideal` feature flag replaces the
> **trivial ideal** used for the carry constraints C7–C9 with the
> **degree-one ideal** $(X-2)$: a polynomial $p(X)$ belongs to $(X-2)$
> iff $p(2) = 0$. This makes the IC membership check non-vacuous for
> carry constraints, strengthening soundness. Because a single
> `BinaryPoly<32>` can only represent non-negative values at $X=2$
> (range $[0, 2^{32}-1]$), each carry constraint needs a **dual
> correction** pair (add/sub) to handle signed residuals:
>
> $$c_{\text{inner}} + \text{corr\_add} - \text{corr\_sub} \in (X-2)$$
>
> where $\text{corr\_add} = \max(-c_{\text{eval}}, 0)$ and
> $\text{corr\_sub} = \max(c_{\text{eval}}, 0)$ ensure exactly one of
> the pair is nonzero at each row. This adds **6 new public bitpoly
> columns** (indices 35–40). The witness generator also **recomputes
> the integer carry columns** $\mu_a, \mu_e, \mu_W$ at every row to
> guarantee $|c_{\text{eval}}| < 2^{32}$ everywhere (including
> inactive boundary rows where the base trace sets $\mu = 0$).
>
> **Note on `boundary`:** The `boundary` feature flag adds a **third
> UAIR** (`Sha256UairBoundaryNoF2x`) that enforces boundary
> constraints — initial state, final output, message input, and zero
> padding — as algebraic constraints rather than relying on implicit
> witness-generation guarantees. It adds **7 new public bitpoly
> columns** (4 selectors + 3 output/expected values) appended after
> the existing bitpoly columns, increasing the total from 35 to 42
> (or 41 to 48 with `true-ideal`). The boundary UAIR has **13
> constraints** (all `assert_zero`, max degree 2) and uses
> `ImpossibleIdeal`. The E2E pipeline gains: (1) a sparse boundary
> Ideal Check (step 2c), (2) a boundary CPR group in the
> multi-degree sumcheck, and (3) the shift sumcheck is replaced by
> the CPR's internal shift handling. The boundary IC uses a **sparse
> evaluation** optimisation that evaluates constraints only at the
> ~22 rows where selectors are active. The boundary CPR reuses the
> already-projected base field trace, projecting only the 7 new
> columns from `BinaryPoly<D>`. A **sentinel shift** (`source_col: 0,
> shift_amount: 1`) avoids triggering legacy all-column shifts.
>
> This document provides a precise mathematical description of the algorithms
> executed in the E2E prover and E2E verifier benchmarks. It is written for
> cryptographers who need to understand the exact algebraic operations, the
> rings and fields in which they take place, and how the sub-protocols compose.

---

## Table of Contents

1. [Overview and Notation](#1-overview-and-notation)
2. [Algebraic Domains](#2-algebraic-domains)
3. [The SHA-256 UAIR Trace](#3-the-sha-256-uair-trace)
4. [E2E Prover Pipeline](#4-e2e-prover-pipeline)
   - 4.0 [Split Columns](#40-step-0-split-columns)
   - 4.1 [PCS Commit](#41-step-1-pcs-commit)
   - 4.2 [Transcript Init & Field Setup](#42-step-2-transcript-init--field-setup)
   - 4.3 [Ideal Check (BinaryPoly UAIR)](#43-step-2a-ideal-check-binarypoly-uair)
   - 4.4 [Ideal Check (Q[X] UAIR)](#44-step-2b-ideal-check-qx-uair)
   - 4.4c [Ideal Check (Boundary UAIR)](#44c-step-2c-ideal-check-boundary-uair)
   - 4.5 [Field Projection & Column Evaluation](#45-step-3a-field-projection--column-evaluation)
   - 4.6 [Hybrid GKR Batched Lookup](#46-step-3b-hybrid-gkr-batched-lookup)
   - 4.7 [Shift Sumcheck](#47-step-3c-shift-sumcheck)
      - 4.7.1 [Soundness Argument for down_evals](#471-soundness-argument-for-down_evals)
   - 4.8 [Folding Protocol (2-Round / 5-Round)](#48-step-4-folding-protocol-2-round--5-round)
   - 4.8b [Ring-Eval Lifting (PIOP↔PCS Bridge)](#48b-step-4b-ring-eval-lifting-pioppcs-bridge)
   - 4.9 [PCS Prove](#49-step-5-pcs-prove)
   - 4.10 [Fiat-Shamir Transcript Flow](#410-fiat-shamir-transcript-flow)
5. [E2E Verifier Pipeline](#5-e2e-verifier-pipeline)
   - 5.1 [Transcript Replay & IC Verify](#51-transcript-replay--ic-verify)
   - 5.2 [Column Evaluation Absorb & Lookup Verify](#52-column-evaluation-absorb--lookup-verify)
   - 5.3 [Shift Sumcheck Verify](#53-shift-sumcheck-verify)
   - 5.4 [Folding Verify (2-Round / 5-Round)](#54-folding-verify-2-round--5-round)
   - 5.4b [Ring-Eval Bridge Verify](#54b-ring-eval-bridge-verify)
   - 5.5 [PCS Verify](#55-pcs-verify)
6. [Appendix A: UAIR Column Layout](#appendix-a-uair-column-layout)
7. [Appendix B: Constraint Catalogue](#appendix-b-constraint-catalogue)
8. [Appendix C: Concrete Parameters](#appendix-c-concrete-parameters)
9. [Appendix D: Security Analysis](#appendix-d-security-analysis)

---

## 1. Overview and Notation

The benchmark measures a **SNARK for 8× SHA-256 compressions** (512 rounds packed into a $2^9$-row trace). The proving system is:

- **Arithmetisation**: Uniform Algebraic Intermediate Representation (UAIR). The trace cells are elements of $\mathbb{F}_2[X]/(X^{32}+1)$ (`BinaryPoly<32>`), but constraints are expressed over $\mathbb{Z}[X]$ (Bp UAIR, with carrier columns for integer arithmetic) and verified via ideal membership in $(X^{32}-1) \subset \mathbb{Z}[X]$ (Qx UAIR) or the zero ideal.
- **PIOP**: Ideal Check + standalone evaluation + Hybrid GKR LogUp lookup + shift sumcheck.
- **PCS**: Zip+ with 4×-folded IPRS codes over BinaryPoly⟨8⟩ (or 32×-folded over BinaryPoly⟨1⟩ with `full-fold`).

**Notation conventions:**

| Symbol | Meaning |
|--------|---------|
| $\mathcal{R}_D$ | The quotient ring $\mathbb{F}_2[X]/(X^D+1)$, elements represented as `BinaryPoly<D>`. Used for trace cell storage; constraints are formulated over $\mathbb{Z}[X]$ (see §2.4) and verified modulo the ideal $(X^{32}-1)$ after lifting to $\mathbb{F}_p[X]$ |
| $\mathbb{F}_p$ | A random 192-bit prime field (3 × 64-bit limbs), derived via Fiat-Shamir (see §2.2) |
| $n$ | Number of MLE variables; $N = 2^n$ rows. Here $n = 9$, $N = 512$ |
| $\alpha \in \mathbb{F}_p$ | The "projecting element" — a random field challenge used to project $\mathbb{Z}[X]$ coefficients to $\mathbb{F}_p$ via evaluation at $\alpha$: each `BinaryPoly<32>` is first coefficient-lifted to $\mathbb{Z}[X]$ (or $\mathbb{F}_p[X]$), then evaluated at $\alpha$ |
| $\mathbf{r} \in \mathbb{F}_p^n$ | The Ideal Check evaluation point |
| $\widetilde{f}$ | The multilinear extension (MLE) of a function $f : \{0,1\}^n \to \mathbb{F}_p$ |
| $\mathrm{eq}(\mathbf{r}, \mathbf{b})$ | $\prod_{i=1}^{n}(r_i b_i + (1-r_i)(1-b_i))$, the MLE of the Kronecker delta |
| $[m]$ | The set $\{0, 1, \ldots, m-1\}$ |

---

## 2. Algebraic Domains

### 2.1. Binary Polynomial Rings

Each UAIR trace cell is an element of $\mathcal{R}_D = \mathbb{F}_2[X]/(X^D+1)$ for some $D \in \{8, 16, 32\}$. Since $\mathrm{char}(\mathbb{F}_2) = 2$, we have $X^D - 1 = X^D + 1$, making this the **cyclotomic ring** of order $D$ over $\mathbb{F}_2$.

A `BinaryPoly<D>` stores $D$ boolean coefficients $(c_0, c_1, \ldots, c_{D-1})$ representing:

$$a(X) = \sum_{i=0}^{D-1} c_i X^i, \quad c_i \in \{0,1\}$$

With the SIMD feature enabled, this is packed into a single `u64` bitmask (for $D \leq 64$). Addition and subtraction are both XOR (characteristic 2).

**Key property for SHA-256:** SHA-256 uses 32-bit word rotations. In $\mathcal{R}_{32}$, right-rotation by $r$ positions corresponds to multiplication by $X^{32-r}$:

$$\mathrm{ROTR}^r(a) \;\longleftrightarrow\; a(X) \cdot X^{32-r} \pmod{X^{32}+1}$$

XOR of multiple rotations becomes addition of the corresponding polynomial products.

### 2.2. The PIOP Prime Field $\mathbb{F}_p$

The PIOP operates over a random prime field $\mathbb{F}_p$ where $p$ is a **192-bit prime** derived from the Fiat-Shamir transcript:

1. The Blake3-based transcript squeezes 24 bytes, reinterprets them as a `Uint<3>` (192-bit) candidate.
2. Makes the candidate odd.
3. Runs trial division against the first 15 odd primes, then a Miller-Rabin base-2 test.
4. Loops until a probable prime is found.

The resulting prime is stored in Montgomery form using `MontyField<3>` (3 × 64-bit limbs = 192-bit representation). Field elements are represented as $a \cdot R \bmod p$ where $R = 2^{192}$.

### 2.3. Projection: $\mathbb{Z}[X] \to \mathbb{F}_p$

Given a random challenge $\alpha \in \mathbb{F}_p$ (the "projecting element"), the **projection homomorphism** is evaluation at $\alpha$:

$$\pi_\alpha : \mathcal{R}_{32} \to \mathbb{F}_p, \quad a(X) \mapsto a(\alpha) = \sum_{i=0}^{31} c_i \cdot \alpha^i$$

This is implemented via a **chunked table lookup**: the 32 coefficients are split into 4 chunks of 8 bits, with a precomputed table of $2^8 = 256$ entries per chunk. The projection reduces to 4 table lookups and 3 field additions per element.

**Algebraic status.** Strictly speaking, $\pi_\alpha$ is *not* a ring homomorphism from $\mathcal{R}_{32}$ to $\mathbb{F}_p$: the source ring has characteristic 2 ($1 \oplus 1 = 0$), but $\pi_\alpha(1) + \pi_\alpha(1) = 2 \neq 0$ in $\mathbb{F}_p$. What the system actually does is:

1. **Coefficient-lift:** embed $\{0,1\}^{32} \hookrightarrow \mathbb{F}_p^{32}$ by the canonical injection $0 \mapsto 0_p, 1 \mapsto 1_p$.
2. **Evaluate:** interpret the lifted vector as a polynomial in $\mathbb{F}_p[X]$ and evaluate at $\alpha$.

Step 2 is the evaluation-at-$\alpha$ ring homomorphism $\mathbb{F}_p[X] \twoheadrightarrow \mathbb{F}_p$, and step 1 is an additive (but *not* multiplicative) map. The composition $\pi_\alpha$ is **$\mathbb{F}_p$-linear on the lifted coefficient vectors**: $\pi_\alpha(a \oplus b) \neq \pi_\alpha(a) + \pi_\alpha(b)$ in general (XOR ≠ integer addition), but for any $\mathbb{F}_p$-linear combination of *already-lifted* columns, linearity does hold. The protocol relies on two key properties:

- **Constraint faithfulness (with quotient correction):** The rotation constraints C1–C4, when computed over $\mathbb{F}_2[X]$, involve polynomial multiplication by constant rotation polynomials $\rho$ (e.g., $\rho_0 = X^{30}+X^{19}+X^{10}$). After coefficient-lifting the binary columns to $\mathbb{F}_p[X]$, the convolution sums differ from $\mathbb{F}_2$ because carries arise: $1+1 = 2 \neq 0$ in $\mathbb{F}_p$. Consequently, the residue of $C_k^{\mathrm{lift}}(\mathrm{cols}) \bmod (X^{32}-1)$ in $\mathbb{F}_p[X]$ is **not** zero but has even coefficients (the constraint holds mod 2, so every coefficient of the residue is divisible by 2). The 8 quotient columns $\mu$ explicitly provide the correction: each constraint's residue is absorbed by $\mu = \mu_\mathrm{lo} + 2\mu_\mathrm{hi}$ (with $\mu_\mathrm{lo}, \mu_\mathrm{hi} \in \{0,1\}^{32}$), so that $C_k^{\mathrm{lift}} - 2\mu_\mathrm{lo} - 4\mu_\mathrm{hi} \equiv 0 \pmod{X^{32}-1}$ in $\mathbb{Z}[X]$ (and hence in $\mathbb{F}_p[X]$). This converts what would be an $\mathbb{F}_2[X]$ ideal-membership check into an exact-zero check over $\mathbb{F}_p[X]$, which is the entire purpose of the `no-f2x` feature and the 8 additional committed columns.
- **MLE–projection commutativity:** Because $\pi_\alpha$ is a fixed $\mathbb{F}_p$-linear map on coefficient vectors, it commutes with MLE evaluation: $\pi_\alpha(\widetilde{\mathrm{col}}(\mathbf{r})) = \widetilde{\pi_\alpha(\mathrm{col})}(\mathbf{r})$.

### 2.4. Integer Polynomial Ring $\mathbb{Z}[X]$

For carry-propagation constraints, the system uses `DensePolynomial<i64, 64>` — polynomials of degree < 64 with `i64` coefficients. At $X = 2$, these recover 32-bit integer values: $\hat{a}(2) = \sum_i c_i \cdot 2^i$.

### 2.5. Dynamic Polynomials $\mathbb{F}_p[X]$

`DynamicPolynomialF<F>` is a univariate polynomial over $\mathbb{F}_p$ with dynamically-sized coefficient vector. It is used as the intermediate representation during the Ideal Check — constraints are built symbolically over $\mathbb{F}_p[X]$ before being evaluated at the projecting element $\alpha$.

---

## 3. The SHA-256 UAIR Trace

### 3.1. Trace Dimensions

The trace has $N = 2^9 = 512$ rows, encoding **8 chained SHA-256 compressions** (each compression = 64 rounds). With `no-f2x` enabled, the trace has **38 columns** (35 `BinaryPoly<32>` + 3 integer columns). Columns 0–26 and integer columns 0–2 are identical to the standard (non-no-f2x) UAIR; columns 27–34 are the 8 new μ quotient decomposition columns added by no-f2x:

- **20 private PCS-committed columns** (the original 12 at bitpoly indices 0, 1, 3–9 and integer indices 0–2, plus the 8 new μ quotient columns at bitpoly indices 27–34).
- **12 public columns** known to the verifier (indices 2, 10–13, 16–20, 25–26).
- **6 shift-source-only columns** (indices 14, 15, 21–24: `D_hat`, `H_hat`, `a_tm1`, `a_tm2`, `e_tm1`, `e_tm2`) — excluded from PCS as shift sources but **not** public. Their evaluation claims are resolved by the shift sumcheck (§4.7), which reduces them to PCS openings of the original source columns at a fresh random point.
- **10 shift-source columns** total (4 overlap with public columns, 6 are the shift-source-only columns above).

Thus: 20 committed + 12 public + 6 shift-source-only = 38 total.

> **Without `no-f2x`:** The trace has 30 columns (27 bitpoly + 3 integer), with 12 committed columns.

> **With `true-ideal`:** 6 additional **public** bitpoly columns (correction pairs at indices 35–40) are appended, bringing the total to **44 columns** (41 bitpoly + 3 integer): 20 committed + 18 public + 6 shift-source-only = 44.

> **With `boundary`:** 7 additional **public** bitpoly columns (4 selectors + 3 output/expected values at indices 33–39, or 39–45 with `true-ideal`) are appended, bringing the total to **45 columns** (42 bitpoly + 3 integer): 20 committed + 19 public + 6 shift-source-only = 45. With `boundary` + `true-ideal`, the total is **51 columns** (48 bitpoly + 3 integer): 20 committed + 25 public + 6 shift-source-only = 51. These columns are consumed by a **separate UAIR** (`Sha256UairBoundaryNoF2x`) that sees all 40 (or 46 with `true-ideal`) bitpoly columns + 3 integer columns in its own column namespace.

### 3.2. Column Layout

The columns are organised into six groups (full table in Appendix A):

1. **Core state columns** (indices 0–9): The SHA-256 working variables $a_t, e_t$, message schedule $W_t$, rotation outputs $\Sigma_0, \Sigma_1, \sigma_0, \sigma_1$, and Boolean function outputs $\mathrm{Maj}, \mathrm{Ch}_1, \mathrm{Ch}_2$. All in $\mathcal{R}_{32}$.

2. **F₂[X] decomposition columns** (indices 10–13): Shift quotients $S_0, S_1$ and remainders $R_0, R_1$ for the SHR operations in $\sigma_0, \sigma_1$. These encode the division $W = R + X^k \cdot S$ needed because SHR is not a rotation.

3. **Lookback columns** (indices 14–24): Virtual shifted copies of source columns, providing cross-row references like $d_t = a_{t-3}$, $h_t = e_{t-3}$, $f_t = e_{t-1}$, etc. Indices 14–15 and 21–24 are shift-source-only (non-public); indices 16–20 are public.

4. **Selector columns** (indices 25–26): Row-selector patterns for boundary handling.

5. **μ quotient columns** (indices 27–34, **no-f2x only**): $\mu_{k,\text{lo}}, \mu_{k,\text{hi}} \in \{0,1\}^{<32}[X]$ for $k = 1,\ldots,4$. These are the binary decomposition of the integer carries that arise when the F₂[X] rotation constraints C1–C4 are lifted to Z[X]. Each $\mu_k = \mu_{k,\text{lo}} + 2 \cdot \mu_{k,\text{hi}}$ has coefficients in $\{0,1,2,3\}$, matching the maximum carry from multiplying two binary polynomials.

6. **Correction columns** (indices 35–40, **true-ideal only**): 3 additive + 3 subtractive correction columns for the carry constraints C7–C9. Each constraint residual $c_{\text{eval}}$ can be positive or negative, but a single `BinaryPoly<32>` only represents non-negative values at $X=2$. The dual pair $(\text{corr\_add}, \text{corr\_sub})$ absorbs the signed residual: $\text{corr\_add}_t = \max(-c_t, 0)$, $\text{corr\_sub}_t = \max(c_t, 0)$, so exactly one is nonzero per row. These are **public** columns.

7. **Integer carry columns** (int indices 0–2): $\mu_a, \mu_e, \mu_W \in \mathbb{Z}$, absorbing overflow from modular-$2^{32}$ additions. As polynomials: $\mu \cdot X^{32}$ cancels the carry at $X = 2$. Although conceptually integer-valued, these columns are **represented as `BinaryPoly<32>`** in the witness (not as a separate integer type): the small integer value is encoded in binary via its bit-decomposition, with only the lowest 2–3 coefficients nonzero (e.g., $\mu_a = 5$ is stored as the bit-vector `[1,0,1,0,\ldots,0]`). This means they are homogeneous with the rest of the trace and undergo the same double-split (§4.0) and PCS commitment path as every other column; the integer content lands in the lowest `BinaryPoly<8>` quarter, with the upper quarters being zero.

8. **Boundary columns** (indices `NO_F2X_NUM_BITPOLY_COLS`+0 through +6, **`boundary` only**): 7 public columns consumed by the boundary UAIR (`Sha256UairBoundaryNoF2x`). Without `true-ideal`, these occupy indices 33–39; with `true-ideal`, indices 39–45. The boundary UAIR sees all base bitpoly columns (0–32 or 0–38) plus these 7 in a contiguous namespace of 40 (or 46) bitpoly columns + 3 integer columns.

   | Offset | Name | Description |
   |--------|------|-------------|
   | +0 | `sel_init` | Binary selector: 1 only at row 0 |
   | +1 | `sel_final_any` | Binary selector: 1 at last 4 active rows ($N-5, N-6, N-7, N-8$) |
   | +2 | `sel_msg` | Binary selector: 1 at rows 0–15 (message schedule input) |
   | +3 | `sel_zero` | Binary selector: 1 at rows $N-3, N-2, N-1$ (zero-padding) |
   | +4 | `out_a` | Expected output: trace value of $\hat{a}$ at final rows |
   | +5 | `out_e` | Expected output: trace value of $\hat{e}$ at final rows |
   | +6 | `msg_expected` | Expected input: trace value of $\hat{W}$ at message rows 0–15 |

> **With `true-ideal`:** The witness generator **recomputes** the integer carry columns at every row using $\mu = \lfloor \max(\text{sum} - \text{next}, 0) / 2^{32} \rfloor$, ensuring $|c_{\text{eval}}| < 2^{32}$ everywhere — including inactive boundary rows where the base trace would set $\mu = 0$ (causing overflow).

### 3.3. Constraint System

The UAIR enforces two classes of constraints via two separate UAIR implementations:

#### 3.3.1. `Sha256UairBpNoF2x` (BinaryPoly UAIR, no-f2x): 12 constraints, all degree 1

With `no-f2x` enabled, the rotation constraints C1–C4 are removed from the Bp UAIR (they move to the Qx UAIR). Only C5–C16 remain, all using the `assert_zero` (zero ideal) check:

| Constraints | Type | Ideal | Description |
|-------------|------|-------|-------------|
| **C5–C6** | Shift decomposition | Zero | $\hat{W}_{t-15} = R_0 + X^3 \cdot S_0$ (exact equality) |
| **C7–C16** | Cross-row linking | Zero | E.g., $d[t+3] = a[t]$ (enforced via shifted MLEs) |

Since all 12 Bp constraints use the zero ideal (no cyclotomic constraints), the IC proof for the Bp UAIR is **trivially empty** — every constraint evaluates to zero, so there is nothing to prove.

> **Without `no-f2x`:** `Sha256Uair` has 16 constraints (C1–C4 cyclotomic + C5–C16 zero). The IC proof contains 4 non-trivial cyclotomic check values.

#### 3.3.2. `Sha256UairQxNoF2x` (Q[X] UAIR, no-f2x): 7 constraints

With `no-f2x`, the Qx UAIR absorbs the rotation constraints C1–C4, reformulated over $\mathbb{Z}[X]$ with explicit μ quotient corrections:

> **Note:** Constraint numbers are scoped per-UAIR. The labels C1–C4 and C7–C9 below are specific to `Sha256UairQxNoF2x` and are processed by the Q[X] Ideal Check (§4.4).

**Rotation constraints with μ correction (cyclotomic ideal $(X^{32}-1)$ over $\mathbb{Z}[X]$):**

> **Note on the ideal:** In $\mathbb{F}_2[X]$, $(X^{32}+1) = (X^{32}-1)$ since $\mathrm{char} = 2$. In $\mathbb{Z}[X]$ and $\mathbb{F}_p[X]$ the system uses $(X^{32}-1)$, i.e., $X^{32} \equiv 1$ (cyclic rotation). The source code confirms additive folding: `reduced[i % 32] += c`.

The key identity: if $Q(\cdot)$ has coefficients in $\{0,1,2,3\}$ and we need $Q \equiv 0 \pmod{2}$ in $(X^{32}-1)$, then $Q - 2\mu \in (X^{32}-1)$ over $\mathbb{Z}[X]$, where $\mu = \mu_{\text{lo}} + 2 \cdot \mu_{\text{hi}}$ absorbs the integer carries.

| Constraint | Ideal | Expression |
|------------|-------|------------|
| **C1** ($\Sigma_0$) | Cyclotomic | $\hat{a} \cdot \rho_0 - \hat{\Sigma}_0 - 2 \cdot \mu_{1,\text{lo}} - 4 \cdot \mu_{1,\text{hi}} \in (X^{32}-1)$ |
| **C2** ($\Sigma_1$) | Cyclotomic | $\hat{e} \cdot \rho_1 - \hat{\Sigma}_1 - 2 \cdot \mu_{2,\text{lo}} - 4 \cdot \mu_{2,\text{hi}} \in (X^{32}-1)$ |
| **C3** ($\sigma_0$) | Cyclotomic | $\hat{W}_{t-15} \cdot \rho_{\sigma_0} + S_0 - \hat{\sigma}_{0,w} - 2 \cdot \mu_{3,\text{lo}} - 4 \cdot \mu_{3,\text{hi}} \in (X^{32}-1)$ |
| **C4** ($\sigma_1$) | Cyclotomic | $\hat{W}_{t-2} \cdot \rho_{\sigma_1} + S_1 - \hat{\sigma}_{1,w} - 2 \cdot \mu_{4,\text{lo}} - 4 \cdot \mu_{4,\text{hi}} \in (X^{32}-1)$ |

**Carry propagation constraints (trivial ideal / DegreeOne ideal):**

| Constraint | Description |
|------------|-------------|
| **C7** | $a$-update: $\hat{a}_{t+1} = h + \Sigma_1 + \mathrm{Ch}_1 + \mathrm{Ch}_2 + K + W + \Sigma_0 + \mathrm{Maj} - \mu_a \cdot X^{32}$ |
| **C8** | $e$-update: $\hat{e}_{t+1} = d + h + \Sigma_1 + \mathrm{Ch}_1 + \mathrm{Ch}_2 + K + W - \mu_e \cdot X^{32}$ |
| **C9** | $W$-schedule: $\hat{W}_t = W_{t-16} + \sigma_0 + W_{t-7} + \sigma_1 - \mu_W \cdot X^{32}$ |

**Without `true-ideal` (default):** The **trivial ideal** (every polynomial is a member) is used for C7–C9, which means constraints hold unconditionally on boundary rows. This design choice drops `max_degree` from 2 to 1 (removing selector gating), enabling the faster MLE-first Ideal Check path.

**With `true-ideal`:** The **degree-one ideal** $(X-2)$ replaces the trivial ideal for C7–C9. The constraint expression is augmented with dual correction columns:

$$c_{\text{inner}} + \text{corr\_add} - \text{corr\_sub} \in (X-2)$$

where $c_{\text{inner}}$ is the original constraint residual (e.g., $\hat{a}_{t+1} - h - \cdots + \mu_a \cdot X^{32}$). The ideal check verifies that evaluating the corrected expression at $X = 2$ yields zero. This is sound because $c_{\text{inner}}(2)$ equals the SHA-256 round equation residual (should be zero), and the correction pair absorbs any signed violations: $\text{corr\_add} = \max(-c_{\text{eval}}, 0)$, $\text{corr\_sub} = \max(c_{\text{eval}}, 0)$. The `max_degree` remains 1.

The **cyclotomic ideal** for C1–C4 uses $(X^{32}-1)$ over $\mathbb{Z}[X]$ (or $\mathbb{F}_p[X]$ after projection): reduce coefficients modulo 32 by folding ($X^{32} \equiv 1$), then check all vanish.

> **Without `no-f2x`:** `Sha256UairQx` has only 3 constraints (C7–C9, all trivial). The IC proof is smaller because there are no cyclotomic check values.

#### 3.3.3. `Sha256UairBoundaryNoF2x` (Boundary UAIR, **`boundary` only**): 13 constraints, all degree 2

When the `boundary` feature is enabled, a third UAIR enforces boundary constraints algebraically. All 13 constraints use `assert_zero` with `ImpossibleIdeal` (zero-only ideal). The maximum degree is **2** (each constraint is a product of a selector column and a linear expression in trace columns).

The boundary UAIR operates on the extended trace: `BDRY_NUM_BITPOLY_COLS` = `NO_F2X_NUM_BITPOLY_COLS` + 7 bitpoly columns + 3 integer columns = **43 total** (without `true-ideal`) or **49 total** (with `true-ideal`).

| # | Name | Expression | Active rows |
|---|------|------------|-------------|
| **B1–B8** | Initial state | $\mathrm{sel\_init} \cdot (\mathrm{col}_i - H[i]) = 0$ for $i \in \{a, e, d, h, a_{t-1}, a_{t-2}, e_{t-1}, e_{t-2}\}$ | Row 0 only |
| **B9–B10** | Final output | $\mathrm{sel\_final\_any} \cdot (\hat{a} - \mathrm{out\_a}) = 0$ and $\mathrm{sel\_final\_any} \cdot (\hat{e} - \mathrm{out\_e}) = 0$ | Last 4 active rows |
| **B11** | Message input | $\mathrm{sel\_msg} \cdot (\hat{W} - \mathrm{msg\_expected}) = 0$ | Rows 0–15 |
| **B12–B13** | Zero padding | $\mathrm{sel\_zero} \cdot \hat{a} = 0$ and $\mathrm{sel\_zero} \cdot \hat{e} = 0$ | Rows $N-3, N-2, N-1$ |

**B1–B8 (initial state):** Verify that the working variables at row 0 match the SHA-256 initial hash values $H[0], \ldots, H[7]$. The constants $H[i]$ are the standard SHA-256 initial hash values encoded as `BinaryPoly<32>` and embedded via `smul_c`. The columns checked are: `a_hat` ($H_0$), `e_hat` ($H_4$), `D_hat` ($H_3$), `H_hat` ($H_7$), `a_tm1` ($H_1$), `a_tm2` ($H_2$), `e_tm1` ($H_5$), `e_tm2` ($H_6$).

**B9–B10 (final output):** Verify that at each of the last 4 active rows (the final state of the last compression), the working variables $\hat{a}$ and $\hat{e}$ match the expected output values stored in `out_a` and `out_e`.

**B11 (message input):** Verify that the message schedule $\hat{W}$ at the first 16 rows of each compression matches the expected input message stored in `msg_expected`.

**B12–B13 (zero padding):** Verify that zero-padding rows at the end of the trace have $\hat{a} = 0$ and $\hat{e} = 0$, ensuring the boundary is clean.

**Shift specification:** The boundary UAIR declares a single **sentinel shift** `ShiftSpec { source_col: 0, shift_amount: 1 }`. No boundary constraint references a shifted column. The sentinel shift exists solely to prevent the legacy empty-shifts codepath (which would erroneously apply shift-by-1 to **all** 40+ columns in the IC/CPR). Because the sentinel is the source column `a_hat` shifted by 1, and `a_hat` is already a CPR column, the sentinel's `down_eval` is handled naturally by the CPR with negligible overhead.

**Max degree = 2:** Because all constraints are selector × linear expression, the IC can use the MLE-first path (skipped in practice — the boundary IC uses a **sparse inline evaluation** instead; see §4.4c).

### 3.4. Lookup Requirements

**10 base lookup columns** (indices 0–9): Each must contain valid `BinaryPoly<32>` values (all coefficients in $\{0,1\}$). This is enforced by a decomposed LogUp argument against a `BitPoly{width:32, chunk_width:8}` table.

**3 affine virtual lookup columns** enforce carry-free arithmetic identities:

| Virtual Column | Expression | SHA-256 Identity |
|----------------|-----------|------------------|
| $\mathrm{Ch}_1$ | $\hat{e} + \hat{e}_{t-1} - 2 \cdot \widehat{\mathrm{ch\_ef}}$ | $e \wedge f$ decomposition (carry-free) |
| $\mathrm{Ch}_2$ | $-\hat{e} + \hat{e}_{t-2} - 2 \cdot \widehat{\mathrm{ch\_neg\_eg}} + \mathbf{1}_{32}$ | $(\neg e) \wedge g$ decomposition |
| $\mathrm{Maj}$ | $\hat{a} + \hat{a}_{t-1} + \hat{a}_{t-2} - 2 \cdot \widehat{\mathrm{Maj}}$ | Majority decomposition (carry-free) |

where $\mathbf{1}_{32} = \sum_{i=0}^{31} X^i = 0\text{xFFFFFFFF}$. These are not stored in the trace — they are computed on-the-fly during the lookup protocol.

**Total lookup columns: 13** (10 base + 3 affine).

### 3.5. Shift Specifications

The BinaryPoly UAIR declares **10 shift specs**, and the Q[X] UAIR declares **2 shift specs** (columns 0 and 1, each shifted by 1). The Q[X] shifts are consumed internally by the Qx IC's `prove_from_binary_poly_at_point` function and do **not** enter the pipeline's shift sumcheck. In total, there are **10 shift claims** resolved by the Bp shift sumcheck.

> **With `boundary`:** The boundary UAIR declares **1 sentinel shift** (`source_col: 0, shift_amount: 1`). This shift does not correspond to any boundary constraint — it exists solely to prevent the legacy empty-shifts codepath in the IC/CPR. The sentinel's `down_eval` is consumed by the boundary CPR group's internal shift handling, not by the standalone shift sumcheck. When `boundary` is enabled, the standalone shift sumcheck is **skipped entirely** and all shift claims (the 10 Bp shifts + 1 boundary sentinel) are resolved within the multi-degree CPR sumcheck.

Each shift represents a **left-shift** (look-ahead): $\mathrm{left\_shift}_c(v)[i] = v[i+c]$ for $i < N-c$, zero-padded otherwise.

---

## 4. E2E Prover Pipeline

The function `prove_hybrid_gkr_logup_4x_folded` (in `snark/src/pipeline.rs`) executes the following steps. With `full-fold`, the function `prove_hybrid_gkr_logup_full_folded` is used instead (see full-fold callouts in each subsection).

### 4.0. Step 0: Split Columns

**Input:** 20 private `BinaryPoly<32>` columns of length $N = 512$ (12 original + 8 μ quotient columns from no-f2x).

> **Without `no-f2x`:** 12 private columns.

**Operation:** Each `BinaryPoly<32>` element $a(X) = \sum_{i=0}^{31} c_i X^i$ is split into low and high halves:

$$a_{\mathrm{lo}}(X) = \sum_{i=0}^{15} c_i X^i \in \mathcal{R}_{16}, \quad a_{\mathrm{hi}}(X) = \sum_{i=0}^{15} c_{16+i} X^i \in \mathcal{R}_{16}$$

A column of length $N$ becomes a column of length $2N$ in $\mathcal{R}_{16}$, with the first $N$ entries being the low halves and the last $N$ entries being the high halves. The key algebraic identity is:

$$a(\alpha) = a_{\mathrm{lo}}(\alpha) + \alpha^{16} \cdot a_{\mathrm{hi}}(\alpha)$$

**Without `full-fold` (default, 4× = 2 rounds):** This is applied twice:
1. `split_columns::<32, 16>`: 20 columns × $N$ → 20 columns × $2N$ in $\mathcal{R}_{16}$
2. `split_columns::<16, 8>`: 20 columns × $2N$ → 20 columns × $4N$ in $\mathcal{R}_{8}$

**Output:** 20 `BinaryPoly<8>` columns of length $4N = 2048$. Each entry has only 8 bits (1 byte), making PCS encoding significantly cheaper.

**MLE dimension:** The original columns have `num_vars` = 9. After double-splitting, the committed columns have `num_vars` = 11 (two extra variables selecting the quarter).

**With `full-fold` (32× = 5 rounds):** Five successive splits are applied:
1. `split_columns::<32, 16>`: 20 columns × $N$ → 20 columns × $2N$ in $\mathcal{R}_{16}$
2. `split_columns::<16, 8>`: 20 columns × $2N$ → 20 columns × $4N$ in $\mathcal{R}_{8}$
3. `split_columns::<8, 4>`: 20 columns × $4N$ → 20 columns × $8N$ in $\mathcal{R}_{4}$
4. `split_columns::<4, 2>`: 20 columns × $8N$ → 20 columns × $16N$ in $\mathcal{R}_{2}$
5. `split_columns::<2, 1>`: 20 columns × $16N$ → 20 columns × $32N$ in $\mathcal{R}_{1}$

**Output:** 20 `BinaryPoly<1>` columns of length $32N = 16384$. Each entry is a single bit.

**MLE dimension:** `num_vars` = $9 + 5 = 14$ (five extra variables selecting the 1/32-th slice).

> **Without `no-f2x`:** 12 columns in both modes.

### 4.1. Step 1: PCS Commit

**Input:** 20 `BinaryPoly<8>` columns of length 2048 (or 20 `BinaryPoly<1>` columns of length 16384 with `full-fold`).

> **Without `no-f2x`:** 12 columns.

**Operation:** Zip+ commit using an IPRS linear code:

**Without `full-fold` (default):**
1. **Encode** each row of evaluations using the IPRS code `PnttConfigF2_16R4B32<2>`:
   - Input length: 2048 (= `BASE_LEN` × $8^{\text{DEPTH}}$ = 32 × 64)
   - Output length: 8192 (= `BASE_DIM` × $8^{\text{DEPTH}}$ = 128 × 64)
   - Rate: 1/4
   - Encoding: Radix-8 pseudo-NTT over $\mathbb{Z}$ using twiddle factors from $\mathbb{F}_{65537}$

**With `full-fold`:**
1. **Encode** using `PnttConfigF2_16R4B32<3>` (one DEPTH level higher):
   - Input length: 16384 (= `BASE_LEN` × $8^3$ = 32 × 512)
   - Output length: 65536 (= `BASE_DIM` × $8^3$ = 128 × 512)
   - Rate: 1/4 (unchanged)
   - Encoding: same radix-8 pseudo-NTT, one more recursion level

2. **Build a Merkle tree** (Blake3) where each leaf hashes a column across all 20 polynomial matrices.

**Output:**
- **Commitment** (sent to verifier): Merkle root (32 bytes) + batch size
- **Hint** (kept by prover): encoded matrices + Merkle tree for opening

### 4.2. Step 2: Transcript Init & Field Setup

1. Initialise a Blake3-based Fiat-Shamir transcript.
2. **Absorb** the Merkle root (32 bytes) into the transcript.
3. **Derive** a random 192-bit prime $p$ via Miller-Rabin: $\mathbb{F}_p$ becomes the PIOP field.
4. Construct `MontyParams<3>` (Montgomery reduction constants for $\mathbb{F}_p$).

### 4.3. Step 2a: Ideal Check (BinaryPoly UAIR)

**Protocol:** The Ideal Check verifies that every UAIR constraint $C_k(\mathrm{row}_j)$ lies in the prescribed ideal $\mathcal{I}_k$ for all rows $j \in [N-1]$.

**MLE-first path** (enabled because `max_degree` = 1 for `Sha256UairBpNoF2x`):

1. **Draw** $\mathbf{r} = (r_1, \ldots, r_n) \in \mathbb{F}_p^n$ from the transcript (the evaluation point).

2. **Precompute** the eq-table: $\mathrm{eq}(\mathbf{r}, j)$ for all $j \in [N]$ — a vector of $N$ field elements.

3. **Evaluate each column MLE at $\mathbf{r}$ directly from BinaryPoly data.** For each column $c$ and coefficient position $d \in [32]$:
   $$\widetilde{\mathrm{col}}_{c,d}(\mathbf{r}) = \sum_{j=0}^{N-1} \mathrm{eq}(\mathbf{r}, j) \cdot \mathbb{1}[\text{bit } d \text{ of trace}[c][j]]$$
   This is a running-sum pass over the trace: if bit $d$ of row $j$ is 1, add $\mathrm{eq}(\mathbf{r}, j)$. This avoids the $O(N \times C \times D)$ cost of projecting the entire trace to $\mathbb{F}_p[X]$.

4. **Apply the linear constraints once** on the MLE-evaluated column values. Since all 12 no-f2x Bp constraints are **strictly degree 1** (affine-linear in the column variables — no products of two witness columns) and use the **zero ideal**, every proof value evaluates to zero [Albert: this is not true is it? some constraints have X^{32}-1 and some X-2]. The IC proof for the Bp UAIR is **trivially empty** (0 bytes).

   > **Without `no-f2x`:** 16 constraints (4 cyclotomic + 12 zero); the IC proof contains 4 non-trivial cyclotomic check values.

   The coefficient-lift is linear, so the constraint output commutes with MLE evaluation:
   $$C_k^{\mathrm{lift}}(\widetilde{\mathrm{cols}}(\mathbf{r})) = \widetilde{C_k^{\mathrm{lift}}(\mathrm{cols})}(\mathbf{r})$$
   This identity would **fail** for degree-$\geq 2$ constraints because the lift does not preserve products (carries in $\mathbb{F}_2$ vs $\mathbb{F}_p$).

5. **Subtract last-row correction:** The Lagrange basis $L_{N-1}(\mathbf{r}) = \prod_i r_i$ evaluates the all-ones index. Compute:
   $$v_k = C_k^{\mathrm{full}}(\mathbf{r}) - L_{N-1}(\mathbf{r}) \cdot C_k(\text{last row})$$
   because constraints only apply to rows $0, \ldots, N-2$.

6. **Send and absorb** the proof $\pi_{\mathrm{IC}} = (v_1, \ldots, v_{12})$ — one $\mathbb{F}_p[X]$ polynomial per constraint. Each $v_k$ is immediately **absorbed into the transcript** (inside the IC prove function), binding these values before any subsequent challenges are drawn. With `no-f2x`, all 12 values are zero (zero ideal), so this step contributes **no non-trivial information** to the proof. (The serialised byte count depends on whether zero polynomials are explicitly encoded; logically the IC proof carries 0 bits of information.)

**Ideal membership:** The verifier checks $v_k \in \mathcal{I}_k$:
- **Zero ideal**: check $v_k = 0$.
- **Trivial ideal**: always passes.

> **Without `no-f2x`:** The verifier additionally checks cyclotomic membership: fold coefficients modulo 32 and check all vanish.

**Output:** The proof values $v_k$ and the evaluation point $\mathbf{r}$.

### 4.4. Step 2b: Ideal Check (Q[X] UAIR)

**When:** Feature `qx-constraints` is enabled.

The Q[X] Ideal Check runs **at the same evaluation point $\mathbf{r}$** produced by step 2a. It uses `prove_from_binary_poly_at_point`, which:

1. Projects the Q[X] scalars (i64 coefficients) to $\mathbb{F}_p[X]$.
2. Evaluates the **7** Q[X] constraints at $\mathbf{r}$ using the MLE-first path.
3. Sends and **absorbs** proof values $v'_1, \ldots, v'_7$ into the transcript. The 4 cyclotomic constraints (C1–C4) are checked against the cyclotomic ideal $(X^{32}-1)$; the 3 carry constraints (C7–C9) use the trivial ideal (or the degree-one ideal $(X-2)$ with `true-ideal`).

> **Without `no-f2x`:** Only 3 constraints (C7–C9, all trivial), producing a smaller IC proof.

**Purpose:** This proves both (a) that the rotation identities hold modulo $(X^{32}-1)$ over $\mathbb{Z}[X]$ (via the μ quotient correction), and (b) that the carry-propagation identities hold in $\mathbb{Z}[X]$ (lifted to $\mathbb{F}_p[X]$), providing soundness for the modular-$2^{32}$ additions.

**Soundness of the trivial ideal for C7–C9 (without `true-ideal`):** The trivial ideal choice means the IC membership check is vacuous for carry constraints — but this is **sound** because the Q[X] constraints hold on *all* 512 rows by construction:

- **Active rows (where selectors = 1):** The SHA-256 round equations hold by correct witness generation, and the carry $\mu$ absorbs the integer overflow.
- **Boundary rows (where selectors = 0):** The witness generator forces $a = e = 0$ on the last 3 rows and $W = 0$ on the last 16 rows. Both sides of each constraint evaluate to zero (all summands are zero and $\mu = 0$), so the constraint holds trivially.
- **Cross-compression boundaries:** At the start of each 64-round block, initial values are loaded as part of the SHA-256 chaining. The carry column compensates for any overflow.

Thus no selector gating is needed for the carry constraints: they hold everywhere, and the trivial ideal is sound.

**Soundness of the degree-one ideal for C7–C9 (with `true-ideal`):** The degree-one ideal $(X-2)$ makes the IC membership check **non-vacuous**: the verifier checks $v'_k(2) = 0$ for each carry constraint. The corrected constraint $c_{\text{inner}} + \text{corr\_add} - \text{corr\_sub}$ must evaluate to zero at $X = 2$, where $\text{BinaryPoly<32>}$ values become their natural integer representation. The correction pair absorbs signed residuals that arise at inactive boundary rows (where the base trace's carries may be incorrect). The witness generator recomputes carries at every row to ensure $|c_{\text{eval}}| < 2^{32}$, so each correction value fits in a single `BinaryPoly<32>`. Soundness error: $\leq 1/|\mathbb{F}_p|$ per constraint (Schwartz-Zippel on a degree-1 factor).

**Soundness of the cyclotomic ideal for C1–C4:** The μ correction $Q - 2\mu \in (X^{32}-1)$ is verified by checking membership in the cyclotomic ideal over $\mathbb{F}_p[X]$ after projection. This gives soundness error $\leq 31/|\mathbb{F}_p|$ per constraint (Schwartz-Zippel on a degree-31 polynomial after folding mod $X^{32}-1$).

### 4.4c. Step 2c: Ideal Check (Boundary UAIR)

**When:** Feature `boundary` is enabled.

The boundary Ideal Check runs **at the same evaluation point $\mathbf{r}$** produced by step 2a, after the Qx IC (step 2b). It verifies the 13 boundary constraints of `Sha256UairBoundaryNoF2x`.

**Sparse inline evaluation (optimisation):** Because all 13 boundary constraints are gated by selectors (`sel_init`, `sel_final_any`, `sel_msg`, `sel_zero`) that are nonzero on at most ~22 of the 512 rows, the prover uses a **sparse evaluation path** instead of iterating over all rows. The algorithm:

1. **Precompute** the eq-table $\mathrm{eq}(\mathbf{r}, j)$ for all $j \in [N]$ (same as §4.3, step 2).

2. **For each active row subset** (determined by which selector is nonzero):
   - **Row 0** (B1–B8): Evaluate all 8 initial-state constraints at row 0, weighted by $\mathrm{eq}(\mathbf{r}, 0)$.
   - **Rows $N-8, \ldots, N-5$** (B9–B10): Evaluate 2 final-output constraints at each of the 4 rows.
   - **Rows 0–15** (B11): Evaluate 1 message-input constraint at each of 16 rows.
   - **Rows $N-3, N-2, N-1$** (B12–B13): Evaluate 2 zero-padding constraints at each of 3 rows.

3. **Accumulate** each constraint's MLE-weighted evaluation: $v_k = \sum_{j \in S_k} \mathrm{eq}(\mathbf{r}, j) \cdot C_k(\mathrm{row}_j)$, where $S_k$ is the support set of the selector for constraint $k$.

4. **Subtract last-row correction** (as in §4.3, step 5): any constraint with nonzero selector at the last row needs correction.

5. **Send and absorb** the boundary IC proof $\pi_{\mathrm{bdry}} = (v_1, \ldots, v_{13})$ — one $\mathbb{F}_p[X]$ polynomial per constraint. Since all 13 constraints use `ImpossibleIdeal` (zero ideal), every proof value must be zero for a valid witness.

**Cost:** The sparse evaluation touches $\sim 22$ rows instead of 512, reducing the boundary IC cost by $\sim 23\times$ compared to a dense evaluation. The eq-table ($N$ field multiplications) is reused from the Bp IC.

**Proof size:** 13 zero-valued $\mathbb{F}_p[X]$ polynomials (logically 0 bits of information, as with the Bp IC under `no-f2x`).

### 4.5. Step 3a: Field Projection & Column Evaluation

> **No CPR sumcheck in E2E.** Because `max_degree` = 1 and the MLE-first Ideal Check
> already verified constraint satisfaction, the E2E prover **skips** the Combined Poly
> Resolver (CPR) sumcheck entirely. Instead, it directly evaluates all column MLEs at
> the IC evaluation point and absorbs the results into the transcript. The individual
> benchmark step "PIOP/Main field sumcheck" exercises the CPR as a standalone sumcheck,
> but this does **not** occur in the E2E prover path.
>
> **Soundness of direct evaluation.** The IC (§4.3–4.4) proves, at a random point
> $\mathbf{r}$, that the polynomial-ring constraints $C_k$ evaluated on the column MLEs
> lie in their prescribed ideals. The folding protocol (§4.8) + PCS (§4.9) proves that
> the committed columns' projected MLEs at $\mathbf{r}$ equal the claimed `up_evals`.
> Public column evaluations are computed directly by the verifier. Because all
> constraints are *linear* ($\max\_\deg = 1$), the constraint outputs are uniquely
> determined by the column MLE values. Therefore:
>
> - **IC** binds the constraint evaluations to the column MLE values at $\mathbf{r}$.
> - **Folding + PCS** binds the committed columns to their claimed evaluations.
> - **Linearity** means these two facts compose without an additional CPR sumcheck.
>
> A cheating prover would have to either fake the IC proof (breaking ideal membership)
> or fake the PCS opening (breaking proximity soundness).
>
> **With `boundary`:** The above direct-evaluation path is **replaced** by a
> multi-degree CPR sumcheck (see §4.5b below). The boundary UAIR has `max_degree` = 2,
> which requires a CPR sumcheck to reduce constraint-plus-column evaluations to a
> single point. The Bp CPR group (degree 1) and boundary CPR group (degree 4 = constraint
> degree 2 + CPR overhead 2) are batched into a single multi-degree sumcheck. The
> standalone shift sumcheck (§4.7) is **skipped** because the CPR handles shifts
> internally via `down_evals`.

1. **Draw** the projecting element $\alpha \in \mathbb{F}_p$ from the transcript.

2. **Project trace to field:** For each column $c$ and each row $j$, compute:
   $$\mathrm{trace}_F[c][j] = \pi_\alpha(\mathrm{trace}[c][j]) = \sum_{i=0}^{31} c_i \cdot \alpha^i \in \mathbb{F}_p$$
   using the 4-chunk table lookup. This produces 38 field-element columns of length $N$.

3. **Extract lookup columns:** Identify the 10 base lookup columns from the field trace. Compute raw integer indices for the decomposed lookup (4 chunks of 8 bits each).

4. **Construct affine virtual columns:** For each of the 3 `AffineLookupSpec`s, compute the virtual column as an integer-linear combination of projected trace columns:
   $$\mathrm{virtual}[j] = \sum_{(\mathrm{col}, \mathrm{coeff}) \in \mathrm{terms}} \mathrm{coeff} \cdot \mathrm{trace}_F[\mathrm{col}][j] + \pi_\alpha(\mathrm{offset})$$

5. **Extract shift-source columns** for the shift sumcheck.

6. **Directly evaluate all field-trace column MLEs at $\mathbf{r}$** (the IC evaluation point) — no sumcheck is needed:
   - `up_evals[c]` = $\widetilde{\mathrm{trace}_F[c]}(\mathbf{r})$ for each column $c$ (the "unshifted" evaluations)
   - `down_evals[i]` = $\widetilde{\mathrm{trace}_F[\mathrm{src}][s+\cdot]}(\mathbf{r})$ for each shift spec $i$ (shifted evaluations)

7. **Absorb** all `up_evals` and `down_evals` into the transcript. These values become part of the proof — the verifier will check them via the folding protocol and PCS.

> **With `boundary`:** Steps 5–7 above are replaced. Instead of direct evaluation and
> absorb, the prover runs the boundary CPR path described in §4.5b below.

#### 4.5b. Boundary CPR Path (with `boundary`)

When the `boundary` feature is enabled, a **multi-degree CPR sumcheck** replaces the direct-evaluation path. The shift sumcheck (§4.7) is also skipped because the CPR handles shifts internally.

**Setup:**

1. **Clone base field trace.** Before the Bp CPR group consumes `field_trace`, it is cloned to `base_field_trace_for_bdry`. This avoids re-projecting 33 base columns from `BinaryPoly<D>` — the most expensive step would otherwise be a redundant $O(33 \times N \times D)$ projection.

2. **Build Bp CPR group (degree 1).** The standard `CombinedPolyResolver::build_prover_group` is called with the BinaryPoly UAIR, consuming `field_trace`.

3. **Build boundary CPR group (degree 4).** The boundary trace is constructed by:
   - Reusing the cloned base field trace (33 bitpoly columns already projected)
   - Generating only the 7 boundary columns via `generate_boundary_columns_only()`, which creates the selector and output MLEs from the raw trace without building a full extended witness
   - Projecting only these 7 new columns from `BinaryPoly<D>` to $\mathbb{F}_p$
   - Assembling `[base_bitpoly(0..33) | boundary(7) | integer(3)]` = 43 columns (or 49 with `true-ideal`)
   - Calling `CombinedPolyResolver::build_prover_group::<Sha256UairBoundaryNoF2x>` with the reconstructed trace; CPR degree = constraint degree 2 + overhead 2 = **4**

4. **Multi-degree sumcheck.** `MultiDegreeSumcheck::prove_as_subprotocol` runs $n = 9$ rounds with exactly **2 groups**: `[Bp CPR (degree 1), boundary CPR (degree 4)]`. In each round:
   - Each group computes its own round polynomial (degree 1 or 4 respectively)
   - Round polynomials are absorbed into the transcript sequentially (group 0, then group 1)
   - A single shared challenge $r_i$ is squeezed and applied to both groups

5. **Finalise Bp CPR (group 0).** Extract `up_evals` (38 values) and `down_evals` (10 values) at the sumcheck point $\mathbf{s}$. Absorb into transcript.

6. **Finalise boundary CPR (group 1).** Extract `bdry_up_evals` and `bdry_down_evals` at $\mathbf{s}$. Absorb into transcript.

**Output:** The evaluation point $\mathbf{s} = (s_1, \ldots, s_9)$ — the challenge point produced by the multi-degree sumcheck — replaces the IC evaluation point $\mathbf{r}$ for subsequent folding and PCS steps. All `up_evals` and `down_evals` from both CPR groups are serialised into the proof.

### 4.6. Step 3b: Hybrid GKR Batched Lookup

**Protocol:** Hybrid GKR Batched Decomposed LogUp with `cutoff` = 2.

**Input:** 13 lookup columns (10 base + 3 affine), each of length $N = 512$, all referencing the same table `BitPoly{width:32, chunk_width:8}`.

#### 4.6.1. Decomposition

Each 32-bit value $w$ projected at $\alpha$ decomposes into 4 chunks:

$$\pi_\alpha(w) = c_0 + \alpha^8 \cdot c_1 + \alpha^{16} \cdot c_2 + \alpha^{24} \cdot c_3$$

where each $c_k$ lies in the **sub-table** $T_{\mathrm{sub}} = \{\pi_\alpha(b) : b \in \mathcal{R}_8\}$ of size $2^8 = 256$.

Shifts: $[1, \alpha^8, \alpha^{16}, \alpha^{24}]$.

#### 4.6.2. LogUp Identity

For each lookup column $\ell \in [13]$, the logarithmic-derivative identity is:

$$\sum_{i=0}^{N-1} \frac{1}{\beta - w_i^{(\ell)}} = \sum_{j=0}^{255} \frac{m_j^{(\ell)}}{\beta - T_{\mathrm{sub}}[j]}$$

where:
- $\beta \in \mathbb{F}_p$ is a random challenge
- $w_i^{(\ell)}$ are the projected witness values
- $m_j^{(\ell)} = \sum_{k=0}^{3} m_{k,j}^{(\ell)}$ is the **aggregated multiplicity** (total count of sub-table entry $j$ across all 4 chunks)

#### 4.6.3. GKR Fraction Tree

For each lookup $\ell$, build a binary **fraction tree** with $K \cdot N = 4 \times 512 = 2048$ leaves (padded to next power of 2):

- **Leaf $(k, i)$**: numerator $p = 1$, denominator $q = \beta - c_k^{(\ell)}[i]$
- **Internal nodes** combine via: $p_{\mathrm{parent}} = p_L q_R + p_R q_L$, $q_{\mathrm{parent}} = q_L q_R$
- **Root**: $(P_w^{(\ell)}, Q_w^{(\ell)})$ encodes $\sum_{k,i} 1/(\beta - c_k^{(\ell)}[i])$

A separate **table tree** (256 leaves) encodes:
- Leaf $j$: $p_j = \sum_\ell \gamma^\ell m_{\mathrm{agg}}^{(\ell)}[j]$, $q_j = \beta - T[j]$

where $\gamma$ is a batching challenge.

#### 4.6.4. Hybrid GKR Protocol (cutoff = 2)

The **top 2 layers** of the witness fraction trees are verified via GKR sumchecks:

- **Layer 0**: Direct algebraic check of the root identity (no sumcheck variables).
- **Layer 1**: Degree-3 sumcheck over 1 variable, checking the internal-node combination identity:
  $$\mathrm{eq}(x, r_0) \cdot \left[(p_L(x) + \gamma' q_L(x)) \cdot q_R(x) + p_R(x) \cdot q_L(x)\right]$$

At the cutoff, the prover sends $2 \times 13 \times 2^2 = 104$ field-element **intermediate values** (the fraction numerators and denominators at layer 2).

The **table tree** (small: depth 8) is verified with **full GKR** (all 8 layers).

#### 4.6.5. Cross-Check

The prover and verifier verify the **root cross-check identity**:

$$\sum_\ell \gamma^\ell \cdot \frac{P_w^{(\ell)}}{Q_w^{(\ell)}} = \frac{P_t}{Q_t}$$

computed via prefix/suffix products to avoid division.

### 4.7. Step 3c: Shift Sumcheck

> **With `boundary`:** This step is **skipped entirely**. When the `boundary` feature is
> enabled, all shift claims (10 Bp + 1 boundary sentinel) are resolved within the
> multi-degree CPR sumcheck (§4.5b). The CPR's internal handling of `down_evals`
> subsumes the shift sumcheck.

**Protocol:** Batched shift sumcheck reducing shifted-column evaluation claims to unshifted MLE claims.

**Input:** 10 shift claims (from BinaryPoly UAIR), each of the form:
$$\widetilde{\mathrm{left\_shift}_c(v)}(\mathbf{r}) = a_i$$

**Algorithm:**

1. **Batch:** Draw random coefficients $\xi_1, \ldots, \xi_{10}$ from the transcript. Form a single sumcheck:
   $$\sum_{\mathbf{b} \in \{0,1\}^n} \left[\sum_{i=1}^{10} \xi_i \cdot h_i(\mathbf{b}) \cdot v_i(\mathbf{b})\right] = \sum_{i=1}^{10} \xi_i \cdot a_i$$
   where $h_i(\mathbf{b})$ is the left-shift predicate: $h_i(j) = \mathrm{eq}(\mathbf{r}, j - c_i)$ for $j \geq c_i$, zero otherwise.

2. **Sumcheck:** Run $n = 9$ rounds. Each round, the prover sends a degree-2 univariate polynomial represented by 3 evaluations $(p_\ell(0), p_\ell(1), p_\ell(2))$. The verifier checks $p_\ell(0) + p_\ell(1) = \mathrm{current\_claim}$ and provides a challenge $s_\ell$.

3. **Final evaluations:** After all rounds, the challenge point $\mathbf{s} = (s_1, \ldots, s_n)$ is produced. The prover computes $v_i^{\mathrm{final}} = \widetilde{v_i}(\mathbf{s})$ for each private source column and sends these.

4. **Verifier check:** Recompute the shift predicate $S_{c_i}(\mathbf{s}, \mathbf{r})$ via the decomposition into "no-carry" and "carry" parts (cost $O(n + c \cdot \log c)$). Verify:
   $$\mathrm{last\_claim} = \sum_{i=1}^{10} \xi_i \cdot S_{c_i}(\mathbf{s}, \mathbf{r}) \cdot v_i^{\mathrm{final}}$$

**Output:** The private source column evaluations $v_i^{\mathrm{final}}$ at point $\mathbf{s}$ are absorbed into the transcript. These values are **not** opened via a separate PCS call. Instead, they are secured by the algebraic reduction chain:

$$\text{PCS at } \mathbf{r}_2 \xrightarrow{\text{folding}} \text{up\_evals at } \mathbf{r} \xrightarrow{\text{IC}} \text{down\_evals at } \mathbf{r} \xrightarrow{\text{shift sumcheck}} v_i^{\mathrm{final}} \text{ at } \mathbf{s}$$

A cheating prover who supplies fake $v_i^{\mathrm{final}}$ must break either the shift sumcheck (contradicting its earlier round messages), the IC (producing wrong `down_evals`), or the PCS (providing inconsistent committed values).

**Proof size:** $3n + k_{\mathrm{private}}$ field elements, where $k_{\mathrm{private}} = 6$ (the shift-source-only columns `D_hat`, `H_hat`, `a_tm1`, `a_tm2`, `e_tm1`, `e_tm2` — indices 14, 15, 21–24). The remaining **4** shift sources are public columns whose $v_i^{\mathrm{final}}$ the verifier recomputes from known data. This makes the shift sumcheck proof compact: $3 \times 9 + 6 = 33$ field elements.

#### 4.7.1. Soundness Argument for `down_evals`

The `down_evals` (shifted-column MLE evaluations at $\mathbf{r}$) are prover-supplied values that are **not** directly opened via PCS. Their correctness is enforced by the following chain:

> **Theorem (Informal).** If the verifier accepts the proof, then with probability $\geq 1 - \varepsilon$ (where $\varepsilon$ is negligible in the security parameter), the prover's `down_evals` are the correct MLE evaluations of the shifted source columns at $\mathbf{r}$.

**Proof sketch.** Each `down_eval[i]` participates in two independent algebraic checks:

1. **IC constraint binding.** The BinaryPoly UAIR constraints C7–C16 enforce:
   $$\mathrm{lookback\_col}[t + \delta] - \mathrm{source}[t] = 0 \quad \forall t \in [N-1]$$
   The IC proves this holds at the random point $\mathbf{r}$, which means:
   $$\widetilde{\mathrm{lookback\_col}}(\mathbf{r}) = \widetilde{\mathrm{source}}(\mathbf{r})$$
   (the lookback column's MLE at $\mathbf{r}$ equals the source column's MLE at $\mathbf{r}$, with the last-row correction being zero because both sides vanish at boundary rows). The `up_evals` for lookback columns and source columns are bound by PCS+folding, so `down_evals` are algebraically determined.

2. **Shift sumcheck binding.** The shift sumcheck asserts:
   $$\sum_i \xi_i \cdot \mathrm{down\_eval}_i = \sum_{\mathbf{b} \in \{0,1\}^n} \sum_i \xi_i \cdot h_i(\mathbf{b}) \cdot \widetilde{v_i}(\mathbf{b})$$
   where $\xi_i$ are random batching coefficients drawn *after* `down_evals` are absorbed. By Schwartz-Zippel, if any `down_eval[i]` is incorrect, the batched sum disagrees with the true sum with probability $\leq 1/|\mathbb{F}_p|$ per incorrect entry. The sumcheck then prevents the prover from faking the reduction.

**Key observation:** The two mechanisms are complementary. The IC provides a *deterministic* algebraic relation between `down_evals` and `up_evals`. The shift sumcheck provides a *probabilistic* check that `down_evals` match the committed source data. Together, a cheating prover must simultaneously break both, which requires either faking the IC proof or the PCS opening. $\square$

### 4.8. Step 4: Folding Protocol (2-Round / 5-Round)

**Purpose:** Reduce the PIOP evaluation claims from the original `BinaryPoly<32>` columns to the committed PCS columns (`BinaryPoly<8>` without `full-fold`; `BinaryPoly<1>` with `full-fold`).

The key algebraic identity is: for $a(X) = a_{\mathrm{lo}}(X) + X^{D/2} \cdot a_{\mathrm{hi}}(X)$:

$$a(\alpha) = a_{\mathrm{lo}}(\alpha) + \alpha^{D/2} \cdot a_{\mathrm{hi}}(\alpha)$$

Since $\pi_\alpha$ is $\mathbb{F}_p$-linear on lifted coefficient vectors (see §2.3), it commutes with MLE evaluation. Combined with MLE linearity:

$$\widetilde{a^{\mathrm{proj}}}(\mathbf{r}) = \widetilde{a_{\mathrm{lo}}^{\mathrm{proj}}}(\mathbf{r}) + \alpha^{D/2} \cdot \widetilde{a_{\mathrm{hi}}^{\mathrm{proj}}}(\mathbf{r})$$

The verifier checks $c_{1,j} + \alpha^{D/2} \cdot c_{2,j} = e_j$ where $e_j = \mathrm{up\_evals}[j]$. This identity holds by the projection-MLE commutativity established in §2.3.

#### Round 1: $D = 32 \to \mathrm{HALF\_D} = 16$

For each of the 20 committed columns $j$ (12 original + 8 μ quotient columns with no-f2x). The split doubles each column's *length* (appending the hi-half evaluations), not the column count, so 20 columns remain throughout:

1. **Prover computes:**
   $$c_{1,j} = \widetilde{u_j^{\mathrm{proj}}}(\mathbf{r}), \quad c_{2,j} = \widetilde{w_j^{\mathrm{proj}}}(\mathbf{r})$$
   where $u_j$ (low half) and $w_j$ (high half) are the `BinaryPoly<16>` split columns, projected at $\alpha$.

2. **Absorb** $(c_{1,j}, c_{2,j})$ into the transcript.

3. **Draw** challenge $\gamma_1 \in \mathbb{F}_p$.

4. **New point:** $\mathbf{r}_1 = (\mathbf{r} \| \gamma_1)$ (10 variables → point in $\mathbb{F}_p^{10}$).

5. **New claims:** $d_{1,j} = (1 - \gamma_1) c_{1,j} + \gamma_1 c_{2,j}$.

**Verifier check (at verification time):** $c_{1,j} + \alpha^{16} \cdot c_{2,j} = e_j$ where $e_j$ is the PIOP's claimed evaluation of column $j$.

#### Round 2: $\mathrm{HALF\_D} = 16 \to \mathrm{QUARTER\_D} = 8$

Identical structure, using the `BinaryPoly<8>` split columns (still 20 columns, now length $4N = 2048$) and verifying $c_{3,j} + \alpha^8 \cdot c_{4,j} = d_{1,j}$.

**New point:** $\mathbf{r}_2 = (\mathbf{r} \| \gamma_1 \| \gamma_2)$ (11 variables).

**Proof cost per round:** 2 field elements per committed column ($c_1, c_2$). Total for 2 rounds: $2 \times 20 \times 2 = 80$ field elements (with no-f2x; 48 without).

#### Full-fold: Rounds 3–5 (with `full-fold`)

With `full-fold`, three additional rounds are applied after Round 2:

- **Round 3** ($8 \to 4$): Verifies $c_{5,j} + \alpha^4 \cdot c_{6,j} = d_{2,j}$ using `BinaryPoly<4>` split columns (length $8N = 4096$). New point: $\mathbf{r}_3 = (\mathbf{r} \| \gamma_1 \| \gamma_2 \| \gamma_3)$ (12 variables).
- **Round 4** ($4 \to 2$): Verifies $c_{7,j} + \alpha^2 \cdot c_{8,j} = d_{3,j}$ using `BinaryPoly<2>` split columns (length $16N = 8192$). New point: 13 variables.
- **Round 5** ($2 \to 1$): Verifies $c_{9,j} + \alpha^1 \cdot c_{10,j} = d_{4,j}$ using `BinaryPoly<1>` split columns (length $32N = 16384$). New point: $\mathbf{r}_5 = (\mathbf{r} \| \gamma_1 \| \cdots \| \gamma_5)$ (14 variables).

**Total proof cost for 5 rounds:** $2 \times 20 \times 5 = 200$ field elements (with no-f2x; 120 without).

#### Bridging scalar claims to ring-valued commitments

The PIOP produces **scalar** $\mathbb{F}_p$ evaluation claims $e_j = \pi_\alpha(\widetilde{\mathrm{col}}_j(\mathbf{r}))$, but the PCS commits to **ring-valued** `BinaryPoly<8>` MLEs — not scalar MLEs. After two rounds of folding, the verifier holds per-column scalar claims $d_{2,j}$ and needs to confirm that these are consistent with the ring-valued MLE evaluations $\widetilde{\mathrm{col}}_j^{(8)}(\mathbf{r}_2)$ that the PCS binds.

The folding identity $c_1 + \alpha^{D/2} \cdot c_2 = e$ is the **projection-evaluation factorisation** of $\pi_\alpha$ through the coefficient splitting:

$$\pi_\alpha(a) = a(\alpha) = \underbrace{a_{\mathrm{lo}}(\alpha)}_{c_1} + \alpha^{D/2} \cdot \underbrace{a_{\mathrm{hi}}(\alpha)}_{c_2}$$

After two rounds, the folding reduces 32-coefficient ring evaluations to 8-coefficient ring evaluations. At this point the `BinaryPoly<8>` columns **are** the committed PCS data, and the PCS proximity check binds them to the Merkle commitment. With `full-fold`, five rounds reduce to 1-coefficient ring evaluations (`BinaryPoly<1>`).

To close the soundness gap between the PIOP's scalar claims and the PCS's ring-valued commitments, the prover supplies **ring-valued MLE evaluations** $v_{j,k}$ — one per coefficient $k \in [8]$ per committed column $j$ — and the verifier checks the **$\alpha$-projection consistency** (see §4.8b and §5.4b).

**Soundness link between folding, lifting, and PCS:** The soundness composition has three parts:

1. **PCS proximity** constrains: the Merkle-committed data is close to a valid codeword, which (by the distance property of the IPRS code) determines the ring-valued evaluations $\widetilde{\mathrm{col}}_j^{(8)}(\mathbf{r}_2) \in \mathcal{R}_8$ uniquely.
2. **Ring-eval bridge** constrains: for each committed column $j$, the prover's claimed coefficient evaluations $v_{j,k}$ must satisfy $\sum_k \alpha^k v_{j,k} = d_{2,j}$ (the folding output). This is a Schwartz-Zippel check on the polynomial $\widetilde{\mathrm{col}}_j^{(8)}(\mathbf{r}_2)(X) - \sum_k v_{j,k} X^k$ evaluated at $\alpha$: if even one $v_{j,k}$ is incorrect, the check fails with probability $\geq 1 - 8/p \approx 1$.
3. **Folding** constrains: the chain $c_1 + \alpha^{D/2} c_2 = e$ decomposes each PIOP scalar claim into half-ring claims, reducing to the $d_{2,j}$ checked by the bridge.

Together: the PCS fixes the ring-valued evaluations (via binding), the ring-eval bridge verifies that the prover's coefficient-level claims are consistent with the folding's scalar claims, and the folding verifies that those scalar claims match the PIOP. A cheating prover would need to find committed data that satisfies PCS proximity AND has $\alpha$-projections inconsistent with the PIOP claims — this contradicts the code's distance guarantee combined with the Schwartz-Zippel bound on the bridge check.

**The PCS's internal alpha-combination (§4.9.2) is unrelated to the PIOP's projecting element $\alpha$.** The PCS alpha challenges are 128-bit i128 values squeezed from the PCS transcript (seeded with the Merkle root); they batch the 20 ring-valued polynomial openings into a single scalar check for efficiency. The PIOP's $\alpha \in \mathbb{F}_p$ is a 192-bit field element squeezed from the PIOP transcript; it projects ring elements to scalars for constraint checking. These two sets of challenges serve different purposes and are drawn from independent transcript instances.

### 4.8b. Step 4b: Ring-Eval Lifting (PIOP↔PCS Bridge)

**Purpose:** Compute the ring-valued MLE evaluations that bridge the PIOP's scalar folding claims to the PCS's ring-valued commitments.

**Input:** The 20 committed `BinaryPoly<8>` columns (each 2048 entries, 11 MLE variables), the doubly-extended point $\mathbf{r}_2 = (\mathbf{r} \| \gamma_1 \| \gamma_2) \in \mathbb{F}_p^{11}$.

> **With `full-fold`:** 20 `BinaryPoly<1>` columns (each 16384 entries, 14 MLE variables), point $\mathbf{r}_5 = (\mathbf{r} \| \gamma_1 \| \cdots \| \gamma_5) \in \mathbb{F}_p^{14}$.

**Algorithm (`compute_ring_evals<Q>` where $Q$ = 8 or 1):**

1. **Eq-tensor precomputation (once):** Build the evaluation table of the equality polynomial $\mathrm{eq}(\mathbf{r}_{\mathrm{final}}, \cdot)$ over all $2^{n_{\mathrm{final}}}$ binary inputs. Starting from $\mathrm{eq}[\mathbf{0}] = 1$, for each variable $i = 0, \ldots, n_{\mathrm{final}}-1$ and each existing index $b < 2^i$:
   $$\mathrm{eq}[b + 2^i] \leftarrow \mathrm{eq}[b] \cdot r_i, \qquad \mathrm{eq}[b] \leftarrow \mathrm{eq}[b] \cdot (1 - r_i)$$
   Cost: $2 \times 2^{n_{\mathrm{final}}}$ field multiplications (4096 for $Q=8$; 32768 for $Q=1$).

2. **Dot-product accumulation (per column, parallelised):** For each committed column $j \in [20]$, maintain $Q$ accumulators $a_0, \ldots, a_{Q-1} = 0$. For each entry index $b \in [2^{n_{\mathrm{final}}}]$, read the packed bits of `BinaryPoly<Q>` entry $b$ and for each set bit $k$:
   $$a_k \mathrel{+}= \mathrm{eq}[b]$$
   The final accumulators give $v_{j,k} = a_k = \sum_{\mathbf{b}: c_k[\mathbf{b}]=1} \mathrm{eq}(\mathbf{r}_{\mathrm{final}}, \mathbf{b})$.
   Cost per column: at most $Q \times 2^{n_{\mathrm{final}}}$ field additions.

**Without `full-fold`:** This produces $20 \times 8 = 160$ field elements (with no-f2x; $12 \times 8 = 96$ without).

**With `full-fold`:** This produces $20 \times 1 = 20$ field elements (with no-f2x; $12 \times 1 = 12$ without). Since $Q = 1$, each `BinaryPoly<1>` has a single coefficient, so the ring-eval for each column is a single field element — which is in fact the projected MLE evaluation itself (the $\alpha$-projection of a 1-bit polynomial $c_0 \cdot X^0$ is just $c_0$).

**Complexity:** $O(2^{n_{\mathrm{final}}})$ field multiplications for the eq tensor (computed once) plus $O(Q \cdot N_{\text{cols}} \cdot 2^{n_{\mathrm{final}}})$ field additions for the accumulation, where $Q \in \{8, 1\}$, $N_{\text{cols}} = 20$, $n_{\mathrm{final}} \in \{11, 14\}$. The previous approach required $O(Q \cdot N_{\text{cols}} \cdot 2^{n_{\mathrm{final}}})$ field multiplications (one full MLE evaluation per coefficient per column); the eq-tensor method replaces these with additions.

**Key identity:** By linearity of MLE evaluation,
$$\sum_{k=0}^{Q-1} \alpha^k \cdot v_{j,k} = \sum_{k=0}^{Q-1} \alpha^k \cdot \widetilde{\mathbf{c}_k}(\mathbf{r}_{\mathrm{final}}) = \pi_\alpha\bigl(\widetilde{\mathrm{col}}_j^{(Q)}(\mathbf{r}_{\mathrm{final}})\bigr) = d_{\mathrm{last},j}$$

The verifier checks this identity in §5.4b.

**Proof cost:** $20 \times Q$ field elements. Without `full-fold`: $20 \times 8 = 160$ ($160 \times 24 = 3840$ bytes). With `full-fold`: $20 \times 1 = 20$ ($20 \times 24 = 480$ bytes).

**Timing:** Measured as the `Lifting` step in the benchmark timing breakdown.

### 4.9. Step 5: PCS Prove

**Input:** 20 `BinaryPoly<8>` columns of length 2048 (each stored as a multilinear polynomial with 11 variables), evaluation point $\mathbf{r}_2 \in \mathbb{F}_p^{11}$, and the commitment hint (20 encoded codeword matrices + shared Merkle tree) from the commit phase.

> **With `full-fold`:** 20 `BinaryPoly<1>` columns of length 16384 (14 variables), evaluation point $\mathbf{r}_5 \in \mathbb{F}_p^{14}$.

> **Without `no-f2x`:** 12 columns.

**Operation:** `ZipPlus::prove_with_seed`, with the Merkle root absorbed as a seed so that the PCS transcript derives the same 192-bit prime $p$ as the PIOP transcript.

#### 4.9.0. PCS Transcript Initialisation

A fresh `PcsTranscript` (Blake3-based) is created. The Merkle root (32 bytes) is absorbed as seed, then:

- **Field derivation:** The transcript squeezes 24 bytes, applies the same Miller-Rabin prime search (§2.2) used by the PIOP, producing a 192-bit prime $p$ and the Montgomery field configuration. Because the seed is identical (same Merkle root), the PCS operates over the same $\mathbb{F}_p$ as the PIOP. The evaluation point $\mathbf{r}_2$ (11 $\mathbb{F}_p$ elements) is passed as an explicit parameter — the PCS never derives it from the transcript.

#### 4.9.1. Tensorisation

The evaluation point (11 variables without `full-fold`, 14 with `full-fold`) is split by `point_to_tensor` into:

**Without `full-fold`:**
- **Row-selector** $q_0 \in \mathbb{F}_p^{512}$: the $\mathrm{eq}$-table for the **low-order** 9 coordinates $(r_3, \ldots, r_{11})$, i.e. $q_0[\mathbf{b}] = \mathrm{eq}((r_3,\ldots,r_{11}), \mathbf{b})$ for $\mathbf{b} \in \{0,1\}^9$.
- **Column-selector** $q_1 \in \mathbb{F}_p^{4}$: the $\mathrm{eq}$-table for the **high-order** 2 coordinates $(r_1, r_2)$, i.e. $q_1[\mathbf{c}] = \mathrm{eq}((r_1, r_2), \mathbf{c})$ for $\mathbf{c} \in \{0,1\}^2$.

The split corresponds to interpreting each polynomial's $2^{11} = 2048$ evaluations as a $512 \times 4$ matrix (512 rows, 4 columns within each `BinaryPoly<8>` coefficient).

**With `full-fold`:**
- **Row-selector** $q_0$: eq-table for the low-order coordinates, with `num_rows` determined by `row_len` = `PnttConfigF2_16R4B32<3>::INPUT_LEN` = 16384. Since each polynomial has $2^{14} = 16384$ evaluations and `row_len` = 16384, we get `num_rows` = 1 and $q_0 = [1]$ (a single-element vector).
- **Column-selector** $q_1 \in \mathbb{F}_p^{16384}$: eq-table for all 14 coordinates.

> **Note on `num_rows` = 1 in full-fold:** When the polynomial length equals `row_len`, the entire polynomial is a single row. The $b$-vector (§4.9.3) degenerates to a single element, and the row-challenge step squeezes just 1 coefficient.

> **Note on "4 columns" (without `full-fold`):** After 4×-folding, each `BinaryPoly<8>` has 8 coefficients, but the IPRS code's `row_len` for `PnttConfigF2_16R4B32<2>` is $32 \times 8^2 = 2048$. Since each polynomial has $2^{11}$ evaluations and `num_rows` = 512, we get `row_len` = $2048 / 512 = 4$, matching the 2 high-order MLE variables.

#### 4.9.2. Alpha-Combination (Polynomial Batching)

For each of the 20 polynomials, the transcript squeezes $D'+1$ i128 challenges $(\alpha_{i,0}, \ldots, \alpha_{i,D'})$ where $D'$ is the `DEGREE_BOUND` of the committed ring type (7 for `BinaryPoly<8>`, 0 for `BinaryPoly<1>`). These project each ring evaluation $\hat{p}(X) = \sum_{j=0}^{Q-1} c_j X^j$ (with $c_j \in \{0,1\}$) into a 384-bit integer:

$$e_i^{\mathrm{comb}} = \sum_{j=0}^{Q-1} c_j \cdot \alpha_{i,j} \in \mathrm{Int}\langle 6 \rangle$$

> **With `full-fold` ($Q = 1$):** Each `BinaryPoly<1>` has a single coefficient $c_0 \in \{0,1\}$, and the alpha-combination degenerates to $e_i^{\mathrm{comb}} = c_0 \cdot \alpha_{i,0}$. The `DEGREE_BOUND` = 0, so only 1 challenge is squeezed per polynomial.

This is computed via `BinaryPolyInnerProduct`, which exploits the binary coefficients (no multiplication — just conditional addition). All 20 × 2048 evaluations are combined in a single parallel batch (rayon `par_iter` over the 20 polynomials).

**Result:** 20 vectors of 2048 `Int<6>` (384-bit) values each.

> **Why independent $\alpha_{i,j}$ instead of powers $\alpha^j$?**
> The PIOP's folding claims use the structured projection $\pi_\alpha(a) = \sum_k c_k \alpha^k$ (powers of a single PIOP element $\alpha \in \mathbb{F}_p$), while the PCS uses independent random 128-bit challenges $\alpha_{i,j}$ here. These are two **different** linear functionals on the coefficients of `BinaryPoly<8>`, and they serve different purposes.
>
> The PCS does not verify the PIOP's structured projection. Instead, the PCS uses the independent $\alpha_{i,j}$ as a **Schwartz-Zippel batching trick**: it reduces "verify 20 × 8 individual coefficient-level MLE openings" to "verify 1 aggregate scalar". If any single coefficient of any single polynomial's MLE evaluation were incorrect, the random linear combination would mismatch with overwhelming probability ($\leq 1/2^{128}$ per coefficient). So the PCS effectively binds **all 8 coefficients of all 20 columns** to the Merkle commitment — not just one particular projection.
>
> The folding check ($c_1 + \alpha^{D/2} c_2 = e_j$ per column) is verified **separately** by the PIOP verifier using the PIOP transcript. The verifier never algebraically links the PCS's `eval_f` to the folding claims $d_{2,j}$. The soundness composition is: (1) the PCS proximity check fixes all ring-valued evaluations $\widetilde{\mathrm{col}}_j^{(8)}(\mathbf{r}_2) \in \mathcal{R}_8$ (via code distance), and (2) the folding check then verifies that the structured $\alpha$-projections of those now-fixed ring values are consistent with the PIOP claims. Since the $\alpha$-projection is a deterministic function of the ring-valued evaluation, once the PCS binds all coefficients, the folding identity is forced to hold.

#### 4.9.3. Compute $b$ (Row-Column Inner Products)

For each of the `num_rows` rows $i$ and each polynomial $m$, compute the inner product of the row's `row_len` `Int<6>` entries (after lifting to $\mathbb{F}_p$ via Montgomery reduction) with the column-selector $q_1$:

$$b_m[i] = \sum_{k=0}^{\mathrm{row\_len}-1} \pi_p\bigl(e_m^{\mathrm{comb}}[\mathrm{row\_len} \cdot i + k]\bigr) \cdot q_1[k]$$

where $\pi_p : \mathrm{Int}\langle 6 \rangle \to \mathbb{F}_p$ is reduction modulo $p$ (Montgomery conversion).

The per-polynomial $b$-vectors are then summed element-wise:

$$b[i] = \sum_{m=0}^{19} b_m[i], \quad i \in [\mathrm{num\_rows}]$$

**Without `full-fold`:** `num_rows` = 512, `row_len` = 4. $b$ is a 512-element vector.

**With `full-fold`:** `num_rows` = 1, `row_len` = 16384. $b$ degenerates to a single element (dot product of the entire polynomial with $q_1$).

**Transcript:** The $b$-vector is written to the proof stream (`write_field_elements`). Each element is a 192-bit Montgomery field element serialised as 24 bytes.

**Proof contribution:** Without `full-fold`: $512 \times 24 = 12\,288$ bytes. With `full-fold`: $1 \times 24 = 24$ bytes.

#### 4.9.4. Compute Evaluation

$$\mathrm{eval} = \langle q_0, b \rangle = \sum_{i=0}^{\mathrm{num\_rows}-1} q_0[i] \cdot b[i] \in \mathbb{F}_p$$

This is the claimed evaluation of the batched polynomial at the PCS evaluation point. It is returned to the pipeline caller and serialised into the proof as `pcs_evals_bytes` (not written to the PCS proof stream itself). The verifier deserialises it from the proof and checks consistency: $\langle q_0, b \rangle = \mathrm{eval}$.

> **With `full-fold` (`num_rows` = 1):** eval = $q_0[0] \cdot b[0] = 1 \cdot b[0] = b[0]$, so the b-vector and evaluation are trivially consistent.

#### 4.9.5. Compute Combined Row

The transcript squeezes `num_rows` i128 challenge coefficients $s_0, \ldots, s_{\mathrm{num\_rows}-1}$. For each polynomial $m$, the row-combination is:

$$w_m[\mathrm{col}] = \sum_{j=0}^{\mathrm{num\_rows}-1} s_j \cdot e_m^{\mathrm{comb}}[\mathrm{row\_len} \cdot j + \mathrm{col}], \quad \mathrm{col} \in [\mathrm{row\_len}]$$

computed via the `combine_rows!` macro (parallel across rows, using checked 384-bit arithmetic with overflow detection when `CHECK_FOR_OVERFLOW` is set). The 20 per-polynomial combined rows are then summed element-wise:

$$w[\mathrm{col}] = \sum_{m=0}^{19} w_m[\mathrm{col}], \quad \mathrm{col} \in [\mathrm{row\_len}]$$

**Without `full-fold`:** `num_rows` = 512, `row_len` = 4. The combined row has 4 message-space entries and 2048 entries in the encoding domain.

**With `full-fold`:** `num_rows` = 1, `row_len` = 16384. Only 1 challenge coefficient is squeezed ($s_0$), so the combined row is just $s_0 \cdot \mathrm{codeword}$.

> **Correction on row length:** The above describes the *message-space* view. In the implementation, `row_len = pp.linear_code.row_len()`, and the combined row has that many `Int<6>` entries. The prove step operates on the *encoded* codeword matrices (which have 4× as many columns per row). The combined row is computed over the message-length portion.

**Transcript:** The combined row is written to the proof stream (`write_const_many`). Each `Int<6>` element contributes 48 bytes (384 bits).

**Proof contribution:** Without `full-fold`: $2048 \times 48 = 98\,304$ bytes. With `full-fold`: $16384 \times 48 = 786\,432$ bytes.

#### 4.9.6. Grinding (Proof-of-Work)

The prover performs a proof-of-work search with `GRINDING_BITS` = 16:

1. **Snapshot FS state:** Squeeze a 32-byte seed from the transcript (commits to all data absorbed so far: Merkle root, field config, $b$ vector, combined row).
2. **Nonce search:** Find the smallest 64-bit nonce $n$ such that `blake3(seed ‖ n)` (where $n$ is little-endian 8 bytes, concatenated into a 40-byte buffer) has $\geq 16$ leading zero bits. The leading-zeros test reads the first 8 bytes of the Blake3 output as a big-endian u64 and uses hardware `lzcnt`.
3. **Parallelism:** With the `parallel` feature, the search is parallelised across $T$ rayon threads using **interleaved assignment**: thread $t \in [T]$ checks nonces $t, t+T, t+2T, \ldots$. An `AtomicU64` upper bound (initialised to `u64::MAX`) is shared across threads; when any thread finds a valid nonce, it calls `fetch_min` to update the bound. All threads terminate once their current nonce exceeds the bound. This guarantees the globally-smallest valid nonce is always found, producing **identical proofs regardless of parallelism** (deterministic output).
4. **Expected cost:** $2^{16} = 65\,536$ Blake3 evaluations (sequential); $\approx 2^{16} / T$ wall-clock (parallel).

**Transcript:** The 8-byte nonce is written to the proof stream and absorbed into the FS transcript (so subsequent challenges depend on it).

**Proof contribution:** 8 bytes.

#### 4.9.7. Column Openings (Proximity Queries)

After grinding, the transcript squeezes 118 pseudo-random column indices, each in $[0, \mathrm{cw\_len})$ (the IPRS codeword length: 8192 without `full-fold`, 65536 with `full-fold`), via `squeeze_challenge_idx`.

For each column index $j$:

1. **Column values:** For each of the 20 polynomial codeword matrices (each `num_rows` $\times$ `cw_len` in `DensePolynomial<i64, Q>` elements), extract the $j$-th column (`num_rows` entries). Write all $20 \times \mathrm{num\_rows}$ codeword elements to the proof stream.

   Without `full-fold`: $20 \times 512 = 10\,240$ entries, each `DensePolynomial<i64, 8>` = 64 bytes → $655\,360$ bytes per opening.

   With `full-fold`: $20 \times 1 = 20$ entries, each `DensePolynomial<i64, 1>` = 8 bytes → $160$ bytes per opening.

2. **Merkle proof:** Generate a Merkle authentication path for leaf $j$ in the shared Merkle tree.

**Merkle tree structure (from commit phase):**
- **Leaf $j$:** `blake3(poly₁_row₀[j] ‖ poly₁_row₁[j] ‖ ⋯ ‖ poly₂₀_row_{num_rows-1}[j])` — all 20 × `num_rows` codeword entries in column $j$ are serialised and hashed.
- **Internal nodes:** Binary tree using `blake3::hazmat::merge_subtrees_non_root` for non-root pairs and `merge_subtrees_root` for the final pair.
- **Root:** 32 bytes (the commitment sent to the verifier).

**Proof contribution per opening:**
- Without `full-fold`: $10\,240 \times 64 + 13 \times 32 + 16 = 656\,192$ bytes (column data + sibling hashes + leaf metadata).
- With `full-fold`: $20 \times 8 + 16 \times 32 + 16 = 672+16 = 688$ bytes per opening (much smaller since `num_rows` = 1 and `BinaryPoly<1>` entries are 8 bytes each).

**Total for 118 openings:** Without `full-fold`: $118 \times 656\,192 \approx 77.4$ MB. With `full-fold`: $118 \times 688 \approx 81$ KB (dramatically smaller due to single-row codewords).

#### 4.9.8. Verifier's Consistency Checks (for reference)

The verifier performs three checks against the proof data:

1. **Evaluation consistency:** $\langle q_0, b \rangle = \mathrm{eval}$.
2. **Eval-proximity link:** $\langle w, q_1 \rangle_{\mathbb{F}_p} = \langle s, b \rangle_{\mathbb{F}_p}$ — links the combined row to the $b$-vector via the row-challenge coefficients.
3. **Proximity check (per opening):** For each of the 118 column indices $j$:
   - Verify the Merkle proof against the commitment root.
   - Encode the combined row at position $j$ using the IPRS encoding matrix (via `encode_wide_at_positions` — evaluates only at the opened positions, not full codeword).
   - Verify: $\mathrm{IPRS}(w)[j] = \sum_m \sum_i s_i \cdot \langle \alpha_m, \mathrm{col}_j^{(m)}[i] \rangle$ — the encoded combined-row entry matches the alpha-weighted, $s$-weighted column values read from the proof.

**Output:** Serialised PCS proof bytes (concatenation of $b$, combined row, grinding nonce, and 118 column openings) + the evaluation value $\mathrm{eval} \in \mathbb{F}_p$.

### 4.10. Fiat-Shamir Transcript Flow

The PIOP transcript operations, in exact order:

```
 PIOP Transcript
 ───────────────────────────────────────────────────────
 [Absorb]  Merkle root (32 bytes)
 [Squeeze] Prime p (192-bit, via Miller-Rabin loop)
 [Squeeze] IC evaluation point r = (r₁, ..., r₉)        §4.3
 [Absorb]  IC proof values v₁, ..., v₁₂  (12 Fp[X])     §4.3
 [Absorb]  QX IC proof values v'₁, ..., v'₇  (7 Fp[X])  §4.4
 ─ With boundary ─
 [Absorb]  Boundary IC values v₁, ..., v₁₃ (13 Fp[X])   §4.4c
 ───────────────────────────────────────────────────────
 [Squeeze] Projecting element α ∈ Fp                     §4.5

 ─ Without boundary (default): direct evaluation path ─
 [Absorb]  up_evals (38 Fp elements)                     §4.5
 [Absorb]  down_evals (10 Fp elements)                   §4.5

 ─ With boundary: CPR path ─
 [Squeeze] Bp CPR folding challenge                      §4.5b
 [Squeeze] Boundary CPR folding challenge                §4.5b
   ⋮       (9 rounds multi-degree sumcheck:              §4.5b
            absorb 2 round polys, squeeze 1 challenge)
 [Absorb]  Bp CPR up_evals + down_evals                  §4.5b
 [Absorb]  Boundary CPR up_evals + down_evals            §4.5b
 ───────────────────────────────────────────────────────

 [Absorb]  Aggregated multiplicities (per-lookup)        §4.6
 [Squeeze] Lookup challenges β, γ                        §4.6
   ⋮       (GKR internal absorbs/squeezes)               §4.6

 ─ Without boundary: shift sumcheck ─
 [Squeeze] Shift batching coefficients ξ₁, ..., ξ₁₀     §4.7
   ⋮       (9 sumcheck rounds: absorb pᵢ, squeeze sᵢ)   §4.7
 [Absorb]  v_finals (6 private Fp elements)              §4.7

 ─ With boundary: shift sumcheck skipped ─

 [Absorb]  Folding round 1: c₁,c₂ for 20 columns        §4.8
 [Squeeze] Folding challenge γ₁                          §4.8
 [Absorb]  Folding round 2: c₃,c₄ for 20 columns        §4.8
 [Squeeze] Folding challenge γ₂                          §4.8
 ─ With full-fold, 3 additional rounds ─
 [Absorb]  Folding round 3: c₅,c₆ for 20 columns        §4.8
 [Squeeze] Folding challenge γ₃                          §4.8
 [Absorb]  Folding round 4: c₇,c₈ for 20 columns        §4.8
 [Squeeze] Folding challenge γ₄                          §4.8
 [Absorb]  Folding round 5: c₉,c₁₀ for 20 columns       §4.8
 [Squeeze] Folding challenge γ₅                          §4.8
 ───────────────────────────────────────────────────────

 PCS Transcript (separate instance, seeded from Merkle root)
 ───────────────────────────────────────────────────────
 [Input]   Evaluation point r₂ (11 Fp elements, from caller)
 [Squeeze] Per-polynomial alpha challenges
 [Absorb]  b vector (inner products)
 [Squeeze] Row-challenge coefficients sⱼ
 [Absorb]  Combined row
 [Squeeze] Grinding nonce verification
 [Squeeze] 118 column-opening indices
 ───────────────────────────────────────────────────────
```

---

## 5. E2E Verifier Pipeline

The function `verify_classic_logup_4x_folded` (in `snark/src/pipeline.rs`) mirrors the prover. With `full-fold`, the function `verify_classic_logup_full_folded` is used instead.

> **Note on function name:** The prover function is named `prove_hybrid_gkr_logup_4x_folded` while the verifier is `verify_classic_logup_4x_folded`. The asymmetric naming is a historical artifact — the verifier was originally written for a "classic" lookup path and was not renamed when Hybrid GKR was added. Both functions implement the same Hybrid GKR LogUp protocol. The full-fold variants are `prove_hybrid_gkr_logup_full_folded` and `verify_classic_logup_full_folded`.

> **Benchmark caveat:** The verifier benchmark passes `TrivialIdeal` for both Bp and Qx ideal-over-field closures (`|_| TrivialIdeal`), meaning ideal membership checks (cyclotomic for Qx C1–C4, zero for Bp) are **skipped** during timing. The benchmark therefore measures transcript replay, column evaluation, lookup, shift sumcheck, folding, and PCS verification — but not the (cheap) ideal membership arithmetic.

### 5.1. Transcript Replay & IC Verify

1. **Initialise transcript** and absorb the Merkle root from the proof.
2. **Derive** the same $\mathbb{F}_p$ and evaluation point $\mathbf{r}$ as the prover.
3. **Deserialise** the Bp IC proof values $(v_1, \ldots, v_{12})$ from the proof bytes.
4. **Verify ideal membership** for each $v_k \in \mathcal{I}_k$. With `no-f2x`, all 12 Bp constraints target the **zero ideal**, so the verifier simply checks each polynomial is identically zero. (The proof is trivially empty — 0 bytes.)
5. **Absorb** IC values into the transcript (same order as prover).
6. **Deserialise and verify** the Qx IC proof (7 constraints):
   - **Cyclotomic** $(X^{32}-1)$ for C1–C4: fold coefficients modulo 32 ($X^{32} \equiv 1$) and check all vanish.
   - **Trivial ideal** for C7–C9: membership is automatic (nothing to check).
   - **With `true-ideal`:** C7–C9 use the **degree-one ideal** $(X-2)$ instead: evaluate $v'_k(2)$ in $\mathbb{F}_p$ and check $= 0$.
7. **With `boundary`:** Deserialise and **absorb** the boundary IC proof (13 values). The verifier calls `verify_at_point_absorb_only` — it absorbs the boundary IC values into the transcript to match the prover's transcript state but defers the zero-ideal membership check (the CPR sumcheck will enforce correctness). This is an **absorb-only** verification step: the IC values are tied to the committed trace through the subsequent CPR, and zero-ideal membership is trivially enforced once the CPR confirms the column evaluations.

**Output:** The evaluation point $\mathbf{r}$ for subsequent steps.

### 5.2. Column Evaluation Absorb & Lookup Verify

> **No CPR sumcheck on the verifier side either.** Since the E2E prover sends column
> evaluations directly (no CPR sumcheck proof), the verifier simply deserialises them,
> absorbs them into the transcript, and proceeds to lookup verification.
>
> **With `boundary`:** The verifier instead replays the multi-degree CPR sumcheck
> (2 groups: Bp CPR + boundary CPR), verifying the round polynomials and deriving
> the shared challenges. After the sumcheck, the verifier finalises both CPR groups:
> reconstructing public column evaluations, merging with private `up_evals`/`down_evals`
> from the proof, and verifying CPR consistency. The boundary CPR verifier projects
> the boundary scalars to field, builds a verifier pre-sumcheck state for
> `Sha256UairBoundaryNoF2x`, and calls `finalize_verifier` with the
> boundary `up_evals`/`down_evals`. The lookup verification then proceeds at the
> CPR sumcheck point $\mathbf{s}$ (not the IC point $\mathbf{r}$).

1. **Deserialise** the private `up_evals` and `down_evals` from the proof.
2. **Draw** the projecting element $\alpha$ from the transcript (same position as the prover — after IC absorptions).
3. **Reconstruct** full `up_evals` by inserting public column evaluations: the verifier projects each known public `BinaryPoly<32>` column at $\alpha$ and evaluates the resulting $\mathbb{F}_p$-valued MLE at $\mathbf{r}$. These are inserted at the public column indices to recover all 38 `up_evals`.
4. **Absorb** full `up_evals` and `down_evals` into the transcript.

5. **Lookup verification** (Hybrid GKR path):
   - Absorb aggregated multiplicities, derive challenges $\beta, \gamma$.
   - **Verify 2 GKR layers** (witness side): check layer proofs (algebraic identity + sumcheck) and MLE consistency of intermediate values.
   - **Verify 8 GKR layers** (table side): standard layer-by-layer verification.
   - **Cross-check** roots: $\sum_\ell \gamma^\ell P_w^{(\ell)} / Q_w^{(\ell)} = P_t / Q_t$.
   - **Verify multiplicity sums:** $\sum_j m_{\mathrm{agg}}^{(\ell)}[j] = K \cdot N$ for each lookup column.

### 5.3. Shift Sumcheck Verify

> **With `boundary`:** This step is **skipped** — shift claims are resolved within the
> CPR sumcheck.

1. **Reconstruct** shift claims from `down_evals` and UAIR shift specs.
2. **Replay** the $n = 9$ rounds of the sumcheck: for each round, absorb the prover's polynomial, check $p(0) + p(1) = \mathrm{claim}$, and draw the challenge.
3. **Compute** shift predicates $S_{c_i}(\mathbf{s}, \mathbf{r})$ at the challenge point.
4. **Check final identity:**
   $$\mathrm{claim} = \sum_i \xi_i \cdot S_{c_i}(\mathbf{s}, \mathbf{r}) \cdot v_i^{\mathrm{final}}$$
5. For **public shifts**: the verifier computes $v_i^{\mathrm{final}}$ by evaluating the known column MLE at $\mathbf{s}$.

### 5.4. Folding Verify (2-Round / 5-Round)

**Without `full-fold` (2 rounds):**

**Round 1** ($D = 32 \to 16$):
1. Deserialise $(c_{1,j}, c_{2,j})$ for each column.
2. **Check consistency:** $c_{1,j} + \alpha^{16} \cdot c_{2,j} = e_j$ (the PIOP evaluation from `up_evals`, corrected for zero-padding if needed).
3. Absorb into transcript, draw $\gamma_1$.
4. Compute new evaluations: $d_{1,j} = (1-\gamma_1)c_{1,j} + \gamma_1 c_{2,j}$.

**Round 2** ($16 \to 8$): Identical structure with $(c_{3,j}, c_{4,j})$.

**Output:** New evaluation point $\mathbf{r}_2 = (\mathbf{r} \| \gamma_1 \| \gamma_2)$ and claimed evaluations $d_{2,j}$ for the `BinaryPoly<8>` columns.

**With `full-fold` (5 rounds):**

The verifier runs an $N$-round loop (where $N = 5$), parameterised by a runtime array `fold_half_sizes = [16, 8, 4, 2, 1]`. For each round $i$:

1. Deserialise $(c_{1,j}^{(i)}, c_{2,j}^{(i)})$ for each column from `folding_rounds[i]`.
2. Compute $\alpha^{\mathrm{half\_sizes}[i]}$ via `compute_alpha_power`.
3. **Check consistency:** $c_{1,j}^{(i)} + \alpha^{\mathrm{half\_sizes}[i]} \cdot c_{2,j}^{(i)} = d_{i-1,j}$ (where $d_{0,j}$ = `up_evals`).
4. **Call** `fold_claims_verify` which absorbs the values, draws $\gamma_i$, and computes $d_{i,j} = (1-\gamma_i)c_{1,j}^{(i)} + \gamma_i c_{2,j}^{(i)}$.

**Output:** New evaluation point $\mathbf{r}_5 = (\mathbf{r} \| \gamma_1 \| \cdots \| \gamma_5)$ (14 variables) and claimed evaluations $d_{5,j}$ for the `BinaryPoly<1>` columns.

> **Implementation note:** The full-fold verifier avoids needing const generics for all intermediate `BinaryPoly` sizes by using a runtime `fold_half_sizes` array. The `compute_alpha_power` function computes $\alpha^{h}$ for any runtime $h$, so the verifier only needs `const D` and `const FINAL` as const generics.

### 5.4b. Ring-Eval Bridge Verify

**Purpose:** Verify that the prover's ring-valued MLE evaluations are consistent with the folding output $d_{\mathrm{last},j}$.

**Without `full-fold`:**
1. **Deserialise** the $20 \times 8$ ring-eval field elements $v_{j,k}$ from `ring_evals_bytes`.
2. **$\alpha$-projection check:** For each committed column $j \in [20]$:
   $$\sum_{k=0}^{7} \alpha^k \cdot v_{j,k} \stackrel{?}{=} d_{2,j}$$
   where $\alpha$ is the PIOP projecting element and $d_{2,j}$ is the folding Round 2 output.

**With `full-fold`:**
1. **Deserialise** the $20 \times 1$ ring-eval field elements from `ring_evals_bytes`.
2. **$\alpha$-projection check:** For each committed column $j \in [20]$:
   $$\alpha^0 \cdot v_{j,0} = v_{j,0} \stackrel{?}{=} d_{5,j}$$
   Since $Q = 1$, this reduces to a direct equality check: the single ring-eval must equal the folding Round 5 output.

If any column fails, the proof is **rejected** (verification returns `Err`).

**Soundness:** Without `full-fold`, this is a Schwartz-Zippel check on the degree-7 polynomial $P_j(X) = \sum_k v_{j,k} X^k - \pi_X(\widetilde{\mathrm{col}}_j^{(8)}(\mathbf{r}_2))$ evaluated at the random point $\alpha$. If even one $v_{j,k}$ differs from the true coefficient-MLE evaluation, the check fails with probability $\geq 1 - 7/p > 1 - 2^{-185}$. With `full-fold` ($Q = 1$), the check is exact (degree 0), so the soundness error from this step is zero.

### 5.5. PCS Verify

1. **Replay** the PCS Fiat-Shamir transcript (using the same transcript state as after prime derivation).
2. **Tensorise** the evaluation point ($\mathbf{r}_2$ without `full-fold`, $\mathbf{r}_5$ with `full-fold`).
3. **Read $b$** from the proof, check $\langle q_0, b \rangle = \mathrm{eval}$.
4. **Read combined row**, check $\langle \mathrm{combined\_row}, q_1 \rangle = \langle s, b \rangle$.
5. **Verify grinding** nonce.
6. **Proximity check** (core soundness):
   - For each of 118 random column indices:
     - Read column values + Merkle proof
     - **Verify Merkle proof** against commitment root
     - **Encode** the combined row at this position using the IPRS encoding matrix
     - **Check** that the encoded value matches the inner product of column values with alpha-challenge vectors

If all checks pass, the proof is **accepted**.

---

## Appendix A: UAIR Column Layout

| Index | Name | Ring | Role | PCS? | Public? |
|-------|------|------|------|------|---------|
| 0 | `a_hat` | $\mathcal{R}_{32}$ | Working variable $a_t$ | ✓ | |
| 1 | `e_hat` | $\mathcal{R}_{32}$ | Working variable $e_t$ | ✓ | |
| 2 | `W_hat` | $\mathcal{R}_{32}$ | Message schedule $W_t$ | | ✓ |
| 3 | `Sigma0_hat` | $\mathcal{R}_{32}$ | $\Sigma_0(a) = \mathrm{ROTR}^2 \oplus \mathrm{ROTR}^{13} \oplus \mathrm{ROTR}^{22}$ | ✓ | |
| 4 | `Sigma1_hat` | $\mathcal{R}_{32}$ | $\Sigma_1(e) = \mathrm{ROTR}^6 \oplus \mathrm{ROTR}^{11} \oplus \mathrm{ROTR}^{25}$ | ✓ | |
| 5 | `Maj_hat` | $\mathcal{R}_{32}$ | $\mathrm{Maj}(a,b,c) = (a \wedge b) \oplus (a \wedge c) \oplus (b \wedge c)$ | ✓ | |
| 6 | `ch_ef_hat` | $\mathcal{R}_{32}$ | $e \wedge f$ (first term of Ch) | ✓ | |
| 7 | `ch_neg_eg_hat` | $\mathcal{R}_{32}$ | $(\neg e) \wedge g$ (second term of Ch) | ✓ | |
| 8 | `sigma0_w_hat` | $\mathcal{R}_{32}$ | $\sigma_0(W_{t-15})$ (schedule hash) | ✓ | |
| 9 | `sigma1_w_hat` | $\mathcal{R}_{32}$ | $\sigma_1(W_{t-2})$ (schedule hash) | ✓ | |
| 10 | `S0` | $\mathcal{R}_{32}$ | Shift quotient for $\sigma_0$: $W_{t-15} \gg 3$ | | ✓ |
| 11 | `S1` | $\mathcal{R}_{32}$ | Shift quotient for $\sigma_1$: $W_{t-2} \gg 10$ | | ✓ |
| 12 | `R0` | $\mathcal{R}_{32}$ | Shift remainder for $\sigma_0$: $W_{t-15} \bmod X^3$ | | ✓ |
| 13 | `R1` | $\mathcal{R}_{32}$ | Shift remainder for $\sigma_1$: $W_{t-2} \bmod X^{10}$ | | ✓ |
| 14 | `D_hat` | $\mathcal{R}_{32}$ | $d_t = a_{t-3}$ (lookback) | | (shift src) |
| 15 | `H_hat` | $\mathcal{R}_{32}$ | $h_t = e_{t-3}$ (lookback) | | (shift src) |
| 16 | `W_tm2` | $\mathcal{R}_{32}$ | $W_{t-2}$ (lookback) | | ✓ |
| 17 | `W_tm7` | $\mathcal{R}_{32}$ | $W_{t-7}$ (lookback) | | ✓ |
| 18 | `W_tm15` | $\mathcal{R}_{32}$ | $W_{t-15}$ (lookback) | | ✓ |
| 19 | `W_tm16` | $\mathcal{R}_{32}$ | $W_{t-16}$ (lookback) | | ✓ |
| 20 | `K_hat` | $\mathcal{R}_{32}$ | Round constant $K_t$ | | ✓ |
| 21 | `a_tm1` | $\mathcal{R}_{32}$ | $b_t = a_{t-1}$ (for Maj) | | (shift src) |
| 22 | `a_tm2` | $\mathcal{R}_{32}$ | $c_t = a_{t-2}$ (for Maj) | | (shift src) |
| 23 | `e_tm1` | $\mathcal{R}_{32}$ | $f_t = e_{t-1}$ (for Ch) | | (shift src) |
| 24 | `e_tm2` | $\mathcal{R}_{32}$ | $g_t = e_{t-2}$ (for Ch) | | (shift src) |
| 25 | `sel_round` | $\mathcal{R}_{32}$ | Active-round selector | | ✓ |
| 26 | `sel_sched` | $\mathcal{R}_{32}$ | Schedule-active selector | | ✓ |
| 27 | `mu_a` | $\mathbb{Z}$ | Carry for $a$-update ($\in \{0,\ldots,6\}$) | ✓ | |
| 28 | `mu_e` | $\mathbb{Z}$ | Carry for $e$-update ($\in \{0,\ldots,5\}$) | ✓ | |
| 29 | `mu_W` | $\mathbb{Z}$ | Carry for $W$-schedule ($\in \{0,\ldots,3\}$) | ✓ | |
| **27–34** | | | **μ quotient columns (no-f2x only)** | | |
| 27 | `mu_c1_lo` | $\mathcal{R}_{32}$ | $\Sigma_0$ rotation quotient, low bit | ✓ | |
| 28 | `mu_c1_hi` | $\mathcal{R}_{32}$ | $\Sigma_0$ rotation quotient, high bit | ✓ | |
| 29 | `mu_c2_lo` | $\mathcal{R}_{32}$ | $\Sigma_1$ rotation quotient, low bit | ✓ | |
| 30 | `mu_c2_hi` | $\mathcal{R}_{32}$ | $\Sigma_1$ rotation quotient, high bit | ✓ | |
| 31 | `mu_c3_lo` | $\mathcal{R}_{32}$ | $\sigma_0$ rotation+shift quotient, low bit | ✓ | |
| 32 | `mu_c3_hi` | $\mathcal{R}_{32}$ | $\sigma_0$ rotation+shift quotient, high bit | ✓ | |
| 33 | `mu_c4_lo` | $\mathcal{R}_{32}$ | $\sigma_1$ rotation+shift quotient, low bit | ✓ | |
| 34 | `mu_c4_hi` | $\mathcal{R}_{32}$ | $\sigma_1$ rotation+shift quotient, high bit | ✓ | |
| **35–40** | | | **Correction columns (true-ideal only)** | | |
| 35 | `corr_add_c7` | $\mathcal{R}_{32}$ | Additive correction for C7 ($a$-update) | | ✓ |
| 36 | `corr_add_c8` | $\mathcal{R}_{32}$ | Additive correction for C8 ($e$-update) | | ✓ |
| 37 | `corr_add_c9` | $\mathcal{R}_{32}$ | Additive correction for C9 ($W$-schedule) | | ✓ |
| 38 | `corr_sub_c7` | $\mathcal{R}_{32}$ | Subtractive correction for C7 | | ✓ |
| 39 | `corr_sub_c8` | $\mathcal{R}_{32}$ | Subtractive correction for C8 | | ✓ |
| 40 | `corr_sub_c9` | $\mathcal{R}_{32}$ | Subtractive correction for C9 | | ✓ |
| **33–39** | | | **Boundary columns (`boundary` only, without `true-ideal`)** | | |
| **39–45** | | | **Boundary columns (`boundary` + `true-ideal`)** | | |
| +0 | `sel_init` | $\mathcal{R}_{32}$ | Binary selector: 1 at row 0 | | ✓ |
| +1 | `sel_final_any` | $\mathcal{R}_{32}$ | Binary selector: 1 at last 4 active rows | | ✓ |
| +2 | `sel_msg` | $\mathcal{R}_{32}$ | Binary selector: 1 at rows 0–15 | | ✓ |
| +3 | `sel_zero` | $\mathcal{R}_{32}$ | Binary selector: 1 at rows $N-3, N-2, N-1$ | | ✓ |
| +4 | `out_a` | $\mathcal{R}_{32}$ | Expected $\hat{a}$ at final rows | | ✓ |
| +5 | `out_e` | $\mathcal{R}_{32}$ | Expected $\hat{e}$ at final rows | | ✓ |
| +6 | `msg_expected` | $\mathcal{R}_{32}$ | Expected $\hat{W}$ at rows 0–15 | | ✓ |

> **Note on column indices 27–29:** With `no-f2x`, the integer carry columns (`mu_a`, `mu_e`, `mu_W`) retain their logical indices as integer columns 0–2 but the bitpoly indices 27–34 are occupied by the μ quotient columns. The total is 35 bitpoly + 3 integer = 38 columns. With `true-ideal`, 6 correction columns (35–40) are appended, giving 41 bitpoly + 3 integer = 44 columns. With `boundary`, 7 boundary columns are appended after all existing bitpoly columns: indices 33–39 (without `true-ideal`, total 42 bitpoly + 3 integer = 45) or 41–47 (with `true-ideal`, total 48 bitpoly + 3 integer = 51).

---

## Appendix B: Constraint Catalogue

### B.1. BinaryPoly UAIR (`Sha256UairBpNoF2x`) — 12 constraints, degree 1

With `no-f2x`, the rotation constraints C1–C4 are moved to the Qx UAIR. Only C5–C16 remain (all `assert_zero`):

**Shift decomposition (zero ideal):**

- **C5**: $\hat{W}_{t-15} - R_0 - X^3 \cdot S_0 = 0$
- **C6**: $\hat{W}_{t-2} - R_1 - X^{10} \cdot S_1 = 0$

**Cross-row linking (zero ideal, via shifted MLEs):**

- **C7–C16**: $\mathrm{shifted}[\mathrm{src}][t + \delta] - \mathrm{src}[t] = 0$ for each of 10 shift specifications.

> **Without `no-f2x`:** `Sha256Uair` has 16 constraints (C1–C4 rotation with cyclotomic ideal, C5–C6 shift decomposition, C7–C16 linking).

### B.2. Q[X] UAIR (`Sha256UairQxNoF2x`) — 7 constraints

**Rotation constraints with μ correction (cyclotomic ideal $(X^{32}-1)$ over $\mathbb{Z}[X]$):**

- **C1** ($\Sigma_0$): $\hat{a} \cdot (X^{30}+X^{19}+X^{10}) - \hat{\Sigma}_0 - 2 \cdot \mu_{1,\text{lo}} - 4 \cdot \mu_{1,\text{hi}} \equiv 0 \pmod{X^{32}-1}$
- **C2** ($\Sigma_1$): $\hat{e} \cdot (X^{26}+X^{21}+X^{7}) - \hat{\Sigma}_1 - 2 \cdot \mu_{2,\text{lo}} - 4 \cdot \mu_{2,\text{hi}} \equiv 0 \pmod{X^{32}-1}$
- **C3** ($\sigma_0$): $\hat{W}_{t-15} \cdot (X^{25}+X^{14}) + S_0 - \hat{\sigma}_{0,w} - 2 \cdot \mu_{3,\text{lo}} - 4 \cdot \mu_{3,\text{hi}} \equiv 0 \pmod{X^{32}-1}$
- **C4** ($\sigma_1$): $\hat{W}_{t-2} \cdot (X^{15}+X^{13}) + S_1 - \hat{\sigma}_{1,w} - 2 \cdot \mu_{4,\text{lo}} - 4 \cdot \mu_{4,\text{hi}} \equiv 0 \pmod{X^{32}-1}$

**Carry propagation (trivial ideal / DegreeOne ideal):**

- **C7**: $\hat{a}[t+1] - \hat{h} - \hat{\Sigma}_1 - \widehat{\mathrm{ch\_ef}} - \widehat{\mathrm{ch\_neg\_eg}} - \hat{K} - \hat{W} - \hat{\Sigma}_0 - \widehat{\mathrm{Maj}} + \mu_a X^{32} = 0$
- **C8**: $\hat{e}[t+1] - \hat{d} - \hat{h} - \hat{\Sigma}_1 - \widehat{\mathrm{ch\_ef}} - \widehat{\mathrm{ch\_neg\_eg}} - \hat{K} - \hat{W} + \mu_e X^{32} = 0$
- **C9**: $\hat{W}_t - \hat{W}_{t-16} - \hat{\sigma}_0 - \hat{W}_{t-7} - \hat{\sigma}_1 + \mu_W X^{32} = 0$

> **With `true-ideal`:** C7–C9 use the degree-one ideal $(X-2)$ with dual correction columns: $c_{\text{inner}} + \text{corr\_add} - \text{corr\_sub} \in (X-2)$.

> **Without `no-f2x`:** `Sha256UairQx` has only 3 constraints (C7–C9, all trivial ideal).

### B.3. Boundary UAIR (`Sha256UairBoundaryNoF2x`, **`boundary` only**) — 13 constraints, degree 2

All constraints use `assert_zero` with `ImpossibleIdeal` (zero-only).

**Initial state (B1–B8, active at row 0):**

| # | Column | Expected value |
|---|--------|---------------|
| B1 | `a_hat` | $H_0$ (`6a09e667`) |
| B2 | `e_hat` | $H_4$ (`510e527f`) |
| B3 | `D_hat` | $H_3$ (`a54ff53a`) |
| B4 | `H_hat` | $H_7$ (`5be0cd19`) |
| B5 | `a_tm1` | $H_1$ (`bb67ae85`) |
| B6 | `a_tm2` | $H_2$ (`3c6ef372`) |
| B7 | `e_tm1` | $H_5$ (`9b05688c`) |
| B8 | `e_tm2` | $H_6$ (`1f83d9ab`) |

Each: $\mathrm{sel\_init} \cdot (\mathrm{col} - H[i]) = 0$, degree 2.

**Final output (B9–B10, active at last 4 active rows):**

- **B9**: $\mathrm{sel\_final\_any} \cdot (\hat{a} - \mathrm{out\_a}) = 0$
- **B10**: $\mathrm{sel\_final\_any} \cdot (\hat{e} - \mathrm{out\_e}) = 0$

**Message input (B11, active at rows 0–15):**

- **B11**: $\mathrm{sel\_msg} \cdot (\hat{W} - \mathrm{msg\_expected}) = 0$

**Zero padding (B12–B13, active at rows $N-3, N-2, N-1$):**

- **B12**: $\mathrm{sel\_zero} \cdot \hat{a} = 0$
- **B13**: $\mathrm{sel\_zero} \cdot \hat{e} = 0$

**Shift spec:** 1 sentinel shift (`source_col: 0, shift_amount: 1`), not referenced by any constraint.

---

## Appendix C: Concrete Parameters

| Parameter | Value (no-f2x) | Value (no-f2x + true-ideal) | Value (standard) | Value (full-fold) |
|-----------|----------------|----------------------------|-------------------|-------------------|
| **Trace rows** | $N = 2^9 = 512$ (8 × 64 SHA-256 rounds) | same | same | same |
| **Total columns** | 38 (35 BinaryPoly<32> + 3 integer); with `boundary`: 45 (42 + 3) | 44 (41 BinaryPoly<32> + 3 integer); with `boundary`: 51 (48 + 3) | 30 (27 + 3) | same as no-f2x |
| **PCS-committed columns** | 20 (12 original + 8 μ) | 20 (same — correction columns are public) | 12 | 20 |
| **Public columns** | 12; with `boundary`: 19 (12 + 7 boundary) | 18 (12 + 6 correction); with `boundary`: 25 (18 + 7) | 12 | 12 |
| **PIOP field** | $\mathbb{F}_p$, $p$ ≈ $2^{192}$ random prime (192-bit Montgomery) | same | same | same |
| **Trace ring** | $\mathcal{R}_{32} = \mathbb{F}_2[X]/(X^{32}+1)$ | same | same | same |
| **PCS evaluation ring** | $\mathcal{R}_8 = \mathbb{F}_2[X]/(X^{8}+1)$ after 4× folding | same | same | $\mathcal{R}_1 = \mathbb{F}_2$ after 32× folding |
| **PCS code** | IPRS `PnttConfigF2_16R4B32<2>`: rate 1/4, input 2048, output 8192 | same | same | IPRS `PnttConfigF2_16R4B32<3>`: rate 1/4, input 16384, output 65536 |
| **PCS code field** | $\mathbb{F}_{65537}$ ($= 2^{16}+1$, Fermat prime) | same | same | same |
| **Column openings** | 118 | 118 | 118 | 118 |
| **Grinding bits** | 16 | 16 | 16 | 16 |
| **Grinding parallelism** | Interleaved nonce search across rayon threads (with `parallel` feature); sequential fallback otherwise. Wall-clock ≈ $2^{16}/T$ Blake3 hashes where $T$ = thread count. | same | same | same |
| **Lookup columns** | 13 (10 base + 3 affine virtual) | same | same | same |
| **Lookup decomposition** | 4 chunks of 8 bits, sub-table size 256 | same | same | same |
| **Hybrid GKR cutoff** | 2 layers | same | same | same |
| **Shift claims** | 10 (shift amounts in $\{1, 2, 3, 7, 15, 16\}$; Qx shifts handled internally by IC) | same | same | same |
| **Folding rounds** | 2 ($32 \to 16 \to 8$) | same | same | 5 ($32 \to 16 \to 8 \to 4 \to 2 \to 1$) |
| **IC constraints (Bp)** | 12, degree 1 (all zero ideal) | 16 (4 cyclotomic + 12 zero) | same as standard | same as no-f2x |
| **IC constraints (Q[X])** | 7 (4 cyclotomic + 3 trivial) | 3 (all trivial) | same as standard | same as no-f2x |
| **IC constraints (Boundary)** | — (unless `boundary` enabled: 13, degree 2, all zero ideal) | same | — | same as no-f2x |
| **Sumcheck degree (shift)** | 2 (3 evaluations per round) | same | same | same |
| **CPR sumcheck** | Skipped (direct eval) unless `boundary` enabled: multi-degree sumcheck with 2 groups (degree 1 + degree 4) | same | same | same |
| **Shift sumcheck** | 10 claims, $n = 9$ rounds; skipped when `boundary` enabled (CPR handles shifts internally) | same | same | same |
| **MLE-first optimization** | Enabled (max_degree = 1 for both UAIRs) | same | same | same |
| **Field element size** | 24 bytes (192-bit Montgomery representation, 3 × 64-bit limbs) | same | same | same |
| **PCS transcript** | Separate Fiat-Shamir instance seeded from the Merkle root. It derives the same prime $p$ as the PIOP transcript but generates its own internal challenges (alpha vectors, column-opening indices, grinding nonce). The evaluation point $\mathbf{r}_2$ is passed as an **explicit parameter** from the PIOP caller — the PCS never derives it. | same | same | same (evaluation point is $\mathbf{r}_5$) |
| **num_vars (PCS)** | 11 ($9 + 2$ folding) | same | same | 14 ($9 + 5$ folding) |
| **PCS matrix shape** | $512 \times 4$ (num_rows × row_len) | same | same | $1 \times 16384$ |

---

## Appendix D: Security Analysis

### Soundness Errors by Sub-protocol

| Sub-protocol | Soundness error | Notes |
|-------------|----------------|-------|
| **IC (Bp UAIR)** | $\leq n / |\mathbb{F}_p| \approx 9/2^{128}$ per constraint | Schwartz-Zippel on MLE evaluation at random $\mathbf{r}$; $n = 9$ variables. With no-f2x, all 12 constraints are zero-ideal — IC proof is trivially empty. |
| **IC (Q[X] UAIR)** | $\leq n / |\mathbb{F}_p|$ per constraint | Same MLE bound; with no-f2x, 4 cyclotomic + 3 trivial constraints (or 3 DegreeOne with `true-ideal`). |
| **IC (Boundary UAIR)** | $\leq n / |\mathbb{F}_p|$ per constraint | With `boundary`: 13 constraints, all zero ideal (`ImpossibleIdeal`). IC proof is trivially empty. Sparse evaluation (~22 active rows) does not affect soundness — the MLE identity holds at the random point $\mathbf{r}$. |
| **IC ideal membership** | $\leq 31 / |\mathbb{F}_p|$ per cyclotomic constraint; $\leq 1 / |\mathbb{F}_p|$ per DegreeOne constraint | After projection at random $\alpha$, a non-zero polynomial of degree $\leq 31$ is detected with probability $\geq 1 - 31/p$. With no-f2x, this applies to the 4 cyclotomic constraints in the Qx UAIR. With `true-ideal`, the 3 carry constraints use $(X-2)$: Schwartz-Zippel on a degree-1 factor gives $\leq 1/p$ per constraint. |
| **Lookup (LogUp)** | $\leq (K \cdot N) / |\mathbb{F}_p|$ per column | $K = 4$ chunks, $N = 512$; degree of the LogUp polynomial in $\beta$ |
| **Shift sumcheck** | $\leq 2n / |\mathbb{F}_p|$ | Degree-2 sumcheck, $n = 9$ rounds; standard sumcheck error. Skipped when `boundary` enabled. |
| **Shift batching** | $\leq 10 / |\mathbb{F}_p|$ | Random $\xi_i$ coefficients; Schwartz-Zippel over 10 Bp shift claims. Subsumed by CPR when `boundary` enabled. |
| **CPR sumcheck** | $\leq d \cdot n / |\mathbb{F}_p|$ per group | With `boundary`: multi-degree sumcheck over 2 groups (degree 1 + degree 4), $n = 9$ rounds. Standard sumcheck error $\leq d/|\mathbb{F}_p|$ per round per group. Without `boundary`: skipped. |
| **Folding** | $\leq 1 / |\mathbb{F}_p|$ per column per round | The folded evaluation $d = (1-\gamma) c_1 + \gamma c_2$ is degree 1 in $\gamma$; Schwartz-Zippel gives $1/|\mathbb{F}_p|$ per check. With 20 columns and 2 rounds, the union bound gives $\leq 40/|\mathbb{F}_p|$. With `full-fold` (5 rounds), the bound is $\leq 100/|\mathbb{F}_p|$ (still negligible). |
| **PCS proximity** | $\leq (1 - \delta)^{118} \cdot 2^{-16}$ | 118 column openings + 16 grinding bits; $\delta$ = relative minimum distance of the IPRS code |
| **Miller-Rabin** | $\leq 2^{-64}$ (heuristic) | Single base-2 test on 192-bit candidate after trial division |

### Overall Security

By a union bound over all sub-protocol invocations (12 + 7 IC constraints with no-f2x, 13 lookup columns, 1 shift sumcheck, 2 folding rounds (or 5 with `full-fold`), 1 PCS; with `boundary`: add 13 boundary IC constraints and 1 multi-degree CPR sumcheck, remove standalone shift sumcheck), the total soundness error is dominated by the PCS proximity term. The PIOP errors are all $O(\mathrm{poly}(N) / 2^{128})$, which is negligible.

The target security level is approximately **128 bits**, achieved through the combination of 118 column openings, 16 grinding bits, and the $\approx 2^{192}$-sized PIOP challenge field.

### Completeness

Completeness (honest execution leads to acceptance) follows from:
1. Correct witness generation produces a trace satisfying all UAIR constraints.
2. MLE linearity ensures IC proof values lie in their prescribed ideals.
3. The shift sumcheck round polynomials are correctly computed from the committed trace.
4. The folding identities $c_1 + \alpha^{D/2} \cdot c_2 = e$ hold algebraically by the split construction.
5. The PCS codewords are valid encodings (the IPRS code is applied to the actual committed data).

---
---

# Part II — Zinc+ SHA-256 8× + ECDSA Dual-Circuit Benchmark: Algorithmic Specification

> **Benchmark covered:**
> ```
> cargo bench --bench steps_sha256_8x_ecdsa_folded -p zinc-snark --features "parallel simd asm qx-constraints no-f2x"
> ```
>
> **Important: feature flags have no effect on this benchmark.** Unlike the SHA-256-only benchmark (Part I),
> this dual-circuit benchmark does **not** use `no-f2x` or `qx-constraints` feature gating.
> It unconditionally uses:
>
> | Circuit | UAIR type | Ring | Columns | Constraints |
> |---------|-----------|------|---------|-------------|
> | **SHA-256** | `Sha256Uair` (standard Bp UAIR) | `BinaryPoly<32>` | 30 (27 bitpoly + 3 integer) | 16 (C1–C4 cyclotomic + C5–C16 zero) |
> | **ECDSA** | `EcdsaUairInt` | `Int<4>` (256-bit) | 11 | 11 (all `assert_zero`) |
>
> The `no-f2x` feature flag activates the `sha256-uair::no_f2x` module but the benchmark never imports
> `Sha256UairBpNoF2x` or `Sha256UairQxNoF2x`. The `qx-constraints` flag is similarly unused — the
> dual-circuit pipeline function `prove_dual_circuit_hybrid_gkr_4x_folded` contains no Qx code path.
>
> This document provides a precise mathematical description of the algorithms
> executed in the dual-circuit E2E prover and E2E verifier benchmarks for a
> cryptographer audience.

---

## Table of Contents (Part II)

1. [Overview](#ii1-overview)
2. [ECDSA Algebraic Domains](#ii2-ecdsa-algebraic-domains)
3. [The ECDSA UAIR Trace](#ii3-the-ecdsa-uair-trace)
4. [Dual-Circuit E2E Prover Pipeline](#ii4-dual-circuit-e2e-prover-pipeline)
5. [Dual-Circuit E2E Verifier Pipeline](#ii5-dual-circuit-e2e-verifier-pipeline)
6. [Appendix E: ECDSA Column Layout](#appendix-e-ecdsa-column-layout)
7. [Appendix F: ECDSA Constraint Catalogue](#appendix-f-ecdsa-constraint-catalogue)
8. [Appendix G: Dual-Circuit Concrete Parameters](#appendix-g-dual-circuit-concrete-parameters)
9. [Appendix H: Dual-Circuit Security Analysis](#appendix-h-dual-circuit-security-analysis)

---

## II.1. Overview

The dual-circuit benchmark proves a **combined statement** consisting of:

1. **SHA-256**: 8 chained SHA-256 compressions (identical to Part I, but using the *standard* `Sha256Uair` — 30 columns, 16 constraints including 4 cyclotomic).
2. **ECDSA**: A secp256k1 Shamir multi-scalar multiplication over 257 steps, verifying an ECDSA signature.

The two circuits are **independent UAIRs** in different scalar rings (`BinaryPoly<32>` for SHA, `Int<4>` for ECDSA) unified through a shared Fiat-Shamir transcript and shared PIOP evaluation point. They share:

- A **single Keccak transcript** (both commitment roots absorbed sequentially).
- A **shared IC evaluation point** (one random $\mathbf{r} \in \mathbb{F}_p^n$ used for both Ideal Checks).
- A **shared projecting element** $\alpha$ (one random $\alpha \in \mathbb{F}_p$ for field projection of both circuits).
- A **single multi-degree sumcheck** (both CPR groups batched with shared verifier challenges).
- A **unified evaluation sumcheck** (all column evaluation claims from both circuits).

They have **separate**:

- **PCS commitments and proofs** (SHA uses 4×-folded `BinaryPoly<8>` over 192-bit PCS field; ECDSA uses unfolded `Int<4>` over 256-bit PCS field).
- **Ideal Check proofs** (SHA uses MLE-first path; ECDSA uses the general `prove_at_point` path).
- **Folding** (SHA only — 2-round protocol; ECDSA columns are not folded).
- **Lookup** (SHA only — Hybrid GKR batched lookup; ECDSA has no lookup requirements).

### Notation

All notation from Part I §1 carries over. Additional notation:

| Symbol | Meaning |
|--------|---------|
| $\mathbb{F}_q$ | secp256k1 base field, $q = 2^{256} - 2^{32} - 977$ |
| $\mathrm{Int}\langle 4 \rangle$ | 256-bit signed integer (4 × 64-bit limbs), representing elements of $\mathbb{F}_q$ |
| $(X, Y, Z)$ | Jacobian projective coordinates on secp256k1 |
| $G = (G_x, G_y)$ | secp256k1 generator point (affine) |
| $Q = (Q_x, Q_y)$ | Public key (affine); set to $G$ in the benchmark |
| $G+Q = (PGQ_x, PGQ_y)$ | Precomputed sum $G + Q$ (affine) |
| $b_1, b_2 \in \{0,1\}$ | Scalar bits of $u_1, u_2$ at each step |
| $R_{\mathrm{SIG}}$ | Expected affine $x$-coordinate of the result point |
| $\mathbb{F}_p$ | PIOP field (192-bit prime, same as Part I) |
| $\mathbb{F}_s$ | ECDSA PCS field (256-bit prime, $s \approx 2^{256}$) |

---

## II.2. ECDSA Algebraic Domains

### II.2.1. The secp256k1 Curve

The ECDSA circuit operates over secp256k1: $y^2 = x^3 + 7$ over $\mathbb{F}_q$ where $q = 2^{256} - 2^{32} - 977$. All trace column values and constraint arithmetic are performed in $\mathbb{F}_q$, represented as `Int<4>` (256-bit integers with modular reduction).

The Jacobian coordinate system represents a point $(x, y)$ in affine coordinates as $(X, Y, Z)$ where $x = X/Z^2$ and $y = Y/Z^3$.

### II.2.2. Scalar Ring `Int<4>`

`Int<4>` is a 256-bit signed integer type (4 × 64-bit limbs). Arithmetic is performed modulo $q$ (the secp256k1 base field prime). Unlike `BinaryPoly<32>`, which is a polynomial ring, `Int<4>` is a scalar (degree-0) type — each trace cell holds a single field element, not a polynomial.

When the PIOP projects `Int<4>` trace values to $\mathbb{F}_p$, each `Int<4>` value $v$ becomes a degree-0 `DynamicPolynomialF<F>` with a single coefficient $v \bmod p$. The projection $\pi_\alpha$ is trivially the identity on scalars (no polynomial evaluation needed).

### II.2.3. PIOP and PCS Fields

The dual-circuit pipeline uses **three different prime fields**:

| Field | Size | Role |
|-------|------|------|
| $\mathbb{F}_p$ (`MontyField<3>`) | 192-bit | Shared PIOP field for both circuits: IC, CPR sumcheck, eval sumcheck |
| $\mathbb{F}_{p_1}$ (`MontyField<3>`) | 192-bit | SHA PCS field (independent prime derived from SHA commitment root) |
| $\mathbb{F}_s$ (`MontyField<4>`) | 256-bit | ECDSA PCS field (independent prime derived from ECDSA commitment root) |

The PIOP operates entirely over $\mathbb{F}_p$. Each PCS creates its own transcript, absorbs only its own Merkle root, and derives its own independent prime modulus.

---

## II.3. The ECDSA UAIR Trace

### II.3.1. Trace Dimensions

The ECDSA trace has $N = 2^9 = 512$ rows and **11 columns** (all `Int<4>`). The logical computation spans 258 rows (rows 0–257):

- **Row 0**: Initialisation — set accumulator to the table point selected by $(b_1[0], b_2[0])$, then double-and-conditionally-add.
- **Rows 1–256**: Pure doubling (all $b_1 = b_2 = 0$ in the benchmark witness).
- **Row 257**: Final accumulator — boundary constraint B4 verifies $X/Z^2 = R_{\mathrm{SIG}}$.
- **Rows 258–511**: Zero padding.

### II.3.2. Column Layout

| Index | Name | Description | PCS? | Public? |
|-------|------|-------------|------|---------|
| 0 | `b1` | Scalar bit of $u_1$ | | ✓ |
| 1 | `b2` | Scalar bit of $u_2$ | | ✓ |
| 2 | `X` | Accumulator $x$-coordinate (Jacobian) | | (shift src) |
| 3 | `Y` | Accumulator $y$-coordinate (Jacobian) | | (shift src) |
| 4 | `Z` | Accumulator $z$-coordinate (Jacobian) | | (shift src) |
| 5 | `X_mid` | Doubled point $2P$ $x$-coordinate | ✓ | |
| 6 | `Y_mid` | Doubled point $2P$ $y$-coordinate | ✓ | |
| 7 | `Z_mid` | Doubled point $2P$ $z$-coordinate | ✓ | |
| 8 | `H` | Addition scratch: chord $x$-difference | ✓ | |
| 9 | `sel_init` | Boundary selector: $1$ at row $0$ | | ✓ |
| 10 | `sel_final` | Boundary selector: $1$ at row $257$ | | ✓ |

Column partition:
- **4 PCS-committed columns**: indices 5–8 (`X_mid`, `Y_mid`, `Z_mid`, `H`).
- **4 public columns**: indices 0, 1, 9, 10 (`b1`, `b2`, `sel_init`, `sel_final`).
- **3 shift-source-only columns**: indices 2, 3, 4 (`X`, `Y`, `Z`) — not public, not PCS-committed. Their evaluation claims are resolved by the unified eval sumcheck.

Thus: 4 committed + 4 public + 3 shift-source-only = 11 total.

### II.3.3. Shift Specifications

| Source column | Shift amount | Produces |
|---------------|-------------|----------|
| `X` (index 2) | 1 | $X[t+1]$ |
| `Y` (index 3) | 1 | $Y[t+1]$ |
| `Z` (index 4) | 1 | $Z[t+1]$ |

All 3 shifts are "genuine" shift claims (non-zero shift amount) that enter the unified evaluation sumcheck.

### II.3.4. Constraint System: `EcdsaUairInt` (11 constraints, all `assert_zero`)

All constraints use the trivial `ImpossibleIdeal` — only zero is in the ideal. The maximum constraint degree is **13** (dominated by C6).

**Shamir selector:** $s = b_1 + b_2 - b_1 b_2$ (logical OR: $s = 1$ iff any bit is set).

**Table point selection** (degree 2 in trace columns, since $G_x, G_y, Q_x, Q_y, PGQ_x, PGQ_y$ are degree-0 constants):
$$T_x = b_1(1-b_2) \cdot G_x + (1-b_1)b_2 \cdot Q_x + b_1 b_2 \cdot PGQ_x$$
$$T_y = b_1(1-b_2) \cdot G_y + (1-b_1)b_2 \cdot Q_y + b_1 b_2 \cdot PGQ_y$$

where $G_x, G_y, Q_x, Q_y, PGQ_x, PGQ_y$ are hardcoded secp256k1 constants embedded via `smul_c` (scalar multiply by constant).

**Doubling constraints (C1–C4):**

| # | Name | Expression | Degree |
|---|------|------------|--------|
| C1 | $Z_{\mathrm{mid}}$ | $Z_{\mathrm{mid}} - 2YZ = 0$ | 2 |
| C2 | $X_{\mathrm{mid}}$ | $X_{\mathrm{mid}} - 9X^4 + 8XY^2 = 0$ | 4 |
| C3 | $Y_{\mathrm{mid}}$ | $Y_{\mathrm{mid}} - 12X^3Y^2 + 3X^2 X_{\mathrm{mid}} + 8Y^4 = 0$ | 5 |
| C4 | $H$ | $H - T_x Z_{\mathrm{mid}}^2 + X_{\mathrm{mid}} = 0$ | 4 |

These compute the intermediate doubled point $2P = (X_{\mathrm{mid}}, Y_{\mathrm{mid}}, Z_{\mathrm{mid}})$ and the addition scratch $H$ using the standard Jacobian doubling formulas with curve parameter $a = 0$ (secp256k1).

**Transition constraints (C5–C7, gated by $1 - \mathrm{sel\_final}$):**

Let $R_a = T_y \cdot Z_{\mathrm{mid}}^3 - Y_{\mathrm{mid}}$ (inlined, degree 5 in columns).

| # | Name | Expression | Degree |
|---|------|------------|--------|
| C5 | $Z[t{+}1]$ | $(1-\mathrm{sel\_final}) \cdot \bigl(Z[t{+}1] - (1{-}s) Z_{\mathrm{mid}} - s \cdot Z_{\mathrm{mid}} H\bigr) = 0$ | 5 |
| C6 | $X[t{+}1]$ | $(1-\mathrm{sel\_final}) \cdot \bigl(X[t{+}1] - (1{-}s) X_{\mathrm{mid}} - s \cdot (R_a^2 - H^3 - 2 X_{\mathrm{mid}} H^2)\bigr) = 0$ | **13** |
| C7 | $Y[t{+}1]$ | $(1-\mathrm{sel\_final}) \cdot \bigl(Y[t{+}1] - (1{-}s) Y_{\mathrm{mid}} - s \cdot (R_a (X_{\mathrm{mid}} H^2 - X[t{+}1]) - Y_{\mathrm{mid}} H^3)\bigr) = 0$ | 11 |

These enforce the conditional point addition using Shamir's trick: when $s = 0$, the next point is $2P$ (pure doubling); when $s = 1$, the next point is $2P + T$ (double-and-add).

**Boundary constraints (B3a–B3c, B4):**

| # | Name | Expression | Degree |
|---|------|------------|--------|
| B3a | Init $X$ | $\mathrm{sel\_init} \cdot (X - T_x) = 0$ | 3 |
| B3b | Init $Y$ | $\mathrm{sel\_init} \cdot (Y - T_y) = 0$ | 3 |
| B3c | Init $Z$ | $\mathrm{sel\_init} \cdot (Z - 1) = 0$ | 2 |
| B4 | Final sig | $\mathrm{sel\_final} \cdot Z \cdot (X - R_{\mathrm{SIG}} \cdot Z^2) = 0$ | 4 |

B3a–B3c enforce the initial accumulator at row 0 equals the table point for $(b_1[0], b_2[0])$ in Jacobian form $(T_x, T_y, 1)$. B4 verifies the final affine $x$-coordinate $X/Z^2 = R_{\mathrm{SIG}}$, rewritten projectively to avoid inversion.

> **Note:** Booleanity of $b_1, b_2$ is **not** enforced algebraically — these are public columns checked directly by the verifier.

### II.3.5. Lookup Requirements

The ECDSA UAIR has **no lookup requirements** — all constraints are algebraic. Lookups apply only to the SHA-256 circuit.

---

## II.4. Dual-Circuit E2E Prover Pipeline

The prover function is `prove_dual_circuit_hybrid_gkr_4x_folded`. The pipeline has the same structure as Part I but with two circuits processed in a unified transcript.

### II.4.0. Witness Generation

Two independent witnesses are generated:

| Circuit | Function | Trace dimensions | Ring |
|---------|----------|-----------------|------|
| SHA-256 | `Sha256Uair::generate_witness(9, rng)` | 512 rows × 30 columns | `BinaryPoly<32>` |
| ECDSA | `EcdsaUairInt::generate_witness(9, rng)` | 512 rows × 11 columns | `Int<4>` |

The ECDSA witness generates a **valid secp256k1 trace**: initialises at $G$ (row 0, $b_1=1, b_2=0$), performs one double-and-add, then 256 pure doublings ($b_1 = b_2 = 0$), computing $3 \cdot 2^{256} \cdot G$. The final $R_{\mathrm{SIG}}$ is lazily computed via the full 257-step trace in $\mathbb{F}_q$ arithmetic, with affine recovery $X/Z^2$ via Fermat inversion.

### II.4.1. Step 0: Double-Split Columns (SHA only)

Identical to Part I §4.0. Only SHA `BinaryPoly<32>` columns are split:
1. `split_columns::<32, 16>`: 12 committed columns × 512 → 12 columns × 1024 in `BinaryPoly<16>`
2. `split_columns::<16, 8>`: 12 columns × 1024 → 12 columns × 2048 in `BinaryPoly<8>`

The ECDSA `Int<4>` columns are **not** split or folded — they enter PCS directly.

### II.4.2. Step 1: PCS Commit (both circuits)

Two independent PCS commitments:

| Circuit | Input | PCS code | Merkle root |
|---------|-------|----------|-------------|
| SHA-256 | 12 `BinaryPoly<8>` columns × 2048 | IPRS `PnttConfigF2_16R4B4<3>`: rate 1/4, input 2048, output 8192 | `commitment1.root` |
| ECDSA | 4 `Int<4>` columns × 512 | IPRS `PnttConfigF2_16R4B64<1>`: rate 1/4, input 512, output 2048 | `commitment2.root` |

Both roots are absorbed into the shared transcript in order: SHA root first, then ECDSA root.

### II.4.3. Step 2: Transcript Init & Field Setup

The shared Keccak transcript absorbs both Merkle roots and derives:
1. **PIOP field config**: A random 192-bit prime $p$ via `get_random_field_cfg` (same mechanism as Part I §4.2).
2. **Shared IC evaluation point**: $\mathbf{r} = (r_1, \ldots, r_9) \in \mathbb{F}_p^9$, squeezed from the transcript.

This single point $\mathbf{r}$ is used for both the SHA and ECDSA Ideal Checks.

### II.4.4. Step 2a: Ideal Check — SHA-256 (BinaryPoly UAIR)

The SHA IC uses `IdealCheckProtocol::prove_mle_first_at_point` — the MLE-first path optimised for `BinaryPoly<D>` traces. This is identical to Part I §4.3 but now with the **standard** `Sha256Uair` (16 constraints: 4 cyclotomic + 12 zero), since `no-f2x` is not active.

The IC proof contains 4 non-trivial cyclotomic check values (for C1–C4) and 12 zero values (for C5–C16).

### II.4.5. Step 2b: Ideal Check — ECDSA

The ECDSA IC uses `IdealCheckProtocol::prove_at_point` — the general path for non-BinaryPoly traces.

**Pre-processing:** Each `Int<4>` trace cell $v$ is wrapped as a degree-0 `DynamicPolynomialF<F>` with a single coefficient $v \bmod p$. The projected ECDSA trace is a matrix of $\mathbb{F}_p$ elements.

Since all 11 ECDSA constraints use `assert_zero` with `ImpossibleIdeal`, the IC simply evaluates each constraint polynomial at the shared point $\mathbf{r}$ using multilinear extensions. Because $\mathrm{max\_degree} > 1$ (degree 13), the IC uses the row-by-row evaluation path: for each row $\mathbf{w} \in \{0,1\}^9$, it computes the eq-weight $\mathrm{eq}(\mathbf{r}, \mathbf{w})$ and accumulates the weighted constraint values.

The IC proof for ECDSA is a vector of 11 field elements — all should be zero for a valid witness.

### II.4.6. Step 3a: Shared Projection

A single projecting element $\alpha \in \mathbb{F}_p$ is squeezed from the transcript. Both circuits' scalar types are projected to $\mathbb{F}_p$:

- **SHA**: Each `BinaryPoly<32>` coefficient vector $(c_0, \ldots, c_{31})$ is evaluated at $\alpha$: $\pi_\alpha(\hat{v}) = \sum_{i=0}^{31} c_i \alpha^i$.
- **ECDSA**: Each `Int<4>` value $v$ becomes $v \bmod p$ (trivial projection — scalars are degree 0).

### II.4.7. Step 3b: Combined Poly Resolver (Multi-Degree Sumcheck)

Both circuits' CPR groups are constructed independently, then batched into a **single multi-degree sumcheck**.

**CPR Group 1 (SHA-256):** Builds the shifted traces (10 shift specs), constructs the eq polynomial, draws a folding challenge, and produces a `(degree, mles, comb_fn)` tuple. SHA constraints have $\mathrm{max\_degree} = 1$.

**CPR Group 2 (ECDSA):** Same process for ECDSA's 3 shift specs. ECDSA constraints have $\mathrm{max\_degree} \leq 13$.

**Multi-degree sumcheck:** `MultiDegreeSumcheck::prove_as_subprotocol` runs $n = 9$ rounds with exactly 2 groups. In each round:
1. Each group computes its own round polynomial of degree $d_i$ (SHA: low degree, ECDSA: up to 13).
2. Round polynomials are absorbed into the transcript **sequentially** (group 0 first, then group 1).
3. A **single shared challenge** $r_i$ is squeezed and applied to both groups.

This produces a shared evaluation point $\mathbf{r}_{\mathrm{cpr}} \in \mathbb{F}_p^9$.

### II.4.8. Step 3c: Finalise CPR & Folding (SHA only)

**SHA CPR finalisation:** Extract `up_evals` (30 values) and `down_evals` (10 shifted column evaluations) at $\mathbf{r}_{\mathrm{cpr}}$.

**Two-round folding protocol** (SHA only, identical to Part I §4.8):
- Round 1: Fold 12 `BinaryPoly<16>` split columns at $\mathbf{r}_{\mathrm{cpr}}$, squeeze $\gamma_1$, extend point.
- Round 2: Fold 12 `BinaryPoly<8>` split columns, squeeze $\gamma_2$, extend point.
- Final PCS point = $(\mathbf{r}_{\mathrm{cpr}}, \gamma_1, \gamma_2) \in \mathbb{F}_p^{11}$.

ECDSA columns are **not folded** — PCS point = $\mathbf{r}_{\mathrm{cpr}} \in \mathbb{F}_p^9$.

### II.4.9. Step 3d: ECDSA CPR Finalisation

Extract ECDSA `up_evals` (11 values) and `down_evals` (3 shifted column evaluations) at $\mathbf{r}_{\mathrm{cpr}}$.

### II.4.10. Step 3e: Unified Evaluation Sumcheck

All column evaluation claims from both circuits are combined into a **single** `shift_sumcheck_prove`:

| Claim type | Count | Source |
|------------|-------|--------|
| SHA `up_evals` (eq claims, shift amount 0) | 30 | Circuit 1 non-shifted columns |
| ECDSA `up_evals` (eq claims, shift amount 0) | 11 | Circuit 2 non-shifted columns |
| ECDSA `down_evals` (genuine shifts) | 3 | Circuit 2 columns X, Y, Z (shift amount 1) |

Total: **44 claims** in the unified eval sumcheck ($n = 9$ rounds, degree 2).

> **SHA shift claims** (10 specs) are handled separately — they are resolved during CPR finalisation via the SHA-specific shift mechanism, not in this unified eval sumcheck.

**Proof size:** $3n + k_{\mathrm{private}}$ field elements. Here $k_{\mathrm{private}}$ includes all private (non-public, non-PCS) source columns whose final evaluations the prover must send:
- SHA: 6 shift-source-only columns (indices 14, 15, 21–24)
- ECDSA: 3 shift-source-only columns (indices 2, 3, 4)
- Total: $k_{\mathrm{private}} = 9$

Proof size: $3 \times 9 + 9 = 36$ field elements.

### II.4.11. Step 4: Hybrid GKR Batched Lookup (SHA only)

Identical to Part I §4.6. The lookup operates only on SHA-256's 10 base columns + 3 affine virtual columns. It runs as a standalone sub-protocol **outside** the multi-degree sumcheck.

### II.4.12. Step 5: PCS Prove (both circuits, separate)

Two independent PCS proofs, each with its own transcript:

| Circuit | PCS data | Eval point | PCS field | Code |
|---------|----------|-----------|-----------|------|
| SHA-256 | 12 `BinaryPoly<8>` cols × 2048 | $(\mathbf{r}_{\mathrm{cpr}}, \gamma_1, \gamma_2) \in \mathbb{F}_p^{11}$ | $\mathbb{F}_{p_1}$ (192-bit) | IPRS `PnttConfigF2_16R4B4<3>` |
| ECDSA | 4 `Int<4>` cols × 512 | $\mathbf{r}_{\mathrm{cpr}} \in \mathbb{F}_p^9$ | $\mathbb{F}_s$ (256-bit) | IPRS `PnttConfigF2_16R4B64<1>` |

Each PCS proof creates its own Keccak transcript, absorbs only its own Merkle root, derives its own field config (via `get_random_field_cfg`), converts the PIOP evaluation point to its own field's representation, and calls `ZipPlus::prove_with_seed`.

**PCS transcript isolation:** The PIOP transcript and PCS transcripts are completely decoupled. The PCS modulus can differ from the PIOP modulus — the PIOP operates over 192-bit $\mathbb{F}_p$ while the ECDSA PCS operates over 256-bit $\mathbb{F}_s$ (since `Int<4>` elements require a 256-bit modulus).

### II.4.13. Fiat-Shamir Transcript Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ SHARED PIOP TRANSCRIPT (Keccak-256)                             │
├─────────────────────────────────────────────────────────────────┤
│ Absorb: commitment1.root (32 bytes)                             │
│ Absorb: commitment2.root (32 bytes)                             │
│ Squeeze: field_cfg (random 192-bit prime p)                     │
│ Squeeze: ic_evaluation_point r ∈ F_p^9                          │
│ [IC₁ SHA: absorb combined MLE values]                           │
│ [IC₂ ECDSA: absorb combined MLE values]                         │
│ Squeeze: projecting element α ∈ F_p                             │
│ [CPR₁ build: squeeze SHA folding challenge]                     │
│ [CPR₂ build: squeeze ECDSA folding challenge]                   │
│ [Multi-degree sumcheck: 9 rounds × (absorb 2 round polys       │
│   + squeeze 1 shared challenge)]                                │
│ [CPR₁ finalise: absorb up/down evals]                           │
│ [Fold round 1: absorb c1s/c2s; squeeze γ₁]                     │
│ [Fold round 2: absorb c3s/c4s; squeeze γ₂]                     │
│ [CPR₂ finalise: absorb up/down evals]                           │
│ [Unified eval sumcheck: 9 rounds]                               │
│ [Hybrid GKR lookup: standalone sub-protocol]                    │
├─────────────────────────────────────────────────────────────────┤
│ PCS₁ TRANSCRIPT (independent Keccak)                            │
│ Absorb: commitment1.root → derive F_{p₁} → prove               │
├─────────────────────────────────────────────────────────────────┤
│ PCS₂ TRANSCRIPT (independent Keccak)                            │
│ Absorb: commitment2.root → derive F_s (256-bit) → prove        │
└─────────────────────────────────────────────────────────────────┘
```

---

## II.5. Dual-Circuit E2E Verifier Pipeline

The verifier function is `verify_dual_circuit_hybrid_gkr_4x_folded`. It replays the shared transcript and verifies each sub-protocol.

### II.5.1. Transcript Replay & IC Verify

1. **Reconstruct transcript:** Absorb both Merkle roots, derive PIOP field config and shared IC point $\mathbf{r}$.
2. **IC₁ verify (SHA):** Deserialise proof, call `IdealCheckProtocol::verify_at_point::<Sha256Uair>`. Verifies 4 cyclotomic ideal membership checks + 12 zero checks.
3. **IC₂ verify (ECDSA):** Deserialise proof, call `IdealCheckProtocol::verify_at_point::<EcdsaUairInt>`. Verifies all 11 constraint evaluations are zero.
4. **Squeeze** shared projecting element $\alpha$.

### II.5.2. Multi-Degree Sumcheck Verify

1. **Build CPR₁ verifier state** (SHA): Reconstruct combination function, compute claimed sum.
2. **Build CPR₂ verifier state** (ECDSA): Same for ECDSA.
3. **Verify multi-degree sumcheck**: Check 2 groups × 9 rounds. Each round: verify $P_i(0) + P_i(1) = \text{claim}_i$ for each group, apply shared challenge.

### II.5.3. CPR Finalisation & Folding Verify

1. **Finalise CPR₁:** Reconstruct public column evaluations (12 public SHA columns). Merge private `up_evals` from proof with verifier-computed public values. Run `finalize_verifier::<Sha256Uair>`.
2. **Two-round folding verify (SHA only):**
   - Round 1: Verify $c_{1,j} + \alpha^{16} \cdot c_{2,j} = e_j$ for each of 12 SHA columns.
   - Round 2: Verify $c_{3,j} + \alpha^{8} \cdot c_{4,j} = d_j$ for each of 12 columns.
3. **Finalise CPR₂:** Reconstruct public column evaluations (4 public ECDSA columns). Run `finalize_verifier::<EcdsaUairInt>`.

### II.5.4. Unified Eval Sumcheck Verify

Reconstruct all 44 claims (30 SHA eq + 11 ECDSA eq + 3 ECDSA shift). Verify the shift sumcheck proof. For public shift sources, compute $v_i^{\mathrm{final}}$ from known data; for private sources, use prover-supplied values.

### II.5.5. Lookup Verify (SHA only)

Verify the standalone Hybrid GKR batched lookup proof for SHA-256's 13 lookup columns.

### II.5.6. PCS Verify (both circuits, parallel)

Two independent PCS verifications (parallelised with `#[cfg(feature = "parallel")]`):

1. **PCS₁ (SHA):** Create transcript from `commitment1.root`, derive 192-bit PCS field, convert folded point, call `ZipPlus::verify_with_field_cfg`.
2. **PCS₂ (ECDSA):** Create transcript from `commitment2.root`, derive 256-bit PCS field, convert point, call `ZipPlus::verify_with_field_cfg`.

> **Note:** The verifier benchmark passes `TrivialIdeal` for both IC closures, skipping ideal membership checks during timing. This means the reported verifier time **underestimates** the cost of a production verifier by the time of 4 cyclotomic membership checks (SHA) + 11 zero checks (ECDSA).

---

## Appendix E: ECDSA Column Layout

| Index | Name | Ring | Role | PCS? | Public? |
|-------|------|------|------|------|---------|
| 0 | `b1` | $\mathrm{Int}\langle 4 \rangle$ | Scalar bit of $u_1$ (bit-index $257-t$) | | ✓ |
| 1 | `b2` | $\mathrm{Int}\langle 4 \rangle$ | Scalar bit of $u_2$ (bit-index $257-t$) | | ✓ |
| 2 | `X` | $\mathrm{Int}\langle 4 \rangle$ | Accumulator $x$-coord (Jacobian) | | (shift src) |
| 3 | `Y` | $\mathrm{Int}\langle 4 \rangle$ | Accumulator $y$-coord (Jacobian) | | (shift src) |
| 4 | `Z` | $\mathrm{Int}\langle 4 \rangle$ | Accumulator $z$-coord (Jacobian) | | (shift src) |
| 5 | `X_mid` | $\mathrm{Int}\langle 4 \rangle$ | Doubled point $x$-coord | ✓ | |
| 6 | `Y_mid` | $\mathrm{Int}\langle 4 \rangle$ | Doubled point $y$-coord | ✓ | |
| 7 | `Z_mid` | $\mathrm{Int}\langle 4 \rangle$ | Doubled point $z$-coord | ✓ | |
| 8 | `H` | $\mathrm{Int}\langle 4 \rangle$ | Addition scratch: chord $x$-diff | ✓ | |
| 9 | `sel_init` | $\mathrm{Int}\langle 4 \rangle$ | Boundary selector ($1$ at row 0) | | ✓ |
| 10 | `sel_final` | $\mathrm{Int}\langle 4 \rangle$ | Boundary selector ($1$ at row 257) | | ✓ |

---

## Appendix F: ECDSA Constraint Catalogue

All 11 constraints use `assert_zero` with `ImpossibleIdeal`.

### Non-boundary constraints (C1–C7)

| # | Target | Degree | Formula |
|---|--------|--------|---------|
| C1 | $Z_{\mathrm{mid}}$ | 2 | $Z_{\mathrm{mid}} - 2YZ$ |
| C2 | $X_{\mathrm{mid}}$ | 4 | $X_{\mathrm{mid}} - 9X^4 + 8XY^2$ |
| C3 | $Y_{\mathrm{mid}}$ | 5 | $Y_{\mathrm{mid}} - 12X^3Y^2 + 3X^2 X_{\mathrm{mid}} + 8Y^4$ |
| C4 | $H$ | 4 | $H - T_x Z_{\mathrm{mid}}^2 + X_{\mathrm{mid}}$ |
| C5 | $Z[t{+}1]$ | 5 | $(1-\mathrm{sf}) \bigl(Z[t{+}1] - (1{-}s) Z_{\mathrm{mid}} - s \cdot Z_{\mathrm{mid}} H\bigr)$ |
| C6 | $X[t{+}1]$ | **13** | $(1-\mathrm{sf}) \bigl(X[t{+}1] - (1{-}s) X_{\mathrm{mid}} - s (R_a^2 - H^3 - 2 X_{\mathrm{mid}} H^2)\bigr)$ |
| C7 | $Y[t{+}1]$ | 11 | $(1-\mathrm{sf}) \bigl(Y[t{+}1] - (1{-}s) Y_{\mathrm{mid}} - s (R_a (X_{\mathrm{mid}} H^2 - X[t{+}1]) - Y_{\mathrm{mid}} H^3)\bigr)$ |

Where: $s = b_1 + b_2 - b_1 b_2$, $\mathrm{sf} = \mathrm{sel\_final}$, $R_a = T_y \cdot Z_{\mathrm{mid}}^3 - Y_{\mathrm{mid}}$.

### Boundary constraints (B3a–B3c, B4)

| # | Target | Degree | Formula |
|---|--------|--------|---------|
| B3a | Init $X$ | 3 | $\mathrm{sel\_init} \cdot (X - T_x)$ |
| B3b | Init $Y$ | 3 | $\mathrm{sel\_init} \cdot (Y - T_y)$ |
| B3c | Init $Z$ | 2 | $\mathrm{sel\_init} \cdot (Z - 1)$ |
| B4 | Final sig | 4 | $\mathrm{sel\_final} \cdot Z \cdot (X - R_{\mathrm{SIG}} Z^2)$ |

### secp256k1 Constants

| Constant | Value |
|----------|-------|
| $G_x$ | `0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798` |
| $G_y$ | `0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8` |
| $Q$ | $= G$ (benchmark uses $Q = G$ for simplicity) |
| $G+Q$ | $= 2G$ (precomputed) |
| $R_{\mathrm{SIG}}$ | Lazily computed: run 257-step trace in $\mathbb{F}_q$, recover $X/Z^2$ via Fermat inversion |

---

## Appendix G: Dual-Circuit Concrete Parameters

| Parameter | SHA-256 (Circuit 1) | ECDSA (Circuit 2) |
|-----------|--------------------|--------------------|
| **Trace rows** | $N = 2^9 = 512$ | $N = 2^9 = 512$ (258 logical + 254 padding) |
| **Total columns** | 30 (27 bitpoly + 3 int) | 11 (all `Int<4>`) |
| **PCS-committed columns** | 12 | 4 |
| **Public columns** | 12 | 4 |
| **Shift-source-only columns** | 6 | 3 |
| **Shift specs (Bp)** | 10 | 3 |
| **Constraints** | 16 (4 cyclotomic + 12 zero) | 11 (all zero) |
| **Max constraint degree** | 1 | 13 |
| **PIOP field** | $\mathbb{F}_p$ (192-bit) | $\mathbb{F}_p$ (192-bit, shared) |
| **PCS field** | $\mathbb{F}_{p_1}$ (192-bit) | $\mathbb{F}_s$ (256-bit) |
| **Trace ring** | `BinaryPoly<32>` | `Int<4>` |
| **PCS eval ring** | `BinaryPoly<8>` (after 4× folding) | `Int<4>` (no folding) |
| **PCS code** | IPRS `PnttConfigF2_16R4B4<3>`: rate 1/4, input 2048, output 8192 | IPRS `PnttConfigF2_16R4B64<1>`: rate 1/4, input 512, output 2048 |
| **PCS code field** | $\mathbb{F}_{65537}$ | $\mathbb{F}_{65537}$ |
| **Column openings** | 118 | 118 |
| **Grinding bits** | 16 | 16 |
| **Lookup columns** | 13 (10 base + 3 affine) | 0 |
| **Folding rounds** | 2 (BP<32>→<16>→<8>) | 0 (no folding) |

### Shared Parameters

| Parameter | Value |
|-----------|-------|
| **num_vars** | 9 (shared) |
| **IC evaluation point** | 1 shared point $\mathbf{r} \in \mathbb{F}_p^9$ |
| **Projecting element** | 1 shared $\alpha \in \mathbb{F}_p$ |
| **Multi-degree sumcheck groups** | 2 (SHA deg 1 + ECDSA deg ≤13) |
| **Multi-degree sumcheck rounds** | 9 |
| **Unified eval sumcheck claims** | 44 (30 SHA eq + 11 ECDSA eq + 3 ECDSA shift) |

---

## Appendix H: Dual-Circuit Security Analysis

| Sub-protocol | Soundness error | Notes |
|-------------|----------------|-------|
| **IC₁ (SHA)** | $\leq n / |\mathbb{F}_p|$ per constraint | 16 constraints (4 cyclotomic + 12 zero) |
| **IC₁ ideal membership** | $\leq 31 / |\mathbb{F}_p|$ per cyclotomic constraint | 4 cyclotomic constraints check $(X^{32}-1)$ membership after projection |
| **IC₂ (ECDSA)** | $\leq n / |\mathbb{F}_p|$ per constraint | 11 constraints, all zero — IC proof is trivially zero-valued |
| **Multi-degree sumcheck** | $\leq d_{\max} \cdot n / |\mathbb{F}_p|$ | $d_{\max} = 13$ (ECDSA C6), $n = 9$; standard sumcheck error per group |
| **Lookup (LogUp)** | $\leq (K \cdot N) / |\mathbb{F}_p|$ per column | SHA only; $K = 4$ chunks, $N = 512$ |
| **Unified eval sumcheck** | $\leq 2n / |\mathbb{F}_p|$ | Degree-2, $n = 9$ rounds |
| **Eval batching** | $\leq 44 / |\mathbb{F}_p|$ | 44 claims batched with random $\xi_i$ |
| **Folding (SHA)** | $\leq 1 / |\mathbb{F}_p|$ per column per round | 12 columns × 2 rounds = 24 checks |
| **PCS₁ proximity** | $\leq (1 - \delta)^{118} \cdot 2^{-16}$ | 118 openings + 16 grinding bits |
| **PCS₂ proximity** | $\leq (1 - \delta')^{118} \cdot 2^{-16}$ | 118 openings + 16 grinding bits; $\delta'$ is the ECDSA IPRS code distance |
| **Miller-Rabin** | $\leq 2^{-64}$ (heuristic) | Base-2 test on 192-bit (PIOP) and 256-bit (ECDSA PCS) primes |

### Overall Security

By a union bound, the total PIOP soundness error is $O(\mathrm{poly}(N, d_{\max}) / 2^{192})$, which is negligible. The PCS proximity terms dominate the overall soundness. The target security level is approximately **128 bits**.

### Completeness

Completeness follows from:
1. The SHA witness satisfies all 16 constraints (identical to Part I).
2. The ECDSA witness is a valid secp256k1 scalar multiplication trace — all 11 constraints are satisfied by construction.
3. The shared IC point, shared projecting element, and multi-degree sumcheck challenges are all consistently derived from the same transcript.
4. The two PCS proofs are valid encodings of the committed data on their respective fields.
5. Folding identities hold algebraically for the SHA split construction.
