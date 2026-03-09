# Zinc+ SHA-256 8× Folded Benchmark: Algorithmic Specification

> **Benchmarks covered:**
> ```
> cargo bench --bench steps_sha256_8x_folded -p zinc-snark --features "parallel simd asm qx-constraints no-f2x" -- "Prover"
> cargo bench --bench steps_sha256_8x_folded -p zinc-snark --features "parallel simd asm qx-constraints no-f2x" -- "Verifier"
> ```
>
> **Note on `no-f2x`:** The `no-f2x` feature flag **changes the UAIR types**
> used by the benchmark. The benchmark file contains `#[cfg(feature = "no-f2x")]`
> gates (lines 60–81) that switch the type aliases:
>
> | | Without `no-f2x` | With `no-f2x` |
> |---|---|---|
> | **BenchBpUair** | `Sha256Uair` (16 constraints: C1–C4 cyclotomic + C5–C16 zero) | `Sha256UairBpNoF2x` (12 constraints: C5–C16, all zero) |
> | **BenchQxUair** | `Sha256UairQx` (3 constraints, all trivial) | `Sha256UairQxNoF2x` (7 constraints: C1–C4 cyclotomic + C7–C9 trivial) |
> | **Trace columns** | 30 (27 bitpoly + 3 integer) | 38 (35 bitpoly + 3 integer) |
> | **Witness generator** | `Sha256Uair::generate_witness()` | `generate_no_f2x_witness()` |
>
> The core mathematical change: the 4 F₂[X] rotation constraints
> (C1–C4) are moved from the Bp UAIR to the Qx UAIR. In Qx, they are
> reformulated over Z[X] with explicit integer-carry quotient columns
> μ = μ\_lo + 2·μ\_hi (8 new bitpoly columns at indices 27–34).
> Since this document describes the benchmark **with** `no-f2x` enabled,
> all parameters, constraint counts, and proof sizes reflect the no-f2x variant.
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
   - 4.0 [Double-Split Columns](#40-step-0-double-split-columns)
   - 4.1 [PCS Commit](#41-step-1-pcs-commit)
   - 4.2 [Transcript Init & Field Setup](#42-step-2-transcript-init--field-setup)
   - 4.3 [Ideal Check (BinaryPoly UAIR)](#43-step-2a-ideal-check-binarypoly-uair)
   - 4.4 [Ideal Check (Q[X] UAIR)](#44-step-2b-ideal-check-qx-uair)
   - 4.5 [Field Projection & Column Evaluation](#45-step-3a-field-projection--column-evaluation)
   - 4.6 [Hybrid GKR Batched Lookup](#46-step-3b-hybrid-gkr-batched-lookup)
   - 4.7 [Shift Sumcheck](#47-step-3c-shift-sumcheck)
      - 4.7.1 [Soundness Argument for down_evals](#471-soundness-argument-for-down_evals)
   - 4.8 [Two-Round Folding Protocol](#48-step-4-two-round-folding-protocol)
   - 4.9 [PCS Prove](#49-step-5-pcs-prove)
   - 4.10 [Fiat-Shamir Transcript Flow](#410-fiat-shamir-transcript-flow)
5. [E2E Verifier Pipeline](#5-e2e-verifier-pipeline)
   - 5.1 [Transcript Replay & IC Verify](#51-transcript-replay--ic-verify)
   - 5.2 [Column Evaluation Absorb & Lookup Verify](#52-column-evaluation-absorb--lookup-verify)
   - 5.3 [Shift Sumcheck Verify](#53-shift-sumcheck-verify)
   - 5.4 [Two-Round Folding Verify](#54-two-round-folding-verify)
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
- **PCS**: Zip+ with 4×-folded IPRS codes over BinaryPoly⟨8⟩.

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

### 3.2. Column Layout

The columns are organised into six groups (full table in Appendix A):

1. **Core state columns** (indices 0–9): The SHA-256 working variables $a_t, e_t$, message schedule $W_t$, rotation outputs $\Sigma_0, \Sigma_1, \sigma_0, \sigma_1$, and Boolean function outputs $\mathrm{Maj}, \mathrm{Ch}_1, \mathrm{Ch}_2$. All in $\mathcal{R}_{32}$.

2. **F₂[X] decomposition columns** (indices 10–13): Shift quotients $S_0, S_1$ and remainders $R_0, R_1$ for the SHR operations in $\sigma_0, \sigma_1$. These encode the division $W = R + X^k \cdot S$ needed because SHR is not a rotation.

3. **Lookback columns** (indices 14–24): Virtual shifted copies of source columns, providing cross-row references like $d_t = a_{t-3}$, $h_t = e_{t-3}$, $f_t = e_{t-1}$, etc. Indices 14–15 and 21–24 are shift-source-only (non-public); indices 16–20 are public.

4. **Selector columns** (indices 25–26): Row-selector patterns for boundary handling.

5. **μ quotient columns** (indices 27–34, **no-f2x only**): $\mu_{k,\text{lo}}, \mu_{k,\text{hi}} \in \{0,1\}^{<32}[X]$ for $k = 1,\ldots,4$. These are the binary decomposition of the integer carries that arise when the F₂[X] rotation constraints C1–C4 are lifted to Z[X]. Each $\mu_k = \mu_{k,\text{lo}} + 2 \cdot \mu_{k,\text{hi}}$ has coefficients in $\{0,1,2,3\}$, matching the maximum carry from multiplying two binary polynomials.

6. **Integer carry columns** (int indices 0–2): $\mu_a, \mu_e, \mu_W \in \mathbb{Z}$, absorbing overflow from modular-$2^{32}$ additions. As polynomials: $\mu \cdot X^{32}$ cancels the carry at $X = 2$. Although conceptually integer-valued, these columns are **represented as `BinaryPoly<32>`** in the witness (not as a separate integer type): the small integer value is encoded in binary via its bit-decomposition, with only the lowest 2–3 coefficients nonzero (e.g., $\mu_a = 5$ is stored as the bit-vector `[1,0,1,0,\ldots,0]`). This means they are homogeneous with the rest of the trace and undergo the same double-split (§4.0) and PCS commitment path as every other column; the integer content lands in the lowest `BinaryPoly<8>` quarter, with the upper quarters being zero.

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

**Carry propagation constraints (trivial ideal):**

| Constraint | Description |
|------------|-------------|
| **C7** | $a$-update: $\hat{a}_{t+1} = h + \Sigma_1 + \mathrm{Ch}_1 + \mathrm{Ch}_2 + K + W + \Sigma_0 + \mathrm{Maj} - \mu_a \cdot X^{32}$ |
| **C8** | $e$-update: $\hat{e}_{t+1} = d + h + \Sigma_1 + \mathrm{Ch}_1 + \mathrm{Ch}_2 + K + W - \mu_e \cdot X^{32}$ |
| **C9** | $W$-schedule: $\hat{W}_t = W_{t-16} + \sigma_0 + W_{t-7} + \sigma_1 - \mu_W \cdot X^{32}$ |

The **trivial ideal** (every polynomial is a member) is used for C7–C9, which means constraints hold unconditionally on boundary rows. This design choice drops `max_degree` from 2 to 1 (removing selector gating), enabling the faster MLE-first Ideal Check path.

The **cyclotomic ideal** for C1–C4 uses $(X^{32}-1)$ over $\mathbb{Z}[X]$ (or $\mathbb{F}_p[X]$ after projection): reduce coefficients modulo 32 by folding ($X^{32} \equiv 1$), then check all vanish.

> **Without `no-f2x`:** `Sha256UairQx` has only 3 constraints (C7–C9, all trivial). The IC proof is smaller because there are no cyclotomic check values.

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

Each shift represents a **left-shift** (look-ahead): $\mathrm{left\_shift}_c(v)[i] = v[i+c]$ for $i < N-c$, zero-padded otherwise.

---

## 4. E2E Prover Pipeline

The function `prove_hybrid_gkr_logup_4x_folded` (in `snark/src/pipeline.rs`) executes the following steps:

### 4.0. Step 0: Double-Split Columns

**Input:** 20 private `BinaryPoly<32>` columns of length $N = 512$ (12 original + 8 μ quotient columns from no-f2x).

> **Without `no-f2x`:** 12 private columns.

**Operation:** Each `BinaryPoly<32>` element $a(X) = \sum_{i=0}^{31} c_i X^i$ is split into low and high halves:

$$a_{\mathrm{lo}}(X) = \sum_{i=0}^{15} c_i X^i \in \mathcal{R}_{16}, \quad a_{\mathrm{hi}}(X) = \sum_{i=0}^{15} c_{16+i} X^i \in \mathcal{R}_{16}$$

A column of length $N$ becomes a column of length $2N$ in $\mathcal{R}_{16}$, with the first $N$ entries being the low halves and the last $N$ entries being the high halves. The key algebraic identity is:

$$a(\alpha) = a_{\mathrm{lo}}(\alpha) + \alpha^{16} \cdot a_{\mathrm{hi}}(\alpha)$$

This is applied twice:
1. `split_columns::<32, 16>`: 20 columns × $N$ → 20 columns × $2N$ in $\mathcal{R}_{16}$
2. `split_columns::<16, 8>`: 20 columns × $2N$ → 20 columns × $4N$ in $\mathcal{R}_{8}$

**Output:** 20 `BinaryPoly<8>` columns of length $4N = 2048$. Each entry has only 8 bits (1 byte), making PCS encoding significantly cheaper.

> **Without `no-f2x`:** 12 columns.

**MLE dimension:** The original columns have `num_vars` = 9. After double-splitting, the committed columns have `num_vars` = 11 (two extra variables selecting the quarter).

### 4.1. Step 1: PCS Commit

**Input:** 20 `BinaryPoly<8>` columns of length 2048.

> **Without `no-f2x`:** 12 columns.

**Operation:** Zip+ commit using an IPRS linear code:

1. **Encode** each row of evaluations using the IPRS code `PnttConfigF2_16R4B32<2>`:
   - Input length: 2048 (= `BASE_LEN` × $8^{\text{DEPTH}}$ = 32 × 64)
   - Output length: 8192 (= `BASE_DIM` × $8^{\text{DEPTH}}$ = 128 × 64)
   - Rate: 1/4
   - Encoding: Radix-8 pseudo-NTT over $\mathbb{Z}$ using twiddle factors from $\mathbb{F}_{65537}$

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
3. Sends and **absorbs** proof values $v'_1, \ldots, v'_7$ into the transcript. The 4 cyclotomic constraints (C1–C4) are checked against the cyclotomic ideal $(X^{32}-1)$; the 3 carry constraints (C7–C9) use the trivial ideal.

> **Without `no-f2x`:** Only 3 constraints (C7–C9, all trivial), producing a smaller IC proof.

**Purpose:** This proves both (a) that the rotation identities hold modulo $(X^{32}-1)$ over $\mathbb{Z}[X]$ (via the μ quotient correction), and (b) that the carry-propagation identities hold in $\mathbb{Z}[X]$ (lifted to $\mathbb{F}_p[X]$), providing soundness for the modular-$2^{32}$ additions.

**Soundness of the trivial ideal for C7–C9:** The trivial ideal choice means the IC membership check is vacuous for carry constraints — but this is **sound** because the Q[X] constraints hold on *all* 512 rows by construction:

- **Active rows (where selectors = 1):** The SHA-256 round equations hold by correct witness generation, and the carry $\mu$ absorbs the integer overflow.
- **Boundary rows (where selectors = 0):** The witness generator forces $a = e = 0$ on the last 3 rows and $W = 0$ on the last 16 rows. Both sides of each constraint evaluate to zero (all summands are zero and $\mu = 0$), so the constraint holds trivially.
- **Cross-compression boundaries:** At the start of each 64-round block, initial values are loaded as part of the SHA-256 chaining. The carry column compensates for any overflow.

Thus no selector gating is needed for the carry constraints: they hold everywhere, and the trivial ideal is sound.

**Soundness of the cyclotomic ideal for C1–C4:** The μ correction $Q - 2\mu \in (X^{32}-1)$ is verified by checking membership in the cyclotomic ideal over $\mathbb{F}_p[X]$ after projection. This gives soundness error $\leq 31/|\mathbb{F}_p|$ per constraint (Schwartz-Zippel on a degree-31 polynomial after folding mod $X^{32}-1$).

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

### 4.8. Step 4: Two-Round Folding Protocol

**Purpose:** Reduce the PIOP evaluation claims from the original `BinaryPoly<32>` columns to the committed `BinaryPoly<8>` columns.

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

**Proof cost per round:** 2 field elements per committed column ($c_1, c_2$). Total: $2 \times 20 \times 2 = 80$ field elements (with no-f2x; 48 without).

### 4.9. Step 5: PCS Prove

**Input:** 20 `BinaryPoly<8>` columns of length 2048, evaluation point $\mathbf{r}_2 \in \mathbb{F}_p^{11}$.

> **Without `no-f2x`:** 12 columns.

**Operation:** Zip+ prove with the seed (Merkle root) absorbed:

1. **Tensorise** the point: split into row-selector $q_0$ and column-selector $q_1$ parts.
2. **Combine** polynomial coefficients with Fiat-Shamir challenges (alpha vectors).
3. **Compute $b$:** Per-row inner products with $q_1$, written to transcript.
4. **Compute evaluation:** $\mathrm{eval} = \langle q_0, b \rangle$.
5. **Compute combined row:** Squeeze row-challenge coefficients $s_j$, form the combined row.
6. **Grinding:** Search for the smallest 64-bit nonce where `Blake3(transcript \| nonce)` has $\geq 16$ leading zero bits. The expected nonce value is $2^{16} = 65\,536$, so on average $\approx 65\,536$ sequential Blake3 hashes are required. When the `parallel` feature is enabled, this search is parallelised across $T$ rayon threads using **interleaved assignment**: thread $t \in [T]$ checks nonces $t, t+T, t+2T, \ldots$, with an `AtomicU64` upper bound that lets all threads terminate as soon as the globally-smallest valid nonce is found. This divides wall-clock grinding time by $\approx T$ while always producing the same nonce (and therefore identical proofs) regardless of parallelism.
7. **Column openings:** Squeeze 118 random column indices. For each, provide the column values across all polynomials + a Merkle authentication path.

**Output:** Serialised PCS proof bytes + evaluation value.

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
 [Squeeze] Projecting element α ∈ Fp                     §4.5
 [Absorb]  up_evals (38 Fp elements)                     §4.5
 [Absorb]  down_evals (10 Fp elements)                   §4.5
 [Absorb]  Aggregated multiplicities (per-lookup)        §4.6
 [Squeeze] Lookup challenges β, γ                        §4.6
   ⋮       (GKR internal absorbs/squeezes)               §4.6
 [Squeeze] Shift batching coefficients ξ₁, ..., ξ₁₀     §4.7
   ⋮       (9 sumcheck rounds: absorb pᵢ, squeeze sᵢ)   §4.7
 [Absorb]  v_finals (6 private Fp elements)              §4.7
 [Absorb]  Folding round 1: c₁,c₂ for 20 columns        §4.8
 [Squeeze] Folding challenge γ₁                          §4.8
 [Absorb]  Folding round 2: c₃,c₄ for 20 columns        §4.8
 [Squeeze] Folding challenge γ₂                          §4.8
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

The function `verify_classic_logup_4x_folded` (in `snark/src/pipeline.rs`) mirrors the prover.

> **Note on function name:** The prover function is named `prove_hybrid_gkr_logup_4x_folded` while the verifier is `verify_classic_logup_4x_folded`. The asymmetric naming is a historical artifact — the verifier was originally written for a "classic" lookup path and was not renamed when Hybrid GKR was added. Both functions implement the same Hybrid GKR LogUp protocol.

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

**Output:** The evaluation point $\mathbf{r}$ for subsequent steps.

### 5.2. Column Evaluation Absorb & Lookup Verify

> **No CPR sumcheck on the verifier side either.** Since the E2E prover sends column
> evaluations directly (no CPR sumcheck proof), the verifier simply deserialises them,
> absorbs them into the transcript, and proceeds to lookup verification.

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

1. **Reconstruct** shift claims from `down_evals` and UAIR shift specs.
2. **Replay** the $n = 9$ rounds of the sumcheck: for each round, absorb the prover's polynomial, check $p(0) + p(1) = \mathrm{claim}$, and draw the challenge.
3. **Compute** shift predicates $S_{c_i}(\mathbf{s}, \mathbf{r})$ at the challenge point.
4. **Check final identity:**
   $$\mathrm{claim} = \sum_i \xi_i \cdot S_{c_i}(\mathbf{s}, \mathbf{r}) \cdot v_i^{\mathrm{final}}$$
5. For **public shifts**: the verifier computes $v_i^{\mathrm{final}}$ by evaluating the known column MLE at $\mathbf{s}$.

### 5.4. Two-Round Folding Verify

**Round 1** ($D = 32 \to 16$):
1. Deserialise $(c_{1,j}, c_{2,j})$ for each column.
2. **Check consistency:** $c_{1,j} + \alpha^{16} \cdot c_{2,j} = e_j$ (the PIOP evaluation from `up_evals`, corrected for zero-padding if needed).
3. Absorb into transcript, draw $\gamma_1$.
4. Compute new evaluations: $d_{1,j} = (1-\gamma_1)c_{1,j} + \gamma_1 c_{2,j}$.

**Round 2** ($16 \to 8$): Identical structure with $(c_{3,j}, c_{4,j})$.

**Output:** New evaluation point $\mathbf{r}_2 = (\mathbf{r} \| \gamma_1 \| \gamma_2)$ and claimed evaluations $d_{2,j}$ for the `BinaryPoly<8>` columns.

### 5.5. PCS Verify

1. **Replay** the PCS Fiat-Shamir transcript (using the same transcript state as after prime derivation).
2. **Tensorise** the point $\mathbf{r}_2$.
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

> **Note on column indices 27–29:** With `no-f2x`, the integer carry columns (`mu_a`, `mu_e`, `mu_W`) retain their logical indices as integer columns 0–2 but the bitpoly indices 27–34 are occupied by the μ quotient columns. The total is 35 bitpoly + 3 integer = 38 columns.

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

**Carry propagation (trivial ideal):**

- **C7**: $\hat{a}[t+1] - \hat{h} - \hat{\Sigma}_1 - \widehat{\mathrm{ch\_ef}} - \widehat{\mathrm{ch\_neg\_eg}} - \hat{K} - \hat{W} - \hat{\Sigma}_0 - \widehat{\mathrm{Maj}} + \mu_a X^{32} = 0$
- **C8**: $\hat{e}[t+1] - \hat{d} - \hat{h} - \hat{\Sigma}_1 - \widehat{\mathrm{ch\_ef}} - \widehat{\mathrm{ch\_neg\_eg}} - \hat{K} - \hat{W} + \mu_e X^{32} = 0$
- **C9**: $\hat{W}_t - \hat{W}_{t-16} - \hat{\sigma}_0 - \hat{W}_{t-7} - \hat{\sigma}_1 + \mu_W X^{32} = 0$

> **Without `no-f2x`:** `Sha256UairQx` has only 3 constraints (C7–C9, all trivial ideal).

---

## Appendix C: Concrete Parameters

| Parameter | Value (no-f2x) | Value (standard) |
|-----------|----------------|-------------------|
| **Trace rows** | $N = 2^9 = 512$ (8 × 64 SHA-256 rounds) | same |
| **Total columns** | 38 (35 BinaryPoly<32> + 3 integer) | 30 (27 + 3) |
| **PCS-committed columns** | 20 (12 original + 8 μ) | 12 |
| **Public columns** | 12 | 12 |
| **PIOP field** | $\mathbb{F}_p$, $p$ ≈ $2^{192}$ random prime (192-bit Montgomery) | same |
| **Trace ring** | $\mathcal{R}_{32} = \mathbb{F}_2[X]/(X^{32}+1)$ | same |
| **PCS evaluation ring** | $\mathcal{R}_8 = \mathbb{F}_2[X]/(X^{8}+1)$ after 4× folding | same |
| **PCS code** | IPRS `PnttConfigF2_16R4B32<2>`: rate 1/4, input 2048, output 8192 | same |
| **PCS code field** | $\mathbb{F}_{65537}$ ($= 2^{16}+1$, Fermat prime) | same |
| **Column openings** | 118 | 118 |
| **Grinding bits** | 16 | 16 |
| **Grinding parallelism** | Interleaved nonce search across rayon threads (with `parallel` feature); sequential fallback otherwise. Wall-clock ≈ $2^{16}/T$ Blake3 hashes where $T$ = thread count. | same |
| **Lookup columns** | 13 (10 base + 3 affine virtual) | same |
| **Lookup decomposition** | 4 chunks of 8 bits, sub-table size 256 | same |
| **Hybrid GKR cutoff** | 2 layers | same |
| **Shift claims** | 10 (shift amounts in $\{1, 2, 3, 7, 15, 16\}$; Qx shifts handled internally by IC) | same |
| **Folding rounds** | 2 ($32 \to 16 \to 8$) | same |
| **IC constraints (Bp)** | 12, degree 1 (all zero ideal) | 16 (4 cyclotomic + 12 zero) |
| **IC constraints (Q[X])** | 7 (4 cyclotomic + 3 trivial) | 3 (all trivial) |
| **Sumcheck degree (shift)** | 2 (3 evaluations per round) | same |
| **MLE-first optimization** | Enabled (max_degree = 1 for both UAIRs) | same |
| **Field element size** | 24 bytes (192-bit Montgomery representation, 3 × 64-bit limbs) | same |
| **PCS transcript** | Separate Fiat-Shamir instance seeded from the Merkle root. It derives the same prime $p$ as the PIOP transcript but generates its own internal challenges (alpha vectors, column-opening indices, grinding nonce). The evaluation point $\mathbf{r}_2$ is passed as an **explicit parameter** from the PIOP caller — the PCS never derives it. | same |

---

## Appendix D: Security Analysis

### Soundness Errors by Sub-protocol

| Sub-protocol | Soundness error | Notes |
|-------------|----------------|-------|
| **IC (Bp UAIR)** | $\leq n / |\mathbb{F}_p| \approx 9/2^{128}$ per constraint | Schwartz-Zippel on MLE evaluation at random $\mathbf{r}$; $n = 9$ variables. With no-f2x, all 12 constraints are zero-ideal — IC proof is trivially empty. |
| **IC (Q[X] UAIR)** | $\leq n / |\mathbb{F}_p|$ per constraint | Same MLE bound; with no-f2x, 4 cyclotomic + 3 trivial constraints. |
| **IC ideal membership** | $\leq 31 / |\mathbb{F}_p|$ per cyclotomic constraint | After projection at random $\alpha$, a non-zero polynomial of degree $\leq 31$ is detected with probability $\geq 1 - 31/p$. With no-f2x, this applies to the 4 cyclotomic constraints in the Qx UAIR. |
| **Lookup (LogUp)** | $\leq (K \cdot N) / |\mathbb{F}_p|$ per column | $K = 4$ chunks, $N = 512$; degree of the LogUp polynomial in $\beta$ |
| **Shift sumcheck** | $\leq 2n / |\mathbb{F}_p|$ | Degree-2 sumcheck, $n = 9$ rounds; standard sumcheck error |
| **Shift batching** | $\leq 10 / |\mathbb{F}_p|$ | Random $\xi_i$ coefficients; Schwartz-Zippel over 10 Bp shift claims |
| **Folding** | $\leq 1 / |\mathbb{F}_p|$ per column per round | The folded evaluation $d = (1-\gamma) c_1 + \gamma c_2$ is degree 1 in $\gamma$; Schwartz-Zippel gives $1/|\mathbb{F}_p|$ per check. With 20 columns and 2 rounds, the union bound gives $\leq 40/|\mathbb{F}_p|$. |
| **PCS proximity** | $\leq (1 - \delta)^{118} \cdot 2^{-16}$ | 118 column openings + 16 grinding bits; $\delta$ = relative minimum distance of the IPRS code |
| **Miller-Rabin** | $\leq 2^{-64}$ (heuristic) | Single base-2 test on 192-bit candidate after trial division |

### Overall Security

By a union bound over all sub-protocol invocations (12 + 7 IC constraints with no-f2x, 13 lookup columns, 1 shift sumcheck, 2 folding rounds, 1 PCS), the total soundness error is dominated by the PCS proximity term. The PIOP errors are all $O(\mathrm{poly}(N) / 2^{128})$, which is negligible.

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
