# Benchmark: `steps_sha256_8x_folded`

```
cargo bench --bench steps_sha256_8x_folded -p zinc-snark --features "parallel simd asm"
```

### `no-f2x` variant

To benchmark the **no-F₂[X]** UAIR variant, add the `no-f2x` feature:

```
cargo bench --bench steps_sha256_8x_folded -p zinc-snark --features "parallel simd asm no-f2x"
```

This switches the dual UAIR from `Sha256Uair`/`Sha256UairQx` to `Sha256UairBpNoF2x`/`Sha256UairQxNoF2x`. The trace grows from 30 to 38 columns (8 new μ decomposition columns). See [sha256-uair/src/no_f2x.rs](sha256-uair/src/no_f2x.rs) for details.

#### Motivation

In the standard UAIR, the 4 rotation constraints (C1–C4) are expressed as ideal membership tests over $\mathbb{F}_2[X]$:

$$Q(\ldots) \in (X^{32} - 1) \quad \text{over } \mathbb{F}_2[X]$$

Because $\mathbb{F}_2[X]$ has characteristic 2, there is no carry when multiplying binary polynomials — each coefficient is automatically reduced mod 2. This naturally handles the "carries that vanish mod 2" produced by the bilinear products $\hat{a} \cdot \rho_0$, etc.

The **no-F₂[X] variant** eliminates the reliance on $\mathbb{F}_2[X]$ arithmetic by explicitly accounting for the carries. The constraints become:

$$Q(\ldots) - 2\mu \in (X^{32} - 1) \quad \text{over } \mathbb{Z}[X]$$

where $\mu$ is a **quotient vector** — the polynomial of carry coefficients that vanishes when the constraint is evaluated mod 2. The quotient $\mu$ satisfies $Q(\ldots) \equiv 0 \pmod{2, X^{32}-1}$, and the correction term $2\mu$ absorbs the non-binary coefficients produced by polynomial multiplication over $\mathbb{Z}[X]$.

#### Mathematical derivation

Consider constraint C1: $\hat{a} \cdot \rho_0 - \hat{\Sigma}_0 \in (X^{32}-1)$ over $\mathbb{F}_2[X]$, where $\rho_0 = X^{30} + X^{19} + X^{10}$.

When evaluated over $\mathbb{Z}[X]$, the product $\hat{a} \cdot \rho_0$ may have coefficients $\in \{0, 1, 2, 3\}$ (since $\rho_0$ has 3 nonzero terms and $\hat{a}$ is binary). After subtracting $\hat{\Sigma}_0$ (binary), the residue $R = \hat{a} \cdot \rho_0 - \hat{\Sigma}_0$ reduced mod $(X^{32}-1)$ has even coefficients (because $R \equiv 0 \pmod{2}$ by the original $\mathbb{F}_2[X]$ constraint). Define:

$$\mu = R / 2 \quad (\text{coefficient-wise integer division})$$

The coefficients of $\mu$ are in $\{0, 1, 2, 3\}$ for C1–C2 (3-term rotation polynomial) and $\{0, 1\}$ for C3–C4 (2-term rotation polynomial). In all cases, $\mu$ can be decomposed into two binary polynomials:

$$\mu = \mu_{\text{lo}} + 2 \cdot \mu_{\text{hi}}, \qquad \mu_{\text{lo}}, \mu_{\text{hi}} \in \{0,1\}^{<32}[X]$$

> **Why degree < 32 (not < 64)?** The unreduced product $\hat{a} \cdot \rho_0$ has degree up to $31 + 30 = 61$, so naïvely $\mu$ would live in $\mathbb{Z}[X]_{<62}$. However, the constraint requires membership in the ideal $(X^{32}-1)$, which means we reduce modulo $X^{32} \equiv 1$. This cyclotomic reduction folds coefficient $i$ onto position $i \bmod 32$:
>
> $$\mu[j] \;=\; \frac{1}{2}\!\left(\sum_{i \,\equiv\, j \!\!\pmod{32}} (\hat{a} \cdot \rho_0)[i] \;-\; \hat{\Sigma}_0[j]\right), \qquad j = 0,\ldots,31$$
>
> Since the product has at most 62 nonzero coefficients (positions 0–61), each position $j$ accumulates at most two values: coefficient $j$ and coefficient $j+32$. The result lives in degree < 32 — it is a representative of the quotient ring $\mathbb{Z}[X]/(X^{32}-1)$, not an element of the free polynomial ring. This is why each $\mu_{\text{lo}}, \mu_{\text{hi}}$ fits in a `BinaryPoly<32>` column.

The constraint becomes:

$$\hat{a} \cdot \rho_0 - \hat{\Sigma}_0 - 2 \cdot \mu_{\text{lo}} - 4 \cdot \mu_{\text{hi}} \in (X^{32}-1) \quad \text{over } \mathbb{Z}[X]$$

This is a degree-1 constraint (linear in the trace columns) and can be checked via the standard cyclotomic ideal check over the integers (or over $\mathbb{F}_p[X]$ after field projection).

#### Why C5–C6 (shift decomposition) do not need μ columns

The shift decomposition constraints C5 and C6 are:
- C5: $\hat{W}_{t-15} = R_0 + X^3 \cdot S_0$
- C6: $\hat{W}_{t-2} = R_1 + X^{10} \cdot S_1$

These involve monomial multiplication ($X^k \cdot S$) rather than multi-term polynomial multiplication. Since $S$ is binary and the monomial $X^k$ has a single nonzero coefficient, the product $X^k \cdot S$ has non-overlapping support with $R$ by construction (the shift and remainder occupy disjoint coefficient ranges). No coefficient exceeds 1, so no carry occurs and the constraint is identical over $\mathbb{F}_2[X]$ and $\mathbb{Z}[X]$.

#### Column layout (38 total: 35 bit-poly + 3 integer)

Columns 0–26 and int columns 0–2 are **identical** to the standard UAIR. The 8 new μ columns are appended at bit-poly positions 27–34:

| Index | Name | Description |
|-------|------|-------------|
| 27 | `mu_c1_lo` | $\Sigma_0$ rotation quotient μ₁, low bit ($\mu_{\text{lo}}$) |
| 28 | `mu_c1_hi` | $\Sigma_0$ rotation quotient μ₁, high bit ($\mu_{\text{hi}}$) |
| 29 | `mu_c2_lo` | $\Sigma_1$ rotation quotient μ₂, low bit |
| 30 | `mu_c2_hi` | $\Sigma_1$ rotation quotient μ₂, high bit |
| 31 | `mu_c3_lo` | $\sigma_0$ rotation+shift quotient μ₃, low bit |
| 32 | `mu_c3_hi` | $\sigma_0$ rotation+shift quotient μ₃, high bit |
| 33 | `mu_c4_lo` | $\sigma_1$ rotation+shift quotient μ₄, low bit |
| 34 | `mu_c4_hi` | $\sigma_1$ rotation+shift quotient μ₄, high bit |

Each cell is a `BinaryPoly<32>` with Boolean coefficients, enforced by lookup (the same BitPoly lookup table that enforces columns 0–9). The total column count increases from 30 to 38 (35 bit-poly + 3 integer).

#### Constraint redistribution

The 19 original SHA-256 UAIR constraints are redistributed between the two sub-UAIRs:

**Standard UAIR** (Bp: 16 constraints, Qx: 3 constraints):
- Bp: C1–C4 (rotation, cyclotomic) + C5–C6 (shift decomp, zero) + C7–C16 (linking, zero)
- Qx: C17–C19 (carry propagation, trivial)

**No-F₂[X] UAIR** (Bp: 12 constraints, Qx: 7 constraints):
- Bp (`Sha256UairBpNoF2x`): C5–C6 (shift decomp, zero) + C7–C16 (linking, zero)
- Qx (`Sha256UairQxNoF2x`): C1–C4 (rotation with μ, **cyclotomic**) + C17–C19 (carry propagation, trivial)

The 4 rotation constraints move from the Bp sub-UAIR (where they used `CyclotomicIdeal` over $\mathbb{F}_2[X]$) to the Qx sub-UAIR (where they use `Sha256QxNoF2xIdeal::Cyclotomic` over $\mathbb{Z}[X]$). The Qx sub-UAIR now has a **non-trivial ideal** for C1–C4 (cyclotomic) while retaining the trivial ideal for C17–C19 (carry propagation).

#### Qx constraint formulas (no-F₂[X])

| # | Name | Formula | Ideal |
|---|------|---------|-------|
| C1 | $\Sigma_0$ rotation | $\hat{a} \cdot \rho_0 - \hat{\Sigma}_0 - 2 \cdot \mu_{1,\text{lo}} - 4 \cdot \mu_{1,\text{hi}} \in (X^{32}-1)$ | Cyclotomic |
| C2 | $\Sigma_1$ rotation | $\hat{e} \cdot \rho_1 - \hat{\Sigma}_1 - 2 \cdot \mu_{2,\text{lo}} - 4 \cdot \mu_{2,\text{hi}} \in (X^{32}-1)$ | Cyclotomic |
| C3 | $\sigma_0$ rot+shift | $\hat{W}_{t-15} \cdot \rho_{\sigma_0} + S_0 - \hat{\sigma}_{0,w} - 2 \cdot \mu_{3,\text{lo}} - 4 \cdot \mu_{3,\text{hi}} \in (X^{32}-1)$ | Cyclotomic |
| C4 | $\sigma_1$ rot+shift | $\hat{W}_{t-2} \cdot \rho_{\sigma_1} + S_1 - \hat{\sigma}_{1,w} - 2 \cdot \mu_{4,\text{lo}} - 4 \cdot \mu_{4,\text{hi}} \in (X^{32}-1)$ | Cyclotomic |
| C17 | a-update carry | $\hat{a}[t{+}1] - \hat{h} - \hat{\Sigma}_1 - \widehat{ch\_ef} - \widehat{ch\_\neg eg} - \hat{K} - \hat{W} - \hat{\Sigma}_0 - \widehat{Maj} + \mu_a \cdot X^{32} \in \text{Trivial}$ | Trivial |
| C18 | e-update carry | $\hat{e}[t{+}1] - \hat{d} - \hat{h} - \hat{\Sigma}_1 - \widehat{ch\_ef} - \widehat{ch\_\neg eg} - \hat{K} - \hat{W} + \mu_e \cdot X^{32} \in \text{Trivial}$ | Trivial |
| C19 | W-schedule carry | $\hat{W} - \hat{W}_{t-16} - \hat{\sigma}_{0,w} - \hat{W}_{t-7} - \hat{\sigma}_{1,w} + \mu_W \cdot X^{32} \in \text{Trivial}$ | Trivial |

All 7 constraints are degree 1, enabling MLE-first IC.

#### Ideal types

| Type | Purpose | Contains check |
|------|---------|----------------|
| `Sha256QxNoF2xIdeal::Cyclotomic` | C1–C4 over $\mathbb{Z}[X]$ | Reduce mod $(X^{32}-1)$: fold coefficients at positions $i$ and $i+32$, check all 32 reduced coefficients are zero |
| `Sha256QxNoF2xIdeal::Trivial` | C17–C19 carry propagation | Always passes |
| `Sha256QxNoF2xIdealOverF::Cyclotomic` | C1–C4 after field projection | Same fold-and-check in $\mathbb{F}_p[X]$ |
| `Sha256QxNoF2xIdealOverF::Trivial` | C17–C19 after field projection | Always passes |

#### Witness generation

`generate_no_f2x_witness(num_vars, rng)` generates the 38-column trace:

1. **Delegate to base**: Calls the standard `Sha256UairBp::generate_witness()` to produce the 30-column base trace (columns 0–26 bit-poly + 3 integer).
2. **Compute μ columns**: For each row $t$ and each rotation constraint $C_k$ ($k = 1, \ldots, 4$):
   - Perform integer polynomial multiplication of the operand by the rotation polynomial (as `i32[64]` coefficient arrays).
   - Add/subtract additional polynomial terms (e.g., $-\hat{\Sigma}_0$, $+S_0$).
   - Reduce mod $(X^{32}-1)$ by folding: `reduced[i % 32] += coeffs[i]`.
   - Divide each reduced coefficient by 2 (asserted to be even).
   - Decompose: $\mu_j = \mu_{\text{lo},j} + 2 \cdot \mu_{\text{hi},j}$ (asserted $0 \leq \mu_j < 4$).
   - Pack $\mu_{\text{lo}}$ and $\mu_{\text{hi}}$ as `BinaryPoly<32>` values.
3. **Assemble**: Concatenate `bp[0..27]` (original bit-poly) + `bp[27..35]` (8 new μ columns) + `int[0..3]` (original integer columns) → 38-column trace.

The computation is implemented in the `compute_mu(poly_a, rho_positions, terms)` helper function in [sha256-uair/src/no_f2x.rs](sha256-uair/src/no_f2x.rs).

#### Benchmark integration

The `no-f2x` feature flag propagates through the crate dependency chain:

- `zinc-snark/Cargo.toml`: `no-f2x = ["zinc-sha256-uair/no-f2x"]`
- `zinc-sha256-uair/Cargo.toml`: `no-f2x = []`

In the benchmark file ([snark/benches/steps_sha256_8x_folded.rs](snark/benches/steps_sha256_8x_folded.rs)), the following are `cfg`-gated on `feature = "no-f2x"`:

| Symbol | Without `no-f2x` | With `no-f2x` |
|--------|-------------------|---------------|
| `BenchBpUair` | `Sha256Uair` | `Sha256UairBpNoF2x` |
| `BenchQxUair` | `Sha256UairQx` | `Sha256UairQxNoF2x` |
| `SHA256_BATCH_SIZE` | 30 | 38 (`NO_F2X_NUM_COLS`) |
| `generate_sha256_trace()` | `Sha256UairBp::generate_witness(9, rng)` | `generate_no_f2x_witness(9, rng)` |

All 15 benchmark functions use the `BenchBpUair`/`BenchQxUair` type aliases, so the same benchmark binary exercises either the standard or no-F₂[X] UAIR depending on the feature flag. The pipeline function, PCS configuration, and folding parameters are unchanged.

#### Impact on proof structure

| Component | Standard | No-F₂[X] | Change |
|-----------|----------|-----------|--------|
| Trace columns | 30 (27 bp + 3 int) | 38 (35 bp + 3 int) | +8 bp columns |
| Bp UAIR constraints | 16 | 12 | −4 (C1–C4 moved) |
| Qx UAIR constraints | 3 (trivial ideal) | 7 (4 cyclotomic + 3 trivial) | +4 (C1–C4 gained) |
| Qx ideal | Trivial only | Cyclotomic + Trivial | Non-trivial ideal check |
| Lookup columns | 10 base + 3 affine = 13 | 10 base + 3 affine = 13 | Unchanged (μ columns need lookup too, but are not in the rotation-value set) |
| PCS-committed columns | 10 | 18 | +8 (μ cols are private, not public, not shift-source) |
| Shift specs (Bp) | 12 | 10 | −2 (â[t+1], ê[t+1] shifts moved to Qx) |
| Shift specs (Qx) | 2 | 2 | Unchanged |
| `max_degree` | 1 (both) | 1 (both) | Unchanged |

## Overview

This benchmark provides a **fine-grained, step-by-step timing breakdown** of the Zinc+ proving stack applied to the **8×SHA-256** workload (8 SHA-256 compressions in parallel) using **4× column folding** and the **Hybrid GKR c=2** lookup protocol to reduce proof size. It uses the [Criterion](https://bheisler.github.io/criterion.rs/book/) framework with a custom harness.

The benchmark exercises a **single pipeline variant**:

| Pipeline variant | Lookup method | Column folding | PCS element type | Chunk decomposition |
|---|---|---|---|---|
| **4× folded Hybrid GKR c=2** | Hybrid GKR (cutoff $c = 2$) | 2 splits (32→16→8) | `BinaryPoly<8>` | 4 chunks × $2^{8}$ |

The benchmark uses a **dual UAIR** architecture: `Sha256Uair` (BinaryPoly constraints) and `Sha256UairQx` (Q[X] carry-propagation constraints). Both have `max_degree == 1`. The pipeline function is `prove_hybrid_gkr_logup_4x_folded<Sha256Uair, Sha256UairQx, FoldedZt4x, FoldedLc4x, 32, 16, 8, UNCHECKED>` / `verify_classic_logup_4x_folded`.

Individual prover steps are benchmarked (13 per-step + 2 E2E = 15 benchmark functions), including **standalone** shift sumcheck and folding steps. Both **prover** and **verifier** end-to-end times are measured.

The benchmark also reports:
- **Proof size** breakdown (PCS, IC, QX IC, CPR, QX CPR, lookup, shift SC, folding, etc.) — raw bytes and Deflate-compressed bytes. Note: QX shift SC is `None` in the current pipeline.
- **Peak memory** usage.
- **Prover pipeline timing** breakdown (printed via `eprintln!`).

---

## The Workload: 8×SHA-256 over `BinaryPoly<32>`

The **UAIR** (Universal Algebraic Intermediate Representation) for SHA-256 is defined in `sha256-uair`. It defines the constraint structure for a trace with $2^9 = 512$ rows (sized for 8 compressions: $8 \times 64$ rounds) and **30 columns**. Note: the benchmark witness generator only populates **1** compression (65 active rows, zero-padded to 512) — the "8×" in the benchmark name refers to the trace dimensions, which determine the proof system's computational cost:

| Column range | Count | Description |
|---|---|---|
| 0–9 | 10 | Bit-polynomial $Q[X]$ columns (`a_hat`, `e_hat`, `W_hat`\*, `Sigma0_hat`, `Sigma1_hat`, `Maj_hat`, `ch_ef_hat`, `ch_neg_eg_hat`, `sigma0_w_hat`, `sigma1_w_hat`) |
| 10–13 | 4 | $\mathbb{F}_2[X]$ shift quotient/remainder columns (`S0`, `S1`, `R0`, `R1`) — **public** |
| 14–20 | 7 | Auxiliary lookback columns (`d_hat`, `h_hat`, `W_{t-2}`\*, `W_{t-7}`\*, `W_{t-15}`\*, `W_{t-16}`\*, `K_hat`\*) |
| 21–24 | 4 | Ch/Maj affine lookback columns (`a_{t-1}`, `a_{t-2}`, `e_{t-1}`, `e_{t-2}`) |
| 25–26 | 2 | Selector columns (`sel_round`\*, `sel_sched`\*) |
| 27–29 | 3 | Integer carry columns ($\mu_a$, $\mu_e$, $\mu_W$) |

\* = **public** (known to the verifier, not PCS-committed). The 12 public columns are: `W_hat` (2), `S0`–`R1` (10–13), `W_{t-2}`–`W_{t-16}` (16–19), `K_hat` (20), `sel_round` (25), `sel_sched` (26).

Each cell is a `BinaryPoly<32>` — a polynomial in $\mathbb{F}_2[X]$ of degree < 32, represented as a packed 32-bit coefficient vector with Boolean coefficients.

### Key parameters

| Parameter | Value | Meaning |
|---|---|---|
| `SHA256_8X_NUM_VARS` | 9 | $2^9 = 512$ rows |
| `SHA256_BATCH_SIZE` | 30 | 27 bit-poly + 3 integer columns |
| `SHA256_LOOKUP_COL_COUNT` | 10 | 10 $Q[X]$ columns requiring lookup |
| `max_degree` | 1 | All `Sha256Uair` constraints are degree ≤ 1; MLE-first IC path. (`Sha256UairQx` also degree 1 — selectors removed, trivial ideal.) Bench assertion: `max_degree >= 1 && max_degree <= 2`. |
| `NUM_COLUMN_OPENINGS` | 131 | PCS proximity-test column openings |
| `GRINDING_BITS` | 8 | Proof-of-work nonce search (adds 8 bits of computational security) |
| Lookup chunk config | 4 chunks × $2^8$ | `chunk_width=8`, `width=32` → 4 sub-tables of 256 entries each (the only config benchmarked) |
| Affine lookups | 3 | Ch, neg-Ch, Maj computed as affine combinations of trace columns |
| Hybrid GKR cutoff | $c = 2$ | Top 2 layers via GKR, intermediates sent in the clear, bottom layers via fresh GKR |

**Security level:** With rate $\rho = 1/4$ and unique decoding radius $\delta = (1 - \rho)/2 = 3/8$, each column opening provides $\log_2(1/(1 - \delta)) = \log_2(8/5) \approx 0.678$ bits of security. With `GRINDING_BITS = 8`: $131 \times 0.678 + 8 \approx 96.8$ bits.

### UAIR Constraints

The SHA-256 UAIR is split into two sub-UAIRs plus lookup constraints, totaling **19 polynomial constraints** and **13 lookup constraints** (32 constraint checks).

#### Sub-UAIR #1: $\mathbb{F}_2[X]$ constraints (`Sha256UairBp`) — 16 constraints, max degree 1

**Rotation constraints (C1–C4)** — ideal $(X^{32} - 1)$, degree 1:

| # | Name | Formula |
|---|------|---------|
| C1 | $\Sigma_0$ rotation | $\hat{a} \cdot \rho_0 - \hat{\Sigma}_0 \in (X^{32}-1)$ |
| C2 | $\Sigma_1$ rotation | $\hat{e} \cdot \rho_1 - \hat{\Sigma}_1 \in (X^{32}-1)$ |
| C3 | $\sigma_0$ rotation+shift | $\hat{W}_{t-15} \cdot \rho_{\sigma_0} + S_0 - \hat{\sigma}_{0,w} \in (X^{32}-1)$ |
| C4 | $\sigma_1$ rotation+shift | $\hat{W}_{t-2} \cdot \rho_{\sigma_1} + S_1 - \hat{\sigma}_{1,w} \in (X^{32}-1)$ |

where $\rho_0 = X^{30}+X^{19}+X^{10}$ (ROTR 2,13,22), $\rho_1 = X^{26}+X^{21}+X^7$ (ROTR 6,11,25), $\rho_{\sigma_0} = X^{25}+X^{14}$ (ROTR 7,18), $\rho_{\sigma_1} = X^{15}+X^{13}$ (ROTR 17,19).

**Shift decomposition constraints (C5–C6)** — zero ideal (exact equality), degree 1:

| # | Name | Formula |
|---|------|---------|
| C5 | $\sigma_0$ shift decomp | $\hat{W}_{t-15} = R_0 + X^3 \cdot S_0$ |
| C6 | $\sigma_1$ shift decomp | $\hat{W}_{t-2} = R_1 + X^{10} \cdot S_1$ |

**Linking constraints (C7–C16)** — zero ideal (exact equality), degree 0:

| # | Source → Aux | Shift | Meaning |
|---|---|---|---|
| C7 | `a_hat` → `d_hat` | 3 | $\hat{d}[t{+}3] = \hat{a}[t]$ |
| C8 | `e_hat` → `h_hat` | 3 | $\hat{h}[t{+}3] = \hat{e}[t]$ |
| C9 | `W_hat` → `W_tm2` | 2 | $\hat{W}_{t-2}[t{+}2] = \hat{W}[t]$ |
| C10 | `W_hat` → `W_tm7` | 7 | $\hat{W}_{t-7}[t{+}7] = \hat{W}[t]$ |
| C11 | `W_hat` → `W_tm15` | 15 | $\hat{W}_{t-15}[t{+}15] = \hat{W}[t]$ |
| C12 | `W_hat` → `W_tm16` | 16 | $\hat{W}_{t-16}[t{+}16] = \hat{W}[t]$ |
| C13 | `a_hat` → `a_tm1` | 1 | $\hat{a}_{t-1}[t{+}1] = \hat{a}[t]$ |
| C14 | `a_hat` → `a_tm2` | 2 | $\hat{a}_{t-2}[t{+}2] = \hat{a}[t]$ |
| C15 | `e_hat` → `e_tm1` | 1 | $\hat{e}_{t-1}[t{+}1] = \hat{e}[t]$ |
| C16 | `e_hat` → `e_tm2` | 2 | $\hat{e}_{t-2}[t{+}2] = \hat{e}[t]$ |

#### Sub-UAIR #2: $\mathbb{Q}[X]$ carry-propagation constraints (`Sha256UairQx`) — 3 constraints, max degree 1

These use the **trivial ideal** (every polynomial passes). Selector gating (`sel_round`, `sel_sched`) has been removed, dropping the constraint degree from 2 to 1 and enabling the MLE-first IdealCheck path. Without selectors the expressions are non-zero at boundary rows (where next-row values are zero-padded), but the trivial ideal makes the ideal check pass unconditionally.

| # | Name | Formula | Ideal |
|---|------|---------|-------|
| C17 | a-update | $\hat{a}[t{+}1] - \hat{h} - \hat{\Sigma}_1 - \widehat{ch\_ef} - \widehat{ch\_\neg eg} - \hat{K} - \hat{W} - \hat{\Sigma}_0 - \widehat{Maj} + \mu_a \cdot X^{32} \in \text{Trivial}$ | Trivial |
| C18 | e-update | $\hat{e}[t{+}1] - \hat{d} - \hat{h} - \hat{\Sigma}_1 - \widehat{ch\_ef} - \widehat{ch\_\neg eg} - \hat{K} - \hat{W} + \mu_e \cdot X^{32} \in \text{Trivial}$ | Trivial |
| C19 | W-schedule | $\hat{W} - \hat{W}_{t-16} - \hat{\sigma}_{0,w} - \hat{W}_{t-7} - \hat{\sigma}_{1,w} + \mu_W \cdot X^{32} \in \text{Trivial}$ | Trivial |

#### Why selectors were removed (placeholder)

> **Note:** The trivial ideal for `Sha256UairQx` is a **placeholder**. The intention is to replace it with a proper ideal that enforces carry correctness once the appropriate ideal is determined. The current trivial ideal is used to unblock benchmarking and pipeline integration.

With the trivial ideal, the carry constraints emit the raw linear expression without selector multiplication. This reduces the max constraint degree from 2 to 1, enabling the more efficient MLE-first IdealCheck path (which avoids projecting the full trace to $F[Y]$). The tradeoff is that the ideal check for these constraints is vacuous — correctness of the carry propagation is not enforced by the PIOP. In practice, carry correctness can be verified by other means (e.g., lookups or direct evaluation at $X=2$) if needed.

#### Lookup constraints (13 total)

**Column lookups (10)**: Columns 0–9 (all 10 primary $Q[X]$ columns) must have every entry in the `BitPoly{32}` table, enforcing that each value is a valid binary polynomial with all coefficients in $\{0,1\}$.

**Affine-combination lookups (3)**: These enforce Ch and Maj correctness by checking that affine linear combinations of trace columns are also valid binary polynomials. For binary vectors $x, y$: $x + y - 2(x \wedge y) = x \oplus y \in \{0,1\}^{<32}[X]$.

| # | Name | Affine expression | Constant |
|---|------|-------------------|----------|
| L1 | Ch part 1 | $\hat{e} + \hat{f} - 2 \cdot \widehat{ch\_ef}$ | 0 |
| L2 | Ch part 2 | $-\hat{e} + \hat{g} - 2 \cdot \widehat{ch\_\neg eg}$ | $\sum_{i=0}^{31} X^i$ |
| L3 | Maj | $\hat{a} + \hat{b} + \hat{c} - 2 \cdot \widehat{Maj}$ | 0 |

---

## Column Folding: The Central Technique

Column folding is the main optimization being benchmarked [Albert: no, it's just one thing in the full stack]. The idea:

1. **Split**: Each `BinaryPoly<D>` column entry $v_i$ is split into a low half $u_i$ (coefficients $0..\text{D}/2$) and a high half $w_i$ (coefficients $\text{D}/2..D$), where $v_i = u_i + X^{D/2} \cdot w_i$.
2. **Commit**: The PCS is only committed over the smaller `BinaryPoly<HALF_D>` columns. A split column has $2n$ entries and $\text{num\_vars} + 1$ MLE variables.
3. **PIOP unchanged**: The Ideal Check, sumcheck, and lookup still operate on the original `BinaryPoly<D>` trace.
4. **Folding round**: After the PIOP produces evaluation claims on the original columns at point $\mathbf{r}$, the prover sends $c_1[j] = \text{MLE}[v'_j](\mathbf{r} \| 0)$ and $c_2[j] = \text{MLE}[v'_j](\mathbf{r} \| 1)$ for each committed column. The verifier checks $c_1[j] + \alpha^{D/2} \cdot c_2[j] = c_j$ (the original eval), then draws a random challenge $\gamma$ to collapse to a single evaluation claim $\text{MLE}[v'_j](\mathbf{r} \| \gamma)$.
5. **PCS prove**: The PCS opening is at the extended point $\mathbf{r} \| \gamma$, over the smaller `BinaryPoly<HALF_D>` columns.

**4× folding** applies this splitting twice: $D \to D/2 \to D/4$, producing two folding rounds with challenges $\gamma_1, \gamma_2$ and a PCS point $\mathbf{r} \| \gamma_1 \| \gamma_2$.

**8× folding** would apply this splitting three times: $D \to D/2 \to D/4 \to D/8$, but this variant is **not implemented** in the current codebase. No `FoldedZt8x` or `FoldedLc8x` types exist.

The benefit: PCS codeword elements are 2× or 4× smaller, so each column opening in the Merkle proof is cheaper. The cost: a small folding proof ($c_1, c_2$ per column per round) is added.

---

## PCS Configurations

A single PCS parameter set is used:

### 4× Folded (`FoldedZt4x` / `FoldedLc4x`)
- **Codeword type**: `BinaryPoly<8>` (64 B per element)
- **Linear code**: `IprsCode` with `PnttConfigF2_16R4B32<2>` — field $\mathbb{F}_{65537}$ ($p = 2^{16}+1$, `TWO_ADICITY=16`), rate 1/4, base length 32, depth 2 → `INPUT_LEN = 2048`, `OUTPUT_LEN = 8192`
- **Folded num_vars**: 11 ($2^{11} = 2048$ evaluations, `row_len = 2048`, 1 row)
- **`NUM_COLUMN_OPENINGS`**: 131
- **`GRINDING_BITS`**: 8

The type aliases are:
```rust
type FoldedZt4x = Sha256ZipTypes<i64, 8>;
type FoldedLc4x = IprsCode<FoldedZt4x, PnttConfigF2_16R4B32<2>, BinaryPolyWideningMulByScalar<i64>, UNCHECKED>;
```

The `PnttConfigF2_16R4B32<2>` config uses `BASE_LEN=32`, `BASE_DIM=128`, `DEPTH=2`, with `i64` codeword coefficients. The field $\mathbb{F}_{65537}$ has small twiddles (max 4096), giving max codeword coefficient $\sim 2^{50}$ (vs $\sim 2^{62}$ at depth 3).

### Main PIOP field

- `F = MontyField<3>` — a 192-bit Montgomery-form prime field selected via Fiat-Shamir transcript (`get_random_field_cfg` with Miller-Rabin primality testing).

---

## E2E Prover Pipeline

The authoritative prover is `prove_hybrid_gkr_logup_4x_folded` (in [snark/src/pipeline.rs](snark/src/pipeline.rs)). It executes the following steps in order:

| # | Pipeline step | Function / operation |
|---|---|---|
| 0 | Double-split columns | `split_columns::<32,16>` then `split_columns::<16,8>` |
| 1 | PCS Commit | `ZipPlus::<PcsZt, PcsLc>::commit(params, &split_trace_quarter)` |
| 2 | Transcript + field setup | `KeccakTranscript::new()` → absorb PCS commitment root → `get_random_field_cfg` |
| 2b | IC (BinaryPoly) | `IdealCheckProtocol::prove_mle_first::<U, D>` (max_degree == 1 path) |
| 2c | QX IC | `IdealCheckProtocol::prove_from_binary_poly_at_point::<U2, D>` at the same eval point |
| 3 | Field projection + extract | Draw $\alpha$ → `project_scalars_to_field` + `project_trace_to_field` → `extract_lookup_columns_from_field_trace` + `append_affine_virtual_columns` → extract shift trace columns |
| 3a | CPR (BinaryPoly) | `CombinedPolyResolver::prove_as_subprotocol::<U>` — standalone degree-3 sumcheck |
| 3b | QX CPR | `CombinedPolyResolver::prove_as_subprotocol::<U2>` — standalone degree-3 sumcheck |
| 3c | Hybrid GKR Lookup | `prove_hybrid_gkr_batched_lookup_with_indices` (cutoff=2) on 13 columns |
| 3d | Shift sumcheck | `shift_sumcheck_prove` — reduces shifted-column claims to unshifted MLE claims |
| 4 | Two-round folding | `fold_claims_prove::<_, _, HALF_D>` then `fold_claims_prove::<_, _, QUARTER_D>` |
| 5 | PCS Prove | `ZipPlus::prove_with_seed` at $\mathbf{r} \| \gamma_1 \| \gamma_2$ (11-dim point) |

> **Note:** The E2E prover sets `qx_shift_sumcheck: None` — no QX shift sumcheck is executed.
>
> Steps 2c and 3b (QX IC and QX CPR) are gated behind `#[cfg(feature = "qx-constraints")]`; this is independent of QX shift sumcheck being `None`.

---

## Individual Prover Steps Benchmarked

These are the per-step Criterion benchmarks. Each isolates one phase of the proving pipeline for fine-grained timing. The individual steps exercise the same algorithms as the E2E prover, but with **independent transcript state** — the FieldSetup step (step 4) does not absorb the PCS commitment root, so all subsequent Fiat-Shamir challenges diverge numerically from the E2E pipeline. This has negligible impact on timing (the computational work is statistically identical regardless of which prime is drawn), but means the individual-step proofs are not verifiable by the E2E verifier.

The individual steps also omit some E2E-only operations:
- **QX IC and QX CPR** have no individual bench steps (only measured via E2E step 14).
- **Affine virtual columns** (`append_affine_virtual_columns`) are not constructed in steps 9–10; only the 10 base lookup columns are extracted.
- **Lookup algorithm**: Step 10 uses the **classic** batched decomposed LogUp prover on 10 columns, whereas the E2E uses **Hybrid GKR** LogUp on 13 columns. Step 10 serves as a comparison baseline.
- **PCS Prove**: Step 13 uses seedless `prove()` rather than the E2E's `prove_with_seed(&root_buf)`.

### 1. `WitnessGen`

**Function**: `Sha256Uair::generate_witness(num_vars=9, rng)`

Generates a 30-column `BinaryPoly<32>` trace sized for $2^9 = 512$ rows. The witness generator runs **8 chained** SHA-256 compressions (each 64 rounds), filling the entire trace with non-trivial computational data. Compression 0 hashes the padded empty-string message; compressions 1–7 use deterministic non-trivial message blocks. The round-function state flows directly from one compression to the next (without the SHA-256 post-processing H-addition), so all linking constraints (`d_hat[t+3] = a_hat[t]`, etc.) are naturally satisfied across compression boundaries.

Due to shift zero-padding at the trace tail, the last compression is slightly truncated: `sel_round = 0` for the last 4 rows (N−4 to N−1), and `sel_sched = 0` for the last 16 rows, giving 508 active round rows and 368 active schedule rows out of 512 total. The 10 $Q[X]$ columns carry the bit-polynomial representations of the round variables ($\hat{a}, \hat{e}, \hat{W}$, intermediate values), while the lookback columns carry copies of earlier rows (needed because the UAIR constraint system is "next-row" only). The integer carry columns ($\mu_a, \mu_e, \mu_W$) store the small integer carries from modular additions. The `rng` parameter is accepted for trait compatibility but unused.

**Output**: `Vec<DenseMultilinearExtension<BinaryPoly<32>>>` — 30 columns, each with $2^9 = 512$ entries.

Before subsequent steps, the benchmark filters out **PCS-excluded columns** (public columns and shift-source columns that the verifier can reconstruct or whose evaluation claims the shift sumcheck resolves). The 12 public columns are: `W_hat` (2), `S0` (10), `S1` (11), `R0` (12), `R1` (13), `W_{t-2}` (16), `W_{t-7}` (17), `W_{t-15}` (18), `W_{t-16}` (19), `K_hat` (20), `sel_round` (25), `sel_sched` (26). The 8 shift-source columns (0–1, 14–15, 21–24) are excluded because the shift sumcheck resolves their evaluation claims. In total 20 columns are excluded, leaving **10 PCS-committed columns** in `sha_pcs_trace`.

### 2. `Folding/SplitColumns`

**Function**: `split_columns::<32, 16>(&sha_pcs_trace)` followed by `split_columns::<16, 8>(&half_trace)`

Performs a **double split** on each PCS-committed column, reducing `BinaryPoly<32>` entries down to `BinaryPoly<8>`:

**First split (32→16)**: For each entry $v_i \in \text{BinaryPoly}<32>$:
1. Extract the **low half** $u_i$: coefficients at positions $0, 1, \ldots, 15$ → a `BinaryPoly<16>`.
2. Extract the **high half** $w_i$: coefficients at positions $16, 17, \ldots, 31$ → a `BinaryPoly<16>`.
3. Such that $v_i = u_i + X^{16} \cdot w_i$.

The output column has 1024 entries and `num_vars = 10`.

**Second split (16→8)**: Each `BinaryPoly<16>` entry from the first split is further split into two `BinaryPoly<8>` halves.

The final output column has 2048 entries and `num_vars = 11`.

With the `parallel` feature, each column is split independently on a Rayon thread.

**Output**: `Vec<DenseMultilinearExtension<BinaryPoly<8>>>` — same number of columns, each with $2^{11} = 2048$ entries, element size quartered from 256 B to 64 B.

### 3. `PCS/Commit (4x-folded)`

**Function**: `ZipPlus::<FoldedZt4x, FoldedLc4x>::commit(&folded_4x_params, &split_trace)`

Commits the double-split `BinaryPoly<8>` columns using the Zip+ polynomial commitment scheme:

1. **Matrix layout**: Each column's 2048 evaluations are arranged into a matrix with `num_rows = 1` row of length `row_len = 2048`.
2. **Linear code encoding**: The single row is encoded through the IPRS code (`PnttConfigF2_16R4B32<2>`), which applies a Number Theoretic Transform over $\mathbb{F}_{65537}$ with rate 1/4. This expands 2048 input symbols to 8192 output codeword symbols per column. With the `parallel` feature, each polynomial's encoding runs on a separate Rayon thread via `cfg_iter!(polys)`, providing up to 10-way parallelism across committed columns. See [§"Implementation Optimizations"](#implementation-optimizations) for details.
3. **Merkle tree**: All encoded columns are interleaved column-wise — leaf $j$ is the hash of `(poly_1_codeword[j] \| poly_2_codeword[j] \| \ldots)` across all committed polynomials. A single Merkle tree of depth 13 (8192 leaves) is constructed.

**Output**: `ZipPlusHint` (codeword matrices + Merkle tree, kept for the proving phase) and `ZipPlusCommitment` (Merkle root + batch size, sent to verifier).

### 4. `PIOP/FieldSetup`

**Function**: `KeccakTranscript::new()` + `transcript.get_random_field_cfg::<F, _, MillerRabin>()`

Initializes the Fiat-Shamir transcript and derives a random 192-bit prime $p$ for the PIOP field `F = MontyField<3>`. The process:

1. Squeeze a candidate integer from the transcript.
2. Run the **Miller-Rabin** primality test. If the candidate is composite, increment and try again.
3. Build the Montgomery-form field configuration (`modulus`, `R`, `R²`, inverses, etc.).

This step is typically fast (a few microseconds) once a prime is found, but has variable latency due to the primality search.

> **Fidelity note**: The E2E pipeline absorbs the PCS commitment root (`HASH_OUT_LEN` bytes) into the transcript **before** calling `get_random_field_cfg`, so the derived prime depends on the commitment. The individual bench step skips this absorption (it has no commitment available at this point). As a result, the derived prime and all subsequent Fiat-Shamir challenges in individual steps 5–13 diverge from those in the E2E pipeline. **This has negligible impact on timing** — the computational work (Montgomery arithmetic, sumcheck rounds, etc.) is statistically identical regardless of which specific prime is drawn, since all candidates are 192-bit primes.

**Output**: `F::Config` — the field configuration used for all subsequent PIOP arithmetic.

### 5. `PIOP/Project Ideal Check`

**Function**: `project_scalars::<F, Sha256Uair>(closure)`

Builds the field-level representation of all UAIR constraint scalars. The SHA-256 UAIR defines constraints whose coefficients are `BinaryPoly<32>` values (each coefficient is a Boolean). This step maps each coefficient to `F` (true → one, false → zero) and wraps them as `DynamicPolynomialF<F>` — univariate polynomials over $F$ that represent the constraint structure.

Concretely, for each distinct `BinaryPoly<32>` scalar appearing in any constraint, the closure maps its 32 Boolean coefficients to a degree-31 polynomial in $F[Y]$, where $Y$ will later be specialized at the projecting element $\alpha$.

**Output**: `HashMap<BinaryPoly<32>, DynamicPolynomialF<F>>` — a map from each scalar to its polynomial representation. This is reused by both the Ideal Check and the main field sumcheck.

### 6. `PIOP/IdealCheck`

**Function**: `IdealCheckProtocol::<F>::prove_mle_first::<Sha256Uair, 32>(...)`

The Ideal Check protocol verifies that the trace satisfies all UAIR constraints modulo the appropriate ideals. Since `Sha256Uair` has `max_degree == 1` (all constraints are linear), the pipeline uses the **MLE-first** path, which avoids projecting the full trace to $F[Y]$:

1. **Build shared eq-table**: Compute $\text{eq}(\mathbf{r}, \cdot)$ once via `build_eq_x_r_inner` in $O(2^n)$ and reuse for all column evaluations.
2. **Evaluate column MLEs at $\mathbf{r}$**: For each of the 30 trace columns and 12 shifted columns, compute $\text{MLE}[v](\mathbf{r}) = \sum_{\mathbf{b}} \text{eq}(\mathbf{r}, \mathbf{b}) \cdot v[\mathbf{b}]$ using the precomputed eq-table. This replaces the expensive full $F[Y]$ projection needed by the standard path.
3. **Draw evaluation point**: Squeeze $\mathbf{r} = (r_0, \ldots, r_8) \in F^9$ from the Fiat-Shamir transcript.
4. **Evaluate combined constraint values**: Using the MLE evaluations from step 2, evaluate each UAIR constraint group to produce combined values.
5. **Absorb proof values**: The combined MLE values are absorbed into the transcript.

> **Note:** With selectors removed from the QX carry constraints, `Sha256UairQx` also has `max_degree == 1` and uses the MLE-first IC path (trivial ideal). The standard IC path (`prove_as_subprotocol`) is no longer needed for either sub-UAIR.

**Output**:
- `Proof<F>` containing `combined_mle_values: Vec<DynamicPolynomialF<F>>` — the constraint evaluations at $\mathbf{r}$, one per combined polynomial.
- `ProverState<F>` containing `evaluation_point: Vec<F>` — the point $\mathbf{r}$ (passed to the next step).

### 7. `PIOP/Project Main field sumcheck`

**Functions**: `project_scalars_to_field(...)` + `project_trace_to_field::<F, 32>(...)`

This step projects everything from the `BinaryPoly<32>` / `DynamicPolynomialF<F>` domain into the scalar PIOP field $F$:

1. **`project_scalars_to_field`**: The Ideal Check drew a projecting element $\alpha \in F$ from the transcript. Each `DynamicPolynomialF<F>` scalar (a polynomial $p(Y) \in F[Y]$) is evaluated at $Y = \alpha$, collapsing it to a single field element $p(\alpha) \in F$. Output: `HashMap<BinaryPoly<32>, F>`.

2. **`project_trace_to_field::<F, 32>`**: Every `BinaryPoly<32>` entry in the 30-column trace is projected to $F$ by evaluating it at $\alpha$. For a `BinaryPoly<32>` with Boolean coefficients $(b_0, b_1, \ldots, b_{31})$, this computes $\sum_{i=0}^{31} b_i \cdot \alpha^i \in F$. This is done using a **precomputed projection closure** (`BinaryPoly::<32>::prepare_projection(&α)`) that caches the powers $\alpha^0, \alpha^1, \ldots, \alpha^{31}$ and computes the dot product via conditional additions (no multiplications needed per entry since $b_i \in \{0, 1\}$).

   Output: `Vec<DenseMultilinearExtension<F::Inner>>` — 30 columns, each with 512 field elements.

After this step, all subsequent PIOP operations work in scalar $F$ arithmetic.

### 8. `PIOP/Main field sumcheck`

**Function**: `CombinedPolyResolver::<F>::prove_as_subprotocol::<Sha256Uair>(...)`

Proves via sumcheck that the Ideal Check evaluation claims are consistent with the trace. This is the most algebraically involved prover step.

**Setup** (before sumcheck):

1. **Build shifted ("down") columns**: For each `ShiftSpec` in the UAIR signature, construct a shifted copy of the source column (drop the first `shift_amount` entries, zero-pad at the end). For SHA-256, these are the lookback columns that reference earlier rows.
2. **Build `eq_r`**: The multilinear extension of the equality function $\text{eq}(\mathbf{r}, \mathbf{b}) = \prod_{i=0}^{n-1} (r_i \cdot b_i + (1-r_i)(1-b_i))$, evaluated at all $2^n$ points. This is a length-512 vector.
3. **Build last-row selector**: $\text{sel}(\mathbf{b}) = \text{eq}(\mathbf{1}, \mathbf{b})$, which is 1 only at the all-ones vertex (last row). The factor $(1 - \text{sel})$ zeroes out the last row, which is a boundary condition.
4. **Draw batching challenge**: Squeeze $\alpha_{\text{fold}} \in F$ from the transcript. This is used to batch all UAIR constraints into one polynomial.
5. **Assemble MLEs**: The sumcheck operates over the array `[sel, eq_r, up_col_0, ..., up_col_{N-1}, down_col_0, ..., down_col_{M-1}]`.

**Sumcheck** (degree `max_degree + 2 = 3`):

The prover runs `MLSumcheck::prove_as_subprotocol` with the combination function:

$$f(\text{mle\_values}) = \text{eq\_r} \cdot (1 - \text{sel}) \cdot \sum_{i=0}^{C-1} \alpha_{\text{fold}}^i \cdot g_i(\text{up}, \text{down})$$

where $g_i$ are the UAIR constraint polynomials applied to the up (current row) and down (shifted row) column values. The `ConstraintFolder` evaluates all constraints in a single pass, accumulating with powers of $\alpha_{\text{fold}}$. The maximum degree is 3 because the `Sha256Uair` constraints are degree 1 (linear), multiplied by `eq_r` (degree 1) and `(1-sel)` (degree 1). (In the E2E pipeline, `Sha256Uair` and `Sha256UairQx` each run a separate standalone CPR sumcheck, both degree 3.)

Over 9 rounds, each round:
- Fixes one variable to a random challenge.
- Sends a degree-3 univariate polynomial (the round message) to the verifier.

**After sumcheck**:

The prover evaluates all trace-column MLEs at the final sumcheck point to produce:
- **`up_evals`**: one $F$ value per original column — $\text{MLE}[\text{col}_j](\mathbf{s})$ where $\mathbf{s}$ is the 9-dimensional sumcheck randomness.
- **`down_evals`**: one $F$ value per shifted column.

These are absorbed into the transcript and included in the proof.

**Output**:
- `Proof<F>` containing `sumcheck_proof` (9 round messages), `up_evals`, `down_evals`.
- `ProverState<F>` containing `evaluation_point` — the sumcheck randomness $\mathbf{s}$.

### 8a. Shift Sumcheck (`PIOP/ShiftSumcheck`)

> **Pipeline status**: The shift sumcheck is active in the `prove_hybrid_gkr_logup_4x_folded` pipeline. In the pipeline it runs after CPR + Lookup finalize and before folding. The shift sumcheck is benchmarked both as a **standalone bench step** (step 11) and as part of the E2E pipeline. This section documents the shift sumcheck prover algorithm for reference.

#### Purpose

The main field sumcheck (step 8) produces **`down_evals`** — one claimed evaluation $\text{MLE}[\text{shift}_{c_i}(v_i)](\mathbf{r})$ per shifted column. These are evaluations of *shifted* versions of committed source columns, but the PCS only commits to the *unshifted* source columns. The **shift sumcheck** bridges this gap: it reduces each shifted-column evaluation claim to a plain MLE evaluation claim on the unshifted source column at a *new* random point $\mathbf{s}$, which the PCS can then open directly.

Formally, for a source column $v$ with left-shift amount $c$, the shifted column is $\text{left}_c(v)[i] = v[i+c]$ for $i < n-c$, zero otherwise. The CPR outputs $\text{MLE}[\text{left}_c(v)](\mathbf{r}) = d$ as a claimed evaluation. The shift sumcheck proves this is consistent with $\text{MLE}[v](\mathbf{s}) = v_{\text{final}}$ at a verifier-chosen point $\mathbf{s}$.

#### SHA-256 shift claims

The `Sha256UairBp` UAIR declares 12 shift specs in `signature().shifts`:

| # | Source column | Shift amount | Source public? |
|---|---|---|---|
| 0 | `COL_D_HAT` (14) | 3 | No |
| 1 | `COL_H_HAT` (15) | 3 | No |
| 2 | `COL_W_TM2` (16) | 2 | **Yes** |
| 3 | `COL_W_TM7` (17) | 7 | **Yes** |
| 4 | `COL_W_TM15` (18) | 15 | **Yes** |
| 5 | `COL_W_TM16` (19) | 16 | **Yes** |
| 6 | `COL_A_TM1` (21) | 1 | No |
| 7 | `COL_A_TM2` (22) | 2 | No |
| 8 | `COL_E_TM1` (23) | 1 | No |
| 9 | `COL_E_TM2` (24) | 2 | No |
| 10 | `COL_A_HAT` (0) | 1 | No |
| 11 | `COL_E_HAT` (1) | 1 | No |

Shifts 2–5 source **public** columns (`W_TM2`, `W_TM7`, `W_TM15`, `W_TM16`), so their `v_finals` are excluded from the proof — the verifier recomputes them from known data. The proof carries only 8 private `v_finals`. Shifts 10–11 (`A_HAT`, `E_HAT`) provide the carry-propagation constraints with access to the *next-row* values â[t+1] and ê[t+1].

Shift-source columns are excluded from PCS commitment (`UairSignature::pcs_excluded_columns()`), because the shift sumcheck resolves their evaluation claims instead.

#### Mathematical formulation

**Function**: `shift_sumcheck_prove(transcript, claims, trace_columns, num_vars, field_cfg)` in [piop/src/shift_sumcheck/prover.rs](piop/src/shift_sumcheck/prover.rs).

Given $k$ shift claims with claimed evaluations $d_i = \text{MLE}[\text{left}_{c_i}(v_i)](\mathbf{r}_i)$, the protocol proves:

$$\sum_{\mathbf{b} \in \{0,1\}^m} \sum_{i=0}^{k-1} \alpha_i \cdot L_{c_i}(\mathbf{b}, \mathbf{r}_i) \cdot v_i(\mathbf{b}) = \sum_{i=0}^{k-1} \alpha_i \cdot d_i$$

where:
- $\alpha_i$ are Fiat-Shamir batching coefficients drawn from the transcript.
- $L_{c_i}(\mathbf{b}, \mathbf{r}_i)$ is the **left-shift predicate** table: $L_c(\mathbf{b}, \mathbf{r}) = \text{eq}^*(\mathbf{r}, \text{bin}(\text{val}(\mathbf{b}) - c))$ when $\text{val}(\mathbf{b}) \geq c$, and 0 otherwise. Implemented by `build_left_shift_table()` in [piop/src/shift_sumcheck/predicate.rs](piop/src/shift_sumcheck/predicate.rs).
- $v_i(\mathbf{b})$ is the MLE of the unshifted source column.

**Round polynomials**: Degree-2 (`ShiftRoundPoly` with 3 evaluations at $X = 0, 1, 2$), using Lagrange interpolation. Over $m = 9$ rounds for SHA-256.

**Key optimisation — table grouping**: Claims that share the same `(eval_point, shift_amount)` reference the same predicate table. The prover builds one table per distinct pair and pre-combines the witness columns within each group: $w_g[j] = \sum_{i \in g} \alpha_i \cdot v_i[j]$. This reduces the number of tables that must be built and folded each round.

**After sumcheck**: The prover evaluates each source column's MLE at the challenge point $\mathbf{s}$ to produce `v_finals[i]` = $\text{MLE}[v_i](\mathbf{s})$, using a shared eq-table $\text{eq}(\mathbf{s}, \cdot)$ built once in $O(n)$. These are absorbed into the transcript.

**Output**: `ShiftSumcheckProverOutput<F>` containing:
- `proof: ShiftSumcheckProof<F>` — $m$ round polynomials (each 3 field elements).
- `challenge_point: Vec<F>` — the random point $\mathbf{s} \in F^m$.
- `final_claim: F` — the final reduced claim value after all sumcheck rounds.
- `v_finals: Vec<F>` — per-claim source column evaluations at $\mathbf{s}$.
- `h_finals: Vec<F>` — per-claim folded predicate values $L_{c_i}(\mathbf{s}, \mathbf{r}_i)$.

**Proof data** (serialized into `SerializedShiftSumcheckProof`):
- `rounds`: $m$ round polynomials, each 3 × `field_elem_size` bytes.
- `v_finals`: one field element per **non-public** shift claim (8 out of 12 for SHA-256).

#### Verifier operation

**Function**: `shift_sumcheck_verify()` (monolithic) or the split API `shift_sumcheck_verify_pre()` + `shift_sumcheck_verify_finalize()` in [piop/src/shift_sumcheck/verifier.rs](piop/src/shift_sumcheck/verifier.rs).

1. Draw the same batching coefficients $\alpha_i$.
2. Replay the $m$ sumcheck rounds: for each round, check $p(0) + p(1) = \text{current\_claim}$, absorb the round poly, draw challenge $s_j$.
3. Absorb `v_finals` (after reconstructing public entries if using the split API).
4. Compute the expected final claim: $\text{expected} = \sum_i \alpha_i \cdot L_{c_i}(\mathbf{s}, \mathbf{r}_i) \cdot v_i(\mathbf{s})$, where $L_{c_i}$ is evaluated via `eval_left_shift_predicate()` — a closed-form $O(m + c \cdot \log c)$ computation decomposing the predicate into carry/no-carry components using `eval_eq_poly`, `eval_next`, and `eval_delta` helpers.
5. Check $\text{current\_claim} = \text{expected}$.

**Output**: `ShiftSumcheckVerifierOutput<F>` containing `challenge_point`, `source_cols`, and `v_finals` — the per-claim evaluation claims that the PCS must open.

The **split API** (`shift_sumcheck_verify_pre` / `shift_sumcheck_verify_finalize`) is used when some shift sources are public columns. It advances the transcript through sumcheck rounds first (yielding $\mathbf{s}$), then the verifier computes public `v_finals` at $\mathbf{s}$ before calling finalize. This matches the CPR pre/finalize pattern.

### 9. `PIOP/LookupExtract`

This step prepares the data needed by the lookup protocol. Two operations per lookup column:

1. **Field columns**: Extract the 10 lookup-relevant columns from `field_trace` (the field-projected trace from step 7). Each is a `Vec<F>` of 512 elements.

2. **Raw integer indices**: Convert each `BinaryPoly<32>` entry from the original trace into a `usize` index by interpreting the 32 Boolean coefficients as a bit-packed integer: $\text{idx} = \sum_{j=0}^{31} b_j \cdot 2^j$. This avoids building a $2^{32}$-entry reverse-lookup table, which would be prohibitive.

3. **Index remapping**: The 10 lookup columns may have duplicate column indices (multiple specs referencing the same column). A `BTreeMap` deduplicates them and builds a remapped `LookupColumnSpec` vector.

**Output**: `(Vec<Vec<F>>, Vec<Vec<usize>>, Vec<LookupColumnSpec>)` — deduplicated field columns, raw indices, and remapped specs.

### 10. `PIOP/Lookup`

**Function**: `prove_batched_lookup_with_indices(&mut transcript, &columns, &raw_indices, &specs, &projecting_element, &field_cfg)`

Proves that each lookup column's values belong to the declared subtable, using the **batched decomposed LogUp** protocol (classic variant). This individual step benchmarks the classic batched prover; the E2E pipeline (step 14) uses the Hybrid GKR c=2 lookup instead. For the GKR-based lookup used in the E2E pipeline, see the [GKR and Hybrid GKR Lookup Protocols](#gkr-and-hybrid-gkr-lookup-protocols) section.

**Grouping**: Specs are grouped by `LookupTableType`. For 8×SHA-256 with 4-chunk configuration, all 10 column lookups use `BitPoly { width: 32, chunk_width: Some(8) }` and form a single group. (Note: the 3 affine lookups — Ch, neg-Ch, Maj — are **not** included in this individual benchmark step; they are only incorporated in the E2E pipeline via `append_affine_virtual_columns`, which adds them as additional virtual columns before calling the lookup prover.)

**For each group**, `BatchedDecompLogupProtocol::prove_as_subprotocol` runs:

1. **Subtable generation**: `generate_bitpoly_table(chunk_width=8, α)` builds a subtable of $2^8 = 256$ entries, where entry $t$ is $\sum_{j=0}^{7} b_j \cdot \alpha^j$ (the projection of an 8-bit polynomial at $\alpha$). The 4 shifts are $\alpha^0, \alpha^8, \alpha^{16}, \alpha^{24}$.

2. **Chunk decomposition**: Each 32-bit index is decomposed into $K = 4$ chunks of 8 bits each. For column $\ell$ and row $i$, chunk $k$ maps to subtable index $\lfloor \text{idx}_i / 2^{8k} \rfloor \bmod 256$. The field-level chunk value is looked up as $c_k^{(\ell)}[i] = \text{subtable}[\text{chunk\_idx}]$.

3. **Multiplicities**: For each column $\ell$ and chunk $k$, compute the multiplicity vector $m_k^{(\ell)}[t]$ = number of times subtable entry $t$ appears. Aggregate across chunks: $m_{\text{agg}}^{(\ell)}[t] = \sum_k m_k^{(\ell)}[t]$. Absorb all aggregated multiplicities.

4. **$\beta$ challenge**: Squeeze from transcript.

5. **Inverse witnesses**: For each chunk value, compute $u_k^{(\ell)}[i] = 1 / (\beta - c_k^{(\ell)}[i])$. For the subtable: $v[t] = 1 / (\beta - T[t])$. Absorb all inverse vectors.

6. **$\gamma$ challenge**: Squeeze batching challenge.

7. **Batched identity polynomial $H$**: Precompute the entire batched identity at all $2^n$ points:

$$H[i] = \sum_\ell \Bigl[ \sum_{k} \gamma^{\ell(K+1)+k} \cdot \bigl((\beta - c_k[i]) \cdot u_k[i] - 1\bigr) + \gamma^{\ell(K+1)+K} \cdot \Bigl(\sum_k u_k[i] - m_{\text{agg}}[i] \cdot v[\text{tbl\_idx}]\Bigr) \Bigr]$$

   This batches two types of sub-identities: **inverse correctness** (each $u_k$ really is $1/(\beta - c_k)$) and **balance** (the sum of chunk inverses equals the multiplicity-weighted table inverse).

8. **Sumcheck** (degree 2): Build just two MLEs — `eq_r` and `H` — and run `MLSumcheck::prove` with combination function $f = \text{eq\_r} \cdot H$. The key optimization: by precomputing $H$ outside the sumcheck loop, the number of MLEs drops from $2 + L(2K+1)$ to just 2 and the degree from 3 to 2.

**Proof data** (serialized into the proof):
- `aggregated_multiplicities`: $L \times T$ field elements ($L$ columns × 256 subtable entries).
- `chunk_inverse_witnesses`: $L \times K \times N$ field elements ($L$ columns × 4 chunks × 512 rows). This is the dominant proof component.
- `inverse_table`: $T = 256$ field elements.
- Sumcheck proof: 9 round messages (degree-2 univariates).

### 11. `PIOP/ShiftSumcheck`

> **Note:** The shift sumcheck is now a **standalone benchmark step** in `steps_sha256_8x_folded`. It benchmarks the shift sumcheck prover independently (after replaying CPR to get the correct transcript state). This step is described in full generality in §8a above; below we detail the algorithm steps.

**Function**: `shift_sumcheck_prove(&mut transcript, &shift_claims, &shift_trace_columns, num_vars, &field_cfg)`

The shift sumcheck reduces the shifted-column evaluation claims from the CPR `down_evals` to plain MLE evaluation claims on the unshifted source columns at a new random point $\mathbf{s}$, which the PCS can then open directly.

#### Setup

1. Re-runs the CPR (untimed) to get `cpr_proof` and `cpr_state` with the correct transcript state (the shift sumcheck reads batching coefficients from the same Fiat-Shamir transcript).
2. Extracts the 12 shift source columns from the field-projected trace — these are the unshifted $v_i$ columns indexed by `ShiftSpec::source_col`.
3. Builds 12 `ShiftClaim` entries from the CPR `down_evals`, one per shift spec:
   - `source_col`: index into the shift trace columns array (0–9).
   - `shift_amount`: the left-shift $c_i$ (1, 2, 3, 7, 15, or 16 for SHA-256).
   - `eval_point`: the CPR sumcheck point $\mathbf{r} \in F^9$ (shared by all claims).
   - `claimed_eval`: $d_i = \text{MLE}[\text{left}_{c_i}(v_i)](\mathbf{r})$ from `cpr_proof.down_evals[i]`.

#### Timed: `shift_sumcheck_prove()`

The prover executes the following steps (all timed):

1. **Draw batching coefficients**: Squeeze $k = 12$ random field elements $\alpha_0, \ldots, \alpha_{11}$ from the transcript. Compute the combined claim $C = \sum_i \alpha_i \cdot d_i$.

2. **Group claims by `(eval_point, shift_amount)`**: Since all 12 claims share the same `eval_point` $\mathbf{r}$, grouping is purely by shift amount. For SHA-256 this produces **6 groups** from 12 claims:

   | Group | Shift amount | Member claims |
   |-------|-------------|---------------|
   | 0 | 1 | {6 (`A_TM1`), 8 (`E_TM1`), 10 (`A_HAT`), 11 (`E_HAT`)} |
   | 1 | 2 | {2 (`W_TM2`), 7 (`A_TM2`), 9 (`E_TM2`)} |
   | 2 | 3 | {0 (`D_HAT`), 1 (`H_HAT`)} |
   | 3 | 7 | {3 (`W_TM7`)} |
   | 4 | 15 | {4 (`W_TM15`)} |
   | 5 | 16 | {5 (`W_TM16`)} |

3. **Build predicate tables** (one per group, 6 total): For each group representative, call `build_left_shift_table(r, c, field_cfg)` which constructs $h[j] = \text{eq}^*(\mathbf{r}, \text{bin}(j - c))$ for $j \geq c$ and $h[j] = 0$ otherwise. Each table has $n = 2^9 = 512$ entries. Internally this builds the full eq-table via `build_eq_x_r_inner` in $O(n)$ and shifts it.

4. **Pre-combine witness columns per group**: For each group $g$, compute $w_g[j] = \sum_{i \in g} \alpha_i \cdot v_i[j]$ for $j = 0, \ldots, n-1$. This reduces the per-round work from 12 column accesses to 6 (one combined column per group). The inner loop is parallelized via `cfg_into_iter!`.

5. **Sumcheck rounds** ($m = 9$ rounds, degree 2): In each round $\ell = 0, \ldots, 8$:
   - Let `half = current_table_size / 2`. Split each group's $h_g$ and $w_g$ tables into low/high halves.
   - **Evaluate round polynomial** at $X = 0, 1, 2$:
     - $e_0 = \sum_g \sum_{j=0}^{\text{half}-1} h_g^{\text{lo}}[j] \cdot w_g^{\text{lo}}[j]$
     - $e_1 = \sum_g \sum_{j=0}^{\text{half}-1} h_g^{\text{hi}}[j] \cdot w_g^{\text{hi}}[j]$
     - $e_2 = \sum_g \sum_{j=0}^{\text{half}-1} (2 h_g^{\text{hi}}[j] - h_g^{\text{lo}}[j]) \cdot (2 w_g^{\text{hi}}[j] - w_g^{\text{lo}}[j])$
   - Each group's $(e_0, e_1, e_2)$ is computed independently with inner parallelism over $j$, then summed across groups.
   - Emit `ShiftRoundPoly { evals: [e_0, e_1, e_2] }`.
   - **Absorb** the 3 field elements into the transcript (3 × `field_elem_size` bytes).
   - **Draw challenge** $s_\ell$ from the transcript. Update the current claim via degree-2 Lagrange interpolation: $\text{claim} \leftarrow p(s_\ell)$.
   - **Fold tables**: For each group in parallel, compute $h_g^{\text{new}}[j] = (1 - s_\ell) \cdot h_g[j] + s_\ell \cdot h_g[j + \text{half}]$ (and the same for $w_g$). Table size halves each round: $512 \to 256 \to \cdots \to 1$.

6. **Compute `h_finals`**: After 9 rounds each group table has a single element. Map each claim to its group's final value: $h_{\text{final},i} = h_{g(i)}[0]$.

7. **Compute `v_finals`** (MLE evaluations at the challenge point $\mathbf{s}$): Build the eq-table $\text{eq}(\mathbf{s}, \cdot)$ once in $O(n)$ using `build_eq_x_r_inner` (with challenges reversed to LE ordering). Then for each claim $i$, compute $v_{\text{final},i} = \sum_{j=0}^{n-1} \text{eq}(\mathbf{s}, j) \cdot v_i[j]$. This step is parallelized across claims. The inner parallel loops use `.with_min_len(256)` to avoid Rayon thread-dispatch overhead when table halves are small (see [§"Implementation Optimizations"](#implementation-optimizations)).

8. **Absorb `v_finals`** into the transcript (12 field elements).

#### Output

`ShiftSumcheckProverOutput<F>` containing:
- `proof`: 9 round polynomials × 3 field elements each = **27 field elements**.
- `challenge_point`: $\mathbf{s} \in F^9$ (derived from transcript, not serialized).
- `final_claim`: the reduced claim value after all rounds (derived, not serialized).
- `h_finals`: 12 field elements (derived, not serialized — the verifier recomputes these via `eval_left_shift_predicate`).
- `v_finals`: 12 field elements — per-claim source column evaluations at $\mathbf{s}$.

**Proof data** (serialized into `SerializedShiftSumcheckProof`):
- `rounds`: $9 \times 3 = 27$ field elements (each `field_elem_size` bytes).
- `v_finals`: only the 8 **non-public** entries (claims 0, 1, 6, 7, 8, 9, 10, 11) — the 4 public-column entries (claims 2–5) are excluded because the verifier recomputes them from known column data at $\mathbf{s}$ via the split API.

#### Computational cost

The dominant cost is the 9 sumcheck rounds. Each round processes 6 group tables of decreasing size. The total work across all rounds is proportional to $6 \times (512 + 256 + 128 + \cdots + 1) = 6 \times 1023 \approx 6{,}138$ multiply-accumulate pairs for the round polynomial, plus the same for folding. The `v_finals` computation is an additional $12 \times 512 = 6{,}144$ inner-product terms. All inner loops are parallelized with rayon when the `parallel` feature is active.

### 12. `Folding/FoldClaims (2-round)`

**Function**: `fold_claims_prove::<F, _, 16>(&mut tr, &half_trace, fold_piop_point, &projecting_elem, &field_cfg)` followed by `fold_claims_prove::<F, _, 8>(&mut tr, &split_trace, &fold1.new_point, &projecting_elem, &field_cfg)`

After the PIOP produces evaluation claims on the original `BinaryPoly<32>` columns, the two-round folding protocol reduces them to claims on the double-split `BinaryPoly<8>` columns:

**Round 1 (32→16)**: For each committed column $j$, the prover sends $c_1[j] = \text{MLE}[v'_j](\mathbf{r} \| 0)$ and $c_2[j] = \text{MLE}[v'_j](\mathbf{r} \| 1)$ (the low/high half evaluations at the CPR sumcheck point extended by 0 and 1). The verifier checks $c_1[j] + \alpha^{16} \cdot c_2[j] = \text{original\_eval}[j]$, then draws $\gamma_1$ to collapse to $\text{MLE}[v'_j](\mathbf{r} \| \gamma_1)$.

**Round 2 (16→8)**: Same procedure on the `BinaryPoly<8>` split trace, producing $c_3[j], c_4[j]$ and drawing $\gamma_2$.

The final PCS point is $(\mathbf{r} \| \gamma_1 \| \gamma_2)$ — an 11-dimensional point for the 4× folded PCS.

**Output**: The folding proof values ($c_1, c_2, c_3, c_4$ per column) and the extended evaluation point.

### 13. `PCS/Prove (4x-folded)`

**Function (individual bench)**: `ZipPlus::<FoldedZt4x, FoldedLc4x>::prove::<F, UNCHECKED>(&folded_4x_params, &split_trace, &folded_4x_pcs_point, &folded_4x_hint)`

**Function (E2E)**: `ZipPlus::<PcsZt, PcsLc>::prove_with_seed::<PiopField, CHECK>(&pcs_params, &split_trace_quarter, &fold2_output.new_point, &hint, &root_buf)`

Opens the committed `BinaryPoly<8>` columns at the extended evaluation point $(\mathbf{r} \| \gamma_1 \| \gamma_2)$ — an 11-dimensional point derived from the PIOP sumcheck point (9 dims) plus the two folding challenges $\gamma_1, \gamma_2$.

> **Fidelity note**: The individual bench step uses `prove()` (no seed) and a synthetic evaluation point with placeholder $\gamma_1 = \gamma_2 = 1$. The E2E uses `prove_with_seed(&root_buf)` which absorbs the PCS commitment root into the PCS transcript before deriving its field config, and the evaluation point comes from the real folding protocol output. The PCS prove cost is point-independent, so this does not affect timing.

**Steps**:

1. **Create PCS transcript**: Initialize a separate `PcsTranscript` (Keccak sponge + byte stream), squeeze its own random field configuration. (In the E2E, the root bytes are absorbed before squeezing.)

2. **Tensor decomposition**: Split the 11-dimensional point into a row-selector $\mathbf{q}_0$ (empty, since `num_rows = 1`) and a column-selector $\mathbf{q}_1$ (all 11 coordinates).

3. **FS challenge combination**: For each committed polynomial, squeeze `degree_bound + 1` challenges $\boldsymbol{\alpha}$ from the PCS transcript. Compute a combined ring element per codeword cell: $\text{CombR}[i] = \langle \text{BinaryPoly<8>}[i], \boldsymbol{\alpha} \rangle$ (inner product of the 8 coefficients with the challenges, projecting `BinaryPoly<8>` to a scalar).

4. **Compute $\mathbf{b}$**: For each row, $b[\text{row}] = \sum_{\text{poly}} \langle \text{CombR}[\text{row}], \mathbf{q}_1 \rangle$ — inner product of the combined row with the column-selector, summed across polynomials. Absorb $\mathbf{b}$.

5. **Evaluation**: $\text{eval} = \langle \mathbf{q}_0, \mathbf{b} \rangle = b[0]$ (since there is only 1 row).

6. **Combined row**: Squeeze `num_rows` coefficients $s_j$ (just 1 here), compute the combined codeword row across all polynomials, write to transcript.

7. **Grinding**: `transcript.grind(GRINDING_BITS=8)` — proof-of-work nonce search. The prover searches for a nonce that makes `H(transcript_state \| nonce)` have 8 leading zero bits, adding 8 bits of computational security.

8. **Column openings**: Squeeze `NUM_COLUMN_OPENINGS = 131` random column indices from the transcript. For each index, open the Merkle path (authentication siblings from leaf to root, depth 13) and include the column values from all committed polynomial codeword matrices. The verifier will check: (a) the Merkle path is valid, (b) the column values are consistent with the combined row, (c) the combined row is a valid codeword.

**Output**: `(F, ZipPlusProof)` — the claimed evaluation and the serialized proof (transcript bytes including Merkle paths, column data, and grinding nonce).

---

## End-to-End Pipeline Benchmarks

### Prover benchmarks

| # | Benchmark ID | What it runs |
|---|---|---|
| 14 | `E2E/Prover (4x Hybrid GKR c=2 4-chunk)` | `prove_hybrid_gkr_logup_4x_folded::<Sha256Uair, Sha256UairQx, FoldedZt4x, FoldedLc4x, 32, 16, 8, UNCHECKED>` — full 4× folded pipeline (double-split → commit → IC (BP) → QX IC → CPR → QX CPR → Hybrid GKR Lookup → Shift SC → Fold (2-round) → PCS prove), 4-chunk. `qx_shift_sumcheck: None`. |

### Verifier benchmarks

| # | Benchmark ID | Verifier function |
|---|---|---|
| 15 | `E2E/Verifier (4x Hybrid GKR c=2 4-chunk)` | `verify_classic_logup_4x_folded::<Sha256Uair, Sha256UairQx, FoldedZt4x, FoldedLc4x, 32, 16, 8, UNCHECKED, ...>` |

For the verifier run, a one-shot timing breakdown is printed to stdout showing: IC verify, CPR+Lookup verify, Lookup verify, PCS verify, and total.

---

## E2E Verifier (`verify_classic_logup_4x_folded`)

The verifier is benchmarked as a single E2E step (step 15). It does not have individual sub-step benchmarks. The function signature is:

```rust
pub fn verify_classic_logup_4x_folded<U, U2, PcsZt, PcsLc,
    const D: usize, const HALF_D: usize, const QUARTER_D: usize, const CHECK: bool,
    IdealOverF, IdealOverFFromRef, QxIdealOverF, QxIdealOverFFromRef>(
    pcs_params, folded_proof, num_vars,
    ideal_over_f_from_ref, qx_ideal_over_f_from_ref, public_column_data,
) -> VerifyResult
```

For the SHA-256 benchmark, both ideal closures return `TrivialIdeal` (placeholder — see [§"Why selectors were removed"](#why-selectors-were-removed-placeholder)).

A one-shot timing breakdown is printed to stdout after the verifier run, covering four timer buckets: **IC verify**, **CPR+Lookup verify**, **Lookup verify**, and **PCS verify**.

### Verifier algorithm (step-by-step)

#### V0. Transcript reconstruction

1. `KeccakTranscript::new()` — fresh Fiat-Shamir transcript.
2. Absorb the PCS commitment root from `folded_proof.commitment.root` (`HASH_OUT_LEN` bytes).
3. `get_random_field_cfg::<PiopField, _, MillerRabin>()` — derive the same 192-bit prime $p$ the prover used (because the root is identical). Primality testing uses trial division by 15 small primes (3–53) as a pre-filter before the Miller-Rabin base-2 test (see [§"Verifier Optimizations"](#verifier-optimizations), V4).
4. `transcript_after_prime = transcript.clone()` — snapshot the FS transcript state immediately after prime derivation, for reuse in V6 (see [§"Verifier Optimizations"](#verifier-optimizations), V1).

This mirrors the prover's step 2. All subsequent transcript operations produce the same challenges as the prover.

#### V1. Ideal Check verify (BinaryPoly, `Sha256Uair`)

**Timer**: `ideal_check_verify`

1. **Deserialize IC proof**: Read `ic_proof_values` from the proof and reconstruct `Vec<DynamicPolynomialF<PiopField>>` — the combined MLE values.
2. **Verify**: `IdealCheckProtocol::verify_as_subprotocol::<U, _, _>(&mut transcript, ic_proof, num_constraints, num_vars, ideal_over_f_from_ref, &field_cfg)`.
   - Absorbs the MLE values into the transcript.
   - Checks that each constraint polynomial lies in the ideal (trivial for SHA-256 — always passes).
   - Squeezes a random evaluation point $\mathbf{r} \in F^{n}$ (where $n = $ `num_vars` $= 9$).
3. **Output**: `ic_subclaim` with `evaluation_point` $\mathbf{r}$.

If verification fails, the function returns early with `accepted: false`.

#### V1b. QX Ideal Check verify (`Sha256UairQx`)

**Timer**: still `ideal_check_verify`

1. **Deserialize**: Read `qx_ic_proof_values` from the proof.
2. **Verify**: `IdealCheckProtocol::verify_at_point::<U2, _, _>(...)` — verifies at the **same** evaluation point $\mathbf{r}$ produced by V1, rather than squeezing a fresh one. This matches the prover's `prove_from_binary_poly_at_point`.
3. Absorbs QX MLE values into the transcript.

Only runs when the `qx-constraints` feature is enabled. For SHA-256, both ideals are trivial.

#### V2. CPR verify (standalone sumcheck, `Sha256Uair`)

**Timer**: `combined_poly_resolver_verify`

The verifier takes the **non-batched path** (since `lookup_proof` is `HybridGkr`, not `BatchedClassic`):

1. **Deserialize CPR proof**: Reconstruct `SumcheckProof { messages, claimed_sum }` from `cpr_sumcheck_messages` and `cpr_sumcheck_claimed_sum`.
2. **Deserialize CPR evals**: `cpr_up_evals` (private column evaluations) and `cpr_down_evals` (shifted column evaluations).
3. **Draw projecting element** $\alpha$: `transcript.get_field_challenge(&field_cfg)` — same $\alpha$ the prover used.
4. **Project scalars**: `project_scalars::<PiopField, U>(closure)` → `project_scalars_to_field(...)`.
5. **Pre-sumcheck check**: `CombinedPolyResolver::build_verifier_pre_sumcheck::<U>(...)` — absorbs the claimed sum and prepares the verification state.
6. **Sumcheck verification**: `MLSumcheck::verify_as_subprotocol(&mut transcript, num_vars, max_degree + 2, &cpr_sumcheck_proof, &field_cfg)` — verifies all 9 round messages (degree 3 each), squeezing one challenge per round, producing a `subclaim { point, expected_evaluation }`.
7. **Reconstruct full up_evals**: If the UAIR has public columns, compute their MLE evaluations at the sumcheck point using `public_column_data`, then call `reconstruct_up_evals` to interleave public and private evaluations.
8. **Finalize**: `CombinedPolyResolver::finalize_verifier::<U>(...)` — checks that the combined polynomial evaluation at the subclaim point equals the expected value, using the up/down evals and projected scalars.

#### V2b. QX CPR verify (standalone sumcheck, `Sha256UairQx`)

**Timer**: still `combined_poly_resolver_verify`

Same 3-step pattern as V2 (pre-sumcheck → `MLSumcheck::verify_as_subprotocol` → finalize) but for the QX sub-UAIR:

1. Deserialize `qx_cpr_sumcheck_messages`, `qx_cpr_sumcheck_claimed_sum`, `qx_cpr_up_evals`, `qx_cpr_down_evals`.
2. `build_verifier_pre_sumcheck::<U2>(...)` with the QX IC subclaim.
3. `MLSumcheck::verify_as_subprotocol(...)` — 9 rounds, degree `qx_max_degree + 2 = 3`.
4. `finalize_verifier::<U2>(...)` with QX up/down evals.

Only runs when `qx-constraints` is enabled.

#### V3. Hybrid GKR Lookup verify

**Timer**: `lookup_verify`

Dispatches to `verify_hybrid_gkr_batched_lookup` based on the `LookupProofData::HybridGkr` variant:

1. **Regenerate subtable**: `generate_bitpoly_table(chunk_width=8, α)` → 256-entry subtable and 4 shifts ($\alpha^0, \alpha^8, \alpha^{16}, \alpha^{24}$).
2. **Absorb multiplicities**: The prover sent aggregated multiplicities (one per subtable entry) — absorb them into the transcript.
3. **Verify GKR layers**: For each layer from root to cutoff $c = 2$:
   - Squeeze challenge $\beta$.
   - Verify the degree-3 GKR sumcheck (one round per layer at the top, multiple rounds for deeper).
   - Cross-check: child evaluations must satisfy the binary fraction relation.
4. **Sent intermediates**: At cutoff $c = 2$, the prover sent $2L \times 2^c$ field elements (the partial products). The verifier absorbs these and verifies consistency with the GKR root claims.
5. **Fresh GKR (bottom layers)**: From the cutoff layer down to the leaves, run a standard GKR verification for the witness tree and table tree separately.
6. **Leaf check**: Verify that the GKR leaf evaluations are consistent with the claimed MLE evaluations (the lookup column values and subtable evaluations).
7. **Multiplicity sum check**: Verify that the total multiplicity sums are consistent across all lookup groups.

#### V4. Shift sumcheck verify

**Timer**: included in `combined_poly_resolver_verify` (same bucket as CPR)

1. **Reconstruct claims**: Build `ShiftClaim` for each of the 12 shift specs from `sig.shifts`, using `cpr_down_evals` as claimed evaluations and `cpr_subclaim.evaluation_point[..num_vars]` as the eval point.
2. **Deserialize proof**: Reconstruct `ShiftSumcheckProof { rounds }` (9 rounds × 3 evaluations each) and the private `v_finals`.
3. **Split-API verify** (used when public shifts exist, which SHA-256 has):
   - `shift_sumcheck_verify_pre(...)` — advances the transcript through all sumcheck rounds, producing the challenge point $\mathbf{s}$.
   - Compute public `v_finals`: For each public shift source column, evaluate the column's MLE at $\mathbf{s}$ using `public_column_data` (with the binary-poly $\alpha$-projection).
   - `reconstruct_shift_v_finals` — interleave public and private v_finals.
   - `shift_sumcheck_verify_finalize(...)` — check final claim consistency: each $v_\text{final}[i]$ must satisfy the shift relation at $\mathbf{s}$.

> **Note:** The E2E prover sets `qx_shift_sumcheck: None`, so no QX shift sumcheck verification is performed.

#### V5. Two-round folding verify

**Timer**: included in `combined_poly_resolver_verify`

1. **Collect original evals**: Extract the per-column evaluations from `cpr_up_evals` for PCS-committed columns only (filter out public + PCS-excluded). If `evaluation_point.len() > num_vars` (shared num_vars padding), divide out the zero-padding factor $\prod_{k=n}^{N-1}(1 - r_k)$.
2. **Deserialize folding data**: Read $c_1, c_2$ (round 1) and $c_3, c_4$ (round 2) from the proof bytes.
3. **Round 1 (D→HALF_D)**: `fold_claims_verify(&mut transcript, &c1s, &c2s, &original_evals, &alpha_power_half, piop_point, &field_cfg)`:
   - Absorb $c_1, c_2$ into the transcript.
   - For each column $j$, verify: $c_1[j] + \alpha^{16} \cdot c_2[j] = \text{original\_eval}[j]$.
   - Squeeze $\gamma_1$.
   - Compute new evals: $\text{new\_eval}[j] = c_1[j] \cdot (1 - \gamma_1) + c_2[j] \cdot \gamma_1$.
   - Produce `new_point = piop_point ‖ γ₁`.
4. **Round 2 (HALF_D→QUARTER_D)**: `fold_claims_verify(&mut transcript, &c3s, &c4s, &fold1.new_evals, &alpha_power_quarter, &fold1.new_point, &field_cfg)`:
   - Same logic, verifying: $c_3[j] + \alpha^{8} \cdot c_4[j] = \text{fold1\_eval}[j]$.
   - Squeeze $\gamma_2$.
   - Produce `new_point = piop_point ‖ γ₁ ‖ γ₂` (11-dimensional).

#### V6. PCS verify

**Timer**: `pcs_verify`

1. **Reuse transcript snapshot**: Create `PcsTranscript { fs_transcript: transcript_after_prime, stream: Cursor<proof_bytes> }`, reusing the FS transcript state saved in V0 step 4 (see [§"Verifier Optimizations"](#verifier-optimizations), V1). This skips the redundant primality search that would otherwise cost ~0.3–0.4 ms.
2. **Reuse field config**: `pcs_field_cfg = field_cfg.clone()` — the PIOP and PCS field configs are identical since both transcripts derive the same prime from the same initial state.
4. **Lift point and eval**: Convert `fold2_output.new_point` (11 coordinates) and the deserialized evaluation to the PCS field's Montgomery representation.
5. **Verify**: `ZipPlus::<PcsZt, PcsLc>::verify_with_field_cfg::<PiopField, CHECK>(pcs_params, &commitment, &point_f, &eval_f, pcs_transcript, &pcs_field_cfg)`:
   - Reads `NUM_COLUMN_OPENINGS` (131) column indices from the proof stream.
   - For each opened column: reads a Merkle authentication path (depth 13) and the codeword values, verifies the path against the committed root.
   - Checks FS combination consistency: the combined scalar matches the expected evaluation at the point.
   - Verifies the proximity relation (encoding consistency via tensor checks + code structure).
   - Checks the **grinding proof**: `GRINDING_BITS` (8) leading zero bits in a hash, ensuring proof-of-work.

### Accept/reject

The verifier returns `VerifyResult { accepted, timing }`. Acceptance requires all sub-steps to pass (IC, QX IC, CPR, QX CPR, lookup, shift SC, folding, PCS). Any failure causes an early return with `accepted: false` and partial timing.

---

## Proof Size Analysis

The benchmark computes and prints a detailed proof size breakdown for the **4× folded Hybrid GKR c=2** pipeline variant. The proof type is `Folded4xZincProof`. The components measured are:

| Component | Description |
|---|---|
| **PCS** | Zip+ opening proof bytes (Merkle paths, column data, consistency hashes, grinding nonce) |
| **IC** | Ideal Check proof for `Sha256Uair` (BP constraints) — combined MLE values at the random point |
| **QX IC** | Ideal Check proof for `Sha256UairQx` (Q[X] carry constraints, trivial ideal) |
| **CPR sumcheck** | Main field sumcheck round messages + claimed sum (standalone `MLSumcheck`, degree 3) |
| **CPR evals** | Up-evaluations and down-evaluations for `Sha256Uair` |
| **QX CPR** | Standalone sumcheck + evals for `Sha256UairQx` |
| **Lookup** | Hybrid GKR c=2 proof data: aggregated multiplicities, witness-side GKR (roots, layer proofs, sent intermediates at cutoff), table-side GKR (root, layer proofs) |
| **Shift SC** | Shift sumcheck proof for `Sha256Uair` — $m$ degree-2 round polynomials + per-claim `v_finals`. For SHA-256, 12 shift claims produce 9 round polys and 8 private `v_finals` (4 are public and excluded). See §8a for details. |
| **QX Shift SC** | `None` — the E2E pipeline sets `qx_shift_sumcheck: None`; no QX shift sumcheck is performed or serialized |
| **Folding** | $c_1, c_2$ values per column (round 1: 32→16) + $c_3, c_4$ values per column (round 2: 16→8) |
| **Eval point** | Serialized evaluation point coordinates |
| **PCS evals** | Serialized claimed PCS evaluation values |

All bytes are concatenated and compressed with Deflate (default compression level) to show the **compressed proof size**, which is the metric that matters for transmission. A single breakdown table is printed (no multi-variant comparison tables).

---

## GKR and Hybrid GKR Lookup Protocols

The **GKR lookup** and its **Hybrid GKR** optimisation are alternative lookup protocols that replace the classic batched decomposition LogUp described in Step 10. They are based on the GKR fractional sumcheck from *Papini & Haböck, "Improving logarithmic derivative lookups using GKR"* ([ePrint 2023/1284](https://eprint.iacr.org/2023/1284)).

### How the GKR lookup differs from classic LogUp (Step 10)

In the **classic** variant (`BatchedDecompLogupProtocol`), the prover sends three families of auxiliary vectors in the clear:

| Classic proof data | Size |
|---|---|
| Aggregated multiplicities $m_{\text{agg}}^{(\ell)}$ | $L \times T$ field elements |
| Chunk inverse witnesses $u_k^{(\ell)}[i] = 1/(\beta - c_k^{(\ell)}[i])$ | $L \times K \times W$ field elements |
| Shared inverse table $v[t] = 1/(\beta - T[t])$ | $T$ field elements |

The identity $H[i] = 0$ (batching inverse-correctness and balance checks) is then verified via a single degree-2 sumcheck over the precomputed polynomial $H$. This sumcheck is **batched with CPR** in the multi-degree sumcheck (`MultiDegreeSumcheck`) — CPR contributes a degree-3 group and lookup contributes a degree-2 group sharing the same random challenges.

In the **GKR** variant (`GkrBatchedDecompLogupProtocol`), the inverse vectors $u_k, v$ are **eliminated entirely**. Only the aggregated multiplicities are sent. Instead, the prover constructs a binary **fraction tree** and runs a layer-by-layer GKR sumcheck to prove that the fractional sums balance. The resulting GKR proof is $O(L \cdot d^2)$ field elements where $d = \log_2(K \cdot W)$, which is much smaller than the classic $O(L \cdot K \cdot W)$ inverse vectors for large traces.

A key architectural difference: **the GKR lookup runs as a separate sumcheck**, not batched with CPR in the multi-degree sumcheck. The pipeline function `prove_hybrid_gkr_logup_4x_folded` runs CPR as a standalone degree-3 `MLSumcheck` first, then runs the Hybrid GKR fractional sumcheck independently.

### The GKR fractional-sum tree

The core data structure is a binary tree of fractions. Given $2^d$ leaf fractions $p_i / q_i$, the tree has $d + 1$ layers:

- **Layer $d$ (leaves)**: $2^d$ entries, each $(p_i, q_i)$.
- **Layer $k$ (internal)**: $2^k$ entries, formed by adding pairs of fractions from layer $k+1$:

$$p_k[i] = p_{k+1}[i] \cdot q_{k+1}[i + 2^k] + p_{k+1}[i + 2^k] \cdot q_{k+1}[i]$$
$$q_k[i] = q_{k+1}[i] \cdot q_{k+1}[i + 2^k]$$

- **Layer 0 (root)**: 1 entry, $(P_{\text{root}}, Q_{\text{root}})$, representing the overall sum $\sum_i p_i / q_i = P_{\text{root}} / Q_{\text{root}}$.

This tree is built bottom-up by `build_fraction_tree()` in [piop/src/lookup/gkr_logup.rs](piop/src/lookup/gkr_logup.rs). The returned array is indexed `[leaves, ..., root]`.

### Per-tree GKR layer protocol

The GKR protocol proves layer-by-layer from root to leaves that the parent values are consistent with the child values. At each layer $k$ (processing GKR round $k$ from $0$ to $d-1$):

**Round 0 ($k = 0$):** The root has two children $(p_L, q_L)$ and $(p_R, q_R)$. The prover sends all four values. The verifier draws a batching challenge $\alpha$ and checks the algebraic identity:

$$P_{\text{root}} + \alpha \cdot Q_{\text{root}} \stackrel{?}{=} (p_L \cdot q_R + p_R \cdot q_L) + \alpha \cdot (q_L \cdot q_R)$$

Then draws a random-linear-combination challenge $\lambda$ and sets the running claim:

$$v_p \leftarrow (1 - \lambda) \cdot p_L + \lambda \cdot p_R, \quad v_q \leftarrow (1 - \lambda) \cdot q_L + \lambda \cdot q_R$$

**Round $k \geq 1$:** The layer has $2^k$ children on each side. The prover builds $1+4$ MLEs over $k$ variables — the eq polynomial $\text{eq}(\mathbf{r}_k, \cdot)$ and the four child-half MLEs $(\tilde{p}_L, \tilde{q}_L, \tilde{p}_R, \tilde{q}_R)$ — then runs a degree-3 sumcheck proving:

$$\sum_{\mathbf{x} \in \{0,1\}^k} \text{eq}(\mathbf{r}_k, \mathbf{x}) \cdot \bigl[(\tilde{p}_L(\mathbf{x}) + \alpha \cdot \tilde{q}_L(\mathbf{x})) \cdot \tilde{q}_R(\mathbf{x}) + \tilde{p}_R(\mathbf{x}) \cdot \tilde{q}_L(\mathbf{x})\bigr] = v_p + \alpha \cdot v_q$$

After the sumcheck, the prover reveals the four child evaluations at the subclaim point $\mathbf{s}$. The verifier checks consistency, draws $\lambda$, and updates the running claim. The evaluation point grows: $\mathbf{r}_{k+1} = (\mathbf{s}, \lambda)$.

After all $d$ rounds, the verifier holds an evaluation point $\mathbf{r} \in F^d$ and expected values $(\tilde{p}(\mathbf{r}), \tilde{q}(\mathbf{r}))$ at the leaf layer. These are checked against the known leaf data (chunk columns from PCS, or direct computation for the table side).

**Implementation**: `gkr_fraction_prove` / `gkr_fraction_verify` in [piop/src/lookup/gkr_logup.rs](piop/src/lookup/gkr_logup.rs).

### Batched GKR for $L$ witness trees

When there are $L$ lookup columns, the protocol builds $L$ separate witness fraction trees (one per column) plus one combined table fraction tree. The **witness-side** trees are batched layer-by-layer into a single sumcheck per layer using a tree-batching challenge $\delta$:

- At each layer, the sumcheck has $1 + 4L$ MLEs (one shared eq, plus 4 per tree).
- The combination function is:

$$f(\text{eq}, \{p_L^{(\ell)}, q_L^{(\ell)}, p_R^{(\ell)}, q_R^{(\ell)}\}_\ell) = \text{eq} \cdot \sum_{\ell=0}^{L-1} \delta^\ell \cdot \bigl[(p_L^{(\ell)} + \alpha \cdot q_L^{(\ell)}) \cdot q_R^{(\ell)} + p_R^{(\ell)} \cdot q_L^{(\ell)}\bigr]$$

- Degree is still 3 (eq × cubic expression of the 4L MLE values).
- All $L$ trees share the same evaluation point trajectory $\mathbf{r}_k$.

The **table-side** tree uses an $\alpha$-batching of multiplicities across all $L$ lookups:

- Leaf numerator: $p_{\text{table}}[j] = \sum_\ell \alpha^\ell \cdot m_{\text{agg}}^{(\ell)}[j]$
- Leaf denominator: $q_{\text{table}}[j] = \beta - T[j]$

The table tree runs a standard (non-batched) GKR fractional sumcheck.

**Cross-check**: After both sides complete, the verifier checks the root identity:

$$\Bigl(\sum_\ell \alpha^\ell \cdot P_w^{(\ell)} \cdot \prod_{j \neq \ell} Q_w^{(j)}\Bigr) \cdot Q_t \stackrel{?}{=} P_t \cdot \prod_\ell Q_w^{(\ell)}$$

This is the cross-multiplication form of $\sum_\ell \alpha^\ell \cdot P_w^{(\ell)} / Q_w^{(\ell)} = P_t / Q_t$.

**Leaf verification**: At the witness-tree leaves, each tree has $p^{(\ell)}[(k,i)] = 1$ (all-ones numerator for each chunk/row pair) and $q^{(\ell)}[(k,i)] = \beta - c_k^{(\ell)}[i]$ (shifted chunk values from PCS commitments). The table leaf verification directly evaluates the multiplicity/subtable MLEs. These leaf checks produce evaluation claims that the PCS resolves.

**Implementation**: `batched_gkr_fraction_prove` / `batched_gkr_fraction_verify` in [piop/src/lookup/gkr_logup.rs](piop/src/lookup/gkr_logup.rs). The full protocol is `GkrBatchedDecompLogupProtocol::prove_as_subprotocol` / `verify_as_subprotocol` in [piop/src/lookup/gkr_batched_decomposition.rs](piop/src/lookup/gkr_batched_decomposition.rs).

### GKR proof data structures

**`GkrFractionProof<F>`** (single tree, used for the table side): root $(P, Q)$, plus one `GkrLayerProof` per layer. Each `GkrLayerProof` contains an optional `SumcheckProof` ($k$ degree-3 round messages for layer $k \geq 1$; `None` for layer 0) and four child evaluations $(p_L, p_R, q_L, q_R)$.

**`BatchedGkrFractionProof<F>`** ($L$ witness trees): per-tree roots $(P_w^{(\ell)}, Q_w^{(\ell)})$, plus one `BatchedGkrLayerProof` per layer. Each layer proof has one shared `SumcheckProof` and per-tree evaluation quadruples: $4L$ field elements per layer.

**`GkrBatchedDecompLogupProof<F>`** (complete GKR lookup proof): aggregated multiplicities ($L \times T$ field elements), a `BatchedGkrFractionProof` for the witness side, and a `GkrFractionProof` for the table side. Chunks are NOT included (bound by the PCS commitment).

All defined in [piop/src/lookup/structs.rs](piop/src/lookup/structs.rs).

### Sumcheck round count

For a single tree of depth $d$, layers $1, 2, \ldots, d-1$ contribute sumchecks with $1, 2, \ldots, d-1$ variables respectively (layer 0 is a direct algebraic check). Total sumcheck rounds:

$$\text{rounds}(d) = \sum_{k=1}^{d-1} k = \frac{d(d-1)}{2}$$

For $L$ batched witness trees of depth $d_w$ plus a single table tree of depth $d_t$:

$$\text{total rounds} = \frac{d_w(d_w-1)}{2} + \frac{d_t(d_t-1)}{2}$$

For SHA-256 with 13 lookup columns (10 base + 3 affine), $K=8$ chunks of width 4, $W=512$ rows: $d_w = \lceil\log_2(8 \times 512)\rceil = 12$, $d_t = 4$. Full GKR uses $66 + 6 = 72$ sumcheck rounds. The round count depends on $d_w$ and $d_t$, not $L$; adding more trees only increases the per-layer proof size.

### The Hybrid GKR optimization

The **Hybrid GKR** protocol (`HybridGkrBatchedDecompLogupProtocol`) exploits the observation that the lower GKR layers have escalating sumcheck sizes (layer $k$ has a $k$-variable sumcheck), so the cost is concentrated in the bottom layers. The hybrid approach:

1. **Top $c$ layers via GKR**: Run the batched GKR sumcheck for layers $0, 1, \ldots, c-1$ from the root, exactly as in the full GKR. This produces an evaluation point $\mathbf{r}_c \in F^c$.

2. **Send intermediates in the clear**: At the cutoff layer $c$, the prover sends the entire intermediate fraction vectors for each tree: $\{\text{sent\_p}^{(\ell)}[j], \text{sent\_q}^{(\ell)}[j]\}$ for $j \in [2^c]$. These are $2L \times 2^c$ field elements total. They are absorbed into the transcript **before** challenges are derived (ensuring commitment before seeing $\mathbf{r}_c$).

3. **Verifier MLE check**: The verifier evaluates the MLE of the sent vectors at $\mathbf{r}_c$ and checks consistency with the running claims $v_p^{(\ell)}, v_q^{(\ell)}$ from the top GKR layers:

$$\tilde{p}_c^{(\ell)}(\mathbf{r}_c) \stackrel{?}{=} v_p^{(\ell)}, \quad \tilde{q}_c^{(\ell)}(\mathbf{r}_c) \stackrel{?}{=} v_q^{(\ell)}$$

This replaces layers $c, c+1, \ldots, d-1$ of the GKR entirely for the purpose of linking root to intermediates.

4. **Bottom verification**: The sent intermediate values at layer $c$ must still be shown consistent with the committed leaf data (chunks). This is application-specific — the current implementation supports running a **fresh GKR** of depth $d - c$ from the intermediates down to the leaves, where sumcheck variables restart at sizes $1, 2, \ldots, d - c - 1$ instead of continuing from $c, c+1, \ldots, d-1$. This is the key savings: a fresh depth-$(d-c)$ GKR uses $\frac{(d-c)(d-c-1)}{2}$ rounds versus the $\sum_{k=c}^{d-1} k = \frac{(d-c)(d+c-1)}{2}$ rounds of the continuing GKR.

**Cost tradeoff at cutoff $c$** (tree depth $d$, $L$ batched trees):

| Component | Field elements |
|---|---|
| Top GKR roots | $2L$ |
| Top GKR layer proofs (layers 0..c-1) | $4L \times c$ evaluations + sumcheck messages for rounds $1, \ldots, c-1$ |
| Sent intermediates | $2L \times 2^c$ |
| Bottom fresh GKR roots | $2L$ |
| Bottom GKR layer proofs (depth $d - c$) | $4L \times (d-c)$ evaluations + sumcheck messages for rounds $1, \ldots, d-c-1$ |

The table-side GKR always runs in full (small depth $d_t$).

### Hybrid GKR proof data structures

**`HybridBatchedGkrFractionProof<F>`**: per-tree roots, per-layer `HybridBatchedGkrLayerProof` entries for the top $c$ layers, the `cutoff` depth, and the sent intermediate vectors `sent_p[ℓ][j]` / `sent_q[ℓ][j]` ($2L \times 2^c$ field elements).

**`HybridGkrBatchedDecompLogupProof<F>`**: aggregated multiplicities ($L \times T$), a `HybridBatchedGkrFractionProof` for the witness side, and a full `GkrFractionProof` for the table side.

All defined in [piop/src/lookup/hybrid_gkr.rs](piop/src/lookup/hybrid_gkr.rs).

### Pipeline integration

This benchmark uses a single pipeline entry point:

| Pipeline function | Folding | Lookup variant | Proof type |
|---|---|---|---|
| `prove_hybrid_gkr_logup_4x_folded` | 4× | Hybrid GKR (cutoff $c = 2$) | `Folded4xZincProof` with `LookupProofData::HybridGkr` |

The verifier is `verify_classic_logup_4x_folded`, which matches on the `LookupProofData` enum and dispatches to `verify_hybrid_gkr_batched_lookup`. These are defined in [piop/src/lookup/pipeline.rs](piop/src/lookup/pipeline.rs) and integrated in [snark/src/pipeline.rs](snark/src/pipeline.rs).

The pipeline step sequence for the 4× folded Hybrid GKR c=2 variant is:

1. Double-split columns (32→16→8) → 2. PCS Commit (4x-folded) → 3. IC (BinaryPoly, `Sha256Uair`) → 4. QX IC (`Sha256UairQx`, trivial ideal) → 5. CPR (standalone `MLSumcheck`, degree 3) → 6. QX CPR (standalone) → 7. Hybrid GKR Lookup → 8. Shift sumcheck (BP) → 9. Folding (2-round) → 10. PCS Prove.

> **Note:** The E2E prover sets `qx_shift_sumcheck: None` — no QX shift sumcheck is executed. The QX sub-UAIR only runs IC (step 4) and CPR (step 6); it does not have its own shift sumcheck.

The Hybrid GKR c=2 approach runs CPR as a standalone sumcheck followed by the Hybrid GKR lookup as a separate step (no multi-degree batching). The QX sub-UAIR (`Sha256UairQx`) gets its own IC and CPR — running independently from the BinaryPoly sub-UAIR. It does **not** get a shift sumcheck (`qx_shift_sumcheck: None`).

### Concrete numbers for SHA-256 8× (13 lookup columns, 4 chunks width-8, 512 rows)

The E2E pipeline includes 10 base $Q[X]$ lookups plus 3 affine lookups (Ch, neg-Ch, Maj), giving $L = 13$ total lookup columns. The individual step 10 benchmark uses only the 10 base lookups.

| Parameter | Value |
|---|---|
| $L$ (lookup columns, E2E) | 13 (10 base + 3 affine) |
| $K$ (chunks) | 4 (`chunk_width=8`) |
| $W$ (witness rows) | 512 |
| Leaf count per witness tree | $K \times W = 2048$, padded to $2^{11}$ |
| Witness tree depth $d_w$ | 11 |
| Subtable size $T$ | $2^8 = 256$ |
| Table tree depth $d_t$ | 8 |
| Hybrid GKR ($c = 2$) top rounds | $1$ (layer 1 only; layer 0 is a direct check) |
| Hybrid GKR ($c = 2$) bottom fresh depth | $11 - 2 = 9$, bottom rounds $= \frac{9 \times 8}{2} = 36$ |
| Hybrid GKR ($c = 2$) sent intermediates | $2 \times 13 \times 4 = 104$ field elements |
| Hybrid GKR ($c = 2$) total witness rounds | $1 + 36 = 37$ |
| Table GKR total rounds | $\frac{8 \times 7}{2} = 28$ |

---

## Hybrid GKR Cost Analysis

> **Note:** This section documents the hybrid GKR cost model for reference. The `steps_sha256_8x_folded` benchmark exercises the Hybrid GKR c=2 pipeline via `prove_hybrid_gkr_logup_4x_folded` in the E2E prover (step 14) and verifier (step 15).

The general cost model for the hybrid GKR LogUp tradeoff described in the preceding section is as follows:

- **Full GKR**: All $d$ layers of the GKR tree are proven via sumcheck. Proof size is $O(L \cdot d^2)$ field elements (sumcheck messages grow per layer) but no auxiliary vectors are sent.
- **Hybrid (cutoff $c$)**: The top $c$ layers are proven via GKR sumcheck, then the $2L \times 2^c$ intermediate fraction values at layer $c$ are sent in the clear. The bottom $d - c$ layers form a fresh (shorter) GKR of depth $d - c$. Sending intermediates trades $O(L \cdot 2^c)$ field elements for a reduction from $\frac{d(d-1)}{2}$ to $\frac{c(c-1)}{2} + \frac{(d-c)(d-c-1)}{2}$ sumcheck rounds.

The benchmark:
1. Prints a cost analysis table with columns: cutoff, top-only proof bytes, full hybrid bytes, top/bottom sumcheck rounds, total rounds, rounds saved, MLE eval ops, sent field elements.
2. **(Not benchmarked here.)** The hybrid GKR prove/verify microbenchmarks at different cutoffs are available in other benchmarks that instantiate the hybrid GKR pipeline.

---

## Implementation Optimizations

The following optimizations have been applied to the proving stack. All affect shared library code and therefore benefit the **4× folded Hybrid GKR c=2** pipeline benchmarked here. The same optimizations also apply to the dual-circuit benchmark ([BENCHMARK_SHA256_8X_ECDSA.md](BENCHMARK_SHA256_8X_ECDSA.md)).

### 1. Ideal Check: shared $\text{eq}(\mathbf{r}, \cdot)$ table

**Location**: `piop/src/ideal_check/combined_poly_builder.rs` — `compute_combined_values_mle_first`.

The MLE-first Ideal Check evaluates every trace column and every shifted column at the same random point $\mathbf{r}$. Each inner-product evaluation $\text{MLE}[v](\mathbf{r}) = \sum_{\mathbf{b}} \text{eq}(\mathbf{r}, \mathbf{b}) \cdot v[\mathbf{b}]$ requires the equality table $\text{eq}(\mathbf{r}, \cdot)$, which has $2^n$ entries.

Previously, the eq table was rebuilt independently for each column evaluation. Now a **single precomputed table** is built once via `build_eq_x_r_inner(evaluation_point)` in $O(n)$ and reused across all 30 column evaluations and all 12 shifted-column evaluations. The new functions `evaluate_binary_poly_column_mle_with_eq` and `evaluate_shifted_binary_poly_column_mle_with_eq` accept the precomputed table as a parameter.

> **Note:** Both `Sha256Uair` and `Sha256UairQx` have `max_degree == 1` (selectors removed from QX constraints, trivial ideal), so the pipeline uses the **MLE-first** IC path for all constraints. The shared eq-table optimisation described here is central to the MLE-first path’s performance.

**Impact**: IC prover time reduced from ~3.2 ms to ~0.82 ms (~4× speedup) for `max_degree == 1` UAIRs.

### 2. Ideal Check: parallel shifted-column evaluation

**Location**: same function.

The shifted-column ("down") evaluations are now parallelized across shift specs via `cfg_iter!(sig.shifts)`. For SHA-256 with 12 shift specs, this distributes the 12 shifted MLE evaluations across Rayon threads.

### 3. PCS commit: parallel encoding across polynomials

**Location**: `zip-plus/src/pcs/phase_commit.rs` — `ZipPlus::commit()`.

The per-polynomial IPRS linear code encoding loop was changed from `polys.iter().map(...)` (sequential) to `cfg_iter!(polys).map(...)` (parallel via Rayon). Each committed polynomial's rows are independently encoded through the NTT-based linear code (`encode_rows`), expanding inputs to 4× codeword lengths. With 10 SHA-256 committed columns, this provides up to 10-way parallelism. The Merkle tree construction (which interleaves all columns) still runs sequentially after encoding.

**Impact**: PCS commit time (folded) reduced from ~5.9 ms to ~2.7 ms (~2× speedup).

### 4. Shift sumcheck: Rayon minimum parallel batch size

**Location**: `piop/src/shift_sumcheck/prover.rs`.

The two inner `into_par_iter()` loops in the shift sumcheck round computation now use `.with_min_len(256)` to prevent Rayon from spawning threads for trivially small work units. At early rounds the table halves are large enough to benefit from parallelism; at later rounds the overhead of thread dispatch exceeds the savings. The minimum batch size of 256 ensures sequential execution once tables shrink below that threshold.

### 5. Commitment absorption into Fiat-Shamir transcript

**Location**: `snark/src/pipeline.rs` (all prove/verify pipeline functions) and `zip-plus/src/pcs/phase_prove.rs` (`prove_with_seed`).

The PCS commitment (Merkle root, `HASH_OUT_LEN` bytes) is now absorbed into **both** the PIOP and PCS Fiat-Shamir transcripts **before** `get_random_field_cfg()`. This makes the randomly derived prime depend on the commitment, preventing a prover from choosing polynomials that exploit a specific field prime.

- **PIOP transcript**: Each pipeline function absorbs the root via `commitment.root.write_transcription_bytes` → `transcript.absorb(&root_buf)` before the PIOP's `get_random_field_cfg()`.
- **PCS transcript**: Pipeline provers call `ZipPlus::prove_with_seed(…, &root_buf)` (new API in `phase_prove.rs`), which absorbs the seed into the PCS transcript before its own `get_random_field_cfg()`. Pipeline verifiers absorb the root into the reconstructed PCS transcript before deriving `pcs_field_cfg`.

Because both PIOP and PCS transcripts absorb the same root bytes from the same initial (empty) state, they deterministically derive the **same** prime. This preserves the existing invariant that PIOP and PCS field elements share a common Montgomery representation.

### 6. Lookup: fraction tree leaf-level optimization for all-ones multiplicities

**Location**: `piop/src/lookup/gkr_logup.rs` — new function `build_fraction_tree_ones_leaf`; callers in `piop/src/lookup/hybrid_gkr.rs` and `piop/src/lookup/gkr_batched_decomposition.rs`.

In the GKR-based LogUp protocol, each lookup column builds a **fraction tree** of depth $d$ from leaf fractions $p_i / q_i$, where $p_i$ is the multiplicity and $q_i = \beta - w_i$ is the denominator. At each internal node, the combination formula is:

$$p_{\text{parent}} = p_L \cdot q_R + p_R \cdot q_L, \qquad q_{\text{parent}} = q_L \cdot q_R$$

This costs **3 field multiplications + 1 addition** per node. The leaf level (layer 0 → layer 1) is the widest, containing $2^{d-1}$ nodes — half the total tree work.

For LogUp witness trees, each witness element appears with multiplicity 1, so **all leaf $p$-values are identically one** (when no zero-padding is needed, i.e. when the leaf count $K \times W$ is already a power of two). In this case the leaf-level formula simplifies:

$$p_{\text{parent}} = 1 \cdot q_R + 1 \cdot q_L = q_L + q_R, \qquad q_{\text{parent}} = q_L \cdot q_R$$

This replaces 2 field multiplications with a single addition at every leaf-level node, reducing the cost to **1 multiplication + 1 addition** per node — a 2× speedup at the widest tree layer.

The new function `build_fraction_tree_ones_leaf(one, leaf_q)` implements this optimization. It applies the simplified formula for the first tree level and falls back to the standard formula for all subsequent levels (where $p$ values are non-trivial sums). The callers in `hybrid_gkr.rs` and `gkr_batched_decomposition.rs` detect when `per_lookup_leaves == w_size` (no padding) and use the optimized path.

For SHA-256 with $L = 13$ lookup columns, $K = 4$ chunks, $W = 512$ rows: leaf count $= 4 \times 512 = 2048 = 2^{11}$ (exactly a power of two, no padding). Each tree has $2^{10} = 1024$ leaf-level nodes. Savings: $1024 \times 2 = 2048$ Montgomery multiplications saved per tree, across 13 trees.

**Impact**: Lookup prover time reduced by ~1.5 ms (from ~4.9 ms to ~3.4 ms in single-run timing).

### 7. Hybrid GKR: proper zero construction in combination function

**Location**: `piop/src/lookup/hybrid_gkr.rs` — `prove_as_subprotocol`, combination function closure.

The GKR sumcheck combination function computes $\text{comb}(\mathbf{vals}) = \sum_{\ell} p_\ell \cdot q_{\ell+1} \cdots q_L - q_\ell \cdot (\ldots)$. The initial accumulator was computed as `vals[0].clone() - eq_val`, which relies on the algebraic identity that `eq_val - eq_val == 0`. This was replaced with an explicit `F::zero_with_cfg(field_cfg)` construction, avoiding one unnecessary field subtraction per evaluation point and improving code clarity.

### 8. Hybrid GKR: clone avoidance in layer proof construction

**Location**: `piop/src/lookup/hybrid_gkr.rs` — `prove_as_subprotocol`, layer proof assembly (both $k = 0$ and $k \geq 1$ branches).

When building each GKR layer proof, the vectors `p_lefts`, `p_rights`, `q_lefts`, `q_rights` were previously **cloned** into the `GkrLayerProof` struct before being used to update the running state (`v_ps`, `v_qs`). Since the state update only reads these vectors (via `evaluate_mle_from_table`), the code was restructured to:

1. Compute the `v_ps`/`v_qs` updates **first** (consuming read references).
2. **Move** the vectors into `GkrLayerProof` (no allocation, no copy).

This eliminates 4 vector clones per GKR layer. For cutoff $c = 2$ with 1 top sumcheck layer, this saves 4 clones of vectors sized $2^{d-c} = 2^9 = 512$ per lookup column.

### 9. Shift sumcheck: parallel inner folding loop

**Location**: `piop/src/shift_sumcheck/prover.rs` — `shift_sumcheck_prove`, table folding step.

After each sumcheck round, the predicate tables $h$ and $w$ are folded by the challenge value $s$:

$$h[j] \leftarrow (1 - s) \cdot h[j] + s \cdot h[j + \text{half}], \qquad j = 0, \ldots, \text{half} - 1$$

Previously, the inner folding loop over $j$ was sequential within each group, even when the `parallel` feature was enabled (only the 2-way group-level parallelism was exploited). The inner loop was restructured using `split_at_mut` + `par_iter_mut` with `.with_min_len(128)` to enable Rayon work-stealing across the table elements. This allows additional cores to participate during early rounds when tables are large (half = 256 at round 0 for $n = 9$).

### 10. Pipeline: parallel shifted-column evaluation

**Location**: `snark/src/pipeline.rs` — `prove_hybrid_gkr_logup_4x_folded`, down-evals computation.

The MLE evaluations for shifted trace columns (`down_evals`) were computed sequentially via `sig.shifts.iter()`. This was changed to `cfg_iter!(sig.shifts)` to parallelize across the 10 shift specifications. Each shifted evaluation constructs a sub-slice of the original column and evaluates the MLE at the IC evaluation point — these are independent and embarrassingly parallel.

### Cumulative Impact

The optimizations 6–10 were applied together and benchmarked against the prior state (which already included optimizations 1–5):

| Metric | Before (opts 1–5) | After (opts 1–10) | Change |
|--------|-------------------|-------------------|--------|
| Criterion median | 15.34 ms | 14.62 ms | **−4.6%** |
| Criterion 95% CI | [15.14, 15.53] ms | [14.52, 14.73] ms | p = 0.00 |

Single-run timing breakdown comparison (representative, not averaged):

| Step | Before | After | Δ |
|------|--------|-------|---|
| Split columns | 0.671 ms | 0.640 ms | −31 µs |
| PCS commit | 4.526 ms | 4.218 ms | −308 µs |
| Ideal Check | 1.761 ms | 1.521 ms | −240 µs |
| Proj+Eval | 0.560 ms | 0.500 ms | −60 µs |
| **Lookup** | **4.888 ms** | **3.374 ms** | **−1514 µs** |
| Shift SC | 1.836 ms | 2.121 ms | +285 µs |
| Folding | 0.400 ms | 0.490 ms | +90 µs |
| PCS prove | 2.714 ms | 2.882 ms | +168 µs |
| **Total (criterion)** | **15.34 ms** | **14.62 ms** | **−720 µs** |

> **Note:** The single-run timing breakdown is from a single warm-up execution and is noisy; only the Criterion statistical result (100 samples, 400 iterations) should be used for regression comparisons. The lookup improvement from optimization 6 is the dominant contributor.

---

## Verifier Optimizations

The following optimizations target the E2E verifier (`verify_classic_logup_4x_folded` and related `verify_*` functions). The verifier's baseline cost is dominated by two operations: the **PCS verify** (~60% of total time, mainly `encode_wide_at_positions` and Merkle proof verification) and the **Hybrid GKR lookup verify** (~33%). The remaining ~7% covers IC verify, CPR verify, shift sumcheck verify, and folding verify.

An additional hidden cost is the **Miller-Rabin primality search** in `get_random_field_cfg`, which takes ~0.3–0.4 ms per call. In the original code, both the PIOP transcript and the PCS transcript independently derive the same 192-bit prime from the same initial state (empty transcript + absorbed commitment root), resulting in a redundant primality search. For dual-circuit verifiers that use two separate PCS instances, each with its own root, up to 3 independent primality searches occur.

### V1. Eliminate duplicate primality search (transcript snapshot reuse)

**Location**: `snark/src/pipeline.rs` — `verify_classic_logup_4x_folded`, `verify_classic_logup_folded`, `verify`, `verify_dual_ring`.

**Problem**: The verifier constructs two independent Fiat-Shamir transcripts — one for the PIOP and one for the PCS. Both start from `KeccakTranscript::new()` (= `Blake3Transcript::default()`), absorb the same PCS commitment root (`HASH_OUT_LEN` bytes), and then call `get_random_field_cfg()`, which invokes the rejection-sampling loop in `get_prime()`. Since Blake3 is deterministic, both transcripts reach identical internal states and derive the **same** prime $p$. The second call is pure waste (~0.3–0.4 ms of Miller-Rabin modular exponentiations on 192-bit candidates).

**Solution**: Immediately after the first `get_random_field_cfg()` call on the PIOP transcript, clone the transcript state:

```rust
let field_cfg = transcript.get_random_field_cfg::<PiopField, _, MillerRabin>();
let transcript_after_prime = transcript.clone();
```

Then, when constructing the PCS transcript (much later, after IC/CPR/lookup/shift/folding verify), reuse the snapshot instead of building a fresh transcript + primality search:

```rust
let mut pcs_transcript = PcsTranscript {
    fs_transcript: transcript_after_prime,  // reuse snapshot
    stream: Cursor::new(proof.pcs_proof_bytes.clone()),
};
let pcs_field_cfg = field_cfg.clone();  // same prime, no search needed
```

**Correctness**: Both transcripts start from `Blake3Transcript::new()`, absorb identical root bytes, and call `get_random_field_cfg()` with the same type parameters. The Blake3 hasher is deterministic, so the snapshot after the first call is byte-identical to the state the PCS transcript would have reached after its own call. The PCS prover (`prove_with_seed`) follows the same sequence (absorb root → get_random_field_cfg), so the reused snapshot is also compatible with the prover's transcript state.

**Applicability**: This optimization applies to verifiers where the PIOP and PCS transcripts share the same commitment root and use the same field type (`PiopField`). It has been applied to:

| Function | PCS field | Same root? | Applied? |
|---|---|---|---|
| `verify_classic_logup_4x_folded` | `PiopField` | Yes (1 root) | **Yes** |
| `verify_classic_logup_folded` | `PiopField` | Yes (1 root) | **Yes** |
| `verify` | `PiopField` | Yes (1 root) | **Yes** |
| `verify_dual_ring` | `PiopField` | Yes (1 root) | **Yes** |
| `verify_generic` | `PcsF` (generic, may differ) | Yes (1 root) | **No** — `PcsF` may differ from `PiopField` |
| `verify_pcs_only` | — | — | **N/A** — no PIOP phase, single search |
| `verify_dual_circuit` (×4 variants) | PCS1=`PiopField`, PCS2=`PcsF2` | No (2 roots, different states) | **No** — PIOP absorbs both roots, PCS1/PCS2 each absorb only their own |

For the dual-circuit verifiers, the three primality searches (PIOP, PCS1, PCS2) derive different primes from different transcript states, so the searches are genuinely independent and cannot be deduplicated.

**Impact**: Saves ~0.3–0.4 ms per single-PCS verify call. For the SHA-256 benchmark, PCS verify dropped from ~1.55 ms to ~1.19 ms (−23%), with the primality search accounting for most of the difference.

### V2. O(L) cross-check via prefix/suffix products

**Location**: `piop/src/lookup/hybrid_gkr.rs` — `HybridGkrBatchedDecompLogupProtocol::verify_as_subprotocol`, Step 6; and `piop/src/lookup/gkr_batched_decomposition.rs` — `GkrBatchedDecompLogupProtocol::verify_as_subprotocol`, same step.

**Problem**: After the GKR verification produces per-lookup root values $(P_w^{(\ell)}, Q_w^{(\ell)})$ and the table root $(P_t, Q_t)$, the verifier must check the **cross-check identity**:

$$\Bigl(\sum_\ell \alpha^\ell \cdot P_w^{(\ell)} \cdot \prod_{j \neq \ell} Q_w^{(j)}\Bigr) \cdot Q_t = P_t \cdot \prod_\ell Q_w^{(\ell)}$$

The original implementation computed $\prod_{j \neq \ell} Q_w^{(j)}$ for each $\ell$ via a nested loop:

```rust
for ell in 0..L {
    let mut others_q = one.clone();
    for j in 0..L {
        if j != ell { others_q *= &roots_q[j]; }
    }
    // use others_q ...
}
```

This costs $O(L^2)$ field multiplications. For $L = 13$ lookup columns, that's 156 multiplications.

**Solution**: Replace with **prefix/suffix product arrays** computed in $O(L)$:

```rust
let mut prefix = vec![one.clone(); L];
for i in 1..L { prefix[i] = prefix[i-1].clone() * &roots_q[i-1]; }
let mut suffix = vec![one.clone(); L];
for i in (0..L-1).rev() { suffix[i] = suffix[i+1].clone() * &roots_q[i+1]; }
// others_q[ell] = prefix[ell] * suffix[ell]
```

This costs $2(L-1) = 24$ multiplications instead of 156 — a 6.5× reduction for $L = 13$.

**Impact**: Small but measurable reduction in lookup verify time. The cross-check itself is a minor fraction of the total lookup verify, but the $O(L^2) \to O(L)$ improvement eliminates a quadratic scaling risk for larger $L$.

### V3. Fused table leaf + multiplicity verification

**Location**: same files as V2, Steps 7 and 8 of `verify_as_subprotocol`.

**Problem**: The GKR verifier performs two separate traversals over the subtable:

- **Step 7 (table leaf check)**: For each subtable entry $j \in [T]$, compute $p_\text{eval} = \sum_\ell \alpha^\ell \cdot m_\ell[j]$ and $q_\text{eval} = \beta - \text{table}[j]$, then dot with $\text{eq}(\mathbf{r}_t, j)$.
- **Step 8 (multiplicity sum check)**: For each lookup $\ell$, verify $\sum_j m_\ell[j] = W$ (each lookup column has exactly $W$ entries). This requires a separate $O(T \times L)$ traversal.

The original code also cloned `alpha_powers` inside the inner loop of Step 7.

**Solution**: Fuse both steps into a single pass over the subtable:

1. **Precompute `combined_mults[j]`**: In one transposed pass (loop over $\ell$, then $j$), compute $\text{combined\_mults}[j] = \sum_\ell \alpha^\ell \cdot m_\ell[j]$ while simultaneously accumulating the multiplicity sums $\text{mult\_sum}[\ell] = \sum_j m_\ell[j]$.

2. **Check multiplicity sums** immediately after the accumulation — no separate Step 8 traversal needed.

3. **Dot with eq-table**: Use the precomputed `combined_mults[j]` directly in the $p_\text{eval}$ dot product:

```rust
for j in 0..table_len {
    p_eval += &(combined_mults[j].clone() * &eq_at_t[j]);
    q_eval += &((beta.clone() - &table[j]) * &eq_at_t[j]);
}
```

This eliminates one full $O(T \times L)$ traversal and removes all `alpha_powers` cloning from the hot inner loop.

**Impact**: Lookup verify dropped from ~0.85 ms to ~0.51–0.70 ms (−18–40%, depending on run). The fused traversal is particularly beneficial because $T = 256$ and $L = 13$, so the eliminated traversal was 3328 multiply-add operations.

### V4. Trial division pre-filter for Miller-Rabin primality testing

**Location**: `primality/src/lib.rs` — `MillerRabin::is_probably_prime`.

**Problem**: The `get_prime()` function in the Fiat-Shamir transcript performs a rejection-sampling loop: it squeezes random 192-bit candidates, makes them odd, and tests each with `MillerRabin::is_probably_prime()`. By the prime number theorem, roughly 1 in $\ln(2^{192}) \approx 133$ odd candidates is prime, so the loop runs ~67 iterations on average. Each failing iteration invokes a full Miller-Rabin base-2 test, which requires a modular exponentiation on a 192-bit integer — approximately $192 \times 3^2 = 1728$ 64-bit multiply-add operations (using schoolbook multiplication on 3-limb integers).

**Solution**: Before the expensive Miller-Rabin test, perform **trial division** by the first 15 small primes (3, 5, 7, …, 53). This is done efficiently in two steps:

1. **Single big-integer remainder**: Compute $r = \text{candidate} \bmod P$, where $P = 3 \times 5 \times 7 \times \cdots \times 53$ (fits in a `u64`). This is one multi-precision division.

2. **15 native u64 checks**: For each small prime $p_i$, check if $r \bmod p_i = 0$. If so, the candidate is composite (unless it *is* $p_i$ itself, which cannot happen for 192-bit candidates).

The probability that a random odd integer is divisible by at least one of {3, 5, …, 53} is:

$$1 - \prod_{p \in \{3, 5, \ldots, 53\}} \frac{p - 1}{p} \approx 77.4\%$$

So ~77% of composite candidates are rejected by trial division alone, skipping the expensive modular exponentiation entirely. The remaining ~23% still fall through to Miller-Rabin.

**Cost**: One `Uint<3> % Uint<3>` operation (≈3 64-bit divisions) plus 15 native `u64 % u64` operations — negligible compared to a Miller-Rabin test.

**Impact**: Speeds up every `get_random_field_cfg()` call across the entire codebase (both prover and verifier, all pipeline variants). For the verifier, where the primality search is a larger fraction of total time, this provides an additional ~3% (0.06 ms) criterion improvement on top of V1. Combined with V1 (which eliminates one of the two searches), the total primality-related savings for the 4x-folded verifier is ~0.4 ms.

### Cumulative Verifier Impact

Baseline (before any verifier optimizations): **2.55 ms** (criterion).

| Optimization batch | Criterion | Step timing (single-run) |
|---|---|---|
| V1 (transcript snapshot) + V2 (O(L) cross-check) + V3 (fused table/mult) | **2.26 ms** | 2.31 ms |
| + V4 (trial division pre-filter) | **2.20 ms** | 2.39 ms |
| **Total improvement** | **−14%** | |

Per-component breakdown (single-run, representative):

| Component | Before | After V1–V3 | After V1–V4 |
|---|---|---|---|
| IC verify | 0.092 ms | 0.052 ms | 0.120 ms (noise) |
| CPR+Lookup verify | 0.080 ms | 0.118 ms | 0.152 ms (noise) |
| Lookup verify | 0.854 ms | 0.510 ms | 0.637 ms |
| PCS verify | 1.546 ms | 1.187 ms | 1.193 ms |
| **Total (step timing)** | **3.046 ms** | **2.305 ms** | **2.698 ms** |
| **Total (criterion)** | **2.55 ms** | **2.26 ms** | **2.20 ms** |

> **Note:** The single-run step timings are from individual warm-up executions and are noisy. The Criterion statistical results (100 samples) are the authoritative measurement. The step-timing "Total" is often higher than the Criterion median due to cold-cache effects in the single warm-up run.

### Remaining Verifier Hot Spots

After V1–V4, the verifier's remaining cost (~2.2 ms) is dominated by:

1. **PCS verify `encode_wide_at_positions`** (~0.5–0.7 ms): Evaluates the IPRS linear code at 131 opened column positions. Each evaluation is a `CombR` dot product of the 2048-element message against code coefficients — 131 × 2048 = ~269K `Int<6>` multiply-add operations. Already parallelized via Rayon. An existing `TODO` in the codebase suggests lifting the dot products to integer arithmetic to reduce Montgomery overhead.

2. **Merkle proof verification** (~0.3–0.5 ms): 131 authentication paths of depth 13. Each path verification requires 13 Blake3 hash invocations.

3. **Primality search** (~0.2–0.3 ms): The single remaining `get_random_field_cfg()` call (the one that cannot be eliminated). Accelerated by trial division (V4) but still requires ~15 Miller-Rabin tests for the ~23% of candidates that pass trial division.

4. **Lookup verify GKR sumcheck replay** (~0.3–0.5 ms): The verifier replays the GKR layer sumchecks (37 witness rounds + 28 table rounds). Each round involves deserializing round messages, checking $p(0) + p(1) = \text{claim}$, absorbing data into the transcript, and squeezing a challenge.

---

## Feature Flags

The three features specified on the command line affect performance:

| Feature | Effect |
|---|---|
| `parallel` | Enables Rayon-based parallelism for column splitting, Merkle hashing, NTT encoding, sumcheck, MLE evaluation, etc. |
| `simd` | Enables SIMD-accelerated `BinaryPoly` arithmetic (XOR, AND, shifts) and inner product operations. |
| `asm` | Enables assembly-optimized routines for field arithmetic and hashing (e.g., SHA-256 intrinsics for the Merkle tree, AES-NI for transcript). |
| `no-f2x` | Switches the dual UAIR to the no-F₂[X] variant: 4 rotation constraints move from Bp to Qx with explicit μ quotient columns, trace grows from 30 to 38 columns. See [§"no-f2x variant"](#no-f2x-variant). |

Additionally, the `Sha256UairQx` carry-propagation constraints are gated behind the `qx-constraints` feature flag in the `sha256-uair` crate.

---

## Criterion Configuration

- **Sample size**: 100 (set via `group.sample_size(100)`)
- **Benchmark group**: `"8xSHA256 4x-Folded Hybrid GKR c=2 Steps"`
- **Harness**: Custom (`harness = false` in `Cargo.toml`)
- **Timing method**: Most step benchmarks use standard `iter` (WitnessGen, SplitColumns, Commit, FieldSetup, Project Ideal Check, Project Main field sumcheck, LookupExtract, PCS/Prove, and E2E Verifier). Stateful steps that need transcript replay setup outside the timed region (IdealCheck, Main field sumcheck, Lookup, ShiftSumcheck, Folding/FoldClaims) and E2E Prover benchmarks use `iter_custom`.

---

## Output

Running the benchmark produces:

1. **Criterion statistical output** (stdout) — for each benchmark function, reports mean time, standard deviation, throughput, and regression detection vs. any saved baseline.
2. **Stdout diagnostic output** — verifier step timing breakdown (printed via `println!` after the E2E Verifier run).
3. **Stderr diagnostic output** — prover pipeline timing breakdown, proof size table, and peak memory usage (printed via `eprintln!`).
4. **Criterion HTML reports** — saved to `target/criterion/8xSHA256 4x-Folded Hybrid GKR c=2 Steps/` for interactive visualization.
