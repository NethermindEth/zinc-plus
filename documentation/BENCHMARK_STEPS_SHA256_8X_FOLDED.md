# Benchmark: `steps_sha256_8x_folded`

```
cargo bench --bench steps_sha256_8x_folded -p zinc-snark --features "parallel simd asm"
```

## Overview

This benchmark provides a **fine-grained, step-by-step timing breakdown** of the Zinc+ proving stack applied to the **8×SHA-256** workload (8 SHA-256 compressions in parallel) using **column folding** to reduce proof size. It uses the [Criterion](https://bheisler.github.io/criterion.rs/book/) framework with a custom harness.

The benchmark compares several pipeline variants side by side:

| Pipeline variant | Lookup method | Column folding | PCS element type |
|---|---|---|---|
| **Original** | Classic batched | None | `BinaryPoly<32>` |
| **2× folded** (classic) | Classic batched | 1 split (32→16) | `BinaryPoly<16>` |
| **2× folded** (GKR) | GKR fractional-sum | 1 split (32→16) | `BinaryPoly<16>` |
| **2× folded** (GKR 4-chunk) | GKR fractional-sum (4-chunk) | 1 split (32→16) | `BinaryPoly<16>` |
| **2× folded** (Hybrid GKR c=2) | Hybrid GKR (cutoff=2) | 1 split (32→16) | `BinaryPoly<16>` |
| **4× folded** (classic) | Classic batched | 2 splits (32→16→8) | `BinaryPoly<8>` |
| **4× folded** (4-chunk) | Classic batched | 2 splits (32→16→8) | `BinaryPoly<8>` |
| **4× Hybrid GKR c=2** (4-chunk) | Hybrid GKR (cutoff=2) | 2 splits (32→16→8) | `BinaryPoly<8>` |
| **4× Hybrid GKR c=1** (4-chunk) | Hybrid GKR (cutoff=1) | 2 splits (32→16→8) | `BinaryPoly<8>` |
| **8× Hybrid GKR c=2** (4-chunk) | Hybrid GKR (cutoff=2) | 3 splits (32→16→8→4) | `BinaryPoly<4>` |

For each variant, both **prover** and **verifier** end-to-end times are measured. The 2× folded classic pipeline is further broken into individual prover steps and verifier steps for a per-phase timing analysis.

The benchmark also reports:
- **Proof size** breakdown (PCS, Ideal Check, sumcheck, lookup, folding, etc.) for every variant — raw bytes and Deflate-compressed bytes.
- **Peak memory** usage.
- **Hybrid GKR cost analysis** for different cutoff values (1–10) and the full GKR baseline.

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
| `max_degree` | 2 | Carry constraints are degree 2 (selector × expression); standard IC path |
| `NUM_COLUMN_OPENINGS` | 131 | PCS proximity-test column openings (96-bit security) |
| `GRINDING_BITS` | 8 | Proof-of-work bits (reduces required column openings) |
| Lookup chunk config (default) | 8 chunks × $2^4$ | `chunk_width=4`, `width=32` → 8 sub-tables of 16 entries each |
| Lookup chunk config (4-chunk) | 4 chunks × $2^8$ | `chunk_width=8`, `width=32` → 4 sub-tables of 256 entries each |
| Affine lookups | 3 | Ch, neg-Ch, Maj computed as affine combinations of trace columns |

**Security level:** These benchmarks target **96-bit security**, matching the configuration used by Binius64. With rate $\rho = 1/4$ and unique decoding radius $\delta = (1 - \rho)/2 = 3/8$, each column opening provides $\log_2(1/(1 - \delta)) = \log_2(8/5) \approx 0.678$ bits of security. Combined with 8 bits of grinding: $131 \times 0.678 + 8 \approx 96.8$ bits.

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

#### Sub-UAIR #2: $\mathbb{Q}[X]$ carry-propagation constraints (`Sha256UairQx`) — 3 constraints, max degree 2

These use the ideal $(X - 2)$ and are **gated by selectors** (`sel_round` or `sel_sched`), making them degree 2 (selector × inner expression). At $X = 2$, bit-polynomial coefficients become binary place values, so membership in $(X-2)$ encodes correct modular-$2^{32}$ arithmetic.

| # | Name | Formula | Gate |
|---|------|---------|------|
| C17 | a-update | $\text{sel\_round} \cdot \bigl(\hat{a}[t{+}1] - \hat{h} - \hat{\Sigma}_1 - \widehat{ch\_ef} - \widehat{ch\_\neg eg} - \hat{K} - \hat{W} - \hat{\Sigma}_0 - \widehat{Maj} + \mu_a \cdot X^{32}\bigr) \in (X{-}2)$ | `sel_round` |
| C18 | e-update | $\text{sel\_round} \cdot \bigl(\hat{e}[t{+}1] - \hat{d} - \hat{h} - \hat{\Sigma}_1 - \widehat{ch\_ef} - \widehat{ch\_\neg eg} - \hat{K} - \hat{W} + \mu_e \cdot X^{32}\bigr) \in (X{-}2)$ | `sel_round` |
| C19 | W-schedule | $\text{sel\_sched} \cdot \bigl(\hat{W} - \hat{W}_{t-16} - \hat{\sigma}_{0,w} - \hat{W}_{t-7} - \hat{\sigma}_{1,w} + \mu_W \cdot X^{32}\bigr) \in (X{-}2)$ | `sel_sched` |

#### Why only the carry constraints are gated

**C1–C6 (rotation & shift decomposition)** reference only same-row columns. The witness generator zeros out all related columns in sync: when the lookback data doesn't exist (e.g., `W_{t-15}` at $t < 15$), all of `W_tm15`, `sigma0_w_hat`, `S0`, and `R0` remain at their default zero. So at $t = 1$, C3 evaluates as $0 \cdot \rho_{\sigma_0} + 0 - 0 = 0$, which trivially satisfies the constraint with no gating needed.

**C7–C16 (linking)** are enforced via the shift sumcheck, which inherently handles boundary rows: the shifted column is zero-padded beyond the trace boundary by definition.

**C17–C19 (carry propagation)** are fundamentally different: they reference **next-row values** via shift-by-1 (`â[t+1]`, `ê[t+1]`). At boundary rows near the trace tail, the next-row values are forced to zero by zero-padding, but the *current-row* values ($\hat{h}, \hat{\Sigma}_1, \hat{K}, \hat{W}$, etc.) are non-zero — they hold valid data from the last active SHA-256 round. Without the selector, the constraint would demand $0 = h_t + \Sigma_1(e_t) + \text{Ch}(e_t,f_t,g_t) + K_t + W_t + \Sigma_0(a_t) + \text{Maj}(a_t,b_t,c_t) - \mu_a \cdot 2^{32}$, which is false. Multiplying by `sel_round = 0` at those rows makes the constraint trivially $0 \in (X{-}2)$.

Concretely: `sel_round = 0` for the last 4 rows (N−4 to N−1), because `d_hat`/`h_hat` use shift-3 and the resulting zero-padded next-row state cannot satisfy the round recurrence. `sel_sched = 0` for the last 16 rows, because `W_{t-16}` uses shift-16.

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

**8× folding** applies this splitting three times: $D \to D/2 \to D/4 \to D/8$, producing three folding rounds with challenges $\gamma_1, \gamma_2, \gamma_3$ and a PCS point $\mathbf{r} \| \gamma_1 \| \gamma_2 \| \gamma_3$. The folding proof carries six values per column: $(c_1, c_2)$ for the first split, $(c_3, c_4)$ for the second, and $(c_5, c_6)$ for the third. With `num_vars = 12` and `row_len = 2048`, the 8× variant requires `num_rows = 2` (vs 1 for 2× and 4×).

The benefit: PCS codeword elements are 2×, 4×, or 8× smaller, so each column opening in the Merkle proof is cheaper. The cost: a small folding proof ($c_1, c_2$ per column per round) is added.

---

## PCS Configurations

Four PCS parameter sets are defined:

### Original (`OrigZt` / `OrigLc`)
- **Codeword type**: `BinaryPoly<32>` (256 B per element)
- **Linear code**: `IprsCode` with `PnttConfigF12289R4B64<1>` — field $\mathbb{F}_{12289}$ ($p = 12289$, $p-1 = 2^{12} \times 3$), rate 1/4, base length 64, depth 1 → `INPUT_LEN = 512`, `OUTPUT_LEN = 2048`
- **num_vars**: 9 ($2^9 = 512$ evaluations, `row_len = 512`, 1 row)

### 2× Folded (`FoldedZt` / `FoldedLc`)
- **Codeword type**: `BinaryPoly<16>` (128 B per element)
- **Linear code**: `IprsCode` with `PnttConfigF12289R4B16<2>` — same field, rate 1/4, base length 16, depth 2 → `INPUT_LEN = 1024`, `OUTPUT_LEN = 4096`
- **Folded num_vars**: 10 ($2^{10} = 1024$ evaluations, `row_len = 1024`, 1 row)

### 4× Folded (`FoldedZt4x` / `FoldedLc4x`)
- **Codeword type**: `BinaryPoly<8>` (64 B per element)
- **Linear code**: `IprsCode` with `PnttConfigF2_16R4B32<2>` — field $\mathbb{F}_{65537}$ ($2^{16}+1$, `TWO_ADICITY=16`), rate 1/4, base length 32, depth 2 → `INPUT_LEN = 2048`, `OUTPUT_LEN = 8192`
- **Folded num_vars**: 11 ($2^{11} = 2048$ evaluations, `row_len = 2048`, 1 row)
- Uses $\mathbb{F}_{65537}$ instead of $\mathbb{F}_{12289}$ because smaller twiddle factors (max 4096) keep codeword coefficients within ~$2^{50}$, vs ~$2^{62}$ at depth 3 with $\mathbb{F}_{12289}$.

### 8× Folded (`FoldedZt8x` / `FoldedLc8x`)
- **Codeword type**: `BinaryPoly<4>` (32 B per element)
- **Linear code**: `IprsCode` with `PnttConfigF2_16R4B4<3>` — field $\mathbb{F}_{65537}$ ($2^{16}+1$, `TWO_ADICITY=16`), rate 1/4, base length 4, depth 3 → `INPUT_LEN = 2048`, `OUTPUT_LEN = 8192`
- **Folded num_vars**: 12 ($2^{12} = 4096$ evaluations; original `num_vars = 9` + 3 folding extensions. With `row_len = 2048` and `num_rows = 2`: $2048 \times 2 = 4096 = 2^{12}$.)
- Uses the same $\mathbb{F}_{65537}$ field as the 4× variant, with an additional split halving the codeword element size from 64 B to 32 B.

### Main PIOP field

- `F = MontyField<3>` — a 192-bit Montgomery-form prime field selected via Fiat-Shamir transcript (`get_random_field_cfg` with Miller-Rabin primality testing).

---

## Prover Steps Benchmarked (Individual)

These are the per-step Criterion benchmarks for the **2× folded classic** pipeline. Each step is described at the implementation level, including the mathematical operations, data flow, and what is produced.

### 1. `WitnessGen`

**Function**: `Sha256Uair::generate_witness(num_vars=9, rng)`

Generates a 30-column `BinaryPoly<32>` trace sized for $2^9 = 512$ rows. The witness generator runs **8 chained** SHA-256 compressions (each 64 rounds), filling the entire trace with non-trivial computational data. Compression 0 hashes the padded empty-string message; compressions 1–7 use deterministic non-trivial message blocks. The round-function state flows directly from one compression to the next (without the SHA-256 post-processing H-addition), so all linking constraints (`d_hat[t+3] = a_hat[t]`, etc.) are naturally satisfied across compression boundaries.

Due to shift zero-padding at the trace tail, the last compression is slightly truncated: `sel_round = 0` for the last 4 rows (N−4 to N−1), and `sel_sched = 0` for the last 16 rows, giving 508 active round rows and 368 active schedule rows out of 512 total. The 10 $Q[X]$ columns carry the bit-polynomial representations of the round variables ($\hat{a}, \hat{e}, \hat{W}$, intermediate values), while the lookback columns carry copies of earlier rows (needed because the UAIR constraint system is "next-row" only). The integer carry columns ($\mu_a, \mu_e, \mu_W$) store the small integer carries from modular additions. The `rng` parameter is accepted for trait compatibility but unused.

**Output**: `Vec<DenseMultilinearExtension<BinaryPoly<32>>>` — 30 columns, each with $2^9 = 512$ entries.

Before subsequent steps, the benchmark filters out **PCS-excluded columns** (public columns and shift-source columns that the verifier can reconstruct or whose evaluation claims the shift sumcheck resolves). The 12 public columns are: `W_hat` (2), `S0` (10), `S1` (11), `R0` (12), `R1` (13), `W_{t-2}` (16), `W_{t-7}` (17), `W_{t-15}` (18), `W_{t-16}` (19), `K_hat` (20), `sel_round` (25), `sel_sched` (26). The 8 shift-source columns (0–1, 14–15, 21–24) are excluded because the shift sumcheck resolves their evaluation claims. In total 20 columns are excluded, leaving **10 PCS-committed columns** in `sha_pcs_trace`.

### 2. `Folding/SplitColumns`

**Function**: `split_columns::<32, 16>(&sha_pcs_trace)`

For each PCS-committed column and each entry $v_i \in \text{BinaryPoly}<32>$ (a polynomial in $\mathbb{F}_2[X]$ with 32 Boolean coefficients):

1. Extract the **low half** $u_i$: coefficients at positions $0, 1, \ldots, 15$ → a `BinaryPoly<16>`.
2. Extract the **high half** $w_i$: coefficients at positions $16, 17, \ldots, 31$ → a `BinaryPoly<16>`.
3. Such that $v_i = u_i + X^{16} \cdot w_i$.

The output column $v' = (u_0, u_1, \ldots, u_{511}, w_0, w_1, \ldots, w_{511})$ has 1024 entries and `num_vars = 10`. Viewed as a multilinear extension, the last variable selects between the low half ($x_{10} = 0$) and the high half ($x_{10} = 1$).

With the `parallel` feature, each column is split independently on a Rayon thread.

**Output**: `Vec<DenseMultilinearExtension<BinaryPoly<16>>>` — same number of columns, each with $2^{10} = 1024$ entries, element size halved from 256 B to 128 B.

### 3. `PCS/Commit (folded)`

**Function**: `ZipPlus::<FoldedZt, FoldedLc>::commit(&folded_params, &split_trace)`

Commits the split `BinaryPoly<16>` columns using the Zip+ polynomial commitment scheme:

1. **Matrix layout**: Each column's 1024 evaluations are arranged into a matrix with `num_rows = 1` row of length `row_len = 1024`.
2. **Linear code encoding**: The single row is encoded through the IPRS code (`PnttConfigF12289R4B16<2>`), which applies a Number Theoretic Transform over $\mathbb{F}_{12289}$ with rate 1/4. This expands 1024 input symbols to 4096 output codeword symbols per column. With the `parallel` feature, each polynomial's encoding runs on a separate Rayon thread via `cfg_iter!(polys)`, providing up to 10-way parallelism across committed columns. See [§"Implementation Optimizations"](#implementation-optimizations) for details.
3. **Merkle tree**: All encoded columns are interleaved column-wise — leaf $j$ is the hash of `(poly_1_codeword[j] \| poly_2_codeword[j] \| \ldots)` across all committed polynomials. A single Merkle tree of depth 12 (4096 leaves) is constructed.

**Output**: `ZipPlusHint` (codeword matrices + Merkle tree, kept for the proving phase) and `ZipPlusCommitment` (Merkle root + batch size, sent to verifier).

### 4. `PCS/Commit (original)` *(comparison)*

**Function**: `ZipPlus::<OrigZt, OrigLc>::commit(&orig_params, &sha_pcs_trace)`

Same as above but on the original `BinaryPoly<32>` columns with different parameters: `row_len = 512`, code `PnttConfigF12289R4B64<1>`, producing a Merkle tree of depth 11 (2048 leaves). Each leaf is 2× larger because the codeword elements are `BinaryPoly<32>` (256 B each). This is benchmarked to quantify the commit-time difference.

### 5. `PIOP/FieldSetup`

**Function**: `KeccakTranscript::new()` + `transcript.get_random_field_cfg::<F, _, MillerRabin>()`

Initializes the Fiat-Shamir transcript (Blake3 [Albert: we should change the name KeccakTranscript before we forget we are actually using Blake3]) and derives a random 192-bit prime $p$ for the PIOP field `F = MontyField<3>`. The process:

1. Squeeze a candidate integer from the transcript.
2. Run the **Miller-Rabin** primality test. If the candidate is composite, increment and try again.
3. Build the Montgomery-form field configuration (`modulus`, `R`, `R²`, inverses, etc.).

This step is typically fast (a few microseconds) once a prime is found, but has variable latency due to the primality search.

**Output**: `F::Config` — the field configuration used for all subsequent PIOP arithmetic.

### 6. `PIOP/Project Ideal Check`

**Function**: `project_scalars::<F, Sha256Uair>(closure)`

Builds the field-level representation of all UAIR constraint scalars. The SHA-256 UAIR defines constraints whose coefficients are `BinaryPoly<32>` values (each coefficient is a Boolean). This step maps each coefficient to `F` (true → one, false → zero) and wraps them as `DynamicPolynomialF<F>` — univariate polynomials over $F$ that represent the constraint structure.

Concretely, for each distinct `BinaryPoly<32>` scalar appearing in any constraint, the closure maps its 32 Boolean coefficients to a degree-31 polynomial in $F[Y]$, where $Y$ will later be specialized at the projecting element $\alpha$.

**Output**: `HashMap<BinaryPoly<32>, DynamicPolynomialF<F>>` — a map from each scalar to its polynomial representation. This is reused by both the Ideal Check and the main field sumcheck.

### 7. `PIOP/IdealCheck`

**Function**: `IdealCheckProtocol::<F>::prove_as_subprotocol::<Sha256Uair>(...)`

The Ideal Check protocol verifies that the trace satisfies all UAIR constraints modulo the appropriate ideals. Since the SHA-256 UAIR has `max_degree == 2` (carry constraints are selector-gated), the pipeline uses the **standard** IC path (not the MLE-first optimisation, which requires linear constraints):

1. **Project trace to $F[Y]$**: Every `BinaryPoly<32>` entry across all 30 columns is lifted to a `DynamicPolynomialF<F>` — a univariate polynomial over $F$ with 32 coefficients. This is the dominant cost: $30 \times 512 = 15{,}360$ projections. (When `max_degree == 1`, the pipeline instead uses the cheaper MLE-first path that avoids this full projection.)
2. **Build constraint polynomials**: For each row $b$, evaluate all 19 UAIR constraints on the projected trace columns, accumulating into combined polynomials (one MLE per constraint group). This handles degree-2 constraints (e.g., `sel_round · (expression)`) correctly via polynomial multiplication.
3. **Draw evaluation point**: Squeeze $\mathbf{r} = (r_0, \ldots, r_8) \in F^9$ from the Fiat-Shamir transcript.
4. **Evaluate combined MLEs at $\mathbf{r}$**: For each combined polynomial MLE, compute its evaluation at $\mathbf{r}$, yielding one `DynamicPolynomialF<F>` per constraint group.
5. **Absorb proof values**: The combined MLE values (one `DynamicPolynomialF<F>` per constraint group) are absorbed into the transcript.

**Output**:
- `Proof<F>` containing `combined_mle_values: Vec<DynamicPolynomialF<F>>` — the constraint evaluations at $\mathbf{r}$, one per combined polynomial.
- `ProverState<F>` containing `evaluation_point: Vec<F>` — the point $\mathbf{r}$ (passed to the next step).

### 8. `PIOP/Project Main field sumcheck`

**Functions**: `project_scalars_to_field(...)` + `project_trace_to_field::<F, 32>(...)`

This step projects everything from the `BinaryPoly<32>` / `DynamicPolynomialF<F>` domain into the scalar PIOP field $F$:

1. **`project_scalars_to_field`**: The Ideal Check drew a projecting element $\alpha \in F$ from the transcript. Each `DynamicPolynomialF<F>` scalar (a polynomial $p(Y) \in F[Y]$) is evaluated at $Y = \alpha$, collapsing it to a single field element $p(\alpha) \in F$. Output: `HashMap<BinaryPoly<32>, F>`.

2. **`project_trace_to_field::<F, 32>`**: Every `BinaryPoly<32>` entry in the 30-column trace is projected to $F$ by evaluating it at $\alpha$. For a `BinaryPoly<32>` with Boolean coefficients $(b_0, b_1, \ldots, b_{31})$, this computes $\sum_{i=0}^{31} b_i \cdot \alpha^i \in F$. This is done using a **precomputed projection closure** (`BinaryPoly::<32>::prepare_projection(&α)`) that caches the powers $\alpha^0, \alpha^1, \ldots, \alpha^{31}$ and computes the dot product via conditional additions (no multiplications needed per entry since $b_i \in \{0, 1\}$).

   Output: `Vec<DenseMultilinearExtension<F::Inner>>` — 30 columns, each with 512 field elements.

After this step, all subsequent PIOP operations work in scalar $F$ arithmetic.

### 9. `PIOP/Main field sumcheck`

**Function**: `CombinedPolyResolver::<F>::prove_as_subprotocol::<Sha256Uair>(...)`

Proves via sumcheck that the Ideal Check evaluation claims are consistent with the trace. This is the most algebraically involved prover step.

**Setup** (before sumcheck):

1. **Build shifted ("down") columns**: For each `ShiftSpec` in the UAIR signature, construct a shifted copy of the source column (drop the first `shift_amount` entries, zero-pad at the end). For SHA-256, these are the lookback columns that reference earlier rows.
2. **Build `eq_r`**: The multilinear extension of the equality function $\text{eq}(\mathbf{r}, \mathbf{b}) = \prod_{i=0}^{n-1} (r_i \cdot b_i + (1-r_i)(1-b_i))$, evaluated at all $2^n$ points. This is a length-512 vector.
3. **Build last-row selector**: $\text{sel}(\mathbf{b}) = \text{eq}(\mathbf{1}, \mathbf{b})$, which is 1 only at the all-ones vertex (last row). The factor $(1 - \text{sel})$ zeroes out the last row, which is a boundary condition.
4. **Draw batching challenge**: Squeeze $\alpha_{\text{fold}} \in F$ from the transcript. This is used to batch all UAIR constraints into one polynomial.
5. **Assemble MLEs**: The sumcheck operates over the array `[sel, eq_r, up_col_0, ..., up_col_{N-1}, down_col_0, ..., down_col_{M-1}]`.

**Sumcheck** (degree `max_degree + 2 = 4`):

The prover runs `MLSumcheck::prove_as_subprotocol` with the combination function:

$$f(\text{mle\_values}) = \text{eq\_r} \cdot (1 - \text{sel}) \cdot \sum_{i=0}^{C-1} \alpha_{\text{fold}}^i \cdot g_i(\text{up}, \text{down})$$

where $g_i$ are the UAIR constraint polynomials applied to the up (current row) and down (shifted row) column values. The `ConstraintFolder` evaluates all constraints in a single pass, accumulating with powers of $\alpha_{\text{fold}}$. The maximum degree is 4 because the carry constraints are degree 2 (selector × expression), multiplied by `eq_r` (degree 1) and `(1-sel)` (degree 1).

Over 9 rounds, each round:
- Fixes one variable to a random challenge.
- Sends a degree-4 univariate polynomial (the round message) to the verifier.

**After sumcheck**:

The prover evaluates all trace-column MLEs at the final sumcheck point to produce:
- **`up_evals`**: one $F$ value per original column — $\text{MLE}[\text{col}_j](\mathbf{s})$ where $\mathbf{s}$ is the 9-dimensional sumcheck randomness.
- **`down_evals`**: one $F$ value per shifted column.

These are absorbed into the transcript and included in the proof.

**Output**:
- `Proof<F>` containing `sumcheck_proof` (9 round messages), `up_evals`, `down_evals`.
- `ProverState<F>` containing `evaluation_point` — the sumcheck randomness $\mathbf{s}$.

### 9a. Shift Sumcheck (`PIOP/ShiftSumcheck` / `V/ShiftSumcheckVerify`)

> **Pipeline status**: The shift sumcheck is now active in **all** pipeline variants — both the original non-folded pipeline (`pipeline::prove` / `pipeline::verify`) and all folded/batched variants (`prove_classic_logup_folded`, `prove_classic_logup_4x_folded`, `prove_gkr_logup_folded`, `prove_hybrid_gkr_logup_folded`, `prove_hybrid_gkr_logup_4x_folded`, `prove_hybrid_gkr_logup_8x_folded`). In the batched path, it runs after CPR + Lookup finalize and before folding. In the non-batched path, it runs after CPR and before lookup. The per-step benchmark includes dedicated `PIOP/ShiftSumcheck` (prover) and `V/ShiftSumcheckVerify` (verifier) steps. At the code level the placement comment is "Step 3a" in [snark/src/pipeline.rs](snark/src/pipeline.rs).

#### Purpose

The main field sumcheck (step 9) produces **`down_evals`** — one claimed evaluation $\text{MLE}[\text{shift}_{c_i}(v_i)](\mathbf{r})$ per shifted column. These are evaluations of *shifted* versions of committed source columns, but the PCS only commits to the *unshifted* source columns. The **shift sumcheck** bridges this gap: it reduces each shifted-column evaluation claim to a plain MLE evaluation claim on the unshifted source column at a *new* random point $\mathbf{s}$, which the PCS can then open directly.

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

### 10. `PIOP/LookupExtract`

This step prepares the data needed by the lookup protocol. Two operations per lookup column:

1. **Field columns**: Extract the 10 lookup-relevant columns from `field_trace` (the field-projected trace from step 8). Each is a `Vec<F>` of 512 elements.

2. **Raw integer indices**: Convert each `BinaryPoly<32>` entry from the original trace into a `usize` index by interpreting the 32 Boolean coefficients as a bit-packed integer: $\text{idx} = \sum_{j=0}^{31} b_j \cdot 2^j$. This avoids building a $2^{32}$-entry reverse-lookup table, which would be prohibitive.

3. **Index remapping**: The 10 lookup columns may have duplicate column indices (multiple specs referencing the same column). A `BTreeMap` deduplicates them and builds a remapped `LookupColumnSpec` vector.

**Output**: `(Vec<Vec<F>>, Vec<Vec<usize>>, Vec<LookupColumnSpec>)` — deduplicated field columns, raw indices, and remapped specs.

### 11. `PIOP/Lookup`

**Function**: `prove_batched_lookup_with_indices(&mut transcript, &columns, &raw_indices, &specs, &projecting_element, &field_cfg)`

Proves that each lookup column's values belong to the declared subtable, using the **batched decomposed LogUp** protocol (classic variant). The protocol works as follows. (For the alternative GKR-based lookup used in the GKR and Hybrid GKR pipeline variants, see the [GKR and Hybrid GKR Lookup Protocols](#gkr-and-hybrid-gkr-lookup-protocols) section.)

**Grouping**: Specs are grouped by `LookupTableType`. For 8×SHA-256 with default configuration, all 10 column lookups use `BitPoly { width: 32, chunk_width: 4 }` and form a single group. (Note: the 3 affine lookups — Ch, neg-Ch, Maj — are **not** included in this individual benchmark step; they are only incorporated in the E2E pipeline via `append_affine_virtual_columns`, which adds them as additional virtual columns before calling the lookup prover.)

**For each group**, `BatchedDecompLogupProtocol::prove_as_subprotocol` runs:

1. **Subtable generation**: `generate_bitpoly_table(chunk_width=4, α)` builds a subtable of $2^4 = 16$ entries, where entry $t$ is $\sum_{j=0}^{3} b_j \cdot \alpha^j$ (the projection of a 4-bit polynomial at $\alpha$). The 8 shifts are $\alpha^0, \alpha^4, \alpha^8, \ldots, \alpha^{28}$.

2. **Chunk decomposition**: Each 32-bit index is decomposed into $K = 8$ chunks of 4 bits each. For column $\ell$ and row $i$, chunk $k$ maps to subtable index $\lfloor \text{idx}_i / 2^{4k} \rfloor \bmod 16$. The field-level chunk value is looked up as $c_k^{(\ell)}[i] = \text{subtable}[\text{chunk\_idx}]$.

3. **Multiplicities**: For each column $\ell$ and chunk $k$, compute the multiplicity vector $m_k^{(\ell)}[t]$ = number of times subtable entry $t$ appears. Aggregate across chunks: $m_{\text{agg}}^{(\ell)}[t] = \sum_k m_k^{(\ell)}[t]$. Absorb all aggregated multiplicities.

4. **$\beta$ challenge**: Squeeze from transcript.

5. **Inverse witnesses**: For each chunk value, compute $u_k^{(\ell)}[i] = 1 / (\beta - c_k^{(\ell)}[i])$. For the subtable: $v[t] = 1 / (\beta - T[t])$. Absorb all inverse vectors.

6. **$\gamma$ challenge**: Squeeze batching challenge.

7. **Batched identity polynomial $H$**: Precompute the entire batched identity at all $2^n$ points:

$$H[i] = \sum_\ell \Bigl[ \sum_{k} \gamma^{\ell(K+1)+k} \cdot \bigl((\beta - c_k[i]) \cdot u_k[i] - 1\bigr) + \gamma^{\ell(K+1)+K} \cdot \Bigl(\sum_k u_k[i] - m_{\text{agg}}[i] \cdot v[\text{tbl\_idx}]\Bigr) \Bigr]$$

   This batches two types of sub-identities: **inverse correctness** (each $u_k$ really is $1/(\beta - c_k)$) and **balance** (the sum of chunk inverses equals the multiplicity-weighted table inverse).

8. **Sumcheck** (degree 2): Build just two MLEs — `eq_r` and `H` — and run `MLSumcheck::prove` with combination function $f = \text{eq\_r} \cdot H$. The key optimization: by precomputing $H$ outside the sumcheck loop, the number of MLEs drops from $2 + L(2K+1)$ to just 2 and the degree from 3 to 2.

**Proof data** (serialized into the proof):
- `aggregated_multiplicities`: $L \times T$ field elements ($L$ columns × 16 subtable entries).
- `chunk_inverse_witnesses`: $L \times K \times N$ field elements ($L$ columns × 8 chunks × 512 rows). This is the dominant proof component.
- `inverse_table`: $T = 16$ field elements.
- Sumcheck proof: 9 round messages (degree-2 univariates).

### 11b. `PIOP/ShiftSumcheck`

**Function**: `shift_sumcheck_prove(&mut transcript, &shift_claims, &shift_trace_columns, num_vars, &field_cfg)`

Benchmarks the **shift sumcheck prover** as a standalone step. The shift sumcheck reduces the shifted-column evaluation claims from the CPR `down_evals` to plain MLE evaluation claims on the unshifted source columns at a new random point $\mathbf{s}$, which the PCS can then open directly. This step is described in full generality in §9a above; below we detail the concrete benchmark invocation and internal algorithm steps.

#### Benchmark setup (untimed)

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

### 12. `PCS/Prove (folded)`

**Function**: `ZipPlus::<FoldedZt, FoldedLc>::prove::<F, UNCHECKED>(&folded_params, &split_trace, &folded_pcs_point, &folded_hint)`

Opens the committed `BinaryPoly<16>` columns at the extended evaluation point $(\mathbf{r} \| \gamma)$ — a 10-dimensional point derived from the PIOP sumcheck point (9 dims) plus the folding challenge $\gamma$.

**Steps**:

1. **Create PCS transcript**: Initialize a separate `PcsTranscript` (Keccak sponge + byte stream), squeeze its own random field configuration.

2. **Tensor decomposition**: Split the 10-dimensional point into a row-selector $\mathbf{q}_0$ (empty, since `num_rows = 1`) and a column-selector $\mathbf{q}_1$ (all 10 coordinates).

3. **FS challenge combination**: For each committed polynomial, squeeze `degree_bound + 1` challenges $\boldsymbol{\alpha}$ from the PCS transcript. Compute a combined ring element per codeword cell: $\text{CombR}[i] = \langle \text{BinaryPoly<16>}[i], \boldsymbol{\alpha} \rangle$ (inner product of the 16 coefficients with the challenges, projecting `BinaryPoly<16>` to a scalar).

4. **Compute $\mathbf{b}$**: For each row, $b[\text{row}] = \sum_{\text{poly}} \langle \text{CombR}[\text{row}], \mathbf{q}_1 \rangle$ — inner product of the combined row with the column-selector, summed across polynomials. Absorb $\mathbf{b}$.

5. **Evaluation**: $\text{eval} = \langle \mathbf{q}_0, \mathbf{b} \rangle = b[0]$ (since there is only 1 row).

6. **Combined row**: Squeeze `num_rows` coefficients $s_j$ (just 1 here), compute the combined codeword row across all polynomials, write to transcript.

7. **Grinding**: `transcript.grind(GRINDING_BITS)` — proof-of-work nonce search. The prover searches for a nonce that makes `H(transcript_state \| nonce)` have the required number of leading zero bits, adding computational security. In this benchmark configuration `GRINDING_BITS = 8`, so the prover searches for a nonce producing 8 leading zero bits.

8. **Column openings**: Squeeze `NUM_COLUMN_OPENINGS = 131` random column indices from the transcript. For each index, open the Merkle path (authentication siblings from leaf to root) and include the column values from all committed polynomial codeword matrices. The verifier will check: (a) the Merkle path is valid, (b) the column values are consistent with the combined row, (c) the combined row is a valid codeword.

**Output**: `(F, ZipPlusProof)` — the claimed evaluation and the serialized proof (transcript bytes including Merkle paths, column data, and grinding nonce).

### 13. `PCS/Prove (original)` *(comparison)*

**Function**: `ZipPlus::<OrigZt, OrigLc>::prove::<F, UNCHECKED>(&orig_params, &sha_pcs_trace, &orig_pcs_point, &orig_hint)`

Same algorithm as step 12, but operating on the original `BinaryPoly<32>` columns with the non-folded parameters:
- Point is 9-dimensional (no folding extension).
- Codeword elements are `BinaryPoly<32>` (256 B each, vs. 128 B for folded).
- Merkle tree depth is 11 (2048 leaves).
- Each of the 131 column openings reveals `BinaryPoly<32>` values, which are 2× larger.

This is benchmarked to measure the PCS proving time difference attributable to column folding. The folded variant has more evaluations (1024 vs 512, i.e. `row_len` doubles) but smaller elements, and the Merkle paths are deeper (12 vs 11) but narrower per leaf. Both variants have `num_rows = 1`.

---

## End-to-End Pipeline Benchmarks

### Prover benchmarks

| # | Benchmark ID | What it runs |
|---|---|---|
| 14 | `E2E/Prover (folded)` | `prove_classic_logup_folded` — full 2× folded pipeline (split → commit → IC → batched CPR+Lookup MD-sumcheck → fold → PCS prove). *CPR = CombinedPolyResolver = "Main field sumcheck" in the per-step breakdown.* |
| 15 | `E2E/Prover (original)` | `pipeline::prove` — non-folded baseline (commit → IC → CPR → lookup → PCS prove) |
| 18 | `E2E/Prover (GKR folded)` | `prove_gkr_logup_folded` — 2× folded with GKR lookup (separate CPR sumcheck + GKR fractional-sumcheck, no multi-degree batching) |
| 19b | `E2E/Prover (GKR folded 4-chunk)` | Same as above but using 4×$2^8$ chunk decomposition instead of 8×$2^4$ |
| 19c | `E2E/Prover (Hybrid GKR c=2 folded)` | `prove_hybrid_gkr_logup_folded` — sends first 2 GKR layers in the clear for proof size reduction |
| 20 | `E2E/Prover (4x folded)` | `prove_classic_logup_4x_folded` — 4× folded with classic lookup, 8-chunk |
| 22 | `E2E/Prover (4x folded 4-chunk)` | Same with 4-chunk lookup |
| 22b | `E2E/Prover (4x Hybrid GKR c=2 4-chunk)` | `prove_hybrid_gkr_logup_4x_folded` — 4× folded + hybrid GKR + 4-chunk |
| 22c | `E2E/Prover (4x Hybrid GKR c=1 4-chunk)` | `prove_hybrid_gkr_logup_4x_folded` — 4× folded + hybrid GKR (cutoff=1) + 4-chunk |
| 23 | `E2E/Prover (8x Hybrid GKR c=2 4-chunk)` | `prove_hybrid_gkr_logup_8x_folded` — 8× folded + hybrid GKR + 4-chunk, 3-round fold (32→16→8→4) |

### Verifier benchmarks

Each prover variant has a corresponding verifier:

| # | Benchmark ID | Verifier function |
|---|---|---|
| 16 | `E2E/Verifier (folded)` | `verify_classic_logup_folded` |
| 17 | `E2E/Verifier (original)` | `pipeline::verify` |
| 19 | `E2E/Verifier (GKR folded)` | `verify_classic_logup_folded` (same verifier handles both classic and GKR proof formats) |
| 19b | `E2E/Verifier (GKR folded 4-chunk)` | `verify_classic_logup_folded` |
| 19c | `E2E/Verifier (Hybrid GKR c=2 folded)` | `verify_classic_logup_folded` |
| 21 | `E2E/Verifier (4x folded)` | `verify_classic_logup_4x_folded` |
| 22 | `E2E/Verifier (4x folded 4-chunk)` | `verify_classic_logup_4x_folded` |
| 22b | `E2E/Verifier (4x Hybrid GKR c=2 4-chunk)` | `verify_classic_logup_4x_folded` |
| 22c | `E2E/Verifier (4x Hybrid GKR c=1 4-chunk)` | `verify_classic_logup_4x_folded` |
| 23 | `E2E/Verifier (8x Hybrid GKR c=2 4-chunk)` | `verify_classic_logup_8x_folded` |

For each verifier run, a one-shot timing breakdown is printed to stdout showing: IC verify, CPR+Lookup verify (CPR = CombinedPolyResolver, i.e. Main field sumcheck), Lookup verify, PCS verify, and total.

---

## Verifier Step-by-Step Breakdown (2× Folded)

Beyond the end-to-end verifier timings, the benchmark dissects the 2× folded verifier into 8 individual steps. Each step replays the Fiat-Shamir transcript through all preceding steps (untimed) before timing the target step, ensuring transcript state is correct.

### V1. `V/FieldSetup`

**Function**: `KeccakTranscript::new()` + `transcript.get_random_field_cfg::<PiopField, _, MillerRabin>()`

Identical to prover step 5. Initializes a fresh Keccak transcript and derives the PIOP field configuration by squeezing a candidate prime and running Miller-Rabin. The verifier must derive the **same** field as the prover (deterministic from the empty transcript state). This is the verifier's "cold start" cost.

### V2. `V/Ideal Check`

**Function**: `IdealCheckProtocol::<PiopField>::verify_as_subprotocol::<Sha256Uair, _, _>(...)`

Verifies the Ideal Check proof (the combined MLE values produced by prover step 7):

1. **Draw evaluation point $\mathbf{r}$**: Squeeze the same $n = 9$ field challenges from the transcript. Since the transcript is in the same state as the prover's at this point, the verifier gets the identical random point.

2. **Absorb proof values**: Read the `combined_mle_values` from the proof (one `DynamicPolynomialF<PiopField>` per constraint group) and absorb them into the transcript in the same order as the prover.

3. **Collect ideals**: `collect_ideals::<Sha256Uair>(num_constraints)` enumerates all ideals from the UAIR definition (for SHA-256, the cyclotomic ideal $\langle X^{32} + 1 \rangle$, the degree-one ideal $\langle X - 2 \rangle$ for carry propagation, and the zero ideal).

4. **Batched ideal membership check**: For each combined MLE value $p(Y)$, verify that $p(Y) \equiv 0 \pmod{I}$ where $I$ is the corresponding ideal. For the cyclotomic ideal, this means checking that $p(Y)$ is divisible by $Y^{32} + 1$ when viewed as a polynomial in the quotient ring. For the degree-one ideal $\langle Y - 2 \rangle$, this means $p(2) = 0$ (evaluation at the root). For the zero ideal, this means $p(Y) = 0$. This is a **cheap algebraic check** — no trace data is needed, just polynomial arithmetic on the short proof values.

5. **Return subclaim**: `VerifierSubClaim { evaluation_point: r, values: combined_mle_values }` — the claim that the main field sumcheck must resolve.

### V3. `V/Main field sumcheck Pre`

**Function**: `CombinedPolyResolver::<PiopField>::build_verifier_pre_sumcheck::<Sha256Uair>(...)`

Prepares the verifier's state for the multi-degree sumcheck that batches the main field sumcheck (CPR) and lookup:

1. **Draw batching challenge $\alpha_{\text{fold}}$**: Squeeze from transcript (same as prover's folding challenge in step 9).

2. **Project IC subclaim values**: The IC subclaim contains `DynamicPolynomialF` values. These are evaluated at the projecting element $\alpha$ to produce scalar expected evaluations. The scalars from step 6 (`project_scalars`) are similarly specialized.

3. **Compute expected claimed sum**: $\text{expected} = \sum_i \alpha_{\text{fold}}^i \cdot v_i(\alpha)$, where $v_i$ are the IC combined MLE values projected to $F$.

4. **Verify claimed sum**: Check that the batched proof's `claimed_sums[0]` matches `expected`. This ensures the prover's sumcheck claim is consistent with the Ideal Check output.

5. **Also prepares lookup**: `replay_lookup_pre()` is called, which for each lookup group:
   - Regenerates the subtable and shifts from the `LookupTableType` and the projecting element $\alpha$.
   - Calls `BatchedDecompLogupProtocol::build_verifier_pre_sumcheck()`, which:
     - Reads and absorbs the group's aggregated multiplicities from the proof.
     - Squeezes the $\beta$ challenge, reads and absorbs the chunk inverse witnesses and table inverses.
     - Squeezes the $\gamma$ challenge.
     - Returns a `LookupVerifierPreSumcheck` containing all the verifier state needed to finalize after the shared sumcheck.

**Output**: `CprVerifierPreSumcheck { folding_challenge_powers, ic_evaluation_point }` plus the lookup pre-sumcheck states and the shared `num_vars` for the multi-degree sumcheck.

### V4. `V/MDSumcheck`

**Function**: `MultiDegreeSumcheck::<PiopField>::verify_as_subprotocol(&mut transcript, shared_nv, &batched_proof.md_proof, &field_cfg)`

Verifies the **multi-degree sumcheck** — a single sumcheck that batches the CPR (degree 4) and lookup (degree 2) into one protocol with shared random challenges:

1. **Absorb metadata**: number of variables, number of degree groups, each group's degree. Must exactly mirror the prover's absorptions.

2. **For each of `shared_nv` rounds** ($= 9$ for SHA-256):
   - For each degree group $g$: read the round message (a degree-$d_g$ univariate polynomial, represented by its $d_g$ "tail evaluations" at points $2, 3, \ldots, d_g$; the evaluation at 0 and 1 are implicit from the running sum).
   - **Squeeze one shared challenge $r_i$**: a single random field element shared across all groups. This is what makes degrees "share" randomness — each group gets the same challenge but uses different degree polynomials.
   - Update each group's running sum: verify $P_g(0) + P_g(1) = \text{running\_sum}_g$, then set $\text{running\_sum}_g \leftarrow P_g(r_i)$.

3. **Generate subclaims**: After all rounds, for each group, check the final round consistency and output the expected evaluation at the shared point $\mathbf{s} = (r_0, r_1, \ldots, r_8)$.

**Output**: `MultiDegreeSubClaims { point, expected_evaluations }` — the shared evaluation point and one expected evaluation per degree group (group 0 = CPR, group 1+ = lookup groups).

### V5. `V/Main field sumcheck Finalize`

**Function**: `CombinedPolyResolver::<PiopField>::finalize_verifier::<Sha256Uair>(...)`

Checks that the CPR subclaim is consistent with the prover-supplied column evaluations:

1. **Reconstruct public column evaluations**: The UAIR declares 12 public columns (known to the verifier in full). For each public column, the verifier: (a) projects all $2^9 = 512$ `BinaryPoly<32>` entries to $F$ using the precomputed projection closure, (b) builds a `DenseMultilinearExtension<F>` from the projected values, and (c) evaluates the MLE at the sumcheck point $\mathbf{s}$. Each MLE evaluation costs $O(2^9)$ field operations. With the `parallel` feature, these 12 evaluations run concurrently via Rayon. The results are merged with the prover-supplied `up_evals` (inserting public evals at the correct column indices via `reconstruct_up_evals`) to form `full_up_evals`.

2. **Compute `eq_r` at $\mathbf{s}$**: Evaluate $\text{eq}(\mathbf{r}, \mathbf{s})$ — this is a single product of linear terms: $\prod_{i=0}^{n-1} (r_i \cdot s_i + (1-r_i)(1-s_i))$. If the sumcheck point is longer than `num_vars` (due to multi-degree padding), extra coordinates are accounted for.

3. **Compute last-row selector at $\mathbf{s}$**: $\text{sel}(\mathbf{s}) = \prod_{i=0}^{n-1} s_i$ (the eq function at the all-ones point).

4. **Re-evaluate constraints**: Apply `ConstraintFolder` to the prover-supplied `up_evals` and `down_evals`, using the projected scalars from the pre-sumcheck phase. This computes $\sum_i \alpha_{\text{fold}}^i \cdot g_i(\text{up\_evals}, \text{down\_evals})$.

5. **Check the identity**: Verify that $\text{eq\_r} \cdot (1 - \text{sel}) \cdot \text{folded\_constraints} = \text{expected\_evaluation}$ (from the sumcheck subclaim). If this fails, the proof is invalid.

6. **Absorb `up_evals` and `down_evals`** into the transcript (must match prover's absorption order).

**Output**: The verifier now holds per-column evaluation claims (`up_evals[j]` = the claimed value of $\text{MLE}[\text{col}_j](\mathbf{s})$), which will be checked by the PCS via the folding protocol.

### V5b. `V/ShiftSumcheckVerify`

After CPR and lookup finalization, if `FoldedZincProof.shift_sumcheck` is `Some`, the verifier:

1. **Reconstructs shift claims**: For each of the 12 `ShiftSpec` entries in the UAIR signature, builds a `ShiftClaim<F>` from the CPR `down_evals` (as claimed evaluations) and the CPR evaluation point.

2. **Deserializes the proof**: Parses `SerializedShiftSumcheckProof.rounds` into `ShiftRoundPoly<F>` (3 field elements each) and `v_finals` into field elements.

3. **Dispatches to the appropriate verifier**:
   - **If public shifts exist** (4 out of 12 for SHA-256: shifts sourcing `W_TM2`, `W_TM7`, `W_TM15`, `W_TM16`): uses the **split API**. Calls `shift_sumcheck_verify_pre()` to replay sumcheck rounds and obtain the challenge point $\mathbf{s}$. Then computes the 4 public `v_finals` by evaluating public source column MLEs at $\mathbf{s}$. Reconstructs full `v_finals` via `reconstruct_shift_v_finals()` (interleaving 8 private + 4 public entries). Calls `shift_sumcheck_verify_finalize()` to check the final claim.
   - **Otherwise**: calls the monolithic `shift_sumcheck_verify()` with all `v_finals` from the proof.

4. The resulting `ShiftSumcheckVerifierOutput.v_finals` provides evaluation claims on the unshifted source columns at $\mathbf{s}$, which must be verified by the PCS.

### V6. `V/FoldingVerify`

**Function**: `fold_claims_verify(&mut transcript, &c1s, &c2s, &original_evals, &alpha_power, &cpr_point, &field_cfg)`

**Transcript replay**: Before timing, the benchmark replays through IC → CPR pre → Lookup pre → MD-sumcheck → CPR finalize → Lookup finalize → **Shift sumcheck verify** (if present). All these are untimed; only the folding verify itself is measured.

This is the step that connects the PIOP claims to the PCS commitment. After the PIOP produces an evaluation claim on the original `BinaryPoly<32>` columns, the folding protocol reduces it to a claim on the split `BinaryPoly<16>` columns:

1. **Deserialize $c_1, c_2$**: Read one pair $(c_1[j], c_2[j])$ per committed column from the proof bytes.

2. **Filter and collect original evals**: From the full `up_evals`, extract only the PCS-committed columns (exclude public and shift-source columns that are not committed).

3. **Precompute $\alpha^{16}$**: `compute_alpha_power(&projecting_element, 16)` — computes $\alpha^{16}$ via repeated squaring (efficient since 16 = $2^4$).

4. **Consistency check** (per column $j$):
   $$c_1[j] + \alpha^{16} \cdot c_2[j] \stackrel{?}{=} \text{original\_eval}[j]$$
   This verifies that the split is valid: the low-half MLE evaluation ($c_1$) plus $X^{16}$ (evaluated at $\alpha$) times the high-half MLE evaluation ($c_2$) reconstructs the original column MLE evaluation. If any check fails, the proof is rejected.

5. **Absorb** all $c_1, c_2$ values into the transcript (same order as the prover).

6. **Squeeze $\beta$**: Random challenge from transcript (advances state but not directly used in 2× folding).

7. **Squeeze $\gamma$**: The folding random challenge.

8. **Build extended PCS point**: $\mathbf{r}_{\text{pcs}} = (s_0, s_1, \ldots, s_8, \gamma)$ — the 10-dimensional point at which the PCS will be opened.

9. **Compute new claimed evaluations**: $d[j] = (1 - \gamma) \cdot c_1[j] + \gamma \cdot c_2[j]$ — the linear interpolation, which equals $\text{MLE}[v'_j](\mathbf{r}_\text{pcs})$.

**Output**: `FoldingVerifierOutput { new_point, new_evals }` — the PCS point and the per-column claims that the PCS opening must match.

### V7. `V/PCSVerify (folded)`

**Function**: `ZipPlus::<FoldedZt, FoldedLc>::verify_with_field_cfg::<PiopField, UNCHECKED>(...)`

Verifies the Zip+ PCS opening proof — confirming that the committed polynomials do indeed evaluate to the claimed values at the point $\mathbf{r}_\text{pcs} = (\mathbf{s} \| \gamma)$:

1. **Create PCS transcript**: Initialize a `PcsTranscript` from the serialized proof bytes (`pcs_proof_bytes`). Squeeze the PCS field configuration (must match the prover's PCS transcript).

2. **Read $\mathbf{b}$ from proof stream**: The prover's combined row-selector inner products. Absorb into PCS transcript.

3. **Recompute evaluation**: $\text{eval} = \langle \mathbf{q}_0, \mathbf{b} \rangle$ (with `num_rows = 1`, this is just $b[0]$). Verify it matches the claimed evaluation from the proof.

4. **Read combined row**: The prover sent one combined codeword row (a vector of 4096 combined coefficients). Absorb it.

5. **Verify grinding**: Check that the proof-of-work nonce produces the required number of leading zero bits.

6. **Column openings** (the main verification cost): Squeeze 131 random column indices. For each:
   - Read the Merkle authentication path and the column values for all committed polynomials.
   - **Verify Merkle path**: Hash the leaf data and check siblings up to the root. The root must match the commitment.
   - **Consistency check**: Verify that the column values, combined with the PCS FS challenges, produce the same combined-row value that the prover claimed.
   - **Proximity check**: The combined row is verified to be a valid codeword of the linear code. This is done by checking the opened column positions against the linear code's parity structure.

The PCS verification does **not** need the original polynomial data — it works entirely from the commitment (Merkle root), the proof, and the claimed evaluation point.

**Output**: `Ok(())` if all checks pass; error otherwise. If this step succeeds, the verifier is convinced that the committed polynomials evaluate to the claimed values at $\mathbf{r}_\text{pcs}$, which (via the folding protocol and PIOP) implies the original trace satisfies all UAIR constraints.

---

## Proof Size Analysis

The benchmark computes and prints detailed proof size breakdowns for every pipeline variant. The components measured are:

| Component | Description |
|---|---|
| **PCS** | Zip+ opening proof bytes (Merkle paths, column data, consistency hashes) |
| **IC** | Ideal Check proof (combined MLE values at the random point) |
| **CPR sumcheck** | Main field sumcheck round messages + claimed sum |
| **CPR evals** | Up-evaluations (column evals at the sumcheck point) and down-evaluations (shifted trace evals) |
| **Lookup** | Aggregated multiplicities, chunk inverse witnesses (classic) or GKR proof data (GKR/Hybrid) |
| **Shift SC** | Shift sumcheck proof — $m$ degree-2 round polynomials (3 field elements each) + per-claim `v_finals` (source column MLE evaluations at the shift sumcheck challenge point). Present in all pipeline variants benchmarked here (both original and all folded/batched variants). For SHA-256, 12 shift claims produce 9 round polys and 8 private `v_finals` (4 are public and excluded). See §9a for details. |
| **Folding** | $c_1, c_2$ values per column (2× folding), $c_1, c_2, c_3, c_4$ (4× folding), or $c_1, c_2, c_3, c_4, c_5, c_6$ (8× folding) |
| **Eval point** | Serialized evaluation point coordinates |
| **PCS evals** | Serialized claimed PCS evaluation values |

All bytes are concatenated and compressed with Deflate (default compression level) to show the **compressed proof size**, which is the metric that matters for transmission.

### Comparison tables printed

1. **4× folded vs 2× folded** — raw and compressed, showing PCS and total savings
2. **Chunk variant comparison** — 2×/4×/8× × 4-chunk/8-chunk and Hybrid GKR (c=1 and c=2)
3. **Folded vs original** — total proof size reduction from folding
4. **Classic vs GKR lookup** — lookup-specific and total proof size differences

---

## GKR and Hybrid GKR Lookup Protocols

The **GKR lookup** and its **Hybrid GKR** optimisation are alternative lookup protocols that replace the classic batched decomposition LogUp described in Step 11. They are based on the GKR fractional sumcheck from *Papini & Haböck, "Improving logarithmic derivative lookups using GKR"* ([ePrint 2023/1284](https://eprint.iacr.org/2023/1284)).

### How the GKR lookup differs from classic LogUp (Step 11)

In the **classic** variant (`BatchedDecompLogupProtocol`), the prover sends three families of auxiliary vectors in the clear:

| Classic proof data | Size |
|---|---|
| Aggregated multiplicities $m_{\text{agg}}^{(\ell)}$ | $L \times T$ field elements |
| Chunk inverse witnesses $u_k^{(\ell)}[i] = 1/(\beta - c_k^{(\ell)}[i])$ | $L \times K \times W$ field elements |
| Shared inverse table $v[t] = 1/(\beta - T[t])$ | $T$ field elements |

The identity $H[i] = 0$ (batching inverse-correctness and balance checks) is then verified via a single degree-2 sumcheck over the precomputed polynomial $H$. This sumcheck is **batched with CPR** in the multi-degree sumcheck (`MultiDegreeSumcheck`) — CPR contributes a degree-4 group and lookup contributes a degree-2 group sharing the same random challenges.

In the **GKR** variant (`GkrBatchedDecompLogupProtocol`), the inverse vectors $u_k, v$ are **eliminated entirely**. Only the aggregated multiplicities are sent. Instead, the prover constructs a binary **fraction tree** and runs a layer-by-layer GKR sumcheck to prove that the fractional sums balance. The resulting GKR proof is $O(L \cdot d^2)$ field elements where $d = \log_2(K \cdot W)$, which is much smaller than the classic $O(L \cdot K \cdot W)$ inverse vectors for large traces.

A key architectural difference: **the GKR lookup runs as a separate sumcheck**, not batched with CPR in the multi-degree sumcheck. The pipeline function `prove_gkr_logup_folded` (and its hybrid variant) runs CPR as a standalone degree-4 `MLSumcheck` first, then runs the GKR fractional sumcheck independently. This is reflected in the E2E tables: row 18 says "separate CPR sumcheck + GKR fractional-sumcheck, no multi-degree batching".

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

Four pipeline entry points use GKR-based lookup, each running CPR as a standalone sumcheck followed by GKR lookup as a separate step (no multi-degree batching):

| Pipeline function | Folding | Lookup variant | Proof type |
|---|---|---|---|
| `prove_gkr_logup_folded` | 2× | Full GKR | `LookupProofData::Gkr` |
| `prove_hybrid_gkr_logup_folded` | 2× | Hybrid GKR (cutoff $c$) | `LookupProofData::HybridGkr` |
| `prove_hybrid_gkr_logup_4x_folded` | 4× | Hybrid GKR (cutoff $c$) | `LookupProofData::HybridGkr` |
| `prove_hybrid_gkr_logup_8x_folded` | 8× | Hybrid GKR (cutoff $c$) | `LookupProofData::HybridGkr` |

All four share the same verifier dispatch: `verify_classic_logup_folded` (for 2×), `verify_classic_logup_4x_folded` (for 4×), or `verify_classic_logup_8x_folded` (for 8×) match on the `LookupProofData` enum and call `verify_gkr_batched_lookup` or `verify_hybrid_gkr_batched_lookup` accordingly. These are defined in [piop/src/lookup/pipeline.rs](piop/src/lookup/pipeline.rs) and integrated in [snark/src/pipeline.rs](snark/src/pipeline.rs).

The pipeline step sequence for GKR/Hybrid GKR folded variants is:

1. Split columns → 2. PCS Commit → 3. Ideal Check → 4. CPR (standalone `MLSumcheck`, degree 4) → 5. GKR/Hybrid GKR Lookup (separate `GkrBatchedDecompLogupProtocol` or `HybridGkrBatchedDecompLogupProtocol`) → 6. Folding → 7. PCS Prove.

Compare with classic folded: steps 4 and 5 are merged into a single `MultiDegreeSumcheck` (CPR at degree 4, lookup at degree 2). The GKR variants sacrifice this batching in exchange for the much smaller lookup proof size.

### Concrete numbers for SHA-256 8× (13 lookup columns, 8 chunks width-4, 512 rows)

The E2E pipeline includes 10 base $Q[X]$ lookups plus 3 affine lookups (Ch, neg-Ch, Maj), giving $L = 13$ total lookup columns. The individual step 11 benchmark uses only the 10 base lookups.

| Parameter | Value |
|---|---|
| $L$ (lookup columns, E2E) | 13 (10 base + 3 affine) |
| $K$ (chunks) | 8 (default) or 4 (4-chunk variant) |
| $W$ (witness rows) | 512 |
| Leaf count per witness tree | $K \times W = 4096$ (8-chunk) or $2048$ (4-chunk), padded to $2^{12}$ / $2^{11}$ |
| Witness tree depth $d_w$ | 12 (8-chunk) or 11 (4-chunk) |
| Subtable size $T$ | $2^4 = 16$ (8-chunk) or $2^8 = 256$ (4-chunk) |
| Table tree depth $d_t$ | 4 (8-chunk) or 8 (4-chunk) |
| Classic lookup inverse witnesses | $13 \times 8 \times 512 = 53{,}248$ field elements (E2E, 8-chunk) |
| Full GKR witness sumcheck rounds | $\frac{12 \times 11}{2} = 66$ (8-chunk) |
| Hybrid GKR ($c = 2$) top rounds | $1$ (layer 1 only; layer 0 is a direct check) |
| Hybrid GKR ($c = 2$) bottom fresh depth | 10, bottom rounds $= 45$ |
| Hybrid GKR ($c = 2$) sent intermediates | $2 \times 13 \times 4 = 104$ field elements |
| Hybrid GKR ($c = 2$) total rounds | $1 + 45 = 46$ (vs 66 for full GKR) |
| Hybrid GKR ($c = 1$) sent intermediates | $2 \times 13 \times 2 = 52$ field elements |
| Hybrid GKR ($c = 1$) total rounds | $0 + 55 = 55$ |

---

## Hybrid GKR Cost Analysis

The benchmark runs a standalone analysis of the **hybrid GKR LogUp** tradeoff described in the preceding section. It instantiates the `analyze_hybrid_costs` function from [piop/src/lookup/hybrid_gkr.rs](piop/src/lookup/hybrid_gkr.rs) and then runs live prove/verify microbenchmarks.

- **Full GKR**: All $d$ layers of the GKR tree are proven via sumcheck. Proof size is $O(L \cdot d^2)$ field elements (sumcheck messages grow per layer) but no auxiliary vectors are sent.
- **Hybrid (cutoff $c$)**: The top $c$ layers are proven via GKR sumcheck, then the $2L \times 2^c$ intermediate fraction values at layer $c$ are sent in the clear. The bottom $d - c$ layers form a fresh (shorter) GKR of depth $d - c$. Sending intermediates trades $O(L \cdot 2^c)$ field elements for a reduction from $\frac{d(d-1)}{2}$ to $\frac{c(c-1)}{2} + \frac{(d-c)(d-c-1)}{2}$ sumcheck rounds.

The benchmark:
1. Prints a cost analysis table with columns: cutoff, top-only proof bytes, full hybrid bytes, top/bottom sumcheck rounds, total rounds, rounds saved, MLE eval ops, sent field elements.
2. Benchmarks prove + verify at cutoffs 1, 2, 3, 4, 5, 6, 8, 10 and the full GKR baseline, each with 100 iterations.

---

## Implementation Optimizations

The following optimizations have been applied to the proving stack. All affect shared library code and therefore benefit **every pipeline variant** benchmarked here (original, 2× folded, 4× folded, 8× folded, classic, GKR, and Hybrid GKR). The same optimizations also apply to the dual-circuit benchmark ([BENCHMARK_SHA256_8X_ECDSA.md](BENCHMARK_SHA256_8X_ECDSA.md)).

### 1. Ideal Check: shared $\text{eq}(\mathbf{r}, \cdot)$ table

**Location**: `piop/src/ideal_check/combined_poly_builder.rs` — `compute_combined_values_mle_first`.

The MLE-first Ideal Check evaluates every trace column and every shifted column at the same random point $\mathbf{r}$. Each inner-product evaluation $\text{MLE}[v](\mathbf{r}) = \sum_{\mathbf{b}} \text{eq}(\mathbf{r}, \mathbf{b}) \cdot v[\mathbf{b}]$ requires the equality table $\text{eq}(\mathbf{r}, \cdot)$, which has $2^n$ entries.

Previously, the eq table was rebuilt independently for each column evaluation. Now a **single precomputed table** is built once via `build_eq_x_r_inner(evaluation_point)` in $O(n)$ and reused across all 30 column evaluations and all 12 shifted-column evaluations. The new functions `evaluate_binary_poly_column_mle_with_eq` and `evaluate_shifted_binary_poly_column_mle_with_eq` accept the precomputed table as a parameter.

> **Note:** Since the unified SHA-256 UAIR has `max_degree == 2`, the pipeline uses the **standard** IC path (`prove_as_subprotocol`) rather than MLE-first. The shared eq-table optimisation described here still applies to any UAIR with `max_degree == 1` (e.g., if the carry constraints were moved to a separate check). For UAIRs with `max_degree > 1`, the standard path projects the full trace to $F[Y]$ before evaluating constraints, which is more expensive but handles nonlinear constraints correctly.

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

---

## Feature Flags

The three features specified on the command line affect performance:

| Feature | Effect |
|---|---|
| `parallel` | Enables Rayon-based parallelism for column splitting, Merkle hashing, NTT encoding, sumcheck, MLE evaluation, etc. |
| `simd` | Enables SIMD-accelerated `BinaryPoly` arithmetic (XOR, AND, shifts) and inner product operations. |
| `asm` | Enables assembly-optimized routines for field arithmetic and hashing (e.g., SHA-256 intrinsics for the Merkle tree, AES-NI for transcript). |

---

## Criterion Configuration

- **Sample size**: 100 (set via `group.sample_size(100)`)
- **Benchmark group**: `"8xSHA256 Folded Steps"`
- **Harness**: Custom (`harness = false` in `Cargo.toml`)
- **Timing method**: Most step benchmarks use standard `iter` (WitnessGen, SplitColumns, Commit, FieldSetup, Project Ideal Check, Project Main field sumcheck, LookupExtract, PCS/Prove, and all E2E Verifier benchmarks). Stateful steps that need transcript replay setup outside the timed region (IdealCheck, Main field sumcheck, Lookup) and E2E Prover benchmarks use `iter_custom`.

---

## Output

Running the benchmark produces:

1. **Criterion statistical output** (stdout) — for each benchmark function, reports mean time, standard deviation, throughput, and regression detection vs. any saved baseline.
2. **Stdout diagnostic output** — verifier step timing breakdowns (printed via `println!` after each E2E Verifier variant).
3. **Stderr diagnostic output** — detailed proof size tables, hybrid GKR cost analysis, pipeline timing summaries, and peak memory usage (printed via `eprintln!`).
4. **Criterion HTML reports** — saved to `target/criterion/8xSHA256 Folded Steps/` for interactive visualization.
