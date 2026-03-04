# Benchmark: `steps_sha256_8x_folded`

```
cargo bench --bench steps_sha256_8x_folded -p zinc-snark --features "parallel simd asm"
```

## Overview

This benchmark provides a **fine-grained, step-by-step timing breakdown** of the Zinc+ proving stack applied to the **8×SHA-256** workload (8 SHA-256 compressions in parallel) using **4× column folding** and the **Hybrid GKR c=2** lookup protocol to reduce proof size. It uses the [Criterion](https://bheisler.github.io/criterion.rs/book/) framework with a custom harness.

The benchmark exercises a **single pipeline variant**:

| Pipeline variant | Lookup method | Column folding | PCS element type | Chunk decomposition |
|---|---|---|---|---|
| **4× folded Hybrid GKR c=2** | Hybrid GKR (cutoff $c = 2$) | 2 splits (32→16→8) | `BinaryPoly<8>` | 4 chunks × $2^{8}$ |

The benchmark uses a **dual UAIR** architecture: `Sha256Uair` (BinaryPoly constraints) and `Sha256UairQx` (Q[X] carry-propagation constraints). Both have `max_degree == 1`. The pipeline function is `prove_hybrid_gkr_logup_4x_folded<Sha256Uair, Sha256UairQx, FoldedZt4x, FoldedLc4x, 32, 16, 8, UNCHECKED>` / `verify_classic_logup_4x_folded`.

Individual prover steps are benchmarked (15 total), including **standalone** shift sumcheck and folding steps. Both **prover** and **verifier** end-to-end times are measured.

The benchmark also reports:
- **Proof size** breakdown (PCS, IC, QX IC, CPR, QX CPR, lookup, shift SC, QX shift SC, folding, etc.) — raw bytes and Deflate-compressed bytes.
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

## Prover Steps Benchmarked (Individual)

These are the per-step Criterion benchmarks for the **4× folded Hybrid GKR c=2** pipeline with 4-chunk lookups. Each step is described at the implementation level, including the mathematical operations, data flow, and what is produced.

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

Initializes the Fiat-Shamir transcript (Blake3-based; the type is still named `KeccakTranscript` for historical reasons) and derives a random 192-bit prime $p$ for the PIOP field `F = MontyField<3>`. The process:

1. Squeeze a candidate integer from the transcript.
2. Run the **Miller-Rabin** primality test. If the candidate is composite, increment and try again.
3. Build the Montgomery-form field configuration (`modulus`, `R`, `R²`, inverses, etc.).

This step is typically fast (a few microseconds) once a prime is found, but has variable latency due to the primality search.

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

**Function**: `ZipPlus::<FoldedZt4x, FoldedLc4x>::prove::<F, UNCHECKED>(&folded_4x_params, &split_trace, &folded_4x_pcs_point, &folded_4x_hint)`

Opens the committed `BinaryPoly<8>` columns at the extended evaluation point $(\mathbf{r} \| \gamma_1 \| \gamma_2)$ — an 11-dimensional point derived from the PIOP sumcheck point (9 dims) plus the two folding challenges $\gamma_1, \gamma_2$.

**Steps**:

1. **Create PCS transcript**: Initialize a separate `PcsTranscript` (Keccak sponge + byte stream), squeeze its own random field configuration.

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
| 14 | `E2E/Prover (4x Hybrid GKR c=2 4-chunk)` | `prove_hybrid_gkr_logup_4x_folded::<Sha256Uair, Sha256UairQx, FoldedZt4x, FoldedLc4x, 32, 16, 8, UNCHECKED>` — full 4× folded pipeline (double-split → commit → IC (BP) → QX IC → CPR → QX CPR → Hybrid GKR Lookup → Shift SC → QX Shift SC → Fold (2-round) → PCS prove), 4-chunk. |

### Verifier benchmarks

| # | Benchmark ID | Verifier function |
|---|---|---|
| 15 | `E2E/Verifier (4x Hybrid GKR c=2 4-chunk)` | `verify_classic_logup_4x_folded::<Sha256Uair, Sha256UairQx, FoldedZt4x, FoldedLc4x, 32, 16, 8, UNCHECKED, ...>` |

For the verifier run, a one-shot timing breakdown is printed to stdout showing: IC verify, CPR+Lookup verify, Lookup verify, PCS verify, and total.

---

## Verifier

The benchmark does **not** include individual verifier step breakdowns (the previous V1–V7 per-step breakdown has been removed). Verifier timing is measured only via the **E2E/Verifier** benchmark (step 15), which runs `verify_classic_logup_4x_folded`.

A one-shot timing breakdown is printed to stdout after each verifier run, showing:
- IC verify
- CPR+Lookup verify
- Lookup verify
- PCS verify
- Total

The verifier internally performs: field setup → IC verify (BP + QX) → CPR verify (BP + QX, standalone sumchecks) → Hybrid GKR lookup verify → shift sumcheck verify (BP + QX) → two-round folding verify → PCS verify. All QX-specific verification steps (QX IC, QX CPR, QX shift sumcheck) run as part of the same pipeline.

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
| **QX Shift SC** | Shift sumcheck proof for `Sha256UairQx` |
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

1. Double-split columns (32→16→8) → 2. PCS Commit (4x-folded) → 3. IC (BinaryPoly, `Sha256Uair`) → 4. QX IC (`Sha256UairQx`, trivial ideal) → 5. CPR (standalone `MLSumcheck`, degree 3) → 6. QX CPR (standalone) → 7. Hybrid GKR Lookup → 8. Shift sumcheck (BP) → 9. QX Shift sumcheck → 10. Folding (2-round) → 11. PCS Prove.

The Hybrid GKR c=2 approach runs CPR as a standalone sumcheck followed by the Hybrid GKR lookup as a separate step (no multi-degree batching). The QX sub-UAIR (`Sha256UairQx`) gets its own IC, CPR, and shift sumcheck — all running independently from the BinaryPoly sub-UAIR.

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

---

## Feature Flags

The three features specified on the command line affect performance:

| Feature | Effect |
|---|---|
| `parallel` | Enables Rayon-based parallelism for column splitting, Merkle hashing, NTT encoding, sumcheck, MLE evaluation, etc. |
| `simd` | Enables SIMD-accelerated `BinaryPoly` arithmetic (XOR, AND, shifts) and inner product operations. |
| `asm` | Enables assembly-optimized routines for field arithmetic and hashing (e.g., SHA-256 intrinsics for the Merkle tree, AES-NI for transcript). |

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
