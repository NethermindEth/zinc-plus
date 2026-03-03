# Benchmark: `steps_sha256_8x_ecdsa`

```
cargo bench --bench steps_sha256_8x_ecdsa -p zinc-snark --features "parallel simd asm"
```

## Overview

This benchmark provides a **fine-grained, step-by-step timing breakdown** of the Zinc+ dual-circuit proving stack applied to **8×SHA-256 + ECDSA** — an SHA-256 hash followed by an ECDSA signature verification — in a single unified proof. It uses the [Criterion](https://bheisler.github.io/criterion.rs/book/) framework with a custom harness.

> For the SHA-256-only folded pipeline see [BENCHMARK_STEPS_SHA256_8X_FOLDED.md](BENCHMARK_STEPS_SHA256_8X_FOLDED.md).

The benchmark compares several dual-circuit pipeline variants:

| Pipeline variant | Lookup method | Column folding | SHA PCS type | ECDSA PCS type |
|---|---|---|---|---|
| **Original** | Classic batched | None | `BinaryPoly<32>` | `Int<4>` |
| **2× folded SHA** | Classic batched | 1 split (32→16) | `BinaryPoly<16>` | `Int<4>` |
| **4× folded SHA** (8-chunk) | Classic batched | 2 splits (32→16→8) | `BinaryPoly<8>` | `Int<4>` |
| **2× folded SHA** (4-chunk) | Classic batched (4-chunk) | 1 split (32→16) | `BinaryPoly<16>` | `Int<4>` |
| **4× folded SHA** (4-chunk) | Classic batched (4-chunk) | 2 splits (32→16→8) | `BinaryPoly<8>` | `Int<4>` |
| **4× Hybrid GKR** (c=2) | Hybrid GKR (cutoff=2) | 2 splits (32→16→8) | `BinaryPoly<8>` | `Int<4>` |
| **4× Dual Hybrid GKR** (c=2) | Hybrid GKR (cutoff=2) | 2 splits (32→16→8) | `BinaryPoly<8>` | `Int<4>` |

For each variant, both **prover** and **verifier** end-to-end times are measured. The 4× Hybrid GKR prover bench is SHA-only (`prove_hybrid_gkr_logup_4x_folded`), while the corresponding verifier bench uses the full dual-circuit proof (`verify_dual_circuit_hybrid_gkr_4x_folded`). The 4× Dual Hybrid GKR entry benchmarks only the dual-circuit prover. The non-folded baseline is further broken into individual prover steps (1–18b) and verifier steps (V1–V9) for a per-phase timing analysis.

The benchmark also reports:
- **Proof size** breakdown (PCS, IC, sumcheck, CPR evals, lookup, eval sumcheck, eval point, folding, etc.) for the baseline, folded, 4×-folded, and Hybrid GKR variants — raw bytes and Deflate-compressed bytes.
- **Classic vs GKR lookup** proof size comparison.
- **Peak memory** usage.
- **Verifier step timing** printed via `eprintln!` for each variant.

---

## The Two Workloads

### Circuit 1: 8×SHA-256 (`Sha256Uair`)

> For a complete description of the SHA-256 UAIR (30 columns, 16 constraints, 10 shift specs, lookup structure, column layout), see [BENCHMARK_STEPS_SHA256_8X_FOLDED.md §"The Workload: 8×SHA-256 over `BinaryPoly<32>`"](BENCHMARK_STEPS_SHA256_8X_FOLDED.md).

The SHA-256 circuit is identical to the single-circuit folded benchmark. Key parameters:

| Parameter | Value |
|---|---|
| `SHA256_8X_NUM_VARS` | 9 ($2^9 = 512$ rows) |
| `SHA256_BATCH_SIZE` | 30 columns (27 bitpoly + 3 integer) |
| `SHA256_LOOKUP_COL_COUNT` | 10 direct + 3 affine virtual |
| `max_degree` | 1 (MLE-first path) |
| PCS-committed columns | 12 (after excluding 12 public + 6 shift-source) |

### Circuit 2: ECDSA Verification (`EcdsaUairInt`)

The ECDSA UAIR defines a trace of **11 columns × 512 rows** (padded from 258 logical rows, `num_vars = 9`). It proves a single ECDSA signature verification over the secp256k1 curve using Jacobian coordinates and Shamir's trick for double-scalar multiplication.

| Index | Column | Description |
|-------|--------|-------------|
| 0 | $b_1$ | Bit of scalar $u_1 = e \cdot s^{-1} \bmod n$ (**public**) |
| 1 | $b_2$ | Bit of scalar $u_2 = r \cdot s^{-1} \bmod n$ (**public**) |
| 2–4 | $X, Y, Z$ | Accumulator point in Jacobian coordinates |
| 5–7 | $X_{\text{mid}}, Y_{\text{mid}}, Z_{\text{mid}}$ | Doubled point $2P$ in Jacobian coordinates |
| 8 | $H$ | Addition scratch (chord x-difference) |
| 9 | `sel_init` | 1 at row 0 (**public**) |
| 10 | `sel_final` | 1 at row 257 (**public**) |

Column values are `Int<4>` (256-bit integers) operating over the secp256k1 base field $\mathbb{F}_p$.

#### Constraints

The ECDSA UAIR specifies **7 non-boundary constraints** and **4 boundary constraints** (11 total):

**Doubling (rows 0–257):**
- C1: $Z_{\text{mid}} = 2 Y Z \pmod{p}$
- C2: $X_{\text{mid}} = 9X^4 - 8XY^2 \pmod{p}$
- C3: $Y_{\text{mid}} = 12X^3 Y^2 - 3X^2 X_{\text{mid}} - 8Y^4 \pmod{p}$

**Addition (Shamir's trick):**
- C4: $H = T_x \cdot Z_{\text{mid}}^2 - X_{\text{mid}} \pmod{p}$ where $T_x$ depends on $(b_1, b_2)$ selecting $G$, $Q$, or $G+Q$
- C5: $(1-\text{sel\_final}) \cdot (Z[t+1] - (1-s) Z_{\text{mid}} - s \cdot Z_{\text{mid}} H) = 0 \pmod{p}$ where $s = b_1 + b_2 - b_1 b_2$
- C6: $(1-\text{sel\_final}) \cdot (X[t+1] - (1-s) X_{\text{mid}} - s (R_a^2 - H^3 - 2 X_{\text{mid}} H^2)) = 0 \pmod{p}$
- C7: $(1-\text{sel\_final}) \cdot (Y[t+1] - (1-s) Y_{\text{mid}} - s (R_a (X_{\text{mid}} H^2 - X[t+1]) - Y_{\text{mid}} H^3)) = 0 \pmod{p}$

> **Transition gating.** C5–C7 are multiplied by $(1 - \text{sel\_final})$ so the transition from the final logical row (257) to the zero-padded tail is not enforced.

**Boundary (4 constraints):**
- B3a: $\text{sel\_init} \cdot (X - T_x) = 0$
- B3b: $\text{sel\_init} \cdot (Y - T_y) = 0$
- B3c: $\text{sel\_init} \cdot (Z - 1) = 0$
- B4: $\text{sel\_final} \cdot Z \cdot (X - R_{\text{SIG}} \cdot Z^2) = 0$

**max_degree**: 13 (degree-12 point-addition formulas × gating by `sel_final`).

**Booleanity** of $b_1, b_2$: checked by the verifier directly via O(N) scan rather than algebraically, since these are public columns.

#### Column Shifts

The ECDSA UAIR declares shift-by-1 specs for columns $X$, $Y$, $Z$ (indices 2–4), since C5–C7 reference $X[t+1], Y[t+1], Z[t+1]$.

#### Column Categories

| Category | Columns | Count |
|---|---|---|
| **Public** | $b_1$ (0), $b_2$ (1), `sel_init` (9), `sel_final` (10) | 4 |
| **Shift source** (PCS-excluded) | $X$ (2), $Y$ (3), $Z$ (4) | 3 |
| **PCS-committed** | $X_{\text{mid}}$ (5), $Y_{\text{mid}}$ (6), $Z_{\text{mid}}$ (7), $H$ (8) | 4 |

#### Witness Generation

**Function**: `EcdsaUairInt::generate_witness(num_vars=9, rng)`

Generates a valid secp256k1 ECDSA trace: a random private key, public key derivation, message hash, signature computation $(r, s)$, and the full scalar-multiplication loop with Shamir's trick. The 258 active rows (rows 0–257) compute the double-and-add loop for $u_1 G + u_2 Q$ in Jacobian coordinates, zero-padded to 512.

---

## Dual-Circuit Architecture

The key difference from [BENCHMARK_STEPS_SHA256_8X_FOLDED.md](BENCHMARK_STEPS_SHA256_8X_FOLDED.md) is the **dual-circuit** design: two independent UAIRs are proved in a single unified proof system, sharing a single Fiat-Shamir transcript and PIOP field.

### Two Fields

| Field | Type | Size | Usage |
|---|---|---|---|
| **PIOP field** $\mathbb{F}_q$ | `F = MontyField<3>` | 192-bit | Shared IC eval point, CPR, lookup, eval sumcheck |
| **ECDSA field** $\mathbb{F}_p$ | `EcdsaField = MontyField<4>` | 256-bit | Per-step ECDSA benchmarks (constraints hold mod secp256k1 $p$) |

In the unified `prove_dual_circuit` pipeline, **both** circuits operate over the same random 192-bit field $\mathbb{F}_q$. The ECDSA trace is projected to $\mathbb{F}_q$ via `project_trace_to_field`, meaning the ECDSA constraints (polynomial identities mod $p$) are evaluated in $\mathbb{F}_q$ rather than $\mathbb{F}_p$. The per-step benchmarks (steps 7–12) measure each circuit in its native field to give realistic per-field cost estimates.

> **Why separate per-step ECDSA field benchmarks?** A production implementation could run separate CPR sumchecks per field to preserve full soundness at the cost of losing batching efficiency. The per-step benchmarks at `EcdsaField` ($\mathbb{F}_p$) quantify the native-field cost. See also the Fixed-C2 pipeline discussion below.

### Two PCS Instances

| PCS | Trace type | Committed columns | Codeword type | Field |
|---|---|---|---|---|
| **PCS₁** (SHA-256) | `BinaryPoly<32>` | 12 | `DensePolynomial<i64, 32>` | `MontyField<3>` (random) |
| **PCS₂** (ECDSA) | `Int<4>` (256-bit) | 4 | `Int<5>` (320-bit) | `MontyField<4>` (secp256k1 $p$) |

Both use `IprsCode` with `PnttConfigF2_16R4B64<1>`, `row_len = 512`, `NUM_COLUMN_OPENINGS = 147`, rate 1/4.

### Shared vs Separate

| Component | Shared or separate? |
|---|---|
| Fiat-Shamir transcript | **Shared** — single `KeccakTranscript` |
| IC evaluation point $\vec{r}$ | **Shared** — one draw of $n = 9$ challenges |
| Projecting element $\alpha$ | **Shared** — one draw |
| Multi-degree sumcheck | **Shared** — Group 0 = SHA CPR, Group 1 = ECDSA CPR, Groups 2+ = SHA lookup |
| Unified eval sumcheck | **Shared** — 44 claims from both circuits |
| PCS transcripts | **Separate** — each PCS has its own independent FS transcript |
| PCS field configs | **Separate** — PCS₁ samples a random prime, PCS₂ uses secp256k1 $p$ |

---

## PCS Configurations

### SHA-256 PCS (Original: `ShaZt` / `ShaLc`)

- **Codeword type**: `BinaryPoly<32>` (256 B per element)
- **Linear code**: `IprsCode` with `PnttConfigF2_16R4B64<1>` — field $\mathbb{F}_{65537}$ ($2^{16}+1$), rate 1/4, base length 64, depth 1 → `INPUT_LEN = 512`, `OUTPUT_LEN = 2048`
- **num_vars**: 9 ($2^9 = 512$ evaluations, `row_len = 512`, 1 row)

> **Note**: This benchmark uses the $\mathbb{F}_{65537}$ PNTT field for all PCS configurations (SHA and ECDSA alike), unlike the folded-only benchmark ([BENCHMARK_STEPS_SHA256_8X_FOLDED.md §"PCS Configurations"](BENCHMARK_STEPS_SHA256_8X_FOLDED.md)) which uses $\mathbb{F}_{12289}$ for the original and 2× folded SHA PCS.

### SHA-256 PCS (2× Folded: `FoldedShaZt` / `FoldedShaLc`)

After splitting `BinaryPoly<32>` → `BinaryPoly<16>`, uses `PnttConfigF2_16R4B16<2>` ($\mathbb{F}_{65537}$), `INPUT_LEN = 1024`, `OUTPUT_LEN = 4096`. Folded `num_vars = 10`.

### SHA-256 PCS (4× Folded: `FoldedSha4xZt` / `FoldedSha4xLc`)

After splitting `BinaryPoly<32>` → `BinaryPoly<8>`, uses `PnttConfigF2_16R4B4<3>` ($\mathbb{F}_{65537}$), `INPUT_LEN = 2048`, `OUTPUT_LEN = 8192`. Folded `num_vars = 11`.

### ECDSA PCS (`EcZt` / `EcLc`)

- **Codeword type**: `Int<5>` (320-bit, the codeword widening of `Int<4>`)
- **Linear code**: `IprsCode` with `PnttConfigF2_16R4B64<1>` — field $\mathbb{F}_{65537}$ ($2^{16}+1$), rate 1/4, `INPUT_LEN = 512`, `OUTPUT_LEN = 2048`
- **num_vars**: 9 ($2^9 = 512$ evaluations, `row_len = 512`, 1 row)
- **Committed columns**: 4 ($X_{\text{mid}}, Y_{\text{mid}}, Z_{\text{mid}}, H$)
- **PCS field** (`FScalar`): `MontyField<4>` — a 256-bit field. The PCS transcript samples its own random prime via Miller-Rabin (independent of both the PIOP $\mathbb{F}_q$ and the secp256k1 base-field $p$).
- **`DEGREE_BOUND = 0`**: `Int<4>` is a scalar (not a polynomial), so $\alpha$-projection in the PCS is a no-op.

> **Note**: `FScalar` is defined separately from `EcdsaField` in the benchmark, even though both are `MontyField<4>`. `FScalar` is the PCS Fiat-Shamir field (random prime from PCS transcript), while `EcdsaField` is the PIOP field for per-step ECDSA benchmarks (fixed secp256k1 $p$).

---

## Column Folding

> See [BENCHMARK_STEPS_SHA256_8X_FOLDED.md §"Column Folding: The Central Technique"](BENCHMARK_STEPS_SHA256_8X_FOLDED.md) for the complete description. The ECDSA circuit is **not folded** — only SHA-256 columns undergo folding.

The folding split applies only to SHA-256's `BinaryPoly<32>` columns. ECDSA's `Int<4>` columns remain as-is, committed via their own PCS₂ instance with no folding overhead.

---

## Prover Steps Benchmarked (Individual)

Steps 1–18b are individually benchmarked in the non-folded pipeline. Each step is timed independently using Criterion, with transcript replay to the correct state for stateful steps.

### 1. `SHA/WitnessGen`

> See [BENCHMARK_STEPS_SHA256_8X_FOLDED.md §1 "WitnessGen"](BENCHMARK_STEPS_SHA256_8X_FOLDED.md) for detail. Generates 30-column `BinaryPoly<32>` trace for $2^9 = 512$ rows.

### 2. `ECDSA/WitnessGen`

**Function**: `EcdsaUairInt::generate_witness(num_vars=9, rng)`

Generates a valid ECDSA trace over secp256k1 with real $\mathbb{F}_p$ witness data. The witness generator:
1. Draws a random private key $d \in [1, n-1]$.
2. Computes the public key $Q = d \cdot G$.
3. Computes a message hash $e$ and signature $(r, s)$ via the ECDSA signing algorithm.
4. Derives scalars $u_1 = e \cdot s^{-1} \bmod n$ and $u_2 = r \cdot s^{-1} \bmod n$.
5. Decomposes $u_1, u_2$ into 256-bit scalar columns $b_1, b_2$.
6. Runs the 258-step double-and-add loop for $u_1 G + u_2 Q$, filling the Jacobian coordinate columns $(X, Y, Z, X_{\text{mid}}, Y_{\text{mid}}, Z_{\text{mid}}, H)$.
7. Fills selector columns: `sel_init[0] = 1`, `sel_final[257] = 1`, all others zero.
8. Zero-pads to 512 rows.

**Output**: `Vec<DenseMultilinearExtension<Int<4>>>` — 11 columns, each $2^9 = 512$ entries.

After generation, the benchmark filters out PCS-excluded columns:
- 4 public columns: $b_1$ (0), $b_2$ (1), `sel_init` (9), `sel_final` (10)
- 3 shift-source columns: $X$ (2), $Y$ (3), $Z$ (4)

Leaving **4 PCS-committed columns** in `ecdsa_pcs_trace`.

### 2b. `SHA/Folding/SplitColumns`

> See [BENCHMARK_STEPS_SHA256_8X_FOLDED.md §2 "Folding/SplitColumns"](BENCHMARK_STEPS_SHA256_8X_FOLDED.md). Splits SHA-256 PCS-committed `BinaryPoly<32>` → `BinaryPoly<16>`.

### 3. `SHA/PCS/Commit`

> See [BENCHMARK_STEPS_SHA256_8X_FOLDED.md §3 "PCS/Commit (folded)"](BENCHMARK_STEPS_SHA256_8X_FOLDED.md) and §4 for original. Both original and folded SHA commits are benchmarked.

### 3b. `SHA/PCS/Commit (folded)`

Commits the split `BinaryPoly<16>` columns using 2× folded parameters.

### 4. `ECDSA/PCS/Commit`

**Function**: `ZipPlus::<EcZt, EcLc>::commit(&ec_params, &ecdsa_pcs_trace)`

Commits the 4 PCS-committed `Int<4>` columns:

1. **Matrix layout**: Each column's 512 evaluations → 1 row of length 512.
2. **Encoding**: IPRS code (`PnttConfigF2_16R4B64<1>`) expands 512 → 2048 `Int<5>` codeword elements per column. With the `parallel` feature, each polynomial's encoding runs on a separate Rayon thread via `cfg_iter!(polys)`. See [§"Implementation Optimizations"](#implementation-optimizations).
3. **Merkle tree**: 2048 leaves, depth 11. Leaf $j$ hashes all 4 polynomials' codeword entries at column $j$.

**Output**: `ZipPlusHint` + `ZipPlusCommitment` (Merkle root).

### 5. `SHA/PIOP/FieldSetup`

> See [BENCHMARK_STEPS_SHA256_8X_FOLDED.md §5 "PIOP/FieldSetup"](BENCHMARK_STEPS_SHA256_8X_FOLDED.md). Transcript init + random 192-bit prime via Miller-Rabin.

### 6. `SHA/PIOP/Project Ideal Check`

> See [BENCHMARK_STEPS_SHA256_8X_FOLDED.md §6 "PIOP/Project Ideal Check"](BENCHMARK_STEPS_SHA256_8X_FOLDED.md). Projects `BinaryPoly<32>` scalars to `DynamicPolynomialF<F>`. Only `project_scalars` is needed (MLE-first path, `max_degree == 1`).

### 7. `ECDSA/PIOP/FieldSetup`

**Function**: `secp256k1_field_config()`

For ECDSA-specific per-step benchmarks, the PIOP field is secp256k1's **fixed** base-field prime $p$ (256-bit). No random prime is sampled; the field configuration is built directly from the known modulus:

$$p = \texttt{0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F}$$

This is instantaneous — no Miller-Rabin primality test is needed.

> In the unified `prove_dual_circuit` pipeline, there is no separate ECDSA field setup; both circuits use the shared 192-bit random field $\mathbb{F}_q$.

### 8. `ECDSA/PIOP/Project Ideal Check`

**Function**: Projects `Int<4>` trace to `DynamicPolynomialF<EcdsaField>` + `project_scalars`.

Since ECDSA constraints have `max_degree = 13 > 1`, the **general path** is used (not MLE-first):

1. **Trace projection**: Each `Int<4>` element is reduced mod $p$ into `EcdsaField`, then wrapped as a degree-0 `DynamicPolynomialF` (a constant polynomial): `DynamicPolynomialF { coeffs: vec![EcdsaField::from_with_cfg(elem, &fcfg)] }`.

2. **Scalar projection**: Each `Int<4>` UAIR scalar is similarly wrapped as a degree-0 `DynamicPolynomialF`.

Since `Int<4>` values are scalar (not polynomial), projection is trivially a field embedding — no coefficient-by-coefficient mapping is needed.

### 9. `SHA/PIOP/IdealCheck`

> See [BENCHMARK_STEPS_SHA256_8X_FOLDED.md §7 "PIOP/IdealCheck"](BENCHMARK_STEPS_SHA256_8X_FOLDED.md). MLE-first path: evaluates 30 column MLEs in `BinaryPoly<32>`, applies linear constraints once, outputs combined MLE values.

### 10. `ECDSA/PIOP/IdealCheck`

**Function**: `IdealCheckProtocol::<EcdsaField>::prove_at_point::<EcdsaUairInt, ...>(...)`

Since `max_degree = 13 > 1`, the **general** (non-MLE-first) path is used:

1. **Draw evaluation point $\vec{r}$**: Squeezed from the transcript (shared with SHA IC in the unified pipeline, separate in per-step benchmark).
2. **Build combined constraint MLEs**: For each row $i \in [0, 512)$, evaluate all 11 ECDSA constraints on the projected column values and combine using the `constrain_general` interface. Each constraint is a polynomial in the 11 column variables with coefficients in $\mathbb{F}_p$.
3. **Evaluate at $\vec{r}$**: Evaluate the combined constraint MLE at the random point.
4. **Absorb proof values** into the transcript.

**Cost**: Dominated by the constraint evaluation loop — 512 rows × 11 constraints, with `max_degree = 13` degree point-addition formulas. This is significantly more expensive than SHA-256's linear constraints.

**Output**: `Proof<EcdsaField>` containing `combined_mle_values` + `ProverState` with `evaluation_point`.

### 11. `SHA/PIOP/Project Main field sumcheck`

> See [BENCHMARK_STEPS_SHA256_8X_FOLDED.md §8 "PIOP/Project Main field sumcheck"](BENCHMARK_STEPS_SHA256_8X_FOLDED.md). Projects scalars to $\mathbb{F}_q$ via $\alpha$ and trace to $\mathbb{F}_q$ via `project_trace_to_field`.

### 12. `ECDSA/PIOP/Project Main field sumcheck`

**Functions**: `project_scalars_to_field(...)` + `project_trace_to_field::<EcdsaField, ...>(...)`

Identical in structure to the SHA step but trivial: since ECDSA's `Int<4>` values project to degree-0 polynomials in `DynamicPolynomialF`, evaluating at $\alpha$ simply returns the constant. The trace projection maps each `Int<4>` → `EcdsaField` (mod $p$ reduction), which is inexpensive as the values are already 256-bit.

### 13. `PIOP/Main field sumcheck` (Unified CPR)

> The single-circuit SHA-256 CPR is described in [BENCHMARK_STEPS_SHA256_8X_FOLDED.md §9 "PIOP/Main field sumcheck"](BENCHMARK_STEPS_SHA256_8X_FOLDED.md). Below we describe the **unified dual-circuit** multi-degree sumcheck.

**Function**: `MultiDegreeSumcheck::prove_as_subprotocol(...)`

The unified pipeline batches both circuits' CPR sumchecks into a **single multi-degree sumcheck** over $\mathbb{F}_q$:

| Group | Contents | Degree |
|---|---|---|
| 0 | SHA-256 CPR (16 constraints of `max_degree=1`) | 3 |
| 1 | ECDSA CPR (11 constraints of `max_degree=13`) | 15 |

The groups run in lockstep with **shared verifier randomness**: one challenge per round, shared across all groups. Each group produces its own round polynomial at its native degree. Over 9 rounds, the multi-degree sumcheck outputs a shared evaluation point $\vec{s} \in \mathbb{F}_q^9$.

**Key difference from single-circuit**: The ECDSA CPR group is degree 15 (13 from constraints + 2 from eq × (1-sel) multiplier), significantly higher than SHA's degree 3. This dominates the per-round cost since higher-degree polynomials require more evaluation points per round.

> **Note on field choice**: In the unified pipeline, ECDSA constraints are evaluated in $\mathbb{F}_q$ (192-bit), not $\mathbb{F}_p$ (256-bit). The per-step benchmark measures the ECDSA CPR over `EcdsaField` separately to quantify native-field cost.

### 13a. `PIOP/UnifiedEvalSumcheck`

> The shift sumcheck protocol is described in full generality in [BENCHMARK_STEPS_SHA256_8X_FOLDED.md §9a "Shift Sumcheck"](BENCHMARK_STEPS_SHA256_8X_FOLDED.md). Below we describe the dual-circuit **unified** evaluation sumcheck, which combines both circuits' eq claims and ECDSA's shift claims into a single instance.

After the multi-degree sumcheck (step 13) and CPR finalization, the prover holds evaluation claims from both circuits. The unified eval sumcheck batches them all:

| Category | Count | Shift | Description |
|---|---|---|---|
| SHA-256 eq claims | 30 | $c = 0$ | One per SHA-256 column at $\vec{s}$ |
| ECDSA eq claims | 11 | $c = 0$ | One per ECDSA column at $\vec{s}$ |
| ECDSA shift claims | 3 | $c = 1$ | For shifted columns $X[t+1], Y[t+1], Z[t+1]$ |
| **Total** | **44** | | |

All 44 claims share the same evaluation point $\vec{s}$ (the multi-degree sumcheck output point).

> **Why no SHA-256 shift claims?** SHA-256 declares 10 shift specs, but each is paired with a **dedicated linking constraint** (C7–C16) in the CPR constraint set that directly equates the shifted column to a committed or public column. This makes shift claims redundant — the CPR already binds every SHA down-evaluation to a PCS-verified value. ECDSA's 3 shift specs have no linking constraints: the shifted values appear directly inside complex recurrence-relation constraints (C5–C7), so the shift claims provide the necessary binding. See [BENCHMARK_STEPS_SHA256_8X_FOLDED.md §9a](BENCHMARK_STEPS_SHA256_8X_FOLDED.md) for the full explanation.
>
> *Note: In the single-circuit pipeline (`prove_generic`), SHA-256 does get its own shift sumcheck for all 10 specs. The dual-circuit pipeline omits circuit 1's shift claims as an optimization enabled by the linking constraints, cutting the total from 54 to 44 claims and the predicate-table groups from 3 to 2.*

#### Table Grouping

Claims sharing the same `(eval_point, shift_amount)` reference an identical predicate table. The prover builds **one predicate table per distinct group**:

| Group | Shift amount | Claims |
|---|---|---|
| 0 | 0 | 41 eq claims (30 SHA + 11 ECDSA) |
| 1 | 1 | 3 ECDSA shift claims ($X, Y, Z$) |

This reduces per-round work from 44 column accesses to 2 group-level operations.

#### Sumcheck Identity

$$\sum_{\vec{b} \in \{0,1\}^n} \left(\sum_{i=0}^{43} \alpha_i \cdot h_i(\vec{b}) \cdot v_i(\vec{b})\right) = \sum_{i=0}^{43} \alpha_i \cdot a_i$$

Round polynomials are degree ≤ 2 over 9 rounds, producing a fresh challenge point $\vec{s}' \in \mathbb{F}_q^9$.

#### Output: `v_finals`

The prover evaluates each source column's MLE at $\vec{s}'$ and absorbs the results. Public column evaluations are excluded from the proof (verifier recomputes them). The fresh point $\vec{s}'$ is forwarded to both PCS instances.

### 14. `SHA/PIOP/LookupExtract`

> See [BENCHMARK_STEPS_SHA256_8X_FOLDED.md §10 "PIOP/LookupExtract"](BENCHMARK_STEPS_SHA256_8X_FOLDED.md). Extracts 10 lookup columns from the field trace + raw integer indices.

### 15. `SHA/PIOP/Lookup`

> See [BENCHMARK_STEPS_SHA256_8X_FOLDED.md §11 "PIOP/Lookup"](BENCHMARK_STEPS_SHA256_8X_FOLDED.md). Classic batched decomposed LogUp for the 10 base lookup columns (3 affine lookups are only included in E2E).

### 15b. `SHA/PIOP/GkrLookup`

> See [BENCHMARK_STEPS_SHA256_8X_FOLDED.md §"GKR and Hybrid GKR Lookup Protocols"](BENCHMARK_STEPS_SHA256_8X_FOLDED.md) for the full GKR lookup description. Benchmarked as a comparison alongside classic lookup.

### 16. `PIOP/Batched Main field sumcheck+Lookup`

**Function**: Multi-degree sumcheck batching CPR + lookup groups in one pass.

This combines step 13 (CPR for both circuits) and step 15 (SHA lookup) into a single `MultiDegreeSumcheck`:

| Group index | Contents | Degree |
|---|---|---|
| 0 | SHA-256 CPR | 3 |
| 1 | ECDSA CPR | 15 |
| 2+ | SHA-256 lookup groups | 2 |

Benchmarked as an A/B comparison with steps 13+15 run separately.

### 17. `SHA/PCS/Prove`

> See [BENCHMARK_STEPS_SHA256_8X_FOLDED.md §12 "PCS/Prove (folded)"](BENCHMARK_STEPS_SHA256_8X_FOLDED.md) and §13 for original.

**Parameters**: 12 committed `BinaryPoly<32>` columns, `row_len = 512`, `DEGREE_BOUND = 31` → squeezes 33 $\alpha$ challenges per polynomial, projecting to `Int<6>`. 147 column openings. PCS evaluation point is $\vec{s}' \in \mathbb{F}_q^9$ (the unified eval sumcheck output point).

### 18. `ECDSA/PCS/Prove`

**Function**: `ZipPlus::<EcZt, EcLc>::prove::<F, UNCHECKED>(...)`

Opens the 4 committed `Int<4>` columns at $\vec{s}'$ (projected from $\mathbb{F}_q$ to `FScalar` via `piop_point_to_pcs_field()`).

**Phase 1 — $\alpha$-projection**: `Int<4>` has `DEGREE_BOUND = 0` (scalar), so each polynomial squeezes a single trivial challenge. Projection is a no-op: `Int<4>` widens to `Int<8>`.

**Phase 2 — Row combination**: Trivial (`num_rows = 1`).

**Phase 3 — Evaluation vector $b$**: Tensor decomposition, inner product with eq table over `FScalar`.

**Phase 4 — Combined row $u'$**: 512 `Int<8>` values (512 × 64 bytes = 32,768 bytes).

**Phase 5 — Column openings (147×)**: 4 polynomials × `Int<5>` (40 bytes) = 160 bytes codeword data + ~344 bytes Merkle proof ≈ 504 bytes per opening. Total: ~74 KB.

**PCS₂ proof total**: ~33 KB (combined row) + ~74 KB (openings) ≈ **107 KB**.

The two PCS instances run **sequentially** in the current implementation (PCS₁ first, then PCS₂). Each has its own independent Fiat-Shamir transcript.

### 18b. `ECDSA/PCS/Prove (via EcdsaField)`

Same as step 18 but opens using the secp256k1 field config directly (for per-step comparison). The evaluation point is projected differently since the PIOP field is `EcdsaField` in this variant.

---

## End-to-End Pipeline Benchmarks

### Prover Benchmarks

| # | Benchmark ID | What it runs |
|---|---|---|
| 19 | `E2E/Prover` | `prove_dual_circuit` — non-folded baseline (commit both PCS → IC both → unified MD-sumcheck → unified eval sumcheck → PCS prove both) |
| 21 | `E2E/Prover (folded SHA)` | `prove_dual_circuit_folded` — 2× folded SHA, 8-chunk lookup |
| 23 | `E2E/Prover (folded SHA 4-chunk)` | `prove_dual_circuit_folded` — 2× folded SHA, 4-chunk lookup |
| 24a | `E2E/Prover (4x-folded SHA)` | `prove_dual_circuit_4x_folded` — 4× folded SHA, 8-chunk lookup |
| 24b | `E2E/Prover (4x-folded SHA 4-chunk)` | Same with 4-chunk lookup |
| 24c | `E2E/Prover (4x Hybrid GKR c=2 4-chunk)` | SHA-only: `prove_hybrid_gkr_logup_4x_folded` |
| 24e | `E2E/Prover (4x Dual Hybrid GKR c=2 4-chunk)` | `prove_dual_circuit_hybrid_gkr_4x_folded` — full dual-circuit prover with Hybrid GKR |

### Verifier Benchmarks

| # | Benchmark ID | Verifier function |
|---|---|---|
| 20 | `E2E/Verifier` | `verify_dual_circuit` |
| 22 | `E2E/Verifier (folded SHA)` | `verify_dual_circuit_folded` |
| 23 | `E2E/Verifier (folded SHA 4-chunk)` | `verify_dual_circuit_folded` |
| 24a | `E2E/Verifier (4x-folded SHA)` | `verify_dual_circuit_4x_folded` |
| 24b | `E2E/Verifier (4x-folded SHA 4-chunk)` | `verify_dual_circuit_4x_folded` |
| 24d | `E2E/Verifier (4x Hybrid GKR c=2 4-chunk)` | `verify_dual_circuit_hybrid_gkr_4x_folded` (dual-circuit) |

Each verifier benchmark also times `feed_forward` and `reconstruct_u1` (step V9) inside the Criterion loop.

For each verifier variant, a one-shot timing breakdown is printed to stderr showing: IC verify, CPR+Lookup verify, Lookup verify, PCS verify, and total.

---

## Fixed-C2 Pipeline (Commented Out)

The benchmark source contains a commented-out `Fixed-C2` pipeline variant that would run the ECDSA circuit over its native secp256k1 field $\mathbb{F}_p$ with **separate** sumchecks:

- **Separate CPR sumcheck** for ECDSA (degree 15 over $\mathbb{F}_p$)
- **Separate eval sumcheck** for ECDSA over $\mathbb{F}_p$
- The proof would contain two separate evaluation points and two separate eval sumcheck proofs

This is not yet implemented in `snark/src/pipeline.rs` but the benchmark scaffolding exists. The primary advantage would be full soundness for ECDSA constraints (evaluation in $\mathbb{F}_p$ where they actually hold, rather than $\mathbb{F}_q$); the cost is losing cross-circuit batching efficiency.

---

## Verifier Steps (Per-Step Breakdown)

The benchmark dissects the non-folded dual-circuit verifier into **9 individual steps**. Each step replays the Fiat-Shamir transcript through all preceding steps (untimed) before timing the target step.

### V1. `V/FieldSetup`

> See [BENCHMARK_STEPS_SHA256_8X_FOLDED.md §V1](BENCHMARK_STEPS_SHA256_8X_FOLDED.md). Transcript init + random 192-bit prime + draw IC evaluation point $\vec{r} \in \mathbb{F}_q^9$.

### V2. `V/Ideal Check`

Verifies IC proofs for **both** circuits:

1. **IC₁ (SHA-256)**: Deserializes `ic1_proof_values`, runs `verify_at_point::<Sha256Uair>` with `TrivialIdeal` (cyclotomic ideal satisfied by construction). Absorbs into transcript.
2. **IC₂ (ECDSA)**: Deserializes `ic2_proof_values`, runs `verify_at_point::<EcdsaUairInt>` with `EcdsaIdealOverF` (zero check for degree-0 polynomials). Absorbs into transcript.

> **Soundness note**: In the unified pipeline, ECDSA constraints are evaluated in $\mathbb{F}_q$ (192-bit) where they do not necessarily vanish. The IC₂ verify absorbs the values for transcript consistency but the algebraic check is weakened.

### V3. `V/Main field sumcheck+LookupPre`

1. **Draw projecting element $\alpha$** from transcript.
2. **CPR₁ pre-sumcheck** (SHA-256): `project_scalars`, `project_scalars_to_field`, then `build_verifier_pre_sumcheck::<Sha256Uair>`. Validates claimed sum against IC evaluation claims.
3. **Lookup pre-sumcheck**: For each lookup group, regenerate subtable + shifts, then `BatchedDecompLogupProtocol::build_verifier_pre_sumcheck` (absorb multiplicities → squeeze $\beta$ → absorb inverses → squeeze $\gamma$).
4. **CPR₂ pre-sumcheck** (ECDSA): Same pattern with `EcdsaUairInt`.

### V4. `V/MDSumcheck`

> See [BENCHMARK_STEPS_SHA256_8X_FOLDED.md §V4](BENCHMARK_STEPS_SHA256_8X_FOLDED.md) for single-circuit. The dual-circuit version verifies the same multi-degree sumcheck structure but with an additional ECDSA CPR group (degree 15).

`MultiDegreeSumcheck::verify_as_subprotocol` replays 9 rounds. Each round absorbs per-group round polynomials (degree 3 for SHA CPR, degree 15 for ECDSA CPR, degree 2 for lookup groups) and draws a shared challenge. Output: shared point $\vec{s} \in \mathbb{F}_q^9$ and per-group expected evaluations.

### V5. `V/Main field sumcheck Finalize`

Finalizes CPR for **both** circuits:

**CPR₁ (SHA-256)**:
1. Deserialize prover-supplied private up/down evaluations.
2. Reconstruct public column evaluations: project each of the 12 public `BinaryPoly<32>` columns to $\mathbb{F}_q$ via the precomputed projection closure, build MLE, evaluate at $\vec{s}$.
3. Merge private + public into `full_up_evals`.
4. Re-evaluate constraints via `ConstraintFolder` and check against sumcheck subclaim.
5. Absorb all evaluations.

**CPR₂ (ECDSA)**:
1. Deserialize prover-supplied private up/down evaluations.
2. Reconstruct public column evaluations: project each of the 4 public `Int<4>` columns to $\mathbb{F}_q$ (via `from_with_cfg`), build MLE, evaluate at $\vec{s}$.
3. Merge, re-evaluate, check, absorb.

> See [BENCHMARK_STEPS_SHA256_8X_FOLDED.md §V5](BENCHMARK_STEPS_SHA256_8X_FOLDED.md) for the SHA-256-only version.

### V6. `V/UnifiedEvalSC`

Verifies the unified evaluation sumcheck (44 claims):

1. **Reconstruct claims**: Build 44 `ShiftClaim` entries — 30 SHA eq + 11 ECDSA eq + 3 ECDSA shift(1).
2. **Deserialize proof**: Parse round polynomials (9 rounds × 3 field elements) and `v_finals`.
3. **Identify public claims**: SHA public columns (indices in `c1_sig.public_columns`) and ECDSA public columns ($b_1, b_2$, `sel_init`, `sel_final`), plus ECDSA public shift sources.
4. **Pre-verify** (`shift_sumcheck_verify_pre`): Replay 9 sumcheck rounds to reconstruct challenge point $\vec{s}'$ and final expected evaluation.
5. **Compute public `v_finals`**: For SHA public columns, project to $\mathbb{F}_q$ and evaluate MLE at $\vec{s}'$. For ECDSA public columns, project `Int<4>` to $\mathbb{F}_q$ and evaluate MLE at $\vec{s}'$.
6. **Reconstruct `full_v_finals`**: Interleave private (from proof) and public (verifier-computed) entries.
7. **Finalize** (`shift_sumcheck_verify_finalize`): Check consistency:
$$\text{final\_claim} = \sum_{i=0}^{43} \alpha_i \cdot S_{c_i}(\vec{s}', \vec{r}_i) \cdot v_{\text{final},i}$$

### V7. `V/LookupFinalize`

> See [BENCHMARK_STEPS_SHA256_8X_FOLDED.md §V5 "V/Main field sumcheck Finalize" (lookup finalize portion)](BENCHMARK_STEPS_SHA256_8X_FOLDED.md). Verifies LogUp decomposition per lookup group. For affine virtual columns, the verifier computes the affine combination from CPR evaluations.

### V8. `V/PCSVerify`

**Both** PCS₁ and PCS₂ verifications run in **parallel** via `rayon::join`:

- **PCS₁ (SHA-256)**: Opens at the evaluation point from the eval sumcheck (9 dimensions for non-folded, 10/11 for 2×/4× folded). Verifies 147 column openings + Merkle paths + proximity test.
- **PCS₂ (ECDSA)**: Opens at $\vec{s}'$ projected to `FScalar`. Verifies 147 column openings for 4 `Int<4>` committed columns.

Running both in parallel is a natural optimization since they have independent transcripts and commitments.

### V9. `V/FeedFwd+Connect`

Two additional verifier-side computations unique to the dual-circuit benchmark:

1. **Feed-forward**: SHA-256 UAIR proves the 64-round state update; the final digest requires $\text{digest}[i] = H_{\text{init}}[i] + \text{state\_final}[i] \pmod{2^{32}}$. The verifier extracts the 8 final working variables and performs 8 wrapping additions.

   The benchmark includes a sanity check: SHA-256("") = `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

2. **Hash-to-ECDSA connection**: The verifier reconstructs $u_1$ from the ECDSA public column $b_1$ bits. In production, the verifier would additionally verify $u_1 = \text{digest} \cdot s^{-1} \bmod n$ using the public signature parameter $s$. The benchmark measures only the extraction + reconstruction cost; the modular-arithmetic assertion is skipped.

---

## Chunk Variant Comparison

The benchmark generates proof size data for 6 configurations, comparing folding depth and chunk decomposition:

| Config | Folding | Chunks | Lookup |
|---|---|---|---|
| 2× folded 8-chunk | 32→16 | 8×$2^4$ | Classic |
| 2× folded 4-chunk | 32→16 | 4×$2^8$ | Classic |
| 4× folded 8-chunk | 32→8 | 8×$2^4$ | Classic |
| 4× folded 4-chunk | 32→8 | 4×$2^8$ | Classic |
| 4× Hybrid GKR (SHA-only) | 32→8 | 4×$2^8$ | Hybrid GKR c=2 |
| 4× Dual Hybrid GKR | 32→8 | 4×$2^8$ | Hybrid GKR c=2 |

For each, a detailed proof size breakdown is printed to stderr with columns for PCS (SHA), PCS (ECDSA), IC, MD sumcheck, CPR evals, eval sumcheck, lookup, folding, eval point, PCS evals, PIOP total, total raw, and Deflate-compressed.

---

## Proof Size Analysis

The benchmark reports detailed proof size breakdowns for the non-folded baseline and all variants:

| Component | Description |
|---|---|
| **PCS (SHA)** | Zip+ opening proof for SHA-256 columns |
| **PCS (ECDSA)** | Zip+ opening proof for ECDSA columns |
| **IC (SHA)** | SHA-256 ideal check proof values |
| **IC (ECDSA)** | ECDSA ideal check proof values |
| **MD sumcheck** | Multi-degree sumcheck messages + claimed sums |
| **CPR evals** | C1 up+down (SHA) + C2 up+down (ECDSA) |
| **Eval sumcheck** | Unified eval sumcheck rounds + private `v_finals` |
| **Lookup** | Classic: aggregated multiplicities + chunk inverses. GKR/Hybrid: GKR layer proofs |
| **Folding** | $c_1, c_2$ per column (2×) or $c_1, c_2, c_3, c_4$ (4×) — SHA only |
| **Eval point** | Serialized evaluation point coordinates |
| **PCS evals** | Claimed PCS evaluation values (both circuits) |

All bytes are Deflate-compressed to show the metric that matters for transmission.

### GKR vs Classic Lookup

A standalone GKR lookup proof is generated on the same data and compared:
- **Classic**: lookup data + MD sumcheck groups 2+ (lookup portion)
- **GKR**: GKR fraction tree proofs (witness-side batched + table-side)

The comparison is printed as "Classic vs GKR Lookup Proof Size" with raw/compressed totals.

> See [BENCHMARK_STEPS_SHA256_8X_FOLDED.md §"GKR and Hybrid GKR Lookup Protocols"](BENCHMARK_STEPS_SHA256_8X_FOLDED.md) for the complete GKR and Hybrid GKR protocol descriptions.

---

## Implementation Optimizations

> See [BENCHMARK_STEPS_SHA256_8X_FOLDED.md §"Implementation Optimizations"](BENCHMARK_STEPS_SHA256_8X_FOLDED.md) for the full description. All optimizations listed there apply equally to this dual-circuit benchmark.

In summary:

| # | Optimization | Affected steps in this benchmark |
|---|---|---|
| 1 | IC shared $\text{eq}(\mathbf{r}, \cdot)$ table | SHA IC (step 9) |
| 2 | IC parallel shifted-column evaluation | SHA IC (step 9) |
| 3 | PCS commit parallel encoding across polynomials | SHA PCS commit (steps 3, 3b), ECDSA PCS commit (step 4) |
| 4 | Shift sumcheck `.with_min_len(256)` | Unified eval sumcheck (step 13a) |
| 5 | Commitment absorption into Fiat-Shamir transcript (before prime derivation) | All E2E prove/verify pipelines |

The ECDSA PCS commit (4 `Int<4>` columns) also benefits from optimization 3, though with only 4 polynomials the speedup is smaller than SHA's 12-column commit.

---

## Feature Flags

> See [BENCHMARK_STEPS_SHA256_8X_FOLDED.md §"Feature Flags"](BENCHMARK_STEPS_SHA256_8X_FOLDED.md).

| Feature | Effect |
|---|---|
| `parallel` | Rayon-based parallelism for column splitting, Merkle hashing, NTT encoding, sumcheck, MLE evaluation, etc. |
| `simd` | SIMD-accelerated `BinaryPoly` arithmetic and inner products. |
| `asm` | Assembly-optimized field arithmetic and hashing. |

---

## Criterion Configuration

- **Sample size**: 100 (`group.sample_size(100)`)
- **Benchmark group**: `"8xSHA256+ECDSA Steps"`
- **Harness**: Custom (`harness = false`)
- **Timing method**: Standard `iter` for stateless steps (WitnessGen, SplitColumns, Commit, FieldSetup, Project IC, LookupExtract, all E2E Verifier benchmarks). `iter_custom` for stateful steps requiring transcript replay (IdealCheck, Main field sumcheck, Lookup, all E2E Prover benchmarks, and per-step Verifier benchmarks V1–V9).

---

## Summary of Differences from Single-Circuit Benchmark

| Aspect | Single-circuit ([BENCHMARK_STEPS_SHA256_8X_FOLDED.md](BENCHMARK_STEPS_SHA256_8X_FOLDED.md)) | Dual-circuit (this benchmark) |
|---|---|---|
| **Circuits** | SHA-256 only | SHA-256 + ECDSA |
| **PCS instances** | 1 (BinaryPoly) | 2 (BinaryPoly + Int<4>) |
| **Trace types** | `BinaryPoly<32>` | `BinaryPoly<32>` + `Int<4>` |
| **PIOP field** | `MontyField<3>` (192-bit, random) | Same — shared by both circuits |
| **IC passes** | 1 | 2 (IC₁ SHA + IC₂ ECDSA) |
| **MD sumcheck groups** | CPR + Lookup | CPR₁ + CPR₂ + Lookup |
| **ECDSA CPR degree** | — | 15 (vs 3 for SHA) |
| **Eval sumcheck claims** | 10 shift claims from SHA | 44 unified (30+11 eq, 3 shift) |
| **Shift claims** | 10 (SHA, various shifts) | 3 (ECDSA shift-by-1 only) |
| **PCS verify** | 1 instance | 2 instances (parallel via `rayon::join`) |
| **Feed-forward** | Not benchmarked | SHA digest extraction + hash-to-ECDSA connection |
| **Booleanity check** | — | Verifier O(N) scan on $b_1, b_2$ |
| **Column folding** | SHA (2×, 4×, 8×) | SHA only (ECDSA Int<4> not folded) |
| **GKR / Hybrid GKR** | Standalone and batched | SHA-only GKR prover + Dual Hybrid GKR prover/verifier |

---

## Output

Running the benchmark produces:

1. **Criterion statistical output** (stdout) — per-benchmark mean time, standard deviation, and regression detection.
2. **Stderr diagnostic output** — proof size breakdowns (baseline, folded, 4× folded, Hybrid GKR, dual Hybrid GKR), classic vs GKR comparison, verifier step timings per variant, SHA-256 digest and ECDSA u₁ values, peak memory usage.
3. **Criterion HTML reports** — saved to `target/criterion/8xSHA256+ECDSA Steps/`.
