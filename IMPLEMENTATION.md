# Zinc+ End-to-End Implementation: What It Does and What It Does Not

**Date:** 2026-02-28

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [The Prover Pipeline in Detail](#3-the-prover-pipeline-in-detail)
4. [The Verifier Pipeline in Detail](#4-the-verifier-pipeline-in-detail)
5. [SHA-256 Arithmetization](#5-sha-256-arithmetization)
6. [ECDSA Arithmetization](#6-ecdsa-arithmetization)
7. [The Dual-Ring Pipeline](#7-the-dual-ring-pipeline)
7b. [The Dual-Circuit Pipeline](#7b-the-dual-circuit-pipeline)
8. [The Split-Trace Architecture](#8-the-split-trace-architecture)
9. [Proof Format and Serialization](#9-proof-format-and-serialization)
10. [Fiat-Shamir Transcript](#10-fiat-shamir-transcript)
11. [Ideal Types and Membership Checks](#11-ideal-types-and-membership-checks)
12. [The Lookup Protocol (LogUp)](#12-the-lookup-protocol-logup)
13. [Witness Generation](#13-witness-generation)
14. [PCS Verifier Optimizations](#14-pcs-verifier-optimizations)
15. [What the Implementation Does NOT Do](#15-what-the-implementation-does-not-do)
16. [Optimization Opportunities](#16-optimization-opportunities)
17. [Test Coverage](#17-test-coverage)
18. [Benchmark Structure](#18-benchmark-structure)
19. [Crate Map](#19-crate-map)

---

## 1. Overview

The Zinc+ implementation is a research SNARK (Succinct Non-interactive Argument of Knowledge) that proves knowledge of a valid SHA-256 hash computation and (partially) an ECDSA signature verification. The system combines:

- A **Polynomial Commitment Scheme (PCS)** called Zip+ (batched, using IPRS codes and FRI-like proximity testing)
- A **Polynomial IOP (PIOP)** consisting of an Ideal Check protocol, a Combined Polynomial Resolver (CPR) with an internal sumcheck, and a **LogUp lookup argument** (with both classic and GKR variants)
- **UAIR constraint systems** (Universal Algebraic Intermediate Representations) for SHA-256 and ECDSA, using a heterogeneous trace model (binary polynomial, arbitrary polynomial, and integer columns)
- A **BLAKE3-based Fiat-Shamir transcript** for non-interactive proof generation

#### What is a "heterogeneous trace"?

In many SNARKs the execution trace is a matrix where every cell holds a value from a single field $\mathbb{F}_p$. In Zinc+ the trace is **heterogeneous**: different columns may hold values from different algebraic structures. Concretely, the `UairSignature` (defined in `zinc-uair`) declares three column categories:

| Category | Rust type | Algebraic domain | Typical usage |
|----------|-----------|-------------------|---------------|
| **Binary polynomial** | `BinaryPoly<D>` | $\{0,1\}^{<D}[X] \subset \mathbb{Z}[X]$ — polynomials with binary coefficients of degree $< D$ | SHA-256 32-bit words (e.g. `a_hat`, `W_hat`) |
| **Arbitrary polynomial** | `DensePolynomial<C, D>` | $\mathbb{Z}^{<D}[X]$ — polynomials with arbitrary integer coefficients | ECDSA field elements over $\mathbb{F}_q[X]$ |
| **Integer** | `Int<N>` | $\mathbb{Z}$ (small bounded integers) | Carry values (e.g. `mu_a`, `mu_e`) |

Each trace row is represented by a `TraceRow` struct that holds three slices — one per category — so constraint logic can operate on all column types in a uniform way. For example, the SHA-256 UAIR has a signature of 14 binary-polynomial columns + 0 arbitrary-polynomial columns + 3 integer columns = 17 total.

This heterogeneity matters because constraints are formulated in the *native ring* of each column type (e.g. bitwise rotation/XOR constraints live naturally in $\mathbb{Z}[X]$, while carry constraints are plain integer equations). Before the PCS can commit and open, every cell is **projected** to $\mathbb{F}_q$: binary and arbitrary polynomials are first coefficient-lifted to $\mathbb{F}_q[X]$ and then evaluated at a random challenge $\alpha$, while integers are directly embedded into $\mathbb{F}_q$. This projection is handled by the decoupled projections module (`piop/src/projections.rs`).

The end-to-end pipeline takes a witness trace (a matrix of ring elements), commits to it via the PCS, proves that the trace satisfies the UAIR constraints via the PIOP, proves that lookup columns belong to prescribed lookup tables via LogUp, and produces a proof that a verifier can check without seeing the trace.

### Key Performance Numbers (MacBook Air M4, 8×SHA-256 + ECDSA, `parallel` + `simd` features)

| Metric | Current | Target |
|--------|---------|--------|
| Combined PCS prover | ~23.8 ms | < 30 ms ✓ |
| Combined PCS verifier (parallel) | ~6.8 ms | < 5 ms |
| Proof size (compressed) | 277 KB (split-SHA, 1×SHA-256) | ≤ 300 KB ✓ |
| Proof size (compressed, 8×SHA-256+ECDSA) | 471 KB (split) / 585 KB (mono) | ≤ 300 KB |

**Lookup protocol comparison (8×SHA-256, 10 BitPoly columns with LogUp):**

| Variant | Prover |  Verifier | Lookup proof size |
|---------|--------|-----------|-------------------|
| Classic LogUp (default in pipeline) | — | — | Larger (sends chunk/inverse vectors) |
| GKR LogUp (available, not wired in) | — | — | Smaller (only multiplicities + GKR layer proofs) |

---

## 2. Architecture

### The Protocol Stack

```
                    ┌─────────────────┐
                    │    Application   │   SHA-256 / ECDSA UAIR
                    └────────┬────────┘
                             │ trace + constraints
                    ┌────────▼────────┐
                    │      PIOP       │   Ideal Check → CPR → (Sumcheck) → Lookup (LogUp)
                    └────────┬────────┘
                             │ evaluation claims + lookup claims
                    ┌────────▼────────┐
                    │       PCS       │   Zip+ (commit → prove → verify)
                    └─────────────────┘
```

### Crate Dependencies

- `zinc-snark` — the top-level pipeline (`pipeline.rs`), glues everything together
- `zinc-piop` — Ideal Check, Combined Poly Resolver, Sumcheck protocols, **LogUp lookup protocol** (classic + GKR variants, decomposition, batched decomposition), decoupled projections
- `zinc-sha256-uair` — SHA-256 constraint system and witness generation (17 columns: 14 BinaryPoly + 3 Int)
- `zinc-ecdsa-uair` — ECDSA constraint system and witness generation (9 columns)
- `zinc-uair` — abstract UAIR trait with heterogeneous trace support (`UairSignature`, `TraceRow`), constraint builders, ideal collectors
- `zinc-poly` — polynomial types: `BinaryPoly<D>`, `DensePolynomial<C, D>`, MLEs
- `zip-plus` — the Zip+ batched PCS (commit, prove, verify — test+evaluate merged into prove/verify)
- `zinc-transcript` — **BLAKE3-based** Fiat-Shamir transcript (backward-compatible `KeccakTranscript` alias)
- `zinc-utils` — shared utilities: field conversion, inner products, `MulByScalar` trait, `Int<N>` type

---

## 3. The Prover Pipeline in Detail

The prover is implemented in `snark/src/pipeline.rs`. There are multiple entry points:

- `prove()` — BinaryPoly-specific, used for SHA-256 single-ring proofs. Includes **classic LogUp** lookup (chunks + inverse vectors, sequential sumchecks).
- `prove_classic_logup()` — Same as `prove()` but **batches** the CPR and classic lookup sumchecks into a single multi-degree sumcheck pass — the CPR's degree-(max_degree+2) sumcheck and each lookup group's degree-2 sumcheck are fused with shared verifier challenges.
- `prove_generic()` — parameterized over any projectable ring `R`, used for ECDSA `Int<4>` proofs. Includes classic LogUp.
- `prove_dual_ring()` — runs two sequential PIOP passes (BinaryPoly + Q[X]) on the same committed trace, used for full SHA-256 proofs.
- `prove_dual_circuit()` — combines two **independent circuits** (e.g. SHA-256 as `BinaryPoly<32>` + ECDSA as `Int<4>`) into a single proving pipeline with shared Fiat-Shamir challenges, shared IC evaluation point, shared projecting element, and a single multi-degree sumcheck spanning both CPR passes plus lookup. Each circuit retains its own PCS.

The trace projection is handled by the **decoupled projections module** (`piop/src/projections.rs`), which supports heterogeneous traces with three column types: binary polynomial, arbitrary polynomial, and integer. Projection occurs in two stages: (1) coefficient lifting to $\mathbb{F}_q[X]$ and (2) evaluation at the projecting element to get $\mathbb{F}_q$ scalars.

The following description covers the common flow.

### Step 1: PCS Commit (Batched)

```
(hint, commitment) = BatchedZipPlus::commit(params, polys)
```

The prover takes the witness trace — a vector of `DenseMultilinearExtension<Eval>`, one per column — and commits to it using the batched Zip+ scheme:

1. **Encode:** For each polynomial (trace column), its evaluations are arranged into a matrix of `num_rows` rows × `row_len` columns, and each row is encoded using the IPRS linear code (expanding from `row_len` to `cw_len` elements, e.g., 512 → 2,048 for rate 1/4). This produces one **codeword matrix** per polynomial, of shape `num_rows × cw_len`.

2. **Concatenate rows:** All codeword matrices are flattened into a single list of rows: all rows of polynomial 0, then all rows of polynomial 1, etc. For a batch of $m$ polynomials with $n$ rows each, this produces $m \times n$ rows total.

3. **Hash columns into Merkle leaves:** The `MerkleTree::new` function hashes **columns**, not rows. For each column index $j \in [0, \text{cw\_len})$, it concatenates the $j$-th element from every row (across all polynomials in the batch): $\text{leaf}_j = \text{blake3}(\text{row}_0[j] \| \text{row}_1[j] \| \ldots \| \text{row}_{mn-1}[j])$. This produces `cw_len` Merkle leaves (e.g., 2,048).

4. **Build Merkle tree:** A single Merkle tree is built over these column-hash leaves. The Merkle root is the **commitment**, binding the prover to all polynomials in the batch simultaneously.

5. A **hint** (all codeword matrices + the Merkle tree) is retained by the prover for later opening phases.

**What this produces:** A single Merkle-root commitment binding the prover to the entire batch of trace polynomials. The batched design means opening a single column index reveals data from all polynomials at once, sharing one Merkle proof per column opening rather than one per polynomial.

**Homogeneous evaluation domain:** All polynomials in the batch must have coefficients in the **same** evaluation ring. The `commit` function accepts `polys: &[DenseMultilinearExtension<Zt::Eval>]`, where `Zt::Eval` is the single associated type fixed by the `ZipTypes` trait instantiation (e.g., `BinaryPoly<32>` or `Int<N>`). There is no support for mixing polynomials with different coefficient domains (e.g., `BinaryPoly<32>` alongside `Int<16>`) within the same batched commitment — Rust's type system enforces that every element of the slice shares the same `Eval` type. Additionally, all polynomials are validated against the same `num_vars` and `expected_num_evals`, further enforcing uniformity across the batch.

### Step 2: PIOP — Ideal Check (IC)

```
(ic_proof, ic_state) = IdealCheckProtocol::prove_as_subprotocol::<BinaryPoly<32>, Sha256Uair>(
    transcript, trace, num_constraints, num_vars, field_cfg
)
```

The Ideal Check protocol operates as follows:

1. **Coefficient lifting $\mathbb{Z}[X] \to \mathbb{F}_q[X]$:** A random prime $q$ is sampled from the Fiat-Shamir transcript via `transcript.get_random_field_cfg()`, establishing the field $\mathbb{F}_q$ for the PIOP. Each trace cell is then lifted from its native ring to a polynomial over $\mathbb{F}_q$ by converting each coefficient individually. For example, a `BinaryPoly<32>` with binary coefficients $p(X) = \sum c_i X^i$ ($c_i \in \{0,1\}$) becomes a `DynamicPolynomialF<F>` with $\mathbb{F}_q$ coefficients $\hat{p}(X) = \sum \hat{c}_i X^i$ where $\hat{c}_i = c_i \bmod q$. This is a coefficient-wise embedding (not evaluation at any challenge), preserving the polynomial structure: the trace lives in $\mathbb{F}_q[X]$ after this step.

2. **Combined polynomial construction:** The UAIR's `constrain_general` method is invoked with `up` and `down` row expressions in $\mathbb{F}_q[X]$. The constraint builder collects the algebraic expressions (products, sums, differences of trace columns and constant scalars) and evaluates them on the lifted trace. Each constraint yields one "combined MLE" — a multilinear extension over the boolean hypercube whose evaluations are the constraint expressions at each row, with each evaluation being a polynomial in $\mathbb{F}_q[X]$.

3. **Evaluation at random point:** A random evaluation point $\mathbf{r} \in \mathbb{F}_q^{\text{num\_vars}}$ is sampled from the transcript. Each combined MLE is evaluated at $\mathbf{r}$, producing a `DynamicPolynomialF<F>` value — still a univariate polynomial over $\mathbb{F}_q$ (the MLE evaluation interpolates the coefficient-level MLEs, yielding one $\mathbb{F}_q$ value per coefficient index, which together form a polynomial in $X$).

4. **Ideal membership check:** The IC prover sends these $\mathbb{F}_q[X]$ evaluated polynomials to the verifier, who checks membership in the corresponding ideals (e.g., $(X^{32}-1)$ for cyclotomic, $(X-2)$ for carry constraints). This check operates on $\mathbb{F}_q[X]$ values, **not** on $\mathbb{F}_q$ scalars.

5. **The IC proof** consists of these evaluated polynomials (`combined_mle_values`).

**Important:** The $\alpha$ challenge that projects $\mathbb{F}_q[X] \to \mathbb{F}_q$ (by evaluating each polynomial at $\alpha$) is sampled later, at the start of the CPR phase (Step 3 below). This ordering is deliberate: the ideal membership check must operate on $\mathbb{F}_q[X]$ polynomials (since ideals like $(X^{32}-1)$ are polynomial ideals), and the projection to scalars happens only after the ideal structure has been verified.

**What this produces:** For each constraint, a claimed evaluation of the combined constraint polynomial at the random point. The verifier will check that each evaluation belongs to the correct ideal.

### Step 3: PIOP — Combined Polynomial Resolver (CPR)

```
(cpr_proof, cpr_state) = CombinedPolyResolver::prove_as_subprotocol::<BinaryPoly<32>, Sha256Uair>(
    transcript, trace_matrix, evaluation_point, projected_scalars,
    num_constraints, num_vars, max_degree, field_cfg
)
```

The CPR reduces the IC's evaluation claims to claims about individual trace column evaluations. Crucially, the CPR is where the **$\alpha$-projection from $\mathbb{F}_q[X]$ to $\mathbb{F}_q$** occurs:

1. **$\alpha$-projection $\mathbb{F}_q[X] \to \mathbb{F}_q$:** A fresh random challenge $\alpha$ is sampled from the Fiat-Shamir transcript (`projecting_element`). The prover evaluates every $\mathbb{F}_q[X]$ trace cell at $\alpha$: each `DynamicPolynomialF<F>` value $\hat{p}(X) = \sum_j c_j X^j$ becomes $\hat{p}(\alpha) = \sum_j c_j \alpha^j \in \mathbb{F}_q$. This flattens the trace from a matrix of $\mathbb{F}_q[X]$ polynomials to a matrix of $\mathbb{F}_q$ scalars. UAIR scalars are also projected from $\mathbb{F}_q[X]$ to $\mathbb{F}_q$ via `project_scalars_to_field`.

2. **Batching:** The verifier (via Fiat-Shamir) samples a random $\beta$, and the prover combines the $k$ constraint claims into a single sumcheck claim: $\sum_{b \in \{0,1\}^n} \text{eq}(\mathbf{r}, b) \cdot [f_0(b) + \beta f_1(b) + \ldots + \beta^k f_k(b)] = v_0 + \beta v_1 + \ldots + \beta^k v_k$

3. **Sumcheck:** The prover runs a sumcheck protocol over this batched sum (now over $\mathbb{F}_q$ scalars). In each round, the prover sends a univariate polynomial (message) that restricts one variable of the multilinear sum. After $n$ rounds, the sum is reduced to a claim about a single point.

4. **Opening:** At the end of sumcheck, the CPR provides the evaluations of each trace column MLE at the sumcheck evaluation point: `up_evals[i]` = column $i$ evaluated at the point, and `down_evals[i]` = column $i$ evaluated at the shifted point (for constraints that reference the next row). These are $\mathbb{F}_q$ scalars.

**What this produces:** A sumcheck proof (round messages + claimed sum) plus the column evaluation vectors `up_evals` and `down_evals`. These constitute the CPR's subclaim: "the committed trace polynomials, when evaluated at this point, yield these values."

### Step 3.5: PIOP — Lookup (LogUp)

```
lookup_proof = prove_batched_lookup_with_indices(
    transcript, columns_f, raw_indices, remapped_specs, projecting_element, field_cfg
)
```

If lookup column specifications are provided, the prover runs a LogUp-based lookup argument to prove that designated trace columns contain only values from prescribed finite lookup tables. The default pipeline (`prove()` and `prove_generic()`) uses **classic Batched Decomposition LogUp** with sequential sumchecks.

1. **Column extraction and projection:** The function `extract_lookup_columns_from_field_trace()` extracts only the trace columns referenced by the lookup specifications from the already-projected field trace (using the same `projecting_element` as the CPR), and collects raw integer indices for efficient decomposition.

2. **Grouping by table type:** Columns with the same `LookupTableType` (e.g., `BitPoly { width: 32 }`) are grouped together into a single lookup batch, amortizing the sumcheck cost across all columns in the group.

3. **Decomposition (for large tables):** Tables larger than $2^8$ entries (controlled by `DECOMP_THRESHOLD = 8`) are decomposed Lasso-style: each $K$-bit entry is split into $\lceil K / 8 \rceil$ chunks of 8 bits each. For a `BitPoly { width: 32 }` table ($2^{32}$ entries), each witness value decomposes into 4 sub-table lookups of $2^8 = 256$ entries, with shift factors ensuring consistency: $w_i = \sum_k \text{shifts}[k] \cdot \text{chunks}[k][i]$.

4. **Classic inverse-vector sumcheck:** The prover sends chunk vectors (decomposed sub-table witnesses) and inverse vectors ($1/(\beta - w_i)$ for witness side, $1/(\beta - T_j)$ for table side) explicitly, then runs a sumcheck proving the log-derivative identity $\sum_i \frac{1}{\beta - w_i} = \sum_j \frac{m_j}{\beta - T_j}$. Aggregated multiplicities $m_j$ are also sent.

5. **What this produces:** A `LookupProofData::Classic` wrapping per-group `BatchedDecompLogupProof` structs (chunk vectors, inverse vectors, aggregated multiplicities, and sumcheck messages) plus `LookupGroupMeta` for verification.

**Batched CPR+Lookup alternative:** The `prove_classic_logup()` entry point also uses classic LogUp but **batches** the CPR and all lookup sumchecks into a single multi-degree sumcheck pass (see Step 3+3.5 below). Instead of running the CPR sumcheck and each lookup sumcheck sequentially (each producing independent transcripts and challenges), the CPR and all lookup groups are fused with shared verifier challenges. This variant produces `LookupProofData::BatchedClassic`.

**GKR LogUp variant:** A GKR-based variant (`prove_gkr_batched_lookup_with_indices`) is implemented in `piop/src/lookup/pipeline.rs` and eliminates the $O(W + N)$ inverse-vector field elements from the proof, replacing them with $O(\log^2 \max(W, N))$ GKR layer proof elements. It is not currently wired into any top-level prover pipeline but can be used directly.

### Step 3+3.5 (Batched): CPR + Lookup Multi-Degree Sumcheck

When using `prove_classic_logup()`, the CPR sumcheck (Step 3) and all classic lookup sumchecks (Step 3.5) are **batched into a single multi-degree sumcheck** rather than run sequentially. This is the `BatchedClassic` proof path.

#### Batched Prover Flow

1. **CPR group construction:** The CPR's `build_prover_group()` method performs all pre-sumcheck work — shifting traces, building `eq(r, ·)` and selector MLEs, drawing the batching challenge $\alpha$ — and returns a `CprSumcheckGroup` containing the degree, MLEs, and combination function, but does NOT run the sumcheck.

2. **Lookup group construction:** For each lookup group (columns sharing the same table type), `BatchedDecompLogupProtocol::build_prover_group()` absorbs chunk vectors and inverses into the transcript, draws challenges $\beta, \gamma, \mathbf{r}$, and returns a `LookupSumcheckGroup` with a degree-2 combination function.

3. **Dimension alignment (zero-padding):** The CPR operates on `num_vars`-dimensional MLEs (e.g., $\log_2(128) = 7$ for 128-row SHA-256 trace), while lookup groups may have larger `num_vars` when the sub-table exceeds the witness length (e.g., $\log_2(256) = 8$ for a `BitPoly { width: 32 }` decomposed sub-table of 256 entries). The prover computes:
   $$\text{shared\_num\_vars} = \max(\text{cpr\_num\_vars}, \max_g \text{lookup\_num\_vars}_g)$$
   If $\text{shared\_num\_vars} > \text{cpr\_num\_vars}$, all CPR MLEs are zero-padded from $2^{\text{cpr\_num\_vars}}$ to $2^{\text{shared\_num\_vars}}$ evaluations. This is sound because the combination function multiplies by both `selector` and `eq(r, ·)`, both of which are zero in the padded region, so the sumcheck claim is unchanged.

4. **Multi-degree sumcheck:** All groups are passed to `MultiDegreeSumcheck::prove_as_subprotocol()` with `shared_num_vars`. The multi-degree sumcheck runs all groups in lockstep — each round, every group sends its round polynomial (of degree equal to the group's degree), and all groups receive the same random challenge. After `shared_num_vars` rounds, every group shares the same evaluation point.

5. **CPR finalize:** `CombinedPolyResolver::finalize_prover()` extracts the column evaluations (`up_evals`, `down_evals`) from the CPR group's 1-variable residual MLEs and absorbs them into the transcript.

6. **Lookup finalize:** For each lookup group, `BatchedDecompLogupProtocol::finalize_prover()` packages the group's sumcheck messages, chunk vectors, inverse vectors, and multiplicities into a `BatchedDecompLogupProof`.

#### Batched Proof Structure

The result is a `BatchedCprLookupProof` containing:
- `md_proof`: the `MultiDegreeSumcheckProof` (per-group round messages and claimed sums),
- `cpr_up_evals` / `cpr_down_evals`: column evaluations at the shared evaluation point,
- `lookup_group_meta`: per-group metadata (table type, column count, witness length),
- `lookup_group_proofs`: per-group `BatchedDecompLogupProof` (chunk vectors, inverses, multiplicities).

#### Benefits

- **Shared randomness:** All sumcheck groups (CPR + lookups) receive the same verifier challenges, tightening the Fiat-Shamir binding.
- **Single pass:** Instead of `1 + G` sequential sumchecks (1 CPR + $G$ lookup groups), there is a single `shared_num_vars`-round sumcheck with multiple degree groups. The prover sends $\sum_g (\text{degree}_g + 1)$ evaluations per round (vs. `max_degree + 1` per round per sequential sumcheck).
- **No padding waste:** Low-degree groups (lookup, degree 2) send only 2 evaluations per round, not padded to the CPR's degree. The multi-degree sumcheck avoids the wasted bandwidth of padding all groups to the maximum degree.

#### No-Lookup Fallback

When no lookup specifications are provided, `prove_classic_logup()` still uses the multi-degree sumcheck with a single CPR group (at `num_vars`, no padding needed). This path is equivalent to the original sequential CPR sumcheck.

### Step 4: PCS Prove (test + evaluate)

```
test_transcript = BatchedZipPlus::test(params, trace, &hint)
(evals_f, proof)  = BatchedZipPlus::evaluate(params, trace, &point, test_transcript)
```

The batched PCS prove phase consists of two sequential calls: **proximity testing** (`test()`) followed by **evaluation opening** (`evaluate()`). The test transcript (containing the Fiat-Shamir sponge state and serialized test-phase data) flows directly into the evaluation call, binding the evaluation-phase randomness to the test-phase data.

Implementation: `BatchedZipPlus::test()` in `zip-plus/src/batched_pcs/phase_test.rs`, `BatchedZipPlus::evaluate()` in `zip-plus/src/batched_pcs/phase_evaluate.rs`. (The non-batched `ZipPlus::prove()` in `zip-plus/src/pcs/phase_prove.rs` merges both phases into a single call, but the pipeline exclusively uses the batched variant.)

#### Phase 1: Proximity Testing (`test()`)

##### Per-polynomial $\alpha$-projection challenges

For each polynomial $i$ in the batch, the prover squeezes `DEGREE_BOUND + 1` challenges $\alpha_{i,0}, \ldots, \alpha_{i,D}$ (of type `Zt::Chal`) from the Fiat-Shamir transcript. These challenges will be used to project each `BPoly<32>` evaluation cell $\hat{p}(X) = \sum_j c_j X^j$ into an integer via the inner product $\langle (c_0, \ldots, c_D), (\alpha_{i,0}, \ldots, \alpha_{i,D}) \rangle$. When `DEGREE_BOUND == 0` (scalar evaluations, e.g., `Int<N>`), this step is trivial: $\alpha = [1]$.

#### Phase 2: Shared row-combination coefficients

A single set of `num_rows` challenges $\gamma_0, \ldots, \gamma_{R-1}$ is squeezed from the Fiat-Shamir transcript. These coefficients are **shared across all polynomials** in the batch — batching works because each polynomial has the same matrix dimensions. When `num_rows == 1`, this step is trivial: $\gamma = [1]$.

#### Phase 3: Compute the combined row $u'$

For each polynomial $i$:

1. View its `num_rows × row_len` evaluation matrix. For each column $j \in [0, \text{row\_len})$, iterate over the rows $k = 0, \ldots, R-1$: project the evaluation cell at $(k, j)$ into an integer via $\langle \text{eval}_{k,j}, \boldsymbol{\alpha}_i \rangle$, scale by the row coefficient $\gamma_k$, and accumulate. This produces a per-polynomial combined row $u_i$ of `row_len` integers in the `CombR` ring.
2. The `combine_rows!` macro performs this operation column-major (`.skip(column).step_by(row_len)`) with checked arithmetic when `CHECK_FOR_OVERFLOW = true`.

All per-polynomial combined rows are summed element-wise: $u' = \sum_{i=0}^{m-1} u_i$ (also with checked addition to detect `CombR` overflow).

The combined row $u'$ is written to the proof byte stream via `transcript.write_const_many(&combined_row)`. This data is **not** absorbed into the Fiat-Shamir sponge — it is only serialized for the verifier to read back.

#### Phase 4: Column openings (repeated `NUM_COLUMN_OPENINGS` times)

For each of `NUM_COLUMN_OPENINGS` iterations:

1. **Sample a column index:** `column = transcript.squeeze_challenge_idx(codeword_len)` — the FS transcript yields a `u32`, and the column index is `u32 % codeword_len`. Note this indexes into the *codeword* (encoded) domain, not the *message* (row) domain.

2. **Open the Merkle tree at that column** via `open_merkle_trees_for_column()`:
   - For each codeword matrix in the batch, write the full column at `column_idx`: all `num_rows` values of type `Zt::Cw`. This gives the verifier the raw codeword entries it needs to recompute the proximity check.
   - Write a **single** Merkle authentication path from the shared tree: `commit_hint.merkle_tree.prove(column_idx)` builds a sibling path by walking up the tree layers. The proof is serialized as `(leaf_index: u64, leaf_count: u64, path_length: u64, siblings: [Blake3Hash; path_length])`, where each hash is 32 bytes.

The number of column openings is a compile-time constant: `NUM_COLUMN_OPENINGS = 147` in the standard configuration (set in the `ZipTypes` trait). This is the primary soundness amplification parameter: each column opening is an independent spot-check, and the linear code's distance guarantees that a codeword far from valid will disagree at a constant fraction of positions. With 147 random spot-checks at code rate $\rho = 1/4$, the false-accept probability is $\leq (1 - \delta)^{147}$ where $\delta$ is the relative distance.

#### Proof byte layout

The transcript is pre-allocated to the exact size:

| Section | Content | Size |
|---------|---------|------|
| Combined row $u'$ | `row_len` elements of `Zt::CombR` | `row_len × CombR::NUM_BYTES` |
| Column opening $\times 147$ | `batch_size × num_rows` codeword elements + 1 Merkle proof | `147 × (batch_size × num_rows × Cw::NUM_BYTES + 3×8 + (tree_height-1)×32)` |

An assertion at the end of `test()` checks that the actual byte count matches the pre-calculated estimate — a compile-time sanity check against serialization bugs.

#### What the verifier does with this (preview)

During `verify_testing()`, the verifier:

1. Re-derives the same $\boldsymbol{\alpha}_i$ and $\gamma_k$ challenges from the FS transcript.
2. Reads the combined row $u'$ from the proof.
3. For each column opening, reads the column values and Merkle proof.
4. **Spot-check encodes** $u'$ using `encode_wide_at_positions()` — this encodes $u'$ with the linear code but only evaluates the encoding at the opened column positions, which is much cheaper than a full encode.
5. For each opened column $j$: alpha-projects and row-combines the column codeword values using the same $\boldsymbol{\alpha}_i$ and $\gamma_k$, and checks that the result equals the encoded value of $u'$ at position $j$.
6. Verifies the Merkle proof against the committed root (the leaf is the Blake3 hash of the concatenated column values across all polynomials and rows).

If the committed codeword matrices are $\delta$-far from valid codewords, a constant fraction of columns will fail the proximity check, and the probability that all 147 random spot-checks miss every disagreement is negligible.

**What this produces:** A `BatchedZipPlusTestTranscript` wrapping a `PcsTranscript` that contains both the serialized proof bytes and the Fiat-Shamir sponge state. This transcript is passed directly to the evaluation sub-phase, which continues squeezing challenges from the same sponge — binding the evaluation-phase randomness to the test-phase data.

#### Phase 2: Evaluation Opening (`evaluate()`)

```
(evals_f, proof) = BatchedZipPlus::evaluate(params, polys, point, test_transcript)
```

The evaluation sub-phase computes the claimed evaluation of each committed polynomial at a shared point, and produces a batched evaluation proof. The key idea is that instead of writing one full `row_len`-sized combined row per polynomial (which would cost $O(m \times \text{row\_len})$ proof data), a single batched row is written, reducing the proof data to $O(\text{row\_len} + m)$.

#### Tensor decomposition of the evaluation point

The evaluation point $\mathbf{p} \in F^{n}$ (where $n = \text{num\_vars}$) is split into two parts via `point_to_tensor(num_rows, point_f, field_cfg)`:

- **High variables** $\mathbf{p}_{\text{hi}} = (p_0, \ldots, p_{n - \log_2(\text{num\_rows}) - 1})$: these index *within* a row.
- **Low variables** $\mathbf{p}_{\text{lo}} = (p_{n - \log_2(\text{num\_rows})}, \ldots, p_{n-1})$: these index *across* rows.

Each part is expanded into an equality polynomial tensor:
- $q_0 = \text{eq}(\mathbf{p}_{\text{lo}}, \cdot)$: a vector of `num_rows` coefficients used to combine rows.
- $q_1 = \text{eq}(\mathbf{p}_{\text{hi}}, \cdot)$: a vector of `row_len` coefficients used to evaluate within a row.

The MLE evaluation satisfies $f(\mathbf{p}) = \langle q_0 \cdot M, q_1 \rangle$ where $M$ is the evaluation matrix (rows × `row_len`).

#### Per-polynomial projection and row combination

For each polynomial $i$ in the batch:

1. **Project to field:** Every evaluation cell $e \in \text{Eval}$ (e.g., `BinaryPoly<32>`) is projected to a field element $F$ using a precomputed projection closure. For `BinaryPoly<32>`, this computes $\sum_j c_j \alpha^j \in F$ where $\alpha$ is the PCS projecting element (sampled from the Fiat-Shamir transcript as a `Zt::Chal`, then converted to $F$). For `Int<N>` evaluations, this is a direct lift $\mathbb{Z} \to F$. The projection closure is created once via `Zt::Eval::prepare_projection(&projecting_element)` and reused for all cells.

2. **Row combination via $q_0$:** Let $e_i$ denote the flat vector of projected field elements for polynomial $i$ (length `num_rows × row_len`), laid out in row-major order: $e_i[k \cdot \text{row\_len} + j]$ is the projected value at row $k$, column $j$. These are combined across rows using the `combine_rows!` macro. For each column index $j \in [0, \text{row\_len})$, this computes:
$$r_i[j] = \sum_{k=0}^{\text{num\_rows}-1} q_0[k] \cdot e_i[k \cdot \text{row\_len} + j]$$
   producing a **combined row** $r_i$ of `row_len` field elements for polynomial $i$ (where $i \in [0, m)$ ranges over the $m$ polynomials in the batch). The macro iterates column-major (`.skip(column).step_by(row_len)`) and accumulates scaled values via `MulByScalar`. When `num_rows == 1`, this step is skipped and the projected evaluations are used directly as $r_i$.

3. **Scalar evaluation:** The polynomial's evaluation at the point is $v_i = \langle r_i, q_1 \rangle$, computed as a standard inner product via `MBSInnerProduct`.

#### Batching across polynomials

After computing all per-polynomial combined rows $r_0, \ldots, r_{m-1}$ and scalar evaluations $v_0, \ldots, v_{m-1}$:

4. **Sample batching challenge:** A challenge $\beta$ is sampled from the Fiat-Shamir transcript (as `Zt::Chal`, converted to $F$).

5. **Compute batched row:** A single row $\hat{r} = \sum_{i=0}^{m-1} \beta^i \cdot r_i$ is computed element-wise. Powers of $\beta$ are accumulated iteratively.

6. **Write to proof transcript:**
   - The batched row $\hat{r}$ (`row_len` field elements, e.g., 512 × 32 bytes = 16 KB for `MontyField<4>`).
   - The $m$ per-polynomial evaluation scalars $v_0, \ldots, v_{m-1}$ ($m$ field elements).

#### Proof size reduction

For $m$ polynomials:
- **Without batching:** $m$ combined rows → $m \times \text{row\_len}$ field elements.
- **With batching:** 1 batched row + $m$ scalars → $\text{row\_len} + m$ field elements.

For the 17-column SHA-256 batch: $17 \times 512 = 8{,}704$ elements → $512 + 17 = 529$ elements (**16× reduction**). At 32 bytes per `MontyField<4>` element: ~272 KB → ~17 KB.

#### Consistency relation (verified by the verifier)

The verifier checks:
$$\langle \hat{r}, q_1 \rangle = \sum_{i=0}^{m-1} \beta^i \cdot v_i$$
This holds because $\langle \hat{r}, q_1 \rangle = \sum_i \beta^i \langle r_i, q_1 \rangle = \sum_i \beta^i v_i$, by linearity of inner products.

**Important note on the evaluation point:** The PCS evaluation point is derived by hashing the CPR's evaluation point into `i128` values via `derive_pcs_point()`. Both prover and verifier call this function to agree on the same deterministic point. This is NOT the same point as the CPR evaluation point (which lives in $\mathbb{F}_q$). See §15 for why this matters.

**Important note on field context:** The PCS samples its own field configuration from its internal Fiat-Shamir transcript via `get_random_field_cfg()`. The evaluation point is passed as raw integers (`&[Zt::Pt]`) and converted to field elements using this PCS-local field config. This field config is independent of the PIOP's field config.

**What this produces:** A `BatchedZipPlusProof` (serialized into `ZincProof.pcs_proof_bytes`) containing the test-phase data, the batched row, and the per-polynomial scalars.

---

## 4. The Verifier Pipeline in Detail

The verifier is implemented as `verify()` / `verify_generic()` / `verify_dual_ring()` / `verify_dual_circuit()` in `pipeline.rs`. It reconstructs the Fiat-Shamir transcript and checks each component:

### Step 1: Ideal Check Verification

1. Reconstruct the same random challenges ($\\alpha$, evaluation point) from a fresh transcript.
2. Deserialize the IC proof (the combined MLE evaluations).
3. Absorb the proof values into the transcript (maintaining Fiat-Shamir consistency).
4. **Ideal membership check:** For each constraint, check that its evaluated combined polynomial belongs to the correct ideal:
   - For `CyclotomicIdeal` $(X^{32} - 1)$: check that the polynomial reduces to zero modulo $X^{32} - 1$.
   - For `DegreeOneIdeal(2)` $(X - 2)$: evaluate the polynomial at $X = 2$ in $\\mathbb{F}_p$ and check the result is zero.
   - For `TrivialIdeal` ({0,1}^{<32}[X] constraints): always passes. See §11 for why.
   - For `BitPolyIdeal`: always passes. See §11 for why.
5. **Output:** A `VerifierSubClaim` containing the evaluation point and projected scalars, which is passed to the CPR verifier.

### Step 2: CPR Verification (Sumcheck)

1. Deserialize the CPR proof (sumcheck messages, claimed sum, up/down evaluations).
2. Verify the sumcheck protocol: replay each round, checking that the prover's univariate polynomial is consistent with the running claim.
3. At the final round, verify that the sumcheck's terminal claim is consistent with the IC's evaluation claims and the column evaluations (`up_evals`, `down_evals`).
4. **Output:** A `VerifierSubclaim` containing the evaluation point where the trace columns should be opened.

### Step 2.5: Lookup Verification (LogUp)

If the proof contains a lookup component (`LookupProofData`), the verifier dispatches to the appropriate variant:

1. **GKR LogUp** (`LookupProofData::Gkr`): Calls `verify_gkr_batched_lookup()`. For each lookup group:
   - Reconstructs the sub-table from `LookupGroupMeta` (table type, width) and the projecting element.
   - Replays the Fiat-Shamir transcript to derive challenges ($\\beta$ for shift, $\\alpha$ for batching).
   - Verifies the GKR fraction tree from root to leaves: at each layer, checks that parent fractions are consistent with child fractions via a sumcheck.
   - At the leaf level, verifies that the claimed evaluations are consistent with the multiplicity and table polynomials.
   - Checks the root cross-condition: $P_w \cdot Q_t = P_t \cdot Q_w$ (the log-derivative sums match).

2. **Classic LogUp** (`LookupProofData::Classic`): Calls `verify_batched_lookup()`. Similar structure but additionally checks the inverse vectors and chunk decomposition consistency that are explicitly provided in the proof.

3. **Batched Classic LogUp** (`LookupProofData::BatchedClassic`): In this path, Steps 2 and 2.5 are **merged** into a single multi-degree sumcheck verification. The verifier:
   1. Draws the projecting element and computes projected scalars (same as non-batched).
   2. Calls `CombinedPolyResolver::build_verifier_pre_sumcheck()` to draw $\\alpha$ and verify the CPR claimed sum against the IC subclaim.
   3. For each lookup group, calls `BatchedDecompLogupProtocol::build_verifier_pre_sumcheck()` to replay transcript operations (absorb chunks/inverses, draw $\\beta, \\gamma, \\mathbf{r}$) and verify table inverse correctness and multiplicity sums.
   4. Computes `shared_num_vars` = $\\max(\\text{num\\_vars}, \\max_g \\text{lookup\\_num\\_vars}_g)$ from the lookup pre-sumcheck data.
   5. Calls `MultiDegreeSumcheck::verify_as_subprotocol()` with `shared_num_vars`, verifying all groups in lockstep — each round checks per-group univariate polynomials against the running claims, and a single shared random challenge is drawn.
   6. **CPR finalize:** Calls `CombinedPolyResolver::finalize_verifier()` with the shared evaluation point and the CPR's expected evaluation. The finalize method pads the IC evaluation point and selector reference with zeros to match `shared_num_vars` dimensions (mirroring the prover's zero-padding of CPR MLEs), recomputes the combination function at the evaluation point, and verifies it matches the subclaim.
   7. **Lookup finalize:** For each lookup group, calls `BatchedDecompLogupProtocol::finalize_verifier()` with the shared evaluation point and the group's expected evaluation. Recomputes the batched identity polynomial $H$ at the evaluation point and verifies consistency.

### Step 3: PCS Verification (Batched)

The PCS verification consists of two sub-phases: **testing** (proximity) and **evaluation** (opening). Both have been heavily optimized (see §14 for details). All notation below references the symbols defined in §3 Steps 1 and 4.

#### Testing Phase (`verify_testing`)

The testing phase checks that the committed codeword matrices are close to valid codewords of the IPRS linear code.

1. **Sample challenges from transcript:**
   - Per-polynomial alpha challenges: for each polynomial $i \in [0, m)$, sample `DEGREE_BOUND + 1` challenges $\alpha_i^{(0)}, \ldots, \alpha_i^{(D)}$ from the Fiat-Shamir transcript (where $D = \text{DEGREE\_BOUND}$). These are used to project each `Cw`-element (a polynomial like `DensePolynomial<i64, 32>`) to a `CombR` integer.
   - Shared row-combination coefficients: sample `num_rows` challenges $c_0, \ldots, c_{\text{num\_rows}-1}$, shared across all polynomials. When `num_rows == 1`, this is just $[1]$.

2. **Read the combined row** from the proof transcript: a vector of `row_len` `CombR` elements. On the prover side (Step 4), this is computed as the sum over all polynomials of the alpha-projected, row-combined encoded rows.

3. **Read `NUM_COLUMN_OPENINGS`** (147) column openings from the proof transcript. For each opening $t \in [0, 147)$:
   - A column index $j_t \in [0, \text{cw\_len})$ squeezed from the Fiat-Shamir transcript.
   - For each polynomial $i \in [0, m)$: the column values $\text{cw}_i[0][j_t], \ldots, \text{cw}_i[\text{num\_rows}-1][j_t]$ — one `Cw`-element per row — totalling $m \times \text{num\_rows}$ `Cw`-elements per opening.
   - A Merkle proof (sibling path from leaf $j_t$ to the root).

4. **Spot-check encoding:** Collect the 147 opened positions $\{j_0, \ldots, j_{146}\}$ and call `encode_wide_at_positions(&combined_row, &positions)`. This computes the IPRS linear-code encoding of the combined row **only** at the 147 opened positions (not the full `cw_len`-element codeword), using the precomputed encoding matrix (see §14). Result: 147 `CombR` values.

5. **Per-column proximity check** (`verify_batched_column_testing`): For each opened column $j_t$:
   - For each polynomial $i$: alpha-project the `Cw`-column entries into `CombR` integers via inner product with $(\alpha_i^{(0)}, \ldots, \alpha_i^{(D)})$, then combine across rows via inner product with $(c_0, \ldots, c_{\text{num\_rows}-1})$. This produces one `CombR` value per polynomial.
   - Sum all $m$ per-polynomial `CombR` values into a total.
   - Check: `total == encode_wide_at_positions(combined_row)[t]`.

6. **Merkle proof verification:** For each opened column $j_t$, concatenate all per-polynomial column values (the same data written during the commit phase to form the leaf) and verify the Merkle proof against the commitment root.

#### Evaluation Phase (`verify_evaluation`)

The evaluation phase checks that the prover's claimed polynomial evaluations $v_0, \ldots, v_{m-1}$ and batched row $\hat{r}$ are consistent with the committed codeword and the evaluation point $\mathbf{p}$.

1. **Tensor decomposition:** Compute $(q_0, q_1) = \text{point\_to\_tensor}(\text{num\_rows}, \mathbf{p}_F, \text{field\_cfg})$, exactly as in §3 Step 4 Phase 2.

2. **Prepare field projection:** Create the projection closure via `Zt::Cw::prepare_projection(&projecting_element)`, where the projecting element $\alpha$ was sampled earlier from the Fiat-Shamir transcript (same $\alpha$ as used by the prover).

3. **Read proof data:**
   - The batched row $\hat{r}$ from the proof transcript: `row_len` field elements (e.g., 512 `MontyField<4>` values).
   - The $m$ per-polynomial evaluation scalars $v_0, \ldots, v_{m-1}$.

4. **Batched evaluation consistency:** Verify $\langle \hat{r}, q_1 \rangle = \sum_{i=0}^{m-1} \beta^i \cdot v_i$. Both sides are computed in the PCS's field. The left side is a standard inner product of the batched row with $q_1$. The right side accumulates $\beta$-weighted scalars.

5. **Spot-check encoding in field:** Call `encode_f_at_positions(&batched_row, &positions)` to compute the field-level IPRS encoding of $\hat{r}$ at the same 147 opened positions. This uses the same precomputed encoding matrix as the testing phase, but operates in the Montgomery field rather than on `CombR` integers.

6. **Per-column evaluation proximity:** For each opened column $j_t$:
   - For each polynomial $i$: project its `Cw`-column entries to field elements via the projection closure, then combine across rows via inner product with $q_0$. This produces one field element per polynomial.
   - Compute the $\beta$-weighted sum: $\hat{v}_{j_t} = \sum_{i=0}^{m-1} \beta^i \cdot (\text{per-poly value for } i)$.
   - Check: $\hat{v}_{j_t} = \text{encode\_f\_at\_positions}(\hat{r})[t]$.

   This verifies that the prover's batched row $\hat{r}$ is consistent with the committed column data at the opened positions, which — together with the testing phase — ensures that $\hat{r}$ is close to the true $\beta$-batched encoding of the committed polynomials.

#### Final Verdict

The verifier returns `accepted: pcs_result.is_ok()`. Both the IC and CPR verifications must succeed (early return on failure), and the PCS must accept.

---

## 5. SHA-256 Arithmetization

### Trace Layout

The SHA-256 witness is a 64-row × 17-column matrix, split into 14 binary polynomial columns and 3 integer columns. Each binary polynomial cell is a `BinaryPoly<32>` — a polynomial in $\\{0,1\\}^{<32}[X] \\subset \\mathbb{Z}[X]$, representing a 32-bit word as $w = \\sum_{i=0}^{31} w_i X^i$ where $w_i \\in \\{0, 1\\}$.

> **Notation.** $\\{0,1\\}^{<32}[X]$ denotes the set of polynomials over $\\mathbb{Z}$ with binary coefficients ($0$ or $1$) of degree $< 32$. This is a subset of $\\mathbb{Z}[X]$, **not** $\\mathbb{F}_2[X]$. The distinction is critical: in $\\mathbb{Z}[X]$ we have $1 + 1 = 2$, whereas in $\\mathbb{F}_2[X]$ we would have $1 + 1 = 0$. The SHA-256 constraints (even the rotation/XOR constraints C1–C6) are formulated over $\\mathbb{Z}[X]$ throughout this system; the projection to $\\mathbb{F}_p[X]$ preserves integer arithmetic.

| Col | Type | Name | Description |
|-----|------|------|-------------|
| 0 | BinaryPoly | `a_hat` | Working variable $a$ |
| 1 | BinaryPoly | `e_hat` | Working variable $e$ |
| 2 | BinaryPoly | `W_hat` | Message schedule word $W_t$ |
| 3 | BinaryPoly | `Sigma0_hat` | $\\Sigma_0(a)$ = ROTR²(a) ⊕ ROTR¹³(a) ⊕ ROTR²²(a) |
| 4 | BinaryPoly | `Sigma1_hat` | $\\Sigma_1(e)$ = ROTR⁶(e) ⊕ ROTR¹¹(e) ⊕ ROTR²⁵(e) |
| 5 | BinaryPoly | `Maj_hat` | Maj(a,b,c) |
| 6 | BinaryPoly | `ch_ef_hat` | $e \\wedge f$ |
| 7 | BinaryPoly | `ch_neg_eg_hat` | $\\neg e \\wedge g$ |
| 8 | BinaryPoly | `sigma0_w_hat` | $\\sigma_0(W_{t-15})$ |
| 9 | BinaryPoly | `sigma1_w_hat` | $\\sigma_1(W_{t-2})$ |
| 10 | BinaryPoly | `S0` | Shift quotient for $\\sigma_0$ |
| 11 | BinaryPoly | `S1` | Shift quotient for $\\sigma_1$ |
| 12 | BinaryPoly | `R0` | Shift remainder for $\\sigma_0$ |
| 13 | BinaryPoly | `R1` | Shift remainder for $\\sigma_1$ |
| 14 | Int | `mu_a` | Carry for $a$ update ($\\in \\{0..6\\}$) |
| 15 | Int | `mu_e` | Carry for $e$ update ($\\in \\{0..5\\}$) |
| 16 | Int | `mu_W` | Carry for $W$ schedule update ($\\in \\{0..3\\}$) |

**Columns removed from the previous 20-column layout:** `d_hat` and `h_hat` (working variables $d, h$) are now inlined via shift-register identities ($d_t = a_{t-3}$, $h_t = e_{t-3}$). Round constant `K_t` is now treated as a public input rather than a trace column.

### Lookup Specifications

10 of the 14 binary polynomial columns (indices 0–9) are designated as lookup columns with `LookupTableType::BitPoly { width: 32 }`. The lookup argument (§12) proves that every cell in these columns is a valid binary polynomial — i.e., all coefficients are in $\\{0, 1\\}$. The 4 shift/remainder-only columns (`S0`, `S1`, `R0`, `R1`) and 3 integer columns do not receive lookup constraints.

### Implemented Constraints (12 of 15)

#### {0,1}^{<32}[X] constraints (6) — enforced via `Uair<BinaryPoly<32>>` (`Sha256UairBp`)

These operate over $\\{0,1\\}^{<32}[X] \\subset \\mathbb{Z}[X]$, with ideal membership checked modulo $(X^{32} - 1)$. Although the XOR semantics of SHA-256 bitwise operations correspond to addition in $\\mathbb{F}_2[X]$, the constraints here live in $\\mathbb{Z}[X]$ because the system projects to $\\mathbb{F}_p[X]$ (where $1 + 1 = 2$, not $0$). The cyclotomic ideal $(X^{32} - 1)$ absorbs the difference: rotation constraints are correct in $\\mathbb{Z}[X]/(X^{32}-1)$ when the inputs have binary coefficients.

| # | Name | Expression | Ideal |
|---|------|-----------|-------|
| C1 | Σ₀ rotation | $\\hat{a} \\cdot \\rho_0 - \\hat{\\Sigma}_0 \\in (X^{32} - 1)$ | Cyclotomic |
| C2 | Σ₁ rotation | $\\hat{e} \\cdot \\rho_1 - \\hat{\\Sigma}_1 \\in (X^{32} - 1)$ | Cyclotomic |
| C3 | σ₀ rotation + shift | $\\hat{W} \\cdot \\rho_{\\sigma_0} + S_0 - \\widehat{\\sigma_0 w} \\in (X^{32} - 1)$ | Cyclotomic |
| C4 | σ₁ rotation + shift | $\\hat{W} \\cdot \\rho_{\\sigma_1} + S_1 - \\widehat{\\sigma_1 w} \\in (X^{32} - 1)$ | Cyclotomic |
| C5 | σ₀ shift decomposition | $\\hat{W} - R_0 - S_0 \\cdot X^3 = 0$ | Zero (exact) |
| C6 | σ₁ shift decomposition | $\\hat{W} - R_1 - S_1 \\cdot X^{10} = 0$ | Zero (exact) |

The rotation polynomials encode ROTR via multiplication by powers of $X$ modulo $(X^{32} - 1)$:
- $\\rho_0 = X^{30} + X^{19} + X^{10}$ encodes ROTR(2, 13, 22) for $\\Sigma_0$
- $\\rho_1 = X^{26} + X^{21} + X^{7}$ encodes ROTR(6, 11, 25) for $\\Sigma_1$
- $\\rho_{\\sigma_0} = X^{25} + X^{14}$ encodes ROTR(7, 18) for $\\sigma_0$
- $\\rho_{\\sigma_1} = X^{15} + X^{13}$ encodes ROTR(17, 19) for $\\sigma_1$

#### Q[X] constraints (3) — enforced via `Uair<DensePolynomial<i64, 64>>` (`Sha256UairQx`)

These operate in $\\mathbb{Z}[X]$ (integer coefficient polynomials, degree < 64). They require viewing the same trace as integer polynomials rather than binary polynomials, because the constant 2 is zero in $\\mathbb{F}_2$.

BitPoly membership (binary coefficient checks for Ch, ¬e∧g, Maj) is now enforced by lookups rather than ideal checks.

| # | Name | Expression | Ideal |
|---|------|-----------|-------|
| C7 | a-update carry | $\\hat{a}[t{+}1] - \\hat{h} - \\hat{\\Sigma}_1 - \\widehat{ch_{ef}} - \\widehat{ch_{neg}} - \\hat{K}_t - \\hat{W} - \\hat{\\Sigma}_0 - \\hat{Maj} + \\mu_a \\cdot X^{32} \\in (X - 2)$ | DegreeOne(2) |
| C8 | e-update carry | $\\hat{e}[t{+}1] - \\hat{d} - \\hat{h} - \\hat{\\Sigma}_1 - \\widehat{ch_{ef}} - \\widehat{ch_{neg}} - \\hat{K}_t - \\hat{W} + \\mu_e \\cdot X^{32} \\in (X - 2)$ | DegreeOne(2) |
| C9 | W-schedule carry | $\\hat{W}[t{+}1] - \\widehat{\\sigma_1 w} - \\hat{W}[t{-}6] - \\widehat{\\sigma_0 w} - \\hat{W}[t{-}15] + \\mu_W \\cdot X^{32} \\in (X - 2)$ | DegreeOne(2) |

The carry constraints use the $(X - 2)$ ideal: a polynomial $p(X) \\in (X - 2)$ iff $p(2) = 0$. Since evaluating a binary-coefficient polynomial at $X = 2$ gives the integer value of the 32-bit word ($\\sum c_i \\cdot 2^i$), this ideal check verifies that the modular addition with carry is correct as an integer equation.

### Not-implemented SHA-256 constraints (3 of 12)

| # | Name | Why deferred |
|---|------|-------------|
| C10 | d-delay ($d \\leftarrow a_{t-3}$) | Requires 3-row lookback; the UAIR only provides `up` (current) and `down` (next row) |
| C11 | h-delay ($h \\leftarrow e_{t-3}$) | Same: 3-row lookback needed |
| C12 | Full message schedule W update | Requires lookbacks of 2, 7, 15, and 16 rows (C9 is a partial approximation using up/down) |

These constraints are algebraically specified in comments but cannot be fully expressed in the current `up`/`down` framework. Implementing them would require either intermediate relay columns ($b, c, f, g$ as separate columns with 1-step delay constraints) or extending the UAIR trait to support multi-row access.

---

## 6. ECDSA Arithmetization

### Trace Layout

The ECDSA witness is a **258-row × 9-column** matrix. The trace is unified under `Int<4>` (256-bit integer) for all contexts — PCS commitment, PIOP constraint checking, and the end-to-end pipeline:

| Context | Evaluation type | Purpose |
|---------|----------------|---------|
| **Unified pipeline** (PCS + IC + CPR) | `Int<4>` (256-bit integer) | All 7 constraints are expressed directly in `Int<4>` via `Uair<Int<4>>`. The same ring is used for PCS commitment, IdealCheck, and CombinedPolyResolver — no ring conversion needed. This eliminates the prior dual-ring architecture. |
| **Legacy constraint checking** | `DensePolynomial<i64, 1>` | Degree-0 polynomials (plain i64 scalars). Retained for the original constraint formulation; the `Int<4>` version is algebraically identical. |

The scalars $u_1 = e \cdot s^{-1} \bmod n$ and $u_2 = r \cdot s^{-1} \bmod n$ are given to the verifier 
in the clear (they are public inputs derived from the message hash and signature). The verifier can independently compute the corresponding bit decomposition columns $b_1, b_2$ and verify that the scalar accumulation is correct without dedicated trace columns or constraints. The quotient bit $k$ (for the final signature check $R_x \equiv r \pmod{n}$) is handled in boundary constraints, not as a trace column.

| Col | Name | Description |
|-----|------|-------------|
| 0 | b₁ | Bit of scalar $u_1$ |
| 1 | b₂ | Bit of scalar $u_2$ |
| 2–4 | X, Y, Z | Accumulator point (Jacobian coordinates) |
| 5–7 | X_mid, Y_mid, Z_mid | Doubled point |
| 8 | H | Addition scratch: chord x-difference $T_x Z_{\text{mid}}^2 - X_{\text{mid}}$ |

The auxiliary values $S = Y^2$ (doubling scratch) and $R_a = T_y Z_{\text{mid}}^3 - Y_{\text{mid}}$ (addition y-difference) are not separate trace columns; they are inlined as sub-expressions in the constraints. This raises the maximum constraint degree (from ~6 to ~10) but saves 2 columns of proof data. $H$ is kept as a column because it appears cubed ($H^3$) in two constraints — inlining would push the degree to ~12+.

### Implemented Constraints (7 of 7)

All 7 ECDSA non-boundary constraints use `assert_zero` (exact integer equality). They use Shamir's trick for simultaneous $u_1 \cdot G + u_2 \cdot Q$ computation on secp256k1 ($a = 0$, so the doubling formula simplifies). The scalars $u_1, u_2$ are public inputs (not in the trace). The former auxiliary definitions $S = Y^2$ and $R_a = T_y Z_{\text{mid}}^3 - Y_{\text{mid}}$ are inlined.

| # | Constraint | Max degree |
|---|-----------|------------|
| C1 | $Z_{\text{mid}} = 2YZ$ | 2 |
| C2 | $X_{\text{mid}} = 9X^4 - 8XY^2$ | 4 |
| C3 | $Y_{\text{mid}} = 12X^3 Y^2 - 3X^2 X_{\text{mid}} - 8Y^4$ | 5 |
| C4 | $H = T_x Z_{\text{mid}}^2 - X_{\text{mid}}$ | ~4 |
| C5 | $Z[t{+}1] = (1{-}s) Z_{\text{mid}} + s \cdot Z_{\text{mid}} H$ | ~4 |
| C6 | $X[t{+}1] = (1{-}s) X_{\text{mid}} + s\bigl((T_y Z_{\text{mid}}^3 - Y_{\text{mid}})^2 - H^3 - 2 X_{\text{mid}} H^2\bigr)$ | 12 |
| C7 | $Y[t{+}1] = (1{-}s) Y_{\text{mid}} + s\bigl((T_y Z_{\text{mid}}^3 - Y_{\text{mid}})(X_{\text{mid}} H^2 - X[t{+}1]) - Y_{\text{mid}} H^3\bigr)$ | 10 |

Where $s = b_1 + b_2 - b_1 b_2$ (Shamir selector: $s = 1$ iff any bit is set), and $T = (T_x, T_y)$ is the table point selected by $(b_1, b_2)$ from $\{\mathcal{O}, G, Q, G{+}Q\}$.

### Limitations

- The constraints operate over `DensePolynomial<i64, 1>` (machine integers) or `Int<4>` (256-bit integers), not over the secp256k1 base field $\mathbb{F}_p$ (256-bit). The current witness uses a toy curve over $\mathbb{F}_{101}$ with the same equation $y^2 = x^3 + 7$.
- For real secp256k1, a 256-bit field type would be needed, which is not yet integrated into the UAIR framework.
- The BinaryPoly<32> UAIR implementation has 0 constraints — it's a placeholder because the ECDSA constraints are integer-valued, not binary-polynomial.
- For PCS benchmarking, the ECDSA trace uses `Int<4>` (256-bit integer) evaluations instead of `BinaryPoly<32>`, since ECDSA values are scalars, not polynomials. This is ~32× cheaper per cell in the IPRS NTT (1 coefficient vs. 32), and is the configuration used in the paper's target benchmark.

---

## 7. The Dual-Ring Pipeline

SHA-256 constraints live in two different rings: {0,1}^{<32}[X] (for bitwise/rotation operations) and Q[X] (for integer carry propagation and AND/Maj checks). The **dual-ring pipeline** (`prove_dual_ring` / `verify_dual_ring`) handles this by running two sequential PIOP passes on the same committed trace. Both passes operate on **all 14 BinaryPoly columns** — the first interprets them as {0,1}^{<32}[X], the second converts them to Q[X] (DensePolynomial<i64, 64>). The 3 integer carry columns participate only in the Q[X] pass:

### Prover: `prove_dual_ring()`

1. **PCS Commit** the `BinaryPoly<32>` trace (once).
2. **Shared field setup:** Sample a PIOP field $\mathbb{F}_q$ and a shared IC evaluation point from the transcript.
3. **IC₁** over `BinaryPoly<32>` ring (constraints C1–C6) — evaluated at the shared point.
4. **IC₂** over `DensePolynomial<i64, 64>` ring (constraints C7–C12): convert trace via `convert_trace_to_qx()`, then run IC at the same shared point.
5. **Shared projecting element:** A single $\alpha$ challenge is drawn from the transcript and used for both passes.
6. **Batched CPR via multi-degree sumcheck:** Both CPR passes are batched into a single `MultiDegreeSumcheck::prove_as_subprotocol` call:
   - **Group 0 (BinaryPoly):** `CombinedPolyResolver::build_prover_group::<BinaryPoly<32>>` constructs the CPR combination function (degree = `bp_max_degree + 2`) without running the sumcheck.
   - **Group 1 (Q[X]):** `CombinedPolyResolver::build_prover_group::<DensePolynomial<i64, 64>>` does the same for Q[X] constraints (degree = `qx_max_degree + 2`).
   - The multi-degree sumcheck runs both groups in lockstep — shared random challenges, single evaluation point.
   - `CombinedPolyResolver::finalize_prover` extracts `up_evals`/`down_evals` for each group.
7. **PCS Prove** (merged test + evaluate).

Both IC and CPR passes share the same Fiat-Shamir transcript, so the random challenges are sequentially derived and verifiably deterministic. The multi-degree sumcheck batching ensures both CPR groups share the same evaluation point, which is then used for PCS opening.

### Verifier: `verify_dual_ring()`

1. **Shared field setup:** Reconstruct $\mathbb{F}_q$ and the IC evaluation point from a fresh transcript.
2. **IC₁ verify (BinaryPoly)** — uses `TrivialIdeal` (always passes; see §11). Evaluated at the shared point via `verify_at_point`.
3. **IC₂ verify (Q[X])** — uses real ideals: `DegreeOneIdeal(2)` for C7–C9. Evaluated at the same shared point.
4. **Shared projecting element** drawn from the transcript (same as prover's).
5. **CPR₁ pre-sumcheck:** Draws $\alpha_1$, verifies the BP claimed sum against IC₁'s subclaim.
6. **CPR₂ pre-sumcheck:** Draws $\alpha_2$, verifies the QX claimed sum against IC₂'s subclaim.
7. **Multi-degree sumcheck verify:** `MultiDegreeSumcheck::verify_as_subprotocol` replays both groups in lockstep, producing a shared evaluation point and per-group expected evaluations.
8. **CPR₁ finalize:** `CombinedPolyResolver::finalize_verifier` checks BP's expected evaluation against the recomputed combination at the shared point.
9. **CPR₂ finalize:** Same for Q[X].
10. **PCS verify.**

---

## 7b. The Dual-Circuit Pipeline

Where the dual-ring pipeline (§7) runs two PIOP passes on the **same** committed trace (same PCS, different rings), the **dual-circuit pipeline** combines two **independent circuits** — each with its own trace and PCS — into a single proving pipeline that shares all PIOP challenges and runs a unified multi-degree sumcheck.

The primary use case is 8×SHA-256 (`BinaryPoly<32>`, 14 columns) + ECDSA (`Int<4>`, 9 columns). Each circuit has different column counts, evaluation types, ZipTypes, and PCS field sizes, but they share a single Fiat-Shamir transcript so that all PIOP randomness is jointly derived.

### Prover: `prove_dual_circuit()`

1. **PCS Commit (both circuits):** Commit circuit 1 (BinaryPoly) and circuit 2 (generic ring) traces independently. Each uses its own `ZipTypes` / `LinearCode` / PCS field.
2. **Shared transcript + field config:** Initialize a single `KeccakTranscript`. Sample the PIOP field $\mathbb{F}_q$.
3. **Shared IC evaluation point:** Sample $\mathbf{r} \in \mathbb{F}_q^{\text{num\_vars}}$ from the transcript (once, used by both circuits).
4. **IC₁ (BinaryPoly)** at the shared point: lift trace to $\mathbb{F}_q[X]$, run IC prover using `prove_at_point`.
5. **IC₂ (generic ring)** at the same shared point: lift trace, run IC prover using `prove_at_point`.
6. **Shared projecting element:** A single $\alpha$ is drawn from the transcript for $\mathbb{F}_q[X] \to \mathbb{F}_q$ projection.
7. **Project both traces** to $\mathbb{F}_q$ scalars using the shared $\alpha$.
8. **Extract lookup columns** from circuit 1 (if lookup specs are provided).
9. **Build CPR sumcheck groups:**
   - **Group 0:** `CombinedPolyResolver::build_prover_group::<BinaryPoly<D>>` for circuit 1 (degree = `c1_max_degree + 2`).
   - **Group 1:** `CombinedPolyResolver::build_prover_group::<R2>` for circuit 2 (degree = `c2_max_degree + 2`).
   - **Groups 2..N+2:** One lookup group per lookup spec (degree 2 each).
10. **Single multi-degree sumcheck** over all groups in lockstep — shared random challenges, single evaluation point.
11. **Finalize** CPR and lookup proofs: extract `up_evals` / `down_evals` for both CPR groups, finalize lookup groups.
12. **PCS Test + Evaluate** for both circuits independently (each at the shared evaluation point).

### Verifier: `verify_dual_circuit()`

1. **Shared field setup:** Reconstruct $\mathbb{F}_q$ and the IC evaluation point from a fresh transcript.
2. **IC₁ verify (BinaryPoly)** at the shared point using `verify_at_point` (uses `TrivialIdeal`).
3. **IC₂ verify (generic ring)** at the shared point using `verify_at_point` (uses `EcdsaIdealOverF`).
4. **Shared projecting element** drawn from the transcript (same as prover's).
5. **CPR₁ pre-sumcheck:** Draws $\alpha_1$, verifies the circuit-1 claimed sum against IC₁'s subclaim.
6. **CPR₂ pre-sumcheck:** Draws $\alpha_2$, verifies the circuit-2 claimed sum against IC₂'s subclaim.
7. **Lookup pre-sumcheck:** For each lookup group, verify claimed sums and build verifier state.
8. **Multi-degree sumcheck verify:** `MultiDegreeSumcheck::verify_as_subprotocol` replays all groups (CPR₁ + CPR₂ + lookup groups) in lockstep.
9. **CPR₁ finalize + CPR₂ finalize + Lookup finalize:** Verify each group's expected evaluation at the shared point.
10. **PCS verify** for both circuits independently.

### Key Differences from Dual-Ring (§7)

| | Dual-Ring (§7) | Dual-Circuit (§7b) |
|---|---|---|
| **Traces** | Same trace, two ring interpretations | Two independent traces |
| **PCS** | One PCS commitment | Two PCS commitments (separate types/fields) |
| **IC passes** | Same columns, different rings | Different columns, different rings |
| **Sumcheck groups** | 2 (CPR₁ + CPR₂) | 2 + N (CPR₁ + CPR₂ + N lookup groups) |
| **Use case** | SHA-256 BinaryPoly + Q\[X\] | SHA-256 + ECDSA (cross-circuit) |

---

## 8. The Split-Trace Architecture

### Motivation

The 17 SHA-256 trace columns use two different evaluation types: {0,1}^{<32}[X] rotation/shift constraints (C1–C6) and Q[X] carry constraints (C7–C9) both need `BinaryPoly<32>` evaluations for the first 14 columns, while the 3 carry columns only need integer evaluations. By splitting the trace into two PCS batches, the 3 integer-only columns can be committed as `Int<1>` (64-bit) instead of `BinaryPoly<32>` (32 coefficients × 8 bytes = 256 bytes per codeword element). This shrinks the column-opening data for these columns from 256 bytes to 16 bytes per element — a **16× reduction per column**.

### Column Classification

The split is defined in `sha256-uair/src/witness.rs`:

**BinaryPoly batch (14 columns)** — `POLY_COLUMN_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]`:
- These participate in {0,1}^{<32}[X] rotation/shift constraints (C1–C6) and/or Q[X] constraints (C7–C9).
- Committed as `BinaryPoly<32>` using `Sha256ZipTypes<i64, 32>`.
- Codeword element: `DensePolynomial<i64, 32>` = 256 bytes.
- CombR: `Int<6>` = 48 bytes (384-bit).
- Columns: a\_hat, e\_hat, W\_hat, Σ0\_hat, Σ1\_hat, Maj\_hat, ch\_ef\_hat, ch\_neg\_eg\_hat, σ0\_w\_hat, σ1\_w\_hat, S0, S1, R0, R1.

**Int batch (3 columns)** — `INT_COLUMN_INDICES = [14, 15, 16]`:
- These only participate as carry values in Q[X] constraints (C10–C12).
- Committed as `Int<1>` (64-bit integer) using `Sha256IntZipTypes`.
- Codeword element: `Int<2>` = 16 bytes (128-bit).
- CombR: `Int<4>` = 32 bytes (256-bit).
- Columns: μ\_a, μ\_e, μ\_W.

### Witness Generators

Two functions in `sha256-uair/src/witness.rs`:

- **`generate_poly_witness(num_vars, rng)`** → `Vec<DenseMultilinearExtension<BinaryPoly<32>>>`: Generates the full 17-column trace, then extracts only the 14 `POLY_COLUMN_INDICES` columns.
- **`generate_int_witness(num_vars, rng)`** → `Vec<DenseMultilinearExtension<Int<1>>>`: Generates the full trace, then for each of the 3 `INT_COLUMN_INDICES`, converts each `BinaryPoly<32>` cell to `Int<1>` via `bp.to_u64() as i64`. All values are small integers that fit in a single u64 limb.

### Proof Size Impact

For 1×SHA-256 (1 compression, 128-row trace with `num_vars = 7`):

| Configuration | PCS Raw | Compressed |
|--------------|---------|------------|
| Monolithic (17 cols as BinaryPoly) | ~700 KB | ~340 KB |
| Split (14 BPoly + 3 Int) | ~490 KB | ~240 KB |
| **Reduction** | **~1.4×** | **~1.4×** |

For 8×SHA-256 + ECDSA (512-row trace with `num_vars = 9`):

| Configuration | PCS Raw | Compressed |
|--------------|---------|------------|
| Monolithic SHA + ECDSA | ~950 KB | ~520 KB |
| Split SHA + ECDSA | ~780 KB | ~430 KB |
| **Reduction** | **~1.2×** | **~1.2×** |

### ZipTypes Instantiation

| Type alias | Eval | Cw | CombR | PcsF |
|---|---|---|---|---|
| `Sha256ZipTypes<i64, 32>` | `BinaryPoly<32>` | `DensePolynomial<i64, 32>` (256 B) | `Int<6>` (48 B) | `MontyField<4>` (128-bit) |
| `Sha256IntZipTypes` | `Int<1>` (64-bit) | `Int<2>` (128-bit) | `Int<4>` (256-bit) | `MontyField<4>` (128-bit) |
| `EcdsaScalarZipTypes` | `Int<4>` (256-bit) | `Int<5>` (320-bit) | `Int<8>` (512-bit) | `MontyField<8>` (512-bit) |

---

## 9. Proof Format and Serialization

### `ZincProof` (single-ring)

| Field | Content | Size |
|-------|---------|------|
| `pcs_proof_bytes` | Serialized PCS transcript (Merkle proofs, queried rows, batched eval row, per-poly scalars) | ~350 KB for 17 cols, DEPTH=1 |
| `commitment` | Merkle root + batch size | Small |
| `ic_proof_values` | Combined MLE evaluations (one `DynamicPolynomialF` per constraint) | Tens of bytes per constraint |
| `cpr_sumcheck_messages` | Sumcheck round polynomials | $O(\text{num\_vars} \times \text{degree})$ field elements |
| `cpr_sumcheck_claimed_sum` | Claimed sum of the batched constraint expression | 1 field element |
| `cpr_up_evals` | Column evaluations at sumcheck point | 1 field element per column |
| `cpr_down_evals` | Column evaluations at shifted point | 1 field element per column |
| `evaluation_point_bytes` | The sumcheck evaluation point | `num_vars` field elements |
| `pcs_evals_bytes` | PCS evaluation values | 1 field element per column |
| `lookup_proof` | Optional lookup proof: `LookupProofData::Gkr`, `Classic`, or `BatchedClassic` | Variable (see below) |

### `LookupProofData` Variants

| Variant | Content | Key data |
|---------|---------|----------|
| `Gkr(GkrPipelineLookupProof)` | Per-group `GkrBatchedDecompLogupProof` + `LookupGroupMeta` | Aggregated multiplicities, GKR fraction proofs (root p/q, per-layer sumcheck + left/right numerator-denominators). **No chunk vectors or inverse vectors.** |
| `Classic(PipelineLookupProof)` | Per-group `BatchedDecompLogupProof` + `LookupGroupMeta` | Chunk vectors, aggregated multiplicities, inverse witness/table vectors, sumcheck proof. |
| `BatchedClassic(BatchedCprLookupProof)` | Multi-degree sumcheck proof + CPR evals + per-group lookup proofs + metadata | `md_proof` (per-group round messages and claimed sums), `cpr_up_evals`, `cpr_down_evals`, per-group `BatchedDecompLogupProof` (chunk/inverse vectors, multiplicities), `LookupGroupMeta`. The CPR sumcheck data is embedded in the multi-degree proof (group 0); lookup groups occupy groups 1..G. When this variant is used, the top-level `cpr_sumcheck_messages` and `cpr_sumcheck_claimed_sum` fields are unused (the CPR data lives inside the `BatchedCprLookupProof`). |

### `DualRingZincProof`

Contains two IC proof passes and a **batched multi-degree sumcheck** for the two CPR passes:

| Field | Content | Notes |
|-------|---------|-------|
| `pcs_proof_bytes` | Serialized PCS transcript | Same as single-ring |
| `commitment` | Merkle root + batch size | |
| `bp_ic_proof_values` | IC₁ combined MLE evaluations (BinaryPoly) | |
| `qx_ic_proof_values` | IC₂ combined MLE evaluations (Q[X]) | |
| `md_group_messages` | Per-group sumcheck round messages | `md_group_messages[0]` = BP, `[1]` = QX |
| `md_claimed_sums` | Claimed sum per group | 2 serialized field elements |
| `md_degrees` | Polynomial degree per group | e.g. `[4, 4]` |
| `bp_cpr_up_evals` / `bp_cpr_down_evals` | BP column evaluations at shared sumcheck point | |
| `qx_cpr_up_evals` / `qx_cpr_down_evals` | QX column evaluations at shared sumcheck point | |
| `evaluation_point_bytes` | Shared sumcheck evaluation point | |
| `pcs_evals_bytes` | PCS evaluation values | |
| `timing` | Per-phase timing breakdown | |

The two CPR passes are batched into a single multi-degree sumcheck (group 0 = BinaryPoly, group 1 = Q[X]), sharing verifier challenges and producing a common evaluation point. This replaces the old sequential design where each CPR ran its own independent sumcheck.

### `DualCircuitZincProof`

Contains two independent PCS instances and a **batched multi-degree sumcheck** covering both CPR passes plus lookup groups:

| Field | Content | Notes |
|-------|---------|-------|
| `pcs1_proof_bytes` | Serialized PCS transcript (circuit 1 — BinaryPoly) | |
| `pcs1_commitment` | Merkle root + batch size (circuit 1) | |
| `pcs2_proof_bytes` | Serialized PCS transcript (circuit 2 — generic ring) | |
| `pcs2_commitment` | Merkle root + batch size (circuit 2) | |
| `ic1_proof_values` | IC₁ combined MLE evaluations (circuit 1) | |
| `ic2_proof_values` | IC₂ combined MLE evaluations (circuit 2) | |
| `md_group_messages` | Per-group sumcheck round messages | `[0]` = CPR₁, `[1]` = CPR₂, `[2..]` = lookup |
| `md_claimed_sums` | Claimed sum per group | 2 + N serialized field elements |
| `md_degrees` | Polynomial degree per group | e.g. `[4, 14, 2, 2, ...]` |
| `cpr1_up_evals` / `cpr1_down_evals` | Circuit 1 column evaluations at shared point | |
| `cpr2_up_evals` / `cpr2_down_evals` | Circuit 2 column evaluations at shared point | |
| `lookup_group_meta` | Per-lookup-group metadata (`LookupGroupMeta`) | Circuit 1 only |
| `lookup_group_proofs` | Per-lookup-group `BatchedDecompLogupProof` | Circuit 1 only |
| `evaluation_point_bytes` | Shared sumcheck evaluation point | Common across all groups |
| `pcs1_evals_bytes` | PCS evaluation values (circuit 1) | |
| `pcs2_evals_bytes` | PCS evaluation values (circuit 2) | |
| `timing` | Per-phase timing breakdown | |

The multi-degree sumcheck groups are ordered: group 0 = CPR for circuit 1 (BinaryPoly), group 1 = CPR for circuit 2 (generic ring), groups 2..N+2 = lookup groups for circuit 1. All groups share verifier challenges and produce a single evaluation point used by both PCS instances.

### PCS Proof Breakdown

The PCS proof transcript is the dominant contributor to proof size. Its structure:

**Test phase:**
- 1 combined row (`row_len × CombR::NUM_BYTES`): e.g., 512 × 48 = 24,576 bytes (BinaryPoly batch) or 512 × 64 = 32,768 bytes (ECDSA batch).
- `NUM_COLUMN_OPENINGS` (147) column openings, each containing:
  - Column index (8 bytes).
  - Per-polynomial column values: `batch_size × num_rows × Cw::NUM_BYTES` per opening. For SHA BinaryPoly: 14 × 1 × 256 = 3,584 bytes; for SHA Int: 3 × 1 × 16 = 48 bytes; for ECDSA: 9 × 1 × 40 = 360 bytes.
  - Merkle proof: ~11 blake3 hashes (~352 bytes per proof for tree depth ≈ log₂(2048) = 11).

**Eval phase (batched):**
- 1 batched row (`row_len × F::NUM_BYTES`): 512 × 32 = 16,384 bytes (MontyField<4>) or 512 × 64 = 32,768 bytes (MontyField<8>).
- `batch_size` per-polynomial evaluation scalars: `batch_size × F::NUM_BYTES`.

### Field Serialization

All `PiopField` elements are serialized in Montgomery representation (32 bytes each for the 128-bit field). This ensures Fiat-Shamir consistency: the verifier can reconstruct the exact same transcript bytes.

---

## 10. Fiat-Shamir Transcript

The system uses a **BLAKE3-based Fiat-Shamir transcript** (`Blake3Transcript`). A backward-compatible type alias `pub type KeccakTranscript = Blake3Transcript` is provided so existing callers continue to compile without changes. The prover and verifier construct identical transcripts by:

1. Initializing a fresh `Blake3Transcript::new()`.
2. Sampling a random prime field configuration: `transcript.get_random_field_cfg()`. This produces a ~128-bit prime $p$ used for all PIOP arithmetic.
3. Sequentially sampling challenges and absorbing proof data in the same order on both sides.

### BLAKE3 Transcript Details

- **Hashing:** BLAKE3 keyed hashing via `blake3::Hasher`. Absorb calls use `hasher.update(v)`.
- **Squeezing:** XOF (extendable output) mode via `clone().finalize_xof()`. The internal state is NOT mutated on squeeze — only absorbs update it. After squeezing, separator bytes `[0x12] || bytes || [0x34]` are absorbed back to bind the squeezed value into the transcript.
- **Optimized absorption:** `absorb_random_field_slice()` concatenates all field elements into a single contiguous buffer before calling `update()`, enabling BLAKE3's internal SIMD-friendly chunk processing.
- **Prime sampling:** `get_prime()` repeatedly squeezes candidates until a probable prime (Miller-Rabin) is found.

The field is a Montgomery-form modular integer with 4×64-bit limbs, using a fixed 128-bit prime `0x860995AE68FC80E1B1BD1E39D54B33`. The Miller-Rabin primality test confirms its primality.

---

## 11. Ideal Types and Membership Checks

The system defines six ideal types, each handling a different kind of constraint:

### `CyclotomicIdeal` — $(X^{32} - 1)$

- **Used by:** C1–C4 (SHA-256 rotation constraints)
- **Prover-side check (`BinaryPoly<32>`):** Only zero is in the ideal for degree ≤ 31 polynomials. But constraint expressions can have degree up to 61 (product of two degree-31 polys), so the check is that the expression reduces to zero modulo $X^{32} - 1$.
- **Verifier-side check (`DynamicPolynomialF`):** Reduces the polynomial modulo $X^{32} - 1$ by folding coefficients: $g_{\text{red},j} = \sum_k g_{j + 32k}$. Checks all 32 reduced coefficients equal zero.

### `DegreeOneIdeal(2)` — $(X - 2)$

- **Used by:** C7–C9 (SHA-256 carry propagation)
- **Prover-side check:** Evaluates $p(2) = \sum c_i \cdot 2^i$ and checks it equals zero.
- **Verifier-side check (`DynamicPolynomialF`):** Evaluates the polynomial at $X = 2$ in $\mathbb{F}_p$ and checks the result is zero. **This IS a real ideal that lifts correctly:** $(X - 2)$ is an ideal in both $\mathbb{Z}[X]$ and $\mathbb{F}_p[X]$ (for $p > 2$), and the evaluation-at-2 check is preserved under the projection from $\mathbb{Z}[X]$ to $\mathbb{F}_p[X]$.

### `TrivialIdeal`

- **Used by:** BinaryPoly IC pass in the verifier
- **Check:** Always returns `true`.
- **Rationale:** The {0,1}^{<32}[X] rotation constraints (C1–C6) are formulated over $\mathbb{Z}[X]/(X^{32}-1)$. The cyclotomic ideal $(X^{32}-1)$ is checked by reducing the polynomial modulo $X^{32}-1$ and verifying all coefficients are zero. However, after the IC protocol's random MLE evaluation and batching, the combined polynomial at the verifier is a random linear combination that no longer has binary coefficients — so the cyclotomic reduction would produce non-zero values even for a valid witness. The `TrivialIdeal` sidesteps this: it always passes the ideal membership check. Soundness is preserved because the sumcheck and PCS verify the algebraic consistency of the projected trace — if any witness row violates the constraint, the random-evaluation-based sumcheck catches it with high probability.

### `EcdsaIdealOverF`

- **Used by:** ECDSA pipeline verification
- **Check:** Checks if value is zero (matching the `assert_zero` semantics of all 7 ECDSA constraints).

### `ImpossibleIdeal`

- **Used by:** ECDSA UAIR at the ring level
- **Never actually checked** — it's the ideal type for `assert_zero` constraints, where the constraint builder calls `assert_zero` instead of `assert_in_ideal`.

### `Sha256QxIdeal` / `Sha256QxIdealOverF`

- **Used by:** Q[X] IC pass
- **Enum wrapping** `DegreeOneIdeal(2)`, mapping to field-level versions via a closure in the verifier.

---

## 12. The Lookup Protocol (LogUp)

The lookup protocol enforces that designated trace columns contain only values from prescribed finite lookup tables. This is essential for SHA-256: the 10 bit-polynomial columns (indices 0–9) must contain valid `BinaryPoly<32>` values where all coefficients are in $\{0, 1\}$. Without the lookup argument, a malicious prover could fill these columns with arbitrary polynomials that happen to satisfy the algebraic constraints but don't represent valid binary words.

### Protocol Variants

The implementation provides four LogUp variants, with increasing sophistication:

| Variant | Module | Key property |
|---------|--------|-------------|
| **Core LogUp** | `logup.rs` | Single witness, single table. Proves $\sum_i 1/(\beta - w_i) = \sum_j m_j/(\beta - T_j)$. |
| **Decomposition + LogUp** | `decomposition.rs` | Large table ($2^{K \cdot c}$) decomposed into $K$ sub-tables of $2^c$. Single sumcheck for consistency + membership. |
| **Batched Decomposition + LogUp** | `batched_decomposition.rs` | $L$ witnesses, same decomposed table, single degree-2 sumcheck. Precomputed batched identity polynomial $H$. |
| **GKR Batched Decomposition + LogUp** | `gkr_batched_decomposition.rs` | $L$ witnesses, GKR fractional sumcheck. **No inverse vectors or chunk vectors in proof.** Available but not currently wired into top-level pipelines. |

### The Log-Derivative Identity

All variants are based on the LogUp identity. Given a witness vector $\mathbf{w} = (w_0, \ldots, w_{W-1})$ and a table $\mathbf{T} = (T_0, \ldots, T_{N-1})$, with multiplicities $m_j = |\{i : w_i = T_j\}|$:

$$\sum_{i=0}^{W-1} \frac{1}{\beta - w_i} = \sum_{j=0}^{N-1} \frac{m_j}{\beta - T_j}$$

This identity holds for a random challenge $\beta$ iff every $w_i$ appears in $\mathbf{T}$.

### Decomposition for Large Tables

For tables larger than $2^8$ entries (the `DECOMP_THRESHOLD`), each witness value is decomposed into $K = \lceil \text{width} / 8 \rceil$ chunks:

$$w_i = \sum_{k=0}^{K-1} \text{shifts}[k] \cdot \text{chunks}[k][i]$$

For `BitPoly { width: 32 }`, $K = 4$ chunks of 8 bits each, with shift factors $\text{shifts}[k] = \alpha^{8k}$ where $\alpha$ is the projecting element. Each chunk is looked up in a shared sub-table of $2^8 = 256$ entries, reducing the table size from $2^{32}$ (infeasible) to $2^8$ (trivial).

### Batched Decomposition (Classic)

Given $L$ witnesses looking up into the same decomposed table:

1. **Chunks and inverses sent:** For each of $L$ witnesses: $K$ chunk vectors ($L \cdot K \cdot W$ field elements) plus inverse vectors ($L \cdot K$ inverse-witness vectors + 1 inverse-table vector).
2. **Single sumcheck:** All $L \cdot (K+1)$ identities (one decomposition + $K$ LogUp per witness) are $\gamma$-batched into a single combination function. A precomputed aggregate polynomial $H$ is evaluated pointwise, yielding a degree-2 sumcheck with only 2 MLEs (`eq` and `H`).
3. **Verification:** The verifier replays the sumcheck, recomputes $H$ at the evaluation point from the provided chunks/inverses, and checks multiplicity sums.

### GKR Batched Decomposition (Available, Not Wired In)

The GKR variant (based on Papini & Haböck, ePrint 2023/1284) eliminates inverse vectors and chunk vectors from the proof:

1. **Only multiplicities sent:** Aggregated multiplicities $m_{\text{agg}}^{(\ell)}$ for each of $L$ witnesses.
2. **Fraction tree construction:** For each leaf $(ℓ, k, i)$, define numerator $\alpha^\ell$ and denominator $\beta - c_k^{(\ell)}[i]$. Build a binary tree of fraction additions:
   $$\frac{p_{\text{parent}}}{q_{\text{parent}}} = \frac{p_{\text{left}} \cdot q_{\text{right}} + p_{\text{right}} \cdot q_{\text{left}}}{q_{\text{left}} \cdot q_{\text{right}}}$$
3. **Two GKR proofs:** One for the witness side (numerator/denominator tree) and one for the table side. At each layer, a sumcheck verifies consistency between parent and child fractions.
4. **Root cross-check:** $P_w \cdot Q_t = P_t \cdot Q_w$ (the witness and table log-derivative sums are equal).
5. **Leaf verification:** At the bottom layer, the verifier checks that claimed evaluations match known input polynomials.

**Proof size savings:** The GKR variant saves $O(W + N)$ field elements (inverse vectors) at the cost of $O(\log^2 \max(W, N))$ GKR layer proof elements. For the 8×SHA-256 workload with 10 lookup columns and $W = 512$ rows each, this is a significant reduction.

### Lookup Table Types

| Type | Table size | Contents | Example use |
|------|-----------|----------|-------------|
| `BitPoly { width: W }` | $2^W$ | All binary polynomials $\sum_{k=0}^{W-1} b_k \alpha^k$ for $b_k \in \{0,1\}$ | SHA-256 columns 0–9 (`width: 32`) |
| `Word { width: W }` | $2^W$ | Integers $\{0, 1, \ldots, 2^W - 1\} \bmod q$ | Integer range checks |

Table generation functions:
- `generate_bitpoly_table(width, projecting_element, field_cfg)` — recursive: $T[n + 2^k] = T[n] + \alpha^k$.
- `generate_word_table(width, field_cfg)` — parallelized enumeration.

### Pipeline Integration

The lookup pipeline (`piop/src/lookup/pipeline.rs`) provides high-level orchestration:

- **`prove_batched_lookup()`** / **`prove_batched_lookup_with_indices()`** — group columns by table type, build lookup instances, run the classic protocol. Used by `prove()`, `prove_classic_logup()`, and `prove_generic()`.
- **`prove_gkr_batched_lookup_with_indices()`** — GKR variant, available but not currently wired into any top-level pipeline.
- **`verify_batched_lookup()`** / **`verify_gkr_batched_lookup()`** — reconstruct sub-tables from metadata, verify each group (dispatched by proof variant).

Constants:
- `DEFAULT_CHUNK_WIDTH = 8` (sub-tables of $2^8 = 256$ entries)
- `DECOMP_THRESHOLD = 8` (tables ≤ $2^8$ entries skip decomposition)

### Multi-Degree Sumcheck

The lookup protocol leverages the **multi-degree sumcheck** (`piop/src/sumcheck/multi_degree.rs`), which runs multiple sumcheck degree groups in lockstep with shared verifier randomness. This avoids padding low-degree identities to `max_degree`:

- A degree-2 identity sends 2 evaluations per round.
- A degree-5 identity sends 5 evaluations per round.
- All groups share the same random challenge each round, producing a common evaluation point.

This is used in two contexts:

1. **Within a single lookup protocol:** Different identity groups (decomposition consistency vs. LogUp fraction checks) may have different degrees.
2. **Batched CPR + Lookup (§3 Step 3+3.5):** The CPR sumcheck (degree $\\text{max\\_degree} + 2$, e.g., 4 for SHA-256) and each classic lookup sumcheck (degree 2) are fused into a single multi-degree sumcheck. The CPR group occupies slot 0 and the lookup groups occupy slots 1..$G$. When the lookup sub-table has more variables than the CPR trace (e.g., 8 vs. 7), the CPR MLEs are zero-padded and a `shared_num_vars` is used for the sumcheck; the verifier similarly pads the IC evaluation point and selector reference in `finalize_verifier`.

## 13. Witness Generation

### SHA-256 (`sha256-uair/src/witness.rs`)

The witness generator:

1. Runs the full SHA-256 compression function on the empty-string message (single 512-bit padded block).
2. At each of the 64 rounds, records all 17 column values:
   - Working variables $a, e$ directly from the SHA-256 state (columns 0–1).
   - Derived values $\Sigma_0, \Sigma_1, \text{Maj}, \text{ch\_ef}, \text{ch\_neg\_eg}$ computed from the state (columns 3–7).
   - Message schedule entries $W_t$ and small-sigma values $\sigma_0, \sigma_1$ (columns 2, 8, 9; for $t \geq 16$).
   - Shift decompositions: $S_0 = W_{t-15} \gg 3$, $R_0 = W_{t-15} \& \text{0x7}$, etc. (columns 10–13).
   - Carry values $\mu_a, \mu_e, \mu_W$ computed as `floor(sum / 2^32)` from the non-wrapping integer sum (columns 14–16).
   - Round constants $K_t$ are no longer stored in the trace; they are public inputs.
3. Sets a boundary row at index 64 with the final $a, e$ values (so that carry constraints at row 63 can reference `down[COL_A_HAT]` and `down[COL_E_HAT]` correctly).
4. Zero-pads remaining rows to $2^{\text{num\_vars}}$.

The witness is verified against NIST test vectors (the hash of the empty string matches `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`).

### SHA-256 Split Witnesses

Two additional generators extract subsets of the full trace:
- `generate_poly_witness()` → 14-column `BinaryPoly<32>` trace for the BinaryPoly batch.
- `generate_int_witness()` → 3-column `Int<1>` trace for the Int batch. Each carry value is converted to its integer representation via `bp.to_u64() as i64`.

### ECDSA (`ecdsa-uair/src/witness.rs`)

The witness generator creates a constant fixed-point trace:
- All 258 rows have $X = 1, Y = 1, Z = 0$ (the point at infinity in Jacobian coordinates).
- Scalar bits $b_1, b_2 = 0$ (no additions performed).
- This satisfies the doubling constraints (everything is zero) but does NOT represent a real ECDSA verification. It is used for testing that the constraint system is self-consistent.
- Two witness generators are available: one producing `DensePolynomial<i64, 1>` traces, another producing `Int<4>` traces.

---

## 14. PCS Verifier Optimizations

Three complementary optimizations were applied to reduce the PCS verifier time from ~17.9 ms to ~6.8 ms for the 8×SHA-256+ECDSA combined benchmark (62% reduction).

### Phase 1: Spot-Check Encoding via Precomputed Encoding Matrix

**Problem:** The PCS verifier's dominant cost was two full PNTT encodings per batch: `encode_wide()` (over CombR integers) and `encode_f()` (over Montgomery field elements). Each encodes the full `row_len` → `cw_len` expansion (512 → 2,048 elements), requiring ~143,360 twiddle multiplications, even though the verifier only needs the encoding at the 147 opened column positions.

**PNTT cost breakdown (PnttConfigF2\_16R4B64<1>):**
- Base layer: 2,048 output elements × 63 twiddle muls each = 129,024 (90%)
- Butterfly stages: 7 × 2,048 × 1 = 14,336 (10%)
- Total: 143,360 twiddle multiplications per `encode_wide` or `encode_f` call

**Solution:** Spot-check encoding — compute the linear code only at the opened positions.

#### Encoding Matrix Precomputation

The IPRS encoding is a linear map: $\text{output}[i] = \sum_j M[i][j] \cdot \text{input}[j]$. The matrix $M$ of shape `[OUTPUT_LEN][INPUT_LEN]` is precomputed once in `Radix8PnttParams::new()` and stored in the `encoding_matrix` field.

The precomputation in `precompute_encoding_matrix()` (in `zip-plus/src/code/iprs/pntt/radix8/params.rs`):

1. **Base layer:** For each output index $i$, determine the chunk $i / \text{BASE\_DIM}$ and row $i \bmod \text{BASE\_DIM}$. Apply octet-reversal to the chunk index. Fill the matrix row from the base Vandermonde matrix: `matrix[i][j] = base_matrix[row][col]` for the relevant input indices. The base Vandermonde matrix is computed from the $\text{BASE\_DIM}$-th roots of unity in $\mathbb{F}_{65537}$.
2. **Butterfly stages:** For each depth $k \in [0, \text{DEPTH})$, apply the radix-8 butterfly transform in-place on each *column* of the matrix using the `BUTTERFLY_TABLE` (a constant $8 \times 7$ table of indices into the twiddle array) and precomputed twiddles. Uses `i128` intermediate arithmetic to avoid overflow during the multiply-accumulate.
3. **Convert** final values from `i128` to `PnttInt` (`i64`).

**Memory:** `OUTPUT_LEN × INPUT_LEN × 8` bytes. For `PnttConfigF2_16R4B64<1>`: $2048 \times 512 \times 8 = 8$ MB per IPRS code instance.

#### Spot-Check Methods on `LinearCode` Trait

Two new methods on the `LinearCode` trait (`zip-plus/src/code.rs`):

```rust
fn encode_wide_at_positions(&self, row: &[Zt::CombR], positions: &[usize]) -> Vec<Zt::CombR>;
fn encode_f_at_positions<F>(&self, row: &[F], positions: &[usize]) -> Vec<F>;
```

Both have default implementations that fall back to full encoding + indexing. The IPRS code overrides both with efficient matrix-row dot products:

For each position $p_k$ in the `positions` list, the encoded value is:
$$\text{result}[k] = \sum_{j=0}^{\text{INPUT\_LEN}-1} \text{encoding\_matrix}[p_k][j] \cdot \text{row}[j]$$

For `encode_wide_at_positions`, each element multiplication is `CombR::mul_by_scalar(&PnttInt)` (i.e., multi-limb integer × i64), which hits the optimized `Int<N> * i64` fast path (see Phase 3 below). For `encode_f_at_positions`, the encoding matrix entry is first converted to a field element via `F::from_with_cfg(coeff, field_cfg)`, then standard field multiplication is used.

Both methods are parallelized via `rayon::par_iter` when the `parallel` feature is enabled.

**Cost reduction:** 147 dot products of length 512 = 75,264 multiplications, versus ~143,360 twiddle multiplications for the full PNTT. This is a **~1.9× reduction** in raw multiplication count, but the actual speedup is larger because each dot product is fully independent (better parallelism), avoids butterfly bookkeeping overhead, and benefits from the Phase 3 multiplication optimization.

### Phase 2: Parallel Verification of Independent PCS Batches

The SHA-256 and ECDSA PCS batches are independent (different commitments, different proofs). The benchmark uses `rayon::join` to verify them concurrently:

```rust
let (sha_r, ec_r) = rayon::join(
    || BatchedZipPlus::<ShaZt, ShaLc>::verify(&sha_params, &sha_comm, &sha_pt, &sha_proof),
    || BatchedZipPlus::<EcZt, EcLc>::verify(&ec_params, &ec_comm, &ec_pt, &ec_proof),
);
```

For the 3-batch split-SHA configuration, nested `rayon::join` is used:

```rust
let ((r1, r2), r3) = rayon::join(
    || rayon::join(
        || BatchedZipPlus::<ShaPolyZt, ShaPolyLc>::verify(/* BinaryPoly batch */),
        || BatchedZipPlus::<ShaIntZt, ShaIntLc>::verify(/* Int batch */),
    ),
    || BatchedZipPlus::<EcZt, EcLc>::verify(/* ECDSA batch */),
);
```

This overlaps the verification work across CPU cores. On the MacBook Air M4 (10 cores), the wall-clock time is approximately `max(batch_times)` rather than `sum(batch_times)`.

**Note:** The parallel verification is currently only implemented in the benchmark harness (`snark/benches/e2e_sha256.rs`), not in the pipeline's `verify()` function. Integrating it into the production pipeline would require the pipeline to be aware of multiple PCS batches.

### Phase 3: Optimized `Int<N>` × `i64` Multiply

**Problem:** The spot-check encoding hot path performs ~75,264 multiplications of `Int<LIMBS> × i64` per batch. The generic `MulByScalar` implementation for `Int<N> × Int<M>` uses full-width $N \times M$ schoolbook multiplication ($\text{LIMBS}^2$ limb multiplications for equal-size operands). Since the IPRS encoding matrix entries are `PnttInt = i64` (a single limb), this wastes $(N-1)$ multiplications per term.

**Solution:** A specialized `MulByScalar<&i64>` implementation for `Int<LIMBS>` in `utils/src/mul_by_scalar.rs`:

1. **Reinterpret** the `i64` scalar as `u64` (two's complement representation).
2. **LIMBS × 1 schoolbook:** For each of the `LIMBS` limbs of `self`, perform a 64×64 → 128 widening multiply with the scalar, propagating carry through the limbs.
3. **Sign correction:** If the scalar is negative, its `u64` representation is $2^{64} - |s|$. The schoolbook therefore computes `self * (2^64 - |s|)` = `self * 2^64 - self * |s|`. The unwanted `self * 2^64` term shifts `self` up by one limb position (into limbs `1..LIMBS`). This extra contribution is subtracted back with a borrow chain: for each limb $i \in [1, \text{LIMBS})$, subtract `self.words[i-1]` with borrow propagation.

**Speedup:** `LIMBS` limb-multiplications instead of `LIMBS²`:
- `Int<6>` (SHA CombR): 6 muls instead of 36 → **~6× faster**
- `Int<8>` (ECDSA CombR): 8 muls instead of 64 → **~8× faster**

This optimization benefits both the new spot-check encoding hot path and the existing full PNTT butterfly stages (where twiddles are also PnttInt = i64).

Three targeted correctness tests verify the implementation for `Int<2>`, `Int<4>`, `Int<6>`, `Int<8>` against naive full-width multiply, including edge cases (`i64::MAX`, `i64::MIN`, `0`, `-1`).

### Combined Impact

| Metric | Before all optimizations | After all optimizations | Improvement |
|--------|--------------------------|------------------------|-|
| Combined PCS verifier (parallel) | ~17.9 ms | ~6.8 ms | 62% faster |
| SHA PCS verifier (alone) | ~8–10 ms | ~5 ms | ~50% faster |
| ECDSA PCS verifier (alone) | ~4–6 ms | ~3 ms | ~50% faster |

The verifier now spends its time roughly as:
- Spot-check encoding (147 dot products × `row_len`): ~40%
- Merkle proof verification (147 blake3 hash walks): ~25%
- Column testing inner products: ~20%
- Transcript deserialization + challenge sampling: ~15%

---

## 15. What the Implementation Does NOT Do

### 15.1 CPR→PCS Binding Gap

**The most significant architectural gap.** The CPR produces evaluation claims: "column $i$ of the committed trace evaluates to `up_evals[i]` at the sumcheck point $\mathbf{s}$, and to `down_evals[i]` at the shifted point." The PCS independently verifies that the committed polynomials pass proximity testing and evaluation consistency at a *different* point (derived by hashing the CPR point into `i128` values via `derive_pcs_point()`).

**What's missing:** There is no check that the PCS-evaluated values are consistent with the CPR's claimed `up_evals` or `down_evals`. Both vectors are prover-supplied values that the CPR verifier plugs into the constraint expression to check algebraic consistency with the sumcheck terminal claim — but neither is verified against the PCS commitment. The CPR operates on projected $\mathbb{F}_q$ values at the sumcheck point; the PCS evaluates committed `BinaryPoly<32>` polynomials at an `i128` point in a different coordinate system.

Specifically for `down_evals`: the "down" (shifted) trace is never separately committed. The prover constructs it by dropping the first row and appending a zero row (`column[1..].iter().cloned().collect()`), but only the original unshifted trace is committed via the PCS. The `down_evals` are absorbed into the Fiat-Shamir transcript (maintaining transcript consistency) but are otherwise taken on faith.

**Impact:** The IC, CPR, and PCS each independently verify their respective claims, but nothing cryptographically binds the CPR's column-evaluation claims (`up_evals` and `down_evals`) to the PCS's committed polynomials. A malicious prover could (in theory) commit one trace via PCS, run the PIOP on a different trace, and both would pass independently.

**What would fix it:** Either (a) the PCS must evaluate at the CPR's sumcheck point $\mathbf{s}$ and the verifier checks `up_evals[i] == pcs_opened_value[i]` (and derives `down_evals` from the committed polynomials evaluated at the appropriately shifted point), or (b) a secondary sumcheck reduces the CPR claims to a single evaluation claim that the PCS can verify directly.

This is documented in `pipeline.rs` with a detailed comment block.

### 15.2 Sumcheck Structure

The PIOP protocol chain described in the paper is: **Ideal Check → CPR → Sumcheck → PCS**. In the non-batched path (`prove()`, `prove_generic()`), the sumcheck is embedded *inside* the CPR (as `MLSumcheck::prove` / `MLSumcheck::verify`), not invoked as a separate top-level protocol step. The CPR internally runs a sumcheck to reduce its batched constraint sum to a single-point claim.

In the **batched classic LogUp path** (`prove_classic_logup()`), the sumcheck is elevated to the top level: `MultiDegreeSumcheck::prove_as_subprotocol` runs at the pipeline level in `pipeline.rs`, receiving pre-built groups from both CPR and lookup. This is a genuine top-level multi-degree sumcheck invocation that subsumes both the CPR's and lookup groups' sumchecks into a single pass (see §3 Step 3+3.5).

### 15.3 Three SHA-256 Constraints Missing

Constraints C13 (d-delay), C14 (h-delay), and C15 (message schedule W update with full lookback) require multi-row lookback. The current UAIR trait only provides `up` (current row) and `down` (next row). C12 (W-schedule carry) is implemented as a partial approximation using up/down rows, but the full constraint requires lookbacks of 2, 7, 15, and 16 rows. The missing constraints don't materially affect prover cost (same algebraic degree as implemented constraints) but mean the system does not fully enforce SHA-256 correctness.

Specifically: without C13–C15, a prover could supply a trace where:
- The register relay ($d_{t+1} = a_{t-3}$) is wrong.
- The message schedule ($W_t$ for $t \geq 16$) is wrong (beyond what C12's partial check catches).
- And the proof would still pass.

### 15.4 ECDSA Not Over Real secp256k1 Field

The ECDSA constraints operate on `Int<4>` (256-bit integers) in the unified pipeline and `DensePolynomial<i64, 1>` (64-bit integers) in the legacy formulation, both using a toy curve over $\mathbb{F}_{101}$, not the real secp256k1 base field (256-bit prime). The constraint *algebra* is correct (it correctly describes Jacobian-coordinate point doubling and addition with Shamir's trick), but:
- The witness values are from $\mathbb{F}_{101}$, not $\mathbb{F}_p$.
- The `Int<4>` pipeline uses `prove_generic`/`verify_generic` with `EcdsaScalarZipTypes` (`Eval = Int<4>`, `CombR = Int<8>`, `PcsF = MontyField<8>`), avoiding the dual-ring architecture entirely.

### 15.5 ECDSA Boundary Constraints Not Enforced

The 7 ECDSA constraints are "non-boundary" constraints applied uniformly to all rows. The implementation does not enforce:
- Row 1 (precomputation): that the initial accumulator is the identity element.
- Row 258 (final check): that the final accumulated point matches the expected signature verification result.

### 15.6 No Compact Proof Serialization

Field elements are serialized at full width (8 bytes per i64 coefficient, 32 bytes per PiopField element in Montgomery form). For DEPTH=1 IPRS coefficients that fit in ~45 bits, this wastes ~23 bits per coefficient. Deflate compression compensates (~1.8–2.1× ratio for DEPTH=1) but not for DEPTH≥2. The eval-phase batching dramatically reduced the eval-phase contribution, so the remaining proof size is dominated by test-phase Merkle openings and column data.

### 15.7 PCS Field Is Not 256-bit

The PIOP field is 128-bit (~120-bit prime). This provides 128-bit soundness for the Schwartz-Zippel lemma (random evaluation). However, a production system over secp256k1 would likely want a 256-bit field to match the curve's security level.

### 15.8 No Zero-Knowledge

The current implementation provides no blinding or zero-knowledge. The PCS proof reveals information about the trace (Merkle openings contain actual committed polynomial evaluations). Adding ZK would require blinding the trace with random polynomials before commitment.

### 15.9 BitPoly Membership Now Handled by Lookups

BitPoly membership (binary coefficient checks for Ch, ¬e∧g, Maj columns) was previously handled by a `BitPolyIdeal` that always returned `true` at the IC verifier level (since binary-coefficient membership is not a mathematical ideal). This has been removed; BitPoly membership is now enforced by the lookup protocol (§12) which directly proves that all coefficients are in $\{0, 1\}$.

### 15.10 Overflow Checking Disabled

The PCS is instantiated with `CHECK = false` (the `UNCHECKED` constant). This skips runtime overflow checks on polynomial coefficient arithmetic during IPRS encoding and evaluation. In principle, if coefficients grow too large, integer overflow could corrupt the proof. In practice, the coefficient bounds are analytically verified to be safe for the configured parameters.

### 15.11 Split-Trace PIOP Not Integrated

The split-trace architecture (§8) currently only operates at the **PCS level** — the two SHA-256 batches are committed and verified independently. The split-trace benchmarks measure PCS-only commit/verify/proof-size, but do **not** run the PIOP (Ideal Check + CPR) on split traces. The full pipeline still uses the monolithic 17-column trace for the PIOP and only supports the non-split PCS.

To integrate the split-trace architecture into the full PIOP pipeline, the IC and CPR would need to be run separately on each batch (BinaryPoly constraints on the 14-column batch, Q[X] constraints on the 3-column batch) with appropriate column index remapping.

### 15.12 Parallel Verification Not in Production Pipeline

The parallel PCS verification via `rayon::join` (§14, Phase 2) is only implemented in the benchmark harness. The `verify()`, `verify_generic()`, `verify_dual_ring()`, and `verify_dual_circuit()` functions in `pipeline.rs` run PCS verification sequentially. Integrating parallelism would require the pipeline to manage multiple PCS contexts.

---

## 16. Optimization Opportunities

### 16.1 Compact Codeword Serialization (Proof Size)

**Estimated savings: ~100 KB for 8×SHA-256 BinaryPoly batch.**

Each `DensePolynomial<i64, 32>` codeword element (used for BinaryPoly columns) serializes as 32 × 8 = 256 bytes. However, for DEPTH=1 IPRS codes, the coefficient bitbound is ~45 bits, so each coefficient fits in 6 bytes. Serializing at 6 bytes per coefficient would give 32 × 6 = 192 bytes per codeword element — a 25% reduction on column-opening data.

After eval-phase batching, column openings dominate the test-phase proof. For 147 openings × 14 BinaryPoly columns:
- Current: 147 × 14 × 256 ≈ 527 KB
- Compact: 147 × 14 × 192 ≈ 395 KB
- **Savings: ~132 KB**

### 16.2 Narrowing the ECDSA Field (Proof Size + Verifier Speed)

**Estimated savings: ~16 KB on eval phase, plus ~2× faster ECDSA field arithmetic.**

Currently ECDSA uses `CombR = Int<8>` (512-bit) which forces `PcsF = MontyField<8>` (64 bytes/element). The actual overflow bound for 9 columns × 1 row × 256-bit evaluations with 128-bit challenges is ~384 bits, fitting in `Int<6>` (384-bit). Narrowing to `CombR = Int<6>` and `PcsF = MontyField<4>` (32 bytes) would:
- Halve the ECDSA eval-phase batched row (512 × 32 = 16 KB vs 512 × 64 = 32 KB).
- Speed up ECDSA field arithmetic by ~2× (128-bit Montgomery multiplication vs 512-bit).
- Reduce the test-phase combined row (512 × 48 vs 512 × 64).

### 16.3 Batch Merkle Verification (Verifier Speed)

**Estimated savings: ~10–20% of Merkle verification time.**

Currently each of the 147 column openings independently hashes the leaf and walks up the Merkle tree (~11 hash levels). Many openings share Merkle path prefixes. Batching the verification to share work on common path segments could save duplicate hash computations.

### 16.4 SIMD-Accelerated Inner Products (Verifier Speed)

The inner-product kernels (`MBSInnerProduct`, `ScalarProduct`, `CombDotChal`) may not fully exploit SIMD. For `MontyField<4>` (256-bit), vectorized Montgomery multiplication using NEON (Apple M-series) or AVX2 intrinsics could accelerate both column testing and evaluation verification.

### 16.5 Reducing NUM_COLUMN_OPENINGS (Proof Size + Verifier Speed)

`NUM_COLUMN_OPENINGS = 147` is a security parameter. Reducing it would proportionally decrease both proof size and verifier time. A formal analysis of the soundness trade-off (proximity testing false-accept probability vs. number of openings) could determine the minimum safe value for the target security level.

### 16.6 Full Split-Trace Pipeline Integration

Integrating the split-trace architecture into the full PIOP pipeline (not just PCS) would require:
1. Running IC₁ + CPR₁ on the 14-column BinaryPoly batch ({0,1}^{<32}[X] constraints C1–C6).
2. Running IC₂ + CPR₂ on the 3-column Int batch (Q[X] constraints C7–C12).
3. Column index remapping between the UAIR constraint expressions (which reference the original 17-column layout) and the split batches.
4. Separate PCS test + evaluate for each batch.

This would yield the split-trace proof size benefits in the full end-to-end pipeline.

### 16.7 Multi-Row UAIR Access

Extending the UAIR trait beyond the current `up`/`down` (1-step lookback) framework to support arbitrary row offsets would enable implementing the missing SHA-256 constraints C12–C14. This could be done via:
- **Relay columns:** Adding intermediate columns $b, c, f, g$ with 1-step delay constraints ($b_{t+1} = a_t$, etc.), increasing the trace width but staying within the `up`/`down` model.
- **Multi-row access trait extension:** Adding `row_at(offset)` to the constraint builder, allowing constraints to reference `row[-3]`, `row[-15]`, etc. directly.

### 16.8 Verifier Target: Sub-5ms

The combined PCS verifier currently runs at ~6.8 ms (parallel, MacBook Air M4). To reach the <5 ms target:
- Narrowing ECDSA to `MontyField<4>` would speed up the ECDSA batch (faster field ops, smaller data).
- Batch Merkle verification could save ~0.5 ms.
- SIMD inner products could save ~0.5 ms.
- Together these could bring the combined time to ~4–5 ms.

### 16.9 Production Pipeline Parallelism

Moving the parallel `rayon::join` verification from the benchmark into the production `verify()` / `verify_generic()` functions. This requires the pipeline to support a multi-PCS-batch model where multiple commitments are verified concurrently.

---

## 17. Test Coverage

Eight integration tests in `snark/tests/`:

| Test | What it exercises |
|------|------------------|
| `round_trip_pcs_sha256` | PCS-only: commit → test → evaluate → verify. Proves the PCS works in isolation. |
| `ideal_check_succeeds_on_valid_sha256_witness` | IC prover on real SHA-256 trace. Verifies that the IC produces a valid proof for the 6 {0,1}^{<32}[X] constraints. Does NOT run verification. |
| `qx_ideal_check_succeeds_on_valid_sha256_witness` | IC prover on Q[X]-projected SHA-256 trace. Verifies the 6 Q[X] constraints (including the $(X-2)$ ideal). Does NOT run verification. |
| `full_pipeline_round_trip` | Single-ring: `prove()` → `verify()` with TrivialIdeal. Full IC + CPR + PCS round-trip for the 6 {0,1}^{<32}[X] constraints. |
| `batched_classic_logup_round_trip` | Batched CPR+Lookup: `prove_classic_logup()` → `verify()` with `BatchedClassic` proof. IC + batched multi-degree sumcheck (CPR degree-4 + lookup degree-2) + PCS round-trip. Uses 10 `BitPoly(32)` lookup columns with `shared_num_vars = 8` (sub-table 256 > trace 128). Verifies the proof is `BatchedClassic` variant and that verification accepts. |
| `dual_ring_pipeline_round_trip` | Dual-ring: `prove_dual_ring()` → `verify_dual_ring()`. BP pass (6 constraints, TrivialIdeal) + QX pass (6 constraints, real DegreeOne(2) ideal) + PCS. **This is the most comprehensive test.** |
| `ecdsa_ideal_check_succeeds_on_valid_witness` | ECDSA IC prover on the constant fixed-point trace. Verifies the 7 i64 constraints produce a valid IC proof. |
| `ecdsa_pipeline_round_trip` | ECDSA single-ring `Int<4>` pipeline: `prove_generic()` → `verify_generic()`. All 7 constraints checked in `Int<4>` ring with `EcdsaScalarZipTypes` (PCS field = `MontyField<8>`). Full IC + CPR + PCS round-trip on a zero trace. |

Additionally:
- Each UAIR crate has unit tests for constraint count, max degree, scalar collection, and (for SHA-256) NIST test vector validation.
- `zinc-utils` has 3 targeted tests for the optimized `Int<N> * i64` multiply covering `Int<2>`, `Int<4>`, `Int<6>`, `Int<8>` against naive full-width multiply, including edge cases (`i64::MAX`, `i64::MIN`, `0`, `-1`).
- The `zip-plus` crate has internal tests for batched PCS verification, single-poly PCS verification, and IPRS code encoding/decoding.
- The `zinc-piop` crate has lookup-specific tests: core LogUp, decomposition LogUp, batched decomposition, and GKR batched decomposition, testing both prover and verifier paths.

---

## 18. Benchmark Structure

### `e2e_sha256` (6 criterion groups)

Six criterion benchmark groups in `snark/benches/e2e_sha256.rs`:

| Suite | What it measures |
|-------|-----------------|
| `sha256_single` | **Headline benchmark.** PCS-only prover/verifier + full-pipeline (`pipeline::prove`/`pipeline::verify`) prover/verifier + proof size for single SHA-256 (17 cols, DEPTH=1). Reports split-PCS proof sizes (14 BPoly + 3 Int). |
| `sha256_8x_ecdsa` | **Paper target benchmark.** Two separate PCS batches: SHA-256 columns as `BinaryPoly<32>` + 9 ECDSA columns as `Int<4>` (256-bit scalar), both at DEPTH=1 (512 rows). Reports combined prover/verifier timing, proof sizes (monolithic and split-SHA variants), and parallel verification. |
| `sha256_piop_only` | IC and CPR prover in isolation (no PCS). |
| `sha256_8x_ecdsa_end_to_end` | Full E2E including PIOP + Lookup + PCS for 8×SHA-256 + ECDSA. |
| `sha256_full_pipeline` | Uses `pipeline::prove()` / `pipeline::verify()` directly, reports detailed proof size breakdown (PCS + PIOP + Lookup components). |
| `sha256_8x_ecdsa_logup_comparison` | **LogUp strategy comparison.** Three variants side-by-side: NoLookup, ClassicLogUp (batched multi-degree sumcheck), SeparateLogUp (independent sequential sumchecks). See below. |

#### LogUp Comparison Benchmark (`sha256_8x_ecdsa_logup_comparison`)

Compares three LogUp strategies for 8×SHA-256 + ECDSA:

| Variant | Pipeline function | Description |
|---------|-------------------|-------------|
| **NoLookup** | `prove_classic_logup(..., &[])` | Baseline with no lookup constraints. Runs the batched multi-degree sumcheck with CPR only. |
| **ClassicLogUp** | `prove_classic_logup(..., &sha_lookup_specs)` | Classic batched decomposition LogUp where the CPR sumcheck and all lookup sumchecks are **batched into a single multi-degree sumcheck** (shared challenges, single evaluation point). |
| **SeparateLogUp** | `prove(..., &sha_lookup_specs)` | Classic batched decomposition LogUp where the CPR and lookup run **independent sequential sumchecks** (separate challenges, separate evaluation points). |

Benchmark functions for each variant:
- `{Variant}/SHAProver` — SHA-256 prover only (no ECDSA)
- `ClassicLogUp/TotalProver` and `SeparateLogUp/TotalProver` — SHA + ECDSA combined prover
- `ClassicLogUp/TotalVerifier` and `SeparateLogUp/TotalVerifier` — SHA + ECDSA combined verifier

Also reports lookup proof sizes and the batched-vs-separate size ratio.

Run with: `cargo bench -p zinc-snark --bench e2e_sha256 --features 'parallel simd asm' -- 'LogUp Comparison'`

### `steps_sha256_8x_ecdsa` (22 granular steps)

A separate criterion benchmark file (`snark/benches/steps_sha256_8x_ecdsa.rs`) providing a **per-step timing breakdown** for the 8×SHA-256 + ECDSA proving stack. Each pipeline phase is isolated into its own benchmark function so that individual step timings sum to the E2E total.

| # | Step | Description |
|---|------|-------------|
| 1 | `SHA/WitnessGen` | Generate the 8×SHA-256 trace (17 columns × 512 rows) |
| 2 | `ECDSA/WitnessGen` | Generate the ECDSA trace (9 columns × 512 rows) |
| 3 | `SHA/PCS/Commit` | PCS commit the SHA-256 trace |
| 4 | `ECDSA/PCS/Commit` | PCS commit the ECDSA trace |
| 5 | `SHA/PIOP/FieldSetup` | Sample PIOP field, IC evaluation point, lift trace coefficients to $\mathbb{F}_q[X]$ |
| 6 | `SHA/PIOP/ProjectIC` | Project lifted $\mathbb{F}_q[X]$ trace to $\mathbb{F}_q$ scalars for IC |
| 7 | `ECDSA/PIOP/FieldSetup` | Same for ECDSA |
| 8 | `ECDSA/PIOP/ProjectIC` | Same for ECDSA |
| 9 | `SHA/PIOP/IdealCheck` | Run IC prover (6 constraints) |
| 10 | `ECDSA/PIOP/IdealCheck` | Run IC prover (7 constraints) |
| 11 | `SHA/PIOP/ProjectCPR` | Draw $\alpha$, project scalars, project trace to $\mathbb{F}_q$ for CPR |
| 12 | `ECDSA/PIOP/ProjectCPR` | Same for ECDSA |
| 13 | `SHA/PIOP/CPR` | Run CPR sumcheck prover |
| 14 | `ECDSA/PIOP/CPR` | Run CPR sumcheck prover |
| 15 | `SHA/PIOP/LookupExtract` | Extract and project lookup columns from the SHA trace |
| 16 | `SHA/PIOP/Lookup` | Run LogUp prover (classic batched decomposition) |
| 17 | `SHA/PCS/Test` | PCS proximity testing (test phase) |
| 18 | `ECDSA/PCS/Test` | PCS proximity testing (test phase) |
| 19 | `SHA/PCS/Evaluate` | PCS evaluation opening (eval phase) |
| 20 | `ECDSA/PCS/Evaluate` | PCS evaluation opening (eval phase) |
| 21 | `E2E/Prover` | Full prover: `prove_dual_circuit` (SHA + ECDSA shared PIOP + separate PCS) |
| 22 | `E2E/Verifier` | Full verifier: `verify_dual_circuit` (SHA + ECDSA shared PIOP + separate PCS) |

The E2E/Prover (step 21) uses `prove_dual_circuit` which shares the Fiat-Shamir transcript, IC evaluation point, and projecting element across both circuits, and batches both CPR passes plus lookup groups into a single multi-degree sumcheck. Each circuit retains its own PCS commitment and proof.

Run with: `cargo bench -p zinc-snark --bench steps_sha256_8x_ecdsa --features 'parallel simd asm'`

### Benchmark Constants

```
SHA256_NUM_VARS        = 7    // 128 rows (64 real + 64 padding)
SHA256_8X_NUM_VARS     = 9    // 512 rows (8 × 64 rounds)
ECDSA_NUM_VARS         = 9    // 512 rows (258 real + 254 padding)
SHA256_BATCH_SIZE      = 17   // 17 SHA-256 columns (monolithic: 14 bitpoly + 3 int)
SHA256_POLY_BATCH_SIZE = 14   // BinaryPoly columns (split)
SHA256_INT_BATCH_SIZE  = 3    // Int columns (split)
ECDSA_BATCH_SIZE       = 9    // 9 ECDSA columns
SHA256_LOOKUP_COL_COUNT = 10  // Q[X] bitpoly columns 0–9 with LogUp
```

### Dual-PCS Architecture for 8×SHA-256 + ECDSA

The `sha256_8x_ecdsa` benchmark uses **two (or three) independent PCS batches** rather than a single combined trace:

| Batch | Columns | Eval type | ZipTypes | IPRS code | Rows | Field |
|-------|---------|-----------|----------|-----------|------|-------|
| SHA-256 (mono) | 17 | `BinaryPoly<32>` | `Sha256ZipTypes<i64, 32>` | R4B64 DEPTH=1 | 512 | `MontyField<4>` (128-bit) |
| SHA-256 BPoly (split) | 14 | `BinaryPoly<32>` | `Sha256ZipTypes<i64, 32>` | R4B64 DEPTH=1 | 512 | `MontyField<4>` (128-bit) |
| SHA-256 Int (split) | 3 | `Int<1>` | `Sha256IntZipTypes` | R4B64 DEPTH=1 | 512 | `MontyField<4>` (128-bit) |
| ECDSA | 9 | `Int<4>` | `EcdsaScalarZipTypes` | R4B64 DEPTH=1 | 512 | `MontyField<8>` (512-bit) |

This design matches the paper's intent: ECDSA values are field elements (scalars), not polynomials, so committing them as `Int<4>` (256-bit signed integer) avoids the 32× overhead of processing 32 binary coefficients per cell.

### IPRS Code Configuration

All benchmarks use `PnttConfigF2_16R4B64<1>`:

| Parameter | Value |
|-----------|-------|
| Field | $\mathbb{F}_{65537}$ ($2^{16} + 1$) |
| BASE\_LEN | 64 |
| BASE\_DIM | 256 |
| DEPTH | 1 |
| INPUT\_LEN (`row_len`) | 512 ($64 \times 8^1$) |
| OUTPUT\_LEN (`cw_len`) | 2,048 ($256 \times 8^1$) |
| Rate | 1/4 |
| Base twiddles | $[1, 4096, -256, 16, -1, -4096, 256, -16]$ |
| Encoding matrix size | 2,048 × 512 × 8 B = 8 MB |

### Benchmark Results (MacBook Air M4, `parallel` + `simd`)

**1×SHA-256 full pipeline:**
- Prover: ~10 ms (IC=5.4ms, CPR=1.2ms, PCS commit=1.8ms, test=0.8ms, eval=1.0ms)
- PCS proof: 790 KB raw, 374 KB compressed
- Split PCS: 533 KB raw, 259 KB compressed

**8×SHA-256 + ECDSA:**
- Combined PCS prover: ~23.8 ms
- Combined PCS verifier (parallel): ~6.8 ms (62% improvement from optimizations)
- SHA PCS: 830 KB, ECDSA PCS: 199 KB
- Combined monolithic: 1,045 KB raw, 571 KB compressed
- Combined split-SHA: 821 KB raw, 460 KB compressed

**LogUp strategy comparison (8×SHA-256 + ECDSA):**
- **ClassicLogUp (batched):** CPR + lookup sumchecks fused into a single multi-degree sumcheck via `prove_classic_logup`. Shared verifier challenges, single evaluation point.
- **SeparateLogUp:** CPR and lookup run independent sequential sumchecks via `prove`. Separate challenges and evaluation points.
- **NoLookup:** Baseline with no lookup constraints (CPR-only multi-degree sumcheck).
- The LogUp comparison benchmark (`sha256_8x_ecdsa_logup_comparison`) reports per-phase timing, lookup proof sizes, and the batched-vs-separate size ratio.

Run all e2e benchmarks:
```bash
cargo bench -p zinc-snark --bench e2e_sha256 --features 'parallel simd asm'
```

Run per-step breakdown:
```bash
cargo bench -p zinc-snark --bench steps_sha256_8x_ecdsa --features 'parallel simd asm'
```

---

## 19. Crate Map

```
zinc-plus-new/
├── snark/              # Top-level pipeline (prove/verify), benchmarks, integration tests
│   ├── src/pipeline.rs # prove(), prove_classic_logup(), prove_generic(),
│   │                   # verify(), verify_generic(), prove_dual_ring(),
│   │                   # verify_dual_ring(), prove_dual_circuit(),
│   │                   # verify_dual_circuit(), derive_pcs_point(),
│   │                   # extract_lookup_columns_from_field_trace(),
│   │                   # BatchedCprLookupProof, DualCircuitZincProof,
│   │                   # LookupProofData::BatchedClassic
│   ├── benches/        # Criterion benchmarks (e2e_sha256: 6 groups,
│   │                   # steps_sha256_8x_ecdsa: 22 granular steps)
│   └── tests/          # 8 integration tests (incl. batched_classic_logup)
├── piop/               # PIOP protocols
│   └── src/
│       ├── ideal_check.rs           # IC prove/verify
│       ├── combined_poly_resolver.rs # CPR prove/verify (contains sumcheck),
│       │                            # split API: build_prover_group(),
│       │                            # finalize_prover(),
│       │                            # build_verifier_pre_sumcheck(),
│       │                            # finalize_verifier()
│       ├── combined_poly_resolver/
│       │   └── structs.rs           # Proof, CprSumcheckGroup,
│       │                            # CprVerifierPreSumcheck
│       ├── sumcheck.rs              # Sumcheck protocol
│       ├── sumcheck/multi_degree.rs # Multi-degree sumcheck (shared challenges)
│       ├── projections.rs           # Decoupled two-stage trace projections
│       │                            # (coefficients → F[X] → F)
│       └── lookup/                  # LogUp lookup argument
│           ├── mod.rs               # Module structure and re-exports
│           ├── logup.rs             # Core LogUp protocol
│           ├── decomposition.rs     # Decomposition + LogUp (large tables)
│           ├── batched_decomposition.rs # Batched decomp (L witnesses, 1 sumcheck),
│           │                            # split API: build_prover_group(),
│           │                            # finalize_prover(),
│           │                            # build_verifier_pre_sumcheck(),
│           │                            # finalize_verifier()
│           ├── gkr_logup.rs         # GKR fractional sumcheck
│           ├── gkr_batched_decomposition.rs # GKR + batched decomp (available, not wired in)
│           ├── pipeline.rs          # High-level prove/verify orchestration
│           ├── structs.rs           # Proof types, instances, errors,
│           │                        # LookupSumcheckGroup,
│           │                        # LookupVerifierPreSumcheck
│           └── tables.rs            # Table generation (BitPoly, Word),
│                                    # decomposition, multiplicity computation
├── sha256-uair/        # SHA-256 UAIR + witness (17 columns: 14 BPoly + 3 Int)
│   └── src/
│       ├── lib.rs      # 6 {0,1}^{<32}[X] + 3 Q[X] constraints, ideal types,
│       │               # Sha256UairBp + Sha256UairQx
│       ├── witness.rs  # Generates 64-row SHA-256 trace, split witnesses
│       │               # (generate_poly_witness, generate_int_witness),
│       │               # POLY_COLUMN_INDICES[14], INT_COLUMN_INDICES[3]
│       └── constants.rs # H[], K[] SHA-256 constants
├── ecdsa-uair/         # ECDSA UAIR + witness (9 columns)
│   └── src/
│       ├── lib.rs       # 7 Int<4> + 7 i64 constraints, ideal types
│       ├── constraints.rs # Mathematical specification of all 7 constraints
│       └── witness.rs   # Constant fixed-point witness generator (Int<4> + i64)
├── uair/               # Abstract UAIR trait with heterogeneous trace
│   └── src/
│       └── lib.rs       # Uair trait, UairSignature (binary_poly_cols,
│                        # arbitrary_poly_cols, int_cols), TraceRow,
│                        # ConstraintBuilder
├── poly/               # Polynomial types (BinaryPoly, DensePolynomial, MLE)
├── zip-plus/           # Zip+ batched PCS (IPRS codes, Merkle trees)
│   └── src/
│       ├── code.rs             # LinearCode trait (encode, encode_wide,
│       │                       # encode_*_at_positions)
│       ├── code/iprs.rs        # IPRS code: spot-check encoding overrides
│       ├── code/iprs/pntt/radix8/
│       │   ├── params.rs       # Radix8PnttParams (encoding_matrix precomputation)
│       │   └── butterfly.rs    # BUTTERFLY_TABLE, radix-8 butterfly
│       ├── batched_pcs/        # Batched PCS (commit, prove, verify)
│       │   ├── phase_commit.rs # Batched commitment (Merkle root)
│       │   └── phase_evaluate.rs # Batched eval (β-batched row)
│       ├── pcs/                # Single-poly PCS
│       │   ├── phase_commit.rs # Single-poly commitment
│       │   ├── phase_prove.rs  # Merged test+evaluate (proximity + opening)
│       │   └── phase_verify.rs # Merged verify (spot-check + eval check)
│       └── merkle.rs           # Merkle tree commit/open/verify (Blake3)
├── transcript/         # BLAKE3 Fiat-Shamir transcript
│   └── src/lib.rs      # Blake3Transcript (+ KeccakTranscript alias)
├── primality/          # Miller-Rabin primality test
└── utils/              # Shared utilities
    └── src/
        ├── mul_by_scalar.rs   # MulByScalar trait, optimized Int<N>*i64
        └── ...                # Field conversion, inner products, Int<N> type
```
