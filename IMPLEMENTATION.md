# Zinc+ End-to-End Implementation: What It Does and What It Does Not

**Date:** 2026-02-23

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [The Prover Pipeline in Detail](#3-the-prover-pipeline-in-detail)
4. [The Verifier Pipeline in Detail](#4-the-verifier-pipeline-in-detail)
5. [SHA-256 Arithmetization](#5-sha-256-arithmetization)
6. [ECDSA Arithmetization](#6-ecdsa-arithmetization)
7. [The Dual-Ring Pipeline](#7-the-dual-ring-pipeline)
8. [The Split-Trace Architecture](#8-the-split-trace-architecture)
9. [Proof Format and Serialization](#9-proof-format-and-serialization)
10. [Fiat-Shamir Transcript](#10-fiat-shamir-transcript)
11. [Ideal Types and Membership Checks](#11-ideal-types-and-membership-checks)
12. [Witness Generation](#12-witness-generation)
13. [PCS Verifier Optimizations](#13-pcs-verifier-optimizations)
14. [What the Implementation Does NOT Do](#14-what-the-implementation-does-not-do)
15. [Optimization Opportunities](#15-optimization-opportunities)
16. [Test Coverage](#16-test-coverage)
17. [Benchmark Structure](#17-benchmark-structure)
18. [Crate Map](#18-crate-map)

---

## 1. Overview

The Zinc+ implementation is a research SNARK (Succinct Non-interactive Argument of Knowledge) that proves knowledge of a valid SHA-256 hash computation and (partially) an ECDSA signature verification. The system combines:

- A **Polynomial Commitment Scheme (PCS)** called Zip+ (batched, using IPRS codes and FRI-like proximity testing)
- A **Polynomial IOP (PIOP)** consisting of an Ideal Check protocol and a Combined Polynomial Resolver (CPR) with an internal sumcheck
- **UAIR constraint systems** (Universal Algebraic Intermediate Representations) for SHA-256 and ECDSA

The end-to-end pipeline takes a witness trace (a matrix of ring elements), commits to it via the PCS, proves that the trace satisfies the UAIR constraints via the PIOP, and produces a proof that a verifier can check without seeing the trace.

### Key Performance Numbers (MacBook Air M4, 8×SHA-256 + ECDSA, `parallel` + `simd` features)

| Metric | Current | Target |
|--------|---------|--------|
| Combined PCS prover | ~23.8 ms | < 30 ms ✓ |
| Combined PCS verifier (parallel) | ~6.8 ms | < 5 ms |
| Proof size (compressed) | 277 KB (split-SHA, 1×SHA-256) | ≤ 300 KB ✓ |
| Proof size (compressed, 8×SHA-256+ECDSA) | 471 KB (split) / 585 KB (mono) | ≤ 300 KB |

---

## 2. Architecture

### The Protocol Stack

```
                    ┌─────────────────┐
                    │    Application   │   SHA-256 / ECDSA UAIR
                    └────────┬────────┘
                             │ trace + constraints
                    ┌────────▼────────┐
                    │      PIOP       │   Ideal Check → CPR → (Sumcheck)
                    └────────┬────────┘
                             │ evaluation claims
                    ┌────────▼────────┐
                    │       PCS       │   Zip+ (commit → test → evaluate → verify)
                    └─────────────────┘
```

### Crate Dependencies

- `zinc-snark` — the top-level pipeline (`pipeline.rs`), glues everything together
- `zinc-piop` — Ideal Check, Combined Poly Resolver, Sumcheck protocols
- `zinc-sha256-uair` — SHA-256 constraint system and witness generation
- `zinc-ecdsa-uair` — ECDSA constraint system and witness generation
- `zinc-uair` — abstract UAIR trait, constraint builders, ideal collectors
- `zinc-poly` — polynomial types: `BinaryPoly<D>`, `DensePolynomial<C, D>`, MLEs
- `zip-plus` — the Zip+ batched PCS (commit, test, evaluate, verify)
- `zinc-transcript` — Keccak-based Fiat-Shamir transcript
- `zinc-utils` — shared utilities: field conversion, inner products, `MulByScalar` trait, `Int<N>` type

---

## 3. The Prover Pipeline in Detail

The prover is implemented in `snark/src/pipeline.rs`. There are multiple entry points:

- `prove()` — BinaryPoly-specific, used for SHA-256 single-ring proofs.
- `prove_generic()` — parameterized over any projectable ring `R`, used for ECDSA `Int<4>` proofs.
- `prove_dual_ring()` — runs two sequential PIOP passes (BinaryPoly + Q[X]) on the same committed trace, used for full SHA-256 proofs.

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

### Step 2: PIOP — Ideal Check (IC)

```
(ic_proof, ic_state) = IdealCheckProtocol::prove_as_subprotocol::<BinaryPoly<32>, Sha256Uair>(
    transcript, trace, num_constraints, num_vars, field_cfg
)
```

The Ideal Check protocol operates as follows:

1. **Coefficient lifting $\mathbb{Z}[X] \to \mathbb{F}_q[X]$:** A random field element $\alpha$ is sampled from the Fiat-Shamir transcript, but it is **not used for evaluation** — it is only used to obtain the field configuration (`α.cfg()`). Each trace cell is lifted from its native ring to a polynomial over $\mathbb{F}_q$ by converting each coefficient individually. For example, a `BinaryPoly<32>` with binary coefficients $p(X) = \sum c_i X^i$ ($c_i \in \{0,1\}$) becomes a `DynamicPolynomialF<F>` with $\mathbb{F}_q$ coefficients $\hat{p}(X) = \sum \hat{c}_i X^i$ where $\hat{c}_i = c_i \bmod q$. This is a coefficient-wise embedding (not evaluation at $\alpha$), preserving the polynomial structure: the trace lives in $\mathbb{F}_q[X]$ after this step.

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

### Step 4: PCS Test

```
test_transcript = BatchedZipPlus::test(params, trace, hint)
```

The PCS "test" phase is the proximity testing step of the Zip+ protocol. Unlike FRI-based systems that use iterative folding rounds, Zip+ performs a **single-round direct spot-check proximity test**: the prover sends one combined row and then opens the committed Merkle tree at randomly sampled column positions. The verifier re-encodes that combined row at only those positions and checks consistency.

Implementation: `BatchedZipPlus::test()` in `zip-plus/src/batched_pcs/phase_test.rs`.

#### Phase 1: Per-polynomial $\alpha$-projection challenges

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

**What this produces:** A `BatchedZipPlusTestTranscript` wrapping a `PcsTranscript` that contains both the serialized proof bytes and the Fiat-Shamir sponge state. This transcript is passed directly to the evaluate phase (Step 5), which continues squeezing challenges from the same sponge — binding the evaluation-phase randomness to the test-phase data.

### Step 5: PCS Evaluate (Batched)

```
(evals_f, proof) = BatchedZipPlus::evaluate(params, polys, point, test_transcript)
```

The PCS evaluation phase computes the claimed evaluation of each committed polynomial at a shared point, and produces a batched evaluation proof. The key idea is that instead of writing one full `row_len`-sized combined row per polynomial (which would cost $O(m \times \text{row\_len})$ proof data), a single batched row is written, reducing the proof data to $O(\text{row\_len} + m)$.

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

For the 20-column SHA-256 batch: $20 \times 512 = 10{,}240$ elements → $512 + 20 = 532$ elements (**19× reduction**). At 32 bytes per `MontyField<4>` element: ~320 KB → ~17 KB.

#### Consistency relation (verified by the verifier)

The verifier checks:
$$\langle \hat{r}, q_1 \rangle = \sum_{i=0}^{m-1} \beta^i \cdot v_i$$
This holds because $\langle \hat{r}, q_1 \rangle = \sum_i \beta^i \langle r_i, q_1 \rangle = \sum_i \beta^i v_i$, by linearity of inner products.

**Important note on the evaluation point:** The PCS evaluation point is derived by hashing the CPR's evaluation point into `i128` values via `derive_pcs_point()`. Both prover and verifier call this function to agree on the same deterministic point. This is NOT the same point as the CPR evaluation point (which lives in $\mathbb{F}_q$). See §14 for why this matters.

**Important note on field context:** The PCS samples its own field configuration from its internal Fiat-Shamir transcript via `get_random_field_cfg()`. The evaluation point is passed as raw integers (`&[Zt::Pt]`) and converted to field elements using this PCS-local field config. This field config is independent of the PIOP's field config.

**What this produces:** A `BatchedZipPlusProof` (serialized into `ZincProof.pcs_proof_bytes`) containing the test-phase data, the batched row, and the per-polynomial scalars.

---

## 4. The Verifier Pipeline in Detail

The verifier is implemented as `verify()` / `verify_generic()` / `verify_dual_ring()` in `pipeline.rs`. It reconstructs the Fiat-Shamir transcript and checks each component:

### Step 1: Ideal Check Verification

1. Reconstruct the same random challenges ($\alpha$, evaluation point) from a fresh transcript.
2. Deserialize the IC proof (the combined MLE evaluations).
3. Absorb the proof values into the transcript (maintaining Fiat-Shamir consistency).
4. **Ideal membership check:** For each constraint, check that its evaluated combined polynomial belongs to the correct ideal:
   - For `CyclotomicIdeal` $(X^{32} - 1)$: check that the polynomial reduces to zero modulo $X^{32} - 1$.
   - For `DegreeOneIdeal(2)` $(X - 2)$: evaluate the polynomial at $X = 2$ in $\mathbb{F}_p$ and check the result is zero.
   - For `TrivialIdeal` (F₂[X] constraints): always passes. See §11 for why.
   - For `BitPolyIdeal`: always passes. See §11 for why.
5. **Output:** A `VerifierSubClaim` containing the evaluation point and projected scalars, which is passed to the CPR verifier.

### Step 2: CPR Verification (Sumcheck)

1. Deserialize the CPR proof (sumcheck messages, claimed sum, up/down evaluations).
2. Verify the sumcheck protocol: replay each round, checking that the prover's univariate polynomial is consistent with the running claim.
3. At the final round, verify that the sumcheck's terminal claim is consistent with the IC's evaluation claims and the column evaluations (`up_evals`, `down_evals`).
4. **Output:** A `VerifierSubclaim` containing the evaluation point where the trace columns should be opened.

### Step 3: PCS Verification (Batched)

The PCS verification consists of two sub-phases: **testing** (proximity) and **evaluation** (opening). Both have been heavily optimized (see §13 for details). All notation below references the symbols defined in §3 Steps 1, 4, and 5.

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

4. **Spot-check encoding:** Collect the 147 opened positions $\{j_0, \ldots, j_{146}\}$ and call `encode_wide_at_positions(&combined_row, &positions)`. This computes the IPRS linear-code encoding of the combined row **only** at the 147 opened positions (not the full `cw_len`-element codeword), using the precomputed encoding matrix (see §13). Result: 147 `CombR` values.

5. **Per-column proximity check** (`verify_batched_column_testing`): For each opened column $j_t$:
   - For each polynomial $i$: alpha-project the `Cw`-column entries into `CombR` integers via inner product with $(\alpha_i^{(0)}, \ldots, \alpha_i^{(D)})$, then combine across rows via inner product with $(c_0, \ldots, c_{\text{num\_rows}-1})$. This produces one `CombR` value per polynomial.
   - Sum all $m$ per-polynomial `CombR` values into a total.
   - Check: `total == encode_wide_at_positions(combined_row)[t]`.

6. **Merkle proof verification:** For each opened column $j_t$, concatenate all per-polynomial column values (the same data written during the commit phase to form the leaf) and verify the Merkle proof against the commitment root.

#### Evaluation Phase (`verify_evaluation`)

The evaluation phase checks that the prover's claimed polynomial evaluations $v_0, \ldots, v_{m-1}$ and batched row $\hat{r}$ are consistent with the committed codeword and the evaluation point $\mathbf{p}$.

1. **Tensor decomposition:** Compute $(q_0, q_1) = \text{point\_to\_tensor}(\text{num\_rows}, \mathbf{p}_F, \text{field\_cfg})$, exactly as in §3 Step 5.

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

The SHA-256 witness is a 64-row × 20-column matrix. Each cell is a `BinaryPoly<32>` — a polynomial over $\mathbb{F}_2$ of degree ≤ 31, representing a 32-bit word as $w = \sum_{i=0}^{31} w_i X^i$ where $w_i \in \{0, 1\}$.

| Col | Name | Description | Split Batch |
|-----|------|-------------|-------------|
| 0 | `a_hat` | Working variable $a$ | BinaryPoly |
| 1 | `e_hat` | Working variable $e$ | BinaryPoly |
| 2 | `W_hat` | Message schedule word $W_t$ | BinaryPoly |
| 3 | `Sigma0_hat` | $\Sigma_0(a)$ = ROTR²(a) ⊕ ROTR¹³(a) ⊕ ROTR²²(a) | BinaryPoly |
| 4 | `Sigma1_hat` | $\Sigma_1(e)$ = ROTR⁶(e) ⊕ ROTR¹¹(e) ⊕ ROTR²⁵(e) | BinaryPoly |
| 5 | `Maj_hat` | Maj(a,b,c) | Int |
| 6 | `ch_ef_hat` | $e \wedge f$ | Int |
| 7 | `ch_neg_eg_hat` | $\neg e \wedge g$ | Int |
| 8 | `sigma0_w_hat` | $\sigma_0(W_{t-15})$ | BinaryPoly |
| 9 | `sigma1_w_hat` | $\sigma_1(W_{t-2})$ | BinaryPoly |
| 10 | `d_hat` | Working variable $d$ | Int |
| 11 | `h_hat` | Working variable $h$ | Int |
| 12 | `mu_a` | Carry polynomial for $a$ update | Int |
| 13 | `mu_e` | Carry polynomial for $e$ update | Int |
| 14 | `mu_W` | Carry polynomial for $W$ update (always zero — deferred) | Int |
| 15 | `S0` | Shift quotient for $\sigma_0$ | BinaryPoly |
| 16 | `S1` | Shift quotient for $\sigma_1$ | BinaryPoly |
| 17 | `R0` | Shift remainder for $\sigma_0$ | BinaryPoly |
| 18 | `R1` | Shift remainder for $\sigma_1$ | BinaryPoly |
| 19 | `K_t` | SHA-256 round constant | Int |

The "Split Batch" column indicates which PCS batch the column belongs to in the split-trace architecture (see §8).

### Implemented Constraints (11 of 14)

#### F₂[X] constraints (6) — enforced via `Uair<BinaryPoly<32>>`

These operate in the polynomial ring $\mathbb{F}_2[X]/(X^{32} - 1)$. Addition in this ring IS XOR, which is the natural operation for SHA-256 bitwise operations.

| # | Name | Expression | Ideal |
|---|------|-----------|-------|
| C1 | Σ₀ rotation | $\hat{a} \cdot \rho_0 - \hat{\Sigma}_0 \in (X^{32} - 1)$ | Cyclotomic |
| C2 | Σ₁ rotation | $\hat{e} \cdot \rho_1 - \hat{\Sigma}_1 \in (X^{32} - 1)$ | Cyclotomic |
| C3 | σ₀ rotation + shift | $\hat{W} \cdot \rho_{\sigma_0} + S_0 - \widehat{\sigma_0 w} \in (X^{32} - 1)$ | Cyclotomic |
| C4 | σ₁ rotation + shift | $\hat{W} \cdot \rho_{\sigma_1} + S_1 - \widehat{\sigma_1 w} \in (X^{32} - 1)$ | Cyclotomic |
| C5 | σ₀ shift decomposition | $\hat{W} - R_0 - S_0 \cdot X^3 = 0$ | Zero (exact) |
| C6 | σ₁ shift decomposition | $\hat{W} - R_1 - S_1 \cdot X^{10} = 0$ | Zero (exact) |

The rotation polynomials encode ROTR via multiplication by powers of $X$ modulo $(X^{32} - 1)$:
- $\rho_0 = X^{30} + X^{19} + X^{10}$ encodes ROTR(2, 13, 22) for $\Sigma_0$
- $\rho_1 = X^{26} + X^{21} + X^{7}$ encodes ROTR(6, 11, 25) for $\Sigma_1$
- $\rho_{\sigma_0} = X^{25} + X^{14}$ encodes ROTR(7, 18) for $\sigma_0$
- $\rho_{\sigma_1} = X^{15} + X^{13}$ encodes ROTR(17, 19) for $\sigma_1$

#### Q[X] constraints (5) — enforced via `Uair<DensePolynomial<i64, 64>>`

These operate in $\mathbb{Z}[X]$ (integer coefficient polynomials, degree < 64). They require viewing the same trace as integer polynomials rather than binary polynomials, because the constant 2 is zero in $\mathbb{F}_2$.

| # | Name | Expression | Ideal |
|---|------|-----------|-------|
| C7 | Ch BitPoly | `ch_ef_hat ∈ BitPoly` | BitPoly |
| C8 | ¬e∧g BitPoly | `ch_neg_eg_hat ∈ BitPoly` | BitPoly |
| C9 | Maj BitPoly | `Maj_hat ∈ BitPoly` | BitPoly |
| C10 | a-update carry | $\hat{a}[t{+}1] - \hat{h} - \hat{\Sigma}_1 - \widehat{ch_{ef}} - \widehat{ch_{neg}} - \hat{K}_t - \hat{W} - \hat{\Sigma}_0 - \hat{Maj} + \mu_a \cdot X^{32} \in (X - 2)$ | DegreeOne(2) |
| C11 | e-update carry | $\hat{e}[t{+}1] - \hat{d} - \hat{h} - \hat{\Sigma}_1 - \widehat{ch_{ef}} - \widehat{ch_{neg}} - \hat{K}_t - \hat{W} + \mu_e \cdot X^{32} \in (X - 2)$ | DegreeOne(2) |

The carry constraints use the $(X - 2)$ ideal: a polynomial $p(X) \in (X - 2)$ iff $p(2) = 0$. Since evaluating a binary-coefficient polynomial at $X = 2$ gives the integer value of the 32-bit word ($\sum c_i \cdot 2^i$), this ideal check verifies that the modular addition with carry is correct as an integer equation.

### Not-implemented SHA-256 constraints (3 of 14)

| # | Name | Why deferred |
|---|------|-------------|
| C12 | d-delay ($d \leftarrow a_{t-3}$) | Requires 3-row lookback; the UAIR only provides `up` (current) and `down` (next row) |
| C13 | h-delay ($h \leftarrow e_{t-3}$) | Same: 3-row lookback needed |
| C14 | Message schedule W update | Requires lookbacks of 2, 7, 15, and 16 rows |

These constraints are algebraically specified in comments but cannot be expressed in the current `up`/`down` framework. Implementing them would require either intermediate relay columns ($b, c, f, g$ as separate columns with 1-step delay constraints) or extending the UAIR trait to support multi-row access.

---

## 6. ECDSA Arithmetization

### Trace Layout

The ECDSA witness is a **258-row × 9-column** matrix. The trace is unified under `Int<4>` (256-bit integer) for all contexts — PCS commitment, PIOP constraint checking, and the end-to-end pipeline:

| Context | Evaluation type | Purpose |
|---------|----------------|---------|
| **Unified pipeline** (PCS + IC + CPR) | `Int<4>` (256-bit integer) | All 7 constraints are expressed directly in `Int<4>` via `Uair<Int<4>>`. The same ring is used for PCS commitment, IdealCheck, and CombinedPolyResolver — no ring conversion needed. This eliminates the prior dual-ring architecture. |
| **Legacy constraint checking** | `DensePolynomial<i64, 1>` | Degree-0 polynomials (plain i64 scalars). Retained for the original constraint formulation; the `Int<4>` version is algebraically identical. |

The scalars $u_1 = e \cdot s^{-1} \bmod n$ and $u_2 = r \cdot s^{-1} \bmod n$ are given to the verifier in the clear (they are public inputs derived from the message hash and signature). The verifier can independently compute the corresponding bit decomposition columns $b_1, b_2$ and verify that the scalar accumulation is correct without dedicated trace columns or constraints. The quotient bit $k$ (for the final signature check $R_x \equiv r \pmod{n}$) is handled in boundary constraints, not as a trace column.

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
- The BinaryPoly<32> UAIR implementation has 0 constraints — it's a placeholder because F₂[X] cannot express F_p arithmetic.
- For PCS benchmarking, the ECDSA trace uses `Int<4>` (256-bit integer) evaluations instead of `BinaryPoly<32>`, since ECDSA values are scalars, not polynomials. This is ~32× cheaper per cell in the IPRS NTT (1 coefficient vs. 32), and is the configuration used in the paper's target benchmark.

---

## 7. The Dual-Ring Pipeline

SHA-256 constraints live in two different rings: F₂[X] (for bitwise/rotation operations) and Q[X] (for integer carry propagation and AND/Maj checks). The **dual-ring pipeline** (`prove_dual_ring` / `verify_dual_ring`) handles this by running two sequential PIOP passes on the same committed trace:

### Prover: `prove_dual_ring()`

1. **PCS Commit** the `BinaryPoly<32>` trace (once).
2. **IC₁ + CPR₁** over `BinaryPoly<32>` ring (constraints C1–C6).
3. **Convert trace** from `BinaryPoly<32>` to `DensePolynomial<i64, 64>` via `convert_trace_to_qx()` — each binary polynomial's bit coefficients are widened to i64.
4. **IC₂ + CPR₂** over `DensePolynomial<i64, 64>` ring (constraints C7–C11).
5. **PCS Test + Evaluate**.

Both IC+CPR passes share the same Fiat-Shamir transcript, so the random challenges are sequentially derived and verifiably deterministic.

### Verifier: `verify_dual_ring()`

1. **IC₁ verify (BinaryPoly)** — uses `TrivialIdeal` (always passes; see §11).
2. **CPR₁ verify** — sumcheck verification.
3. **IC₂ verify (Q[X])** — uses real ideals: `BitPolyIdeal` (always passes) for C7–C9, `DegreeOneIdeal(2)` (evaluates at 2, checks zero) for C10–C11.
4. **CPR₂ verify** — sumcheck verification.
5. **PCS verify**.

---

## 8. The Split-Trace Architecture

### Motivation

The 20 SHA-256 trace columns use two different evaluation types: F₂[X] rotation/shift constraints (C1–C6) need `BinaryPoly<32>` evaluations, while Q[X] carry/BitPoly constraints (C7–C11) only need integer evaluations. By splitting the trace into two PCS batches, the 9 integer-only columns can be committed as `Int<1>` (64-bit) instead of `BinaryPoly<32>` (32 coefficients × 8 bytes = 256 bytes per codeword element). This shrinks the column-opening data for these columns from 256 bytes to 16 bytes per element — a **16× reduction per column**.

### Column Classification

The split is defined in `sha256-uair/src/witness.rs`:

**BinaryPoly batch (11 columns)** — `POLY_COLUMN_INDICES = [0, 1, 2, 3, 4, 8, 9, 15, 16, 17, 18]`:
- These participate in F₂[X] rotation/shift constraints (C1–C6).
- Committed as `BinaryPoly<32>` using `Sha256ZipTypes<i64, 32>`.
- Codeword element: `DensePolynomial<i64, 32>` = 256 bytes.
- CombR: `Int<6>` = 48 bytes (384-bit).
- Columns: a\_hat, e\_hat, W\_hat, Σ0\_hat, Σ1\_hat, σ0\_w\_hat, σ1\_w\_hat, S0, S1, R0, R1.

**Int batch (9 columns)** — `INT_COLUMN_INDICES = [5, 6, 7, 10, 11, 12, 13, 14, 19]`:
- These only participate in Q[X] carry/BitPoly constraints (C7–C11).
- Committed as `Int<1>` (64-bit integer) using `Sha256IntZipTypes`.
- Codeword element: `Int<2>` = 16 bytes (128-bit).
- CombR: `Int<4>` = 32 bytes (256-bit).
- Columns: Maj\_hat, ch\_ef\_hat, ch\_neg\_eg\_hat, d\_hat, h\_hat, μ\_a, μ\_e, μ\_W, K\_t.

### Witness Generators

Two functions in `sha256-uair/src/witness.rs`:

- **`generate_poly_witness(num_vars, rng)`** → `Vec<DenseMultilinearExtension<BinaryPoly<32>>>`: Generates the full 20-column trace, then extracts only the 11 `POLY_COLUMN_INDICES` columns.
- **`generate_int_witness(num_vars, rng)`** → `Vec<DenseMultilinearExtension<Int<1>>>`: Generates the full trace, then for each of the 9 `INT_COLUMN_INDICES`, converts each `BinaryPoly<32>` cell to `Int<1>` via `bp.to_u64() as i64`. All values are 32-bit unsigned integers that fit in a single u64 limb.

### Proof Size Impact

For 1×SHA-256 (1 compression, 128-row trace with `num_vars = 7`):

| Configuration | PCS Raw | Compressed |
|--------------|---------|------------|
| Monolithic (20 cols as BinaryPoly) | 790 KB | 374 KB |
| Split (11 BPoly + 9 Int) | 533 KB | 259 KB |
| **Reduction** | **1.48×** | **1.44×** |

For 8×SHA-256 + ECDSA (512-row trace with `num_vars = 9`):

| Configuration | PCS Raw | Compressed |
|--------------|---------|------------|
| Monolithic SHA + ECDSA | 1,045 KB | 571 KB |
| Split SHA + ECDSA | 821 KB | 460 KB |
| **Reduction** | **1.27×** | **1.24×** |

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
| `pcs_proof_bytes` | Serialized PCS transcript (Merkle proofs, queried rows, batched eval row, per-poly scalars) | ~350 KB for 20 cols, DEPTH=1 |
| `commitment` | Merkle root + batch size | Small |
| `ic_proof_values` | Combined MLE evaluations (one `DynamicPolynomialF` per constraint) | Tens of bytes per constraint |
| `cpr_sumcheck_messages` | Sumcheck round polynomials | $O(\text{num\_vars} \times \text{degree})$ field elements |
| `cpr_sumcheck_claimed_sum` | Claimed sum of the batched constraint expression | 1 field element |
| `cpr_up_evals` | Column evaluations at sumcheck point | 1 field element per column |
| `cpr_down_evals` | Column evaluations at shifted point | 1 field element per column |
| `evaluation_point_bytes` | The sumcheck evaluation point | `num_vars` field elements |
| `pcs_evals_bytes` | PCS evaluation values | 1 field element per column |

### `DualRingZincProof`

Same structure but with two sets of IC + CPR fields: `bp_*` (BinaryPoly pass) and `qx_*` (Q[X] pass).

### PCS Proof Breakdown

The PCS proof transcript is the dominant contributor to proof size. Its structure:

**Test phase:**
- 1 combined row (`row_len × CombR::NUM_BYTES`): e.g., 512 × 48 = 24,576 bytes (BinaryPoly batch) or 512 × 64 = 32,768 bytes (ECDSA batch).
- `NUM_COLUMN_OPENINGS` (147) column openings, each containing:
  - Column index (8 bytes).
  - Per-polynomial column values: `batch_size × num_rows × Cw::NUM_BYTES` per opening. For SHA BinaryPoly: 11 × 1 × 256 = 2,816 bytes; for SHA Int: 9 × 1 × 16 = 144 bytes; for ECDSA: 9 × 1 × 40 = 360 bytes.
  - Merkle proof: ~11 blake3 hashes (~352 bytes per proof for tree depth ≈ log₂(2048) = 11).

**Eval phase (batched):**
- 1 batched row (`row_len × F::NUM_BYTES`): 512 × 32 = 16,384 bytes (MontyField<4>) or 512 × 64 = 32,768 bytes (MontyField<8>).
- `batch_size` per-polynomial evaluation scalars: `batch_size × F::NUM_BYTES`.

### Field Serialization

All `PiopField` elements are serialized in Montgomery representation (32 bytes each for the 128-bit field). This ensures Fiat-Shamir consistency: the verifier can reconstruct the exact same transcript bytes.

---

## 10. Fiat-Shamir Transcript

The system uses a Keccak-based Fiat-Shamir transcript (`KeccakTranscript`). The prover and verifier construct identical transcripts by:

1. Initializing a fresh `KeccakTranscript::new()`.
2. Sampling a random prime field configuration: `transcript.get_random_field_cfg()`. This produces a ~128-bit prime $p$ used for all PIOP arithmetic.
3. Sequentially sampling challenges and absorbing proof data in the same order on both sides.

The field is a Montgomery-form modular integer with 4×64-bit limbs, using a fixed 128-bit prime `0x860995AE68FC80E1B1BD1E39D54B33`. The Miller-Rabin primality test confirms its primality.

---

## 11. Ideal Types and Membership Checks

The system defines seven ideal types, each handling a different kind of constraint:

### `CyclotomicIdeal` — $(X^{32} - 1)$

- **Used by:** C1–C4 (SHA-256 rotation constraints)
- **Prover-side check (`BinaryPoly<32>`):** Only zero is in the ideal for degree ≤ 31 polynomials. But constraint expressions can have degree up to 61 (product of two degree-31 polys), so the check is that the expression reduces to zero modulo $X^{32} - 1$.
- **Verifier-side check (`DynamicPolynomialF`):** Reduces the polynomial modulo $X^{32} - 1$ by folding coefficients: $g_{\text{red},j} = \sum_k g_{j + 32k}$. Checks all 32 reduced coefficients equal zero.

### `BitPolyIdeal`

- **Used by:** C7–C9 (Ch/Maj binary coefficient checks)
- **Prover-side check (`DensePolynomial<i64, 64>`):** All coefficients in $\{0, 1\}$.
- **Verifier-side check (`DynamicPolynomialF`):** **Always returns true.** This is documented and intentional: the BitPoly property is NOT a mathematical ideal (not closed under linear combination). After the IC protocol's MLE evaluation at a random point, the combined value has MLE-interpolated coefficients that are not binary even if each individual row's coefficients are. Soundness for BitPoly constraints relies on the sumcheck + PCS binding, not the ideal membership check.

### `DegreeOneIdeal(2)` — $(X - 2)$

- **Used by:** C10–C11 (SHA-256 carry propagation)
- **Prover-side check:** Evaluates $p(2) = \sum c_i \cdot 2^i$ and checks it equals zero.
- **Verifier-side check (`DynamicPolynomialF`):** Evaluates the polynomial at $X = 2$ in $\mathbb{F}_p$ and checks the result is zero. **This IS a real ideal that lifts correctly:** $(X - 2)$ is an ideal in both $\mathbb{Z}[X]$ and $\mathbb{F}_p[X]$ (for $p > 2$), and the evaluation-at-2 check is preserved under the projection from $\mathbb{Z}[X]$ to $\mathbb{F}_p[X]$.

### `TrivialIdeal`

- **Used by:** BinaryPoly IC pass in the verifier
- **Check:** Always returns `true`.
- **Rationale:** F₂[X] constraints (where addition = XOR) do NOT lift to $\mathbb{F}_p$ arithmetic (where $1 + 1 = 2 \neq 0$). The rotation constraint $\hat{a} \cdot \rho_0 - \hat{\Sigma}_0$ is zero in $\mathbb{F}_2[X]/(X^{32}-1)$ but not in $\mathbb{F}_p[X]/(X^{32}-1)$ after projection. Soundness is preserved because the sumcheck and PCS verify the algebraic consistency of the projected trace — if any witness row violates the constraint, the random-evaluation-based sumcheck catches it with high probability.

### `EcdsaIdealOverF`

- **Used by:** ECDSA pipeline verification
- **Check:** Checks if value is zero (matching the `assert_zero` semantics of all 7 ECDSA constraints).

### `ImpossibleIdeal`

- **Used by:** ECDSA UAIR at the ring level
- **Never actually checked** — it's the ideal type for `assert_zero` constraints, where the constraint builder calls `assert_zero` instead of `assert_in_ideal`.

### `Sha256QxIdeal` / `Sha256QxIdealOverF`

- **Used by:** Q[X] IC pass
- **Enum wrapping** `BitPolyIdeal` and `DegreeOneIdeal(2)`, mapping to field-level versions via a closure in the verifier.

---

## 12. Witness Generation

### SHA-256 (`sha256-uair/src/witness.rs`)

The witness generator:

1. Runs the full SHA-256 compression function on the empty-string message (single 512-bit padded block).
2. At each of the 64 rounds, records all 20 column values:
   - Working variables $a, e, d, h$ directly from the SHA-256 state.
   - Derived values $\Sigma_0, \Sigma_1, \text{Maj}, \text{ch\_ef}, \text{ch\_neg\_eg}$ computed from the state.
   - Message schedule entries $W_t$ and small-sigma values $\sigma_0, \sigma_1$ (for $t \geq 16$).
   - Carry polynomials $\mu_a, \mu_e$ computed as `floor(sum / 2^32)` from the non-wrapping integer sum.
   - Shift decompositions: $S_0 = W_{t-15} \gg 3$, $R_0 = W_{t-15} \& \text{0x7}$, etc.
   - Round constants $K_t$ from the SHA-256 specification.
3. Sets a boundary row at index 64 with the final $a, e$ values (so that carry constraints at row 63 can reference `down[COL_A_HAT]` and `down[COL_E_HAT]` correctly).
4. Zero-pads remaining rows to $2^{\text{num\_vars}}$.

The witness is verified against NIST test vectors (the hash of the empty string matches `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`).

### SHA-256 Split Witnesses

Two additional generators extract subsets of the full trace:
- `generate_poly_witness()` → 11-column `BinaryPoly<32>` trace for the F₂[X] batch.
- `generate_int_witness()` → 9-column `Int<1>` trace for the Q[X] batch. Each `BinaryPoly<32>` value is converted to its integer representation via `bp.to_u64() as i64`.

### ECDSA (`ecdsa-uair/src/witness.rs`)

The witness generator creates a constant fixed-point trace:
- All 258 rows have $X = 1, Y = 1, Z = 0$ (the point at infinity in Jacobian coordinates).
- Scalar bits $b_1, b_2 = 0$ (no additions performed).
- This satisfies the doubling constraints (everything is zero) but does NOT represent a real ECDSA verification. It is used for testing that the constraint system is self-consistent.
- Two witness generators are available: one producing `DensePolynomial<i64, 1>` traces, another producing `Int<4>` traces.

---

## 13. PCS Verifier Optimizations

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

## 14. What the Implementation Does NOT Do

### 14.1 CPR→PCS Binding Gap

**The most significant architectural gap.** The CPR produces evaluation claims: "column $i$ of the committed trace evaluates to `up_evals[i]` at the sumcheck point $\mathbf{s}$, and to `down_evals[i]` at the shifted point." The PCS independently verifies that the committed polynomials pass proximity testing and evaluation consistency at a *different* point (derived by hashing the CPR point into `i128` values via `derive_pcs_point()`).

**What's missing:** There is no check that the PCS-evaluated values are consistent with the CPR's claimed `up_evals` or `down_evals`. Both vectors are prover-supplied values that the CPR verifier plugs into the constraint expression to check algebraic consistency with the sumcheck terminal claim — but neither is verified against the PCS commitment. The CPR operates on projected $\mathbb{F}_q$ values at the sumcheck point; the PCS evaluates committed `BinaryPoly<32>` polynomials at an `i128` point in a different coordinate system.

Specifically for `down_evals`: the "down" (shifted) trace is never separately committed. The prover constructs it by dropping the first row and appending a zero row (`column[1..].iter().cloned().collect()`), but only the original unshifted trace is committed via the PCS. The `down_evals` are absorbed into the Fiat-Shamir transcript (maintaining transcript consistency) but are otherwise taken on faith.

**Impact:** The IC, CPR, and PCS each independently verify their respective claims, but nothing cryptographically binds the CPR's column-evaluation claims (`up_evals` and `down_evals`) to the PCS's committed polynomials. A malicious prover could (in theory) commit one trace via PCS, run the PIOP on a different trace, and both would pass independently.

**What would fix it:** Either (a) the PCS must evaluate at the CPR's sumcheck point $\mathbf{s}$ and the verifier checks `up_evals[i] == pcs_opened_value[i]` (and derives `down_evals` from the committed polynomials evaluated at the appropriately shifted point), or (b) a secondary sumcheck reduces the CPR claims to a single evaluation claim that the PCS can verify directly.

This is documented in `pipeline.rs` with a detailed comment block.

### 14.2 No Actual Sumcheck Invocation at the Top Level

The PIOP protocol chain described in the paper is: **Ideal Check → CPR → Sumcheck → PCS**. In the implementation, the sumcheck is embedded *inside* the CPR (as `MLSumcheck::prove` / `MLSumcheck::verify`), not invoked as a separate top-level protocol step. The CPR internally runs a sumcheck to reduce its batched constraint sum to a single-point claim. There is no separate, standalone sumcheck phase between the CPR and PCS.

### 14.3 Three SHA-256 Constraints Missing

Constraints C12 (d-delay), C13 (h-delay), and C14 (message schedule W update) require multi-row lookback. The current UAIR trait only provides `up` (current row) and `down` (next row). The missing constraints don't materially affect prover cost (same algebraic degree as implemented constraints) but mean the system does not fully enforce SHA-256 correctness.

Specifically: without C12–C14, a prover could supply a trace where:
- The register relay ($d_{t+1} = a_{t-3}$) is wrong.
- The message schedule ($W_t$ for $t \geq 16$) is wrong.
- And the proof would still pass.

### 14.4 ECDSA Not Over Real secp256k1 Field

The ECDSA constraints operate on `Int<4>` (256-bit integers) in the unified pipeline and `DensePolynomial<i64, 1>` (64-bit integers) in the legacy formulation, both using a toy curve over $\mathbb{F}_{101}$, not the real secp256k1 base field (256-bit prime). The constraint *algebra* is correct (it correctly describes Jacobian-coordinate point doubling and addition with Shamir's trick), but:
- The witness values are from $\mathbb{F}_{101}$, not $\mathbb{F}_p$.
- The `Int<4>` pipeline uses `prove_generic`/`verify_generic` with `EcdsaScalarZipTypes` (`Eval = Int<4>`, `CombR = Int<8>`, `PcsF = MontyField<8>`), avoiding the dual-ring architecture entirely.

### 14.5 ECDSA Boundary Constraints Not Enforced

The 7 ECDSA constraints are "non-boundary" constraints applied uniformly to all rows. The implementation does not enforce:
- Row 1 (precomputation): that the initial accumulator is the identity element.
- Row 258 (final check): that the final accumulated point matches the expected signature verification result.

### 14.6 No Compact Proof Serialization

Field elements are serialized at full width (8 bytes per i64 coefficient, 32 bytes per PiopField element in Montgomery form). For DEPTH=1 IPRS coefficients that fit in ~45 bits, this wastes ~23 bits per coefficient. Deflate compression compensates (~1.8–2.1× ratio for DEPTH=1) but not for DEPTH≥2. The eval-phase batching dramatically reduced the eval-phase contribution, so the remaining proof size is dominated by test-phase Merkle openings and column data.

### 14.7 PCS Field Is Not 256-bit

The PIOP field is 128-bit (~120-bit prime). This provides 128-bit soundness for the Schwartz-Zippel lemma (random evaluation). However, a production system over secp256k1 would likely want a 256-bit field to match the curve's security level.

### 14.8 No Zero-Knowledge

The current implementation provides no blinding or zero-knowledge. The PCS proof reveals information about the trace (Merkle openings contain actual committed polynomial evaluations). Adding ZK would require blinding the trace with random polynomials before commitment.

### 14.9 BitPoly Ideal Not Sound at the IC Level

As discussed in §11, the BitPoly ideal membership check always returns `true` in the verifier. This means C7–C9 (Ch/Maj binary checks) are verified only through sumcheck + PCS, not through ideal membership testing. This is theoretically sound (the combined polynomial resolver's algebraic check catches constraint violations at random evaluation points) but represents weakened checking compared to "real" ideals like $(X - 2)$ or $(X^{32} - 1)$.

### 14.10 Overflow Checking Disabled

The PCS is instantiated with `CHECK = false` (the `UNCHECKED` constant). This skips runtime overflow checks on polynomial coefficient arithmetic during IPRS encoding and evaluation. In principle, if coefficients grow too large, integer overflow could corrupt the proof. In practice, the coefficient bounds are analytically verified to be safe for the configured parameters.

### 14.11 Split-Trace PIOP Not Integrated

The split-trace architecture (§8) currently only operates at the **PCS level** — the two SHA-256 batches are committed and verified independently. The split-trace benchmarks measure PCS-only commit/verify/proof-size, but do **not** run the PIOP (Ideal Check + CPR) on split traces. The full pipeline still uses the monolithic 20-column trace for the PIOP and only supports the non-split PCS.

To integrate the split-trace architecture into the full PIOP pipeline, the IC and CPR would need to be run separately on each batch (BinaryPoly constraints on the 11-column batch, Q[X] constraints on the 9-column batch) with appropriate column index remapping.

### 14.12 Parallel Verification Not in Production Pipeline

The parallel PCS verification via `rayon::join` (§13, Phase 2) is only implemented in the benchmark harness. The `verify()`, `verify_generic()`, and `verify_dual_ring()` functions in `pipeline.rs` run PCS verification sequentially. Integrating parallelism would require the pipeline to manage multiple PCS contexts.

---

## 15. Optimization Opportunities

### 15.1 Compact Codeword Serialization (Proof Size)

**Estimated savings: ~100 KB for 8×SHA-256 BinaryPoly batch.**

Each `DensePolynomial<i64, 32>` codeword element (used for BinaryPoly columns) serializes as 32 × 8 = 256 bytes. However, for DEPTH=1 IPRS codes, the coefficient bitbound is ~45 bits, so each coefficient fits in 6 bytes. Serializing at 6 bytes per coefficient would give 32 × 6 = 192 bytes per codeword element — a 25% reduction on column-opening data.

After eval-phase batching, column openings dominate the test-phase proof. For 147 openings × 11 BinaryPoly columns:
- Current: 147 × 11 × 256 ≈ 414 KB
- Compact: 147 × 11 × 192 ≈ 310 KB
- **Savings: ~104 KB**

### 15.2 Narrowing the ECDSA Field (Proof Size + Verifier Speed)

**Estimated savings: ~16 KB on eval phase, plus ~2× faster ECDSA field arithmetic.**

Currently ECDSA uses `CombR = Int<8>` (512-bit) which forces `PcsF = MontyField<8>` (64 bytes/element). The actual overflow bound for 9 columns × 1 row × 256-bit evaluations with 128-bit challenges is ~384 bits, fitting in `Int<6>` (384-bit). Narrowing to `CombR = Int<6>` and `PcsF = MontyField<4>` (32 bytes) would:
- Halve the ECDSA eval-phase batched row (512 × 32 = 16 KB vs 512 × 64 = 32 KB).
- Speed up ECDSA field arithmetic by ~2× (128-bit Montgomery multiplication vs 512-bit).
- Reduce the test-phase combined row (512 × 48 vs 512 × 64).

### 15.3 Batch Merkle Verification (Verifier Speed)

**Estimated savings: ~10–20% of Merkle verification time.**

Currently each of the 147 column openings independently hashes the leaf and walks up the Merkle tree (~11 hash levels). Many openings share Merkle path prefixes. Batching the verification to share work on common path segments could save duplicate hash computations.

### 15.4 SIMD-Accelerated Inner Products (Verifier Speed)

The inner-product kernels (`MBSInnerProduct`, `ScalarProduct`, `CombDotChal`) may not fully exploit SIMD. For `MontyField<4>` (256-bit), vectorized Montgomery multiplication using NEON (Apple M-series) or AVX2 intrinsics could accelerate both column testing and evaluation verification.

### 15.5 Reducing NUM_COLUMN_OPENINGS (Proof Size + Verifier Speed)

`NUM_COLUMN_OPENINGS = 147` is a security parameter. Reducing it would proportionally decrease both proof size and verifier time. A formal analysis of the soundness trade-off (proximity testing false-accept probability vs. number of openings) could determine the minimum safe value for the target security level.

### 15.6 Full Split-Trace Pipeline Integration

Integrating the split-trace architecture into the full PIOP pipeline (not just PCS) would require:
1. Running IC₁ + CPR₁ on the 11-column BinaryPoly batch (F₂[X] constraints C1–C6).
2. Running IC₂ + CPR₂ on the 9-column Int batch (Q[X] constraints C7–C11).
3. Column index remapping between the UAIR constraint expressions (which reference the original 20-column layout) and the split batches.
4. Separate PCS test + evaluate for each batch.

This would yield the split-trace proof size benefits in the full end-to-end pipeline.

### 15.7 Multi-Row UAIR Access

Extending the UAIR trait beyond the current `up`/`down` (1-step lookback) framework to support arbitrary row offsets would enable implementing the missing SHA-256 constraints C12–C14. This could be done via:
- **Relay columns:** Adding intermediate columns $b, c, f, g$ with 1-step delay constraints ($b_{t+1} = a_t$, etc.), increasing the trace width but staying within the `up`/`down` model.
- **Multi-row access trait extension:** Adding `row_at(offset)` to the constraint builder, allowing constraints to reference `row[-3]`, `row[-15]`, etc. directly.

### 15.8 Verifier Target: Sub-5ms

The combined PCS verifier currently runs at ~6.8 ms (parallel, MacBook Air M4). To reach the <5 ms target:
- Narrowing ECDSA to `MontyField<4>` would speed up the ECDSA batch (faster field ops, smaller data).
- Batch Merkle verification could save ~0.5 ms.
- SIMD inner products could save ~0.5 ms.
- Together these could bring the combined time to ~4–5 ms.

### 15.9 Production Pipeline Parallelism

Moving the parallel `rayon::join` verification from the benchmark into the production `verify()` / `verify_generic()` functions. This requires the pipeline to support a multi-PCS-batch model where multiple commitments are verified concurrently.

---

## 16. Test Coverage

Seven integration tests in `snark/tests/`:

| Test | What it exercises |
|------|------------------|
| `round_trip_pcs_sha256` | PCS-only: commit → test → evaluate → verify. Proves the PCS works in isolation. |
| `ideal_check_succeeds_on_valid_sha256_witness` | IC prover on real SHA-256 trace. Verifies that the IC produces a valid proof for the 6 F₂[X] constraints. Does NOT run verification. |
| `qx_ideal_check_succeeds_on_valid_sha256_witness` | IC prover on Q[X]-projected SHA-256 trace. Verifies the 5 Q[X] constraints (including the $(X-2)$ ideal). Does NOT run verification. |
| `full_pipeline_round_trip` | Single-ring: `prove()` → `verify()` with TrivialIdeal. Full IC + CPR + PCS round-trip for the 6 F₂[X] constraints. |
| `dual_ring_pipeline_round_trip` | Dual-ring: `prove_dual_ring()` → `verify_dual_ring()`. BP pass (6 constraints, TrivialIdeal) + QX pass (5 constraints, real DegreeOne(2) ideal) + PCS. **This is the most comprehensive test.** |
| `ecdsa_ideal_check_succeeds_on_valid_witness` | ECDSA IC prover on the constant fixed-point trace. Verifies the 7 i64 constraints produce a valid IC proof. |
| `ecdsa_pipeline_round_trip` | ECDSA single-ring `Int<4>` pipeline: `prove_generic()` → `verify_generic()`. All 7 constraints checked in `Int<4>` ring with `EcdsaScalarZipTypes` (PCS field = `MontyField<8>`). Full IC + CPR + PCS round-trip on a zero trace. |

Additionally:
- Each UAIR crate has unit tests for constraint count, max degree, scalar collection, and (for SHA-256) NIST test vector validation.
- `zinc-utils` has 3 targeted tests for the optimized `Int<N> * i64` multiply covering `Int<2>`, `Int<4>`, `Int<6>`, `Int<8>` against naive full-width multiply, including edge cases (`i64::MAX`, `i64::MIN`, `0`, `-1`).
- The `zip-plus` crate has internal tests for batched PCS verification, single-poly PCS verification, and IPRS code encoding/decoding.

---

## 17. Benchmark Structure

Five criterion benchmark suites in `snark/benches/e2e_sha256.rs`:

| Suite | What it measures |
|-------|-----------------|
| `sha256_single` | **Headline benchmark.** PCS-only prover/verifier + full-pipeline (`pipeline::prove`/`pipeline::verify`) prover/verifier + proof size for single SHA-256 (20 cols, DEPTH=1). Reports split-PCS proof sizes (11 BPoly + 9 Int). |
| `sha256_8x_ecdsa` | **Paper target benchmark.** Two separate PCS batches: SHA-256 columns as `BinaryPoly<32>` + 9 ECDSA columns as `Int<4>` (256-bit scalar), both at DEPTH=1 (512 rows). Reports combined prover/verifier timing, proof sizes (monolithic and split-SHA variants), and parallel verification. |
| `sha256_piop_only` | IC and CPR prover in isolation (no PCS). |
| `sha256_end_to_end` | Manual PIOP+PCS composition (IC + CPR + PCS commit/test/evaluate). |
| `sha256_full_pipeline` | Uses `pipeline::prove()` / `pipeline::verify()` directly, reports detailed proof size breakdown (PCS + PIOP components). |

### Benchmark Constants

```
SHA256_NUM_VARS       = 7    // 128 rows (64 real + 64 padding)
SHA256_8X_NUM_VARS    = 9    // 512 rows (8 × 64 rounds)
ECDSA_NUM_VARS        = 9    // 512 rows (258 real + 254 padding)
SHA256_BATCH_SIZE     = 20   // 20 SHA-256 columns (monolithic)
SHA256_POLY_BATCH_SIZE = 11  // BinaryPoly columns (split)
SHA256_INT_BATCH_SIZE  = 9   // Int columns (split)
ECDSA_BATCH_SIZE      = 9    // 9 ECDSA columns
```

### Dual-PCS Architecture for 8×SHA-256 + ECDSA

The `sha256_8x_ecdsa` benchmark uses **two (or three) independent PCS batches** rather than a single combined trace:

| Batch | Columns | Eval type | ZipTypes | IPRS code | Rows | Field |
|-------|---------|-----------|----------|-----------|------|-------|
| SHA-256 (mono) | 20 | `BinaryPoly<32>` | `Sha256ZipTypes<i64, 32>` | R4B64 DEPTH=1 | 512 | `MontyField<4>` (128-bit) |
| SHA-256 BPoly (split) | 11 | `BinaryPoly<32>` | `Sha256ZipTypes<i64, 32>` | R4B64 DEPTH=1 | 512 | `MontyField<4>` (128-bit) |
| SHA-256 Int (split) | 9 | `Int<1>` | `Sha256IntZipTypes` | R4B64 DEPTH=1 | 512 | `MontyField<4>` (128-bit) |
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

Run all benchmarks:
```bash
cargo bench -p zinc-snark --bench e2e_sha256 --features=zinc-snark/parallel,zinc-snark/simd
```

---

## 18. Crate Map

```
zinc-plus-new/
├── snark/              # Top-level pipeline (prove/verify), benchmarks, integration tests
│   ├── src/pipeline.rs # prove(), verify(), prove_generic(), verify_generic(),
│   │                   # prove_dual_ring(), verify_dual_ring(), derive_pcs_point()
│   ├── benches/        # Criterion benchmarks (5 suites)
│   └── tests/          # 7 integration tests
├── piop/               # PIOP protocols
│   └── src/
│       ├── ideal_check.rs           # IC prove/verify
│       ├── combined_poly_resolver.rs # CPR prove/verify (contains sumcheck)
│       └── sumcheck.rs              # Sumcheck protocol
├── sha256-uair/        # SHA-256 UAIR + witness
│   └── src/
│       ├── lib.rs      # 6 F₂[X] + 5 Q[X] constraints, ideal types
│       ├── witness.rs  # Generates 64-row SHA-256 trace, split witnesses
│       │               # (generate_poly_witness, generate_int_witness),
│       │               # POLY_COLUMN_INDICES, INT_COLUMN_INDICES
│       └── constants.rs # H[], K[] SHA-256 constants
├── ecdsa-uair/         # ECDSA UAIR + witness
│   └── src/
│       ├── lib.rs       # 7 Int<4> + 7 i64 constraints, ideal types
│       ├── constraints.rs # Mathematical specification of all 7 constraints
│       └── witness.rs   # Constant fixed-point witness generator (Int<4> + i64)
├── uair/               # Abstract UAIR trait, constraint builders
├── poly/               # Polynomial types (BinaryPoly, DensePolynomial, MLE)
├── zip-plus/           # Zip+ batched PCS (IPRS codes, Merkle trees, FRI-like testing)
│   └── src/
│       ├── code.rs             # LinearCode trait (encode, encode_wide,
│       │                       # encode_*_at_positions)
│       ├── code/iprs.rs        # IPRS code: spot-check encoding overrides
│       ├── code/iprs/pntt/radix8/
│       │   ├── params.rs       # Radix8PnttParams (encoding_matrix precomputation)
│       │   └── butterfly.rs    # BUTTERFLY_TABLE, radix-8 butterfly
│       ├── batched_pcs/        # Batched PCS (commit, test, evaluate, verify)
│       │   ├── phase_evaluate.rs  # Batched eval: β-batched row + per-poly scalars
│       │   └── phase_verify.rs    # Batched verify: spot-check + parallel
│       ├── pcs/                # Single-poly PCS (reference implementation)
│       │   └── phase_verify.rs # Single-poly verify: spot-check encoding
│       └── merkle.rs           # Merkle tree commit/open/verify
├── transcript/         # Keccak Fiat-Shamir transcript
├── primality/          # Miller-Rabin primality test
└── utils/              # Shared utilities
    └── src/
        ├── mul_by_scalar.rs   # MulByScalar trait, optimized Int<N>*i64
        └── ...                # Field conversion, inner products, Int<N> type
```
