# Instructions for Implementing the Lookup Protocol in Zinc+

## 1. Goal

Implement the **lookup argument** for the Zinc+ PIOP. This is the component that enforces **typing constraints** on trace columns — i.e., that every entry of a witness vector belongs to a prescribed finite set (a "lookup table"). Two primary lookup tables arise in practice:

- **`BitPoly(w)`**: the set `{0,1}^{<w}[X]` — binary polynomials of degree < w. After projection ψ_{q,a}: F_q[X] → F_q (evaluation at a random point a), this becomes a subset of F_q of size 2^w.
- **`Word(w)`**: the set `{0, 1, ..., 2^w - 1}` ⊂ Z. After projection φ_q: Z → F_q (mod q), this becomes a subset of F_q of size 2^w.

For w = 32, these projected tables have 2^32 entries, which is too large to handle directly. The standard approach is **Lasso-style table decomposition**: decompose a 2^32-entry table into two 2^16-entry sub-tables, prove decomposition consistency, then run LogUp on each sub-table.

The lookup argument operates **purely over F_q** (the random projected prime field), after the Zinc+ compiler has already projected the trace from Q[X] → F_q[X] → F_q.

**Regarding commitments**: For any auxiliary vectors arising from the LogUp protocol (multiplicity vectors, inverse vectors, decomposition chunk vectors), the prover sends these **in the clear** to the verifier. No polynomial commitment is needed for these vectors.

---

## 2. Background: Where Lookups Fit in the Protocol Pipeline

The Zinc+ PIOP pipeline (see `piop/src/` and the paper's Section on the PIOP compiler) is:

1. **Ideal Check** (`ideal_check.rs`): Evaluates UAIR constraint polynomials at a random MLE point, checks ideal membership.
2. **Combined Poly Resolver** (`combined_poly_resolver.rs`): Projects F_q[X] → F_q, batches constraint claims, runs sumcheck.
3. **Lookup** ← **THIS IS WHAT YOU IMPLEMENT**: After projection to F_q, enforce that projected trace columns belong to projected lookup tables.
4. *(Future: Shift reduction — not yet implemented either.)*

The lookup step receives:
- The **projected trace columns** as `DenseMultilinearExtension<F::Inner>` vectors over F_q (base field elements, not polynomials).
- Information about **which columns need which lookup tables** (inferred from the UAIR signature: `binary_poly_cols` → BitPoly table, `int_cols` → Word table).
- The **projecting element** `a ∈ F_q` used in the ψ_a projection (needed to compute the projected BitPoly table).

---

## 3. Protocol Description

### 3.1 LogUp PIOP (Protocol `prot:logup` in the paper)

**Relation**: Given a witness vector w ∈ F_q^W and a table T ∈ F_q^N, prove that w_i ∈ {T_j : j ∈ [N]} for all i ∈ [W].

**Protocol**:

1. **Multiplicities**: P computes the multiplicity vector m ∈ F_q^N, where m_j counts the number of occurrences of T_j in w. P sends m in the clear.

2. **Challenge**: V samples β ←$ F_q.

3. **Log-derivative identity**: P and V verify:
   ```
   Σ_{i ∈ [W]} 1/(β - w_i) = Σ_{j ∈ [N]} m_j/(β - T_j)
   ```
   
   **Inverse-commitment variant** (recommended for this implementation since we send things in the clear anyway):
   - P computes u_i = 1/(β - w_i) for i ∈ [W] and v_j = 1/(β - T_j) for j ∈ [N]. P sends u and v in the clear.
   - Run a sumcheck to verify: `Σ_{y ∈ {0,1}^μ} [(β - w̃(y)) · ũ(y) - 1] · eq(y, r) = 0` (i.e., (β - w_i)·u_i = 1 for all i).
   - Run a sumcheck to verify: `Σ_{y ∈ {0,1}^ν} [(β - T̃(y)) · ṽ(y) - 1] · eq(y, r') = 0` (i.e., (β - T_j)·v_j = 1 for all j).
   - Run a sumcheck to verify: `Σ_{i} ũ(y) - Σ_{j} m̃(y)·ṽ(y) = 0`.
   
   These three sumchecks can be **batched** into a single sumcheck using random linear combination (sample a batching challenge γ).

4. **Multiplicity sum check**: Run a sumcheck to verify `Σ_j m_j = W`.

5. V accepts iff all sumcheck verifiers accept.

### 3.2 Decomposition + LogUp for Large Tables

For tables of size 2^32 (e.g., BitPoly(32) or Word(32)), decompose into c = 2 chunks of size 2^16 each.

**Table decomposition for BitPoly(32) projected to F_q**:

After projection at evaluation point `a`, the table element corresponding to binary polynomial b_{31}X^{31} + ... + b_1 X + b_0 maps to:
```
b_0 + b_1·a + b_2·a^2 + ... + b_{31}·a^{31}
```

This decomposes as:
```
(b_0 + b_1·a + ... + b_{15}·a^{15}) + a^{16} · (b_{16} + b_{17}·a + ... + b_{31}·a^{15})
```

So the decomposition is:
- g(y_1, y_2) = y_1 + a^{16} · y_2 (a multilinear polynomial in two variables)
- T_1 = T_2 = {projected BitPoly(16)} — a table of size 2^{16}
- Each entry of the witness w decomposes as w = chunk_lo + a^{16} · chunk_hi, where chunk_lo, chunk_hi ∈ T_1.

**Table decomposition for Word(32)**:

An integer n ∈ [0, 2^{32}-1] decomposes as n = lo + 2^{16} · hi where lo, hi ∈ [0, 2^{16}-1]. After mod-q projection:
- g(y_1, y_2) = y_1 + (2^{16} mod q) · y_2
- T_1 = T_2 = {0, 1, ..., 2^{16}-1} mod q — a table of size 2^{16}

**Protocol (Decomposition + LogUp)**:

1. **Decomposition step**: P computes chunk vectors w_lo, w_hi ∈ F_q^W satisfying w_j = g(w_lo,j, w_hi,j) for all j. P sends w_lo, w_hi in the clear.

2. **Decomposition consistency sumcheck**: V samples random r ∈ F_q^μ. Run sumcheck on:
   ```
   Σ_{y ∈ {0,1}^μ} [w̃(y) - g(w̃_lo(y), w̃_hi(y))] · eq(y, r) = 0
   ```

3. **Sub-table lookups**: For each chunk (lo and hi), invoke LogUp with the sub-table T_i (size 2^{16}) and witness chunk w_i.

4. V accepts iff decomposition consistency and all LogUp invocations accept.

All sumchecks (decomposition + LogUp for both chunks) can be **batched** into a single sumcheck.

---

## 4. Implementation Plan

### 4.1 New Crate or Module?

Add a new module inside the `piop` crate: `piop/src/lookup/`. This module will contain:

```
piop/src/lookup/
├── mod.rs           // Public API: LogupProtocol, DecompLogupProtocol
├── logup.rs         // Core LogUp protocol (prove/verify)
├── decomposition.rs // Table decomposition + LogUp wrapper
├── tables.rs        // Projected lookup table generation (BitPoly, Word)
└── structs.rs       // Proof, ProverState, VerifierSubClaim structs
```

Add `pub mod lookup;` to `piop/src/lib.rs`.

### 4.2 Key Types and Structs

```rust
// In piop/src/lookup/structs.rs

/// Describes a lookup table and which columns use it.
pub struct LookupInstance<F> {
    /// The projected lookup table entries (e.g., projected BitPoly(16) as Vec<F>).
    pub table: Vec<F>,
    /// The witness column MLEs that must have entries in this table.
    /// These are DenseMultilinearExtension<F::Inner> — the projected trace columns.
    pub witness_columns: Vec<DenseMultilinearExtension<F::Inner>>,
}

/// Proof for the LogUp protocol.
/// Since all auxiliary vectors are sent in the clear, the proof
/// just contains these vectors plus the sumcheck proof(s).
pub struct LogupProof<F: PrimeField> {
    /// Multiplicity vector for each table entry.
    pub multiplicities: Vec<F>,
    /// Inverse witness vector: u_i = 1/(β - w_i).
    pub inverse_witness: Vec<F>,
    /// Inverse table vector: v_j = 1/(β - T_j).
    pub inverse_table: Vec<F>,
    /// The batched sumcheck proof.
    pub sumcheck_proof: SumcheckProof<F>,
}

/// Proof for the Decomposition+LogUp protocol.
pub struct DecompLogupProof<F: PrimeField> {
    /// The decomposition chunk vectors (sent in the clear).
    /// For a 2-chunk decomposition: [chunk_lo, chunk_hi].
    pub chunk_vectors: Vec<Vec<F>>,
    /// Sumcheck proof for decomposition consistency.
    pub decomp_sumcheck_proof: SumcheckProof<F>,
    /// LogUp proofs for each chunk's sub-table lookup.
    pub chunk_logup_proofs: Vec<LogupProof<F>>,
}
```

### 4.3 Table Generation (`tables.rs`)

Implement functions that generate the projected lookup tables:

```rust
/// Generate the projected BitPoly(w) table over F_q.
/// For BitPoly(w), the table has 2^w entries. Each entry is the evaluation
/// of a binary polynomial b_{w-1}X^{w-1} + ... + b_0 at the projecting 
/// element `a`.
/// Entry indexed by integer n (0 <= n < 2^w) corresponds to the binary
/// polynomial whose coefficients are the bits of n, evaluated at a:
///   T[n] = Σ_{k=0}^{w-1} bit_k(n) · a^k
pub fn generate_bitpoly_table<F: PrimeField>(
    width: usize,
    projecting_element: &F,
) -> Vec<F> { ... }

/// Generate the projected Word(w) table over F_q.
/// For Word(w), the table has 2^w entries: {0, 1, ..., 2^w - 1} mod q.
pub fn generate_word_table<F: PrimeField>(
    width: usize,
    field_cfg: &F::Config,
) -> Vec<F> { ... }

/// Generate the projected BitPoly(w) sub-table for decomposition.
/// This is just BitPoly(w/2) (the lower or upper half bits).
pub fn generate_bitpoly_subtable<F: PrimeField>(
    half_width: usize,
    projecting_element: &F,
) -> Vec<F> { ... }

/// Generate the projected Word(w) sub-table for decomposition.
/// This is just Word(w/2) = {0, ..., 2^{w/2} - 1} mod q.
pub fn generate_word_subtable<F: PrimeField>(
    half_width: usize,
    field_cfg: &F::Config,
) -> Vec<F> { ... }
```

For BitPoly tables, the generation can be done efficiently:
- Precompute powers of `a`: a^0, a^1, ..., a^{w-1}.
- For each n ∈ [0, 2^w), compute T[n] = Σ_k bit_k(n) · a^k.
- Optimization: use the recursive structure — T[n + 2^k] = T[n] + a^k.

### 4.4 Core LogUp Implementation (`logup.rs`)

Follow the existing pattern of `prove_as_subprotocol` / `verify_as_subprotocol`:

```rust
pub struct LogupProtocol<F: InnerTransparentField>(PhantomData<F>);

impl<F: InnerTransparentField + FromPrimitiveWithConfig + Send + Sync> LogupProtocol<F> {
    /// Prover for the LogUp protocol.
    ///
    /// # Arguments
    /// - `transcript`: Fiat-Shamir transcript
    /// - `witness`: The witness vector as DenseMultilinearExtension<F::Inner>
    ///   (the projected trace column). Length W = 2^μ.
    /// - `table`: The lookup table entries. Length N = 2^ν.
    /// - `field_cfg`: Field configuration.
    ///
    /// # Returns
    /// (LogupProof, LogupProverState)
    pub fn prove_as_subprotocol(
        transcript: &mut impl Transcript,
        witness: &DenseMultilinearExtension<F::Inner>,
        table: &[F],
        field_cfg: &F::Config,
    ) -> Result<(LogupProof<F>, LogupProverState<F>), LogupError<F>> {
        // 1. Compute multiplicity vector
        // 2. Send multiplicities in the clear (absorb into transcript)
        // 3. Get challenge β from transcript
        // 4. Compute inverse vectors u, v
        // 5. Send u, v in the clear (absorb into transcript)
        // 6. Run batched sumcheck for the three LogUp identities
        // 7. Run multiplicity-sum sumcheck
    }

    /// Verifier for the LogUp protocol.
    pub fn verify_as_subprotocol(
        transcript: &mut impl Transcript,
        proof: LogupProof<F>,
        table: &[F],
        witness_len: usize,
        field_cfg: &F::Config,
    ) -> Result<LogupVerifierSubClaim<F>, LogupError<F>> {
        // 1. Absorb multiplicities from proof into transcript
        // 2. Get challenge β from transcript
        // 3. Absorb inverse vectors from proof into transcript
        // 4. Verify batched sumcheck
        // 5. Verify multiplicity sum = witness_len
        // 6. Verify final evaluation claims against proof data
    }
}
```

**Key implementation details for the prover**:

a) **Computing multiplicities**: Build a HashMap<F, usize> from the witness entries, then for each table entry T_j, look up how many times it appears → m_j.

b) **Computing inverses**: For each w_i, compute u_i = (β - w_i)^{-1}. For each T_j, compute v_j = (β - T_j)^{-1}. Use batch inversion for efficiency.

c) **Batched sumcheck**: The three identities to verify are:
   - `(β - w_i) · u_i = 1` for all i
   - `(β - T_j) · v_j = 1` for all j  
   - `Σ_i u_i = Σ_j m_j · v_j`

   Batch them with a random challenge γ:
   ```
   Σ_y [γ^0 · ((β - w̃(y))·ũ(y) - 1)·eq(y,r) 
      + γ^1 · ((β - T̃(y))·ṽ(y) - 1)·eq(y,r')
      + γ^2 · (ũ(y) - m̃(y)·ṽ(y))] = 0
   ```
   
   Note: The witness and table may have different lengths (2^μ vs 2^ν). Handle this by padding or running separate sumchecks. The simplest approach: run the first and third sumchecks over the witness hypercube {0,1}^μ, and the second over the table hypercube {0,1}^ν. Alternatively, if witness and table have the same MLE dimension, batch everything.

   **Simplest correct approach**: Run the identity `Σ_i u_i - Σ_j m_j · v_j = 0` as a combined sumcheck where u is padded to max(W, N) and m·v is padded to max(W, N), OR run two separate sumchecks. The constraint sumchecks for inverse correctness can be run separately over their respective domains.

d) **Multiplicity sum**: Verify `Σ_j m_j = W`. Since m is sent in the clear, the verifier can just sum the entries directly — no sumcheck needed.

### 4.5 Decomposition + LogUp (`decomposition.rs`)

```rust
pub struct DecompLogupProtocol<F: InnerTransparentField>(PhantomData<F>);

impl<F: InnerTransparentField + FromPrimitiveWithConfig + Send + Sync> DecompLogupProtocol<F> {
    /// Prover for the Decomposition+LogUp protocol.
    ///
    /// Handles tables of size 2^{2k} by decomposing into two sub-tables
    /// of size 2^k each.
    ///
    /// # Arguments
    /// - `transcript`: Fiat-Shamir transcript
    /// - `witness`: The witness column (DenseMultilinearExtension<F::Inner>)
    /// - `decompose_fn`: Function that decomposes a witness entry into
    ///   (chunk_lo, chunk_hi) ∈ sub-table × sub-table
    /// - `combine_fn`: The multilinear g such that w = g(chunk_lo, chunk_hi)
    ///   i.e., w = chunk_lo + shift · chunk_hi for some `shift`
    /// - `subtable`: The sub-table entries (size 2^k)
    /// - `field_cfg`: Field configuration
    pub fn prove_as_subprotocol(
        transcript: &mut impl Transcript,
        witness: &DenseMultilinearExtension<F::Inner>,
        decompose_fn: impl Fn(&F) -> (F, F),
        combine_fn: impl Fn(&F, &F) -> F + Send + Sync,
        subtable: &[F],
        field_cfg: &F::Config,
    ) -> Result<(DecompLogupProof<F>, DecompLogupProverState<F>), DecompLogupError<F>> {
        // 1. Compute chunk vectors by decomposing each witness entry
        // 2. Send chunk vectors in the clear (absorb into transcript)
        // 3. Run decomposition consistency sumcheck:
        //    Σ_y [w̃(y) - g(chunk_lo_tilde(y), chunk_hi_tilde(y))] · eq(y,r) = 0
        // 4. Run LogUp for chunk_lo against subtable
        // 5. Run LogUp for chunk_hi against subtable
    }

    pub fn verify_as_subprotocol(
        transcript: &mut impl Transcript,
        proof: DecompLogupProof<F>,
        combine_fn: impl Fn(&F, &F) -> F,
        subtable: &[F],
        witness_len: usize,
        field_cfg: &F::Config,
    ) -> Result<DecompLogupVerifierSubClaim<F>, DecompLogupError<F>> {
        // Mirror the prover's transcript operations
        // Verify decomposition sumcheck + LogUp sub-proofs
    }
}
```

**Decomposition functions for specific tables**:

For **BitPoly(32)** projected at evaluation point `a`:
```rust
/// Decompose a projected BitPoly(32) entry into two BitPoly(16) chunks.
/// Given w = Σ_{k=0}^{31} b_k · a^k, decompose as:
///   chunk_lo = Σ_{k=0}^{15} b_k · a^k
///   chunk_hi = Σ_{k=0}^{15} b_{k+16} · a^k
/// So w = chunk_lo + a^{16} · chunk_hi.
fn decompose_bitpoly32(w: &F, a: &F, a_pow_16: &F, subtable: &[F]) -> (F, F) {
    // Find chunk_lo, chunk_hi such that w = chunk_lo + a^16 * chunk_hi
    // and both are in the subtable.
    // Since subtable = projected BitPoly(16), and P knows the original 
    // binary polynomial, P can compute this directly from the original 
    // witness bits.
}
```

**Important note on decomposition**: The prover needs access to the **original unprojected witness** (the binary polynomial or integer) to compute the decomposition. The decomposition cannot be recovered solely from the projected F_q value (that would require solving a discrete-log-like problem). Therefore, the decomposition function should operate on the original trace data before projection, or the prover should precompute and store the chunk vectors alongside the projected trace.

The recommended approach:
- During `project_trace_to_field` (or right after), also compute the decomposition chunks for columns that require lookup.
- For BitPoly columns: the original `BinaryPoly<32>` gives you the bits directly. Split bits [0..16) and [16..32), project each half at `a`.
- For int columns: the original integer n gives you lo = n mod 2^16, hi = n / 2^16. Project each as F_q elements.

### 4.6 Integration Point: Connecting Lookups to the UAIR

The UAIR signature (`UairSignature`) tells you which columns need lookups:
- `binary_poly_cols` columns → each needs a BitPoly(w) lookup (w is the BinaryPoly const generic, typically 32)
- `int_cols` columns → each needs a Word(w) lookup (w is typically 32)
- `arbitrary_poly_cols` → no lookup needed (these are free-form polynomial columns)

Create a function that, given a UAIR signature and the projecting element, determines the lookup instances:

```rust
/// Determine which columns need lookups and generate the corresponding
/// projected tables and decomposition parameters.
pub fn build_lookup_instances<F, U: Uair>(
    projected_trace: &[DenseMultilinearExtension<F::Inner>],
    projecting_element: &F,
    field_cfg: &F::Config,
) -> Vec<DecompLookupInstance<F>> {
    let sig = U::signature();
    let mut instances = Vec::new();
    
    // First `binary_poly_cols` columns need BitPoly lookup
    for col_idx in 0..sig.binary_poly_cols {
        instances.push(DecompLookupInstance {
            witness: projected_trace[col_idx].clone(),
            subtable: generate_bitpoly_subtable(16, projecting_element),
            // g(lo, hi) = lo + a^16 * hi
            shift: projecting_element.pow(16),
        });
    }
    
    // Last `int_cols` columns need Word lookup
    let int_start = sig.binary_poly_cols + sig.arbitrary_poly_cols;
    for col_idx in int_start..(int_start + sig.int_cols) {
        instances.push(DecompLookupInstance {
            witness: projected_trace[col_idx].clone(),
            subtable: generate_word_subtable(16, field_cfg),
            // g(lo, hi) = lo + 2^16 * hi
            shift: F::from_with_cfg(1u64 << 16, field_cfg),
        });
    }
    
    instances
}
```

### 4.7 Interaction with Existing Sumcheck

The existing `MLSumcheck` (`piop/src/sumcheck.rs`) should be used for all sumcheck invocations within the lookup protocol. Follow the same pattern:

```rust
let (sumcheck_proof, sumcheck_prover_state) = MLSumcheck::prove_as_subprotocol(
    transcript,
    mles,       // Vec<DenseMultilinearExtension<F::Inner>>
    num_vars,
    degree,     // degree of the combination function
    |mle_values: &[F]| { /* combination function */ },
    field_cfg,
);
```

The LogUp sumcheck has degree 2 (products of two MLEs) or degree 3 if you include the eq polynomial.

### 4.8 Transcript Interaction

Follow the existing pattern for Fiat-Shamir:
- **Absorb** all prover messages (vectors sent in the clear) into the transcript before deriving challenges.
- **Derive** challenges (β, γ, random evaluation points) from the transcript.
- Use `transcript.absorb_random_field_slice(&vec, &mut buf)` for absorbing field element vectors.
- Use `transcript.get_field_challenge(field_cfg)` for sampling challenges.

### 4.9 Batching Multiple Lookup Instances

When multiple columns need lookups (e.g., 10 BitPoly32 columns + 3 Word32 columns in the SHA-256 UAIR), batch all lookup arguments:

1. All columns with the same table type share one table (and one set of table inverses v_j).
2. Sample a single β challenge for all LogUp instances.
3. Use random linear combination (with batching challenge γ) to combine all LogUp sumcheck identities into one sumcheck.
4. Similarly batch all decomposition consistency sumchecks.

---

## 5. Concrete Implementation Steps

### Step 1: Create the module structure

Create `piop/src/lookup/mod.rs`, `tables.rs`, `logup.rs`, `decomposition.rs`, `structs.rs`. Add `pub mod lookup;` to `piop/src/lib.rs`.

### Step 2: Implement table generation (`tables.rs`)

Implement `generate_bitpoly_table`, `generate_word_table`, and their sub-table variants. Write unit tests verifying:
- BitPoly table has correct size (2^w).
- Each entry equals the evaluation of the corresponding binary polynomial at the projecting element.
- Word table entries are {0, 1, ..., 2^w - 1} mod q.

### Step 3: Implement LogUp core (`logup.rs`)

Implement `LogupProtocol::prove_as_subprotocol` and `LogupProtocol::verify_as_subprotocol`. Start with a simple (unbatched, single-column) version:

a) Prover:
   - Compute multiplicities (HashMap-based counting).
   - Absorb multiplicities into transcript.
   - Get β from transcript.
   - Compute inverse vectors u, v (use batch inversion).
   - Absorb u, v into transcript.
   - Build MLE representations of u, v, m, witness, table.
   - Run sumcheck for the LogUp identity.

b) Verifier:
   - Mirror transcript operations.
   - Verify sumchecks.
   - Check final evaluation claims.

Write tests with small tables (e.g., table = {0, 1, 2, 3}, witness = valid entries).

### Step 4: Implement Decomposition+LogUp (`decomposition.rs`)

Implement `DecompLogupProtocol` that wraps the LogUp protocol with a decomposition step:

a) Prover:
   - Decompose each witness entry into chunks.
   - Send chunks in the clear.
   - Run decomposition consistency sumcheck.
   - Run LogUp for each chunk.

b) Verifier:
   - Verify decomposition sumcheck.
   - Verify chunk LogUp proofs.

Write tests with a 2^8 table decomposed into two 2^4 sub-tables.

### Step 5: Implement batched lookup for multiple columns

Create a high-level `BatchedLookupProtocol` that handles multiple columns with different (or same) tables:
- Group columns by table type.
- Batch all LogUp identities into a single sumcheck.
- Batch all decomposition consistency checks.

### Step 6: Integration with the PIOP pipeline

Add a function (or extend the existing pipeline) that:
1. Takes the projected trace from `CombinedPolyResolver`.
2. Determines which columns need lookups based on the UAIR signature.
3. Runs the batched lookup protocol.
4. Returns a `LookupVerifierSubClaim` that the outer protocol can check.

### Step 7: End-to-end tests

Write integration tests that:
1. Define a test UAIR with binary_poly and int columns.
2. Generate a valid witness.
3. Run the full PIOP pipeline: ideal check → combined poly resolver → lookup.
4. Verify the complete proof.

Also test rejection: modify a witness entry to be outside the lookup table and verify that the protocol rejects.

---

## 6. Detailed API Conventions to Follow

### 6.1 Code Style

- Follow the existing code style in the repository (see `rustfmt.toml`, `clippy.toml`).
- Use the `#[cfg(feature = "parallel")]` pattern with rayon for parallelism (see `zinc_utils::cfg_iter!`).
- Use `thiserror` for error types.
- Use `PhantomData` for type-level parameters on protocol structs.
- Use the `InnerTransparentField` trait bound for F (see `utils/src/inner_transparent_field.rs`).

### 6.2 Field Operations

- Use `F::zero_with_cfg(field_cfg)`, `F::one_with_cfg(field_cfg)` for constants.
- Use `F::from_with_cfg(value, field_cfg)` for converting integers to field elements.
- Use `transcript.get_field_challenge(field_cfg)` for Fiat-Shamir challenges.
- Use `transcript.absorb_random_field_slice(&vec, &mut buf)` for absorbing vectors. Remember to allocate `buf` as `vec![0u8; F::Inner::NUM_BYTES]`.

### 6.3 MLE Construction

```rust
let mle = DenseMultilinearExtension::from_evaluations_vec(
    num_vars,           // log2 of the vector length
    evaluations,        // Vec<F::Inner>
    F::Inner::default() // padding value
);
```

### 6.4 Dependencies

The lookup module will need these dependencies (already available in the `piop` crate's Cargo.toml):
- `crypto-primitives` (field traits)
- `zinc-poly` (MLE, polynomials)
- `zinc-transcript` (Fiat-Shamir)
- `zinc-uair` (UAIR signature)
- `zinc-utils` (parallel iterators, field helpers)
- `itertools`, `thiserror`, `num-traits`

You may also need to add `std::collections::HashMap` for multiplicity counting.

### 6.5 Batch Inversion

For computing inverse vectors efficiently, implement (or use if available) Montgomery's batch inversion trick:
```rust
/// Compute the multiplicative inverse of each element in `values`.
/// Uses Montgomery's trick: O(n) multiplications + 1 inversion.
fn batch_inverse<F: PrimeField>(values: &[F]) -> Vec<F> {
    // 1. Compute prefix products: p[i] = values[0] * ... * values[i]
    // 2. Invert p[n-1]
    // 3. Recover individual inverses by backward pass
}
```

---

## 7. Example: Projected BitPoly(32) Decomposition Step-by-Step

Suppose the UAIR has a `binary_poly` column with `BinaryPoly<32>`.

1. **Original witness**: A column of `BinaryPoly<32>` values, e.g., `[0b1010...0, 0b1100...1, ...]`.

2. **Step 1 projection** (φ_q: Z → F_q): Each bit becomes 0 or 1 in F_q. The binary polynomial's coefficients become F_q elements. This produces `DenseMultilinearExtension<DynamicPolynomialF<F>>`.

3. **Step 3 projection** (ψ_a: F_q[X] → F_q): Evaluate each polynomial at a random point `a`. This produces `DenseMultilinearExtension<F::Inner>` — a vector of field elements.

4. **Lookup table**: The projected BitPoly(32) table is {Σ b_k · a^k : b ∈ {0,1}^32}, which has 2^32 entries.

5. **Decomposition**: Split each 32-bit binary polynomial into low 16 bits and high 16 bits:
   - chunk_lo = Σ_{k=0}^{15} b_k · a^k (a projected BitPoly(16) element)
   - chunk_hi = Σ_{k=0}^{15} b_{k+16} · a^k (also a projected BitPoly(16) element)
   - w = chunk_lo + a^{16} · chunk_hi

6. **Sub-table**: projected BitPoly(16) = {Σ_{k=0}^{15} b_k · a^k : b ∈ {0,1}^16}, which has 2^16 entries.

7. **P sends** chunk_lo_vec, chunk_hi_vec in the clear.

8. **Decomposition sumcheck**: Verify w_i = chunk_lo_i + a^16 · chunk_hi_i for all i.

9. **LogUp on chunks**: Run LogUp(chunk_lo_vec, subtable) and LogUp(chunk_hi_vec, subtable).

---

## 8. Important Notes

1. **The prover needs the original (unprojected) witness** to compute decomposition chunks. The decomposition cannot be derived from the projected field element alone. Design the API so that the prover receives either:
   - The original `BinaryPoly<32>` / integer trace alongside the projected trace, OR
   - Precomputed chunk vectors.

2. **Table entries must be distinct** for LogUp to work (each T_j is unique). Both BitPoly and Word tables have this property by construction.

3. **Characteristic** must be sufficiently large: the field F_q must have characteristic > max table entry value, and β must not collide with any table entry. With a random 128-bit prime q, the collision probability is negligible.

4. **Padding**: Trace columns are padded to power-of-two length. Padded entries must also be valid table entries (e.g., pad with 0, which is in both BitPoly and Word tables).

5. **Testing**: Start with small examples (4-bit tables, 2-entry witnesses) to validate correctness before scaling up.

6. **The sumcheck for multiplicity sum** (`Σ m_j = W`) can be checked directly by the verifier since m is sent in the clear — no sumcheck needed.

7. **All sumchecks in the lookup can potentially be batched** with the main constraint sumcheck from `CombinedPolyResolver`, but implementing them as separate sub-protocols first is recommended for correctness and modularity.
