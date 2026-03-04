# E2E Verifier Specification: `verify_classic_logup_4x_folded`

> **Comprehensive step-by-step specification of the Zinc+ verifier for the 4× folded pipeline.**
> Produced from analysis of [snark/src/pipeline.rs](../snark/src/pipeline.rs) (lines 3652–4700).
> Target audience: Worker Agent writing detailed documentation.

---

## 1. Function Signature and Generic Parameters

```rust
pub fn verify_classic_logup_4x_folded<
    U, U2, PcsZt, PcsLc,
    const D: usize, const HALF_D: usize, const QUARTER_D: usize,
    const CHECK: bool,
    IdealOverF, IdealOverFFromRef,
    QxIdealOverF, QxIdealOverFFromRef,
>(
    pcs_params:              &ZipPlusParams<PcsZt, PcsLc>,
    folded_proof:            &Folded4xZincProof,
    num_vars:                usize,
    ideal_over_f_from_ref:   IdealOverFFromRef,
    qx_ideal_over_f_from_ref: QxIdealOverFFromRef,
    public_column_data:      &[DenseMultilinearExtension<BinaryPoly<D>>],
) -> VerifyResult
```

### Type parameter roles

| Parameter | Role | SHA-256 8× instantiation |
|---|---|---|
| `U` | Primary UAIR (BinaryPoly constraints) | `Sha256Uair` |
| `U2` | Secondary UAIR (Q[X] carry constraints) | `Sha256UairQx` |
| `PcsZt` | Zip+ PCS type parameters | `FoldedZt4x` |
| `PcsLc` | Zip+ linear code | `FoldedLc4x` |
| `D` | Original BinaryPoly width | 32 |
| `HALF_D` | After 1st fold | 16 |
| `QUARTER_D` | After 2nd fold (PCS element width) | 8 |
| `CHECK` | Enable/disable overflow checks in PCS | `false` (`UNCHECKED`) |
| `IdealOverF` / `IdealOverFFromRef` | Ideal membership checker for `U` | `TrivialIdeal` (max_degree==1 → trivial) |
| `QxIdealOverF` / `QxIdealOverFFromRef` | Ideal membership checker for `U2` | `TrivialIdeal` |

### Key trait bounds

- `U: Uair<Scalar = BinaryPoly<D>>` — primary UAIR over BinaryPoly ring
- `U2: Uair` with `U2::Scalar: Deref<Target = [i64]>` — Q[X] UAIR over integer ring
- `BinaryPoly<QUARTER_D>: ProjectableToField<PiopField>` — PCS elements project to PIOP field
- `PiopField: FromPrimitiveWithConfig + FromWithConfig<i64>` — field construction from integers
- `MillerRabin: PrimalityTest<...>` — primality test for random field config
- `PiopField` is `MontyField<3>` (192-bit Montgomery field, 3 × 64-bit limbs)

### Arguments

| Argument | Description |
|---|---|
| `pcs_params` | PCS parameters (num_vars, num_rows, linear code config, etc.) |
| `folded_proof` | The `Folded4xZincProof` struct containing all serialized proof data |
| `num_vars` | Number of MLE variables (= $\log_2(\text{trace rows})$, e.g. 9 for 512 rows) |
| `ideal_over_f_from_ref` | Closure converting UAIR-defined ideals to field ideals for `U` |
| `qx_ideal_over_f_from_ref` | Closure converting UAIR-defined ideals to field ideals for `U2` |
| `public_column_data` | MLE polynomials for public columns (known to verifier, not PCS-committed) |

### Return type

```rust
pub struct VerifyResult {
    pub accepted: bool,
    pub timing: VerifyTimingBreakdown,
}

pub struct VerifyTimingBreakdown {
    pub ideal_check_verify: Duration,
    pub combined_poly_resolver_verify: Duration,
    pub lookup_verify: Duration,
    pub pcs_verify: Duration,
    pub total: Duration,
}
```

---

## 2. Proof Structure: `Folded4xZincProof`

All fields are serialized byte vectors. The verifier deserializes them lazily at each step using `field_from_bytes(&bytes, &field_cfg)`.

```rust
pub struct Folded4xZincProof {
    // PCS
    pub pcs_proof_bytes: Vec<u8>,
    pub commitment: ZipPlusCommitment,              // Merkle root + batch_size

    // Ideal Check (BinaryPoly UAIR)
    pub ic_proof_values: Vec<Vec<u8>>,              // Combined MLE values

    // Ideal Check (Q[X] UAIR)
    pub qx_ic_proof_values: Vec<Vec<u8>>,           // Combined MLE values (empty if no U2)

    // CPR (Combined Poly Resolver)
    pub cpr_sumcheck_messages: Vec<Vec<u8>>,         // Round polynomials
    pub cpr_sumcheck_claimed_sum: Vec<u8>,            // Claimed sum
    pub cpr_up_evals: Vec<Vec<u8>>,                   // Up-evaluations (column MLEs at sumcheck point)
    pub cpr_down_evals: Vec<Vec<u8>>,                 // Down-evaluations (shifted column MLEs)

    // Q[X] CPR
    pub qx_cpr_sumcheck_messages: Vec<Vec<u8>>,
    pub qx_cpr_sumcheck_claimed_sum: Vec<u8>,
    pub qx_cpr_up_evals: Vec<Vec<u8>>,
    pub qx_cpr_down_evals: Vec<Vec<u8>>,

    // Lookup
    pub lookup_proof: Option<LookupProofData>,       // HybridGkr variant for this pipeline

    // Shift sumcheck (BP UAIR only; qx_shift_sumcheck is always None here)
    pub shift_sumcheck: Option<SerializedShiftSumcheckProof>,
    pub qx_shift_sumcheck: Option<SerializedShiftSumcheckProof>, // Always None

    // PIOP evaluation point
    pub evaluation_point_bytes: Vec<Vec<u8>>,         // Not used by verifier (reconstructed)

    // PCS evaluation claims
    pub pcs_evals_bytes: Vec<Vec<u8>>,                // 1 element: the combined PCS eval

    // 2-round folding
    pub folding_c1s_bytes: Vec<Vec<u8>>,              // Round 1: c₁[j] per PCS-committed column
    pub folding_c2s_bytes: Vec<Vec<u8>>,              // Round 1: c₂[j]
    pub folding_c3s_bytes: Vec<Vec<u8>>,              // Round 2: c₃[j]
    pub folding_c4s_bytes: Vec<Vec<u8>>,              // Round 2: c₄[j]

    pub timing: TimingBreakdown,                      // Prover-side timing (informational)
}
```

### `LookupProofData` enum

For the Hybrid GKR pipeline, the variant is:
```rust
LookupProofData::HybridGkr(HybridGkrPipelineLookupProof<PiopField>)
```

The verifier dispatches on this enum:
- `BatchedClassic` → batched CPR+Lookup path (multi-degree sumcheck)
- `HybridGkr` → standalone CPR sumcheck + separate Hybrid GKR lookup ← **this pipeline**
- `Gkr` → standalone CPR + separate full GKR lookup
- `Classic` → standalone CPR + separate classic LogUp lookup

---

## 3. Complete Step-by-Step Verifier Algorithm

### Overview

The verifier has **two main paths** depending on the `LookupProofData` variant. For the SHA-256 8× Hybrid GKR c=2 pipeline, `is_batched = false` (the proof uses `LookupProofData::HybridGkr`). The non-batched path is:

1. **Setup**: Transcript initialization + field config derivation
2. **Step 1**: IC verify (BinaryPoly UAIR)
3. **Step 1b**: QX IC verify (Q[X] UAIR) `[cfg(qx-constraints)]`
4. **Step 2**: CPR verify (standalone sumcheck)
   - 2a. QX CPR verify `[cfg(qx-constraints)]`
   - 2b. Lookup verify (Hybrid GKR)
5. **Step 2c**: Shift sumcheck verify (BP UAIR)
6. **Step 3**: Two-round folding verify (D→HALF_D→QUARTER_D)
7. **Step 4**: PCS verify (Zip+)

---

### Step 0: Setup — Transcript Initialization and Field Config

**Timing**: Part of `total` (no separate bucket)

**Logic**:
1. Start total timer: `let total_start = Instant::now()`
2. Count constraints and max degree from UAIR signature: `count_constraints::<U>()`, `count_max_degree::<U>()`
3. Create a fresh Keccak transcript: `KeccakTranscript::new()`
4. **Absorb the PCS commitment root** into the transcript:
   - `folded_proof.commitment.root.write_transcription_bytes(&mut root_buf)` → `transcript.absorb(&root_buf)`
   - This is `HASH_OUT_LEN` bytes (the Merkle root of the PCS commitment)
   - **Critical**: This makes the randomly derived prime depend on the commitment, preventing a prover from choosing polynomials that exploit a specific field prime.
5. **Derive the random field configuration**: `transcript.get_random_field_cfg::<PiopField, ...>()`
   - Uses Miller-Rabin primality testing to find a 128-bit prime
   - All subsequent field operations use this configuration (`field_cfg`)
6. Compute `field_elem_size` = number of bytes per field element (`ConstTranscribable::NUM_BYTES` for `Uint<3>`)

**Outputs**:
- `transcript` — Fiat-Shamir transcript (threaded through all subsequent steps)
- `field_cfg` — Montgomery field configuration (prime + Montgomery parameters)
- `num_constraints`, `max_degree` — from `U::signature()`

**State flowing forward**: `transcript`, `field_cfg`

---

### Step 1: IC Verify (BinaryPoly UAIR)

**Timing bucket**: `ideal_check_verify`

**Function**: `IdealCheckProtocol::<PiopField>::verify_as_subprotocol::<U, _, _>()`

**Proof data consumed**: `folded_proof.ic_proof_values`

**Deserialization**:
- Each entry in `ic_proof_values` is a byte vector encoding one `DynamicPolynomialF<PiopField>` (a polynomial with `num_coeffs` field element coefficients)
- Decoded into `ic_combined_mle_values: Vec<DynamicPolynomialF<PiopField>>`
- Wrapped into `zinc_piop::ideal_check::Proof { combined_mle_values }`

**Verification logic** (inside `verify_as_subprotocol`):
1. **Squeeze random evaluation point**: `evaluation_point = transcript.get_field_challenges(num_vars, &field_cfg)`
   - This is the random point $\mathbf{r} \in F^{n}$ at which all MLEs will be evaluated
2. **Absorb proof values** into transcript: for each `DynamicPolynomialF`, absorb all coefficients
3. **Collect ideals**: extract the ideals defined by `U`'s constraints via `collect_ideals::<U>(num_constraints)`
4. **Batched ideal check**: verify that each `combined_mle_values[i]` belongs to the corresponding ideal (via `ideal_over_f_from_ref`)
   - For SHA-256 with `max_degree == 1` and `TrivialIdeal`, this is a no-op (everything is in the trivial ideal)

**Outputs**:
- `ic_subclaim: VerifierSubClaim<F>` containing:
  - `evaluation_point: Vec<F>` — the random point $\mathbf{r}$
  - `values: Vec<DynamicPolynomialF<F>>` — the combined MLE values (claim data for CPR)
- `ic_evaluation_point = ic_subclaim.evaluation_point.clone()`

**Error condition**: If `batched_ideal_check` fails → return `VerifyResult { accepted: false, ... }`

**Connection to next step**: The `ic_subclaim` (evaluation point + MLE values) feeds into CPR.

---

### Step 1b: QX IC Verify (Q[X] UAIR) `[cfg(feature = "qx-constraints")]`

**Timing bucket**: Still `ideal_check_verify` (accumulated)

**Function**: `IdealCheckProtocol::<PiopField>::verify_at_point::<U2, _, _>()`

Note: This uses `verify_at_point` (NOT `verify_as_subprotocol`) because the QX IC reuses the **same** evaluation point $\mathbf{r}$ from Step 1.

**Proof data consumed**: `folded_proof.qx_ic_proof_values`

**Deserialization**: Same pattern as Step 1, decoded into `qx_ic_combined_mle_values`

**Verification logic** (inside `verify_at_point`):
1. `num_constraints` and `max_degree` from `U2::signature()`
2. **Absorb proof values** into transcript (same pattern)
3. **Batched ideal check** against `U2`'s ideals

**Key difference from Step 1**: No new random point is squeezed; the existing `ic_evaluation_point` is passed in directly.

**Outputs**:
- `qx_ic_subclaim` — same structure, shares the same evaluation point
- `ic_verify_time = t0.elapsed()` — total time for both IC steps

**Error condition**: If batched ideal check fails → early return with `accepted: false`

**Connection to next step**: `qx_ic_subclaim` feeds into QX CPR (Step 2a).

---

### Step 2: CPR + Lookup Verify (Non-Batched Path)

**Timing bucket**: `combined_poly_resolver_verify` (for CPR), `lookup_verify` (for lookup)

The verifier checks `is_batched = matches!(folded_proof.lookup_proof, Some(LookupProofData::BatchedClassic(_)))`. For `HybridGkr`, this is `false`, so the **non-batched path** is taken.

#### Step 2: CPR Verify (Standalone Sumcheck)

**Functions**:
1. `CombinedPolyResolver::<PiopField>::build_verifier_pre_sumcheck::<U>()`
2. `MLSumcheck::<PiopField>::verify_as_subprotocol()`
3. `CombinedPolyResolver::<PiopField>::finalize_verifier::<U>()`

**Proof data consumed**:
- `folded_proof.cpr_sumcheck_messages` — round polynomials for the degree-3 sumcheck
- `folded_proof.cpr_sumcheck_claimed_sum` — claimed sum of the sumcheck
- `folded_proof.cpr_up_evals` — private MLE evaluations of trace columns at sumcheck point
- `folded_proof.cpr_down_evals` — MLE evaluations of shifted columns

**Deserialization**:
- Round messages → `Vec<ProverMsg<PiopField>>` (each ProverMsg wraps `NatEvaluatedPolyWithoutConstant`)
- Claimed sum → single `PiopField`
- Up/down evals → `Vec<PiopField>`

**Verification logic**:

**(a) Pre-sumcheck setup** (`build_verifier_pre_sumcheck`):
1. Squeeze **projecting element** $\alpha$: `projecting_element = transcript.get_field_challenge(&field_cfg)`
   - This is used to project `BinaryPoly<D>` scalars to field elements
2. **Project UAIR scalars**: `project_scalars::<PiopField, U>()` maps each BinaryPoly coefficient to field element, then `project_scalars_to_field()` evaluates at `projecting_element`
3. Squeeze **folding challenge** $\beta$ from transcript (inside `build_verifier_pre_sumcheck`)
4. Compute **expected CPR claimed sum**: $\sum_{i=0}^{C-1} \beta^i \cdot \text{combined\_mle\_values}[i](\alpha)$
   - Evaluates each IC combined value at the projecting element, scales by folding challenge powers
5. Check: `cpr_claimed_sum == expected_sum` → error if mismatch

**(b) Sumcheck verification** (`MLSumcheck::verify_as_subprotocol`):
1. Verify the degree-(`max_degree + 2`) sumcheck proof over `num_vars` rounds
2. Each round: check `p(0) + p(1) == current_claim`, absorb polynomial, squeeze challenge
3. Returns `subclaim` with `point` (challenge vector) and `expected_evaluation`

**(c) Post-sumcheck finalization** (`finalize_verifier`):
1. Compute $\text{eq}(\mathbf{x}^*, \mathbf{r})$ where $\mathbf{x}^*$ is the sumcheck point and $\mathbf{r}$ is the IC evaluation point
2. Compute selector value $\text{eq}(\mathbf{x}^*, \mathbf{1})$
3. If there are **public columns**: compute their MLE evaluations at the sumcheck point from `public_column_data`, then reconstruct full `up_evals` via `reconstruct_up_evals()` (splicing public evals into private evals at the correct positions)
4. Re-evaluate all UAIR constraints via `U::constrain_general()` using the up/down evals, fold with $\beta$ powers
5. Check: $\text{eq}(\mathbf{x}^*, \mathbf{r}) \cdot (1 - \text{selector}) \cdot \text{folded\_constraints} = \text{subclaim.expected\_evaluation}$
6. **Absorb** up_evals and down_evals into transcript

**Outputs**:
- `cpr_subclaim: VerifierSubclaim<F>` containing:
  - `evaluation_point` — the sumcheck challenge point (= the PIOP evaluation point used for folding)
  - `up_evals` — trace column evaluations
  - `down_evals` — shifted column evaluations
- `cpr_verify_time`

**Error conditions**: Wrong claimed sum, sumcheck round check failure, constraint mismatch

---

#### Step 2a: QX CPR Verify `[cfg(feature = "qx-constraints")]`

**Functions**: Same trio as CPR but parameterized by `U2`:
1. `CombinedPolyResolver::<PiopField>::build_verifier_pre_sumcheck::<U2>()`
2. `MLSumcheck::<PiopField>::verify_as_subprotocol()`
3. `CombinedPolyResolver::<PiopField>::finalize_verifier::<U2>()`

**Proof data consumed**:
- `folded_proof.qx_cpr_sumcheck_messages`
- `folded_proof.qx_cpr_sumcheck_claimed_sum`
- `folded_proof.qx_cpr_up_evals`
- `folded_proof.qx_cpr_down_evals`

**Logic**: Identical to Step 2 but:
- Uses `qx_ic_subclaim` instead of `ic_subclaim`
- Uses `qx_num_constraints`, `qx_max_degree`, `qx_field_projected_scalars`
- Operates over `U2`'s constraint system (integer-valued carry propagation)
- Shares the **same** `projecting_element` (squeezed once in Step 2 and reused)

**Error conditions**: Same as Step 2

---

#### Step 2b: Lookup Verify (Hybrid GKR)

**Timing bucket**: `lookup_verify`

**Dispatch**: The verifier matches on `folded_proof.lookup_proof`:
```rust
LookupProofData::HybridGkr(proof) => verify_hybrid_gkr_batched_lookup(...)
```

**Function**: `verify_hybrid_gkr_batched_lookup()` → dispatches to `HybridGkrBatchedDecompLogupProtocol::<PiopField>::verify_as_subprotocol()` per lookup group.

**Proof data consumed**: `HybridGkrPipelineLookupProof<PiopField>` containing:
- Per-group: `HybridGkrBatchedDecompLogupProof<F>` with:
  - `aggregated_multiplicities`: $L \times T$ field elements
  - `witness_gkr`: `HybridBatchedGkrFractionProof` (roots, top-layer proofs, sent intermediates, bottom-layer GKR)
  - `table_gkr`: `GkrFractionProof` (root, layer proofs)

**Verification logic** (per lookup group, inside `verify_as_subprotocol`):

1. **Absorb aggregated multiplicities** into transcript
2. **Squeeze challenges** $\beta$ (lookup randomness) and $\alpha$ (tree-batching challenge)
3. **Compute tree dimensions**: witness depth $d_w$, table depth $d_t$
4. **Verify hybrid witness GKR** (`hybrid_batched_gkr_fraction_verify`):
   - Top $c$ layers: layer-by-layer GKR verification with shared sumchecks
   - MLE consistency check: sent intermediates must match running claims at cutoff
   - Bottom $d_w - c$ layers: fresh depth-$(d_w-c)$ batched GKR from intermediates to leaves
5. **Verify full table GKR** (`gkr_fraction_verify`): standard layer-by-layer verification
6. **Cross-check roots**: $\sum_\ell \alpha^\ell \cdot P_w^{(\ell)} / Q_w^{(\ell)} = P_t / Q_t$ (via cross-multiplication)
7. **Verify table-side leaf claims**: evaluate multiplicity/subtable MLEs at the table GKR's leaf point
8. **Verify multiplicity sums**: $\sum_j m^{(\ell)}[j] = K \times W$ for each lookup $\ell$

**Outputs**: `Vec<HybridGkrVerifierSubClaim<F>>` (not directly used for folding — transcript is the key output)

**Error conditions**: `GkrRootMismatch`, `GkrLeafMismatch`, `MultiplicitySumMismatch`, sumcheck failures at any GKR layer

---

### Step 2c: Shift Sumcheck Verify (BP UAIR)

**Timing bucket**: Part of `combined_poly_resolver_verify` (same timer as CPR in non-batched path)

**Condition**: Only runs when `folded_proof.shift_sumcheck.is_some()` (i.e., `U::signature().shifts` is non-empty)

**Proof data consumed**:
- `folded_proof.shift_sumcheck.rounds` — serialized round polynomials (3 evals each, degree-2)
- `folded_proof.shift_sumcheck.v_finals` — per-claim source column evaluations at challenge point (private only)
- `folded_proof.cpr_down_evals` — re-deserialized for shift claims

**Claim construction**:
```rust
claims[i] = ShiftClaim {
    source_col: i,
    shift_amount: sig.shifts[i].shift_amount,
    eval_point: cpr_subclaim.evaluation_point[..num_vars],  // PIOP point
    claimed_eval: cpr_down_evals[i],                        // from CPR step
}
```

**Verification logic** (split API when public shifts exist, monolithic otherwise):

**(a) When public shifts exist** (`has_public_shifts = true` for SHA-256):

1. `shift_sumcheck_verify_pre(&mut transcript, &ss_proof, &claims, num_vars, &field_cfg)`:
   - Draw batching coefficients $\alpha_i$ (one per claim)
   - Compute combined claim $\sum_i \alpha_i \cdot d_i$ where $d_i$ is the CPR down_eval
   - Replay `num_vars` sumcheck rounds: check $p(0) + p(1) = \text{claim}$, absorb, squeeze challenge $s_k$
   - Returns `ShiftSumcheckPreOutput { challenge_point: s, current_claim, alphas }`
   - **Does NOT absorb v_finals yet** — allows verifier to compute public entries

2. **Compute public v_finals** at challenge point $\mathbf{s}$:
   - For each public shift source column: evaluate its MLE at $\mathbf{s}$ (LE ordering) using `public_column_data`
   - Project via `BinaryPoly::<D>::prepare_projection(&projecting_element)`

3. **Reconstruct full v_finals**: `reconstruct_shift_v_finals()` splices public and private entries

4. `shift_sumcheck_verify_finalize(&mut transcript, &ss_pre, &claims, &full_v_finals, &field_cfg)`:
   - Absorb all v_finals into transcript
   - Compute expected final: $\sum_i \alpha_i \cdot L_{c_i}(\mathbf{s}, \mathbf{r}_i) \cdot v_i(\mathbf{s})$
     - $L_c$ = left-shift predicate (`eval_left_shift_predicate`)
   - Check: `current_claim == expected_final`

**(b) When no public shifts** (monolithic):

- `shift_sumcheck_verify()` does everything in one call

**Error conditions**: Round check failure, final claim mismatch

**Note**: The QX shift sumcheck (`qx_shift_sumcheck`) is **always None** in the `prove_hybrid_gkr_logup_4x_folded` prover, so it is skipped entirely.

---

### Step 3: Two-Round Folding Verification

**Timing bucket**: Not separately timed (falls between CPR and PCS buckets)

The folding reduces the PIOP point from $n$ dimensions to $n+2$ dimensions, while reducing the polynomial width from `D` to `QUARTER_D` in two rounds.

**Setup**:
1. `piop_point = cpr_subclaim.evaluation_point[..num_vars]`
2. `pcs_excluded = sig.pcs_excluded_columns()` — columns not PCS-committed
3. Collect `original_evals`: filter `cpr_up_evals` to include only PCS-committed columns (excluding public and shift-source columns)
4. If `cpr_subclaim.evaluation_point.len() > num_vars`: divide out the zero-padding factor $\prod_{k=n}^{\text{shared}-1}(1 - r_k)$ to normalize evals

#### Round 1: D → HALF_D

**Proof data consumed**: `folded_proof.folding_c1s_bytes`, `folded_proof.folding_c2s_bytes`

**Function**: `fold_claims_verify::<PiopField, _>(&mut transcript, &c1s, &c2s, &original_evals, &alpha_power_half, piop_point, &field_cfg)`

`alpha_power_half = compute_alpha_power(projecting_element, HALF_D)` = $\alpha^{16}$

**Verification logic** (inside `fold_claims_verify`):
1. **Consistency check**: for each column $j$: $c_1[j] + \alpha^{\text{HALF\_D}} \cdot c_2[j] = \text{original\_eval}[j]$
2. **Absorb** all $c_1$ and $c_2$ values into transcript
3. **Squeeze** $\beta$ (unused, for transcript alignment) and $\gamma_1$ (folding challenge)
4. **Compute new point**: $\mathbf{r}_1 = (\mathbf{r} \| \gamma_1)$
5. **Compute new evals**: $d[j] = (1 - \gamma_1) \cdot c_1[j] + \gamma_1 \cdot c_2[j]$

**Output**: `fold1_output: FoldingVerifierOutput { new_point, new_evals }`

#### Round 2: HALF_D → QUARTER_D

**Proof data consumed**: `folded_proof.folding_c3s_bytes`, `folded_proof.folding_c4s_bytes`

**Function**: Same `fold_claims_verify` but with round 1 outputs as inputs

`alpha_power_quarter = compute_alpha_power(projecting_element, QUARTER_D)` = $\alpha^{8}$

**Verification logic**: Same as Round 1 but:
- Input `original_evals` = `fold1_output.new_evals`
- Input `point` = `fold1_output.new_point`
- Consistency: $c_3[j] + \alpha^{\text{QUARTER\_D}} \cdot c_4[j] = d_1[j]$
- Squeezes $\gamma_2$, extends point to $(\mathbf{r} \| \gamma_1 \| \gamma_2)$

**Output**: `fold2_output: FoldingVerifierOutput { new_point, new_evals }`
- `new_point` = the final PCS evaluation point of dimension $n + 2$
- `new_evals` = the claimed PCS evaluations at that point

**Error conditions**: Folding consistency check fails for any column

---

### Step 4: PCS Verify (Zip+)

**Timing bucket**: `pcs_verify`

**Function**: `ZipPlus::<PcsZt, PcsLc>::verify_with_field_cfg::<PiopField, CHECK>()`

**Proof data consumed**:
- `folded_proof.pcs_proof_bytes` — the full Zip+ opening proof (serialized in transcript-stream form)
- `folded_proof.commitment` — the PCS commitment (Merkle root + batch_size)
- `folded_proof.pcs_evals_bytes` — 1 field element: the combined PCS eval claim

**Setup**:
1. Create a fresh PCS transcript: `PcsTranscript { fs_transcript: KeccakTranscript::default(), stream: Cursor::new(pcs_proof_bytes) }`
2. **Absorb the same commitment root** into the PCS transcript (same bytes as PIOP transcript)
3. Derive PCS field config: `pcs_transcript.fs_transcript.get_random_field_cfg()`
   - This produces the **same** prime as the PIOP field config (both transcripts absorb the same root from the same initial state)
4. Convert the folded point to PCS field representation
5. Read `eval_f` from `pcs_evals_bytes[0]`

**Verification logic** (inside `verify_with_field_cfg`):
1. **Tensor decomposition**: split point into row selector $q_0$ and column selector $q_1$
2. **Per-polynomial alpha challenges**: squeeze degree-bound+1 challenges per polynomial
3. **Read combined row** $b$ (from proof stream): `num_rows` field elements
4. **Eval consistency**: $\langle q_0, b \rangle = \text{eval\_f}$ (inner product check)
5. **Squeeze row coefficients** $s$
6. **Read combined row** $w$ from proof stream
7. **Eval-proximity link**: $\langle w, q_1 \rangle = \langle s, b \rangle$ (links eval to proximity)
8. **Grinding verification**: check proof-of-work nonce
9. **Column openings** (`NUM_COLUMN_OPENINGS` = 131 for SHA-256 params):
   - For each opening: squeeze random column index, read column values + Merkle proof
   - **Proximity check**: encode combined row at opened position, verify column values are consistent
   - **Merkle check**: verify authentication path from leaf to committed root

**Final result**: `pcs_result: Result<(), ZipError>`

**Error conditions**: Eval consistency failure, eval-proximity link failure, grinding failure, column testing failure, Merkle proof failure

---

## 4. Final Accept/Reject

```rust
VerifyResult {
    accepted: pcs_result.is_ok(),
    timing: VerifyTimingBreakdown { ... },
}
```

The proof is accepted if and only if **all** of:
1. IC verify passed (both BP and QX)
2. CPR verify passed (both BP and QX)
3. Lookup verify passed (Hybrid GKR)
4. Shift sumcheck verify passed
5. Folding rounds 1 and 2 passed
6. PCS verify passed

Any failure at steps 1–5 causes an **early return** with `accepted: false`. PCS failure sets `accepted: pcs_result.is_ok() = false`.

---

## 5. Fiat-Shamir Transcript Threading

The Keccak-based transcript is threaded sequentially through all verification steps. The exact absorption/squeeze order must match the prover exactly:

| Phase | Absorb / Squeeze |
|---|---|
| Setup | Absorb: commitment root (`HASH_OUT_LEN` bytes). Squeeze: `field_cfg` (random prime) |
| IC (BP) | Squeeze: evaluation point $\mathbf{r}$ (`num_vars` challenges). Absorb: IC proof values (MLE coefficients) |
| IC (QX) | Absorb: QX IC proof values |
| CPR pre | Squeeze: projecting element $\alpha$. Squeeze: folding challenge $\beta$ |
| CPR sumcheck | Per round: absorb round polynomial, squeeze challenge. (Repeated `num_vars` times) |
| CPR finalize | Absorb: up_evals, down_evals |
| QX CPR pre | Squeeze: QX folding challenge |
| QX CPR sumcheck | Per round: absorb + squeeze |
| QX CPR finalize | Absorb: QX up/down evals |
| Lookup (Hybrid GKR) | Absorb: aggregated multiplicities. Squeeze: $\beta$, $\alpha$. Then GKR layer-by-layer: absorb intermediates/evals, squeeze challenges. Table GKR similarly. |
| Shift sumcheck | Squeeze: $k$ batching alphas. Per round: absorb round poly, squeeze challenge. Absorb: v_finals |
| Fold round 1 | Absorb: $c_1$s, $c_2$s. Squeeze: $\beta$ (unused), $\gamma_1$ |
| Fold round 2 | Absorb: $c_3$s, $c_4$s. Squeeze: $\beta$ (unused), $\gamma_2$ |

The PCS uses a **separate** transcript (`PcsTranscript`) that does NOT share state with the PIOP transcript. The PCS transcript absorbs the same commitment root independently to derive the same field prime.

---

## 6. Timing Breakpoints

The verifier measures 4 timing buckets via `Instant::now()`:

| Bucket | Timer start | Timer stop | What it covers |
|---|---|---|---|
| `ideal_check_verify` | `t0 = Instant::now()` (before IC) | After QX IC verify | IC (BP) + IC (QX) |
| `combined_poly_resolver_verify` | `t1 = Instant::now()` (before CPR) | After shift sumcheck | CPR (BP) + QX CPR + Shift SC (non-batched path). In batched path: CPR + QX CPR + lookup finalize |
| `lookup_verify` | `t1b = Instant::now()` (before lookup dispatch) | After lookup dispatch | Hybrid GKR / GKR / Classic lookup only (non-batched). Zero in batched path |
| `pcs_verify` | `t2 = Instant::now()` (before PCS) | After PCS verify | Zip+ verify |
| `total` | `total_start = Instant::now()` (very first line) | After PCS verify | Everything |

The benchmark also prints these via `println!` for diagnostic purposes.

---

## 7. Assertions and Invariants

| Location | Assertion |
|---|---|
| Shift sumcheck | `assert!(!sig.shifts.is_empty())` when `shift_sumcheck.is_some()` |
| Shift rounds | Each round poly has exactly 3 evaluations |
| Folding | `c1s.len() == c2s.len() == original_evals.len()` |
| PCS | Various: eval consistency, proximity link, Merkle paths |
| CPR finalize | `expected_claim_value == subclaim_expected_evaluation` |
| IC verify | Combined MLE values belong to UAIR ideals |
| Lookup | Multiplicity sums = $K \times W$, root cross-check, leaf MLE matches |

---

## 8. Data Flow Diagram (Simplified)

```
folded_proof
    │
    ├─ commitment.root ──► transcript.absorb ──► field_cfg
    │
    ├─ ic_proof_values ──► IC verify ──► ic_subclaim (eval_point r, values)
    ├─ qx_ic_proof_values ──► QX IC verify ──► qx_ic_subclaim
    │
    ├─ cpr_sumcheck_* ──► CPR verify ──► cpr_subclaim (eval_point x*, up/down_evals)
    │   └─ ic_subclaim feeds expected sum
    │
    ├─ qx_cpr_* ──► QX CPR verify
    │   └─ qx_ic_subclaim feeds expected sum
    │
    ├─ lookup_proof ──► Hybrid GKR verify ──► (transcript advanced)
    │
    ├─ shift_sumcheck ──► Shift SC verify ──► (transcript advanced)
    │   └─ cpr_down_evals feed claims, public_column_data feed public v_finals
    │
    ├─ folding_c1s/c2s ──► Fold round 1 ──► fold1_output (point r‖γ₁, new_evals)
    ├─ folding_c3s/c4s ──► Fold round 2 ──► fold2_output (point r‖γ₁‖γ₂, final_evals)
    │   └─ cpr_up_evals (PCS-committed only, padding-corrected) feed original_evals
    │
    ├─ pcs_proof_bytes ──► PCS verify at fold2_output.new_point
    └─ pcs_evals_bytes ──► eval claim for PCS
```

---

## 9. Batched Path (Brief Summary)

When `folded_proof.lookup_proof` is `LookupProofData::BatchedClassic(bp)`, the verifier takes a different code path where CPR and classic lookup share a **multi-degree sumcheck**:

1. CPR pre-sumcheck + QX CPR pre-sumcheck + Lookup pre-sumcheck (all set up independently)
2. **Single** `MultiDegreeSumcheck::verify_as_subprotocol()` with shared challenges
   - Group 0 = CPR (degree `max_degree + 2`)
   - Group 1 = QX CPR (when enabled)
   - Groups 2+ = lookup groups (degree 2)
3. CPR finalize + QX CPR finalize + Lookup finalize (all use the shared subclaim point)

This path is **not used** by the SHA-256 Hybrid GKR pipeline (which uses the non-batched path described above), but the same `verify_classic_logup_4x_folded` function supports both.

---

## 10. Concrete Numbers for SHA-256 8× (4× Folded Hybrid GKR c=2)

| Parameter | Value |
|---|---|
| `D` | 32 |
| `HALF_D` | 16 |
| `QUARTER_D` | 8 |
| `num_vars` | 9 (512 rows) |
| BP constraints (`U`) | from `Sha256Uair` |
| QX constraints (`U2`) | from `Sha256UairQx` |
| Total trace columns | 30 |
| Public columns | 12 |
| PCS-excluded columns | 20 (12 public + 8 shift-source) |
| PCS-committed columns | 10 |
| Shift claims (BP) | 12 |
| Public shifts | 4 (claims 2–5: W_TM2, W_TM7, W_TM15, W_TM16) |
| Private shift v_finals | 8 |
| Lookup columns (E2E) | 13 (10 base + 3 affine) |
| Chunks per lookup | 4 |
| Chunk width | 8 bits |
| Subtable size | $2^8 = 256$ |
| Witness tree depth $d_w$ | 11 |
| Table tree depth $d_t$ | 8 |
| Hybrid cutoff $c$ | 2 |
| Fold round 1: $\alpha^{16}$ | Used for D→HALF_D consistency |
| Fold round 2: $\alpha^{8}$ | Used for HALF_D→QUARTER_D consistency |
| PCS eval point dimension | 11 ($= 9 + 2$ folding challenges) |
| NUM_COLUMN_OPENINGS | 131 |
| Field | 192-bit Montgomery (3×64-bit limbs), random 128-bit prime |
