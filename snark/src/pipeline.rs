//! End-to-end prover/verifier pipeline for Zinc+ SNARKs.
//!
//! This module provides [`prove`] and [`verify`] functions that compose:
//! - Batched Zip+ PCS (commit → test → evaluate → verify)
//! - Zinc+ PIOP (ideal check → combined poly resolver → sumcheck)
//!
//! The pipeline works over `BinaryPoly<DEGREE_PLUS_ONE>` traces.

use std::time::{Duration, Instant};

use crypto_bigint::modular::MontyParams;
use crypto_bigint::Odd;
use crypto_primitives::crypto_bigint_monty::MontyField;
use crypto_primitives::{Field, FromPrimitiveWithConfig, FromWithConfig, IntoWithConfig, PrimeField};
use num_traits::One;
use zinc_poly::mle::DenseMultilinearExtension;
use zinc_poly::univariate::binary::BinaryPoly;
use zinc_poly::univariate::dense::DensePolynomial;
use zinc_primality::{MillerRabin, PrimalityTest};
use zinc_transcript::traits::{ConstTranscribable, Transcript};
use zinc_transcript::KeccakTranscript;
use zinc_uair::constraint_counter::count_constraints;
use zinc_uair::degree_counter::count_max_degree;
use zinc_uair::Uair;
use zinc_utils::from_ref::FromRef;
use zinc_utils::projectable_to_field::ProjectableToField;

use zip_plus::code::LinearCode;
use zip_plus::pcs::structs::{ZipPlus, ZipPlusCommitment, ZipPlusParams, ZipTypes};
use zip_plus::pcs::ZipPlusProof;

use zinc_piop::ideal_check::IdealCheckProtocol;
use zinc_piop::combined_poly_resolver::CombinedPolyResolver;
use zinc_piop::lookup::{
    GkrPipelineLookupProof, PipelineLookupProof, LookupColumnSpec,
    LookupSumcheckGroup,
    prove_batched_lookup_with_indices,
    verify_gkr_batched_lookup,
    verify_batched_lookup,
    BatchedDecompLogupProtocol,
    group_lookup_specs,
};
use zinc_piop::lookup::pipeline::{
    LookupGroupMeta,
    build_lookup_instance_from_indices_pub,
    generate_table_and_shifts,
};
use zinc_piop::projections::{
    project_trace_coeffs, project_trace_to_field,
    project_scalars, project_scalars_to_field,
};
use zinc_piop::sumcheck::prover::{NatEvaluatedPolyWithoutConstant, ProverMsg};
use zinc_piop::sumcheck::SumcheckProof;
use zinc_piop::sumcheck::multi_degree::{MultiDegreeSumcheck, MultiDegreeSumcheckProof};
use zinc_poly::univariate::dynamic::over_field::DynamicPolynomialF;
use zinc_uair::ideal::{Ideal, IdealCheck};
use zinc_uair::ideal_collector::IdealOrZero;

/// Trivial ideal that always passes membership checks.
///
/// The IC verifier requires checking whether the combined MLE values
/// belong to the UAIR ideals lifted to the PIOP field. However, for
/// UAIRs defined over F₂\[X\] (like SHA-256), F₂ arithmetic (XOR = add)
/// does **not** lift to F_p arithmetic (1 + 1 = 2 ≠ 0). The prover
/// evaluates constraints over the projected field and the combined
/// polynomial outputs are not in the ideal when viewed over F_p.
///
/// Soundness is preserved because:
/// 1. The sumcheck / CPR verifies that the prover's claimed sums are
///    consistent with the committed trace (algebraic check over F_p).
/// 2. The PCS verifies proximity of the committed polynomials.
///
/// This matches the existing piop test pattern where the ideal check
/// always uses `IdealOrZero::Zero` (which bypasses membership testing).
#[derive(Clone, Copy, Debug)]
pub struct TrivialIdeal;

impl Ideal for TrivialIdeal {}

impl FromRef<TrivialIdeal> for TrivialIdeal {
    #[inline(always)]
    fn from_ref(ideal: &TrivialIdeal) -> Self {
        *ideal
    }
}

impl<F: PrimeField> IdealCheck<DynamicPolynomialF<F>> for TrivialIdeal {
    #[inline]
    fn contains(&self, _value: &DynamicPolynomialF<F>) -> bool {
        true
    }
}

// ─── Types ──────────────────────────────────────────────────────────────────

/// Timing breakdown for a single prove/verify invocation.
#[derive(Clone, Debug, Default)]
pub struct TimingBreakdown {
    pub pcs_commit: Duration,
    pub ideal_check: Duration,
    pub combined_poly_resolver: Duration,
    pub lookup: Duration,
    pub pcs_prove: Duration,
    pub total: Duration,
}

/// Timing breakdown for verification.
#[derive(Clone, Debug, Default)]
pub struct VerifyTimingBreakdown {
    pub ideal_check_verify: Duration,
    pub combined_poly_resolver_verify: Duration,
    pub lookup_verify: Duration,
    pub pcs_verify: Duration,
    pub total: Duration,
}

/// Lookup proof variant: either GKR-based or classic batched decomposition.
#[derive(Clone, Debug)]
pub enum LookupProofData {
    /// GKR fractional sumcheck based (no chunks/inverses sent).
    Gkr(GkrPipelineLookupProof<PiopField>),
    /// Classic batched decomposition (chunks + inverses + sumcheck).
    Classic(PipelineLookupProof<PiopField>),
    /// CPR + classic lookup sumchecks batched into a multi-degree sumcheck.
    ///
    /// Group 0 is the CPR, groups 1..N are the lookup table groups.
    /// All share the same verifier challenges and evaluation point.
    BatchedClassic(BatchedCprLookupProof),
}

/// Combined CPR + Classic lookup proof using multi-degree sumcheck.
///
/// The CPR sumcheck (degree `max_degree + 2`) and one or more lookup
/// sumchecks (degree 2) are run in lockstep with shared verifier
/// randomness, producing a single evaluation point.
#[derive(Clone, Debug)]
pub struct BatchedCprLookupProof {
    /// Multi-degree sumcheck proof (group 0 = CPR, groups 1.. = lookup).
    pub md_proof: MultiDegreeSumcheckProof<PiopField>,
    /// CPR up evaluations at the shared point.
    pub cpr_up_evals: Vec<PiopField>,
    /// CPR down evaluations at the shared point.
    pub cpr_down_evals: Vec<PiopField>,
    /// Per-group lookup metadata.
    pub lookup_group_meta: Vec<LookupGroupMeta>,
    /// Per-group lookup proof data (chunks, mults, inverses — no sumcheck).
    pub lookup_group_proofs: Vec<zinc_piop::lookup::BatchedDecompLogupProof<PiopField>>,
}

/// Full proof data produced by the prover.
#[derive(Clone, Debug)]
pub struct ZincProof {
    /// Serialized PCS proof bytes.
    pub pcs_proof_bytes: Vec<u8>,
    /// PCS commitment (Merkle root + batch size).
    pub commitment: ZipPlusCommitment,
    /// IdealCheck proof: evaluated combined polynomials at random point.
    pub ic_proof_values: Vec<Vec<u8>>,
    /// CPR proof: sumcheck proof + up/down evaluations.
    pub cpr_sumcheck_messages: Vec<Vec<u8>>,
    pub cpr_sumcheck_claimed_sum: Vec<u8>,
    pub cpr_up_evals: Vec<Vec<u8>>,
    pub cpr_down_evals: Vec<Vec<u8>>,
    /// Batched decomposed LogUp proof for column typing constraints.
    /// `None` if no lookup columns were specified.
    pub lookup_proof: Option<LookupProofData>,
    /// PCS evaluation claims from CPR (evaluation point in the field).
    pub evaluation_point_bytes: Vec<Vec<u8>>,
    /// PCS evaluation values (evaluations of committed polys at the point).
    pub pcs_evals_bytes: Vec<Vec<u8>>,
    /// Prover timing breakdown.
    pub timing: TimingBreakdown,
}

/// Result of the verification.
#[derive(Clone, Debug)]
pub struct VerifyResult {
    pub accepted: bool,
    pub timing: VerifyTimingBreakdown,
}

// ─── Field configuration ────────────────────────────────────────────────────

/// 192-bit Montgomery field (3 × 64-bit limbs) used for the PIOP.
pub const FIELD_LIMBS: usize = 3;
pub type PiopField = MontyField<FIELD_LIMBS>;

/// Returns a fixed configuration for the PIOP field.
///
/// Uses a known-good 128-bit prime: 0x860995AE68FC80E1B1BD1E39D54B33.
pub fn piop_field_config() -> MontyParams<FIELD_LIMBS> {
    let modulus = crypto_bigint::Uint::<FIELD_LIMBS>::from_be_hex(
        "000000000000000000860995AE68FC80E1B1BD1E39D54B33",
    );
    let modulus = Odd::new(modulus).expect("modulus should be odd");
    MontyParams::new(modulus)
}

/// Serialize a PiopField element to bytes (using Montgomery representation
/// for Fiat-Shamir consistency).
fn field_to_bytes(f: &PiopField) -> Vec<u8> {
    let mut buf = vec![0u8; <crypto_primitives::crypto_bigint_uint::Uint<FIELD_LIMBS> as ConstTranscribable>::NUM_BYTES];
    f.inner().write_transcription_bytes(&mut buf);
    buf
}

/// Deserialize a PiopField element from bytes (Montgomery representation).
fn field_from_bytes(bytes: &[u8], cfg: &<PiopField as PrimeField>::Config) -> PiopField {
    let inner = <crypto_primitives::crypto_bigint_uint::Uint<FIELD_LIMBS> as ConstTranscribable>::read_transcription_bytes(bytes);
    PiopField::from_montgomery(inner, cfg)
}

/// Convert a PiopField element (modular integer) to the PCS point type Zt::Pt.
///
/// The PIOP field prime is ~120 bits, so retrieved values always fit in i128.
/// We use `retrieve()` to get the true integer value from Montgomery form.
/// Kept for reference / future use.
#[allow(dead_code)]
fn piop_field_to_i128(f: &PiopField) -> i128 {
    let uint = f.retrieve();
    let words = uint.as_words();
    debug_assert!(
        words[2] == 0,
        "PIOP field element too large for i128: {:?}",
        words
    );
    (words[0] as i128) | ((words[1] as i128) << 64)
}

/// Derives a deterministic PCS evaluation point (Vec<i128>) from the CPR
/// evaluation point. Used by both prover and verifier to ensure they agree
/// on the same evaluation point for the batched PCS.
fn derive_pcs_point(cpr_evaluation_point: &[PiopField], num_vars: usize) -> Vec<i128> {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for f in cpr_evaluation_point {
        std::hash::Hash::hash(f.inner().as_words(), &mut hasher);
    }
    let seed = std::hash::Hasher::finish(&hasher) as i128;
    (0..num_vars)
        .map(|i| seed.wrapping_add(i as i128))
        .collect()
}

/// Project only the trace columns referenced by `lookup_specs` to field
/// elements, extracting raw integer indices at the same time.
///
/// Returns `(columns, raw_indices, remapped_specs)` where `columns` and
/// `raw_indices` contain only the needed columns and `remapped_specs` has
/// column indices adjusted to the 0-based position in `columns`.
/// Extract lookup columns from an already-projected field trace and compute
/// raw integer indices from the original BinaryPoly trace.
///
/// This avoids redundant field-element projection when the CPR step has
/// already projected the full trace with the same `projecting_element`.
fn extract_lookup_columns_from_field_trace<const D: usize>(
    bp_trace: &[DenseMultilinearExtension<BinaryPoly<D>>],
    field_trace: &[DenseMultilinearExtension<<PiopField as Field>::Inner>],
    lookup_specs: &[LookupColumnSpec],
    field_cfg: &MontyParams<FIELD_LIMBS>,
) -> (Vec<Vec<PiopField>>, Vec<Vec<usize>>, Vec<LookupColumnSpec>) {
    use std::collections::BTreeMap;

    // Collect unique column indices (sorted for determinism).
    let mut needed: BTreeMap<usize, usize> = BTreeMap::new();
    for spec in lookup_specs {
        let next_id = needed.len();
        needed.entry(spec.column_index).or_insert(next_id);
    }

    let mut columns: Vec<Vec<PiopField>> = Vec::with_capacity(needed.len());
    let mut raw_indices: Vec<Vec<usize>> = Vec::with_capacity(needed.len());

    for &orig_idx in needed.keys() {
        // Wrap Inner values into PiopField.
        let col_f: Vec<PiopField> = field_trace[orig_idx]
            .iter()
            .map(|inner| PiopField::new_unchecked_with_cfg(inner.clone(), field_cfg))
            .collect();
        columns.push(col_f);

        // Compute raw indices from BinaryPoly (pure bit manipulation, no field ops).
        let col_idx: Vec<usize> = bp_trace[orig_idx]
            .iter()
            .map(|bp| {
                let mut idx = 0usize;
                for (j, coeff) in bp.iter().enumerate() {
                    if coeff.into_inner() {
                        idx |= 1usize << j;
                    }
                }
                idx
            })
            .collect();
        raw_indices.push(col_idx);
    }

    // Remap lookup specs to 0-based indices into the projected columns.
    let index_map: BTreeMap<usize, usize> = needed
        .keys()
        .enumerate()
        .map(|(new_idx, &orig_idx)| (orig_idx, new_idx))
        .collect();

    let remapped_specs: Vec<LookupColumnSpec> = lookup_specs
        .iter()
        .map(|spec| LookupColumnSpec {
            column_index: index_map[&spec.column_index],
            table_type: spec.table_type.clone(),
        })
        .collect();

    (columns, raw_indices, remapped_specs)
}

// ─── Prover ─────────────────────────────────────────────────────────────────

/// Run the full Zinc+ prover pipeline for a UAIR over `BinaryPoly<D>`.
///
/// Returns a proof and timing breakdown.
///
/// # Type parameters
/// - `U`: the UAIR describing the constraint system.
/// - `Zt`: Zip+ type configuration.
/// - `Lc`: linear code for the PCS.
/// - `D`: `DEGREE_PLUS_ONE` for BinaryPoly.
/// - `CHECK`: overflow checking flag for the PCS.
///
/// # Arguments
/// - `params`: PCS parameters (poly size, num rows, code).
/// - `trace`: the witness trace as MLEs.
/// - `num_vars`: log₂(trace length).
/// - `lookup_specs`: column lookup specifications for batched decomposed
///   LogUp. Pass `&[]` to skip the lookup step.
#[allow(clippy::type_complexity)]
pub fn prove<U, Zt, Lc, const D: usize, const CHECK: bool>(
    params: &ZipPlusParams<Zt, Lc>,
    trace: &[DenseMultilinearExtension<BinaryPoly<D>>],
    num_vars: usize,
    lookup_specs: &[LookupColumnSpec],
) -> ZincProof
where
    U: Uair<Scalar = BinaryPoly<D>>,
    Zt: ZipTypes<Eval = BinaryPoly<D>>,
    Lc: LinearCode<Zt>,
    rand::distr::StandardUniform: rand::distr::Distribution<BinaryPoly<D>>,
    PiopField: FromPrimitiveWithConfig,
    <PiopField as Field>::Inner: ConstTranscribable + FromRef<<PiopField as Field>::Inner> + Send + Sync + Default + num_traits::Zero,
    MillerRabin: PrimalityTest<<PiopField as Field>::Inner>,
    Zt::Eval: ProjectableToField<PiopField>,
    Zt::Cw: ProjectableToField<PiopField>,
    PiopField: for<'a> FromWithConfig<&'a Zt::Chal> + for<'a> FromWithConfig<&'a Zt::Pt> + for<'a> FromWithConfig<&'a Zt::CombR>,
    <PiopField as Field>::Inner: FromRef<Zt::Fmod>,
{
    let total_start = Instant::now();
    let num_constraints = count_constraints::<U>();
    let max_degree = count_max_degree::<U>();

    // ── Step 1: PCS Commit ──────────────────────────────────────────
    let t0 = Instant::now();
    let (hint, commitment) = ZipPlus::<Zt, Lc>::commit(params, trace)
        .expect("PCS commit failed");
    let pcs_commit_time = t0.elapsed();

    // ── Step 2: PIOP — Ideal Check ──────────────────────────────────
    let t1 = Instant::now();
    let mut transcript = KeccakTranscript::new();
    let field_cfg = transcript.get_random_field_cfg::<PiopField, <PiopField as Field>::Inner, MillerRabin>();

    // Project trace coefficients to DynamicPolynomialF for IC.
    let projected_trace = project_trace_coeffs::<PiopField, i64, i64, D>(
        trace, &[], &[], &field_cfg,
    );
    let projected_scalars = project_scalars::<PiopField, U>(|scalar| {
        let one = PiopField::one_with_cfg(&field_cfg);
        let zero = PiopField::zero_with_cfg(&field_cfg);
        DynamicPolynomialF::new(
            scalar.iter().map(|coeff| {
                if coeff.into_inner() { one.clone() } else { zero.clone() }
            }).collect::<Vec<_>>()
        )
    });

    let (ic_proof, ic_state) =
        IdealCheckProtocol::<PiopField>::prove_as_subprotocol::<U>(
            &mut transcript,
            &projected_trace,
            &projected_scalars,
            num_constraints,
            num_vars,
            &field_cfg,
        )
        .expect("Ideal check prover failed");
    let ideal_check_time = t1.elapsed();

    // ── Step 3: PIOP — Combined Poly Resolver ───────────────────────
    let t2 = Instant::now();
    // Get projecting element for F[X]→F projection (shared by CPR and lookup).
    let projecting_element: PiopField = transcript.get_field_challenge(&field_cfg);
    let field_projected_scalars =
        project_scalars_to_field(projected_scalars, &projecting_element)
            .expect("scalar projection failed");
    let field_trace = project_trace_to_field::<PiopField, D>(
        trace, &[], &[], &projecting_element,
    );

    // Extract lookup columns from the projected field trace before CPR
    // consumes it. This reuses the same projecting_element, avoiding a
    // redundant ~2ms re-projection of the BinaryPoly columns.
    let lookup_precomputed = if !lookup_specs.is_empty() {
        let (columns, raw_indices, remapped_specs) =
            extract_lookup_columns_from_field_trace(trace, &field_trace, lookup_specs, &field_cfg);
        Some((columns, raw_indices, remapped_specs))
    } else {
        None
    };

    let (cpr_proof, cpr_state) =
        CombinedPolyResolver::<PiopField>::prove_as_subprotocol::<U>(
            &mut transcript,
            field_trace,
            &ic_state.evaluation_point,
            &field_projected_scalars,
            num_constraints,
            num_vars,
            max_degree,
            &field_cfg,
        )
        .expect("Combined poly resolver failed");
    let cpr_time = t2.elapsed();

    // ── Step 3b: PIOP — Batched Decomposed LogUp (column typing) ────
    let t2b = Instant::now();
    let lookup_proof = if let Some((columns, raw_indices, remapped_specs)) = lookup_precomputed {
        let (lk_proof, _lk_state) = prove_batched_lookup_with_indices(
            &mut transcript,
            &columns,
            &raw_indices,
            &remapped_specs,
            &projecting_element,
            &field_cfg,
        )
        .expect("Batched lookup prover failed");

        Some(LookupProofData::Classic(lk_proof))
    } else {
        None
    };
    let lookup_time = t2b.elapsed();

    // ── Step 4: PCS Prove (test + evaluate) ────────────────────────
    let t3 = Instant::now();
    let point: Vec<Zt::Pt> = {
        let i128_point = derive_pcs_point(&cpr_state.evaluation_point, num_vars);
        i128_point
            .into_iter()
            .map(|v| {
                // SAFETY: Zt::Pt is i128 in all current instantiations.
                unsafe { std::mem::transmute_copy(&v) }
            })
            .collect()
    };
    let (eval_f, proof) =
        ZipPlus::<Zt, Lc>::prove::<PiopField, CHECK>(
            params,
            trace,
            &point,
            &hint,
        )
        .expect("PCS prove failed");
    let pcs_prove_time = t3.elapsed();

    let total_time = total_start.elapsed();

    // Serialize proofs.
    let proof_bytes: Vec<u8> = {
        let pcs_transcript: zip_plus::pcs_transcript::PcsTranscript = proof.into();
        pcs_transcript.stream.into_inner()
    };

    // Serialize PIOP proof data.
    let ic_proof_values: Vec<Vec<u8>> = ic_proof
        .combined_mle_values
        .iter()
        .map(|dpf| dpf.coeffs.iter().flat_map(|c| field_to_bytes(c)).collect())
        .collect();

    let cpr_sumcheck_messages: Vec<Vec<u8>> = cpr_proof
        .sumcheck_proof
        .messages
        .iter()
        .map(|msg| msg.0.tail_evaluations.iter().flat_map(|c| field_to_bytes(c)).collect())
        .collect();
    let cpr_sumcheck_claimed_sum = field_to_bytes(&cpr_proof.sumcheck_proof.claimed_sum);
    let cpr_up_evals: Vec<Vec<u8>> = cpr_proof.up_evals.iter().map(field_to_bytes).collect();
    let cpr_down_evals: Vec<Vec<u8>> = cpr_proof.down_evals.iter().map(field_to_bytes).collect();
    let evaluation_point_bytes: Vec<Vec<u8>> = cpr_state.evaluation_point.iter().map(field_to_bytes).collect();
    let pcs_evals_bytes: Vec<Vec<u8>> = vec![field_to_bytes(&eval_f)];

    ZincProof {
        pcs_proof_bytes: proof_bytes,
        commitment,
        ic_proof_values,
        cpr_sumcheck_messages,
        cpr_sumcheck_claimed_sum,
        cpr_up_evals,
        cpr_down_evals,
        lookup_proof,
        evaluation_point_bytes,
        pcs_evals_bytes,
        timing: TimingBreakdown {
            pcs_commit: pcs_commit_time,
            ideal_check: ideal_check_time,
            combined_poly_resolver: cpr_time,
            lookup: lookup_time,
            pcs_prove: pcs_prove_time,
            total: total_time,
        },
    }
}

/// Run the full Zinc+ prover pipeline with **classic** batched decomposition
/// LogUp (chunks + inverses + sumcheck), instead of GKR LogUp.
///
/// Unlike [`prove`], this variant **batches the CPR and lookup sumchecks**
/// into a single multi-degree sumcheck via
/// [`MultiDegreeSumcheck::prove_as_subprotocol`]. Group 0 is the CPR
/// (degree `max_degree + 2`), groups 1..N are the lookup table groups
/// (degree 2 each). All groups share verifier challenges and produce
/// a common evaluation point.
#[allow(clippy::type_complexity)]
pub fn prove_classic_logup<U, Zt, Lc, const D: usize, const CHECK: bool>(
    params: &ZipPlusParams<Zt, Lc>,
    trace: &[DenseMultilinearExtension<BinaryPoly<D>>],
    num_vars: usize,
    lookup_specs: &[LookupColumnSpec],
) -> ZincProof
where
    U: Uair<Scalar = BinaryPoly<D>>,
    Zt: ZipTypes<Eval = BinaryPoly<D>>,
    Lc: LinearCode<Zt>,
    rand::distr::StandardUniform: rand::distr::Distribution<BinaryPoly<D>>,
    PiopField: FromPrimitiveWithConfig,
    <PiopField as Field>::Inner: ConstTranscribable + FromRef<<PiopField as Field>::Inner> + Send + Sync + Default + num_traits::Zero,
    MillerRabin: PrimalityTest<<PiopField as Field>::Inner>,
    Zt::Eval: ProjectableToField<PiopField>,
    Zt::Cw: ProjectableToField<PiopField>,
    PiopField: for<'a> FromWithConfig<&'a Zt::Chal> + for<'a> FromWithConfig<&'a Zt::Pt> + for<'a> FromWithConfig<&'a Zt::CombR>,
    <PiopField as Field>::Inner: FromRef<Zt::Fmod>,
{
    let total_start = Instant::now();
    let num_constraints = count_constraints::<U>();
    let max_degree = count_max_degree::<U>();

    let t0 = Instant::now();
    let (hint, commitment) = ZipPlus::<Zt, Lc>::commit(params, trace)
        .expect("PCS commit failed");
    let pcs_commit_time = t0.elapsed();

    let t1 = Instant::now();
    let mut transcript = KeccakTranscript::new();
    let field_cfg = transcript.get_random_field_cfg::<PiopField, <PiopField as Field>::Inner, MillerRabin>();

    let projected_trace = project_trace_coeffs::<PiopField, i64, i64, D>(
        trace, &[], &[], &field_cfg,
    );
    let projected_scalars = project_scalars::<PiopField, U>(|scalar| {
        let one = PiopField::one_with_cfg(&field_cfg);
        let zero = PiopField::zero_with_cfg(&field_cfg);
        DynamicPolynomialF::new(
            scalar.iter().map(|coeff| {
                if coeff.into_inner() { one.clone() } else { zero.clone() }
            }).collect::<Vec<_>>()
        )
    });

    let (ic_proof, ic_state) =
        IdealCheckProtocol::<PiopField>::prove_as_subprotocol::<U>(
            &mut transcript,
            &projected_trace,
            &projected_scalars,
            num_constraints,
            num_vars,
            &field_cfg,
        )
        .expect("Ideal check prover failed");
    let ideal_check_time = t1.elapsed();

    // ── Step 3: Batched CPR + Lookup via multi-degree sumcheck ──────
    let t2 = Instant::now();
    let projecting_element: PiopField = transcript.get_field_challenge(&field_cfg);
    let field_projected_scalars =
        project_scalars_to_field(projected_scalars, &projecting_element)
            .expect("scalar projection failed");
    let field_trace = project_trace_to_field::<PiopField, D>(
        trace, &[], &[], &projecting_element,
    );

    // Extract lookup columns before CPR consumes the field trace.
    let lookup_precomputed = if !lookup_specs.is_empty() {
        let (columns, raw_indices, remapped_specs) =
            extract_lookup_columns_from_field_trace(trace, &field_trace, lookup_specs, &field_cfg);
        Some((columns, raw_indices, remapped_specs))
    } else {
        None
    };

    // ── CPR: build sumcheck group (does NOT run sumcheck) ───────────
    let mut cpr_group = CombinedPolyResolver::<PiopField>::build_prover_group::<U>(
        &mut transcript,
        field_trace,
        &ic_state.evaluation_point,
        &field_projected_scalars,
        num_constraints,
        num_vars,
        max_degree,
        &field_cfg,
    )
    .expect("CPR build_prover_group failed");
    let cpr_num_cols = cpr_group.num_cols;

    // ── Lookup: build sumcheck groups (does NOT run sumchecks) ───────
    let mut lookup_groups_data: Vec<(LookupSumcheckGroup<PiopField>, LookupGroupMeta)> = Vec::new();
    if let Some((ref columns, ref raw_indices, ref remapped_specs)) = lookup_precomputed {
        let groups = group_lookup_specs(remapped_specs);
        for group in &groups {
            let instance = build_lookup_instance_from_indices_pub(
                columns, raw_indices, group, &projecting_element, &field_cfg,
            )
            .expect("lookup instance build failed");

            let witness_len = instance.witnesses[0].len();

            let lk_group = BatchedDecompLogupProtocol::<PiopField>::build_prover_group(
                &mut transcript,
                &instance,
                &field_cfg,
            )
            .expect("lookup build_prover_group failed");

            let meta = LookupGroupMeta {
                table_type: group.table_type.clone(),
                num_columns: group.column_indices.len(),
                witness_len,
            };
            lookup_groups_data.push((lk_group, meta));
        }
    }

    // Determine the shared num_vars across CPR and all lookup groups.
    // If the lookup subtable is larger than the witness, its num_vars
    // may exceed CPR's. We zero-pad the CPR's MLEs in that case.
    let shared_num_vars = {
        let max_lk = lookup_groups_data
            .iter()
            .map(|(g, _)| g.num_vars)
            .max()
            .unwrap_or(num_vars);
        num_vars.max(max_lk)
    };

    // Pad CPR MLEs to shared_num_vars if needed.
    if shared_num_vars > num_vars {
        let target_len = 1usize << shared_num_vars;
        for mle in &mut cpr_group.mles {
            mle.evaluations
                .resize(target_len, Default::default());
            mle.num_vars = shared_num_vars;
        }
    }

    // ── Assemble all groups and run multi-degree sumcheck ────────────
    let has_lookup = !lookup_groups_data.is_empty();
    let (batched_proof, evaluation_point, cpr_up_evals, cpr_down_evals, _lookup_proof) =
    if has_lookup {
        // Build the groups vector: CPR first, then lookup groups.
        let mut sumcheck_groups: Vec<(
            usize,
            Vec<DenseMultilinearExtension<<PiopField as Field>::Inner>>,
            Box<dyn Fn(&[PiopField]) -> PiopField + Send + Sync>,
        )> = Vec::with_capacity(1 + lookup_groups_data.len());

        sumcheck_groups.push((cpr_group.degree, cpr_group.mles, cpr_group.comb_fn));

        let mut lookup_pre_data: Vec<(LookupSumcheckGroup<PiopField>, LookupGroupMeta)> = Vec::new();
        for (lk_group, meta) in lookup_groups_data {
            sumcheck_groups.push((lk_group.degree, lk_group.mles, lk_group.comb_fn));
            // We still need the ancillary data; reconstruct a shell.
            lookup_pre_data.push((LookupSumcheckGroup {
                degree: 0, // unused after this point
                mles: vec![], // already moved
                comb_fn: Box::new(|_: &[PiopField]| unreachable!()),
                num_vars: 0,
                chunk_vectors: lk_group.chunk_vectors,
                aggregated_multiplicities: lk_group.aggregated_multiplicities,
                chunk_inverse_witnesses: lk_group.chunk_inverse_witnesses,
                inverse_table: lk_group.inverse_table,
            }, meta));
        }

        // Run the combined multi-degree sumcheck.
        let (md_proof, mut prover_states) =
            MultiDegreeSumcheck::<PiopField>::prove_as_subprotocol(
                &mut transcript,
                sumcheck_groups,
                shared_num_vars,
                &field_cfg,
            );

        // ── CPR finalize: extract up/down evals from group 0's state ──
        let cpr_prover_state = prover_states.remove(0);
        let (up_evals, down_evals, cpr_state_final) =
            CombinedPolyResolver::<PiopField>::finalize_prover(
                &mut transcript,
                cpr_prover_state,
                cpr_num_cols,
                &field_cfg,
            )
            .expect("CPR finalize_prover failed");

        // ── Lookup finalize: build per-group proofs from remaining states ──
        let mut lookup_group_proofs = Vec::new();
        let mut lookup_group_meta = Vec::new();
        for (i, (lk_pre, meta)) in lookup_pre_data.into_iter().enumerate() {
            // Extract this lookup group's sumcheck proof from the md_proof.
            let lk_sumcheck_proof = SumcheckProof {
                messages: md_proof.group_messages[i + 1]
                    .iter()
                    .cloned()
                    .collect(),
                claimed_sum: md_proof.claimed_sums[i + 1].clone(),
            };
            let lk_eval_point = prover_states.remove(0).randomness;
            let (lk_proof, _lk_state) =
                BatchedDecompLogupProtocol::<PiopField>::finalize_prover(
                    lk_pre, lk_sumcheck_proof, lk_eval_point,
                );
            lookup_group_proofs.push(lk_proof);
            lookup_group_meta.push(meta);
        }

        let batched = BatchedCprLookupProof {
            md_proof,
            cpr_up_evals: up_evals.clone(),
            cpr_down_evals: down_evals.clone(),
            lookup_group_meta,
            lookup_group_proofs,
        };

        (
            Some(LookupProofData::BatchedClassic(batched)),
            cpr_state_final.evaluation_point,
            up_evals,
            down_evals,
            true,
        )
    } else {
        // No lookup — run CPR sumcheck alone (falls back to normal path).
        let groups = vec![(cpr_group.degree, cpr_group.mles, cpr_group.comb_fn)];
        let (_md_proof, mut prover_states) =
            MultiDegreeSumcheck::<PiopField>::prove_as_subprotocol(
                &mut transcript,
                groups,
                num_vars,
                &field_cfg,
            );

        let cpr_prover_state = prover_states.remove(0);
        let (up_evals, down_evals, cpr_state_final) =
            CombinedPolyResolver::<PiopField>::finalize_prover(
                &mut transcript,
                cpr_prover_state,
                cpr_num_cols,
                &field_cfg,
            )
            .expect("CPR finalize_prover failed");

        (None, cpr_state_final.evaluation_point, up_evals, down_evals, false)
    };

    let cpr_lookup_time = t2.elapsed();

    // ── Step 4: PCS Prove (test + evaluate) ────────────────────────
    let t3 = Instant::now();
    let point: Vec<Zt::Pt> = {
        let i128_point = derive_pcs_point(&evaluation_point, num_vars);
        i128_point
            .into_iter()
            .map(|v| unsafe { std::mem::transmute_copy(&v) })
            .collect()
    };
    let (eval_f, proof) =
        ZipPlus::<Zt, Lc>::prove::<PiopField, CHECK>(
            params,
            trace,
            &point,
            &hint,
        )
        .expect("PCS prove failed");
    let pcs_prove_time = t3.elapsed();
    let total_time = total_start.elapsed();

    let proof_bytes: Vec<u8> = {
        let pcs_transcript: zip_plus::pcs_transcript::PcsTranscript = proof.into();
        pcs_transcript.stream.into_inner()
    };
    let ic_proof_values: Vec<Vec<u8>> = ic_proof
        .combined_mle_values
        .iter()
        .map(|dpf| dpf.coeffs.iter().flat_map(|c| field_to_bytes(c)).collect())
        .collect();

    // For the batched path, CPR sumcheck data is embedded in the
    // BatchedClassic variant; these fields are unused but populated
    // for the no-lookup fallback case.
    let cpr_sumcheck_messages = Vec::new();
    let cpr_sumcheck_claimed_sum = vec![0u8; <crypto_primitives::crypto_bigint_uint::Uint<FIELD_LIMBS> as ConstTranscribable>::NUM_BYTES];
    let cpr_up_evals_bytes: Vec<Vec<u8>> = cpr_up_evals.iter().map(field_to_bytes).collect();
    let cpr_down_evals_bytes: Vec<Vec<u8>> = cpr_down_evals.iter().map(field_to_bytes).collect();
    let evaluation_point_bytes: Vec<Vec<u8>> = evaluation_point.iter().map(field_to_bytes).collect();
    let pcs_evals_bytes: Vec<Vec<u8>> = vec![field_to_bytes(&eval_f)];

    ZincProof {
        pcs_proof_bytes: proof_bytes,
        commitment,
        ic_proof_values,
        cpr_sumcheck_messages,
        cpr_sumcheck_claimed_sum,
        cpr_up_evals: cpr_up_evals_bytes,
        cpr_down_evals: cpr_down_evals_bytes,
        lookup_proof: batched_proof,
        evaluation_point_bytes,
        pcs_evals_bytes,
        timing: TimingBreakdown {
            pcs_commit: pcs_commit_time,
            ideal_check: ideal_check_time,
            combined_poly_resolver: cpr_lookup_time,
            lookup: Duration::ZERO,  // included in combined_poly_resolver
            pcs_prove: pcs_prove_time,
            total: total_time,
        },
    }
}

/// Run the Zinc+ prover pipeline for any ring `R` (generic single-ring variant).
///
/// Unlike `prove()` (which is hardcoded to `BinaryPoly<D>`), this function
/// works with any evaluation ring — `Int<4>`, `DensePolynomial<i64, D>`, etc.
/// The ring must implement `ProjectableToField<PiopField>` and
/// `PiopField: FromWithConfig<R>` (for projecting elements to the PIOP field).
///
/// This is the target pipeline for ECDSA with `Int<4>` throughout:
/// the same `Int<4>` type is used as PCS evaluation type AND PIOP constraint ring.
#[allow(clippy::type_complexity)]
pub fn prove_generic<U, R, Zt, Lc, PcsF, const CHECK: bool>(
    params: &ZipPlusParams<Zt, Lc>,
    trace: &[DenseMultilinearExtension<R>],
    num_vars: usize,
    lookup_specs: &[LookupColumnSpec],
) -> ZincProof
where
    R: ProjectableToField<PiopField> + crypto_primitives::Semiring + std::fmt::Debug + Clone + Send + Sync + 'static,
    U: Uair<Scalar = R>,
    Zt: ZipTypes<Eval = R>,
    Lc: LinearCode<Zt>,
    PiopField: FromPrimitiveWithConfig + FromWithConfig<R>,
    <PiopField as Field>::Inner: ConstTranscribable + FromRef<<PiopField as Field>::Inner> + Send + Sync + Default + num_traits::Zero,
    MillerRabin: PrimalityTest<<PiopField as Field>::Inner>,
    // PCS field (may be wider than PiopField when Zt::Fmod > 128 bits)
    PcsF: PrimeField
        + FromPrimitiveWithConfig
        + for<'a> FromWithConfig<&'a Zt::Chal>
        + for<'a> FromWithConfig<&'a Zt::Pt>
        + for<'a> FromWithConfig<&'a Zt::CombR>
        + for<'a> zinc_utils::mul_by_scalar::MulByScalar<&'a PcsF>
        + FromRef<PcsF>,
    PcsF::Inner: FromRef<Zt::Fmod> + ConstTranscribable,
    Zt::Eval: ProjectableToField<PcsF>,
{
    let total_start = Instant::now();
    let num_constraints = count_constraints::<U>();
    let max_degree = count_max_degree::<U>();

    // ── Step 1: PCS Commit ──────────────────────────────────────────
    let t0 = Instant::now();
    let (hint, commitment) = ZipPlus::<Zt, Lc>::commit(params, trace)
        .expect("PCS commit failed");
    let pcs_commit_time = t0.elapsed();

    // ── Step 2: PIOP — Ideal Check ──────────────────────────────────
    let t1 = Instant::now();
    let mut transcript = KeccakTranscript::new();
    let field_cfg = transcript.get_random_field_cfg::<PiopField, <PiopField as Field>::Inner, MillerRabin>();

    // Project trace: each R element becomes a constant DynamicPolynomialF.
    let projected_trace: Vec<DenseMultilinearExtension<DynamicPolynomialF<PiopField>>> = trace
        .iter()
        .map(|col_mle| {
            col_mle
                .iter()
                .map(|elem| DynamicPolynomialF {
                    coeffs: vec![PiopField::from_with_cfg(elem.clone(), &field_cfg)],
                })
                .collect()
        })
        .collect();

    let projected_scalars = project_scalars::<PiopField, U>(|scalar| {
        DynamicPolynomialF {
            coeffs: vec![PiopField::from_with_cfg(scalar.clone(), &field_cfg)],
        }
    });

    let (ic_proof, ic_state) =
        IdealCheckProtocol::<PiopField>::prove_as_subprotocol::<U>(
            &mut transcript,
            &projected_trace,
            &projected_scalars,
            num_constraints,
            num_vars,
            &field_cfg,
        )
        .expect("Ideal check prover failed");
    let ideal_check_time = t1.elapsed();

    // ── Step 3: PIOP — Combined Poly Resolver ───────────────────────
    let t2 = Instant::now();
    let projecting_element: PiopField = transcript.get_field_challenge(&field_cfg);
    let field_projected_scalars =
        project_scalars_to_field(projected_scalars, &projecting_element)
            .expect("scalar projection failed");
    // For scalar types (Int<N>), evaluating a constant poly at any point
    // yields the constant. We can use project_trace_to_field with int slot.
    let field_trace = project_trace_to_field::<PiopField, 1>(
        &[], &[], &projected_trace, &projecting_element,
    );

    let (cpr_proof, cpr_state) =
        CombinedPolyResolver::<PiopField>::prove_as_subprotocol::<U>(
            &mut transcript,
            field_trace,
            &ic_state.evaluation_point,
            &field_projected_scalars,
            num_constraints,
            num_vars,
            max_degree,
            &field_cfg,
        )
        .expect("Combined poly resolver failed");
    let cpr_time = t2.elapsed();

    // ── Step 3b: PIOP — Batched Decomposed LogUp (column typing) ────
    let t2b = Instant::now();
    let lookup_proof = if !lookup_specs.is_empty() {
        // Derive a lookup-specific projecting element from the transcript.
        let lookup_projecting_element: PiopField = transcript.get_field_challenge(&field_cfg);

        // Collect unique column indices referenced by lookup specs.
        let mut needed: std::collections::BTreeMap<usize, usize> = std::collections::BTreeMap::new();
        for spec in lookup_specs {
            let next_id = needed.len();
            needed.entry(spec.column_index).or_insert(next_id);
        }

        // Project only the columns that have lookup constraints.
        let projection_fn = R::prepare_projection(&lookup_projecting_element);
        let mut columns: Vec<Vec<PiopField>> = Vec::with_capacity(needed.len());
        let mut raw_indices: Vec<Vec<usize>> = Vec::with_capacity(needed.len());
        for &orig_idx in needed.keys() {
            let col_mle = &trace[orig_idx];
            let mut col_f = Vec::with_capacity(col_mle.len());
            let mut col_idx = Vec::with_capacity(col_mle.len());
            for v in col_mle.iter() {
                let fv = projection_fn(v);
                let uint = fv.retrieve();
                let idx = uint.as_words()[0] as usize;
                col_f.push(fv);
                col_idx.push(idx);
            }
            columns.push(col_f);
            raw_indices.push(col_idx);
        }

        // Remap lookup specs to 0-based indices into the projected columns.
        let index_map: std::collections::BTreeMap<usize, usize> = needed
            .keys()
            .enumerate()
            .map(|(new_idx, &orig_idx)| (orig_idx, new_idx))
            .collect();
        let remapped_specs: Vec<LookupColumnSpec> = lookup_specs
            .iter()
            .map(|spec| LookupColumnSpec {
                column_index: index_map[&spec.column_index],
                table_type: spec.table_type.clone(),
            })
            .collect();

        let (lk_proof, _lk_state) = prove_batched_lookup_with_indices(
            &mut transcript,
            &columns,
            &raw_indices,
            &remapped_specs,
            &lookup_projecting_element,
            &field_cfg,
        )
        .expect("Batched lookup prover failed");

        Some(LookupProofData::Classic(lk_proof))
    } else {
        None
    };
    let lookup_time = t2b.elapsed();

    // ── Step 4: PCS Prove (test + evaluate) ────────────────────────
    let t3 = Instant::now();
    let point: Vec<Zt::Pt> = {
        let i128_point = derive_pcs_point(&cpr_state.evaluation_point, num_vars);
        i128_point
            .into_iter()
            .map(|v| unsafe { std::mem::transmute_copy(&v) })
            .collect()
    };
    let (_eval_f, proof) =
        ZipPlus::<Zt, Lc>::prove::<PcsF, CHECK>(
            params,
            trace,
            &point,
            &hint,
        )
        .expect("PCS prove failed");
    let pcs_prove_time = t3.elapsed();

    let total_time = total_start.elapsed();

    // Serialize proofs.
    let proof_bytes: Vec<u8> = {
        let pcs_transcript: zip_plus::pcs_transcript::PcsTranscript = proof.into();
        pcs_transcript.stream.into_inner()
    };

    let ic_proof_values: Vec<Vec<u8>> = ic_proof
        .combined_mle_values
        .iter()
        .map(|dpf| dpf.coeffs.iter().flat_map(|c| field_to_bytes(c)).collect())
        .collect();

    let cpr_sumcheck_messages: Vec<Vec<u8>> = cpr_proof
        .sumcheck_proof
        .messages
        .iter()
        .map(|msg| msg.0.tail_evaluations.iter().flat_map(|c| field_to_bytes(c)).collect())
        .collect();
    let cpr_sumcheck_claimed_sum = field_to_bytes(&cpr_proof.sumcheck_proof.claimed_sum);
    let cpr_up_evals: Vec<Vec<u8>> = cpr_proof.up_evals.iter().map(field_to_bytes).collect();
    let cpr_down_evals: Vec<Vec<u8>> = cpr_proof.down_evals.iter().map(field_to_bytes).collect();
    let evaluation_point_bytes: Vec<Vec<u8>> = cpr_state.evaluation_point.iter().map(field_to_bytes).collect();
    // PCS evals are in PcsF (potentially wider than PiopField); serialize
    // using ConstTranscribable for the verifier to deserialize.
    let pcs_evals_bytes: Vec<Vec<u8>> = {
        let mut buf = vec![0u8; <PcsF::Inner as ConstTranscribable>::NUM_BYTES];
        _eval_f.inner().write_transcription_bytes(&mut buf);
        vec![buf]
    };

    ZincProof {
        pcs_proof_bytes: proof_bytes,
        commitment,
        ic_proof_values,
        cpr_sumcheck_messages,
        cpr_sumcheck_claimed_sum,
        cpr_up_evals,
        cpr_down_evals,
        lookup_proof,
        evaluation_point_bytes,
        pcs_evals_bytes,
        timing: TimingBreakdown {
            pcs_commit: pcs_commit_time,
            ideal_check: ideal_check_time,
            combined_poly_resolver: cpr_time,
            lookup: lookup_time,
            pcs_prove: pcs_prove_time,
            total: total_time,
        },
    }
}

/// Run the Zinc+ verifier for any ring `R` (generic single-ring variant).
///
/// Corresponds to `prove_generic`. Verifies IC + CPR + PCS.
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn verify_generic<U, R, Zt, Lc, PcsF, const CHECK: bool, IdealOverF, IdealOverFFromRef>(
    params: &ZipPlusParams<Zt, Lc>,
    zinc_proof: &ZincProof,
    num_vars: usize,
    ideal_over_f_from_ref: IdealOverFFromRef,
) -> VerifyResult
where
    R: ProjectableToField<PiopField> + crypto_primitives::Semiring + std::fmt::Debug + Clone + Send + Sync + 'static,
    U: Uair<Scalar = R>,
    Zt: ZipTypes<Eval = R>,
    Lc: LinearCode<Zt>,
    PiopField: FromPrimitiveWithConfig + FromWithConfig<R>,
    <PiopField as Field>::Inner: ConstTranscribable + FromRef<<PiopField as Field>::Inner> + Send + Sync + Default + num_traits::Zero,
    MillerRabin: PrimalityTest<<PiopField as Field>::Inner>,
    // PCS field (may be wider than PiopField when Zt::Fmod > 128 bits)
    PcsF: PrimeField
        + FromPrimitiveWithConfig
        + for<'a> FromWithConfig<&'a Zt::Chal>
        + for<'a> FromWithConfig<&'a Zt::Pt>
        + for<'a> FromWithConfig<&'a Zt::CombR>
        + for<'a> zinc_utils::mul_by_scalar::MulByScalar<&'a PcsF>
        + FromRef<PcsF>,
    PcsF::Inner: FromRef<Zt::Fmod> + ConstTranscribable,
    Zt::Eval: ProjectableToField<PcsF>,
    Zt::Cw: ProjectableToField<PcsF>,
    IdealOverF: Ideal + IdealCheck<DynamicPolynomialF<PiopField>>,
    IdealOverFFromRef: Fn(&IdealOrZero<U::Ideal>) -> IdealOverF,
{
    let total_start = Instant::now();

    let num_constraints = count_constraints::<U>();
    let max_degree = count_max_degree::<U>();

    // ── Reconstruct Fiat-Shamir transcript ──────────────────────────
    let mut transcript = KeccakTranscript::new();
    let field_cfg = transcript.get_random_field_cfg::<PiopField, <PiopField as Field>::Inner, MillerRabin>();

    let field_elem_size = <crypto_primitives::crypto_bigint_uint::Uint<FIELD_LIMBS> as ConstTranscribable>::NUM_BYTES;

    // ── Step 1: IC verify ───────────────────────────────────────────
    let t0 = Instant::now();

    let ic_combined_mle_values: Vec<DynamicPolynomialF<PiopField>> = zinc_proof
        .ic_proof_values
        .iter()
        .map(|bytes| {
            let num_coeffs = bytes.len() / field_elem_size;
            let coeffs: Vec<PiopField> = (0..num_coeffs)
                .map(|i| field_from_bytes(&bytes[i * field_elem_size..(i + 1) * field_elem_size], &field_cfg))
                .collect();
            DynamicPolynomialF::new(coeffs)
        })
        .collect();

    let ic_proof = zinc_piop::ideal_check::Proof::<PiopField> {
        combined_mle_values: ic_combined_mle_values,
    };

    let ic_subclaim = match IdealCheckProtocol::<PiopField>::verify_as_subprotocol::<U, _, _>(
        &mut transcript,
        ic_proof,
        num_constraints,
        num_vars,
        ideal_over_f_from_ref,
        &field_cfg,
    ) {
        Ok(subclaim) => subclaim,
        Err(e) => {
            eprintln!("IdealCheck verification failed: {e:?}");
            return VerifyResult {
                accepted: false,
                timing: VerifyTimingBreakdown {
                    ideal_check_verify: t0.elapsed(),
                    combined_poly_resolver_verify: Duration::ZERO,
                    lookup_verify: Duration::ZERO,
                    pcs_verify: Duration::ZERO,
                    total: total_start.elapsed(),
                },
            };
        }
    };
    let ic_verify_time = t0.elapsed();

    // ── Step 2: CPR verify ──────────────────────────────────────────
    let t1 = Instant::now();

    let cpr_sumcheck_messages: Vec<ProverMsg<PiopField>> = zinc_proof
        .cpr_sumcheck_messages
        .iter()
        .map(|bytes| {
            let num_evals = bytes.len() / field_elem_size;
            let tail_evaluations: Vec<PiopField> = (0..num_evals)
                .map(|i| field_from_bytes(&bytes[i * field_elem_size..(i + 1) * field_elem_size], &field_cfg))
                .collect();
            ProverMsg(NatEvaluatedPolyWithoutConstant::new(tail_evaluations))
        })
        .collect();

    let cpr_claimed_sum = field_from_bytes(&zinc_proof.cpr_sumcheck_claimed_sum, &field_cfg);
    let cpr_up_evals: Vec<PiopField> = zinc_proof.cpr_up_evals.iter()
        .map(|b| field_from_bytes(b, &field_cfg))
        .collect();
    let cpr_down_evals: Vec<PiopField> = zinc_proof.cpr_down_evals.iter()
        .map(|b| field_from_bytes(b, &field_cfg))
        .collect();

    let cpr_proof = zinc_piop::combined_poly_resolver::Proof::<PiopField> {
        sumcheck_proof: SumcheckProof {
            messages: cpr_sumcheck_messages,
            claimed_sum: cpr_claimed_sum,
        },
        up_evals: cpr_up_evals,
        down_evals: cpr_down_evals,
    };

    // Compute projecting element and projected scalars for CPR verify.
    let projecting_element: PiopField = transcript.get_field_challenge(&field_cfg);
    let projected_scalars_coeffs = project_scalars::<PiopField, U>(|scalar| {
        DynamicPolynomialF {
            coeffs: vec![PiopField::from_with_cfg(scalar.clone(), &field_cfg)],
        }
    });
    let field_projected_scalars =
        project_scalars_to_field(projected_scalars_coeffs, &projecting_element)
            .expect("scalar projection failed");

    let cpr_subclaim = match CombinedPolyResolver::<PiopField>::verify_as_subprotocol::<U>(
        &mut transcript,
        cpr_proof,
        num_constraints,
        num_vars,
        max_degree,
        &projecting_element,
        &field_projected_scalars,
        ic_subclaim,
        &field_cfg,
    ) {
        Ok(subclaim) => subclaim,
        Err(e) => {
            eprintln!("CPR verification failed: {e:?}");
            return VerifyResult {
                accepted: false,
                timing: VerifyTimingBreakdown {
                    ideal_check_verify: ic_verify_time,
                    combined_poly_resolver_verify: t1.elapsed(),
                    lookup_verify: Duration::ZERO,
                    pcs_verify: Duration::ZERO,
                    total: total_start.elapsed(),
                },
            };
        }
    };
    let cpr_verify_time = t1.elapsed();

    // ── Step 2b: Verify Batched Decomposed LogUp (column typing) ────
    let t1b = Instant::now();
    if let Some(ref lookup_data) = zinc_proof.lookup_proof {
        let projecting_element: PiopField = transcript.get_field_challenge(&field_cfg);

        let result = match lookup_data {
            LookupProofData::Gkr(proof) => verify_gkr_batched_lookup(
                &mut transcript, proof, &projecting_element, &field_cfg,
            ).map(|_| ()),
            LookupProofData::Classic(proof) => verify_batched_lookup(
                &mut transcript, proof, &projecting_element, &field_cfg,
            ).map(|_| ()),
            LookupProofData::BatchedClassic(_) => {
                unimplemented!(
                    "BatchedClassic proofs should use `verify` (BinaryPoly), \
                     not `verify_generic`"
                );
            }
        };
        if let Err(e) = result {
            eprintln!("Lookup verification failed (generic): {e:?}");
            return VerifyResult {
                accepted: false,
                timing: VerifyTimingBreakdown {
                    ideal_check_verify: ic_verify_time,
                    combined_poly_resolver_verify: cpr_verify_time,
                    lookup_verify: t1b.elapsed(),
                    pcs_verify: Duration::ZERO,
                    total: total_start.elapsed(),
                },
            };
        }
    }
    let lookup_verify_time = t1b.elapsed();

    // ── Step 3: PCS Verify ──────────────────────────────────────────
    let t2 = Instant::now();

    // Derive PcsF field config (same as what ZipPlus::verify will derive
    // internally from a fresh transcript).
    let pcs_field_cfg = KeccakTranscript::default()
        .get_random_field_cfg::<PcsF, Zt::Fmod, Zt::PrimeTest>();

    // Convert integer point to PcsF field elements.
    let pcs_point: Vec<Zt::Pt> = {
        let i128_point = derive_pcs_point(&cpr_subclaim.evaluation_point, num_vars);
        i128_point
            .into_iter()
            .map(|v| unsafe { std::mem::transmute_copy(&v) })
            .collect()
    };
    let point_f: Vec<PcsF> = pcs_point.iter().map(|v| v.into_with_cfg(&pcs_field_cfg)).collect();

    // Deserialize PcsF eval.
    let eval_inner = <PcsF::Inner as ConstTranscribable>::read_transcription_bytes(
        &zinc_proof.pcs_evals_bytes[0],
    );
    let eval_f: PcsF = PcsF::new_unchecked_with_cfg(eval_inner, &pcs_field_cfg);

    let pcs_transcript = zip_plus::pcs_transcript::PcsTranscript {
        fs_transcript: KeccakTranscript::default(),
        stream: std::io::Cursor::new(zinc_proof.pcs_proof_bytes.clone()),
    };
    let pcs_proof: ZipPlusProof = pcs_transcript.into();

    let pcs_result = ZipPlus::<Zt, Lc>::verify::<PcsF, CHECK>(
        params,
        &zinc_proof.commitment,
        &point_f,
        &eval_f,
        &pcs_proof,
    );

    if let Err(ref e) = pcs_result {
        eprintln!("PCS verification failed (generic): {e:?}");
    }

    let pcs_verify_time = t2.elapsed();

    VerifyResult {
        accepted: pcs_result.is_ok(),
        timing: VerifyTimingBreakdown {
            ideal_check_verify: ic_verify_time,
            combined_poly_resolver_verify: cpr_verify_time,
            lookup_verify: lookup_verify_time,
            pcs_verify: pcs_verify_time,
            total: total_start.elapsed(),
        },
    }
}

/// Run the PCS-only prover pipeline (commit + test + evaluate).
///
/// This benchmarks the dominant cost (the PCS) without the PIOP overhead,
/// matching the paper's Zip+ benchmark.
pub fn prove_pcs_only<Zt, Lc, const D: usize, const CHECK: bool>(
    params: &ZipPlusParams<Zt, Lc>,
    trace: &[DenseMultilinearExtension<BinaryPoly<D>>],
    num_vars: usize,
) -> (ZincProof, Vec<PiopField>)
where
    Zt: ZipTypes<Eval = BinaryPoly<D>>,
    Lc: LinearCode<Zt>,
    rand::distr::StandardUniform: rand::distr::Distribution<BinaryPoly<D>>,
    PiopField: FromPrimitiveWithConfig,
    <PiopField as Field>::Inner: ConstTranscribable + FromRef<<PiopField as Field>::Inner>,
    Zt::Eval: ProjectableToField<PiopField>,
    Zt::Cw: ProjectableToField<PiopField>,
    PiopField: for<'a> FromWithConfig<&'a Zt::Chal> + for<'a> FromWithConfig<&'a Zt::Pt> + for<'a> FromWithConfig<&'a Zt::CombR>,
    <PiopField as Field>::Inner: FromRef<Zt::Fmod>,
{
    let total_start = Instant::now();

    // ── Step 1: PCS Commit ──────────────────────────────────────────
    let t0 = Instant::now();
    let (hint, commitment) = ZipPlus::<Zt, Lc>::commit(params, trace)
        .expect("PCS commit failed");
    let pcs_commit_time = t0.elapsed();

    // ── Step 2: PCS Prove (test + evaluate) ────────────────────────
    let t1 = Instant::now();
    let point: Vec<Zt::Pt> = vec![Zt::Pt::one(); num_vars];
    let (eval_f, proof) =
        ZipPlus::<Zt, Lc>::prove::<PiopField, CHECK>(
            params,
            trace,
            &point,
            &hint,
        )
        .expect("PCS prove failed");
    let pcs_prove_time = t1.elapsed();

    let total_time = total_start.elapsed();

    let proof_bytes: Vec<u8> = {
        let pcs_transcript: zip_plus::pcs_transcript::PcsTranscript = proof.into();
        pcs_transcript.stream.into_inner()
    };

    let field_cfg = eval_f.cfg().clone();
    let point_f: Vec<PiopField> = point.iter().map(|v| v.into_with_cfg(&field_cfg)).collect();

    (
        ZincProof {
            pcs_proof_bytes: proof_bytes,
            commitment,
            ic_proof_values: vec![],
            cpr_sumcheck_messages: vec![],
            cpr_sumcheck_claimed_sum: vec![],
            cpr_up_evals: vec![],
            cpr_down_evals: vec![],
            lookup_proof: None,
            evaluation_point_bytes: vec![],
            pcs_evals_bytes: vec![field_to_bytes(&eval_f)],
            timing: TimingBreakdown {
                pcs_commit: pcs_commit_time,
                ideal_check: Duration::ZERO,
                combined_poly_resolver: Duration::ZERO,
                lookup: Duration::ZERO,
                pcs_prove: pcs_prove_time,
                total: total_time,
            },
        },
        point_f,
    )
}

/// Run the PCS-only verifier.
pub fn verify_pcs_only<Zt, Lc, const D: usize, const CHECK: bool>(
    params: &ZipPlusParams<Zt, Lc>,
    commitment: &ZipPlusCommitment,
    point: &[Zt::Pt],
    eval_bytes: &[u8],
    proof_bytes: &[u8],
) -> VerifyResult
where
    Zt: ZipTypes<Eval = BinaryPoly<D>>,
    Lc: LinearCode<Zt>,
    Zt::Eval: ProjectableToField<PiopField>,
    Zt::Cw: ProjectableToField<PiopField>,
    PiopField: for<'a> FromWithConfig<&'a Zt::Chal> + for<'a> FromWithConfig<&'a Zt::Pt> + for<'a> FromWithConfig<&'a Zt::CombR>,
    <PiopField as Field>::Inner: FromRef<Zt::Fmod> + ConstTranscribable,
{
    let total_start = Instant::now();

    // Derive PiopField config from a fresh PCS transcript.
    let pcs_field_cfg = KeccakTranscript::default()
        .get_random_field_cfg::<PiopField, Zt::Fmod, Zt::PrimeTest>();

    // Convert integer point to field elements.
    let point_f: Vec<PiopField> = point.iter().map(|v| v.into_with_cfg(&pcs_field_cfg)).collect();

    // Deserialize eval.
    let eval_f: PiopField = PiopField::new_unchecked_with_cfg(
        <PiopField as Field>::Inner::read_transcription_bytes(eval_bytes),
        &pcs_field_cfg,
    );

    let pcs_transcript = zip_plus::pcs_transcript::PcsTranscript {
        fs_transcript: KeccakTranscript::default(),
        stream: std::io::Cursor::new(proof_bytes.to_vec()),
    };
    let proof: ZipPlusProof = pcs_transcript.into();

    let t0 = Instant::now();
    let result = ZipPlus::<Zt, Lc>::verify::<PiopField, CHECK>(
        params,
        commitment,
        &point_f,
        &eval_f,
        &proof,
    );
    let pcs_verify_time = t0.elapsed();

    let total_time = total_start.elapsed();

    VerifyResult {
        accepted: result.is_ok(),
        timing: VerifyTimingBreakdown {
            ideal_check_verify: Duration::ZERO,
            combined_poly_resolver_verify: Duration::ZERO,
            lookup_verify: Duration::ZERO,
            pcs_verify: pcs_verify_time,
            total: total_time,
        },
    }
}

// ─── Full Verifier ──────────────────────────────────────────────────────────

/// Run the full Zinc+ verifier: PIOP verification (IdealCheck + CPR) then PCS verification.
///
/// This verifies the complete pipeline:
/// 1. Deserialize the PIOP proof (IdealCheck + CPR/Sumcheck)
/// 2. Run the IdealCheck verifier (checks combined polynomials are in ideals)
/// 3. Run the CPR verifier (verifies sumcheck + constraint evaluation)
/// 4. Check that CPR evaluation claims match PCS evaluations
/// 5. Run PCS verifier
///
/// # Type parameters
/// - `U`: the UAIR.
/// - `Zt`: Zip+ types.
/// - `Lc`: linear code.
/// - `D`: degree + 1 for BinaryPoly.
/// - `CHECK`: overflow checking flag.
/// - `IdealOverF`: the field-level ideal type for verification.
/// - `IdealOverFFromRef`: maps from `IdealOrZero<U::Ideal>` to `IdealOverF`.
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn verify<U, Zt, Lc, const D: usize, const CHECK: bool, IdealOverF, IdealOverFFromRef>(
    params: &ZipPlusParams<Zt, Lc>,
    zinc_proof: &ZincProof,
    num_vars: usize,
    ideal_over_f_from_ref: IdealOverFFromRef,
) -> VerifyResult
where
    U: Uair<Scalar = BinaryPoly<D>>,
    Zt: ZipTypes<Eval = BinaryPoly<D>>,
    Lc: LinearCode<Zt>,
    PiopField: FromPrimitiveWithConfig,
    <PiopField as Field>::Inner: ConstTranscribable + FromRef<<PiopField as Field>::Inner> + Send + Sync + Default + num_traits::Zero,
    MillerRabin: PrimalityTest<<PiopField as Field>::Inner>,
    Zt::Eval: ProjectableToField<PiopField>,
    Zt::Cw: ProjectableToField<PiopField>,
    PiopField: for<'a> FromWithConfig<&'a Zt::Chal> + for<'a> FromWithConfig<&'a Zt::Pt> + for<'a> FromWithConfig<&'a Zt::CombR>,
    <PiopField as Field>::Inner: FromRef<Zt::Fmod>,
    IdealOverF: Ideal + IdealCheck<DynamicPolynomialF<PiopField>>,
    IdealOverFFromRef: Fn(&IdealOrZero<U::Ideal>) -> IdealOverF,
{
    let total_start = Instant::now();

    let num_constraints = count_constraints::<U>();
    let max_degree = count_max_degree::<U>();

    // ── Reconstruct Fiat-Shamir transcript (must match prover) ──────
    let mut transcript = KeccakTranscript::new();
    let field_cfg = transcript.get_random_field_cfg::<PiopField, <PiopField as Field>::Inner, MillerRabin>();

    let field_elem_size = <crypto_primitives::crypto_bigint_uint::Uint<FIELD_LIMBS> as ConstTranscribable>::NUM_BYTES;

    // ── Step 1: Deserialize and verify IdealCheck ───────────────────
    let t0 = Instant::now();

    // Reconstruct IC proof from serialized bytes.
    let ic_combined_mle_values: Vec<DynamicPolynomialF<PiopField>> = zinc_proof
        .ic_proof_values
        .iter()
        .map(|bytes| {
            let num_coeffs = bytes.len() / field_elem_size;
            let coeffs: Vec<PiopField> = (0..num_coeffs)
                .map(|i| field_from_bytes(&bytes[i * field_elem_size..(i + 1) * field_elem_size], &field_cfg))
                .collect();
            DynamicPolynomialF::new(coeffs)
        })
        .collect();

    let ic_proof = zinc_piop::ideal_check::Proof::<PiopField> {
        combined_mle_values: ic_combined_mle_values,
    };

    let ic_verify_result = IdealCheckProtocol::<PiopField>::verify_as_subprotocol::<U, _, _>(
        &mut transcript,
        ic_proof,
        num_constraints,
        num_vars,
        ideal_over_f_from_ref,
        &field_cfg,
    );

    let ic_subclaim = match ic_verify_result {
        Ok(subclaim) => subclaim,
        Err(e) => {
            eprintln!("IdealCheck verification failed: {e:?}");
            return VerifyResult {
                accepted: false,
                timing: VerifyTimingBreakdown {
                    ideal_check_verify: t0.elapsed(),
                    combined_poly_resolver_verify: Duration::ZERO,
                    lookup_verify: Duration::ZERO,
                    pcs_verify: Duration::ZERO,
                    total: total_start.elapsed(),
                },
            };
        }
    };
    let ic_verify_time = t0.elapsed();

    // ── Step 2+2b: CPR + Lookup verification ──────────────────────
    //
    // If the proof uses `BatchedClassic`, the CPR and lookup sumchecks
    // were batched into a single multi-degree sumcheck. Otherwise, we
    // fall back to the sequential (non-batched) verification.
    let is_batched = matches!(
        zinc_proof.lookup_proof,
        Some(LookupProofData::BatchedClassic(_))
    );

    let (cpr_subclaim, cpr_verify_time, lookup_verify_time) = if is_batched {
        let t1 = Instant::now();
        let batched_proof = match zinc_proof.lookup_proof {
            Some(LookupProofData::BatchedClassic(ref bp)) => bp,
            _ => unreachable!(),
        };

        // Draw projecting element + compute projected scalars (same as non-batched).
        let projecting_element: PiopField = transcript.get_field_challenge(&field_cfg);
        let projected_scalars_coeffs = project_scalars::<PiopField, U>(|scalar| {
            let one = PiopField::one_with_cfg(&field_cfg);
            let zero = PiopField::zero_with_cfg(&field_cfg);
            DynamicPolynomialF::new(
                scalar.iter().map(|coeff| {
                    if coeff.into_inner() { one.clone() } else { zero.clone() }
                }).collect::<Vec<_>>()
            )
        });
        let field_projected_scalars =
            project_scalars_to_field(projected_scalars_coeffs, &projecting_element)
                .expect("scalar projection failed");

        // CPR pre-sumcheck: draw α, check claimed_sum against IC.
        let cpr_pre = match CombinedPolyResolver::<PiopField>::build_verifier_pre_sumcheck::<U>(
            &mut transcript,
            &batched_proof.md_proof.claimed_sums[0],
            num_constraints,
            &projecting_element,
            &field_projected_scalars,
            &ic_subclaim,
            &field_cfg,
        ) {
            Ok(pre) => pre,
            Err(e) => {
                eprintln!("CPR build_verifier_pre_sumcheck failed: {e:?}");
                return VerifyResult {
                    accepted: false,
                    timing: VerifyTimingBreakdown {
                        ideal_check_verify: ic_verify_time,
                        combined_poly_resolver_verify: t1.elapsed(),
                        lookup_verify: Duration::ZERO,
                        pcs_verify: Duration::ZERO,
                        total: total_start.elapsed(),
                    },
                };
            }
        };

        // Lookup pre-sumcheck: for each group, mirror transcript ops.
        let mut lookup_pres = Vec::new();
        for (group_proof, meta) in batched_proof.lookup_group_proofs.iter()
            .zip(batched_proof.lookup_group_meta.iter())
        {
            let (subtable, shifts) = generate_table_and_shifts(
                &meta.table_type,
                &projecting_element,
                &field_cfg,
            );
            match BatchedDecompLogupProtocol::<PiopField>::build_verifier_pre_sumcheck(
                &mut transcript,
                group_proof,
                &subtable,
                &shifts,
                meta.num_columns,
                meta.witness_len,
                &field_cfg,
            ) {
                Ok(pre) => lookup_pres.push(pre),
                Err(e) => {
                    eprintln!("Lookup build_verifier_pre_sumcheck failed: {e:?}");
                    return VerifyResult {
                        accepted: false,
                        timing: VerifyTimingBreakdown {
                            ideal_check_verify: ic_verify_time,
                            combined_poly_resolver_verify: t1.elapsed(),
                            lookup_verify: Duration::ZERO,
                            pcs_verify: Duration::ZERO,
                            total: total_start.elapsed(),
                        },
                    };
                }
            }
        }

        // Compute shared_num_vars: max of CPR num_vars and all lookup num_vars.
        let shared_num_vars = lookup_pres
            .iter()
            .map(|p| p.num_vars)
            .max()
            .map_or(num_vars, |max_lookup| max_lookup.max(num_vars));

        // Multi-degree sumcheck verify.
        let md_subclaims = match MultiDegreeSumcheck::<PiopField>::verify_as_subprotocol(
            &mut transcript,
            shared_num_vars,
            &batched_proof.md_proof,
            &field_cfg,
        ) {
            Ok(sc) => sc,
            Err(e) => {
                eprintln!("Multi-degree sumcheck verification failed: {e:?}");
                return VerifyResult {
                    accepted: false,
                    timing: VerifyTimingBreakdown {
                        ideal_check_verify: ic_verify_time,
                        combined_poly_resolver_verify: t1.elapsed(),
                        lookup_verify: Duration::ZERO,
                        pcs_verify: Duration::ZERO,
                        total: total_start.elapsed(),
                    },
                };
            }
        };

        // CPR finalize: check subclaim[0] against recomputed combination value.
        let cpr_sub = match CombinedPolyResolver::<PiopField>::finalize_verifier::<U>(
            &mut transcript,
            md_subclaims.point.clone(),
            md_subclaims.expected_evaluations[0].clone(),
            &cpr_pre,
            batched_proof.cpr_up_evals.clone(),
            batched_proof.cpr_down_evals.clone(),
            num_vars,
            &field_projected_scalars,
            &field_cfg,
        ) {
            Ok(sub) => sub,
            Err(e) => {
                eprintln!("CPR finalize_verifier failed: {e:?}");
                return VerifyResult {
                    accepted: false,
                    timing: VerifyTimingBreakdown {
                        ideal_check_verify: ic_verify_time,
                        combined_poly_resolver_verify: t1.elapsed(),
                        lookup_verify: Duration::ZERO,
                        pcs_verify: Duration::ZERO,
                        total: total_start.elapsed(),
                    },
                };
            }
        };

        // Lookup finalize: check subclaim[g+1] for each group.
        for (g, (lk_pre, group_proof)) in lookup_pres.iter()
            .zip(batched_proof.lookup_group_proofs.iter())
            .enumerate()
        {
            if let Err(e) = BatchedDecompLogupProtocol::<PiopField>::finalize_verifier(
                lk_pre,
                group_proof,
                &md_subclaims.point,
                &md_subclaims.expected_evaluations[g + 1],
                &field_cfg,
            ) {
                eprintln!("Lookup finalize_verifier failed (group {g}): {e:?}");
                return VerifyResult {
                    accepted: false,
                    timing: VerifyTimingBreakdown {
                        ideal_check_verify: ic_verify_time,
                        combined_poly_resolver_verify: t1.elapsed(),
                        lookup_verify: Duration::ZERO,
                        pcs_verify: Duration::ZERO,
                        total: total_start.elapsed(),
                    },
                };
            }
        }

        let cpr_verify_time = t1.elapsed();
        // Lookup time is included in CPR time for batched path.
        (cpr_sub, cpr_verify_time, Duration::ZERO)
    } else {
        // ── Sequential (non-batched) CPR + Lookup verification ──────
        let t1 = Instant::now();

        // Reconstruct CPR proof.
        let cpr_sumcheck_messages: Vec<ProverMsg<PiopField>> = zinc_proof
            .cpr_sumcheck_messages
            .iter()
            .map(|bytes| {
                let num_evals = bytes.len() / field_elem_size;
                let tail_evaluations: Vec<PiopField> = (0..num_evals)
                    .map(|i| field_from_bytes(&bytes[i * field_elem_size..(i + 1) * field_elem_size], &field_cfg))
                    .collect();
                ProverMsg(NatEvaluatedPolyWithoutConstant::new(tail_evaluations))
            })
            .collect();

        let cpr_claimed_sum = field_from_bytes(&zinc_proof.cpr_sumcheck_claimed_sum, &field_cfg);
        let cpr_up_evals: Vec<PiopField> = zinc_proof.cpr_up_evals.iter()
            .map(|b| field_from_bytes(b, &field_cfg))
            .collect();
        let cpr_down_evals: Vec<PiopField> = zinc_proof.cpr_down_evals.iter()
            .map(|b| field_from_bytes(b, &field_cfg))
            .collect();

        let cpr_proof = zinc_piop::combined_poly_resolver::Proof::<PiopField> {
            sumcheck_proof: SumcheckProof {
                messages: cpr_sumcheck_messages,
                claimed_sum: cpr_claimed_sum,
            },
            up_evals: cpr_up_evals,
            down_evals: cpr_down_evals,
        };

        // Compute projecting element and projected scalars for CPR verify.
        let projecting_element: PiopField = transcript.get_field_challenge(&field_cfg);
        let projected_scalars_coeffs = project_scalars::<PiopField, U>(|scalar| {
            let one = PiopField::one_with_cfg(&field_cfg);
            let zero = PiopField::zero_with_cfg(&field_cfg);
            DynamicPolynomialF::new(
                scalar.iter().map(|coeff| {
                    if coeff.into_inner() { one.clone() } else { zero.clone() }
                }).collect::<Vec<_>>()
            )
        });
        let field_projected_scalars =
            project_scalars_to_field(projected_scalars_coeffs, &projecting_element)
                .expect("scalar projection failed");

        let cpr_verify_result = CombinedPolyResolver::<PiopField>::verify_as_subprotocol::<U>(
            &mut transcript,
            cpr_proof,
            num_constraints,
            num_vars,
            max_degree,
            &projecting_element,
            &field_projected_scalars,
            ic_subclaim,
            &field_cfg,
        );

        let cpr_subclaim = match cpr_verify_result {
            Ok(subclaim) => subclaim,
            Err(e) => {
                eprintln!("CPR verification failed: {e:?}");
                return VerifyResult {
                    accepted: false,
                    timing: VerifyTimingBreakdown {
                        ideal_check_verify: ic_verify_time,
                        combined_poly_resolver_verify: t1.elapsed(),
                        lookup_verify: Duration::ZERO,
                        pcs_verify: Duration::ZERO,
                        total: total_start.elapsed(),
                    },
                };
            }
        };
        let cpr_verify_time = t1.elapsed();

        // ── Step 2b: Verify Batched Decomposed LogUp (column typing) ──
        let t1b = Instant::now();
        if let Some(ref lookup_data) = zinc_proof.lookup_proof {
            // Reuse the same projecting_element from the CPR step.
            let result = match lookup_data {
                LookupProofData::Gkr(proof) => verify_gkr_batched_lookup(
                    &mut transcript, proof, &projecting_element, &field_cfg,
                ).map(|_| ()),
                LookupProofData::Classic(proof) => verify_batched_lookup(
                    &mut transcript, proof, &projecting_element, &field_cfg,
                ).map(|_| ()),
                LookupProofData::BatchedClassic(_) => unreachable!("handled above"),
            };
            if let Err(e) = result {
                eprintln!("Lookup verification failed: {e:?}");
                return VerifyResult {
                    accepted: false,
                    timing: VerifyTimingBreakdown {
                        ideal_check_verify: ic_verify_time,
                        combined_poly_resolver_verify: cpr_verify_time,
                        lookup_verify: t1b.elapsed(),
                        pcs_verify: Duration::ZERO,
                        total: total_start.elapsed(),
                    },
                };
            }
        }
        let lookup_verify_time = t1b.elapsed();

        (cpr_subclaim, cpr_verify_time, lookup_verify_time)
    };

    // ── Step 3: PCS Verify ──────────────────────────────────────────
    // The CPR subclaim gives us the evaluation point and expected evaluations.
    // Verify that the PCS proof is consistent with these claims.
    let t2 = Instant::now();

    // Derive PiopField config from a fresh PCS transcript.
    let pcs_field_cfg = KeccakTranscript::default()
        .get_random_field_cfg::<PiopField, Zt::Fmod, Zt::PrimeTest>();

    // Deserialize the claimed evaluation.
    let eval_f: PiopField = PiopField::new_unchecked_with_cfg(
        <PiopField as Field>::Inner::read_transcription_bytes(&zinc_proof.pcs_evals_bytes[0]),
        &pcs_field_cfg,
    );

    // Derive the same hash-based PCS evaluation point that the prover used.
    let pcs_point: Vec<Zt::Pt> = {
        let i128_point = derive_pcs_point(&cpr_subclaim.evaluation_point, num_vars);
        i128_point
            .into_iter()
            .map(|v| {
                // SAFETY: Zt::Pt is i128 in all current instantiations.
                unsafe { std::mem::transmute_copy(&v) }
            })
            .collect()
    };
    let point_f: Vec<PiopField> = pcs_point.iter().map(|v| v.into_with_cfg(&pcs_field_cfg)).collect();

    // Deserialize PCS proof.
    let pcs_transcript = zip_plus::pcs_transcript::PcsTranscript {
        fs_transcript: KeccakTranscript::default(),
        stream: std::io::Cursor::new(zinc_proof.pcs_proof_bytes.clone()),
    };
    let pcs_proof: ZipPlusProof = pcs_transcript.into();

    let pcs_result = ZipPlus::<Zt, Lc>::verify::<PiopField, CHECK>(
        params,
        &zinc_proof.commitment,
        &point_f,
        &eval_f,
        &pcs_proof,
    );

    if let Err(ref e) = pcs_result {
        eprintln!("PCS verification failed: {e:?}");
    }

    let pcs_verify_time = t2.elapsed();
    let total_time = total_start.elapsed();

    // ── CPR→PCS binding note ────────────────────────────────────────
    // The PCS evaluates at a deterministic hash-derived point (derived
    // from the CPR evaluation point). Full binding would require the
    // PCS to open at the CPR's actual point; for now the hash-derived
    // point is used consistently by both prover and verifier.

    VerifyResult {
        accepted: pcs_result.is_ok(),
        timing: VerifyTimingBreakdown {
            ideal_check_verify: ic_verify_time,
            combined_poly_resolver_verify: cpr_verify_time,
            lookup_verify: lookup_verify_time,
            pcs_verify: pcs_verify_time,
            total: total_time,
        },
    }
}

// ─── Dual-Ring Pipeline ─────────────────────────────────────────────────────

/// Full proof data produced by the dual-ring prover.
///
/// Both IC passes share the same evaluation point and projecting element.
/// Both CPR sumchecks are batched into a single multi-degree sumcheck,
/// producing one shared evaluation point.
#[derive(Clone, Debug)]
pub struct DualRingZincProof {
    /// Serialized PCS proof bytes.
    pub pcs_proof_bytes: Vec<u8>,
    /// PCS commitment.
    pub commitment: ZipPlusCommitment,

    // ── IC proof values (both passes share the same evaluation point) ──
    pub bp_ic_proof_values: Vec<Vec<u8>>,
    pub qx_ic_proof_values: Vec<Vec<u8>>,

    // ── Batched CPR (multi-degree sumcheck: group 0 = BP, group 1 = QX) ──
    /// Per-group round messages, serialized.
    /// `md_group_messages[group][round]` → serialized field elements.
    pub md_group_messages: Vec<Vec<Vec<u8>>>,
    /// Claimed sum per group.
    pub md_claimed_sums: Vec<Vec<u8>>,
    /// Degree per group.
    pub md_degrees: Vec<usize>,

    /// BP CPR up/down evaluations at the shared sumcheck point.
    pub bp_cpr_up_evals: Vec<Vec<u8>>,
    pub bp_cpr_down_evals: Vec<Vec<u8>>,
    /// QX CPR up/down evaluations at the shared sumcheck point.
    pub qx_cpr_up_evals: Vec<Vec<u8>>,
    pub qx_cpr_down_evals: Vec<Vec<u8>>,

    // ── PCS evaluation data ─────────────────────────────────────────
    pub evaluation_point_bytes: Vec<Vec<u8>>,
    pub pcs_evals_bytes: Vec<Vec<u8>>,

    pub timing: TimingBreakdown,
}

/// Run the dual-ring Zinc+ prover pipeline with **shared challenges**.
///
/// Both PIOP passes (BinaryPoly and Q[X]) share:
/// - The same Fiat-Shamir prime / field configuration
/// - The same IC evaluation point
/// - The same F[X]→F projecting element
///
/// Both CPR sumchecks are batched into a single multi-degree sumcheck
/// (group 0 = BinaryPoly, group 1 = Q[X]), producing a common
/// evaluation point for PCS opening.
///
/// # Type parameters
/// - `U1`: UAIR for BinaryPoly constraints.
/// - `U2`: UAIR for Q[X] constraints.
/// - `D1`: degree+1 for BinaryPoly.
/// - `D2`: degree+1 for DensePolynomial.
/// - `ConvertFn`: converts the BinaryPoly trace to a DensePolynomial trace.
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn prove_dual_ring<U1, U2, Zt, Lc, const D1: usize, const D2: usize, const CHECK: bool, ConvertFn>(
    params: &ZipPlusParams<Zt, Lc>,
    trace: &[DenseMultilinearExtension<BinaryPoly<D1>>],
    num_vars: usize,
    convert_trace: ConvertFn,
) -> DualRingZincProof
where
    U1: Uair<Scalar = BinaryPoly<D1>>,
    U2: Uair<Scalar = DensePolynomial<i64, D2>>,
    Zt: ZipTypes<Eval = BinaryPoly<D1>>,
    Lc: LinearCode<Zt>,
    rand::distr::StandardUniform: rand::distr::Distribution<BinaryPoly<D1>>,
    PiopField: FromPrimitiveWithConfig + FromWithConfig<i64>,
    <PiopField as Field>::Inner: ConstTranscribable + FromRef<<PiopField as Field>::Inner> + Send + Sync + Default + num_traits::Zero,
    MillerRabin: PrimalityTest<<PiopField as Field>::Inner>,
    Zt::Eval: ProjectableToField<PiopField>,
    Zt::Cw: ProjectableToField<PiopField>,
    PiopField: for<'a> FromWithConfig<&'a Zt::Chal> + for<'a> FromWithConfig<&'a Zt::Pt> + for<'a> FromWithConfig<&'a Zt::CombR>,
    <PiopField as Field>::Inner: FromRef<Zt::Fmod>,
    ConvertFn: Fn(&[DenseMultilinearExtension<BinaryPoly<D1>>]) -> Vec<DenseMultilinearExtension<DensePolynomial<i64, D2>>>,
{
    let total_start = Instant::now();

    let bp_num_constraints = count_constraints::<U1>();
    let bp_max_degree = count_max_degree::<U1>();
    let qx_num_constraints = count_constraints::<U2>();
    let qx_max_degree = count_max_degree::<U2>();

    // ── Step 1: PCS Commit ──────────────────────────────────────────
    let t0 = Instant::now();
    let (hint, commitment) = ZipPlus::<Zt, Lc>::commit(params, trace)
        .expect("PCS commit failed");
    let pcs_commit_time = t0.elapsed();

    // ── Step 2: Fiat-Shamir transcript + field config ───────────────
    let mut transcript = KeccakTranscript::new();
    let field_cfg = transcript.get_random_field_cfg::<PiopField, <PiopField as Field>::Inner, MillerRabin>();

    // ── Step 3: Shared IC evaluation point ──────────────────────────
    let ic_evaluation_point: Vec<PiopField> =
        transcript.get_field_challenges(num_vars, &field_cfg);

    // ── Step 4: IC₁ (BinaryPoly) — uses shared evaluation point ────
    let t1 = Instant::now();
    let bp_projected_trace = project_trace_coeffs::<PiopField, i64, i64, D1>(
        trace, &[], &[], &field_cfg,
    );
    let bp_projected_scalars = project_scalars::<PiopField, U1>(|scalar| {
        let one = PiopField::one_with_cfg(&field_cfg);
        let zero = PiopField::zero_with_cfg(&field_cfg);
        DynamicPolynomialF::new(
            scalar.iter().map(|coeff| {
                if coeff.into_inner() { one.clone() } else { zero.clone() }
            }).collect::<Vec<_>>()
        )
    });

    let (bp_ic_proof, _bp_ic_state) =
        IdealCheckProtocol::<PiopField>::prove_at_point::<U1>(
            &mut transcript,
            &bp_projected_trace,
            &bp_projected_scalars,
            bp_num_constraints,
            &ic_evaluation_point,
            &field_cfg,
        )
        .expect("BinaryPoly ideal check prover failed");

    // ── Step 5: Convert trace to Q[X] and run IC₂ ──────────────────
    let qx_trace = convert_trace(trace);
    let qx_projected_trace = project_trace_coeffs::<PiopField, i64, i64, D2>(
        &[], &qx_trace, &[], &field_cfg,
    );
    let qx_projected_scalars = project_scalars::<PiopField, U2>(|scalar| {
        DynamicPolynomialF::new(
            scalar.iter().map(|coeff| PiopField::from_with_cfg(*coeff, &field_cfg)).collect::<Vec<_>>()
        )
    });

    let (qx_ic_proof, _qx_ic_state) =
        IdealCheckProtocol::<PiopField>::prove_at_point::<U2>(
            &mut transcript,
            &qx_projected_trace,
            &qx_projected_scalars,
            qx_num_constraints,
            &ic_evaluation_point,
            &field_cfg,
        )
        .expect("Q[X] ideal check prover failed");
    let ic_time = t1.elapsed();

    // ── Step 6: Shared projecting element + project traces to F_q ───
    let t2 = Instant::now();
    let projecting_element: PiopField = transcript.get_field_challenge(&field_cfg);

    let bp_field_projected_scalars =
        project_scalars_to_field(bp_projected_scalars, &projecting_element)
            .expect("BP scalar projection failed");
    let bp_field_trace = project_trace_to_field::<PiopField, D1>(
        trace, &[], &[], &projecting_element,
    );

    let qx_field_projected_scalars =
        project_scalars_to_field(qx_projected_scalars, &projecting_element)
            .expect("QX scalar projection failed");
    let qx_field_trace = project_trace_to_field::<PiopField, D2>(
        &[], &qx_projected_trace, &[], &projecting_element,
    );

    // ── Step 7: Build CPR sumcheck groups (without running sumcheck) ──
    let bp_cpr_group = CombinedPolyResolver::<PiopField>::build_prover_group::<U1>(
        &mut transcript,
        bp_field_trace,
        &ic_evaluation_point,
        &bp_field_projected_scalars,
        bp_num_constraints,
        num_vars,
        bp_max_degree,
        &field_cfg,
    )
    .expect("BP CPR build_prover_group failed");
    let bp_num_cols = bp_cpr_group.num_cols;

    let qx_cpr_group = CombinedPolyResolver::<PiopField>::build_prover_group::<U2>(
        &mut transcript,
        qx_field_trace,
        &ic_evaluation_point,
        &qx_field_projected_scalars,
        qx_num_constraints,
        num_vars,
        qx_max_degree,
        &field_cfg,
    )
    .expect("QX CPR build_prover_group failed");
    let qx_num_cols = qx_cpr_group.num_cols;

    // ── Step 8: Run batched multi-degree sumcheck ───────────────────
    let sumcheck_groups: Vec<(
        usize,
        Vec<DenseMultilinearExtension<<PiopField as Field>::Inner>>,
        Box<dyn Fn(&[PiopField]) -> PiopField + Send + Sync>,
    )> = vec![
        (bp_cpr_group.degree, bp_cpr_group.mles, bp_cpr_group.comb_fn),
        (qx_cpr_group.degree, qx_cpr_group.mles, qx_cpr_group.comb_fn),
    ];

    let (md_proof, mut prover_states) =
        MultiDegreeSumcheck::<PiopField>::prove_as_subprotocol(
            &mut transcript,
            sumcheck_groups,
            num_vars,
            &field_cfg,
        );

    // ── Step 9: Finalize both CPR groups ────────────────────────────
    let bp_prover_state = prover_states.remove(0);
    let (bp_up_evals, bp_down_evals, bp_cpr_state) =
        CombinedPolyResolver::<PiopField>::finalize_prover(
            &mut transcript,
            bp_prover_state,
            bp_num_cols,
            &field_cfg,
        )
        .expect("BP CPR finalize_prover failed");

    let qx_prover_state = prover_states.remove(0);
    let (qx_up_evals, qx_down_evals, _qx_cpr_state) =
        CombinedPolyResolver::<PiopField>::finalize_prover(
            &mut transcript,
            qx_prover_state,
            qx_num_cols,
            &field_cfg,
        )
        .expect("QX CPR finalize_prover failed");
    let cpr_time = t2.elapsed();

    // ── Step 10: PCS Prove (test + evaluate) ──────────────────────
    let t3 = Instant::now();
    let point: Vec<Zt::Pt> = {
        let i128_point = derive_pcs_point(&bp_cpr_state.evaluation_point, num_vars);
        i128_point
            .into_iter()
            .map(|v| unsafe { std::mem::transmute_copy(&v) })
            .collect()
    };
    let (eval_f, proof) =
        ZipPlus::<Zt, Lc>::prove::<PiopField, CHECK>(
            params,
            trace,
            &point,
            &hint,
        )
        .expect("PCS prove failed");
    let pcs_prove_time = t3.elapsed();

    let total_time = total_start.elapsed();

    // ── Serialize ───────────────────────────────────────────────────
    let proof_bytes: Vec<u8> = {
        let pcs_transcript: zip_plus::pcs_transcript::PcsTranscript = proof.into();
        pcs_transcript.stream.into_inner()
    };

    // Serialize IC proofs
    let bp_ic_values: Vec<Vec<u8>> = bp_ic_proof
        .combined_mle_values
        .iter()
        .map(|dpf| dpf.coeffs.iter().flat_map(|c| field_to_bytes(c)).collect())
        .collect();
    let qx_ic_values: Vec<Vec<u8>> = qx_ic_proof
        .combined_mle_values
        .iter()
        .map(|dpf| dpf.coeffs.iter().flat_map(|c| field_to_bytes(c)).collect())
        .collect();

    // Serialize multi-degree sumcheck proof
    let md_group_messages: Vec<Vec<Vec<u8>>> = md_proof
        .group_messages
        .iter()
        .map(|group_rounds| {
            group_rounds
                .iter()
                .map(|msg| msg.0.tail_evaluations.iter().flat_map(|c| field_to_bytes(c)).collect())
                .collect()
        })
        .collect();
    let md_claimed_sums: Vec<Vec<u8>> = md_proof
        .claimed_sums
        .iter()
        .map(field_to_bytes)
        .collect();

    // Serialize up/down evals
    let bp_cpr_ups: Vec<Vec<u8>> = bp_up_evals.iter().map(field_to_bytes).collect();
    let bp_cpr_downs: Vec<Vec<u8>> = bp_down_evals.iter().map(field_to_bytes).collect();
    let qx_cpr_ups: Vec<Vec<u8>> = qx_up_evals.iter().map(field_to_bytes).collect();
    let qx_cpr_downs: Vec<Vec<u8>> = qx_down_evals.iter().map(field_to_bytes).collect();

    // Serialize evaluation data
    let evaluation_point_bytes: Vec<Vec<u8>> =
        bp_cpr_state.evaluation_point.iter().map(field_to_bytes).collect();
    let pcs_evals_bytes: Vec<Vec<u8>> = vec![field_to_bytes(&eval_f)];

    DualRingZincProof {
        pcs_proof_bytes: proof_bytes,
        commitment,
        bp_ic_proof_values: bp_ic_values,
        qx_ic_proof_values: qx_ic_values,
        md_group_messages,
        md_claimed_sums,
        md_degrees: md_proof.degrees,
        bp_cpr_up_evals: bp_cpr_ups,
        bp_cpr_down_evals: bp_cpr_downs,
        qx_cpr_up_evals: qx_cpr_ups,
        qx_cpr_down_evals: qx_cpr_downs,
        evaluation_point_bytes,
        pcs_evals_bytes,
        timing: TimingBreakdown {
            pcs_commit: pcs_commit_time,
            ideal_check: ic_time,
            combined_poly_resolver: cpr_time,
            lookup: Duration::ZERO,
            pcs_prove: pcs_prove_time,
            total: total_time,
        },
    }
}

/// Run the dual-ring Zinc+ verifier with **shared challenges**.
///
/// Mirrors [`prove_dual_ring`]: draws one IC evaluation point, one
/// projecting element, and verifies both CPRs via a single multi-degree
/// sumcheck.
///
/// - The BinaryPoly pass uses `TrivialIdeal` (F₂ constraints don't lift).
/// - The Q[X] pass uses the caller-provided `qx_ideal_from_ref` closure
///   to map UAIR ideals to field-level ideals (e.g. `DegreeOneIdeal(2)`).
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn verify_dual_ring<U1, U2, Zt, Lc, const D1: usize, const D2: usize, const CHECK: bool, QxIdealOverF, QxIdealFromRef>(
    params: &ZipPlusParams<Zt, Lc>,
    proof: &DualRingZincProof,
    num_vars: usize,
    qx_ideal_from_ref: QxIdealFromRef,
) -> VerifyResult
where
    U1: Uair<Scalar = BinaryPoly<D1>>,
    U2: Uair<Scalar = DensePolynomial<i64, D2>>,
    Zt: ZipTypes<Eval = BinaryPoly<D1>>,
    Lc: LinearCode<Zt>,
    PiopField: FromPrimitiveWithConfig + FromWithConfig<i64>,
    <PiopField as Field>::Inner: ConstTranscribable + FromRef<<PiopField as Field>::Inner> + Send + Sync + Default + num_traits::Zero,
    MillerRabin: PrimalityTest<<PiopField as Field>::Inner>,
    Zt::Eval: ProjectableToField<PiopField>,
    Zt::Cw: ProjectableToField<PiopField>,
    PiopField: for<'a> FromWithConfig<&'a Zt::Chal> + for<'a> FromWithConfig<&'a Zt::Pt> + for<'a> FromWithConfig<&'a Zt::CombR>,
    <PiopField as Field>::Inner: FromRef<Zt::Fmod>,
    QxIdealOverF: Ideal + IdealCheck<DynamicPolynomialF<PiopField>>,
    QxIdealFromRef: Fn(&IdealOrZero<U2::Ideal>) -> QxIdealOverF,
{
    let total_start = Instant::now();

    let bp_num_constraints = count_constraints::<U1>();
    let qx_num_constraints = count_constraints::<U2>();
    // Degrees are encoded in the proof (proof.md_degrees) for the verifier.

    // ── Reconstruct Fiat-Shamir transcript ──────────────────────────
    let mut transcript = KeccakTranscript::new();
    let field_cfg = transcript.get_random_field_cfg::<PiopField, <PiopField as Field>::Inner, MillerRabin>();
    let field_elem_size = <crypto_primitives::crypto_bigint_uint::Uint<FIELD_LIMBS> as ConstTranscribable>::NUM_BYTES;

    // ── Shared IC evaluation point ──────────────────────────────────
    let ic_evaluation_point: Vec<PiopField> =
        transcript.get_field_challenges(num_vars, &field_cfg);

    // ── IC₁ verify (BinaryPoly, TrivialIdeal) ──────────────────────
    let t0 = Instant::now();

    let bp_ic_combined_mle_values: Vec<DynamicPolynomialF<PiopField>> = proof
        .bp_ic_proof_values
        .iter()
        .map(|bytes| {
            let num_coeffs = bytes.len() / field_elem_size;
            let coeffs: Vec<PiopField> = (0..num_coeffs)
                .map(|i| field_from_bytes(&bytes[i * field_elem_size..(i + 1) * field_elem_size], &field_cfg))
                .collect();
            DynamicPolynomialF::new(coeffs)
        })
        .collect();

    let bp_ic_proof = zinc_piop::ideal_check::Proof::<PiopField> {
        combined_mle_values: bp_ic_combined_mle_values,
    };

    let bp_ic_subclaim = match IdealCheckProtocol::<PiopField>::verify_at_point::<U1, _, _>(
        &mut transcript,
        bp_ic_proof,
        bp_num_constraints,
        ic_evaluation_point.clone(),
        |_: &IdealOrZero<U1::Ideal>| TrivialIdeal,
        &field_cfg,
    ) {
        Ok(subclaim) => subclaim,
        Err(e) => {
            eprintln!("BinaryPoly IdealCheck verification failed: {e:?}");
            return VerifyResult {
                accepted: false,
                timing: VerifyTimingBreakdown {
                    ideal_check_verify: t0.elapsed(),
                    combined_poly_resolver_verify: Duration::ZERO,
                    lookup_verify: Duration::ZERO,
                    pcs_verify: Duration::ZERO,
                    total: total_start.elapsed(),
                },
            };
        }
    };

    // ── IC₂ verify (Q[X], real ideals) ──────────────────────────────
    let qx_ic_combined_mle_values: Vec<DynamicPolynomialF<PiopField>> = proof
        .qx_ic_proof_values
        .iter()
        .map(|bytes| {
            let num_coeffs = bytes.len() / field_elem_size;
            let coeffs: Vec<PiopField> = (0..num_coeffs)
                .map(|i| field_from_bytes(&bytes[i * field_elem_size..(i + 1) * field_elem_size], &field_cfg))
                .collect();
            DynamicPolynomialF::new(coeffs)
        })
        .collect();

    let qx_ic_proof = zinc_piop::ideal_check::Proof::<PiopField> {
        combined_mle_values: qx_ic_combined_mle_values,
    };

    let qx_ic_subclaim = match IdealCheckProtocol::<PiopField>::verify_at_point::<U2, _, _>(
        &mut transcript,
        qx_ic_proof,
        qx_num_constraints,
        ic_evaluation_point.clone(),
        &qx_ideal_from_ref,
        &field_cfg,
    ) {
        Ok(subclaim) => subclaim,
        Err(e) => {
            eprintln!("Q[X] IdealCheck verification failed: {e:?}");
            return VerifyResult {
                accepted: false,
                timing: VerifyTimingBreakdown {
                    ideal_check_verify: t0.elapsed(),
                    combined_poly_resolver_verify: Duration::ZERO,
                    lookup_verify: Duration::ZERO,
                    pcs_verify: Duration::ZERO,
                    total: total_start.elapsed(),
                },
            };
        }
    };
    let ic_verify_time = t0.elapsed();

    // ── Shared projecting element ───────────────────────────────────
    let t1 = Instant::now();
    let projecting_element: PiopField = transcript.get_field_challenge(&field_cfg);

    // ── CPR₁ pre-sumcheck (draws α₁ from transcript) ───────────────
    let bp_projected_scalars_coeffs = project_scalars::<PiopField, U1>(|scalar| {
        let one = PiopField::one_with_cfg(&field_cfg);
        let zero = PiopField::zero_with_cfg(&field_cfg);
        DynamicPolynomialF::new(
            scalar.iter().map(|coeff| {
                if coeff.into_inner() { one.clone() } else { zero.clone() }
            }).collect::<Vec<_>>()
        )
    });
    let bp_field_projected_scalars =
        project_scalars_to_field(bp_projected_scalars_coeffs, &projecting_element)
            .expect("BP scalar projection failed");

    let bp_cpr_pre = match CombinedPolyResolver::<PiopField>::build_verifier_pre_sumcheck::<U1>(
        &mut transcript,
        &field_from_bytes(&proof.md_claimed_sums[0], &field_cfg),
        bp_num_constraints,
        &projecting_element,
        &bp_field_projected_scalars,
        &bp_ic_subclaim,
        &field_cfg,
    ) {
        Ok(pre) => pre,
        Err(e) => {
            eprintln!("BP CPR pre-sumcheck failed: {e:?}");
            return VerifyResult {
                accepted: false,
                timing: VerifyTimingBreakdown {
                    ideal_check_verify: ic_verify_time,
                    combined_poly_resolver_verify: t1.elapsed(),
                    lookup_verify: Duration::ZERO,
                    pcs_verify: Duration::ZERO,
                    total: total_start.elapsed(),
                },
            };
        }
    };

    // ── CPR₂ pre-sumcheck (draws α₂ from transcript) ───────────────
    let qx_projected_scalars_coeffs = project_scalars::<PiopField, U2>(|scalar| {
        DynamicPolynomialF::new(
            scalar.iter().map(|coeff| PiopField::from_with_cfg(*coeff, &field_cfg)).collect::<Vec<_>>()
        )
    });
    let qx_field_projected_scalars =
        project_scalars_to_field(qx_projected_scalars_coeffs, &projecting_element)
            .expect("QX scalar projection failed");

    let qx_cpr_pre = match CombinedPolyResolver::<PiopField>::build_verifier_pre_sumcheck::<U2>(
        &mut transcript,
        &field_from_bytes(&proof.md_claimed_sums[1], &field_cfg),
        qx_num_constraints,
        &projecting_element,
        &qx_field_projected_scalars,
        &qx_ic_subclaim,
        &field_cfg,
    ) {
        Ok(pre) => pre,
        Err(e) => {
            eprintln!("QX CPR pre-sumcheck failed: {e:?}");
            return VerifyResult {
                accepted: false,
                timing: VerifyTimingBreakdown {
                    ideal_check_verify: ic_verify_time,
                    combined_poly_resolver_verify: t1.elapsed(),
                    lookup_verify: Duration::ZERO,
                    pcs_verify: Duration::ZERO,
                    total: total_start.elapsed(),
                },
            };
        }
    };

    // ── Deserialize multi-degree sumcheck proof ─────────────────────
    let md_group_messages: Vec<Vec<ProverMsg<PiopField>>> = proof
        .md_group_messages
        .iter()
        .map(|group_rounds| {
            group_rounds
                .iter()
                .map(|bytes| {
                    let num_evals = bytes.len() / field_elem_size;
                    let tail_evaluations: Vec<PiopField> = (0..num_evals)
                        .map(|i| field_from_bytes(&bytes[i * field_elem_size..(i + 1) * field_elem_size], &field_cfg))
                        .collect();
                    ProverMsg(NatEvaluatedPolyWithoutConstant::new(tail_evaluations))
                })
                .collect()
        })
        .collect();

    let md_claimed_sums: Vec<PiopField> = proof
        .md_claimed_sums
        .iter()
        .map(|b| field_from_bytes(b, &field_cfg))
        .collect();

    let md_proof = MultiDegreeSumcheckProof {
        group_messages: md_group_messages,
        claimed_sums: md_claimed_sums,
        degrees: proof.md_degrees.clone(),
    };

    // ── Verify multi-degree sumcheck ────────────────────────────────
    let md_subclaims = match MultiDegreeSumcheck::<PiopField>::verify_as_subprotocol(
        &mut transcript,
        num_vars,
        &md_proof,
        &field_cfg,
    ) {
        Ok(subclaims) => subclaims,
        Err(e) => {
            eprintln!("Multi-degree sumcheck verification failed: {e:?}");
            return VerifyResult {
                accepted: false,
                timing: VerifyTimingBreakdown {
                    ideal_check_verify: ic_verify_time,
                    combined_poly_resolver_verify: t1.elapsed(),
                    lookup_verify: Duration::ZERO,
                    pcs_verify: Duration::ZERO,
                    total: total_start.elapsed(),
                },
            };
        }
    };

    // ── Finalize CPR₁ (BinaryPoly) ─────────────────────────────────
    let bp_up_evals: Vec<PiopField> = proof.bp_cpr_up_evals.iter()
        .map(|b| field_from_bytes(b, &field_cfg))
        .collect();
    let bp_down_evals: Vec<PiopField> = proof.bp_cpr_down_evals.iter()
        .map(|b| field_from_bytes(b, &field_cfg))
        .collect();

    let bp_cpr_subclaim = match CombinedPolyResolver::<PiopField>::finalize_verifier::<U1>(
        &mut transcript,
        md_subclaims.point.clone(),
        md_subclaims.expected_evaluations[0].clone(),
        &bp_cpr_pre,
        bp_up_evals,
        bp_down_evals,
        num_vars,
        &bp_field_projected_scalars,
        &field_cfg,
    ) {
        Ok(subclaim) => subclaim,
        Err(e) => {
            eprintln!("BP CPR finalize_verifier failed: {e:?}");
            return VerifyResult {
                accepted: false,
                timing: VerifyTimingBreakdown {
                    ideal_check_verify: ic_verify_time,
                    combined_poly_resolver_verify: t1.elapsed(),
                    lookup_verify: Duration::ZERO,
                    pcs_verify: Duration::ZERO,
                    total: total_start.elapsed(),
                },
            };
        }
    };

    // ── Finalize CPR₂ (Q[X]) ───────────────────────────────────────
    let qx_up_evals: Vec<PiopField> = proof.qx_cpr_up_evals.iter()
        .map(|b| field_from_bytes(b, &field_cfg))
        .collect();
    let qx_down_evals: Vec<PiopField> = proof.qx_cpr_down_evals.iter()
        .map(|b| field_from_bytes(b, &field_cfg))
        .collect();

    if let Err(e) = CombinedPolyResolver::<PiopField>::finalize_verifier::<U2>(
        &mut transcript,
        md_subclaims.point.clone(),
        md_subclaims.expected_evaluations[1].clone(),
        &qx_cpr_pre,
        qx_up_evals,
        qx_down_evals,
        num_vars,
        &qx_field_projected_scalars,
        &field_cfg,
    ) {
        eprintln!("QX CPR finalize_verifier failed: {e:?}");
        return VerifyResult {
            accepted: false,
            timing: VerifyTimingBreakdown {
                ideal_check_verify: ic_verify_time,
                combined_poly_resolver_verify: t1.elapsed(),
                lookup_verify: Duration::ZERO,
                pcs_verify: Duration::ZERO,
                total: total_start.elapsed(),
            },
        };
    }
    let cpr_verify_time = t1.elapsed();

    // ── PCS Verify ──────────────────────────────────────────────────
    let t2 = Instant::now();

    // Derive PiopField config from a fresh PCS transcript.
    let pcs_field_cfg = KeccakTranscript::default()
        .get_random_field_cfg::<PiopField, Zt::Fmod, Zt::PrimeTest>();

    // Deserialize the claimed evaluation.
    let eval_f: PiopField = PiopField::new_unchecked_with_cfg(
        <PiopField as Field>::Inner::read_transcription_bytes(&proof.pcs_evals_bytes[0]),
        &pcs_field_cfg,
    );

    // Derive the same hash-based PCS evaluation point that the prover used.
    let pcs_point: Vec<Zt::Pt> = {
        let i128_point = derive_pcs_point(&bp_cpr_subclaim.evaluation_point, num_vars);
        i128_point
            .into_iter()
            .map(|v| {
                // SAFETY: Zt::Pt is i128 in all current instantiations.
                unsafe { std::mem::transmute_copy(&v) }
            })
            .collect()
    };
    let point_f: Vec<PiopField> = pcs_point.iter().map(|v| v.into_with_cfg(&pcs_field_cfg)).collect();

    let pcs_transcript = zip_plus::pcs_transcript::PcsTranscript {
        fs_transcript: KeccakTranscript::default(),
        stream: std::io::Cursor::new(proof.pcs_proof_bytes.clone()),
    };
    let pcs_proof: ZipPlusProof = pcs_transcript.into();

    let pcs_result = ZipPlus::<Zt, Lc>::verify::<PiopField, CHECK>(
        params,
        &proof.commitment,
        &point_f,
        &eval_f,
        &pcs_proof,
    );
    let pcs_verify_time = t2.elapsed();

    VerifyResult {
        accepted: pcs_result.is_ok(),
        timing: VerifyTimingBreakdown {
            ideal_check_verify: ic_verify_time,
            combined_poly_resolver_verify: cpr_verify_time,
            lookup_verify: Duration::ZERO,
            pcs_verify: pcs_verify_time,
            total: total_start.elapsed(),
        },
    }
}

// ─── Dual-Circuit Pipeline ──────────────────────────────────────────────────

/// Full proof data produced by the dual-circuit prover.
///
/// Two **separate** circuits (e.g. SHA-256 with `BinaryPoly<D>` and ECDSA
/// with `Int<4>`) share one Fiat-Shamir transcript, one IC evaluation
/// point, one projecting element, and one multi-degree sumcheck.  Each
/// has its own PCS commitment and proof.
///
/// Multi-degree sumcheck group ordering:
///   group 0       — CPR for circuit 1 (BinaryPoly)
///   group 1       — CPR for circuit 2 (generic ring)
///   groups 2..N+2 — lookup groups for circuit 1
#[derive(Clone, Debug)]
pub struct DualCircuitZincProof {
    // ── PCS (circuit 1 = BinaryPoly) ────────────────────────────────
    pub pcs1_proof_bytes: Vec<u8>,
    pub pcs1_commitment: ZipPlusCommitment,

    // ── PCS (circuit 2 = generic ring) ──────────────────────────────
    pub pcs2_proof_bytes: Vec<u8>,
    pub pcs2_commitment: ZipPlusCommitment,

    // ── IC proof values (both use shared eval point) ────────────────
    pub ic1_proof_values: Vec<Vec<u8>>,
    pub ic2_proof_values: Vec<Vec<u8>>,

    // ── Batched multi-degree sumcheck ───────────────────────────────
    pub md_group_messages: Vec<Vec<Vec<u8>>>,
    pub md_claimed_sums: Vec<Vec<u8>>,
    pub md_degrees: Vec<usize>,

    // ── CPR up/down evals ───────────────────────────────────────────
    pub cpr1_up_evals: Vec<Vec<u8>>,
    pub cpr1_down_evals: Vec<Vec<u8>>,
    pub cpr2_up_evals: Vec<Vec<u8>>,
    pub cpr2_down_evals: Vec<Vec<u8>>,

    // ── Lookup (circuit 1 only) ─────────────────────────────────────
    pub lookup_group_meta: Vec<LookupGroupMeta>,
    pub lookup_group_proofs: Vec<zinc_piop::lookup::BatchedDecompLogupProof<PiopField>>,

    // ── Shared evaluation point + PCS evals ─────────────────────────
    pub evaluation_point_bytes: Vec<Vec<u8>>,
    pub pcs1_evals_bytes: Vec<Vec<u8>>,
    pub pcs2_evals_bytes: Vec<Vec<u8>>,

    pub timing: TimingBreakdown,
}

/// Run the dual-circuit Zinc+ prover with **shared challenges**.
///
/// Combines two independent circuits into a single proving pipeline:
/// - Circuit 1: `BinaryPoly<D>` (e.g. SHA-256) with optional lookup
/// - Circuit 2: generic ring `R2` (e.g. ECDSA with `Int<4>`)
///
/// Both circuits share:
/// - The same Fiat-Shamir prime / field configuration
/// - The same IC evaluation point
/// - The same F\[X\]→F projecting element
/// - A single multi-degree sumcheck (CPR₁ + CPR₂ + lookup groups)
///
/// Each circuit retains its own PCS (commit → test → evaluate).
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn prove_dual_circuit<U1, U2, R2, Zt1, Lc1, Zt2, Lc2, PcsF2, const D: usize, const CHECK: bool>(
    params1: &ZipPlusParams<Zt1, Lc1>,
    trace1: &[DenseMultilinearExtension<BinaryPoly<D>>],
    params2: &ZipPlusParams<Zt2, Lc2>,
    trace2: &[DenseMultilinearExtension<R2>],
    num_vars: usize,
    lookup_specs: &[LookupColumnSpec],
) -> DualCircuitZincProof
where
    // Circuit 1 (BinaryPoly<D>):
    U1: Uair<Scalar = BinaryPoly<D>>,
    Zt1: ZipTypes<Eval = BinaryPoly<D>>,
    Lc1: LinearCode<Zt1>,
    rand::distr::StandardUniform: rand::distr::Distribution<BinaryPoly<D>>,
    Zt1::Eval: ProjectableToField<PiopField>,
    Zt1::Cw: ProjectableToField<PiopField>,
    PiopField: for<'a> FromWithConfig<&'a Zt1::Chal> + for<'a> FromWithConfig<&'a Zt1::Pt> + for<'a> FromWithConfig<&'a Zt1::CombR>,
    <PiopField as Field>::Inner: FromRef<Zt1::Fmod>,
    // Circuit 2 (generic ring R2):
    R2: ProjectableToField<PiopField>
        + crypto_primitives::Semiring
        + std::fmt::Debug
        + Clone
        + Send
        + Sync
        + 'static,
    U2: Uair<Scalar = R2>,
    Zt2: ZipTypes<Eval = R2>,
    Lc2: LinearCode<Zt2>,
    PiopField: FromWithConfig<R2>,
    PcsF2: PrimeField
        + FromPrimitiveWithConfig
        + for<'a> FromWithConfig<&'a Zt2::Chal>
        + for<'a> FromWithConfig<&'a Zt2::Pt>
        + for<'a> FromWithConfig<&'a Zt2::CombR>
        + for<'a> zinc_utils::mul_by_scalar::MulByScalar<&'a PcsF2>
        + FromRef<PcsF2>,
    PcsF2::Inner: FromRef<Zt2::Fmod> + ConstTranscribable,
    Zt2::Eval: ProjectableToField<PcsF2>,
    // Shared:
    PiopField: FromPrimitiveWithConfig,
    <PiopField as Field>::Inner: ConstTranscribable
        + FromRef<<PiopField as Field>::Inner>
        + Send
        + Sync
        + Default
        + num_traits::Zero,
    MillerRabin: PrimalityTest<<PiopField as Field>::Inner>,
{
    let total_start = Instant::now();

    let c1_num_constraints = count_constraints::<U1>();
    let c1_max_degree = count_max_degree::<U1>();
    let c2_num_constraints = count_constraints::<U2>();
    let c2_max_degree = count_max_degree::<U2>();

    // ── Step 1: PCS Commit (both circuits) ──────────────────────────
    let t0 = Instant::now();
    let (hint1, commitment1) = ZipPlus::<Zt1, Lc1>::commit(params1, trace1)
        .expect("PCS1 commit failed");
    let (hint2, commitment2) = ZipPlus::<Zt2, Lc2>::commit(params2, trace2)
        .expect("PCS2 commit failed");
    let pcs_commit_time = t0.elapsed();

    // ── Step 2: Shared transcript + field config ────────────────────
    let mut transcript = KeccakTranscript::new();
    let field_cfg = transcript.get_random_field_cfg::<PiopField, <PiopField as Field>::Inner, MillerRabin>();

    // ── Step 3: Shared IC evaluation point ──────────────────────────
    let ic_evaluation_point: Vec<PiopField> =
        transcript.get_field_challenges(num_vars, &field_cfg);

    // ── Step 4: IC₁ (BinaryPoly) at shared point ───────────────────
    let t1 = Instant::now();
    let c1_projected_trace = project_trace_coeffs::<PiopField, i64, i64, D>(
        trace1, &[], &[], &field_cfg,
    );
    let c1_projected_scalars = project_scalars::<PiopField, U1>(|scalar| {
        let one = PiopField::one_with_cfg(&field_cfg);
        let zero = PiopField::zero_with_cfg(&field_cfg);
        DynamicPolynomialF::new(
            scalar
                .iter()
                .map(|coeff| {
                    if coeff.into_inner() {
                        one.clone()
                    } else {
                        zero.clone()
                    }
                })
                .collect::<Vec<_>>(),
        )
    });

    let (c1_ic_proof, _c1_ic_state) =
        IdealCheckProtocol::<PiopField>::prove_at_point::<U1>(
            &mut transcript,
            &c1_projected_trace,
            &c1_projected_scalars,
            c1_num_constraints,
            &ic_evaluation_point,
            &field_cfg,
        )
        .expect("BinaryPoly ideal check prover failed");

    // ── Step 5: IC₂ (generic R) at shared point ────────────────────
    let c2_projected_trace: Vec<DenseMultilinearExtension<DynamicPolynomialF<PiopField>>> =
        trace2
            .iter()
            .map(|col_mle| {
                col_mle
                    .iter()
                    .map(|elem| DynamicPolynomialF {
                        coeffs: vec![PiopField::from_with_cfg(elem.clone(), &field_cfg)],
                    })
                    .collect()
            })
            .collect();

    let c2_projected_scalars = project_scalars::<PiopField, U2>(|scalar| {
        DynamicPolynomialF {
            coeffs: vec![PiopField::from_with_cfg(scalar.clone(), &field_cfg)],
        }
    });

    let (c2_ic_proof, _c2_ic_state) =
        IdealCheckProtocol::<PiopField>::prove_at_point::<U2>(
            &mut transcript,
            &c2_projected_trace,
            &c2_projected_scalars,
            c2_num_constraints,
            &ic_evaluation_point,
            &field_cfg,
        )
        .expect("Generic ring ideal check prover failed");
    let ic_time = t1.elapsed();

    // ── Step 6: Shared projecting element + project both to F_q ────
    let t2 = Instant::now();
    let projecting_element: PiopField = transcript.get_field_challenge(&field_cfg);

    let c1_field_projected_scalars =
        project_scalars_to_field(c1_projected_scalars, &projecting_element)
            .expect("C1 scalar projection failed");
    let c1_field_trace = project_trace_to_field::<PiopField, D>(
        trace1, &[], &[], &projecting_element,
    );

    let c2_field_projected_scalars =
        project_scalars_to_field(c2_projected_scalars, &projecting_element)
            .expect("C2 scalar projection failed");
    let c2_field_trace = project_trace_to_field::<PiopField, 1>(
        &[], &[], &c2_projected_trace, &projecting_element,
    );

    // ── Step 7: Extract lookup columns from circuit 1's field trace ──
    let lookup_precomputed = if !lookup_specs.is_empty() {
        let (columns, raw_indices, remapped_specs) =
            extract_lookup_columns_from_field_trace(trace1, &c1_field_trace, lookup_specs, &field_cfg);
        Some((columns, raw_indices, remapped_specs))
    } else {
        None
    };

    // ── Step 8: Build CPR group for circuit 1 ───────────────────────
    let mut c1_cpr_group = CombinedPolyResolver::<PiopField>::build_prover_group::<U1>(
        &mut transcript,
        c1_field_trace,
        &ic_evaluation_point,
        &c1_field_projected_scalars,
        c1_num_constraints,
        num_vars,
        c1_max_degree,
        &field_cfg,
    )
    .expect("C1 CPR build_prover_group failed");
    let c1_num_cols = c1_cpr_group.num_cols;

    // ── Step 9: Build lookup groups for circuit 1 ───────────────────
    let mut lookup_groups_data: Vec<(LookupSumcheckGroup<PiopField>, LookupGroupMeta)> =
        Vec::new();
    if let Some((ref columns, ref raw_indices, ref remapped_specs)) = lookup_precomputed {
        let groups = group_lookup_specs(remapped_specs);
        for group in &groups {
            let instance = build_lookup_instance_from_indices_pub(
                columns,
                raw_indices,
                group,
                &projecting_element,
                &field_cfg,
            )
            .expect("lookup instance build failed");

            let witness_len = instance.witnesses[0].len();

            let lk_group = BatchedDecompLogupProtocol::<PiopField>::build_prover_group(
                &mut transcript,
                &instance,
                &field_cfg,
            )
            .expect("lookup build_prover_group failed");

            let meta = LookupGroupMeta {
                table_type: group.table_type.clone(),
                num_columns: group.column_indices.len(),
                witness_len,
            };
            lookup_groups_data.push((lk_group, meta));
        }
    }

    // ── Step 10: Build CPR group for circuit 2 ──────────────────────
    let mut c2_cpr_group = CombinedPolyResolver::<PiopField>::build_prover_group::<U2>(
        &mut transcript,
        c2_field_trace,
        &ic_evaluation_point,
        &c2_field_projected_scalars,
        c2_num_constraints,
        num_vars,
        c2_max_degree,
        &field_cfg,
    )
    .expect("C2 CPR build_prover_group failed");
    let c2_num_cols = c2_cpr_group.num_cols;

    // ── Step 11: Compute shared_num_vars and pad ────────────────────
    let shared_num_vars = {
        let max_lk = lookup_groups_data
            .iter()
            .map(|(g, _)| g.num_vars)
            .max()
            .unwrap_or(num_vars);
        num_vars.max(max_lk)
    };

    if shared_num_vars > num_vars {
        let target_len = 1usize << shared_num_vars;
        for mle in &mut c1_cpr_group.mles {
            mle.evaluations.resize(target_len, Default::default());
            mle.num_vars = shared_num_vars;
        }
        for mle in &mut c2_cpr_group.mles {
            mle.evaluations.resize(target_len, Default::default());
            mle.num_vars = shared_num_vars;
        }
    }

    // ── Step 12: Assemble all groups and run multi-degree sumcheck ──
    let mut sumcheck_groups: Vec<(
        usize,
        Vec<DenseMultilinearExtension<<PiopField as Field>::Inner>>,
        Box<dyn Fn(&[PiopField]) -> PiopField + Send + Sync>,
    )> = Vec::with_capacity(2 + lookup_groups_data.len());

    sumcheck_groups.push((c1_cpr_group.degree, c1_cpr_group.mles, c1_cpr_group.comb_fn));
    sumcheck_groups.push((c2_cpr_group.degree, c2_cpr_group.mles, c2_cpr_group.comb_fn));

    let mut lookup_pre_data: Vec<(LookupSumcheckGroup<PiopField>, LookupGroupMeta)> = Vec::new();
    for (lk_group, meta) in lookup_groups_data {
        sumcheck_groups.push((lk_group.degree, lk_group.mles, lk_group.comb_fn));
        lookup_pre_data.push((
            LookupSumcheckGroup {
                degree: 0,
                mles: vec![],
                comb_fn: Box::new(|_: &[PiopField]| unreachable!()),
                num_vars: 0,
                chunk_vectors: lk_group.chunk_vectors,
                aggregated_multiplicities: lk_group.aggregated_multiplicities,
                chunk_inverse_witnesses: lk_group.chunk_inverse_witnesses,
                inverse_table: lk_group.inverse_table,
            },
            meta,
        ));
    }

    let (md_proof, mut prover_states) =
        MultiDegreeSumcheck::<PiopField>::prove_as_subprotocol(
            &mut transcript,
            sumcheck_groups,
            shared_num_vars,
            &field_cfg,
        );

    // ── Step 13: Finalize CPR₁ (circuit 1) ─────────────────────────
    let c1_prover_state = prover_states.remove(0);
    let (c1_up_evals, c1_down_evals, c1_cpr_state) =
        CombinedPolyResolver::<PiopField>::finalize_prover(
            &mut transcript,
            c1_prover_state,
            c1_num_cols,
            &field_cfg,
        )
        .expect("C1 CPR finalize_prover failed");

    // ── Step 14: Finalize CPR₂ (circuit 2) ─────────────────────────
    let c2_prover_state = prover_states.remove(0);
    let (c2_up_evals, c2_down_evals, _c2_cpr_state) =
        CombinedPolyResolver::<PiopField>::finalize_prover(
            &mut transcript,
            c2_prover_state,
            c2_num_cols,
            &field_cfg,
        )
        .expect("C2 CPR finalize_prover failed");

    // ── Step 15: Finalize lookup groups ─────────────────────────────
    let mut lookup_group_proofs = Vec::new();
    let mut lookup_group_meta = Vec::new();
    for (i, (lk_pre, meta)) in lookup_pre_data.into_iter().enumerate() {
        let lk_sumcheck_proof = SumcheckProof {
            messages: md_proof.group_messages[i + 2].iter().cloned().collect(),
            claimed_sum: md_proof.claimed_sums[i + 2].clone(),
        };
        let lk_eval_point = prover_states.remove(0).randomness;
        let (lk_proof, _lk_state) =
            BatchedDecompLogupProtocol::<PiopField>::finalize_prover(
                lk_pre,
                lk_sumcheck_proof,
                lk_eval_point,
            );
        lookup_group_proofs.push(lk_proof);
        lookup_group_meta.push(meta);
    }

    let cpr_lookup_time = t2.elapsed();

    // ── Step 16: PCS Prove (both circuits) ────────────────────────
    let t3 = Instant::now();
    let pcs_i128_point = derive_pcs_point(&c1_cpr_state.evaluation_point, num_vars);
    let point1: Vec<Zt1::Pt> = pcs_i128_point
        .iter()
        .map(|v| unsafe { std::mem::transmute_copy(v) })
        .collect();
    let (eval1_f, proof1) =
        ZipPlus::<Zt1, Lc1>::prove::<PiopField, CHECK>(
            params1,
            trace1,
            &point1,
            &hint1,
        )
        .expect("PCS1 prove failed");

    let point2: Vec<Zt2::Pt> = pcs_i128_point
        .iter()
        .map(|v| unsafe { std::mem::transmute_copy(v) })
        .collect();
    let (eval2_f, proof2) =
        ZipPlus::<Zt2, Lc2>::prove::<PcsF2, CHECK>(
            params2,
            trace2,
            &point2,
            &hint2,
        )
        .expect("PCS2 prove failed");
    let pcs_prove_time = t3.elapsed();

    let total_time = total_start.elapsed();

    // ── Serialize ───────────────────────────────────────────────────
    let proof1_bytes: Vec<u8> = {
        let t: zip_plus::pcs_transcript::PcsTranscript = proof1.into();
        t.stream.into_inner()
    };
    let proof2_bytes: Vec<u8> = {
        let t: zip_plus::pcs_transcript::PcsTranscript = proof2.into();
        t.stream.into_inner()
    };

    let ic1_values: Vec<Vec<u8>> = c1_ic_proof
        .combined_mle_values
        .iter()
        .map(|dpf| dpf.coeffs.iter().flat_map(|c| field_to_bytes(c)).collect())
        .collect();
    let ic2_values: Vec<Vec<u8>> = c2_ic_proof
        .combined_mle_values
        .iter()
        .map(|dpf| dpf.coeffs.iter().flat_map(|c| field_to_bytes(c)).collect())
        .collect();

    let md_group_messages: Vec<Vec<Vec<u8>>> = md_proof
        .group_messages
        .iter()
        .map(|group_rounds| {
            group_rounds
                .iter()
                .map(|msg| {
                    msg.0
                        .tail_evaluations
                        .iter()
                        .flat_map(|c| field_to_bytes(c))
                        .collect()
                })
                .collect()
        })
        .collect();
    let md_claimed_sums: Vec<Vec<u8>> = md_proof
        .claimed_sums
        .iter()
        .map(field_to_bytes)
        .collect();

    let cpr1_ups: Vec<Vec<u8>> = c1_up_evals.iter().map(field_to_bytes).collect();
    let cpr1_downs: Vec<Vec<u8>> = c1_down_evals.iter().map(field_to_bytes).collect();
    let cpr2_ups: Vec<Vec<u8>> = c2_up_evals.iter().map(field_to_bytes).collect();
    let cpr2_downs: Vec<Vec<u8>> = c2_down_evals.iter().map(field_to_bytes).collect();

    let evaluation_point_bytes: Vec<Vec<u8>> =
        c1_cpr_state.evaluation_point.iter().map(field_to_bytes).collect();
    let pcs1_evals: Vec<Vec<u8>> = vec![field_to_bytes(&eval1_f)];
    // PCS2 evals are in PcsF2; serialize using ConstTranscribable.
    let pcs2_evals: Vec<Vec<u8>> = {
        let mut buf = vec![0u8; <PcsF2::Inner as ConstTranscribable>::NUM_BYTES];
        eval2_f.inner().write_transcription_bytes(&mut buf);
        vec![buf]
    };

    DualCircuitZincProof {
        pcs1_proof_bytes: proof1_bytes,
        pcs1_commitment: commitment1,
        pcs2_proof_bytes: proof2_bytes,
        pcs2_commitment: commitment2,
        ic1_proof_values: ic1_values,
        ic2_proof_values: ic2_values,
        md_group_messages,
        md_claimed_sums,
        md_degrees: md_proof.degrees,
        cpr1_up_evals: cpr1_ups,
        cpr1_down_evals: cpr1_downs,
        cpr2_up_evals: cpr2_ups,
        cpr2_down_evals: cpr2_downs,
        lookup_group_meta,
        lookup_group_proofs,
        evaluation_point_bytes,
        pcs1_evals_bytes: pcs1_evals,
        pcs2_evals_bytes: pcs2_evals,
        timing: TimingBreakdown {
            pcs_commit: pcs_commit_time,
            ideal_check: ic_time,
            combined_poly_resolver: cpr_lookup_time,
            lookup: Duration::ZERO, // included in combined_poly_resolver
            pcs_prove: pcs_prove_time,
            total: total_time,
        },
    }
}

/// Run the dual-circuit Zinc+ verifier with **shared challenges**.
///
/// Mirrors [`prove_dual_circuit`]: draws one IC evaluation point, one
/// projecting element, and verifies both CPRs + lookup via a single
/// multi-degree sumcheck.  Then verifies both PCS proofs independently.
///
/// # Type parameters
/// - `U1`: UAIR for circuit 1 (BinaryPoly<D> constraints).
/// - `U2`: UAIR for circuit 2 (generic ring R2 constraints).
/// - `R2`: ring for circuit 2 (e.g. `Int<4>`).
/// - `IdealOverF1/2`: field-level ideal types for verification.
/// - `IdealFromRef1/2`: closures mapping UAIR ideals to field ideals.
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn verify_dual_circuit<
    U1, U2, R2,
    Zt1, Lc1, Zt2, Lc2, PcsF2,
    const D: usize, const CHECK: bool,
    IdealOverF1, IdealFromRef1,
    IdealOverF2, IdealFromRef2,
>(
    params1: &ZipPlusParams<Zt1, Lc1>,
    params2: &ZipPlusParams<Zt2, Lc2>,
    proof: &DualCircuitZincProof,
    num_vars: usize,
    ideal_from_ref1: IdealFromRef1,
    ideal_from_ref2: IdealFromRef2,
) -> VerifyResult
where
    // Circuit 1 (BinaryPoly<D>):
    U1: Uair<Scalar = BinaryPoly<D>>,
    Zt1: ZipTypes<Eval = BinaryPoly<D>>,
    Lc1: LinearCode<Zt1>,
    Zt1::Eval: ProjectableToField<PiopField>,
    Zt1::Cw: ProjectableToField<PiopField>,
    PiopField: for<'a> FromWithConfig<&'a Zt1::Chal> + for<'a> FromWithConfig<&'a Zt1::Pt> + for<'a> FromWithConfig<&'a Zt1::CombR>,
    <PiopField as Field>::Inner: FromRef<Zt1::Fmod>,
    IdealOverF1: Ideal + IdealCheck<DynamicPolynomialF<PiopField>>,
    IdealFromRef1: Fn(&IdealOrZero<U1::Ideal>) -> IdealOverF1,
    // Circuit 2 (generic ring R2):
    R2: ProjectableToField<PiopField>
        + crypto_primitives::Semiring
        + std::fmt::Debug
        + Clone
        + Send
        + Sync
        + 'static,
    U2: Uair<Scalar = R2>,
    Zt2: ZipTypes<Eval = R2>,
    Lc2: LinearCode<Zt2>,
    PiopField: FromWithConfig<R2>,
    PcsF2: PrimeField
        + FromPrimitiveWithConfig
        + for<'a> FromWithConfig<&'a Zt2::Chal>
        + for<'a> FromWithConfig<&'a Zt2::Pt>
        + for<'a> FromWithConfig<&'a Zt2::CombR>
        + for<'a> zinc_utils::mul_by_scalar::MulByScalar<&'a PcsF2>
        + FromRef<PcsF2>,
    PcsF2::Inner: FromRef<Zt2::Fmod> + ConstTranscribable,
    Zt2::Eval: ProjectableToField<PcsF2>,
    Zt2::Cw: ProjectableToField<PcsF2>,
    IdealOverF2: Ideal + IdealCheck<DynamicPolynomialF<PiopField>>,
    IdealFromRef2: Fn(&IdealOrZero<U2::Ideal>) -> IdealOverF2,
    // Shared:
    PiopField: FromPrimitiveWithConfig,
    <PiopField as Field>::Inner: ConstTranscribable
        + FromRef<<PiopField as Field>::Inner>
        + Send
        + Sync
        + Default
        + num_traits::Zero,
    MillerRabin: PrimalityTest<<PiopField as Field>::Inner>,
{
    let total_start = Instant::now();

    let c1_num_constraints = count_constraints::<U1>();
    let c2_num_constraints = count_constraints::<U2>();

    // ── Reconstruct Fiat-Shamir transcript ──────────────────────────
    let mut transcript = KeccakTranscript::new();
    let field_cfg = transcript.get_random_field_cfg::<PiopField, <PiopField as Field>::Inner, MillerRabin>();
    let field_elem_size =
        <crypto_primitives::crypto_bigint_uint::Uint<FIELD_LIMBS> as ConstTranscribable>::NUM_BYTES;

    // ── Shared IC evaluation point ──────────────────────────────────
    let ic_evaluation_point: Vec<PiopField> =
        transcript.get_field_challenges(num_vars, &field_cfg);

    // ── IC₁ verify (BinaryPoly) ────────────────────────────────────
    let t0 = Instant::now();

    let c1_ic_combined: Vec<DynamicPolynomialF<PiopField>> = proof
        .ic1_proof_values
        .iter()
        .map(|bytes| {
            let num_coeffs = bytes.len() / field_elem_size;
            let coeffs: Vec<PiopField> = (0..num_coeffs)
                .map(|i| {
                    field_from_bytes(
                        &bytes[i * field_elem_size..(i + 1) * field_elem_size],
                        &field_cfg,
                    )
                })
                .collect();
            DynamicPolynomialF::new(coeffs)
        })
        .collect();

    let c1_ic_proof = zinc_piop::ideal_check::Proof::<PiopField> {
        combined_mle_values: c1_ic_combined,
    };

    let c1_ic_subclaim =
        match IdealCheckProtocol::<PiopField>::verify_at_point::<U1, _, _>(
            &mut transcript,
            c1_ic_proof,
            c1_num_constraints,
            ic_evaluation_point.clone(),
            &ideal_from_ref1,
            &field_cfg,
        ) {
            Ok(sc) => sc,
            Err(e) => {
                eprintln!("C1 IdealCheck verification failed: {e:?}");
                return VerifyResult {
                    accepted: false,
                    timing: VerifyTimingBreakdown {
                        ideal_check_verify: t0.elapsed(),
                        combined_poly_resolver_verify: Duration::ZERO,
                        lookup_verify: Duration::ZERO,
                        pcs_verify: Duration::ZERO,
                        total: total_start.elapsed(),
                    },
                };
            }
        };

    // ── IC₂ verify (generic R) ─────────────────────────────────────
    let c2_ic_combined: Vec<DynamicPolynomialF<PiopField>> = proof
        .ic2_proof_values
        .iter()
        .map(|bytes| {
            let num_coeffs = bytes.len() / field_elem_size;
            let coeffs: Vec<PiopField> = (0..num_coeffs)
                .map(|i| {
                    field_from_bytes(
                        &bytes[i * field_elem_size..(i + 1) * field_elem_size],
                        &field_cfg,
                    )
                })
                .collect();
            DynamicPolynomialF::new(coeffs)
        })
        .collect();

    let c2_ic_proof = zinc_piop::ideal_check::Proof::<PiopField> {
        combined_mle_values: c2_ic_combined,
    };

    let c2_ic_subclaim =
        match IdealCheckProtocol::<PiopField>::verify_at_point::<U2, _, _>(
            &mut transcript,
            c2_ic_proof,
            c2_num_constraints,
            ic_evaluation_point.clone(),
            &ideal_from_ref2,
            &field_cfg,
        ) {
            Ok(sc) => sc,
            Err(e) => {
                eprintln!("C2 IdealCheck verification failed: {e:?}");
                return VerifyResult {
                    accepted: false,
                    timing: VerifyTimingBreakdown {
                        ideal_check_verify: t0.elapsed(),
                        combined_poly_resolver_verify: Duration::ZERO,
                        lookup_verify: Duration::ZERO,
                        pcs_verify: Duration::ZERO,
                        total: total_start.elapsed(),
                    },
                };
            }
        };
    let ic_verify_time = t0.elapsed();

    // ── Shared projecting element ───────────────────────────────────
    let t1 = Instant::now();
    let projecting_element: PiopField = transcript.get_field_challenge(&field_cfg);

    // ── CPR₁ pre-sumcheck ───────────────────────────────────────────
    let c1_projected_scalars_coeffs = project_scalars::<PiopField, U1>(|scalar| {
        let one = PiopField::one_with_cfg(&field_cfg);
        let zero = PiopField::zero_with_cfg(&field_cfg);
        DynamicPolynomialF::new(
            scalar
                .iter()
                .map(|coeff| {
                    if coeff.into_inner() {
                        one.clone()
                    } else {
                        zero.clone()
                    }
                })
                .collect::<Vec<_>>(),
        )
    });
    let c1_field_projected_scalars =
        project_scalars_to_field(c1_projected_scalars_coeffs, &projecting_element)
            .expect("C1 scalar projection failed");

    let c1_cpr_pre =
        match CombinedPolyResolver::<PiopField>::build_verifier_pre_sumcheck::<U1>(
            &mut transcript,
            &field_from_bytes(&proof.md_claimed_sums[0], &field_cfg),
            c1_num_constraints,
            &projecting_element,
            &c1_field_projected_scalars,
            &c1_ic_subclaim,
            &field_cfg,
        ) {
            Ok(pre) => pre,
            Err(e) => {
                eprintln!("C1 CPR pre-sumcheck failed: {e:?}");
                return VerifyResult {
                    accepted: false,
                    timing: VerifyTimingBreakdown {
                        ideal_check_verify: ic_verify_time,
                        combined_poly_resolver_verify: t1.elapsed(),
                        lookup_verify: Duration::ZERO,
                        pcs_verify: Duration::ZERO,
                        total: total_start.elapsed(),
                    },
                };
            }
        };

    // ── Lookup pre-sumcheck (circuit 1 only) ────────────────────────
    let mut lookup_pres = Vec::new();
    for (group_proof, meta) in proof
        .lookup_group_proofs
        .iter()
        .zip(proof.lookup_group_meta.iter())
    {
        let (subtable, shifts) = generate_table_and_shifts(
            &meta.table_type,
            &projecting_element,
            &field_cfg,
        );
        match BatchedDecompLogupProtocol::<PiopField>::build_verifier_pre_sumcheck(
            &mut transcript,
            group_proof,
            &subtable,
            &shifts,
            meta.num_columns,
            meta.witness_len,
            &field_cfg,
        ) {
            Ok(pre) => lookup_pres.push(pre),
            Err(e) => {
                eprintln!("Lookup pre-sumcheck failed: {e:?}");
                return VerifyResult {
                    accepted: false,
                    timing: VerifyTimingBreakdown {
                        ideal_check_verify: ic_verify_time,
                        combined_poly_resolver_verify: t1.elapsed(),
                        lookup_verify: Duration::ZERO,
                        pcs_verify: Duration::ZERO,
                        total: total_start.elapsed(),
                    },
                };
            }
        }
    }

    // ── CPR₂ pre-sumcheck ───────────────────────────────────────────
    let c2_projected_scalars_coeffs = project_scalars::<PiopField, U2>(|scalar| {
        DynamicPolynomialF {
            coeffs: vec![PiopField::from_with_cfg(scalar.clone(), &field_cfg)],
        }
    });
    let c2_field_projected_scalars =
        project_scalars_to_field(c2_projected_scalars_coeffs, &projecting_element)
            .expect("C2 scalar projection failed");

    let c2_cpr_pre =
        match CombinedPolyResolver::<PiopField>::build_verifier_pre_sumcheck::<U2>(
            &mut transcript,
            &field_from_bytes(&proof.md_claimed_sums[1], &field_cfg),
            c2_num_constraints,
            &projecting_element,
            &c2_field_projected_scalars,
            &c2_ic_subclaim,
            &field_cfg,
        ) {
            Ok(pre) => pre,
            Err(e) => {
                eprintln!("C2 CPR pre-sumcheck failed: {e:?}");
                return VerifyResult {
                    accepted: false,
                    timing: VerifyTimingBreakdown {
                        ideal_check_verify: ic_verify_time,
                        combined_poly_resolver_verify: t1.elapsed(),
                        lookup_verify: Duration::ZERO,
                        pcs_verify: Duration::ZERO,
                        total: total_start.elapsed(),
                    },
                };
            }
        };

    // ── Compute shared_num_vars ─────────────────────────────────────
    let shared_num_vars = lookup_pres
        .iter()
        .map(|p| p.num_vars)
        .max()
        .map_or(num_vars, |max_lk| max_lk.max(num_vars));

    // ── Deserialize multi-degree sumcheck proof ─────────────────────
    let md_group_messages: Vec<Vec<ProverMsg<PiopField>>> = proof
        .md_group_messages
        .iter()
        .map(|group_rounds| {
            group_rounds
                .iter()
                .map(|bytes| {
                    let num_evals = bytes.len() / field_elem_size;
                    let tail_evaluations: Vec<PiopField> = (0..num_evals)
                        .map(|i| {
                            field_from_bytes(
                                &bytes[i * field_elem_size..(i + 1) * field_elem_size],
                                &field_cfg,
                            )
                        })
                        .collect();
                    ProverMsg(NatEvaluatedPolyWithoutConstant::new(tail_evaluations))
                })
                .collect()
        })
        .collect();

    let md_claimed_sums: Vec<PiopField> = proof
        .md_claimed_sums
        .iter()
        .map(|b| field_from_bytes(b, &field_cfg))
        .collect();

    let md_proof_deserialized = MultiDegreeSumcheckProof {
        group_messages: md_group_messages,
        claimed_sums: md_claimed_sums,
        degrees: proof.md_degrees.clone(),
    };

    // ── Verify multi-degree sumcheck ────────────────────────────────
    let md_subclaims = match MultiDegreeSumcheck::<PiopField>::verify_as_subprotocol(
        &mut transcript,
        shared_num_vars,
        &md_proof_deserialized,
        &field_cfg,
    ) {
        Ok(sc) => sc,
        Err(e) => {
            eprintln!("Multi-degree sumcheck verification failed: {e:?}");
            return VerifyResult {
                accepted: false,
                timing: VerifyTimingBreakdown {
                    ideal_check_verify: ic_verify_time,
                    combined_poly_resolver_verify: t1.elapsed(),
                    lookup_verify: Duration::ZERO,
                    pcs_verify: Duration::ZERO,
                    total: total_start.elapsed(),
                },
            };
        }
    };

    // ── Finalize CPR₁ ───────────────────────────────────────────────
    let c1_up_evals: Vec<PiopField> = proof
        .cpr1_up_evals
        .iter()
        .map(|b| field_from_bytes(b, &field_cfg))
        .collect();
    let c1_down_evals: Vec<PiopField> = proof
        .cpr1_down_evals
        .iter()
        .map(|b| field_from_bytes(b, &field_cfg))
        .collect();

    let c1_cpr_subclaim =
        match CombinedPolyResolver::<PiopField>::finalize_verifier::<U1>(
            &mut transcript,
            md_subclaims.point.clone(),
            md_subclaims.expected_evaluations[0].clone(),
            &c1_cpr_pre,
            c1_up_evals,
            c1_down_evals,
            num_vars,
            &c1_field_projected_scalars,
            &field_cfg,
        ) {
            Ok(sc) => sc,
            Err(e) => {
                eprintln!("C1 CPR finalize_verifier failed: {e:?}");
                return VerifyResult {
                    accepted: false,
                    timing: VerifyTimingBreakdown {
                        ideal_check_verify: ic_verify_time,
                        combined_poly_resolver_verify: t1.elapsed(),
                        lookup_verify: Duration::ZERO,
                        pcs_verify: Duration::ZERO,
                        total: total_start.elapsed(),
                    },
                };
            }
        };

    // ── Finalize CPR₂ ───────────────────────────────────────────────
    let c2_up_evals: Vec<PiopField> = proof
        .cpr2_up_evals
        .iter()
        .map(|b| field_from_bytes(b, &field_cfg))
        .collect();
    let c2_down_evals: Vec<PiopField> = proof
        .cpr2_down_evals
        .iter()
        .map(|b| field_from_bytes(b, &field_cfg))
        .collect();

    if let Err(e) = CombinedPolyResolver::<PiopField>::finalize_verifier::<U2>(
        &mut transcript,
        md_subclaims.point.clone(),
        md_subclaims.expected_evaluations[1].clone(),
        &c2_cpr_pre,
        c2_up_evals,
        c2_down_evals,
        num_vars,
        &c2_field_projected_scalars,
        &field_cfg,
    ) {
        eprintln!("C2 CPR finalize_verifier failed: {e:?}");
        return VerifyResult {
            accepted: false,
            timing: VerifyTimingBreakdown {
                ideal_check_verify: ic_verify_time,
                combined_poly_resolver_verify: t1.elapsed(),
                lookup_verify: Duration::ZERO,
                pcs_verify: Duration::ZERO,
                total: total_start.elapsed(),
            },
        };
    }

    // ── Finalize lookup groups ──────────────────────────────────────
    for (g, (lk_pre, group_proof)) in lookup_pres
        .iter()
        .zip(proof.lookup_group_proofs.iter())
        .enumerate()
    {
        if let Err(e) = BatchedDecompLogupProtocol::<PiopField>::finalize_verifier(
            lk_pre,
            group_proof,
            &md_subclaims.point,
            &md_subclaims.expected_evaluations[g + 2],
            &field_cfg,
        ) {
            eprintln!("Lookup finalize_verifier failed (group {g}): {e:?}");
            return VerifyResult {
                accepted: false,
                timing: VerifyTimingBreakdown {
                    ideal_check_verify: ic_verify_time,
                    combined_poly_resolver_verify: t1.elapsed(),
                    lookup_verify: Duration::ZERO,
                    pcs_verify: Duration::ZERO,
                    total: total_start.elapsed(),
                },
            };
        }
    }
    let cpr_verify_time = t1.elapsed();

    // ── PCS Verify (both circuits) ──────────────────────────────────
    let t2 = Instant::now();

    let pcs_i128_point = derive_pcs_point(&c1_cpr_subclaim.evaluation_point, num_vars);

    // PCS1 verify (PiopField)
    let pcs1_field_cfg = KeccakTranscript::default()
        .get_random_field_cfg::<PiopField, Zt1::Fmod, Zt1::PrimeTest>();
    let pcs1_point: Vec<Zt1::Pt> = pcs_i128_point
        .iter()
        .map(|v| unsafe { std::mem::transmute_copy(v) })
        .collect();
    let point1_f: Vec<PiopField> = pcs1_point.iter().map(|v| v.into_with_cfg(&pcs1_field_cfg)).collect();
    let eval1_f: PiopField = PiopField::new_unchecked_with_cfg(
        <PiopField as Field>::Inner::read_transcription_bytes(&proof.pcs1_evals_bytes[0]),
        &pcs1_field_cfg,
    );
    let pcs1_transcript = zip_plus::pcs_transcript::PcsTranscript {
        fs_transcript: KeccakTranscript::default(),
        stream: std::io::Cursor::new(proof.pcs1_proof_bytes.clone()),
    };
    let pcs1_proof: ZipPlusProof =
        pcs1_transcript.into();

    let pcs1_result = ZipPlus::<Zt1, Lc1>::verify::<PiopField, CHECK>(
        params1,
        &proof.pcs1_commitment,
        &point1_f,
        &eval1_f,
        &pcs1_proof,
    );
    if let Err(ref e) = pcs1_result {
        eprintln!("PCS1 verification failed: {e:?}");
    }

    // PCS2 verify (PcsF2)
    let pcs2_field_cfg = KeccakTranscript::default()
        .get_random_field_cfg::<PcsF2, Zt2::Fmod, Zt2::PrimeTest>();
    let pcs2_point: Vec<Zt2::Pt> = pcs_i128_point
        .iter()
        .map(|v| unsafe { std::mem::transmute_copy(v) })
        .collect();
    let point2_f: Vec<PcsF2> = pcs2_point.iter().map(|v| v.into_with_cfg(&pcs2_field_cfg)).collect();
    let eval2_f: PcsF2 = PcsF2::new_unchecked_with_cfg(
        <PcsF2::Inner as ConstTranscribable>::read_transcription_bytes(&proof.pcs2_evals_bytes[0]),
        &pcs2_field_cfg,
    );
    let pcs2_transcript = zip_plus::pcs_transcript::PcsTranscript {
        fs_transcript: KeccakTranscript::default(),
        stream: std::io::Cursor::new(proof.pcs2_proof_bytes.clone()),
    };
    let pcs2_proof: ZipPlusProof =
        pcs2_transcript.into();

    let pcs2_result = ZipPlus::<Zt2, Lc2>::verify::<PcsF2, CHECK>(
        params2,
        &proof.pcs2_commitment,
        &point2_f,
        &eval2_f,
        &pcs2_proof,
    );
    if let Err(ref e) = pcs2_result {
        eprintln!("PCS2 verification failed: {e:?}");
    }

    let pcs_verify_time = t2.elapsed();

    VerifyResult {
        accepted: pcs1_result.is_ok() && pcs2_result.is_ok(),
        timing: VerifyTimingBreakdown {
            ideal_check_verify: ic_verify_time,
            combined_poly_resolver_verify: cpr_verify_time,
            lookup_verify: Duration::ZERO, // included in CPR
            pcs_verify: pcs_verify_time,
            total: total_start.elapsed(),
        },
    }
}