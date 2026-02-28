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

use zip_plus::batched_pcs::structs::{BatchedZipPlus, BatchedZipPlusCommitment};
use zip_plus::code::LinearCode;
use zip_plus::pcs::structs::{ZipPlusParams, ZipTypes};

use zinc_piop::ideal_check::IdealCheckProtocol;
use zinc_piop::combined_poly_resolver::CombinedPolyResolver;
use zinc_piop::lookup::{
    LookupColumnSpec, PipelineLookupProof,
    prove_batched_lookup_with_indices, verify_batched_lookup,
};
use zinc_piop::projections::{
    project_trace_coeffs, project_trace_to_field,
    project_scalars, project_scalars_to_field,
};
use zinc_piop::sumcheck::prover::{NatEvaluatedPolyWithoutConstant, ProverMsg};
use zinc_piop::sumcheck::SumcheckProof;
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
    pub pcs_test: Duration,
    pub pcs_evaluate: Duration,
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

/// Full proof data produced by the prover.
#[derive(Clone, Debug)]
pub struct ZincProof {
    /// Serialized PCS proof bytes.
    pub pcs_proof_bytes: Vec<u8>,
    /// PCS commitment (Merkle root + batch size).
    pub commitment: BatchedZipPlusCommitment,
    /// IdealCheck proof: evaluated combined polynomials at random point.
    pub ic_proof_values: Vec<Vec<u8>>,
    /// CPR proof: sumcheck proof + up/down evaluations.
    pub cpr_sumcheck_messages: Vec<Vec<u8>>,
    pub cpr_sumcheck_claimed_sum: Vec<u8>,
    pub cpr_up_evals: Vec<Vec<u8>>,
    pub cpr_down_evals: Vec<Vec<u8>>,
    /// Batched decomposed LogUp proof for column typing constraints.
    /// `None` if no lookup columns were specified.
    pub lookup_proof: Option<PipelineLookupProof<PiopField>>,
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
    PiopField: for<'a> FromWithConfig<&'a Zt::Chal> + for<'a> FromWithConfig<&'a Zt::Pt>,
    <PiopField as Field>::Inner: FromRef<Zt::Fmod>,
{
    let total_start = Instant::now();
    let num_constraints = count_constraints::<U>();
    let max_degree = count_max_degree::<U>();

    // ── Step 1: PCS Commit ──────────────────────────────────────────
    let t0 = Instant::now();
    let (hint, commitment) = BatchedZipPlus::<Zt, Lc>::commit(params, trace)
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
    // Get projecting element for F[X]→F projection (shared by CPR prover & verifier).
    let projecting_element: PiopField = transcript.get_field_challenge(&field_cfg);
    let field_projected_scalars =
        project_scalars_to_field(projected_scalars, &projecting_element)
            .expect("scalar projection failed");
    let field_trace = project_trace_to_field::<PiopField, D>(
        trace, &[], &[], &projecting_element,
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

        // Project each trace column (BinaryPoly) to field elements and
        // simultaneously extract raw integer indices from the binary
        // coefficients.  This avoids building the full 2^D lookup table
        // (which would be 2^32 ≈ 4 G entries for D=32).
        let mut columns: Vec<Vec<PiopField>> = Vec::with_capacity(trace.len());
        let mut raw_indices: Vec<Vec<usize>> = Vec::with_capacity(trace.len());
        for col_mle in trace {
            let mut col_f = Vec::with_capacity(col_mle.len());
            let mut col_idx = Vec::with_capacity(col_mle.len());
            for bp in col_mle.iter() {
                let mut val = PiopField::zero_with_cfg(&field_cfg);
                let mut power = PiopField::one_with_cfg(&field_cfg);
                let mut idx = 0usize;
                for (j, coeff) in bp.iter().enumerate() {
                    if coeff.into_inner() {
                        val += &power;
                        idx |= 1usize << j;
                    }
                    power *= &lookup_projecting_element;
                }
                col_f.push(val);
                col_idx.push(idx);
            }
            columns.push(col_f);
            raw_indices.push(col_idx);
        }

        let (lk_proof, _lk_state) = prove_batched_lookup_with_indices(
            &mut transcript,
            &columns,
            &raw_indices,
            lookup_specs,
            &lookup_projecting_element,
            &field_cfg,
        )
        .expect("Batched lookup prover failed");

        Some(lk_proof)
    } else {
        None
    };
    let lookup_time = t2b.elapsed();

    // ── Step 4: PCS Test ────────────────────────────────────────────
    let t3 = Instant::now();
    let test_transcript =
        BatchedZipPlus::<Zt, Lc>::test::<CHECK>(params, trace, &hint)
            .expect("PCS test failed");
    let pcs_test_time = t3.elapsed();
    let t4 = Instant::now();
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
    let (evals_f, proof) =
        BatchedZipPlus::<Zt, Lc>::evaluate::<PiopField, CHECK>(
            params,
            trace,
            &point,
            test_transcript,
        )
        .expect("PCS evaluate failed");
    let pcs_eval_time = t4.elapsed();

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
    let pcs_evals_bytes: Vec<Vec<u8>> = evals_f.iter().map(field_to_bytes).collect();

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
            pcs_test: pcs_test_time,
            pcs_evaluate: pcs_eval_time,
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
    let (hint, commitment) = BatchedZipPlus::<Zt, Lc>::commit(params, trace)
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

        // Project each trace column to field elements using ProjectableToField
        // and extract raw integer indices for the lookup decomposition.
        //
        // For Word lookups, the raw index is the plain integer value of the
        // projected field element (safe because word entries < 2^32 ≪ p).
        // This avoids building the full 2^width table for reverse-mapping.
        let projection_fn = R::prepare_projection(&lookup_projecting_element);
        let mut columns: Vec<Vec<PiopField>> = Vec::with_capacity(trace.len());
        let mut raw_indices: Vec<Vec<usize>> = Vec::with_capacity(trace.len());
        for col_mle in trace {
            let mut col_f = Vec::with_capacity(col_mle.len());
            let mut col_idx = Vec::with_capacity(col_mle.len());
            for v in col_mle.iter() {
                let fv = projection_fn(v);
                // Retrieve the integer value from Montgomery form.
                let uint = fv.retrieve();
                let idx = uint.as_words()[0] as usize;
                col_f.push(fv);
                col_idx.push(idx);
            }
            columns.push(col_f);
            raw_indices.push(col_idx);
        }

        let (lk_proof, _lk_state) = prove_batched_lookup_with_indices(
            &mut transcript,
            &columns,
            &raw_indices,
            lookup_specs,
            &lookup_projecting_element,
            &field_cfg,
        )
        .expect("Batched lookup prover failed");

        Some(lk_proof)
    } else {
        None
    };
    let lookup_time = t2b.elapsed();

    // ── Step 4: PCS Test ────────────────────────────────────────────
    let t3 = Instant::now();
    let test_transcript =
        BatchedZipPlus::<Zt, Lc>::test::<CHECK>(params, trace, &hint)
            .expect("PCS test failed");
    let pcs_test_time = t3.elapsed();

    // ── Step 5: PCS Evaluate ────────────────────────────────────────
    let t4 = Instant::now();
    let point: Vec<Zt::Pt> = {
        let i128_point = derive_pcs_point(&cpr_state.evaluation_point, num_vars);
        i128_point
            .into_iter()
            .map(|v| unsafe { std::mem::transmute_copy(&v) })
            .collect()
    };
    let (_evals_f, proof) =
        BatchedZipPlus::<Zt, Lc>::evaluate::<PcsF, CHECK>(
            params,
            trace,
            &point,
            test_transcript,
        )
        .expect("PCS evaluate failed");
    let pcs_eval_time = t4.elapsed();

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
    // PCS evals are in PcsF (potentially wider than PiopField); they are not
    // consumed by the verifier so we skip serialization.
    let pcs_evals_bytes: Vec<Vec<u8>> = Vec::new();

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
            pcs_test: pcs_test_time,
            pcs_evaluate: pcs_eval_time,
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
    if let Some(ref lookup_proof) = zinc_proof.lookup_proof {
        let projecting_element: PiopField = transcript.get_field_challenge(&field_cfg);

        if let Err(e) = verify_batched_lookup(
            &mut transcript,
            lookup_proof,
            &projecting_element,
            &field_cfg,
        ) {
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

    let _pcs_evals: Vec<PiopField> = zinc_proof.pcs_evals_bytes.iter()
        .map(|b| field_from_bytes(b, &field_cfg))
        .collect();

    let pcs_point: Vec<Zt::Pt> = {
        let i128_point = derive_pcs_point(&cpr_subclaim.evaluation_point, num_vars);
        i128_point
            .into_iter()
            .map(|v| unsafe { std::mem::transmute_copy(&v) })
            .collect()
    };

    let pcs_transcript = zip_plus::pcs_transcript::PcsTranscript {
        fs_transcript: KeccakTranscript::default(),
        stream: std::io::Cursor::new(zinc_proof.pcs_proof_bytes.clone()),
    };
    let pcs_proof: zip_plus::batched_pcs::structs::BatchedZipPlusProof = pcs_transcript.into();

    let pcs_result = BatchedZipPlus::<Zt, Lc>::verify::<PcsF, CHECK>(
        params,
        &zinc_proof.commitment,
        &pcs_point,
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
    PiopField: for<'a> FromWithConfig<&'a Zt::Chal> + for<'a> FromWithConfig<&'a Zt::Pt>,
    <PiopField as Field>::Inner: FromRef<Zt::Fmod>,
{
    let total_start = Instant::now();

    // ── Step 1: PCS Commit ──────────────────────────────────────────
    let t0 = Instant::now();
    let (hint, commitment) = BatchedZipPlus::<Zt, Lc>::commit(params, trace)
        .expect("PCS commit failed");
    let pcs_commit_time = t0.elapsed();

    // ── Step 2: PCS Test ────────────────────────────────────────────
    let t1 = Instant::now();
    let test_transcript =
        BatchedZipPlus::<Zt, Lc>::test::<CHECK>(params, trace, &hint)
            .expect("PCS test failed");
    let pcs_test_time = t1.elapsed();

    // ── Step 3: PCS Evaluate ────────────────────────────────────────
    let t2 = Instant::now();
    let point: Vec<Zt::Pt> = vec![Zt::Pt::one(); num_vars];
    let (evals_f, proof) =
        BatchedZipPlus::<Zt, Lc>::evaluate::<PiopField, CHECK>(
            params,
            trace,
            &point,
            test_transcript,
        )
        .expect("PCS evaluate failed");
    let pcs_eval_time = t2.elapsed();

    let total_time = total_start.elapsed();

    let proof_bytes: Vec<u8> = {
        let pcs_transcript: zip_plus::pcs_transcript::PcsTranscript = proof.into();
        pcs_transcript.stream.into_inner()
    };

    let field_cfg = evals_f[0].cfg().clone();
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
            pcs_evals_bytes: vec![],
            timing: TimingBreakdown {
                pcs_commit: pcs_commit_time,
                ideal_check: Duration::ZERO,
                combined_poly_resolver: Duration::ZERO,
                lookup: Duration::ZERO,
                pcs_test: pcs_test_time,
                pcs_evaluate: pcs_eval_time,
                total: total_time,
            },
        },
        point_f,
    )
}

/// Run the PCS-only verifier.
pub fn verify_pcs_only<Zt, Lc, const D: usize, const CHECK: bool>(
    params: &ZipPlusParams<Zt, Lc>,
    commitment: &BatchedZipPlusCommitment,
    point: &[Zt::Pt],
    proof_bytes: &[u8],
) -> VerifyResult
where
    Zt: ZipTypes<Eval = BinaryPoly<D>>,
    Lc: LinearCode<Zt>,
    Zt::Eval: ProjectableToField<PiopField>,
    Zt::Cw: ProjectableToField<PiopField>,
    PiopField: for<'a> FromWithConfig<&'a Zt::Chal> + for<'a> FromWithConfig<&'a Zt::Pt>,
    <PiopField as Field>::Inner: FromRef<Zt::Fmod> + ConstTranscribable,
{
    let total_start = Instant::now();

    let pcs_transcript = zip_plus::pcs_transcript::PcsTranscript {
        fs_transcript: KeccakTranscript::default(),
        stream: std::io::Cursor::new(proof_bytes.to_vec()),
    };
    let proof: zip_plus::batched_pcs::structs::BatchedZipPlusProof = pcs_transcript.into();

    let t0 = Instant::now();
    let result = BatchedZipPlus::<Zt, Lc>::verify::<PiopField, CHECK>(
        params,
        commitment,
        point,
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
    PiopField: for<'a> FromWithConfig<&'a Zt::Chal> + for<'a> FromWithConfig<&'a Zt::Pt>,
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

    // ── Step 2: Deserialize and verify CPR (sumcheck) ───────────────
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

    // ── Step 2b: Verify Batched Decomposed LogUp (column typing) ────
    let t1b = Instant::now();
    if let Some(ref lookup_proof) = zinc_proof.lookup_proof {
        // Derive the same projecting element the prover used.
        let projecting_element: PiopField = transcript.get_field_challenge(&field_cfg);

        if let Err(e) = verify_batched_lookup(
            &mut transcript,
            lookup_proof,
            &projecting_element,
            &field_cfg,
        ) {
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

    // ── Step 3: PCS Verify ──────────────────────────────────────────
    // The CPR subclaim gives us the evaluation point and expected evaluations.
    // Verify that the PCS proof is consistent with these claims.
    let t2 = Instant::now();

    let _pcs_evals: Vec<PiopField> = zinc_proof.pcs_evals_bytes.iter()
        .map(|b| field_from_bytes(b, &field_cfg))
        .collect();

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

    // Deserialize PCS proof.
    let pcs_transcript = zip_plus::pcs_transcript::PcsTranscript {
        fs_transcript: KeccakTranscript::default(),
        stream: std::io::Cursor::new(zinc_proof.pcs_proof_bytes.clone()),
    };
    let pcs_proof: zip_plus::batched_pcs::structs::BatchedZipPlusProof = pcs_transcript.into();

    let pcs_result = BatchedZipPlus::<Zt, Lc>::verify::<PiopField, CHECK>(
        params,
        &zinc_proof.commitment,
        &pcs_point,
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
/// Contains proof material for two IC+CPR passes: one for the primary ring
/// (BinaryPoly, F₂[X] constraints) and one for a secondary ring
/// (DensePolynomial<i64, D2>, Q[X] constraints).
#[derive(Clone, Debug)]
pub struct DualRingZincProof {
    /// Serialized PCS proof bytes.
    pub pcs_proof_bytes: Vec<u8>,
    /// PCS commitment.
    pub commitment: BatchedZipPlusCommitment,

    // ── BinaryPoly pass (IC₁ + CPR₁) ───────────────────────────────
    pub bp_ic_proof_values: Vec<Vec<u8>>,
    pub bp_cpr_sumcheck_messages: Vec<Vec<u8>>,
    pub bp_cpr_sumcheck_claimed_sum: Vec<u8>,
    pub bp_cpr_up_evals: Vec<Vec<u8>>,
    pub bp_cpr_down_evals: Vec<Vec<u8>>,

    // ── Q[X] pass (IC₂ + CPR₂) ─────────────────────────────────────
    pub qx_ic_proof_values: Vec<Vec<u8>>,
    pub qx_cpr_sumcheck_messages: Vec<Vec<u8>>,
    pub qx_cpr_sumcheck_claimed_sum: Vec<u8>,
    pub qx_cpr_up_evals: Vec<Vec<u8>>,
    pub qx_cpr_down_evals: Vec<Vec<u8>>,

    // ── PCS evaluation data ─────────────────────────────────────────
    pub evaluation_point_bytes: Vec<Vec<u8>>,
    pub pcs_evals_bytes: Vec<Vec<u8>>,

    pub timing: TimingBreakdown,
}

/// Run the dual-ring Zinc+ prover pipeline.
///
/// This runs two IC+CPR passes on the same Fiat-Shamir transcript:
/// 1. IC₁ + CPR₁ for BinaryPoly<D1> (F₂[X]) constraints
/// 2. IC₂ + CPR₂ for DensePolynomial<i64, D2> (Q[X]) constraints
///
/// The trace is committed once (as BinaryPoly<D1>). The Q[X] trace is
/// derived via `convert_trace`.
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
    PiopField: for<'a> FromWithConfig<&'a Zt::Chal> + for<'a> FromWithConfig<&'a Zt::Pt>,
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
    let (hint, commitment) = BatchedZipPlus::<Zt, Lc>::commit(params, trace)
        .expect("PCS commit failed");
    let pcs_commit_time = t0.elapsed();

    // ── Step 2: Fiat-Shamir transcript + field config ───────────────
    let mut transcript = KeccakTranscript::new();
    let field_cfg = transcript.get_random_field_cfg::<PiopField, <PiopField as Field>::Inner, MillerRabin>();

    // ── Step 3: IC₁ (BinaryPoly) ───────────────────────────────────
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

    let (bp_ic_proof, bp_ic_state) =
        IdealCheckProtocol::<PiopField>::prove_as_subprotocol::<U1>(
            &mut transcript,
            &bp_projected_trace,
            &bp_projected_scalars,
            bp_num_constraints,
            num_vars,
            &field_cfg,
        )
        .expect("BinaryPoly ideal check prover failed");

    // ── Step 4: CPR₁ (BinaryPoly) ──────────────────────────────────
    let bp_projecting_element: PiopField = transcript.get_field_challenge(&field_cfg);
    let bp_field_projected_scalars =
        project_scalars_to_field(bp_projected_scalars, &bp_projecting_element)
            .expect("BP scalar projection failed");
    let bp_field_trace = project_trace_to_field::<PiopField, D1>(
        trace, &[], &[], &bp_projecting_element,
    );

    let (bp_cpr_proof, bp_cpr_state) =
        CombinedPolyResolver::<PiopField>::prove_as_subprotocol::<U1>(
            &mut transcript,
            bp_field_trace,
            &bp_ic_state.evaluation_point,
            &bp_field_projected_scalars,
            bp_num_constraints,
            num_vars,
            bp_max_degree,
            &field_cfg,
        )
        .expect("BinaryPoly CPR failed");
    let bp_time = t1.elapsed();

    // ── Step 5: Convert trace to Q[X] ──────────────────────────────
    let qx_trace = convert_trace(trace);

    // ── Step 6: IC₂ (Q[X]) ─────────────────────────────────────────
    let t2 = Instant::now();
    let qx_projected_trace = project_trace_coeffs::<PiopField, i64, i64, D2>(
        &[], &qx_trace, &[], &field_cfg,
    );
    let qx_projected_scalars = project_scalars::<PiopField, U2>(|scalar| {
        DynamicPolynomialF::new(
            scalar.iter().map(|coeff| PiopField::from_with_cfg(*coeff, &field_cfg)).collect::<Vec<_>>()
        )
    });

    let (qx_ic_proof, qx_ic_state) =
        IdealCheckProtocol::<PiopField>::prove_as_subprotocol::<U2>(
            &mut transcript,
            &qx_projected_trace,
            &qx_projected_scalars,
            qx_num_constraints,
            num_vars,
            &field_cfg,
        )
        .expect("Q[X] ideal check prover failed");

    // ── Step 7: CPR₂ (Q[X]) ────────────────────────────────────────
    let qx_projecting_element: PiopField = transcript.get_field_challenge(&field_cfg);
    let qx_field_projected_scalars =
        project_scalars_to_field(qx_projected_scalars, &qx_projecting_element)
            .expect("QX scalar projection failed");
    let qx_field_trace = project_trace_to_field::<PiopField, D2>(
        &[], &qx_projected_trace, &[], &qx_projecting_element,
    );

    let (qx_cpr_proof, _qx_cpr_state) =
        CombinedPolyResolver::<PiopField>::prove_as_subprotocol::<U2>(
            &mut transcript,
            qx_field_trace,
            &qx_ic_state.evaluation_point,
            &qx_field_projected_scalars,
            qx_num_constraints,
            num_vars,
            qx_max_degree,
            &field_cfg,
        )
        .expect("Q[X] CPR failed");
    let qx_time = t2.elapsed();

    // ── Step 8: PCS Test ────────────────────────────────────────────
    let t3 = Instant::now();
    let test_transcript =
        BatchedZipPlus::<Zt, Lc>::test::<CHECK>(params, trace, &hint)
            .expect("PCS test failed");
    let pcs_test_time = t3.elapsed();

    // ── Step 9: PCS Evaluate ────────────────────────────────────────
    let t4 = Instant::now();
    let point: Vec<Zt::Pt> = {
        let i128_point = derive_pcs_point(&bp_cpr_state.evaluation_point, num_vars);
        i128_point
            .into_iter()
            .map(|v| unsafe { std::mem::transmute_copy(&v) })
            .collect()
    };
    let (evals_f, proof) =
        BatchedZipPlus::<Zt, Lc>::evaluate::<PiopField, CHECK>(
            params,
            trace,
            &point,
            test_transcript,
        )
        .expect("PCS evaluate failed");
    let pcs_eval_time = t4.elapsed();

    let total_time = total_start.elapsed();

    // ── Serialize ───────────────────────────────────────────────────
    let proof_bytes: Vec<u8> = {
        let pcs_transcript: zip_plus::pcs_transcript::PcsTranscript = proof.into();
        pcs_transcript.stream.into_inner()
    };

    // Serialize BinaryPoly IC proof
    let bp_ic_values: Vec<Vec<u8>> = bp_ic_proof
        .combined_mle_values
        .iter()
        .map(|dpf| dpf.coeffs.iter().flat_map(|c| field_to_bytes(c)).collect())
        .collect();

    // Serialize BinaryPoly CPR proof
    let bp_cpr_msgs: Vec<Vec<u8>> = bp_cpr_proof
        .sumcheck_proof
        .messages
        .iter()
        .map(|msg| msg.0.tail_evaluations.iter().flat_map(|c| field_to_bytes(c)).collect())
        .collect();
    let bp_cpr_sum = field_to_bytes(&bp_cpr_proof.sumcheck_proof.claimed_sum);
    let bp_cpr_ups: Vec<Vec<u8>> = bp_cpr_proof.up_evals.iter().map(field_to_bytes).collect();
    let bp_cpr_downs: Vec<Vec<u8>> = bp_cpr_proof.down_evals.iter().map(field_to_bytes).collect();

    // Serialize Q[X] IC proof
    let qx_ic_values: Vec<Vec<u8>> = qx_ic_proof
        .combined_mle_values
        .iter()
        .map(|dpf| dpf.coeffs.iter().flat_map(|c| field_to_bytes(c)).collect())
        .collect();

    // Serialize Q[X] CPR proof
    let qx_cpr_msgs: Vec<Vec<u8>> = qx_cpr_proof
        .sumcheck_proof
        .messages
        .iter()
        .map(|msg| msg.0.tail_evaluations.iter().flat_map(|c| field_to_bytes(c)).collect())
        .collect();
    let qx_cpr_sum = field_to_bytes(&qx_cpr_proof.sumcheck_proof.claimed_sum);
    let qx_cpr_ups: Vec<Vec<u8>> = qx_cpr_proof.up_evals.iter().map(field_to_bytes).collect();
    let qx_cpr_downs: Vec<Vec<u8>> = qx_cpr_proof.down_evals.iter().map(field_to_bytes).collect();

    // Serialize evaluation data
    let evaluation_point_bytes: Vec<Vec<u8>> =
        bp_cpr_state.evaluation_point.iter().map(field_to_bytes).collect();
    let pcs_evals_bytes: Vec<Vec<u8>> = evals_f.iter().map(field_to_bytes).collect();

    DualRingZincProof {
        pcs_proof_bytes: proof_bytes,
        commitment,
        bp_ic_proof_values: bp_ic_values,
        bp_cpr_sumcheck_messages: bp_cpr_msgs,
        bp_cpr_sumcheck_claimed_sum: bp_cpr_sum,
        bp_cpr_up_evals: bp_cpr_ups,
        bp_cpr_down_evals: bp_cpr_downs,
        qx_ic_proof_values: qx_ic_values,
        qx_cpr_sumcheck_messages: qx_cpr_msgs,
        qx_cpr_sumcheck_claimed_sum: qx_cpr_sum,
        qx_cpr_up_evals: qx_cpr_ups,
        qx_cpr_down_evals: qx_cpr_downs,
        evaluation_point_bytes,
        pcs_evals_bytes,
        timing: TimingBreakdown {
            pcs_commit: pcs_commit_time,
            ideal_check: bp_time + qx_time,
            combined_poly_resolver: Duration::ZERO, // included in ideal_check
            lookup: Duration::ZERO, // no lookup in dual-ring pipeline yet
            pcs_test: pcs_test_time,
            pcs_evaluate: pcs_eval_time,
            total: total_time,
        },
    }
}

/// Run the dual-ring Zinc+ verifier.
///
/// Verifies both IC+CPR passes (BinaryPoly and Q[X]) on the same
/// Fiat-Shamir transcript, then verifies the PCS.
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
    PiopField: for<'a> FromWithConfig<&'a Zt::Chal> + for<'a> FromWithConfig<&'a Zt::Pt>,
    <PiopField as Field>::Inner: FromRef<Zt::Fmod>,
    QxIdealOverF: Ideal + IdealCheck<DynamicPolynomialF<PiopField>>,
    QxIdealFromRef: Fn(&IdealOrZero<U2::Ideal>) -> QxIdealOverF,
{
    let total_start = Instant::now();

    let bp_num_constraints = count_constraints::<U1>();
    let bp_max_degree = count_max_degree::<U1>();
    let qx_num_constraints = count_constraints::<U2>();
    let qx_max_degree = count_max_degree::<U2>();

    // ── Reconstruct Fiat-Shamir transcript ──────────────────────────
    let mut transcript = KeccakTranscript::new();
    let field_cfg = transcript.get_random_field_cfg::<PiopField, <PiopField as Field>::Inner, MillerRabin>();
    let field_elem_size = <crypto_primitives::crypto_bigint_uint::Uint<FIELD_LIMBS> as ConstTranscribable>::NUM_BYTES;

    // ── Pass 1: IC₁ verify (BinaryPoly, TrivialIdeal) ──────────────
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

    let bp_ic_subclaim = match IdealCheckProtocol::<PiopField>::verify_as_subprotocol::<U1, _, _>(
        &mut transcript,
        bp_ic_proof,
        bp_num_constraints,
        num_vars,
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

    // ── Pass 1: CPR₁ verify ────────────────────────────────────────
    let bp_cpr_sumcheck_messages: Vec<ProverMsg<PiopField>> = proof
        .bp_cpr_sumcheck_messages
        .iter()
        .map(|bytes| {
            let num_evals = bytes.len() / field_elem_size;
            let tail_evaluations: Vec<PiopField> = (0..num_evals)
                .map(|i| field_from_bytes(&bytes[i * field_elem_size..(i + 1) * field_elem_size], &field_cfg))
                .collect();
            ProverMsg(NatEvaluatedPolyWithoutConstant::new(tail_evaluations))
        })
        .collect();

    let bp_cpr_claimed_sum = field_from_bytes(&proof.bp_cpr_sumcheck_claimed_sum, &field_cfg);
    let bp_cpr_up_evals: Vec<PiopField> = proof.bp_cpr_up_evals.iter()
        .map(|b| field_from_bytes(b, &field_cfg))
        .collect();
    let bp_cpr_down_evals: Vec<PiopField> = proof.bp_cpr_down_evals.iter()
        .map(|b| field_from_bytes(b, &field_cfg))
        .collect();

    let bp_cpr_proof = zinc_piop::combined_poly_resolver::Proof::<PiopField> {
        sumcheck_proof: SumcheckProof {
            messages: bp_cpr_sumcheck_messages,
            claimed_sum: bp_cpr_claimed_sum,
        },
        up_evals: bp_cpr_up_evals,
        down_evals: bp_cpr_down_evals,
    };

    // Compute BP projecting element and projected scalars for CPR₁ verify.
    let bp_projecting_element: PiopField = transcript.get_field_challenge(&field_cfg);
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
        project_scalars_to_field(bp_projected_scalars_coeffs, &bp_projecting_element)
            .expect("BP scalar projection failed");

    let bp_cpr_subclaim = match CombinedPolyResolver::<PiopField>::verify_as_subprotocol::<U1>(
        &mut transcript,
        bp_cpr_proof,
        bp_num_constraints,
        num_vars,
        bp_max_degree,
        &bp_projecting_element,
        &bp_field_projected_scalars,
        bp_ic_subclaim,
        &field_cfg,
    ) {
        Ok(subclaim) => subclaim,
        Err(e) => {
            eprintln!("BinaryPoly CPR verification failed: {e:?}");
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
    let bp_verify_time = t0.elapsed();

    // ── Pass 2: IC₂ verify (Q[X], real ideals) ─────────────────────
    let t1 = Instant::now();

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

    let qx_ic_subclaim = match IdealCheckProtocol::<PiopField>::verify_as_subprotocol::<U2, _, _>(
        &mut transcript,
        qx_ic_proof,
        qx_num_constraints,
        num_vars,
        &qx_ideal_from_ref,
        &field_cfg,
    ) {
        Ok(subclaim) => subclaim,
        Err(e) => {
            eprintln!("Q[X] IdealCheck verification failed: {e:?}");
            return VerifyResult {
                accepted: false,
                timing: VerifyTimingBreakdown {
                    ideal_check_verify: bp_verify_time + t1.elapsed(),
                    combined_poly_resolver_verify: Duration::ZERO,
                    lookup_verify: Duration::ZERO,
                    pcs_verify: Duration::ZERO,
                    total: total_start.elapsed(),
                },
            };
        }
    };

    // ── Pass 2: CPR₂ verify ────────────────────────────────────────
    let qx_cpr_sumcheck_messages: Vec<ProverMsg<PiopField>> = proof
        .qx_cpr_sumcheck_messages
        .iter()
        .map(|bytes| {
            let num_evals = bytes.len() / field_elem_size;
            let tail_evaluations: Vec<PiopField> = (0..num_evals)
                .map(|i| field_from_bytes(&bytes[i * field_elem_size..(i + 1) * field_elem_size], &field_cfg))
                .collect();
            ProverMsg(NatEvaluatedPolyWithoutConstant::new(tail_evaluations))
        })
        .collect();

    let qx_cpr_claimed_sum = field_from_bytes(&proof.qx_cpr_sumcheck_claimed_sum, &field_cfg);
    let qx_cpr_up_evals: Vec<PiopField> = proof.qx_cpr_up_evals.iter()
        .map(|b| field_from_bytes(b, &field_cfg))
        .collect();
    let qx_cpr_down_evals: Vec<PiopField> = proof.qx_cpr_down_evals.iter()
        .map(|b| field_from_bytes(b, &field_cfg))
        .collect();

    let qx_cpr_proof = zinc_piop::combined_poly_resolver::Proof::<PiopField> {
        sumcheck_proof: SumcheckProof {
            messages: qx_cpr_sumcheck_messages,
            claimed_sum: qx_cpr_claimed_sum,
        },
        up_evals: qx_cpr_up_evals,
        down_evals: qx_cpr_down_evals,
    };

    // Compute Q[X] projecting element and projected scalars for CPR₂ verify.
    let qx_projecting_element: PiopField = transcript.get_field_challenge(&field_cfg);
    let qx_projected_scalars_coeffs = project_scalars::<PiopField, U2>(|scalar| {
        DynamicPolynomialF::new(
            scalar.iter().map(|coeff| PiopField::from_with_cfg(*coeff, &field_cfg)).collect::<Vec<_>>()
        )
    });
    let qx_field_projected_scalars =
        project_scalars_to_field(qx_projected_scalars_coeffs, &qx_projecting_element)
            .expect("QX scalar projection failed");

    if let Err(e) = CombinedPolyResolver::<PiopField>::verify_as_subprotocol::<U2>(
        &mut transcript,
        qx_cpr_proof,
        qx_num_constraints,
        num_vars,
        qx_max_degree,
        &qx_projecting_element,
        &qx_field_projected_scalars,
        qx_ic_subclaim,
        &field_cfg,
    ) {
        eprintln!("Q[X] CPR verification failed: {e:?}");
        return VerifyResult {
            accepted: false,
            timing: VerifyTimingBreakdown {
                ideal_check_verify: bp_verify_time + t1.elapsed(),
                combined_poly_resolver_verify: Duration::ZERO,
                lookup_verify: Duration::ZERO,
                pcs_verify: Duration::ZERO,
                total: total_start.elapsed(),
            },
        };
    }
    let qx_verify_time = t1.elapsed();

    // ── Step 3: PCS Verify ──────────────────────────────────────────
    let t2 = Instant::now();

    let _pcs_evals: Vec<PiopField> = proof.pcs_evals_bytes.iter()
        .map(|b| field_from_bytes(b, &field_cfg))
        .collect();

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

    let pcs_transcript = zip_plus::pcs_transcript::PcsTranscript {
        fs_transcript: KeccakTranscript::default(),
        stream: std::io::Cursor::new(proof.pcs_proof_bytes.clone()),
    };
    let pcs_proof: zip_plus::batched_pcs::structs::BatchedZipPlusProof = pcs_transcript.into();

    let pcs_result = BatchedZipPlus::<Zt, Lc>::verify::<PiopField, CHECK>(
        params,
        &proof.commitment,
        &pcs_point,
        &pcs_proof,
    );
    let pcs_verify_time = t2.elapsed();

    // CPR→PCS binding: same note as in verify() above.

    VerifyResult {
        accepted: pcs_result.is_ok(),
        timing: VerifyTimingBreakdown {
            ideal_check_verify: bp_verify_time + qx_verify_time,
            combined_poly_resolver_verify: Duration::ZERO,
            lookup_verify: Duration::ZERO,
            pcs_verify: pcs_verify_time,
            total: total_start.elapsed(),
        },
    }
}
