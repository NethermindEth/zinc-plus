//! Zinc+ PIOP for UCS — end-to-end protocol (without PCS).
//!
//! Implements the four steps of the Zinc+ compiler from
//! Section 2.2 "Combining the three steps" of the paper:
//!
//! ```text
//! Q[X]  ──φ_q──>  F_q[X]  ──MLE eval──>  F_q[X]  ──ψ_a──>  F_q
//!       Step 1             Step 2                  Step 3
//! ```
//!
//! Step 4 runs a finite-field PIOP (sumcheck) over F_q.
//!
//! The verifier's output is a [`Subclaim`] containing evaluation
//! claims about the trace column MLEs. In the full protocol,
//! these would be resolved by the Zip+ PCS.

use crypto_primitives::{
    ConstIntRing, ConstIntSemiring, FromPrimitiveWithConfig, FromWithConfig, IntoWithConfig,
    PrimeField,
};
use num_traits::Zero;
use std::io::Cursor;
use thiserror::Error;
use zinc_piop::{
    combined_poly_resolver::{self, CombinedPolyResolver, CombinedPolyResolverError},
    ideal_check::{self, IdealCheckProtocol},
    projections::{
        project_scalars, project_scalars_to_field, project_trace_coeffs, project_trace_to_field,
    },
};
use zinc_poly::{
    ConstCoeffBitWidth,
    mle::{DenseMultilinearExtension, MultilinearExtensionWithConfig},
    univariate::{
        binary::BinaryPoly, dense::DensePolynomial, dynamic::over_field::DynamicPolynomialF,
    },
};
use zinc_primality::PrimalityTest;
use zinc_transcript::{
    KeccakTranscript,
    traits::{ConstTranscribable, Transcribable, Transcript},
};
use zinc_uair::{
    Uair,
    constraint_counter::count_constraints,
    degree_counter::count_max_degree,
    ideal::{Ideal, IdealCheck},
    ideal_collector::IdealOrZero,
};
use zinc_utils::{
    from_ref::FromRef, inner_transparent_field::InnerTransparentField, mul_by_scalar::MulByScalar,
    named::Named, projectable_to_field::ProjectableToField,
};
use zip_plus::{
    ZipError,
    code::LinearCode,
    pcs::structs::{ZipPlus, ZipPlusCommitment, ZipPlusHint, ZipPlusParams, ZipTypes},
    pcs_transcript::{PcsProverTranscript, PcsVerifierTranscript},
};

//
// Data structures
//

/// Proof produced by the Zinc+ PIOP for UCS (without PCS).
///
/// Contains the two subproofs from Steps 2 and 4:
/// - `ideal_check`: MLE evaluations in F_q\[X\] (Step 2).
/// - `resolver`: sumcheck proof + trace evaluation claims (Step 4).
#[derive(Clone, Debug)]
pub struct Proof<F: PrimeField> {
    pub num_witness_cols: (usize, usize, usize),
    pub zip_commitments: Vec<ZipPlusCommitment>,
    /// Serialized PCS proof data (Zip+ proving transcripts).
    pub zip_proof: Vec<u8>,
    pub ideal_check: ideal_check::Proof<F>,
    pub resolver: combined_poly_resolver::Proof<F>,
}

/// Subclaim returned by the verifier, to be resolved by PCS.
///
/// Contains evaluation claims: "the trace column MLEs, evaluated at
/// `evaluation_point`, should yield `up_evals` (current row) and
/// `down_evals` (next row)."
#[derive(Clone, Debug)]
pub struct Subclaim<F: PrimeField> {
    pub evaluation_point: Vec<F>,
    pub up_evals: Vec<F>,
    pub down_evals: Vec<F>,
}

/// Prover auxiliary data for subclaim resolution without PCS.
pub struct ProverAux<F: PrimeField> {
    /// The random field configuration (derived from transcript in Step 1).
    pub field_cfg: F::Config,
    /// The trace projected to F_q (after Steps 1 and 3).
    pub projected_trace_f: Vec<DenseMultilinearExtension<F::Inner>>,
}

/// Trait bundling the various type parameters for the witness and Zip+ PCS.
pub trait WitnessZipTypes<const DEGREE_PLUS_ONE: usize> {
    type Int: Named + ConstCoeffBitWidth + Default + Clone + Send + Sync;
    type Chal: ConstIntRing + ConstTranscribable + Named;
    type Pt: ConstIntRing;
    type Fmod: ConstIntSemiring + ConstTranscribable + Named;
    type PrimeTest: PrimalityTest<Self::Fmod>;

    type BinaryZt: ZipTypes<
            Eval = BinaryPoly<DEGREE_PLUS_ONE>,
            Chal = Self::Chal,
            Pt = Self::Pt,
            Fmod = Self::Fmod,
            PrimeTest = Self::PrimeTest,
        >;
    type ArbitraryZt: ZipTypes<
            Eval = DensePolynomial<Self::Int, DEGREE_PLUS_ONE>,
            Chal = Self::Chal,
            Pt = Self::Pt,
            Fmod = Self::Fmod,
            PrimeTest = Self::PrimeTest,
        >;
    type IntZt: ZipTypes<
            Eval = Self::Int,
            Chal = Self::Chal,
            Pt = Self::Pt,
            Fmod = Self::Fmod,
            PrimeTest = Self::PrimeTest,
        >;

    type BinaryLc: LinearCode<Self::BinaryZt>;
    type ArbitraryLc: LinearCode<Self::ArbitraryZt>;
    type IntLc: LinearCode<Self::IntZt>;
}

//
// Prover
//

/// Zinc+ PIOP Prover (Algorithm 1 from the paper, Steps 1–5).
///
/// # Protocol flow (paper Section 2.2 "Combining the three steps"):
///
/// 0. **Commit**: commit each witness column via Zip+ PCS, absorb roots.
/// 1. **Prime projection** (φ_q: Q\[X\] → F_q\[X\]): sample random prime q from
///    transcript, project trace and scalars.
/// 2. **Ideal check**: sample r ∈ F_q^μ, prover sends MLE evaluations, verifier
///    checks ideal membership.
/// 3. **Evaluation projection** (ψ_a: F_q\[X\] → F_q): sample a ∈ F_q, evaluate
///    polynomials at X = a.
/// 4. **Finite-field PIOP**: sumcheck over F_q to prove the projected claim.
/// 5. **PCS open**: Zip+ test + evaluate for each committed column, proving
///    witness MLE evaluations at the sumcheck challenge point.
///
/// Returns the proof and auxiliary data (for subclaim resolution without PCS).
#[allow(clippy::too_many_arguments)]
pub fn prove<Zt, U, F, ProjectScalar, const D: usize, const CHECK_FOR_OVERFLOW: bool>(
    (pp_bin, pp_arb, pp_int): &(
        ZipPlusParams<Zt::BinaryZt, Zt::BinaryLc>,
        ZipPlusParams<Zt::ArbitraryZt, Zt::ArbitraryLc>,
        ZipPlusParams<Zt::IntZt, Zt::IntLc>,
    ),
    trace_bin_poly: &[DenseMultilinearExtension<<Zt::BinaryZt as ZipTypes>::Eval>],
    trace_arb_poly: &[DenseMultilinearExtension<<Zt::ArbitraryZt as ZipTypes>::Eval>],
    trace_int: &[DenseMultilinearExtension<<Zt::IntZt as ZipTypes>::Eval>],
    num_vars: usize,
    project_scalar: ProjectScalar,
) -> Result<(Proof<F>, ProverAux<F>), ProtocolError<F>>
where
    Zt: WitnessZipTypes<D>,
    Zt::Int: ProjectableToField<F>,
    <Zt::ArbitraryZt as ZipTypes>::Eval: ProjectableToField<F>,
    F: InnerTransparentField
        + FromPrimitiveWithConfig
        + FromWithConfig<Zt::Int>
        + for<'a> FromWithConfig<&'a Zt::Chal>
        + for<'a> FromWithConfig<&'a Zt::Pt>
        + for<'a> MulByScalar<&'a F>
        + FromRef<F>
        + Send
        + Sync
        + 'static,
    F::Inner:
        ConstIntSemiring + ConstTranscribable + FromRef<Zt::Fmod> + Send + Sync + Zero + Default,
    U: Uair + 'static,
    ProjectScalar: Fn(&U::Scalar, &F::Config) -> DynamicPolynomialF<F>,
{
    // === Step 0: Commit to witness traces ===

    // Commit each witness column via Zip+ PCS.
    fn pcs_commit_witness<Zt, Lc>(
        pp: &ZipPlusParams<Zt, Lc>,
        witness: &[DenseMultilinearExtension<Zt::Eval>],
    ) -> Result<(Vec<ZipPlusHint<Zt::Cw>>, Vec<ZipPlusCommitment>), ZipError>
    where
        Zt: ZipTypes,
        Lc: LinearCode<Zt>,
    {
        let mut hints = Vec::with_capacity(witness.len());
        let mut commitments = Vec::with_capacity(witness.len());
        for col in witness {
            let (hint, comm) = ZipPlus::<Zt, Lc>::commit(pp, col)?;
            hints.push(hint);
            commitments.push(comm);
        }
        Ok((hints, commitments))
    }

    let (hints_bin, commitments_bin) = pcs_commit_witness(pp_bin, trace_bin_poly)?;
    let (hints_arb, commitments_arb) = pcs_commit_witness(pp_arb, trace_arb_poly)?;
    let (hints_int, commitments_int) = pcs_commit_witness(pp_int, trace_int)?;

    // Create the main transcript
    let mut pcs_transcript = PcsProverTranscript::new_from_commitments(
        commitments_bin
            .iter()
            .chain(commitments_arb.iter())
            .chain(commitments_int.iter()),
    )?;
    // TODO: Absorb public inputs as well once they are part of the protocol,
    //       or this will open up a soundness vulnerability!

    // === Step 1: Prime projection (φ_q: Q[X] → F_q[X]) ===

    // Sample a random prime modulus from the Fiat-Shamir transcript.
    let field_cfg = pcs_transcript
        .fs_transcript
        .get_random_field_cfg::<F, Zt::Fmod, Zt::PrimeTest>();

    // Project the witness trace from Q[X] to F_q[X].
    let projected_trace = project_trace_coeffs::<F, Zt::Int, Zt::Int, D>(
        trace_bin_poly,
        trace_arb_poly,
        trace_int,
        &field_cfg,
    );

    // Project UAIR scalars from Q[X] to F_q[X].
    let projected_scalars_fx = project_scalars::<F, U>(|s| project_scalar(s, &field_cfg));

    let num_constraints = count_constraints::<U>();

    // ── Step 2: Randomized ideal check ──────────────────────────
    let (ic_proof, ic_prover_state) = IdealCheckProtocol::prove_as_subprotocol::<U>(
        &mut pcs_transcript.fs_transcript,
        &projected_trace,
        &projected_scalars_fx,
        num_constraints,
        num_vars,
        &field_cfg,
    )
    .map_err(|e| ProtocolError::IdealCheck(format!("{e:?}")))?;

    // === Step 3: Evaluation projection (ψ_a: F_q[X] → F_q) ===
    // Sample the projecting element as Zt::Chal (matching the Zip+ PCS convention),
    // then convert to F for PIOP use.
    let projecting_element: Zt::Chal = pcs_transcript.fs_transcript.get_challenge();
    let projecting_element_f: F = F::from_with_cfg(&projecting_element, &field_cfg);

    // Project trace from F_q[X] to F_q by evaluating each polynomial at X = a.
    let projected_trace_f =
        project_trace_to_field::<F, D>(&[], &projected_trace, &[], &projecting_element_f);

    // Project scalars from F_q[X] to F_q.
    let projected_scalars_f = project_scalars_to_field(projected_scalars_fx, &projecting_element_f)
        .map_err(|(_s, _f, e)| ProtocolError::ScalarProjection(format!("{e:?}")))?;

    let max_degree = count_max_degree::<U>();

    // === Step 4: Finite-field PIOP (sumcheck over F_q) ===
    let (cpr_proof, cpr_prover_state) = CombinedPolyResolver::prove_as_subprotocol::<U>(
        &mut pcs_transcript.fs_transcript,
        projected_trace_f.clone(),
        &ic_prover_state.evaluation_point,
        &projected_scalars_f,
        num_constraints,
        num_vars,
        max_degree,
        &field_cfg,
    )?;

    // === Step 5: PCS open (prove witness MLE evaluations) ===
    // After the sumcheck, the prover must prove that the committed
    // witness MLEs evaluate to the claimed values (up_evals / down_evals)
    // at the sumcheck challenge point r'.
    // This corresponds to Step 12 "Oracle evaluations at sumcheck point"
    // in the AirZinc protocol (rolled_out_argument.tex).
    //
    // TODO: Once we add public inputs, the verifier will compute public
    //       input MLE evaluations at the sumcheck point directly from
    //       public data. The PCS only covers witness columns.
    let eval_point = &cpr_prover_state.evaluation_point;

    fn pcs_prove_witness_evaluations<Zt, Lc, F, const CHECK_FOR_OVERFLOW: bool>(
        pcs_transcript: &mut PcsProverTranscript,
        pp: &ZipPlusParams<Zt, Lc>,
        witness: &[DenseMultilinearExtension<Zt::Eval>],
        hints: &[ZipPlusHint<Zt::Cw>],
        commitments: &[ZipPlusCommitment],
        eval_point: &[F],
        field_cfg: &F::Config,
        projecting_element: &Zt::Chal,
    ) -> Result<(), ProtocolError<F>>
    where
        Zt: ZipTypes,
        Zt::Eval: ProjectableToField<F>,
        Lc: LinearCode<Zt>,
        F: PrimeField
            + for<'a> FromWithConfig<&'a Zt::Chal>
            + for<'a> MulByScalar<&'a F>
            + FromRef<F>,
        F::Inner: FromRef<Zt::Fmod> + Transcribable,
    {
        for ((hint, col), _comm) in hints.iter().zip(witness.iter()).zip(commitments.iter()) {
            // Proximity test
            ZipPlus::<Zt, Lc>::test::<CHECK_FOR_OVERFLOW>(pcs_transcript, pp, col, hint)?;

            // Evaluation proof
            let _eval_f: F = ZipPlus::<Zt, Lc>::evaluate_f::<F, CHECK_FOR_OVERFLOW>(
                pcs_transcript,
                pp,
                col,
                eval_point,
                &field_cfg,
                &projecting_element,
            )?;
        }
        Ok(())
    }

    pcs_prove_witness_evaluations::<Zt::BinaryZt, Zt::BinaryLc, F, CHECK_FOR_OVERFLOW>(
        &mut pcs_transcript,
        pp_bin,
        trace_bin_poly,
        &hints_bin,
        &commitments_bin,
        eval_point,
        &field_cfg,
        &projecting_element,
    )?;
    pcs_prove_witness_evaluations::<Zt::ArbitraryZt, Zt::ArbitraryLc, F, CHECK_FOR_OVERFLOW>(
        &mut pcs_transcript,
        pp_arb,
        trace_arb_poly,
        &hints_arb,
        &commitments_arb,
        eval_point,
        &field_cfg,
        &projecting_element,
    )?;
    pcs_prove_witness_evaluations::<Zt::IntZt, Zt::IntLc, F, CHECK_FOR_OVERFLOW>(
        &mut pcs_transcript,
        pp_int,
        trace_int,
        &hints_int,
        &commitments_int,
        eval_point,
        &field_cfg,
        &projecting_element,
    )?;

    let zip_proof = pcs_transcript.stream.into_inner();
    let commitments = commitments_bin
        .into_iter()
        .chain(commitments_arb.into_iter())
        .chain(commitments_int.into_iter())
        .collect();

    Ok((
        Proof {
            num_witness_cols: (trace_bin_poly.len(), trace_arb_poly.len(), trace_int.len()),
            zip_commitments: commitments,
            ideal_check: ic_proof,
            resolver: cpr_proof,
            zip_proof,
        },
        ProverAux {
            field_cfg,
            projected_trace_f,
        },
    ))
}

// ─── Verifier ───────────────────────────────────────────────────

/// Zinc+ PIOP Verifier (Algorithm 1, verification side, Steps 0–5).
///
/// Verifies all steps and returns a [`Subclaim`] containing
/// evaluation claims (already verified by the PCS).
#[allow(clippy::too_many_arguments)]
pub fn verify<
    Zt,
    U,
    F,
    IdealOverF,
    ProjectScalar,
    ProjectIdeal,
    const D: usize,
    const CHECK_FOR_OVERFLOW: bool,
>(
    (vp_bin, vp_arb, vp_int): &(
        ZipPlusParams<Zt::BinaryZt, Zt::BinaryLc>,
        ZipPlusParams<Zt::ArbitraryZt, Zt::ArbitraryLc>,
        ZipPlusParams<Zt::IntZt, Zt::IntLc>,
    ),
    proof: Proof<F>,
    num_vars: usize,
    project_scalar: ProjectScalar,
    project_ideal: ProjectIdeal,
) -> Result<Subclaim<F>, ProtocolError<F>>
where
    Zt: WitnessZipTypes<D>,
    F: InnerTransparentField
        + FromPrimitiveWithConfig
        + for<'a> FromWithConfig<&'a Zt::Chal>
        + for<'a> MulByScalar<&'a F>
        + FromRef<F>
        + Send
        + Sync
        + 'static,
    <Zt::BinaryZt as ZipTypes>::Cw: ProjectableToField<F>,
    <Zt::ArbitraryZt as ZipTypes>::Cw: ProjectableToField<F>,
    <Zt::IntZt as ZipTypes>::Cw: ProjectableToField<F>,
    F::Inner:
        ConstIntSemiring + ConstTranscribable + FromRef<Zt::Fmod> + Send + Sync + Zero + Default,
    U: Uair + 'static,
    IdealOverF: Ideal + IdealCheck<DynamicPolynomialF<F>>,
    ProjectScalar: Fn(&U::Scalar, &F::Config) -> DynamicPolynomialF<F>,
    ProjectIdeal: Fn(&IdealOrZero<U::Ideal>, &F::Config) -> IdealOverF,
{
    // === Step 0: Reconstruct transcript from commitments ===
    // The verifier creates a PcsVerifierTranscript from the PCS proof bytes.
    let mut pcs_transcript = PcsVerifierTranscript {
        fs_transcript: KeccakTranscript::default(),
        stream: Cursor::new(proof.zip_proof),
    };
    for comm in &proof.zip_commitments {
        pcs_transcript.fs_transcript.absorb_slice(&comm.root);
    }

    // === Step 1: Prime projection ===
    let field_cfg = pcs_transcript
        .fs_transcript
        .get_random_field_cfg::<F, Zt::Fmod, Zt::PrimeTest>();

    let num_constraints = count_constraints::<U>();

    // === Step 2: Verify ideal check ===
    let ic_subclaim = IdealCheckProtocol::verify_as_subprotocol::<U, IdealOverF, _>(
        &mut pcs_transcript.fs_transcript,
        proof.ideal_check,
        num_constraints,
        num_vars,
        |ideal| project_ideal(ideal, &field_cfg),
        &field_cfg,
    )
    .map_err(|e| ProtocolError::IdealCheck(format!("{e:?}")))?;

    // === Step 3: Evaluation projection ===
    // Sample projecting element as Zt::Chal (matching the prover).
    let projecting_element: Zt::Chal = pcs_transcript.fs_transcript.get_challenge();
    let projecting_element_f: F = F::from_with_cfg(&projecting_element, &field_cfg);

    // Verifier independently computes projected scalars (public data).
    let projected_scalars_fx = project_scalars::<F, U>(|s| project_scalar(s, &field_cfg));
    let projected_scalars_f = project_scalars_to_field(projected_scalars_fx, &projecting_element_f)
        .map_err(|(_s, _f, e)| ProtocolError::ScalarProjection(format!("{e:?}")))?;

    let max_degree = count_max_degree::<U>();

    // === Step 4: Verify finite-field PIOP ===
    let cpr_subclaim = CombinedPolyResolver::verify_as_subprotocol::<U>(
        &mut pcs_transcript.fs_transcript,
        proof.resolver,
        num_constraints,
        num_vars,
        max_degree,
        &projecting_element_f,
        &projected_scalars_f,
        ic_subclaim,
        &field_cfg,
    )?;

    // === Step 5: PCS verify (check witness MLE evaluation claims) ====
    // After the sumcheck, the verifier uses the Zip+ PCS to confirm
    // that the committed witness MLEs actually evaluate to the claimed
    // up_evals at the sumcheck challenge point.
    //
    // TODO: Once we add public inputs, compute public input MLE evaluations
    //       at cpr_subclaim.evaluation_point directly from public data here,
    //       then include them in the constraint recomputation check.

    for (i, comm) in proof.zip_commitments.iter().enumerate() {
        macro_rules! zip_verify {
            ($zt:ident, $lc:ident, $vp:ident) => {
                ZipPlus::<Zt::$zt, Zt::$lc>::verify::<F, CHECK_FOR_OVERFLOW>(
                    &mut pcs_transcript,
                    $vp,
                    comm,
                    &field_cfg,
                    &projecting_element,
                    &cpr_subclaim.evaluation_point,
                    &cpr_subclaim.up_evals[i],
                )
                .map_err(|e| ProtocolError::PcsVerification(i, e))?;
            };
        }

        if i < proof.num_witness_cols.0 {
            zip_verify!(BinaryZt, BinaryLc, vp_bin);
        } else if i < proof.num_witness_cols.0 + proof.num_witness_cols.1 {
            zip_verify!(ArbitraryZt, ArbitraryLc, vp_arb);
        } else {
            zip_verify!(IntZt, IntLc, vp_int);
        }
    }

    Ok(Subclaim {
        evaluation_point: cpr_subclaim.evaluation_point,
        up_evals: cpr_subclaim.up_evals,
        down_evals: cpr_subclaim.down_evals,
    })
}

// ─── Subclaim resolution (placeholder for PCS) ─────────────────

/// Resolve the verifier's subclaim by evaluating the actual trace MLEs
/// and checking they match the claimed evaluations.
///
/// In the full Zinc+ protocol, this step would be handled by the
/// Zip+ PCS (polynomial commitment scheme). Here we verify directly
/// against the prover's auxiliary data.
pub fn resolve_subclaim<F>(
    subclaim: &Subclaim<F>,
    projected_trace_f: &[DenseMultilinearExtension<F::Inner>],
    field_cfg: &F::Config,
) -> Result<(), ProtocolError<F>>
where
    F: InnerTransparentField,
    F::Inner: ConstTranscribable + Send + Sync + Zero + Default,
    DenseMultilinearExtension<F::Inner>: MultilinearExtensionWithConfig<F>,
{
    let num_cols = projected_trace_f.len();

    // Check "up" evaluations (current row columns).
    for (i, (mle, expected)) in projected_trace_f
        .iter()
        .zip(subclaim.up_evals.iter())
        .enumerate()
    {
        let actual = mle.evaluate_with_config(&subclaim.evaluation_point, field_cfg)?;

        if actual != *expected {
            return Err(ProtocolError::SubclaimMismatch {
                column: i,
                expected: expected.clone(),
                actual,
            });
        }
    }

    // Check "down" evaluations (shifted/next-row columns).
    // The shifted trace drops the first row and zero-pads, matching
    // the CombinedPolyResolver's convention.
    for (i, (mle, expected)) in projected_trace_f
        .iter()
        .zip(subclaim.down_evals.iter())
        .enumerate()
    {
        let shifted: DenseMultilinearExtension<F::Inner> = mle.iter().skip(1).cloned().collect();

        let actual = shifted.evaluate_with_config(&subclaim.evaluation_point, field_cfg)?;

        if actual != *expected {
            return Err(ProtocolError::SubclaimMismatch {
                column: num_cols + i,
                expected: expected.clone(),
                actual,
            });
        }
    }

    Ok(())
}

//
// Error type and conversion
//

// TODO: Convert other error types to proper wrappers
#[derive(Debug, Error)]
pub enum ProtocolError<F: PrimeField> {
    #[error("ideal check failed: {0}")]
    IdealCheck(String),
    #[error("combined poly resolver failed: {0}")]
    Resolver(#[from] CombinedPolyResolverError<F>),
    #[error("scalar projection failed: {0}")]
    ScalarProjection(String),
    #[error("subclaim resolution: MLE evaluation failed: {0}")]
    MleEvaluation(#[from] zinc_poly::EvaluationError),
    #[error("subclaim mismatch at column {column}: expected {expected:?}, got {actual:?}")]
    SubclaimMismatch {
        column: usize,
        expected: F,
        actual: F,
    },
    #[error("PCS error: {0}")]
    Pcs(ZipError),
    #[error("PCS verification failed at column {0}: {1}")]
    PcsVerification(usize, ZipError),
}

impl<F: PrimeField> From<ZipError> for ProtocolError<F> {
    fn from(e: ZipError) -> Self {
        ProtocolError::Pcs(e)
    }
}

//
// Tests
//

#[cfg(test)]
mod tests {
    use super::*;
    use crypto_bigint::U64;

    use crypto_primitives::{
        Field, crypto_bigint_int::Int, crypto_bigint_monty::MontyField, crypto_bigint_uint::Uint,
    };
    use rand::rng;
    use zinc_poly::univariate::{binary::BinaryPolyWideningMulByScalar, ideal::DegreeOneIdeal};
    use zinc_poly::univariate::binary::BinaryPolyInnerProduct;
    use zinc_poly::univariate::dense::DensePolyInnerProduct;
    use zinc_primality::MillerRabin;
    use zinc_test_uair::{
        BigLinearUair, BinaryDecompositionUair, GenerateMultiTypeWitness,
        GenerateSingleTypeWitness, TestAirNoMultiplication, TestUairSimpleMultiplication,
    };
    use zinc_utils::{
        CHECKED,
        mul_by_scalar::{WideningMulByScalar},
    };
    use zinc_utils::inner_product::{MBSInnerProduct, ScalarProduct};
    use zinc_utils::mul_by_scalar::PrimtiveWideningMulByScalar;
    use zip_plus::{
        code::{
            iprs::{IprsCode, PnttConfigF2_16_1},
        },
    };

    const INT_LIMBS: usize = U64::LIMBS;
    const FIELD_LIMBS: usize = 4;
    const DEGREE_PLUS_ONE: usize = 32;

    // Zip+ type parameters.

    const N: usize = INT_LIMBS;
    const K: usize = INT_LIMBS * 4;
    const M: usize = INT_LIMBS * 8;
    const IPRS_DEPTH: usize = 1;

    type F = MontyField<FIELD_LIMBS>;
    type Witness = DensePolynomial<Int<INT_LIMBS>, DEGREE_PLUS_ONE>;


    pub struct BinPolyZipTypes {}
    impl ZipTypes for BinPolyZipTypes {
        const NUM_COLUMN_OPENINGS: usize = 200;
        type Eval = BinaryPoly<DEGREE_PLUS_ONE>;
        type Cw = DensePolynomial<i64, DEGREE_PLUS_ONE>;
        type Fmod = Uint<K>;
        type PrimeTest = MillerRabin;
        type Chal = i128;
        type Pt = i128;
        type CombR = Int<M>;
        type Comb = DensePolynomial<Self::CombR, DEGREE_PLUS_ONE>;
        type EvalDotChal = BinaryPolyInnerProduct<Self::Chal, DEGREE_PLUS_ONE>;
        type CombDotChal = DensePolyInnerProduct<
            Self::CombR,
            Self::Chal,
            Self::CombR,
            MBSInnerProduct,
            DEGREE_PLUS_ONE,
        >;
        type ArrCombRDotChal = MBSInnerProduct;
    }

    pub struct ArbitraryPolyZipTypes {}
    impl ZipTypes for ArbitraryPolyZipTypes {
        const NUM_COLUMN_OPENINGS: usize = 200;
        type Eval = DensePolynomial<i64, DEGREE_PLUS_ONE>;
        type Cw = DensePolynomial<i64, DEGREE_PLUS_ONE>;
        type Fmod = Uint<K>;
        type PrimeTest = MillerRabin;
        type Chal = i128;
        type Pt = i128;
        type CombR = Int<M>;
        type Comb = DensePolynomial<Self::CombR, DEGREE_PLUS_ONE>;
        type EvalDotChal = DensePolyInnerProduct<i64, Self::Chal, Self::CombR, MBSInnerProduct, DEGREE_PLUS_ONE>;
        type CombDotChal = DensePolyInnerProduct<
            Self::CombR,
            Self::Chal,
            Self::CombR,
            MBSInnerProduct,
            DEGREE_PLUS_ONE,
        >;
        type ArrCombRDotChal = MBSInnerProduct;
    }

    pub struct IntZipTypes {}
    impl ZipTypes for IntZipTypes {
        const NUM_COLUMN_OPENINGS: usize = 200;
        type Eval = i64;
        type Cw = i128;
        type Fmod = Uint<K>;
        type PrimeTest = MillerRabin;
        type Chal = i128;
        type Pt = i128;
        type CombR = Int<M>;
        type Comb = Self::CombR;
        type EvalDotChal = ScalarProduct;
        type CombDotChal = ScalarProduct;
        type ArrCombRDotChal = MBSInnerProduct;
    }

    struct TestWitnessZipTypes;

    impl WitnessZipTypes<DEGREE_PLUS_ONE> for TestWitnessZipTypes {
        type Int = i64;
        type Chal = i128;
        type Pt = i128;
        type Fmod = Uint<K>;
        type PrimeTest = MillerRabin;

        type BinaryZt = BinPolyZipTypes;
        type ArbitraryZt = ArbitraryPolyZipTypes;
        type IntZt = IntZipTypes;

        type BinaryLc = IprsCode<
            Self::BinaryZt,
            PnttConfigF2_16_1<IPRS_DEPTH>,
            BinaryPolyWideningMulByScalar<i64>,
            CHECKED,
        >;
        type ArbitraryLc = ();
        type IntLc = IprsCode<
            Self::IntZt,
            PnttConfigF2_16_1<IPRS_DEPTH>,
            PrimtiveWideningMulByScalar,
            CHECKED,
        >;
    }

    /// Helper: project a DensePolynomial scalar to DynamicPolynomialF
    /// by projecting each coefficient via φ_q.
    fn project_scalar_fn(
        scalar: &Witness,
        field_cfg: &<F as PrimeField>::Config,
    ) -> DynamicPolynomialF<F> {
        scalar
            .iter()
            .map(|coeff| F::from_with_cfg(coeff, field_cfg))
            .collect()
    }

    /// Set up Zip+ PCS parameters for a given number of MLE variables.
    fn setup_pp<Wzt>(
        num_vars: usize,
    ) -> (
        ZipPlusParams<Wzt::BinaryZt, Wzt::BinaryLc>,
        ZipPlusParams<Wzt::ArbitraryZt, Wzt::ArbitraryLc>,
        ZipPlusParams<Wzt::IntZt, Wzt::IntLc>,
    )
    where
        Wzt: WitnessZipTypes<DEGREE_PLUS_ONE>,
    {
        let poly_size = 1 << num_vars;
        (
            ZipPlus::<Wzt::BinaryZt, Wzt::BinaryLc>::setup(
                poly_size,
                Wzt::BinaryLc::new(poly_size),
            ),
            ZipPlus::<Wzt::ArbitraryZt, Wzt::ArbitraryLc>::setup(
                poly_size,
                Wzt::ArbitraryLc::new(poly_size),
            ),
            ZipPlus::<Wzt::IntZt, Wzt::IntLc>::setup(poly_size, Wzt::IntLc::new(poly_size)),
        )
    }

    /// End-to-end test: TestAirNoMultiplication.
    ///
    /// UAIR constraint: a + b - c ∈ (X - 2)
    /// (one constraint, no polynomial multiplication, ideal = ⟨X - 2⟩).
    ///
    /// NOTE: The witness_columns type (DensePolynomial<Int<5>, 32>) does not
    /// match Zt::Eval (Int<5>) so this won't compile yet. The protocol
    /// structure is correct — only the type plumbing needs resolving.
    #[test]
    fn test_end_to_end_no_multiplication() {
        let mut rng = rng();
        let num_vars = 4;
        let pp = setup_pp(num_vars);

        // Generate a valid witness satisfying the UAIR constraints.
        let trace = TestAirNoMultiplication::<INT_LIMBS>::generate_witness(num_vars, &mut rng);

        // ── Prover ──
        let (proof, prover_aux) = prove::<
            TestWitnessZipTypes,
            TestAirNoMultiplication<INT_LIMBS>,
            F,
            _,
            DEGREE_PLUS_ONE,
            CHECKED,
        >(
            &pp,
            &[],
            &trace,
            &[],
            num_vars,
            project_scalar_fn,
        )
        .expect("Prover failed");

        // ── Verifier ──
        let subclaim = verify::<
            TestWitnessZipTypes,
            TestAirNoMultiplication<INT_LIMBS>,
            F,
            _,
            _,
            _,
            DEGREE_PLUS_ONE,
            CHECKED,
        >(
            &pp,
            proof,
            num_vars,
            project_scalar_fn,
            |ideal, field_cfg| ideal.map(|i| DegreeOneIdeal::from_with_cfg(i, field_cfg)),
        )
        .expect("Verifier failed");

        // ── Subclaim resolution (in lieu of PCS) ──
        resolve_subclaim(
            &subclaim,
            &prover_aux.projected_trace_f,
            &prover_aux.field_cfg,
        )
        .expect("Subclaim resolution failed");
    }

    /*
    /// End-to-end test: TestUairSimpleMultiplication.
    ///
    /// UAIR constraints (3 total, no ideals):
    ///   up[0] * up[1] = down[0]
    ///   up[1] * up[2] = down[1]
    ///   up[0] * up[2] = down[2]
    #[test]
    fn test_end_to_end_simple_multiplication() {
        let mut rng = rng();
        let num_vars = 2;
        let pp = setup_pp(num_vars);

        let trace =
            TestUairSimpleMultiplication::<Int<INT_LIMBS>>::generate_witness(num_vars, &mut rng);

        // ── Prover ──
        let (proof, prover_aux) = prove::<
            Zt,
            Lc,
            TestUairSimpleMultiplication<Int<INT_LIMBS>>,
            F,
            Int<INT_LIMBS>,
            Int<INT_LIMBS>,
            _,
            CHECKED,
            DEGREE_PLUS_ONE,
        >(
            &pp,
            &trace, // witness_columns — type mismatch with Zt::Eval
            &[],
            &trace,
            &[],
            num_vars,
            project_scalar_fn,
        )
        .expect("Prover failed");

        // ── Verifier ──
        let subclaim = verify::<
            Zt,
            Lc,
            TestUairSimpleMultiplication<Int<INT_LIMBS>>,
            F,
            _,
            _,
            _,
            CHECKED,
            DEGREE_PLUS_ONE,
        >(
            &pp,
            proof,
            num_vars,
            project_scalar_fn,
            |_ideal, _field_cfg| IdealOrZero::<DegreeOneIdeal<F>>::zero(),
        )
        .expect("Verifier failed");

        // ── Subclaim resolution (in lieu of PCS) ──
        resolve_subclaim(
            &subclaim,
            &prover_aux.projected_trace_f,
            &prover_aux.field_cfg,
        )
        .expect("Subclaim resolution failed");
    }

    /// End-to-end test: BinaryDecompositionUair.
    ///
    /// Uses binary_poly (1 col) and int (1 col) trace types.
    /// UAIR constraint: binary_poly[0] - int[0] ∈ ⟨X - 2⟩
    #[test]
    fn test_end_to_end_binary_decomposition() {
        let mut rng = rng();
        let num_vars = 4;
        let pp = setup_pp(num_vars);

        let (binary_trace, arb_trace, int_trace) =
            BinaryDecompositionUair::generate_witness(num_vars, &mut rng);

        // BinaryDecompositionUair uses u32 coefficients.
        let project_scalar_u32 = |scalar: &DensePolynomial<u32, 32>,
                                  field_cfg: &<F as PrimeField>::Config|
         -> DynamicPolynomialF<F> {
            scalar
                .iter()
                .map(|coeff| F::from_with_cfg(coeff, field_cfg))
                .collect()
        };

        // ── Prover ──
        // TODO: witness_columns should come from the original trace columns.
        // Currently passing empty since we can't convert trace column types
        // to DenseMultilinearExtension<Zt::Eval> yet.
        let (proof, prover_aux) =
            prove::<Zt, Lc, BinaryDecompositionUair, F, u32, u32, _, CHECKED, DEGREE_PLUS_ONE>(
                &pp,
                &[], // witness_columns — empty for now
                &binary_trace,
                &arb_trace,
                &int_trace,
                num_vars,
                project_scalar_u32,
            )
            .expect("Prover failed");

        // ── Verifier ──
        let subclaim =
            verify::<Zt, Lc, BinaryDecompositionUair, F, _, _, _, CHECKED, DEGREE_PLUS_ONE>(
                &pp,
                proof,
                num_vars,
                project_scalar_u32,
                |ideal, field_cfg| ideal.map(|i| DegreeOneIdeal::from_with_cfg(i, field_cfg)),
            )
            .expect("Verifier failed");

        // ── Subclaim resolution (in lieu of PCS) ──
        resolve_subclaim(
            &subclaim,
            &prover_aux.projected_trace_f,
            &prover_aux.field_cfg,
        )
        .expect("Subclaim resolution failed");
    }

    /// End-to-end test: BigLinearUair.
    ///
    /// Uses 16 binary_poly cols and 1 int col.
    /// UAIR constraints:
    ///   sum(up.binary_poly[0..16]) - up.int[0] ∈ ⟨X - 1⟩
    ///   down.binary_poly[0] - up.int[0] ∈ ⟨X - 2⟩
    ///   up.binary_poly[i] - down.binary_poly[i] = 0, for i=1..15
    #[test]
    fn test_end_to_end_big_linear() {
        let mut rng = rng();
        let num_vars = 4;
        let pp = setup_pp(num_vars);

        let (binary_trace, arb_trace, int_trace) =
            BigLinearUair::generate_witness(num_vars, &mut rng);

        let project_scalar_u32 = |scalar: &DensePolynomial<u32, 32>,
                                  field_cfg: &<F as PrimeField>::Config|
         -> DynamicPolynomialF<F> {
            scalar
                .iter()
                .map(|coeff| F::from_with_cfg(coeff, field_cfg))
                .collect()
        };

        // ── Prover ──
        let (proof, prover_aux) =
            prove::<Zt, Lc, BigLinearUair, F, u32, u32, _, CHECKED, DEGREE_PLUS_ONE>(
                &pp,
                &[], // witness_columns — empty for now
                &binary_trace,
                &arb_trace,
                &int_trace,
                num_vars,
                project_scalar_u32,
            )
            .expect("Prover failed");

        // ── Verifier ──
        let subclaim = verify::<Zt, Lc, BigLinearUair, F, _, _, _, CHECKED, DEGREE_PLUS_ONE>(
            &pp,
            proof,
            num_vars,
            project_scalar_u32,
            |ideal, field_cfg| ideal.map(|i| DegreeOneIdeal::from_with_cfg(i, field_cfg)),
        )
        .expect("Verifier failed");

        // ── Subclaim resolution (in lieu of PCS) ──
        resolve_subclaim(
            &subclaim,
            &prover_aux.projected_trace_f,
            &prover_aux.field_cfg,
        )
        .expect("Subclaim resolution failed");
    }
     */
}
