//! Zinc+ PIOP for UCS — end-to-end protocol (without PCS).
//!
//! Implements the four steps of the Zinc+ compiler from
//! Section 2.2 "Combining the three steps" of the paper:
//!
//! ```text
//! Q[X]  ──φ_q──▶  F_q[X]  ──MLE eval──▶  F_q[X]  ──ψ_a──▶  F_q
//!       Step 1             Step 2                  Step 3
//! ```
//!
//! Step 4 runs a finite-field PIOP (sumcheck) over F_q.
//!
//! The verifier's output is a [`Subclaim`] containing evaluation
//! claims about the trace column MLEs. In the full protocol,
//! these would be resolved by the Zip+ PCS.

use crypto_primitives::{ConstIntSemiring, FromPrimitiveWithConfig, FromWithConfig, PrimeField};
use num_traits::Zero;
use thiserror::Error;
use zip_plus::merkle::MtHash;
use zinc_piop::{
    combined_poly_resolver::{self, CombinedPolyResolver, CombinedPolyResolverError},
    ideal_check::{self, IdealCheckProtocol},
    projections::{
        project_scalars, project_scalars_to_field, project_trace_coeffs, project_trace_to_field,
    },
};
use zinc_poly::{
    mle::{DenseMultilinearExtension, MultilinearExtensionWithConfig},
    univariate::{
        binary::BinaryPoly, dense::DensePolynomial, dynamic::over_field::DynamicPolynomialF,
    },
};
use zinc_primality::{MillerRabin, PrimalityTest};
use zinc_transcript::{
    KeccakTranscript,
    traits::{ConstTranscribable, Transcript},
};
use zinc_uair::{
    Uair,
    constraint_counter::count_constraints,
    degree_counter::count_max_degree,
    ideal::{Ideal, IdealCheck},
    ideal_collector::IdealOrZero,
};
use zinc_utils::{from_ref::FromRef, inner_transparent_field::InnerTransparentField};

// ─── Data structures ────────────────────────────────────────────

/// Proof produced by the Zinc+ PIOP for UCS (without PCS).
///
/// Contains the two subproofs from Steps 2 and 4:
/// - `ideal_check`: MLE evaluations in F_q\[X\] (Step 2).
/// - `resolver`: sumcheck proof + trace evaluation claims (Step 4).
#[derive(Clone, Debug)]
pub struct Proof<F: PrimeField> {
    pub commitments: Vec<MtHash>,
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

// ─── Error type ─────────────────────────────────────────────────

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
}

// ─── Prover ─────────────────────────────────────────────────────

/// Zinc+ PIOP Prover (Algorithm 1 from the paper, Steps 1–4).
///
/// # Protocol flow (paper Section 2.2 "Combining the three steps"):
///
/// 1. **Prime projection** (φ_q: Q\[X\] → F_q\[X\]): sample random prime q from
///    transcript, project trace and scalars.
/// 2. **Ideal check**: sample r ∈ F_q^μ, prover sends MLE evaluations, verifier
///    checks ideal membership.
/// 3. **Evaluation projection** (ψ_a: F_q\[X\] → F_q): sample a ∈ F_q, evaluate
///    polynomials at X = a.
/// 4. **Finite-field PIOP**: sumcheck over F_q to prove the projected claim.
///
/// Returns the proof and auxiliary data (for subclaim resolution without PCS).
#[allow(clippy::too_many_arguments)]
pub fn prove<U, F, FMod, PolyCoeff, IntType, ProjectScalar, CommitFn, const D: usize>(
    binary_poly_trace: &[DenseMultilinearExtension<BinaryPoly<D>>],
    arbitrary_poly_trace: &[DenseMultilinearExtension<DensePolynomial<PolyCoeff, D>>],
    int_trace: &[DenseMultilinearExtension<IntType>],
    num_vars: usize,
    project_scalar: ProjectScalar,
    commit_traces: CommitFn,
) -> Result<(Proof<F>, ProverAux<F>), ProtocolError<F>>
where
    F: InnerTransparentField
        + FromPrimitiveWithConfig
        + FromWithConfig<PolyCoeff>
        + FromWithConfig<IntType>
        + FromRef<F>
        + Send
        + Sync
        + 'static,
    F::Inner: ConstIntSemiring + ConstTranscribable + FromRef<FMod> + Send + Sync + Zero + Default,
    FMod: ConstTranscribable + ConstIntSemiring,
    PolyCoeff: Clone + Send + Sync,
    IntType: Clone + Send + Sync,
    U: Uair + 'static,
    ProjectScalar: Fn(&U::Scalar, &F::Config) -> DynamicPolynomialF<F>,
    CommitFn: FnOnce(
        &[DenseMultilinearExtension<BinaryPoly<D>>],
        &[DenseMultilinearExtension<DensePolynomial<PolyCoeff, D>>],
        &[DenseMultilinearExtension<IntType>],
    ) -> Vec<MtHash>,
    MillerRabin: PrimalityTest<FMod>,
{
    // ── Step 0: Commit to witness traces ────────────────────────
    // Create a fresh Fiat-Shamir transcript. The prover commits to
    // all witness columns via Zip+ PCS, then absorbs the Merkle roots
    // into the transcript before any challenges are derived.
    let mut transcript = KeccakTranscript::new();
    let commitments = commit_traces(binary_poly_trace, arbitrary_poly_trace, int_trace);
    for comm in &commitments {
        transcript.absorb_slice(&comm);
    }
    // TODO: We have to absorb public inputs as well once we add them to the protocol,
    //       or this will open up a soundness vulnerability!

    // ── Step 1: Prime projection (φ_q: Q[X] → F_q[X]) ──────────
    // Sample a random Ω(λ)-bit prime q from the Fiat-Shamir transcript.
    let field_cfg = transcript.get_random_field_cfg::<F, FMod, MillerRabin>();

    // Project the witness trace from Q[X] to F_q[X].
    let projected_trace = project_trace_coeffs::<F, PolyCoeff, IntType, D>(
        binary_poly_trace,
        arbitrary_poly_trace,
        int_trace,
        &field_cfg,
    );

    // Project UAIR scalars from Q[X] to F_q[X].
    let projected_scalars_fx = project_scalars::<F, U>(|s| project_scalar(s, &field_cfg));

    let num_constraints = count_constraints::<U>();

    // ── Step 2: Randomized ideal check ──────────────────────────
    // The verifier samples r ∈ F_q^μ via the transcript. The prover
    // computes MLE evaluations of the combined constraint polynomials
    // at r and sends them. The verifier checks each e_i ∈ p_i'.
    let (ic_proof, ic_prover_state) = IdealCheckProtocol::prove_as_subprotocol::<U>(
        &mut transcript,
        &projected_trace,
        &projected_scalars_fx,
        num_constraints,
        num_vars,
        &field_cfg,
    )
    .map_err(|e| ProtocolError::IdealCheck(format!("{e:?}")))?;

    // ── Step 3: Evaluation projection (ψ_a: F_q[X] → F_q) ─────
    // Sample a ∈ F_q from the transcript.
    let projecting_element: F = transcript.get_field_challenge(&field_cfg);

    // Project trace from F_q[X] to F_q by evaluating each polynomial at X = a.
    // After project_trace_coeffs all column types are unified as
    // DynamicPolynomialF<F>, so we pass them as "arbitrary" polynomials for the
    // field projection.
    let projected_trace_f =
        project_trace_to_field::<F, D>(&[], &projected_trace, &[], &projecting_element);

    // Project scalars from F_q[X] to F_q.
    let projected_scalars_f =
        project_scalars_to_field(projected_scalars_fx, &projecting_element)
            .map_err(|(_s, _f, e)| ProtocolError::ScalarProjection(format!("{e:?}")))?;

    let max_degree = count_max_degree::<U>();

    // ── Step 4: Finite-field PIOP (sumcheck over F_q) ───────────
    // Prove the claim ψ_{q,a}(Σ^{r,e})(ψ_{q,a}(y), ψ_{q,a}(f_0)) = 0
    // in F_q via a sumcheck protocol.
    let (cpr_proof, _) = CombinedPolyResolver::prove_as_subprotocol::<U>(
        &mut transcript,
        projected_trace_f.clone(),
        &ic_prover_state.evaluation_point,
        &projected_scalars_f,
        num_constraints,
        num_vars,
        max_degree,
        &field_cfg,
    )?;

    Ok((
        Proof {
            commitments,
            ideal_check: ic_proof,
            resolver: cpr_proof,
        },
        ProverAux {
            field_cfg,
            projected_trace_f,
        },
    ))
}

// ─── Verifier ───────────────────────────────────────────────────

/// Zinc+ PIOP Verifier (Algorithm 1, verification side).
///
/// Verifies all four steps and returns a [`Subclaim`] containing
/// evaluation claims to be resolved by the PCS (or directly for testing).
#[allow(clippy::too_many_arguments)]
pub fn verify<U, F, FMod, IdealOverF, ProjectScalar, ProjectIdeal, const D: usize>(
    proof: Proof<F>,
    num_vars: usize,
    project_scalar: ProjectScalar,
    project_ideal: ProjectIdeal,
) -> Result<Subclaim<F>, ProtocolError<F>>
where
    F: InnerTransparentField + FromPrimitiveWithConfig + FromRef<F> + Send + Sync + 'static,
    F::Inner: ConstIntSemiring + ConstTranscribable + FromRef<FMod> + Send + Sync + Zero + Default,
    FMod: ConstTranscribable + ConstIntSemiring,
    U: Uair + 'static,
    IdealOverF: Ideal + IdealCheck<DynamicPolynomialF<F>>,
    ProjectScalar: Fn(&U::Scalar, &F::Config) -> DynamicPolynomialF<F>,
    ProjectIdeal: Fn(&IdealOrZero<U::Ideal>, &F::Config) -> IdealOverF,
    MillerRabin: PrimalityTest<FMod>,
{
    // ── Step 0: Reconstruct transcript from commitments ─────────
    // The verifier creates the same fresh transcript and absorbs the
    // commitments so that all subsequent challenges match the prover's.
    let mut transcript = KeccakTranscript::new();
    for comm in &proof.commitments {
        transcript.absorb_slice(&comm);
    }

    // ── Step 1: Prime projection ────────────────────────────────
    // Both parties derive the same random prime from the transcript.
    let field_cfg = transcript.get_random_field_cfg::<F, FMod, MillerRabin>();

    let num_constraints = count_constraints::<U>();

    // ── Step 2: Verify ideal check ──────────────────────────────
    // Verifier checks that the prover's MLE evaluations belong to the
    // corresponding ideals.
    let ic_subclaim = IdealCheckProtocol::verify_as_subprotocol::<U, IdealOverF, _>(
        &mut transcript,
        proof.ideal_check,
        num_constraints,
        num_vars,
        |ideal| project_ideal(ideal, &field_cfg),
        &field_cfg,
    )
    .map_err(|e| ProtocolError::IdealCheck(format!("{e:?}")))?;

    // ── Step 3: Evaluation projection ───────────────────────────
    // Sample projecting element a ∈ F_q.
    let projecting_element: F = transcript.get_field_challenge(&field_cfg);

    // Verifier independently computes projected scalars (public data).
    let projected_scalars_fx = project_scalars::<F, U>(|s| project_scalar(s, &field_cfg));
    let projected_scalars_f =
        project_scalars_to_field(projected_scalars_fx, &projecting_element)
            .map_err(|(_s, _f, e)| ProtocolError::ScalarProjection(format!("{e:?}")))?;

    let max_degree = count_max_degree::<U>();

    // ── Step 4: Verify finite-field PIOP ────────────────────────
    let cpr_subclaim = CombinedPolyResolver::verify_as_subprotocol::<U>(
        &mut transcript,
        proof.resolver,
        num_constraints,
        num_vars,
        max_degree,
        &projecting_element,
        &projected_scalars_f,
        ic_subclaim,
        &field_cfg,
    )?;

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

// ─── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    use crypto_primitives::{Field, crypto_bigint_int::Int, crypto_bigint_monty::MontyField};
    use rand::rng;
    use zinc_poly::univariate::ideal::DegreeOneIdeal;
    use zinc_test_uair::{
        BigLinearUair, BinaryDecompositionUair, GenerateMultiTypeWitness,
        GenerateSingleTypeWitness, TestAirNoMultiplication, TestUairSimpleMultiplication,
    };

    const INT_LIMBS: usize = 5;
    const FIELD_LIMBS: usize = 4;
    const DEGREE_PLUS_ONE: usize = 32;

    type F = MontyField<FIELD_LIMBS>;
    type Witness = DensePolynomial<Int<INT_LIMBS>, DEGREE_PLUS_ONE>;

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

    /// No-op commit closure for tests without real Zip+ PCS.
    fn no_commit<B, P, I>(_: &[B], _: &[P], _: &[I]) -> Vec<MtHash> {
        vec![]
    }

    /// End-to-end test: TestAirNoMultiplication.
    ///
    /// UAIR constraint: a + b - c ∈ (X - 2)
    /// (one constraint, no polynomial multiplication, ideal = ⟨X - 2⟩).
    #[test]
    fn test_end_to_end_no_multiplication() {
        let mut rng = rng();
        let num_vars = 4;

        // Generate a valid witness satisfying the UAIR constraints.
        let trace = TestAirNoMultiplication::<INT_LIMBS>::generate_witness(num_vars, &mut rng);

        // ── Prover ──
        let (proof, prover_aux) = prove::<
            TestAirNoMultiplication<INT_LIMBS>,
            F,
            <F as Field>::Inner,
            Int<INT_LIMBS>,
            Int<INT_LIMBS>,
            _,
            _,
            DEGREE_PLUS_ONE,
        >(
            &[],
            &trace,
            &[],
            num_vars,
            project_scalar_fn,
            no_commit,
        )
        .expect("Prover failed");

        // ── Verifier ──
        let subclaim = verify::<
            TestAirNoMultiplication<INT_LIMBS>,
            F,
            <F as Field>::Inner,
            _,
            _,
            _,
            DEGREE_PLUS_ONE,
        >(
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

        let trace =
            TestUairSimpleMultiplication::<Int<INT_LIMBS>>::generate_witness(num_vars, &mut rng);

        // ── Prover ──
        let (proof, prover_aux) = prove::<
            TestUairSimpleMultiplication<Int<INT_LIMBS>>,
            F,
            <F as Field>::Inner,
            Int<INT_LIMBS>,
            Int<INT_LIMBS>,
            _,
            _,
            DEGREE_PLUS_ONE,
        >(
            &[],
            &trace,
            &[],
            num_vars,
            project_scalar_fn,
            no_commit,
        )
        .expect("Prover failed");

        // ── Verifier ──
        let subclaim = verify::<
            TestUairSimpleMultiplication<Int<INT_LIMBS>>,
            F,
            <F as Field>::Inner,
            _,
            _,
            _,
            DEGREE_PLUS_ONE,
        >(
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
        let (proof, prover_aux) = prove::<
            BinaryDecompositionUair,
            F,
            <F as Field>::Inner,
            u32,
            u32,
            _,
            _,
            DEGREE_PLUS_ONE,
        >(
            &binary_trace,
            &arb_trace,
            &int_trace,
            num_vars,
            project_scalar_u32,
            no_commit,
        )
        .expect("Prover failed");

        // ── Verifier ──
        let subclaim =
            verify::<BinaryDecompositionUair, F, <F as Field>::Inner, _, _, _, DEGREE_PLUS_ONE>(
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
        let (proof, prover_aux) = prove::<
            BigLinearUair,
            F,
            <F as Field>::Inner,
            u32,
            u32,
            _,
            _,
            DEGREE_PLUS_ONE,
        >(
            &binary_trace,
            &arb_trace,
            &int_trace,
            num_vars,
            project_scalar_u32,
            no_commit,
        )
        .expect("Prover failed");

        // ── Verifier ──
        let subclaim = verify::<BigLinearUair, F, <F as Field>::Inner, _, _, _, DEGREE_PLUS_ONE>(
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
}
