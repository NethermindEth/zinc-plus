//! Test that the SHA-256 Q[X] IdealCheck + CPR pipeline succeeds on a valid
//! witness with real ideal membership verification.
//!
//! This demonstrates that the multi-ring IdealCheck generalization works:
//! 1. Generate a valid SHA-256 witness (BinaryPoly<32> trace)
//! 2. Convert the trace to DensePolynomial<i64, 64> for Q[X] constraints
//! 3. Run IdealCheck prover on the 5 Q[X] constraints
//!    - 3 BitPoly checks (ch_ef, ch_neg_eg, Maj ∈ {0,1} coefficients)
//!    - 2 carry propagation checks (a-update and e-update via (X−2) ideal)
//! 4. Run CombinedPolyResolver
//! 5. Run IdealCheck verifier with REAL ideal checks (not TrivialIdeal)
//! 6. Run CPR verifier
//! 7. Assert that all steps succeed

#![allow(clippy::arithmetic_side_effects)]

use crypto_bigint::U64;
use crypto_primitives::{
    crypto_bigint_monty::MontyField,
    Field, FromWithConfig,
};
use zinc_poly::univariate::binary::BinaryPoly;
use zinc_poly::univariate::dynamic::over_field::DynamicPolynomialF;
use zinc_primality::MillerRabin;
use zinc_transcript::traits::Transcript;
use zinc_uair::constraint_counter::count_constraints;
use zinc_uair::degree_counter::count_max_degree;
use zinc_uair::ideal_collector::IdealOrZero;

use zinc_sha256_uair::{
    Sha256QxIdeal, Sha256QxIdealOverF, Sha256Uair, Sha256UairQx, convert_trace_to_qx,
    witness::GenerateWitness,
};

use zinc_piop::ideal_check::IdealCheckProtocol;
use zinc_piop::projections::{
    project_trace_coeffs, project_trace_to_field,
    project_scalars, project_scalars_to_field,
};

const INT_LIMBS: usize = U64::LIMBS;
type F = MontyField<{ INT_LIMBS * 4 }>;

#[test]
fn qx_ideal_check_succeeds_on_valid_sha256_witness() {
    let num_vars = 7; // poly_size = 128 (64 real SHA-256 rows + 64 padding)
    let mut rng = rand::rng();

    // Step 1: Generate BinaryPoly<32> witness trace
    let bp_trace = <Sha256Uair as GenerateWitness<BinaryPoly<32>>>::generate_witness(num_vars, &mut rng);

    // Step 2: Convert to DensePolynomial<i64, 64> for Q[X] constraints
    let qx_trace = convert_trace_to_qx(&bp_trace);
    assert_eq!(qx_trace.len(), bp_trace.len(), "trace column count mismatch");
    assert_eq!(qx_trace[0].evaluations.len(), bp_trace[0].evaluations.len(), "row count mismatch");

    let num_constraints = count_constraints::<Sha256UairQx>();
    let max_degree = count_max_degree::<Sha256UairQx>();

    println!("SHA-256 Q[X] UAIR: {} constraints, max degree {}", num_constraints, max_degree);
    assert_eq!(num_constraints, 5, "Expected 5 Q[X] constraints");
    assert_eq!(max_degree, 1, "Expected max degree 1");

    // Step 3: Project trace and scalars for IdealCheck
    let mut transcript = zinc_transcript::KeccakTranscript::new();
    let field_cfg = transcript.get_random_field_cfg::<F, <F as Field>::Inner, MillerRabin>();

    let projected_trace = project_trace_coeffs::<F, i64, i64, 64>(
        &[], &qx_trace, &[], &field_cfg,
    );
    let projected_scalars = project_scalars::<F, Sha256UairQx>(|scalar| {
        DynamicPolynomialF::new(
            scalar.iter().map(|coeff| F::from_with_cfg(*coeff, &field_cfg)).collect::<Vec<_>>()
        )
    });

    // Step 4: IdealCheck prover
    let ic_result =
        IdealCheckProtocol::<F>::prove_as_subprotocol::<Sha256UairQx>(
            &mut transcript,
            &projected_trace,
            &projected_scalars,
            num_constraints,
            num_vars,
            &field_cfg,
        );

    assert!(ic_result.is_ok(), "Q[X] IdealCheck prover FAILED: {:?}", ic_result.err());
    let (ic_proof, ic_state) = ic_result.unwrap();
    println!("Q[X] IdealCheck prover PASSED ✓ ({num_constraints} constraints)");

    // Step 5: Project to field for CPR
    let projecting_element: F = transcript.get_field_challenge(&field_cfg);
    let field_trace = project_trace_to_field::<F, 64>(
        &[], &projected_trace, &[], &projecting_element,
    );
    let field_projected_scalars =
        project_scalars_to_field(projected_scalars.clone(), &projecting_element)
            .expect("scalar projection failed");

    // Step 6: CombinedPolyResolver prover
    let cpr_result =
        zinc_piop::combined_poly_resolver::CombinedPolyResolver::<F>::prove_as_subprotocol::<
            Sha256UairQx,
        >(
            &mut transcript,
            field_trace,
            &ic_state.evaluation_point,
            &field_projected_scalars,
            num_constraints,
            num_vars,
            max_degree,
            &field_cfg,
        );

    assert!(cpr_result.is_ok(), "Q[X] CombinedPolyResolver FAILED: {:?}", cpr_result.err());
    let (cpr_proof, _cpr_state) = cpr_result.unwrap();
    println!("Q[X] CombinedPolyResolver prover PASSED ✓");

    // Step 7: IdealCheck verifier with REAL ideal checks
    let mut verify_transcript = zinc_transcript::KeccakTranscript::new();
    let verify_field_cfg = verify_transcript.get_random_field_cfg::<F, <F as Field>::Inner, MillerRabin>();

    let qx_ideal_from_ref = |ideal: &IdealOrZero<Sha256QxIdeal>| -> Sha256QxIdealOverF<F> {
        match ideal {
            IdealOrZero::Zero => Sha256QxIdealOverF::Zero,
            IdealOrZero::Ideal(Sha256QxIdeal::BitPoly(_)) => Sha256QxIdealOverF::BitPoly,
            IdealOrZero::Ideal(Sha256QxIdeal::DegreeOne(_)) => {
                let two = F::from_with_cfg(2i64, &verify_field_cfg);
                Sha256QxIdealOverF::DegreeOne(two)
            }
        }
    };

    let ic_verify_result =
        IdealCheckProtocol::<F>::verify_as_subprotocol::<Sha256UairQx, _, _>(
            &mut verify_transcript,
            ic_proof,
            num_constraints,
            num_vars,
            qx_ideal_from_ref,
            &verify_field_cfg,
        );

    assert!(
        ic_verify_result.is_ok(),
        "Q[X] IdealCheck verifier FAILED: {:?}",
        ic_verify_result.err()
    );
    let ic_subclaim = ic_verify_result.unwrap();
    println!("Q[X] IdealCheck verifier PASSED ✓ (real ideal checks!)");

    // Step 8: CPR verifier
    let verify_projected_scalars_coeffs = project_scalars::<F, Sha256UairQx>(|scalar| {
        DynamicPolynomialF::new(
            scalar.iter().map(|coeff| F::from_with_cfg(*coeff, &verify_field_cfg)).collect::<Vec<_>>()
        )
    });
    let verify_projecting_element: F = verify_transcript.get_field_challenge(&verify_field_cfg);
    let verify_field_projected_scalars =
        project_scalars_to_field(verify_projected_scalars_coeffs, &verify_projecting_element)
            .expect("verify scalar projection failed");

    let cpr_verify_result =
        zinc_piop::combined_poly_resolver::CombinedPolyResolver::<F>::verify_as_subprotocol::<
            Sha256UairQx,
        >(
            &mut verify_transcript,
            cpr_proof,
            num_constraints,
            num_vars,
            max_degree,
            &verify_projecting_element,
            &verify_field_projected_scalars,
            ic_subclaim,
            &verify_field_cfg,
        );

    assert!(
        cpr_verify_result.is_ok(),
        "Q[X] CPR verifier FAILED: {:?}",
        cpr_verify_result.err()
    );
    println!("Q[X] CombinedPolyResolver verifier PASSED ✓");

    println!("\n✓ Full Q[X] IC + CPR pipeline verified for SHA-256 (5 constraints with REAL ideal checks)");
    println!("  3 BitPoly checks (ch_ef, ch_neg_eg, Maj have binary coefficients)");
    println!("  2 carry propagation checks (a-update, e-update via (X−2) ideal)");
}
