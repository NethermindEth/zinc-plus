//! Test that the ECDSA IC + CPR pipeline succeeds on a valid witness.
//!
//! All 9 ECDSA DensePolynomial constraints use `assert_zero` (ImpossibleIdeal),
//! so the verifier only needs to check that each combined MLE value is exactly
//! zero (no non-trivial ideal membership).
//!
//! Steps:
//! 1. Generate a valid DensePolynomial<i64, 1> witness (constant-row fixed point)
//! 2. Project trace and scalars for IdealCheck
//! 3. Run IdealCheck prover (9 constraints, max degree ≤ 12)
//! 4. Project to field for CombinedPolyResolver
//! 5. Run CombinedPolyResolver prover
//! 6. Run IdealCheck verifier with zero-only ideal checks
//! 7. Run CPR verifier
//! 8. Assert all steps succeed

#![allow(clippy::arithmetic_side_effects)]

use crypto_bigint::U64;
use crypto_primitives::{Field, FromWithConfig, PrimeField};
use zinc_poly::univariate::dense::DensePolynomial;
use zinc_poly::univariate::dynamic::over_field::DynamicPolynomialF;
use zinc_primality::MillerRabin;
use zinc_transcript::traits::Transcript;
use zinc_uair::ideal::{Ideal, IdealCheck, ImpossibleIdeal};
use zinc_uair::ideal_collector::IdealOrZero;
use zinc_utils::from_ref::FromRef;

use zinc_ecdsa_uair::{
    EcdsaUairDp, NUM_CONSTRAINTS_I64,
    witness::GenerateWitness,
};

use zinc_piop::ideal_check::IdealCheckProtocol;
use zinc_piop::projections::{
    project_trace_coeffs, project_trace_to_field,
    project_scalars, project_scalars_to_field,
};
use zinc_uair::constraint_counter::count_constraints;
use zinc_uair::degree_counter::count_max_degree;

const INT_LIMBS: usize = U64::LIMBS;
type F = crypto_primitives::crypto_bigint_monty::MontyField<{ INT_LIMBS * 4 }>;

// ─── Zero-only ideal for the verifier ──────────────────────────────────────

/// An ideal that only contains the zero polynomial.
/// Used for verifying ECDSA constraints (all assert_zero).
#[derive(Clone, Debug)]
struct ZeroOnlyIdeal;

impl FromRef<ZeroOnlyIdeal> for ZeroOnlyIdeal {
    fn from_ref(_: &ZeroOnlyIdeal) -> Self {
        ZeroOnlyIdeal
    }
}

impl Ideal for ZeroOnlyIdeal {}

impl<Fld: PrimeField> IdealCheck<DynamicPolynomialF<Fld>> for ZeroOnlyIdeal {
    fn contains(&self, value: &DynamicPolynomialF<Fld>) -> bool {
        num_traits::Zero::is_zero(value)
    }
}

#[test]
fn ecdsa_ideal_check_succeeds_on_valid_witness() {
    let num_vars = 4; // 2^4 = 16 rows
    let mut rng = rand::rng();

    // Step 1: Generate valid i64 witness (constant-row fixed point).
    let trace = <EcdsaUairDp as GenerateWitness<DensePolynomial<i64, 1>>>::generate_witness(
        num_vars, &mut rng,
    );

    let num_constraints = count_constraints::<EcdsaUairDp>();
    let max_degree = count_max_degree::<EcdsaUairDp>();

    println!(
        "ECDSA i64 UAIR: {} constraints, max degree {}",
        num_constraints, max_degree
    );
    assert_eq!(num_constraints, NUM_CONSTRAINTS_I64);
    assert!(max_degree >= 4 && max_degree <= 12);
    // Step 2: Project trace and scalars for IdealCheck.
    let mut transcript = zinc_transcript::KeccakTranscript::new();
    let field_cfg = transcript.get_random_field_cfg::<F, <F as Field>::Inner, MillerRabin>();

    let projected_trace = project_trace_coeffs::<F, i64, i64, 1>(
        &[], &trace, &[], &field_cfg,
    );
    let projected_scalars = project_scalars::<F, EcdsaUairDp>(|scalar| {
        DynamicPolynomialF::new(
            scalar.iter().map(|coeff| F::from_with_cfg(*coeff, &field_cfg)).collect::<Vec<_>>()
        )
    });

    // Step 3: IdealCheck prover.
    let ic_result =
        IdealCheckProtocol::<F>::prove_as_subprotocol::<EcdsaUairDp>(
            &mut transcript,
            &projected_trace,
            &projected_scalars,
            num_constraints,
            num_vars,
            &field_cfg,
        );

    assert!(
        ic_result.is_ok(),
        "ECDSA IdealCheck prover FAILED: {:?}",
        ic_result.err()
    );
    let (ic_proof, ic_state) = ic_result.unwrap();
    println!("ECDSA IdealCheck prover PASSED ✓ ({num_constraints} constraints)");

    // Step 4: Project to field for CPR.
    let projecting_element: F = transcript.get_field_challenge(&field_cfg);
    let field_trace = project_trace_to_field::<F, 1>(
        &[], &projected_trace, &[], &projecting_element,
    );
    let field_projected_scalars =
        project_scalars_to_field(projected_scalars.clone(), &projecting_element)
            .expect("scalar projection failed");

    // Step 5: CombinedPolyResolver prover.
    let cpr_result =
        zinc_piop::combined_poly_resolver::CombinedPolyResolver::<F>::prove_as_subprotocol::<
            EcdsaUairDp,
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

    assert!(
        cpr_result.is_ok(),
        "ECDSA CPR prover FAILED: {:?}",
        cpr_result.err()
    );
    let (cpr_proof, _cpr_state) = cpr_result.unwrap();
    println!("ECDSA CombinedPolyResolver prover PASSED ✓");

    // Step 6: IdealCheck verifier with zero-only ideal checks.
    let mut verify_transcript = zinc_transcript::KeccakTranscript::new();
    let verify_field_cfg =
        verify_transcript.get_random_field_cfg::<F, <F as Field>::Inner, MillerRabin>();

    let ideal_from_ref = |ideal: &IdealOrZero<ImpossibleIdeal>| -> ZeroOnlyIdeal {
        match ideal {
            IdealOrZero::Zero => ZeroOnlyIdeal,
            IdealOrZero::Ideal(_) => panic!("ECDSA has no non-zero ideal constraints"),
        }
    };

    let ic_verify_result = IdealCheckProtocol::<F>::verify_as_subprotocol::<
        EcdsaUairDp,
        _,
        _,
    >(
        &mut verify_transcript,
        ic_proof,
        num_constraints,
        num_vars,
        ideal_from_ref,
        &verify_field_cfg,
    );

    assert!(
        ic_verify_result.is_ok(),
        "ECDSA IdealCheck verifier FAILED: {:?}",
        ic_verify_result.err()
    );
    let ic_subclaim = ic_verify_result.unwrap();
    println!("ECDSA IdealCheck verifier PASSED ✓ (all zero-ideal checks)");

    // Step 7: CPR verifier.
    let verify_projected_scalars_coeffs = project_scalars::<F, EcdsaUairDp>(|scalar| {
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
            EcdsaUairDp,
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
        "ECDSA CPR verifier FAILED: {:?}",
        cpr_verify_result.err()
    );
    println!("ECDSA CombinedPolyResolver verifier PASSED ✓");

    println!("\n✓ Full ECDSA IC + CPR pipeline verified (9 assert_zero constraints, max degree {max_degree})");
}
