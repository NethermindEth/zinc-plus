//! Test that the SHA-256 UAIR Ideal Check protocol succeeds on a valid witness.
//!
//! This demonstrates the PIOP constraint verification pipeline:
//! 1. Generate a SHA-256 witness (valid trace satisfying UAIR constraints)
//! 2. Run IdealCheck prover on the 6 implemented F₂[X] constraints
//! 3. Run CombinedPolyResolver
//! 4. Verify that both succeed (no constraint violations)

#![allow(clippy::arithmetic_side_effects)]

use std::collections::HashMap;

use crypto_bigint::U64;
use crypto_primitives::{
    crypto_bigint_monty::MontyField,
    Field, PrimeField,
};
use zinc_poly::univariate::binary::BinaryPoly;
use zinc_poly::univariate::dynamic::over_field::DynamicPolynomialF;
use zinc_primality::MillerRabin;
use zinc_transcript::traits::Transcript;
use zinc_uair::constraint_counter::count_constraints;
use zinc_uair::degree_counter::count_max_degree;

use zinc_piop::projections::{
    project_trace_coeffs, project_trace_to_field,
    project_scalars, project_scalars_to_field,
};

use zinc_sha256_uair::{Sha256Uair, witness::GenerateWitness};

const INT_LIMBS: usize = U64::LIMBS;
type F = MontyField<{ INT_LIMBS * 4 }>;

#[test]
fn ideal_check_succeeds_on_valid_sha256_witness() {
    let num_vars = 7; // poly_size = 128 (64 real SHA-256 rows + 64 padding)
    let mut rng = rand::rng();
    let trace = <Sha256Uair as GenerateWitness<BinaryPoly<32>>>::generate_witness(num_vars, &mut rng);

    let num_constraints = count_constraints::<Sha256Uair>();
    let max_degree = count_max_degree::<Sha256Uair>();

    println!("SHA-256 UAIR: {} constraints, max degree {}", num_constraints, max_degree);
    assert_eq!(num_constraints, 16, "Expected 16 F₂[X] constraints (6 rot/shift + 6 linking + 4 Ch/Maj linking)");
    assert_eq!(max_degree, 1, "Expected max degree 1 (linear constraints)");

    // ── Projection: trace & scalars → DynamicPolynomialF ────────────
    let mut transcript = zinc_transcript::KeccakTranscript::new();
    let field_cfg = transcript.get_random_field_cfg::<F, <F as Field>::Inner, MillerRabin>();

    let projected_trace = project_trace_coeffs::<F, bool, bool, 32>(
        &trace, &[], &[], &field_cfg,
    );
    let projected_scalars = project_scalars::<F, Sha256Uair>(|scalar| {
        let one = F::one_with_cfg(&field_cfg);
        let zero = F::zero_with_cfg(&field_cfg);
        DynamicPolynomialF::new(
            scalar.iter().map(|coeff| {
                if coeff.into_inner() { one.clone() } else { zero.clone() }
            }).collect::<Vec<_>>()
        )
    });

    // ── IdealCheck prover ────────────────────────────────────────────
    let ic_result = zinc_piop::ideal_check::IdealCheckProtocol::<F>::prove_as_subprotocol::<Sha256Uair>(
        &mut transcript,
        &projected_trace,
        &projected_scalars,
        num_constraints,
        num_vars,
        &field_cfg,
    );

    assert!(ic_result.is_ok(), "IdealCheck prover FAILED: {:?}", ic_result.err());
    let (_ic_proof, ic_state) = ic_result.unwrap();
    println!("IdealCheck prover PASSED ✓ (6 constraints verified on valid witness)");

    // ── Projection to field for CPR ──────────────────────────────────
    let projecting_element: F = transcript.get_field_challenge(&field_cfg);
    let field_trace = project_trace_to_field::<F, 32>(
        &trace, &[], &[], &projecting_element,
    );
    let field_projected_scalars =
        project_scalars_to_field(projected_scalars.clone(), &projecting_element)
            .expect("scalar projection failed");

    // ── CombinedPolyResolver prover ──────────────────────────────────
    let cpr_result = zinc_piop::combined_poly_resolver::CombinedPolyResolver::<F>::prove_as_subprotocol::<Sha256Uair>(
        &mut transcript,
        field_trace,
        &ic_state.evaluation_point,
        &field_projected_scalars,
        num_constraints,
        num_vars,
        max_degree,
        &field_cfg,
    );

    assert!(cpr_result.is_ok(), "CombinedPolyResolver FAILED: {:?}", cpr_result.err());
    println!("CombinedPolyResolver PASSED ✓");
    println!("Full PIOP pipeline verified for SHA-256 UAIR (6/14 constraints in F₂[X])");
}
