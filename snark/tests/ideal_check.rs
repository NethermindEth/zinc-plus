//! Test that the SHA-256 UAIR Ideal Check protocol succeeds on a valid witness.
//!
//! This demonstrates the PIOP constraint verification pipeline:
//! 1. Generate a SHA-256 witness (valid trace satisfying UAIR constraints)
//! 2. Run IdealCheck prover on the 6 implemented F₂[X] constraints
//! 3. Run CombinedPolyResolver
//! 4. Verify that both succeed (no constraint violations)

#![allow(clippy::arithmetic_side_effects)]

use crypto_bigint::U64;
use crypto_primitives::{
    crypto_bigint_monty::MontyField,
    Field,
};
use zinc_poly::univariate::binary::BinaryPoly;
use zinc_primality::MillerRabin;
use zinc_transcript::traits::Transcript;
use zinc_uair::constraint_counter::count_constraints;
use zinc_uair::degree_counter::count_max_degree;

use zinc_sha256_uair::{Sha256Uair, witness::GenerateWitness};

const INT_LIMBS: usize = U64::LIMBS;
type F = MontyField<{ INT_LIMBS * 4 }>;

#[test]
fn ideal_check_succeeds_on_valid_sha256_witness() {
    let num_vars = 7; // poly_size = 128 (64 real SHA-256 rows + 64 padding)
    let mut rng = rand::rng();
    let trace = <Sha256Uair as GenerateWitness<BinaryPoly<32>>>::generate_witness(num_vars, &mut rng);

    let num_constraints = count_constraints::<BinaryPoly<32>, Sha256Uair>();
    let max_degree = count_max_degree::<BinaryPoly<32>, Sha256Uair>();

    println!("SHA-256 UAIR: {} constraints, max degree {}", num_constraints, max_degree);
    assert_eq!(num_constraints, 6, "Expected 6 F₂[X] constraints");
    assert_eq!(max_degree, 1, "Expected max degree 1 (linear constraints)");

    // ── IdealCheck prover ────────────────────────────────────────────
    let mut transcript = zinc_transcript::KeccakTranscript::new();
    let field_cfg = transcript.get_random_field_cfg::<F, <F as Field>::Inner, MillerRabin>();

    let ic_result = zinc_piop::ideal_check::IdealCheckProtocol::<F>::prove_as_subprotocol::<BinaryPoly<32>, Sha256Uair>(
        &mut transcript,
        &trace,
        num_constraints,
        num_vars,
        &field_cfg,
    );

    assert!(ic_result.is_ok(), "IdealCheck prover FAILED: {:?}", ic_result.err());
    let (_ic_proof, ic_state) = ic_result.unwrap();
    println!("IdealCheck prover PASSED ✓ (6 constraints verified on valid witness)");

    // ── CombinedPolyResolver prover ──────────────────────────────────
    let cpr_result = zinc_piop::combined_poly_resolver::CombinedPolyResolver::<F>::prove_as_subprotocol::<BinaryPoly<32>, Sha256Uair>(
        &mut transcript,
        &ic_state.trace_matrix,
        &ic_state.evaluation_point,
        ic_state.projected_scalars,
        num_constraints,
        num_vars,
        max_degree,
        &field_cfg,
    );

    assert!(cpr_result.is_ok(), "CombinedPolyResolver FAILED: {:?}", cpr_result.err());
    println!("CombinedPolyResolver PASSED ✓");
    println!("Full PIOP pipeline verified for SHA-256 UAIR (6/14 constraints in F₂[X])");
}
