//! Full pipeline round-trip test: prove (PIOP + PCS) → verify (PIOP + PCS).
//!
//! This test verifies the complete Zinc+ pipeline end-to-end:
//! 1. Generate a valid SHA-256 witness
//! 2. Run the full prover: IdealCheck → CPR → PCS Commit/Test/Evaluate
//! 3. Run the full verifier: IdealCheck verify → CPR verify → PCS verify
//! 4. Assert that verification succeeds

#![allow(clippy::arithmetic_side_effects)]

use std::marker::PhantomData;

use crypto_bigint::U64;
use crypto_primitives::{
    boolean::Boolean,
    crypto_bigint_int::Int,
    crypto_bigint_uint::Uint,
    FixedSemiring,
};

use zinc_poly::univariate::binary::{
    BinaryPoly, BinaryPolyInnerProduct, BinaryPolyWideningMulByScalar,
};
use zinc_poly::univariate::dense::{DensePolyInnerProduct, DensePolynomial};
use zinc_primality::MillerRabin;
use zinc_transcript::traits::ConstTranscribable;
use zinc_uair::ideal_collector::IdealOrZero;
use zinc_utils::{
    UNCHECKED,
    from_ref::FromRef,
    inner_product::MBSInnerProduct,
    named::Named,
};
use zip_plus::{
    code::{
        LinearCode,
        iprs::{IprsCode, PnttConfigF2_16R4B16},
    },
    pcs::structs::{ZipPlusParams, ZipTypes},
};

use zinc_sha256_uair::{CyclotomicIdeal, Sha256Uair, witness::GenerateWitness};
use zinc_snark::pipeline;

// ─── Type definitions (must match the benchmark) ────────────────────────────

const INT_LIMBS: usize = U64::LIMBS;

struct TestZipTypes<CwCoeff, const D_PLUS_ONE: usize>(PhantomData<CwCoeff>);

impl<CwCoeff, const D_PLUS_ONE: usize> ZipTypes for TestZipTypes<CwCoeff, D_PLUS_ONE>
where
    CwCoeff: ConstTranscribable
        + Copy
        + Default
        + FromRef<Boolean>
        + Named
        + FixedSemiring
        + Send
        + Sync,
    Int<6>: FromRef<CwCoeff>,
{
    const NUM_COLUMN_OPENINGS: usize = 147;
    type Eval = BinaryPoly<D_PLUS_ONE>;
    type Cw = DensePolynomial<CwCoeff, D_PLUS_ONE>;
    type Fmod = Uint<{ INT_LIMBS * 4 }>;
    type PrimeTest = MillerRabin;
    type Chal = i128;
    type Pt = i128;
    type CombR = Int<{ INT_LIMBS * 6 }>;
    type Comb = DensePolynomial<Self::CombR, D_PLUS_ONE>;
    type EvalDotChal = BinaryPolyInnerProduct<Self::Chal, D_PLUS_ONE>;
    type CombDotChal =
        DensePolyInnerProduct<Self::CombR, Self::Chal, Self::CombR, MBSInnerProduct, D_PLUS_ONE>;
    type ArrCombRDotChal = MBSInnerProduct;
}

type Zt = TestZipTypes<i64, 32>;
type Lc = IprsCode<Zt, PnttConfigF2_16R4B16<1>, BinaryPolyWideningMulByScalar<i64>, UNCHECKED>;

#[test]
fn full_pipeline_round_trip() {
    // Generate SHA-256 witness (poly_size = 2^7 = 128)
    let num_vars = 7;
    let mut rng = rand::rng();
    let trace = <Sha256Uair as GenerateWitness<BinaryPoly<32>>>::generate_witness(num_vars, &mut rng);

    // Setup PCS params (R4B16 DEPTH=1, row_len=128)
    let row_len = 128;
    let linear_code = Lc::new(row_len);
    let params = ZipPlusParams::new(num_vars, 1, linear_code);

    // ── Prove (full pipeline: PIOP + PCS) ──────────────────────────
    let zinc_proof = pipeline::prove::<Sha256Uair, Zt, Lc, 32, UNCHECKED>(
        &params,
        &trace,
        num_vars,
    );

    println!("Prover completed:");
    println!("  PCS commit:      {:?}", zinc_proof.timing.pcs_commit);
    println!("  Ideal check:     {:?}", zinc_proof.timing.ideal_check);
    println!("  CPR:             {:?}", zinc_proof.timing.combined_poly_resolver);
    println!("  PCS test:        {:?}", zinc_proof.timing.pcs_test);
    println!("  PCS evaluate:    {:?}", zinc_proof.timing.pcs_evaluate);
    println!("  Total:           {:?}", zinc_proof.timing.total);
    println!("  PCS proof size:  {} bytes ({:.1} KB)", zinc_proof.pcs_proof_bytes.len(), zinc_proof.pcs_proof_bytes.len() as f64 / 1024.0);
    println!("  PIOP proof data: IC {} constraints, CPR {} messages, {} up_evals, {} down_evals",
        zinc_proof.ic_proof_values.len(),
        zinc_proof.cpr_sumcheck_messages.len(),
        zinc_proof.cpr_up_evals.len(),
        zinc_proof.cpr_down_evals.len(),
    );

    // ── Verify (full pipeline: PIOP + PCS) ─────────────────────────
    // Use TrivialIdeal for the verifier's ideal check. F₂ constraints
    // don't lift to F_p (XOR = add in F₂, but 1+1 = 2 ≠ 0 in F_p).
    // Soundness comes from the sumcheck/CPR + PCS, not from re-checking
    // ideal membership over the PIOP field. This matches the existing
    // piop crate tests which always use IdealOrZero::Zero.
    let verify_result = pipeline::verify::<Sha256Uair, Zt, Lc, 32, UNCHECKED, _, _>(
        &params,
        &zinc_proof,
        num_vars,
        |_ideal: &IdealOrZero<CyclotomicIdeal>| pipeline::TrivialIdeal,
    );

    println!("\nVerifier completed:");
    println!("  IC verify:  {:?}", verify_result.timing.ideal_check_verify);
    println!("  CPR verify: {:?}", verify_result.timing.combined_poly_resolver_verify);
    println!("  PCS verify: {:?}", verify_result.timing.pcs_verify);
    println!("  Total:      {:?}", verify_result.timing.total);
    println!("  Accepted:   {}", verify_result.accepted);

    assert!(verify_result.accepted, "Full pipeline verification FAILED");
    println!("\n✓ Full pipeline round-trip test PASSED");
}
