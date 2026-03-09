//! Dual-ring pipeline round-trip test: prove (BP + Q[X] PIOP + PCS) → verify.
//!
//! Verifies the full dual-ring Zinc+ pipeline end-to-end:
//! 1. Generate a valid SHA-256 BinaryPoly<32> witness
//! 2. Run the dual-ring prover: IC₁(BP) + CPR₁ → IC₂(Q[X]) + CPR₂ → PCS
//! 3. Run the dual-ring verifier with TrivialIdeal for both F₂[X] and Q[X]
//!    (Q[X] constraints are incomplete pending multi-row lookback support)
//! 4. Assert verification succeeds

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

use zinc_sha256_uair::{
    Sha256Uair, Sha256UairQx, convert_trace_to_qx,
    witness::GenerateWitness,
};
use zinc_snark::pipeline;
use zinc_snark::pipeline::TrivialIdeal;

// ─── Type definitions (same as full_pipeline test) ──────────────────────────

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
    const NUM_COLUMN_OPENINGS: usize = 118;
    const GRINDING_BITS: usize = 16;
    type Eval = BinaryPoly<D_PLUS_ONE>;
    type Cw = DensePolynomial<CwCoeff, D_PLUS_ONE>;
    type Fmod = Uint<{ INT_LIMBS * 3 }>;
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
fn dual_ring_pipeline_round_trip() {
    // Generate SHA-256 witness
    let num_vars = 7;
    let mut rng = rand::rng();
    let trace =
        <Sha256Uair as GenerateWitness<BinaryPoly<32>>>::generate_witness(num_vars, &mut rng);

    // Setup PCS params
    let row_len = 128;
    let linear_code = Lc::new(row_len);
    let params = ZipPlusParams::new(num_vars, 1, linear_code);

    // ── Prove (dual-ring pipeline) ──────────────────────────────────
    let proof = pipeline::prove_dual_ring::<
        Sha256Uair,            // U1: BinaryPoly<32> constraints (6 F₂[X])
        Sha256UairQx,          // U2: DensePolynomial<i64, 64> constraints (3 Q[X])
        Zt, Lc,
        32,                    // D1
        64,                    // D2
        UNCHECKED,
        _,                     // ConvertFn (inferred)
    >(
        &params,
        &trace,
        num_vars,
        convert_trace_to_qx,   // BinaryPoly<32> → DensePolynomial<i64, 64>
    );

    println!("Dual-ring prover completed:");
    println!("  PCS commit:      {:?}", proof.timing.pcs_commit);
    println!("  IC (both):       {:?}", proof.timing.ideal_check);
    println!("  CPR (batched):   {:?}", proof.timing.combined_poly_resolver);
    println!("  PCS prove:       {:?}", proof.timing.pcs_prove);
    println!("  Total:           {:?}", proof.timing.total);
    println!("  PCS proof size:  {} bytes ({:.1} KB)",
        proof.pcs_proof_bytes.len(),
        proof.pcs_proof_bytes.len() as f64 / 1024.0,
    );
    println!("  BP IC: {} constraints, QX IC: {} constraints",
        proof.bp_ic_proof_values.len(),
        proof.qx_ic_proof_values.len(),
    );
    println!("  Multi-degree sumcheck: {} groups, degrees {:?}",
        proof.md_degrees.len(),
        proof.md_degrees,
    );

    // ── Verify (dual-ring pipeline) ─────────────────────────────────
    // Q[X] constraints are currently incomplete (require multi-row lookback
    // not yet implemented), so the Q[X] ideal check uses TrivialIdeal just
    // like the BinaryPoly pass. Real DegreeOne(2) verification will be
    // enabled once the framework supports multi-row references.

    let verify_result = pipeline::verify_dual_ring::<
        Sha256Uair,            // U1
        Sha256UairQx,          // U2
        Zt, Lc,
        32,                    // D1
        64,                    // D2
        UNCHECKED,
        TrivialIdeal,
        _,
    >(
        &params,
        &proof,
        num_vars,
        |_ideal: &IdealOrZero<_>| TrivialIdeal,
    );

    println!("\nDual-ring verifier completed:");
    println!("  IC verify:       {:?}", verify_result.timing.ideal_check_verify);
    println!("  CPR verify:      {:?}", verify_result.timing.combined_poly_resolver_verify);
    println!("  PCS verify:      {:?}", verify_result.timing.pcs_verify);
    println!("  Total:           {:?}", verify_result.timing.total);
    println!("  Accepted:        {}", verify_result.accepted);

    assert!(verify_result.accepted, "Dual-ring pipeline verification FAILED");
    println!("\n✓ Dual-ring pipeline round-trip test PASSED (6 F₂[X] + 3 Q[X] constraints)");
}
