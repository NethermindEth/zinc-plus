//! ECDSA full pipeline round-trip: PCS + IC + CPR → verify.
//!
//! Demonstrates end-to-end proving/verification for ECDSA constraints:
//! 1. Commit the BinaryPoly<32> trace via PCS
//! 2. IC₁ on BinaryPoly<32> (0 constraints — trivial pass)
//! 3. Convert trace to DensePolynomial<i64, 1> (evaluate at X=2)
//! 4. IC₂ on i64 (11 assert_zero constraints)
//! 5. PCS test + evaluate
//! 6. Verify IC₁, IC₂, CPRs, and PCS
//!
//! **Note:** Uses the all-zero BinaryPoly<32> trace. The all-zero i64 trace
//! trivially satisfies all 11 constraints. For non-trivial constraint testing,
//! see `ecdsa_ideal_check.rs` which uses the constant-row (1,1,0) witness
//! directly in `DensePolynomial<i64, 1>`.

#![allow(clippy::arithmetic_side_effects)]

use std::marker::PhantomData;

use crypto_bigint::U64;
use crypto_primitives::{
    boolean::Boolean,
    crypto_bigint_int::Int,
    crypto_bigint_uint::Uint,
    FixedSemiring,
};

use zinc_poly::mle::DenseMultilinearExtension;
use zinc_poly::univariate::binary::{
    BinaryPoly, BinaryPolyInnerProduct, BinaryPolyWideningMulByScalar,
};
use zinc_poly::univariate::dense::{DensePolyInnerProduct, DensePolynomial};
use zinc_primality::MillerRabin;
use zinc_transcript::traits::ConstTranscribable;
use zinc_uair::ideal::ImpossibleIdeal;
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

use zinc_ecdsa_uair::{
    EcdsaIdealOverF, EcdsaUair, NUM_COLS,
    convert_trace_bp_to_i64,
};
use zinc_snark::pipeline;

// ─── Type definitions (reused from other pipeline tests) ────────────────────

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
    const NUM_COLUMN_OPENINGS: usize = 64;
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
fn ecdsa_pipeline_round_trip() {
    // Use num_vars=7 (128 rows) to match PCS config (row_len=128, DEPTH=1).
    let num_vars = 7;

    // Generate all-zero BinaryPoly<32> trace (14 columns × 128 rows).
    // All-zero satisfies all 11 ECDSA constraints when converted to i64.
    let trace: Vec<DenseMultilinearExtension<BinaryPoly<32>>> = (0..NUM_COLS)
        .map(|_| {
            DenseMultilinearExtension::from_evaluations_vec(
                num_vars,
                vec![BinaryPoly::from(0u32); 1 << num_vars],
                BinaryPoly::from(0u32),
            )
        })
        .collect();

    // PCS params.
    let row_len = 128;
    let linear_code = Lc::new(row_len);
    let params = ZipPlusParams::new(num_vars, 1, linear_code);

    // ── Prove (dual-ring: 0 BP constraints + 11 i64 constraints) ────
    let proof = pipeline::prove_dual_ring::<
        EcdsaUair,          // U1: BinaryPoly<32> (0 constraints)
        EcdsaUair,          // U2: DensePolynomial<i64, 1> (11 constraints)
        Zt,
        Lc,
        32,                 // D1 (BinaryPoly<32>)
        1,                  // D2 (DensePolynomial<i64, 1>)
        UNCHECKED,
        _,                  // ConvertFn (inferred)
    >(
        &params,
        &trace,
        num_vars,
        convert_trace_bp_to_i64,
    );

    println!("ECDSA pipeline prover completed:");
    println!("  PCS commit:      {:?}", proof.timing.pcs_commit);
    println!("  IC+CPR (both):   {:?}", proof.timing.ideal_check);
    println!("  PCS test:        {:?}", proof.timing.pcs_test);
    println!("  PCS evaluate:    {:?}", proof.timing.pcs_evaluate);
    println!("  Total:           {:?}", proof.timing.total);
    println!(
        "  PCS proof size:  {} bytes ({:.1} KB)",
        proof.pcs_proof_bytes.len(),
        proof.pcs_proof_bytes.len() as f64 / 1024.0,
    );
    println!(
        "  BP IC: {} constraints, i64 IC: {} constraints",
        proof.bp_ic_proof_values.len(),
        proof.qx_ic_proof_values.len(),
    );

    // ── Verify ──────────────────────────────────────────────────────
    let verify_result = pipeline::verify_dual_ring::<
        EcdsaUair,          // U1
        EcdsaUair,          // U2
        Zt,
        Lc,
        32,                 // D1
        1,                  // D2
        UNCHECKED,
        EcdsaIdealOverF,
        _,
    >(
        &params,
        &proof,
        num_vars,
        |ideal: &IdealOrZero<ImpossibleIdeal>| -> EcdsaIdealOverF {
            match ideal {
                IdealOrZero::Zero => EcdsaIdealOverF,
                IdealOrZero::Ideal(_) => panic!("ECDSA has no non-zero ideal constraints"),
            }
        },
    );

    println!("\nECDSA pipeline verifier completed:");
    println!(
        "  IC+CPR verify: {:?}",
        verify_result.timing.ideal_check_verify
    );
    println!("  PCS verify:    {:?}", verify_result.timing.pcs_verify);
    println!("  Total:         {:?}", verify_result.timing.total);
    println!("  Accepted:      {}", verify_result.accepted);

    assert!(
        verify_result.accepted,
        "ECDSA pipeline verification FAILED"
    );
    println!(
        "\n✓ ECDSA pipeline round-trip PASSED (0 F₂[X] + 11 i64 assert_zero constraints)"
    );
}
