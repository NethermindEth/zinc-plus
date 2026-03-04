//! Public-column round-trip test: prove → verify with verifier-computed MLE evals.
//!
//! Uses `PublicColumnTestUair` which has two binary-poly columns:
//! - Column 0: private witness
//! - Column 1: public input (verifier recomputes the MLE evaluation)
//!
//! The single constraint is `a - b = 0` (both columns must be equal).

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

use zinc_test_uair::{GenerateMultiTypeWitness, PublicColumnTestUair, PublicShiftTestUair};
use zinc_snark::pipeline;

// ─── Type definitions ───────────────────────────────────────────────────────

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
    const NUM_COLUMN_OPENINGS: usize = 131;
    const GRINDING_BITS: usize = 8;
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
fn public_column_round_trip() {
    let num_vars = 7; // 2^7 = 128 rows (matches PnttConfigF2_16R4B16<1>)
    let mut rng = rand::rng();
    let (bp_cols, _arb_cols, _int_cols) =
        PublicColumnTestUair::generate_witness(num_vars, &mut rng);

    // bp_cols[0] = private column (a), bp_cols[1] = public column (b).
    // Both are identical (satisfying a = b).
    assert_eq!(bp_cols.len(), 2);

    // PCS setup
    let row_len = 128; // R4B16 DEPTH=1
    let linear_code = Lc::new(row_len);
    let params = ZipPlusParams::new(num_vars, 1, linear_code);

    // ── Prove ───────────────────────────────────────────────────────
    let zinc_proof = pipeline::prove::<PublicColumnTestUair, Zt, Lc, 32, UNCHECKED>(
        &params,
        &bp_cols,
        num_vars,
        &[],
    );

    // The proof should contain only 1 up_eval (column 0), not 2,
    // because column 1 is public and was stripped by the prover.
    println!("cpr_up_evals count: {}", zinc_proof.cpr_up_evals.len());

    // ── Verify ──────────────────────────────────────────────────────
    // Pass column 1 as public column data so the verifier can
    // recompute its MLE evaluation.
    let public_column_data = &[bp_cols[1].clone()];

    let verify_result = pipeline::verify::<PublicColumnTestUair, Zt, Lc, 32, UNCHECKED, _, _>(
        &params,
        &zinc_proof,
        num_vars,
        |_ideal: &IdealOrZero<ImpossibleIdeal>| pipeline::TrivialIdeal,
        public_column_data,
    );

    println!("Accepted: {}", verify_result.accepted);
    assert!(verify_result.accepted, "Public-column pipeline verification FAILED");
    println!("✓ Public-column round-trip test PASSED");
}

/// Round-trip test for a UAIR whose **only shift** sources a public column.
///
/// `PublicShiftTestUair`:
///  - Column 0 (a): private, Column 1 (b): public.
///  - Shift: left-shift-by-1 of column 1.
///  - Constraint: `a = shift_1(b)`.
///
/// The prover must NOT include the shift v_final (MLE eval of column 1
/// at the shift sumcheck challenge point) in the proof; the verifier
/// must recompute it from the public column data.
#[test]
fn public_shift_column_round_trip() {
    let num_vars = 7;
    let mut rng = rand::rng();
    let (bp_cols, _arb_cols, _int_cols) =
        PublicShiftTestUair::generate_witness(num_vars, &mut rng);

    assert_eq!(bp_cols.len(), 2);

    let row_len = 128;
    let linear_code = Lc::new(row_len);
    let params = ZipPlusParams::new(num_vars, 1, linear_code);

    // ── Prove ───────────────────────────────────────────────────────
    let zinc_proof = pipeline::prove::<PublicShiftTestUair, Zt, Lc, 32, UNCHECKED>(
        &params,
        &bp_cols,
        num_vars,
        &[],
    );

    // Shift v_finals for public source columns should have been stripped.
    // The UAIR has 1 shift whose source_col = 1 (public), so the proof
    // should contain 0 shift v_finals.
    println!("shift_sumcheck v_finals count in proof: {}", zinc_proof.shift_sumcheck.as_ref().map_or(0, |s| s.v_finals.len()));

    // ── Verify ──────────────────────────────────────────────────────
    let public_column_data = &[bp_cols[1].clone()];

    let verify_result = pipeline::verify::<PublicShiftTestUair, Zt, Lc, 32, UNCHECKED, _, _>(
        &params,
        &zinc_proof,
        num_vars,
        |_ideal: &IdealOrZero<ImpossibleIdeal>| pipeline::TrivialIdeal,
        public_column_data,
    );

    println!("Accepted: {}", verify_result.accepted);
    assert!(verify_result.accepted, "Public-shift-column pipeline verification FAILED");
    println!("✓ Public-shift-column round-trip test PASSED");
}
