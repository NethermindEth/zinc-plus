//! Folded pipeline round-trip test: prove (PIOP + folding + PCS) → verify.
//!
//! This test verifies the folded Zinc+ pipeline end-to-end:
//! 1. Generate a valid SHA-256 witness with BinaryPoly<32> columns
//! 2. Run the folded prover:
//!    a. Split BinaryPoly<32> → BinaryPoly<16> (halved columns)
//!    b. Commit split columns via PCS (smaller codewords)
//!    c. IdealCheck → CPR on original trace
//!    d. Folding protocol reduces eval claims to split columns
//!    e. PCS prove on split columns at extended point
//! 3. Run the folded verifier
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
use zinc_uair::Uair;
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
        iprs::{IprsCode, PnttConfigF2_16R4B32},
    },
    pcs::structs::{ZipPlusParams, ZipTypes},
};

use zinc_sha256_uair::{CyclotomicIdeal, Sha256Uair, witness::GenerateWitness};
use zinc_snark::pipeline;

// ─── Type definitions ───────────────────────────────────────────────────────

const INT_LIMBS: usize = U64::LIMBS;

/// ZipTypes for BinaryPoly<D> — same as the normal Sha256ZipTypes.
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

/// PCS types for the folded (half-width) columns: BinaryPoly<16>.
/// Uses PnttConfigF2_16R4B32<1>: INPUT_LEN = 32 × 8 = 256 (doubled row_len),
/// OUTPUT_LEN = 128 × 8 = 1024 (rate 1/4, same as original).
type FoldedZt = TestZipTypes<i64, 16>;
type FoldedLc = IprsCode<FoldedZt, PnttConfigF2_16R4B32<1>, BinaryPolyWideningMulByScalar<i64>, UNCHECKED>;

#[test]
fn folded_pipeline_round_trip() {
    // Generate SHA-256 witness (poly_size = 2^7 = 128)
    let num_vars = 7;
    let mut rng = rand::rng();
    let trace = <Sha256Uair as GenerateWitness<BinaryPoly<32>>>::generate_witness(num_vars, &mut rng);

    // Setup PCS params for the *folded* columns (BinaryPoly<16>).
    // The split columns have num_vars + 1 variables (2^{nv+1} evaluations).
    // Double row_len (256) so num_rows = 2^{nv+1} / 256 = 1, matching the
    // original. BinaryPoly<16> codeword elements (half-width) then yield
    // 2x smaller column openings.
    let folded_num_vars = num_vars + 1; // 8
    let row_len = 256; // doubled: 2^8
    let num_rows = (1usize << folded_num_vars) / row_len; // 256 / 256 = 1
    let linear_code = FoldedLc::new(row_len);
    let params = ZipPlusParams::new(folded_num_vars, num_rows, linear_code);

    // ── Prove (folded pipeline: PIOP + folding + PCS) ──────────────
    let folded_proof = pipeline::prove_classic_logup_folded::<
        Sha256Uair,
        FoldedZt,
        FoldedLc,
        32,   // D (original width)
        16,   // HALF_D (folded width)
        UNCHECKED,
    >(
        &params,
        &trace,
        num_vars,
        &[],  // lookup_specs
        &[],  // affine_lookup_specs
    );

    println!("Folded prover completed:");
    println!("  PCS commit:      {:?}", folded_proof.timing.pcs_commit);
    println!("  Ideal check:     {:?}", folded_proof.timing.ideal_check);
    println!("  CPR:             {:?}", folded_proof.timing.combined_poly_resolver);
    println!("  PCS prove:       {:?}", folded_proof.timing.pcs_prove);
    println!("  Total:           {:?}", folded_proof.timing.total);
    println!("  PCS proof size:  {} bytes ({:.1} KB)",
        folded_proof.pcs_proof_bytes.len(),
        folded_proof.pcs_proof_bytes.len() as f64 / 1024.0,
    );
    println!("  Folding data:    {} c1s, {} c2s",
        folded_proof.folding_c1s_bytes.len(),
        folded_proof.folding_c2s_bytes.len(),
    );

    // ── Verify (folded pipeline: PIOP + folding + PCS) ─────────────
    let sha_sig = Sha256Uair::signature();
    let sha_public_cols: Vec<_> = sha_sig.public_columns.iter()
        .map(|&i| trace[i].clone())
        .collect();

    let verify_result = pipeline::verify_classic_logup_folded::<
        Sha256Uair,
        FoldedZt,
        FoldedLc,
        32,   // D
        16,   // HALF_D
        UNCHECKED,
        _,
        _,
    >(
        &params,
        &folded_proof,
        num_vars,
        |_ideal: &IdealOrZero<CyclotomicIdeal>| pipeline::TrivialIdeal,
        &sha_public_cols,
    );

    println!("\nFolded verifier completed:");
    println!("  IC verify:  {:?}", verify_result.timing.ideal_check_verify);
    println!("  CPR verify: {:?}", verify_result.timing.combined_poly_resolver_verify);
    println!("  PCS verify: {:?}", verify_result.timing.pcs_verify);
    println!("  Total:      {:?}", verify_result.timing.total);
    println!("  Accepted:   {}", verify_result.accepted);

    assert!(verify_result.accepted, "Folded pipeline verification FAILED");
    println!("\n✓ Folded pipeline round-trip test PASSED");
}
