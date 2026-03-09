//! Round-trip proof test: prove → serialize → deserialize → verify.
//!
//! This test ensures the complete PCS pipeline produces a valid proof
//! that survives serialization and can be successfully verified.

#![allow(clippy::arithmetic_side_effects)]

use std::marker::PhantomData;

use crypto_bigint::U64;
use crypto_primitives::{
    boolean::Boolean,
    crypto_bigint_int::Int,
    crypto_bigint_uint::Uint,
    FixedSemiring,
};
use crypto_primitives::PrimeField;
use zinc_transcript::traits::Transcript;

use zinc_poly::univariate::binary::{
    BinaryPoly, BinaryPolyInnerProduct, BinaryPolyWideningMulByScalar,
};
use zinc_poly::univariate::dense::{DensePolyInnerProduct, DensePolynomial};
use zinc_primality::MillerRabin;
use zinc_transcript::traits::ConstTranscribable;
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
    pcs::structs::{ZipPlus, ZipPlusParams, ZipTypes},
    pcs::ZipPlusProof,
};

use zinc_sha256_uair::{Sha256Uair, witness::GenerateWitness};
use zinc_snark::pipeline::PiopField;

// ─── Type definitions (must match the benchmark) ────────────────────────────

const INT_LIMBS: usize = U64::LIMBS;
type F = PiopField;

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
fn round_trip_pcs_sha256() {
    // Generate SHA-256 witness (poly_size = 2^7 = 128)
    let num_vars = 7;
    let mut rng = rand::rng();
    let trace = <Sha256Uair as GenerateWitness<BinaryPoly<32>>>::generate_witness(num_vars, &mut rng);

    // Setup PCS params
    let row_len = 128; // R4B16 DEPTH=1
    let linear_code = Lc::new(row_len);
    let params = ZipPlusParams::new(num_vars, 1, linear_code);

    // ── Prove ────────────────────────────────────────────────────────
    let (hint, commitment) = ZipPlus::<Zt, Lc>::commit(&params, &trace)
        .expect("commit failed");
    let pcs_field_cfg = zinc_transcript::KeccakTranscript::default()
        .get_random_field_cfg::<F, <Zt as ZipTypes>::Fmod, <Zt as ZipTypes>::PrimeTest>();
    let point: Vec<F> = vec![F::one_with_cfg(&pcs_field_cfg); num_vars];
    let (eval_f, proof) = ZipPlus::<Zt, Lc>::prove::<F, UNCHECKED>(
        &params, &trace, &point, &hint,
    )
    .expect("prove failed");

    // ── Serialize proof ──────────────────────────────────────────────
    let proof_bytes: Vec<u8> = {
        let tx: zip_plus::pcs_transcript::PcsTranscript = proof.into();
        tx.stream.into_inner()
    };
    println!("Proof size: {} bytes ({:.1} KB)", proof_bytes.len(), proof_bytes.len() as f64 / 1024.0);

    // ── Deserialize and Verify ───────────────────────────────────────
    let point_f: Vec<F> = point.clone();

    let deserialized_proof: ZipPlusProof = {
        let tx = zip_plus::pcs_transcript::PcsTranscript {
            fs_transcript: zinc_transcript::KeccakTranscript::default(),
            stream: std::io::Cursor::new(proof_bytes),
        };
        tx.into()
    };

    let verify_result = ZipPlus::<Zt, Lc>::verify::<F, UNCHECKED>(
        &params,
        &commitment,
        &point_f,
        &eval_f,
        &deserialized_proof,
    );

    assert!(verify_result.is_ok(), "Verification failed: {:?}", verify_result.err());
    println!("Round-trip verification PASSED ✓");
}
