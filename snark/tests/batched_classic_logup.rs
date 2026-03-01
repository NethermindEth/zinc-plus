//! Batched classic LogUp round-trip test.
//!
//! Exercises the `prove_classic_logup` → `verify` pipeline where the CPR
//! and lookup sumchecks are batched into a single multi-degree sumcheck.

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
        iprs::{IprsCode, PnttConfigF2_16R4B16},
    },
    pcs::structs::{ZipPlusParams, ZipTypes},
};

use zinc_piop::lookup::{LookupColumnSpec, LookupTableType, AffineLookupSpec};
use zinc_sha256_uair::{CyclotomicIdeal, Sha256Uair, witness::GenerateWitness};
use zinc_sha256_uair::{
    COL_E_HAT, COL_E_TM1, COL_CH_EF_HAT, COL_E_TM2, COL_CH_NEG_EG_HAT,
    COL_A_HAT, COL_A_TM1, COL_A_TM2, COL_MAJ_HAT,
};
use zinc_snark::pipeline;

// ─── Type definitions (from full_pipeline.rs) ───────────────────────────────

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

fn sha256_lookup_specs() -> Vec<LookupColumnSpec> {
    // First 10 bitpoly columns have lookup constraints (BitPoly(32)).
    (0..10)
        .map(|i| LookupColumnSpec {
            column_index: i,
            table_type: LookupTableType::BitPoly { width: 32 },
        })
        .collect()
}

/// Affine-combination lookup specs for Ch and Maj (3 lookups).
///
/// Each spec checks that an affine combination of trace columns is a
/// valid BitPoly{32}, which enforces the correctness of the AND/Maj
/// operations without F₂ multiplication gates.
fn sha256_affine_lookup_specs() -> Vec<AffineLookupSpec> {
    let bp32 = LookupTableType::BitPoly { width: 32 };
    vec![
        // Ch lookup 1: ê[t] + ê[t−1] − 2·ch_ef[t] ∈ BitPoly{32}
        AffineLookupSpec {
            terms: vec![
                (COL_E_HAT, 1),
                (COL_E_TM1, 1),
                (COL_CH_EF_HAT, -2),
            ],
            constant_offset_bits: 0,
            table_type: bp32.clone(),
        },
        // Ch lookup 2: (1_w − ê[t]) + ê[t−2] − 2·ch_¬e,g[t] ∈ BitPoly{32}
        AffineLookupSpec {
            terms: vec![
                (COL_E_HAT, -1),
                (COL_E_TM2, 1),
                (COL_CH_NEG_EG_HAT, -2),
            ],
            constant_offset_bits: 0xFFFF_FFFF,
            table_type: bp32.clone(),
        },
        // Maj lookup: â[t] + â[t−1] + â[t−2] − 2·Maj[t] ∈ BitPoly{32}
        AffineLookupSpec {
            terms: vec![
                (COL_A_HAT, 1),
                (COL_A_TM1, 1),
                (COL_A_TM2, 1),
                (COL_MAJ_HAT, -2),
            ],
            constant_offset_bits: 0,
            table_type: bp32,
        },
    ]
}

#[test]
fn batched_classic_logup_round_trip() {
    // Generate SHA-256 witness (poly_size = 2^7 = 128).
    let num_vars = 7;
    let mut rng = rand::rng();
    let trace = <Sha256Uair as GenerateWitness<BinaryPoly<32>>>::generate_witness(num_vars, &mut rng);

    // Setup PCS params.
    let row_len = 128;
    let linear_code = Lc::new(row_len);
    let params = ZipPlusParams::new(num_vars, 1, linear_code);

    let lookup_specs = sha256_lookup_specs();
    let affine_specs = sha256_affine_lookup_specs();

    // ── Prove (batched classic LogUp) ──────────────────────────────
    let zinc_proof = pipeline::prove_classic_logup::<Sha256Uair, Zt, Lc, 32, UNCHECKED>(
        &params,
        &trace,
        num_vars,
        &lookup_specs,
        &affine_specs,
    );

    println!("Batched classic LogUp prover completed:");
    println!("  PCS commit:  {:?}", zinc_proof.timing.pcs_commit);
    println!("  IC:          {:?}", zinc_proof.timing.ideal_check);
    println!("  CPR+Lookup:  {:?}", zinc_proof.timing.combined_poly_resolver);
    println!("  PCS prove:   {:?}", zinc_proof.timing.pcs_prove);
    println!("  Total:       {:?}", zinc_proof.timing.total);
    println!("  Proof size:  {} bytes ({:.1} KB)", zinc_proof.pcs_proof_bytes.len(), zinc_proof.pcs_proof_bytes.len() as f64 / 1024.0);

    // Check that the proof uses the BatchedClassic variant.
    assert!(
        matches!(zinc_proof.lookup_proof, Some(pipeline::LookupProofData::BatchedClassic(_))),
        "Expected BatchedClassic proof variant"
    );

    // ── Verify ─────────────────────────────────────────────────────
    // Public columns from the UAIR signature.
    let sha_sig = Sha256Uair::signature();
    let sha_public_cols: Vec<_> = sha_sig.public_columns.iter()
        .map(|&i| trace[i].clone())
        .collect();

    let verify_result = pipeline::verify::<Sha256Uair, Zt, Lc, 32, UNCHECKED, _, _>(
        &params,
        &zinc_proof,
        num_vars,
        |_ideal: &IdealOrZero<CyclotomicIdeal>| pipeline::TrivialIdeal,
        &sha_public_cols,
    );

    println!("\nBatched classic LogUp verifier completed:");
    println!("  IC+CPR verify: {:?}", verify_result.timing.combined_poly_resolver_verify);
    println!("  PCS verify:    {:?}", verify_result.timing.pcs_verify);
    println!("  Total:         {:?}", verify_result.timing.total);
    println!("  Accepted:      {}", verify_result.accepted);

    assert!(verify_result.accepted, "Batched classic LogUp verification FAILED");
    println!("\n✓ Batched classic LogUp round-trip test PASSED");
}
