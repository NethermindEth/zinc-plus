//! ECDSA full pipeline round-trip: PCS + IC + CPR → verify.
//!
//! Demonstrates end-to-end proving/verification for ECDSA constraints
//! using `Int<4>` throughout — the same 256-bit integer type is used as
//! PCS evaluation type AND PIOP constraint ring (single-ring pipeline).
//!
//! 1. Generate an all-zero `Int<4>` trace (14 columns × 128 rows)
//! 2. Commit the trace via PCS (using `EcdsaScalarZipTypes`)
//! 3. Run PIOP: IdealCheck (11 constraints) + CombinedPolyResolver
//! 4. PCS test + evaluate
//! 5. Verify IC, CPR, and PCS
//!
//! All-zero `Int<4>` trivially satisfies all 11 constraints.
//! For non-trivial constraint testing, see `ecdsa_ideal_check.rs`.

#![allow(clippy::arithmetic_side_effects)]

use crypto_bigint::U64;
use crypto_primitives::{
    crypto_bigint_int::Int,
    crypto_bigint_monty::MontyField,
    crypto_bigint_uint::Uint,
};

use zinc_poly::mle::DenseMultilinearExtension;
use zinc_primality::MillerRabin;
use zinc_uair::ideal::ImpossibleIdeal;
use zinc_uair::ideal_collector::IdealOrZero;
use zinc_utils::{
    UNCHECKED,
    inner_product::{MBSInnerProduct, ScalarProduct},
    mul_by_scalar::ScalarWideningMulByScalar,
};
use zip_plus::{
    code::{
        LinearCode,
        iprs::{IprsCode, PnttConfigF2_16R4B16},
    },
    pcs::structs::{ZipPlusParams, ZipTypes},
};

use zinc_ecdsa_uair::{
    EcdsaIdealOverF, EcdsaUairInt, NUM_COLS,
};
use zinc_snark::pipeline;

// ─── Type definitions: Int<4> single-ring PCS configuration ─────────────────

const INT_LIMBS: usize = U64::LIMBS;

/// ZipTypes for ECDSA using `Int<4>` (256-bit) evaluations throughout.
///
/// ECDSA values are field-element scalars — not polynomials — so `Int<4>`
/// is the natural evaluation ring. All PIOP constraints are also expressed
/// in `Int<4>`, making this a **single-ring** pipeline.
struct EcdsaScalarZipTypes;

impl ZipTypes for EcdsaScalarZipTypes {
    const NUM_COLUMN_OPENINGS: usize = 147;
    type Eval = Int<{ INT_LIMBS * 4 }>;          // 256-bit integer
    type Cw = Int<{ INT_LIMBS * 5 }>;            // 320-bit codeword
    type Fmod = Uint<{ INT_LIMBS * 3 }>;         // 192-bit modulus search
    type PrimeTest = MillerRabin;
    type Chal = i128;
    type Pt = i128;
    type CombR = Int<{ INT_LIMBS * 8 }>;         // 512-bit combination ring
    type Comb = Self::CombR;
    type EvalDotChal = ScalarProduct;
    type CombDotChal = ScalarProduct;
    type ArrCombRDotChal = MBSInnerProduct;
}

type Zt = EcdsaScalarZipTypes;
/// 192-bit PCS field — matches pipeline's MontyField<3>.
type PcsF = MontyField<{ U64::LIMBS * 3 }>;
type Lc = IprsCode<Zt, PnttConfigF2_16R4B16<1>, ScalarWideningMulByScalar<Int<{ U64::LIMBS * 5 }>>, UNCHECKED>;

#[test]
fn ecdsa_pipeline_round_trip() {
    // Use num_vars=7 (128 rows) to match PCS config (row_len=128, DEPTH=1).
    let num_vars = 7;

    // Generate all-zero Int<4> trace (14 columns × 128 rows).
    // All-zero trivially satisfies all 11 ECDSA constraints.
    let zero = Int::<{ INT_LIMBS * 4 }>::default();
    let trace: Vec<DenseMultilinearExtension<Int<{ INT_LIMBS * 4 }>>> = (0..NUM_COLS)
        .map(|_| {
            DenseMultilinearExtension::from_evaluations_vec(
                num_vars,
                vec![zero; 1 << num_vars],
                zero,
            )
        })
        .collect();

    // PCS params.
    let row_len = 128;
    let linear_code = Lc::new(row_len);
    let params = ZipPlusParams::new(num_vars, 1, linear_code);

    // ── Prove (single-ring: 11 Int<4> constraints) ──────────────────
    let proof = pipeline::prove_generic::<
        EcdsaUairInt,                           // U: Uair<Scalar=Int<4>>
        Int<{ INT_LIMBS * 4 }>,                 // R = Int<4>
        Zt,
        Lc,
        PcsF,                                   // PCS field (512-bit)
        UNCHECKED,
    >(
        &params,
        &trace,
        num_vars,
        &[],
    );

    println!("ECDSA Int<4> pipeline prover completed:");
    println!("  PCS commit:      {:?}", proof.timing.pcs_commit);
    println!("  Ideal check:     {:?}", proof.timing.ideal_check);
    println!("  CPR:             {:?}", proof.timing.combined_poly_resolver);
    println!("  PCS prove:       {:?}", proof.timing.pcs_prove);
    println!("  Total:           {:?}", proof.timing.total);
    println!(
        "  PCS proof size:  {} bytes ({:.1} KB)",
        proof.pcs_proof_bytes.len(),
        proof.pcs_proof_bytes.len() as f64 / 1024.0,
    );

    // ── Verify ──────────────────────────────────────────────────────
    // Public columns b_1 (col 0) and b_2 (col 1) — the verifier
    // must supply these to reconstruct the full evaluation set.
    let public_column_data: Vec<DenseMultilinearExtension<Int<{ INT_LIMBS * 4 }>>> = vec![
        trace[0].clone(),  // b_1
        trace[1].clone(),  // b_2
    ];

    let verify_result = pipeline::verify_generic::<
        EcdsaUairInt,                           // U: Uair<Scalar=Int<4>>
        Int<{ INT_LIMBS * 4 }>,                 // R = Int<4>
        Zt,
        Lc,
        PcsF,                                   // PCS field (512-bit)
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
        &public_column_data,
    );

    println!("\nECDSA Int<4> pipeline verifier completed:");
    println!(
        "  IC+CPR verify: {:?}",
        verify_result.timing.ideal_check_verify
    );
    println!("  PCS verify:    {:?}", verify_result.timing.pcs_verify);
    println!("  Total:         {:?}", verify_result.timing.total);
    println!("  Accepted:      {}", verify_result.accepted);

    assert!(
        verify_result.accepted,
        "ECDSA Int<4> pipeline verification FAILED"
    );
    println!(
        "\n✓ ECDSA Int<4> single-ring pipeline round-trip PASSED (11 assert_zero constraints)"
    );
}
