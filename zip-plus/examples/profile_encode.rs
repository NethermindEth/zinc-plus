//! Profiling harness for encode_rows with IPRS codes
//! Run with: cargo run --example profile_encode --release --features "asm simd parallel pntt-timing" -p zip-plus
#![allow(non_local_definitions)]

use std::marker::PhantomData;
use std::hint::black_box;

use zinc_poly::univariate::{
    binary::{BinaryPoly, BinaryPolyInnerProduct, BinaryPolyWideningMulByScalar},
    dense::{DensePolyInnerProduct, DensePolynomial},
};
use zinc_poly::mle::{DenseMultilinearExtension, MultilinearExtensionRand};
use zinc_primality::MillerRabin;
use zinc_transcript::traits::ConstTranscribable;
use zinc_utils::{from_ref::FromRef, inner_product::MBSInnerProduct, named::Named};

use crypto_bigint::U64;
use crypto_primitives::{
    FixedSemiring, boolean::Boolean, crypto_bigint_int::Int, crypto_bigint_uint::Uint,
};
use zip_plus::{
    code::LinearCode,
    code::iprs::{IprsCode, PnttConfigF2_16_1_Depth2_Rate1_2},
    pcs::structs::{ZipPlus, ZipTypes},
};
#[cfg(feature = "pntt-timing")]
use zip_plus::code::iprs::{reset_timing, print_timing};

use rand::prelude::*;

const INT_LIMBS: usize = U64::LIMBS;

struct ProfileZipPlusTypes<CwCoeff, const D_PLUS_ONE: usize>(PhantomData<CwCoeff>);

impl<CwCoeff, const D_PLUS_ONE: usize> ZipTypes for ProfileZipPlusTypes<CwCoeff, D_PLUS_ONE>
where
    CwCoeff: ConstTranscribable
        + Copy
        + Default
        + FromRef<Boolean>
        + Named
        + FixedSemiring
        + Send
        + Sync,
    Int<5>: FromRef<CwCoeff>,
{
    const NUM_COLUMN_OPENINGS: usize = 147;
    type Eval = BinaryPoly<D_PLUS_ONE>;
    type Cw = DensePolynomial<CwCoeff, D_PLUS_ONE>;
    type Fmod = Uint<{ INT_LIMBS * 4 }>;
    type PrimeTest = MillerRabin;
    type Chal = i128;
    type Pt = i128;
    type CombR = Int<{ INT_LIMBS * 5 }>;
    type Comb = DensePolynomial<Self::CombR, D_PLUS_ONE>;
    type EvalDotChal = BinaryPolyInnerProduct<Self::Chal, D_PLUS_ONE>;
    type CombDotChal =
        DensePolyInnerProduct<Self::CombR, Self::Chal, Self::CombR, MBSInnerProduct, D_PLUS_ONE>;
    type ArrCombRDotChal = MBSInnerProduct;
}

type ProfileIprsCode = IprsCode<
    ProfileZipPlusTypes<i64, 32>,
    PnttConfigF2_16_1_Depth2_Rate1_2,
    BinaryPolyWideningMulByScalar<i64>,
>;

// Prevent inlining to make profiling clearer
#[inline(never)]
fn do_encode_rows(
    params: &zip_plus::pcs::structs::ZipPlusParams<ProfileZipPlusTypes<i64, 32>, ProfileIprsCode>,
    row_len: usize,
    poly: &DenseMultilinearExtension<BinaryPoly<32>>,
) -> crypto_primitives::DenseRowMatrix<DensePolynomial<i64, 32>> {
    ZipPlus::encode_rows(params, row_len, poly)
}

fn main() {
    const P: usize = 16; // 2^16 = 65536 elements
    const ITERATIONS: usize = 100;
    
    let poly_size = 1 << P;
    let linear_code = ProfileIprsCode::new(poly_size);
    let params = ZipPlus::setup(poly_size, linear_code);
    let row_len = params.linear_code.row_len();
    let rows = poly_size / row_len;
    
    println!("Profiling encode_rows with IPRS codes");
    println!("  poly_size: 2^{} = {}", P, poly_size);
    println!("  matrix: {}x{}", rows, row_len);
    println!("  row_len (INPUT_LEN): {}", row_len);
    println!("  output per row (OUTPUT_LEN): {}", row_len * 2); // rate 1/2
    println!("  iterations: {}", ITERATIONS);
    println!();
    
    let mut rng = ThreadRng::default();
    let poly = DenseMultilinearExtension::<BinaryPoly<32>>::rand(P, &mut rng);
    
    // Warm up
    println!("Warming up...");
    for _ in 0..5 {
        let cw = do_encode_rows(&params, row_len, &poly);
        black_box(cw);
    }
    
    // Reset timing counters after warmup
    #[cfg(feature = "pntt-timing")]
    reset_timing();
    
    println!("Starting timed run...");
    let start = std::time::Instant::now();
    
    for i in 0..ITERATIONS {
        let cw = do_encode_rows(&params, row_len, &poly);
        black_box(cw);
        if (i + 1) % 20 == 0 {
            println!("  completed {} iterations", i + 1);
        }
    }
    
    let elapsed = start.elapsed();
    println!();
    println!("Wall clock time: {:?}", elapsed);
    println!("Average per encode_rows call: {:?}", elapsed / ITERATIONS as u32);
    println!("Average per row: {:?}", elapsed / (ITERATIONS * rows) as u32);
    
    // Print PNTT timing breakdown
    #[cfg(feature = "pntt-timing")]
    print_timing();
    
    #[cfg(not(feature = "pntt-timing"))]
    println!("\nNote: Run with --features pntt-timing for detailed PNTT breakdown");
}
