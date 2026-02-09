//! Benchmark: Zip commit to 40 polynomials with 6–11 variables.
//! Uses depth-1 and depth-2 IPRS codes with rate 1/4 over F65537.
//! Each variable count gets a matching IPRS code so the matrix has 1 row.
//!
//! Note: With rate 1/4 IPRS codes over F65537, 32-bit i32 evals × 16-bit
//! twiddles produce intermediates that exceed i64 during butterfly stages.
//! We use Int<2> (128-bit) codewords to avoid overflow.

#![allow(non_local_definitions)]
#![allow(clippy::eq_op, clippy::arithmetic_side_effects, clippy::unwrap_used)]

mod zip_common;

use zinc_primality::MillerRabin;
use zinc_utils::UNCHECKED;
use zinc_utils::inner_product::{MBSInnerProduct, ScalarProduct};
use zip_common::*;

use criterion::{Criterion, criterion_group, criterion_main};
use crypto_bigint::U64;
use crypto_primitives::{crypto_bigint_int::Int, crypto_bigint_uint::Uint};
use zinc_utils::from_ref::FromRef;
use zinc_utils::mul_by_scalar::WideningMulByScalar;
use zip_plus::{
    code::iprs::{IprsCode, PnttConfigF2_16_1_Rate1_4_Base},
    pcs::structs::ZipTypes,
};

const INT_LIMBS: usize = U64::LIMBS;

/// ZipTypes for 32-bit integer evaluations with 128-bit codewords.
struct BenchZipTypes32Bit {}
impl ZipTypes for BenchZipTypes32Bit {
    const NUM_COLUMN_OPENINGS: usize = 200;
    type Eval = i32;
    type Cw = Int<2>; // 128-bit codewords
    type Fmod = Uint<{ INT_LIMBS * 4 }>;
    type PrimeTest = MillerRabin;
    type Chal = i128;
    type Pt = i128;
    type CombR = Int<4>; // Cw × Chal = 128 × 128 → 256 bits
    type Comb = Self::CombR;
    type EvalDotChal = ScalarProduct;
    type CombDotChal = ScalarProduct;
    type ArrCombRDotChal = MBSInnerProduct;
}

/// Widening multiplication: i32 × i64 → Int<2>.
#[derive(Clone, Default)]
struct I32ToInt2WideningMulByScalar;

impl WideningMulByScalar<i32, i64> for I32ToInt2WideningMulByScalar {
    type Output = Int<2>;

    #[allow(clippy::arithmetic_side_effects)]
    fn mul_by_scalar_widen(lhs: &i32, rhs: &i64) -> Self::Output {
        let wide_lhs: Int<2> = Int::<2>::from_ref(lhs);
        let wide_rhs: Int<2> = Int::<2>::from_ref(rhs);
        wide_lhs * wide_rhs
    }
}

type Zt = BenchZipTypes32Bit;
type Mul = I32ToInt2WideningMulByScalar;

// Depth-1 codes: INPUT_LEN = BASE_LEN * 8 = 2^num_vars
type IprsD1V6  = IprsCode<Zt, PnttConfigF2_16_1_Rate1_4_Base<8, 1>, Mul>;
type IprsD1V7  = IprsCode<Zt, PnttConfigF2_16_1_Rate1_4_Base<16, 1>, Mul>;
type IprsD1V8  = IprsCode<Zt, PnttConfigF2_16_1_Rate1_4_Base<32, 1>, Mul>;
type IprsD1V9  = IprsCode<Zt, PnttConfigF2_16_1_Rate1_4_Base<64, 1>, Mul>;
type IprsD1V10 = IprsCode<Zt, PnttConfigF2_16_1_Rate1_4_Base<128, 1>, Mul>;
type IprsD1V11 = IprsCode<Zt, PnttConfigF2_16_1_Rate1_4_Base<256, 1>, Mul>;

// Depth-2 codes: INPUT_LEN = BASE_LEN * 64 = 2^num_vars
type IprsD2V6  = IprsCode<Zt, PnttConfigF2_16_1_Rate1_4_Base<1, 2>, Mul>;
type IprsD2V7  = IprsCode<Zt, PnttConfigF2_16_1_Rate1_4_Base<2, 2>, Mul>;
type IprsD2V8  = IprsCode<Zt, PnttConfigF2_16_1_Rate1_4_Base<4, 2>, Mul>;
type IprsD2V9  = IprsCode<Zt, PnttConfigF2_16_1_Rate1_4_Base<8, 2>, Mul>;
type IprsD2V10 = IprsCode<Zt, PnttConfigF2_16_1_Rate1_4_Base<16, 2>, Mul>;
type IprsD2V11 = IprsCode<Zt, PnttConfigF2_16_1_Rate1_4_Base<32, 2>, Mul>;

fn zip_commit_40_polys(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip Commit 40 Polys");

    commit_n_polys::<Zt, IprsD1V6, 6, 40>(&mut group, "IPRS depth-1 rate-1/4");
    commit_n_polys::<Zt, IprsD1V7, 7, 40>(&mut group, "IPRS depth-1 rate-1/4");
    commit_n_polys::<Zt, IprsD1V8, 8, 40>(&mut group, "IPRS depth-1 rate-1/4");
    commit_n_polys::<Zt, IprsD1V9, 9, 40>(&mut group, "IPRS depth-1 rate-1/4");
    commit_n_polys::<Zt, IprsD1V10, 10, 40>(&mut group, "IPRS depth-1 rate-1/4");
    commit_n_polys::<Zt, IprsD1V11, 11, 40>(&mut group, "IPRS depth-1 rate-1/4");

    commit_n_polys::<Zt, IprsD2V6, 6, 40>(&mut group, "IPRS depth-2 rate-1/4");
    commit_n_polys::<Zt, IprsD2V7, 7, 40>(&mut group, "IPRS depth-2 rate-1/4");
    commit_n_polys::<Zt, IprsD2V8, 8, 40>(&mut group, "IPRS depth-2 rate-1/4");
    commit_n_polys::<Zt, IprsD2V9, 9, 40>(&mut group, "IPRS depth-2 rate-1/4");
    commit_n_polys::<Zt, IprsD2V10, 10, 40>(&mut group, "IPRS depth-2 rate-1/4");
    commit_n_polys::<Zt, IprsD2V11, 11, 40>(&mut group, "IPRS depth-2 rate-1/4");

    group.finish();
}

fn zip_test_40_polys(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip Test 40 Polys");

    test_n_polys::<Zt, IprsD1V6, { UNCHECKED }, 6, 40>(&mut group, "IPRS depth-1 rate-1/4");
    test_n_polys::<Zt, IprsD1V7, { UNCHECKED }, 7, 40>(&mut group, "IPRS depth-1 rate-1/4");
    test_n_polys::<Zt, IprsD1V8, { UNCHECKED }, 8, 40>(&mut group, "IPRS depth-1 rate-1/4");
    test_n_polys::<Zt, IprsD1V9, { UNCHECKED }, 9, 40>(&mut group, "IPRS depth-1 rate-1/4");
    test_n_polys::<Zt, IprsD1V10, { UNCHECKED }, 10, 40>(&mut group, "IPRS depth-1 rate-1/4");
    test_n_polys::<Zt, IprsD1V11, { UNCHECKED }, 11, 40>(&mut group, "IPRS depth-1 rate-1/4");

    test_n_polys::<Zt, IprsD2V6, { UNCHECKED }, 6, 40>(&mut group, "IPRS depth-2 rate-1/4");
    test_n_polys::<Zt, IprsD2V7, { UNCHECKED }, 7, 40>(&mut group, "IPRS depth-2 rate-1/4");
    test_n_polys::<Zt, IprsD2V8, { UNCHECKED }, 8, 40>(&mut group, "IPRS depth-2 rate-1/4");
    test_n_polys::<Zt, IprsD2V9, { UNCHECKED }, 9, 40>(&mut group, "IPRS depth-2 rate-1/4");
    test_n_polys::<Zt, IprsD2V10, { UNCHECKED }, 10, 40>(&mut group, "IPRS depth-2 rate-1/4");
    test_n_polys::<Zt, IprsD2V11, { UNCHECKED }, 11, 40>(&mut group, "IPRS depth-2 rate-1/4");

    group.finish();
}

fn zip_evaluate_40_polys(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip Evaluate 40 Polys");

    evaluate_n_polys::<Zt, IprsD1V6, { UNCHECKED }, 6, 40>(&mut group, "IPRS depth-1 rate-1/4");
    evaluate_n_polys::<Zt, IprsD1V7, { UNCHECKED }, 7, 40>(&mut group, "IPRS depth-1 rate-1/4");
    evaluate_n_polys::<Zt, IprsD1V8, { UNCHECKED }, 8, 40>(&mut group, "IPRS depth-1 rate-1/4");
    evaluate_n_polys::<Zt, IprsD1V9, { UNCHECKED }, 9, 40>(&mut group, "IPRS depth-1 rate-1/4");
    evaluate_n_polys::<Zt, IprsD1V10, { UNCHECKED }, 10, 40>(&mut group, "IPRS depth-1 rate-1/4");
    evaluate_n_polys::<Zt, IprsD1V11, { UNCHECKED }, 11, 40>(&mut group, "IPRS depth-1 rate-1/4");

    evaluate_n_polys::<Zt, IprsD2V6, { UNCHECKED }, 6, 40>(&mut group, "IPRS depth-2 rate-1/4");
    evaluate_n_polys::<Zt, IprsD2V7, { UNCHECKED }, 7, 40>(&mut group, "IPRS depth-2 rate-1/4");
    evaluate_n_polys::<Zt, IprsD2V8, { UNCHECKED }, 8, 40>(&mut group, "IPRS depth-2 rate-1/4");
    evaluate_n_polys::<Zt, IprsD2V9, { UNCHECKED }, 9, 40>(&mut group, "IPRS depth-2 rate-1/4");
    evaluate_n_polys::<Zt, IprsD2V10, { UNCHECKED }, 10, 40>(&mut group, "IPRS depth-2 rate-1/4");
    evaluate_n_polys::<Zt, IprsD2V11, { UNCHECKED }, 11, 40>(&mut group, "IPRS depth-2 rate-1/4");

    group.finish();
}

fn zip_verify_40_polys(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip Verify 40 Polys");

    verify_n_polys::<Zt, IprsD1V6, { UNCHECKED }, 6, 40>(&mut group, "IPRS depth-1 rate-1/4");
    verify_n_polys::<Zt, IprsD1V7, { UNCHECKED }, 7, 40>(&mut group, "IPRS depth-1 rate-1/4");
    verify_n_polys::<Zt, IprsD1V8, { UNCHECKED }, 8, 40>(&mut group, "IPRS depth-1 rate-1/4");
    verify_n_polys::<Zt, IprsD1V9, { UNCHECKED }, 9, 40>(&mut group, "IPRS depth-1 rate-1/4");
    verify_n_polys::<Zt, IprsD1V10, { UNCHECKED }, 10, 40>(&mut group, "IPRS depth-1 rate-1/4");
    verify_n_polys::<Zt, IprsD1V11, { UNCHECKED }, 11, 40>(&mut group, "IPRS depth-1 rate-1/4");

    verify_n_polys::<Zt, IprsD2V6, { UNCHECKED }, 6, 40>(&mut group, "IPRS depth-2 rate-1/4");
    verify_n_polys::<Zt, IprsD2V7, { UNCHECKED }, 7, 40>(&mut group, "IPRS depth-2 rate-1/4");
    verify_n_polys::<Zt, IprsD2V8, { UNCHECKED }, 8, 40>(&mut group, "IPRS depth-2 rate-1/4");
    verify_n_polys::<Zt, IprsD2V9, { UNCHECKED }, 9, 40>(&mut group, "IPRS depth-2 rate-1/4");
    verify_n_polys::<Zt, IprsD2V10, { UNCHECKED }, 10, 40>(&mut group, "IPRS depth-2 rate-1/4");
    verify_n_polys::<Zt, IprsD2V11, { UNCHECKED }, 11, 40>(&mut group, "IPRS depth-2 rate-1/4");

    group.finish();
}

criterion_group!(
    benches,
    zip_commit_40_polys,
    zip_test_40_polys,
    zip_evaluate_40_polys,
    zip_verify_40_polys
);
criterion_main!(benches);