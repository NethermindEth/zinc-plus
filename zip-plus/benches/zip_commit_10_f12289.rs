//! Benchmark: Zip commit to 10 polynomials with 6–10 variables.
//! Uses depth-1 and depth-2 IPRS codes with rate 1/2 and 1/4 over F12289.
//! Each variable count gets a matching IPRS code so the matrix has 1 row.
//!
//! F12289 = 3·2^12 + 1, so the max NTT domain is 2^12 = 4096.
//! Rate 1/4 at 11 vars would need OUTPUT_LEN = 2^13 > 4096, so we cap at 10.
//!
//! Note: With rate 1/4 IPRS codes over F12289, 32-bit i32 evals × 14-bit
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
    code::iprs::{IprsCode, PnttConfigF12289_Rate1_2, PnttConfigF12289_Rate1_4},
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

// --- Rate 1/2 ---

// Depth-1 codes: INPUT_LEN = BASE_LEN * 8 = 2^num_vars
type IprsR12D1V6  = IprsCode<Zt, PnttConfigF12289_Rate1_2<8, 1>, Mul>;
type IprsR12D1V7  = IprsCode<Zt, PnttConfigF12289_Rate1_2<16, 1>, Mul>;
type IprsR12D1V8  = IprsCode<Zt, PnttConfigF12289_Rate1_2<32, 1>, Mul>;
type IprsR12D1V9  = IprsCode<Zt, PnttConfigF12289_Rate1_2<64, 1>, Mul>;
type IprsR12D1V10 = IprsCode<Zt, PnttConfigF12289_Rate1_2<128, 1>, Mul>;

// Depth-2 codes: INPUT_LEN = BASE_LEN * 64 = 2^num_vars
type IprsR12D2V6  = IprsCode<Zt, PnttConfigF12289_Rate1_2<1, 2>, Mul>;
type IprsR12D2V7  = IprsCode<Zt, PnttConfigF12289_Rate1_2<2, 2>, Mul>;
type IprsR12D2V8  = IprsCode<Zt, PnttConfigF12289_Rate1_2<4, 2>, Mul>;
type IprsR12D2V9  = IprsCode<Zt, PnttConfigF12289_Rate1_2<8, 2>, Mul>;
type IprsR12D2V10 = IprsCode<Zt, PnttConfigF12289_Rate1_2<16, 2>, Mul>;

// --- Rate 1/4 ---

// Depth-1 codes: INPUT_LEN = BASE_LEN * 8 = 2^num_vars
type IprsR14D1V6  = IprsCode<Zt, PnttConfigF12289_Rate1_4<8, 1>, Mul>;
type IprsR14D1V7  = IprsCode<Zt, PnttConfigF12289_Rate1_4<16, 1>, Mul>;
type IprsR14D1V8  = IprsCode<Zt, PnttConfigF12289_Rate1_4<32, 1>, Mul>;
type IprsR14D1V9  = IprsCode<Zt, PnttConfigF12289_Rate1_4<64, 1>, Mul>;
type IprsR14D1V10 = IprsCode<Zt, PnttConfigF12289_Rate1_4<128, 1>, Mul>;

// Depth-2 codes: INPUT_LEN = BASE_LEN * 64 = 2^num_vars
type IprsR14D2V6  = IprsCode<Zt, PnttConfigF12289_Rate1_4<1, 2>, Mul>;
type IprsR14D2V7  = IprsCode<Zt, PnttConfigF12289_Rate1_4<2, 2>, Mul>;
type IprsR14D2V8  = IprsCode<Zt, PnttConfigF12289_Rate1_4<4, 2>, Mul>;
type IprsR14D2V9  = IprsCode<Zt, PnttConfigF12289_Rate1_4<8, 2>, Mul>;
type IprsR14D2V10 = IprsCode<Zt, PnttConfigF12289_Rate1_4<16, 2>, Mul>;

fn zip_commit_10_polys_r12d1(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip Commit F12289 10 Polys IPRS-1-1/2-F12289");

    commit_n_polys::<Zt, IprsR12D1V6, 6, 10>(&mut group, "IPRS-1-1/2-F12289");
    commit_n_polys::<Zt, IprsR12D1V7, 7, 10>(&mut group, "IPRS-1-1/2-F12289");
    commit_n_polys::<Zt, IprsR12D1V8, 8, 10>(&mut group, "IPRS-1-1/2-F12289");
    commit_n_polys::<Zt, IprsR12D1V9, 9, 10>(&mut group, "IPRS-1-1/2-F12289");
    commit_n_polys::<Zt, IprsR12D1V10, 10, 10>(&mut group, "IPRS-1-1/2-F12289");

    group.finish();
}

fn zip_commit_10_polys_r12d2(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip Commit F12289 10 Polys IPRS-2-1/2-F12289");

    commit_n_polys::<Zt, IprsR12D2V6, 6, 10>(&mut group, "IPRS-2-1/2-F12289");
    commit_n_polys::<Zt, IprsR12D2V7, 7, 10>(&mut group, "IPRS-2-1/2-F12289");
    commit_n_polys::<Zt, IprsR12D2V8, 8, 10>(&mut group, "IPRS-2-1/2-F12289");
    commit_n_polys::<Zt, IprsR12D2V9, 9, 10>(&mut group, "IPRS-2-1/2-F12289");
    commit_n_polys::<Zt, IprsR12D2V10, 10, 10>(&mut group, "IPRS-2-1/2-F12289");

    group.finish();
}

fn zip_commit_10_polys_r14d1(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip Commit F12289 10 Polys IPRS-1-1/4-F12289");

    commit_n_polys::<Zt, IprsR14D1V6, 6, 10>(&mut group, "IPRS-1-1/4-F12289");
    commit_n_polys::<Zt, IprsR14D1V7, 7, 10>(&mut group, "IPRS-1-1/4-F12289");
    commit_n_polys::<Zt, IprsR14D1V8, 8, 10>(&mut group, "IPRS-1-1/4-F12289");
    commit_n_polys::<Zt, IprsR14D1V9, 9, 10>(&mut group, "IPRS-1-1/4-F12289");
    commit_n_polys::<Zt, IprsR14D1V10, 10, 10>(&mut group, "IPRS-1-1/4-F12289");

    group.finish();
}

fn zip_commit_10_polys_r14d2(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip Commit F12289 10 Polys IPRS-2-1/4-F12289");

    commit_n_polys::<Zt, IprsR14D2V6, 6, 10>(&mut group, "IPRS-2-1/4-F12289");
    commit_n_polys::<Zt, IprsR14D2V7, 7, 10>(&mut group, "IPRS-2-1/4-F12289");
    commit_n_polys::<Zt, IprsR14D2V8, 8, 10>(&mut group, "IPRS-2-1/4-F12289");
    commit_n_polys::<Zt, IprsR14D2V9, 9, 10>(&mut group, "IPRS-2-1/4-F12289");
    commit_n_polys::<Zt, IprsR14D2V10, 10, 10>(&mut group, "IPRS-2-1/4-F12289");

    group.finish();
}

fn zip_test_10_polys_r12d1(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip Test F12289 10 Polys IPRS-1-1/2-F12289");

    test_n_polys::<Zt, IprsR12D1V6, { UNCHECKED }, 6, 10>(&mut group, "IPRS-1-1/2-F12289");
    test_n_polys::<Zt, IprsR12D1V7, { UNCHECKED }, 7, 10>(&mut group, "IPRS-1-1/2-F12289");
    test_n_polys::<Zt, IprsR12D1V8, { UNCHECKED }, 8, 10>(&mut group, "IPRS-1-1/2-F12289");
    test_n_polys::<Zt, IprsR12D1V9, { UNCHECKED }, 9, 10>(&mut group, "IPRS-1-1/2-F12289");
    test_n_polys::<Zt, IprsR12D1V10, { UNCHECKED }, 10, 10>(&mut group, "IPRS-1-1/2-F12289");

    group.finish();
}

fn zip_test_10_polys_r12d2(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip Test F12289 10 Polys IPRS-2-1/2-F12289");

    test_n_polys::<Zt, IprsR12D2V6, { UNCHECKED }, 6, 10>(&mut group, "IPRS-2-1/2-F12289");
    test_n_polys::<Zt, IprsR12D2V7, { UNCHECKED }, 7, 10>(&mut group, "IPRS-2-1/2-F12289");
    test_n_polys::<Zt, IprsR12D2V8, { UNCHECKED }, 8, 10>(&mut group, "IPRS-2-1/2-F12289");
    test_n_polys::<Zt, IprsR12D2V9, { UNCHECKED }, 9, 10>(&mut group, "IPRS-2-1/2-F12289");
    test_n_polys::<Zt, IprsR12D2V10, { UNCHECKED }, 10, 10>(&mut group, "IPRS-2-1/2-F12289");

    group.finish();
}

fn zip_test_10_polys_r14d1(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip Test F12289 10 Polys IPRS-1-1/4-F12289");

    test_n_polys::<Zt, IprsR14D1V6, { UNCHECKED }, 6, 10>(&mut group, "IPRS-1-1/4-F12289");
    test_n_polys::<Zt, IprsR14D1V7, { UNCHECKED }, 7, 10>(&mut group, "IPRS-1-1/4-F12289");
    test_n_polys::<Zt, IprsR14D1V8, { UNCHECKED }, 8, 10>(&mut group, "IPRS-1-1/4-F12289");
    test_n_polys::<Zt, IprsR14D1V9, { UNCHECKED }, 9, 10>(&mut group, "IPRS-1-1/4-F12289");
    test_n_polys::<Zt, IprsR14D1V10, { UNCHECKED }, 10, 10>(&mut group, "IPRS-1-1/4-F12289");

    group.finish();
}

fn zip_test_10_polys_r14d2(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip Test F12289 10 Polys IPRS-2-1/4-F12289");

    test_n_polys::<Zt, IprsR14D2V6, { UNCHECKED }, 6, 10>(&mut group, "IPRS-2-1/4-F12289");
    test_n_polys::<Zt, IprsR14D2V7, { UNCHECKED }, 7, 10>(&mut group, "IPRS-2-1/4-F12289");
    test_n_polys::<Zt, IprsR14D2V8, { UNCHECKED }, 8, 10>(&mut group, "IPRS-2-1/4-F12289");
    test_n_polys::<Zt, IprsR14D2V9, { UNCHECKED }, 9, 10>(&mut group, "IPRS-2-1/4-F12289");
    test_n_polys::<Zt, IprsR14D2V10, { UNCHECKED }, 10, 10>(&mut group, "IPRS-2-1/4-F12289");

    group.finish();
}

fn zip_evaluate_10_polys_r12d1(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip Evaluate F12289 10 Polys IPRS-1-1/2-F12289");

    evaluate_n_polys::<Zt, IprsR12D1V6, { UNCHECKED }, 6, 10>(&mut group, "IPRS-1-1/2-F12289");
    evaluate_n_polys::<Zt, IprsR12D1V7, { UNCHECKED }, 7, 10>(&mut group, "IPRS-1-1/2-F12289");
    evaluate_n_polys::<Zt, IprsR12D1V8, { UNCHECKED }, 8, 10>(&mut group, "IPRS-1-1/2-F12289");
    evaluate_n_polys::<Zt, IprsR12D1V9, { UNCHECKED }, 9, 10>(&mut group, "IPRS-1-1/2-F12289");
    evaluate_n_polys::<Zt, IprsR12D1V10, { UNCHECKED }, 10, 10>(&mut group, "IPRS-1-1/2-F12289");

    group.finish();
}

fn zip_evaluate_10_polys_r12d2(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip Evaluate F12289 10 Polys IPRS-2-1/2-F12289");

    evaluate_n_polys::<Zt, IprsR12D2V6, { UNCHECKED }, 6, 10>(&mut group, "IPRS-2-1/2-F12289");
    evaluate_n_polys::<Zt, IprsR12D2V7, { UNCHECKED }, 7, 10>(&mut group, "IPRS-2-1/2-F12289");
    evaluate_n_polys::<Zt, IprsR12D2V8, { UNCHECKED }, 8, 10>(&mut group, "IPRS-2-1/2-F12289");
    evaluate_n_polys::<Zt, IprsR12D2V9, { UNCHECKED }, 9, 10>(&mut group, "IPRS-2-1/2-F12289");
    evaluate_n_polys::<Zt, IprsR12D2V10, { UNCHECKED }, 10, 10>(&mut group, "IPRS-2-1/2-F12289");

    group.finish();
}

fn zip_evaluate_10_polys_r14d1(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip Evaluate F12289 10 Polys IPRS-1-1/4-F12289");

    evaluate_n_polys::<Zt, IprsR14D1V6, { UNCHECKED }, 6, 10>(&mut group, "IPRS-1-1/4-F12289");
    evaluate_n_polys::<Zt, IprsR14D1V7, { UNCHECKED }, 7, 10>(&mut group, "IPRS-1-1/4-F12289");
    evaluate_n_polys::<Zt, IprsR14D1V8, { UNCHECKED }, 8, 10>(&mut group, "IPRS-1-1/4-F12289");
    evaluate_n_polys::<Zt, IprsR14D1V9, { UNCHECKED }, 9, 10>(&mut group, "IPRS-1-1/4-F12289");
    evaluate_n_polys::<Zt, IprsR14D1V10, { UNCHECKED }, 10, 10>(&mut group, "IPRS-1-1/4-F12289");

    group.finish();
}

fn zip_evaluate_10_polys_r14d2(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip Evaluate F12289 10 Polys IPRS-2-1/4-F12289");

    evaluate_n_polys::<Zt, IprsR14D2V6, { UNCHECKED }, 6, 10>(&mut group, "IPRS-2-1/4-F12289");
    evaluate_n_polys::<Zt, IprsR14D2V7, { UNCHECKED }, 7, 10>(&mut group, "IPRS-2-1/4-F12289");
    evaluate_n_polys::<Zt, IprsR14D2V8, { UNCHECKED }, 8, 10>(&mut group, "IPRS-2-1/4-F12289");
    evaluate_n_polys::<Zt, IprsR14D2V9, { UNCHECKED }, 9, 10>(&mut group, "IPRS-2-1/4-F12289");
    evaluate_n_polys::<Zt, IprsR14D2V10, { UNCHECKED }, 10, 10>(&mut group, "IPRS-2-1/4-F12289");

    group.finish();
}

fn zip_verify_10_polys_r12d1(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip Verify F12289 10 Polys IPRS-1-1/2-F12289");

    verify_n_polys::<Zt, IprsR12D1V6, { UNCHECKED }, 6, 10>(&mut group, "IPRS-1-1/2-F12289");
    verify_n_polys::<Zt, IprsR12D1V7, { UNCHECKED }, 7, 10>(&mut group, "IPRS-1-1/2-F12289");
    verify_n_polys::<Zt, IprsR12D1V8, { UNCHECKED }, 8, 10>(&mut group, "IPRS-1-1/2-F12289");
    verify_n_polys::<Zt, IprsR12D1V9, { UNCHECKED }, 9, 10>(&mut group, "IPRS-1-1/2-F12289");
    verify_n_polys::<Zt, IprsR12D1V10, { UNCHECKED }, 10, 10>(&mut group, "IPRS-1-1/2-F12289");

    group.finish();
}

fn zip_verify_10_polys_r12d2(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip Verify F12289 10 Polys IPRS-2-1/2-F12289");

    verify_n_polys::<Zt, IprsR12D2V6, { UNCHECKED }, 6, 10>(&mut group, "IPRS-2-1/2-F12289");
    verify_n_polys::<Zt, IprsR12D2V7, { UNCHECKED }, 7, 10>(&mut group, "IPRS-2-1/2-F12289");
    verify_n_polys::<Zt, IprsR12D2V8, { UNCHECKED }, 8, 10>(&mut group, "IPRS-2-1/2-F12289");
    verify_n_polys::<Zt, IprsR12D2V9, { UNCHECKED }, 9, 10>(&mut group, "IPRS-2-1/2-F12289");
    verify_n_polys::<Zt, IprsR12D2V10, { UNCHECKED }, 10, 10>(&mut group, "IPRS-2-1/2-F12289");

    group.finish();
}

fn zip_verify_10_polys_r14d1(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip Verify F12289 10 Polys IPRS-1-1/4-F12289");

    verify_n_polys::<Zt, IprsR14D1V6, { UNCHECKED }, 6, 10>(&mut group, "IPRS-1-1/4-F12289");
    verify_n_polys::<Zt, IprsR14D1V7, { UNCHECKED }, 7, 10>(&mut group, "IPRS-1-1/4-F12289");
    verify_n_polys::<Zt, IprsR14D1V8, { UNCHECKED }, 8, 10>(&mut group, "IPRS-1-1/4-F12289");
    verify_n_polys::<Zt, IprsR14D1V9, { UNCHECKED }, 9, 10>(&mut group, "IPRS-1-1/4-F12289");
    verify_n_polys::<Zt, IprsR14D1V10, { UNCHECKED }, 10, 10>(&mut group, "IPRS-1-1/4-F12289");

    group.finish();
}

fn zip_verify_10_polys_r14d2(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip Verify F12289 10 Polys IPRS-2-1/4-F12289");

    verify_n_polys::<Zt, IprsR14D2V6, { UNCHECKED }, 6, 10>(&mut group, "IPRS-2-1/4-F12289");
    verify_n_polys::<Zt, IprsR14D2V7, { UNCHECKED }, 7, 10>(&mut group, "IPRS-2-1/4-F12289");
    verify_n_polys::<Zt, IprsR14D2V8, { UNCHECKED }, 8, 10>(&mut group, "IPRS-2-1/4-F12289");
    verify_n_polys::<Zt, IprsR14D2V9, { UNCHECKED }, 9, 10>(&mut group, "IPRS-2-1/4-F12289");
    verify_n_polys::<Zt, IprsR14D2V10, { UNCHECKED }, 10, 10>(&mut group, "IPRS-2-1/4-F12289");

    group.finish();
}

criterion_group!(
    benches,
    zip_commit_10_polys_r12d1,
    zip_commit_10_polys_r12d2,
    zip_commit_10_polys_r14d1,
    zip_commit_10_polys_r14d2,
    zip_test_10_polys_r12d1,
    zip_test_10_polys_r12d2,
    zip_test_10_polys_r14d1,
    zip_test_10_polys_r14d2,
    zip_evaluate_10_polys_r12d1,
    zip_evaluate_10_polys_r12d2,
    zip_evaluate_10_polys_r14d1,
    zip_evaluate_10_polys_r14d2,
    zip_verify_10_polys_r12d1,
    zip_verify_10_polys_r12d2,
    zip_verify_10_polys_r14d1,
    zip_verify_10_polys_r14d2
);
criterion_main!(benches);
