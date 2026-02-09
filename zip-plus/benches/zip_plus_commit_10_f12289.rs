//! Benchmark: Zip+ commit to 10 polynomials with 6–10 variables.
//! Uses depth-1 and depth-2 IPRS codes with rate 1/2 and 1/4 over F12289.
//! Each variable count gets a matching IPRS code so the matrix has 1 row.
//!
//! F12289 = 3·2^12 + 1, so the max NTT domain is 2^12 = 4096.
//! Rate 1/4 at 11 vars would need OUTPUT_LEN = 2^13 > 4096, so we cap at 10.

#![allow(non_local_definitions)]
#![allow(clippy::eq_op, clippy::arithmetic_side_effects, clippy::unwrap_used)]

mod zip_common;

use std::marker::PhantomData;

use zinc_poly::univariate::{
    binary::{BinaryPoly, BinaryPolyInnerProduct, BinaryPolyWideningMulByScalar},
    dense::{DensePolyInnerProduct, DensePolynomial},
};
use zinc_primality::MillerRabin;
use zinc_transcript::traits::ConstTranscribable;
use zinc_utils::{UNCHECKED, from_ref::FromRef, inner_product::MBSInnerProduct, named::Named};
use zip_common::*;

use criterion::{Criterion, criterion_group, criterion_main};
use crypto_bigint::U64;
use crypto_primitives::{
    FixedSemiring, boolean::Boolean, crypto_bigint_int::Int, crypto_bigint_uint::Uint,
};
use zip_plus::{
    code::iprs::{IprsCode, PnttConfigF12289_Rate1_2, PnttConfigF12289_Rate1_4},
    pcs::structs::ZipTypes,
};

const INT_LIMBS: usize = U64::LIMBS;

struct BenchZipPlusTypes<CwCoeff, const D_PLUS_ONE: usize>(PhantomData<CwCoeff>);

impl<CwCoeff, const D_PLUS_ONE: usize> ZipTypes for BenchZipPlusTypes<CwCoeff, D_PLUS_ONE>
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
    const NUM_COLUMN_OPENINGS: usize = 200;
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

type Zt = BenchZipPlusTypes<i64, 32>;
type Mul = BinaryPolyWideningMulByScalar<i64>;

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

fn zip_plus_commit_10_polys_r12d1(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip+ Commit F12289 10 Polys IPRS-1-1/2-F12289");

    commit_n_polys::<Zt, IprsR12D1V6, 6, 10>(&mut group, "IPRS-1-1/2-F12289");
    commit_n_polys::<Zt, IprsR12D1V7, 7, 10>(&mut group, "IPRS-1-1/2-F12289");
    commit_n_polys::<Zt, IprsR12D1V8, 8, 10>(&mut group, "IPRS-1-1/2-F12289");
    commit_n_polys::<Zt, IprsR12D1V9, 9, 10>(&mut group, "IPRS-1-1/2-F12289");
    commit_n_polys::<Zt, IprsR12D1V10, 10, 10>(&mut group, "IPRS-1-1/2-F12289");

    group.finish();
}

fn zip_plus_commit_10_polys_r12d2(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip+ Commit F12289 10 Polys IPRS-2-1/2-F12289");

    commit_n_polys::<Zt, IprsR12D2V6, 6, 10>(&mut group, "IPRS-2-1/2-F12289");
    commit_n_polys::<Zt, IprsR12D2V7, 7, 10>(&mut group, "IPRS-2-1/2-F12289");
    commit_n_polys::<Zt, IprsR12D2V8, 8, 10>(&mut group, "IPRS-2-1/2-F12289");
    commit_n_polys::<Zt, IprsR12D2V9, 9, 10>(&mut group, "IPRS-2-1/2-F12289");
    commit_n_polys::<Zt, IprsR12D2V10, 10, 10>(&mut group, "IPRS-2-1/2-F12289");

    group.finish();
}

fn zip_plus_commit_10_polys_r14d1(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip+ Commit F12289 10 Polys IPRS-1-1/4-F12289");

    commit_n_polys::<Zt, IprsR14D1V6, 6, 10>(&mut group, "IPRS-1-1/4-F12289");
    commit_n_polys::<Zt, IprsR14D1V7, 7, 10>(&mut group, "IPRS-1-1/4-F12289");
    commit_n_polys::<Zt, IprsR14D1V8, 8, 10>(&mut group, "IPRS-1-1/4-F12289");
    commit_n_polys::<Zt, IprsR14D1V9, 9, 10>(&mut group, "IPRS-1-1/4-F12289");
    commit_n_polys::<Zt, IprsR14D1V10, 10, 10>(&mut group, "IPRS-1-1/4-F12289");

    group.finish();
}

fn zip_plus_commit_10_polys_r14d2(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip+ Commit F12289 10 Polys IPRS-2-1/4-F12289");

    commit_n_polys::<Zt, IprsR14D2V6, 6, 10>(&mut group, "IPRS-2-1/4-F12289");
    commit_n_polys::<Zt, IprsR14D2V7, 7, 10>(&mut group, "IPRS-2-1/4-F12289");
    commit_n_polys::<Zt, IprsR14D2V8, 8, 10>(&mut group, "IPRS-2-1/4-F12289");
    commit_n_polys::<Zt, IprsR14D2V9, 9, 10>(&mut group, "IPRS-2-1/4-F12289");
    commit_n_polys::<Zt, IprsR14D2V10, 10, 10>(&mut group, "IPRS-2-1/4-F12289");

    group.finish();
}

fn zip_plus_test_10_polys_r12d1(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip+ Test F12289 10 Polys IPRS-1-1/2-F12289");

    test_n_polys::<Zt, IprsR12D1V6, { UNCHECKED }, 6, 10>(&mut group, "IPRS-1-1/2-F12289");
    test_n_polys::<Zt, IprsR12D1V7, { UNCHECKED }, 7, 10>(&mut group, "IPRS-1-1/2-F12289");
    test_n_polys::<Zt, IprsR12D1V8, { UNCHECKED }, 8, 10>(&mut group, "IPRS-1-1/2-F12289");
    test_n_polys::<Zt, IprsR12D1V9, { UNCHECKED }, 9, 10>(&mut group, "IPRS-1-1/2-F12289");
    test_n_polys::<Zt, IprsR12D1V10, { UNCHECKED }, 10, 10>(&mut group, "IPRS-1-1/2-F12289");

    group.finish();
}

fn zip_plus_test_10_polys_r12d2(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip+ Test F12289 10 Polys IPRS-2-1/2-F12289");

    test_n_polys::<Zt, IprsR12D2V6, { UNCHECKED }, 6, 10>(&mut group, "IPRS-2-1/2-F12289");
    test_n_polys::<Zt, IprsR12D2V7, { UNCHECKED }, 7, 10>(&mut group, "IPRS-2-1/2-F12289");
    test_n_polys::<Zt, IprsR12D2V8, { UNCHECKED }, 8, 10>(&mut group, "IPRS-2-1/2-F12289");
    test_n_polys::<Zt, IprsR12D2V9, { UNCHECKED }, 9, 10>(&mut group, "IPRS-2-1/2-F12289");
    test_n_polys::<Zt, IprsR12D2V10, { UNCHECKED }, 10, 10>(&mut group, "IPRS-2-1/2-F12289");

    group.finish();
}

fn zip_plus_test_10_polys_r14d1(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip+ Test F12289 10 Polys IPRS-1-1/4-F12289");

    test_n_polys::<Zt, IprsR14D1V6, { UNCHECKED }, 6, 10>(&mut group, "IPRS-1-1/4-F12289");
    test_n_polys::<Zt, IprsR14D1V7, { UNCHECKED }, 7, 10>(&mut group, "IPRS-1-1/4-F12289");
    test_n_polys::<Zt, IprsR14D1V8, { UNCHECKED }, 8, 10>(&mut group, "IPRS-1-1/4-F12289");
    test_n_polys::<Zt, IprsR14D1V9, { UNCHECKED }, 9, 10>(&mut group, "IPRS-1-1/4-F12289");
    test_n_polys::<Zt, IprsR14D1V10, { UNCHECKED }, 10, 10>(&mut group, "IPRS-1-1/4-F12289");

    group.finish();
}

fn zip_plus_test_10_polys_r14d2(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip+ Test F12289 10 Polys IPRS-2-1/4-F12289");

    test_n_polys::<Zt, IprsR14D2V6, { UNCHECKED }, 6, 10>(&mut group, "IPRS-2-1/4-F12289");
    test_n_polys::<Zt, IprsR14D2V7, { UNCHECKED }, 7, 10>(&mut group, "IPRS-2-1/4-F12289");
    test_n_polys::<Zt, IprsR14D2V8, { UNCHECKED }, 8, 10>(&mut group, "IPRS-2-1/4-F12289");
    test_n_polys::<Zt, IprsR14D2V9, { UNCHECKED }, 9, 10>(&mut group, "IPRS-2-1/4-F12289");
    test_n_polys::<Zt, IprsR14D2V10, { UNCHECKED }, 10, 10>(&mut group, "IPRS-2-1/4-F12289");

    group.finish();
}

fn zip_plus_evaluate_10_polys_r12d1(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip+ Evaluate F12289 10 Polys IPRS-1-1/2-F12289");

    evaluate_n_polys::<Zt, IprsR12D1V6, { UNCHECKED }, 6, 10>(&mut group, "IPRS-1-1/2-F12289");
    evaluate_n_polys::<Zt, IprsR12D1V7, { UNCHECKED }, 7, 10>(&mut group, "IPRS-1-1/2-F12289");
    evaluate_n_polys::<Zt, IprsR12D1V8, { UNCHECKED }, 8, 10>(&mut group, "IPRS-1-1/2-F12289");
    evaluate_n_polys::<Zt, IprsR12D1V9, { UNCHECKED }, 9, 10>(&mut group, "IPRS-1-1/2-F12289");
    evaluate_n_polys::<Zt, IprsR12D1V10, { UNCHECKED }, 10, 10>(&mut group, "IPRS-1-1/2-F12289");

    group.finish();
}

fn zip_plus_evaluate_10_polys_r12d2(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip+ Evaluate F12289 10 Polys IPRS-2-1/2-F12289");

    evaluate_n_polys::<Zt, IprsR12D2V6, { UNCHECKED }, 6, 10>(&mut group, "IPRS-2-1/2-F12289");
    evaluate_n_polys::<Zt, IprsR12D2V7, { UNCHECKED }, 7, 10>(&mut group, "IPRS-2-1/2-F12289");
    evaluate_n_polys::<Zt, IprsR12D2V8, { UNCHECKED }, 8, 10>(&mut group, "IPRS-2-1/2-F12289");
    evaluate_n_polys::<Zt, IprsR12D2V9, { UNCHECKED }, 9, 10>(&mut group, "IPRS-2-1/2-F12289");
    evaluate_n_polys::<Zt, IprsR12D2V10, { UNCHECKED }, 10, 10>(&mut group, "IPRS-2-1/2-F12289");

    group.finish();
}

fn zip_plus_evaluate_10_polys_r14d1(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip+ Evaluate F12289 10 Polys IPRS-1-1/4-F12289");

    evaluate_n_polys::<Zt, IprsR14D1V6, { UNCHECKED }, 6, 10>(&mut group, "IPRS-1-1/4-F12289");
    evaluate_n_polys::<Zt, IprsR14D1V7, { UNCHECKED }, 7, 10>(&mut group, "IPRS-1-1/4-F12289");
    evaluate_n_polys::<Zt, IprsR14D1V8, { UNCHECKED }, 8, 10>(&mut group, "IPRS-1-1/4-F12289");
    evaluate_n_polys::<Zt, IprsR14D1V9, { UNCHECKED }, 9, 10>(&mut group, "IPRS-1-1/4-F12289");
    evaluate_n_polys::<Zt, IprsR14D1V10, { UNCHECKED }, 10, 10>(&mut group, "IPRS-1-1/4-F12289");

    group.finish();
}

fn zip_plus_evaluate_10_polys_r14d2(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip+ Evaluate F12289 10 Polys IPRS-2-1/4-F12289");

    evaluate_n_polys::<Zt, IprsR14D2V6, { UNCHECKED }, 6, 10>(&mut group, "IPRS-2-1/4-F12289");
    evaluate_n_polys::<Zt, IprsR14D2V7, { UNCHECKED }, 7, 10>(&mut group, "IPRS-2-1/4-F12289");
    evaluate_n_polys::<Zt, IprsR14D2V8, { UNCHECKED }, 8, 10>(&mut group, "IPRS-2-1/4-F12289");
    evaluate_n_polys::<Zt, IprsR14D2V9, { UNCHECKED }, 9, 10>(&mut group, "IPRS-2-1/4-F12289");
    evaluate_n_polys::<Zt, IprsR14D2V10, { UNCHECKED }, 10, 10>(&mut group, "IPRS-2-1/4-F12289");

    group.finish();
}

fn zip_plus_verify_10_polys_r12d1(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip+ Verify F12289 10 Polys IPRS-1-1/2-F12289");

    verify_n_polys::<Zt, IprsR12D1V6, { UNCHECKED }, 6, 10>(&mut group, "IPRS-1-1/2-F12289");
    verify_n_polys::<Zt, IprsR12D1V7, { UNCHECKED }, 7, 10>(&mut group, "IPRS-1-1/2-F12289");
    verify_n_polys::<Zt, IprsR12D1V8, { UNCHECKED }, 8, 10>(&mut group, "IPRS-1-1/2-F12289");
    verify_n_polys::<Zt, IprsR12D1V9, { UNCHECKED }, 9, 10>(&mut group, "IPRS-1-1/2-F12289");
    verify_n_polys::<Zt, IprsR12D1V10, { UNCHECKED }, 10, 10>(&mut group, "IPRS-1-1/2-F12289");

    group.finish();
}

fn zip_plus_verify_10_polys_r12d2(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip+ Verify F12289 10 Polys IPRS-2-1/2-F12289");

    verify_n_polys::<Zt, IprsR12D2V6, { UNCHECKED }, 6, 10>(&mut group, "IPRS-2-1/2-F12289");
    verify_n_polys::<Zt, IprsR12D2V7, { UNCHECKED }, 7, 10>(&mut group, "IPRS-2-1/2-F12289");
    verify_n_polys::<Zt, IprsR12D2V8, { UNCHECKED }, 8, 10>(&mut group, "IPRS-2-1/2-F12289");
    verify_n_polys::<Zt, IprsR12D2V9, { UNCHECKED }, 9, 10>(&mut group, "IPRS-2-1/2-F12289");
    verify_n_polys::<Zt, IprsR12D2V10, { UNCHECKED }, 10, 10>(&mut group, "IPRS-2-1/2-F12289");

    group.finish();
}

fn zip_plus_verify_10_polys_r14d1(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip+ Verify F12289 10 Polys IPRS-1-1/4-F12289");

    verify_n_polys::<Zt, IprsR14D1V6, { UNCHECKED }, 6, 10>(&mut group, "IPRS-1-1/4-F12289");
    verify_n_polys::<Zt, IprsR14D1V7, { UNCHECKED }, 7, 10>(&mut group, "IPRS-1-1/4-F12289");
    verify_n_polys::<Zt, IprsR14D1V8, { UNCHECKED }, 8, 10>(&mut group, "IPRS-1-1/4-F12289");
    verify_n_polys::<Zt, IprsR14D1V9, { UNCHECKED }, 9, 10>(&mut group, "IPRS-1-1/4-F12289");
    verify_n_polys::<Zt, IprsR14D1V10, { UNCHECKED }, 10, 10>(&mut group, "IPRS-1-1/4-F12289");

    group.finish();
}

fn zip_plus_verify_10_polys_r14d2(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip+ Verify F12289 10 Polys IPRS-2-1/4-F12289");

    verify_n_polys::<Zt, IprsR14D2V6, { UNCHECKED }, 6, 10>(&mut group, "IPRS-2-1/4-F12289");
    verify_n_polys::<Zt, IprsR14D2V7, { UNCHECKED }, 7, 10>(&mut group, "IPRS-2-1/4-F12289");
    verify_n_polys::<Zt, IprsR14D2V8, { UNCHECKED }, 8, 10>(&mut group, "IPRS-2-1/4-F12289");
    verify_n_polys::<Zt, IprsR14D2V9, { UNCHECKED }, 9, 10>(&mut group, "IPRS-2-1/4-F12289");
    verify_n_polys::<Zt, IprsR14D2V10, { UNCHECKED }, 10, 10>(&mut group, "IPRS-2-1/4-F12289");

    group.finish();
}

criterion_group!(
    benches,
    zip_plus_commit_10_polys_r12d1,
    zip_plus_commit_10_polys_r12d2,
    zip_plus_commit_10_polys_r14d1,
    zip_plus_commit_10_polys_r14d2,
    zip_plus_test_10_polys_r12d1,
    zip_plus_test_10_polys_r12d2,
    zip_plus_test_10_polys_r14d1,
    zip_plus_test_10_polys_r14d2,
    zip_plus_evaluate_10_polys_r12d1,
    zip_plus_evaluate_10_polys_r12d2,
    zip_plus_evaluate_10_polys_r14d1,
    zip_plus_evaluate_10_polys_r14d2,
    zip_plus_verify_10_polys_r12d1,
    zip_plus_verify_10_polys_r12d2,
    zip_plus_verify_10_polys_r14d1,
    zip_plus_verify_10_polys_r14d2
);
criterion_main!(benches);
