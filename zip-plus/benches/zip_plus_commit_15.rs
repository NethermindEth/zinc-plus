//! Benchmark: Zip+ commit to 15 polynomials with 6–11 variables.
//! Uses depth-1 and depth-2 IPRS codes with rate 1/4 over F65537.
//! Each variable count gets a matching IPRS code so the matrix has 1 row.

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
    code::iprs::{IprsCode, PnttConfigF2_16_1_Rate1_4_Base},
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

type Zt = BenchZipPlusTypes<i64, 32>;
type Mul = BinaryPolyWideningMulByScalar<i64>;

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

fn zip_plus_commit_15_polys_d1(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip+ Commit 15 Polys IPRS-1-1/4-F65537");

    commit_n_polys::<Zt, IprsD1V6, 6, 15>(&mut group, "IPRS-1-1/4-F65537");
    commit_n_polys::<Zt, IprsD1V7, 7, 15>(&mut group, "IPRS-1-1/4-F65537");
    commit_n_polys::<Zt, IprsD1V8, 8, 15>(&mut group, "IPRS-1-1/4-F65537");
    commit_n_polys::<Zt, IprsD1V9, 9, 15>(&mut group, "IPRS-1-1/4-F65537");
    commit_n_polys::<Zt, IprsD1V10, 10, 15>(&mut group, "IPRS-1-1/4-F65537");
    commit_n_polys::<Zt, IprsD1V11, 11, 15>(&mut group, "IPRS-1-1/4-F65537");

    group.finish();
}

fn zip_plus_commit_15_polys_d2(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip+ Commit 15 Polys IPRS-2-1/4-F65537");

    commit_n_polys::<Zt, IprsD2V6, 6, 15>(&mut group, "IPRS-2-1/4-F65537");
    commit_n_polys::<Zt, IprsD2V7, 7, 15>(&mut group, "IPRS-2-1/4-F65537");
    commit_n_polys::<Zt, IprsD2V8, 8, 15>(&mut group, "IPRS-2-1/4-F65537");
    commit_n_polys::<Zt, IprsD2V9, 9, 15>(&mut group, "IPRS-2-1/4-F65537");
    commit_n_polys::<Zt, IprsD2V10, 10, 15>(&mut group, "IPRS-2-1/4-F65537");
    commit_n_polys::<Zt, IprsD2V11, 11, 15>(&mut group, "IPRS-2-1/4-F65537");

    group.finish();
}

fn zip_plus_test_15_polys_d1(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip+ Test 15 Polys IPRS-1-1/4-F65537");

    test_n_polys::<Zt, IprsD1V6, { UNCHECKED }, 6, 15>(&mut group, "IPRS-1-1/4-F65537");
    test_n_polys::<Zt, IprsD1V7, { UNCHECKED }, 7, 15>(&mut group, "IPRS-1-1/4-F65537");
    test_n_polys::<Zt, IprsD1V8, { UNCHECKED }, 8, 15>(&mut group, "IPRS-1-1/4-F65537");
    test_n_polys::<Zt, IprsD1V9, { UNCHECKED }, 9, 15>(&mut group, "IPRS-1-1/4-F65537");
    test_n_polys::<Zt, IprsD1V10, { UNCHECKED }, 10, 15>(&mut group, "IPRS-1-1/4-F65537");
    test_n_polys::<Zt, IprsD1V11, { UNCHECKED }, 11, 15>(&mut group, "IPRS-1-1/4-F65537");

    group.finish();
}

fn zip_plus_test_15_polys_d2(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip+ Test 15 Polys IPRS-2-1/4-F65537");

    test_n_polys::<Zt, IprsD2V6, { UNCHECKED }, 6, 15>(&mut group, "IPRS-2-1/4-F65537");
    test_n_polys::<Zt, IprsD2V7, { UNCHECKED }, 7, 15>(&mut group, "IPRS-2-1/4-F65537");
    test_n_polys::<Zt, IprsD2V8, { UNCHECKED }, 8, 15>(&mut group, "IPRS-2-1/4-F65537");
    test_n_polys::<Zt, IprsD2V9, { UNCHECKED }, 9, 15>(&mut group, "IPRS-2-1/4-F65537");
    test_n_polys::<Zt, IprsD2V10, { UNCHECKED }, 10, 15>(&mut group, "IPRS-2-1/4-F65537");
    test_n_polys::<Zt, IprsD2V11, { UNCHECKED }, 11, 15>(&mut group, "IPRS-2-1/4-F65537");

    group.finish();
}

fn zip_plus_evaluate_15_polys_d1(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip+ Evaluate 15 Polys IPRS-1-1/4-F65537");

    evaluate_n_polys::<Zt, IprsD1V6, { UNCHECKED }, 6, 15>(&mut group, "IPRS-1-1/4-F65537");
    evaluate_n_polys::<Zt, IprsD1V7, { UNCHECKED }, 7, 15>(&mut group, "IPRS-1-1/4-F65537");
    evaluate_n_polys::<Zt, IprsD1V8, { UNCHECKED }, 8, 15>(&mut group, "IPRS-1-1/4-F65537");
    evaluate_n_polys::<Zt, IprsD1V9, { UNCHECKED }, 9, 15>(&mut group, "IPRS-1-1/4-F65537");
    evaluate_n_polys::<Zt, IprsD1V10, { UNCHECKED }, 10, 15>(&mut group, "IPRS-1-1/4-F65537");
    evaluate_n_polys::<Zt, IprsD1V11, { UNCHECKED }, 11, 15>(&mut group, "IPRS-1-1/4-F65537");

    group.finish();
}

fn zip_plus_evaluate_15_polys_d2(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip+ Evaluate 15 Polys IPRS-2-1/4-F65537");

    evaluate_n_polys::<Zt, IprsD2V6, { UNCHECKED }, 6, 15>(&mut group, "IPRS-2-1/4-F65537");
    evaluate_n_polys::<Zt, IprsD2V7, { UNCHECKED }, 7, 15>(&mut group, "IPRS-2-1/4-F65537");
    evaluate_n_polys::<Zt, IprsD2V8, { UNCHECKED }, 8, 15>(&mut group, "IPRS-2-1/4-F65537");
    evaluate_n_polys::<Zt, IprsD2V9, { UNCHECKED }, 9, 15>(&mut group, "IPRS-2-1/4-F65537");
    evaluate_n_polys::<Zt, IprsD2V10, { UNCHECKED }, 10, 15>(&mut group, "IPRS-2-1/4-F65537");
    evaluate_n_polys::<Zt, IprsD2V11, { UNCHECKED }, 11, 15>(&mut group, "IPRS-2-1/4-F65537");

    group.finish();
}

fn zip_plus_verify_15_polys_d1(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip+ Verify 15 Polys IPRS-1-1/4-F65537");

    verify_n_polys::<Zt, IprsD1V6, { UNCHECKED }, 6, 15>(&mut group, "IPRS-1-1/4-F65537");
    verify_n_polys::<Zt, IprsD1V7, { UNCHECKED }, 7, 15>(&mut group, "IPRS-1-1/4-F65537");
    verify_n_polys::<Zt, IprsD1V8, { UNCHECKED }, 8, 15>(&mut group, "IPRS-1-1/4-F65537");
    verify_n_polys::<Zt, IprsD1V9, { UNCHECKED }, 9, 15>(&mut group, "IPRS-1-1/4-F65537");
    verify_n_polys::<Zt, IprsD1V10, { UNCHECKED }, 10, 15>(&mut group, "IPRS-1-1/4-F65537");
    verify_n_polys::<Zt, IprsD1V11, { UNCHECKED }, 11, 15>(&mut group, "IPRS-1-1/4-F65537");

    group.finish();
}

fn zip_plus_verify_15_polys_d2(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip+ Verify 15 Polys IPRS-2-1/4-F65537");

    verify_n_polys::<Zt, IprsD2V6, { UNCHECKED }, 6, 15>(&mut group, "IPRS-2-1/4-F65537");
    verify_n_polys::<Zt, IprsD2V7, { UNCHECKED }, 7, 15>(&mut group, "IPRS-2-1/4-F65537");
    verify_n_polys::<Zt, IprsD2V8, { UNCHECKED }, 8, 15>(&mut group, "IPRS-2-1/4-F65537");
    verify_n_polys::<Zt, IprsD2V9, { UNCHECKED }, 9, 15>(&mut group, "IPRS-2-1/4-F65537");
    verify_n_polys::<Zt, IprsD2V10, { UNCHECKED }, 10, 15>(&mut group, "IPRS-2-1/4-F65537");
    verify_n_polys::<Zt, IprsD2V11, { UNCHECKED }, 11, 15>(&mut group, "IPRS-2-1/4-F65537");

    group.finish();
}

criterion_group!(
    benches,
    zip_plus_commit_15_polys_d1,
    zip_plus_commit_15_polys_d2,
    zip_plus_test_15_polys_d1,
    zip_plus_test_15_polys_d2,
    zip_plus_evaluate_15_polys_d1,
    zip_plus_evaluate_15_polys_d2,
    zip_plus_verify_15_polys_d1,
    zip_plus_verify_15_polys_d2
);
criterion_main!(benches);
