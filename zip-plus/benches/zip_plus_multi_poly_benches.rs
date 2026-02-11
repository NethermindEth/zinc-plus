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
    code::iprs::{IprsCode, PnttConfigF2_16_1},
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

type IprsCodeType<Twiddle, const DEPTH: usize, const D_PLUS_ONE: usize, const CHECK: bool> =
    IprsCode<
        BenchZipPlusTypes<Twiddle, D_PLUS_ONE>,
        PnttConfigF2_16_1<DEPTH>,
        BinaryPolyWideningMulByScalar<Twiddle>,
        CHECK,
    >;

/// Polynomial counts to benchmark.
const POLY_COUNTS: &[usize] = &[8];

/// Fixed polynomial size exponent (2^P evaluations per polynomial).
const P: usize = 12;

fn multi_poly_commit(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip+ MultiPoly Commit");
    group.sample_size(10);

    for &num_polys in POLY_COUNTS {
        batch_commit::<
            BenchZipPlusTypes<i64, 32>,
            IprsCodeType<i64, 1, 32, UNCHECKED>,
            P,
        >(&mut group, num_polys);
    }

    group.finish();
}

fn multi_poly_test(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip+ MultiPoly Test");
    group.sample_size(10);

    for &num_polys in POLY_COUNTS {
        batch_test::<
            BenchZipPlusTypes<i64, 32>,
            IprsCodeType<i64, 1, 32, UNCHECKED>,
            UNCHECKED,
            P,
        >(&mut group, num_polys);
    }

    group.finish();
}

fn multi_poly_evaluate(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip+ MultiPoly Evaluate");
    group.sample_size(10);

    for &num_polys in POLY_COUNTS {
        batch_evaluate::<
            BenchZipPlusTypes<i64, 32>,
            IprsCodeType<i64, 1, 32, UNCHECKED>,
            UNCHECKED,
            P,
        >(&mut group, num_polys);
    }

    group.finish();
}

fn multi_poly_verify(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip+ MultiPoly Verify");
    group.sample_size(10);

    for &num_polys in POLY_COUNTS {
        batch_verify::<
            BenchZipPlusTypes<i64, 32>,
            IprsCodeType<i64, 1, 32, UNCHECKED>,
            UNCHECKED,
            P,
        >(&mut group, num_polys);
    }

    group.finish();
}

criterion_group!(
    benches,
    multi_poly_commit,
    multi_poly_test,
    // multi_poly_evaluate,
    // multi_poly_verify,
);
criterion_main!(benches);
