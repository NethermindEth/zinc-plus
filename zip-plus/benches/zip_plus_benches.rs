#![allow(non_local_definitions)]
#![allow(clippy::eq_op, clippy::arithmetic_side_effects, clippy::unwrap_used)]

mod zip_common;

use zip_common::*;

use criterion::{Criterion, criterion_group, criterion_main};
use crypto_bigint::U64;
use crypto_primitives::{
    FixedSemiring, boolean::Boolean, crypto_bigint_int::Int, crypto_bigint_uint::Uint,
};
use std::marker::PhantomData;
use zinc_poly::univariate::{
    binary::{BinaryPoly, BinaryPolyInnerProduct},
    dense::{DensePolyInnerProduct, DensePolynomial},
};
use zinc_primality::MillerRabin;
use zinc_transcript::traits::ConstTranscribable;
use zinc_utils::{from_ref::FromRef, inner_product::MBSInnerProduct, named::Named};
use zip_plus::{
    code::{
        iprs::{IprsCode, PnttConfigF65537},
        raa::{RaaCode, RaaConfig},
    },
    pcs::structs::ZipTypes,
};

const PERFORM_CHECKS: bool = if cfg!(feature = "unchecked") {
    zinc_utils::UNCHECKED
} else {
    zinc_utils::CHECKED
};

const INT_LIMBS: usize = U64::LIMBS;

#[derive(Debug, Clone)]
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

#[derive(Clone, Copy)]
struct BenchRaaConfig;
impl RaaConfig for BenchRaaConfig {
    const PERMUTE_IN_PLACE: bool = true;
    const CHECK_FOR_OVERFLOWS: bool = PERFORM_CHECKS;
}

type BenchRaaCode<const D_PLUS_ONE: usize> =
    RaaCode<BenchZipPlusTypes<i32, D_PLUS_ONE>, BenchRaaConfig, 4>;

type BenchIprsCode<Twiddle, const D_PLUS_ONE: usize> =
    IprsCode<BenchZipPlusTypes<Twiddle, D_PLUS_ONE>, PnttConfigF65537, 4, PERFORM_CHECKS>;

fn zip_plus_benchmarks_raa(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip+ RAA");

    do_bench::<BenchZipPlusTypes<i32, 32>, _, PERFORM_CHECKS>(&mut group, |poly_size| {
        Some(BenchRaaCode::new(poly_size.isqrt().next_power_of_two()))
    });
    do_bench::<BenchZipPlusTypes<i32, 64>, _, PERFORM_CHECKS>(&mut group, |poly_size| {
        Some(BenchRaaCode::new(poly_size.isqrt().next_power_of_two()))
    });

    group.finish();
}

fn zip_plus_benchmarks_iprs(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip+ IPRS");

    // Use flat single-row Zip+ matrix
    do_bench::<BenchZipPlusTypes<i64, 32>, _, PERFORM_CHECKS>(&mut group, |poly_size| {
        BenchIprsCode::new_with_optimal_depth(poly_size).ok()
    });
    do_bench::<BenchZipPlusTypes<i64, 64>, _, PERFORM_CHECKS>(&mut group, |poly_size| {
        BenchIprsCode::new_with_optimal_depth(poly_size).ok()
    });

    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(500);
    targets = zip_plus_benchmarks_raa, zip_plus_benchmarks_iprs
}
criterion_main!(benches);
