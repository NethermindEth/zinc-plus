#![allow(non_local_definitions)]
#![allow(clippy::eq_op, clippy::arithmetic_side_effects, clippy::unwrap_used)]

mod zip_common;

use zinc_poly::univariate::dense::{
    DensePolyInnerProduct, DensePolynomial, HornerProjection, InnerProductProjection,
};
use zinc_primality::MillerRabin;
use zinc_utils::inner_product::{
    BooleanInnerProductCheckedAdd, BooleanInnerProductUncheckedAdd, MBSInnerProductChecked,
};
use zip_common::*;

use criterion::{Criterion, criterion_group, criterion_main};
use crypto_bigint::U64;
use crypto_primitives::{boolean::Boolean, crypto_bigint_int::Int, crypto_bigint_uint::Uint};
use zip_plus::{
    code::raa::{RaaCode, RaaConfig},
    pcs::structs::ZipTypes,
};

const INT_LIMBS: usize = U64::LIMBS;

struct BenchZipPlusTypes<const D_PLUS_ONE: usize> {}
impl<const D_PLUS_ONE: usize> ZipTypes for BenchZipPlusTypes<D_PLUS_ONE> {
    const NUM_COLUMN_OPENINGS: usize = 650;
    type Eval = DensePolynomial<Boolean, D_PLUS_ONE>;
    type Cw = DensePolynomial<i32, D_PLUS_ONE>;
    type Fmod = Uint<{ INT_LIMBS * 4 }>;
    type PrimeTest = MillerRabin;
    type Chal = i128;
    type Pt = i128;
    type CombR = Int<{ INT_LIMBS * 5 }>;
    type Comb = DensePolynomial<Self::CombR, D_PLUS_ONE>;
    type EvalDotChal = DensePolyInnerProduct<
        Boolean,
        Self::Chal,
        Self::CombR,
        BooleanInnerProductCheckedAdd,
        D_PLUS_ONE,
    >;
    type CombDotChal = DensePolyInnerProduct<
        Self::CombR,
        Self::Chal,
        Self::CombR,
        MBSInnerProductChecked,
        D_PLUS_ONE,
    >;
}

#[derive(Clone, Copy)]
struct BenchRaaConfig;
impl RaaConfig for BenchRaaConfig {
    const PERMUTE_IN_PLACE: bool = true;
    const CHECK_FOR_OVERFLOWS: bool = false;
}

type Code<const D_PLUS_ONE: usize> = RaaCode<BenchZipPlusTypes<D_PLUS_ONE>, BenchRaaConfig, 4>;

fn zip_plus_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip+");

    do_bench::<
        BenchZipPlusTypes<32>,
        Code<_>,
        InnerProductProjection<Boolean, BooleanInnerProductUncheckedAdd, _>,
        HornerProjection<_, _>,
    >(&mut group);
    do_bench::<
        BenchZipPlusTypes<64>,
        Code<_>,
        InnerProductProjection<Boolean, BooleanInnerProductUncheckedAdd, _>,
        HornerProjection<_, _>,
    >(&mut group);

    group.finish();
}

criterion_group!(benches, zip_plus_benchmarks);
criterion_main!(benches);
