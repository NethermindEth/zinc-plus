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
use zinc_utils::{
    from_ref::FromRef,
    inner_product::{BooleanInnerProductUncheckedAdd, MBSInnerProductUnchecked},
    named::Named,
};
use zip_common::*;

use criterion::{Criterion, criterion_group, criterion_main};
use crypto_bigint::U64;
use crypto_primitives::{
    FixedSemiring, boolean::Boolean, crypto_bigint_int::Int, crypto_bigint_uint::Uint,
};
use zip_plus::{
    code::{
        iprs::{IprsCode, PnttConfigF2_16_1},
        raa::{RaaCode, RaaConfig},
    },
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
    const NUM_COLUMN_OPENINGS: usize = 650;
    type Eval = BinaryPoly<D_PLUS_ONE>;
    type Cw = DensePolynomial<CwCoeff, D_PLUS_ONE>;
    type Fmod = Uint<{ INT_LIMBS * 4 }>;
    type PrimeTest = MillerRabin;
    type Chal = Int<{ INT_LIMBS *  2 }>;
    type Pt = i128;
    type CombR = Int<{ INT_LIMBS * 5 }>;
    type Comb = DensePolynomial<Self::CombR, D_PLUS_ONE>;
    type EvalDotChal =
        BinaryPolyInnerProduct<Self::Chal, D_PLUS_ONE>;
    type CombDotChal = DensePolyInnerProduct<
        Self::CombR,
        Self::Chal,
        Self::CombR,
        MBSInnerProductUnchecked,
        D_PLUS_ONE,
    >;
    type ArrCombRDotChal = MBSInnerProductUnchecked;
}

#[derive(Clone, Copy)]
struct BenchRaaConfig;
impl RaaConfig for BenchRaaConfig {
    const PERMUTE_IN_PLACE: bool = true;
    const CHECK_FOR_OVERFLOWS: bool = false;
}

type SomeRaaCode<const D_PLUS_ONE: usize> =
    RaaCode<BenchZipPlusTypes<i32, D_PLUS_ONE>, BenchRaaConfig, 4>;

type SomeIprsCode<Twiddle, const DEPTH: usize, const D_PLUS_ONE: usize> = IprsCode<
    BenchZipPlusTypes<Twiddle, D_PLUS_ONE>,
    PnttConfigF2_16_1<DEPTH>,
    BinaryPolyWideningMulByScalar<Twiddle>,
>;

fn zip_plus_benchmarks_raa(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip+ RAA");

    do_bench::<BenchZipPlusTypes<i32, 32>, SomeRaaCode<_>>(&mut group);
    // do_bench::<BenchZipPlusTypes<i32, 64>, SomeRaaCode<_>>(&mut group);

    group.finish();
}

// fn zip_plus_benchmarks_iprs(c: &mut Criterion) {
//     let mut group = c.benchmark_group("Zip+ IPRS");
//
//     do_bench::<BenchZipPlusTypes<i64, 32>, SomeIprsCode<i64, 1, 32>>(&mut group);
//     do_bench::<BenchZipPlusTypes<i64, 64>, SomeIprsCode<i64, 1, 64>>(&mut group);
//
//     group.finish();
// }

criterion_group!(benches, zip_plus_benchmarks_raa/* , zip_plus_benchmarks_iprs */);
criterion_main!(benches);
