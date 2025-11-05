#![allow(non_local_definitions)]
#![allow(clippy::eq_op, clippy::arithmetic_side_effects, clippy::unwrap_used)]

mod zip_common;

use std::marker::PhantomData;
use zip_common::*;

use criterion::{Criterion, criterion_group, criterion_main};
use crypto_bigint::U64;
use crypto_primitives::{crypto_bigint_int::Int, crypto_bigint_uint::Uint, FixedSemiring};
use zip_plus::{
    code::raa::{RaaCode, RaaConfig},
    pcs::structs::ZipTypes,
    poly::{bit_decomposed::BitDecomposedPolynomial, dense::DensePolynomial},
    primality::MillerRabin,
};
use zip_plus::code::raa_sign_flip::RaaSignFlippingCode;
use zip_plus::poly::ConstCoeffBitWidth;
use zip_plus::traits::{ConstTranscribable, FromRef, Named};

const INT_LIMBS: usize = U64::LIMBS;

type CombR = Int<{ INT_LIMBS * 5 }>;
type Comb = DensePolynomial<CombR, 31>;

struct BenchZipPlusTypes<EvalCw, const D: usize>
where
    EvalCw: FixedSemiring
    + ConstCoeffBitWidth
    + ConstTranscribable
    + FromRef<EvalCw>
    + Named
    + Copy,
    Comb: FromRef<EvalCw>
{
    phantom_data: PhantomData<EvalCw>
}

impl<EvalCw, const D: usize> ZipTypes<D> for BenchZipPlusTypes<EvalCw, D>
where
    EvalCw: FixedSemiring
    + ConstCoeffBitWidth
    + ConstTranscribable
    + FromRef<EvalCw>
    + Named
    + Copy,
    Comb: FromRef<EvalCw>
{
    const NUM_COLUMN_OPENINGS: usize = 650;
    type Eval = EvalCw;
    type Cw = EvalCw;
    type Fmod = Uint<{ INT_LIMBS * 4 }>;
    type PrimeTest = MillerRabin;
    type Chal = i128;
    type Pt = i128;
    type CombR = CombR;
    type Comb = Comb;
}

const RAA_CONFIG: RaaConfig = RaaConfig {
    check_for_overflows: false,
    permute_in_place: false,
};

fn zip_plus_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip+");

    do_bench::<BenchZipPlusTypes<DensePolynomial<i64, 31>, 31>, RaaCode<_, 4, _>, _>(&mut group, RAA_CONFIG);
    do_bench::<BenchZipPlusTypes<DensePolynomial<i64, 31>, 31>, RaaSignFlippingCode<_, 4, _>, _>(&mut group, RAA_CONFIG);
    do_bench::<BenchZipPlusTypes<BitDecomposedPolynomial<20, 31>, 31>, RaaCode<_, 4, _>, _>(&mut group, RAA_CONFIG);
    do_bench::<BenchZipPlusTypes<BitDecomposedPolynomial<14, 31>, 31>, RaaSignFlippingCode<_, 4, _>, _>(&mut group, RAA_CONFIG);

    group.finish();
}

criterion_group!(benches, zip_plus_benchmarks);
criterion_main!(benches);
