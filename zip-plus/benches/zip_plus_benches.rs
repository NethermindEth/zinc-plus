#![allow(non_local_definitions)]
#![allow(clippy::eq_op, clippy::arithmetic_side_effects, clippy::unwrap_used)]

mod zip_common;

use zip_common::*;

use criterion::{Criterion, criterion_group, criterion_main};
use crypto_primitives::crypto_bigint_int::Int;
use zip_plus::{
    code::raa::RaaCode, pcs::structs::ZipTypes, poly::dense::DensePolynomial, utils::WORD_FACTOR,
};

const INT_LIMBS: usize = WORD_FACTOR;

struct BenchZipPlusTypes<const D: usize> {}
impl<const D: usize> ZipTypes for BenchZipPlusTypes<D> {
    type EvalR = i32;
    type Eval = DensePolynomial<Self::EvalR, D>;
    type CwR = i64;
    type Cw = DensePolynomial<Self::CwR, D>;
    type Chal = i128;
    type Pt = i128;
    type CombR = Int<{ INT_LIMBS * 4 }>;
    type Comb = DensePolynomial<Self::CombR, D>;
    type Code = RaaCode<Self::Eval, Self::Cw, Self::CombR>;
}

fn zip_plus_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip+");

    do_bench::<BenchZipPlusTypes<31>>(&mut group);
    do_bench::<BenchZipPlusTypes<63>>(&mut group);

    group.finish();
}

criterion_group!(benches, zip_plus_benchmarks);
criterion_main!(benches);
