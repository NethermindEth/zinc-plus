#![allow(non_local_definitions)]
#![allow(clippy::eq_op, clippy::arithmetic_side_effects, clippy::unwrap_used)]

mod zip_common;

use zip_common::*;

use criterion::{Criterion, criterion_group, criterion_main};
use crypto_bigint::{U64, Uint};
use crypto_primes::hazmat::MillerRabin;
use crypto_primitives::crypto_bigint_int::Int;
use zip_plus::{
    code::raa::RaaCode, pcs::structs::ZipTypes, poly::dense::DensePolynomial, utils::UintSemiring,
};

const INT_LIMBS: usize = U64::LIMBS;

struct BenchZipPlusTypes<const D: usize> {}
impl<const D: usize> ZipTypes for BenchZipPlusTypes<D> {
    const NUM_COLUMN_OPENINGS: usize = 650;
    type EvalR = i8; // TODO: Use binary polynomials for evaluations
    type Eval = DensePolynomial<Self::EvalR, D>;
    type CwR = i32;
    type Cw = DensePolynomial<Self::CwR, D>;
    type Fmod = UintSemiring<{ INT_LIMBS * 4 }>;
    type PrimeTest = MillerRabin<Uint<{ INT_LIMBS * 4 }>>;
    type Chal = i128;
    type Pt = i128;
    type CombR = Int<{ INT_LIMBS * 5 }>;
    type Comb = DensePolynomial<Self::CombR, D>;
}
type Code<const D: usize> = RaaCode<BenchZipPlusTypes<D>, 4>;

fn zip_plus_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip+");

    do_bench::<BenchZipPlusTypes<31>, Code<31>>(&mut group);
    do_bench::<BenchZipPlusTypes<63>, Code<63>>(&mut group);

    group.finish();
}

criterion_group!(benches, zip_plus_benchmarks);
criterion_main!(benches);
