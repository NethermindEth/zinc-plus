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

const DEGREE_BOUND: usize = 31;

struct BenchZipPlusTypes {}
impl ZipTypes for BenchZipPlusTypes {
    type EvalR = Int<{ INT_LIMBS }>;
    type Eval = DensePolynomial<Self::EvalR, DEGREE_BOUND>;
    type CwR = Int<{ INT_LIMBS * 4 }>;
    type Cw = DensePolynomial<Self::CwR, DEGREE_BOUND>;
    type Chal = Int<{ INT_LIMBS }>;
    type CombR = Int<{ INT_LIMBS * 8 }>;
    type Comb = DensePolynomial<Self::CombR, DEGREE_BOUND>;
    type Code = RaaCode<Self::Eval, Self::Cw, Self::CombR>;
}

fn zip_plus_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip");

    encode_rows::<BenchZipPlusTypes, 12>(&mut group);
    encode_rows::<BenchZipPlusTypes, 13>(&mut group);
    encode_rows::<BenchZipPlusTypes, 14>(&mut group);
    encode_rows::<BenchZipPlusTypes, 15>(&mut group);
    encode_rows::<BenchZipPlusTypes, 16>(&mut group);

    encode_single_row::<BenchZipPlusTypes, 128>(&mut group);
    encode_single_row::<BenchZipPlusTypes, 256>(&mut group);
    encode_single_row::<BenchZipPlusTypes, 512>(&mut group);
    encode_single_row::<BenchZipPlusTypes, 1024>(&mut group);
    encode_single_row::<BenchZipPlusTypes, 2048>(&mut group);
    encode_single_row::<BenchZipPlusTypes, 4096>(&mut group);

    merkle_root::<BenchZipPlusTypes, 12>(&mut group);
    merkle_root::<BenchZipPlusTypes, 13>(&mut group);
    merkle_root::<BenchZipPlusTypes, 14>(&mut group);
    merkle_root::<BenchZipPlusTypes, 15>(&mut group);
    merkle_root::<BenchZipPlusTypes, 16>(&mut group);

    commit::<BenchZipPlusTypes, 12>(&mut group);
    commit::<BenchZipPlusTypes, 13>(&mut group);
    commit::<BenchZipPlusTypes, 14>(&mut group);
    commit::<BenchZipPlusTypes, 15>(&mut group);
    commit::<BenchZipPlusTypes, 16>(&mut group);

    open::<BenchZipPlusTypes, 12>(&mut group);
    open::<BenchZipPlusTypes, 13>(&mut group);
    open::<BenchZipPlusTypes, 14>(&mut group);
    open::<BenchZipPlusTypes, 15>(&mut group);
    open::<BenchZipPlusTypes, 16>(&mut group);

    verify::<BenchZipPlusTypes, 12>(&mut group);
    verify::<BenchZipPlusTypes, 13>(&mut group);
    verify::<BenchZipPlusTypes, 14>(&mut group);
    verify::<BenchZipPlusTypes, 15>(&mut group);
    verify::<BenchZipPlusTypes, 16>(&mut group);

    group.finish();
}

criterion_group!(benches, zip_plus_benchmarks);
criterion_main!(benches);
