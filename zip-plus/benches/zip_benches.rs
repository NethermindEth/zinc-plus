#![allow(non_local_definitions)]
#![allow(clippy::eq_op, clippy::arithmetic_side_effects, clippy::unwrap_used)]

mod zip_common;

use zip_common::*;

use criterion::{Criterion, criterion_group, criterion_main};
use crypto_primitives::crypto_bigint_int::Int;
use zip_plus::{code::raa::RaaCode, pcs::structs::ZipTypes, utils::WORD_FACTOR};

const INT_LIMBS: usize = WORD_FACTOR;

struct BenchZipTypes {}
impl ZipTypes for BenchZipTypes {
    const NUM_COLUMN_OPENINGS: usize = 650;
    type EvalR = i32;
    type Eval = Self::EvalR;
    type CwR = i64;
    type Cw = Self::CwR;
    type Chal = i128;
    type Pt = i128;
    type CombR = Int<{ INT_LIMBS * 3 }>;
    type Comb = Self::CombR;
    type Code = RaaCode<Self::Eval, Self::Cw, Self::Comb, 4>;
}

fn zip_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip+");
    do_bench::<BenchZipTypes>(&mut group);
    group.finish();
}

criterion_group!(benches, zip_benchmarks);
criterion_main!(benches);
