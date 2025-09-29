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
    type EvalR = Int<{ INT_LIMBS }>;
    type Eval = Self::EvalR;
    type CwR = Int<{ INT_LIMBS * 4 }>;
    type Cw = Self::CwR;
    type Chal = Int<{ INT_LIMBS }>;
    type CombR = Int<{ INT_LIMBS * 8 }>;
    type Comb = Self::CombR;
    type Code = RaaCode<Self::Eval, Self::Cw, Self::Comb>;
}

fn zip_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip");

    encode_rows::<BenchZipTypes, 12>(&mut group);
    encode_rows::<BenchZipTypes, 13>(&mut group);
    encode_rows::<BenchZipTypes, 14>(&mut group);
    encode_rows::<BenchZipTypes, 15>(&mut group);
    encode_rows::<BenchZipTypes, 16>(&mut group);

    encode_single_row::<BenchZipTypes, 128>(&mut group);
    encode_single_row::<BenchZipTypes, 256>(&mut group);
    encode_single_row::<BenchZipTypes, 512>(&mut group);
    encode_single_row::<BenchZipTypes, 1024>(&mut group);
    encode_single_row::<BenchZipTypes, 2048>(&mut group);
    encode_single_row::<BenchZipTypes, 4096>(&mut group);

    merkle_root::<BenchZipTypes, 12>(&mut group);
    merkle_root::<BenchZipTypes, 13>(&mut group);
    merkle_root::<BenchZipTypes, 14>(&mut group);
    merkle_root::<BenchZipTypes, 15>(&mut group);
    merkle_root::<BenchZipTypes, 16>(&mut group);

    commit::<BenchZipTypes, 12>(&mut group);
    commit::<BenchZipTypes, 13>(&mut group);
    commit::<BenchZipTypes, 14>(&mut group);
    commit::<BenchZipTypes, 15>(&mut group);
    commit::<BenchZipTypes, 16>(&mut group);

    open::<BenchZipTypes, 12>(&mut group);
    open::<BenchZipTypes, 13>(&mut group);
    open::<BenchZipTypes, 14>(&mut group);
    open::<BenchZipTypes, 15>(&mut group);
    open::<BenchZipTypes, 16>(&mut group);

    verify::<BenchZipTypes, 12>(&mut group);
    verify::<BenchZipTypes, 13>(&mut group);
    verify::<BenchZipTypes, 14>(&mut group);
    verify::<BenchZipTypes, 15>(&mut group);
    verify::<BenchZipTypes, 16>(&mut group);

    group.finish();
}

criterion_group!(benches, zip_benchmarks);
criterion_main!(benches);
