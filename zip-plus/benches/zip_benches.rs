#![allow(non_local_definitions)]
#![allow(clippy::eq_op, clippy::arithmetic_side_effects, clippy::unwrap_used)]

mod zip_common;

use zinc_primality::MillerRabin;
use zinc_utils::{inner_product::ScalarProduct, projection_to_field::SimpleProjection};
use zip_common::*;

use criterion::{Criterion, criterion_group, criterion_main};
use crypto_bigint::U64;
use crypto_primitives::{crypto_bigint_int::Int, crypto_bigint_uint::Uint};
use zip_plus::{
    code::raa::{RaaCode, RaaConfigGeneric},
    pcs::structs::ZipTypes,
};

const INT_LIMBS: usize = U64::LIMBS;

struct BenchZipTypes {}
impl ZipTypes for BenchZipTypes {
    const NUM_COLUMN_OPENINGS: usize = 650;
    type Eval = i32;
    type Cw = i64;
    type Fmod = Uint<{ INT_LIMBS * 4 }>;
    type PrimeTest = MillerRabin;
    type Chal = i128;
    type Pt = i128;
    type CombR = Int<{ INT_LIMBS * 3 }>;
    type Comb = Self::CombR;
    type CombDotChal = ScalarProduct;
    type EvalDotChal = ScalarProduct;
}
type Code = RaaCode<BenchZipTypes, RaaConfigGeneric<false, false>, 4>;

fn zip_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip+");
    do_bench::<BenchZipTypes, Code, SimpleProjection<_>, SimpleProjection<_>>(&mut group);
    group.finish();
}

criterion_group!(benches, zip_benchmarks);
criterion_main!(benches);
