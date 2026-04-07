#![allow(non_local_definitions)]
#![allow(clippy::eq_op, clippy::arithmetic_side_effects, clippy::unwrap_used)]

mod zip_common;

use criterion::{Criterion, criterion_group, criterion_main};
use crypto_bigint::U64;
use crypto_primitives::{crypto_bigint_int::Int, crypto_bigint_uint::Uint};
use zinc_primality::MillerRabin;
use zinc_utils::inner_product::{MBSInnerProduct, ScalarProduct};
use zip_common::*;
use zip_plus::{
    code::iprs::{IprsCode, PnttConfigF65537},
    pcs::structs::ZipTypes,
};

const PERFORM_CHECKS: bool = if cfg!(feature = "unchecked") {
    zinc_utils::UNCHECKED
} else {
    zinc_utils::CHECKED
};

const INT_LIMBS: usize = U64::LIMBS;

struct BenchZipTypes {}
impl ZipTypes for BenchZipTypes {
    const NUM_COLUMN_OPENINGS: usize = 147;
    type Eval = i32;
    type Cw = i64;
    type Fmod = Uint<{ INT_LIMBS * 4 }>;
    type PrimeTest = MillerRabin;
    type Chal = i128;
    type Pt = i128;
    type CombR = Int<{ INT_LIMBS * 3 }>;
    type Comb = Self::CombR;
    type EvalDotChal = ScalarProduct;
    type CombDotChal = ScalarProduct;
    type ArrCombRDotChal = MBSInnerProduct;
}

type BenchIprsCode = IprsCode<BenchZipTypes, PnttConfigF65537, 4, PERFORM_CHECKS>;

fn zip_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip+ IPRS");
    do_bench::<BenchZipTypes, _, PERFORM_CHECKS>(&mut group, |poly_size| {
        BenchIprsCode::new_with_optimal_depth(poly_size).unwrap()
    });
    group.finish();
}

criterion_group!(benches, zip_benchmarks);
criterion_main!(benches);
