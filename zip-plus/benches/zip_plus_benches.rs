#![allow(non_local_definitions)]
#![allow(clippy::eq_op, clippy::arithmetic_side_effects, clippy::unwrap_used)]

mod zip_common;

use zinc_poly::univariate::dense::DensePolynomial;
use zinc_primality::MillerRabin;
use zip_common::*;

use criterion::{Criterion, criterion_group, criterion_main};
use crypto_bigint::U64;
use crypto_primitives::{boolean::Boolean, crypto_bigint_int::Int, crypto_bigint_uint::Uint};
use zip_plus::{
    code::{
        iprs::{IprsCode, IprsConfig},
        raa::{RaaCode, RaaConfig},
    },
    pcs::structs::ZipTypes,
};

const INT_LIMBS: usize = U64::LIMBS;

struct BenchZipPlusTypes<const D_PLUS_ONE: usize> {}
impl<const D_PLUS_ONE: usize> ZipTypes for BenchZipPlusTypes<D_PLUS_ONE> {
    const NUM_COLUMN_OPENINGS: usize = 650;
    type Eval = DensePolynomial<Boolean, D_PLUS_ONE>;
    type Twiddle = i32;
    type Cw = DensePolynomial<i32, D_PLUS_ONE>;
    type Fmod = Uint<{ INT_LIMBS * 4 }>;
    type PrimeTest = MillerRabin;
    type Chal = i128;
    type Pt = i128;
    type CombR = Int<{ INT_LIMBS * 5 }>;
    type Comb = DensePolynomial<Self::CombR, D_PLUS_ONE>;
}

struct BenchZipPlusTypesIPRS<const D_PLUS_ONE: usize> {}
impl<const D_PLUS_ONE: usize> ZipTypes for BenchZipPlusTypesIPRS<D_PLUS_ONE> {
    const NUM_COLUMN_OPENINGS: usize = 650;
    type Eval = DensePolynomial<Boolean, D_PLUS_ONE>;
    type Twiddle = i128;
    type Cw = DensePolynomial<i128, D_PLUS_ONE>;
    type Fmod = Uint<{ INT_LIMBS * 4 }>;
    type PrimeTest = MillerRabin;
    type Chal = i128;
    type Pt = i128;
    type CombR = Int<{ INT_LIMBS * 5 }>;
    type Comb = DensePolynomial<Self::CombR, D_PLUS_ONE>;
}
type SomeRaaCode<const D_PLUS_ONE: usize> = RaaCode<BenchZipPlusTypes<D_PLUS_ONE>, 4>;
type SomeIprsCode<const D_PLUS_ONE: usize> = IprsCode<BenchZipPlusTypesIPRS<D_PLUS_ONE>, 2>;

const RAA_CONFIG: RaaConfig = RaaConfig {
    check_for_overflows: false,
    permute_in_place: true,
};

fn zip_plus_benchmarks_raa(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip+ RAA");

    do_bench::<BenchZipPlusTypes<32>, SomeRaaCode<_>>(&mut group, RAA_CONFIG);
    do_bench::<BenchZipPlusTypes<64>, SomeRaaCode<_>>(&mut group, RAA_CONFIG);

    group.finish();
}

fn zip_plus_benchmarks_iprs(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip+ IPRS");

    let config = IprsConfig::new_any_m(1 << 12, 2, 3);
    encode_rows::<BenchZipPlusTypesIPRS<32>, SomeIprsCode<_>, 12>(&mut group, config);
}

criterion_group!(benches, zip_plus_benchmarks_raa, zip_plus_benchmarks_iprs);
criterion_main!(benches);
