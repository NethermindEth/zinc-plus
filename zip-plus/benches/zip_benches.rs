#![allow(non_local_definitions)]
#![allow(clippy::eq_op, clippy::arithmetic_side_effects, clippy::unwrap_used)]

mod zip_common;

use zinc_primality::MillerRabin;
use zinc_utils::inner_product::{MBSInnerProduct, ScalarProduct};
use zip_common::*;

use criterion::{Criterion, criterion_group, criterion_main};
use crypto_bigint::U64;
use crypto_primitives::{crypto_bigint_int::Int, crypto_bigint_uint::Uint};
use zinc_utils::UNCHECKED;
use zip_plus::{
    code::{
        iprs::{
            IprsCode,
            PnttConfigF2_16_1_Base16_Depth1_Rate1_2,
            PnttConfigF2_16_1_Base16_Depth1_Rate1_4,
            PnttConfigF2_16_1_Base32_Depth1_Rate1_2,
            PnttConfigF2_16_1_Base32_Depth1_Rate1_4,
            PnttConfigF2_16_1_Base64_Depth1_Rate1_2,
            PnttConfigF2_16_1_Base64_Depth1_Rate1_4,
            PnttConfigF2_16_1_Depth2_Rate1_2,
            PnttConfigF2_16_1_Depth2_Rate1_4,
        },
        raa::{RaaCode, RaaConfig},
    },
    pcs::structs::ZipTypes,
};
use zinc_utils::mul_by_scalar::WideningMulByScalar;

const INT_LIMBS: usize = U64::LIMBS;

struct BenchZipTypes {}
impl ZipTypes for BenchZipTypes {
    const NUM_COLUMN_OPENINGS: usize = 200;
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

#[derive(Clone, Copy)]
struct BenchRaaConfig;
impl RaaConfig for BenchRaaConfig {
    const PERMUTE_IN_PLACE: bool = false;
    const CHECK_FOR_OVERFLOWS: bool = UNCHECKED;
}

type Code = RaaCode<BenchZipTypes, BenchRaaConfig, 4>;

#[derive(Clone, Default)]
struct I32WideningMulByScalar;

impl WideningMulByScalar<i32, i64> for I32WideningMulByScalar {
    type Output = i64;

    #[allow(clippy::arithmetic_side_effects)]
    fn mul_by_scalar_widen(lhs: &i32, rhs: &i64) -> Self::Output {
        i64::from(*lhs) * *rhs
    }
}

type IprsCodeDepth2 = IprsCode<BenchZipTypes, PnttConfigF2_16_1_Depth2_Rate1_2, I32WideningMulByScalar>;
type IprsCodeDepth2Rate1_4 =
    IprsCode<BenchZipTypes, PnttConfigF2_16_1_Depth2_Rate1_4, I32WideningMulByScalar>;

type IprsCodeDepth1Base16 =
    IprsCode<BenchZipTypes, PnttConfigF2_16_1_Base16_Depth1_Rate1_2, I32WideningMulByScalar>;
type IprsCodeDepth1Base32 =
    IprsCode<BenchZipTypes, PnttConfigF2_16_1_Base32_Depth1_Rate1_2, I32WideningMulByScalar>;
type IprsCodeDepth1Base64 =
    IprsCode<BenchZipTypes, PnttConfigF2_16_1_Base64_Depth1_Rate1_2, I32WideningMulByScalar>;

type IprsCodeDepth1Base16Rate1_4 =
    IprsCode<BenchZipTypes, PnttConfigF2_16_1_Base16_Depth1_Rate1_4, I32WideningMulByScalar>;
type IprsCodeDepth1Base32Rate1_4 =
    IprsCode<BenchZipTypes, PnttConfigF2_16_1_Base32_Depth1_Rate1_4, I32WideningMulByScalar>;
type IprsCodeDepth1Base64Rate1_4 =
    IprsCode<BenchZipTypes, PnttConfigF2_16_1_Base64_Depth1_Rate1_4, I32WideningMulByScalar>;

fn zip_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip+");
    do_bench::<BenchZipTypes, Code, UNCHECKED>(&mut group);
    group.finish();
}

fn zip_benchmarks_iprs(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip IPRS");
    do_bench_iprs_matrices::<BenchZipTypes, IprsCodeDepth2, UNCHECKED>(&mut group);
    group.finish();
}

fn zip_benchmarks_iprs_matrix_shapes(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip IPRS Matrix Shapes");
    do_bench_iprs_matrix_shapes::<BenchZipTypes, IprsCodeDepth1Base16, UNCHECKED>(&mut group);
    do_bench_iprs_matrix_shapes::<BenchZipTypes, IprsCodeDepth1Base32, UNCHECKED>(&mut group);
    do_bench_iprs_matrix_shapes::<BenchZipTypes, IprsCodeDepth1Base64, UNCHECKED>(&mut group);
    group.finish();
}

fn zip_benchmarks_iprs_rate1_4(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip IPRS rate1_4");
    do_bench_iprs_matrices::<BenchZipTypes, IprsCodeDepth2Rate1_4, UNCHECKED>(&mut group);
    group.finish();
}

fn zip_benchmarks_iprs_rate1_4_matrix_shapes(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip IPRS rate1_4 Matrix Shapes");
    do_bench_iprs_matrix_shapes::<BenchZipTypes, IprsCodeDepth1Base16Rate1_4, UNCHECKED>(
        &mut group,
    );
    do_bench_iprs_matrix_shapes::<BenchZipTypes, IprsCodeDepth1Base32Rate1_4, UNCHECKED>(
        &mut group,
    );
    do_bench_iprs_matrix_shapes::<BenchZipTypes, IprsCodeDepth1Base64Rate1_4, UNCHECKED>(
        &mut group,
    );
    group.finish();
}

criterion_group!(
    benches,
    zip_benchmarks,
    zip_benchmarks_iprs,
    zip_benchmarks_iprs_matrix_shapes,
    zip_benchmarks_iprs_rate1_4,
    zip_benchmarks_iprs_rate1_4_matrix_shapes
);
criterion_main!(benches);
