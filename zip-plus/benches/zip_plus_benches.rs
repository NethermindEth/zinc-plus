#![allow(non_local_definitions)]
#![allow(clippy::eq_op, clippy::arithmetic_side_effects, clippy::unwrap_used)]

mod zip_common;

use std::marker::PhantomData;

use zinc_poly::univariate::{
    binary::{BinaryPoly, BinaryPolyInnerProduct, BinaryPolyWideningMulByScalar},
    dense::{DensePolyInnerProduct, DensePolynomial},
};
use zinc_primality::MillerRabin;
use zinc_transcript::traits::ConstTranscribable;
use zinc_utils::{UNCHECKED, from_ref::FromRef, inner_product::MBSInnerProduct, named::Named};
use zip_common::*;

use criterion::{Criterion, criterion_group, criterion_main};
use crypto_bigint::U64;
use crypto_primitives::{
    FixedSemiring, boolean::Boolean, crypto_bigint_int::Int, crypto_bigint_uint::Uint,
};
use zip_plus::{
    code::{
        iprs::{
            IprsCode,
            PnttConfigF12289_Depth3_Rate1_2,
            PnttConfigF257_Base16_Depth1_Rate1_2,
            PnttConfigF257_Base8_Depth1_Rate1_2,
            PnttConfigF257_Depth2_Rate1_2,
            PnttConfigF2_16_1_Base16_Depth1_Rate1_2,
            PnttConfigF2_16_1_Base16_Depth1_Rate1_4,
            PnttConfigF2_16_1_Base32_Depth1_Rate1_2,
            PnttConfigF2_16_1_Base32_Depth1_Rate1_4,
            PnttConfigF2_16_1_Base64_Depth1_Rate1_2,
            PnttConfigF2_16_1_Base64_Depth1_Rate1_4,
            PnttConfigF2_16_1_Base8_Depth3_Rate1_2,
            PnttConfigF2_16_1_Depth2_Rate1_2,
            PnttConfigF2_16_1_Depth2_Rate1_4,
        },
        raa::{RaaCode, RaaConfig},
    },
    pcs::structs::ZipTypes,
};

const INT_LIMBS: usize = U64::LIMBS;

struct BenchZipPlusTypes<CwCoeff, const D_PLUS_ONE: usize>(PhantomData<CwCoeff>);

impl<CwCoeff, const D_PLUS_ONE: usize> ZipTypes for BenchZipPlusTypes<CwCoeff, D_PLUS_ONE>
where
    CwCoeff: ConstTranscribable
        + Copy
        + Default
        + FromRef<Boolean>
        + Named
        + FixedSemiring
        + Send
        + Sync,
    Int<5>: FromRef<CwCoeff>,
{
    const NUM_COLUMN_OPENINGS: usize = 200;
    type Eval = BinaryPoly<D_PLUS_ONE>;
    type Cw = DensePolynomial<CwCoeff, D_PLUS_ONE>;
    type Fmod = Uint<{ INT_LIMBS * 4 }>;
    type PrimeTest = MillerRabin;
    type Chal = i128;
    type Pt = i128;
    type CombR = Int<{ INT_LIMBS * 5 }>;
    type Comb = DensePolynomial<Self::CombR, D_PLUS_ONE>;
    type EvalDotChal = BinaryPolyInnerProduct<Self::Chal, D_PLUS_ONE>;
    type CombDotChal =
        DensePolyInnerProduct<Self::CombR, Self::Chal, Self::CombR, MBSInnerProduct, D_PLUS_ONE>;
    type ArrCombRDotChal = MBSInnerProduct;
}

#[derive(Clone, Copy)]
struct BenchRaaConfig;
impl RaaConfig for BenchRaaConfig {
    const PERMUTE_IN_PLACE: bool = true;
    const CHECK_FOR_OVERFLOWS: bool = UNCHECKED;
}

type SomeRaaCode<const D_PLUS_ONE: usize> =
    RaaCode<BenchZipPlusTypes<i32, D_PLUS_ONE>, BenchRaaConfig, 4>;

type SomeIprsCodeDepth2<Twiddle, const D_PLUS_ONE: usize> = IprsCode<
    BenchZipPlusTypes<Twiddle, D_PLUS_ONE>,
    PnttConfigF2_16_1_Depth2_Rate1_2,
    BinaryPolyWideningMulByScalar<Twiddle>,
>;

type SomeIprsCodeDepth1Base16<Twiddle, const D_PLUS_ONE: usize> = IprsCode<
    BenchZipPlusTypes<Twiddle, D_PLUS_ONE>,
    PnttConfigF2_16_1_Base16_Depth1_Rate1_2,
    BinaryPolyWideningMulByScalar<Twiddle>,
>;

type SomeIprsCodeDepth1Base32<Twiddle, const D_PLUS_ONE: usize> = IprsCode<
    BenchZipPlusTypes<Twiddle, D_PLUS_ONE>,
    PnttConfigF2_16_1_Base32_Depth1_Rate1_2,
    BinaryPolyWideningMulByScalar<Twiddle>,
>;

type SomeIprsCodeDepth1Base64<Twiddle, const D_PLUS_ONE: usize> = IprsCode<
    BenchZipPlusTypes<Twiddle, D_PLUS_ONE>,
    PnttConfigF2_16_1_Base64_Depth1_Rate1_2,
    BinaryPolyWideningMulByScalar<Twiddle>,
>;

type SomeIprsCodeDepth2Rate1_4<Twiddle, const D_PLUS_ONE: usize> = IprsCode<
    BenchZipPlusTypes<Twiddle, D_PLUS_ONE>,
    PnttConfigF2_16_1_Depth2_Rate1_4,
    BinaryPolyWideningMulByScalar<Twiddle>,
>;

type SomeIprsCodeDepth1Base16Rate1_4<Twiddle, const D_PLUS_ONE: usize> = IprsCode<
    BenchZipPlusTypes<Twiddle, D_PLUS_ONE>,
    PnttConfigF2_16_1_Base16_Depth1_Rate1_4,
    BinaryPolyWideningMulByScalar<Twiddle>,
>;

type SomeIprsCodeDepth1Base32Rate1_4<Twiddle, const D_PLUS_ONE: usize> = IprsCode<
    BenchZipPlusTypes<Twiddle, D_PLUS_ONE>,
    PnttConfigF2_16_1_Base32_Depth1_Rate1_4,
    BinaryPolyWideningMulByScalar<Twiddle>,
>;

type SomeIprsCodeDepth1Base64Rate1_4<Twiddle, const D_PLUS_ONE: usize> = IprsCode<
    BenchZipPlusTypes<Twiddle, D_PLUS_ONE>,
    PnttConfigF2_16_1_Base64_Depth1_Rate1_4,
    BinaryPolyWideningMulByScalar<Twiddle>,
>;

type SomeIprsCodeF12289Depth3<Twiddle, const D_PLUS_ONE: usize> = IprsCode<
    BenchZipPlusTypes<Twiddle, D_PLUS_ONE>,
    PnttConfigF12289_Depth3_Rate1_2,
    BinaryPolyWideningMulByScalar<Twiddle>,
>;

/// F257 config for message size 2^7 with depth 2 (base matrix 4x2)
type SomeIprsCodeF257Depth2<Twiddle, const D_PLUS_ONE: usize> = IprsCode<
    BenchZipPlusTypes<Twiddle, D_PLUS_ONE>,
    PnttConfigF257_Depth2_Rate1_2,
    BinaryPolyWideningMulByScalar<Twiddle>,
>;

/// F257 config for message size 2^7 with depth 1 (base matrix 32x16)
type SomeIprsCodeF257Base16Depth1<Twiddle, const D_PLUS_ONE: usize> = IprsCode<
    BenchZipPlusTypes<Twiddle, D_PLUS_ONE>,
    PnttConfigF257_Base16_Depth1_Rate1_2,
    BinaryPolyWideningMulByScalar<Twiddle>,
>;

/// F257 config for message size 2^6, yielding 32 rows for poly_size=2^11
type SomeIprsCodeF257Base8Depth1<Twiddle, const D_PLUS_ONE: usize> = IprsCode<
    BenchZipPlusTypes<Twiddle, D_PLUS_ONE>,
    PnttConfigF257_Base8_Depth1_Rate1_2,
    BinaryPolyWideningMulByScalar<Twiddle>,
>;

/// F65537 config for message size 2^12, yielding 16 rows for poly_size=2^16
type SomeIprsCodeF65537Base8Depth3<Twiddle, const D_PLUS_ONE: usize> = IprsCode<
    BenchZipPlusTypes<Twiddle, D_PLUS_ONE>,
    PnttConfigF2_16_1_Base8_Depth3_Rate1_2,
    BinaryPolyWideningMulByScalar<Twiddle>,
>;

fn zip_plus_benchmarks_raa(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip+ RAA");
    do_bench::<BenchZipPlusTypes<i32, 32>, SomeRaaCode<_>, UNCHECKED>(&mut group);
    // Skipped 64-bit benchmarks.
    // do_bench::<BenchZipPlusTypes<i32, 64>, SomeRaaCode<_>, UNCHECKED>(&mut group);

    group.finish();
}

fn zip_plus_benchmarks_iprs(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip+ IPRS");

    do_bench_iprs_matrices::<BenchZipPlusTypes<i64, 32>, SomeIprsCodeDepth2<i64, 32>, UNCHECKED>(
        &mut group,
    );
    // Skipped 64-bit benchmarks.
    // do_bench_iprs_matrices::<BenchZipPlusTypes<i64, 64>, SomeIprsCodeDepth2<i64, 64>, UNCHECKED>(
    //     &mut group,
    // );

    group.finish();
}

fn zip_plus_benchmarks_iprs_matrix_shapes(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip+ IPRS Matrix Shapes");

    do_bench_iprs_matrix_shapes::<
        BenchZipPlusTypes<i64, 32>,
        SomeIprsCodeDepth1Base16<i64, 32>,
        UNCHECKED,
    >(&mut group);
    do_bench_iprs_matrix_shapes::<
        BenchZipPlusTypes<i64, 32>,
        SomeIprsCodeDepth1Base32<i64, 32>,
        UNCHECKED,
    >(&mut group);
    do_bench_iprs_matrix_shapes::<
        BenchZipPlusTypes<i64, 32>,
        SomeIprsCodeDepth1Base64<i64, 32>,
        UNCHECKED,
    >(&mut group);

    group.finish();
}

fn zip_plus_benchmarks_iprs_rate1_4(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip+ IPRS rate1_4");

    do_bench_iprs_matrices::<
        BenchZipPlusTypes<i64, 32>,
        SomeIprsCodeDepth2Rate1_4<i64, 32>,
        UNCHECKED,
    >(&mut group);

    // Skipped 64-bit benchmarks.
    // do_bench_iprs_matrices::<
    //     BenchZipPlusTypes<i64, 64>,
    //     SomeIprsCodeDepth2Rate1_4<i64, 64>,
    //     UNCHECKED,
    // >(&mut group);

    group.finish();
}

fn zip_plus_benchmarks_iprs_rate1_4_matrix_shapes(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip+ IPRS rate1_4 Matrix Shapes");

    do_bench_iprs_matrix_shapes::<
        BenchZipPlusTypes<i64, 32>,
        SomeIprsCodeDepth1Base16Rate1_4<i64, 32>,
        UNCHECKED,
    >(&mut group);
    do_bench_iprs_matrix_shapes::<
        BenchZipPlusTypes<i64, 32>,
        SomeIprsCodeDepth1Base32Rate1_4<i64, 32>,
        UNCHECKED,
    >(&mut group);
    do_bench_iprs_matrix_shapes::<
        BenchZipPlusTypes<i64, 32>,
        SomeIprsCodeDepth1Base64Rate1_4<i64, 32>,
        UNCHECKED,
    >(&mut group);

    group.finish();
}

fn zip_plus_benchmarks_iprs_f12289_depth3(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip+ IPRS F12289 Depth3");

    do_bench_iprs_matrices::<
        BenchZipPlusTypes<i64, 32>,
        SomeIprsCodeF12289Depth3<i64, 32>,
        UNCHECKED,
    >(&mut group);

    group.finish();
}

/// Benchmarks for F65537 IPRS configuration with depth 3:
/// - 2^4 x 2^12 = 16 rows x 4096 message size (using Base8_Depth3)
/// Total poly_size = 2^16
fn zip_plus_benchmarks_iprs_f65537_depth3(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip+ IPRS F65537 Depth3");

    // 16 rows x 4096 cols (2^4 x 2^12)
    do_bench_iprs_f257_matrix_shapes::<
        BenchZipPlusTypes<i64, 32>,
        SomeIprsCodeF65537Base8Depth3<i64, 32>,
        UNCHECKED,
        16,
    >(&mut group, "F65537_Base8_Depth3");

    group.finish();
}

/// Benchmarks for F257 IPRS configurations with matrix sizes:
/// - 2^4 x 2^7 = 16 rows x 128 message size (using Depth2, base matrix 4x2)
/// - 2^4 x 2^7 = 16 rows x 128 message size (using Base16_Depth1, base matrix 32x16)
/// - 2^5 x 2^6 = 32 rows x 64 message size (using Base8_Depth1)
///
/// Note: F257 only supports up to 256 roots of unity, so message size 2^8
/// (which would need OUTPUT_LEN=512) is not possible with this field.
fn zip_plus_benchmarks_iprs_f257_matrix_shapes(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip+ IPRS F257 Matrix Shapes");

    // 16 rows x 128 cols (2^4 x 2^7) with depth 2
    do_bench_iprs_f257_matrix_shapes::<
        BenchZipPlusTypes<i64, 32>,
        SomeIprsCodeF257Depth2<i64, 32>,
        UNCHECKED,
        11,
    >(&mut group, "F257_Depth2");

    // 16 rows x 128 cols (2^4 x 2^7) with depth 1 (larger base matrix 32x16)
    do_bench_iprs_f257_matrix_shapes::<
        BenchZipPlusTypes<i64, 32>,
        SomeIprsCodeF257Base16Depth1<i64, 32>,
        UNCHECKED,
        11,
    >(&mut group, "F257_Base16_Depth1");

    // 32 rows x 64 cols (2^5 x 2^6)
    do_bench_iprs_f257_matrix_shapes::<
        BenchZipPlusTypes<i64, 32>,
        SomeIprsCodeF257Base8Depth1<i64, 32>,
        UNCHECKED,
        11,
    >(&mut group, "F257_Base8_Depth1");

    group.finish();
}

criterion_group!(
    benches,
    zip_plus_benchmarks_raa,
    zip_plus_benchmarks_iprs,
    zip_plus_benchmarks_iprs_matrix_shapes,
    zip_plus_benchmarks_iprs_rate1_4,
    zip_plus_benchmarks_iprs_rate1_4_matrix_shapes,
    zip_plus_benchmarks_iprs_f12289_depth3,
    zip_plus_benchmarks_iprs_f65537_depth3,
    zip_plus_benchmarks_iprs_f257_matrix_shapes
);
criterion_main!(benches);
