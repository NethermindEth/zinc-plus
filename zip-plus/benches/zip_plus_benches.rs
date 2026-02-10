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
            PnttConfigF2_16_1_Base8_Depth3_Rate1_2,
            PnttConfigF2_16_1_Base8_Depth3_Rate1_4,
            PnttConfigF2_16_1_Base16_Depth3_Rate1_4,
            PnttConfigF2_16_1_Base64_Depth2_Rate1_4,
            PnttConfigF2_16_1_Base16_Depth1_Rate1_2,
            PnttConfigF2_16_1_Base16_Depth1_Rate1_4,
            PnttConfigF2_16_1_Base32_Depth1_Rate1_2,
            PnttConfigF2_16_1_Base32_Depth1_Rate1_4,
            PnttConfigF2_16_1_Base64_Depth1_Rate1_2,
            PnttConfigF2_16_1_Base64_Depth1_Rate1_4,
            PnttConfigF2_16_1_Depth2_Rate1_2,
            PnttConfigF2_16_1_Depth2_Rate1_4,
            PnttConfigF2_16_1_Depth3_Rate1_2,
            PnttConfigF2_16_1_Depth3_Rate1_4,
            PnttConfigF2_16_1_Base1_Depth4_Rate1_4,
            PnttConfigF2_16_1_Base2_Depth4_Rate1_4,
            PnttConfigF2_16_1_Base4_Depth4_Rate1_4,
        },
        raa::{RaaCode, RaaConfig},
    },
    pcs::structs::ZipTypes,
};
use zip_plus::code::iprs::{
    PnttConfigF2_16_1_Base64_Depth2_Rate1_2,
    PnttConfigF2_16_1_Base128_Depth2_Rate1_2,
    PnttConfigF2_16_1_Base256_Depth2_Rate1_2,
    PnttConfigF2_16_1_Base512_Depth2_Rate1_2,
    PnttConfigF2_16_1_Base128_Depth2_Rate1_4,
    PnttConfigF2_16_1_Base256_Depth2_Rate1_4,
    PnttConfigF2_16_1_Base4_Depth2_Rate1_4,
};
use zip_plus::code::iprs::pntt::radix8::params::PnttConfigF12289_Depth3_Rate1_2;

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
    const NUM_COLUMN_OPENINGS: usize = 147;
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

/// F65537 config for message size 2^12 with depth 3 (base matrix 16x8)
type SomeIprsCodeF65537Base8Depth3<Twiddle, const D_PLUS_ONE: usize> = IprsCode<
    BenchZipPlusTypes<Twiddle, D_PLUS_ONE>,
    PnttConfigF2_16_1_Base8_Depth3_Rate1_2,
    BinaryPolyWideningMulByScalar<Twiddle>,
>;

/// F65537 config for message size 2^12 with depth 3, rate 1/4 (base matrix 32x8)
/// **Warning:** Overflows i64 at butterfly stage 2. Use the depth-2 variant below.
type SomeIprsCodeF65537Base8Depth3Rate1_4<Twiddle, const D_PLUS_ONE: usize> = IprsCode<
    BenchZipPlusTypes<Twiddle, D_PLUS_ONE>,
    PnttConfigF2_16_1_Base8_Depth3_Rate1_4,
    BinaryPolyWideningMulByScalar<Twiddle>,
>;

/// F65537 config for message size 2^12 with depth 2, rate 1/4 (base matrix 256x64)
/// This is the i64-safe alternative to the depth-3 config above.
type SomeIprsCodeF65537Base64Depth2Rate1_4<Twiddle, const D_PLUS_ONE: usize> = IprsCode<
    BenchZipPlusTypes<Twiddle, D_PLUS_ONE>,
    PnttConfigF2_16_1_Base64_Depth2_Rate1_4,
    BinaryPolyWideningMulByScalar<Twiddle>,
>;

/// F12289 config for message size 2^11 with depth 3 (base matrix 8x4)
type SomeIprsCodeF12289Depth3<Twiddle, const D_PLUS_ONE: usize> = IprsCode<
    BenchZipPlusTypes<Twiddle, D_PLUS_ONE>,
    PnttConfigF12289_Depth3_Rate1_2,
    BinaryPolyWideningMulByScalar<Twiddle>,
>;

/// F65537 depth-2 rate-1/4 config for message size 2^13 (base matrix 512x128)
type SomeIprsCodeF65537Base128Depth2Rate1_4<Twiddle, const D_PLUS_ONE: usize> = IprsCode<
    BenchZipPlusTypes<Twiddle, D_PLUS_ONE>,
    PnttConfigF2_16_1_Base128_Depth2_Rate1_4,
    BinaryPolyWideningMulByScalar<Twiddle>,
>;

/// F65537 depth-2 rate-1/4 config for message size 2^14 (base matrix 1024x256)
type SomeIprsCodeF65537Base256Depth2Rate1_4<Twiddle, const D_PLUS_ONE: usize> = IprsCode<
    BenchZipPlusTypes<Twiddle, D_PLUS_ONE>,
    PnttConfigF2_16_1_Base256_Depth2_Rate1_4,
    BinaryPolyWideningMulByScalar<Twiddle>,
>;

/// F65537 depth-1 rate-1/4 config for message size 2^8 (base matrix 128x32)
type SomeIprsCodeF65537Base32Depth1Rate1_4<Twiddle, const D_PLUS_ONE: usize> = IprsCode<
    BenchZipPlusTypes<Twiddle, D_PLUS_ONE>,
    PnttConfigF2_16_1_Base32_Depth1_Rate1_4,
    BinaryPolyWideningMulByScalar<Twiddle>,
>;

/// F65537 depth-2 rate-1/4 config for message size 2^8 (base matrix 16x4)
type SomeIprsCodeF65537Base4Depth2Rate1_4<Twiddle, const D_PLUS_ONE: usize> = IprsCode<
    BenchZipPlusTypes<Twiddle, D_PLUS_ONE>,
    PnttConfigF2_16_1_Base4_Depth2_Rate1_4,
    BinaryPolyWideningMulByScalar<Twiddle>,
>;

/// F65537 depth-3 rate-1/4 config for message size 2^13 (base matrix 64x16)
/// Requires i128 codeword coefficients to avoid overflow.
type SomeIprsCodeF65537Base16Depth3Rate1_4<Twiddle, const D_PLUS_ONE: usize> = IprsCode<
    BenchZipPlusTypes<Twiddle, D_PLUS_ONE>,
    PnttConfigF2_16_1_Base16_Depth3_Rate1_4,
    BinaryPolyWideningMulByScalar<Twiddle>,
>;

/// F65537 depth-3 rate-1/4 config for message size 2^14 (base matrix 128x32)
/// Requires i128 codeword coefficients to avoid overflow.
type SomeIprsCodeF65537Depth3Rate1_4<Twiddle, const D_PLUS_ONE: usize> = IprsCode<
    BenchZipPlusTypes<Twiddle, D_PLUS_ONE>,
    PnttConfigF2_16_1_Depth3_Rate1_4,
    BinaryPolyWideningMulByScalar<Twiddle>,
>;

/// F65537 depth-4 rate-1/4 config for message size 2^12 (base matrix 4x1)
/// Requires i128 codeword coefficients (~86-bit intermediates).
type SomeIprsCodeF65537Base1Depth4Rate1_4<Twiddle, const D_PLUS_ONE: usize> = IprsCode<
    BenchZipPlusTypes<Twiddle, D_PLUS_ONE>,
    PnttConfigF2_16_1_Base1_Depth4_Rate1_4,
    BinaryPolyWideningMulByScalar<Twiddle>,
>;

/// F65537 depth-4 rate-1/4 config for message size 2^13 (base matrix 8x2)
/// Requires i128 codeword coefficients (~87-bit intermediates).
type SomeIprsCodeF65537Base2Depth4Rate1_4<Twiddle, const D_PLUS_ONE: usize> = IprsCode<
    BenchZipPlusTypes<Twiddle, D_PLUS_ONE>,
    PnttConfigF2_16_1_Base2_Depth4_Rate1_4,
    BinaryPolyWideningMulByScalar<Twiddle>,
>;

/// F65537 depth-4 rate-1/4 config for message size 2^14 (base matrix 16x4)
/// Requires i128 codeword coefficients (~88-bit intermediates).
type SomeIprsCodeF65537Base4Depth4Rate1_4<Twiddle, const D_PLUS_ONE: usize> = IprsCode<
    BenchZipPlusTypes<Twiddle, D_PLUS_ONE>,
    PnttConfigF2_16_1_Base4_Depth4_Rate1_4,
    BinaryPolyWideningMulByScalar<Twiddle>,
>;

/// F65537 depth-3 rate-1/2 config for message size 2^14 (base matrix 64x32)
type SomeIprsCodeF65537Depth3<Twiddle, const D_PLUS_ONE: usize> = IprsCode<
    BenchZipPlusTypes<Twiddle, D_PLUS_ONE>,
    PnttConfigF2_16_1_Depth3_Rate1_2,
    BinaryPolyWideningMulByScalar<Twiddle>,
>;

/// F65537 depth-2 rate-1/2 config for message size 2^12 (base matrix 128x64)
type SomeIprsCodeF65537Base64Depth2<Twiddle, const D_PLUS_ONE: usize> = IprsCode<
    BenchZipPlusTypes<Twiddle, D_PLUS_ONE>,
    PnttConfigF2_16_1_Base64_Depth2_Rate1_2,
    BinaryPolyWideningMulByScalar<Twiddle>,
>;

/// F65537 depth-2 rate-1/2 config for message size 2^13 (base matrix 256x128)
type SomeIprsCodeF65537Base128Depth2<Twiddle, const D_PLUS_ONE: usize> = IprsCode<
    BenchZipPlusTypes<Twiddle, D_PLUS_ONE>,
    PnttConfigF2_16_1_Base128_Depth2_Rate1_2,
    BinaryPolyWideningMulByScalar<Twiddle>,
>;

/// F65537 depth-2 rate-1/2 config for message size 2^14 (base matrix 512x256)
type SomeIprsCodeF65537Base256Depth2<Twiddle, const D_PLUS_ONE: usize> = IprsCode<
    BenchZipPlusTypes<Twiddle, D_PLUS_ONE>,
    PnttConfigF2_16_1_Base256_Depth2_Rate1_2,
    BinaryPolyWideningMulByScalar<Twiddle>,
>;

/// F65537 depth-2 rate-1/2 config for message size 2^15 (base matrix 1024x512)
type SomeIprsCodeF65537Base512Depth2<Twiddle, const D_PLUS_ONE: usize> = IprsCode<
    BenchZipPlusTypes<Twiddle, D_PLUS_ONE>,
    PnttConfigF2_16_1_Base512_Depth2_Rate1_2,
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

/// Benchmarks for F65537 IPRS configuration with various matrix dimensions.
/// All configurations use row_len = 4096 (2^12) with rate 1/2.
/// Matrix sizes: 2^1 x 2^12, 2^2 x 2^12, 2^3 x 2^12, 2^4 x 2^12, 2^5 x 2^12
fn zip_plus_benchmarks_iprs_f65537_depth3(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip+ IPRS F65537 Depth3");

    // 2 rows x 4096 cols (2^1 x 2^12), poly_size=2^13
    commit::<BenchZipPlusTypes<i64, 32>, SomeIprsCodeF65537Base8Depth3<i64, 32>, 13>(&mut group);
    // 4 rows x 4096 cols (2^2 x 2^12), poly_size=2^14
    commit::<BenchZipPlusTypes<i64, 32>, SomeIprsCodeF65537Base8Depth3<i64, 32>, 14>(&mut group);
    // 8 rows x 4096 cols (2^3 x 2^12), poly_size=2^15
    commit::<BenchZipPlusTypes<i64, 32>, SomeIprsCodeF65537Base8Depth3<i64, 32>, 15>(&mut group);
    // 16 rows x 4096 cols (2^4 x 2^12), poly_size=2^16
    commit::<BenchZipPlusTypes<i64, 32>, SomeIprsCodeF65537Base8Depth3<i64, 32>, 16>(&mut group);
    // 32 rows x 4096 cols (2^5 x 2^12), poly_size=2^17
    commit::<BenchZipPlusTypes<i64, 32>, SomeIprsCodeF65537Base8Depth3<i64, 32>, 17>(&mut group);

    group.finish();
}

/// Benchmarks for F65537 IPRS Depth-2 configuration with various matrix dimensions.
/// All configurations use row_len = 2048 (2^11) with rate 1/2.
/// Matrix sizes: 2^2 x 2^11, 2^3 x 2^11, 2^4 x 2^11, 2^5 x 2^11, 2^6 x 2^11
fn zip_plus_benchmarks_iprs_f65537_depth2(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip+ IPRS F65537 Depth2");

    // 1 row x 2048 cols (2^0 x 2^11), poly_size=2^11
    commit::<BenchZipPlusTypes<i64, 32>, SomeIprsCodeDepth2<i64, 32>, 11>(&mut group);
    test::<BenchZipPlusTypes<i64, 32>, SomeIprsCodeDepth2<i64, 32>, UNCHECKED, 11>(&mut group);
    evaluate::<BenchZipPlusTypes<i64, 32>, SomeIprsCodeDepth2<i64, 32>, UNCHECKED, 11>(&mut group);
    // 4 rows x 2048 cols (2^2 x 2^11), poly_size=2^13
    commit::<BenchZipPlusTypes<i64, 32>, SomeIprsCodeDepth2<i64, 32>, 13>(&mut group);
    // 8 rows x 2048 cols (2^3 x 2^11), poly_size=2^14
    commit::<BenchZipPlusTypes<i64, 32>, SomeIprsCodeDepth2<i64, 32>, 14>(&mut group);
    // 16 rows x 2048 cols (2^4 x 2^11), poly_size=2^15
    commit::<BenchZipPlusTypes<i64, 32>, SomeIprsCodeDepth2<i64, 32>, 15>(&mut group);
    // 32 rows x 2048 cols (2^5 x 2^11), poly_size=2^16
    commit::<BenchZipPlusTypes<i64, 32>, SomeIprsCodeDepth2<i64, 32>, 16>(&mut group);
    // 64 rows x 2048 cols (2^6 x 2^11), poly_size=2^17
    commit::<BenchZipPlusTypes<i64, 32>, SomeIprsCodeDepth2<i64, 32>, 17>(&mut group);

    group.finish();
}

/// Benchmarks for F12289 IPRS Depth-3 configuration with various matrix dimensions.
/// All configurations use row_len = 2048 (2^11) with rate 1/2.
/// Matrix sizes: 2^2 x 2^11, 2^3 x 2^11, 2^4 x 2^11, 2^5 x 2^11, 2^6 x 2^11
fn zip_plus_benchmarks_iprs_f12289_depth3(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip+ IPRS F12289 Depth3 2^11");

    // 4 rows x 2048 cols (2^2 x 2^11), poly_size=2^13
    commit::<BenchZipPlusTypes<i64, 32>, SomeIprsCodeF12289Depth3<i64, 32>, 13>(&mut group);
    // 8 rows x 2048 cols (2^3 x 2^11), poly_size=2^14
    commit::<BenchZipPlusTypes<i64, 32>, SomeIprsCodeF12289Depth3<i64, 32>, 14>(&mut group);
    // 16 rows x 2048 cols (2^4 x 2^11), poly_size=2^15
    commit::<BenchZipPlusTypes<i64, 32>, SomeIprsCodeF12289Depth3<i64, 32>, 15>(&mut group);
    // 32 rows x 2048 cols (2^5 x 2^11), poly_size=2^16
    commit::<BenchZipPlusTypes<i64, 32>, SomeIprsCodeF12289Depth3<i64, 32>, 16>(&mut group);
    // 64 rows x 2048 cols (2^6 x 2^11), poly_size=2^17
    commit::<BenchZipPlusTypes<i64, 32>, SomeIprsCodeF12289Depth3<i64, 32>, 17>(&mut group);

    group.finish();
}

/// Benchmarks for F65537 IPRS Depth-2 configuration with rate 1/4.
/// Uses BASE_LEN=64, BASE_DIM=256 (depth-2 alternative to avoid i64 overflow).
/// All configurations use row_len = 4096 (2^12) with rate 1/4.
/// Matrix sizes: 2^2 x 2^12, 2^3 x 2^12, 2^4 x 2^12, 2^5 x 2^12
fn zip_plus_benchmarks_iprs_f65537_depth2_rate1_4(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip+ IPRS F65537 Depth2 Rate1_4");

    // 4 rows x 4096 cols (2^2 x 2^12), poly_size=2^14
    commit::<BenchZipPlusTypes<i64, 32>, SomeIprsCodeF65537Base64Depth2Rate1_4<i64, 32>, 14>(&mut group);
    // 8 rows x 4096 cols (2^3 x 2^12), poly_size=2^15
    commit::<BenchZipPlusTypes<i64, 32>, SomeIprsCodeF65537Base64Depth2Rate1_4<i64, 32>, 15>(&mut group);
    // 16 rows x 4096 cols (2^4 x 2^12), poly_size=2^16
    commit::<BenchZipPlusTypes<i64, 32>, SomeIprsCodeF65537Base64Depth2Rate1_4<i64, 32>, 16>(&mut group);
    // 32 rows x 4096 cols (2^5 x 2^12), poly_size=2^17
    commit::<BenchZipPlusTypes<i64, 32>, SomeIprsCodeF65537Base64Depth2Rate1_4<i64, 32>, 17>(&mut group);

    group.finish();
}

/// Benchmarks for F65537 IPRS Depth-2 rate-1/2 configurations
/// with 4 rows and varying message sizes (row_len = 2^12..2^15).
/// Matrix sizes: 4×2^12, 4×2^13, 4×2^14, 4×2^15
fn zip_plus_benchmarks_iprs_f65537_depth2_wide(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip+ IPRS F65537 Depth2 Wide");

    // 4 rows x 4096 cols (4 x 2^12), poly_size=2^14
    commit::<BenchZipPlusTypes<i64, 32>, SomeIprsCodeF65537Base64Depth2<i64, 32>, 14>(&mut group);
    // 4 rows x 8192 cols (4 x 2^13), poly_size=2^15
    commit::<BenchZipPlusTypes<i64, 32>, SomeIprsCodeF65537Base128Depth2<i64, 32>, 15>(&mut group);
    // 4 rows x 16384 cols (4 x 2^14), poly_size=2^16
    commit::<BenchZipPlusTypes<i64, 32>, SomeIprsCodeF65537Base256Depth2<i64, 32>, 16>(&mut group);
    // 4 rows x 32768 cols (4 x 2^15), poly_size=2^17
    commit::<BenchZipPlusTypes<i64, 32>, SomeIprsCodeF65537Base512Depth2<i64, 32>, 17>(&mut group);

    group.finish();
}

/// Benchmarks for Zip+ commit with optimal matrix shapes.
/// Uses depth-2 IPRS codes for message sizes 2^12–2^14, and depth-3 for 2^14.
/// Matrix shapes: 2×2^12, 2×2^13, 4×2^13, 4×2^14, 8×2^14.
fn zip_plus_benchmarks_commit_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip+ Commit Comparison");

    // 2 rows × 4096 cols (2×2^12), depth 2, msg_size=2^12, poly_size=2^13
    commit::<BenchZipPlusTypes<i64, 32>, SomeIprsCodeF65537Base64Depth2<i64, 32>, 13>(&mut group);
    // 2 rows × 8192 cols (2×2^13), depth 2, msg_size=2^13, poly_size=2^14
    commit::<BenchZipPlusTypes<i64, 32>, SomeIprsCodeF65537Base128Depth2<i64, 32>, 14>(&mut group);
    // 4 rows × 8192 cols (4×2^13), depth 2, msg_size=2^13, poly_size=2^15
    commit::<BenchZipPlusTypes<i64, 32>, SomeIprsCodeF65537Base128Depth2<i64, 32>, 15>(&mut group);
    // 4 rows × 16384 cols (4×2^14), depth 3, msg_size=2^14, poly_size=2^16
    commit::<BenchZipPlusTypes<i64, 32>, SomeIprsCodeF65537Depth3<i64, 32>, 16>(&mut group);
    // 8 rows × 16384 cols (8×2^14), depth 3, msg_size=2^14, poly_size=2^17
    commit::<BenchZipPlusTypes<i64, 32>, SomeIprsCodeF65537Depth3<i64, 32>, 17>(&mut group);

    group.finish();
}

/// Benchmarks for the Zip+ test phase with optimal matrix shapes (same configs as Commit Comparison).
fn zip_plus_benchmarks_test_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip+ Test Comparison");

    // 2 rows × 4096 cols (2×2^12), depth 2, msg_size=2^12, poly_size=2^13
    test::<BenchZipPlusTypes<i64, 32>, SomeIprsCodeF65537Base64Depth2<i64, 32>, UNCHECKED, 13>(&mut group);
    // 2 rows × 8192 cols (2×2^13), depth 2, msg_size=2^13, poly_size=2^14
    test::<BenchZipPlusTypes<i64, 32>, SomeIprsCodeF65537Base128Depth2<i64, 32>, UNCHECKED, 14>(&mut group);
    // 4 rows × 8192 cols (4×2^13), depth 2, msg_size=2^13, poly_size=2^15
    test::<BenchZipPlusTypes<i64, 32>, SomeIprsCodeF65537Base128Depth2<i64, 32>, UNCHECKED, 15>(&mut group);
    // 4 rows × 16384 cols (4×2^14), depth 3, msg_size=2^14, poly_size=2^16
    test::<BenchZipPlusTypes<i64, 32>, SomeIprsCodeF65537Depth3<i64, 32>, UNCHECKED, 16>(&mut group);
    // 8 rows × 16384 cols (8×2^14), depth 3, msg_size=2^14, poly_size=2^17
    test::<BenchZipPlusTypes<i64, 32>, SomeIprsCodeF65537Depth3<i64, 32>, UNCHECKED, 17>(&mut group);

    group.finish();
}

/// Benchmarks for the Zip+ evaluate phase with optimal matrix shapes (same configs as Commit Comparison).
fn zip_plus_benchmarks_evaluate_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip+ Evaluate Comparison");

    // 2 rows × 4096 cols (2×2^12), depth 2, msg_size=2^12, poly_size=2^13
    evaluate::<BenchZipPlusTypes<i64, 32>, SomeIprsCodeF65537Base64Depth2<i64, 32>, UNCHECKED, 13>(&mut group);
    // 2 rows × 8192 cols (2×2^13), depth 2, msg_size=2^13, poly_size=2^14
    evaluate::<BenchZipPlusTypes<i64, 32>, SomeIprsCodeF65537Base128Depth2<i64, 32>, UNCHECKED, 14>(&mut group);
    // 4 rows × 8192 cols (4×2^13), depth 2, msg_size=2^13, poly_size=2^15
    evaluate::<BenchZipPlusTypes<i64, 32>, SomeIprsCodeF65537Base128Depth2<i64, 32>, UNCHECKED, 15>(&mut group);
    // 4 rows × 16384 cols (4×2^14), depth 3, msg_size=2^14, poly_size=2^16
    evaluate::<BenchZipPlusTypes<i64, 32>, SomeIprsCodeF65537Depth3<i64, 32>, UNCHECKED, 16>(&mut group);
    // 8 rows × 16384 cols (8×2^14), depth 3, msg_size=2^14, poly_size=2^17
    evaluate::<BenchZipPlusTypes<i64, 32>, SomeIprsCodeF65537Depth3<i64, 32>, UNCHECKED, 17>(&mut group);

    group.finish();
}

/// Benchmarks for Zip+ commit with optimal matrix shapes at rate 1/4.
/// Uses depth-2 IPRS codes for all message sizes (2^12, 2^13, 2^14).
/// Matrix shapes: 2×2^12, 2×2^13, 4×2^13, 4×2^14, 8×2^14.
fn zip_plus_benchmarks_commit_comparison_rate1_4(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip+ Commit Comparison Rate1_4");

    // 2 rows × 4096 cols (2×2^12), depth 2, msg_size=2^12, poly_size=2^13
    commit::<BenchZipPlusTypes<i64, 32>, SomeIprsCodeF65537Base64Depth2Rate1_4<i64, 32>, 13>(&mut group);
    // 2 rows × 8192 cols (2×2^13), depth 2, msg_size=2^13, poly_size=2^14
    commit::<BenchZipPlusTypes<i64, 32>, SomeIprsCodeF65537Base128Depth2Rate1_4<i64, 32>, 14>(&mut group);
    // 4 rows × 8192 cols (4×2^13), depth 2, msg_size=2^13, poly_size=2^15
    commit::<BenchZipPlusTypes<i64, 32>, SomeIprsCodeF65537Base128Depth2Rate1_4<i64, 32>, 15>(&mut group);
    // 4 rows × 16384 cols (4×2^14), depth 2, msg_size=2^14, poly_size=2^16
    commit::<BenchZipPlusTypes<i64, 32>, SomeIprsCodeF65537Base256Depth2Rate1_4<i64, 32>, 16>(&mut group);
    // 8 rows × 16384 cols (8×2^14), depth 2, msg_size=2^14, poly_size=2^17
    commit::<BenchZipPlusTypes<i64, 32>, SomeIprsCodeF65537Base256Depth2Rate1_4<i64, 32>, 17>(&mut group);

    group.finish();
}

/// Benchmarks for the Zip+ test phase with optimal matrix shapes at rate 1/4 (same configs as Commit Comparison Rate1_4).
fn zip_plus_benchmarks_test_comparison_rate1_4(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip+ Test Comparison Rate1_4");

    // 2 rows × 4096 cols (2×2^12), depth 2, msg_size=2^12, poly_size=2^13
    test::<BenchZipPlusTypes<i64, 32>, SomeIprsCodeF65537Base64Depth2Rate1_4<i64, 32>, UNCHECKED, 13>(&mut group);
    // 2 rows × 8192 cols (2×2^13), depth 2, msg_size=2^13, poly_size=2^14
    test::<BenchZipPlusTypes<i64, 32>, SomeIprsCodeF65537Base128Depth2Rate1_4<i64, 32>, UNCHECKED, 14>(&mut group);
    // 4 rows × 8192 cols (4×2^13), depth 2, msg_size=2^13, poly_size=2^15
    test::<BenchZipPlusTypes<i64, 32>, SomeIprsCodeF65537Base128Depth2Rate1_4<i64, 32>, UNCHECKED, 15>(&mut group);
    // 4 rows × 16384 cols (4×2^14), depth 2, msg_size=2^14, poly_size=2^16
    test::<BenchZipPlusTypes<i64, 32>, SomeIprsCodeF65537Base256Depth2Rate1_4<i64, 32>, UNCHECKED, 16>(&mut group);
    // 8 rows × 16384 cols (8×2^14), depth 2, msg_size=2^14, poly_size=2^17
    test::<BenchZipPlusTypes<i64, 32>, SomeIprsCodeF65537Base256Depth2Rate1_4<i64, 32>, UNCHECKED, 17>(&mut group);

    group.finish();
}

/// Benchmarks for the Zip+ evaluate phase with optimal matrix shapes at rate 1/4 (same configs as Commit Comparison Rate1_4).
fn zip_plus_benchmarks_evaluate_comparison_rate1_4(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip+ Evaluate Comparison Rate1_4");

    // 2 rows × 4096 cols (2×2^12), depth 2, msg_size=2^12, poly_size=2^13
    evaluate::<BenchZipPlusTypes<i64, 32>, SomeIprsCodeF65537Base64Depth2Rate1_4<i64, 32>, UNCHECKED, 13>(&mut group);
    // 2 rows × 8192 cols (2×2^13), depth 2, msg_size=2^13, poly_size=2^14
    evaluate::<BenchZipPlusTypes<i64, 32>, SomeIprsCodeF65537Base128Depth2Rate1_4<i64, 32>, UNCHECKED, 14>(&mut group);
    // 4 rows × 8192 cols (4×2^13), depth 2, msg_size=2^13, poly_size=2^15
    evaluate::<BenchZipPlusTypes<i64, 32>, SomeIprsCodeF65537Base128Depth2Rate1_4<i64, 32>, UNCHECKED, 15>(&mut group);
    // 4 rows × 16384 cols (4×2^14), depth 2, msg_size=2^14, poly_size=2^16
    evaluate::<BenchZipPlusTypes<i64, 32>, SomeIprsCodeF65537Base256Depth2Rate1_4<i64, 32>, UNCHECKED, 16>(&mut group);
    // 8 rows × 16384 cols (8×2^14), depth 2, msg_size=2^14, poly_size=2^17
    evaluate::<BenchZipPlusTypes<i64, 32>, SomeIprsCodeF65537Base256Depth2Rate1_4<i64, 32>, UNCHECKED, 17>(&mut group);

    group.finish();
}

/// Benchmarks for Zip+ commit with optimal matrix shapes at rate 1/4, depth 3.
/// Uses depth-3 IPRS codes with i128 coefficients to avoid overflow.
/// Matrix shapes: 2×2^12, 2×2^13, 4×2^13, 4×2^14, 8×2^14.
fn zip_plus_benchmarks_commit_comparison_rate1_4_depth3(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip+ Commit Comparison Rate1_4 Depth3");

    // 2 rows × 4096 cols (2×2^12), depth 3, msg_size=2^12, poly_size=2^13
    commit::<BenchZipPlusTypes<i128, 32>, SomeIprsCodeF65537Base8Depth3Rate1_4<i128, 32>, 13>(&mut group);
    // 2 rows × 8192 cols (2×2^13), depth 3, msg_size=2^13, poly_size=2^14
    commit::<BenchZipPlusTypes<i128, 32>, SomeIprsCodeF65537Base16Depth3Rate1_4<i128, 32>, 14>(&mut group);
    // 4 rows × 8192 cols (4×2^13), depth 3, msg_size=2^13, poly_size=2^15
    commit::<BenchZipPlusTypes<i128, 32>, SomeIprsCodeF65537Base16Depth3Rate1_4<i128, 32>, 15>(&mut group);
    // 4 rows × 16384 cols (4×2^14), depth 3, msg_size=2^14, poly_size=2^16
    commit::<BenchZipPlusTypes<i128, 32>, SomeIprsCodeF65537Depth3Rate1_4<i128, 32>, 16>(&mut group);
    // 8 rows × 16384 cols (8×2^14), depth 3, msg_size=2^14, poly_size=2^17
    commit::<BenchZipPlusTypes<i128, 32>, SomeIprsCodeF65537Depth3Rate1_4<i128, 32>, 17>(&mut group);

    group.finish();
}

/// Benchmarks for the Zip+ test phase with optimal matrix shapes at rate 1/4 depth 3.
fn zip_plus_benchmarks_test_comparison_rate1_4_depth3(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip+ Test Comparison Rate1_4 Depth3");

    // 2 rows × 4096 cols (2×2^12), depth 3, msg_size=2^12, poly_size=2^13
    test::<BenchZipPlusTypes<i128, 32>, SomeIprsCodeF65537Base8Depth3Rate1_4<i128, 32>, UNCHECKED, 13>(&mut group);
    // 2 rows × 8192 cols (2×2^13), depth 3, msg_size=2^13, poly_size=2^14
    test::<BenchZipPlusTypes<i128, 32>, SomeIprsCodeF65537Base16Depth3Rate1_4<i128, 32>, UNCHECKED, 14>(&mut group);
    // 4 rows × 8192 cols (4×2^13), depth 3, msg_size=2^13, poly_size=2^15
    test::<BenchZipPlusTypes<i128, 32>, SomeIprsCodeF65537Base16Depth3Rate1_4<i128, 32>, UNCHECKED, 15>(&mut group);
    // 4 rows × 16384 cols (4×2^14), depth 3, msg_size=2^14, poly_size=2^16
    test::<BenchZipPlusTypes<i128, 32>, SomeIprsCodeF65537Depth3Rate1_4<i128, 32>, UNCHECKED, 16>(&mut group);
    // 8 rows × 16384 cols (8×2^14), depth 3, msg_size=2^14, poly_size=2^17
    test::<BenchZipPlusTypes<i128, 32>, SomeIprsCodeF65537Depth3Rate1_4<i128, 32>, UNCHECKED, 17>(&mut group);

    group.finish();
}

/// Benchmarks for the Zip+ evaluate phase with optimal matrix shapes at rate 1/4 depth 3.
fn zip_plus_benchmarks_evaluate_comparison_rate1_4_depth3(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip+ Evaluate Comparison Rate1_4 Depth3");

    // 2 rows × 4096 cols (2×2^12), depth 3, msg_size=2^12, poly_size=2^13
    evaluate::<BenchZipPlusTypes<i128, 32>, SomeIprsCodeF65537Base8Depth3Rate1_4<i128, 32>, UNCHECKED, 13>(&mut group);
    // 2 rows × 8192 cols (2×2^13), depth 3, msg_size=2^13, poly_size=2^14
    evaluate::<BenchZipPlusTypes<i128, 32>, SomeIprsCodeF65537Base16Depth3Rate1_4<i128, 32>, UNCHECKED, 14>(&mut group);
    // 4 rows × 8192 cols (4×2^13), depth 3, msg_size=2^13, poly_size=2^15
    evaluate::<BenchZipPlusTypes<i128, 32>, SomeIprsCodeF65537Base16Depth3Rate1_4<i128, 32>, UNCHECKED, 15>(&mut group);
    // 4 rows × 16384 cols (4×2^14), depth 3, msg_size=2^14, poly_size=2^16
    evaluate::<BenchZipPlusTypes<i128, 32>, SomeIprsCodeF65537Depth3Rate1_4<i128, 32>, UNCHECKED, 16>(&mut group);
    // 8 rows × 16384 cols (8×2^14), depth 3, msg_size=2^14, poly_size=2^17
    evaluate::<BenchZipPlusTypes<i128, 32>, SomeIprsCodeF65537Depth3Rate1_4<i128, 32>, UNCHECKED, 17>(&mut group);

    group.finish();
}

/// Benchmarks for Zip+ commit with optimal matrix shapes at rate 1/4, depth 4.
/// Uses depth-4 IPRS codes with i128 coefficients (smallest possible base matrices).
/// Matrix shapes: 2×2^12, 2×2^13, 4×2^13, 4×2^14, 8×2^14.
fn zip_plus_benchmarks_commit_comparison_rate1_4_depth4(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip+ Commit Comparison Rate1_4 Depth4");

    // 2 rows × 4096 cols (2×2^12), depth 4, msg_size=2^12, poly_size=2^13
    commit::<BenchZipPlusTypes<i128, 32>, SomeIprsCodeF65537Base1Depth4Rate1_4<i128, 32>, 13>(&mut group);
    // 2 rows × 8192 cols (2×2^13), depth 4, msg_size=2^13, poly_size=2^14
    commit::<BenchZipPlusTypes<i128, 32>, SomeIprsCodeF65537Base2Depth4Rate1_4<i128, 32>, 14>(&mut group);
    // 4 rows × 8192 cols (4×2^13), depth 4, msg_size=2^13, poly_size=2^15
    commit::<BenchZipPlusTypes<i128, 32>, SomeIprsCodeF65537Base2Depth4Rate1_4<i128, 32>, 15>(&mut group);
    // 4 rows × 16384 cols (4×2^14), depth 4, msg_size=2^14, poly_size=2^16
    commit::<BenchZipPlusTypes<i128, 32>, SomeIprsCodeF65537Base4Depth4Rate1_4<i128, 32>, 16>(&mut group);
    // 8 rows × 16384 cols (8×2^14), depth 4, msg_size=2^14, poly_size=2^17
    commit::<BenchZipPlusTypes<i128, 32>, SomeIprsCodeF65537Base4Depth4Rate1_4<i128, 32>, 17>(&mut group);

    group.finish();
}

/// Benchmarks for the Zip+ test phase with optimal matrix shapes at rate 1/4 depth 4.
fn zip_plus_benchmarks_test_comparison_rate1_4_depth4(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip+ Test Comparison Rate1_4 Depth4");

    // 2 rows × 4096 cols (2×2^12), depth 4, msg_size=2^12, poly_size=2^13
    test::<BenchZipPlusTypes<i128, 32>, SomeIprsCodeF65537Base1Depth4Rate1_4<i128, 32>, UNCHECKED, 13>(&mut group);
    // 2 rows × 8192 cols (2×2^13), depth 4, msg_size=2^13, poly_size=2^14
    test::<BenchZipPlusTypes<i128, 32>, SomeIprsCodeF65537Base2Depth4Rate1_4<i128, 32>, UNCHECKED, 14>(&mut group);
    // 4 rows × 8192 cols (4×2^13), depth 4, msg_size=2^13, poly_size=2^15
    test::<BenchZipPlusTypes<i128, 32>, SomeIprsCodeF65537Base2Depth4Rate1_4<i128, 32>, UNCHECKED, 15>(&mut group);
    // 4 rows × 16384 cols (4×2^14), depth 4, msg_size=2^14, poly_size=2^16
    test::<BenchZipPlusTypes<i128, 32>, SomeIprsCodeF65537Base4Depth4Rate1_4<i128, 32>, UNCHECKED, 16>(&mut group);
    // 8 rows × 16384 cols (8×2^14), depth 4, msg_size=2^14, poly_size=2^17
    test::<BenchZipPlusTypes<i128, 32>, SomeIprsCodeF65537Base4Depth4Rate1_4<i128, 32>, UNCHECKED, 17>(&mut group);

    group.finish();
}

/// Benchmarks for the Zip+ evaluate phase with optimal matrix shapes at rate 1/4 depth 4.
fn zip_plus_benchmarks_evaluate_comparison_rate1_4_depth4(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip+ Evaluate Comparison Rate1_4 Depth4");

    // 2 rows × 4096 cols (2×2^12), depth 4, msg_size=2^12, poly_size=2^13
    evaluate::<BenchZipPlusTypes<i128, 32>, SomeIprsCodeF65537Base1Depth4Rate1_4<i128, 32>, UNCHECKED, 13>(&mut group);
    // 2 rows × 8192 cols (2×2^13), depth 4, msg_size=2^13, poly_size=2^14
    evaluate::<BenchZipPlusTypes<i128, 32>, SomeIprsCodeF65537Base2Depth4Rate1_4<i128, 32>, UNCHECKED, 14>(&mut group);
    // 4 rows × 8192 cols (4×2^13), depth 4, msg_size=2^13, poly_size=2^15
    evaluate::<BenchZipPlusTypes<i128, 32>, SomeIprsCodeF65537Base2Depth4Rate1_4<i128, 32>, UNCHECKED, 15>(&mut group);
    // 4 rows × 16384 cols (4×2^14), depth 4, msg_size=2^14, poly_size=2^16
    evaluate::<BenchZipPlusTypes<i128, 32>, SomeIprsCodeF65537Base4Depth4Rate1_4<i128, 32>, UNCHECKED, 16>(&mut group);
    // 8 rows × 16384 cols (8×2^14), depth 4, msg_size=2^14, poly_size=2^17
    evaluate::<BenchZipPlusTypes<i128, 32>, SomeIprsCodeF65537Base4Depth4Rate1_4<i128, 32>, UNCHECKED, 17>(&mut group);

    group.finish();
}

/// Benchmarks for Zip+ commit of 10 polynomials with 8 variables (poly_size=2^8),
/// matrix shape 1×256, with degree <32 polynomial entries.
/// Uses depth-1 and depth-2 IPRS codes with rate 1/4.
fn zip_plus_benchmarks_commit_10_polys_8vars(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip+ Commit 10 Polys 8 Vars");

    // Depth 1, rate 1/4: msg_size=2^8, 1 row × 256 cols, poly_size=2^8
    commit_n_polys::<BenchZipPlusTypes<i64, 32>, SomeIprsCodeF65537Base32Depth1Rate1_4<i64, 32>, 8, 10>(&mut group, "IPRS-1-1/4-F65537");
    // Depth 2, rate 1/4: msg_size=2^8, 1 row × 256 cols, poly_size=2^8
    commit_n_polys::<BenchZipPlusTypes<i64, 32>, SomeIprsCodeF65537Base4Depth2Rate1_4<i64, 32>, 8, 10>(&mut group, "IPRS-2-1/4-F65537");

    group.finish();
}

criterion_group!(
    benches,
    zip_plus_benchmarks_raa,
    zip_plus_benchmarks_iprs,
    zip_plus_benchmarks_iprs_matrix_shapes,
    zip_plus_benchmarks_iprs_rate1_4,
    zip_plus_benchmarks_iprs_rate1_4_matrix_shapes,
    zip_plus_benchmarks_iprs_f65537_depth3,
    zip_plus_benchmarks_iprs_f65537_depth2,
    zip_plus_benchmarks_iprs_f12289_depth3,
    zip_plus_benchmarks_iprs_f65537_depth2_rate1_4,
    zip_plus_benchmarks_iprs_f65537_depth2_wide,
    zip_plus_benchmarks_commit_comparison,
    zip_plus_benchmarks_test_comparison,
    zip_plus_benchmarks_evaluate_comparison,
    zip_plus_benchmarks_commit_comparison_rate1_4,
    zip_plus_benchmarks_test_comparison_rate1_4,
    zip_plus_benchmarks_evaluate_comparison_rate1_4,
    zip_plus_benchmarks_commit_comparison_rate1_4_depth3,
    zip_plus_benchmarks_test_comparison_rate1_4_depth3,
    zip_plus_benchmarks_evaluate_comparison_rate1_4_depth3,
    zip_plus_benchmarks_commit_comparison_rate1_4_depth4,
    zip_plus_benchmarks_test_comparison_rate1_4_depth4,
    zip_plus_benchmarks_evaluate_comparison_rate1_4_depth4,
    zip_plus_benchmarks_commit_10_polys_8vars
);
criterion_main!(benches);
