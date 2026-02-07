#![allow(non_local_definitions)]
#![allow(clippy::eq_op, clippy::arithmetic_side_effects, clippy::unwrap_used)]

mod zip_common;

use std::hint::black_box;

use zinc_primality::MillerRabin;
use zinc_transcript::traits::ConstTranscribable;
use zinc_utils::inner_product::{MBSInnerProduct, ScalarProduct};
use zinc_utils::named::Named;
use zip_common::*;

use criterion::{Criterion, criterion_group, criterion_main};
use crypto_bigint::U64;
use crypto_primitives::{crypto_bigint_int::Int, crypto_bigint_uint::Uint};
use rand::{distr::StandardUniform, prelude::*};
use zinc_utils::UNCHECKED;
use zip_plus::{
    code::{
        LinearCode,
        iprs::{
            IprsCode,
            PnttConfigF2_16_1_Base8_Depth3_Rate1_2,
            PnttConfigF2_16_1_Base8_Depth3_Rate1_4,
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

// --- 256-bit integer types for encoding benchmarks ---

/// ZipTypes for 256-bit integer evaluations (Int<4> = 4×64 = 256 bits).
/// Codeword type is Int<5> (320 bits) to hold the widened result after
/// multiplying by i64 twiddle factors and accumulating.
struct BenchZipTypes256Bit {}
impl ZipTypes for BenchZipTypes256Bit {
    const NUM_COLUMN_OPENINGS: usize = 200;
    type Eval = Int<4>;
    type Cw = Int<5>;
    type Fmod = Uint<{ INT_LIMBS * 4 }>;
    type PrimeTest = MillerRabin;
    type Chal = i128;
    type Pt = i128;
    type CombR = Int<8>;
    type Comb = Self::CombR;
    type EvalDotChal = ScalarProduct;
    type CombDotChal = ScalarProduct;
    type ArrCombRDotChal = MBSInnerProduct;
}

/// Widening multiplication: Int<4> × i64 → Int<5>.
/// This widens the 256-bit input to 320 bits, then multiplies by the i64
/// twiddle factor.
#[derive(Clone, Default)]
struct Int4WideningMulByScalar;

impl WideningMulByScalar<Int<4>, i64> for Int4WideningMulByScalar {
    type Output = Int<5>;

    #[allow(clippy::arithmetic_side_effects)]
    fn mul_by_scalar_widen(lhs: &Int<4>, rhs: &i64) -> Self::Output {
        use zinc_utils::from_ref::FromRef;
        let wide: Int<5> = Int::<5>::from_ref(lhs);
        let rhs_wide: Int<5> = Int::<5>::from_ref(rhs);
        wide * rhs_wide
    }
}

/// IPRS code for F65537, depth 3, rate 1/2, with 256-bit integer inputs.
/// INPUT_LEN = 4096 (2^12), OUTPUT_LEN = 8192.
type IprsCode256BitRate1_2 =
    IprsCode<BenchZipTypes256Bit, PnttConfigF2_16_1_Base8_Depth3_Rate1_2, Int4WideningMulByScalar>;

/// IPRS code for F65537, depth 3, rate 1/4, with 256-bit integer inputs.
/// INPUT_LEN = 4096 (2^12), OUTPUT_LEN = 16384.
type IprsCode256BitRate1_4 =
    IprsCode<BenchZipTypes256Bit, PnttConfigF2_16_1_Base8_Depth3_Rate1_4, Int4WideningMulByScalar>;

/// Benchmark encoding a single vector of 2^12 random 256-bit integers
/// with IPRS at rate 1/2 and rate 1/4 over F_{65537}.
fn zip_benchmarks_encode_256bit(c: &mut Criterion) {
    let mut rng = ThreadRng::default();
    let mut group = c.benchmark_group("Zip Encode 256-bit");

    // Rate 1/2: 4096 → 8192
    {
        let code = IprsCode256BitRate1_2::new(1 << 12);
        let message: Vec<Int<4>> = (0..code.row_len())
            .map(|_| rng.random())
            .collect();

        group.bench_function(
            format!(
                "Encode: {} -> {}, len=4096, rate=1/2, field=F65537",
                Int::<4>::type_name(),
                Int::<5>::type_name(),
            ),
            |b| {
                b.iter(|| {
                    let encoded = code.encode(&message);
                    black_box(encoded);
                })
            },
        );
    }

    // Rate 1/4: 4096 → 16384
    {
        let code = IprsCode256BitRate1_4::new(1 << 12);
        let message: Vec<Int<4>> = (0..code.row_len())
            .map(|_| rng.random())
            .collect();

        group.bench_function(
            format!(
                "Encode: {} -> {}, len=4096, rate=1/4, field=F65537",
                Int::<4>::type_name(),
                Int::<5>::type_name(),
            ),
            |b| {
                b.iter(|| {
                    let encoded = code.encode(&message);
                    black_box(encoded);
                })
            },
        );
    }

    group.finish();
}

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
    zip_benchmarks_iprs_rate1_4_matrix_shapes,
    zip_benchmarks_encode_256bit
);
criterion_main!(benches);
