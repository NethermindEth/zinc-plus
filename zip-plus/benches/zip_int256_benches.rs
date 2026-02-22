#![allow(non_local_definitions)]
#![allow(clippy::eq_op, clippy::arithmetic_side_effects, clippy::unwrap_used)]

//! Benchmarks for Zip with 256-bit integer evaluations (`Int<4>`) using IPRS codes.
//!
//! All benchmarks use num_rows=1 (row_len = poly_size), with poly_size from 2^7 to 2^10.
//!
//! Type sizing rationale (on 64-bit platforms, where `U64::LIMBS = 1`):
//! - `Eval  = Int<4>`  = 256 bits — the polynomial evaluation ring
//! - `Cw    = Int<5>`  = 320 bits — codeword ring (wider than Eval for IPRS encoding headroom)
//! - `Chal  = i128`    = 128 bits — challenge coefficients
//! - `CombR = Int<8>`  = 512 bits — linear combination ring
//!     Cw_bits (320) + Chal_bits (128) = 448; plus accumulation headroom → 512
//! - `Fmod  = Uint<8>` = 512 bits — modulus search space (must be >= CombR width)
//!
//! IPRS configs (num_rows=1, so row_len = poly_size):
//!   2^7  = 128  → PnttConfigF2_16R4B16<1>
//!   2^8  = 256  → PnttConfigF2_16R4B32<1>
//!   2^9  = 512  → PnttConfigF2_16R4B64<1>
//!   2^10 = 1024 → PnttConfigF2_16R4B16<2>

use zinc_primality::MillerRabin;
use zinc_transcript::traits::ConstTranscribable;
use zinc_utils::{
    UNCHECKED,
    inner_product::{MBSInnerProduct, ScalarProduct},
    mul_by_scalar::ScalarWideningMulByScalar,
    named::Named,
};

use criterion::{BenchmarkGroup, Criterion, criterion_group, criterion_main, measurement::WallTime};
use crypto_bigint::U64;
use crypto_primitives::{
    IntoWithConfig, PrimeField,
    crypto_bigint_int::Int,
    crypto_bigint_monty::MontyField,
    crypto_bigint_uint::Uint,
};
use num_traits::One;
use rand::prelude::*;
use std::{
    hint::black_box,
    time::{Duration, Instant},
};
use zinc_poly::mle::{DenseMultilinearExtension, MultilinearExtensionRand};
use zip_plus::{
    code::{
        LinearCode,
        iprs::{
            IprsCode, PnttConfigF2_16R4B16, PnttConfigF2_16R4B32, PnttConfigF2_16R4B64,
        },
    },
    pcs::structs::{ZipPlus, ZipTypes},
};

const INT_LIMBS: usize = U64::LIMBS;
// F must be at least as wide as Fmod so the sampled prime fits.
type F = MontyField<{ INT_LIMBS * 8 }>;

/// ZipTypes configuration with 256-bit integer evaluations.
struct IntZipTypes256 {}

impl ZipTypes for IntZipTypes256 {
    const NUM_COLUMN_OPENINGS: usize = 200;

    // 256-bit evaluations (4 × 64-bit limbs)
    type Eval = Int<{ INT_LIMBS * 4 }>;

    // 320-bit codewords — wider than Eval for IPRS encoding accumulation
    type Cw = Int<{ INT_LIMBS * 5 }>;

    // 512-bit unsigned modulus search space
    type Fmod = Uint<{ INT_LIMBS * 8 }>;
    type PrimeTest = MillerRabin;

    // 128-bit challenges
    type Chal = i128;
    type Pt = i128;

    // 512-bit linear combination ring:
    //   Cw (320) × Chal (128) = 448-bit product + accumulation headroom
    type CombR = Int<{ INT_LIMBS * 8 }>;
    type Comb = Self::CombR;

    type EvalDotChal = ScalarProduct;
    type CombDotChal = ScalarProduct;
    type ArrCombRDotChal = MBSInnerProduct;
}

// IPRS code types for each row length.
// ScalarWideningMulByScalar<Cw> widens Eval (Int<4>) × PnttInt (i64) → Cw (Int<5>).

/// Row length 2^7 = 128
type Iprs7 =
    IprsCode<IntZipTypes256, PnttConfigF2_16R4B16<1>, ScalarWideningMulByScalar<Int<{ INT_LIMBS * 5 }>>, UNCHECKED>;

/// Row length 2^8 = 256
type Iprs8 =
    IprsCode<IntZipTypes256, PnttConfigF2_16R4B32<1>, ScalarWideningMulByScalar<Int<{ INT_LIMBS * 5 }>>, UNCHECKED>;

/// Row length 2^9 = 512
type Iprs9 =
    IprsCode<IntZipTypes256, PnttConfigF2_16R4B64<1>, ScalarWideningMulByScalar<Int<{ INT_LIMBS * 5 }>>, UNCHECKED>;

/// Row length 2^10 = 1024
type Iprs10 =
    IprsCode<IntZipTypes256, PnttConfigF2_16R4B16<2>, ScalarWideningMulByScalar<Int<{ INT_LIMBS * 5 }>>, UNCHECKED>;

// ---------------------------------------------------------------------------
// Benchmark helpers (num_rows=1: row_len = poly_size)
// ---------------------------------------------------------------------------

fn bench_encode<Lc: LinearCode<IntZipTypes256>, const P: usize>(
    group: &mut BenchmarkGroup<WallTime>,
) {
    let poly_size: usize = 1 << P;
    let row_len = poly_size; // num_rows = 1
    let linear_code = Lc::new(row_len);
    let params = ZipPlus::setup(poly_size, linear_code);

    group.bench_function(
        format!(
            "EncodeRows: {} -> {}, poly_size=2^{P}, 1row",
            <IntZipTypes256 as ZipTypes>::Eval::type_name(),
            <IntZipTypes256 as ZipTypes>::Cw::type_name(),
        ),
        |b| {
            let mut rng = ThreadRng::default();
            let poly = DenseMultilinearExtension::<<IntZipTypes256 as ZipTypes>::Eval>::rand(P, &mut rng);
            b.iter(|| {
                let cw = ZipPlus::encode_rows(&params, row_len, &poly);
                black_box(cw)
            })
        },
    );
}

fn bench_commit<Lc: LinearCode<IntZipTypes256>, const P: usize>(
    group: &mut BenchmarkGroup<WallTime>,
) {
    let poly_size: usize = 1 << P;
    let row_len = poly_size;
    let linear_code = Lc::new(row_len);
    let params = ZipPlus::setup(poly_size, linear_code);

    group.bench_function(
        format!(
            "Commit: Eval={}, Cw={}, Comb={}, poly_size=2^{P}, 1row",
            <IntZipTypes256 as ZipTypes>::Eval::type_name(),
            <IntZipTypes256 as ZipTypes>::Cw::type_name(),
            <IntZipTypes256 as ZipTypes>::Comb::type_name(),
        ),
        |b| {
            let mut rng = ThreadRng::default();
            b.iter_custom(|iters| {
                let mut total_duration = Duration::ZERO;
                for _ in 0..iters {
                    let poly = DenseMultilinearExtension::rand(P, &mut rng);
                    let timer = Instant::now();
                    let res = ZipPlus::commit(&params, &poly).expect("Failed to commit");
                    black_box(res);
                    total_duration += timer.elapsed();
                }
                total_duration
            })
        },
    );
}

fn bench_test<Lc: LinearCode<IntZipTypes256>, const P: usize>(
    group: &mut BenchmarkGroup<WallTime>,
) {
    let poly_size: usize = 1 << P;
    let row_len = poly_size;
    let linear_code = Lc::new(row_len);
    let params = ZipPlus::setup(poly_size, linear_code);

    let mut rng = ThreadRng::default();
    let poly = DenseMultilinearExtension::rand(P, &mut rng);
    let (data, _) = ZipPlus::commit(&params, &poly).unwrap();

    group.bench_function(
        format!(
            "Test: Eval={}, Cw={}, Comb={}, poly_size=2^{P}, 1row",
            <IntZipTypes256 as ZipTypes>::Eval::type_name(),
            <IntZipTypes256 as ZipTypes>::Cw::type_name(),
            <IntZipTypes256 as ZipTypes>::Comb::type_name(),
        ),
        |b| {
            b.iter(|| {
                let test_transcript = ZipPlus::test::<UNCHECKED>(&params, &poly, &data)
                    .expect("Test phase failed");
                black_box(test_transcript);
            })
        },
    );
}

fn bench_evaluate<Lc: LinearCode<IntZipTypes256>, const P: usize>(
    group: &mut BenchmarkGroup<WallTime>,
) {
    let poly_size: usize = 1 << P;
    let row_len = poly_size;
    let linear_code = Lc::new(row_len);
    let params = ZipPlus::setup(poly_size, linear_code);

    let mut rng = ThreadRng::default();
    let poly = DenseMultilinearExtension::rand(P, &mut rng);
    let (data, _) = ZipPlus::commit(&params, &poly).unwrap();
    let point = vec![<IntZipTypes256 as ZipTypes>::Pt::one(); P];
    let test_transcript =
        ZipPlus::test::<UNCHECKED>(&params, &poly, &data).expect("Test phase failed");

    group.bench_function(
        format!(
            "Evaluate: Eval={}, Cw={}, Comb={}, poly_size=2^{P}, 1row, modulus=({} bits)",
            <IntZipTypes256 as ZipTypes>::Eval::type_name(),
            <IntZipTypes256 as ZipTypes>::Cw::type_name(),
            <IntZipTypes256 as ZipTypes>::Comb::type_name(),
            <IntZipTypes256 as ZipTypes>::Fmod::NUM_BYTES * 8,
        ),
        |b| {
            b.iter_custom(|iters| {
                let mut total_duration = Duration::ZERO;
                for _ in 0..iters {
                    let proof = test_transcript.clone();
                    let timer = Instant::now();
                    let (eval_f, proof) =
                        ZipPlus::evaluate::<F, UNCHECKED>(&params, &poly, &point, proof)
                            .expect("Evaluation phase failed");
                    total_duration += timer.elapsed();
                    black_box((eval_f, proof));
                }
                total_duration
            })
        },
    );
}

fn bench_verify<Lc: LinearCode<IntZipTypes256>, const P: usize>(
    group: &mut BenchmarkGroup<WallTime>,
) {
    let poly_size: usize = 1 << P;
    let row_len = poly_size;
    let linear_code = Lc::new(row_len);
    let params = ZipPlus::setup(poly_size, linear_code);

    let mut rng = ThreadRng::default();
    let poly = DenseMultilinearExtension::rand(P, &mut rng);
    let (data, commitment) = ZipPlus::commit(&params, &poly).unwrap();
    let point = vec![<IntZipTypes256 as ZipTypes>::Pt::one(); P];
    let test_transcript =
        ZipPlus::test::<UNCHECKED>(&params, &poly, &data).expect("Test phase failed");
    let (eval_f, proof) =
        ZipPlus::evaluate::<F, UNCHECKED>(&params, &poly, &point, test_transcript)
            .expect("Evaluation phase failed");
    let field_cfg = *eval_f.cfg();
    let point_f: Vec<F> = point.iter().map(|v| v.into_with_cfg(&field_cfg)).collect();

    group.bench_function(
        format!(
            "Verify: Eval={}, Cw={}, Comb={}, poly_size=2^{P}, 1row, modulus=({} bits)",
            <IntZipTypes256 as ZipTypes>::Eval::type_name(),
            <IntZipTypes256 as ZipTypes>::Cw::type_name(),
            <IntZipTypes256 as ZipTypes>::Comb::type_name(),
            <IntZipTypes256 as ZipTypes>::Fmod::NUM_BYTES * 8,
        ),
        |b| {
            b.iter(|| {
                ZipPlus::verify::<_, UNCHECKED>(
                    &params,
                    &commitment,
                    &point_f,
                    &eval_f,
                    &proof,
                )
                .expect("Verification failed");
            })
        },
    );
}

// ---------------------------------------------------------------------------
// Criterion entry point
// ---------------------------------------------------------------------------

fn zip_int256_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip Int256 IPRS 1row");

    // poly_size=2^7=128, row_len=128, num_rows=1
    bench_encode::<Iprs7, 7>(&mut group);
    bench_commit::<Iprs7, 7>(&mut group);
    bench_test::<Iprs7, 7>(&mut group);
    bench_evaluate::<Iprs7, 7>(&mut group);
    bench_verify::<Iprs7, 7>(&mut group);

    // poly_size=2^8=256, row_len=256, num_rows=1
    bench_encode::<Iprs8, 8>(&mut group);
    bench_commit::<Iprs8, 8>(&mut group);
    bench_test::<Iprs8, 8>(&mut group);
    bench_evaluate::<Iprs8, 8>(&mut group);
    bench_verify::<Iprs8, 8>(&mut group);

    // poly_size=2^9=512, row_len=512, num_rows=1
    bench_encode::<Iprs9, 9>(&mut group);
    bench_commit::<Iprs9, 9>(&mut group);
    bench_test::<Iprs9, 9>(&mut group);
    bench_evaluate::<Iprs9, 9>(&mut group);
    bench_verify::<Iprs9, 9>(&mut group);

    // poly_size=2^10=1024, row_len=1024, num_rows=1
    bench_encode::<Iprs10, 10>(&mut group);
    bench_commit::<Iprs10, 10>(&mut group);
    bench_test::<Iprs10, 10>(&mut group);
    bench_evaluate::<Iprs10, 10>(&mut group);
    bench_verify::<Iprs10, 10>(&mut group);

    group.finish();
}

criterion_group!(benches, zip_int256_benchmarks);
criterion_main!(benches);
