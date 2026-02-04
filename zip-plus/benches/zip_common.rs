//! This module contains common benchmarking code for the Zip+ PCS,
//! both for Zip (integer coefficients) and Zip+ (polynomial coefficients).

#![allow(non_local_definitions)]
#![allow(clippy::eq_op, clippy::arithmetic_side_effects, clippy::unwrap_used)]

use criterion::{BenchmarkGroup, measurement::WallTime};
use crypto_bigint::U64;
use crypto_primitives::{
    DenseRowMatrix, Field, FromWithConfig, IntoWithConfig, PrimeField,
    crypto_bigint_monty::MontyField,
};
use itertools::Itertools;
use num_traits::One;
use rand::{distr::StandardUniform, prelude::*};
use std::{
    hint::black_box,
    time::{Duration, Instant},
};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use zinc_poly::mle::{DenseMultilinearExtension, MultilinearExtensionRand};
use zinc_transcript::traits::ConstTranscribable;
use zinc_utils::{cfg_iter, from_ref::FromRef, named::Named, projectable_to_field::ProjectableToField};
use zip_plus::{
    code::LinearCode,
    merkle::MerkleTree,
    pcs::structs::{ZipPlus, ZipPlusParams, ZipTypes},
};

const INT_LIMBS: usize = U64::LIMBS;
type F = MontyField<{ INT_LIMBS * 4 }>;

pub fn do_bench<Zt: ZipTypes, Lc: LinearCode<Zt>, const CHECK_FOR_OVERFLOWS: bool>(
    group: &mut BenchmarkGroup<WallTime>,
) where
    StandardUniform: Distribution<Zt::Eval> + Distribution<Zt::Cw>,
    F: for<'a> FromWithConfig<&'a Zt::Chal> + for<'a> FromWithConfig<&'a Zt::Pt>,
    <F as Field>::Inner: FromRef<Zt::Fmod>,
    Zt::Eval: ProjectableToField<F>,
    Zt::Cw: ProjectableToField<F>,
{

    // cargo bench --bench zip_plus_benches --features "asm simd parallel" -- IPRS
    
    // encode_rows::<Zt, Lc, 12>(group);
    // encode_rows::<Zt, Lc, 13>(group);
    // encode_rows::<Zt, Lc, 14>(group);
    // encode_rows::<Zt, Lc, 15>(group);
    // encode_rows::<Zt, Lc, 13>(group);

    // encode_single_row::<Zt, Lc, 128>(group);
    // encode_single_row::<Zt, Lc, 256>(group);
    // encode_single_row::<Zt, Lc, 512>(group);
    // encode_single_row::<Zt, Lc, 1024>(group);

    // merkle_root::<Zt, 12>(group);
    merkle_root::<Zt, 13>(group);
    // merkle_root::<Zt, 14>(group);
    // merkle_root::<Zt, 15>(group);
    // merkle_root::<Zt, 16>(group);

    // commit::<Zt, Lc, 12>(group);
    commit::<Zt, Lc, 13>(group);
    commit_batch::<Zt, Lc, 13, 2>(group);
    test::<Zt, Lc, CHECK_FOR_OVERFLOWS, 13>(group);
    test_batch::<Zt, Lc, CHECK_FOR_OVERFLOWS, 13, 2>(group);

    // evaluate::<Zt, Lc, CHECK_FOR_OVERFLOWS, 12>(group);
    // evaluate::<Zt, Lc, CHECK_FOR_OVERFLOWS, 13>(group);
    // evaluate::<Zt, Lc, CHECK_FOR_OVERFLOWS, 14>(group);
    // evaluate::<Zt, Lc, CHECK_FOR_OVERFLOWS, 15>(group);
    // evaluate::<Zt, Lc, CHECK_FOR_OVERFLOWS, 16>(group);

    // verify_only_test::<Zt, Lc, CHECK_FOR_OVERFLOWS, 12>(group);
    verify_only_test::<Zt, Lc, CHECK_FOR_OVERFLOWS, 13>(group);
    // verify_only_test::<Zt, Lc, CHECK_FOR_OVERFLOWS, 14>(group);
    // verify_only_test::<Zt, Lc, CHECK_FOR_OVERFLOWS, 15>(group);
    // verify_only_test::<Zt, Lc, CHECK_FOR_OVERFLOWS, 16>(group);
}

pub fn encode_rows<Zt: ZipTypes, Lc: LinearCode<Zt>, const P: usize>(
    group: &mut BenchmarkGroup<WallTime>,
) where
    StandardUniform: Distribution<Zt::Eval>,
{
    group.bench_function(
        format!(
            "EncodeRows: {} -> {}, poly_size = 2^{P}",
            Zt::Eval::type_name(),
            Zt::Cw::type_name()
        ),
        |b| {
            let mut rng = ThreadRng::default();
            let poly_size = 1 << P;
            let linear_code = Lc::new(poly_size);
            let params = ZipPlus::setup(poly_size, linear_code);
            let row_len = params.linear_code.row_len();
            let poly = DenseMultilinearExtension::<<Zt as ZipTypes>::Eval>::rand(P, &mut rng);
            b.iter(|| {
                let cw = ZipPlus::encode_rows(&params, row_len, &poly);
                black_box(cw)
            })
        },
    );
}

pub fn commit_matrix_4x1024_raa<Zt: ZipTypes, Lc: LinearCode<Zt>>(
    group: &mut BenchmarkGroup<WallTime>,
) where
    StandardUniform: Distribution<Zt::Eval>,
{
    const NUM_ROWS: usize = 4;
    const ROW_LEN: usize = 1024;
    const P: usize = 12; // 2^12 = 4096 = 4 * 1024
    const POLY_SIZE_FOR_ROW_LEN: usize = 1 << 20; // Ensures RAA row_len = 1024

    let mut rng = ThreadRng::default();
    let poly_size = 1 << P;
    let linear_code = Lc::new(POLY_SIZE_FOR_ROW_LEN);
    let row_len = linear_code.row_len();
    assert_eq!(
        row_len, ROW_LEN,
        "Expected row_len to be {ROW_LEN}, got {row_len}"
    );
    let params = ZipPlusParams::new(P, NUM_ROWS, linear_code);

    group.bench_function(
        format!(
            "CommitMatrix/4x1024/{}",
            Zt::Eval::type_name()
        ),
        |b| {
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

pub fn commit_matrix_8x1024_raa<Zt: ZipTypes, Lc: LinearCode<Zt>>(
    group: &mut BenchmarkGroup<WallTime>,
) where
    StandardUniform: Distribution<Zt::Eval>,
{
    const NUM_ROWS: usize = 8;
    const ROW_LEN: usize = 1024;
    const P: usize = 13; // 2^13 = 8192 = 8 * 1024
    const POLY_SIZE_FOR_ROW_LEN: usize = 1 << 20; // Ensures RAA row_len = 1024

    let mut rng = ThreadRng::default();
    let poly_size = 1 << P;
    let linear_code = Lc::new(POLY_SIZE_FOR_ROW_LEN);
    let row_len = linear_code.row_len();
    assert_eq!(
        row_len, ROW_LEN,
        "Expected row_len to be {ROW_LEN}, got {row_len}"
    );
    let params = ZipPlusParams::new(P, NUM_ROWS, linear_code);

    group.bench_function(
        format!(
            "CommitMatrix/8x1024/{}",
            Zt::Eval::type_name()
        ),
        |b| {
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

pub fn commit_matrix_32x256_raa<Zt: ZipTypes, Lc: LinearCode<Zt>>(
    group: &mut BenchmarkGroup<WallTime>,
) where
    StandardUniform: Distribution<Zt::Eval>,
{
    const NUM_ROWS: usize = 32;
    const ROW_LEN: usize = 256;
    const P: usize = 13; // 2^13 = 8192 = 32 * 256
    const POLY_SIZE_FOR_ROW_LEN: usize = 1 << 16; // Ensures RAA row_len = 256

    let mut rng = ThreadRng::default();
    let poly_size = 1 << P;
    let linear_code = Lc::new(POLY_SIZE_FOR_ROW_LEN);
    let row_len = linear_code.row_len();
    assert_eq!(
        row_len, ROW_LEN,
        "Expected row_len to be {ROW_LEN}, got {row_len}"
    );
    let params = ZipPlusParams::new(P, NUM_ROWS, linear_code);

    group.bench_function(
        format!(
            "CommitMatrix/32x256/{}",
            Zt::Eval::type_name()
        ),
        |b| {
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

pub fn test_matrix_8x1024_raa<Zt: ZipTypes, Lc: LinearCode<Zt>, const CHECK_FOR_OVERFLOWS: bool>(
    group: &mut BenchmarkGroup<WallTime>,
) where
    StandardUniform: Distribution<Zt::Eval>,
{
    const NUM_ROWS: usize = 8;
    const ROW_LEN: usize = 1024;
    const P: usize = 13; // 2^13 = 8192 = 8 * 1024
    const POLY_SIZE_FOR_ROW_LEN: usize = 1 << 20; // Ensures RAA row_len = 1024

    let mut rng = ThreadRng::default();
    let poly_size = 1 << P;
    let linear_code = Lc::new(POLY_SIZE_FOR_ROW_LEN);
    let row_len = linear_code.row_len();
    assert_eq!(
        row_len, ROW_LEN,
        "Expected row_len to be {ROW_LEN}, got {row_len}"
    );
    let params = ZipPlusParams::new(P, NUM_ROWS, linear_code);

    let poly = DenseMultilinearExtension::rand(P, &mut rng);
    let (data, _) = ZipPlus::commit(&params, &poly).unwrap();

    group.bench_function(
        format!(
            "TestMatrix/8x1024/{}",
            Zt::Eval::type_name()
        ),
        |b| {
            b.iter(|| {
                let test_transcript =
                    ZipPlus::test::<CHECK_FOR_OVERFLOWS>(&params, &poly, &data)
                        .expect("Test phase failed");
                black_box(test_transcript);
            })
        },
    );
}

pub fn test_matrix_32x256_raa<Zt: ZipTypes, Lc: LinearCode<Zt>, const CHECK_FOR_OVERFLOWS: bool>(
    group: &mut BenchmarkGroup<WallTime>,
) where
    StandardUniform: Distribution<Zt::Eval>,
{
    const NUM_ROWS: usize = 32;
    const ROW_LEN: usize = 256;
    const P: usize = 13; // 2^13 = 8192 = 32 * 256
    const POLY_SIZE_FOR_ROW_LEN: usize = 1 << 16; // Ensures RAA row_len = 256

    let mut rng = ThreadRng::default();
    let poly_size = 1 << P;
    let linear_code = Lc::new(POLY_SIZE_FOR_ROW_LEN);
    let row_len = linear_code.row_len();
    assert_eq!(
        row_len, ROW_LEN,
        "Expected row_len to be {ROW_LEN}, got {row_len}"
    );
    let params = ZipPlusParams::new(P, NUM_ROWS, linear_code);

    let poly = DenseMultilinearExtension::rand(P, &mut rng);
    let (data, _) = ZipPlus::commit(&params, &poly).unwrap();

    group.bench_function(
        format!(
            "TestMatrix/32x256/{}",
            Zt::Eval::type_name()
        ),
        |b| {
            b.iter(|| {
                let test_transcript =
                    ZipPlus::test::<CHECK_FOR_OVERFLOWS>(&params, &poly, &data)
                        .expect("Test phase failed");
                black_box(test_transcript);
            })
        },
    );
}

pub fn verify_only_test_matrix_8x1024_raa<
    Zt: ZipTypes,
    Lc: LinearCode<Zt>,
    const CHECK_FOR_OVERFLOWS: bool,
>(
    group: &mut BenchmarkGroup<WallTime>,
) where
    StandardUniform: Distribution<Zt::Eval>,
{
    const NUM_ROWS: usize = 8;
    const ROW_LEN: usize = 1024;
    const P: usize = 13; // 2^13 = 8192 = 8 * 1024
    const POLY_SIZE_FOR_ROW_LEN: usize = 1 << 20; // Ensures RAA row_len = 1024

    let mut rng = ThreadRng::default();
    let poly_size = 1 << P;
    let linear_code = Lc::new(POLY_SIZE_FOR_ROW_LEN);
    let row_len = linear_code.row_len();
    assert_eq!(
        row_len, ROW_LEN,
        "Expected row_len to be {ROW_LEN}, got {row_len}"
    );
    let params = ZipPlusParams::new(P, NUM_ROWS, linear_code);

    let poly = DenseMultilinearExtension::rand(P, &mut rng);
    let (data, commitment) = ZipPlus::commit(&params, &poly).unwrap();
    let test_transcript =
        ZipPlus::test::<CHECK_FOR_OVERFLOWS>(&params, &poly, &data).expect("Test phase failed");

    group.bench_function(
        format!(
            "VerifyOnlyTestMatrix/8x1024/{}",
            Zt::Eval::type_name()
        ),
        |b| {
            b.iter(|| {
                let proof = test_transcript.clone();
                ZipPlus::verify_test_phase::<CHECK_FOR_OVERFLOWS>(&params, &commitment, proof)
                    .expect("Test phase verification failed");
            })
        },
    );
}

pub fn verify_only_test_matrix_32x256_raa<
    Zt: ZipTypes,
    Lc: LinearCode<Zt>,
    const CHECK_FOR_OVERFLOWS: bool,
>(
    group: &mut BenchmarkGroup<WallTime>,
) where
    StandardUniform: Distribution<Zt::Eval>,
{
    const NUM_ROWS: usize = 32;
    const ROW_LEN: usize = 256;
    const P: usize = 13; // 2^13 = 8192 = 32 * 256
    const POLY_SIZE_FOR_ROW_LEN: usize = 1 << 16; // Ensures RAA row_len = 256

    let mut rng = ThreadRng::default();
    let poly_size = 1 << P;
    let linear_code = Lc::new(POLY_SIZE_FOR_ROW_LEN);
    let row_len = linear_code.row_len();
    assert_eq!(
        row_len, ROW_LEN,
        "Expected row_len to be {ROW_LEN}, got {row_len}"
    );
    let params = ZipPlusParams::new(P, NUM_ROWS, linear_code);

    let poly = DenseMultilinearExtension::rand(P, &mut rng);
    let (data, commitment) = ZipPlus::commit(&params, &poly).unwrap();
    let test_transcript =
        ZipPlus::test::<CHECK_FOR_OVERFLOWS>(&params, &poly, &data).expect("Test phase failed");

    group.bench_function(
        format!(
            "VerifyOnlyTestMatrix/32x256/{}",
            Zt::Eval::type_name()
        ),
        |b| {
            b.iter(|| {
                let proof = test_transcript.clone();
                ZipPlus::verify_test_phase::<CHECK_FOR_OVERFLOWS>(&params, &commitment, proof)
                    .expect("Test phase verification failed");
            })
        },
    );
}

pub fn commit_matrix_4x1024_iprs<Zt: ZipTypes, Lc: LinearCode<Zt>>(
    group: &mut BenchmarkGroup<WallTime>,
) where
    StandardUniform: Distribution<Zt::Eval>,
{
    const ROW_LEN: usize = 1024;
    const P: usize = 12; // 2^12 = 4096 = 4 * 1024

    let mut rng = ThreadRng::default();
    let poly_size = 1 << P;
    let linear_code = Lc::new(poly_size);
    let row_len = linear_code.row_len();
    assert_eq!(
        row_len, ROW_LEN,
        "Expected row_len to be {ROW_LEN}, got {row_len}"
    );
    let params = ZipPlus::setup(poly_size, linear_code);

    group.bench_function(
        format!(
            "CommitMatrix/4x1024/{}",
            Zt::Eval::type_name()
        ),
        |b| {
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

pub fn commit_matrix_8x1024_iprs<Zt: ZipTypes, Lc: LinearCode<Zt>>(
    group: &mut BenchmarkGroup<WallTime>,
) where
    StandardUniform: Distribution<Zt::Eval>,
{
    const ROW_LEN: usize = 1024;
    const P: usize = 13; // 2^13 = 8192 = 8 * 1024

    let mut rng = ThreadRng::default();
    let poly_size = 1 << P;
    let linear_code = Lc::new(poly_size);
    let row_len = linear_code.row_len();
    assert_eq!(
        row_len, ROW_LEN,
        "Expected row_len to be {ROW_LEN}, got {row_len}"
    );
    let params = ZipPlus::setup(poly_size, linear_code);

    group.bench_function(
        format!(
            "CommitMatrix/8x1024/{}",
            Zt::Eval::type_name()
        ),
        |b| {
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

pub fn commit_matrix_32x256_iprs<Zt: ZipTypes, Lc: LinearCode<Zt>>(
    group: &mut BenchmarkGroup<WallTime>,
) where
    StandardUniform: Distribution<Zt::Eval>,
{
    const ROW_LEN: usize = 256;
    const P: usize = 13; // 2^13 = 8192 = 32 * 256

    let mut rng = ThreadRng::default();
    let poly_size = 1 << P;
    let linear_code = Lc::new(poly_size);
    let row_len = linear_code.row_len();
    assert_eq!(
        row_len, ROW_LEN,
        "Expected row_len to be {ROW_LEN}, got {row_len}"
    );
    let params = ZipPlus::setup(poly_size, linear_code);

    group.bench_function(
        format!(
            "CommitMatrix/32x256/{}",
            Zt::Eval::type_name()
        ),
        |b| {
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

pub fn test_matrix_8x1024_iprs<Zt: ZipTypes, Lc: LinearCode<Zt>, const CHECK_FOR_OVERFLOWS: bool>(
    group: &mut BenchmarkGroup<WallTime>,
) where
    StandardUniform: Distribution<Zt::Eval>,
{
    const ROW_LEN: usize = 1024;
    const P: usize = 13; // 2^13 = 8192 = 8 * 1024

    let mut rng = ThreadRng::default();
    let poly_size = 1 << P;
    let linear_code = Lc::new(poly_size);
    let row_len = linear_code.row_len();
    assert_eq!(
        row_len, ROW_LEN,
        "Expected row_len to be {ROW_LEN}, got {row_len}"
    );
    let params = ZipPlus::setup(poly_size, linear_code);

    let poly = DenseMultilinearExtension::rand(P, &mut rng);
    let (data, _) = ZipPlus::commit(&params, &poly).unwrap();

    group.bench_function(
        format!(
            "TestMatrix/8x1024/{}",
            Zt::Eval::type_name()
        ),
        |b| {
            b.iter(|| {
                let test_transcript =
                    ZipPlus::test::<CHECK_FOR_OVERFLOWS>(&params, &poly, &data)
                        .expect("Test phase failed");
                black_box(test_transcript);
            })
        },
    );
}

pub fn test_matrix_32x256_iprs<Zt: ZipTypes, Lc: LinearCode<Zt>, const CHECK_FOR_OVERFLOWS: bool>(
    group: &mut BenchmarkGroup<WallTime>,
) where
    StandardUniform: Distribution<Zt::Eval>,
{
    const ROW_LEN: usize = 256;
    const P: usize = 13; // 2^13 = 8192 = 32 * 256

    let mut rng = ThreadRng::default();
    let poly_size = 1 << P;
    let linear_code = Lc::new(poly_size);
    let row_len = linear_code.row_len();
    assert_eq!(
        row_len, ROW_LEN,
        "Expected row_len to be {ROW_LEN}, got {row_len}"
    );
    let params = ZipPlus::setup(poly_size, linear_code);

    let poly = DenseMultilinearExtension::rand(P, &mut rng);
    let (data, _) = ZipPlus::commit(&params, &poly).unwrap();

    group.bench_function(
        format!(
            "TestMatrix/32x256/{}",
            Zt::Eval::type_name()
        ),
        |b| {
            b.iter(|| {
                let test_transcript =
                    ZipPlus::test::<CHECK_FOR_OVERFLOWS>(&params, &poly, &data)
                        .expect("Test phase failed");
                black_box(test_transcript);
            })
        },
    );
}

pub fn verify_only_test_matrix_8x1024_iprs<
    Zt: ZipTypes,
    Lc: LinearCode<Zt>,
    const CHECK_FOR_OVERFLOWS: bool,
>(
    group: &mut BenchmarkGroup<WallTime>,
) where
    StandardUniform: Distribution<Zt::Eval>,
{
    const ROW_LEN: usize = 1024;
    const P: usize = 13; // 2^13 = 8192 = 8 * 1024

    let mut rng = ThreadRng::default();
    let poly_size = 1 << P;
    let linear_code = Lc::new(poly_size);
    let row_len = linear_code.row_len();
    assert_eq!(
        row_len, ROW_LEN,
        "Expected row_len to be {ROW_LEN}, got {row_len}"
    );
    let params = ZipPlus::setup(poly_size, linear_code);

    let poly = DenseMultilinearExtension::rand(P, &mut rng);
    let (data, commitment) = ZipPlus::commit(&params, &poly).unwrap();
    let test_transcript =
        ZipPlus::test::<CHECK_FOR_OVERFLOWS>(&params, &poly, &data).expect("Test phase failed");

    group.bench_function(
        format!(
            "VerifyOnlyTestMatrix/8x1024/{}",
            Zt::Eval::type_name()
        ),
        |b| {
            b.iter(|| {
                let proof = test_transcript.clone();
                ZipPlus::verify_test_phase::<CHECK_FOR_OVERFLOWS>(&params, &commitment, proof)
                    .expect("Test phase verification failed");
            })
        },
    );
}

pub fn verify_only_test_matrix_32x256_iprs<
    Zt: ZipTypes,
    Lc: LinearCode<Zt>,
    const CHECK_FOR_OVERFLOWS: bool,
>(
    group: &mut BenchmarkGroup<WallTime>,
) where
    StandardUniform: Distribution<Zt::Eval>,
{
    const ROW_LEN: usize = 256;
    const P: usize = 13; // 2^13 = 8192 = 32 * 256

    let mut rng = ThreadRng::default();
    let poly_size = 1 << P;
    let linear_code = Lc::new(poly_size);
    let row_len = linear_code.row_len();
    assert_eq!(
        row_len, ROW_LEN,
        "Expected row_len to be {ROW_LEN}, got {row_len}"
    );
    let params = ZipPlus::setup(poly_size, linear_code);

    let poly = DenseMultilinearExtension::rand(P, &mut rng);
    let (data, commitment) = ZipPlus::commit(&params, &poly).unwrap();
    let test_transcript =
        ZipPlus::test::<CHECK_FOR_OVERFLOWS>(&params, &poly, &data).expect("Test phase failed");

    group.bench_function(
        format!(
            "VerifyOnlyTestMatrix/32x256/{}",
            Zt::Eval::type_name()
        ),
        |b| {
            b.iter(|| {
                let proof = test_transcript.clone();
                ZipPlus::verify_test_phase::<CHECK_FOR_OVERFLOWS>(&params, &commitment, proof)
                    .expect("Test phase verification failed");
            })
        },
    );
}

pub fn encode_single_row<Zt: ZipTypes, Lc: LinearCode<Zt>, const ROW_LEN: usize>(
    group: &mut BenchmarkGroup<WallTime>,
) where
    StandardUniform: Distribution<Zt::Eval>,
{
    let mut rng = ThreadRng::default();
    let poly_size = ROW_LEN * ROW_LEN;
    let linear_code = Lc::new(poly_size);
    if linear_code.row_len() != ROW_LEN {
        // TODO(Ilia): Since IPRS codes require
        //             the input size to be known at compile time
        //             this detects IPRS benches.
        //             Ofc, it's a lame way to handle this and
        //             one can come up with a more elegant type safe way
        //             but for the sake of a fast solution it's good enough.
        //             Once we have time address this pls.

        return;
    }

    group.bench_function(
        format!(
            "EncodeMessage: {} -> {}, row_len = {ROW_LEN}",
            Zt::Eval::type_name(),
            Zt::Cw::type_name()
        ),
        |b| {
            let message: Vec<<Zt as ZipTypes>::Eval> =
                (0..ROW_LEN).map(|_i| rng.random()).collect();
            b.iter(|| {
                let encoded_row: Vec<<Zt as ZipTypes>::Cw> = linear_code.encode(&message);
                black_box(encoded_row);
            })
        },
    );
}

pub fn merkle_root<Zt: ZipTypes, const P: usize>(group: &mut BenchmarkGroup<WallTime>)
where
    StandardUniform: Distribution<Zt::Cw>,
{
    let mut rng = ThreadRng::default();

    let num_leaves = 1 << P;
    let leaves = (0..num_leaves)
        .map(|_| rng.random::<<Zt as ZipTypes>::Cw>())
        .collect_vec();
    let matrix: DenseRowMatrix<_> = vec![leaves.clone()].into();
    let rows = matrix.to_rows_slices();

    group.bench_function(
        format!("MerkleRoot: {}, leaves=2^{P}", Zt::Cw::type_name()),
        |b| {
            b.iter(|| {
                let tree = MerkleTree::new(&rows);
                black_box(tree.root());
            })
        },
    );
}

pub fn commit<Zt: ZipTypes, Lc: LinearCode<Zt>, const P: usize>(
    group: &mut BenchmarkGroup<WallTime>,
) where
    StandardUniform: Distribution<Zt::Eval>,
{
    let mut rng = ThreadRng::default();
    let poly_size = 1 << P;
    let linear_code = Lc::new(poly_size);
    let params = ZipPlus::setup(poly_size, linear_code);

    group.bench_function(
        format!(
            "Commit: Eval={}, Cw={}, Comb={}, poly_size=2^{P}",
            Zt::Eval::type_name(),
            Zt::Cw::type_name(),
            Zt::Comb::type_name()
        ),
        |b| {
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

pub fn commit_batch<Zt: ZipTypes, Lc: LinearCode<Zt>, const P: usize, const B: usize>(
    group: &mut BenchmarkGroup<WallTime>,
) where
    StandardUniform: Distribution<Zt::Eval>,
{
    let mut rng = ThreadRng::default();
    let poly_size = 1 << P;
    let linear_code = Lc::new(poly_size);
    let params = ZipPlus::setup(poly_size, linear_code);

    let polys: Vec<_> = (0..B)
        .map(|_| DenseMultilinearExtension::rand(P, &mut rng))
        .collect();

    group.bench_function(
        format!(
            "CommitBatch: Eval={}, Cw={}, Comb={}, poly_size=2^{P}, batch={B}",
            Zt::Eval::type_name(),
            Zt::Cw::type_name(),
            Zt::Comb::type_name()
        ),
        |b| {
            b.iter_custom(|iters| {
                let mut total_duration = Duration::ZERO;
                for _ in 0..iters {
                    let timer = Instant::now();
                    let results: Vec<_> = cfg_iter!(&polys)
                        .map(|poly| ZipPlus::commit(&params, poly).expect("Failed to commit"))
                        .collect();
                    black_box(results);
                    total_duration += timer.elapsed();
                }

                total_duration
            })
        },
    );
}

pub fn test<Zt: ZipTypes, Lc: LinearCode<Zt>, const CHECK_FOR_OVERFLOWS: bool, const P: usize>(
    group: &mut BenchmarkGroup<WallTime>,
) where
    StandardUniform: Distribution<Zt::Eval>,
{
    let mut rng = ThreadRng::default();

    let poly_size = 1 << P;
    let linear_code = Lc::new(poly_size);
    let params = ZipPlus::setup(poly_size, linear_code);

    let poly = DenseMultilinearExtension::rand(P, &mut rng);
    let (data, _) = ZipPlus::commit(&params, &poly).unwrap();

    group.bench_function(
        format!(
            "Test: Eval={}, Cw={}, Comb={}, poly_size=2^{P}",
            Zt::Eval::type_name(),
            Zt::Cw::type_name(),
            Zt::Comb::type_name(),
        ),
        |b| {
            b.iter(|| {
                let test_transcript = ZipPlus::test::<CHECK_FOR_OVERFLOWS>(&params, &poly, &data)
                    .expect("Test phase failed");
                black_box(test_transcript);
            })
        },
    );
}

pub fn test_batch<
    Zt: ZipTypes,
    Lc: LinearCode<Zt>,
    const CHECK_FOR_OVERFLOWS: bool,
    const P: usize,
    const B: usize,
>(
    group: &mut BenchmarkGroup<WallTime>,
) where
    StandardUniform: Distribution<Zt::Eval>,
{
    let mut rng = ThreadRng::default();

    let poly_size = 1 << P;
    let linear_code = Lc::new(poly_size);
    let params = ZipPlus::setup(poly_size, linear_code);

    let batch: Vec<_> = (0..B)
        .map(|_| {
            let poly = DenseMultilinearExtension::rand(P, &mut rng);
            let (data, _) = ZipPlus::commit(&params, &poly).unwrap();
            (poly, data)
        })
        .collect();

    group.bench_function(
        format!(
            "TestBatch: Eval={}, Cw={}, Comb={}, poly_size=2^{P}, batch={B}",
            Zt::Eval::type_name(),
            Zt::Cw::type_name(),
            Zt::Comb::type_name(),
        ),
        |b| {
            b.iter(|| {
                let results: Vec<_> = cfg_iter!(&batch)
                    .map(|pair| {
                        let (poly, data) = pair;
                        ZipPlus::test::<CHECK_FOR_OVERFLOWS>(&params, poly, data)
                            .expect("Test phase failed")
                    })
                    .collect();
                black_box(results);
            })
        },
    );
}

pub fn evaluate<Zt: ZipTypes, Lc: LinearCode<Zt>, const CHECK_FOR_OVERFLOWS: bool, const P: usize>(
    group: &mut BenchmarkGroup<WallTime>,
) where
    StandardUniform: Distribution<Zt::Eval>,
    F: for<'a> FromWithConfig<&'a Zt::Chal> + for<'a> FromWithConfig<&'a Zt::Pt>,
    <F as Field>::Inner: FromRef<Zt::Fmod>,
    Zt::Eval: ProjectableToField<F>,
{
    let mut rng = ThreadRng::default();

    let poly_size = 1 << P;
    let linear_code = Lc::new(poly_size);
    let params = ZipPlus::setup(poly_size, linear_code);

    let poly = DenseMultilinearExtension::rand(P, &mut rng);
    let (data, _) = ZipPlus::commit(&params, &poly).unwrap();
    let point = vec![Zt::Pt::one(); P];

    let test_transcript =
        ZipPlus::test::<CHECK_FOR_OVERFLOWS>(&params, &poly, &data).expect("Test phase failed");

    group.bench_function(
        format!(
            "Evaluate: Eval={}, Cw={}, Comb={}, poly_size=2^{P}, modulus=({} bits)",
            Zt::Eval::type_name(),
            Zt::Cw::type_name(),
            Zt::Comb::type_name(),
            Zt::Fmod::NUM_BYTES * 8
        ),
        |b| {
            b.iter_custom(|iters| {
                let mut total_duration = Duration::ZERO;
                for _ in 0..iters {
                    let proof = test_transcript.clone();
                    let timer = Instant::now();
                    let (eval_f, proof) =
                        ZipPlus::evaluate::<F, CHECK_FOR_OVERFLOWS>(&params, &poly, &point, proof)
                            .expect("Evaluation phase failed");
                    total_duration += timer.elapsed();
                    black_box((eval_f, proof));
                }
                total_duration
            })
        },
    );
}

pub fn verify_only_test<
    Zt: ZipTypes,
    Lc: LinearCode<Zt>,
    const CHECK_FOR_OVERFLOWS: bool,
    const P: usize,
>(
    group: &mut BenchmarkGroup<WallTime>,
) where
    StandardUniform: Distribution<Zt::Eval>,
{
    let mut rng = ThreadRng::default();
    let poly_size = 1 << P;
    let linear_code = Lc::new(poly_size);
    let params = ZipPlus::setup(poly_size, linear_code);

    let poly = DenseMultilinearExtension::rand(P, &mut rng);
    let (data, commitment) = ZipPlus::commit(&params, &poly).unwrap();
    let test_transcript =
        ZipPlus::test::<CHECK_FOR_OVERFLOWS>(&params, &poly, &data).expect("Test phase failed");

    group.bench_function(
        format!(
            "VerifyOnlyTest: Eval={}, Cw={}, Comb={}, poly_size=2^{P}",
            Zt::Eval::type_name(),
            Zt::Cw::type_name(),
            Zt::Comb::type_name()
        ),
        |b| {
            b.iter(|| {
                let proof = test_transcript.clone();
                ZipPlus::verify_test_phase::<CHECK_FOR_OVERFLOWS>(&params, &commitment, proof)
                    .expect("Test phase verification failed");
            })
        },
    );
}
