//! Minimal batched Zip+ PCS benchmarks for BPoly<31> and BPoly<15> with poly sizes 2^9, 2^10, 2^11.
//!
//! Measures Encode, Merkle, Commit, Test, and Verify for batches of 14 (or 7)
//! polynomials using a single shared Merkle tree with num_rows=1.

#![allow(non_local_definitions)]
#![allow(clippy::eq_op, clippy::arithmetic_side_effects, clippy::unwrap_used)]

use std::hint::black_box;
use std::marker::PhantomData;
use std::time::{Duration, Instant};

use criterion::{BenchmarkGroup, Criterion, criterion_group, criterion_main, measurement::WallTime};
use crypto_bigint::U64;
use crypto_primitives::{
    Field, FromWithConfig, IntoWithConfig, PrimeField,
    FixedSemiring,
    boolean::Boolean,
    crypto_bigint_int::Int,
    crypto_bigint_uint::Uint,
    crypto_bigint_monty::MontyField,
};
use rand::{distr::StandardUniform, prelude::*};

use zinc_poly::mle::{DenseMultilinearExtension, MultilinearExtensionRand};
use zinc_poly::univariate::{
    binary::{BinaryPoly, BinaryPolyInnerProduct, BinaryPolyWideningMulByScalar},
    dense::{DensePolyInnerProduct, DensePolynomial},
};
use zinc_primality::MillerRabin;
use zinc_transcript::traits::{ConstTranscribable, Transcribable, Transcript};
use zinc_utils::{
    UNCHECKED,
    from_ref::FromRef,
    inner_product::MBSInnerProduct,
    mul_by_scalar::MulByScalar,
    named::Named,
};
use zip_plus::{
    code::{
        LinearCode,
        iprs::{
            IprsCode,
            PnttConfigF2_16R4B16, PnttConfigF2_16R4B32, PnttConfigF2_16R4B64,
        },
    },
    merkle::MerkleTree,
    pcs::structs::{ZipPlus, ZipPlusParams, ZipTypes},
};

const INT_LIMBS: usize = U64::LIMBS;
type F = MontyField<{ INT_LIMBS * 4 }>;

// ---------- ZipTypes for BPoly<31> benchmarks --------------------------------

struct BenchZipPlusTypes<CwCoeff, const D_PLUS_ONE: usize>(PhantomData<CwCoeff>);

impl<CwCoeff, const D_PLUS_ONE: usize> ZipTypes for BenchZipPlusTypes<CwCoeff, D_PLUS_ONE>
where
    CwCoeff: ConstTranscribable + Copy + Default + FromRef<Boolean> + Named + FixedSemiring + Send + Sync,
    Int<6>: FromRef<CwCoeff>,
{
    const NUM_COLUMN_OPENINGS: usize = 118;
    const GRINDING_BITS: usize = 16;
    type Eval = BinaryPoly<D_PLUS_ONE>;
    type Cw = DensePolynomial<CwCoeff, D_PLUS_ONE>;
    type Fmod = Uint<{ INT_LIMBS * 4 }>;
    type PrimeTest = MillerRabin;
    type Chal = i128;
    type Pt = i128;
    type CombR = Int<{ INT_LIMBS * 6 }>;
    type Comb = DensePolynomial<Self::CombR, D_PLUS_ONE>;
    type EvalDotChal = BinaryPolyInnerProduct<Self::Chal, D_PLUS_ONE>;
    type CombDotChal = DensePolyInnerProduct<Self::CombR, Self::Chal, Self::CombR, MBSInnerProduct, D_PLUS_ONE>;
    type ArrCombRDotChal = MBSInnerProduct;
}

// ---------- IPRS code type aliases for BPoly<31> (F65537 rate 1/4) -----------

// P=9: R4B64 D=1, row_len=512
type IprsBPolyR4B64<Twiddle, const DEPTH: usize, const D_PLUS_ONE: usize, const CHECK: bool> =
    IprsCode<
        BenchZipPlusTypes<Twiddle, D_PLUS_ONE>,
        PnttConfigF2_16R4B64<DEPTH>,
        BinaryPolyWideningMulByScalar<Twiddle>,
        CHECK,
    >;

// P=10: R4B16 D=2, row_len=1024
type IprsBPolyR4B16<Twiddle, const DEPTH: usize, const D_PLUS_ONE: usize, const CHECK: bool> =
    IprsCode<
        BenchZipPlusTypes<Twiddle, D_PLUS_ONE>,
        PnttConfigF2_16R4B16<DEPTH>,
        BinaryPolyWideningMulByScalar<Twiddle>,
        CHECK,
    >;

// P=11: R4B32 D=2, row_len=2048
type IprsBPolyR4B32<Twiddle, const DEPTH: usize, const D_PLUS_ONE: usize, const CHECK: bool> =
    IprsCode<
        BenchZipPlusTypes<Twiddle, D_PLUS_ONE>,
        PnttConfigF2_16R4B32<DEPTH>,
        BinaryPolyWideningMulByScalar<Twiddle>,
        CHECK,
    >;

// ---------- Batched benchmark helpers ----------------------------------------

const BATCH_SIZE: usize = 14;
const BATCH_SIZE_7: usize = 7;

fn batched_encode_nrows<Zt: ZipTypes, Lc: LinearCode<Zt>, const P: usize>(
    group: &mut BenchmarkGroup<WallTime>,
    num_rows: usize,
    batch_size: usize,
) where
    StandardUniform: Distribution<Zt::Eval>,
{
    let mut rng = ThreadRng::default();
    let poly_size = 1 << P;
    let row_len = poly_size / num_rows;
    let linear_code = Lc::new(row_len);
    let params = ZipPlusParams::new(P, num_rows, linear_code);
    let row_len = params.linear_code.row_len();

    group.bench_function(
        format!("Encode poly_size=2^{P} num_rows={num_rows}"),
        |b| {
            b.iter_custom(|iters| {
                let mut total_duration = Duration::ZERO;
                for _ in 0..iters {
                    let polys: Vec<_> = (0..batch_size)
                        .map(|_| DenseMultilinearExtension::rand(P, &mut rng))
                        .collect();
                    let timer = Instant::now();
                    for poly in &polys {
                        let cw = ZipPlus::<Zt, Lc>::encode_rows(&params, row_len, poly);
                        black_box(&cw);
                    }
                    total_duration += timer.elapsed();
                }
                total_duration
            })
        },
    );
}

fn batched_merkle_nrows<Zt: ZipTypes, Lc: LinearCode<Zt>, const P: usize>(
    group: &mut BenchmarkGroup<WallTime>,
    num_rows: usize,
    batch_size: usize,
) where
    StandardUniform: Distribution<Zt::Eval>,
{
    let mut rng = ThreadRng::default();
    let poly_size = 1 << P;
    let row_len = poly_size / num_rows;
    let linear_code = Lc::new(row_len);
    let params = ZipPlusParams::new(P, num_rows, linear_code);
    let row_len = params.linear_code.row_len();

    let cw_matrices: Vec<_> = (0..batch_size)
        .map(|_| {
            let poly = DenseMultilinearExtension::rand(P, &mut rng);
            ZipPlus::<Zt, Lc>::encode_rows(&params, row_len, &poly)
        })
        .collect();

    group.bench_function(
        format!("Merkle poly_size=2^{P} num_rows={num_rows}"),
        |b| {
            b.iter(|| {
                let all_rows: Vec<&[Zt::Cw]> = cw_matrices
                    .iter()
                    .flat_map(|m| m.to_rows_slices())
                    .collect();
                let tree = MerkleTree::new(&all_rows);
                black_box(tree.root());
            })
        },
    );
}

fn batched_commit_nrows<Zt: ZipTypes, Lc: LinearCode<Zt>, const P: usize>(
    group: &mut BenchmarkGroup<WallTime>,
    num_rows: usize,
    batch_size: usize,
) where
    StandardUniform: Distribution<Zt::Eval>,
{
    let mut rng = ThreadRng::default();
    let poly_size = 1 << P;
    let row_len = poly_size / num_rows;
    let linear_code = Lc::new(row_len);
    let params = ZipPlusParams::new(P, num_rows, linear_code);

    group.bench_function(
        format!("Commit poly_size=2^{P} num_rows={num_rows}"),
        |b| {
            b.iter_custom(|iters| {
                let mut total_duration = Duration::ZERO;
                for _ in 0..iters {
                    let polys: Vec<_> = (0..batch_size)
                        .map(|_| DenseMultilinearExtension::rand(P, &mut rng))
                        .collect();
                    let timer = Instant::now();
                    let res = ZipPlus::<Zt, Lc>::commit(&params, &polys)
                        .expect("Commit failed");
                    black_box(res);
                    total_duration += timer.elapsed();
                }
                total_duration
            })
        },
    );
}

fn batched_test_nrows<
    Zt: ZipTypes,
    Lc: LinearCode<Zt>,
    const CHECK_FOR_OVERFLOWS: bool,
    const P: usize,
>(
    group: &mut BenchmarkGroup<WallTime>,
    num_rows: usize,
    batch_size: usize,
) where
    StandardUniform: Distribution<Zt::Eval>,
    F: for<'a> FromWithConfig<&'a Zt::CombR>
        + for<'a> FromWithConfig<&'a Zt::Chal>
        + for<'a> MulByScalar<&'a F>
        + FromRef<F>,
    <F as Field>::Inner: FromRef<Zt::Fmod> + Transcribable,
{
    let mut rng = ThreadRng::default();
    let poly_size = 1 << P;
    let row_len = poly_size / num_rows;
    let linear_code = Lc::new(row_len);
    let params = ZipPlusParams::new(P, num_rows, linear_code);

    let polys: Vec<_> = (0..batch_size)
        .map(|_| DenseMultilinearExtension::rand(P, &mut rng))
        .collect();
    let (hint, _) = ZipPlus::<Zt, Lc>::commit(&params, &polys).unwrap();

    group.bench_function(
        format!("Test poly_size=2^{P} num_rows={num_rows}"),
        |b| {
            b.iter(|| {
                let field_cfg = zinc_transcript::KeccakTranscript::default()
                    .get_random_field_cfg::<F, Zt::Fmod, Zt::PrimeTest>();
                let point: Vec<F> = vec![F::one_with_cfg(&field_cfg); P];
                let result =
                    ZipPlus::<Zt, Lc>::prove::<F, CHECK_FOR_OVERFLOWS>(&params, &polys, &point, &hint)
                        .expect("Prove failed");
                black_box(result);
            })
        },
    );
}

fn batched_verify_nrows<
    Zt: ZipTypes,
    Lc: LinearCode<Zt>,
    const CHECK_FOR_OVERFLOWS: bool,
    const P: usize,
>(
    group: &mut BenchmarkGroup<WallTime>,
    num_rows: usize,
    batch_size: usize,
) where
    StandardUniform: Distribution<Zt::Eval>,
    F: PrimeField
        + for<'a> FromWithConfig<&'a Zt::CombR>
        + for<'a> FromWithConfig<&'a Zt::Chal>
        + for<'a> MulByScalar<&'a F>
        + FromRef<F>,
    <F as Field>::Inner: FromRef<Zt::Fmod> + Transcribable,
{
    let mut rng = ThreadRng::default();
    let poly_size = 1 << P;
    let row_len = poly_size / num_rows;
    let linear_code = Lc::new(row_len);
    let params = ZipPlusParams::new(P, num_rows, linear_code);

    let polys: Vec<_> = (0..batch_size)
        .map(|_| DenseMultilinearExtension::rand(P, &mut rng))
        .collect();
    let (hint, commitment) = ZipPlus::<Zt, Lc>::commit(&params, &polys).unwrap();
    let pcs_field_cfg = zinc_transcript::KeccakTranscript::default()
        .get_random_field_cfg::<F, Zt::Fmod, Zt::PrimeTest>();
    let point: Vec<F> = vec![F::one_with_cfg(&pcs_field_cfg); P];

    let (eval_f, proof) = ZipPlus::<Zt, Lc>::prove::<F, CHECK_FOR_OVERFLOWS>(
        &params,
        &polys,
        &point,
        &hint,
    )
    .expect("Prove failed");

    let point_f: Vec<F> = point.clone();

    group.bench_function(
        format!("Verify poly_size=2^{P} num_rows={num_rows}"),
        |b| {
            b.iter(|| {
                ZipPlus::<Zt, Lc>::verify::<F, CHECK_FOR_OVERFLOWS>(
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

// ---------- Criterion entry point --------------------------------------------

/// Batched PCS pipeline suite for BPoly<31>, poly sizes 2^9, 2^10, 2^11 only.
///
/// poly_size (2^P)   Config                 Field        row_len
/// ───────────────   ──────                 ─────        ───────
/// 2^9  = 512        R4B64 D=1 (rate 1/4)   F65537       512
/// 2^10 = 1024       R4B16 D=2 (rate 1/4)   F65537       1024
/// 2^11 = 2048       R4B32 D=2 (rate 1/4)   F65537       2048
fn batched_pcs_pipeline_suite_bpoly31_1row(c: &mut Criterion) {
    let mut group = c.benchmark_group("Batched PCS Pipeline Suite BPoly31 1row");
    group.sample_size(100);

    // ── Encode ───────────────────────────────────────────────────────
    batched_encode_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B64<i64, 1, 32, UNCHECKED>,  9>(&mut group, 1, BATCH_SIZE);
    batched_encode_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B16<i64, 2, 32, UNCHECKED>, 10>(&mut group, 1, BATCH_SIZE);
    batched_encode_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B32<i64, 2, 32, UNCHECKED>, 11>(&mut group, 1, BATCH_SIZE);

    // ── Merkle ───────────────────────────────────────────────────────
    batched_merkle_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B64<i64, 1, 32, UNCHECKED>,  9>(&mut group, 1, BATCH_SIZE);
    batched_merkle_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B16<i64, 2, 32, UNCHECKED>, 10>(&mut group, 1, BATCH_SIZE);
    batched_merkle_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B32<i64, 2, 32, UNCHECKED>, 11>(&mut group, 1, BATCH_SIZE);

    // ── Commit ───────────────────────────────────────────────────────
    batched_commit_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B64<i64, 1, 32, UNCHECKED>,  9>(&mut group, 1, BATCH_SIZE);
    batched_commit_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B16<i64, 2, 32, UNCHECKED>, 10>(&mut group, 1, BATCH_SIZE);
    batched_commit_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B32<i64, 2, 32, UNCHECKED>, 11>(&mut group, 1, BATCH_SIZE);

    // ── Test ─────────────────────────────────────────────────────────
    batched_test_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B64<i64, 1, 32, UNCHECKED>, UNCHECKED,  9>(&mut group, 1, BATCH_SIZE);
    batched_test_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B16<i64, 2, 32, UNCHECKED>, UNCHECKED, 10>(&mut group, 1, BATCH_SIZE);
    batched_test_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B32<i64, 2, 32, UNCHECKED>, UNCHECKED, 11>(&mut group, 1, BATCH_SIZE);

    // ── Verify ───────────────────────────────────────────────────────
    batched_verify_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B64<i64, 1, 32, UNCHECKED>, UNCHECKED,  9>(&mut group, 1, BATCH_SIZE);
    batched_verify_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B16<i64, 2, 32, UNCHECKED>, UNCHECKED, 10>(&mut group, 1, BATCH_SIZE);
    batched_verify_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B32<i64, 2, 32, UNCHECKED>, UNCHECKED, 11>(&mut group, 1, BATCH_SIZE);

    group.finish();
}

/// Batched PCS pipeline suite for BPoly<31>, poly sizes 2^9, 2^10, 2^11 with num_rows=2.
///
/// poly_size (2^P)   Config                 Field        row_len
/// ───────────────   ──────                 ─────        ───────
/// 2^9  = 512        R4B32 D=1 (rate 1/4)   F65537       256
/// 2^10 = 1024       R4B64 D=1 (rate 1/4)   F65537       512
/// 2^11 = 2048       R4B16 D=2 (rate 1/4)   F65537       1024
fn batched_pcs_pipeline_suite_bpoly31_2row(c: &mut Criterion) {
    let mut group = c.benchmark_group("Batched PCS Pipeline Suite BPoly31 2row");
    group.sample_size(100);

    // ── Encode ───────────────────────────────────────────────────────
    batched_encode_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B32<i64, 1, 32, UNCHECKED>,  9>(&mut group, 2, BATCH_SIZE);
    batched_encode_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B64<i64, 1, 32, UNCHECKED>, 10>(&mut group, 2, BATCH_SIZE);
    batched_encode_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B16<i64, 2, 32, UNCHECKED>, 11>(&mut group, 2, BATCH_SIZE);

    // ── Merkle ───────────────────────────────────────────────────────
    batched_merkle_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B32<i64, 1, 32, UNCHECKED>,  9>(&mut group, 2, BATCH_SIZE);
    batched_merkle_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B64<i64, 1, 32, UNCHECKED>, 10>(&mut group, 2, BATCH_SIZE);
    batched_merkle_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B16<i64, 2, 32, UNCHECKED>, 11>(&mut group, 2, BATCH_SIZE);

    // ── Commit ───────────────────────────────────────────────────────
    batched_commit_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B32<i64, 1, 32, UNCHECKED>,  9>(&mut group, 2, BATCH_SIZE);
    batched_commit_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B64<i64, 1, 32, UNCHECKED>, 10>(&mut group, 2, BATCH_SIZE);
    batched_commit_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B16<i64, 2, 32, UNCHECKED>, 11>(&mut group, 2, BATCH_SIZE);

    // ── Test ─────────────────────────────────────────────────────────
    batched_test_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B32<i64, 1, 32, UNCHECKED>, UNCHECKED,  9>(&mut group, 2, BATCH_SIZE);
    batched_test_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B64<i64, 1, 32, UNCHECKED>, UNCHECKED, 10>(&mut group, 2, BATCH_SIZE);
    batched_test_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B16<i64, 2, 32, UNCHECKED>, UNCHECKED, 11>(&mut group, 2, BATCH_SIZE);

    // ── Verify ───────────────────────────────────────────────────────
    batched_verify_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B32<i64, 1, 32, UNCHECKED>, UNCHECKED,  9>(&mut group, 2, BATCH_SIZE);
    batched_verify_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B64<i64, 1, 32, UNCHECKED>, UNCHECKED, 10>(&mut group, 2, BATCH_SIZE);
    batched_verify_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B16<i64, 2, 32, UNCHECKED>, UNCHECKED, 11>(&mut group, 2, BATCH_SIZE);

    group.finish();
}

/// Batched PCS pipeline suite for BPoly<31>, batch_size=7, poly sizes 2^9, 2^10, 2^11 only.
///
/// poly_size (2^P)   Config                 Field        row_len
/// ───────────────   ──────                 ─────        ───────
/// 2^9  = 512        R4B64 D=1 (rate 1/4)   F65537       512
/// 2^10 = 1024       R4B16 D=2 (rate 1/4)   F65537       1024
/// 2^11 = 2048       R4B32 D=2 (rate 1/4)   F65537       2048
fn batched_pcs_pipeline_suite_bpoly31_1row_batch7(c: &mut Criterion) {
    let mut group = c.benchmark_group("Batched PCS Pipeline Suite BPoly31 1row batch7");
    group.sample_size(100);

    // ── Encode ───────────────────────────────────────────────────────
    batched_encode_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B64<i64, 1, 32, UNCHECKED>,  9>(&mut group, 1, BATCH_SIZE_7);
    batched_encode_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B16<i64, 2, 32, UNCHECKED>, 10>(&mut group, 1, BATCH_SIZE_7);
    batched_encode_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B32<i64, 2, 32, UNCHECKED>, 11>(&mut group, 1, BATCH_SIZE_7);

    // ── Merkle ───────────────────────────────────────────────────────
    batched_merkle_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B64<i64, 1, 32, UNCHECKED>,  9>(&mut group, 1, BATCH_SIZE_7);
    batched_merkle_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B16<i64, 2, 32, UNCHECKED>, 10>(&mut group, 1, BATCH_SIZE_7);
    batched_merkle_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B32<i64, 2, 32, UNCHECKED>, 11>(&mut group, 1, BATCH_SIZE_7);

    // ── Commit ───────────────────────────────────────────────────────
    batched_commit_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B64<i64, 1, 32, UNCHECKED>,  9>(&mut group, 1, BATCH_SIZE_7);
    batched_commit_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B16<i64, 2, 32, UNCHECKED>, 10>(&mut group, 1, BATCH_SIZE_7);
    batched_commit_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B32<i64, 2, 32, UNCHECKED>, 11>(&mut group, 1, BATCH_SIZE_7);

    // ── Test ─────────────────────────────────────────────────────────
    batched_test_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B64<i64, 1, 32, UNCHECKED>, UNCHECKED,  9>(&mut group, 1, BATCH_SIZE_7);
    batched_test_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B16<i64, 2, 32, UNCHECKED>, UNCHECKED, 10>(&mut group, 1, BATCH_SIZE_7);
    batched_test_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B32<i64, 2, 32, UNCHECKED>, UNCHECKED, 11>(&mut group, 1, BATCH_SIZE_7);

    // ── Verify ───────────────────────────────────────────────────────
    batched_verify_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B64<i64, 1, 32, UNCHECKED>, UNCHECKED,  9>(&mut group, 1, BATCH_SIZE_7);
    batched_verify_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B16<i64, 2, 32, UNCHECKED>, UNCHECKED, 10>(&mut group, 1, BATCH_SIZE_7);
    batched_verify_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B32<i64, 2, 32, UNCHECKED>, UNCHECKED, 11>(&mut group, 1, BATCH_SIZE_7);

    group.finish();
}

/// Batched PCS pipeline suite for BPoly<31>, batch_size=7, poly sizes 2^9, 2^10, 2^11 with num_rows=2.
///
/// poly_size (2^P)   Config                 Field        row_len
/// ───────────────   ──────                 ─────        ───────
/// 2^9  = 512        R4B32 D=1 (rate 1/4)   F65537       256
/// 2^10 = 1024       R4B64 D=1 (rate 1/4)   F65537       512
/// 2^11 = 2048       R4B16 D=2 (rate 1/4)   F65537       1024
fn batched_pcs_pipeline_suite_bpoly31_2row_batch7(c: &mut Criterion) {
    let mut group = c.benchmark_group("Batched PCS Pipeline Suite BPoly31 2row batch7");
    group.sample_size(100);

    // ── Encode ───────────────────────────────────────────────────────
    batched_encode_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B32<i64, 1, 32, UNCHECKED>,  9>(&mut group, 2, BATCH_SIZE_7);
    batched_encode_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B64<i64, 1, 32, UNCHECKED>, 10>(&mut group, 2, BATCH_SIZE_7);
    batched_encode_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B16<i64, 2, 32, UNCHECKED>, 11>(&mut group, 2, BATCH_SIZE_7);

    // ── Merkle ───────────────────────────────────────────────────────
    batched_merkle_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B32<i64, 1, 32, UNCHECKED>,  9>(&mut group, 2, BATCH_SIZE_7);
    batched_merkle_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B64<i64, 1, 32, UNCHECKED>, 10>(&mut group, 2, BATCH_SIZE_7);
    batched_merkle_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B16<i64, 2, 32, UNCHECKED>, 11>(&mut group, 2, BATCH_SIZE_7);

    // ── Commit ───────────────────────────────────────────────────────
    batched_commit_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B32<i64, 1, 32, UNCHECKED>,  9>(&mut group, 2, BATCH_SIZE_7);
    batched_commit_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B64<i64, 1, 32, UNCHECKED>, 10>(&mut group, 2, BATCH_SIZE_7);
    batched_commit_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B16<i64, 2, 32, UNCHECKED>, 11>(&mut group, 2, BATCH_SIZE_7);

    // ── Test ─────────────────────────────────────────────────────────
    batched_test_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B32<i64, 1, 32, UNCHECKED>, UNCHECKED,  9>(&mut group, 2, BATCH_SIZE_7);
    batched_test_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B64<i64, 1, 32, UNCHECKED>, UNCHECKED, 10>(&mut group, 2, BATCH_SIZE_7);
    batched_test_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B16<i64, 2, 32, UNCHECKED>, UNCHECKED, 11>(&mut group, 2, BATCH_SIZE_7);

    // ── Verify ───────────────────────────────────────────────────────
    batched_verify_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B32<i64, 1, 32, UNCHECKED>, UNCHECKED,  9>(&mut group, 2, BATCH_SIZE_7);
    batched_verify_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B64<i64, 1, 32, UNCHECKED>, UNCHECKED, 10>(&mut group, 2, BATCH_SIZE_7);
    batched_verify_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B16<i64, 2, 32, UNCHECKED>, UNCHECKED, 11>(&mut group, 2, BATCH_SIZE_7);

    group.finish();
}

/// Batched PCS pipeline suite for BPoly<15>, poly sizes 2^9, 2^10, 2^11 only.
///
/// poly_size (2^P)   Config                 Field        row_len
/// ───────────────   ──────                 ─────        ───────
/// 2^9  = 512        R4B64 D=1 (rate 1/4)   F65537       512
/// 2^10 = 1024       R4B16 D=2 (rate 1/4)   F65537       1024
/// 2^11 = 2048       R4B32 D=2 (rate 1/4)   F65537       2048
fn batched_pcs_pipeline_suite_bpoly15_1row(c: &mut Criterion) {
    let mut group = c.benchmark_group("Batched PCS Pipeline Suite BPoly15 1row");
    group.sample_size(100);

    // ── Encode ───────────────────────────────────────────────────────
    batched_encode_nrows::<BenchZipPlusTypes<i64, 16>, IprsBPolyR4B64<i64, 1, 16, UNCHECKED>,  9>(&mut group, 1, BATCH_SIZE);
    batched_encode_nrows::<BenchZipPlusTypes<i64, 16>, IprsBPolyR4B16<i64, 2, 16, UNCHECKED>, 10>(&mut group, 1, BATCH_SIZE);
    batched_encode_nrows::<BenchZipPlusTypes<i64, 16>, IprsBPolyR4B32<i64, 2, 16, UNCHECKED>, 11>(&mut group, 1, BATCH_SIZE);

    // ── Merkle ───────────────────────────────────────────────────────
    batched_merkle_nrows::<BenchZipPlusTypes<i64, 16>, IprsBPolyR4B64<i64, 1, 16, UNCHECKED>,  9>(&mut group, 1, BATCH_SIZE);
    batched_merkle_nrows::<BenchZipPlusTypes<i64, 16>, IprsBPolyR4B16<i64, 2, 16, UNCHECKED>, 10>(&mut group, 1, BATCH_SIZE);
    batched_merkle_nrows::<BenchZipPlusTypes<i64, 16>, IprsBPolyR4B32<i64, 2, 16, UNCHECKED>, 11>(&mut group, 1, BATCH_SIZE);

    // ── Commit ───────────────────────────────────────────────────────
    batched_commit_nrows::<BenchZipPlusTypes<i64, 16>, IprsBPolyR4B64<i64, 1, 16, UNCHECKED>,  9>(&mut group, 1, BATCH_SIZE);
    batched_commit_nrows::<BenchZipPlusTypes<i64, 16>, IprsBPolyR4B16<i64, 2, 16, UNCHECKED>, 10>(&mut group, 1, BATCH_SIZE);
    batched_commit_nrows::<BenchZipPlusTypes<i64, 16>, IprsBPolyR4B32<i64, 2, 16, UNCHECKED>, 11>(&mut group, 1, BATCH_SIZE);

    // ── Test ─────────────────────────────────────────────────────────
    batched_test_nrows::<BenchZipPlusTypes<i64, 16>, IprsBPolyR4B64<i64, 1, 16, UNCHECKED>, UNCHECKED,  9>(&mut group, 1, BATCH_SIZE);
    batched_test_nrows::<BenchZipPlusTypes<i64, 16>, IprsBPolyR4B16<i64, 2, 16, UNCHECKED>, UNCHECKED, 10>(&mut group, 1, BATCH_SIZE);
    batched_test_nrows::<BenchZipPlusTypes<i64, 16>, IprsBPolyR4B32<i64, 2, 16, UNCHECKED>, UNCHECKED, 11>(&mut group, 1, BATCH_SIZE);

    // ── Verify ───────────────────────────────────────────────────────
    batched_verify_nrows::<BenchZipPlusTypes<i64, 16>, IprsBPolyR4B64<i64, 1, 16, UNCHECKED>, UNCHECKED,  9>(&mut group, 1, BATCH_SIZE);
    batched_verify_nrows::<BenchZipPlusTypes<i64, 16>, IprsBPolyR4B16<i64, 2, 16, UNCHECKED>, UNCHECKED, 10>(&mut group, 1, BATCH_SIZE);
    batched_verify_nrows::<BenchZipPlusTypes<i64, 16>, IprsBPolyR4B32<i64, 2, 16, UNCHECKED>, UNCHECKED, 11>(&mut group, 1, BATCH_SIZE);

    group.finish();
}

/// Batched PCS pipeline suite for BPoly<15>, poly sizes 2^9, 2^10, 2^11 with num_rows=2.
///
/// poly_size (2^P)   Config                 Field        row_len
/// ───────────────   ──────                 ─────        ───────
/// 2^9  = 512        R4B32 D=1 (rate 1/4)   F65537       256
/// 2^10 = 1024       R4B64 D=1 (rate 1/4)   F65537       512
/// 2^11 = 2048       R4B16 D=2 (rate 1/4)   F65537       1024
fn batched_pcs_pipeline_suite_bpoly15_2row(c: &mut Criterion) {
    let mut group = c.benchmark_group("Batched PCS Pipeline Suite BPoly15 2row");
    group.sample_size(100);

    // ── Encode ───────────────────────────────────────────────────────
    batched_encode_nrows::<BenchZipPlusTypes<i64, 16>, IprsBPolyR4B32<i64, 1, 16, UNCHECKED>,  9>(&mut group, 2, BATCH_SIZE);
    batched_encode_nrows::<BenchZipPlusTypes<i64, 16>, IprsBPolyR4B64<i64, 1, 16, UNCHECKED>, 10>(&mut group, 2, BATCH_SIZE);
    batched_encode_nrows::<BenchZipPlusTypes<i64, 16>, IprsBPolyR4B16<i64, 2, 16, UNCHECKED>, 11>(&mut group, 2, BATCH_SIZE);

    // ── Merkle ───────────────────────────────────────────────────────
    batched_merkle_nrows::<BenchZipPlusTypes<i64, 16>, IprsBPolyR4B32<i64, 1, 16, UNCHECKED>,  9>(&mut group, 2, BATCH_SIZE);
    batched_merkle_nrows::<BenchZipPlusTypes<i64, 16>, IprsBPolyR4B64<i64, 1, 16, UNCHECKED>, 10>(&mut group, 2, BATCH_SIZE);
    batched_merkle_nrows::<BenchZipPlusTypes<i64, 16>, IprsBPolyR4B16<i64, 2, 16, UNCHECKED>, 11>(&mut group, 2, BATCH_SIZE);

    // ── Commit ───────────────────────────────────────────────────────
    batched_commit_nrows::<BenchZipPlusTypes<i64, 16>, IprsBPolyR4B32<i64, 1, 16, UNCHECKED>,  9>(&mut group, 2, BATCH_SIZE);
    batched_commit_nrows::<BenchZipPlusTypes<i64, 16>, IprsBPolyR4B64<i64, 1, 16, UNCHECKED>, 10>(&mut group, 2, BATCH_SIZE);
    batched_commit_nrows::<BenchZipPlusTypes<i64, 16>, IprsBPolyR4B16<i64, 2, 16, UNCHECKED>, 11>(&mut group, 2, BATCH_SIZE);

    // ── Test ─────────────────────────────────────────────────────────
    batched_test_nrows::<BenchZipPlusTypes<i64, 16>, IprsBPolyR4B32<i64, 1, 16, UNCHECKED>, UNCHECKED,  9>(&mut group, 2, BATCH_SIZE);
    batched_test_nrows::<BenchZipPlusTypes<i64, 16>, IprsBPolyR4B64<i64, 1, 16, UNCHECKED>, UNCHECKED, 10>(&mut group, 2, BATCH_SIZE);
    batched_test_nrows::<BenchZipPlusTypes<i64, 16>, IprsBPolyR4B16<i64, 2, 16, UNCHECKED>, UNCHECKED, 11>(&mut group, 2, BATCH_SIZE);

    // ── Verify ───────────────────────────────────────────────────────
    batched_verify_nrows::<BenchZipPlusTypes<i64, 16>, IprsBPolyR4B32<i64, 1, 16, UNCHECKED>, UNCHECKED,  9>(&mut group, 2, BATCH_SIZE);
    batched_verify_nrows::<BenchZipPlusTypes<i64, 16>, IprsBPolyR4B64<i64, 1, 16, UNCHECKED>, UNCHECKED, 10>(&mut group, 2, BATCH_SIZE);
    batched_verify_nrows::<BenchZipPlusTypes<i64, 16>, IprsBPolyR4B16<i64, 2, 16, UNCHECKED>, UNCHECKED, 11>(&mut group, 2, BATCH_SIZE);

    group.finish();
}

/// Batched PCS pipeline suite for BPoly<15>, batch_size=7, poly sizes 2^9, 2^10, 2^11 only.
///
/// poly_size (2^P)   Config                 Field        row_len
/// ───────────────   ──────                 ─────        ───────
/// 2^9  = 512        R4B64 D=1 (rate 1/4)   F65537       512
/// 2^10 = 1024       R4B16 D=2 (rate 1/4)   F65537       1024
/// 2^11 = 2048       R4B32 D=2 (rate 1/4)   F65537       2048
fn batched_pcs_pipeline_suite_bpoly15_1row_batch7(c: &mut Criterion) {
    let mut group = c.benchmark_group("Batched PCS Pipeline Suite BPoly15 1row batch7");
    group.sample_size(100);

    // ── Encode ───────────────────────────────────────────────────────
    batched_encode_nrows::<BenchZipPlusTypes<i64, 16>, IprsBPolyR4B64<i64, 1, 16, UNCHECKED>,  9>(&mut group, 1, BATCH_SIZE_7);
    batched_encode_nrows::<BenchZipPlusTypes<i64, 16>, IprsBPolyR4B16<i64, 2, 16, UNCHECKED>, 10>(&mut group, 1, BATCH_SIZE_7);
    batched_encode_nrows::<BenchZipPlusTypes<i64, 16>, IprsBPolyR4B32<i64, 2, 16, UNCHECKED>, 11>(&mut group, 1, BATCH_SIZE_7);

    // ── Merkle ───────────────────────────────────────────────────────
    batched_merkle_nrows::<BenchZipPlusTypes<i64, 16>, IprsBPolyR4B64<i64, 1, 16, UNCHECKED>,  9>(&mut group, 1, BATCH_SIZE_7);
    batched_merkle_nrows::<BenchZipPlusTypes<i64, 16>, IprsBPolyR4B16<i64, 2, 16, UNCHECKED>, 10>(&mut group, 1, BATCH_SIZE_7);
    batched_merkle_nrows::<BenchZipPlusTypes<i64, 16>, IprsBPolyR4B32<i64, 2, 16, UNCHECKED>, 11>(&mut group, 1, BATCH_SIZE_7);

    // ── Commit ───────────────────────────────────────────────────────
    batched_commit_nrows::<BenchZipPlusTypes<i64, 16>, IprsBPolyR4B64<i64, 1, 16, UNCHECKED>,  9>(&mut group, 1, BATCH_SIZE_7);
    batched_commit_nrows::<BenchZipPlusTypes<i64, 16>, IprsBPolyR4B16<i64, 2, 16, UNCHECKED>, 10>(&mut group, 1, BATCH_SIZE_7);
    batched_commit_nrows::<BenchZipPlusTypes<i64, 16>, IprsBPolyR4B32<i64, 2, 16, UNCHECKED>, 11>(&mut group, 1, BATCH_SIZE_7);

    // ── Test ─────────────────────────────────────────────────────────
    batched_test_nrows::<BenchZipPlusTypes<i64, 16>, IprsBPolyR4B64<i64, 1, 16, UNCHECKED>, UNCHECKED,  9>(&mut group, 1, BATCH_SIZE_7);
    batched_test_nrows::<BenchZipPlusTypes<i64, 16>, IprsBPolyR4B16<i64, 2, 16, UNCHECKED>, UNCHECKED, 10>(&mut group, 1, BATCH_SIZE_7);
    batched_test_nrows::<BenchZipPlusTypes<i64, 16>, IprsBPolyR4B32<i64, 2, 16, UNCHECKED>, UNCHECKED, 11>(&mut group, 1, BATCH_SIZE_7);

    // ── Verify ───────────────────────────────────────────────────────
    batched_verify_nrows::<BenchZipPlusTypes<i64, 16>, IprsBPolyR4B64<i64, 1, 16, UNCHECKED>, UNCHECKED,  9>(&mut group, 1, BATCH_SIZE_7);
    batched_verify_nrows::<BenchZipPlusTypes<i64, 16>, IprsBPolyR4B16<i64, 2, 16, UNCHECKED>, UNCHECKED, 10>(&mut group, 1, BATCH_SIZE_7);
    batched_verify_nrows::<BenchZipPlusTypes<i64, 16>, IprsBPolyR4B32<i64, 2, 16, UNCHECKED>, UNCHECKED, 11>(&mut group, 1, BATCH_SIZE_7);

    group.finish();
}

/// Batched PCS pipeline suite for BPoly<15>, batch_size=7, poly sizes 2^9, 2^10, 2^11 with num_rows=2.
///
/// poly_size (2^P)   Config                 Field        row_len
/// ───────────────   ──────                 ─────        ───────
/// 2^9  = 512        R4B32 D=1 (rate 1/4)   F65537       256
/// 2^10 = 1024       R4B64 D=1 (rate 1/4)   F65537       512
/// 2^11 = 2048       R4B16 D=2 (rate 1/4)   F65537       1024
fn batched_pcs_pipeline_suite_bpoly15_2row_batch7(c: &mut Criterion) {
    let mut group = c.benchmark_group("Batched PCS Pipeline Suite BPoly15 2row batch7");
    group.sample_size(100);

    // ── Encode ───────────────────────────────────────────────────────
    batched_encode_nrows::<BenchZipPlusTypes<i64, 16>, IprsBPolyR4B32<i64, 1, 16, UNCHECKED>,  9>(&mut group, 2, BATCH_SIZE_7);
    batched_encode_nrows::<BenchZipPlusTypes<i64, 16>, IprsBPolyR4B64<i64, 1, 16, UNCHECKED>, 10>(&mut group, 2, BATCH_SIZE_7);
    batched_encode_nrows::<BenchZipPlusTypes<i64, 16>, IprsBPolyR4B16<i64, 2, 16, UNCHECKED>, 11>(&mut group, 2, BATCH_SIZE_7);

    // ── Merkle ───────────────────────────────────────────────────────
    batched_merkle_nrows::<BenchZipPlusTypes<i64, 16>, IprsBPolyR4B32<i64, 1, 16, UNCHECKED>,  9>(&mut group, 2, BATCH_SIZE_7);
    batched_merkle_nrows::<BenchZipPlusTypes<i64, 16>, IprsBPolyR4B64<i64, 1, 16, UNCHECKED>, 10>(&mut group, 2, BATCH_SIZE_7);
    batched_merkle_nrows::<BenchZipPlusTypes<i64, 16>, IprsBPolyR4B16<i64, 2, 16, UNCHECKED>, 11>(&mut group, 2, BATCH_SIZE_7);

    // ── Commit ───────────────────────────────────────────────────────
    batched_commit_nrows::<BenchZipPlusTypes<i64, 16>, IprsBPolyR4B32<i64, 1, 16, UNCHECKED>,  9>(&mut group, 2, BATCH_SIZE_7);
    batched_commit_nrows::<BenchZipPlusTypes<i64, 16>, IprsBPolyR4B64<i64, 1, 16, UNCHECKED>, 10>(&mut group, 2, BATCH_SIZE_7);
    batched_commit_nrows::<BenchZipPlusTypes<i64, 16>, IprsBPolyR4B16<i64, 2, 16, UNCHECKED>, 11>(&mut group, 2, BATCH_SIZE_7);

    // ── Test ─────────────────────────────────────────────────────────
    batched_test_nrows::<BenchZipPlusTypes<i64, 16>, IprsBPolyR4B32<i64, 1, 16, UNCHECKED>, UNCHECKED,  9>(&mut group, 2, BATCH_SIZE_7);
    batched_test_nrows::<BenchZipPlusTypes<i64, 16>, IprsBPolyR4B64<i64, 1, 16, UNCHECKED>, UNCHECKED, 10>(&mut group, 2, BATCH_SIZE_7);
    batched_test_nrows::<BenchZipPlusTypes<i64, 16>, IprsBPolyR4B16<i64, 2, 16, UNCHECKED>, UNCHECKED, 11>(&mut group, 2, BATCH_SIZE_7);

    // ── Verify ───────────────────────────────────────────────────────
    batched_verify_nrows::<BenchZipPlusTypes<i64, 16>, IprsBPolyR4B32<i64, 1, 16, UNCHECKED>, UNCHECKED,  9>(&mut group, 2, BATCH_SIZE_7);
    batched_verify_nrows::<BenchZipPlusTypes<i64, 16>, IprsBPolyR4B64<i64, 1, 16, UNCHECKED>, UNCHECKED, 10>(&mut group, 2, BATCH_SIZE_7);
    batched_verify_nrows::<BenchZipPlusTypes<i64, 16>, IprsBPolyR4B16<i64, 2, 16, UNCHECKED>, UNCHECKED, 11>(&mut group, 2, BATCH_SIZE_7);

    group.finish();
}

criterion_group!(benches,
    batched_pcs_pipeline_suite_bpoly31_1row,
    batched_pcs_pipeline_suite_bpoly31_2row,
    batched_pcs_pipeline_suite_bpoly31_1row_batch7,
    batched_pcs_pipeline_suite_bpoly31_2row_batch7,
    batched_pcs_pipeline_suite_bpoly15_1row,
    batched_pcs_pipeline_suite_bpoly15_2row,
    batched_pcs_pipeline_suite_bpoly15_1row_batch7,
    batched_pcs_pipeline_suite_bpoly15_2row_batch7,
);
criterion_main!(benches);
