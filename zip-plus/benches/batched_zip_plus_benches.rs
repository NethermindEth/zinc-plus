//! Benchmarks for the Batched Zip+ PCS.
//!
//! Measures commit, test, evaluate, and verify for batches of polynomials
//! using a single shared Merkle tree.

#![allow(non_local_definitions)]
#![allow(clippy::eq_op, clippy::arithmetic_side_effects, clippy::unwrap_used)]

mod zip_common;

use criterion::{BenchmarkGroup, Criterion, criterion_group, criterion_main, measurement::WallTime};
use crypto_bigint::U64;
use crypto_primitives::{
    Field, FromWithConfig, IntoWithConfig, PrimeField, crypto_bigint_int::Int,
    crypto_bigint_monty::MontyField, crypto_bigint_uint::Uint,
};
use num_traits::One;
use rand::{distr::StandardUniform, prelude::*};
use std::{
    hint::black_box,
    time::{Duration, Instant},
};
use zinc_poly::mle::{DenseMultilinearExtension, MultilinearExtensionRand};
use zinc_primality::MillerRabin;
use zinc_transcript::traits::ConstTranscribable;
use zinc_utils::{
    UNCHECKED, from_ref::FromRef, inner_product::{MBSInnerProduct, ScalarProduct},
    named::Named, projectable_to_field::ProjectableToField,
};
use zip_plus::{
    batched_pcs::structs::BatchedZipPlus,
    code::{
        LinearCode,
        raa::{RaaCode, RaaConfig},
    },
    merkle::MerkleTree,
    pcs::structs::{ZipPlus, ZipTypes},
};

const INT_LIMBS: usize = U64::LIMBS;
type F = MontyField<{ INT_LIMBS * 4 }>;

// ---------- ZipTypes for scalar benchmarks ----------

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

// ---------- Batched benchmark helpers ----------

/// Benchmark the batched commit phase for a given polynomial size and batch size.
fn batched_commit<Zt: ZipTypes, Lc: LinearCode<Zt>, const P: usize>(
    group: &mut BenchmarkGroup<WallTime>,
    batch_size: usize,
) where
    StandardUniform: Distribution<Zt::Eval>,
{
    let mut rng = ThreadRng::default();
    let poly_size = 1 << P;
    let linear_code = Lc::new(poly_size);
    let params = ZipPlus::<Zt, Lc>::setup(poly_size, linear_code);

    group.bench_function(
        format!(
            "BatchedCommit: Eval={}, Cw={}, poly_size=2^{P}, batch={batch_size}",
            Zt::Eval::type_name(),
            Zt::Cw::type_name(),
        ),
        |b| {
            b.iter_custom(|iters| {
                let mut total_duration = Duration::ZERO;
                for _ in 0..iters {
                    let polys: Vec<_> = (0..batch_size)
                        .map(|_| DenseMultilinearExtension::rand(P, &mut rng))
                        .collect();
                    let timer = Instant::now();
                    let res = BatchedZipPlus::<Zt, Lc>::commit(&params, &polys)
                        .expect("Batched commit failed");
                    black_box(res);
                    total_duration += timer.elapsed();
                }
                total_duration
            })
        },
    );
}

/// Benchmark the batched test phase for a given polynomial size and batch size.
fn batched_test<
    Zt: ZipTypes,
    Lc: LinearCode<Zt>,
    const CHECK_FOR_OVERFLOWS: bool,
    const P: usize,
>(
    group: &mut BenchmarkGroup<WallTime>,
    batch_size: usize,
) where
    StandardUniform: Distribution<Zt::Eval>,
{
    let mut rng = ThreadRng::default();
    let poly_size = 1 << P;
    let linear_code = Lc::new(poly_size);
    let params = ZipPlus::<Zt, Lc>::setup(poly_size, linear_code);

    let polys: Vec<_> = (0..batch_size)
        .map(|_| DenseMultilinearExtension::rand(P, &mut rng))
        .collect();
    let (hint, _) = BatchedZipPlus::<Zt, Lc>::commit(&params, &polys).unwrap();

    group.bench_function(
        format!(
            "BatchedTest: Eval={}, Cw={}, Comb={}, poly_size=2^{P}, batch={batch_size}",
            Zt::Eval::type_name(),
            Zt::Cw::type_name(),
            Zt::Comb::type_name(),
        ),
        |b| {
            b.iter(|| {
                let transcript =
                    BatchedZipPlus::<Zt, Lc>::test::<CHECK_FOR_OVERFLOWS>(&params, &polys, &hint)
                        .expect("Batched test phase failed");
                black_box(transcript);
            })
        },
    );
}

/// Benchmark the batched evaluate phase.
fn batched_evaluate<
    Zt: ZipTypes,
    Lc: LinearCode<Zt>,
    const CHECK_FOR_OVERFLOWS: bool,
    const P: usize,
>(
    group: &mut BenchmarkGroup<WallTime>,
    batch_size: usize,
) where
    StandardUniform: Distribution<Zt::Eval>,
    F: for<'a> FromWithConfig<&'a Zt::Chal> + for<'a> FromWithConfig<&'a Zt::Pt>,
    <F as Field>::Inner: FromRef<Zt::Fmod>,
    Zt::Eval: ProjectableToField<F>,
{
    let mut rng = ThreadRng::default();
    let poly_size = 1 << P;
    let linear_code = Lc::new(poly_size);
    let params = ZipPlus::<Zt, Lc>::setup(poly_size, linear_code);

    let polys: Vec<_> = (0..batch_size)
        .map(|_| DenseMultilinearExtension::rand(P, &mut rng))
        .collect();
    let (hint, _) = BatchedZipPlus::<Zt, Lc>::commit(&params, &polys).unwrap();
    let point = vec![Zt::Pt::one(); P];

    let test_transcript =
        BatchedZipPlus::<Zt, Lc>::test::<CHECK_FOR_OVERFLOWS>(&params, &polys, &hint)
            .expect("Batched test phase failed");

    group.bench_function(
        format!(
            "BatchedEvaluate: Eval={}, Cw={}, Comb={}, poly_size=2^{P}, batch={batch_size}, modulus=({} bits)",
            Zt::Eval::type_name(),
            Zt::Cw::type_name(),
            Zt::Comb::type_name(),
            Zt::Fmod::NUM_BYTES * 8,
        ),
        |b| {
            b.iter_custom(|iters| {
                let mut total_duration = Duration::ZERO;
                for _ in 0..iters {
                    let transcript = test_transcript.clone();
                    let timer = Instant::now();
                    let (evals_f, proof) = BatchedZipPlus::<Zt, Lc>::evaluate::<
                        F,
                        CHECK_FOR_OVERFLOWS,
                    >(
                        &params, &polys, &point, transcript
                    )
                    .expect("Batched evaluate failed");
                    total_duration += timer.elapsed();
                    black_box((evals_f, proof));
                }
                total_duration
            })
        },
    );
}

/// Benchmark the batched verify phase.
fn batched_verify<
    Zt: ZipTypes,
    Lc: LinearCode<Zt>,
    const CHECK_FOR_OVERFLOWS: bool,
    const P: usize,
>(
    group: &mut BenchmarkGroup<WallTime>,
    batch_size: usize,
) where
    StandardUniform: Distribution<Zt::Eval>,
    F: for<'a> FromWithConfig<&'a Zt::Chal> + for<'a> FromWithConfig<&'a Zt::Pt>,
    <F as Field>::Inner: FromRef<Zt::Fmod>,
    Zt::Eval: ProjectableToField<F>,
    Zt::Cw: ProjectableToField<F>,
{
    let mut rng = ThreadRng::default();
    let poly_size = 1 << P;
    let linear_code = Lc::new(poly_size);
    let params = ZipPlus::<Zt, Lc>::setup(poly_size, linear_code);

    let polys: Vec<_> = (0..batch_size)
        .map(|_| DenseMultilinearExtension::rand(P, &mut rng))
        .collect();
    let (hint, commitment) = BatchedZipPlus::<Zt, Lc>::commit(&params, &polys).unwrap();
    let point = vec![Zt::Pt::one(); P];

    let test_transcript =
        BatchedZipPlus::<Zt, Lc>::test::<CHECK_FOR_OVERFLOWS>(&params, &polys, &hint)
            .expect("Batched test phase failed");
    let (evals_f, proof) = BatchedZipPlus::<Zt, Lc>::evaluate::<F, CHECK_FOR_OVERFLOWS>(
        &params,
        &polys,
        &point,
        test_transcript,
    )
    .expect("Batched evaluate failed");
    let field_cfg = *evals_f[0].cfg();
    let point_f: Vec<F> = point.iter().map(|v| v.into_with_cfg(&field_cfg)).collect();

    group.bench_function(
        format!(
            "BatchedVerify: Eval={}, Cw={}, Comb={}, poly_size=2^{P}, batch={batch_size}, modulus=({} bits)",
            Zt::Eval::type_name(),
            Zt::Cw::type_name(),
            Zt::Comb::type_name(),
            Zt::Fmod::NUM_BYTES * 8,
        ),
        |b| {
            b.iter(|| {
                BatchedZipPlus::<Zt, Lc>::verify::<_, CHECK_FOR_OVERFLOWS>(
                    &params,
                    &commitment,
                    &point_f,
                    &evals_f,
                    &proof,
                )
                .expect("Batched verification failed");
            })
        },
    );
}

/// Benchmark only the Merkle tree construction for a batched commit.
///
/// This isolates the cost of `MerkleTree::new` on the concatenated codeword
/// rows of `batch_size` polynomials.  Encoding is done once in setup so the
/// measurement captures only the tree building.
fn batched_merkle_tree<Zt: ZipTypes, Lc: LinearCode<Zt>, const P: usize>(
    group: &mut BenchmarkGroup<WallTime>,
    batch_size: usize,
) where
    StandardUniform: Distribution<Zt::Eval>,
{
    let mut rng = ThreadRng::default();
    let poly_size = 1 << P;
    let linear_code = Lc::new(poly_size);
    let params = ZipPlus::<Zt, Lc>::setup(poly_size, linear_code);

    let row_len = params.linear_code.row_len();

    // Pre-encode all polynomials
    let cw_matrices: Vec<_> = (0..batch_size)
        .map(|_| {
            let poly = DenseMultilinearExtension::rand(P, &mut rng);
            ZipPlus::<Zt, Lc>::encode_rows(&params, row_len, &poly)
        })
        .collect();

    // Collect row slices for MerkleTree::new (same layout as BatchedZipPlus::commit)
    let all_rows: Vec<&[Zt::Cw]> = cw_matrices
        .iter()
        .flat_map(|m| m.to_rows_slices())
        .collect();

    group.bench_function(
        format!(
            "BatchedMerkleTree: Cw={}, poly_size=2^{P}, batch={batch_size}, total_rows={}",
            Zt::Cw::type_name(),
            all_rows.len(),
        ),
        |b| {
            b.iter(|| {
                let tree = MerkleTree::new(&all_rows);
                black_box(tree.root());
            })
        },
    );
}

/// Benchmark building `batch_size` separate Merkle trees (one per polynomial)
/// vs the batched single-tree approach.  This lets us directly compare the
/// cost of m independent trees against 1 batched tree.
fn separate_merkle_trees<Zt: ZipTypes, Lc: LinearCode<Zt>, const P: usize>(
    group: &mut BenchmarkGroup<WallTime>,
    batch_size: usize,
) where
    StandardUniform: Distribution<Zt::Eval>,
{
    let mut rng = ThreadRng::default();
    let poly_size = 1 << P;
    let linear_code = Lc::new(poly_size);
    let params = ZipPlus::<Zt, Lc>::setup(poly_size, linear_code);

    let row_len = params.linear_code.row_len();

    // Pre-encode all polynomials
    let cw_matrices: Vec<_> = (0..batch_size)
        .map(|_| {
            let poly = DenseMultilinearExtension::rand(P, &mut rng);
            ZipPlus::<Zt, Lc>::encode_rows(&params, row_len, &poly)
        })
        .collect();

    // Pre-collect row slices per polynomial
    let per_poly_rows: Vec<Vec<&[Zt::Cw]>> = cw_matrices
        .iter()
        .map(|m| m.to_rows_slices().into_iter().collect())
        .collect();

    group.bench_function(
        format!(
            "SeparateMerkleTrees: Cw={}, poly_size=2^{P}, batch={batch_size}, rows_per_tree={}",
            Zt::Cw::type_name(),
            per_poly_rows[0].len(),
        ),
        |b| {
            b.iter(|| {
                for rows in &per_poly_rows {
                    let tree = MerkleTree::new(rows);
                    black_box(tree.root());
                }
            })
        },
    );
}

/// Run all batched benchmarks for a given ZipTypes/Code/P combination
/// with several batch sizes.
fn do_batched_bench<
    Zt: ZipTypes,
    Lc: LinearCode<Zt>,
    const CHECK_FOR_OVERFLOWS: bool,
    const P: usize,
>(
    group: &mut BenchmarkGroup<WallTime>,
) where
    StandardUniform: Distribution<Zt::Eval> + Distribution<Zt::Cw>,
    F: for<'a> FromWithConfig<&'a Zt::Chal> + for<'a> FromWithConfig<&'a Zt::Pt>,
    <F as Field>::Inner: FromRef<Zt::Fmod>,
    Zt::Eval: ProjectableToField<F>,
    Zt::Cw: ProjectableToField<F>,
{
    for &batch_size in &[1, 2, 5, 10] {
        batched_commit::<Zt, Lc, P>(group, batch_size);
        batched_merkle_tree::<Zt, Lc, P>(group, batch_size);
        separate_merkle_trees::<Zt, Lc, P>(group, batch_size);
        batched_test::<Zt, Lc, CHECK_FOR_OVERFLOWS, P>(group, batch_size);
        batched_evaluate::<Zt, Lc, CHECK_FOR_OVERFLOWS, P>(group, batch_size);
        batched_verify::<Zt, Lc, CHECK_FOR_OVERFLOWS, P>(group, batch_size);
    }
}

// ---------- Criterion entry point ----------

fn batched_zip_plus_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("Batched Zip+");

    do_batched_bench::<BenchZipTypes, Code, UNCHECKED, 12>(&mut group);
    do_batched_bench::<BenchZipTypes, Code, UNCHECKED, 14>(&mut group);
    do_batched_bench::<BenchZipTypes, Code, UNCHECKED, 16>(&mut group);

    group.finish();
}

criterion_group!(benches, batched_zip_plus_benchmarks);
criterion_main!(benches);
