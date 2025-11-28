//! This module contains common benchmarking code for the Zip+ PCS,
//! both for Zip (integer coefficients) and Zip+ (polynomial coefficients).

#![allow(non_local_definitions)]
#![allow(clippy::eq_op, clippy::arithmetic_side_effects, clippy::unwrap_used)]

use ark_std::{
    hint::black_box,
    time::{Duration, Instant},
};
use criterion::{BenchmarkGroup, measurement::WallTime};
use crypto_primitives::{
    DenseRowMatrix, Field, FromWithConfig, IntoWithConfig, PrimeField,
    crypto_bigint_boxed_monty::BoxedMontyField,
};
use itertools::Itertools;
use num_traits::One;
use rand::{distr::StandardUniform, prelude::*};
use zinc_poly::mle::{DenseMultilinearExtension, MultilinearExtensionRand};
use zinc_transcript::traits::ConstTranscribable;
use zinc_utils::{from_ref::FromRef, named::Named, projectable_to_field::ProjectableToField};
use zip_plus::{
    code::LinearCode,
    merkle::MerkleTree,
    pcs::structs::{ZipPlus, ZipTypes},
};

type F = BoxedMontyField;

pub fn do_bench<Zt: ZipTypes, Lc: LinearCode<Zt>>(
    group: &mut BenchmarkGroup<WallTime>,
    code_config: Lc::Config,
) where
    StandardUniform: Distribution<Zt::Eval> + Distribution<Zt::Cw>,
    F: for<'a> FromWithConfig<&'a Zt::Chal> + for<'a> FromWithConfig<&'a Zt::Pt>,
    <F as Field>::Inner: FromRef<Zt::Fmod>,
    Zt::Eval: ProjectableToField<F>,
    Zt::Cw: ProjectableToField<F>,
{
    encode_rows::<Zt, Lc, 12>(group, code_config);
    encode_rows::<Zt, Lc, 13>(group, code_config);
    encode_rows::<Zt, Lc, 14>(group, code_config);
    encode_rows::<Zt, Lc, 15>(group, code_config);
    encode_rows::<Zt, Lc, 16>(group, code_config);

    encode_single_row::<Zt, Lc, 128>(group, code_config);
    encode_single_row::<Zt, Lc, 256>(group, code_config);
    encode_single_row::<Zt, Lc, 512>(group, code_config);
    encode_single_row::<Zt, Lc, 1024>(group, code_config);

    merkle_root::<Zt, 12>(group);
    merkle_root::<Zt, 13>(group);
    merkle_root::<Zt, 14>(group);
    merkle_root::<Zt, 15>(group);
    merkle_root::<Zt, 16>(group);

    commit::<Zt, Lc, 12>(group, code_config);
    commit::<Zt, Lc, 13>(group, code_config);
    commit::<Zt, Lc, 14>(group, code_config);
    commit::<Zt, Lc, 15>(group, code_config);
    commit::<Zt, Lc, 16>(group, code_config);

    test::<Zt, Lc, 12>(group, code_config);
    test::<Zt, Lc, 13>(group, code_config);
    test::<Zt, Lc, 14>(group, code_config);
    test::<Zt, Lc, 15>(group, code_config);
    test::<Zt, Lc, 16>(group, code_config);

    evaluate::<Zt, Lc, 12>(group, code_config);
    evaluate::<Zt, Lc, 13>(group, code_config);
    evaluate::<Zt, Lc, 14>(group, code_config);
    evaluate::<Zt, Lc, 15>(group, code_config);
    evaluate::<Zt, Lc, 16>(group, code_config);

    verify::<Zt, Lc, 12>(group, code_config);
    verify::<Zt, Lc, 13>(group, code_config);
    verify::<Zt, Lc, 14>(group, code_config);
    verify::<Zt, Lc, 15>(group, code_config);
    verify::<Zt, Lc, 16>(group, code_config);
}

pub fn encode_rows<Zt: ZipTypes, Lc: LinearCode<Zt>, const P: usize>(
    group: &mut BenchmarkGroup<WallTime>,
    code_config: Lc::Config,
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
            let linear_code = Lc::new(poly_size, code_config);
            let params = ZipPlus::setup(poly_size, linear_code);
            let row_len = params.linear_code.row_len();
            let poly = DenseMultilinearExtension::<<Zt as ZipTypes>::Eval>::rand(P, &mut rng);
            b.iter(|| {
                let cw = ZipPlus::encode_rows(&params, row_len, &poly.evaluations);
                black_box(cw)
            })
        },
    );
}

pub fn encode_single_row<Zt: ZipTypes, Lc: LinearCode<Zt>, const ROW_LEN: usize>(
    group: &mut BenchmarkGroup<WallTime>,
    code_config: Lc::Config,
) where
    StandardUniform: Distribution<Zt::Eval>,
{
    group.bench_function(
        format!(
            "EncodeMessage: {} -> {}, row_len = {ROW_LEN}",
            Zt::Eval::type_name(),
            Zt::Cw::type_name()
        ),
        |b| {
            let mut rng = ThreadRng::default();
            let poly_size = ROW_LEN * ROW_LEN;
            let linear_code = Lc::new(poly_size, code_config);
            assert_eq!(linear_code.row_len(), ROW_LEN, "Unexpected row_len");
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
    code_config: Lc::Config,
) where
    StandardUniform: Distribution<Zt::Eval>,
{
    let mut rng = ThreadRng::default();
    let poly_size = 1 << P;
    let linear_code = Lc::new(poly_size, code_config);
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

pub fn test<Zt: ZipTypes, Lc: LinearCode<Zt>, const P: usize>(
    group: &mut BenchmarkGroup<WallTime>,
    code_config: Lc::Config,
) where
    StandardUniform: Distribution<Zt::Eval>,
{
    let mut rng = ThreadRng::default();

    let poly_size = 1 << P;
    let linear_code = Lc::new(poly_size, code_config);
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
                let test_transcript =
                    ZipPlus::test(&params, &poly, &data).expect("Test phase failed");
                black_box(test_transcript);
            })
        },
    );
}

pub fn evaluate<Zt: ZipTypes, Lc: LinearCode<Zt>, const P: usize>(
    group: &mut BenchmarkGroup<WallTime>,
    code_config: Lc::Config,
) where
    StandardUniform: Distribution<Zt::Eval>,
    F: for<'a> FromWithConfig<&'a Zt::Chal> + for<'a> FromWithConfig<&'a Zt::Pt>,
    <F as Field>::Inner: FromRef<Zt::Fmod>,
    Zt::Eval: ProjectableToField<F>,
{
    let mut rng = ThreadRng::default();

    let poly_size = 1 << P;
    let linear_code = Lc::new(poly_size, code_config);
    let params = ZipPlus::setup(poly_size, linear_code);

    let poly = DenseMultilinearExtension::rand(P, &mut rng);
    let (data, _) = ZipPlus::commit(&params, &poly).unwrap();
    let point = vec![Zt::Pt::one(); P];

    let test_transcript = ZipPlus::test(&params, &poly, &data).expect("Test phase failed");

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
                    let (eval_f, proof) = ZipPlus::evaluate::<F>(&params, &poly, &point, proof)
                        .expect("Evaluation phase failed");
                    total_duration += timer.elapsed();
                    black_box((eval_f, proof));
                }
                total_duration
            })
        },
    );
}

pub fn verify<Zt: ZipTypes, Lc: LinearCode<Zt>, const P: usize>(
    group: &mut BenchmarkGroup<WallTime>,
    code_config: Lc::Config,
) where
    StandardUniform: Distribution<Zt::Eval>,
    F: for<'a> FromWithConfig<&'a Zt::Chal> + for<'a> FromWithConfig<&'a Zt::Pt>,
    <F as Field>::Inner: FromRef<Zt::Fmod>,
    Zt::Eval: ProjectableToField<F>,
    Zt::Cw: ProjectableToField<F>,
{
    let mut rng = ThreadRng::default();
    let poly_size = 1 << P;
    let linear_code = Lc::new(poly_size, code_config);
    let params = ZipPlus::setup(poly_size, linear_code);

    let poly = DenseMultilinearExtension::rand(P, &mut rng);
    let (data, commitment) = ZipPlus::commit(&params, &poly).unwrap();
    let point = vec![Zt::Pt::one(); P];

    let test_transcript = ZipPlus::test(&params, &poly, &data).expect("Test phase failed");
    let (eval_f, proof) = ZipPlus::evaluate::<F>(&params, &poly, &point, test_transcript)
        .expect("Evaluation phase failed");
    let field_cfg = eval_f.cfg().clone();
    let point_f: Vec<F> = point.iter().map(|v| v.into_with_cfg(&field_cfg)).collect();

    group.bench_function(
        format!(
            "Verify: Eval={}, Cw={}, Comb={}, poly_size=2^{P}, modulus=({} bits)",
            Zt::Eval::type_name(),
            Zt::Cw::type_name(),
            Zt::Comb::type_name(),
            Zt::Fmod::NUM_BYTES * 8
        ),
        |b| {
            b.iter(|| {
                ZipPlus::verify(&params, &commitment, &point_f, &eval_f, &proof)
                    .expect("Verification failed");
            })
        },
    );
}
