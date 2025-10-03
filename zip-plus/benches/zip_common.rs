//! This module contains common benchmarking code for the Zip+ PCS,
//! both for Zip (integer coefficients) and Zip+ (polynomial coefficients).

#![allow(non_local_definitions)]
#![allow(clippy::eq_op, clippy::arithmetic_side_effects, clippy::unwrap_used)]

use ark_std::{
    hint::black_box,
    time::{Duration, Instant},
};
use criterion::{BenchmarkGroup, measurement::WallTime};
use crypto_bigint::{U256, const_monty_params, modular::ConstMontyParams};
use itertools::Itertools;
use num_traits::ConstOne;
use rand::{distr::StandardUniform, prelude::*};
use zip_plus::{
    code::LinearCode,
    field::F256,
    merkle::MerkleTree,
    pcs::structs::{ProjectableToField, ZipPlus, ZipTypes},
    poly::mle::{DenseMultilinearExtension, MultilinearExtensionRand},
    traits::{FromRef, Named},
};

const_monty_params!(
    ModP,
    U256,
    "EB0E9F20F7BFC231327A11792F585AC6C20C74ACCCAB538BE6B0C3AB2E3D176F"
);
type F = F256<ModP>;

pub fn do_bench<Zt: ZipTypes, Lc: LinearCode<Zt>>(group: &mut BenchmarkGroup<WallTime>)
where
    StandardUniform: Distribution<Zt::Eval> + Distribution<Zt::Cw>,
    F: FromRef<Zt::Chal> + FromRef<Zt::Pt>,
    Zt::Eval: ProjectableToField<F>,
    Zt::Cw: ProjectableToField<F>,
{
    encode_rows::<Zt, Lc, 12>(group);
    encode_rows::<Zt, Lc, 13>(group);
    encode_rows::<Zt, Lc, 14>(group);
    encode_rows::<Zt, Lc, 15>(group);
    encode_rows::<Zt, Lc, 16>(group);

    encode_single_row::<Zt, Lc, 128>(group);
    encode_single_row::<Zt, Lc, 256>(group);
    encode_single_row::<Zt, Lc, 512>(group);
    encode_single_row::<Zt, Lc, 1024>(group);
    encode_single_row::<Zt, Lc, 2048>(group);
    encode_single_row::<Zt, Lc, 4096>(group);

    merkle_root::<Zt, 12>(group);
    merkle_root::<Zt, 13>(group);
    merkle_root::<Zt, 14>(group);
    merkle_root::<Zt, 15>(group);
    merkle_root::<Zt, 16>(group);

    commit::<Zt, Lc, 12>(group);
    commit::<Zt, Lc, 13>(group);
    commit::<Zt, Lc, 14>(group);
    commit::<Zt, Lc, 15>(group);
    commit::<Zt, Lc, 16>(group);

    test::<Zt, Lc, 12>(group);
    test::<Zt, Lc, 13>(group);
    test::<Zt, Lc, 14>(group);
    test::<Zt, Lc, 15>(group);
    test::<Zt, Lc, 16>(group);

    evaluate::<Zt, Lc, 12>(group);
    evaluate::<Zt, Lc, 13>(group);
    evaluate::<Zt, Lc, 14>(group);
    evaluate::<Zt, Lc, 15>(group);
    evaluate::<Zt, Lc, 16>(group);

    verify::<Zt, Lc, 12>(group);
    verify::<Zt, Lc, 13>(group);
    verify::<Zt, Lc, 14>(group);
    verify::<Zt, Lc, 15>(group);
    verify::<Zt, Lc, 16>(group);
}

pub fn encode_rows<Zt: ZipTypes, Lc: LinearCode<Zt>, const P: usize>(
    group: &mut BenchmarkGroup<WallTime>,
) where
    StandardUniform: Distribution<Zt::Eval>,
{
    group.bench_function(
        format!(
            "EncodeRows: {} -> {}, poly_size = 2^{P})",
            Zt::Eval::type_name(),
            Zt::Cw::type_name()
        ),
        |b| {
            let mut rng = ThreadRng::default();
            let poly_size = 1 << P;
            let linear_code = Lc::new(poly_size, false);
            let params = ZipPlus::setup(poly_size, linear_code);
            let row_len = params.linear_code.row_len();
            let codeword_len = params.linear_code.codeword_len();
            let poly = DenseMultilinearExtension::<<Zt as ZipTypes>::Eval>::rand(P, &mut rng);
            b.iter(|| {
                let cw = ZipPlus::encode_rows(&params, codeword_len, row_len, &poly.evaluations);
                black_box(cw)
            })
        },
    );
}

pub fn encode_single_row<Zt: ZipTypes, Lc: LinearCode<Zt>, const ROW_LEN: usize>(
    group: &mut BenchmarkGroup<WallTime>,
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
            let linear_code = Lc::new(poly_size, false);
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

    group.bench_function(
        format!("MerkleRoot: {}, leaves=2^{P}", Zt::Cw::type_name()),
        |b| {
            b.iter(|| {
                let tree = MerkleTree::new(&leaves, num_leaves);
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
    let linear_code = Lc::new(poly_size, false);
    let params = ZipPlus::setup(poly_size, linear_code);

    group.bench_function(
        format!(
            "Commit: Eval={}, Codeword={}, Combination={}, poly_size=2^{P}",
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

pub fn test<Zt: ZipTypes, Lc: LinearCode<Zt>, const P: usize>(group: &mut BenchmarkGroup<WallTime>)
where
    StandardUniform: Distribution<Zt::Eval>,
{
    let mut rng = ThreadRng::default();

    let poly_size = 1 << P;
    let linear_code = Lc::new(poly_size, false);
    let params = ZipPlus::setup(poly_size, linear_code);

    let poly = DenseMultilinearExtension::rand(P, &mut rng);
    let (data, _) = ZipPlus::commit(&params, &poly).unwrap();

    group.bench_function(
        format!(
            "Test: Eval={}, Codeword={}, Combination={}, poly_size=2^{P}, modulus={}",
            Zt::Eval::type_name(),
            Zt::Cw::type_name(),
            Zt::Comb::type_name(),
            ModP::PARAMS.modulus()
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
) where
    StandardUniform: Distribution<Zt::Eval>,
    Zt::Eval: ProjectableToField<F>,
    F: FromRef<Zt::Chal> + FromRef<Zt::Pt>,
{
    let mut rng = ThreadRng::default();

    let poly_size = 1 << P;
    let linear_code = Lc::new(poly_size, false);
    let params = ZipPlus::setup(poly_size, linear_code);

    let poly = DenseMultilinearExtension::rand(P, &mut rng);
    let (data, _) = ZipPlus::commit(&params, &poly).unwrap();
    let point = vec![Zt::Pt::ONE; P];

    let test_transcript = ZipPlus::test(&params, &poly, &data).expect("Test phase failed");

    group.bench_function(
        format!(
            "Evaluate: Eval={}, Codeword={}, Combination={}, poly_size=2^{P}, modulus={}",
            Zt::Eval::type_name(),
            Zt::Cw::type_name(),
            Zt::Comb::type_name(),
            ModP::PARAMS.modulus()
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
) where
    StandardUniform: Distribution<Zt::Eval>,
    F: FromRef<Zt::Chal> + FromRef<Zt::Pt>,
    Zt::Eval: ProjectableToField<F>,
    Zt::Cw: ProjectableToField<F>,
{
    let mut rng = ThreadRng::default();
    let poly_size = 1 << P;
    let linear_code = Lc::new(poly_size, false);
    let params = ZipPlus::setup(poly_size, linear_code);

    let poly = DenseMultilinearExtension::rand(P, &mut rng);
    let (data, commitment) = ZipPlus::commit(&params, &poly).unwrap();
    let point = vec![Zt::Pt::ONE; P];
    let point_f: Vec<F> = point.iter().map(F::from_ref).collect();

    let test_transcript = ZipPlus::test(&params, &poly, &data).expect("Test phase failed");
    let (eval_f, proof) = ZipPlus::evaluate::<F>(&params, &poly, &point, test_transcript)
        .expect("Evaluation phase failed");

    group.bench_function(
        format!(
            "Verify: Eval={}, Codeword={}, Combination={}, poly_size=2^{P}, modulus={}",
            Zt::Eval::type_name(),
            Zt::Cw::type_name(),
            Zt::Comb::type_name(),
            ModP::PARAMS.modulus()
        ),
        |b| {
            b.iter(|| {
                ZipPlus::verify(&params, &commitment, &point_f, &eval_f, &proof)
                    .expect("Verification failed");
            })
        },
    );
}
