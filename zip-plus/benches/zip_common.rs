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
use rand::{distr::StandardUniform, prelude::*};
use zip_plus::{
    code::{DefaultLinearCodeSpec, LinearCode},
    field::F256,
    merkle::MerkleTree,
    pcs::structs::{ProjectableToField, ZipPlus, ZipTypes},
    pcs_transcript::PcsTranscript,
    poly::mle::{DenseMultilinearExtension, MultilinearExtensionRand},
    traits::{FromRef, Named},
    transcript::KeccakTranscript,
};

const_monty_params!(
    ModP,
    U256,
    "EB0E9F20F7BFC231327A11792F585AC6C20C74ACCCAB538BE6B0C3AB2E3D176F"
);
type F = F256<ModP>;

pub fn encode_rows<Zt: ZipTypes, const P: usize>(group: &mut BenchmarkGroup<WallTime>)
where
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
            type T = KeccakTranscript;
            let mut keccak_transcript = T::new();
            let poly_size = 1 << P;
            let linear_code = Zt::Code::new(
                &DefaultLinearCodeSpec,
                poly_size,
                false,
                &mut keccak_transcript,
            );
            let params = ZipPlus::<Zt>::setup(poly_size, linear_code);
            let row_len = params.linear_code.row_len();
            let codeword_len = params.linear_code.codeword_len();
            let poly = DenseMultilinearExtension::<<Zt as ZipTypes>::Eval>::rand(P, &mut rng);
            b.iter(|| {
                let _ = ZipPlus::encode_rows(&params, codeword_len, row_len, &poly.evaluations);
            })
        },
    );
}

pub fn encode_single_row<Zt: ZipTypes, const ROW_LEN: usize>(group: &mut BenchmarkGroup<WallTime>)
where
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
            let mut keccak_transcript = KeccakTranscript::new();
            let poly_size = ROW_LEN * ROW_LEN;
            let linear_code = <Zt as ZipTypes>::Code::new(
                &DefaultLinearCodeSpec,
                poly_size,
                false,
                &mut keccak_transcript,
            );
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

pub fn commit<Zt: ZipTypes, const P: usize>(group: &mut BenchmarkGroup<WallTime>)
where
    StandardUniform: Distribution<Zt::Eval>,
{
    let mut rng = ThreadRng::default();
    type T = KeccakTranscript;
    let mut keccak_transcript = T::new();
    let poly_size = 1 << P;
    let linear_code = <Zt as ZipTypes>::Code::new(
        &DefaultLinearCodeSpec,
        poly_size,
        false,
        &mut keccak_transcript,
    );
    let params = ZipPlus::<Zt>::setup(poly_size, linear_code);

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

pub fn open<Zt: ZipTypes, const P: usize>(group: &mut BenchmarkGroup<WallTime>)
where
    StandardUniform: Distribution<Zt::Eval>,
    Zt::Eval: ProjectableToField<F>,
{
    let mut rng = ThreadRng::default();

    type T = KeccakTranscript;
    let mut keccak_transcript = T::new();
    let poly_size = 1 << P;
    let linear_code = <Zt as ZipTypes>::Code::new(
        &DefaultLinearCodeSpec,
        poly_size,
        false,
        &mut keccak_transcript,
    );
    let params = ZipPlus::<Zt>::setup(poly_size, linear_code);

    let poly = DenseMultilinearExtension::rand(P, &mut rng);
    let (data, _) = ZipPlus::commit(&params, &poly).unwrap();
    let point = vec![1i64; P];
    let field_point: Vec<F> = point.iter().map(F::from).collect();

    group.bench_function(
        format!(
            "Open: Eval={}, Codeword={}, Combination={}, poly_size=2^{P}, modulus={}",
            Zt::Eval::type_name(),
            Zt::Cw::type_name(),
            Zt::Comb::type_name(),
            ModP::PARAMS.modulus()
        ),
        |b| {
            b.iter_custom(|iters| {
                let mut total_duration = Duration::ZERO;
                for _ in 0..iters {
                    let mut transcript = PcsTranscript::new();
                    let timer = Instant::now();
                    ZipPlus::open(&params, &poly, &data, &field_point, &mut transcript)
                        .expect("Failed to make opening");
                    total_duration += timer.elapsed();
                }
                total_duration
            })
        },
    );
}

pub fn verify<Zt: ZipTypes, const P: usize>(group: &mut BenchmarkGroup<WallTime>)
where
    StandardUniform: Distribution<Zt::Eval>,
    F: FromRef<<Zt::Code as LinearCode<Zt::Eval, Zt::Cw, Zt::CombR>>::Inner>,
    Zt::Eval: ProjectableToField<F>,
    Zt::Cw: ProjectableToField<F>,
{
    let mut rng = ThreadRng::default();
    type T = KeccakTranscript;
    let mut keccak_transcript = T::new();
    let poly_size = 1 << P;
    let linear_code = <Zt as ZipTypes>::Code::new(
        &DefaultLinearCodeSpec,
        poly_size,
        false,
        &mut keccak_transcript,
    );
    let code_row_len = linear_code.row_len();
    let params = ZipPlus::<Zt>::setup(poly_size, linear_code);

    let poly = DenseMultilinearExtension::rand(P, &mut rng);
    let (data, commitment) = ZipPlus::commit(&params, &poly).unwrap();
    let point = vec![1i64; P];
    let field_point: Vec<F> = point.iter().map(F::from).collect();
    let mut transcript = PcsTranscript::new();

    ZipPlus::open(&params, &poly, &data, &field_point, &mut transcript).unwrap();

    let project = {
        let mut transcript = transcript.clone();
        let _ = transcript.read_field_elements::<F>(code_row_len);
        Zt::Eval::read_projection(&mut transcript)
    };
    let eval = poly.evaluations.last().unwrap();
    let eval_field: F = project(eval);

    let proof = transcript.into_proof();

    group.bench_function(
        format!(
            "Verify: Eval={}, Codeword={}, Combination={}, poly_size=2^{P}, modulus={}",
            Zt::Eval::type_name(),
            Zt::Cw::type_name(),
            Zt::Comb::type_name(),
            ModP::PARAMS.modulus()
        ),
        |b| {
            b.iter_custom(|iters| {
                let mut total_duration = Duration::ZERO;
                for _ in 0..iters {
                    let mut transcript = PcsTranscript::from_proof(&proof);
                    let timer = Instant::now();
                    ZipPlus::verify(
                        &params,
                        &commitment,
                        &field_point,
                        &eval_field,
                        &mut transcript,
                    )
                    .expect("Failed to verify");
                    total_duration += timer.elapsed();
                }
                total_duration
            })
        },
    );
}
