#![allow(non_local_definitions)]
#![allow(clippy::eq_op)]

use std::{hint::black_box, ops::{Add, Mul}};

use criterion::{
    AxisScale, BatchSize, BenchmarkGroup, BenchmarkId, Criterion, PlotConfiguration,
    criterion_group, criterion_main, measurement::WallTime,
};
use crypto_primitives::{
    ConstIntSemiring, Field, FromPrimitiveWithConfig, crypto_bigint_monty::MontyField,
};
use num_traits::Zero;
use rand::{Rng, rng};
use zinc_piop::random_field_sumcheck::{RFSumcheck, RFSumcheckProof};
use zinc_poly::{
    mle::DenseMultilinearExtension, univariate::binary::BinaryPoly, utils::build_eq_x_r_inner,
};
use zinc_primality::{MillerRabin, PrimalityTest};
use zinc_transcript::{
    KeccakTranscript,
    traits::{ConstTranscribable, Transcript},
};
use zinc_utils::{from_ref::FromRef, inner_transparent_field::InnerTransparentField};

#[allow(clippy::arithmetic_side_effects)]
pub fn bench_simple_product<F, const LIMBS: usize>(
    group: &mut BenchmarkGroup<WallTime>,
    witness_size: usize,
) where
    F: FromPrimitiveWithConfig + InnerTransparentField + FromRef<F> + 'static,
    F::Inner: FromRef<F::Inner> + ConstTranscribable + ConstIntSemiring,
    MillerRabin: PrimalityTest<F::Inner>,
    for<'a> &'a F: Mul<&'a F, Output = F>,
{
    let mut rng = rng();
    let a: Vec<u32> = (0..witness_size).map(|_| rng.random()).collect();
    let b: Vec<u32> = (0..witness_size).map(|_| rng.random()).collect();
    let c: Vec<u32> = (0..witness_size).map(|_| rng.random()).collect();

    let nvars = zinc_utils::log2(witness_size) as usize;

    let params = format!("LIMBS={}/nvars={}", LIMBS, nvars);

    let a: DenseMultilinearExtension<BinaryPoly<32>> =
        DenseMultilinearExtension::from_evaluations_vec(
            nvars,
            a.into_iter().map(BinaryPoly::from).collect(),
            BinaryPoly::zero(),
        );

    let b: DenseMultilinearExtension<BinaryPoly<32>> =
        DenseMultilinearExtension::from_evaluations_vec(
            nvars,
            b.into_iter().map(BinaryPoly::from).collect(),
            BinaryPoly::zero(),
        );

    let c: DenseMultilinearExtension<BinaryPoly<32>> =
        DenseMultilinearExtension::from_evaluations_vec(
            nvars,
            c.into_iter().map(BinaryPoly::from).collect(),
            BinaryPoly::zero(),
        );

    let transcript = KeccakTranscript::new();

    let prove = |(a, b, c, mut transcript):(_,_,_,KeccakTranscript)| -> RFSumcheckProof<F, BinaryPoly<32>> {
        let field_cfg = transcript.get_random_field_cfg::<F, <F as Field>::Inner, MillerRabin>();

        let eq_r = build_eq_x_r_inner(&vec![F::from_with_cfg(2u32, &field_cfg); nvars], &field_cfg)
            .expect("Failed to build eq_r");

        (RFSumcheck::<F, _>::prove_as_subprotocol(
            &mut transcript,
            vec![a, b, c],
            vec![eq_r],
            nvars,
            3,
            |_x, vals| (&vals[0] * &vals[1] - &vals[2]) * &vals[3],
            field_cfg,
        ))
        .0
    };

    group.bench_with_input(
        BenchmarkId::new("Simple Product Sumcheck Prover", &params),
        &(a.clone(), b.clone(), c.clone(), transcript.clone()),
        |bench, (a, b, c, transcript)| {
            bench.iter_batched(
                || (a.clone(), b.clone(), c.clone(), transcript.clone()),
                |(a, b, c, transcript)| {
                    let _ = black_box(&prove((a, b, c, transcript)));
                },
                BatchSize::SmallInput,
            );
        },
    );

    let proof = prove((a, b, c, transcript.clone()));

    group.bench_with_input(
        BenchmarkId::new("Simple Product Sumcheck Verifier", &params),
        &(proof, transcript),
        |bench, (proof, transcript)| {
            bench.iter_batched(
                || (proof.clone(), transcript.clone()),
                |(proof, mut transcript)| {
                    let field_cfg =
                        transcript.get_random_field_cfg::<F, <F as Field>::Inner, MillerRabin>();

                    let _ = black_box(
                        RFSumcheck::<F, _>::verify_as_subprotocol(
                            &mut transcript,
                            nvars,
                            3,
                            &proof,
                            field_cfg,
                        )
                        .expect("Failed to verify"),
                    );
                },
                BatchSize::SmallInput,
            );
        },
    );
}

#[allow(clippy::arithmetic_side_effects)]
pub fn bench_sum_of_40<F, const LIMBS: usize>(
    group: &mut BenchmarkGroup<WallTime>,
    witness_size: usize,
) where
    F: FromPrimitiveWithConfig + InnerTransparentField + FromRef<F> + 'static,
    F::Inner: FromRef<F::Inner> + ConstTranscribable + ConstIntSemiring,
    MillerRabin: PrimalityTest<F::Inner>,
    for<'a> &'a F: Mul<&'a F, Output = F> + Add<&'a F, Output = F>,
{
    const N_POLYS: usize = 40;

    let mut rng = rng();
    let nvars = zinc_utils::log2(witness_size) as usize;
    let params = format!("LIMBS={}/nvars={}/npolys={}", LIMBS, nvars, N_POLYS);

    // Pre-generate the 40 random MLEs (not benchmarked).
    let polys: Vec<DenseMultilinearExtension<BinaryPoly<32>>> = (0..N_POLYS)
        .map(|_| {
            let evals: Vec<u32> = (0..witness_size).map(|_| rng.random()).collect();
            DenseMultilinearExtension::from_evaluations_vec(
                nvars,
                evals.into_iter().map(BinaryPoly::from).collect(),
                BinaryPoly::zero(),
            )
        })
        .collect();

    let transcript = KeccakTranscript::new();

    // Prover closure: eq(x,r) * sum_i a_i(x)
    let prove =
        |(polys, mut transcript): (Vec<DenseMultilinearExtension<BinaryPoly<32>>>, KeccakTranscript)| -> RFSumcheckProof<F, BinaryPoly<32>> {
            let field_cfg =
                transcript.get_random_field_cfg::<F, <F as Field>::Inner, MillerRabin>();

            let eq_r =
                build_eq_x_r_inner(&vec![F::from_with_cfg(2u32, &field_cfg); nvars], &field_cfg)
                    .expect("Failed to build eq_r");

            // polys go into mles (projected), eq_r goes into mles_f.
            // comb_fn receives vals[0..N_POLYS] = a_i's, vals[N_POLYS] = eq.
            // G(alpha, x) = eq(x,r) * sum_{i=0}^{N_POLYS-1} a_i(x)
            (RFSumcheck::<F, _>::prove_as_subprotocol(
                &mut transcript,
                polys,
                vec![eq_r],
                nvars,
                1,
                |_alpha, vals| {
                    let mut sum = vals[0].clone();
                    for v in &vals[1..N_POLYS] {
                        sum = &sum + v;
                    }
                    sum * &vals[N_POLYS]
                },
                field_cfg,
            ))
            .0
        };

    group.bench_with_input(
        BenchmarkId::new("Sum-of-40 Sumcheck Prover", &params),
        &(polys.clone(), transcript.clone()),
        |bench, (polys, transcript)| {
            bench.iter_batched(
                || (polys.clone(), transcript.clone()),
                |(polys, transcript)| {
                    let _ = black_box(&prove((polys, transcript)));
                },
                BatchSize::SmallInput,
            );
        },
    );

    let proof = prove((polys, transcript.clone()));

    group.bench_with_input(
        BenchmarkId::new("Sum-of-40 Sumcheck Verifier", &params),
        &(proof, transcript),
        |bench, (proof, transcript)| {
            bench.iter_batched(
                || (proof.clone(), transcript.clone()),
                |(proof, mut transcript)| {
                    let field_cfg =
                        transcript.get_random_field_cfg::<F, <F as Field>::Inner, MillerRabin>();

                    let _ = black_box(
                        RFSumcheck::<F, _>::verify_as_subprotocol(
                            &mut transcript,
                            nvars,
                            1,
                            &proof,
                            field_cfg,
                        )
                        .expect("Failed to verify"),
                    );
                },
                BatchSize::SmallInput,
            );
        },
    );
}

pub fn sumcheck_benches(c: &mut Criterion) {
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);

    let mut group = c.benchmark_group("Sumcheck benchmarks");
    group.plot_config(plot_config);

    // bench_simple_product::<MontyField<3>, 3>(&mut group, 1 << 13);
    // bench_simple_product::<MontyField<4>, 4>(&mut group, 1 << 13);
    // bench_simple_product::<MontyField<3>, 3>(&mut group, 1 << 14);
    // bench_simple_product::<MontyField<4>, 4>(&mut group, 1 << 14);
    // bench_simple_product::<MontyField<3>, 3>(&mut group, 1 << 15);
    // bench_simple_product::<MontyField<4>, 4>(&mut group, 1 << 15);
    // bench_simple_product::<MontyField<3>, 3>(&mut group, 1 << 16);
    // bench_simple_product::<MontyField<4>, 4>(&mut group, 1 << 16);
    // bench_simple_product::<MontyField<3>, 3>(&mut group, 1 << 17);
    // bench_simple_product::<MontyField<4>, 4>(&mut group, 1 << 17);

    bench_sum_of_40::<MontyField<3>, 3>(&mut group, 1 << 6);
    bench_sum_of_40::<MontyField<4>, 4>(&mut group, 1 << 6);
    bench_sum_of_40::<MontyField<3>, 3>(&mut group, 1 << 7);
    bench_sum_of_40::<MontyField<4>, 4>(&mut group, 1 << 7);
    bench_sum_of_40::<MontyField<3>, 3>(&mut group, 1 << 8);
    bench_sum_of_40::<MontyField<4>, 4>(&mut group, 1 << 8);
    bench_sum_of_40::<MontyField<3>, 3>(&mut group, 1 << 9);
    bench_sum_of_40::<MontyField<4>, 4>(&mut group, 1 << 9);
    bench_sum_of_40::<MontyField<3>, 3>(&mut group, 1 << 10);
    bench_sum_of_40::<MontyField<4>, 4>(&mut group, 1 << 10);

    group.finish();
}

criterion_group!(benches, sumcheck_benches);
criterion_main!(benches);
