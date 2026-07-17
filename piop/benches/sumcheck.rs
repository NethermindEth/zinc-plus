#![allow(non_local_definitions)]
#![allow(clippy::eq_op)]

use std::hint::black_box;

use criterion::{
    AxisScale, BatchSize, BenchmarkGroup, BenchmarkId, Criterion, PlotConfiguration,
    criterion_group, criterion_main, measurement::WallTime,
};
use crypto_primitives::{
    BaseFieldConfig, ConstIntSemiring, ProjectPrimitiveIntegersWithConfig,
    crypto_bigint_monty::MontyField,
};
use num_traits::Zero;
use rand::prelude::*;
use zinc_piop::random_field_sumcheck::{RFSumcheck, RFSumcheckProof};
use zinc_poly::{
    mle::DenseMultilinearExtension, univariate::binary::BinaryPoly, utils::build_eq_x_r,
};
use zinc_primality::{MillerRabin, PrimalityTest};
use zinc_transcript::{
    Blake3Transcript,
    traits::{ConstTranscribable, Transcript},
};
use zinc_utils::from_ref::FromRef;

#[allow(clippy::arithmetic_side_effects)]
pub fn bench_simple_product<C, const LIMBS: usize>(
    group: &mut BenchmarkGroup<WallTime>,
    witness_size: usize,
) where
    C: BaseFieldConfig + ProjectPrimitiveIntegersWithConfig + 'static,
    C::Element: ConstTranscribable,
    C::Integer: ConstTranscribable + ConstIntSemiring + FromRef<C::Integer>,
    MillerRabin: PrimalityTest<C::Integer>,
{
    let mut rng = rand::rng();
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

    let transcript = Blake3Transcript::new();

    let prove = |(a, b, c, mut transcript): (
        _,
        _,
        _,
        Blake3Transcript,
    )|
     -> RFSumcheckProof<C::Element, BinaryPoly<32>> {
        let field_cfg = transcript.get_random_field_cfg::<C, C::Integer, MillerRabin>();

        let eq_r = build_eq_x_r(&field_cfg, &vec![field_cfg.project(&2u32); nvars])
            .expect("Failed to build eq_r");

        (RFSumcheck::<C, _>::prove_as_subprotocol(
            &mut transcript,
            vec![a, b, c],
            vec![eq_r],
            nvars,
            3,
            |_x, vals| {
                field_cfg.mul(
                    &field_cfg.sub(&field_cfg.mul(&vals[0], &vals[1]), &vals[2]),
                    &vals[3],
                )
            },
            &field_cfg,
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
                    let field_cfg = transcript.get_random_field_cfg::<C, C::Integer, MillerRabin>();

                    let _ = black_box(
                        RFSumcheck::<C, _>::verify_as_subprotocol(
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

pub fn sumcheck_benches(c: &mut Criterion) {
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);

    let mut group = c.benchmark_group("Sumcheck benchmarks");
    group.plot_config(plot_config);

    bench_simple_product::<MontyField<3>, 3>(&mut group, 1 << 13);
    bench_simple_product::<MontyField<4>, 4>(&mut group, 1 << 13);
    bench_simple_product::<MontyField<3>, 3>(&mut group, 1 << 14);
    bench_simple_product::<MontyField<4>, 4>(&mut group, 1 << 14);
    bench_simple_product::<MontyField<3>, 3>(&mut group, 1 << 15);
    bench_simple_product::<MontyField<4>, 4>(&mut group, 1 << 15);
    bench_simple_product::<MontyField<3>, 3>(&mut group, 1 << 16);
    bench_simple_product::<MontyField<4>, 4>(&mut group, 1 << 16);
    bench_simple_product::<MontyField<3>, 3>(&mut group, 1 << 17);
    bench_simple_product::<MontyField<4>, 4>(&mut group, 1 << 17);
    group.finish();
}

criterion_group!(benches, sumcheck_benches);
criterion_main!(benches);
