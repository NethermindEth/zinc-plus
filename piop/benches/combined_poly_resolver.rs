use criterion::{
    AxisScale, BatchSize, BenchmarkGroup, BenchmarkId, Criterion, PlotConfiguration,
    criterion_group, criterion_main, measurement::WallTime,
};
use crypto_primitives::{FromPrimitiveWithConfig, crypto_bigint_monty::MontyField};
use rand::rng;
use std::hint::black_box;
use zinc_piop::{
    combined_poly_resolver::CombinedPolyResolver,
    ideal_check::{IdealCheckField, IdealCheckProtocol},
};
use zinc_poly::univariate::{binary::BinaryPoly, dynamic::over_field::DynamicPolynomialF};
use zinc_primality::{MillerRabin, PrimalityTest};
use zinc_test_uair::{GenerateWitness, TestAirBinary, TestUairSimpleMultiplication};
use zinc_transcript::{KeccakTranscript, traits::Transcript};
use zinc_uair::{
    Uair,
    constraint_counter::count_constraints,
    degree_counter::count_max_degree,
    ideal::{Ideal, IdealCheck},
    ideal_collector::IdealOrZero,
};

const DEGREE_PLUS_ONE: usize = 32;

#[allow(clippy::arithmetic_side_effects)]
fn do_bench<F, Air, IdealOverFFromRef, IdealOverF>(
    group: &mut BenchmarkGroup<WallTime>,
    bench_title: &str,
    witness_size: usize,
    ideal_over_f_from_ref: IdealOverFFromRef,
) where
    F: IdealCheckField + FromPrimitiveWithConfig,
    MillerRabin: PrimalityTest<F::Inner>,
    Air: Uair<BinaryPoly<DEGREE_PLUS_ONE>> + GenerateWitness<BinaryPoly<DEGREE_PLUS_ONE>>,
    IdealOverF: Ideal + IdealCheck<DynamicPolynomialF<F>>,
    IdealOverFFromRef:
        Fn(&IdealOrZero<<Air as Uair<BinaryPoly<DEGREE_PLUS_ONE>>>::Ideal>) -> IdealOverF + Copy,
{
    let mut rng = rng();
    let num_vars = zinc_utils::log2(witness_size) as usize;
    let trace = Air::generate_witness(num_vars, &mut rng);

    let params = format!("{}/nvars={}", bench_title, num_vars);

    let mut prover_transcript = KeccakTranscript::new();
    let mut verifier_transcript = prover_transcript.clone();

    let prover_field_cfg = prover_transcript.get_random_field_cfg::<F, _, MillerRabin>();
    let _ = verifier_transcript.get_random_field_cfg::<F, _, MillerRabin>();

    let num_constraints = count_constraints::<BinaryPoly<DEGREE_PLUS_ONE>, Air>();
    let max_degree = count_max_degree::<BinaryPoly<DEGREE_PLUS_ONE>, Air>();

    let (ic_proof, ic_prover_state) =
        IdealCheckProtocol::<F, DEGREE_PLUS_ONE>::prove_as_subprotocol::<Air>(
            &mut prover_transcript,
            &trace,
            num_constraints,
            num_vars,
            &prover_field_cfg,
        )
        .expect("IC Prover failed");

    let ic_check_subclaim =
        IdealCheckProtocol::<F, DEGREE_PLUS_ONE>::verify_as_subprotocol::<Air, _, _>(
            &mut verifier_transcript,
            ic_proof,
            num_constraints,
            num_vars,
            ideal_over_f_from_ref,
            &prover_field_cfg,
        )
        .expect("IC Verifier failed");

    let verifier_transcript_after_ic = verifier_transcript.clone();

    group.bench_with_input(
        BenchmarkId::new("CPR Prover", &params),
        &(
            prover_transcript.clone(),
            ic_prover_state.projected_scalars.clone(),
        ),
        |bench, (transcript, scalars)| {
            bench.iter_batched(
                || (scalars.clone(), transcript.clone()),
                |(scalars, mut transcript)| {
                    let _ = black_box(CombinedPolyResolver::<F>::prove_as_subprotocol::<_, Air>(
                        &mut transcript,
                        &ic_prover_state.trace_matrix,
                        &ic_prover_state.evaluation_point,
                        scalars.clone(),
                        num_constraints,
                        num_vars,
                        max_degree,
                        &prover_field_cfg,
                    ))
                    .expect("CPR Prover failed");
                },
                BatchSize::SmallInput,
            );
        },
    );

    let (cpr_proof, _) = CombinedPolyResolver::<F>::prove_as_subprotocol::<_, Air>(
        &mut prover_transcript,
        &ic_prover_state.trace_matrix,
        &ic_prover_state.evaluation_point,
        ic_prover_state.projected_scalars.clone(),
        num_constraints,
        num_vars,
        max_degree,
        &prover_field_cfg,
    )
    .expect("CPR Prover failed");

    group.bench_with_input(
        BenchmarkId::new("CPR Verifier", &params),
        &(cpr_proof, ic_check_subclaim, verifier_transcript_after_ic),
        |bench, (proof, subclaim, transcript)| {
            bench.iter_batched(
                || (proof.clone(), subclaim.clone(), transcript.clone()),
                |(proof, subclaim, mut transcript)| {
                    let _ = black_box(CombinedPolyResolver::<F>::verify_as_subprotocol::<_, Air>(
                        &mut transcript,
                        proof,
                        num_constraints,
                        num_vars,
                        max_degree,
                        subclaim,
                        &prover_field_cfg,
                    ))
                    .expect("CPR Verifier failed");
                },
                BatchSize::SmallInput,
            );
        },
    );
}

pub fn bench_bin<const FIELD_LIMBS: usize>(
    group: &mut BenchmarkGroup<WallTime>,
    witness_size: usize,
) {
    do_bench::<MontyField<FIELD_LIMBS>, TestAirBinary, _, _>(
        group,
        &format!("Binary/LIMBS={FIELD_LIMBS}"),
        witness_size,
        |_ideal_over_ring| IdealOrZero::Zero,
    )
}

pub fn bench_simple_mul<const FIELD_LIMBS: usize>(
    group: &mut BenchmarkGroup<WallTime>,
    witness_size: usize,
) {
    do_bench::<MontyField<FIELD_LIMBS>, TestUairSimpleMultiplication, _, _>(
        group,
        &format!("SimpleMul/LIMBS={FIELD_LIMBS}"),
        witness_size,
        |_ideal_over_ring| IdealOrZero::Zero,
    )
}

pub fn combined_poly_resolver_benches(c: &mut Criterion) {
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);

    let mut group = c.benchmark_group("Combined poly resolver benchmarks");
    group.plot_config(plot_config);

    bench_bin::<3>(&mut group, 1 << 13);
    bench_bin::<4>(&mut group, 1 << 13);
    bench_bin::<3>(&mut group, 1 << 14);
    bench_bin::<4>(&mut group, 1 << 14);
    bench_bin::<3>(&mut group, 1 << 15);
    bench_bin::<4>(&mut group, 1 << 15);
    bench_bin::<3>(&mut group, 1 << 16);
    bench_bin::<4>(&mut group, 1 << 16);
    bench_bin::<3>(&mut group, 1 << 17);
    bench_bin::<4>(&mut group, 1 << 17);

    bench_simple_mul::<3>(&mut group, 1 << 2);
    bench_simple_mul::<4>(&mut group, 1 << 2);

    group.finish();
}

criterion_group!(benches, combined_poly_resolver_benches);
criterion_main!(benches);
