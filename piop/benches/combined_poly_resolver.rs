use std::hint::black_box;

use criterion::{
    AxisScale, BatchSize, BenchmarkGroup, BenchmarkId, Criterion, PlotConfiguration,
    criterion_group, criterion_main, measurement::WallTime,
};
use crypto_primitives::{
    ConstIntSemiring, Field, FromWithConfig, PrimeField, crypto_bigint_int::Int,
    crypto_bigint_monty::MontyField,
};
use rand::rng;
use zinc_piop::{
    combined_poly_resolver::CombinedPolyResolver,
    ideal_check::IdealCheckProtocol,
    projections::{
        evaluate_trace_to_column_mles, project_scalars, project_scalars_to_field,
        project_trace_coeffs_row_major,
    },
};
use zinc_poly::univariate::dense::DensePolynomial;
use zinc_primality::{MillerRabin, PrimalityTest};
use zinc_test_uair::{
    GenerateSingleTypeWitness, TestAirNoMultiplication, TestUairSimpleMultiplication,
};
use zinc_transcript::{
    KeccakTranscript,
    traits::{ConstTranscribable, Transcript},
};
use zinc_uair::{
    Uair, constraint_counter::count_constraints, degree_counter::count_max_degree,
    ideal::degree_one::DegreeOneIdeal, ideal_collector::IdealOrZero,
};

const DEGREE_PLUS_ONE: usize = 32;

type WitnessCoeff<const INT_LIMBS: usize> = Int<INT_LIMBS>;
type Witness<const INT_LIMBS: usize> = DensePolynomial<WitnessCoeff<INT_LIMBS>, DEGREE_PLUS_ONE>;
type F<const FIELD_LIMBS: usize> = MontyField<FIELD_LIMBS>;

#[allow(clippy::arithmetic_side_effects)]
fn bench_no_mult<const INT_LIMBS: usize, const FIELD_LIMBS: usize>(
    group: &mut BenchmarkGroup<WallTime>,
    witness_size: usize,
) where
    <F<FIELD_LIMBS> as Field>::Inner: ConstIntSemiring + ConstTranscribable,
    TestAirNoMultiplication<INT_LIMBS>: Uair<Scalar = Witness<INT_LIMBS>, Ideal = DegreeOneIdeal<WitnessCoeff<INT_LIMBS>>>
        + GenerateSingleTypeWitness<Witness = Witness<INT_LIMBS>>,
    MillerRabin: PrimalityTest<<F<FIELD_LIMBS> as Field>::Inner>,
{
    let mut rng = rng();
    let num_vars = zinc_utils::log2(witness_size) as usize;
    let trace = TestAirNoMultiplication::generate_witness(num_vars, &mut rng);

    let params = format!("NoMult/LIMBS={}/nvars={}", FIELD_LIMBS, num_vars);

    let num_constraints = count_constraints::<TestAirNoMultiplication<INT_LIMBS>>();
    let max_degree = count_max_degree::<TestAirNoMultiplication<INT_LIMBS>>();

    let prove_cpr = |field_cfg: &<F<FIELD_LIMBS> as PrimeField>::Config,
                     trace: &[_],
                     transcript: &mut KeccakTranscript| {
        let projected_trace =
            project_trace_coeffs_row_major::<_, _, Int<5>, _>(&[], trace, &[], field_cfg);

        let projected_scalars =
            project_scalars::<F<FIELD_LIMBS>, TestAirNoMultiplication<INT_LIMBS>>(|scalar| {
                scalar
                    .iter()
                    .map(|coeff| F::from_with_cfg(coeff, field_cfg))
                    .collect()
            });

        let (ic_proof, ic_prover_state) =
            <TestAirNoMultiplication<INT_LIMBS> as IdealCheckProtocol>::prove_combined(
                transcript,
                &projected_trace,
                &projected_scalars,
                num_constraints,
                num_vars,
                field_cfg,
            )
            .expect("IC Prover failed");

        let projecting_element: F<FIELD_LIMBS> = transcript.get_field_challenge(field_cfg);

        let trace_f = evaluate_trace_to_column_mles(&projected_trace, &projecting_element);
        let scalars_f = project_scalars_to_field(projected_scalars, &projecting_element)
            .expect("failed to project scalars to field");

        let (cpr_proof, cpr_state) =
            CombinedPolyResolver::prove_as_subprotocol::<TestAirNoMultiplication<INT_LIMBS>>(
                transcript,
                trace_f,
                &ic_prover_state.evaluation_point,
                &scalars_f,
                num_constraints,
                num_vars,
                max_degree,
                field_cfg,
            )
            .expect("CPR Prover failed");

        (
            ic_proof,
            cpr_proof,
            cpr_state,
            scalars_f,
            projecting_element,
        )
    };

    group.bench_with_input(
        BenchmarkId::new("CPR Prover", &params),
        &trace,
        |bench, trace| {
            let mut transcript = KeccakTranscript::new();
            let field_cfg = transcript.get_random_field_cfg::<F<FIELD_LIMBS>, _, MillerRabin>();
            bench.iter_batched(
                || (trace, transcript.clone()),
                |(trace, mut transcript)| {
                    let _ = black_box(prove_cpr(&field_cfg, trace, &mut transcript));
                },
                BatchSize::SmallInput,
            );
        },
    );

    group.bench_with_input(
        BenchmarkId::new("CPR Verifier", &params),
        &trace,
        |bench, trace| {
            let mut prover_transcript = KeccakTranscript::new();
            let mut verifier_transcript = prover_transcript.clone();
            let field_cfg =
                prover_transcript.get_random_field_cfg::<F<FIELD_LIMBS>, _, MillerRabin>();
            let _ = verifier_transcript.get_random_field_cfg::<F<FIELD_LIMBS>, _, MillerRabin>();

            let (ic_proof, cpr_proof, _, scalars_f, _) =
                prove_cpr(&field_cfg, trace, &mut prover_transcript);

            let ic_check_subclaim =
                <TestAirNoMultiplication<INT_LIMBS> as IdealCheckProtocol>::verify_as_subprotocol::<
                    F<FIELD_LIMBS>,
                    _,
                    _,
                >(
                    &mut verifier_transcript,
                    ic_proof,
                    num_constraints,
                    num_vars,
                    |ideal_over_ring| {
                        ideal_over_ring.map(|i| DegreeOneIdeal::from_with_cfg(i, &field_cfg))
                    },
                    &field_cfg,
                )
                .expect("IC Verifier failed");

            let verifier_projecting_element: F<FIELD_LIMBS> =
                verifier_transcript.get_field_challenge(&field_cfg);

            bench.iter_batched(
                || {
                    (
                        cpr_proof.clone(),
                        ic_check_subclaim.clone(),
                        verifier_transcript.clone(),
                    )
                },
                |(proof, subclaim, mut transcript)| {
                    let _ = black_box(CombinedPolyResolver::verify_as_subprotocol::<
                        TestAirNoMultiplication<INT_LIMBS>,
                    >(
                        &mut transcript,
                        proof,
                        num_constraints,
                        num_vars,
                        max_degree,
                        &verifier_projecting_element,
                        &scalars_f,
                        subclaim,
                        &field_cfg,
                    ))
                    .expect("CPR Verifier failed");
                },
                BatchSize::SmallInput,
            );
        },
    );
}

#[allow(clippy::arithmetic_side_effects)]
fn bench_simple_mult<const INT_LIMBS: usize, const FIELD_LIMBS: usize>(
    group: &mut BenchmarkGroup<WallTime>,
    witness_size: usize,
) where
    <F<FIELD_LIMBS> as Field>::Inner: ConstIntSemiring + ConstTranscribable,
    TestUairSimpleMultiplication<Int<INT_LIMBS>>:
        Uair<Scalar = Witness<INT_LIMBS>> + GenerateSingleTypeWitness<Witness = Witness<INT_LIMBS>>,
    MillerRabin: PrimalityTest<<F<FIELD_LIMBS> as Field>::Inner>,
{
    let mut rng = rng();
    let num_vars = zinc_utils::log2(witness_size) as usize;
    let trace = TestUairSimpleMultiplication::generate_witness(num_vars, &mut rng);

    let params = format!("SimpleMult/LIMBS={}/nvars={}", FIELD_LIMBS, num_vars);

    let num_constraints = count_constraints::<TestUairSimpleMultiplication<Int<INT_LIMBS>>>();
    let max_degree = count_max_degree::<TestUairSimpleMultiplication<Int<INT_LIMBS>>>();

    let prove_cpr = |field_cfg: &<F<FIELD_LIMBS> as PrimeField>::Config,
                     trace: &[_],
                     transcript: &mut KeccakTranscript| {
        let projected_trace =
            project_trace_coeffs_row_major::<_, _, Int<5>, _>(&[], trace, &[], field_cfg);

        let projected_scalars = project_scalars::<
            F<FIELD_LIMBS>,
            TestUairSimpleMultiplication<Int<INT_LIMBS>>,
        >(|scalar| {
            scalar
                .iter()
                .map(|coeff| F::from_with_cfg(coeff, field_cfg))
                .collect()
        });

        let (ic_proof, ic_prover_state) =
            <TestUairSimpleMultiplication<Int<INT_LIMBS>> as IdealCheckProtocol>::prove_combined(
                transcript,
                &projected_trace,
                &projected_scalars,
                num_constraints,
                num_vars,
                field_cfg,
            )
            .expect("IC Prover failed");

        let projecting_element: F<FIELD_LIMBS> = transcript.get_field_challenge(field_cfg);

        let trace_f = evaluate_trace_to_column_mles(&projected_trace, &projecting_element);
        let scalars_f = project_scalars_to_field(projected_scalars, &projecting_element)
            .expect("failed to project scalars to field");

        let (cpr_proof, cpr_state) = CombinedPolyResolver::prove_as_subprotocol::<
            TestUairSimpleMultiplication<Int<INT_LIMBS>>,
        >(
            transcript,
            trace_f,
            &ic_prover_state.evaluation_point,
            &scalars_f,
            num_constraints,
            num_vars,
            max_degree,
            field_cfg,
        )
        .expect("CPR Prover failed");

        (
            ic_proof,
            cpr_proof,
            cpr_state,
            scalars_f,
            projecting_element,
        )
    };

    group.bench_with_input(
        BenchmarkId::new("CPR Prover", &params),
        &trace,
        |bench, trace| {
            let mut transcript = KeccakTranscript::new();
            let field_cfg = transcript.get_random_field_cfg::<F<FIELD_LIMBS>, _, MillerRabin>();
            bench.iter_batched(
                || (trace, transcript.clone()),
                |(trace, mut transcript)| {
                    let _ = black_box(prove_cpr(&field_cfg, trace, &mut transcript));
                },
                BatchSize::SmallInput,
            );
        },
    );

    group.bench_with_input(
        BenchmarkId::new("CPR Verifier", &params),
        &trace,
        |bench, trace| {
            let mut prover_transcript = KeccakTranscript::new();
            let mut verifier_transcript = prover_transcript.clone();
            let field_cfg =
                prover_transcript.get_random_field_cfg::<F<FIELD_LIMBS>, _, MillerRabin>();
            let _ = verifier_transcript.get_random_field_cfg::<F<FIELD_LIMBS>, _, MillerRabin>();

            let (ic_proof, cpr_proof, _, scalars_f, _) =
                prove_cpr(&field_cfg, trace, &mut prover_transcript);

            let ic_check_subclaim =
                <TestUairSimpleMultiplication<Int<INT_LIMBS>> as IdealCheckProtocol>::verify_as_subprotocol::<
                    F<FIELD_LIMBS>,
                    _,
                    _,
                >(
                    &mut verifier_transcript,
                    ic_proof,
                    num_constraints,
                    num_vars,
                    |_ideal_over_ring| IdealOrZero::zero(),
                    &field_cfg,
                )
                .expect("IC Verifier failed");

            let verifier_projecting_element: F<FIELD_LIMBS> =
                verifier_transcript.get_field_challenge(&field_cfg);

            bench.iter_batched(
                || {
                    (
                        cpr_proof.clone(),
                        ic_check_subclaim.clone(),
                        verifier_transcript.clone(),
                    )
                },
                |(proof, subclaim, mut transcript)| {
                    let _ = black_box(CombinedPolyResolver::verify_as_subprotocol::<
                        TestUairSimpleMultiplication<Int<INT_LIMBS>>,
                    >(
                        &mut transcript,
                        proof,
                        num_constraints,
                        num_vars,
                        max_degree,
                        &verifier_projecting_element,
                        &scalars_f,
                        subclaim,
                        &field_cfg,
                    ))
                    .expect("CPR Verifier failed");
                },
                BatchSize::SmallInput,
            );
        },
    );
}

pub fn combined_poly_resolver_benches(c: &mut Criterion) {
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);

    let mut group = c.benchmark_group("Combined poly resolver benchmarks");
    group.plot_config(plot_config);

    bench_no_mult::<3, 4>(&mut group, 1 << 14);
    bench_no_mult::<4, 5>(&mut group, 1 << 14);
    bench_no_mult::<3, 4>(&mut group, 1 << 15);
    bench_no_mult::<4, 5>(&mut group, 1 << 15);
    bench_no_mult::<3, 4>(&mut group, 1 << 16);
    bench_no_mult::<4, 5>(&mut group, 1 << 16);
    bench_no_mult::<3, 4>(&mut group, 1 << 17);
    bench_no_mult::<4, 5>(&mut group, 1 << 17);

    bench_simple_mult::<3, 4>(&mut group, 1 << 2);
    bench_simple_mult::<4, 5>(&mut group, 1 << 2);

    group.finish();
}

criterion_group!(benches, combined_poly_resolver_benches);
criterion_main!(benches);
