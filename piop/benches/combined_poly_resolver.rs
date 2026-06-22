use std::hint::black_box;

use criterion::{
    AxisScale, BatchSize, BenchmarkGroup, BenchmarkId, Criterion, PlotConfiguration,
    criterion_group, criterion_main, measurement::WallTime,
};
use crypto_primitives::{
    Field, FromWithConfig, HasPrimeFieldConfig, crypto_bigint_int::Int,
    crypto_bigint_monty::MontyField,
};
use rand::rng;
use zinc_piop::{
    combined_poly_resolver::CombinedPolyResolver,
    ideal_check::IdealCheckProtocol,
    projections::{
        ProjectedTrace, evaluate_trace_to_column_mles, project_scalars, project_scalars_to_field,
        project_trace_coeffs_row_major,
    },
    sumcheck::multi_degree::MultiDegreeSumcheck,
};
use zinc_poly::univariate::dense::DensePolynomial;
use zinc_primality::{MillerRabin, PrimalityTest};
use zinc_test_uair::{GenerateRandomTrace, TestUairNoMultiplication, TestUairSimpleMultiplication};
use zinc_transcript::{Blake3Transcript, traits::Transcript};
use zinc_uair::{
    Uair, UairTrace, constraint_counter::count_constraints, degree_counter::count_max_degree,
    ideal::DegreeOneIdeal, ideal_collector::IdealOrZero,
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
    TestUairNoMultiplication<Int<INT_LIMBS>, <F<FIELD_LIMBS> as Field>::Integer>:
        Uair<Scalar = Witness<INT_LIMBS>, Ideal = DegreeOneIdeal<WitnessCoeff<INT_LIMBS>>>
            + GenerateRandomTrace<DEGREE_PLUS_ONE, PolyCoeff = Int<INT_LIMBS>, Int = Int<INT_LIMBS>>,
    MillerRabin: PrimalityTest<<F<FIELD_LIMBS> as Field>::Integer>,
{
    let mut rng = rng();
    let num_vars = zinc_utils::log2(witness_size) as usize;
    let trace = TestUairNoMultiplication::generate_random_trace(num_vars, &mut rng);

    let params = format!("NoMult/LIMBS={}/nvars={}", FIELD_LIMBS, num_vars);

    let num_constraints = count_constraints::<TestUairNoMultiplication<_, _>>();
    let max_degree = count_max_degree::<TestUairNoMultiplication<_, _>>();

    let prove_cpr = |field_cfg: &<F<FIELD_LIMBS> as HasPrimeFieldConfig>::Config,
                     trace: &UairTrace<_, _, _, _>,
                     transcript: &mut Blake3Transcript| {
        let projected_trace = project_trace_coeffs_row_major(trace, field_cfg);

        let projected_scalars =
            project_scalars::<F<FIELD_LIMBS>, TestUairNoMultiplication<_, _>>(|scalar| {
                scalar
                    .iter()
                    .map(|coeff| F::from_with_cfg(coeff, field_cfg))
                    .collect()
            });

        let evaluation_point: Vec<F<FIELD_LIMBS>> =
            transcript.get_field_challenges(num_vars, field_cfg);
        let ic_proof = IdealCheckProtocol::<TestUairNoMultiplication<_, _>>::prove_combined::<
            _,
            DEGREE_PLUS_ONE,
        >(
            transcript,
            &projected_trace,
            &projected_scalars,
            /* family_idx = */ 0,
            num_constraints.q,
            &evaluation_point,
            field_cfg,
        )
        .expect("IC Prover failed");

        let projecting_element: F<FIELD_LIMBS> = transcript.get_field_challenge(field_cfg);

        let trace_f = evaluate_trace_to_column_mles(
            &ProjectedTrace::RowMajor(projected_trace),
            &projecting_element,
        );
        let scalars_f = project_scalars_to_field(projected_scalars, &projecting_element)
            .expect("failed to project scalars to field");

        let folding_challenge: F<FIELD_LIMBS> = transcript.get_field_challenge(field_cfg);
        let (cpr_group, cpr_ancillary) =
            CombinedPolyResolver::prepare_sumcheck_group::<TestUairNoMultiplication<_, _>>(
                trace_f,
                Vec::new(),
                &evaluation_point,
                &scalars_f,
                /* family_idx = */ 0,
                num_constraints.q,
                num_vars,
                max_degree,
                &folding_challenge,
                field_cfg,
            )
            .expect("CPR prepare failed");

        let mut sumcheck_outputs = MultiDegreeSumcheck::prove_as_subprotocol(
            transcript,
            vec![(vec![cpr_group], field_cfg)],
            num_vars,
            field_cfg,
        );
        let (md_proof, md_states) = sumcheck_outputs.pop().expect("single family");

        let (cpr_proof, cpr_state) =
            CombinedPolyResolver::finalize_prover::<TestUairNoMultiplication<_, _>>(
                transcript,
                md_states.into_iter().next().expect("one CPR group"),
                cpr_ancillary,
                field_cfg,
            )
            .expect("CPR finalize failed");

        (
            ic_proof,
            cpr_proof,
            md_proof,
            cpr_state,
            scalars_f,
            projecting_element,
        )
    };

    group.bench_function(BenchmarkId::new("CPR Prover", &params), |bench| {
        let mut transcript = Blake3Transcript::new();
        let field_cfg = transcript.get_random_field_cfg::<F<FIELD_LIMBS>, _, MillerRabin>();
        bench.iter_batched(
            || transcript.clone(),
            |mut transcript| {
                let _ = black_box(prove_cpr(&field_cfg, &trace, &mut transcript));
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function(BenchmarkId::new("CPR Verifier", &params), |bench| {
        let mut prover_transcript = Blake3Transcript::new();
        let mut verifier_transcript = prover_transcript.clone();
        let field_cfg = prover_transcript.get_random_field_cfg::<F<FIELD_LIMBS>, _, MillerRabin>();
        let _ = verifier_transcript.get_random_field_cfg::<F<FIELD_LIMBS>, _, MillerRabin>();

        let (ic_proof, cpr_proof, md_proof, _, scalars_f, _) =
            prove_cpr(&field_cfg, &trace, &mut prover_transcript);

        let ic_evaluation_point: Vec<F<FIELD_LIMBS>> =
            verifier_transcript.get_field_challenges(num_vars, &field_cfg);
        let ic_check_subclaim =
            IdealCheckProtocol::<TestUairNoMultiplication<_, _>>::verify_as_subprotocol::<
                F<FIELD_LIMBS>,
                _,
                _,
                _,
            >(
                &mut verifier_transcript,
                ic_proof,
                /* family_idx = */ 0,
                num_constraints.q,
                &ic_evaluation_point,
                |ideal_over_ring| {
                    ideal_over_ring.map(|i| DegreeOneIdeal::from_with_cfg(i, &field_cfg))
                },
                |_| unreachable!("not used here"),
            )
            .expect("IC Verifier failed");

        let verifier_projecting_element: F<FIELD_LIMBS> =
            verifier_transcript.get_field_challenge(&field_cfg);
        let verifier_folding_challenge: F<FIELD_LIMBS> =
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
                let ancillary =
                    CombinedPolyResolver::prepare_verifier::<TestUairNoMultiplication<_, _>>(
                        &proof,
                        md_proof.claimed_sums()[0].clone(),
                        &subclaim,
                        num_constraints.q,
                        num_vars,
                        &verifier_projecting_element,
                        &verifier_folding_challenge,
                        &field_cfg,
                    )
                    .expect("CPR prepare_verifier failed");

                let mut md_subclaims_vec = MultiDegreeSumcheck::verify_as_subprotocol(
                    &mut transcript,
                    num_vars,
                    &[(&md_proof, &field_cfg)],
                    &field_cfg,
                )
                .expect("MultiDegreeSumcheck verify failed");
                let md_subclaims = md_subclaims_vec.pop().expect("single family");

                let _ = black_box(
                    CombinedPolyResolver::finalize_verifier::<TestUairNoMultiplication<_, _>>(
                        &mut transcript,
                        proof,
                        md_subclaims.point().to_vec(),
                        md_subclaims.expected_evaluations()[0].clone(),
                        ancillary,
                        &scalars_f,
                        /* family_idx = */ 0,
                        &field_cfg,
                    )
                    .expect("CPR finalize_verifier failed"),
                );
            },
            BatchSize::SmallInput,
        );
    });
}

#[allow(clippy::arithmetic_side_effects)]
fn bench_simple_mult<const INT_LIMBS: usize, const FIELD_LIMBS: usize>(
    group: &mut BenchmarkGroup<WallTime>,
    witness_size: usize,
) where
    TestUairSimpleMultiplication<Int<INT_LIMBS>, <F<FIELD_LIMBS> as Field>::Integer>:
        Uair<Scalar = Witness<INT_LIMBS>>
            + GenerateRandomTrace<DEGREE_PLUS_ONE, PolyCoeff = Int<INT_LIMBS>, Int = Int<INT_LIMBS>>,
    MillerRabin: PrimalityTest<<F<FIELD_LIMBS> as Field>::Integer>,
{
    let mut rng = rng();
    let num_vars = zinc_utils::log2(witness_size) as usize;
    let trace = TestUairSimpleMultiplication::generate_random_trace(num_vars, &mut rng);

    let params = format!("SimpleMult/LIMBS={}/nvars={}", FIELD_LIMBS, num_vars);

    let num_constraints = count_constraints::<TestUairSimpleMultiplication<_, _>>();
    let max_degree = count_max_degree::<TestUairSimpleMultiplication<_, _>>();

    let prove_cpr = |field_cfg: &<F<FIELD_LIMBS> as HasPrimeFieldConfig>::Config,
                     trace: &UairTrace<_, _, _, _>,
                     transcript: &mut Blake3Transcript| {
        let projected_trace = project_trace_coeffs_row_major(trace, field_cfg);

        let projected_scalars =
            project_scalars::<F<FIELD_LIMBS>, TestUairSimpleMultiplication<_, _>>(|scalar| {
                scalar
                    .iter()
                    .map(|coeff| F::from_with_cfg(coeff, field_cfg))
                    .collect()
            });

        let evaluation_point: Vec<F<FIELD_LIMBS>> =
            transcript.get_field_challenges(num_vars, field_cfg);
        let ic_proof = IdealCheckProtocol::<TestUairSimpleMultiplication<_, _>>::prove_combined::<
            _,
            DEGREE_PLUS_ONE,
        >(
            transcript,
            &projected_trace,
            &projected_scalars,
            /* family_idx = */ 0,
            num_constraints.q,
            &evaluation_point,
            field_cfg,
        )
        .expect("IC Prover failed");

        let projecting_element: F<FIELD_LIMBS> = transcript.get_field_challenge(field_cfg);

        let trace_f = evaluate_trace_to_column_mles(
            &ProjectedTrace::RowMajor(projected_trace),
            &projecting_element,
        );
        let scalars_f = project_scalars_to_field(projected_scalars, &projecting_element)
            .expect("failed to project scalars to field");

        let folding_challenge: F<FIELD_LIMBS> = transcript.get_field_challenge(field_cfg);
        let (cpr_group, cpr_ancillary) =
            CombinedPolyResolver::prepare_sumcheck_group::<TestUairSimpleMultiplication<_, _>>(
                trace_f,
                Vec::new(),
                &evaluation_point,
                &scalars_f,
                /* family_idx = */ 0,
                num_constraints.q,
                num_vars,
                max_degree,
                &folding_challenge,
                field_cfg,
            )
            .expect("CPR prepare failed");

        let (md_proof, md_states) = MultiDegreeSumcheck::prove_as_subprotocol(
            transcript,
            vec![(vec![cpr_group], field_cfg)],
            num_vars,
            field_cfg,
        )
        .pop()
        .expect("single family");

        let (cpr_proof, cpr_state) =
            CombinedPolyResolver::finalize_prover::<TestUairSimpleMultiplication<_, _>>(
                transcript,
                md_states.into_iter().next().expect("one CPR group"),
                cpr_ancillary,
                field_cfg,
            )
            .expect("CPR finalize failed");

        (
            ic_proof,
            cpr_proof,
            md_proof,
            cpr_state,
            scalars_f,
            projecting_element,
        )
    };

    group.bench_function(BenchmarkId::new("CPR Prover", &params), |bench| {
        let mut transcript = Blake3Transcript::new();
        let field_cfg = transcript.get_random_field_cfg::<F<FIELD_LIMBS>, _, MillerRabin>();
        bench.iter_batched(
            || transcript.clone(),
            |mut transcript| {
                let _ = black_box(prove_cpr(&field_cfg, &trace, &mut transcript));
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function(BenchmarkId::new("CPR Verifier", &params), |bench| {
        let mut prover_transcript = Blake3Transcript::new();
        let mut verifier_transcript = prover_transcript.clone();
        let field_cfg = prover_transcript.get_random_field_cfg::<F<FIELD_LIMBS>, _, MillerRabin>();
        let _ = verifier_transcript.get_random_field_cfg::<F<FIELD_LIMBS>, _, MillerRabin>();

        let (ic_proof, cpr_proof, md_proof, _, scalars_f, _) =
            prove_cpr(&field_cfg, &trace, &mut prover_transcript);

        let ic_evaluation_point = verifier_transcript.get_field_challenges(num_vars, &field_cfg);
        let ic_check_subclaim =
            IdealCheckProtocol::<TestUairSimpleMultiplication<_, _>>::verify_as_subprotocol::<
                F<FIELD_LIMBS>,
                _,
                _,
                _,
            >(
                &mut verifier_transcript,
                ic_proof,
                /* family_idx = */ 0,
                num_constraints.q,
                &ic_evaluation_point,
                |_ideal_over_ring| IdealOrZero::<DegreeOneIdeal<_>>::zero(),
                |_| unreachable!("not used here"),
            )
            .expect("IC Verifier failed");

        let verifier_projecting_element: F<FIELD_LIMBS> =
            verifier_transcript.get_field_challenge(&field_cfg);
        let verifier_folding_challenge: F<FIELD_LIMBS> =
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
                let ancillary =
                    CombinedPolyResolver::prepare_verifier::<TestUairSimpleMultiplication<_, _>>(
                        &proof,
                        md_proof.claimed_sums()[0].clone(),
                        &subclaim,
                        num_constraints.q,
                        num_vars,
                        &verifier_projecting_element,
                        &verifier_folding_challenge,
                        &field_cfg,
                    )
                    .expect("CPR prepare_verifier failed");

                let md_subclaims = MultiDegreeSumcheck::verify_as_subprotocol(
                    &mut transcript,
                    num_vars,
                    &[(&md_proof, &field_cfg)],
                    &field_cfg,
                )
                .expect("MultiDegreeSumcheck verify failed")
                .pop()
                .expect("single family");

                let _ = black_box(
                    CombinedPolyResolver::finalize_verifier::<TestUairSimpleMultiplication<_, _>>(
                        &mut transcript,
                        proof,
                        md_subclaims.point().to_vec(),
                        md_subclaims.expected_evaluations()[0].clone(),
                        ancillary,
                        &scalars_f,
                        /* family_idx = */ 0,
                        &field_cfg,
                    )
                    .expect("CPR finalize_verifier failed"),
                );
            },
            BatchSize::SmallInput,
        );
    });
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
