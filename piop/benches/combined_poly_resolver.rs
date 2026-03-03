use criterion::{
    AxisScale, BatchSize, BenchmarkGroup, BenchmarkId, Criterion, PlotConfiguration,
    criterion_group, criterion_main, measurement::WallTime,
};
use crypto_primitives::{
    ConstIntSemiring, Field, FromWithConfig, PrimeField, crypto_bigint_int::Int,
    crypto_bigint_monty::MontyField,
};
use zinc_utils::from_ref::FromRef;
use rand::rng;
use std::hint::black_box;
use zinc_piop::{
    combined_poly_resolver::CombinedPolyResolver,
    ideal_check::IdealCheckProtocol,
    projections::{project_scalars, project_scalars_to_field, project_trace_coeffs, project_trace_to_field},
};
use zinc_poly::univariate::dense::DensePolynomial;
use zinc_primality::{MillerRabin, PrimalityTest};
use zinc_test_uair::{GenerateSingleTypeWitness, TestAirNoMultiplication};
use zinc_transcript::{
    KeccakTranscript,
    traits::{ConstTranscribable, Transcript},
};
use zinc_uair::{
    Uair,
    constraint_counter::count_constraints,
    degree_counter::count_max_degree,
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
    <F<FIELD_LIMBS> as Field>::Inner: ConstIntSemiring + ConstTranscribable + FromRef<<F<FIELD_LIMBS> as Field>::Inner>,
    TestAirNoMultiplication<INT_LIMBS>: Uair<Scalar = Witness<INT_LIMBS>, Ideal = zinc_poly::univariate::ideal::DegreeOneIdeal<WitnessCoeff<INT_LIMBS>>>
        + GenerateSingleTypeWitness<Witness = Witness<INT_LIMBS>>,
    MillerRabin: PrimalityTest<<F<FIELD_LIMBS> as Field>::Inner>,
    F<FIELD_LIMBS>: FromWithConfig<Int<INT_LIMBS>> + PrimeField + FromRef<F<FIELD_LIMBS>>,
{
    let mut rng = rng();
    let num_vars = zinc_utils::log2(witness_size) as usize;
    let trace = TestAirNoMultiplication::generate_witness(num_vars, &mut rng);

    let params = format!("NoMult/LIMBS={}/nvars={}", FIELD_LIMBS, num_vars);

    let num_constraints = count_constraints::<TestAirNoMultiplication<INT_LIMBS>>();
    let max_degree = count_max_degree::<TestAirNoMultiplication<INT_LIMBS>>();

    let mut prover_transcript = KeccakTranscript::new();

    let prover_field_cfg = prover_transcript.get_random_field_cfg::<F<FIELD_LIMBS>, _, MillerRabin>();

    let projected_trace = project_trace_coeffs::<F<FIELD_LIMBS>, _, Int<INT_LIMBS>, DEGREE_PLUS_ONE>(&[], &trace, &[], &prover_field_cfg);

    let projected_scalars =
        project_scalars::<F<FIELD_LIMBS>, TestAirNoMultiplication<INT_LIMBS>>(|scalar| {
            scalar
                .iter()
                .map(|coeff| F::from_with_cfg(*coeff, &prover_field_cfg))
                .collect()
        });

    let (_ic_proof, ic_prover_state) =
        IdealCheckProtocol::prove_as_subprotocol::<TestAirNoMultiplication<INT_LIMBS>>(
            &mut prover_transcript,
            &projected_trace,
            &projected_scalars,
            num_constraints,
            num_vars,
            &prover_field_cfg,
        )
        .expect("Ideal Check Prover failed");

    let projecting_element: F<FIELD_LIMBS> =
        prover_transcript.get_field_challenge(&prover_field_cfg);

    let projected_scalars_f =
        project_scalars_to_field(projected_scalars, &projecting_element).unwrap();

    let trace_f = project_trace_to_field::<_, DEGREE_PLUS_ONE>(&[], &projected_trace, &[], &projecting_element);

    group.bench_with_input(
        BenchmarkId::new("Main field sumcheck Prover", &params),
        &prover_transcript.clone(),
        |bench, transcript| {
            bench.iter_batched(
                || transcript.clone(),
                |mut transcript| {
                    let _ = black_box(CombinedPolyResolver::<F<FIELD_LIMBS>>::prove_as_subprotocol::<TestAirNoMultiplication<INT_LIMBS>>(
                        &mut transcript,
                        trace_f.clone(),
                        &ic_prover_state.evaluation_point,
                        &projected_scalars_f,
                        num_constraints,
                        num_vars,
                        max_degree,
                        &prover_field_cfg,
                    ))
                    .expect("Main field sumcheck Prover failed");
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

    bench_no_mult::<5, 3>(&mut group, 1 << 13);
    bench_no_mult::<5, 4>(&mut group, 1 << 13);
    bench_no_mult::<5, 3>(&mut group, 1 << 14);
    bench_no_mult::<5, 4>(&mut group, 1 << 14);
    bench_no_mult::<5, 3>(&mut group, 1 << 15);
    bench_no_mult::<5, 4>(&mut group, 1 << 15);
    bench_no_mult::<5, 3>(&mut group, 1 << 16);
    bench_no_mult::<5, 4>(&mut group, 1 << 16);
    bench_no_mult::<5, 3>(&mut group, 1 << 17);
    bench_no_mult::<5, 4>(&mut group, 1 << 17);

    group.finish();
}

criterion_group!(benches, combined_poly_resolver_benches);
criterion_main!(benches);
