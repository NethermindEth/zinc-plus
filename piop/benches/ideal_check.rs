use std::hint::black_box;

use criterion::{
    AxisScale, BatchSize, BenchmarkGroup, BenchmarkId, Criterion, PlotConfiguration,
    criterion_group, criterion_main, measurement::WallTime,
};
use crypto_primitives::{
    ConstIntSemiring, Field, Semiring, crypto_bigint_int::Int, crypto_bigint_monty::MontyField,
};
use rand::rng;
use zinc_piop::ideal_check::{IdealCheckProtocol, IdealCheckTypes, Proof};
use zinc_poly::univariate::{
    dense::DensePolynomial, dynamic::over_field::DynamicPolynomialF, ideal::DegreeOneIdeal,
};
use zinc_primality::{MillerRabin, PrimalityTest};
use zinc_test_uair::{GenerateWitness, TestAirNoMultiplication, TestUairSimpleMultiplication};
use zinc_transcript::{
    KeccakTranscript,
    traits::{ConstTranscribable, Transcript},
};
use zinc_uair::{
    Uair, constraint_counter::count_constraints, ideal::IdealCheck, ideal_collector::IdealOrZero,
};
use zinc_utils::from_ref::FromRef;

const DEGREE_PLUS_ONE: usize = 32;

#[derive(Clone)]
struct BenchIcTypes<const FIELD_LIMBS: usize, const INT_LIMBS: usize>;

trait BenchIcTrait<const FIELD_LIMBS: usize, const INT_LIMBS: usize>:
    IdealCheckTypes<
        Int<INT_LIMBS>,
        DEGREE_PLUS_ONE,
        Witness = DensePolynomial<Int<INT_LIMBS>, DEGREE_PLUS_ONE>,
        F = MontyField<FIELD_LIMBS>,
    > + Clone
{
    const FIELD_LIMBS: usize;
}

impl<const FIELD_LIMBS: usize, const INT_LIMBS: usize> IdealCheckTypes<Int<INT_LIMBS>, DEGREE_PLUS_ONE>
    for BenchIcTypes<FIELD_LIMBS, INT_LIMBS>
{
    type Witness = DensePolynomial<Int<INT_LIMBS>, DEGREE_PLUS_ONE>;
    type F = MontyField<FIELD_LIMBS>;
}

impl<const FIELD_LIMBS: usize, const INT_LIMBS: usize> BenchIcTrait<FIELD_LIMBS, INT_LIMBS>
    for BenchIcTypes<FIELD_LIMBS, INT_LIMBS>
{
    const FIELD_LIMBS: usize = FIELD_LIMBS;
}

#[allow(clippy::arithmetic_side_effects)]
fn bench_no_mult<IcTypes, const FIELD_LIMBS: usize, const INT_LIMBS: usize>(group: &mut BenchmarkGroup<WallTime>, witness_size: usize)
where
    IcTypes: BenchIcTrait<FIELD_LIMBS, INT_LIMBS>,
{
    let mut rng = rng();
    let num_vars = zinc_utils::log2(witness_size) as usize;
    let trace = TestAirNoMultiplication::generate_witness(num_vars, &mut rng);

    let params = format!("NoMult/LIMBS={}/nvars={}", IcTypes::FIELD_LIMBS, num_vars);

    let transcript = KeccakTranscript::new();

    let num_constraints = count_constraints::<<IcTypes as IdealCheckTypes<_, _>>::Witness, TestAirNoMultiplication>();

    let prove =
        |(trace, mut transcript): (Vec<_>, KeccakTranscript)| -> Proof<_, IcTypes, DEGREE_PLUS_ONE> {
            let field_cfg = transcript
                .get_random_field_cfg::<<IcTypes as IdealCheckTypes<_, _>>::F, <<IcTypes as IdealCheckTypes<_, _>>::F as Field>::Inner, MillerRabin>();
            IdealCheckProtocol::prove_as_subprotocol::<TestAirNoMultiplication>(
                &mut transcript,
                &trace,
                num_constraints,
                num_vars,
                &field_cfg,
            )
            .expect("Prover failed")
            .0
        };

    group.bench_with_input(
        BenchmarkId::new("Ideal Check Prover", &params),
        &(trace.clone(), transcript.clone()),
        |bench, (trace, transcript)| {
            bench.iter_batched(
                || (trace.clone(), transcript.clone()),
                |(trace, transcript)| {
                    let _ = black_box(&prove((trace, transcript)));
                },
                BatchSize::SmallInput,
            );
        },
    );

    let proof = prove((trace, transcript.clone()));

    group.bench_with_input(
        BenchmarkId::new("Ideal Check Verifier", &params),
        &(proof, transcript),
        |bench, (proof, transcript)| {
            bench.iter_batched(
                || (proof.clone(), transcript.clone()),
                |(proof, mut transcript)| {
                    let field_cfg = transcript.get_random_field_cfg::<
                        <IcTypes as IdealCheckTypes<_, _>>::F,
                        <<IcTypes as IdealCheckTypes<_, _>>::F as Field>::Inner,
                        MillerRabin,
                    >();
                    let _ = black_box(IdealCheckProtocol::verify_as_subprotocol::<
                        TestAirNoMultiplication,
                        _,
                        _,
                    >(
                        &mut transcript,
                        proof,
                        num_constraints,
                        num_vars,
                        |ideal_over_ring| {
                            ideal_over_ring.map(|i| DegreeOneIdeal::from_with_cfg(i, &field_cfg))
                        },
                        &field_cfg,
                    ))
                    .expect("Failed to verify");
                },
                BatchSize::SmallInput,
            );
        },
    );
}

pub fn bench_no_mult_3(group: &mut BenchmarkGroup<WallTime>, witness_size: usize) {
    bench_no_mult::<BenchIcTypes<3, 4>, 3, 4>(group, witness_size)
}

pub fn bench_no_mult_4(group: &mut BenchmarkGroup<WallTime>, witness_size: usize) {
    bench_no_mult::<BenchIcTypes<4, 5>, 4, 5>(group, witness_size)
}

#[allow(clippy::arithmetic_side_effects)]
fn bench_simple_mult<IcTypes, const FIELD_LIMBS: usize, const INT_LIMBS: usize>(group: &mut BenchmarkGroup<WallTime>, witness_size: usize)
where
    IcTypes: BenchIcTrait<FIELD_LIMBS, INT_LIMBS>,
{
    let mut rng = rng();
    let num_vars = zinc_utils::log2(witness_size) as usize;
    let trace = TestUairSimpleMultiplication::generate_witness(num_vars, &mut rng);

    let params = format!(
        "SimpleMult/LIMBS={}/nvars={}",
        IcTypes::FIELD_LIMBS,
        num_vars
    );

    let transcript = KeccakTranscript::new();

    let num_constraints = count_constraints::<<IcTypes as IdealCheckTypes<_, _>>::Witness, TestUairSimpleMultiplication>();

    let prove =
        |(trace, mut transcript): (Vec<_>, KeccakTranscript)| -> Proof<_, IcTypes, DEGREE_PLUS_ONE> {
            let field_cfg = transcript
                .get_random_field_cfg::<<IcTypes as IdealCheckTypes<_, _>>::F, <<IcTypes as IdealCheckTypes<_, _>>::F as Field>::Inner, MillerRabin>();
            IdealCheckProtocol::prove_as_subprotocol::<TestUairSimpleMultiplication>(
                &mut transcript,
                &trace,
                num_constraints,
                num_vars,
                &field_cfg,
            )
            .expect("Prover failed")
            .0
        };

    group.bench_with_input(
        BenchmarkId::new("Ideal Check Prover", &params),
        &(trace.clone(), transcript.clone()),
        |bench, (trace, transcript)| {
            bench.iter_batched(
                || (trace.clone(), transcript.clone()),
                |(trace, transcript)| {
                    let _ = black_box(&prove((trace, transcript)));
                },
                BatchSize::SmallInput,
            );
        },
    );

    let proof = prove((trace, transcript.clone()));

    group.bench_with_input(
        BenchmarkId::new("Ideal Check Verifier", &params),
        &(proof, transcript),
        |bench, (proof, transcript)| {
            bench.iter_batched(
                || (proof.clone(), transcript.clone()),
                |(proof, mut transcript)| {
                    let field_cfg = transcript.get_random_field_cfg::<
                        <IcTypes as IdealCheckTypes<_, _>>::F,
                        <<IcTypes as IdealCheckTypes<_, _>>::F as Field>::Inner,
                        MillerRabin,
                    >();
                    let _ = black_box(IdealCheckProtocol::verify_as_subprotocol::<
                        TestUairSimpleMultiplication,
                        _,
                        _,
                    >(
                        &mut transcript,
                        proof,
                        num_constraints,
                        num_vars,
                        |_ideal_over_ring| IdealOrZero::zero(),
                        &field_cfg,
                    ))
                    .expect("Failed to verify");
                },
                BatchSize::SmallInput,
            );
        },
    );
}

pub fn bench_simple_mult_3(group: &mut BenchmarkGroup<WallTime>, witness_size: usize) {
    bench_simple_mult::<BenchIcTypes<3, 4>, 3, 4>(group, witness_size)
}

pub fn bench_simple_mult_4(group: &mut BenchmarkGroup<WallTime>, witness_size: usize) {
    bench_simple_mult::<BenchIcTypes<4, 5>, 4, 5>(group, witness_size)
}

/// Before/after diff for combined_poly_builder (parallel vs sequential):
///   1. cargo bench -p zinc-piop --bench ideal_check -- "Ideal Check Prover"
///      --save-baseline sequential
///   2. cargo bench -p zinc-piop --bench ideal_check --features parallel --
///      "Ideal Check Prover" --baseline sequential
pub fn ideal_check_benches(c: &mut Criterion) {
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);

    let mut group = c.benchmark_group("Ideal check benchmarks");
    group.plot_config(plot_config);

    bench_no_mult_3(&mut group, 1 << 13);
    bench_no_mult_4(&mut group, 1 << 13);
    bench_no_mult_3(&mut group, 1 << 14);
    bench_no_mult_4(&mut group, 1 << 14);
    bench_no_mult_3(&mut group, 1 << 15);
    bench_no_mult_4(&mut group, 1 << 15);
    bench_no_mult_3(&mut group, 1 << 16);
    bench_no_mult_4(&mut group, 1 << 16);
    bench_no_mult_3(&mut group, 1 << 17);
    bench_no_mult_4(&mut group, 1 << 17);

    bench_simple_mult_3(&mut group, 1 << 2);
    bench_simple_mult_4(&mut group, 1 << 2);

    group.finish();
}

criterion_group!(benches, ideal_check_benches);
criterion_main!(benches);
