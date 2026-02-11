use criterion::{
    AxisScale, BatchSize, BenchmarkGroup, BenchmarkId, Criterion, PlotConfiguration,
    criterion_group, criterion_main, measurement::WallTime,
};
use crypto_primitives::{Field, crypto_bigint_monty::MontyField};
use rand::rng;
use std::hint::black_box;
use zinc_piop::ideal_check::{IdealCheckProtocol, IdealCheckTypes, Proof};
use zinc_poly::univariate::{binary::BinaryPoly, dynamic::over_field::DynamicPolynomialF};
use zinc_primality::MillerRabin;
use zinc_test_uair::{GenerateWitness, TestAirBinary, TestUairSimpleMultiplication};
use zinc_transcript::{
    KeccakTranscript,
    traits::{ConstTranscribable, Transcript},
};
use zinc_uair::{
    Uair,
    constraint_counter::count_constraints,
    ideal::{Ideal, IdealCheck},
    ideal_collector::IdealOrZero,
};

const DEGREE_PLUS_ONE: usize = 32;

#[derive(Clone)]
struct BenchIcTypes<const FIELD_LIMBS: usize>;

trait BenchIcTrait<const FIELD_LIMBS: usize>:
    IdealCheckTypes<F = MontyField<FIELD_LIMBS>> + Clone
{
    const FIELD_LIMBS: usize;
}

impl<const FIELD_LIMBS: usize> IdealCheckTypes for BenchIcTypes<FIELD_LIMBS> {
    type F = MontyField<FIELD_LIMBS>;
}

impl<const FIELD_LIMBS: usize> BenchIcTrait<FIELD_LIMBS> for BenchIcTypes<FIELD_LIMBS> {
    const FIELD_LIMBS: usize = FIELD_LIMBS;
}

#[allow(clippy::arithmetic_side_effects)]
fn do_bench<IcTypes, Air, IdealOverFFromRef, IdealOverF, const FIELD_LIMBS: usize>(
    group: &mut BenchmarkGroup<WallTime>,
    air_name: &str,
    witness_size: usize,
    ideal_over_f_from_ref: IdealOverFFromRef,
) where
    IcTypes: BenchIcTrait<FIELD_LIMBS> + IdealCheckTypes,
    <<IcTypes as IdealCheckTypes>::F as Field>::Inner: ConstTranscribable,
    Air: Uair<BinaryPoly<DEGREE_PLUS_ONE>> + GenerateWitness<BinaryPoly<DEGREE_PLUS_ONE>>,
    IdealOverF: Ideal,
    IdealOverF: IdealCheck<DynamicPolynomialF<MontyField<FIELD_LIMBS>>>,
    IdealOverFFromRef:
        Fn(&IdealOrZero<<Air as Uair<BinaryPoly<DEGREE_PLUS_ONE>>>::Ideal>) -> IdealOverF + Copy,
{
    let mut rng = rng();
    let num_vars = zinc_utils::log2(witness_size) as usize;
    let trace = Air::generate_witness(num_vars, &mut rng);

    let params = format!(
        "{}/LIMBS={}/nvars={}",
        air_name,
        IcTypes::FIELD_LIMBS,
        num_vars
    );

    let transcript = KeccakTranscript::new();

    let num_constraints = count_constraints::<BinaryPoly<DEGREE_PLUS_ONE>, Air>();

    let prove =
        |(trace, mut transcript): (Vec<_>, KeccakTranscript)| -> Proof<IcTypes, DEGREE_PLUS_ONE> {
            let field_cfg = transcript
                .get_random_field_cfg::<<IcTypes as IdealCheckTypes>::F, <<IcTypes as IdealCheckTypes>::F as Field>::Inner, MillerRabin>();
            IdealCheckProtocol::prove_as_subprotocol::<Air>(
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
                        <IcTypes as IdealCheckTypes>::F,
                        <<IcTypes as IdealCheckTypes>::F as Field>::Inner,
                        MillerRabin,
                    >();
                    let _ = black_box(IdealCheckProtocol::verify_as_subprotocol::<Air, _, _>(
                        &mut transcript,
                        proof,
                        num_constraints,
                        num_vars,
                        ideal_over_f_from_ref,
                        &field_cfg,
                    ))
                    .expect("Failed to verify");
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
    do_bench::<BenchIcTypes<FIELD_LIMBS>, TestAirBinary, _, _, FIELD_LIMBS>(
        group,
        "Binary",
        witness_size,
        |_ideal_over_ring| IdealOrZero::zero(),
    )
}

pub fn bench_simple_mul<const FIELD_LIMBS: usize>(
    group: &mut BenchmarkGroup<WallTime>,
    witness_size: usize,
) {
    do_bench::<BenchIcTypes<FIELD_LIMBS>, TestUairSimpleMultiplication, _, _, FIELD_LIMBS>(
        group,
        "SimpleMul",
        witness_size,
        |_ideal_over_ring| IdealOrZero::zero(),
    )
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

criterion_group!(benches, ideal_check_benches);
criterion_main!(benches);
