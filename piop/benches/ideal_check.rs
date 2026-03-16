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
    ideal_check::{IdealCheckProtocol, Proof},
    projections::{
        project_scalars, project_trace_coeffs_column_major, project_trace_coeffs_row_major,
    },
};
use zinc_poly::univariate::dense::DensePolynomial;
use zinc_primality::{MillerRabin, PrimalityTest};
use zinc_test_uair::{
    BigLinearUair, BinaryDecompositionUair, GenerateMultiTypeWitness, GenerateSingleTypeWitness,
    TestAirNoMultiplication, TestUairSimpleMultiplication,
};
use zinc_transcript::{
    KeccakTranscript,
    traits::{ConstTranscribable, Transcript},
};
use zinc_uair::{
    Uair, constraint_counter::count_constraints, ideal::degree_one::DegreeOneIdeal,
    ideal_collector::IdealOrZero,
};

const DEGREE_PLUS_ONE: usize = 32;

type Witness<const INT_LIMBS: usize> = DensePolynomial<Int<INT_LIMBS>, DEGREE_PLUS_ONE>;
type F<const FIELD_LIMBS: usize> = MontyField<FIELD_LIMBS>;

#[allow(clippy::arithmetic_side_effects)]
fn bench_no_mult<const INT_LIMBS: usize, const FIELD_LIMBS: usize>(
    group: &mut BenchmarkGroup<WallTime>,
    witness_size: usize,
) where
    <F<FIELD_LIMBS> as Field>::Inner: ConstIntSemiring + ConstTranscribable,
    TestAirNoMultiplication<Int<INT_LIMBS>>: Uair<Scalar = Witness<INT_LIMBS>, Ideal = DegreeOneIdeal<Int<INT_LIMBS>>>
        + GenerateSingleTypeWitness<Witness = Witness<INT_LIMBS>>
        + IdealCheckProtocol,
    MillerRabin: PrimalityTest<<F<FIELD_LIMBS> as Field>::Inner>,
{
    let mut rng = rng();
    let num_vars = zinc_utils::log2(witness_size) as usize;
    let trace = TestAirNoMultiplication::generate_witness(num_vars, &mut rng);

    let params = format!("NoMult/LIMBS={}/nvars={}", FIELD_LIMBS, num_vars);

    let num_constraints = count_constraints::<TestAirNoMultiplication<Int<INT_LIMBS>>>();

    let prove = |field_cfg: &<F<FIELD_LIMBS> as PrimeField>::Config,
                 trace: &[_],
                 transcript: &mut KeccakTranscript|
     -> Proof<F<FIELD_LIMBS>> {
        let trace = project_trace_coeffs_row_major::<_, _, Int<5>, _>(&[], trace, &[], field_cfg);

        let projected_scalars =
            project_scalars::<F<FIELD_LIMBS>, TestAirNoMultiplication<Int<INT_LIMBS>>>(|scalar| {
                scalar
                    .iter()
                    .map(|coeff| F::from_with_cfg(coeff, field_cfg))
                    .collect()
            });

        // Even though this UAIR is linear, using prove_combined yields much better
        // prover performance for it.
        TestAirNoMultiplication::prove_combined(
            transcript,
            &trace,
            &projected_scalars,
            num_constraints,
            num_vars,
            field_cfg,
        )
        .expect("Prover failed")
        .0
    };

    group.bench_with_input(
        BenchmarkId::new("Ideal Check Prover", &params),
        &trace,
        |bench, trace| {
            let mut transcript = KeccakTranscript::new();
            let field_cfg = transcript.get_random_field_cfg::<F<FIELD_LIMBS>, _, MillerRabin>();
            bench.iter_batched(
                || (trace, transcript.clone()),
                |(trace, mut transcript)| {
                    let _ = black_box(prove(&field_cfg, trace, &mut transcript));
                },
                BatchSize::SmallInput,
            );
        },
    );

    group.bench_with_input(
        BenchmarkId::new("Ideal Check Verifier", &params),
        &trace,
        |bench, trace| {
            let mut transcript = KeccakTranscript::new();
            let field_cfg = transcript.get_random_field_cfg::<F<FIELD_LIMBS>, _, MillerRabin>();
            let proof = prove(&field_cfg, trace, &mut transcript);

            bench.iter_batched(
                || (proof.clone(), transcript.clone()),
                |(proof, mut transcript)| {
                    let _ = black_box(TestAirNoMultiplication::verify_as_subprotocol(
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
    bench_no_mult::<3, 4>(group, witness_size)
}

pub fn bench_no_mult_4(group: &mut BenchmarkGroup<WallTime>, witness_size: usize) {
    bench_no_mult::<4, 5>(group, witness_size)
}

#[allow(clippy::arithmetic_side_effects)]
fn bench_simple_mult<const INT_LIMBS: usize, const FIELD_LIMBS: usize>(
    group: &mut BenchmarkGroup<WallTime>,
    witness_size: usize,
) where
    <F<FIELD_LIMBS> as Field>::Inner: ConstIntSemiring + ConstTranscribable,
    TestUairSimpleMultiplication<Int<INT_LIMBS>>: Uair<Scalar = Witness<INT_LIMBS>>
        + GenerateSingleTypeWitness<Witness = Witness<INT_LIMBS>>
        + IdealCheckProtocol,
    MillerRabin: PrimalityTest<<F<FIELD_LIMBS> as Field>::Inner>,
{
    let mut rng = rng();
    let num_vars = zinc_utils::log2(witness_size) as usize;
    let trace = TestUairSimpleMultiplication::generate_witness(num_vars, &mut rng);

    let params = format!("SimpleMult/LIMBS={}/nvars={}", FIELD_LIMBS, num_vars);

    let num_constraints = count_constraints::<TestUairSimpleMultiplication<Int<INT_LIMBS>>>();

    let prove = |field_cfg: &<F<FIELD_LIMBS> as PrimeField>::Config,
                 trace: &[_],
                 transcript: &mut KeccakTranscript|
     -> Proof<F<FIELD_LIMBS>> {
        let trace = project_trace_coeffs_row_major::<_, _, Int<5>, _>(&[], trace, &[], field_cfg);

        let projected_scalars = project_scalars::<
            F<FIELD_LIMBS>,
            TestUairSimpleMultiplication<Int<INT_LIMBS>>,
        >(|scalar| {
            scalar
                .iter()
                .map(|coeff| F::from_with_cfg(coeff, field_cfg))
                .collect()
        });

        TestUairSimpleMultiplication::prove_combined(
            transcript,
            &trace,
            &projected_scalars,
            num_constraints,
            num_vars,
            field_cfg,
        )
        .expect("Prover failed")
        .0
    };

    group.bench_with_input(
        BenchmarkId::new("Ideal Check Prover", &params),
        &trace,
        |bench, trace| {
            let mut transcript = KeccakTranscript::new();
            let field_cfg = transcript.get_random_field_cfg::<F<FIELD_LIMBS>, _, MillerRabin>();
            bench.iter_batched(
                || (trace, transcript.clone()),
                |(trace, mut transcript)| {
                    let _ = black_box(prove(&field_cfg, trace, &mut transcript));
                },
                BatchSize::SmallInput,
            );
        },
    );

    group.bench_with_input(
        BenchmarkId::new("Ideal Check Verifier", &params),
        &trace,
        |bench, trace| {
            let mut transcript = KeccakTranscript::new();
            let field_cfg = transcript.get_random_field_cfg::<F<FIELD_LIMBS>, _, MillerRabin>();
            let proof = prove(&field_cfg, trace, &mut transcript);

            bench.iter_batched(
                || (proof.clone(), transcript.clone()),
                |(proof, mut transcript)| {
                    let _ = black_box(TestUairSimpleMultiplication::verify_as_subprotocol(
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
    bench_simple_mult::<3, 4>(group, witness_size)
}

pub fn bench_simple_mult_4(group: &mut BenchmarkGroup<WallTime>, witness_size: usize) {
    bench_simple_mult::<4, 5>(group, witness_size)
}

#[allow(clippy::arithmetic_side_effects)]
fn bench_binary_decomposition<const FIELD_LIMBS: usize>(
    group: &mut BenchmarkGroup<WallTime>,
    witness_size: usize,
) where
    <F<FIELD_LIMBS> as Field>::Inner: ConstIntSemiring + ConstTranscribable,
    MillerRabin: PrimalityTest<<F<FIELD_LIMBS> as Field>::Inner>,
{
    let mut rng = rng();
    let num_vars = zinc_utils::log2(witness_size) as usize;
    let (binary_poly_trace, _, int_trace) =
        BinaryDecompositionUair::generate_witness(num_vars, &mut rng);

    let params = format!(
        "BinaryDecomposition/LIMBS={}/nvars={}",
        FIELD_LIMBS, num_vars
    );

    let num_constraints = count_constraints::<BinaryDecompositionUair<u32>>();

    let prove = |field_cfg: &<F<FIELD_LIMBS> as PrimeField>::Config,
                 binary_poly_trace: &[_],
                 int_trace: &[_],
                 transcript: &mut KeccakTranscript|
     -> Proof<F<FIELD_LIMBS>> {
        let trace = project_trace_coeffs_row_major::<_, u32, u32, _>(
            binary_poly_trace,
            &[],
            int_trace,
            field_cfg,
        );

        let projected_scalars =
            project_scalars::<F<FIELD_LIMBS>, BinaryDecompositionUair<u32>>(|scalar| {
                scalar
                    .iter()
                    .map(|coeff| F::from_with_cfg(coeff, field_cfg))
                    .collect()
            });

        BinaryDecompositionUair::prove_combined(
            transcript,
            &trace,
            &projected_scalars,
            num_constraints,
            num_vars,
            field_cfg,
        )
        .expect("Prover failed")
        .0
    };

    group.bench_with_input(
        BenchmarkId::new("Ideal Check Prover", &params),
        &(&binary_poly_trace, &int_trace),
        |bench, (binary_poly_trace, int_trace)| {
            let mut transcript = KeccakTranscript::new();
            let field_cfg = transcript.get_random_field_cfg::<F<FIELD_LIMBS>, _, MillerRabin>();
            bench.iter_batched(
                || (&field_cfg, binary_poly_trace, int_trace, transcript.clone()),
                |(field_cfg, binary_poly_trace, int_trace, mut transcript)| {
                    let _ = black_box(prove(
                        field_cfg,
                        binary_poly_trace,
                        int_trace,
                        &mut transcript,
                    ));
                },
                BatchSize::SmallInput,
            );
        },
    );

    group.bench_with_input(
        BenchmarkId::new("Ideal Check Verifier", &params),
        &(&binary_poly_trace, &int_trace),
        |bench, (binary_poly_trace, int_trace)| {
            let mut transcript = KeccakTranscript::new();
            let field_cfg = transcript.get_random_field_cfg::<F<FIELD_LIMBS>, _, MillerRabin>();
            let proof = prove(&field_cfg, binary_poly_trace, int_trace, &mut transcript);

            bench.iter_batched(
                || (proof.clone(), transcript.clone()),
                |(proof, mut transcript)| {
                    let _ = black_box(BinaryDecompositionUair::<u32>::verify_as_subprotocol(
                        &mut transcript,
                        proof,
                        num_constraints,
                        num_vars,
                        |ideal_over_ring| {
                            ideal_over_ring.map(|ideal_over_ring| {
                                DegreeOneIdeal::from_with_cfg(ideal_over_ring, &field_cfg)
                            })
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

#[allow(clippy::arithmetic_side_effects)]
fn bench_big_linear_uair<const FIELD_LIMBS: usize>(
    group: &mut BenchmarkGroup<WallTime>,
    witness_size: usize,
) where
    <F<FIELD_LIMBS> as Field>::Inner: ConstIntSemiring + ConstTranscribable,
    MillerRabin: PrimalityTest<<F<FIELD_LIMBS> as Field>::Inner>,
{
    let mut rng = rng();
    let num_vars = zinc_utils::log2(witness_size) as usize;
    let (binary_poly_trace, _, int_trace) = BigLinearUair::generate_witness(num_vars, &mut rng);

    let params = format!("BigLinearUair/LIMBS={}/nvars={}", FIELD_LIMBS, num_vars);

    let num_constraints = count_constraints::<BigLinearUair<u32>>();

    macro_rules! prove {
        ($transcript:expr, $field_cfg:expr, $gen_trace:ident, $prove_fn:ident) => {{
            let trace =
                $gen_trace::<_, u32, u32, _>(&binary_poly_trace, &[], &int_trace, $field_cfg);

            let projected_scalars =
                project_scalars::<F<FIELD_LIMBS>, BigLinearUair<u32>>(|scalar| {
                    scalar
                        .iter()
                        .map(|coeff| F::from_with_cfg(coeff, $field_cfg))
                        .collect()
                });

            BigLinearUair::$prove_fn(
                $transcript,
                &trace,
                &projected_scalars,
                num_constraints,
                num_vars,
                $field_cfg,
            )
            .expect("Prover failed")
            .0
        }};
    }

    group.bench_function(
        BenchmarkId::new("Ideal Check Prover (MLE-first)", &params),
        |bench| {
            let mut transcript = KeccakTranscript::new();
            let field_cfg = transcript.get_random_field_cfg::<F<FIELD_LIMBS>, _, MillerRabin>();
            bench.iter_batched(
                || (&field_cfg, transcript.clone()),
                |(field_cfg, mut transcript)| {
                    let proof = prove!(
                        &mut transcript,
                        field_cfg,
                        project_trace_coeffs_column_major,
                        prove_linear
                    );
                    black_box(proof);
                },
                BatchSize::SmallInput,
            );
        },
    );

    group.bench_function(
        BenchmarkId::new("Ideal Check Prover (Combined)", &params),
        |bench| {
            let mut transcript = KeccakTranscript::new();
            let field_cfg = transcript.get_random_field_cfg::<F<FIELD_LIMBS>, _, MillerRabin>();
            bench.iter_batched(
                || (&field_cfg, transcript.clone()),
                |(field_cfg, mut transcript)| {
                    let proof = prove!(
                        &mut transcript,
                        field_cfg,
                        project_trace_coeffs_row_major,
                        prove_combined
                    );
                    black_box(proof);
                },
                BatchSize::SmallInput,
            );
        },
    );

    let mut transcript = KeccakTranscript::new();
    let field_cfg = transcript.get_random_field_cfg::<F<FIELD_LIMBS>, _, MillerRabin>();
    let proof = prove!(
        &mut transcript,
        &field_cfg,
        project_trace_coeffs_column_major,
        prove_linear
    );

    group.bench_function(BenchmarkId::new("Ideal Check Verifier", &params), |bench| {
        bench.iter_batched(
            || (proof.clone(), transcript.clone()),
            |(proof, mut transcript)| {
                let _ = black_box(BigLinearUair::<u32>::verify_as_subprotocol(
                    &mut transcript,
                    proof,
                    num_constraints,
                    num_vars,
                    |ideal_over_ring| {
                        ideal_over_ring.map(|ideal_over_ring| {
                            DegreeOneIdeal::from_with_cfg(ideal_over_ring, &field_cfg)
                        })
                    },
                    &field_cfg,
                ))
                .expect("Failed to verify");
            },
            BatchSize::SmallInput,
        );
    });
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

    bench_binary_decomposition::<3>(&mut group, 1 << 12);
    bench_binary_decomposition::<4>(&mut group, 1 << 12);
    bench_binary_decomposition::<3>(&mut group, 1 << 13);
    bench_binary_decomposition::<4>(&mut group, 1 << 13);
    bench_binary_decomposition::<3>(&mut group, 1 << 14);
    bench_binary_decomposition::<4>(&mut group, 1 << 14);
    bench_binary_decomposition::<3>(&mut group, 1 << 15);
    bench_binary_decomposition::<4>(&mut group, 1 << 15);
    bench_binary_decomposition::<3>(&mut group, 1 << 16);
    bench_binary_decomposition::<4>(&mut group, 1 << 16);
    bench_binary_decomposition::<3>(&mut group, 1 << 17);
    bench_binary_decomposition::<4>(&mut group, 1 << 17);

    bench_big_linear_uair::<3>(&mut group, 1 << 12);
    bench_big_linear_uair::<4>(&mut group, 1 << 12);
    bench_big_linear_uair::<3>(&mut group, 1 << 13);
    bench_big_linear_uair::<4>(&mut group, 1 << 13);
    bench_big_linear_uair::<3>(&mut group, 1 << 14);
    bench_big_linear_uair::<4>(&mut group, 1 << 14);
    bench_big_linear_uair::<3>(&mut group, 1 << 15);
    bench_big_linear_uair::<4>(&mut group, 1 << 15);
    bench_big_linear_uair::<3>(&mut group, 1 << 16);
    bench_big_linear_uair::<4>(&mut group, 1 << 16);
    bench_big_linear_uair::<3>(&mut group, 1 << 17);
    bench_big_linear_uair::<4>(&mut group, 1 << 17);

    group.finish();
}

criterion_group!(benches, ideal_check_benches);
criterion_main!(benches);
