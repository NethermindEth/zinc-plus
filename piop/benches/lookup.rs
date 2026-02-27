#![allow(non_local_definitions)]
#![allow(clippy::eq_op)]

use std::hint::black_box;

use criterion::{
    AxisScale, BatchSize, BenchmarkGroup, BenchmarkId, Criterion, PlotConfiguration,
    criterion_group, criterion_main, measurement::WallTime,
};
use crypto_bigint::{U192, const_monty_params};
use crypto_primitives::{
    FromPrimitiveWithConfig, PrimeField, crypto_bigint_const_monty::ConstMontyField,
};
use num_traits::Zero;
use rand::{Rng, rng};
use zinc_piop::lookup::{
    BatchedDecompLogupProtocol, BatchedDecompLookupInstance, DecompLogupProtocol,
    DecompLookupInstance, LogupProtocol,
    tables::{bitpoly_shift, generate_bitpoly_table, generate_word_table},
};
use zinc_transcript::{KeccakTranscript, traits::ConstTranscribable};
use zinc_utils::inner_transparent_field::InnerTransparentField;

// Compile-time 192-bit (3-limb) Montgomery field for benchmarks.
// Each element is 24 bytes, vs ~108 bytes for runtime MontyField<3>.
const_monty_params!(BenchModulus, U192, "fffffffffffffffffffffffffffffffeffffffffffffffff");
type BenchField = ConstMontyField<BenchModulus, 3>;

// ---------------------------------------------------------------------------
// LogUp benchmark (Word table)
// ---------------------------------------------------------------------------

/// Benchmark the core LogUp protocol with a Word(table_width) table.
#[allow(clippy::arithmetic_side_effects)]
fn bench_logup<F, const LIMBS: usize>(
    group: &mut BenchmarkGroup<WallTime>,
    table_width: usize,
    witness_size: usize,
    field_cfg: &F::Config,
) where
    F: InnerTransparentField + FromPrimitiveWithConfig + PrimeField + Send + Sync + 'static,
    F::Inner: ConstTranscribable + Zero + Default + Send + Sync,
{
    let mut rng = rng();
    let base_transcript = KeccakTranscript::new();

    // Generate the lookup table.
    let table = generate_word_table::<F>(table_width, field_cfg);

    // Generate a random witness whose entries are all valid table entries.
    let witness: Vec<F> = (0..witness_size)
        .map(|_| table[rng.random_range(0..table.len())].clone())
        .collect();

    let nvars = zinc_utils::log2(witness_size.next_power_of_two()) as usize;
    let params = format!("LIMBS={}/tw={}/nvars={}", LIMBS, table_width, nvars);

    // ---- Prover benchmark ----
    group.bench_with_input(
        BenchmarkId::new("LogUp Prover", &params),
        &base_transcript,
        |bench, transcript| {
            bench.iter_batched(
                || transcript.clone(),
                |mut transcript| {
                    black_box(
                        LogupProtocol::<F>::prove_as_subprotocol(
                            &mut transcript,
                            &witness,
                            &table,
                            field_cfg,
                        )
                        .expect("prover failed"),
                    );
                },
                BatchSize::SmallInput,
            );
        },
    );

    // Produce a proof for the verifier benchmark.
    let proof = {
        let mut t = base_transcript.clone();
        LogupProtocol::<F>::prove_as_subprotocol(&mut t, &witness, &table, field_cfg)
            .expect("prover failed")
            .0
    };

    // ---- Verifier benchmark ----
    group.bench_with_input(
        BenchmarkId::new("LogUp Verifier", &params),
        &(proof, base_transcript),
        |bench, (proof, transcript)| {
            bench.iter_batched(
                || (proof.clone(), transcript.clone()),
                |(proof, mut transcript)| {
                    black_box(
                        LogupProtocol::<F>::verify_as_subprotocol(
                            &mut transcript,
                            &proof,
                            &table,
                            witness_size,
                            field_cfg,
                        )
                        .expect("verifier failed"),
                    );
                },
                BatchSize::SmallInput,
            );
        },
    );
}

// ---------------------------------------------------------------------------
// Decomposition + LogUp benchmark (BitPoly table)
// ---------------------------------------------------------------------------

/// Benchmark the Decomposition + LogUp protocol with a
/// BitPoly(chunk_width * num_chunks) table decomposed into `num_chunks`
/// BitPoly(chunk_width) sub-tables.
#[allow(clippy::arithmetic_side_effects)]
fn bench_decomp_logup<F, const LIMBS: usize>(
    group: &mut BenchmarkGroup<WallTime>,
    chunk_width: usize,
    num_chunks: usize,
    witness_size: usize,
    field_cfg: &F::Config,
) where
    F: InnerTransparentField + FromPrimitiveWithConfig + PrimeField + Send + Sync + 'static,
    F::Inner: ConstTranscribable + Zero + Default + Send + Sync,
{
    let mut rng = rng();
    let base_transcript = KeccakTranscript::new();

    // Use a = 3 as the projecting element for BitPoly.
    let a = F::from_with_cfg(3u32, field_cfg);

    let full_width = chunk_width * num_chunks;

    // Generate the sub-table (2^chunk_width entries).
    let subtable = generate_bitpoly_table(chunk_width, &a, field_cfg);

    // Compute shifts: shifts[k] = a^{k * chunk_width}.
    let shifts: Vec<F> = (0..num_chunks)
        .map(|k| bitpoly_shift(k * chunk_width, &a))
        .collect();

    // Generate random chunks from the sub-table, then reconstruct witness.
    let chunks: Vec<Vec<F>> = (0..num_chunks)
        .map(|_| {
            (0..witness_size)
                .map(|_| subtable[rng.random_range(0..subtable.len())].clone())
                .collect()
        })
        .collect();

    let witness: Vec<F> = (0..witness_size)
        .map(|i| {
            let mut val = F::zero_with_cfg(field_cfg);
            for k in 0..num_chunks {
                val = val + &(shifts[k].clone() * &chunks[k][i]);
            }
            val
        })
        .collect();

    let nvars = zinc_utils::log2(witness_size.next_power_of_two()) as usize;
    let params = format!(
        "LIMBS={}/bpoly={}/nvars={}",
        LIMBS, full_width, nvars
    );

    let instance = DecompLookupInstance {
        witness: witness.clone(),
        subtable: subtable.clone(),
        shifts: shifts.clone(),
        chunks,
    };

    // ---- Prover benchmark ----
    group.bench_with_input(
        BenchmarkId::new("Decomp+LogUp Prover", &params),
        &base_transcript,
        |bench, transcript| {
            bench.iter_batched(
                || transcript.clone(),
                |mut transcript| {
                    black_box(
                        DecompLogupProtocol::<F>::prove_as_subprotocol(
                            &mut transcript,
                            &instance,
                            field_cfg,
                        )
                        .expect("prover failed"),
                    );
                },
                BatchSize::SmallInput,
            );
        },
    );

    // Produce a proof for the verifier benchmark.
    let proof = {
        let mut t = base_transcript.clone();
        DecompLogupProtocol::<F>::prove_as_subprotocol(&mut t, &instance, field_cfg)
            .expect("prover failed")
            .0
    };

    // ---- Verifier benchmark ----
    group.bench_with_input(
        BenchmarkId::new("Decomp+LogUp Verifier", &params),
        &(proof, base_transcript),
        |bench, (proof, transcript)| {
            bench.iter_batched(
                || (proof.clone(), transcript.clone()),
                |(proof, mut transcript)| {
                    black_box(
                        DecompLogupProtocol::<F>::verify_as_subprotocol(
                            &mut transcript,
                            &proof,
                            &subtable,
                            &shifts,
                            witness_size,
                            field_cfg,
                        )
                        .expect("verifier failed"),
                    );
                },
                BatchSize::SmallInput,
            );
        },
    );
}

// ---------------------------------------------------------------------------
// Batched Decomposition + LogUp benchmark
// ---------------------------------------------------------------------------

/// Benchmark the batched Decomposition + LogUp protocol with L
/// lookups into a shared BitPoly(chunk_width * num_chunks) table.
#[allow(clippy::arithmetic_side_effects)]
fn bench_batched_decomp_logup<F, const LIMBS: usize>(
    group: &mut BenchmarkGroup<WallTime>,
    chunk_width: usize,
    num_chunks: usize,
    num_lookups: usize,
    witness_size: usize,
    field_cfg: &F::Config,
) where
    F: InnerTransparentField + FromPrimitiveWithConfig + PrimeField + Send + Sync + 'static,
    F::Inner: ConstTranscribable + Zero + Default + Send + Sync,
{
    let mut rng = rng();
    let base_transcript = KeccakTranscript::new();

    let a = F::from_with_cfg(3u32, field_cfg);
    let full_width = chunk_width * num_chunks;

    let subtable = generate_bitpoly_table(chunk_width, &a, field_cfg);
    let shifts: Vec<F> = (0..num_chunks)
        .map(|k| bitpoly_shift(k * chunk_width, &a))
        .collect();

    // Build L random witnesses with their chunk decompositions.
    let mut witnesses = Vec::with_capacity(num_lookups);
    let mut all_chunks = Vec::with_capacity(num_lookups);
    for _ in 0..num_lookups {
        let chunks: Vec<Vec<F>> = (0..num_chunks)
            .map(|_| {
                (0..witness_size)
                    .map(|_| subtable[rng.random_range(0..subtable.len())].clone())
                    .collect()
            })
            .collect();
        let witness: Vec<F> = (0..witness_size)
            .map(|i| {
                let mut val = F::zero_with_cfg(field_cfg);
                for k in 0..num_chunks {
                    val = val + &(shifts[k].clone() * &chunks[k][i]);
                }
                val
            })
            .collect();
        witnesses.push(witness);
        all_chunks.push(chunks);
    }

    let nvars = zinc_utils::log2(witness_size.next_power_of_two()) as usize;
    let params = format!(
        "LIMBS={}/bpoly={}/L={}/nvars={}",
        LIMBS, full_width, num_lookups, nvars
    );

    let instance = BatchedDecompLookupInstance {
        witnesses: witnesses.clone(),
        subtable: subtable.clone(),
        shifts: shifts.clone(),
        chunks: all_chunks,
    };

    // ---- Prover benchmark ----
    group.bench_with_input(
        BenchmarkId::new("Batched Prover", &params),
        &base_transcript,
        |bench, transcript| {
            bench.iter_batched(
                || transcript.clone(),
                |mut transcript| {
                    black_box(
                        BatchedDecompLogupProtocol::<F>::prove_as_subprotocol(
                            &mut transcript,
                            &instance,
                            field_cfg,
                        )
                        .expect("prover failed"),
                    );
                },
                BatchSize::SmallInput,
            );
        },
    );

    // Produce a proof for the verifier benchmark.
    let proof = {
        let mut t = base_transcript.clone();
        BatchedDecompLogupProtocol::<F>::prove_as_subprotocol(&mut t, &instance, field_cfg)
            .expect("prover failed")
            .0
    };

    // ---- Verifier benchmark ----
    group.bench_with_input(
        BenchmarkId::new("Batched Verifier", &params),
        &(proof, base_transcript),
        |bench, (proof, transcript)| {
            bench.iter_batched(
                || (proof.clone(), transcript.clone()),
                |(proof, mut transcript)| {
                    black_box(
                        BatchedDecompLogupProtocol::<F>::verify_as_subprotocol(
                            &mut transcript,
                            &proof,
                            &subtable,
                            &shifts,
                            num_lookups,
                            witness_size,
                            field_cfg,
                        )
                        .expect("verifier failed"),
                    );
                },
                BatchSize::SmallInput,
            );
        },
    );
}

// ---------------------------------------------------------------------------
// Benchmark entry points
// ---------------------------------------------------------------------------

pub fn lookup_benches(c: &mut Criterion) {
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);

    // --- LogUp benchmarks (Word table, width=8 → 256 entries) ---
    {
        let mut group = c.benchmark_group("LogUp Word(8)");
        group.plot_config(plot_config.clone());

        //bench_logup::<BenchField, 3>(&mut group, 8, 1 << 10, &());
        // bench_logup::<BenchField, 3>(&mut group, 8, 1 << 12, &());
        // bench_logup::<BenchField, 3>(&mut group, 8, 1 << 14, &());

        group.finish();
    }

    // --- Decomposition + LogUp benchmarks (BitPoly, 4 chunks × 2^8 → full=32) ---
    {
        let mut group = c.benchmark_group("Decomp+LogUp BitPoly(32)");
        group.plot_config(plot_config);

        // bench_decomp_logup::<BenchField, 3>(&mut group, 8, 4, 1 << 10, &());
        // bench_decomp_logup::<BenchField, 3>(&mut group, 8, 4, 1 << 14, &());

        group.finish();
    }

    // --- Batched Decomposition + LogUp benchmarks (L lookups, BitPoly(32), 4×2^8) ---
    {
        let mut group = c.benchmark_group("Batched Decomp+LogUp BitPoly(32)");
        group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

        bench_batched_decomp_logup::<BenchField, 3>(&mut group, 8, 4, 5, 1 << 10, &());
        bench_batched_decomp_logup::<BenchField, 3>(&mut group, 8, 4, 10, 1 << 10, &());
        // bench_batched_decomp_logup::<BenchField, 3>(&mut group, 8, 4, 10, 1 << 14, &());
        // bench_batched_decomp_logup::<BenchField, 3>(&mut group, 8, 4, 20, 1 << 10, &());

        group.finish();
    }
}

criterion_group!(benches, lookup_benches);
criterion_main!(benches);
