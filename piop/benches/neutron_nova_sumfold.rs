#![allow(non_local_definitions)]

use std::hint::black_box;

use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use crypto_primitives::{Field, FromWithConfig, PrimeField, crypto_bigint_monty::MontyField};
use zinc_piop::{
    neutron_nova::{
        ProjectedShaPublic, ProjectedShaTrace, SHA_ROW_COUNT, SHA_WORD_BITS, ShaBitSliceColumns,
        ShaBooleanitySource, ShaIntCol, ShaIntColumns, ShaPublicCol, ShaPublicColumns, ShaWordCol,
        build_dense_sha_sumfold_group, build_production_sha_sumfold_group_owned,
        scalarize_trace_words,
    },
    sumcheck::multi_degree::MultiDegreeSumcheck,
};
use zinc_primality::{MillerRabin, PrimalityTest};
use zinc_transcript::{Blake3Transcript, traits::Transcript};

type F = MontyField<4>;

fn bench_config() -> <F as PrimeField>::Config
where
    MillerRabin: PrimalityTest<<F as Field>::Modulus>,
{
    let mut transcript = Blake3Transcript::new();
    transcript.get_random_field_cfg::<F, <F as Field>::Modulus, MillerRabin>()
}

fn f(value: u64, cfg: &<F as PrimeField>::Config) -> F {
    F::from_with_cfg(value, cfg)
}

fn synthetic_boolean_trace(
    instance_idx: u64,
    a: &F,
    cfg: &<F as PrimeField>::Config,
) -> ProjectedShaTrace<F> {
    let zero = F::zero_with_cfg(cfg);
    let mut bits = vec![vec![vec![zero.clone(); SHA_WORD_BITS]; SHA_ROW_COUNT]; ShaWordCol::COUNT];
    for (col_idx, col) in bits.iter_mut().enumerate() {
        for (row_idx, row) in col.iter_mut().enumerate() {
            for (bit_idx, bit) in row.iter_mut().enumerate() {
                let selector = instance_idx
                    + u64::try_from(col_idx * 17 + row_idx * 3 + bit_idx)
                        .expect("bench selector fits u64");
                if selector % 2 == 1 {
                    *bit = f(1, cfg);
                }
            }
        }
    }
    let bit_slices = ShaBitSliceColumns { columns: bits };
    let scalarized_words = scalarize_trace_words(&bit_slices, a, cfg).unwrap();
    ProjectedShaTrace {
        rows: SHA_ROW_COUNT,
        bit_slices,
        scalarized_words,
        int_columns: ShaIntColumns {
            columns: vec![vec![zero.clone(); SHA_ROW_COUNT]; ShaIntCol::COUNT],
        },
        public_columns: ShaPublicColumns {
            columns: vec![vec![zero; SHA_ROW_COUNT]; ShaPublicCol::COUNT],
        },
    }
}

fn zero_public(cfg: &<F as PrimeField>::Config) -> ProjectedShaPublic<F> {
    ProjectedShaPublic {
        columns: ShaPublicColumns {
            columns: vec![vec![F::zero_with_cfg(cfg); SHA_ROW_COUNT]; ShaPublicCol::COUNT],
        },
    }
}

fn booleanity_sources() -> Vec<ShaBooleanitySource> {
    vec![
        ShaBooleanitySource::WordBit {
            col: ShaWordCol::A,
            bit: 0,
        },
        ShaBooleanitySource::WordBit {
            col: ShaWordCol::A,
            bit: 7,
        },
        ShaBooleanitySource::WordBit {
            col: ShaWordCol::E,
            bit: 1,
        },
        ShaBooleanitySource::WordBit {
            col: ShaWordCol::E,
            bit: 9,
        },
        ShaBooleanitySource::WordBit {
            col: ShaWordCol::W,
            bit: 2,
        },
        ShaBooleanitySource::WordBit {
            col: ShaWordCol::W,
            bit: 13,
        },
        ShaBooleanitySource::WordBit {
            col: ShaWordCol::Sigma0,
            bit: 3,
        },
        ShaBooleanitySource::WordBit {
            col: ShaWordCol::Sigma1,
            bit: 5,
        },
    ]
}

#[allow(clippy::too_many_lines)]
fn neutron_nova_sumfold_benches(c: &mut Criterion) {
    let cfg = bench_config();
    let ell = 7usize;
    let prefix_vars = 2usize;
    let a = f(3, &cfg);
    let traces = (0..(1usize << ell))
        .map(|idx| synthetic_boolean_trace(u64::try_from(idx).unwrap(), &a, &cfg))
        .collect::<Vec<_>>();
    let publics = vec![zero_public(&cfg); traces.len()];
    let beta = vec![
        f(5, &cfg),
        f(7, &cfg),
        f(11, &cfg),
        f(13, &cfg),
        f(17, &cfg),
        f(19, &cfg),
        f(37, &cfg),
    ];
    let r_ic = [
        f(2, &cfg),
        f(3, &cfg),
        f(5, &cfg),
        f(7, &cfg),
        f(11, &cfg),
        f(13, &cfg),
        f(17, &cfg),
    ];
    let lambda = f(23, &cfg);
    let rho = f(29, &cfg);
    let xi = f(31, &cfg);
    let sources = booleanity_sources();

    let mut group = c.benchmark_group("NeutronNova SHA SumFold");
    group.sample_size(10);

    group.bench_function(BenchmarkId::new("dense_build_and_prove", ell), |bench| {
        bench.iter_batched(
            Blake3Transcript::new,
            |mut transcript| {
                let group = build_dense_sha_sumfold_group(
                    &traces, &publics, &beta, &r_ic, &a, &lambda, &rho, &xi, &sources, &cfg,
                )
                .unwrap();
                let (proof, _) = MultiDegreeSumcheck::prove_as_subprotocol(
                    &mut transcript,
                    vec![group],
                    ell,
                    &cfg,
                );
                black_box(proof.claimed_sums()[0].clone())
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function(
        BenchmarkId::new("production_prefix_tail_build_and_prove", prefix_vars),
        |bench| {
            bench.iter_batched(
                || (traces.clone().into_boxed_slice(), Blake3Transcript::new()),
                |(owned_traces, mut transcript)| {
                    let group = build_production_sha_sumfold_group_owned(
                        owned_traces,
                        &publics,
                        &beta,
                        &r_ic,
                        &a,
                        &lambda,
                        &rho,
                        &xi,
                        &sources,
                        prefix_vars,
                        &cfg,
                    )
                    .unwrap();
                    let (proof, _) = MultiDegreeSumcheck::prove_as_subprotocol(
                        &mut transcript,
                        vec![group],
                        ell,
                        &cfg,
                    );
                    black_box(proof.claimed_sums()[0].clone())
                },
                BatchSize::SmallInput,
            );
        },
    );

    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = neutron_nova_sumfold_benches
}
criterion_main!(benches);
