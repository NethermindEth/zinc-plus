#![allow(non_local_definitions)]
#![allow(clippy::eq_op)]

use ark_std::{
    test_rng,
    time::{Duration, Instant},
};
use criterion::{criterion_group, criterion_main, measurement::WallTime, AxisScale, BenchmarkGroup, BenchmarkId, Criterion, PlotConfiguration};
use crypto_bigint::Random;
use itertools::Itertools;
use std::hint::black_box;
use std::iter::{Product, Sum};
use zip_plus::field::config::FieldConfigBase;
use zip_plus::{code::{DefaultLinearCodeSpec, LinearCode}, code_raa::RaaCode, define_field_config, define_random_field_zip_types, field::RandomField, implement_random_field_zip_types, pcs::{structs::MultilinearZip, MerkleTree}, pcs_transcript::PcsTranscript, poly_z::mle::{DenseMultilinearExtension, MultilinearExtension}, traits::{FieldMap, ZipTypes}, transcript::KeccakTranscript};

define_field_config!(Fc, "695962179703626800597079116051991347");

fn bench_random_field(group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>) {
    let field_elem: RandomField<4, Fc<4>> = 695962179703_u64.map_to_field();
    group.bench_with_input(
        BenchmarkId::new("Multiply", "Random128BitFieldElement"),
        &field_elem,
        |b, unop_elem| {
            b.iter(|| {
                for _ in 0..10000 {
                    let _ = black_box(unop_elem.clone() * unop_elem.clone());
                }
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("Addition", "Random128BitFieldElement"),
        &field_elem,
        |b, unop_elem| {
            b.iter(|| {
                for _ in 0..10000 {
                    let _ = black_box(unop_elem.clone() + unop_elem.clone());
                }
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("Division", "Random128BitFieldElement"),
        &field_elem,
        |b, unop_elem| {
            b.iter(|| {
                for _ in 0..10000 {
                    let _ = black_box(unop_elem.clone() / unop_elem.clone());
                }
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("Negation", "Random128BitFieldElement"),
        &field_elem,
        |b, unop_elem| {
            b.iter(|| {
                for _ in 0..10000 {
                    let _ = black_box(-unop_elem.clone());
                }
            });
        },
    );

    let v = vec![field_elem; 10];

    group.bench_with_input(
        BenchmarkId::new("Sum", "Random128BitFieldElement"),
        &v,
        |b, v| {
            b.iter(|| {
                for _ in 0..10000 {
                    let _ = black_box(RandomField::sum(v.iter()));
                }
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("Product", "Random128BitFieldElement"),
        &v,
        |b, v| {
            b.iter(|| {
                for _ in 0..10000 {
                    let _ = black_box(RandomField::product(v.iter()));
                }
            });
        },
    );
}

pub fn field_benchmarks(c: &mut Criterion) {
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);

    let mut group = c.benchmark_group("RandomFieldArithmetic");
    group.plot_config(plot_config);

    bench_random_field(&mut group);
    group.finish();
}

criterion_group!(benches, field_benchmarks);
criterion_main!(benches);
