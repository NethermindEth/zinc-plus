#![allow(non_local_definitions)]
#![allow(clippy::eq_op)]

use std::hint::black_box;
use ark_std::{
    iter::{Product, Sum},
    iterable::Iterable,
};
use criterion::{
    criterion_group, criterion_main, AxisScale, BenchmarkId, Criterion,
    PlotConfiguration,
};
use zip_plus::{big_int, define_field_config, field::{ConfigRef, RandomField}, field_config, random_field};
use zip_plus::field::config::ConstFieldConfig;
use zip_plus::traits::{ConfigReference, FieldMap};

define_field_config!(Fc, "695962179703626800597079116051991347");

fn bench_random_field(group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>) {
    let field_config = unsafe { ConfigRef::new(Box::leak(Box::new(Fc::<4>::field_config()))) };

    let field_elem: RandomField<4, Fc<4>> = 695962179703_u64.map_to_field(field_config);
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
