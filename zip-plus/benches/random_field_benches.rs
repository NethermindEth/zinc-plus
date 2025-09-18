#![allow(non_local_definitions)]
#![allow(clippy::eq_op)]

use std::{
    hint::black_box,
    iter::{Product, Sum},
};

use criterion::{
    AxisScale, BenchmarkId, Criterion, PlotConfiguration, criterion_group, criterion_main,
};
use crypto_bigint::{U256, const_monty_params};
use zip_plus::{field::ConstMontyField, utils::WORD_FACTOR};

const_monty_params!(
    Params,
    U256,
    "0000000000000000000000000000000000860995AE68FC80E1B1BD1E39D54B33"
);

type F = ConstMontyField<Params, { 4 * WORD_FACTOR }>;

#[allow(clippy::arithmetic_side_effects)]
fn bench_random_field(group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>) {
    let field_elem = F::from(695962179703_u64);

    group.bench_with_input(
        BenchmarkId::new("Multiply", "Random128BitFieldElement"),
        &field_elem,
        |b, unop_elem| {
            b.iter(|| {
                for _ in 0..10000 {
                    let _ = black_box(*unop_elem * *unop_elem);
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
                    let _ = black_box(*unop_elem + *unop_elem);
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
                    let _ = black_box(*unop_elem / unop_elem);
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
                    let _ = black_box(-*unop_elem);
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
                    let _ = black_box(F::sum(v.iter()));
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
                    let _ = black_box(F::product(v.iter()));
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
