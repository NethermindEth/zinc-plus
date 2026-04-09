#![allow(non_local_definitions)]
#![allow(clippy::eq_op)]

use std::{
    hint::black_box,
    iter::{Product, Sum},
    ops::{Add, Div, Mul},
};

use criterion::{AxisScale, BatchSize, BenchmarkId, Criterion, PlotConfiguration};
use crypto_primitives::FromPrimitiveWithConfig;

fn bench_random_field<F>(
    group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>,
    num: u64,
    config: &F::Config,
) where
    for<'a> &'a F: Add<&'a F> + Mul<&'a F> + Div<&'a F>,
    F: FromPrimitiveWithConfig + for<'a> Sum<&'a F> + for<'a> Product<&'a F>,
{
    let field_elem = F::from_with_cfg(num, config);
    let param = format!("Param = {}", num);

    group.bench_with_input(
        BenchmarkId::new("Mul owned by owned", &param),
        &field_elem,
        |b, unop_elem| {
            b.iter_batched(
                || {
                    (
                        vec![unop_elem.clone(); 10000],
                        vec![unop_elem.clone(); 10000],
                    )
                },
                |(lhs, rhs)| {
                    for (lhs, rhs) in lhs.into_iter().zip(rhs.into_iter()) {
                        let _ = black_box(lhs * rhs);
                    }
                },
                BatchSize::SmallInput,
            );
        },
    );

    group.bench_with_input(
        BenchmarkId::new("Mul owned by ref", &param),
        &field_elem,
        |b, unop_elem| {
            b.iter_batched(
                || {
                    (
                        vec![unop_elem.clone(); 10000],
                        vec![unop_elem.clone(); 10000],
                    )
                },
                |(lhs, rhs)| {
                    for (lhs, rhs) in lhs.into_iter().zip(rhs.into_iter()) {
                        let _ = black_box(lhs * &rhs);
                    }
                },
                BatchSize::SmallInput,
            );
        },
    );

    group.bench_with_input(
        BenchmarkId::new("Mul ref by ref", &param),
        &field_elem,
        |b, unop_elem| {
            b.iter(|| {
                for _ in 0..10000 {
                    let _ = black_box(unop_elem * unop_elem);
                }
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("Add owned to owned", &param),
        &field_elem,
        |b, unop_elem| {
            b.iter_batched(
                || {
                    (
                        vec![unop_elem.clone(); 10000],
                        vec![unop_elem.clone(); 10000],
                    )
                },
                |(lhs, rhs)| {
                    for (lhs, rhs) in lhs.into_iter().zip(rhs.into_iter()) {
                        let _ = black_box(lhs + rhs);
                    }
                },
                BatchSize::SmallInput,
            );
        },
    );

    group.bench_with_input(
        BenchmarkId::new("Add owned to ref", &param),
        &field_elem,
        |b, unop_elem| {
            b.iter_batched(
                || {
                    (
                        vec![unop_elem.clone(); 10000],
                        vec![unop_elem.clone(); 10000],
                    )
                },
                |(lhs, rhs)| {
                    for (lhs, rhs) in lhs.into_iter().zip(rhs.into_iter()) {
                        let _ = black_box(lhs + &rhs);
                    }
                },
                BatchSize::SmallInput,
            );
        },
    );

    group.bench_with_input(
        BenchmarkId::new("Add ref to ref", &param),
        &field_elem,
        |b, unop_elem| {
            b.iter(|| {
                for _ in 0..10000 {
                    let _ = black_box(unop_elem + unop_elem);
                }
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("Div owned by owned", &param),
        &field_elem,
        |b, unop_elem| {
            b.iter_batched(
                || {
                    (
                        vec![unop_elem.clone(); 10000],
                        vec![unop_elem.clone(); 10000],
                    )
                },
                |(lhs, rhs)| {
                    for (lhs, rhs) in lhs.into_iter().zip(rhs.into_iter()) {
                        let _ = black_box(lhs / rhs);
                    }
                },
                BatchSize::SmallInput,
            );
        },
    );

    group.bench_with_input(
        BenchmarkId::new("Div owned by ref", &param),
        &field_elem,
        |b, unop_elem| {
            b.iter_batched(
                || {
                    (
                        vec![unop_elem.clone(); 10000],
                        vec![unop_elem.clone(); 10000],
                    )
                },
                |(lhs, rhs)| {
                    for (lhs, rhs) in lhs.into_iter().zip(rhs.into_iter()) {
                        let _ = black_box(lhs / &rhs);
                    }
                },
                BatchSize::SmallInput,
            );
        },
    );

    group.bench_with_input(
        BenchmarkId::new("Div ref by ref", &param),
        &field_elem,
        |b, unop_elem| {
            b.iter(|| {
                for _ in 0..10000 {
                    let _ = black_box(unop_elem / unop_elem);
                }
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("Negation", &param),
        &field_elem,
        |b, unop_elem| {
            b.iter_batched(
                || vec![unop_elem.clone(); 10000],
                |unop_elem| {
                    for x in unop_elem.into_iter() {
                        let _ = black_box(-x);
                    }
                },
                BatchSize::SmallInput,
            );
        },
    );

    let v = vec![field_elem; 10];

    group.bench_with_input(BenchmarkId::new("Sum", &param), &v, |b, v| {
        b.iter(|| {
            for _ in 0..10000 {
                let _ = black_box(F::sum(v.iter()));
            }
        });
    });

    group.bench_with_input(BenchmarkId::new("Product", &param), &v, |b, v| {
        b.iter(|| {
            for _ in 0..10000 {
                let _ = black_box(F::product(v.iter()));
            }
        });
    });
}

pub fn field_benchmarks<F>(c: &mut Criterion, name: &str, config: &F::Config)
where
    for<'a> &'a F: Add<&'a F> + Mul<&'a F> + Div<&'a F>,
    F: FromPrimitiveWithConfig + for<'a> Sum<&'a F> + for<'a> Product<&'a F>,
{
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);

    let mut group = c.benchmark_group(name);
    group.plot_config(plot_config);

    bench_random_field::<F>(&mut group, 695962179703_u64, config);
    bench_random_field::<F>(&mut group, 2345695962179703_u64, config);
    bench_random_field::<F>(&mut group, 111111111111111111_u64, config);
    bench_random_field::<F>(&mut group, 12345678124578658568_u64, config);
    group.finish();
}
