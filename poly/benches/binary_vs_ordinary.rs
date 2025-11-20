#![allow(non_local_definitions)]
#![allow(clippy::eq_op)]

use std::hint::black_box;

use criterion::{
    AxisScale, BenchmarkId, Criterion, PlotConfiguration, criterion_group, criterion_main,
};
use crypto_bigint::{Odd, modular::MontyParams};
use crypto_primitives::{FromWithConfig, PrimeField, crypto_bigint_monty::F256};
use itertools::Itertools;
use zinc_poly::{
    EvaluatablePolynomial,
    univariate::{binary::BinaryPoly256, dense::DensePolynomial},
};
use zinc_utils::projectable_to_field::ProjectableToField;

const LIMBS: usize = 4;

type F = F256;

fn bench_config() -> MontyParams<LIMBS> {
    let modulus = crypto_bigint::Uint::<LIMBS>::from_be_hex(
        "0000000000000000000000000000000000860995AE68FC80E1B1BD1E39D54B33",
    );
    let modulus = Odd::new(modulus).expect("modulus should be odd");
    MontyParams::new(modulus)
}

#[allow(clippy::arithmetic_side_effects)]
fn bench_dense_poly_projection(
    group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>,
) {
    let v: Vec<DensePolynomial<u32, 32>> = (0..1024)
        .map(|i| {
            DensePolynomial::new(
                (0..32)
                    .map(|j| if i & (1 << j) == 0 { 0u32 } else { 1 })
                    .collect_vec(),
            )
        })
        .collect_vec();

    let project = DensePolynomial::prepare_projection(&F::from_with_cfg(235325, &bench_config()));

    group.bench_with_input(BenchmarkId::new("Project", "Dense version"), &v, |b, v| {
        b.iter(|| {
            for x in v {
                let _ = black_box(project(x));
            }
        });
    });

    let v: Vec<BinaryPoly256<u32>> = (0..1024).map(|i| i.into()).collect_vec();

    let project =
        BinaryPoly256::<u32>::prepare_projection(&F::from_with_cfg(235325, &bench_config()));

    group.bench_with_input(BenchmarkId::new("Project", "Binary version"), &v, |b, v| {
        b.iter(|| {
            for x in v {
                let _ = black_box(project(x));
            }
        });
    });
}

#[allow(clippy::arithmetic_side_effects)]
fn bench_dense_poly_evaluation(
    group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>,
) {
    let zero = F::zero_with_cfg(&bench_config());
    let one = F::one_with_cfg(&bench_config());
    let v: Vec<DensePolynomial<F, 32>> = (0..1024)
        .map(|i| {
            DensePolynomial::new_with_zero(
                (0..32)
                    .map(|j| {
                        if i & (1 << j) == 0 {
                            zero.clone()
                        } else {
                            one.clone()
                        }
                    })
                    .collect_vec(),
                zero.clone(),
            )
        })
        .collect_vec();

    let point = F::from_with_cfg(235325, &bench_config());

    group.bench_with_input(BenchmarkId::new("Evaluate", "Dense version"), &v, |b, v| {
        b.iter(|| {
            for x in v {
                let _ = black_box(x.evaluate_at_point(&point));
            }
        });
    });

    let v: Vec<BinaryPoly256<u32>> = (0..1024).map(|i| i.into()).collect_vec();

    group.bench_with_input(
        BenchmarkId::new("Evaluate", "Binary version"),
        &v,
        |b, v| {
            b.iter(|| {
                for x in v {
                    let _ = black_box(x.evaluate_at_point(&point));
                }
            });
        },
    );
}

pub fn binary_poly_benchmarks(c: &mut Criterion) {
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);

    let mut group = c.benchmark_group("Binary poly benches");
    group.plot_config(plot_config);

    bench_dense_poly_projection(&mut group);
    bench_dense_poly_evaluation(&mut group);
    group.finish();
}

criterion_group!(benches, binary_poly_benchmarks);
criterion_main!(benches);
