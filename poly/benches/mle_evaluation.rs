#![allow(non_local_definitions)]
#![allow(clippy::eq_op)]

use std::hint::black_box;

use criterion::{
    AxisScale, BenchmarkId, Criterion, PlotConfiguration, criterion_group, criterion_main,
};
use crypto_bigint::{Odd, modular::MontyParams};
use crypto_primitives::{FromWithConfig, PrimeField, crypto_bigint_monty::F256};
use itertools::Itertools;
use num_traits::ConstZero;
use rand::{Rng, rng};
use zinc_poly::mle::{DenseMultilinearExtension, MultilinearExtensionWithConfig};

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
fn bench_dense_mle_evaluation(
    group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>,
) {
    let mut rng = rng();
    for i in 0..20 {
        let v = DenseMultilinearExtension::from_evaluations_vec(
            i,
            (0..(1 << i))
                .map(|i| F::from_with_cfg(i, &bench_config()))
                .collect_vec(),
            F::zero_with_cfg(&bench_config()),
        );

        let point = (0..i)
            .map(|_| F::from_with_cfg(rng.random::<u128>(), &bench_config()))
            .collect_vec();

        let zero = F::zero_with_cfg(&bench_config());

        group.bench_with_input(
            BenchmarkId::new("Evaluate", format!("nvars={}", i)),
            &v,
            |b, v| {
                b.iter(|| {
                    let _ = black_box(v.evaluate(&point, zero.clone()));
                });
            },
        );

        let v = DenseMultilinearExtension::from_evaluations_vec(
            i,
            (0..(1 << i))
                .map(|i| F::from_with_cfg(i, &bench_config()).into_inner())
                .collect_vec(),
            crypto_primitives::semiring::crypto_bigint_uint::Uint::ZERO,
        );

        let point = (0..i)
            .map(|_| F::from_with_cfg(rng.random::<u128>(), &bench_config()))
            .collect_vec();

        let config = &bench_config();
        group.bench_with_input(
            BenchmarkId::new("Evaluate With Config", format!("nvars={}", i)),
            &v,
            |b, v| {
                b.iter(|| {
                    let _ = black_box(v.evaluate_with_config(&point, config));
                });
            },
        );
    }
}

pub fn binary_poly_benchmarks(c: &mut Criterion) {
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);

    let mut group = c.benchmark_group("Mle evaluation benches");
    group.plot_config(plot_config);

    bench_dense_mle_evaluation(&mut group);
    group.finish();
}

criterion_group!(benches, binary_poly_benchmarks);
criterion_main!(benches);
