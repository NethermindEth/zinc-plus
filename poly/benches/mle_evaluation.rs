#![allow(non_local_definitions)]
#![allow(clippy::eq_op)]

use std::hint::black_box;

use criterion::{
    AxisScale, BenchmarkId, Criterion, PlotConfiguration, criterion_group, criterion_main,
};
use crypto_primitives::{
    BaseFieldConfig, ProjectElementWithConfig, SemiringConfig, crypto_bigint_monty::MontyField,
    crypto_bigint_uint::Uint,
};
use itertools::Itertools;
use rand::prelude::*;
use zinc_poly::mle::DenseMultilinearExtension;

const LIMBS: usize = 4;

type F = MontyField<LIMBS>;

fn bench_config() -> F {
    let modulus =
        Uint::from_be_hex("0000000000000000000000000000000000860995AE68FC80E1B1BD1E39D54B33");
    F::new(&modulus).expect("modulus should be a valid odd prime")
}

#[allow(clippy::arithmetic_side_effects)]
fn bench_dense_mle_evaluation(
    group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>,
) {
    let mut rng = rand::rng();
    let cfg = bench_config();
    for i in 0..20 {
        let v = DenseMultilinearExtension::from_evaluations_vec(
            i,
            (0..(1u64 << i)).map(|x| cfg.project(&x)).collect_vec(),
            cfg.zero(),
        );

        let point = (0..i)
            .map(|_| cfg.project(&rng.random::<u128>()))
            .collect_vec();

        group.bench_with_input(
            BenchmarkId::new("Evaluate With Config", format!("nvars={}", i)),
            &v,
            |b, v| {
                b.iter(|| {
                    let _ = black_box(v.clone().evaluate(&cfg, &point));
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
