#![allow(non_local_definitions)]
#![allow(clippy::eq_op)]

use std::hint::black_box;

use criterion::{
    AxisScale, BenchmarkId, Criterion, PlotConfiguration, criterion_group, criterion_main,
};
use crypto_bigint::{Odd, modular::MontyParams};
use crypto_primitives::{FromWithConfig, crypto_bigint_monty::F256};
use itertools::Itertools;
use zinc_poly::{EvaluatablePolynomial, univariate::nat_evaluation::NatEvaluatedPoly};

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
fn bench_evaluation(group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>) {
    let field_elem = F::from_with_cfg(695962179703_u64, &bench_config());

    for i in 0..16 {
        let poly = NatEvaluatedPoly::new(
            (0..(1 << i))
                .map(|x| F::from_with_cfg(x, &bench_config()))
                .collect_vec(),
        );

        group.bench_with_input(
            BenchmarkId::new("Evaluate", format!("deg={}", 1 << i)),
            &field_elem,
            |b, field_elem| {
                b.iter(|| {
                    let _ = black_box(poly.evaluate_at_point(field_elem));
                });
            },
        );
    }
}

pub fn nat_evaluation_benchmarks(c: &mut Criterion) {
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);

    let mut group = c.benchmark_group("Natural Evaluation Domain poly");
    group.plot_config(plot_config);

    bench_evaluation(&mut group);
    group.finish();
}

criterion_group!(benches, nat_evaluation_benchmarks);
criterion_main!(benches);
