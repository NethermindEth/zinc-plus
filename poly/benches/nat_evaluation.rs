#![allow(non_local_definitions)]
#![allow(clippy::eq_op)]

use std::hint::black_box;

use criterion::{
    AxisScale, BenchmarkId, Criterion, PlotConfiguration, criterion_group, criterion_main,
};
use crypto_primitives::{
    BaseFieldConfig, ProjectElementWithConfig, crypto_bigint_monty::MontyField,
    crypto_bigint_uint::Uint,
};
use itertools::Itertools;
use zinc_poly::univariate::nat_evaluation::NatEvaluatedPoly;

const LIMBS: usize = 4;

type F = MontyField<LIMBS>;

fn bench_config() -> F {
    let modulus =
        Uint::from_be_hex("0000000000000000000000000000000000860995AE68FC80E1B1BD1E39D54B33");
    F::new(&modulus).expect("modulus should be a valid odd prime")
}

#[allow(clippy::arithmetic_side_effects)]
fn bench_evaluation(group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>) {
    let cfg = bench_config();
    let field_elem = cfg.project(&695962179703_u64);

    for i in 0..16 {
        let poly = NatEvaluatedPoly::new((0..(1u64 << i)).map(|x| cfg.project(&x)).collect_vec());

        group.bench_with_input(
            BenchmarkId::new("Evaluate", format!("deg={}", 1 << i)),
            &field_elem,
            |b, field_elem| {
                b.iter(|| {
                    let _ = black_box(poly.evaluate_at_point(&cfg, field_elem));
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
