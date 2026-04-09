#![allow(non_local_definitions)]
#![allow(clippy::eq_op)]

mod field_ops_bench_common;

use criterion::{Criterion, criterion_group, criterion_main};
use crypto_bigint::{Odd, modular::BoxedMontyParams};
use crypto_primitives::crypto_bigint_boxed_monty::BoxedMontyField;

use crate::field_ops_bench_common::field_benchmarks;

fn bench_config() -> BoxedMontyParams {
    let modulus = crypto_bigint::BoxedUint::from_be_hex(
        "0000000000000000000000000000000000860995AE68FC80E1B1BD1E39D54B33",
        256,
    )
    .unwrap();
    let modulus = Odd::new(modulus).expect("modulus should be odd");
    BoxedMontyParams::new(modulus)
}

pub fn boxed_monty_field_benches(c: &mut Criterion) {
    field_benchmarks::<BoxedMontyField>(c, "Boxed Monty Field Arithmetic", &bench_config());
}

criterion_group!(benches, boxed_monty_field_benches);
criterion_main!(benches);
