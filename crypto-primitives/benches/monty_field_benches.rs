#![allow(non_local_definitions)]
#![allow(clippy::eq_op)]

mod field_ops_bench_common;

use criterion::{Criterion, criterion_group, criterion_main};
use crypto_bigint::{Odd, modular::MontyParams};
use crypto_primitives::crypto_bigint_monty::MontyField;

use crate::field_ops_bench_common::field_benchmarks;

const LIMBS: usize = 4;

fn bench_config() -> MontyParams<LIMBS> {
    let modulus = crypto_bigint::Uint::<LIMBS>::from_be_hex(
        "0000000000000000000000000000000000860995AE68FC80E1B1BD1E39D54B33",
    );
    let modulus = Odd::new(modulus).expect("modulus should be odd");
    MontyParams::new(modulus)
}

fn monty_field_benches(c: &mut Criterion) {
    field_benchmarks::<MontyField<_>>(c, "Monty Field Arithmetic", &bench_config());
}

criterion_group!(benches, monty_field_benches);
criterion_main!(benches);
