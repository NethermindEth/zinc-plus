#![allow(clippy::arithmetic_side_effects, clippy::unwrap_used)]

use criterion::{Criterion, criterion_group, criterion_main};
use crypto_bigint::{Odd, modular::MontyParams};
use crypto_primitives::{IntoWithConfig, crypto_bigint_monty::MontyField, crypto_bigint_uint::Uint};
use rand::{Rng, rng};
use rayon::prelude::*;
use std::hint::black_box;

const N: usize = 179;
const LIMBS: usize = 3; // 3 × 64 = 192 bits

type F = MontyField<LIMBS>;

/// Returns a config for a 192-bit prime field.
/// Using a 192-bit prime: 2^191 - 19 (a known safe prime).
fn bench_config() -> MontyParams<LIMBS> {
    let modulus = crypto_bigint::Uint::<LIMBS>::from_be_hex(
        "7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFED", // 2^191 - 19
    );
    let modulus = Odd::new(modulus).expect("modulus should be odd");
    MontyParams::new(modulus)
}

fn square_mod_sequential(values: &[F; N]) -> Vec<F> {
    values.iter().map(|v| v * v).collect()
}

fn square_mod_parallel(values: &[F; N]) -> Vec<F> {
    values.par_iter().map(|v| v * v).collect()
}

fn bench_square_mod(c: &mut Criterion) {
    let mut rng = rng();
    let cfg = bench_config();

    // Generate N random 192-bit field elements
    // We combine two random u128 to get full 192-bit coverage
    let values: [F; N] = std::array::from_fn(|_| {
        let lo: u128 = rng.random();
        let hi: u64 = rng.random();
        let uint = Uint::<LIMBS>::new(crypto_bigint::Uint::<LIMBS>::from_words([
            lo as u64,
            (lo >> 64) as u64,
            hi,
        ]));
        uint.into_with_cfg(&cfg)
    });

    let mut group = c.benchmark_group("Square mod q (179 x 192-bit integers)");

    group.bench_function("sequential", |b| {
        b.iter(|| black_box(square_mod_sequential(black_box(&values))))
    });

    group.bench_function("parallel (rayon)", |b| {
        b.iter(|| black_box(square_mod_parallel(black_box(&values))))
    });

    group.finish();
}

criterion_group!(benches, bench_square_mod);
criterion_main!(benches);
