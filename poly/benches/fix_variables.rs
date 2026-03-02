//! Micro-benchmark for `fix_variables_with_config` to isolate the impact
//! of the SIMD-optimized Montgomery multiplication path.

use criterion::{
    BenchmarkId, Criterion, criterion_group, criterion_main, BatchSize,
};
use crypto_primitives::{
    Field, PrimeField, crypto_bigint_monty::MontyField, crypto_bigint_uint::Uint,
    FromWithConfig,
};
use zinc_poly::mle::{DenseMultilinearExtension, MultilinearExtensionWithConfig};

const LIMBS: usize = 3;
type F = MontyField<LIMBS>;

// Zinc+ 128-bit prime in hex
const MODULUS_HEX: &str = "000000860995ae68fc80e1b1bd1e39d54b33";

fn make_config() -> <F as PrimeField>::Config {
    let modulus = Uint::new(
        crypto_bigint::Uint::<LIMBS>::from_str_radix_vartime(MODULUS_HEX, 16).unwrap(),
    );
    F::make_cfg(&modulus).expect("Failed to create field config")
}

fn bench_fix_variables(c: &mut Criterion) {
    let mut group = c.benchmark_group("fix_variables_with_config");

    let cfg = make_config();

    // Create a challenge value (arbitrary, non-zero Montgomery form)
    let r: F = F::from_with_cfg(&0x1234_5678_ABCD_EF01u64, &cfg);

    for nvars in [14, 16, 18, 20] {
        let n = 1usize << nvars;

        // Generate pseudo-random evaluations
        let evals: Vec<<F as Field>::Inner> = (0..n)
            .map(|i| {
                F::from_with_cfg(&((i as u64).wrapping_mul(0x9E3779B97F4A7C15)), &cfg)
                    .into_inner()
            })
            .collect();

        let zero_inner = F::zero_with_cfg(&cfg).into_inner();
        let mle: DenseMultilinearExtension<<F as Field>::Inner> =
            DenseMultilinearExtension::from_evaluations_vec(nvars, evals, zero_inner);

        group.bench_with_input(
            BenchmarkId::new("MontyField<3>", nvars),
            &nvars,
            |bench, _| {
                bench.iter_batched(
                    || mle.clone(),
                    |mut m: DenseMultilinearExtension<<F as Field>::Inner>| {
                        m.fix_variables_with_config(std::slice::from_ref(&r), &cfg);
                        m
                    },
                    BatchSize::LargeInput,
                );
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_fix_variables);
criterion_main!(benches);
