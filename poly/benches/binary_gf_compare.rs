//! Benchmark: prototype `BinaryFieldGF128` vs the production
//! `BinaryFieldGF192`. Covers the primitives that dominate the F_2
//! proving path's hot loops:
//!
//! - `mul`     (single-element, throughput in scalar loop)
//! - `square`  (used heavily in inverse + Frobenius-style chains)
//! - `inverse` (Fermat chain — sanity for the worst case)
//! - `batch_mul` 1024-elt loop, exercises ILP and cache behaviour
//!
//! Plus an α-projection group sized to match the F_2 SHA-256 prover:
//!
//! - `alpha_precompute` (`α^0..α^{D-1}`, D=32; runs `D-1` mults)
//! - `project_col` (one column of 2^16 cells, D=32 bits each, XOR-only inner)
//! - `project_trace` (41 cols × 2^16 cells; matches `prove_f2_uair`'s shape)
//!
//! A full `ideal-check` GF128 port is the next step — gated on
//! implementing the `Semiring`/`Field`/`PrimeField`/`InnerTransparentField`/
//! `ConstTranscribable` trait surface on `BinaryFieldGF128`. Until that
//! lands, the α-projection numbers are the cleanest signal for the
//! field-size win on the IC side.
//!
//! Run:
//! ```sh
//! cargo bench -p zinc-poly --bench binary_gf_compare
//! ```

#![allow(non_local_definitions)]

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use rand::{Rng, SeedableRng, rngs::StdRng};
use zinc_poly::univariate::binary::BinaryPoly;
use zinc_poly::univariate::binary_gf128::{
    self as gf128, BinaryFieldGF128,
};
use zinc_poly::univariate::binary_gf192::{
    self as gf192, BinaryFieldGF192,
};

const BATCH: usize = 1024;

/// SHA-256 F_2 prove path: trace is `D × num_cols × 2^num_vars` bits.
/// `D = 32` and `num_vars = 16` (≈ 65k cells per column) is what
/// `prove_f2_uair_with_groups` runs on for the SHA F_2 UAIR.
const D: usize = 32;
const NUM_CELLS: usize = 1 << 16;
const NUM_COLS: usize = 41;

fn rand_gf128(rng: &mut StdRng) -> BinaryFieldGF128 {
    BinaryFieldGF128::from_words([rng.random(), rng.random()])
}

fn rand_gf192(rng: &mut StdRng) -> BinaryFieldGF192 {
    BinaryFieldGF192::from_words([rng.random(), rng.random(), rng.random()])
}

fn bench_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("binary_gf/mul");
    let mut rng = StdRng::seed_from_u64(0xB1B1_C5);
    let a128 = rand_gf128(&mut rng);
    let b128 = rand_gf128(&mut rng);
    let a192 = rand_gf192(&mut rng);
    let b192 = rand_gf192(&mut rng);

    group.bench_function("GF128", |bench| {
        bench.iter(|| {
            let mut x = black_box(a128);
            x *= &black_box(b128);
            black_box(x)
        });
    });
    group.bench_function("GF192", |bench| {
        bench.iter(|| {
            let mut x = black_box(a192);
            x *= &black_box(b192);
            black_box(x)
        });
    });
    group.finish();
}

fn bench_square(c: &mut Criterion) {
    let mut group = c.benchmark_group("binary_gf/square");
    let mut rng = StdRng::seed_from_u64(0xB1B1_C5);
    let a128 = rand_gf128(&mut rng);
    let a192 = rand_gf192(&mut rng);

    group.bench_function("GF128", |bench| {
        bench.iter(|| black_box(black_box(a128).square()));
    });
    group.bench_function("GF192", |bench| {
        bench.iter(|| black_box(black_box(a192).square()));
    });
    group.finish();
}

fn bench_inverse(c: &mut Criterion) {
    let mut group = c.benchmark_group("binary_gf/inverse");
    // `inverse` runs a Fermat chain (~127 sq + ~126 mul for GF128;
    // ~191 sq + ~190 mul for GF192). Drop sample count — each
    // iteration is multi-microsecond.
    group.sample_size(20);
    let mut rng = StdRng::seed_from_u64(0xB1B1_C5);
    let a128 = rand_gf128(&mut rng);
    let a192 = rand_gf192(&mut rng);

    group.bench_function("GF128", |bench| {
        bench.iter(|| black_box(black_box(a128).inverse()));
    });
    group.bench_function("GF192", |bench| {
        bench.iter(|| black_box(black_box(a192).inverse()));
    });
    group.finish();
}

fn bench_batch_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("binary_gf/batch_mul_1024");
    let mut rng = StdRng::seed_from_u64(0xB1B1_C5);
    let lhs128: Vec<BinaryFieldGF128> = (0..BATCH).map(|_| rand_gf128(&mut rng)).collect();
    let rhs128: Vec<BinaryFieldGF128> = (0..BATCH).map(|_| rand_gf128(&mut rng)).collect();
    let lhs192: Vec<BinaryFieldGF192> = (0..BATCH).map(|_| rand_gf192(&mut rng)).collect();
    let rhs192: Vec<BinaryFieldGF192> = (0..BATCH).map(|_| rand_gf192(&mut rng)).collect();

    group.throughput(criterion::Throughput::Elements(BATCH as u64));
    group.bench_function("GF128", |bench| {
        bench.iter(|| {
            let mut acc = BinaryFieldGF128::zero();
            for i in 0..BATCH {
                let mut x = lhs128[i];
                x *= &rhs128[i];
                acc += &x;
            }
            black_box(acc)
        });
    });
    group.bench_function("GF192", |bench| {
        bench.iter(|| {
            let mut acc = BinaryFieldGF192::from_words([0, 0, 0]);
            for i in 0..BATCH {
                let mut x = lhs192[i];
                x *= &rhs192[i];
                acc += &x;
            }
            black_box(acc)
        });
    });
    group.finish();
}

// -- α-projection group ----------------------------------------------
//
// The F_2 prove path projects every trace cell at a random α drawn
// from the embedding field — a per-cell `O(D)` XOR-add accumulation
// over precomputed `α^i` powers. The kernel is XOR-heavy (no
// inner-loop mults), so the speedup from GF128 vs GF192 comes from
// the smaller element (16 vs 24 bytes ↔ fewer u64 XORs per add)
// and the lighter cache pressure on the powers table.

fn rand_cell(rng: &mut StdRng) -> BinaryPoly<D> {
    BinaryPoly::<D>::from(rng.random::<u32>())
}

fn bench_alpha_precompute(c: &mut Criterion) {
    let mut group = c.benchmark_group("binary_gf/alpha_precompute");
    let mut rng = StdRng::seed_from_u64(0xA0A0_A0A0);
    let a128 = BinaryFieldGF128::from_words([rng.random(), rng.random()]);
    let a192 = BinaryFieldGF192::from_words([rng.random(), rng.random(), rng.random()]);

    group.bench_function("GF128", |bench| {
        bench.iter(|| black_box(gf128::alpha_powers(&black_box(a128), D)));
    });
    group.bench_function("GF192", |bench| {
        bench.iter(|| black_box(gf192::alpha_powers(&black_box(a192), D)));
    });
    group.finish();
}

fn bench_project_col(c: &mut Criterion) {
    let mut group = c.benchmark_group("binary_gf/project_col_65536");
    // Heavy iteration — bound it.
    group.sample_size(20);
    let mut rng = StdRng::seed_from_u64(0xA0A0_A0A0);
    let cells: Vec<BinaryPoly<D>> = (0..NUM_CELLS).map(|_| rand_cell(&mut rng)).collect();
    let a128 = BinaryFieldGF128::from_words([rng.random(), rng.random()]);
    let a192 = BinaryFieldGF192::from_words([rng.random(), rng.random(), rng.random()]);
    let pows128 = gf128::alpha_powers(&a128, D);
    let pows192 = gf192::alpha_powers(&a192, D);

    group.throughput(criterion::Throughput::Elements(NUM_CELLS as u64));
    // Branchy baseline: `if bit_set { acc += pow }` per bit.
    //
    // Accumulator XORs every per-cell result into `acc` so DCE can't
    // hoist the loop body out (the bench closure's `black_box(acc)`
    // observes a value that depends on every cell). Black-boxing the
    // cell pointer also blocks address-folding optimisations across
    // iterations.
    group.bench_function("GF128_branchy", |bench| {
        bench.iter(|| {
            let mut acc = BinaryFieldGF128::zero();
            for cell in &cells {
                acc += &gf128::eval_f2_poly_d_at_with_powers::<D>(black_box(cell), &pows128);
            }
            black_box(acc)
        });
    });
    group.bench_function("GF192_branchy", |bench| {
        bench.iter(|| {
            let mut acc = BinaryFieldGF192::from_words([0, 0, 0]);
            for cell in &cells {
                acc += &gf192::eval_f2_poly_d_at_with_powers::<D>(black_box(cell), &pows192);
            }
            black_box(acc)
        });
    });
    // Branchless: mask `pow` by `-(bit) as u64`, unconditional XOR.
    // Same inputs / inputs / outputs, no branch misprediction tax.
    group.bench_function("GF128_branchless", |bench| {
        bench.iter(|| {
            let mut acc = BinaryFieldGF128::zero();
            for cell in &cells {
                acc += &gf128::eval_f2_poly_d_at_with_powers_branchless::<D>(
                    black_box(cell),
                    &pows128,
                );
            }
            black_box(acc)
        });
    });
    group.bench_function("GF192_branchless", |bench| {
        bench.iter(|| {
            let mut acc = BinaryFieldGF192::from_words([0, 0, 0]);
            for cell in &cells {
                acc += &gf192::eval_f2_poly_d_at_with_powers_branchless::<D>(
                    black_box(cell),
                    &pows192,
                );
            }
            black_box(acc)
        });
    });
    group.finish();
}

fn bench_project_trace(c: &mut Criterion) {
    let mut group = c.benchmark_group("binary_gf/project_trace_41x65536");
    group.sample_size(10);
    let mut rng = StdRng::seed_from_u64(0xA0A0_A0A0);
    // 41 cols × 65 536 cells ≈ 2.7M cells (matches `num_vars=16` SHA F_2).
    let cols: Vec<Vec<BinaryPoly<D>>> = (0..NUM_COLS)
        .map(|_| (0..NUM_CELLS).map(|_| rand_cell(&mut rng)).collect())
        .collect();
    let a128 = BinaryFieldGF128::from_words([rng.random(), rng.random()]);
    let a192 = BinaryFieldGF192::from_words([rng.random(), rng.random(), rng.random()]);

    group.throughput(criterion::Throughput::Elements((NUM_CELLS * NUM_COLS) as u64));
    group.bench_function("GF128", |bench| {
        bench.iter(|| {
            let pows = gf128::alpha_powers(&a128, D);
            let mut acc = BinaryFieldGF128::zero();
            for col in &cols {
                for cell in col {
                    acc += &gf128::eval_f2_poly_d_at_with_powers::<D>(cell, &pows);
                }
            }
            black_box(acc)
        });
    });
    group.bench_function("GF192", |bench| {
        bench.iter(|| {
            let pows = gf192::alpha_powers(&a192, D);
            let mut acc = BinaryFieldGF192::from_words([0, 0, 0]);
            for col in &cols {
                for cell in col {
                    acc += &gf192::eval_f2_poly_d_at_with_powers::<D>(cell, &pows);
                }
            }
            black_box(acc)
        });
    });
    group.finish();
}

criterion_group! {
    name = compare;
    config = Criterion::default();
    targets =
        bench_mul,
        bench_square,
        bench_inverse,
        bench_batch_mul,
        bench_alpha_precompute,
        bench_project_col,
        bench_project_trace
}
criterion_main!(compare);
