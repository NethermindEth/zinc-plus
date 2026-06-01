//! Benchmark: the (now-production) `BinaryFieldGF128` vs the
//! deprecated `BinaryFieldGF192`. Covers the primitives that dominate the F_2
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
//! After the GF128 trait surface (`Semiring`/`Field`/`PrimeField`/
//! `InnerTransparentField`/`ConstTranscribable`) was promoted to
//! production, the F_2 prover path itself runs on GF128; this bench
//! kept its GF192 baseline so we can keep watching the field-size
//! win on the IC + α-projection side.
//!
//! Run:
//! ```sh
//! cargo bench -p zinc-poly --bench binary_gf_compare
//! ```

#![allow(non_local_definitions)]
// GF192 is the deprecated baseline this bench compares against; silence
// the deprecation warnings at the bench level.
#![allow(deprecated)]

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use rand::{Rng, SeedableRng, rngs::StdRng};
use zinc_poly::univariate::F2PackU64;
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
    // SIMD-batched 4-cell: branchless body, NEON-vectorised on aarch64.
    // One α-load shared across 4 accumulators per `i`. Beats both the
    // scalar branchy form (loop overhead amortised 4×) and the scalar
    // branchless form (NEON XOR is 1 op vs 2 scalar XORs) once the
    // batch hides the per-iteration ILP.
    group.bench_function("GF128_simd_x4", |bench| {
        bench.iter(|| {
            let mut acc = BinaryFieldGF128::zero();
            let mut chunks = cells.chunks_exact(4);
            for chunk in &mut chunks {
                let packed = [
                    chunk[0].pack_u64(),
                    chunk[1].pack_u64(),
                    chunk[2].pack_u64(),
                    chunk[3].pack_u64(),
                ];
                let r = gf128::eval_f2_poly_d_at_with_powers_simd_x4::<D>(
                    black_box(packed),
                    &pows128,
                );
                acc += &r[0];
                acc += &r[1];
                acc += &r[2];
                acc += &r[3];
            }
            for cell in chunks.remainder() {
                acc += &gf128::eval_f2_poly_d_at_with_powers::<D>(black_box(cell), &pows128);
            }
            black_box(acc)
        });
    });
    // SIMD-batched 4-cell + union-bit skip: same NEON body as `simd_x4`
    // but iterates only positions where ≥1 cell has the bit set
    // (`trailing_zeros` over `cells[0]|...|cells[3]`). Wins on low-popcount
    // cells (e.g. SHA-256 trace columns) where the union is well below
    // `D`; loses on random/high-popcount where the union saturates.
    group.bench_function("GF128_simd_x4_sparse", |bench| {
        bench.iter(|| {
            let mut acc = BinaryFieldGF128::zero();
            let mut chunks = cells.chunks_exact(4);
            for chunk in &mut chunks {
                let packed = [
                    chunk[0].pack_u64(),
                    chunk[1].pack_u64(),
                    chunk[2].pack_u64(),
                    chunk[3].pack_u64(),
                ];
                let r = gf128::eval_f2_poly_d_at_with_powers_simd_x4_sparse::<D>(
                    black_box(packed),
                    &pows128,
                );
                acc += &r[0];
                acc += &r[1];
                acc += &r[2];
                acc += &r[3];
            }
            for cell in chunks.remainder() {
                acc += &gf128::eval_f2_poly_d_at_with_powers::<D>(black_box(cell), &pows128);
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

/// Number of Hadamard relations in the SHA-256 F_2 discharge.
const K: usize = 16;

/// PHASE-0 PROBE for the small-value sum-check prover (see
/// `documentation/f2-hadamard-univariate-skip-design.md`). The discharge comb
/// at ONE sumcheck point is `eq · Σ_k γ^k Σ_b σ^b (U_{k,b}·V_{k,b} − W_{k,b})`
/// over `K=16` relations × `D=32` bits = 512 terms.
///
/// - **`bb_GF128_products`**: the standard prover's representation — *after* a
///   round-1 GF(2¹²⁸) fold the slices are general field elements, so each of the
///   512 `U_b·V_b` terms is a full GF(2¹²⁸) multiply (`bb`). ~1041 GF128 muls.
/// - **`sv_packed_psi_x4`**: the small-value prover keeps the operands as the
///   *bits* they are on the hypercube. Then `U_b·V_b − W_b = ((U_k & V_k)⊕W_k)_b`
///   (one packed `u64` AND+XOR per relation), and `Σ_b σ^b·(that)` is exactly a
///   **ψ_σ projection** of the packed word — done 4 relations at a time by the
///   existing `eval_f2_poly_d_at_with_powers_simd_x4` NEON kernel. ~16 packed
///   ops + 4 ψ-x4 calls + 17 GF128 muls.
///
/// The ratio bounds the realized `√κ` on this aarch64 target: it's the ceiling
/// of the small-value win on the per-point inner comb (the full prover adds
/// off-hypercube-grid overhead on top, which only lowers it). Gate the full
/// (byte-identical) build on this clearing ≳4×.
fn bench_discharge_comb(c: &mut Criterion) {
    let mut group = c.benchmark_group("binary_gf/discharge_comb_512term");
    group.sample_size(50);
    let mut rng = StdRng::seed_from_u64(0x5C5C_5C5C);

    let sigma = rand_gf128(&mut rng);
    let gamma = rand_gf128(&mut rng);
    let sigma_pows = gf128::alpha_powers(&sigma, D); // [1, σ, …, σ^{D-1}]
    let gamma_pows = gf128::alpha_powers(&gamma, K); // [1, γ, …, γ^{K-1}]
    let eq = rand_gf128(&mut rng);

    // bb: general (post-fold) GF128 slice values — 512 (U,V,W) triples.
    let u: Vec<BinaryFieldGF128> = (0..K * D).map(|_| rand_gf128(&mut rng)).collect();
    let v: Vec<BinaryFieldGF128> = (0..K * D).map(|_| rand_gf128(&mut rng)).collect();
    let w: Vec<BinaryFieldGF128> = (0..K * D).map(|_| rand_gf128(&mut rng)).collect();
    // sv: packed bit operands (pre-fold) — one D-bit word per relation.
    let up: Vec<u64> = (0..K).map(|_| rng.random()).collect();
    let vp: Vec<u64> = (0..K).map(|_| rng.random()).collect();
    let wp: Vec<u64> = (0..K).map(|_| rng.random()).collect();

    group.bench_function("bb_GF128_products", |bench| {
        bench.iter(|| {
            let mut acc = BinaryFieldGF128::zero();
            for k in 0..K {
                let mut s = BinaryFieldGF128::zero();
                for b in 0..D {
                    let idx = k * D + b;
                    let prod = black_box(u[idx]) * black_box(v[idx]);
                    s += &(sigma_pows[b] * (prod - black_box(w[idx])));
                }
                acc += &(gamma_pows[k] * s);
            }
            black_box(eq * acc)
        });
    });

    group.bench_function("sv_packed_psi_x4", |bench| {
        bench.iter(|| {
            let mut acc = BinaryFieldGF128::zero();
            for q in 0..(K / 4) {
                // d_k = (U_k & V_k) ^ W_k, four relations packed for the x4 kernel.
                let quad = [
                    (black_box(up[4 * q]) & black_box(vp[4 * q])) ^ black_box(wp[4 * q]),
                    (black_box(up[4 * q + 1]) & black_box(vp[4 * q + 1])) ^ black_box(wp[4 * q + 1]),
                    (black_box(up[4 * q + 2]) & black_box(vp[4 * q + 2])) ^ black_box(wp[4 * q + 2]),
                    (black_box(up[4 * q + 3]) & black_box(vp[4 * q + 3])) ^ black_box(wp[4 * q + 3]),
                ];
                let psi = gf128::eval_f2_poly_d_at_with_powers_simd_x4::<D>(quad, &sigma_pows);
                acc += &(gamma_pows[4 * q] * psi[0]);
                acc += &(gamma_pows[4 * q + 1] * psi[1]);
                acc += &(gamma_pows[4 * q + 2] * psi[2]);
                acc += &(gamma_pows[4 * q + 3] * psi[3]);
            }
            black_box(eq * acc)
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
        bench_project_trace,
        bench_discharge_comb
}
criterion_main!(compare);
