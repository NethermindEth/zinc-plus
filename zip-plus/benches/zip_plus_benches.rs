#![allow(non_local_definitions)]
#![allow(clippy::eq_op, clippy::arithmetic_side_effects, clippy::unwrap_used)]

mod zip_common;

use zip_common::*;

use criterion::{Criterion, criterion_group, criterion_main};
use crypto_bigint::U64;
use crypto_primitives::{
    FixedSemiring, boolean::Boolean, crypto_bigint_int::Int, crypto_bigint_uint::Uint,
};
use std::marker::PhantomData;
use zinc_poly::univariate::{
    binary::{BinaryPoly, BinaryPolyInnerProduct},
    dense::{DensePolyInnerProduct, DensePolynomial},
};
use zinc_primality::MillerRabin;
use zinc_transcript::traits::ConstTranscribable;
use zinc_utils::{from_ref::FromRef, inner_product::MBSInnerProduct, named::Named};
use zip_plus::{
    code::{
        f2_lin_comb::{F2X128, F2X32, f2_lin_comb},
        iprs::{IprsCode, PnttConfigF65537},
        raa::{RaaCode, RaaConfig},
        raa_f2::RaaF2Code,
    },
    pcs::structs::ZipTypes,
};

const PERFORM_CHECKS: bool = if cfg!(feature = "unchecked") {
    zinc_utils::UNCHECKED
} else {
    zinc_utils::CHECKED
};

const INT_LIMBS: usize = U64::LIMBS;

#[derive(Debug, Clone)]
struct BenchZipPlusTypes<CwCoeff, const D_PLUS_ONE: usize>(PhantomData<CwCoeff>);

impl<CwCoeff, const D_PLUS_ONE: usize> ZipTypes for BenchZipPlusTypes<CwCoeff, D_PLUS_ONE>
where
    CwCoeff: ConstTranscribable
        + Copy
        + Default
        + FromRef<Boolean>
        + Named
        + FixedSemiring
        + Send
        + Sync,
    Int<5>: FromRef<CwCoeff>,
{
    const NUM_COLUMN_OPENINGS: usize = 147;
    type Eval = BinaryPoly<D_PLUS_ONE>;
    type Cw = DensePolynomial<CwCoeff, D_PLUS_ONE>;
    type Fmod = Uint<{ INT_LIMBS * 4 }>;
    type PrimeTest = MillerRabin;
    type Chal = i128;
    type Pt = i128;
    type CombR = Int<{ INT_LIMBS * 5 }>;
    type Comb = DensePolynomial<Self::CombR, D_PLUS_ONE>;
    type EvalDotChal = BinaryPolyInnerProduct<Self::Chal, D_PLUS_ONE>;
    type CombDotChal =
        DensePolyInnerProduct<Self::CombR, Self::Chal, Self::CombR, MBSInnerProduct, D_PLUS_ONE>;
    type ArrCombRDotChal = MBSInnerProduct;
}

/// `ZipTypes` for the `F_2`-RAA binary commit lane. Mirrors
/// `BenchZipPlusTypes` but uses `Cw = BinaryPoly<D_PLUS_ONE>` — the
/// codeword stays in `F_2[X]/<X^D>` (no integer widening) because
/// accumulation runs in `F_2`. `Comb`/`CombR` still live in a wide
/// integer ring; the random linear combination of codewords lifts the
/// `F_2` cells to `CombR` (multi-poly addition with integer
/// challenges does not fit back in `F_2`).
#[derive(Debug, Clone)]
struct BenchZipPlusTypesF2<const D_PLUS_ONE: usize>;

impl<const D_PLUS_ONE: usize> ZipTypes for BenchZipPlusTypesF2<D_PLUS_ONE> {
    const NUM_COLUMN_OPENINGS: usize = 147;
    type Eval = BinaryPoly<D_PLUS_ONE>;
    type Cw = BinaryPoly<D_PLUS_ONE>;
    type Fmod = Uint<{ INT_LIMBS * 4 }>;
    type PrimeTest = MillerRabin;
    type Chal = i128;
    type Pt = i128;
    type CombR = Int<{ INT_LIMBS * 5 }>;
    type Comb = DensePolynomial<Self::CombR, D_PLUS_ONE>;
    type EvalDotChal = BinaryPolyInnerProduct<Self::Chal, D_PLUS_ONE>;
    type CombDotChal =
        DensePolyInnerProduct<Self::CombR, Self::Chal, Self::CombR, MBSInnerProduct, D_PLUS_ONE>;
    type ArrCombRDotChal = MBSInnerProduct;
}

#[derive(Clone, Copy)]
struct BenchRaaConfig;
impl RaaConfig for BenchRaaConfig {
    const PERMUTE_IN_PLACE: bool = true;
    const CHECK_FOR_OVERFLOWS: bool = PERFORM_CHECKS;
}

type BenchRaaCode<const D_PLUS_ONE: usize> =
    RaaCode<BenchZipPlusTypes<i32, D_PLUS_ONE>, BenchRaaConfig, 4>;

type BenchRaaF2Code<const D_PLUS_ONE: usize> =
    RaaF2Code<BenchZipPlusTypesF2<D_PLUS_ONE>, BenchRaaConfig, 4>;

type BenchIprsCode<Twiddle, const D_PLUS_ONE: usize> =
    IprsCode<BenchZipPlusTypes<Twiddle, D_PLUS_ONE>, PnttConfigF65537, 4, PERFORM_CHECKS>;

fn zip_plus_benchmarks_raa(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip+ RAA");

    do_bench::<BenchZipPlusTypes<i32, 32>, _, PERFORM_CHECKS>(&mut group, |poly_size| {
        Some(BenchRaaCode::new(poly_size.isqrt().next_power_of_two()))
    });
    do_bench::<BenchZipPlusTypes<i32, 64>, _, PERFORM_CHECKS>(&mut group, |poly_size| {
        Some(BenchRaaCode::new(poly_size.isqrt().next_power_of_two()))
    });

    group.finish();
}

fn zip_plus_benchmarks_iprs(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip+ IPRS");

    // Use flat single-row Zip+ matrix
    do_bench::<BenchZipPlusTypes<i64, 32>, _, PERFORM_CHECKS>(&mut group, |poly_size| {
        BenchIprsCode::new_with_optimal_depth(poly_size).ok()
    });
    do_bench::<BenchZipPlusTypes<i64, 64>, _, PERFORM_CHECKS>(&mut group, |poly_size| {
        BenchIprsCode::new_with_optimal_depth(poly_size).ok()
    });

    group.finish();
}

/// Commit-only benches for the `F_2`-RAA binary lane. Prove/verify are
/// not exercised because the existing proximity test was designed for
/// integer-RAA codewords; see `BenchZipPlusTypesF2` and the
/// `RaaF2Code` notes.
fn zip_plus_benchmarks_raa_f2(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip+ RAA F_2");

    do_bench_commit_only::<BenchZipPlusTypesF2<32>, _>(&mut group, |poly_size| {
        Some(BenchRaaF2Code::new(poly_size.isqrt().next_power_of_two()))
    });
    do_bench_commit_only::<BenchZipPlusTypesF2<64>, _>(&mut group, |poly_size| {
        Some(BenchRaaF2Code::new(poly_size.isqrt().next_power_of_two()))
    });

    group.finish();
}

/// Benches the F_2[X]-coefficient linear combination that the F_2-RAA
/// prove path would compute: each opened column of an `F_2[X]<32>`-
/// valued commit matrix combined under `F_2[X]<128>` per-row
/// coefficients, producing one `F_2[X]<160>` entry per column.
///
/// Reported metrics are wall-clock time for varying `row_len` (the
/// number of column entries in a codeword row) and `num_rows` (the
/// number of rows; equals the per-row challenge count).
fn f2_lin_comb_benchmarks(c: &mut Criterion) {
    use rand::{Rng, rngs::ThreadRng};
    use std::hint::black_box;

    let mut group = c.benchmark_group("F_2 LinComb");

    // The scales mirror the F_2-RAA commit benches: rows × row_len ≈
    // poly_size, with `row_len` at the codeword length (= row_len ×
    // REP after encoding; but the lin-comb here works on a row of the
    // codeword matrix, so the relevant size is `codeword_len`).
    //
    // We use `(num_rows, row_len)` covering 2^12 .. 2^16 cells total.
    for (num_rows, row_len) in [
        (16, 1 << 8),  // 2^12 cells
        (32, 1 << 8),  // 2^13
        (64, 1 << 8),  // 2^14
        (128, 1 << 8), // 2^15
        (256, 1 << 8), // 2^16
        // Also a few "fatter" shapes:
        (4, 1 << 10), // 2^12, wide rows
        (4, 1 << 12), // 2^14, wide rows
    ] {
        let mut rng = ThreadRng::default();
        let n = num_rows * row_len;
        let cells: Vec<F2X32> = (0..n).map(|_| F2X32::from(rng.random::<u32>())).collect();
        let coeffs: Vec<F2X128> = (0..num_rows)
            .map(|_| {
                use zinc_poly::univariate::binary_f2_wide::BinaryF2Poly;
                BinaryF2Poly::from_words([rng.random::<u64>(), rng.random::<u64>()])
            })
            .collect();

        group.bench_function(
            format!("LinComb: num_rows={num_rows}, row_len={row_len}"),
            |b| {
                b.iter(|| {
                    let out = f2_lin_comb(&cells, &coeffs, row_len);
                    black_box(out);
                })
            },
        );
    }

    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(500);
    targets = zip_plus_benchmarks_raa,
              zip_plus_benchmarks_raa_f2,
              f2_lin_comb_benchmarks,
              zip_plus_benchmarks_iprs
}
criterion_main!(benches);
