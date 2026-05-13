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
        binary_add_fft::{AddFftConfigGF2_32, BinaryAddFftCode},
        iprs::{IprsCode, PnttConfigF65537},
        raa::{RaaCode, RaaConfig},
    },
    pcs::structs::ZipTypes,
};
use zinc_utils::inner_product::ScalarProduct;

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

#[derive(Clone, Copy)]
struct BenchRaaConfig;
impl RaaConfig for BenchRaaConfig {
    const PERMUTE_IN_PLACE: bool = true;
    const CHECK_FOR_OVERFLOWS: bool = PERFORM_CHECKS;
}

type BenchRaaCode<const D_PLUS_ONE: usize> =
    RaaCode<BenchZipPlusTypes<i32, D_PLUS_ONE>, BenchRaaConfig, 4>;

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

/// ZipTypes for the binary additive-FFT bench. Mirrors
/// `BinPolyAddFftZipTypes` from `pcs::test_utils` (which is `#[cfg(test)]`
/// and not visible to benches). `D_PLUS_ONE = 32` is fixed (the additive
/// FFT operates over `GF(2^32) = F_2[X]/(f)` with `deg(f) = 32`).
#[derive(Debug, Clone)]
struct BenchAddFftZipPlusTypes<const K: usize, const M: usize>(PhantomData<()>);

impl<const K: usize, const M: usize> ZipTypes for BenchAddFftZipPlusTypes<K, M> {
    const NUM_COLUMN_OPENINGS: usize = 147;
    type Eval = BinaryPoly<32>;
    type Cw = DensePolynomial<Int<K>, 32>;
    type Fmod = Uint<{ INT_LIMBS * 4 }>;
    type PrimeTest = MillerRabin;
    type Chal = i128;
    type Pt = i128;
    type CombR = DensePolynomial<Int<M>, 32>;
    type Comb = Self::CombR;
    type EvalDotChal = ScalarProduct;
    type CombDotChal = ScalarProduct;
    type ArrCombRDotChal = MBSInnerProduct;
}

type BenchBinaryAddFftCode<const K: usize, const M: usize> =
    BinaryAddFftCode<BenchAddFftZipPlusTypes<K, M>, AddFftConfigGF2_32, 4, PERFORM_CHECKS>;

/// Variant of `BenchAddFftZipPlusTypes` using native `i32` for the
/// codeword / combined-row coefficients — the tightest single
/// primitive that's safe for our codeword bit-budget. The codeword
/// `P(v_k) = ∑_i c_i · X_i(v_k)` is a sum of at most `codeword_len`
/// products of `{0,1}`-coefficient lifted GF elements (each product
/// has per-coefficient magnitude ≤ ~200 in `Z[X]/f̃`), so the
/// worst-case grows linearly with `codeword_len`. At
/// `codeword_len = 2^16`, worst case ≈ `2^16 × 200 ≈ 2^24`, well
/// within `i32`'s `±2^31` range; empirical max at
/// `poly_size = 2^14` is `≈ 2^21` (see `column_size_report` test).
/// 4× smaller storage than `i128` and operates on native CPU 32-bit
/// adds.
#[derive(Debug, Clone)]
struct BenchAddFftI32ZipPlusTypes;

impl ZipTypes for BenchAddFftI32ZipPlusTypes {
    const NUM_COLUMN_OPENINGS: usize = 147;
    type Eval = BinaryPoly<32>;
    type Cw = DensePolynomial<i32, 32>;
    type Fmod = Uint<{ INT_LIMBS * 4 }>;
    type PrimeTest = MillerRabin;
    // i32 (vs the usual i128) — must match Cw's coefficient width so that
    // `MulByScalar<&Chal>` on `DensePolynomial<i32, 32>` resolves via the
    // existing `i32: MulByScalar<&i32, i32>` impl. Chal is only used by
    // the inner-product traits (`EvalDotChal`, etc.); the encode/commit
    // bench path doesn't actually consume it.
    type Chal = i32;
    type Pt = i32;
    type CombR = DensePolynomial<i32, 32>;
    type Comb = Self::CombR;
    type EvalDotChal = ScalarProduct;
    type CombDotChal = ScalarProduct;
    type ArrCombRDotChal = MBSInnerProduct;
}

type BenchBinaryAddFftI32Code =
    BinaryAddFftCode<BenchAddFftI32ZipPlusTypes, AddFftConfigGF2_32, 4, PERFORM_CHECKS>;

/// Bench harness for the binary additive-FFT linear code. Exercises the
/// encode-side operations only (encode_rows / encode_single_row /
/// merkle_root / commit). Prove/verify benches need an `F`-projection
/// for `CombR = DensePolynomial<Int<M>, 32>` which is not yet
/// implemented — see the plan file's "Blocked — full prove/verify
/// integration" section.
fn zip_plus_benchmarks_add_fft(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip+ AddFFT");

    // K=2 (128-bit) Cw coefficients, M=3 (192-bit) CombR coefficients.
    // Bit budget: each butterfly stage grows max-abs by a factor of at
    // most ~33, so after `m` stages the worst-case coefficient is
    // bounded by `33^m`. For `row_len = 2^16` (m = 18 with REP=4),
    // `33^18 ≈ 2^91`, which fits comfortably in `Int<2>`. K and M were
    // previously 5/8 (320/512-bit), wildly over-budget for the row
    // sizes we benchmark — the smaller widths roughly halve the per-
    // butterfly `Int<N>::add` cost.
    bench_encode_only::<BenchAddFftZipPlusTypes<2, 3>, _>(&mut group, |poly_size| {
        BenchBinaryAddFftCode::<2, 3>::new(poly_size / 4).ok()
    });

    group.finish();
}

fn zip_plus_benchmarks_add_fft_i32(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip+ AddFFT (i32)");

    // Native i32 coefficients — tight to the actual bit-budget of the
    // codeword (real max ≈ 2^24 at codeword_len = 2^16, comfortably
    // inside i32's ±2^31). 4× smaller per-coefficient storage than
    // i128, with native CPU 32-bit ALU ops.
    bench_encode_only::<BenchAddFftI32ZipPlusTypes, _>(&mut group, |poly_size| {
        BenchBinaryAddFftI32Code::new(poly_size / 4).ok()
    });

    group.finish();
}

/// Encode-only bench harness. Mirrors the encode/merkle/commit subset of
/// `do_bench` from `zip_common.rs`. Used by `zip_plus_benchmarks_add_fft`
/// because the full `do_bench` exercises prove/verify which currently
/// requires `F: FromWithConfig<&Zt::CombR>` — not available for
/// polynomial `CombR`.
fn bench_encode_only<Zt: ZipTypes, Lc: zip_plus::code::LinearCode<Zt>>(
    group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>,
    make_linear_code: impl Fn(usize) -> Option<Lc> + Copy,
) where
    rand::distr::StandardUniform:
        rand::prelude::Distribution<Zt::Eval> + rand::prelude::Distribution<Zt::Cw>,
{
    use itertools::Itertools as _;
    use zip_common::{commit, encode_rows, encode_single_row, merkle_root};

    // encode_rows::<Zt, Lc, 9>(group, make_linear_code);
    //encode_rows::<Zt, Lc, 10>(group, make_linear_code);
    // encode_rows::<Zt, Lc, 14>(group, make_linear_code);
    // encode_rows::<Zt, Lc, 15>(group, make_linear_code);
    // encode_rows::<Zt, Lc, 16>(group, make_linear_code);

    for lc in (9..=9)
        .filter_map(|row_len_ilog2| {
            let row_len = 1usize << row_len_ilog2;
            make_linear_code(row_len)
        })
        .dedup_by(|a, b| a.row_len() == b.row_len() && a.codeword_len() == b.codeword_len())
    {
        //encode_single_row::<Zt, Lc>(group, lc);
    }

    //merkle_root::<Zt, 9>(group);
    // merkle_root::<Zt, 13>(group);
    // merkle_root::<Zt, 14>(group);
    // merkle_root::<Zt, 15>(group);
    // merkle_root::<Zt, 16>(group);

    commit::<Zt, Lc, 17, 11>(group, make_linear_code);
    //commit::<Zt, Lc, 13, 1>(group, make_linear_code);
    //commit::<Zt, Lc, 14, 1>(group, make_linear_code);
    //commit::<Zt, Lc, 15, 1>(group, make_linear_code);
    //commit::<Zt, Lc, 16, 1>(group, make_linear_code);
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(500);
    targets = zip_plus_benchmarks_raa, zip_plus_benchmarks_iprs, zip_plus_benchmarks_add_fft, zip_plus_benchmarks_add_fft_i32
}
criterion_main!(benches);
