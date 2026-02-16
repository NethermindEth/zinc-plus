//! Exhaustive rate-1/4 IPRS configuration benchmarks.
//!
//! For each poly_size from 2^8 to 2^11, this benchmark tests every viable
//! rate-1/4 IPRS code configuration by varying:
//!   - The twiddle field (F3329, F65537, F1179649, F7340033, F167772161)
//!   - The base matrix size (BASE_LEN ∈ {2, 4, 8, 16, 32, 64})
//!   - The recursion depth (DEPTH ∈ {1, 2, 3})
//!
//! Rate 1/4 means BASE_DIM = 4 × BASE_LEN (REPETITION_FACTOR = 4).
//!
//! Row length = BASE_LEN × 8^DEPTH. For a given poly_size = 2^P and
//! row_len, the matrix has num_rows = poly_size / row_len rows.
//!
//! The benchmark runs the full PCS pipeline (Commit, Test, Verify) for
//! BPoly<31> evaluations, as well as encoding-only benchmarks for scalar i32.
//!
//! # Available rate-1/4 row lengths
//!
//! | Config           | Field       | D=1 | D=2  | D=3   |
//! |------------------|-------------|-----|------|-------|
//! | R4B2  (F3329)    | F3329       |  16 |  —   |  —    |
//! | R4B4  (F3329)    | F3329       |  32 |  —   |  —    |
//! | R4B8  (F3329)    | F3329       |  64 |  —   |  —    |
//! | R4B16 (F65537)   | F65537      | 128 | 1024 | 8192  |
//! | R4B32 (F65537)   | F65537      | 256 | 2048 |  —    |
//! | R4B64 (F65537)   | F65537      | 512 | 4096 |  —    |
//! | R4B16 (F1179649) | F1179649    | 128 | 1024 | 8192  |
//! | R4B32 (F1179649) | F1179649    | 256 | 2048 |  —    |
//! | R4B64 (F1179649) | F1179649    | 512 | 4096 |  —    |
//! | R4B16 (F7340033) | F7340033    | 128 | 1024 | 8192  |
//! | R4B32 (F7340033) | F7340033    | 256 | 2048 |  —    |
//! | R4B64 (F7340033) | F7340033    | 512 | 4096 |  —    |
//! | R4B16 (F167M)    | F167772161  | 128 | 1024 | 8192  |
//! | R4B32 (F167M)    | F167772161  | 256 | 2048 |  —    |
//! | R4B64 (F167M)    | F167772161  | 512 | 4096 |  —    |
//!
//! For poly_size = 2^P with row_len dividing 2^P:
//!   - P=8  (256):  row_len ∈ {16, 32, 64, 128, 256}
//!   - P=9  (512):  row_len ∈ {16, 32, 64, 128, 256, 512}
//!   - P=10 (1024): row_len ∈ {16, 32, 64, 128, 256, 512, 1024}
//!   - P=11 (2048): row_len ∈ {16, 32, 64, 128, 256, 512, 1024, 2048}

#![allow(non_local_definitions)]
#![allow(clippy::eq_op, clippy::arithmetic_side_effects, clippy::unwrap_used)]

mod zip_common;

use std::marker::PhantomData;

use zinc_poly::{
    mle::{DenseMultilinearExtension, MultilinearExtensionRand},
    univariate::{
        binary::{BinaryPoly, BinaryPolyInnerProduct, BinaryPolyWideningMulByScalar},
        dense::{DensePolyInnerProduct, DensePolynomial},
    },
};
use zinc_primality::MillerRabin;
use zinc_transcript::traits::ConstTranscribable;
use zinc_utils::{UNCHECKED, from_ref::FromRef, inner_product::{MBSInnerProduct, ScalarProduct}, mul_by_scalar::ScalarWideningMulByScalar, named::Named};
use zip_common::*;

use criterion::{Criterion, criterion_group, criterion_main};
use crypto_bigint::U64;
use crypto_primitives::{
    FixedSemiring, boolean::Boolean, crypto_bigint_int::Int, crypto_bigint_uint::Uint,
};
use zip_plus::{
    code::{
        LinearCode,
        iprs::{
            IprsCode,
            // F3329 rate 1/4
            PnttConfigF3329R4B2, PnttConfigF3329R4B4, PnttConfigF3329R4B8,
            // F65537 rate 1/4
            PnttConfigF2_16R4B16, PnttConfigF2_16R4B32, PnttConfigF2_16R4B64,
            // F1179649 rate 1/4
            PnttConfigF1179649R4B16, PnttConfigF1179649R4B32, PnttConfigF1179649R4B64,
            // F7340033 rate 1/4
            PnttConfigF7340033R4B16, PnttConfigF7340033R4B32, PnttConfigF7340033R4B64,
            // F167772161 rate 1/4
            PnttConfigF167772161R4B16, PnttConfigF167772161R4B32, PnttConfigF167772161R4B64,
        },
    },
    pcs::structs::{ZipPlus, ZipTypes},
};

const INT_LIMBS: usize = U64::LIMBS;

// ==================== ZipTypes implementations ====================

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

struct BenchZipScalarTypes;

impl ZipTypes for BenchZipScalarTypes {
    const NUM_COLUMN_OPENINGS: usize = 147;
    type Eval = i32;
    type Cw = i128;
    type Fmod = Uint<{ INT_LIMBS * 4 }>;
    type PrimeTest = MillerRabin;
    type Chal = i128;
    type Pt = i128;
    type CombR = Int<{ INT_LIMBS * 3 }>;
    type Comb = Self::CombR;
    type EvalDotChal = ScalarProduct;
    type CombDotChal = ScalarProduct;
    type ArrCombRDotChal = MBSInnerProduct;
}

// ==================== Rate 1/4 code type aliases ====================

// ── F3329 rate 1/4 (small field, 12-bit modulus) ──
// R4B2 (BASE_LEN=2, BASE_DIM=8): row_len = 2 × 8^D = 16 (D=1)
type F3329R4B2<const D: usize, const CHK: bool> =
    IprsCode<BenchZipScalarTypes, PnttConfigF3329R4B2<D>, ScalarWideningMulByScalar<i128>, CHK>;
// R4B4 (BASE_LEN=4, BASE_DIM=16): row_len = 4 × 8^D = 32 (D=1)
type F3329R4B4<const D: usize, const CHK: bool> =
    IprsCode<BenchZipScalarTypes, PnttConfigF3329R4B4<D>, ScalarWideningMulByScalar<i128>, CHK>;
// R4B8 (BASE_LEN=8, BASE_DIM=32): row_len = 8 × 8^D = 64 (D=1)
type F3329R4B8<const D: usize, const CHK: bool> =
    IprsCode<BenchZipScalarTypes, PnttConfigF3329R4B8<D>, ScalarWideningMulByScalar<i128>, CHK>;

// BPoly<31> variants for F3329
type F3329R4B2Bp<Tw, const D: usize, const DP1: usize, const CHK: bool> =
    IprsCode<BenchZipPlusTypes<Tw, DP1>, PnttConfigF3329R4B2<D>, BinaryPolyWideningMulByScalar<Tw>, CHK>;
type F3329R4B4Bp<Tw, const D: usize, const DP1: usize, const CHK: bool> =
    IprsCode<BenchZipPlusTypes<Tw, DP1>, PnttConfigF3329R4B4<D>, BinaryPolyWideningMulByScalar<Tw>, CHK>;
type F3329R4B8Bp<Tw, const D: usize, const DP1: usize, const CHK: bool> =
    IprsCode<BenchZipPlusTypes<Tw, DP1>, PnttConfigF3329R4B8<D>, BinaryPolyWideningMulByScalar<Tw>, CHK>;

// ── F65537 rate 1/4 (16-bit modulus) ──
// R4B16 (BASE_LEN=16, BASE_DIM=64): row_len = 128 (D=1), 1024 (D=2), 8192 (D=3)
type F65537R4B16<const D: usize, const CHK: bool> =
    IprsCode<BenchZipScalarTypes, PnttConfigF2_16R4B16<D>, ScalarWideningMulByScalar<i128>, CHK>;
// R4B32 (BASE_LEN=32, BASE_DIM=128): row_len = 256 (D=1), 2048 (D=2)
type F65537R4B32<const D: usize, const CHK: bool> =
    IprsCode<BenchZipScalarTypes, PnttConfigF2_16R4B32<D>, ScalarWideningMulByScalar<i128>, CHK>;
// R4B64 (BASE_LEN=64, BASE_DIM=256): row_len = 512 (D=1), 4096 (D=2)
type F65537R4B64<const D: usize, const CHK: bool> =
    IprsCode<BenchZipScalarTypes, PnttConfigF2_16R4B64<D>, ScalarWideningMulByScalar<i128>, CHK>;

// BPoly<31> variants for F65537
type F65537R4B16Bp<Tw, const D: usize, const DP1: usize, const CHK: bool> =
    IprsCode<BenchZipPlusTypes<Tw, DP1>, PnttConfigF2_16R4B16<D>, BinaryPolyWideningMulByScalar<Tw>, CHK>;
type F65537R4B32Bp<Tw, const D: usize, const DP1: usize, const CHK: bool> =
    IprsCode<BenchZipPlusTypes<Tw, DP1>, PnttConfigF2_16R4B32<D>, BinaryPolyWideningMulByScalar<Tw>, CHK>;
type F65537R4B64Bp<Tw, const D: usize, const DP1: usize, const CHK: bool> =
    IprsCode<BenchZipPlusTypes<Tw, DP1>, PnttConfigF2_16R4B64<D>, BinaryPolyWideningMulByScalar<Tw>, CHK>;

// ── F1179649 rate 1/4 (21-bit modulus) ──
// R4B16 (BASE_LEN=16, BASE_DIM=64): row_len = 128 (D=1), 1024 (D=2), 8192 (D=3)
type F1179649R4B16<const D: usize, const CHK: bool> =
    IprsCode<BenchZipScalarTypes, PnttConfigF1179649R4B16<D>, ScalarWideningMulByScalar<i128>, CHK>;
// R4B32 (BASE_LEN=32, BASE_DIM=128): row_len = 256 (D=1), 2048 (D=2)
type F1179649R4B32<const D: usize, const CHK: bool> =
    IprsCode<BenchZipScalarTypes, PnttConfigF1179649R4B32<D>, ScalarWideningMulByScalar<i128>, CHK>;
// R4B64 (BASE_LEN=64, BASE_DIM=256): row_len = 512 (D=1), 4096 (D=2)
type F1179649R4B64<const D: usize, const CHK: bool> =
    IprsCode<BenchZipScalarTypes, PnttConfigF1179649R4B64<D>, ScalarWideningMulByScalar<i128>, CHK>;

// BPoly<31> variants for F1179649
type F1179649R4B16Bp<Tw, const D: usize, const DP1: usize, const CHK: bool> =
    IprsCode<BenchZipPlusTypes<Tw, DP1>, PnttConfigF1179649R4B16<D>, BinaryPolyWideningMulByScalar<Tw>, CHK>;
type F1179649R4B32Bp<Tw, const D: usize, const DP1: usize, const CHK: bool> =
    IprsCode<BenchZipPlusTypes<Tw, DP1>, PnttConfigF1179649R4B32<D>, BinaryPolyWideningMulByScalar<Tw>, CHK>;
type F1179649R4B64Bp<Tw, const D: usize, const DP1: usize, const CHK: bool> =
    IprsCode<BenchZipPlusTypes<Tw, DP1>, PnttConfigF1179649R4B64<D>, BinaryPolyWideningMulByScalar<Tw>, CHK>;

// ── F7340033 rate 1/4 (23-bit modulus) ──
// R4B16 (BASE_LEN=16, BASE_DIM=64): row_len = 128 (D=1), 1024 (D=2), 8192 (D=3)
type F7340033R4B16<const D: usize, const CHK: bool> =
    IprsCode<BenchZipScalarTypes, PnttConfigF7340033R4B16<D>, ScalarWideningMulByScalar<i128>, CHK>;
// R4B32 (BASE_LEN=32, BASE_DIM=128): row_len = 256 (D=1), 2048 (D=2)
type F7340033R4B32<const D: usize, const CHK: bool> =
    IprsCode<BenchZipScalarTypes, PnttConfigF7340033R4B32<D>, ScalarWideningMulByScalar<i128>, CHK>;
// R4B64 (BASE_LEN=64, BASE_DIM=256): row_len = 512 (D=1), 4096 (D=2)
type F7340033R4B64<const D: usize, const CHK: bool> =
    IprsCode<BenchZipScalarTypes, PnttConfigF7340033R4B64<D>, ScalarWideningMulByScalar<i128>, CHK>;

// BPoly<31> variants for F7340033
type F7340033R4B16Bp<Tw, const D: usize, const DP1: usize, const CHK: bool> =
    IprsCode<BenchZipPlusTypes<Tw, DP1>, PnttConfigF7340033R4B16<D>, BinaryPolyWideningMulByScalar<Tw>, CHK>;
type F7340033R4B32Bp<Tw, const D: usize, const DP1: usize, const CHK: bool> =
    IprsCode<BenchZipPlusTypes<Tw, DP1>, PnttConfigF7340033R4B32<D>, BinaryPolyWideningMulByScalar<Tw>, CHK>;
type F7340033R4B64Bp<Tw, const D: usize, const DP1: usize, const CHK: bool> =
    IprsCode<BenchZipPlusTypes<Tw, DP1>, PnttConfigF7340033R4B64<D>, BinaryPolyWideningMulByScalar<Tw>, CHK>;

// ── F167772161 rate 1/4 (28-bit modulus) ──
// R4B16 (BASE_LEN=16, BASE_DIM=64): row_len = 128 (D=1), 1024 (D=2), 8192 (D=3)
type F167772161R4B16<const D: usize, const CHK: bool> =
    IprsCode<BenchZipScalarTypes, PnttConfigF167772161R4B16<D>, ScalarWideningMulByScalar<i128>, CHK>;
// R4B32 (BASE_LEN=32, BASE_DIM=128): row_len = 256 (D=1), 2048 (D=2)
type F167772161R4B32<const D: usize, const CHK: bool> =
    IprsCode<BenchZipScalarTypes, PnttConfigF167772161R4B32<D>, ScalarWideningMulByScalar<i128>, CHK>;
// R4B64 (BASE_LEN=64, BASE_DIM=256): row_len = 512 (D=1), 4096 (D=2)
type F167772161R4B64<const D: usize, const CHK: bool> =
    IprsCode<BenchZipScalarTypes, PnttConfigF167772161R4B64<D>, ScalarWideningMulByScalar<i128>, CHK>;

// BPoly<31> variants for F167772161
type F167772161R4B16Bp<Tw, const D: usize, const DP1: usize, const CHK: bool> =
    IprsCode<BenchZipPlusTypes<Tw, DP1>, PnttConfigF167772161R4B16<D>, BinaryPolyWideningMulByScalar<Tw>, CHK>;
type F167772161R4B32Bp<Tw, const D: usize, const DP1: usize, const CHK: bool> =
    IprsCode<BenchZipPlusTypes<Tw, DP1>, PnttConfigF167772161R4B32<D>, BinaryPolyWideningMulByScalar<Tw>, CHK>;
type F167772161R4B64Bp<Tw, const D: usize, const DP1: usize, const CHK: bool> =
    IprsCode<BenchZipPlusTypes<Tw, DP1>, PnttConfigF167772161R4B64<D>, BinaryPolyWideningMulByScalar<Tw>, CHK>;

// ==================== Encoding benchmarks (scalar i32) ====================

/// Encoding-only benchmark for all rate-1/4 configs at each target row_len.
/// This measures pure encoding time without the PCS overhead.
fn r4_encode_config_search_i32(c: &mut Criterion) {
    let mut group = c.benchmark_group("R4 Encode i32 Config Search");
    group.sample_size(20);

    // ── row_len = 16 (F3329 R4B2 D=1) ──
    encode_single_row::<BenchZipScalarTypes, F3329R4B2<1, UNCHECKED>, 16>(&mut group, "R4-F3329-B2-D1");

    // ── row_len = 32 (F3329 R4B4 D=1) ──
    encode_single_row::<BenchZipScalarTypes, F3329R4B4<1, UNCHECKED>, 32>(&mut group, "R4-F3329-B4-D1");

    // ── row_len = 64 (F3329 R4B8 D=1) ──
    encode_single_row::<BenchZipScalarTypes, F3329R4B8<1, UNCHECKED>, 64>(&mut group, "R4-F3329-B8-D1");

    // ── row_len = 128 (R4B16, D=1) ── 4 field options
    encode_single_row::<BenchZipScalarTypes, F65537R4B16<1, UNCHECKED>,    128>(&mut group, "R4-F65537-B16-D1");
    encode_single_row::<BenchZipScalarTypes, F1179649R4B16<1, UNCHECKED>,  128>(&mut group, "R4-F1179649-B16-D1");
    encode_single_row::<BenchZipScalarTypes, F7340033R4B16<1, UNCHECKED>,  128>(&mut group, "R4-F7340033-B16-D1");
    encode_single_row::<BenchZipScalarTypes, F167772161R4B16<1, UNCHECKED>,128>(&mut group, "R4-F167772161-B16-D1");

    // ── row_len = 256 (R4B32, D=1) ── 4 field options
    encode_single_row::<BenchZipScalarTypes, F65537R4B32<1, UNCHECKED>,    256>(&mut group, "R4-F65537-B32-D1");
    encode_single_row::<BenchZipScalarTypes, F1179649R4B32<1, UNCHECKED>,  256>(&mut group, "R4-F1179649-B32-D1");
    encode_single_row::<BenchZipScalarTypes, F7340033R4B32<1, UNCHECKED>,  256>(&mut group, "R4-F7340033-B32-D1");
    encode_single_row::<BenchZipScalarTypes, F167772161R4B32<1, UNCHECKED>,256>(&mut group, "R4-F167772161-B32-D1");

    // ── row_len = 512 (R4B64, D=1) ── 4 field options
    encode_single_row::<BenchZipScalarTypes, F65537R4B64<1, UNCHECKED>,    512>(&mut group, "R4-F65537-B64-D1");
    encode_single_row::<BenchZipScalarTypes, F1179649R4B64<1, UNCHECKED>,  512>(&mut group, "R4-F1179649-B64-D1");
    encode_single_row::<BenchZipScalarTypes, F7340033R4B64<1, UNCHECKED>,  512>(&mut group, "R4-F7340033-B64-D1");
    encode_single_row::<BenchZipScalarTypes, F167772161R4B64<1, UNCHECKED>,512>(&mut group, "R4-F167772161-B64-D1");

    // ── row_len = 1024 (R4B16, D=2) ── 4 field options
    encode_single_row::<BenchZipScalarTypes, F65537R4B16<2, UNCHECKED>,    1024>(&mut group, "R4-F65537-B16-D2");
    encode_single_row::<BenchZipScalarTypes, F1179649R4B16<2, UNCHECKED>,  1024>(&mut group, "R4-F1179649-B16-D2");
    encode_single_row::<BenchZipScalarTypes, F7340033R4B16<2, UNCHECKED>,  1024>(&mut group, "R4-F7340033-B16-D2");
    encode_single_row::<BenchZipScalarTypes, F167772161R4B16<2, UNCHECKED>,1024>(&mut group, "R4-F167772161-B16-D2");

    // ── row_len = 2048 (R4B32, D=2) ── 4 field options
    encode_single_row::<BenchZipScalarTypes, F65537R4B32<2, UNCHECKED>,    2048>(&mut group, "R4-F65537-B32-D2");
    encode_single_row::<BenchZipScalarTypes, F1179649R4B32<2, UNCHECKED>,  2048>(&mut group, "R4-F1179649-B32-D2");
    encode_single_row::<BenchZipScalarTypes, F7340033R4B32<2, UNCHECKED>,  2048>(&mut group, "R4-F7340033-B32-D2");
    encode_single_row::<BenchZipScalarTypes, F167772161R4B32<2, UNCHECKED>,2048>(&mut group, "R4-F167772161-B32-D2");

    group.finish();
}

/// Encoding-only benchmark for BPoly<31> at each target row_len.
fn r4_encode_config_search_bpoly(c: &mut Criterion) {
    let mut group = c.benchmark_group("R4 Encode BPoly31 Config Search");
    group.sample_size(20);

    // ── row_len = 16 ──
    encode_single_row::<BenchZipPlusTypes<i128, 32>, F3329R4B2Bp<i128, 1, 32, UNCHECKED>, 16>(&mut group, "R4-F3329-B2-D1");

    // ── row_len = 32 ──
    encode_single_row::<BenchZipPlusTypes<i128, 32>, F3329R4B4Bp<i128, 1, 32, UNCHECKED>, 32>(&mut group, "R4-F3329-B4-D1");

    // ── row_len = 64 ──
    encode_single_row::<BenchZipPlusTypes<i128, 32>, F3329R4B8Bp<i128, 1, 32, UNCHECKED>, 64>(&mut group, "R4-F3329-B8-D1");

    // ── row_len = 128 ──
    encode_single_row::<BenchZipPlusTypes<i128, 32>, F65537R4B16Bp<i128, 1, 32, UNCHECKED>,    128>(&mut group, "R4-F65537-B16-D1");
    encode_single_row::<BenchZipPlusTypes<i128, 32>, F1179649R4B16Bp<i128, 1, 32, UNCHECKED>,  128>(&mut group, "R4-F1179649-B16-D1");
    encode_single_row::<BenchZipPlusTypes<i128, 32>, F7340033R4B16Bp<i128, 1, 32, UNCHECKED>,  128>(&mut group, "R4-F7340033-B16-D1");
    encode_single_row::<BenchZipPlusTypes<i128, 32>, F167772161R4B16Bp<i128, 1, 32, UNCHECKED>,128>(&mut group, "R4-F167772161-B16-D1");

    // ── row_len = 256 ──
    encode_single_row::<BenchZipPlusTypes<i128, 32>, F65537R4B32Bp<i128, 1, 32, UNCHECKED>,    256>(&mut group, "R4-F65537-B32-D1");
    encode_single_row::<BenchZipPlusTypes<i128, 32>, F1179649R4B32Bp<i128, 1, 32, UNCHECKED>,  256>(&mut group, "R4-F1179649-B32-D1");
    encode_single_row::<BenchZipPlusTypes<i128, 32>, F7340033R4B32Bp<i128, 1, 32, UNCHECKED>,  256>(&mut group, "R4-F7340033-B32-D1");
    encode_single_row::<BenchZipPlusTypes<i128, 32>, F167772161R4B32Bp<i128, 1, 32, UNCHECKED>,256>(&mut group, "R4-F167772161-B32-D1");

    // ── row_len = 512 ──
    encode_single_row::<BenchZipPlusTypes<i128, 32>, F65537R4B64Bp<i128, 1, 32, UNCHECKED>,    512>(&mut group, "R4-F65537-B64-D1");
    encode_single_row::<BenchZipPlusTypes<i128, 32>, F1179649R4B64Bp<i128, 1, 32, UNCHECKED>,  512>(&mut group, "R4-F1179649-B64-D1");
    encode_single_row::<BenchZipPlusTypes<i128, 32>, F7340033R4B64Bp<i128, 1, 32, UNCHECKED>,  512>(&mut group, "R4-F7340033-B64-D1");
    encode_single_row::<BenchZipPlusTypes<i128, 32>, F167772161R4B64Bp<i128, 1, 32, UNCHECKED>,512>(&mut group, "R4-F167772161-B64-D1");

    // ── row_len = 1024 ──
    encode_single_row::<BenchZipPlusTypes<i128, 32>, F65537R4B16Bp<i128, 2, 32, UNCHECKED>,    1024>(&mut group, "R4-F65537-B16-D2");
    encode_single_row::<BenchZipPlusTypes<i128, 32>, F1179649R4B16Bp<i128, 2, 32, UNCHECKED>,  1024>(&mut group, "R4-F1179649-B16-D2");
    encode_single_row::<BenchZipPlusTypes<i128, 32>, F7340033R4B16Bp<i128, 2, 32, UNCHECKED>,  1024>(&mut group, "R4-F7340033-B16-D2");
    encode_single_row::<BenchZipPlusTypes<i128, 32>, F167772161R4B16Bp<i128, 2, 32, UNCHECKED>,1024>(&mut group, "R4-F167772161-B16-D2");

    // ── row_len = 2048 ──
    encode_single_row::<BenchZipPlusTypes<i128, 32>, F65537R4B32Bp<i128, 2, 32, UNCHECKED>,    2048>(&mut group, "R4-F65537-B32-D2");
    encode_single_row::<BenchZipPlusTypes<i128, 32>, F1179649R4B32Bp<i128, 2, 32, UNCHECKED>,  2048>(&mut group, "R4-F1179649-B32-D2");
    encode_single_row::<BenchZipPlusTypes<i128, 32>, F7340033R4B32Bp<i128, 2, 32, UNCHECKED>,  2048>(&mut group, "R4-F7340033-B32-D2");
    encode_single_row::<BenchZipPlusTypes<i128, 32>, F167772161R4B32Bp<i128, 2, 32, UNCHECKED>,2048>(&mut group, "R4-F167772161-B32-D2");

    group.finish();
}

// ==================== PCS Pipeline benchmarks ====================

/// Full PCS pipeline (Commit, Test, Verify) for poly_size = 2^8 = 256.
///
/// All rate-1/4 configs where row_len divides 256:
///   row=16  (F3329 R4B2 D=1):   num_rows = 16
///   row=32  (F3329 R4B4 D=1):   num_rows = 8
///   row=64  (F3329 R4B8 D=1):   num_rows = 4
///   row=128 (R4B16 D=1):        num_rows = 2  [F65537, F1179649, F7340033, F167772161]
///   row=256 (R4B32 D=1):        num_rows = 1  [F65537, F1179649, F7340033, F167772161]
fn r4_pcs_pipeline_p8(c: &mut Criterion) {
    let mut group = c.benchmark_group("R4 PCS P=8");
    group.sample_size(10);

    // ── Commit ──
    // F3329
    commit_named::<BenchZipPlusTypes<i128, 32>, F3329R4B2Bp<i128, 1, 32, UNCHECKED>, 8>(&mut group, "F3329-R4B2-D1");
    commit_named::<BenchZipPlusTypes<i128, 32>, F3329R4B4Bp<i128, 1, 32, UNCHECKED>, 8>(&mut group, "F3329-R4B4-D1");
    commit_named::<BenchZipPlusTypes<i128, 32>, F3329R4B8Bp<i128, 1, 32, UNCHECKED>, 8>(&mut group, "F3329-R4B8-D1");
    // F65537
    commit_named::<BenchZipPlusTypes<i128, 32>, F65537R4B16Bp<i128, 1, 32, UNCHECKED>, 8>(&mut group, "F65537-R4B16-D1");
    commit_named::<BenchZipPlusTypes<i128, 32>, F65537R4B32Bp<i128, 1, 32, UNCHECKED>, 8>(&mut group, "F65537-R4B32-D1");
    // F1179649
    commit_named::<BenchZipPlusTypes<i128, 32>, F1179649R4B16Bp<i128, 1, 32, UNCHECKED>, 8>(&mut group, "F1179649-R4B16-D1");
    commit_named::<BenchZipPlusTypes<i128, 32>, F1179649R4B32Bp<i128, 1, 32, UNCHECKED>, 8>(&mut group, "F1179649-R4B32-D1");
    // F7340033
    commit_named::<BenchZipPlusTypes<i128, 32>, F7340033R4B16Bp<i128, 1, 32, UNCHECKED>, 8>(&mut group, "F7340033-R4B16-D1");
    commit_named::<BenchZipPlusTypes<i128, 32>, F7340033R4B32Bp<i128, 1, 32, UNCHECKED>, 8>(&mut group, "F7340033-R4B32-D1");
    // F167772161
    commit_named::<BenchZipPlusTypes<i128, 32>, F167772161R4B16Bp<i128, 1, 32, UNCHECKED>, 8>(&mut group, "F167772161-R4B16-D1");
    commit_named::<BenchZipPlusTypes<i128, 32>, F167772161R4B32Bp<i128, 1, 32, UNCHECKED>, 8>(&mut group, "F167772161-R4B32-D1");

    // ── Test ──
    test_named::<BenchZipPlusTypes<i128, 32>, F3329R4B2Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 8>(&mut group, "F3329-R4B2-D1");
    test_named::<BenchZipPlusTypes<i128, 32>, F3329R4B4Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 8>(&mut group, "F3329-R4B4-D1");
    test_named::<BenchZipPlusTypes<i128, 32>, F3329R4B8Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 8>(&mut group, "F3329-R4B8-D1");
    test_named::<BenchZipPlusTypes<i128, 32>, F65537R4B16Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 8>(&mut group, "F65537-R4B16-D1");
    test_named::<BenchZipPlusTypes<i128, 32>, F65537R4B32Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 8>(&mut group, "F65537-R4B32-D1");
    test_named::<BenchZipPlusTypes<i128, 32>, F1179649R4B16Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 8>(&mut group, "F1179649-R4B16-D1");
    test_named::<BenchZipPlusTypes<i128, 32>, F1179649R4B32Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 8>(&mut group, "F1179649-R4B32-D1");
    test_named::<BenchZipPlusTypes<i128, 32>, F7340033R4B16Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 8>(&mut group, "F7340033-R4B16-D1");
    test_named::<BenchZipPlusTypes<i128, 32>, F7340033R4B32Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 8>(&mut group, "F7340033-R4B32-D1");
    test_named::<BenchZipPlusTypes<i128, 32>, F167772161R4B16Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 8>(&mut group, "F167772161-R4B16-D1");
    test_named::<BenchZipPlusTypes<i128, 32>, F167772161R4B32Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 8>(&mut group, "F167772161-R4B32-D1");

    // ── Verify ──
    verify_named::<BenchZipPlusTypes<i128, 32>, F3329R4B2Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 8>(&mut group, "F3329-R4B2-D1");
    verify_named::<BenchZipPlusTypes<i128, 32>, F3329R4B4Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 8>(&mut group, "F3329-R4B4-D1");
    verify_named::<BenchZipPlusTypes<i128, 32>, F3329R4B8Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 8>(&mut group, "F3329-R4B8-D1");
    verify_named::<BenchZipPlusTypes<i128, 32>, F65537R4B16Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 8>(&mut group, "F65537-R4B16-D1");
    verify_named::<BenchZipPlusTypes<i128, 32>, F65537R4B32Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 8>(&mut group, "F65537-R4B32-D1");
    verify_named::<BenchZipPlusTypes<i128, 32>, F1179649R4B16Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 8>(&mut group, "F1179649-R4B16-D1");
    verify_named::<BenchZipPlusTypes<i128, 32>, F1179649R4B32Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 8>(&mut group, "F1179649-R4B32-D1");
    verify_named::<BenchZipPlusTypes<i128, 32>, F7340033R4B16Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 8>(&mut group, "F7340033-R4B16-D1");
    verify_named::<BenchZipPlusTypes<i128, 32>, F7340033R4B32Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 8>(&mut group, "F7340033-R4B32-D1");
    verify_named::<BenchZipPlusTypes<i128, 32>, F167772161R4B16Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 8>(&mut group, "F167772161-R4B16-D1");
    verify_named::<BenchZipPlusTypes<i128, 32>, F167772161R4B32Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 8>(&mut group, "F167772161-R4B32-D1");

    group.finish();
}

/// Full PCS pipeline for poly_size = 2^9 = 512.
///
/// All rate-1/4 configs where row_len divides 512:
///   row=16  (F3329 R4B2 D=1):   num_rows = 32
///   row=32  (F3329 R4B4 D=1):   num_rows = 16
///   row=64  (F3329 R4B8 D=1):   num_rows = 8
///   row=128 (R4B16 D=1):        num_rows = 4   [F65537, F1179649, F7340033, F167772161]
///   row=256 (R4B32 D=1):        num_rows = 2   [F65537, F1179649, F7340033, F167772161]
///   row=512 (R4B64 D=1):        num_rows = 1   [F65537, F1179649, F7340033, F167772161]
fn r4_pcs_pipeline_p9(c: &mut Criterion) {
    let mut group = c.benchmark_group("R4 PCS P=9");
    group.sample_size(10);

    // ── Commit ──
    commit_named::<BenchZipPlusTypes<i128, 32>, F3329R4B2Bp<i128, 1, 32, UNCHECKED>, 9>(&mut group, "F3329-R4B2-D1");
    commit_named::<BenchZipPlusTypes<i128, 32>, F3329R4B4Bp<i128, 1, 32, UNCHECKED>, 9>(&mut group, "F3329-R4B4-D1");
    commit_named::<BenchZipPlusTypes<i128, 32>, F3329R4B8Bp<i128, 1, 32, UNCHECKED>, 9>(&mut group, "F3329-R4B8-D1");
    commit_named::<BenchZipPlusTypes<i128, 32>, F65537R4B16Bp<i128, 1, 32, UNCHECKED>, 9>(&mut group, "F65537-R4B16-D1");
    commit_named::<BenchZipPlusTypes<i128, 32>, F65537R4B32Bp<i128, 1, 32, UNCHECKED>, 9>(&mut group, "F65537-R4B32-D1");
    commit_named::<BenchZipPlusTypes<i128, 32>, F65537R4B64Bp<i128, 1, 32, UNCHECKED>, 9>(&mut group, "F65537-R4B64-D1");
    commit_named::<BenchZipPlusTypes<i128, 32>, F1179649R4B16Bp<i128, 1, 32, UNCHECKED>, 9>(&mut group, "F1179649-R4B16-D1");
    commit_named::<BenchZipPlusTypes<i128, 32>, F1179649R4B32Bp<i128, 1, 32, UNCHECKED>, 9>(&mut group, "F1179649-R4B32-D1");
    commit_named::<BenchZipPlusTypes<i128, 32>, F1179649R4B64Bp<i128, 1, 32, UNCHECKED>, 9>(&mut group, "F1179649-R4B64-D1");
    commit_named::<BenchZipPlusTypes<i128, 32>, F7340033R4B16Bp<i128, 1, 32, UNCHECKED>, 9>(&mut group, "F7340033-R4B16-D1");
    commit_named::<BenchZipPlusTypes<i128, 32>, F7340033R4B32Bp<i128, 1, 32, UNCHECKED>, 9>(&mut group, "F7340033-R4B32-D1");
    commit_named::<BenchZipPlusTypes<i128, 32>, F7340033R4B64Bp<i128, 1, 32, UNCHECKED>, 9>(&mut group, "F7340033-R4B64-D1");
    commit_named::<BenchZipPlusTypes<i128, 32>, F167772161R4B16Bp<i128, 1, 32, UNCHECKED>, 9>(&mut group, "F167772161-R4B16-D1");
    commit_named::<BenchZipPlusTypes<i128, 32>, F167772161R4B32Bp<i128, 1, 32, UNCHECKED>, 9>(&mut group, "F167772161-R4B32-D1");
    commit_named::<BenchZipPlusTypes<i128, 32>, F167772161R4B64Bp<i128, 1, 32, UNCHECKED>, 9>(&mut group, "F167772161-R4B64-D1");

    // ── Test ──
    test_named::<BenchZipPlusTypes<i128, 32>, F3329R4B2Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 9>(&mut group, "F3329-R4B2-D1");
    test_named::<BenchZipPlusTypes<i128, 32>, F3329R4B4Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 9>(&mut group, "F3329-R4B4-D1");
    test_named::<BenchZipPlusTypes<i128, 32>, F3329R4B8Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 9>(&mut group, "F3329-R4B8-D1");
    test_named::<BenchZipPlusTypes<i128, 32>, F65537R4B16Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 9>(&mut group, "F65537-R4B16-D1");
    test_named::<BenchZipPlusTypes<i128, 32>, F65537R4B32Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 9>(&mut group, "F65537-R4B32-D1");
    test_named::<BenchZipPlusTypes<i128, 32>, F65537R4B64Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 9>(&mut group, "F65537-R4B64-D1");
    test_named::<BenchZipPlusTypes<i128, 32>, F1179649R4B16Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 9>(&mut group, "F1179649-R4B16-D1");
    test_named::<BenchZipPlusTypes<i128, 32>, F1179649R4B32Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 9>(&mut group, "F1179649-R4B32-D1");
    test_named::<BenchZipPlusTypes<i128, 32>, F1179649R4B64Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 9>(&mut group, "F1179649-R4B64-D1");
    test_named::<BenchZipPlusTypes<i128, 32>, F7340033R4B16Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 9>(&mut group, "F7340033-R4B16-D1");
    test_named::<BenchZipPlusTypes<i128, 32>, F7340033R4B32Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 9>(&mut group, "F7340033-R4B32-D1");
    test_named::<BenchZipPlusTypes<i128, 32>, F7340033R4B64Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 9>(&mut group, "F7340033-R4B64-D1");
    test_named::<BenchZipPlusTypes<i128, 32>, F167772161R4B16Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 9>(&mut group, "F167772161-R4B16-D1");
    test_named::<BenchZipPlusTypes<i128, 32>, F167772161R4B32Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 9>(&mut group, "F167772161-R4B32-D1");
    test_named::<BenchZipPlusTypes<i128, 32>, F167772161R4B64Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 9>(&mut group, "F167772161-R4B64-D1");

    // ── Verify ──
    verify_named::<BenchZipPlusTypes<i128, 32>, F3329R4B2Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 9>(&mut group, "F3329-R4B2-D1");
    verify_named::<BenchZipPlusTypes<i128, 32>, F3329R4B4Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 9>(&mut group, "F3329-R4B4-D1");
    verify_named::<BenchZipPlusTypes<i128, 32>, F3329R4B8Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 9>(&mut group, "F3329-R4B8-D1");
    verify_named::<BenchZipPlusTypes<i128, 32>, F65537R4B16Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 9>(&mut group, "F65537-R4B16-D1");
    verify_named::<BenchZipPlusTypes<i128, 32>, F65537R4B32Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 9>(&mut group, "F65537-R4B32-D1");
    verify_named::<BenchZipPlusTypes<i128, 32>, F65537R4B64Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 9>(&mut group, "F65537-R4B64-D1");
    verify_named::<BenchZipPlusTypes<i128, 32>, F1179649R4B16Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 9>(&mut group, "F1179649-R4B16-D1");
    verify_named::<BenchZipPlusTypes<i128, 32>, F1179649R4B32Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 9>(&mut group, "F1179649-R4B32-D1");
    verify_named::<BenchZipPlusTypes<i128, 32>, F1179649R4B64Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 9>(&mut group, "F1179649-R4B64-D1");
    verify_named::<BenchZipPlusTypes<i128, 32>, F7340033R4B16Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 9>(&mut group, "F7340033-R4B16-D1");
    verify_named::<BenchZipPlusTypes<i128, 32>, F7340033R4B32Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 9>(&mut group, "F7340033-R4B32-D1");
    verify_named::<BenchZipPlusTypes<i128, 32>, F7340033R4B64Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 9>(&mut group, "F7340033-R4B64-D1");
    verify_named::<BenchZipPlusTypes<i128, 32>, F167772161R4B16Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 9>(&mut group, "F167772161-R4B16-D1");
    verify_named::<BenchZipPlusTypes<i128, 32>, F167772161R4B32Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 9>(&mut group, "F167772161-R4B32-D1");
    verify_named::<BenchZipPlusTypes<i128, 32>, F167772161R4B64Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 9>(&mut group, "F167772161-R4B64-D1");

    group.finish();
}

/// Full PCS pipeline for poly_size = 2^10 = 1024.
///
/// All rate-1/4 configs where row_len divides 1024:
///   row=16   (F3329 R4B2 D=1):    num_rows = 64
///   row=32   (F3329 R4B4 D=1):    num_rows = 32
///   row=64   (F3329 R4B8 D=1):    num_rows = 16
///   row=128  (R4B16 D=1):         num_rows = 8    [F65537, F1179649, F7340033, F167772161]
///   row=256  (R4B32 D=1):         num_rows = 4    [F65537, F1179649, F7340033, F167772161]
///   row=512  (R4B64 D=1):         num_rows = 2    [F65537, F1179649, F7340033, F167772161]
///   row=1024 (R4B16 D=2):         num_rows = 1    [F65537, F1179649, F7340033, F167772161]
fn r4_pcs_pipeline_p10(c: &mut Criterion) {
    let mut group = c.benchmark_group("R4 PCS P=10");
    group.sample_size(10);

    // ── Commit ──
    commit_named::<BenchZipPlusTypes<i128, 32>, F3329R4B2Bp<i128, 1, 32, UNCHECKED>, 10>(&mut group, "F3329-R4B2-D1");
    commit_named::<BenchZipPlusTypes<i128, 32>, F3329R4B4Bp<i128, 1, 32, UNCHECKED>, 10>(&mut group, "F3329-R4B4-D1");
    commit_named::<BenchZipPlusTypes<i128, 32>, F3329R4B8Bp<i128, 1, 32, UNCHECKED>, 10>(&mut group, "F3329-R4B8-D1");
    // D=1 configs
    commit_named::<BenchZipPlusTypes<i128, 32>, F65537R4B16Bp<i128, 1, 32, UNCHECKED>, 10>(&mut group, "F65537-R4B16-D1");
    commit_named::<BenchZipPlusTypes<i128, 32>, F65537R4B32Bp<i128, 1, 32, UNCHECKED>, 10>(&mut group, "F65537-R4B32-D1");
    commit_named::<BenchZipPlusTypes<i128, 32>, F65537R4B64Bp<i128, 1, 32, UNCHECKED>, 10>(&mut group, "F65537-R4B64-D1");
    commit_named::<BenchZipPlusTypes<i128, 32>, F1179649R4B16Bp<i128, 1, 32, UNCHECKED>, 10>(&mut group, "F1179649-R4B16-D1");
    commit_named::<BenchZipPlusTypes<i128, 32>, F1179649R4B32Bp<i128, 1, 32, UNCHECKED>, 10>(&mut group, "F1179649-R4B32-D1");
    commit_named::<BenchZipPlusTypes<i128, 32>, F1179649R4B64Bp<i128, 1, 32, UNCHECKED>, 10>(&mut group, "F1179649-R4B64-D1");
    commit_named::<BenchZipPlusTypes<i128, 32>, F7340033R4B16Bp<i128, 1, 32, UNCHECKED>, 10>(&mut group, "F7340033-R4B16-D1");
    commit_named::<BenchZipPlusTypes<i128, 32>, F7340033R4B32Bp<i128, 1, 32, UNCHECKED>, 10>(&mut group, "F7340033-R4B32-D1");
    commit_named::<BenchZipPlusTypes<i128, 32>, F7340033R4B64Bp<i128, 1, 32, UNCHECKED>, 10>(&mut group, "F7340033-R4B64-D1");
    commit_named::<BenchZipPlusTypes<i128, 32>, F167772161R4B16Bp<i128, 1, 32, UNCHECKED>, 10>(&mut group, "F167772161-R4B16-D1");
    commit_named::<BenchZipPlusTypes<i128, 32>, F167772161R4B32Bp<i128, 1, 32, UNCHECKED>, 10>(&mut group, "F167772161-R4B32-D1");
    commit_named::<BenchZipPlusTypes<i128, 32>, F167772161R4B64Bp<i128, 1, 32, UNCHECKED>, 10>(&mut group, "F167772161-R4B64-D1");
    // D=2 configs (row_len=1024)
    commit_named::<BenchZipPlusTypes<i128, 32>, F65537R4B16Bp<i128, 2, 32, UNCHECKED>, 10>(&mut group, "F65537-R4B16-D2");
    commit_named::<BenchZipPlusTypes<i128, 32>, F1179649R4B16Bp<i128, 2, 32, UNCHECKED>, 10>(&mut group, "F1179649-R4B16-D2");
    commit_named::<BenchZipPlusTypes<i128, 32>, F7340033R4B16Bp<i128, 2, 32, UNCHECKED>, 10>(&mut group, "F7340033-R4B16-D2");
    commit_named::<BenchZipPlusTypes<i128, 32>, F167772161R4B16Bp<i128, 2, 32, UNCHECKED>, 10>(&mut group, "F167772161-R4B16-D2");

    // ── Test ──
    test_named::<BenchZipPlusTypes<i128, 32>, F3329R4B2Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 10>(&mut group, "F3329-R4B2-D1");
    test_named::<BenchZipPlusTypes<i128, 32>, F3329R4B4Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 10>(&mut group, "F3329-R4B4-D1");
    test_named::<BenchZipPlusTypes<i128, 32>, F3329R4B8Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 10>(&mut group, "F3329-R4B8-D1");
    test_named::<BenchZipPlusTypes<i128, 32>, F65537R4B16Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 10>(&mut group, "F65537-R4B16-D1");
    test_named::<BenchZipPlusTypes<i128, 32>, F65537R4B32Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 10>(&mut group, "F65537-R4B32-D1");
    test_named::<BenchZipPlusTypes<i128, 32>, F65537R4B64Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 10>(&mut group, "F65537-R4B64-D1");
    test_named::<BenchZipPlusTypes<i128, 32>, F1179649R4B16Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 10>(&mut group, "F1179649-R4B16-D1");
    test_named::<BenchZipPlusTypes<i128, 32>, F1179649R4B32Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 10>(&mut group, "F1179649-R4B32-D1");
    test_named::<BenchZipPlusTypes<i128, 32>, F1179649R4B64Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 10>(&mut group, "F1179649-R4B64-D1");
    test_named::<BenchZipPlusTypes<i128, 32>, F7340033R4B16Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 10>(&mut group, "F7340033-R4B16-D1");
    test_named::<BenchZipPlusTypes<i128, 32>, F7340033R4B32Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 10>(&mut group, "F7340033-R4B32-D1");
    test_named::<BenchZipPlusTypes<i128, 32>, F7340033R4B64Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 10>(&mut group, "F7340033-R4B64-D1");
    test_named::<BenchZipPlusTypes<i128, 32>, F167772161R4B16Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 10>(&mut group, "F167772161-R4B16-D1");
    test_named::<BenchZipPlusTypes<i128, 32>, F167772161R4B32Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 10>(&mut group, "F167772161-R4B32-D1");
    test_named::<BenchZipPlusTypes<i128, 32>, F167772161R4B64Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 10>(&mut group, "F167772161-R4B64-D1");
    test_named::<BenchZipPlusTypes<i128, 32>, F65537R4B16Bp<i128, 2, 32, UNCHECKED>, UNCHECKED, 10>(&mut group, "F65537-R4B16-D2");
    test_named::<BenchZipPlusTypes<i128, 32>, F1179649R4B16Bp<i128, 2, 32, UNCHECKED>, UNCHECKED, 10>(&mut group, "F1179649-R4B16-D2");
    test_named::<BenchZipPlusTypes<i128, 32>, F7340033R4B16Bp<i128, 2, 32, UNCHECKED>, UNCHECKED, 10>(&mut group, "F7340033-R4B16-D2");
    test_named::<BenchZipPlusTypes<i128, 32>, F167772161R4B16Bp<i128, 2, 32, UNCHECKED>, UNCHECKED, 10>(&mut group, "F167772161-R4B16-D2");

    // ── Verify ──
    verify_named::<BenchZipPlusTypes<i128, 32>, F3329R4B2Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 10>(&mut group, "F3329-R4B2-D1");
    verify_named::<BenchZipPlusTypes<i128, 32>, F3329R4B4Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 10>(&mut group, "F3329-R4B4-D1");
    verify_named::<BenchZipPlusTypes<i128, 32>, F3329R4B8Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 10>(&mut group, "F3329-R4B8-D1");
    verify_named::<BenchZipPlusTypes<i128, 32>, F65537R4B16Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 10>(&mut group, "F65537-R4B16-D1");
    verify_named::<BenchZipPlusTypes<i128, 32>, F65537R4B32Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 10>(&mut group, "F65537-R4B32-D1");
    verify_named::<BenchZipPlusTypes<i128, 32>, F65537R4B64Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 10>(&mut group, "F65537-R4B64-D1");
    verify_named::<BenchZipPlusTypes<i128, 32>, F1179649R4B16Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 10>(&mut group, "F1179649-R4B16-D1");
    verify_named::<BenchZipPlusTypes<i128, 32>, F1179649R4B32Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 10>(&mut group, "F1179649-R4B32-D1");
    verify_named::<BenchZipPlusTypes<i128, 32>, F1179649R4B64Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 10>(&mut group, "F1179649-R4B64-D1");
    verify_named::<BenchZipPlusTypes<i128, 32>, F7340033R4B16Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 10>(&mut group, "F7340033-R4B16-D1");
    verify_named::<BenchZipPlusTypes<i128, 32>, F7340033R4B32Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 10>(&mut group, "F7340033-R4B32-D1");
    verify_named::<BenchZipPlusTypes<i128, 32>, F7340033R4B64Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 10>(&mut group, "F7340033-R4B64-D1");
    verify_named::<BenchZipPlusTypes<i128, 32>, F167772161R4B16Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 10>(&mut group, "F167772161-R4B16-D1");
    verify_named::<BenchZipPlusTypes<i128, 32>, F167772161R4B32Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 10>(&mut group, "F167772161-R4B32-D1");
    verify_named::<BenchZipPlusTypes<i128, 32>, F167772161R4B64Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 10>(&mut group, "F167772161-R4B64-D1");
    verify_named::<BenchZipPlusTypes<i128, 32>, F65537R4B16Bp<i128, 2, 32, UNCHECKED>, UNCHECKED, 10>(&mut group, "F65537-R4B16-D2");
    verify_named::<BenchZipPlusTypes<i128, 32>, F1179649R4B16Bp<i128, 2, 32, UNCHECKED>, UNCHECKED, 10>(&mut group, "F1179649-R4B16-D2");
    verify_named::<BenchZipPlusTypes<i128, 32>, F7340033R4B16Bp<i128, 2, 32, UNCHECKED>, UNCHECKED, 10>(&mut group, "F7340033-R4B16-D2");
    verify_named::<BenchZipPlusTypes<i128, 32>, F167772161R4B16Bp<i128, 2, 32, UNCHECKED>, UNCHECKED, 10>(&mut group, "F167772161-R4B16-D2");

    group.finish();
}

/// Full PCS pipeline for poly_size = 2^11 = 2048.
///
/// All rate-1/4 configs where row_len divides 2048:
///   row=16   (F3329 R4B2 D=1):    num_rows = 128
///   row=32   (F3329 R4B4 D=1):    num_rows = 64
///   row=64   (F3329 R4B8 D=1):    num_rows = 32
///   row=128  (R4B16 D=1):         num_rows = 16   [F65537, F1179649, F7340033, F167772161]
///   row=256  (R4B32 D=1):         num_rows = 8    [F65537, F1179649, F7340033, F167772161]
///   row=512  (R4B64 D=1):         num_rows = 4    [F65537, F1179649, F7340033, F167772161]
///   row=1024 (R4B16 D=2):         num_rows = 2    [F65537, F1179649, F7340033, F167772161]
///   row=2048 (R4B32 D=2):         num_rows = 1    [F65537, F1179649, F7340033, F167772161]
fn r4_pcs_pipeline_p11(c: &mut Criterion) {
    let mut group = c.benchmark_group("R4 PCS P=11");
    group.sample_size(10);

    // ── Commit ──
    commit_named::<BenchZipPlusTypes<i128, 32>, F3329R4B2Bp<i128, 1, 32, UNCHECKED>, 11>(&mut group, "F3329-R4B2-D1");
    commit_named::<BenchZipPlusTypes<i128, 32>, F3329R4B4Bp<i128, 1, 32, UNCHECKED>, 11>(&mut group, "F3329-R4B4-D1");
    commit_named::<BenchZipPlusTypes<i128, 32>, F3329R4B8Bp<i128, 1, 32, UNCHECKED>, 11>(&mut group, "F3329-R4B8-D1");
    // D=1 configs
    commit_named::<BenchZipPlusTypes<i128, 32>, F65537R4B16Bp<i128, 1, 32, UNCHECKED>, 11>(&mut group, "F65537-R4B16-D1");
    commit_named::<BenchZipPlusTypes<i128, 32>, F65537R4B32Bp<i128, 1, 32, UNCHECKED>, 11>(&mut group, "F65537-R4B32-D1");
    commit_named::<BenchZipPlusTypes<i128, 32>, F65537R4B64Bp<i128, 1, 32, UNCHECKED>, 11>(&mut group, "F65537-R4B64-D1");
    commit_named::<BenchZipPlusTypes<i128, 32>, F1179649R4B16Bp<i128, 1, 32, UNCHECKED>, 11>(&mut group, "F1179649-R4B16-D1");
    commit_named::<BenchZipPlusTypes<i128, 32>, F1179649R4B32Bp<i128, 1, 32, UNCHECKED>, 11>(&mut group, "F1179649-R4B32-D1");
    commit_named::<BenchZipPlusTypes<i128, 32>, F1179649R4B64Bp<i128, 1, 32, UNCHECKED>, 11>(&mut group, "F1179649-R4B64-D1");
    commit_named::<BenchZipPlusTypes<i128, 32>, F7340033R4B16Bp<i128, 1, 32, UNCHECKED>, 11>(&mut group, "F7340033-R4B16-D1");
    commit_named::<BenchZipPlusTypes<i128, 32>, F7340033R4B32Bp<i128, 1, 32, UNCHECKED>, 11>(&mut group, "F7340033-R4B32-D1");
    commit_named::<BenchZipPlusTypes<i128, 32>, F7340033R4B64Bp<i128, 1, 32, UNCHECKED>, 11>(&mut group, "F7340033-R4B64-D1");
    commit_named::<BenchZipPlusTypes<i128, 32>, F167772161R4B16Bp<i128, 1, 32, UNCHECKED>, 11>(&mut group, "F167772161-R4B16-D1");
    commit_named::<BenchZipPlusTypes<i128, 32>, F167772161R4B32Bp<i128, 1, 32, UNCHECKED>, 11>(&mut group, "F167772161-R4B32-D1");
    commit_named::<BenchZipPlusTypes<i128, 32>, F167772161R4B64Bp<i128, 1, 32, UNCHECKED>, 11>(&mut group, "F167772161-R4B64-D1");
    // D=2 configs
    commit_named::<BenchZipPlusTypes<i128, 32>, F65537R4B16Bp<i128, 2, 32, UNCHECKED>, 11>(&mut group, "F65537-R4B16-D2");
    commit_named::<BenchZipPlusTypes<i128, 32>, F65537R4B32Bp<i128, 2, 32, UNCHECKED>, 11>(&mut group, "F65537-R4B32-D2");
    commit_named::<BenchZipPlusTypes<i128, 32>, F1179649R4B16Bp<i128, 2, 32, UNCHECKED>, 11>(&mut group, "F1179649-R4B16-D2");
    commit_named::<BenchZipPlusTypes<i128, 32>, F1179649R4B32Bp<i128, 2, 32, UNCHECKED>, 11>(&mut group, "F1179649-R4B32-D2");
    commit_named::<BenchZipPlusTypes<i128, 32>, F7340033R4B16Bp<i128, 2, 32, UNCHECKED>, 11>(&mut group, "F7340033-R4B16-D2");
    commit_named::<BenchZipPlusTypes<i128, 32>, F7340033R4B32Bp<i128, 2, 32, UNCHECKED>, 11>(&mut group, "F7340033-R4B32-D2");
    commit_named::<BenchZipPlusTypes<i128, 32>, F167772161R4B16Bp<i128, 2, 32, UNCHECKED>, 11>(&mut group, "F167772161-R4B16-D2");
    commit_named::<BenchZipPlusTypes<i128, 32>, F167772161R4B32Bp<i128, 2, 32, UNCHECKED>, 11>(&mut group, "F167772161-R4B32-D2");

    // ── Test ──
    test_named::<BenchZipPlusTypes<i128, 32>, F3329R4B2Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 11>(&mut group, "F3329-R4B2-D1");
    test_named::<BenchZipPlusTypes<i128, 32>, F3329R4B4Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 11>(&mut group, "F3329-R4B4-D1");
    test_named::<BenchZipPlusTypes<i128, 32>, F3329R4B8Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 11>(&mut group, "F3329-R4B8-D1");
    test_named::<BenchZipPlusTypes<i128, 32>, F65537R4B16Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 11>(&mut group, "F65537-R4B16-D1");
    test_named::<BenchZipPlusTypes<i128, 32>, F65537R4B32Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 11>(&mut group, "F65537-R4B32-D1");
    test_named::<BenchZipPlusTypes<i128, 32>, F65537R4B64Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 11>(&mut group, "F65537-R4B64-D1");
    test_named::<BenchZipPlusTypes<i128, 32>, F1179649R4B16Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 11>(&mut group, "F1179649-R4B16-D1");
    test_named::<BenchZipPlusTypes<i128, 32>, F1179649R4B32Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 11>(&mut group, "F1179649-R4B32-D1");
    test_named::<BenchZipPlusTypes<i128, 32>, F1179649R4B64Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 11>(&mut group, "F1179649-R4B64-D1");
    test_named::<BenchZipPlusTypes<i128, 32>, F7340033R4B16Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 11>(&mut group, "F7340033-R4B16-D1");
    test_named::<BenchZipPlusTypes<i128, 32>, F7340033R4B32Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 11>(&mut group, "F7340033-R4B32-D1");
    test_named::<BenchZipPlusTypes<i128, 32>, F7340033R4B64Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 11>(&mut group, "F7340033-R4B64-D1");
    test_named::<BenchZipPlusTypes<i128, 32>, F167772161R4B16Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 11>(&mut group, "F167772161-R4B16-D1");
    test_named::<BenchZipPlusTypes<i128, 32>, F167772161R4B32Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 11>(&mut group, "F167772161-R4B32-D1");
    test_named::<BenchZipPlusTypes<i128, 32>, F167772161R4B64Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 11>(&mut group, "F167772161-R4B64-D1");
    test_named::<BenchZipPlusTypes<i128, 32>, F65537R4B16Bp<i128, 2, 32, UNCHECKED>, UNCHECKED, 11>(&mut group, "F65537-R4B16-D2");
    test_named::<BenchZipPlusTypes<i128, 32>, F65537R4B32Bp<i128, 2, 32, UNCHECKED>, UNCHECKED, 11>(&mut group, "F65537-R4B32-D2");
    test_named::<BenchZipPlusTypes<i128, 32>, F1179649R4B16Bp<i128, 2, 32, UNCHECKED>, UNCHECKED, 11>(&mut group, "F1179649-R4B16-D2");
    test_named::<BenchZipPlusTypes<i128, 32>, F1179649R4B32Bp<i128, 2, 32, UNCHECKED>, UNCHECKED, 11>(&mut group, "F1179649-R4B32-D2");
    test_named::<BenchZipPlusTypes<i128, 32>, F7340033R4B16Bp<i128, 2, 32, UNCHECKED>, UNCHECKED, 11>(&mut group, "F7340033-R4B16-D2");
    test_named::<BenchZipPlusTypes<i128, 32>, F7340033R4B32Bp<i128, 2, 32, UNCHECKED>, UNCHECKED, 11>(&mut group, "F7340033-R4B32-D2");
    test_named::<BenchZipPlusTypes<i128, 32>, F167772161R4B16Bp<i128, 2, 32, UNCHECKED>, UNCHECKED, 11>(&mut group, "F167772161-R4B16-D2");
    test_named::<BenchZipPlusTypes<i128, 32>, F167772161R4B32Bp<i128, 2, 32, UNCHECKED>, UNCHECKED, 11>(&mut group, "F167772161-R4B32-D2");

    // ── Verify ──
    verify_named::<BenchZipPlusTypes<i128, 32>, F3329R4B2Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 11>(&mut group, "F3329-R4B2-D1");
    verify_named::<BenchZipPlusTypes<i128, 32>, F3329R4B4Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 11>(&mut group, "F3329-R4B4-D1");
    verify_named::<BenchZipPlusTypes<i128, 32>, F3329R4B8Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 11>(&mut group, "F3329-R4B8-D1");
    verify_named::<BenchZipPlusTypes<i128, 32>, F65537R4B16Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 11>(&mut group, "F65537-R4B16-D1");
    verify_named::<BenchZipPlusTypes<i128, 32>, F65537R4B32Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 11>(&mut group, "F65537-R4B32-D1");
    verify_named::<BenchZipPlusTypes<i128, 32>, F65537R4B64Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 11>(&mut group, "F65537-R4B64-D1");
    verify_named::<BenchZipPlusTypes<i128, 32>, F1179649R4B16Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 11>(&mut group, "F1179649-R4B16-D1");
    verify_named::<BenchZipPlusTypes<i128, 32>, F1179649R4B32Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 11>(&mut group, "F1179649-R4B32-D1");
    verify_named::<BenchZipPlusTypes<i128, 32>, F1179649R4B64Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 11>(&mut group, "F1179649-R4B64-D1");
    verify_named::<BenchZipPlusTypes<i128, 32>, F7340033R4B16Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 11>(&mut group, "F7340033-R4B16-D1");
    verify_named::<BenchZipPlusTypes<i128, 32>, F7340033R4B32Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 11>(&mut group, "F7340033-R4B32-D1");
    verify_named::<BenchZipPlusTypes<i128, 32>, F7340033R4B64Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 11>(&mut group, "F7340033-R4B64-D1");
    verify_named::<BenchZipPlusTypes<i128, 32>, F167772161R4B16Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 11>(&mut group, "F167772161-R4B16-D1");
    verify_named::<BenchZipPlusTypes<i128, 32>, F167772161R4B32Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 11>(&mut group, "F167772161-R4B32-D1");
    verify_named::<BenchZipPlusTypes<i128, 32>, F167772161R4B64Bp<i128, 1, 32, UNCHECKED>, UNCHECKED, 11>(&mut group, "F167772161-R4B64-D1");
    verify_named::<BenchZipPlusTypes<i128, 32>, F65537R4B16Bp<i128, 2, 32, UNCHECKED>, UNCHECKED, 11>(&mut group, "F65537-R4B16-D2");
    verify_named::<BenchZipPlusTypes<i128, 32>, F65537R4B32Bp<i128, 2, 32, UNCHECKED>, UNCHECKED, 11>(&mut group, "F65537-R4B32-D2");
    verify_named::<BenchZipPlusTypes<i128, 32>, F1179649R4B16Bp<i128, 2, 32, UNCHECKED>, UNCHECKED, 11>(&mut group, "F1179649-R4B16-D2");
    verify_named::<BenchZipPlusTypes<i128, 32>, F1179649R4B32Bp<i128, 2, 32, UNCHECKED>, UNCHECKED, 11>(&mut group, "F1179649-R4B32-D2");
    verify_named::<BenchZipPlusTypes<i128, 32>, F7340033R4B16Bp<i128, 2, 32, UNCHECKED>, UNCHECKED, 11>(&mut group, "F7340033-R4B16-D2");
    verify_named::<BenchZipPlusTypes<i128, 32>, F7340033R4B32Bp<i128, 2, 32, UNCHECKED>, UNCHECKED, 11>(&mut group, "F7340033-R4B32-D2");
    verify_named::<BenchZipPlusTypes<i128, 32>, F167772161R4B16Bp<i128, 2, 32, UNCHECKED>, UNCHECKED, 11>(&mut group, "F167772161-R4B16-D2");
    verify_named::<BenchZipPlusTypes<i128, 32>, F167772161R4B32Bp<i128, 2, 32, UNCHECKED>, UNCHECKED, 11>(&mut group, "F167772161-R4B32-D2");

    group.finish();
}

// ==================== Scalar i32 PCS Pipeline ====================

/// Scalar i32 PCS pipeline for P=8 to P=11 with all rate-1/4 configs.
fn r4_pcs_pipeline_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("R4 PCS i32");
    group.sample_size(10);

    // ── P=8 ──
    commit_named::<BenchZipScalarTypes, F3329R4B2<1, UNCHECKED>, 8>(&mut group, "F3329-R4B2-D1");
    commit_named::<BenchZipScalarTypes, F3329R4B4<1, UNCHECKED>, 8>(&mut group, "F3329-R4B4-D1");
    commit_named::<BenchZipScalarTypes, F3329R4B8<1, UNCHECKED>, 8>(&mut group, "F3329-R4B8-D1");
    commit_named::<BenchZipScalarTypes, F65537R4B16<1, UNCHECKED>, 8>(&mut group, "F65537-R4B16-D1");
    commit_named::<BenchZipScalarTypes, F65537R4B32<1, UNCHECKED>, 8>(&mut group, "F65537-R4B32-D1");
    commit_named::<BenchZipScalarTypes, F1179649R4B16<1, UNCHECKED>, 8>(&mut group, "F1179649-R4B16-D1");
    commit_named::<BenchZipScalarTypes, F1179649R4B32<1, UNCHECKED>, 8>(&mut group, "F1179649-R4B32-D1");
    commit_named::<BenchZipScalarTypes, F7340033R4B16<1, UNCHECKED>, 8>(&mut group, "F7340033-R4B16-D1");
    commit_named::<BenchZipScalarTypes, F7340033R4B32<1, UNCHECKED>, 8>(&mut group, "F7340033-R4B32-D1");
    commit_named::<BenchZipScalarTypes, F167772161R4B16<1, UNCHECKED>, 8>(&mut group, "F167772161-R4B16-D1");
    commit_named::<BenchZipScalarTypes, F167772161R4B32<1, UNCHECKED>, 8>(&mut group, "F167772161-R4B32-D1");

    test_named::<BenchZipScalarTypes, F3329R4B2<1, UNCHECKED>, UNCHECKED, 8>(&mut group, "F3329-R4B2-D1");
    test_named::<BenchZipScalarTypes, F3329R4B4<1, UNCHECKED>, UNCHECKED, 8>(&mut group, "F3329-R4B4-D1");
    test_named::<BenchZipScalarTypes, F3329R4B8<1, UNCHECKED>, UNCHECKED, 8>(&mut group, "F3329-R4B8-D1");
    test_named::<BenchZipScalarTypes, F65537R4B16<1, UNCHECKED>, UNCHECKED, 8>(&mut group, "F65537-R4B16-D1");
    test_named::<BenchZipScalarTypes, F65537R4B32<1, UNCHECKED>, UNCHECKED, 8>(&mut group, "F65537-R4B32-D1");
    test_named::<BenchZipScalarTypes, F1179649R4B16<1, UNCHECKED>, UNCHECKED, 8>(&mut group, "F1179649-R4B16-D1");
    test_named::<BenchZipScalarTypes, F1179649R4B32<1, UNCHECKED>, UNCHECKED, 8>(&mut group, "F1179649-R4B32-D1");
    test_named::<BenchZipScalarTypes, F7340033R4B16<1, UNCHECKED>, UNCHECKED, 8>(&mut group, "F7340033-R4B16-D1");
    test_named::<BenchZipScalarTypes, F7340033R4B32<1, UNCHECKED>, UNCHECKED, 8>(&mut group, "F7340033-R4B32-D1");
    test_named::<BenchZipScalarTypes, F167772161R4B16<1, UNCHECKED>, UNCHECKED, 8>(&mut group, "F167772161-R4B16-D1");
    test_named::<BenchZipScalarTypes, F167772161R4B32<1, UNCHECKED>, UNCHECKED, 8>(&mut group, "F167772161-R4B32-D1");

    verify_named::<BenchZipScalarTypes, F3329R4B2<1, UNCHECKED>, UNCHECKED, 8>(&mut group, "F3329-R4B2-D1");
    verify_named::<BenchZipScalarTypes, F3329R4B4<1, UNCHECKED>, UNCHECKED, 8>(&mut group, "F3329-R4B4-D1");
    verify_named::<BenchZipScalarTypes, F3329R4B8<1, UNCHECKED>, UNCHECKED, 8>(&mut group, "F3329-R4B8-D1");
    verify_named::<BenchZipScalarTypes, F65537R4B16<1, UNCHECKED>, UNCHECKED, 8>(&mut group, "F65537-R4B16-D1");
    verify_named::<BenchZipScalarTypes, F65537R4B32<1, UNCHECKED>, UNCHECKED, 8>(&mut group, "F65537-R4B32-D1");
    verify_named::<BenchZipScalarTypes, F1179649R4B16<1, UNCHECKED>, UNCHECKED, 8>(&mut group, "F1179649-R4B16-D1");
    verify_named::<BenchZipScalarTypes, F1179649R4B32<1, UNCHECKED>, UNCHECKED, 8>(&mut group, "F1179649-R4B32-D1");
    verify_named::<BenchZipScalarTypes, F7340033R4B16<1, UNCHECKED>, UNCHECKED, 8>(&mut group, "F7340033-R4B16-D1");
    verify_named::<BenchZipScalarTypes, F7340033R4B32<1, UNCHECKED>, UNCHECKED, 8>(&mut group, "F7340033-R4B32-D1");
    verify_named::<BenchZipScalarTypes, F167772161R4B16<1, UNCHECKED>, UNCHECKED, 8>(&mut group, "F167772161-R4B16-D1");
    verify_named::<BenchZipScalarTypes, F167772161R4B32<1, UNCHECKED>, UNCHECKED, 8>(&mut group, "F167772161-R4B32-D1");

    // ── P=9 ──
    commit_named::<BenchZipScalarTypes, F3329R4B2<1, UNCHECKED>, 9>(&mut group, "F3329-R4B2-D1");
    commit_named::<BenchZipScalarTypes, F3329R4B4<1, UNCHECKED>, 9>(&mut group, "F3329-R4B4-D1");
    commit_named::<BenchZipScalarTypes, F3329R4B8<1, UNCHECKED>, 9>(&mut group, "F3329-R4B8-D1");
    commit_named::<BenchZipScalarTypes, F65537R4B16<1, UNCHECKED>, 9>(&mut group, "F65537-R4B16-D1");
    commit_named::<BenchZipScalarTypes, F65537R4B32<1, UNCHECKED>, 9>(&mut group, "F65537-R4B32-D1");
    commit_named::<BenchZipScalarTypes, F65537R4B64<1, UNCHECKED>, 9>(&mut group, "F65537-R4B64-D1");
    commit_named::<BenchZipScalarTypes, F1179649R4B16<1, UNCHECKED>, 9>(&mut group, "F1179649-R4B16-D1");
    commit_named::<BenchZipScalarTypes, F1179649R4B32<1, UNCHECKED>, 9>(&mut group, "F1179649-R4B32-D1");
    commit_named::<BenchZipScalarTypes, F1179649R4B64<1, UNCHECKED>, 9>(&mut group, "F1179649-R4B64-D1");
    commit_named::<BenchZipScalarTypes, F7340033R4B16<1, UNCHECKED>, 9>(&mut group, "F7340033-R4B16-D1");
    commit_named::<BenchZipScalarTypes, F7340033R4B32<1, UNCHECKED>, 9>(&mut group, "F7340033-R4B32-D1");
    commit_named::<BenchZipScalarTypes, F7340033R4B64<1, UNCHECKED>, 9>(&mut group, "F7340033-R4B64-D1");
    commit_named::<BenchZipScalarTypes, F167772161R4B16<1, UNCHECKED>, 9>(&mut group, "F167772161-R4B16-D1");
    commit_named::<BenchZipScalarTypes, F167772161R4B32<1, UNCHECKED>, 9>(&mut group, "F167772161-R4B32-D1");
    commit_named::<BenchZipScalarTypes, F167772161R4B64<1, UNCHECKED>, 9>(&mut group, "F167772161-R4B64-D1");

    test_named::<BenchZipScalarTypes, F3329R4B2<1, UNCHECKED>, UNCHECKED, 9>(&mut group, "F3329-R4B2-D1");
    test_named::<BenchZipScalarTypes, F3329R4B4<1, UNCHECKED>, UNCHECKED, 9>(&mut group, "F3329-R4B4-D1");
    test_named::<BenchZipScalarTypes, F3329R4B8<1, UNCHECKED>, UNCHECKED, 9>(&mut group, "F3329-R4B8-D1");
    test_named::<BenchZipScalarTypes, F65537R4B16<1, UNCHECKED>, UNCHECKED, 9>(&mut group, "F65537-R4B16-D1");
    test_named::<BenchZipScalarTypes, F65537R4B32<1, UNCHECKED>, UNCHECKED, 9>(&mut group, "F65537-R4B32-D1");
    test_named::<BenchZipScalarTypes, F65537R4B64<1, UNCHECKED>, UNCHECKED, 9>(&mut group, "F65537-R4B64-D1");
    test_named::<BenchZipScalarTypes, F1179649R4B16<1, UNCHECKED>, UNCHECKED, 9>(&mut group, "F1179649-R4B16-D1");
    test_named::<BenchZipScalarTypes, F1179649R4B32<1, UNCHECKED>, UNCHECKED, 9>(&mut group, "F1179649-R4B32-D1");
    test_named::<BenchZipScalarTypes, F1179649R4B64<1, UNCHECKED>, UNCHECKED, 9>(&mut group, "F1179649-R4B64-D1");
    test_named::<BenchZipScalarTypes, F7340033R4B16<1, UNCHECKED>, UNCHECKED, 9>(&mut group, "F7340033-R4B16-D1");
    test_named::<BenchZipScalarTypes, F7340033R4B32<1, UNCHECKED>, UNCHECKED, 9>(&mut group, "F7340033-R4B32-D1");
    test_named::<BenchZipScalarTypes, F7340033R4B64<1, UNCHECKED>, UNCHECKED, 9>(&mut group, "F7340033-R4B64-D1");
    test_named::<BenchZipScalarTypes, F167772161R4B16<1, UNCHECKED>, UNCHECKED, 9>(&mut group, "F167772161-R4B16-D1");
    test_named::<BenchZipScalarTypes, F167772161R4B32<1, UNCHECKED>, UNCHECKED, 9>(&mut group, "F167772161-R4B32-D1");
    test_named::<BenchZipScalarTypes, F167772161R4B64<1, UNCHECKED>, UNCHECKED, 9>(&mut group, "F167772161-R4B64-D1");

    verify_named::<BenchZipScalarTypes, F3329R4B2<1, UNCHECKED>, UNCHECKED, 9>(&mut group, "F3329-R4B2-D1");
    verify_named::<BenchZipScalarTypes, F3329R4B4<1, UNCHECKED>, UNCHECKED, 9>(&mut group, "F3329-R4B4-D1");
    verify_named::<BenchZipScalarTypes, F3329R4B8<1, UNCHECKED>, UNCHECKED, 9>(&mut group, "F3329-R4B8-D1");
    verify_named::<BenchZipScalarTypes, F65537R4B16<1, UNCHECKED>, UNCHECKED, 9>(&mut group, "F65537-R4B16-D1");
    verify_named::<BenchZipScalarTypes, F65537R4B32<1, UNCHECKED>, UNCHECKED, 9>(&mut group, "F65537-R4B32-D1");
    verify_named::<BenchZipScalarTypes, F65537R4B64<1, UNCHECKED>, UNCHECKED, 9>(&mut group, "F65537-R4B64-D1");
    verify_named::<BenchZipScalarTypes, F1179649R4B16<1, UNCHECKED>, UNCHECKED, 9>(&mut group, "F1179649-R4B16-D1");
    verify_named::<BenchZipScalarTypes, F1179649R4B32<1, UNCHECKED>, UNCHECKED, 9>(&mut group, "F1179649-R4B32-D1");
    verify_named::<BenchZipScalarTypes, F1179649R4B64<1, UNCHECKED>, UNCHECKED, 9>(&mut group, "F1179649-R4B64-D1");
    verify_named::<BenchZipScalarTypes, F7340033R4B16<1, UNCHECKED>, UNCHECKED, 9>(&mut group, "F7340033-R4B16-D1");
    verify_named::<BenchZipScalarTypes, F7340033R4B32<1, UNCHECKED>, UNCHECKED, 9>(&mut group, "F7340033-R4B32-D1");
    verify_named::<BenchZipScalarTypes, F7340033R4B64<1, UNCHECKED>, UNCHECKED, 9>(&mut group, "F7340033-R4B64-D1");
    verify_named::<BenchZipScalarTypes, F167772161R4B16<1, UNCHECKED>, UNCHECKED, 9>(&mut group, "F167772161-R4B16-D1");
    verify_named::<BenchZipScalarTypes, F167772161R4B32<1, UNCHECKED>, UNCHECKED, 9>(&mut group, "F167772161-R4B32-D1");
    verify_named::<BenchZipScalarTypes, F167772161R4B64<1, UNCHECKED>, UNCHECKED, 9>(&mut group, "F167772161-R4B64-D1");

    // ── P=10 ──
    commit_named::<BenchZipScalarTypes, F3329R4B2<1, UNCHECKED>, 10>(&mut group, "F3329-R4B2-D1");
    commit_named::<BenchZipScalarTypes, F3329R4B4<1, UNCHECKED>, 10>(&mut group, "F3329-R4B4-D1");
    commit_named::<BenchZipScalarTypes, F3329R4B8<1, UNCHECKED>, 10>(&mut group, "F3329-R4B8-D1");
    commit_named::<BenchZipScalarTypes, F65537R4B16<1, UNCHECKED>, 10>(&mut group, "F65537-R4B16-D1");
    commit_named::<BenchZipScalarTypes, F65537R4B32<1, UNCHECKED>, 10>(&mut group, "F65537-R4B32-D1");
    commit_named::<BenchZipScalarTypes, F65537R4B64<1, UNCHECKED>, 10>(&mut group, "F65537-R4B64-D1");
    commit_named::<BenchZipScalarTypes, F65537R4B16<2, UNCHECKED>, 10>(&mut group, "F65537-R4B16-D2");
    commit_named::<BenchZipScalarTypes, F1179649R4B16<1, UNCHECKED>, 10>(&mut group, "F1179649-R4B16-D1");
    commit_named::<BenchZipScalarTypes, F1179649R4B32<1, UNCHECKED>, 10>(&mut group, "F1179649-R4B32-D1");
    commit_named::<BenchZipScalarTypes, F1179649R4B64<1, UNCHECKED>, 10>(&mut group, "F1179649-R4B64-D1");
    commit_named::<BenchZipScalarTypes, F1179649R4B16<2, UNCHECKED>, 10>(&mut group, "F1179649-R4B16-D2");
    commit_named::<BenchZipScalarTypes, F7340033R4B16<1, UNCHECKED>, 10>(&mut group, "F7340033-R4B16-D1");
    commit_named::<BenchZipScalarTypes, F7340033R4B32<1, UNCHECKED>, 10>(&mut group, "F7340033-R4B32-D1");
    commit_named::<BenchZipScalarTypes, F7340033R4B64<1, UNCHECKED>, 10>(&mut group, "F7340033-R4B64-D1");
    commit_named::<BenchZipScalarTypes, F7340033R4B16<2, UNCHECKED>, 10>(&mut group, "F7340033-R4B16-D2");
    commit_named::<BenchZipScalarTypes, F167772161R4B16<1, UNCHECKED>, 10>(&mut group, "F167772161-R4B16-D1");
    commit_named::<BenchZipScalarTypes, F167772161R4B32<1, UNCHECKED>, 10>(&mut group, "F167772161-R4B32-D1");
    commit_named::<BenchZipScalarTypes, F167772161R4B64<1, UNCHECKED>, 10>(&mut group, "F167772161-R4B64-D1");
    commit_named::<BenchZipScalarTypes, F167772161R4B16<2, UNCHECKED>, 10>(&mut group, "F167772161-R4B16-D2");

    test_named::<BenchZipScalarTypes, F65537R4B16<2, UNCHECKED>, UNCHECKED, 10>(&mut group, "F65537-R4B16-D2");
    test_named::<BenchZipScalarTypes, F1179649R4B16<2, UNCHECKED>, UNCHECKED, 10>(&mut group, "F1179649-R4B16-D2");
    test_named::<BenchZipScalarTypes, F7340033R4B16<2, UNCHECKED>, UNCHECKED, 10>(&mut group, "F7340033-R4B16-D2");
    test_named::<BenchZipScalarTypes, F167772161R4B16<2, UNCHECKED>, UNCHECKED, 10>(&mut group, "F167772161-R4B16-D2");

    verify_named::<BenchZipScalarTypes, F65537R4B16<2, UNCHECKED>, UNCHECKED, 10>(&mut group, "F65537-R4B16-D2");
    verify_named::<BenchZipScalarTypes, F1179649R4B16<2, UNCHECKED>, UNCHECKED, 10>(&mut group, "F1179649-R4B16-D2");
    verify_named::<BenchZipScalarTypes, F7340033R4B16<2, UNCHECKED>, UNCHECKED, 10>(&mut group, "F7340033-R4B16-D2");
    verify_named::<BenchZipScalarTypes, F167772161R4B16<2, UNCHECKED>, UNCHECKED, 10>(&mut group, "F167772161-R4B16-D2");

    // ── P=11 ──
    commit_named::<BenchZipScalarTypes, F3329R4B2<1, UNCHECKED>, 11>(&mut group, "F3329-R4B2-D1");
    commit_named::<BenchZipScalarTypes, F3329R4B4<1, UNCHECKED>, 11>(&mut group, "F3329-R4B4-D1");
    commit_named::<BenchZipScalarTypes, F3329R4B8<1, UNCHECKED>, 11>(&mut group, "F3329-R4B8-D1");
    commit_named::<BenchZipScalarTypes, F65537R4B16<1, UNCHECKED>, 11>(&mut group, "F65537-R4B16-D1");
    commit_named::<BenchZipScalarTypes, F65537R4B32<1, UNCHECKED>, 11>(&mut group, "F65537-R4B32-D1");
    commit_named::<BenchZipScalarTypes, F65537R4B64<1, UNCHECKED>, 11>(&mut group, "F65537-R4B64-D1");
    commit_named::<BenchZipScalarTypes, F65537R4B16<2, UNCHECKED>, 11>(&mut group, "F65537-R4B16-D2");
    commit_named::<BenchZipScalarTypes, F65537R4B32<2, UNCHECKED>, 11>(&mut group, "F65537-R4B32-D2");
    commit_named::<BenchZipScalarTypes, F1179649R4B16<1, UNCHECKED>, 11>(&mut group, "F1179649-R4B16-D1");
    commit_named::<BenchZipScalarTypes, F1179649R4B32<1, UNCHECKED>, 11>(&mut group, "F1179649-R4B32-D1");
    commit_named::<BenchZipScalarTypes, F1179649R4B64<1, UNCHECKED>, 11>(&mut group, "F1179649-R4B64-D1");
    commit_named::<BenchZipScalarTypes, F1179649R4B16<2, UNCHECKED>, 11>(&mut group, "F1179649-R4B16-D2");
    commit_named::<BenchZipScalarTypes, F1179649R4B32<2, UNCHECKED>, 11>(&mut group, "F1179649-R4B32-D2");
    commit_named::<BenchZipScalarTypes, F7340033R4B16<1, UNCHECKED>, 11>(&mut group, "F7340033-R4B16-D1");
    commit_named::<BenchZipScalarTypes, F7340033R4B32<1, UNCHECKED>, 11>(&mut group, "F7340033-R4B32-D1");
    commit_named::<BenchZipScalarTypes, F7340033R4B64<1, UNCHECKED>, 11>(&mut group, "F7340033-R4B64-D1");
    commit_named::<BenchZipScalarTypes, F7340033R4B16<2, UNCHECKED>, 11>(&mut group, "F7340033-R4B16-D2");
    commit_named::<BenchZipScalarTypes, F7340033R4B32<2, UNCHECKED>, 11>(&mut group, "F7340033-R4B32-D2");
    commit_named::<BenchZipScalarTypes, F167772161R4B16<1, UNCHECKED>, 11>(&mut group, "F167772161-R4B16-D1");
    commit_named::<BenchZipScalarTypes, F167772161R4B32<1, UNCHECKED>, 11>(&mut group, "F167772161-R4B32-D1");
    commit_named::<BenchZipScalarTypes, F167772161R4B64<1, UNCHECKED>, 11>(&mut group, "F167772161-R4B64-D1");
    commit_named::<BenchZipScalarTypes, F167772161R4B16<2, UNCHECKED>, 11>(&mut group, "F167772161-R4B16-D2");
    commit_named::<BenchZipScalarTypes, F167772161R4B32<2, UNCHECKED>, 11>(&mut group, "F167772161-R4B32-D2");

    test_named::<BenchZipScalarTypes, F65537R4B16<2, UNCHECKED>, UNCHECKED, 11>(&mut group, "F65537-R4B16-D2");
    test_named::<BenchZipScalarTypes, F65537R4B32<2, UNCHECKED>, UNCHECKED, 11>(&mut group, "F65537-R4B32-D2");
    test_named::<BenchZipScalarTypes, F1179649R4B16<2, UNCHECKED>, UNCHECKED, 11>(&mut group, "F1179649-R4B16-D2");
    test_named::<BenchZipScalarTypes, F1179649R4B32<2, UNCHECKED>, UNCHECKED, 11>(&mut group, "F1179649-R4B32-D2");
    test_named::<BenchZipScalarTypes, F7340033R4B16<2, UNCHECKED>, UNCHECKED, 11>(&mut group, "F7340033-R4B16-D2");
    test_named::<BenchZipScalarTypes, F7340033R4B32<2, UNCHECKED>, UNCHECKED, 11>(&mut group, "F7340033-R4B32-D2");
    test_named::<BenchZipScalarTypes, F167772161R4B16<2, UNCHECKED>, UNCHECKED, 11>(&mut group, "F167772161-R4B16-D2");
    test_named::<BenchZipScalarTypes, F167772161R4B32<2, UNCHECKED>, UNCHECKED, 11>(&mut group, "F167772161-R4B32-D2");

    verify_named::<BenchZipScalarTypes, F65537R4B16<2, UNCHECKED>, UNCHECKED, 11>(&mut group, "F65537-R4B16-D2");
    verify_named::<BenchZipScalarTypes, F65537R4B32<2, UNCHECKED>, UNCHECKED, 11>(&mut group, "F65537-R4B32-D2");
    verify_named::<BenchZipScalarTypes, F1179649R4B16<2, UNCHECKED>, UNCHECKED, 11>(&mut group, "F1179649-R4B16-D2");
    verify_named::<BenchZipScalarTypes, F1179649R4B32<2, UNCHECKED>, UNCHECKED, 11>(&mut group, "F1179649-R4B32-D2");
    verify_named::<BenchZipScalarTypes, F7340033R4B16<2, UNCHECKED>, UNCHECKED, 11>(&mut group, "F7340033-R4B16-D2");
    verify_named::<BenchZipScalarTypes, F7340033R4B32<2, UNCHECKED>, UNCHECKED, 11>(&mut group, "F7340033-R4B32-D2");
    verify_named::<BenchZipScalarTypes, F167772161R4B16<2, UNCHECKED>, UNCHECKED, 11>(&mut group, "F167772161-R4B16-D2");
    verify_named::<BenchZipScalarTypes, F167772161R4B32<2, UNCHECKED>, UNCHECKED, 11>(&mut group, "F167772161-R4B32-D2");

    group.finish();
}

criterion_group!(
    benches,
    r4_encode_config_search_i32,
    r4_encode_config_search_bpoly,
    r4_pcs_pipeline_p8,
    r4_pcs_pipeline_p9,
    r4_pcs_pipeline_p10,
    r4_pcs_pipeline_p11,
    r4_pcs_pipeline_scalar,
);
criterion_main!(benches);
