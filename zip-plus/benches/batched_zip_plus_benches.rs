//! Benchmarks for the Batched Zip+ PCS.
//!
//! Measures commit, test, evaluate, and verify for batches of polynomials
//! using a single shared Merkle tree.

#![allow(non_local_definitions)]
#![allow(clippy::eq_op, clippy::arithmetic_side_effects, clippy::unwrap_used)]

mod zip_common;

use std::marker::PhantomData;

use criterion::{Criterion, criterion_group, criterion_main};
use crypto_bigint::U64;
use crypto_primitives::{
    FixedSemiring,
    boolean::Boolean, crypto_bigint_int::Int,
    crypto_bigint_uint::Uint,
};

use zinc_poly::univariate::{
    binary::{BinaryPoly, BinaryPolyInnerProduct, BinaryPolyWideningMulByScalar},
    dense::{DensePolyInnerProduct, DensePolynomial},
};
use zinc_primality::MillerRabin;
use zinc_transcript::traits::ConstTranscribable;
use zinc_utils::{
    UNCHECKED, from_ref::FromRef, inner_product::{MBSInnerProduct, ScalarProduct},
    mul_by_scalar::ScalarWideningMulByScalar,
    named::Named,
};
use zip_plus::{
    code::iprs::{
        IprsCode,
        PnttConfigF2_16_1, PnttConfigF2_16B16, PnttConfigF2_16B64,
        PnttConfigF2_16R4B16, PnttConfigF2_16R4B32, PnttConfigF2_16R4B64,
        PnttConfigF3329R4B2, PnttConfigF3329R4B4, PnttConfigF3329R4B8,
        PnttConfigF1179649B16,
        PnttConfigF167772161, PnttConfigF167772161B64,
    },
    pcs::structs::ZipTypes,
};

const INT_LIMBS: usize = U64::LIMBS;

// ---------- ZipTypes for scalar i32 benchmarks (matching PCS Pipeline i32) ----

/// ZipTypes implementation for scalar i32 evaluations, matching the
/// configuration used in the PCS Pipeline Suite i32 1row.
struct BenchZipScalarTypes;

impl ZipTypes for BenchZipScalarTypes {
    const NUM_COLUMN_OPENINGS: usize = 147;
    type Eval = i32;
    type Cw = i128;
    type Fmod = Uint<{ INT_LIMBS * 4 }>;
    type PrimeTest = MillerRabin;
    type Chal = i128;
    type Pt = i128;
    // Extra limb vs non-batched (3→4) to accommodate summing BATCH_SIZE=24
    // combined rows without overflow (need ceil(log2(24))=5 extra bits).
    type CombR = Int<{ INT_LIMBS * 4 }>;
    type Comb = Self::CombR;
    type EvalDotChal = ScalarProduct;
    type CombDotChal = ScalarProduct;
    type ArrCombRDotChal = MBSInnerProduct;
}

// IPRS code types for scalar i32 evaluations (same as zip_plus_benches.rs)
// F65537 B16 (BASE_LEN=16): row_len = 16 × 8^D
type IprsScalarB16<const DEPTH: usize, const CHECK: bool> =
    IprsCode<BenchZipScalarTypes, PnttConfigF2_16B16<DEPTH>, ScalarWideningMulByScalar<i128>, CHECK>;

// F65537 B64 (BASE_LEN=64): row_len = 64 × 8^D
type IprsScalarB64<const DEPTH: usize, const CHECK: bool> =
    IprsCode<BenchZipScalarTypes, PnttConfigF2_16B64<DEPTH>, ScalarWideningMulByScalar<i128>, CHECK>;

// F3329 rate 1/4 scalar i32 codes
type IprsScalarSmallR4B2<const DEPTH: usize, const CHECK: bool> =
    IprsCode<BenchZipScalarTypes, PnttConfigF3329R4B2<DEPTH>, ScalarWideningMulByScalar<i128>, CHECK>;
type IprsScalarSmallR4B4<const DEPTH: usize, const CHECK: bool> =
    IprsCode<BenchZipScalarTypes, PnttConfigF3329R4B4<DEPTH>, ScalarWideningMulByScalar<i128>, CHECK>;
type IprsScalarSmallR4B8<const DEPTH: usize, const CHECK: bool> =
    IprsCode<BenchZipScalarTypes, PnttConfigF3329R4B8<DEPTH>, ScalarWideningMulByScalar<i128>, CHECK>;

// F65537 rate 1/4 scalar i32 codes
type IprsScalarR4B16<const DEPTH: usize, const CHECK: bool> =
    IprsCode<BenchZipScalarTypes, PnttConfigF2_16R4B16<DEPTH>, ScalarWideningMulByScalar<i128>, CHECK>;
type IprsScalarR4B32<const DEPTH: usize, const CHECK: bool> =
    IprsCode<BenchZipScalarTypes, PnttConfigF2_16R4B32<DEPTH>, ScalarWideningMulByScalar<i128>, CHECK>;
type IprsScalarR4B64<const DEPTH: usize, const CHECK: bool> =
    IprsCode<BenchZipScalarTypes, PnttConfigF2_16R4B64<DEPTH>, ScalarWideningMulByScalar<i128>, CHECK>;

// F1179649 B16: scalar i32
type IprsScalarMidB16<const DEPTH: usize, const CHECK: bool> =
    IprsCode<BenchZipScalarTypes, PnttConfigF1179649B16<DEPTH>, ScalarWideningMulByScalar<i128>, CHECK>;

// F167772161 (5 × 2^25 + 1) scalar i32 codes
type IprsScalarLargeB32<const DEPTH: usize, const CHECK: bool> =
    IprsCode<BenchZipScalarTypes, PnttConfigF167772161<DEPTH>, ScalarWideningMulByScalar<i128>, CHECK>;
type IprsScalarLargeB64<const DEPTH: usize, const CHECK: bool> =
    IprsCode<BenchZipScalarTypes, PnttConfigF167772161B64<DEPTH>, ScalarWideningMulByScalar<i128>, CHECK>;

// ---------- ZipTypes for BPoly<31> benchmarks (matching PCS Pipeline Suite 1row) ----

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
    Int<6>: FromRef<CwCoeff>,
{
    const NUM_COLUMN_OPENINGS: usize = 147;
    type Eval = BinaryPoly<D_PLUS_ONE>;
    type Cw = DensePolynomial<CwCoeff, D_PLUS_ONE>;
    type Fmod = Uint<{ INT_LIMBS * 4 }>;
    type PrimeTest = MillerRabin;
    type Chal = i128;
    type Pt = i128;
    // Extra limb vs non-batched (5→6) to accommodate summing batch_size=5
    // combined rows without overflow (need ceil(log2(5))=3 extra bits).
    type CombR = Int<{ INT_LIMBS * 6 }>;
    type Comb = DensePolynomial<Self::CombR, D_PLUS_ONE>;
    type EvalDotChal = BinaryPolyInnerProduct<Self::Chal, D_PLUS_ONE>;
    type CombDotChal =
        DensePolyInnerProduct<Self::CombR, Self::Chal, Self::CombR, MBSInnerProduct, D_PLUS_ONE>;
    type ArrCombRDotChal = MBSInnerProduct;
}

// IPRS BPoly code types (matching zip_plus_benches.rs)
// F65537 B16 (BASE_LEN=16): row_len = 16 × 8^D
type IprsBPolyB16<Twiddle, const DEPTH: usize, const D_PLUS_ONE: usize, const CHECK: bool> =
    IprsCode<
        BenchZipPlusTypes<Twiddle, D_PLUS_ONE>,
        PnttConfigF2_16B16<DEPTH>,
        BinaryPolyWideningMulByScalar<Twiddle>,
        CHECK,
    >;

// F65537 B32 (BASE_LEN=32): row_len = 32 × 8^D
type IprsBPolyB32<Twiddle, const DEPTH: usize, const D_PLUS_ONE: usize, const CHECK: bool> =
    IprsCode<
        BenchZipPlusTypes<Twiddle, D_PLUS_ONE>,
        PnttConfigF2_16_1<DEPTH>,
        BinaryPolyWideningMulByScalar<Twiddle>,
        CHECK,
    >;

// F65537 B64 (BASE_LEN=64): row_len = 64 × 8^D
type IprsBPolyB64<Twiddle, const DEPTH: usize, const D_PLUS_ONE: usize, const CHECK: bool> =
    IprsCode<
        BenchZipPlusTypes<Twiddle, D_PLUS_ONE>,
        PnttConfigF2_16B64<DEPTH>,
        BinaryPolyWideningMulByScalar<Twiddle>,
        CHECK,
    >;

// F3329 rate 1/4 R4B2 (BASE_LEN=2, BASE_DIM=8): row_len = 2 × 8^D
type IprsBPolySmallR4B2<Twiddle, const DEPTH: usize, const D_PLUS_ONE: usize, const CHECK: bool> =
    IprsCode<
        BenchZipPlusTypes<Twiddle, D_PLUS_ONE>,
        PnttConfigF3329R4B2<DEPTH>,
        BinaryPolyWideningMulByScalar<Twiddle>,
        CHECK,
    >;

// F3329 rate 1/4 R4B4 (BASE_LEN=4, BASE_DIM=16): row_len = 4 × 8^D
type IprsBPolySmallR4B4<Twiddle, const DEPTH: usize, const D_PLUS_ONE: usize, const CHECK: bool> =
    IprsCode<
        BenchZipPlusTypes<Twiddle, D_PLUS_ONE>,
        PnttConfigF3329R4B4<DEPTH>,
        BinaryPolyWideningMulByScalar<Twiddle>,
        CHECK,
    >;

// F3329 rate 1/4 R4B8 (BASE_LEN=8, BASE_DIM=32): row_len = 8 × 8^D
type IprsBPolySmallR4B8<Twiddle, const DEPTH: usize, const D_PLUS_ONE: usize, const CHECK: bool> =
    IprsCode<
        BenchZipPlusTypes<Twiddle, D_PLUS_ONE>,
        PnttConfigF3329R4B8<DEPTH>,
        BinaryPolyWideningMulByScalar<Twiddle>,
        CHECK,
    >;

// F65537 rate 1/4 R4B16 (BASE_LEN=16, BASE_DIM=64): row_len = 16 × 8^D
type IprsBPolyR4B16<Twiddle, const DEPTH: usize, const D_PLUS_ONE: usize, const CHECK: bool> =
    IprsCode<
        BenchZipPlusTypes<Twiddle, D_PLUS_ONE>,
        PnttConfigF2_16R4B16<DEPTH>,
        BinaryPolyWideningMulByScalar<Twiddle>,
        CHECK,
    >;

// F65537 rate 1/4 R4B32 (BASE_LEN=32, BASE_DIM=128): row_len = 32 × 8^D
type IprsBPolyR4B32<Twiddle, const DEPTH: usize, const D_PLUS_ONE: usize, const CHECK: bool> =
    IprsCode<
        BenchZipPlusTypes<Twiddle, D_PLUS_ONE>,
        PnttConfigF2_16R4B32<DEPTH>,
        BinaryPolyWideningMulByScalar<Twiddle>,
        CHECK,
    >;

// F65537 rate 1/4 R4B64 (BASE_LEN=64, BASE_DIM=256): row_len = 64 × 8^D
type IprsBPolyR4B64<Twiddle, const DEPTH: usize, const D_PLUS_ONE: usize, const CHECK: bool> =
    IprsCode<
        BenchZipPlusTypes<Twiddle, D_PLUS_ONE>,
        PnttConfigF2_16R4B64<DEPTH>,
        BinaryPolyWideningMulByScalar<Twiddle>,
        CHECK,
    >;

// F1179649 (9 × 2^17 + 1) BPoly B16
type IprsBPolyMidB16<Twiddle, const DEPTH: usize, const D_PLUS_ONE: usize, const CHECK: bool> =
    IprsCode<
        BenchZipPlusTypes<Twiddle, D_PLUS_ONE>,
        PnttConfigF1179649B16<DEPTH>,
        BinaryPolyWideningMulByScalar<Twiddle>,
        CHECK,
    >;

// F167772161 (5 × 2^25 + 1) BPoly B32
type IprsBPolyLargeB32<Twiddle, const DEPTH: usize, const D_PLUS_ONE: usize, const CHECK: bool> =
    IprsCode<
        BenchZipPlusTypes<Twiddle, D_PLUS_ONE>,
        PnttConfigF167772161<DEPTH>,
        BinaryPolyWideningMulByScalar<Twiddle>,
        CHECK,
    >;

// F167772161 (5 × 2^25 + 1) BPoly B64
type IprsBPolyLargeB64<Twiddle, const DEPTH: usize, const D_PLUS_ONE: usize, const CHECK: bool> =
    IprsCode<
        BenchZipPlusTypes<Twiddle, D_PLUS_ONE>,
        PnttConfigF167772161B64<DEPTH>,
        BinaryPolyWideningMulByScalar<Twiddle>,
        CHECK,
    >;

/// Batch size: number of polynomials committed together in the batched PCS.
const BATCH_SIZE: usize = 24;

// ---------- Criterion entry point ----------

/// Batched PCS pipeline suite for scalar i32, rate 1/4 IPRS codes.
/// Mirrors the "PCS Pipeline Suite i32 1row" configs but uses BatchedZipPlus
/// with `BATCH_SIZE` polynomials sharing a single Merkle tree.
///
/// poly_size (2^P)   Config                Field        row_len     num_rows
/// ───────────────   ──────                ─────        ───────     ────────
/// 2^4  = 16         R4B2 D=1 (rate 1/4)   F3329        16          1
/// 2^5  = 32         R4B4 D=1 (rate 1/4)   F3329        32          1
/// 2^6  = 64         R4B8 D=1 (rate 1/4)   F3329        64          1
/// 2^10 = 1024       R4B16 D=2 (rate 1/4)  F65537       1024        1
/// 2^10 = 1024       R4B64 D=1 (rate 1/4)  F65537       512         2
/// 2^11 = 2048       R4B16 D=2 (rate 1/4)  F65537       1024        2
/// 2^12 = 4096       R4B16 D=2 (rate 1/4)  F65537       1024        4
/// 2^13 = 8192       R4B32 D=2 (rate 1/4)  F65537       2048        4
fn batched_pcs_pipeline_suite_scalar_1row(c: &mut Criterion) {
    use zip_common::*;

    let mut group = c.benchmark_group("Batched PCS Pipeline Suite i32 1row");
    group.sample_size(10);

    // ── Encode ───────────────────────────────────────────────────────
    // F3329 rate 1/4: small row lengths (P=4,5,6)
    batched_encode_nrows::<BenchZipScalarTypes, IprsScalarSmallR4B2<1, UNCHECKED>, 4>(&mut group, 1, BATCH_SIZE);
    batched_encode_nrows::<BenchZipScalarTypes, IprsScalarSmallR4B4<1, UNCHECKED>, 5>(&mut group, 1, BATCH_SIZE);
    batched_encode_nrows::<BenchZipScalarTypes, IprsScalarSmallR4B8<1, UNCHECKED>, 6>(&mut group, 1, BATCH_SIZE);

    // F65537 rate 1/4: medium row lengths (P=10,11,12,13)
    batched_encode_nrows::<BenchZipScalarTypes, IprsScalarR4B16<2, UNCHECKED>, 10>(&mut group, 1, BATCH_SIZE);
    batched_encode_nrows::<BenchZipScalarTypes, IprsScalarR4B64<1, UNCHECKED>, 10>(&mut group, 2, BATCH_SIZE);
    batched_encode_nrows::<BenchZipScalarTypes, IprsScalarR4B16<2, UNCHECKED>, 11>(&mut group, 2, BATCH_SIZE);
    batched_encode_nrows::<BenchZipScalarTypes, IprsScalarR4B16<2, UNCHECKED>, 12>(&mut group, 4, BATCH_SIZE);
    batched_encode_nrows::<BenchZipScalarTypes, IprsScalarR4B32<2, UNCHECKED>, 13>(&mut group, 4, BATCH_SIZE);

    // ── Merkle ───────────────────────────────────────────────────────
    // F3329 rate 1/4: small row lengths (P=4,5,6)
    batched_merkle_nrows::<BenchZipScalarTypes, IprsScalarSmallR4B2<1, UNCHECKED>, 4>(&mut group, 1, BATCH_SIZE);
    batched_merkle_nrows::<BenchZipScalarTypes, IprsScalarSmallR4B4<1, UNCHECKED>, 5>(&mut group, 1, BATCH_SIZE);
    batched_merkle_nrows::<BenchZipScalarTypes, IprsScalarSmallR4B8<1, UNCHECKED>, 6>(&mut group, 1, BATCH_SIZE);

    // F65537 rate 1/4: medium row lengths (P=10,11,12,13)
    batched_merkle_nrows::<BenchZipScalarTypes, IprsScalarR4B16<2, UNCHECKED>, 10>(&mut group, 1, BATCH_SIZE);
    batched_merkle_nrows::<BenchZipScalarTypes, IprsScalarR4B64<1, UNCHECKED>, 10>(&mut group, 2, BATCH_SIZE);
    batched_merkle_nrows::<BenchZipScalarTypes, IprsScalarR4B16<2, UNCHECKED>, 11>(&mut group, 2, BATCH_SIZE);
    batched_merkle_nrows::<BenchZipScalarTypes, IprsScalarR4B16<2, UNCHECKED>, 12>(&mut group, 4, BATCH_SIZE);
    batched_merkle_nrows::<BenchZipScalarTypes, IprsScalarR4B32<2, UNCHECKED>, 13>(&mut group, 4, BATCH_SIZE);

    // ── Commit ───────────────────────────────────────────────────────
    // F3329 rate 1/4: small row lengths (P=4,5,6)
    batched_commit_nrows::<BenchZipScalarTypes, IprsScalarSmallR4B2<1, UNCHECKED>, 4>(&mut group, 1, BATCH_SIZE);
    batched_commit_nrows::<BenchZipScalarTypes, IprsScalarSmallR4B4<1, UNCHECKED>, 5>(&mut group, 1, BATCH_SIZE);
    batched_commit_nrows::<BenchZipScalarTypes, IprsScalarSmallR4B8<1, UNCHECKED>, 6>(&mut group, 1, BATCH_SIZE);

    // F65537 rate 1/4: medium row lengths (P=10,11,12,13)
    batched_commit_nrows::<BenchZipScalarTypes, IprsScalarR4B16<2, UNCHECKED>, 10>(&mut group, 1, BATCH_SIZE);
    batched_commit_nrows::<BenchZipScalarTypes, IprsScalarR4B64<1, UNCHECKED>, 10>(&mut group, 2, BATCH_SIZE);
    batched_commit_nrows::<BenchZipScalarTypes, IprsScalarR4B16<2, UNCHECKED>, 11>(&mut group, 2, BATCH_SIZE);
    batched_commit_nrows::<BenchZipScalarTypes, IprsScalarR4B16<2, UNCHECKED>, 12>(&mut group, 4, BATCH_SIZE);
    batched_commit_nrows::<BenchZipScalarTypes, IprsScalarR4B32<2, UNCHECKED>, 13>(&mut group, 4, BATCH_SIZE);

    // ── Test ─────────────────────────────────────────────────────────
    // F3329 rate 1/4
    batched_test_nrows::<BenchZipScalarTypes, IprsScalarSmallR4B2<1, UNCHECKED>, UNCHECKED, 4>(&mut group, 1, BATCH_SIZE);
    batched_test_nrows::<BenchZipScalarTypes, IprsScalarSmallR4B4<1, UNCHECKED>, UNCHECKED, 5>(&mut group, 1, BATCH_SIZE);
    batched_test_nrows::<BenchZipScalarTypes, IprsScalarSmallR4B8<1, UNCHECKED>, UNCHECKED, 6>(&mut group, 1, BATCH_SIZE);

    // F65537 rate 1/4
    batched_test_nrows::<BenchZipScalarTypes, IprsScalarR4B16<2, UNCHECKED>, UNCHECKED, 10>(&mut group, 1, BATCH_SIZE);
    batched_test_nrows::<BenchZipScalarTypes, IprsScalarR4B64<1, UNCHECKED>, UNCHECKED, 10>(&mut group, 2, BATCH_SIZE);
    batched_test_nrows::<BenchZipScalarTypes, IprsScalarR4B16<2, UNCHECKED>, UNCHECKED, 11>(&mut group, 2, BATCH_SIZE);
    batched_test_nrows::<BenchZipScalarTypes, IprsScalarR4B16<2, UNCHECKED>, UNCHECKED, 12>(&mut group, 4, BATCH_SIZE);
    batched_test_nrows::<BenchZipScalarTypes, IprsScalarR4B32<2, UNCHECKED>, UNCHECKED, 13>(&mut group, 4, BATCH_SIZE);

    // ── Verify ───────────────────────────────────────────────────────
    // F3329 rate 1/4
    batched_verify_nrows::<BenchZipScalarTypes, IprsScalarSmallR4B2<1, UNCHECKED>, UNCHECKED, 4>(&mut group, 1, BATCH_SIZE);
    batched_verify_nrows::<BenchZipScalarTypes, IprsScalarSmallR4B4<1, UNCHECKED>, UNCHECKED, 5>(&mut group, 1, BATCH_SIZE);
    batched_verify_nrows::<BenchZipScalarTypes, IprsScalarSmallR4B8<1, UNCHECKED>, UNCHECKED, 6>(&mut group, 1, BATCH_SIZE);

    // F65537 rate 1/4
    batched_verify_nrows::<BenchZipScalarTypes, IprsScalarR4B16<2, UNCHECKED>, UNCHECKED, 10>(&mut group, 1, BATCH_SIZE);
    batched_verify_nrows::<BenchZipScalarTypes, IprsScalarR4B64<1, UNCHECKED>, UNCHECKED, 10>(&mut group, 2, BATCH_SIZE);
    batched_verify_nrows::<BenchZipScalarTypes, IprsScalarR4B16<2, UNCHECKED>, UNCHECKED, 11>(&mut group, 2, BATCH_SIZE);
    batched_verify_nrows::<BenchZipScalarTypes, IprsScalarR4B16<2, UNCHECKED>, UNCHECKED, 12>(&mut group, 4, BATCH_SIZE);
    batched_verify_nrows::<BenchZipScalarTypes, IprsScalarR4B32<2, UNCHECKED>, UNCHECKED, 13>(&mut group, 4, BATCH_SIZE);

    group.finish();
}

/// Batched PCS pipeline suite for BPoly<31>, various IPRS codes, num_rows=1.
/// Mirrors the "PCS Pipeline Suite BPoly31 1row" configs but uses BatchedZipPlus
/// with 5 polynomials sharing a single Merkle tree.
///
/// poly_size (2^P)   Config                 Field        row_len
/// ───────────────   ──────                 ─────        ───────
/// 2^4  = 16         R4B2 D=1 (rate 1/4)    F3329        16
/// 2^5  = 32         R4B4 D=1 (rate 1/4)    F3329        32
/// 2^6  = 64         R4B8 D=1 (rate 1/4)    F3329        64
/// 2^7  = 128        R4B16 D=1 (rate 1/4)   F65537       128
/// 2^8  = 256        R4B32 D=1 (rate 1/4)   F65537       256
/// 2^9  = 512        R4B64 D=1 (rate 1/4)   F65537       512
/// 2^10 = 1024       R4B16 D=2 (rate 1/4)   F65537       1024
/// 2^11 = 2048       R4B32 D=2 (rate 1/4)   F65537       2048
/// 2^12 = 4096       R4B64 D=2 (rate 1/4)   F65537       4096
/// 2^13 = 8192       R4B16 D=3 (rate 1/4)   F65537       8192
/// 2^14 = 16384      B32 D=3 (rate 1/2)     F65537       16384
/// 2^16 = 65536      B16 D=4 (rate 1/2)     F1179649     65536
/// 2^17 = 131072     B32 D=4 (rate 1/2)     F167772161   131072
/// 2^18 = 262144     B64 D=4 (rate 1/2)     F167772161   262144
fn batched_pcs_pipeline_suite_bpoly31_1row(c: &mut Criterion) {
    use zip_common::*;

    let mut group = c.benchmark_group("Batched PCS Pipeline Suite BPoly31 1row");
    group.sample_size(10);

    // ── Encode ───────────────────────────────────────────────────────
    // F3329 rate 1/4: small row lengths (P=4,5,6)
    batched_encode_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolySmallR4B2<i64, 1, 32, UNCHECKED>, 4>(&mut group, 1, 5);
    batched_encode_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolySmallR4B4<i64, 1, 32, UNCHECKED>, 5>(&mut group, 1, 5);
    batched_encode_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolySmallR4B8<i64, 1, 32, UNCHECKED>, 6>(&mut group, 1, 5);

    // F65537 rate 1/4: medium row lengths (P=7..13)
    batched_encode_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B16<i64, 1, 32, UNCHECKED>,  7>(&mut group, 1, 5);
    batched_encode_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B32<i64, 1, 32, UNCHECKED>,  8>(&mut group, 1, 5);
    batched_encode_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B64<i64, 1, 32, UNCHECKED>,  9>(&mut group, 1, 5);
    batched_encode_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B16<i64, 2, 32, UNCHECKED>, 10>(&mut group, 1, 5);
    batched_encode_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B32<i64, 2, 32, UNCHECKED>, 11>(&mut group, 1, 5);
    batched_encode_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B64<i64, 2, 32, UNCHECKED>, 12>(&mut group, 1, 5);
    batched_encode_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B16<i64, 3, 32, UNCHECKED>, 13>(&mut group, 1, 5);

    // F65537 rate 1/2 (P=14)
    batched_encode_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyB32<i64, 3, 32, UNCHECKED>, 14>(&mut group, 1, 5);

    // F1179649: larger row lengths (P=16)
    batched_encode_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyMidB16<i64, 4, 32, UNCHECKED>, 16>(&mut group, 1, 5);

    // F167772161: largest row lengths (P=17,18)
    batched_encode_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyLargeB32<i64, 4, 32, UNCHECKED>, 17>(&mut group, 1, 5);
    batched_encode_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyLargeB64<i64, 4, 32, UNCHECKED>, 18>(&mut group, 1, 5);

    // ── Merkle ───────────────────────────────────────────────────────
    // F3329 rate 1/4: small row lengths (P=4,5,6)
    batched_merkle_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolySmallR4B2<i64, 1, 32, UNCHECKED>, 4>(&mut group, 1, 5);
    batched_merkle_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolySmallR4B4<i64, 1, 32, UNCHECKED>, 5>(&mut group, 1, 5);
    batched_merkle_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolySmallR4B8<i64, 1, 32, UNCHECKED>, 6>(&mut group, 1, 5);

    // F65537 rate 1/4
    batched_merkle_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B16<i64, 1, 32, UNCHECKED>,  7>(&mut group, 1, 5);
    batched_merkle_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B32<i64, 1, 32, UNCHECKED>,  8>(&mut group, 1, 5);
    batched_merkle_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B64<i64, 1, 32, UNCHECKED>,  9>(&mut group, 1, 5);
    batched_merkle_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B16<i64, 2, 32, UNCHECKED>, 10>(&mut group, 1, 5);
    batched_merkle_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B32<i64, 2, 32, UNCHECKED>, 11>(&mut group, 1, 5);
    batched_merkle_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B64<i64, 2, 32, UNCHECKED>, 12>(&mut group, 1, 5);
    batched_merkle_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B16<i64, 3, 32, UNCHECKED>, 13>(&mut group, 1, 5);

    // F65537 rate 1/2
    batched_merkle_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyB32<i64, 3, 32, UNCHECKED>, 14>(&mut group, 1, 5);

    // F1179649
    batched_merkle_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyMidB16<i64, 4, 32, UNCHECKED>, 16>(&mut group, 1, 5);

    // F167772161
    batched_merkle_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyLargeB32<i64, 4, 32, UNCHECKED>, 17>(&mut group, 1, 5);
    batched_merkle_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyLargeB64<i64, 4, 32, UNCHECKED>, 18>(&mut group, 1, 5);

    // ── Commit ───────────────────────────────────────────────────────
    // F3329 rate 1/4
    batched_commit_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolySmallR4B2<i64, 1, 32, UNCHECKED>, 4>(&mut group, 1, 5);
    batched_commit_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolySmallR4B4<i64, 1, 32, UNCHECKED>, 5>(&mut group, 1, 5);
    batched_commit_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolySmallR4B8<i64, 1, 32, UNCHECKED>, 6>(&mut group, 1, 5);

    // F65537 rate 1/4
    batched_commit_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B16<i64, 1, 32, UNCHECKED>,  7>(&mut group, 1, 5);
    batched_commit_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B32<i64, 1, 32, UNCHECKED>,  8>(&mut group, 1, 5);
    batched_commit_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B64<i64, 1, 32, UNCHECKED>,  9>(&mut group, 1, 5);
    batched_commit_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B16<i64, 2, 32, UNCHECKED>, 10>(&mut group, 1, 5);
    batched_commit_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B32<i64, 2, 32, UNCHECKED>, 11>(&mut group, 1, 5);
    batched_commit_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B64<i64, 2, 32, UNCHECKED>, 12>(&mut group, 1, 5);
    batched_commit_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B16<i64, 3, 32, UNCHECKED>, 13>(&mut group, 1, 5);

    // F65537 rate 1/2
    batched_commit_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyB32<i64, 3, 32, UNCHECKED>, 14>(&mut group, 1, 5);

    // F1179649
    batched_commit_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyMidB16<i64, 4, 32, UNCHECKED>, 16>(&mut group, 1, 5);

    // F167772161
    batched_commit_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyLargeB32<i64, 4, 32, UNCHECKED>, 17>(&mut group, 1, 5);
    batched_commit_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyLargeB64<i64, 4, 32, UNCHECKED>, 18>(&mut group, 1, 5);

    // ── Test ─────────────────────────────────────────────────────────
    // F3329 rate 1/4
    batched_test_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolySmallR4B2<i64, 1, 32, UNCHECKED>, UNCHECKED, 4>(&mut group, 1, 5);
    batched_test_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolySmallR4B4<i64, 1, 32, UNCHECKED>, UNCHECKED, 5>(&mut group, 1, 5);
    batched_test_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolySmallR4B8<i64, 1, 32, UNCHECKED>, UNCHECKED, 6>(&mut group, 1, 5);

    // F65537 rate 1/4
    batched_test_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B16<i64, 1, 32, UNCHECKED>, UNCHECKED,  7>(&mut group, 1, 5);
    batched_test_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B32<i64, 1, 32, UNCHECKED>, UNCHECKED,  8>(&mut group, 1, 5);
    batched_test_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B64<i64, 1, 32, UNCHECKED>, UNCHECKED,  9>(&mut group, 1, 5);
    batched_test_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B16<i64, 2, 32, UNCHECKED>, UNCHECKED, 10>(&mut group, 1, 5);
    batched_test_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B32<i64, 2, 32, UNCHECKED>, UNCHECKED, 11>(&mut group, 1, 5);
    batched_test_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B64<i64, 2, 32, UNCHECKED>, UNCHECKED, 12>(&mut group, 1, 5);
    batched_test_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B16<i64, 3, 32, UNCHECKED>, UNCHECKED, 13>(&mut group, 1, 5);

    // F65537 rate 1/2
    batched_test_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyB32<i64, 3, 32, UNCHECKED>, UNCHECKED, 14>(&mut group, 1, 5);

    // F1179649
    batched_test_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyMidB16<i64, 4, 32, UNCHECKED>, UNCHECKED, 16>(&mut group, 1, 5);

    // F167772161
    batched_test_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyLargeB32<i64, 4, 32, UNCHECKED>, UNCHECKED, 17>(&mut group, 1, 5);
    batched_test_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyLargeB64<i64, 4, 32, UNCHECKED>, UNCHECKED, 18>(&mut group, 1, 5);

    // ── Verify ───────────────────────────────────────────────────────
    // F3329 rate 1/4
    batched_verify_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolySmallR4B2<i64, 1, 32, UNCHECKED>, UNCHECKED, 4>(&mut group, 1, 5);
    batched_verify_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolySmallR4B4<i64, 1, 32, UNCHECKED>, UNCHECKED, 5>(&mut group, 1, 5);
    batched_verify_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolySmallR4B8<i64, 1, 32, UNCHECKED>, UNCHECKED, 6>(&mut group, 1, 5);

    // F65537 rate 1/4
    batched_verify_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B16<i64, 1, 32, UNCHECKED>, UNCHECKED,  7>(&mut group, 1, 5);
    batched_verify_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B32<i64, 1, 32, UNCHECKED>, UNCHECKED,  8>(&mut group, 1, 5);
    batched_verify_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B64<i64, 1, 32, UNCHECKED>, UNCHECKED,  9>(&mut group, 1, 5);
    batched_verify_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B16<i64, 2, 32, UNCHECKED>, UNCHECKED, 10>(&mut group, 1, 5);
    batched_verify_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B32<i64, 2, 32, UNCHECKED>, UNCHECKED, 11>(&mut group, 1, 5);
    batched_verify_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B64<i64, 2, 32, UNCHECKED>, UNCHECKED, 12>(&mut group, 1, 5);
    batched_verify_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyR4B16<i64, 3, 32, UNCHECKED>, UNCHECKED, 13>(&mut group, 1, 5);

    // F65537 rate 1/2
    batched_verify_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyB32<i64, 3, 32, UNCHECKED>, UNCHECKED, 14>(&mut group, 1, 5);

    // F1179649
    batched_verify_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyMidB16<i64, 4, 32, UNCHECKED>, UNCHECKED, 16>(&mut group, 1, 5);

    // F167772161
    batched_verify_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyLargeB32<i64, 4, 32, UNCHECKED>, UNCHECKED, 17>(&mut group, 1, 5);
    batched_verify_nrows::<BenchZipPlusTypes<i64, 32>, IprsBPolyLargeB64<i64, 4, 32, UNCHECKED>, UNCHECKED, 18>(&mut group, 1, 5);

    group.finish();
}

criterion_group!(benches, batched_pcs_pipeline_suite_scalar_1row, batched_pcs_pipeline_suite_bpoly31_1row);
criterion_main!(benches);
