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
            IprsCode, PnttConfigF2_16_1, PnttConfigF2_16B16, PnttConfigF2_16B64,
            PnttConfigF1179649B16,
            PnttConfigF3329B8, PnttConfigF3329B16,
            PnttConfigF3329R4B2, PnttConfigF3329R4B4, PnttConfigF3329R4B8,
            PnttConfigF167772161, PnttConfigF167772161B16, PnttConfigF167772161B64,
            PnttConfigF7340033B16, PnttConfigF7340033B32, PnttConfigF7340033B64,
        },
        raa::{RaaCode, RaaConfig},
    },
    pcs::structs::{ZipPlus, ZipTypes},
};

const INT_LIMBS: usize = U64::LIMBS;

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
    const CHECK_FOR_OVERFLOWS: bool = UNCHECKED;
}

/// ZipTypes implementation for scalar i32 evaluations (non-polynomial)
struct BenchZipScalarTypes;

impl ZipTypes for BenchZipScalarTypes {
    const NUM_COLUMN_OPENINGS: usize = 147;
    type Eval = i32;
    type Cw = i128;  // Use i128 to avoid overflow when accumulating products
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

// IPRS code types for scalar i32 evaluations
type IprsScalarB16<const DEPTH: usize, const CHECK: bool> =
    IprsCode<BenchZipScalarTypes, PnttConfigF2_16B16<DEPTH>, ScalarWideningMulByScalar<i128>, CHECK>;

// F65537 B32 (BASE_LEN=32, BASE_DIM=64): row_len = 32 × 8^D
type IprsScalarB32<const DEPTH: usize, const CHECK: bool> =
    IprsCode<BenchZipScalarTypes, PnttConfigF2_16_1<DEPTH>, ScalarWideningMulByScalar<i128>, CHECK>;

// F65537 B64 (BASE_LEN=64, BASE_DIM=128): row_len = 64 × 8^D
type IprsScalarB64<const DEPTH: usize, const CHECK: bool> =
    IprsCode<BenchZipScalarTypes, PnttConfigF2_16B64<DEPTH>, ScalarWideningMulByScalar<i128>, CHECK>;

// F3329 B8 (BASE_LEN=8, BASE_DIM=16): row_len = 8 × 8^D
type IprsScalarSmallB8<const DEPTH: usize, const CHECK: bool> =
    IprsCode<BenchZipScalarTypes, PnttConfigF3329B8<DEPTH>, ScalarWideningMulByScalar<i128>, CHECK>;

// F3329 B16 (BASE_LEN=16, BASE_DIM=32): row_len = 16 × 8^D
type IprsScalarSmallB16<const DEPTH: usize, const CHECK: bool> =
    IprsCode<BenchZipScalarTypes, PnttConfigF3329B16<DEPTH>, ScalarWideningMulByScalar<i128>, CHECK>;

// F1179649 B16: scalar i32 safe up to D=3 (~120 bits)
type IprsScalarMidB16<const DEPTH: usize, const CHECK: bool> =
    IprsCode<BenchZipScalarTypes, PnttConfigF1179649B16<DEPTH>, ScalarWideningMulByScalar<i128>, CHECK>;

// F167772161 (5 × 2^25 + 1) scalar i32 codes: NTT domain up to 2^25
// B16: row_len = 16 × 8^D (D=5 → 524288 = 2^19)
type IprsScalarLargeB16<const DEPTH: usize, const CHECK: bool> =
    IprsCode<BenchZipScalarTypes, PnttConfigF167772161B16<DEPTH>, ScalarWideningMulByScalar<i128>, CHECK>;

// B32: row_len = 32 × 8^D (D=4 → 131072 = 2^17)
type IprsScalarLargeB32<const DEPTH: usize, const CHECK: bool> =
    IprsCode<BenchZipScalarTypes, PnttConfigF167772161<DEPTH>, ScalarWideningMulByScalar<i128>, CHECK>;

// B64: row_len = 64 × 8^D (D=4 → 262144 = 2^18)
type IprsScalarLargeB64<const DEPTH: usize, const CHECK: bool> =
    IprsCode<BenchZipScalarTypes, PnttConfigF167772161B64<DEPTH>, ScalarWideningMulByScalar<i128>, CHECK>;

// F3329 rate 1/4 scalar i32 codes
// R4B2 (BASE_LEN=2, BASE_DIM=8): row_len = 2 × 8^D
type IprsScalarSmallR4B2<const DEPTH: usize, const CHECK: bool> =
    IprsCode<BenchZipScalarTypes, PnttConfigF3329R4B2<DEPTH>, ScalarWideningMulByScalar<i128>, CHECK>;

// R4B4 (BASE_LEN=4, BASE_DIM=16): row_len = 4 × 8^D
type IprsScalarSmallR4B4<const DEPTH: usize, const CHECK: bool> =
    IprsCode<BenchZipScalarTypes, PnttConfigF3329R4B4<DEPTH>, ScalarWideningMulByScalar<i128>, CHECK>;

// R4B8 (BASE_LEN=8, BASE_DIM=32): row_len = 8 × 8^D
type IprsScalarSmallR4B8<const DEPTH: usize, const CHECK: bool> =
    IprsCode<BenchZipScalarTypes, PnttConfigF3329R4B8<DEPTH>, ScalarWideningMulByScalar<i128>, CHECK>;

#[allow(dead_code)]
type SomeRaaCode<const D_PLUS_ONE: usize> =
    RaaCode<BenchZipPlusTypes<i32, D_PLUS_ONE>, BenchRaaConfig, 4>;

// All IPRS codes use BASE_LEN=16, BASE_DIM=32 (rate 1/2)
// F65537 (2^16+1): Row lengths 128 (D=1), 1024 (D=2), 8192 (D=3)
type IprsB16<Twiddle, const DEPTH: usize, const D_PLUS_ONE: usize, const CHECK: bool> =
    IprsCode<
        BenchZipPlusTypes<Twiddle, D_PLUS_ONE>,
        PnttConfigF2_16B16<DEPTH>,
        BinaryPolyWideningMulByScalar<Twiddle>,
        CHECK,
    >;

// F65537 B32 (BASE_LEN=32): row_len = 32 × 8^D
type IprsB32<Twiddle, const DEPTH: usize, const D_PLUS_ONE: usize, const CHECK: bool> =
    IprsCode<
        BenchZipPlusTypes<Twiddle, D_PLUS_ONE>,
        PnttConfigF2_16_1<DEPTH>,
        BinaryPolyWideningMulByScalar<Twiddle>,
        CHECK,
    >;

// F65537 B64 (BASE_LEN=64): row_len = 64 × 8^D
type IprsB64<Twiddle, const DEPTH: usize, const D_PLUS_ONE: usize, const CHECK: bool> =
    IprsCode<
        BenchZipPlusTypes<Twiddle, D_PLUS_ONE>,
        PnttConfigF2_16B64<DEPTH>,
        BinaryPolyWideningMulByScalar<Twiddle>,
        CHECK,
    >;

// F3329 B8 (BASE_LEN=8): row_len = 8 × 8^D
type IprsSmallB8<Twiddle, const DEPTH: usize, const D_PLUS_ONE: usize, const CHECK: bool> =
    IprsCode<
        BenchZipPlusTypes<Twiddle, D_PLUS_ONE>,
        PnttConfigF3329B8<DEPTH>,
        BinaryPolyWideningMulByScalar<Twiddle>,
        CHECK,
    >;

// F3329 B16 (BASE_LEN=16): row_len = 16 × 8^D
type IprsSmallB16<Twiddle, const DEPTH: usize, const D_PLUS_ONE: usize, const CHECK: bool> =
    IprsCode<
        BenchZipPlusTypes<Twiddle, D_PLUS_ONE>,
        PnttConfigF3329B16<DEPTH>,
        BinaryPolyWideningMulByScalar<Twiddle>,
        CHECK,
    >;
// F3329 rate 1/4 B2 (BASE_LEN=2, BASE_DIM=8): row_len = 2 × 8^D
type IprsSmallR4B2<Twiddle, const DEPTH: usize, const D_PLUS_ONE: usize, const CHECK: bool> =
    IprsCode<
        BenchZipPlusTypes<Twiddle, D_PLUS_ONE>,
        PnttConfigF3329R4B2<DEPTH>,
        BinaryPolyWideningMulByScalar<Twiddle>,
        CHECK,
    >;

// F3329 rate 1/4 B4 (BASE_LEN=4, BASE_DIM=16): row_len = 4 × 8^D
type IprsSmallR4B4<Twiddle, const DEPTH: usize, const D_PLUS_ONE: usize, const CHECK: bool> =
    IprsCode<
        BenchZipPlusTypes<Twiddle, D_PLUS_ONE>,
        PnttConfigF3329R4B4<DEPTH>,
        BinaryPolyWideningMulByScalar<Twiddle>,
        CHECK,
    >;

// F3329 rate 1/4 B8 (BASE_LEN=8, BASE_DIM=32): row_len = 8 × 8^D
type IprsSmallR4B8<Twiddle, const DEPTH: usize, const D_PLUS_ONE: usize, const CHECK: bool> =
    IprsCode<
        BenchZipPlusTypes<Twiddle, D_PLUS_ONE>,
        PnttConfigF3329R4B8<DEPTH>,
        BinaryPolyWideningMulByScalar<Twiddle>,
        CHECK,
    >;
// F1179649 (9 × 2^17 + 1): Row lengths 128 (D=1), 1024 (D=2), 8192 (D=3), 65536 (D=4)
type IprsMidB16<Twiddle, const DEPTH: usize, const D_PLUS_ONE: usize, const CHECK: bool> =
    IprsCode<
        BenchZipPlusTypes<Twiddle, D_PLUS_ONE>,
        PnttConfigF1179649B16<DEPTH>,
        BinaryPolyWideningMulByScalar<Twiddle>,
        CHECK,
    >;

// F167772161 (5 × 2^25 + 1) BPoly codes: NTT domain up to 2^25
type IprsLargeB16<Twiddle, const DEPTH: usize, const D_PLUS_ONE: usize, const CHECK: bool> =
    IprsCode<
        BenchZipPlusTypes<Twiddle, D_PLUS_ONE>,
        PnttConfigF167772161B16<DEPTH>,
        BinaryPolyWideningMulByScalar<Twiddle>,
        CHECK,
    >;

type IprsLargeB32<Twiddle, const DEPTH: usize, const D_PLUS_ONE: usize, const CHECK: bool> =
    IprsCode<
        BenchZipPlusTypes<Twiddle, D_PLUS_ONE>,
        PnttConfigF167772161<DEPTH>,
        BinaryPolyWideningMulByScalar<Twiddle>,
        CHECK,
    >;

type IprsLargeB64<Twiddle, const DEPTH: usize, const D_PLUS_ONE: usize, const CHECK: bool> =
    IprsCode<
        BenchZipPlusTypes<Twiddle, D_PLUS_ONE>,
        PnttConfigF167772161B64<DEPTH>,
        BinaryPolyWideningMulByScalar<Twiddle>,
        CHECK,
    >;

// F7340033 (7 × 2^20 + 1) BPoly codes: NTT domain up to 2^20
type IprsMid2B16<Twiddle, const DEPTH: usize, const D_PLUS_ONE: usize, const CHECK: bool> =
    IprsCode<
        BenchZipPlusTypes<Twiddle, D_PLUS_ONE>,
        PnttConfigF7340033B16<DEPTH>,
        BinaryPolyWideningMulByScalar<Twiddle>,
        CHECK,
    >;

type IprsMid2B32<Twiddle, const DEPTH: usize, const D_PLUS_ONE: usize, const CHECK: bool> =
    IprsCode<
        BenchZipPlusTypes<Twiddle, D_PLUS_ONE>,
        PnttConfigF7340033B32<DEPTH>,
        BinaryPolyWideningMulByScalar<Twiddle>,
        CHECK,
    >;

type IprsMid2B64<Twiddle, const DEPTH: usize, const D_PLUS_ONE: usize, const CHECK: bool> =
    IprsCode<
        BenchZipPlusTypes<Twiddle, D_PLUS_ONE>,
        PnttConfigF7340033B64<DEPTH>,
        BinaryPolyWideningMulByScalar<Twiddle>,
        CHECK,
    >;

#[allow(dead_code)]
fn zip_plus_benchmarks_raa(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip+ RAA");

    do_bench::<BenchZipPlusTypes<i32, 32>, SomeRaaCode<_>, UNCHECKED>(&mut group, "RAA");
    do_bench::<BenchZipPlusTypes<i32, 64>, SomeRaaCode<_>, UNCHECKED>(&mut group, "RAA");

    group.finish();
}

fn zip_plus_benchmarks_iprs(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip+ Encode");

    // ========== BPoly<31> benchmarks (BASE_LEN=16, Cw coeff = i128) ==========
    // F65537: DEPTH 1-3 (max ~72 bits after D=3)
    encode_single_row::<BenchZipPlusTypes<i128, 32>, IprsB16<i128, 1, 32, UNCHECKED>, 128>(&mut group, "IPRS-1-1_2-F65537");
    encode_single_row::<BenchZipPlusTypes<i128, 32>, IprsB16<i128, 2, 32, UNCHECKED>, 1024>(&mut group, "IPRS-2-1_2-F65537");
    encode_single_row::<BenchZipPlusTypes<i128, 32>, IprsB16<i128, 3, 32, UNCHECKED>, 8192>(&mut group, "IPRS-3-1_2-F65537");
    // F1179649: DEPTH 4 (max ~111 bits)
    encode_single_row::<BenchZipPlusTypes<i128, 32>, IprsMidB16<i128, 4, 32, UNCHECKED>, 65536>(&mut group, "IPRS-4-1_2-F1179649");

    // ========== Degree-0 (constant) polynomial benchmarks ==========
    // F65537: DEPTH 1-3
    encode_single_row::<BenchZipPlusTypes<i128, 1>, IprsB16<i128, 1, 1, UNCHECKED>, 128>(&mut group, "IPRS-1-1_2-F65537-Deg0");
    encode_single_row::<BenchZipPlusTypes<i128, 1>, IprsB16<i128, 2, 1, UNCHECKED>, 1024>(&mut group, "IPRS-2-1_2-F65537-Deg0");
    encode_single_row::<BenchZipPlusTypes<i128, 1>, IprsB16<i128, 3, 1, UNCHECKED>, 8192>(&mut group, "IPRS-3-1_2-F65537-Deg0");
    // F1179649: DEPTH 4
    encode_single_row::<BenchZipPlusTypes<i128, 1>, IprsMidB16<i128, 4, 1, UNCHECKED>, 65536>(&mut group, "IPRS-4-1_2-F1179649-Deg0");

    // ========== Scalar i32 benchmarks (Cw = i128) ==========
    // B16 F65537: DEPTH 1-3 (row_len 128, 1024, 8192; max ~103 bits)
    encode_single_row::<BenchZipScalarTypes, IprsScalarB16<1, UNCHECKED>, 128>(&mut group, "IPRS-1-1_2-F65537-i32");
    encode_single_row::<BenchZipScalarTypes, IprsScalarB16<2, UNCHECKED>, 1024>(&mut group, "IPRS-2-1_2-F65537-i32");
    encode_single_row::<BenchZipScalarTypes, IprsScalarB16<3, UNCHECKED>, 8192>(&mut group, "IPRS-3-1_2-F65537-i32");
    // B16 F1179649: DEPTH 1-3 (row_len 128, 1024, 8192; max ~120 bits)
    encode_single_row::<BenchZipScalarTypes, IprsScalarMidB16<1, UNCHECKED>, 128>(&mut group, "IPRS-1-1_2-F1179649-i32");
    encode_single_row::<BenchZipScalarTypes, IprsScalarMidB16<2, UNCHECKED>, 1024>(&mut group, "IPRS-2-1_2-F1179649-i32");
    encode_single_row::<BenchZipScalarTypes, IprsScalarMidB16<3, UNCHECKED>, 8192>(&mut group, "IPRS-3-1_2-F1179649-i32");
    // B32 F65537: D=3 -> row_len=16384=2^14 (~104 bits)
    encode_single_row::<BenchZipScalarTypes, IprsScalarB32<3, UNCHECKED>, 16384>(&mut group, "IPRS-3-1_2-F65537-B32-i32");
    // B64 F65537: D=3 -> row_len=32768=2^15 (~105 bits)
    encode_single_row::<BenchZipScalarTypes, IprsScalarB64<3, UNCHECKED>, 32768>(&mut group, "IPRS-3-1_2-F65537-B64-i32");

    // ========== Batched EncodeMessage benchmarks (Scalar i32, Cw = i128) ==========
    for &num_rows in &[16, 64, 256] {
        // B16 F65537: DEPTH 1-3
        encode_message_batch::<BenchZipScalarTypes, IprsScalarB16<1, UNCHECKED>, 128>(&mut group, "IPRS-1-1_2-F65537-i32", num_rows);
        encode_message_batch::<BenchZipScalarTypes, IprsScalarB16<2, UNCHECKED>, 1024>(&mut group, "IPRS-2-1_2-F65537-i32", num_rows);
        encode_message_batch::<BenchZipScalarTypes, IprsScalarB16<3, UNCHECKED>, 8192>(&mut group, "IPRS-3-1_2-F65537-i32", num_rows);
        // B16 F1179649: DEPTH 1-3
        encode_message_batch::<BenchZipScalarTypes, IprsScalarMidB16<1, UNCHECKED>, 128>(&mut group, "IPRS-1-1_2-F1179649-i32", num_rows);
        encode_message_batch::<BenchZipScalarTypes, IprsScalarMidB16<2, UNCHECKED>, 1024>(&mut group, "IPRS-2-1_2-F1179649-i32", num_rows);
        encode_message_batch::<BenchZipScalarTypes, IprsScalarMidB16<3, UNCHECKED>, 8192>(&mut group, "IPRS-3-1_2-F1179649-i32", num_rows);
        // B32 F65537: D=3
        encode_message_batch::<BenchZipScalarTypes, IprsScalarB32<3, UNCHECKED>, 16384>(&mut group, "IPRS-3-1_2-F65537-B32-i32", num_rows);
        // B64 F65537: D=3
        encode_message_batch::<BenchZipScalarTypes, IprsScalarB64<3, UNCHECKED>, 32768>(&mut group, "IPRS-3-1_2-F65537-B64-i32", num_rows);
    }

    group.finish();
}

/// EncodeMessage suite: Batch=1, row_len from 2^6 to 2^19.
/// - F3329 for 2^6 to 2^7
/// - F65537 for 2^8 to 2^15
/// - F1179649 for 2^16
/// - F167772161 for 2^17 to 2^19 (F1179649 NTT domain 2^17 cannot support row_len > 2^16)
fn encode_message_suite(c: &mut Criterion) {
    let mut group = c.benchmark_group("EncodeMessage Suite");

    // ========== Scalar i32 (Eval=i32, Cw=i128) ==========
    // F3329: row_len 2^6 to 2^7
    encode_message_batch::<BenchZipScalarTypes, IprsScalarSmallB8<1, UNCHECKED>,    64>(&mut group, "F3329-i32", 1);    // 2^6
    encode_message_batch::<BenchZipScalarTypes, IprsScalarSmallB16<1, UNCHECKED>,  128>(&mut group, "F3329-i32", 1);    // 2^7
    // F65537: row_len 2^8 to 2^15
    encode_message_batch::<BenchZipScalarTypes, IprsScalarB32<1, UNCHECKED>,   256>(&mut group, "F65537-i32", 1);   // 2^8
    encode_message_batch::<BenchZipScalarTypes, IprsScalarB64<1, UNCHECKED>,   512>(&mut group, "F65537-i32", 1);   // 2^9
    encode_message_batch::<BenchZipScalarTypes, IprsScalarB16<2, UNCHECKED>,  1024>(&mut group, "F65537-i32", 1);   // 2^10
    encode_message_batch::<BenchZipScalarTypes, IprsScalarB32<2, UNCHECKED>,  2048>(&mut group, "F65537-i32", 1);   // 2^11
    encode_message_batch::<BenchZipScalarTypes, IprsScalarB64<2, UNCHECKED>,  4096>(&mut group, "F65537-i32", 1);   // 2^12
    encode_message_batch::<BenchZipScalarTypes, IprsScalarB16<3, UNCHECKED>,  8192>(&mut group, "F65537-i32", 1);   // 2^13
    encode_message_batch::<BenchZipScalarTypes, IprsScalarB32<3, UNCHECKED>, 16384>(&mut group, "F65537-i32", 1);   // 2^14
    encode_message_batch::<BenchZipScalarTypes, IprsScalarB64<3, UNCHECKED>, 32768>(&mut group, "F65537-i32", 1);   // 2^15
    // F1179649: row_len 2^16
    encode_message_batch::<BenchZipScalarTypes, IprsScalarMidB16<4, UNCHECKED>, 65536>(&mut group, "F1179649-i32", 1);  // 2^16
    // F167772161: row_len 2^17 to 2^19
    encode_message_batch::<BenchZipScalarTypes, IprsScalarLargeB32<4, UNCHECKED>, 131072>(&mut group, "F167772161-i32", 1); // 2^17
    encode_message_batch::<BenchZipScalarTypes, IprsScalarLargeB64<4, UNCHECKED>, 262144>(&mut group, "F167772161-i32", 1); // 2^18
    encode_message_batch::<BenchZipScalarTypes, IprsScalarLargeB16<5, UNCHECKED>, 524288>(&mut group, "F167772161-i32", 1); // 2^19

    // ========== BPoly<31> (Eval=BinaryPoly<32>, Cw=DensePolynomial<i128,32>) ==========
    // F3329: row_len 2^6 to 2^7
    encode_message_batch::<BenchZipPlusTypes<i128, 32>, IprsSmallB8<i128, 1, 32, UNCHECKED>,    64>(&mut group, "F3329-BPoly31", 1);    // 2^6
    encode_message_batch::<BenchZipPlusTypes<i128, 32>, IprsSmallB16<i128, 1, 32, UNCHECKED>,  128>(&mut group, "F3329-BPoly31", 1);    // 2^7
    // F65537: row_len 2^8 to 2^15
    encode_message_batch::<BenchZipPlusTypes<i128, 32>, IprsB32<i128, 1, 32, UNCHECKED>,   256>(&mut group, "F65537-BPoly31", 1);   // 2^8
    encode_message_batch::<BenchZipPlusTypes<i128, 32>, IprsB64<i128, 1, 32, UNCHECKED>,   512>(&mut group, "F65537-BPoly31", 1);   // 2^9
    encode_message_batch::<BenchZipPlusTypes<i128, 32>, IprsB16<i128, 2, 32, UNCHECKED>,  1024>(&mut group, "F65537-BPoly31", 1);   // 2^10
    encode_message_batch::<BenchZipPlusTypes<i128, 32>, IprsB32<i128, 2, 32, UNCHECKED>,  2048>(&mut group, "F65537-BPoly31", 1);   // 2^11
    encode_message_batch::<BenchZipPlusTypes<i128, 32>, IprsB64<i128, 2, 32, UNCHECKED>,  4096>(&mut group, "F65537-BPoly31", 1);   // 2^12
    encode_message_batch::<BenchZipPlusTypes<i128, 32>, IprsB16<i128, 3, 32, UNCHECKED>,  8192>(&mut group, "F65537-BPoly31", 1);   // 2^13
    encode_message_batch::<BenchZipPlusTypes<i128, 32>, IprsB32<i128, 3, 32, UNCHECKED>, 16384>(&mut group, "F65537-BPoly31", 1);   // 2^14
    encode_message_batch::<BenchZipPlusTypes<i128, 32>, IprsB64<i128, 3, 32, UNCHECKED>, 32768>(&mut group, "F65537-BPoly31", 1);   // 2^15
    // F1179649: row_len 2^16
    encode_message_batch::<BenchZipPlusTypes<i128, 32>, IprsMidB16<i128, 4, 32, UNCHECKED>, 65536>(&mut group, "F1179649-BPoly31", 1);  // 2^16
    // F167772161: row_len 2^17 to 2^19
    encode_message_batch::<BenchZipPlusTypes<i128, 32>, IprsLargeB32<i128, 4, 32, UNCHECKED>, 131072>(&mut group, "F167772161-BPoly31", 1); // 2^17
    encode_message_batch::<BenchZipPlusTypes<i128, 32>, IprsLargeB64<i128, 4, 32, UNCHECKED>, 262144>(&mut group, "F167772161-BPoly31", 1); // 2^18
    encode_message_batch::<BenchZipPlusTypes<i128, 32>, IprsLargeB16<i128, 5, 32, UNCHECKED>, 524288>(&mut group, "F167772161-BPoly31", 1); // 2^19

    group.finish();
}

/// Full PCS pipeline suite: Commit, Test, Verify for BPoly<31>.
/// Uses IPRS codes at rate 1/4 over F3329.
///
/// Row length   Config              poly_size exponent (P)
/// ──────────   ──────              ─────────────────────
/// 2^4  (16)    F3329 R4B2 D=1      P = 8, 9
/// 2^5  (32)    F3329 R4B4 D=1      P = 10, 11
/// 2^6  (64)    F3329 R4B8 D=1      P = 12
fn pcs_pipeline_suite(c: &mut Criterion) {
    let mut group = c.benchmark_group("PCS Pipeline Suite");
    group.sample_size(10);

    // ── Commit ───────────────────────────────────────────────────────
    // F3329 rate 1/4: P=8 to P=12
    commit::<BenchZipPlusTypes<i128, 32>, IprsSmallR4B2<i128, 1, 32, UNCHECKED>,  8>(&mut group);  // row_len=16
    commit::<BenchZipPlusTypes<i128, 32>, IprsSmallR4B2<i128, 1, 32, UNCHECKED>,  9>(&mut group);  // row_len=16
    commit::<BenchZipPlusTypes<i128, 32>, IprsSmallR4B4<i128, 1, 32, UNCHECKED>, 10>(&mut group);  // row_len=32
    commit::<BenchZipPlusTypes<i128, 32>, IprsSmallR4B4<i128, 1, 32, UNCHECKED>, 11>(&mut group);  // row_len=32
    commit::<BenchZipPlusTypes<i128, 32>, IprsSmallR4B8<i128, 1, 32, UNCHECKED>, 12>(&mut group);  // row_len=64

    // ── Test ─────────────────────────────────────────────────────────
    // F3329 rate 1/4: P=8 to P=12
    test::<BenchZipPlusTypes<i128, 32>, IprsSmallR4B2<i128, 1, 32, UNCHECKED>, UNCHECKED,  8>(&mut group);
    test::<BenchZipPlusTypes<i128, 32>, IprsSmallR4B2<i128, 1, 32, UNCHECKED>, UNCHECKED,  9>(&mut group);
    test::<BenchZipPlusTypes<i128, 32>, IprsSmallR4B4<i128, 1, 32, UNCHECKED>, UNCHECKED, 10>(&mut group);
    test::<BenchZipPlusTypes<i128, 32>, IprsSmallR4B4<i128, 1, 32, UNCHECKED>, UNCHECKED, 11>(&mut group);
    test::<BenchZipPlusTypes<i128, 32>, IprsSmallR4B8<i128, 1, 32, UNCHECKED>, UNCHECKED, 12>(&mut group);

    // ── Verify ───────────────────────────────────────────────────────
    // F3329 rate 1/4: P=8 to P=12
    verify::<BenchZipPlusTypes<i128, 32>, IprsSmallR4B2<i128, 1, 32, UNCHECKED>, UNCHECKED,  8>(&mut group);
    verify::<BenchZipPlusTypes<i128, 32>, IprsSmallR4B2<i128, 1, 32, UNCHECKED>, UNCHECKED,  9>(&mut group);
    verify::<BenchZipPlusTypes<i128, 32>, IprsSmallR4B4<i128, 1, 32, UNCHECKED>, UNCHECKED, 10>(&mut group);
    verify::<BenchZipPlusTypes<i128, 32>, IprsSmallR4B4<i128, 1, 32, UNCHECKED>, UNCHECKED, 11>(&mut group);
    verify::<BenchZipPlusTypes<i128, 32>, IprsSmallR4B8<i128, 1, 32, UNCHECKED>, UNCHECKED, 12>(&mut group);

    group.finish();
}

/// Full PCS pipeline suite: Commit, Test, Verify for i32 (scalar).
/// Uses IPRS codes at rate 1/4 over F3329.
///
/// Row length   Config              poly_size exponent (P)
/// ──────────   ──────              ─────────────────────
/// 2^4  (16)    F3329 R4B2 D=1      P = 8, 9
/// 2^5  (32)    F3329 R4B4 D=1      P = 10, 11
/// 2^6  (64)    F3329 R4B8 D=1      P = 12
fn pcs_pipeline_suite_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("PCS Pipeline Suite i32");
    group.sample_size(10);

    // ── Commit ───────────────────────────────────────────────────────
    commit::<BenchZipScalarTypes, IprsScalarSmallR4B2<1, UNCHECKED>,  8>(&mut group);
    commit::<BenchZipScalarTypes, IprsScalarSmallR4B2<1, UNCHECKED>,  9>(&mut group);
    commit::<BenchZipScalarTypes, IprsScalarSmallR4B4<1, UNCHECKED>, 10>(&mut group);
    commit::<BenchZipScalarTypes, IprsScalarSmallR4B4<1, UNCHECKED>, 11>(&mut group);
    commit::<BenchZipScalarTypes, IprsScalarSmallR4B8<1, UNCHECKED>, 12>(&mut group);

    // ── Test ─────────────────────────────────────────────────────────
    test::<BenchZipScalarTypes, IprsScalarSmallR4B2<1, UNCHECKED>, UNCHECKED,  8>(&mut group);
    test::<BenchZipScalarTypes, IprsScalarSmallR4B2<1, UNCHECKED>, UNCHECKED,  9>(&mut group);
    test::<BenchZipScalarTypes, IprsScalarSmallR4B4<1, UNCHECKED>, UNCHECKED, 10>(&mut group);
    test::<BenchZipScalarTypes, IprsScalarSmallR4B4<1, UNCHECKED>, UNCHECKED, 11>(&mut group);
    test::<BenchZipScalarTypes, IprsScalarSmallR4B8<1, UNCHECKED>, UNCHECKED, 12>(&mut group);

    // ── Verify ───────────────────────────────────────────────────────
    verify::<BenchZipScalarTypes, IprsScalarSmallR4B2<1, UNCHECKED>, UNCHECKED,  8>(&mut group);
    verify::<BenchZipScalarTypes, IprsScalarSmallR4B2<1, UNCHECKED>, UNCHECKED,  9>(&mut group);
    verify::<BenchZipScalarTypes, IprsScalarSmallR4B4<1, UNCHECKED>, UNCHECKED, 10>(&mut group);
    verify::<BenchZipScalarTypes, IprsScalarSmallR4B4<1, UNCHECKED>, UNCHECKED, 11>(&mut group);
    verify::<BenchZipScalarTypes, IprsScalarSmallR4B8<1, UNCHECKED>, UNCHECKED, 12>(&mut group);

    group.finish();
}

/// Full PCS pipeline suite with num_rows=1: Commit, Test, Verify for BPoly<31>.
/// Forces the matrix to have a single row (1 x n), using various IPRS codes.
///
/// With num_rows=1, poly_size must equal row_len. We use different fields
/// to support different row lengths:
///
/// poly_size (2^P)   Config                Field        row_len formula
/// ───────────────   ──────                ─────        ───────────────
/// 2^4  = 16         R4B2 D=1 (rate 1/4)   F3329        2 × 8^1 = 16
/// 2^5  = 32         R4B4 D=1 (rate 1/4)   F3329        4 × 8^1 = 32
/// 2^6  = 64         R4B8 D=1 (rate 1/4)   F3329        8 × 8^1 = 64
/// 2^7  = 128        B16 D=1 (rate 1/2)    F65537       16 × 8^1 = 128
/// 2^8  = 256        B32 D=1 (rate 1/2)    F65537       32 × 8^1 = 256
/// 2^9  = 512        B64 D=1 (rate 1/2)    F65537       64 × 8^1 = 512
/// 2^10 = 1024       B16 D=2 (rate 1/2)    F65537       16 × 8^2 = 1024
/// 2^11 = 2048       B32 D=2 (rate 1/2)    F65537       32 × 8^2 = 2048
/// 2^12 = 4096       B64 D=2 (rate 1/2)    F65537       64 × 8^2 = 4096
/// 2^13 = 8192       B16 D=3 (rate 1/2)    F65537       16 × 8^3 = 8192
/// 2^14 = 16384      B32 D=3 (rate 1/2)    F65537       32 × 8^3 = 16384
/// 2^16 = 65536      B16 D=4 (rate 1/2)    F1179649     16 × 8^4 = 65536
/// 2^17 = 131072     B32 D=4 (rate 1/2)    F167772161   32 × 8^4 = 131072
/// 2^18 = 262144     B64 D=4 (rate 1/2)    F167772161   64 × 8^4 = 262144
fn pcs_pipeline_suite_1row(c: &mut Criterion) {
    let mut group = c.benchmark_group("PCS Pipeline Suite 1row");
    group.sample_size(10);

    // ── Commit ───────────────────────────────────────────────────────
    // F3329 rate 1/4: small row lengths (P=4,5,6)
    commit_1row::<BenchZipPlusTypes<i128, 32>, IprsSmallR4B2<i128, 1, 32, UNCHECKED>, 4>(&mut group);  // row_len=16
    commit_1row::<BenchZipPlusTypes<i128, 32>, IprsSmallR4B4<i128, 1, 32, UNCHECKED>, 5>(&mut group);  // row_len=32
    commit_1row::<BenchZipPlusTypes<i128, 32>, IprsSmallR4B8<i128, 1, 32, UNCHECKED>, 6>(&mut group);  // row_len=64

    // F65537 rate 1/2: medium row lengths (P=7,8,9,10,11,12,13,14)
    commit_1row::<BenchZipPlusTypes<i128, 32>, IprsB16<i128, 1, 32, UNCHECKED>,  7>(&mut group);  // row_len=128
    commit_1row::<BenchZipPlusTypes<i128, 32>, IprsB32<i128, 1, 32, UNCHECKED>,  8>(&mut group);  // row_len=256
    commit_1row::<BenchZipPlusTypes<i128, 32>, IprsB64<i128, 1, 32, UNCHECKED>,  9>(&mut group);  // row_len=512
    commit_1row::<BenchZipPlusTypes<i128, 32>, IprsB16<i128, 2, 32, UNCHECKED>, 10>(&mut group);  // row_len=1024
    commit_1row::<BenchZipPlusTypes<i128, 32>, IprsB32<i128, 2, 32, UNCHECKED>, 11>(&mut group);  // row_len=2048
    commit_1row::<BenchZipPlusTypes<i128, 32>, IprsB64<i128, 2, 32, UNCHECKED>, 12>(&mut group);  // row_len=4096
    commit_1row::<BenchZipPlusTypes<i128, 32>, IprsB16<i128, 3, 32, UNCHECKED>, 13>(&mut group);  // row_len=8192
    commit_1row::<BenchZipPlusTypes<i128, 32>, IprsB32<i128, 3, 32, UNCHECKED>, 14>(&mut group);  // row_len=16384

    // F1179649: larger row lengths (P=16)
    commit_1row::<BenchZipPlusTypes<i128, 32>, IprsMidB16<i128, 4, 32, UNCHECKED>, 16>(&mut group);  // row_len=65536

    // F167772161: largest row lengths (P=17,18)
    commit_1row::<BenchZipPlusTypes<i128, 32>, IprsLargeB32<i128, 4, 32, UNCHECKED>, 17>(&mut group);  // row_len=131072
    commit_1row::<BenchZipPlusTypes<i128, 32>, IprsLargeB64<i128, 4, 32, UNCHECKED>, 18>(&mut group);  // row_len=262144

    // ── Test ─────────────────────────────────────────────────────────
    // F3329 rate 1/4
    test_1row::<BenchZipPlusTypes<i128, 32>, IprsSmallR4B2<i128, 1, 32, UNCHECKED>, UNCHECKED, 4>(&mut group);
    test_1row::<BenchZipPlusTypes<i128, 32>, IprsSmallR4B4<i128, 1, 32, UNCHECKED>, UNCHECKED, 5>(&mut group);
    test_1row::<BenchZipPlusTypes<i128, 32>, IprsSmallR4B8<i128, 1, 32, UNCHECKED>, UNCHECKED, 6>(&mut group);

    // F65537 rate 1/2
    test_1row::<BenchZipPlusTypes<i128, 32>, IprsB16<i128, 1, 32, UNCHECKED>, UNCHECKED,  7>(&mut group);
    test_1row::<BenchZipPlusTypes<i128, 32>, IprsB32<i128, 1, 32, UNCHECKED>, UNCHECKED,  8>(&mut group);
    test_1row::<BenchZipPlusTypes<i128, 32>, IprsB64<i128, 1, 32, UNCHECKED>, UNCHECKED,  9>(&mut group);
    test_1row::<BenchZipPlusTypes<i128, 32>, IprsB16<i128, 2, 32, UNCHECKED>, UNCHECKED, 10>(&mut group);
    test_1row::<BenchZipPlusTypes<i128, 32>, IprsB32<i128, 2, 32, UNCHECKED>, UNCHECKED, 11>(&mut group);
    test_1row::<BenchZipPlusTypes<i128, 32>, IprsB64<i128, 2, 32, UNCHECKED>, UNCHECKED, 12>(&mut group);
    test_1row::<BenchZipPlusTypes<i128, 32>, IprsB16<i128, 3, 32, UNCHECKED>, UNCHECKED, 13>(&mut group);
    test_1row::<BenchZipPlusTypes<i128, 32>, IprsB32<i128, 3, 32, UNCHECKED>, UNCHECKED, 14>(&mut group);

    // F1179649
    test_1row::<BenchZipPlusTypes<i128, 32>, IprsMidB16<i128, 4, 32, UNCHECKED>, UNCHECKED, 16>(&mut group);

    // F167772161
    test_1row::<BenchZipPlusTypes<i128, 32>, IprsLargeB32<i128, 4, 32, UNCHECKED>, UNCHECKED, 17>(&mut group);
    test_1row::<BenchZipPlusTypes<i128, 32>, IprsLargeB64<i128, 4, 32, UNCHECKED>, UNCHECKED, 18>(&mut group);

    // ── Verify ───────────────────────────────────────────────────────
    // F3329 rate 1/4
    verify_1row::<BenchZipPlusTypes<i128, 32>, IprsSmallR4B2<i128, 1, 32, UNCHECKED>, UNCHECKED, 4>(&mut group);
    verify_1row::<BenchZipPlusTypes<i128, 32>, IprsSmallR4B4<i128, 1, 32, UNCHECKED>, UNCHECKED, 5>(&mut group);
    verify_1row::<BenchZipPlusTypes<i128, 32>, IprsSmallR4B8<i128, 1, 32, UNCHECKED>, UNCHECKED, 6>(&mut group);

    // F65537 rate 1/2
    verify_1row::<BenchZipPlusTypes<i128, 32>, IprsB16<i128, 1, 32, UNCHECKED>, UNCHECKED,  7>(&mut group);
    verify_1row::<BenchZipPlusTypes<i128, 32>, IprsB32<i128, 1, 32, UNCHECKED>, UNCHECKED,  8>(&mut group);
    verify_1row::<BenchZipPlusTypes<i128, 32>, IprsB64<i128, 1, 32, UNCHECKED>, UNCHECKED,  9>(&mut group);
    verify_1row::<BenchZipPlusTypes<i128, 32>, IprsB16<i128, 2, 32, UNCHECKED>, UNCHECKED, 10>(&mut group);
    verify_1row::<BenchZipPlusTypes<i128, 32>, IprsB32<i128, 2, 32, UNCHECKED>, UNCHECKED, 11>(&mut group);
    verify_1row::<BenchZipPlusTypes<i128, 32>, IprsB64<i128, 2, 32, UNCHECKED>, UNCHECKED, 12>(&mut group);
    verify_1row::<BenchZipPlusTypes<i128, 32>, IprsB16<i128, 3, 32, UNCHECKED>, UNCHECKED, 13>(&mut group);
    verify_1row::<BenchZipPlusTypes<i128, 32>, IprsB32<i128, 3, 32, UNCHECKED>, UNCHECKED, 14>(&mut group);

    // F1179649
    verify_1row::<BenchZipPlusTypes<i128, 32>, IprsMidB16<i128, 4, 32, UNCHECKED>, UNCHECKED, 16>(&mut group);

    // F167772161
    verify_1row::<BenchZipPlusTypes<i128, 32>, IprsLargeB32<i128, 4, 32, UNCHECKED>, UNCHECKED, 17>(&mut group);
    verify_1row::<BenchZipPlusTypes<i128, 32>, IprsLargeB64<i128, 4, 32, UNCHECKED>, UNCHECKED, 18>(&mut group);

    group.finish();
}

/// Full PCS pipeline suite with num_rows=1: Commit, Test, Verify for scalar i32.
/// Forces the matrix to have a single row (1 x n), using various IPRS codes.
///
/// With num_rows=1, poly_size must equal row_len. We use different fields
/// to support different row lengths:
///
/// poly_size (2^P)   Config                Field        row_len formula
/// ───────────────   ──────                ─────        ───────────────
/// 2^4  = 16         R4B2 D=1 (rate 1/4)   F3329        2 × 8^1 = 16
/// 2^5  = 32         R4B4 D=1 (rate 1/4)   F3329        4 × 8^1 = 32
/// 2^6  = 64         R4B8 D=1 (rate 1/4)   F3329        8 × 8^1 = 64
/// 2^7  = 128        B16 D=1 (rate 1/2)    F65537       16 × 8^1 = 128
/// 2^8  = 256        B32 D=1 (rate 1/2)    F65537       32 × 8^1 = 256
/// 2^9  = 512        B64 D=1 (rate 1/2)    F65537       64 × 8^1 = 512
/// 2^10 = 1024       B16 D=2 (rate 1/2)    F65537       16 × 8^2 = 1024
/// 2^11 = 2048       B32 D=2 (rate 1/2)    F65537       32 × 8^2 = 2048
/// 2^12 = 4096       B64 D=2 (rate 1/2)    F65537       64 × 8^2 = 4096
/// 2^13 = 8192       B16 D=3 (rate 1/2)    F65537       16 × 8^3 = 8192
/// 2^14 = 16384      B32 D=3 (rate 1/2)    F65537       32 × 8^3 = 16384
/// 2^16 = 65536      B16 D=4 (rate 1/2)    F1179649     16 × 8^4 = 65536
/// 2^17 = 131072     B32 D=4 (rate 1/2)    F167772161   32 × 8^4 = 131072
/// 2^18 = 262144     B64 D=4 (rate 1/2)    F167772161   64 × 8^4 = 262144
fn pcs_pipeline_suite_scalar_1row(c: &mut Criterion) {
    let mut group = c.benchmark_group("PCS Pipeline Suite i32 1row");
    group.sample_size(10);

    // ── Commit ───────────────────────────────────────────────────────
    // F3329 rate 1/4: small row lengths (P=4,5,6)
    commit_1row::<BenchZipScalarTypes, IprsScalarSmallR4B2<1, UNCHECKED>, 4>(&mut group);  // row_len=16
    commit_1row::<BenchZipScalarTypes, IprsScalarSmallR4B4<1, UNCHECKED>, 5>(&mut group);  // row_len=32
    commit_1row::<BenchZipScalarTypes, IprsScalarSmallR4B8<1, UNCHECKED>, 6>(&mut group);  // row_len=64

    // F65537 rate 1/2: medium row lengths (P=7,8,9,10,11,12,13,14)
    commit_1row::<BenchZipScalarTypes, IprsScalarB16<1, UNCHECKED>,  7>(&mut group);  // row_len=128
    commit_1row::<BenchZipScalarTypes, IprsScalarB32<1, UNCHECKED>,  8>(&mut group);  // row_len=256
    commit_1row::<BenchZipScalarTypes, IprsScalarB64<1, UNCHECKED>,  9>(&mut group);  // row_len=512
    commit_1row::<BenchZipScalarTypes, IprsScalarB16<2, UNCHECKED>, 10>(&mut group);  // row_len=1024
    commit_1row::<BenchZipScalarTypes, IprsScalarB32<2, UNCHECKED>, 11>(&mut group);  // row_len=2048
    commit_1row::<BenchZipScalarTypes, IprsScalarB64<2, UNCHECKED>, 12>(&mut group);  // row_len=4096
    commit_1row::<BenchZipScalarTypes, IprsScalarB16<3, UNCHECKED>, 13>(&mut group);  // row_len=8192
    commit_1row::<BenchZipScalarTypes, IprsScalarB32<3, UNCHECKED>, 14>(&mut group);  // row_len=16384

    // F1179649: larger row lengths (P=16)
    commit_1row::<BenchZipScalarTypes, IprsScalarMidB16<4, UNCHECKED>, 16>(&mut group);  // row_len=65536

    // F167772161: largest row lengths (P=17,18)
    commit_1row::<BenchZipScalarTypes, IprsScalarLargeB32<4, UNCHECKED>, 17>(&mut group);  // row_len=131072
    commit_1row::<BenchZipScalarTypes, IprsScalarLargeB64<4, UNCHECKED>, 18>(&mut group);  // row_len=262144

    // ── Test ─────────────────────────────────────────────────────────
    // F3329 rate 1/4
    test_1row::<BenchZipScalarTypes, IprsScalarSmallR4B2<1, UNCHECKED>, UNCHECKED, 4>(&mut group);
    test_1row::<BenchZipScalarTypes, IprsScalarSmallR4B4<1, UNCHECKED>, UNCHECKED, 5>(&mut group);
    test_1row::<BenchZipScalarTypes, IprsScalarSmallR4B8<1, UNCHECKED>, UNCHECKED, 6>(&mut group);

    // F65537 rate 1/2
    test_1row::<BenchZipScalarTypes, IprsScalarB16<1, UNCHECKED>, UNCHECKED,  7>(&mut group);
    test_1row::<BenchZipScalarTypes, IprsScalarB32<1, UNCHECKED>, UNCHECKED,  8>(&mut group);
    test_1row::<BenchZipScalarTypes, IprsScalarB64<1, UNCHECKED>, UNCHECKED,  9>(&mut group);
    test_1row::<BenchZipScalarTypes, IprsScalarB16<2, UNCHECKED>, UNCHECKED, 10>(&mut group);
    test_1row::<BenchZipScalarTypes, IprsScalarB32<2, UNCHECKED>, UNCHECKED, 11>(&mut group);
    test_1row::<BenchZipScalarTypes, IprsScalarB64<2, UNCHECKED>, UNCHECKED, 12>(&mut group);
    test_1row::<BenchZipScalarTypes, IprsScalarB16<3, UNCHECKED>, UNCHECKED, 13>(&mut group);
    test_1row::<BenchZipScalarTypes, IprsScalarB32<3, UNCHECKED>, UNCHECKED, 14>(&mut group);

    // F1179649
    test_1row::<BenchZipScalarTypes, IprsScalarMidB16<4, UNCHECKED>, UNCHECKED, 16>(&mut group);

    // F167772161
    test_1row::<BenchZipScalarTypes, IprsScalarLargeB32<4, UNCHECKED>, UNCHECKED, 17>(&mut group);
    test_1row::<BenchZipScalarTypes, IprsScalarLargeB64<4, UNCHECKED>, UNCHECKED, 18>(&mut group);

    // ── Verify ───────────────────────────────────────────────────────
    // F3329 rate 1/4
    verify_1row::<BenchZipScalarTypes, IprsScalarSmallR4B2<1, UNCHECKED>, UNCHECKED, 4>(&mut group);
    verify_1row::<BenchZipScalarTypes, IprsScalarSmallR4B4<1, UNCHECKED>, UNCHECKED, 5>(&mut group);
    verify_1row::<BenchZipScalarTypes, IprsScalarSmallR4B8<1, UNCHECKED>, UNCHECKED, 6>(&mut group);

    // F65537 rate 1/2
    verify_1row::<BenchZipScalarTypes, IprsScalarB16<1, UNCHECKED>, UNCHECKED,  7>(&mut group);
    verify_1row::<BenchZipScalarTypes, IprsScalarB32<1, UNCHECKED>, UNCHECKED,  8>(&mut group);
    verify_1row::<BenchZipScalarTypes, IprsScalarB64<1, UNCHECKED>, UNCHECKED,  9>(&mut group);
    verify_1row::<BenchZipScalarTypes, IprsScalarB16<2, UNCHECKED>, UNCHECKED, 10>(&mut group);
    verify_1row::<BenchZipScalarTypes, IprsScalarB32<2, UNCHECKED>, UNCHECKED, 11>(&mut group);
    verify_1row::<BenchZipScalarTypes, IprsScalarB64<2, UNCHECKED>, UNCHECKED, 12>(&mut group);
    verify_1row::<BenchZipScalarTypes, IprsScalarB16<3, UNCHECKED>, UNCHECKED, 13>(&mut group);
    verify_1row::<BenchZipScalarTypes, IprsScalarB32<3, UNCHECKED>, UNCHECKED, 14>(&mut group);

    // F1179649
    verify_1row::<BenchZipScalarTypes, IprsScalarMidB16<4, UNCHECKED>, UNCHECKED, 16>(&mut group);

    // F167772161
    verify_1row::<BenchZipScalarTypes, IprsScalarLargeB32<4, UNCHECKED>, UNCHECKED, 17>(&mut group);
    verify_1row::<BenchZipScalarTypes, IprsScalarLargeB64<4, UNCHECKED>, UNCHECKED, 18>(&mut group);

    group.finish();
}

/// Compare different code configurations for encoding a single BPoly<31> message
/// of length 2^18 = 262144. Only configs that produce exactly row_len = 262144
/// are compared (i.e. BASE_LEN=64, DEPTH=4 with different fields).
fn encode_full_poly_2_18_config_search(c: &mut Criterion) {
    use std::hint::black_box;
    use rand::Rng;

    let mut group = c.benchmark_group("Encode 2^18 BPoly31 Config Search");
    group.sample_size(10);

    const ROW_LEN: usize = 262144; // 2^18

    let mut rng = rand::rng();
    let message: Vec<BinaryPoly<32>> = (0..ROW_LEN).map(|_| rng.random()).collect();

    // Config A (CURRENT): B64-D4, F167772161 (~28-bit twiddles)
    {
        type Code = IprsLargeB64<i128, 4, 32, UNCHECKED>;
        let lc = <Code as zip_plus::code::LinearCode<BenchZipPlusTypes<i128, 32>>>::new(ROW_LEN * ROW_LEN);
        group.bench_function("B64-D4-F167772161 (CURRENT)", |b| {
            b.iter(|| {
                let encoded = lc.encode(&message);
                black_box(encoded);
            })
        });
    }

    // Config B (NEW): B64-D4, F7340033 (~22-bit twiddles)
    {
        type Code = IprsMid2B64<i128, 4, 32, UNCHECKED>;
        let lc = <Code as zip_plus::code::LinearCode<BenchZipPlusTypes<i128, 32>>>::new(ROW_LEN * ROW_LEN);
        group.bench_function("B64-D4-F7340033 (NEW)", |b| {
            b.iter(|| {
                let encoded = lc.encode(&message);
                black_box(encoded);
            })
        });
    }

    group.finish();
}

criterion_group!(benches, zip_plus_benchmarks_iprs, encode_message_suite, pcs_pipeline_suite, pcs_pipeline_suite_scalar, pcs_pipeline_suite_1row, pcs_pipeline_suite_scalar_1row, encode_full_poly_2_18_config_search);
criterion_main!(benches);
