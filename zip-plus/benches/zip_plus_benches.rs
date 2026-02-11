#![allow(non_local_definitions)]
#![allow(clippy::eq_op, clippy::arithmetic_side_effects, clippy::unwrap_used)]

mod zip_common;

use std::marker::PhantomData;

use zinc_poly::univariate::{
    binary::{BinaryPoly, BinaryPolyInnerProduct, BinaryPolyWideningMulByScalar},
    dense::{DensePolyInnerProduct, DensePolynomial},
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
        iprs::{
            IprsCode, PnttConfigF2_16_1, PnttConfigF2_16B16, PnttConfigF2_16B64,
            PnttConfigF167772161, PnttConfigF167772161B16, PnttConfigF167772161B64,
        },
        raa::{RaaCode, RaaConfig},
    },
    pcs::structs::ZipTypes,
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

// IPRS code types for scalar i32 evaluations (BASE_LEN > 16 only)
type IprsScalarB32<const DEPTH: usize, const CHECK: bool> =
    IprsCode<BenchZipScalarTypes, PnttConfigF2_16_1<DEPTH>, ScalarWideningMulByScalar<i128>, CHECK>;

type IprsScalarB64<const DEPTH: usize, const CHECK: bool> =
    IprsCode<BenchZipScalarTypes, PnttConfigF2_16B64<DEPTH>, ScalarWideningMulByScalar<i128>, CHECK>;

type IprsScalarLargeB32<const DEPTH: usize, const CHECK: bool> =
    IprsCode<BenchZipScalarTypes, PnttConfigF167772161<DEPTH>, ScalarWideningMulByScalar<i128>, CHECK>;

type IprsScalarLargeB64<const DEPTH: usize, const CHECK: bool> =
    IprsCode<BenchZipScalarTypes, PnttConfigF167772161B64<DEPTH>, ScalarWideningMulByScalar<i128>, CHECK>;

#[allow(dead_code)]
type SomeRaaCode<const D_PLUS_ONE: usize> =
    RaaCode<BenchZipPlusTypes<i32, D_PLUS_ONE>, BenchRaaConfig, 4>;

// IPRS code with BASE_LEN=32, BASE_DIM=64 (rate 1/2)
// Row lengths: 256 (D=1), 2048 (D=2), 16384 (D=3)
type IprsB32<Twiddle, const DEPTH: usize, const D_PLUS_ONE: usize, const CHECK: bool> =
    IprsCode<
        BenchZipPlusTypes<Twiddle, D_PLUS_ONE>,
        PnttConfigF2_16_1<DEPTH>,
        BinaryPolyWideningMulByScalar<Twiddle>,
        CHECK,
    >;

// IPRS code with BASE_LEN=16, BASE_DIM=32 (rate 1/2)
// Row lengths: 128 (D=1), 1024 (D=2), 8192 (D=3), 65536 (D=4)
type IprsB16<Twiddle, const DEPTH: usize, const D_PLUS_ONE: usize, const CHECK: bool> =
    IprsCode<
        BenchZipPlusTypes<Twiddle, D_PLUS_ONE>,
        PnttConfigF2_16B16<DEPTH>,
        BinaryPolyWideningMulByScalar<Twiddle>,
        CHECK,
    >;

// IPRS code with BASE_LEN=64, BASE_DIM=128 (rate 1/2)
// Row lengths: 512 (D=1), 4096 (D=2), 32768 (D=3)
type IprsB64<Twiddle, const DEPTH: usize, const D_PLUS_ONE: usize, const CHECK: bool> =
    IprsCode<
        BenchZipPlusTypes<Twiddle, D_PLUS_ONE>,
        PnttConfigF2_16B64<DEPTH>,
        BinaryPolyWideningMulByScalar<Twiddle>,
        CHECK,
    >;

// F167772161 = 5 × 2^25 + 1 configurations (supports NTT up to 2^25)
// IPRS code with BASE_LEN=32, BASE_DIM=64 (rate 1/2)
// Row lengths: 256 (D=1), 2048 (D=2), 16384 (D=3), 131072 (D=4), 1048576 (D=5)
type IprsLargeB32<Twiddle, const DEPTH: usize, const D_PLUS_ONE: usize, const CHECK: bool> =
    IprsCode<
        BenchZipPlusTypes<Twiddle, D_PLUS_ONE>,
        PnttConfigF167772161<DEPTH>,
        BinaryPolyWideningMulByScalar<Twiddle>,
        CHECK,
    >;

// IPRS code with BASE_LEN=16, BASE_DIM=32 (rate 1/2)
// Row lengths: 128 (D=1), 1024 (D=2), 8192 (D=3), 65536 (D=4), 524288 (D=5)
type IprsLargeB16<Twiddle, const DEPTH: usize, const D_PLUS_ONE: usize, const CHECK: bool> =
    IprsCode<
        BenchZipPlusTypes<Twiddle, D_PLUS_ONE>,
        PnttConfigF167772161B16<DEPTH>,
        BinaryPolyWideningMulByScalar<Twiddle>,
        CHECK,
    >;

// IPRS code with BASE_LEN=64, BASE_DIM=128 (rate 1/2)
// Row lengths: 512 (D=1), 4096 (D=2), 32768 (D=3), 262144 (D=4)
type IprsLargeB64<Twiddle, const DEPTH: usize, const D_PLUS_ONE: usize, const CHECK: bool> =
    IprsCode<
        BenchZipPlusTypes<Twiddle, D_PLUS_ONE>,
        PnttConfigF167772161B64<DEPTH>,
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

    // Benchmark encoding for all row lengths with rate 1/2 IPRS codes
    // Field F2^16+1 = F65537
    // Row length 128: BASE_LEN=16, DEPTH=1
    encode_single_row::<BenchZipPlusTypes<i64, 32>, IprsB16<i64, 1, 32, UNCHECKED>, 128>(&mut group, "IPRS-1-1_2-F65537");
    // Row length 256: BASE_LEN=32, DEPTH=1
    encode_single_row::<BenchZipPlusTypes<i64, 32>, IprsB32<i64, 1, 32, UNCHECKED>, 256>(&mut group, "IPRS-1-1_2-F65537");
    // Row length 512: BASE_LEN=64, DEPTH=1
    encode_single_row::<BenchZipPlusTypes<i64, 32>, IprsB64<i64, 1, 32, UNCHECKED>, 512>(&mut group, "IPRS-1-1_2-F65537");
    // Row length 1024: BASE_LEN=16, DEPTH=2
    encode_single_row::<BenchZipPlusTypes<i64, 32>, IprsB16<i64, 2, 32, UNCHECKED>, 1024>(&mut group, "IPRS-2-1_2-F65537");
    // Row length 2048: BASE_LEN=32, DEPTH=2
    encode_single_row::<BenchZipPlusTypes<i64, 32>, IprsB32<i64, 2, 32, UNCHECKED>, 2048>(&mut group, "IPRS-2-1_2-F65537");
    // Row length 4096: BASE_LEN=64, DEPTH=2
    encode_single_row::<BenchZipPlusTypes<i64, 32>, IprsB64<i64, 2, 32, UNCHECKED>, 4096>(&mut group, "IPRS-2-1_2-F65537");
    // Row length 8192: BASE_LEN=16, DEPTH=3
    encode_single_row::<BenchZipPlusTypes<i64, 32>, IprsB16<i64, 3, 32, UNCHECKED>, 8192>(&mut group, "IPRS-3-1_2-F65537");
    // Row length 16384: BASE_LEN=32, DEPTH=3
    encode_single_row::<BenchZipPlusTypes<i64, 32>, IprsB32<i64, 3, 32, UNCHECKED>, 16384>(&mut group, "IPRS-3-1_2-F65537");
    // Row length 32768: BASE_LEN=64, DEPTH=3
    encode_single_row::<BenchZipPlusTypes<i64, 32>, IprsB64<i64, 3, 32, UNCHECKED>, 32768>(&mut group, "IPRS-3-1_2-F65537");

    // Larger row lengths using F167772161 (5 × 2^25 + 1)
    // Row length 65536: BASE_LEN=16, DEPTH=4
    encode_single_row::<BenchZipPlusTypes<i64, 32>, IprsLargeB16<i64, 4, 32, UNCHECKED>, 65536>(&mut group, "IPRS-4-1_2-F167772161");
    // Row length 131072: BASE_LEN=32, DEPTH=4
    encode_single_row::<BenchZipPlusTypes<i64, 32>, IprsLargeB32<i64, 4, 32, UNCHECKED>, 131072>(&mut group, "IPRS-4-1_2-F167772161");
    // Row length 262144: BASE_LEN=64, DEPTH=4
    encode_single_row::<BenchZipPlusTypes<i64, 32>, IprsLargeB64<i64, 4, 32, UNCHECKED>, 262144>(&mut group, "IPRS-4-1_2-F167772161");
    // Row length 524288: BASE_LEN=16, DEPTH=5
    encode_single_row::<BenchZipPlusTypes<i64, 32>, IprsLargeB16<i64, 5, 32, UNCHECKED>, 524288>(&mut group, "IPRS-5-1_2-F167772161");
    // Row length 1048576 (2^20): BASE_LEN=32, DEPTH=5
    encode_single_row::<BenchZipPlusTypes<i64, 32>, IprsLargeB32<i64, 5, 32, UNCHECKED>, 1048576>(&mut group, "IPRS-5-1_2-F167772161");

    // ========== Degree-0 (constant) polynomial benchmarks ==========
    // Using D_PLUS_ONE = 1 for BPoly<1> (constant polynomials)
    // Field F2^16+1 = F65537
    encode_single_row::<BenchZipPlusTypes<i64, 1>, IprsB16<i64, 1, 1, UNCHECKED>, 128>(&mut group, "IPRS-1-1_2-F65537-Deg0");
    encode_single_row::<BenchZipPlusTypes<i64, 1>, IprsB32<i64, 1, 1, UNCHECKED>, 256>(&mut group, "IPRS-1-1_2-F65537-Deg0");
    encode_single_row::<BenchZipPlusTypes<i64, 1>, IprsB64<i64, 1, 1, UNCHECKED>, 512>(&mut group, "IPRS-1-1_2-F65537-Deg0");
    encode_single_row::<BenchZipPlusTypes<i64, 1>, IprsB16<i64, 2, 1, UNCHECKED>, 1024>(&mut group, "IPRS-2-1_2-F65537-Deg0");
    encode_single_row::<BenchZipPlusTypes<i64, 1>, IprsB32<i64, 2, 1, UNCHECKED>, 2048>(&mut group, "IPRS-2-1_2-F65537-Deg0");
    encode_single_row::<BenchZipPlusTypes<i64, 1>, IprsB64<i64, 2, 1, UNCHECKED>, 4096>(&mut group, "IPRS-2-1_2-F65537-Deg0");
    encode_single_row::<BenchZipPlusTypes<i64, 1>, IprsB16<i64, 3, 1, UNCHECKED>, 8192>(&mut group, "IPRS-3-1_2-F65537-Deg0");
    encode_single_row::<BenchZipPlusTypes<i64, 1>, IprsB32<i64, 3, 1, UNCHECKED>, 16384>(&mut group, "IPRS-3-1_2-F65537-Deg0");
    encode_single_row::<BenchZipPlusTypes<i64, 1>, IprsB64<i64, 3, 1, UNCHECKED>, 32768>(&mut group, "IPRS-3-1_2-F65537-Deg0");

    // Larger row lengths using F167772161 (5 × 2^25 + 1) - Degree 0
    encode_single_row::<BenchZipPlusTypes<i64, 1>, IprsLargeB16<i64, 4, 1, UNCHECKED>, 65536>(&mut group, "IPRS-4-1_2-F167772161-Deg0");
    encode_single_row::<BenchZipPlusTypes<i64, 1>, IprsLargeB32<i64, 4, 1, UNCHECKED>, 131072>(&mut group, "IPRS-4-1_2-F167772161-Deg0");
    encode_single_row::<BenchZipPlusTypes<i64, 1>, IprsLargeB64<i64, 4, 1, UNCHECKED>, 262144>(&mut group, "IPRS-4-1_2-F167772161-Deg0");
    encode_single_row::<BenchZipPlusTypes<i64, 1>, IprsLargeB16<i64, 5, 1, UNCHECKED>, 524288>(&mut group, "IPRS-5-1_2-F167772161-Deg0");
    encode_single_row::<BenchZipPlusTypes<i64, 1>, IprsLargeB32<i64, 5, 1, UNCHECKED>, 1048576>(&mut group, "IPRS-5-1_2-F167772161-Deg0");

    // ========== Scalar i32 benchmarks (BASE_LEN > 16 only) ==========
    // Using i32 as eval type (non-polynomial)
    // Field F2^16+1 = F65537
    // Row length 256: BASE_LEN=32, DEPTH=1
    encode_single_row::<BenchZipScalarTypes, IprsScalarB32<1, UNCHECKED>, 256>(&mut group, "IPRS-1-1_2-F65537-i32");
    // Row length 512: BASE_LEN=64, DEPTH=1
    encode_single_row::<BenchZipScalarTypes, IprsScalarB64<1, UNCHECKED>, 512>(&mut group, "IPRS-1-1_2-F65537-i32");
    // Row length 2048: BASE_LEN=32, DEPTH=2
    encode_single_row::<BenchZipScalarTypes, IprsScalarB32<2, UNCHECKED>, 2048>(&mut group, "IPRS-2-1_2-F65537-i32");
    // Row length 4096: BASE_LEN=64, DEPTH=2
    encode_single_row::<BenchZipScalarTypes, IprsScalarB64<2, UNCHECKED>, 4096>(&mut group, "IPRS-2-1_2-F65537-i32");
    // Row length 16384: BASE_LEN=32, DEPTH=3
    encode_single_row::<BenchZipScalarTypes, IprsScalarB32<3, UNCHECKED>, 16384>(&mut group, "IPRS-3-1_2-F65537-i32");
    // Row length 32768: BASE_LEN=64, DEPTH=3
    encode_single_row::<BenchZipScalarTypes, IprsScalarB64<3, UNCHECKED>, 32768>(&mut group, "IPRS-3-1_2-F65537-i32");

    // Larger row lengths using F167772161 (5 × 2^25 + 1) - i32
    // Row length 131072: BASE_LEN=32, DEPTH=4
    encode_single_row::<BenchZipScalarTypes, IprsScalarLargeB32<4, UNCHECKED>, 131072>(&mut group, "IPRS-4-1_2-F167772161-i32");
    // Row length 262144: BASE_LEN=64, DEPTH=4
    encode_single_row::<BenchZipScalarTypes, IprsScalarLargeB64<4, UNCHECKED>, 262144>(&mut group, "IPRS-4-1_2-F167772161-i32");
    // Row length 1048576: BASE_LEN=32, DEPTH=5
    encode_single_row::<BenchZipScalarTypes, IprsScalarLargeB32<5, UNCHECKED>, 1048576>(&mut group, "IPRS-5-1_2-F167772161-i32");

    group.finish();
}

criterion_group!(benches, zip_plus_benchmarks_iprs);
criterion_main!(benches);
