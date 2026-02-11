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
            IprsCode, PnttConfigF2_16B16,
            PnttConfigF1179649B16,
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

// IPRS code types for scalar i32 evaluations (BASE_LEN=16 only)
type IprsScalarB16<const DEPTH: usize, const CHECK: bool> =
    IprsCode<BenchZipScalarTypes, PnttConfigF2_16B16<DEPTH>, ScalarWideningMulByScalar<i128>, CHECK>;

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

// F1179649 (9 × 2^17 + 1): Row lengths 128 (D=1), 1024 (D=2), 8192 (D=3), 65536 (D=4)
type IprsMidB16<Twiddle, const DEPTH: usize, const D_PLUS_ONE: usize, const CHECK: bool> =
    IprsCode<
        BenchZipPlusTypes<Twiddle, D_PLUS_ONE>,
        PnttConfigF1179649B16<DEPTH>,
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
    // F65537: DEPTH 1-3 (max ~103 bits after D=3)
    encode_single_row::<BenchZipScalarTypes, IprsScalarB16<1, UNCHECKED>, 128>(&mut group, "IPRS-1-1_2-F65537-i32");
    encode_single_row::<BenchZipScalarTypes, IprsScalarB16<2, UNCHECKED>, 1024>(&mut group, "IPRS-2-1_2-F65537-i32");
    encode_single_row::<BenchZipScalarTypes, IprsScalarB16<3, UNCHECKED>, 8192>(&mut group, "IPRS-3-1_2-F65537-i32");

    group.finish();
}

criterion_group!(benches, zip_plus_benchmarks_iprs);
criterion_main!(benches);
