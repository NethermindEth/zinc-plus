#![allow(non_local_definitions)]
#![allow(clippy::eq_op, clippy::arithmetic_side_effects, clippy::unwrap_used)]

mod zip_common;

use num_traits::{One, Zero};
use std::marker::PhantomData;
use zinc_transcript::traits::ConstTranscribable;

use zinc_poly::univariate::dense::{
    DensePolyInnerProduct, DensePolynomial, HornerProjection, InnerProductProjection,
};
use zinc_primality::MillerRabin;
use zinc_utils::{
    from_ref::FromRef,
    inner_product::{
        BooleanInnerProductCheckedAdd, BooleanInnerProductUncheckedAdd, MBSInnerProductChecked,
    },
    named::Named,
};
use zip_common::*;

use criterion::{Criterion, criterion_group, criterion_main};
use crypto_bigint::U64;
use crypto_primitives::{
    Semiring, boolean::Boolean, crypto_bigint_int::Int, crypto_bigint_uint::Uint,
};
use zip_plus::{
    code::{
        iprs::{AnyMConfig, IprsCode},
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
        + Semiring
        + Send
        + Sync
        + Zero
        + One,
    Int<5>: FromRef<CwCoeff>,
{
    const NUM_COLUMN_OPENINGS: usize = 650;
    type Eval = DensePolynomial<Boolean, D_PLUS_ONE>;
    type Cw = DensePolynomial<CwCoeff, D_PLUS_ONE>;
    type Fmod = Uint<{ INT_LIMBS * 4 }>;
    type PrimeTest = MillerRabin;
    type Chal = i128;
    type Pt = i128;
    type CombR = Int<{ INT_LIMBS * 5 }>;
    type Comb = DensePolynomial<Self::CombR, D_PLUS_ONE>;
    type EvalDotChal = DensePolyInnerProduct<
        Boolean,
        Self::Chal,
        Self::CombR,
        BooleanInnerProductCheckedAdd,
        D_PLUS_ONE,
    >;
    type CombDotChal = DensePolyInnerProduct<
        Self::CombR,
        Self::Chal,
        Self::CombR,
        MBSInnerProductChecked,
        D_PLUS_ONE,
    >;
}

type SomeRaaCode<const D_PLUS_ONE: usize> = RaaCode<BenchZipPlusTypes<i32, D_PLUS_ONE>, 4>;
type SomeIprsCode<Twiddle, const M: usize, const D_PLUS_ONE: usize> =
    IprsCode<BenchZipPlusTypes<i128, D_PLUS_ONE>, AnyMConfig<M, 2, 3>, Twiddle, 2>;

const RAA_CONFIG: RaaConfig = RaaConfig {
    check_for_overflows: false,
    permute_in_place: true,
};

fn zip_plus_benchmarks_raa(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip+ RAA");

    do_bench::<
        BenchZipPlusTypes<i32, 32>,
        SomeRaaCode<_>,
        InnerProductProjection<Boolean, BooleanInnerProductUncheckedAdd, _>,
        HornerProjection<_, _>,
    >(&mut group, RAA_CONFIG);
    do_bench::<
        BenchZipPlusTypes<i32, 64>,
        SomeRaaCode<_>,
        InnerProductProjection<Boolean, BooleanInnerProductUncheckedAdd, _>,
        HornerProjection<_, _>,
    >(&mut group, RAA_CONFIG);

    group.finish();
}

fn zip_plus_benchmarks_iprs(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip+ IPRS");

    encode_rows::<BenchZipPlusTypes<i128, 32>, SomeIprsCode<i128, { 1 << 12 }, _>, 12>(
        &mut group, AnyMConfig,
    );
    encode_rows::<BenchZipPlusTypes<i128, 32>, SomeIprsCode<i128, { 1 << 13 }, _>, 13>(
        &mut group, AnyMConfig,
    );
    encode_rows::<BenchZipPlusTypes<i128, 32>, SomeIprsCode<i128, { 1 << 14 }, _>, 14>(
        &mut group, AnyMConfig,
    );
    encode_rows::<BenchZipPlusTypes<i128, 32>, SomeIprsCode<i128, { 1 << 15 }, _>, 15>(
        &mut group, AnyMConfig,
    );
    encode_rows::<BenchZipPlusTypes<i128, 32>, SomeIprsCode<i128, { 1 << 16 }, _>, 16>(
        &mut group, AnyMConfig,
    );
}

criterion_group!(benches, zip_plus_benchmarks_raa, zip_plus_benchmarks_iprs);
criterion_main!(benches);
