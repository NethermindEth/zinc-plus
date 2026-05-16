#![allow(non_local_definitions)]
#![allow(clippy::eq_op, clippy::arithmetic_side_effects, clippy::unwrap_used)]

mod zip_common;

use zip_common::*;

use criterion::{Criterion, criterion_group, criterion_main};
use crypto_bigint::U64;
use crypto_primitives::{
    FixedSemiring, boolean::Boolean, crypto_bigint_int::Int, crypto_bigint_uint::Uint,
};
use rand::prelude::*;
use std::{hint::black_box, marker::PhantomData, time::{Duration, Instant}};
use zinc_poly::mle::{DenseMultilinearExtension, MultilinearExtensionRand};
use zinc_poly::univariate::{
    binary::{BinaryPoly, BinaryPolyInnerProduct},
    dense::{DensePolyInnerProduct, DensePolynomial},
};
use zinc_primality::MillerRabin;
use zinc_transcript::traits::{ConstTranscribable, Transcript};
use zinc_utils::{
    from_ref::FromRef,
    inner_product::{MBSInnerProduct, ScalarProduct},
    named::Named,
};
use zip_plus::{
    code::{
        binary_add_fft_16::{
            AddFftConfigGF2_16, BinaryAddFft16Code,
            ext_field::{Bit4Poly16, Gf2_16Ext, P_16_DEFAULT},
        },
        iprs::{IprsCode, PnttConfigF65537},
        raa::{RaaCode, RaaConfig},
    },
    pcs::structs::{ZipPlus, ZipTypes},
    pcs_transcript::PcsProverTranscript,
};

const PERFORM_CHECKS: bool = if cfg!(feature = "unchecked") {
    zinc_utils::UNCHECKED
} else {
    zinc_utils::CHECKED
};

const INT_LIMBS: usize = U64::LIMBS;

/// Column-opening counts by inverse-rate, matching the protocol's IPRS
/// table (`bench_real_*` in `protocol/benches/e2e.rs`: 150 / 100 / 75 for
/// REP = 4 / 8 / 16). We use 147 at REP=4 as the polychal benches have
/// historically. Any Zt struct whose `NUM_COLUMN_OPENINGS` references one
/// of these constants is implicitly tied to that REP — wiring it to a
/// different REP requires picking the matching constant.
const OPENINGS_REP_4: usize = 147;
const OPENINGS_REP_8: usize = 100;
#[allow(dead_code)]
const OPENINGS_REP_16: usize = 75;

// ===========================================================================
// Reference codes (RAA, IPRS) and the ZipTypes they use.
// ===========================================================================

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
    // `BenchZipPlusTypes` is only used by REP=4 reference codes (RAA, IPRS).
    const NUM_COLUMN_OPENINGS: usize = OPENINGS_REP_4;
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

    do_bench::<BenchZipPlusTypes<i64, 32>, _, PERFORM_CHECKS>(&mut group, |poly_size| {
        BenchIprsCode::new_with_optimal_depth(poly_size).ok()
    });
    do_bench::<BenchZipPlusTypes<i64, 64>, _, PERFORM_CHECKS>(&mut group, |poly_size| {
        BenchIprsCode::new_with_optimal_depth(poly_size).ok()
    });

    group.finish();
}

/// IPRS at the SHA-256 binary-lane dimensions (`num_vars=9, batch=11`),
/// kept as a reference baseline alongside the D=16 radix-8 polychal bench.
fn zip_plus_benchmarks_sha_iprs(c: &mut Criterion) {
    use criterion::{BenchmarkGroup, measurement::WallTime};
    let mut group: BenchmarkGroup<WallTime> =
        c.benchmark_group("Zip+ SHA-style (IPRS, i64/Int<5>)");
    group.sample_size(20);

    zip_common::commit::<BenchZipPlusTypes<i64, 32>, _, 9, 11>(&mut group, |poly_size| {
        BenchIprsCode::<i64, 32>::new_with_optimal_depth(poly_size).ok()
    });
    zip_common::prove::<BenchZipPlusTypes<i64, 32>, _, PERFORM_CHECKS, 9, 11>(
        &mut group,
        |poly_size| BenchIprsCode::<i64, 32>::new_with_optimal_depth(poly_size).ok(),
    );
    zip_common::verify::<BenchZipPlusTypes<i64, 32>, _, PERFORM_CHECKS, 9, 11>(
        &mut group,
        |poly_size| BenchIprsCode::<i64, 32>::new_with_optimal_depth(poly_size).ok(),
    );
    group.finish();
}

/// IPRS at SHA+ECDSA binary-lane dimensions (`num_vars=9, batch=15`).
fn zip_plus_benchmarks_sha_ecdsa_iprs(c: &mut Criterion) {
    use criterion::{BenchmarkGroup, measurement::WallTime};
    let mut group: BenchmarkGroup<WallTime> =
        c.benchmark_group("Zip+ SHA+ECDSA-style (IPRS, i64/Int<5>)");
    group.sample_size(20);

    zip_common::commit::<BenchZipPlusTypes<i64, 32>, _, 9, 15>(&mut group, |poly_size| {
        BenchIprsCode::<i64, 32>::new_with_optimal_depth(poly_size).ok()
    });
    zip_common::prove::<BenchZipPlusTypes<i64, 32>, _, PERFORM_CHECKS, 9, 15>(
        &mut group,
        |poly_size| BenchIprsCode::<i64, 32>::new_with_optimal_depth(poly_size).ok(),
    );
    zip_common::verify::<BenchZipPlusTypes<i64, 32>, _, PERFORM_CHECKS, 9, 15>(
        &mut group,
        |poly_size| BenchIprsCode::<i64, 32>::new_with_optimal_depth(poly_size).ok(),
    );
    group.finish();
}

// ===========================================================================
// D=16 polychal Zip+ — the binary-additive-FFT variant we ship.
//
// `Eval = BinaryPoly<16>` (1× folded — each `BinaryPoly<32>` witness column
// is split into 2 halves at the protocol layer). The encoder is a radix-8
// additive FFT over `GF(2^16)` lifted to `Z[X]/f̃`: per-meta-stage 8×8
// Vandermonde matrices are precomputed in GF and lifted as 0/1-coef polys,
// and the matvec is carried out in `Z[X]/f̃`. The verifier-side field is
// `F = F_p[X]/f̃` with `P = P_16_DEFAULT`.
//
// Cw and CombR are bit-packed `PackedI64Poly16<BITS>`. The codeword is
// encoded with the signed `{-1,0,1}` twiddle lift (`new_signed`); the
// certified worst-case codeword coefficient for the folded-lane sizes
// is < 2^23, so BITS_CW=24 is lossless (see `certified_worst_case_bound`).
// CombR coefs are bounded by `batch × max_chal_coef` (≤ batch · 15 at
// num_rows=1) so BITS_CR=8 is safe at batch=11.
// ===========================================================================

const BITS_CW: u32 = 24;
const BITS_CR: u32 = 8;

#[derive(Debug, Clone)]
struct BenchAddFft16PolyChalPackedZipTypes<const PP: u64, const NUM_OPENINGS: usize>;
impl<const PP: u64, const NUM_OPENINGS: usize> ZipTypes
    for BenchAddFft16PolyChalPackedZipTypes<PP, NUM_OPENINGS>
{
    const NUM_COLUMN_OPENINGS: usize = NUM_OPENINGS;
    type Eval = BinaryPoly<16>;
    type Cw = zip_plus::code::binary_add_fft_16::packed_cw::PackedI64Poly16<BITS_CW>;
    type Fmod = Uint<16>;
    type PrimeTest = MillerRabin;
    type Chal = Bit4Poly16;
    type Pt = Bit4Poly16;
    type CombR = zip_plus::code::binary_add_fft_16::packed_cw::PackedI64Poly16<BITS_CR>;
    type Comb = Self::CombR;
    type EvalDotChal = ScalarProduct;
    type CombDotChal = ScalarProduct;
    type ArrCombRDotChal = MBSInnerProduct;
}

/// D=16 polychal commit/prove/verify bench. `num_rows = 1`. Radix-8 FFT
/// requires `log2(row_len · REP) % 3 == 0`.
fn bench_polychal_16<const REP: usize, const NUM_OPENINGS: usize>(
    c: &mut Criterion,
    label: &str,
    num_vars: usize,
    batch: usize,
) {
    use criterion::{BenchmarkGroup, measurement::WallTime};

    type Zt<const NO: usize> = BenchAddFft16PolyChalPackedZipTypes<P_16_DEFAULT, NO>;
    type Code<const NO: usize, const R: usize> =
        BinaryAddFft16Code<Zt<NO>, AddFftConfigGF2_16, R, PERFORM_CHECKS>;
    type F = Gf2_16Ext<P_16_DEFAULT>;

    let mut group: BenchmarkGroup<WallTime> = c.benchmark_group(label);
    group.sample_size(20);

    let poly_size: usize = 1 << num_vars;
    let row_len = poly_size; // num_rows = 1
    assert!(
        poly_size.checked_mul(REP).map(|c| c <= (1 << 16)).unwrap_or(false),
        "D=16 + num_rows=1 needs poly_size*REP <= 2^16; got poly_size={poly_size}, REP={REP}",
    );
    let code = Code::<NUM_OPENINGS, REP>::new_signed(row_len).expect("valid params");
    let pp = ZipPlus::<Zt<NUM_OPENINGS>, Code<NUM_OPENINGS, REP>>::setup(poly_size, code);

    let mut rng = StdRng::seed_from_u64(0xcafe_f00d);
    let polys: Vec<DenseMultilinearExtension<BinaryPoly<16>>> = (0..batch)
        .map(|_| DenseMultilinearExtension::<BinaryPoly<16>>::rand(num_vars, &mut rng))
        .collect();

    let tag = format!("D16, REP={REP}, batch={batch}, poly_size=2^{num_vars}");

    group.bench_function(format!("Commit({tag})"), |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let timer = Instant::now();
                let res = ZipPlus::<Zt<NUM_OPENINGS>, Code<NUM_OPENINGS, REP>>::commit(&pp, &polys).unwrap();
                total += timer.elapsed();
                black_box(res);
            }
            total
        })
    });

    let (hint, comm) = ZipPlus::<Zt<NUM_OPENINGS>, Code<NUM_OPENINGS, REP>>::commit(&pp, &polys).unwrap();
    let field_cfg = ();
    let point: Vec<Bit4Poly16> = (0..num_vars).map(|i| Bit4Poly16(2 + i as i64)).collect();

    group.bench_function(format!("Prove({tag})"), |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let mut transcript = PcsProverTranscript::new_from_commitment(&comm);
                let timer = Instant::now();
                let res = ZipPlus::<Zt<NUM_OPENINGS>, Code<NUM_OPENINGS, REP>>::prove::<F, PERFORM_CHECKS>(
                    &mut transcript, &pp, &polys, &point, &hint, &field_cfg,
                ).unwrap();
                total += timer.elapsed();
                black_box(res);
            }
            total
        })
    });

    let mut transcript = PcsProverTranscript::new_from_commitment(&comm);
    let eval_f = ZipPlus::<Zt<NUM_OPENINGS>, Code<NUM_OPENINGS, REP>>::prove::<F, PERFORM_CHECKS>(
        &mut transcript, &pp, &polys, &point, &hint, &field_cfg,
    ).unwrap();

    let combined_proof = CombinedProof {
        comm: comm.clone(),
        proof_transcript: transcript.stream.get_ref().clone(),
    };
    zip_plus::utils::eprint_proof_size(format!("{tag}"), &combined_proof);

    let point_f: Vec<F> = point.iter()
        .map(|v| <F as crypto_primitives::FromWithConfig<&Bit4Poly16>>::from_with_cfg(v, &field_cfg))
        .collect();
    let v_transcript_template = transcript.into_verification_transcript();

    group.bench_function(format!("Verify({tag})"), |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let mut transcript = v_transcript_template.clone();
                let timer = Instant::now();
                transcript.fs_transcript.absorb_slice(&comm.root);
                let res = ZipPlus::<Zt<NUM_OPENINGS>, Code<NUM_OPENINGS, REP>>::verify::<F, PERFORM_CHECKS>(
                    &mut transcript, &pp, &comm, &field_cfg, &point_f, &eval_f,
                ).unwrap();
                total += timer.elapsed();
                black_box(res);
            }
            total
        })
    });

    group.finish();
}

// ===========================================================================
// D=16 polychal entry points.
//
// `num_vars = 10` mirrors the protocol's "folded" binary-lane dim at
// `protocol_num_vars=9` (each `BinaryPoly<32>` cell becomes 2 `BinaryPoly<16>`
// cells). Radix-8 constraint: `log2(poly_size · REP) % 3 == 0`.
//
// - R=4: `log2(2^10 · 4) = 12` ✓
// - R=8 at `num_vars=10` would give `m=13`, which fails. To keep `num_rows=1`
//   with R=8 we drop to `num_vars=9` (`m=12` ✓), committing HALF the bits —
//   not bit-budget-matched but a useful rate-axis sanity at the same `m`.
// ===========================================================================

fn zip_plus_benchmarks_polychal_d16_r4(c: &mut Criterion) {
    bench_polychal_16::<4, OPENINGS_REP_4>(
        c, "Zip+ PolyChal D16 R4 (Cw32/CombR8)",
        /* num_vars = */ 10, /* batch = */ 11,
    );
}

fn zip_plus_benchmarks_polychal_d16_r8(c: &mut Criterion) {
    bench_polychal_16::<8, OPENINGS_REP_8>(
        c, "Zip+ PolyChal D16 R8 (Cw32/CombR8)",
        /* num_vars = */ 9, /* batch = */ 11,
    );
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(500);
    targets =
        zip_plus_benchmarks_raa,
        zip_plus_benchmarks_iprs,
        zip_plus_benchmarks_sha_iprs,
        zip_plus_benchmarks_sha_ecdsa_iprs,
        zip_plus_benchmarks_polychal_d16_r4,
        zip_plus_benchmarks_polychal_d16_r8,
}
criterion_main!(benches);
