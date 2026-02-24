//! End-to-end SHA-256 benchmarks for the Zinc+ SNARK.
//!
//! Measures prover time, verifier time, and proof size for SHA-256
//! hashing plus (simulated) ECDSA verification using the Zinc+ pipeline:
//! Batched PCS (Zip+ with IPRS codes) + PIOP (ideal check + CPR + sumcheck).
//!
//! Target metrics (from the Zinc+ paper, MacBook Air M4):
//!   - Prover:  < 30 ms
//!   - Verifier: < 5 ms
//!   - Proof:   200–300 KB

#![allow(clippy::arithmetic_side_effects, clippy::unwrap_used)]

use std::hint::black_box;
use std::io::Write;
use std::marker::PhantomData;
use std::time::{Duration, Instant};

use criterion::{
    criterion_group, criterion_main, BenchmarkGroup, Criterion, measurement::WallTime,
};
use crypto_bigint::U64;
use crypto_primitives::{
    boolean::Boolean,
    crypto_bigint_int::Int,
    crypto_bigint_monty::MontyField,
    crypto_bigint_uint::Uint,
    Field, FixedSemiring, FromWithConfig,
};
use num_traits::One;


use zinc_poly::mle::{DenseMultilinearExtension, MultilinearExtensionRand};
use zinc_poly::univariate::binary::{
    BinaryPoly, BinaryPolyInnerProduct, BinaryPolyWideningMulByScalar,
};
use zinc_poly::univariate::dense::{DensePolyInnerProduct, DensePolynomial};
use zinc_primality::MillerRabin;
use zinc_transcript::traits::{ConstTranscribable, Transcript};
use zinc_utils::{
    UNCHECKED,
    from_ref::FromRef,
    inner_product::{MBSInnerProduct, ScalarProduct},
    mul_by_scalar::ScalarWideningMulByScalar,
    named::Named,
    projectable_to_field::ProjectableToField,
};
use zip_plus::{
    batched_pcs::structs::BatchedZipPlus,
    code::{
        LinearCode,
        iprs::{IprsCode, PnttConfigF2_16R4B16, PnttConfigF2_16R4B32, PnttConfigF2_16R4B64},
    },
    pcs::structs::{ZipPlusParams, ZipTypes},
};

use zinc_ecdsa_uair::EcdsaUair;
use zinc_sha256_uair::{Sha256Uair, witness::GenerateWitness};
use zinc_sha256_uair::witness::{generate_poly_witness, generate_int_witness};

// ─── Type definitions (matching batched_zip_plus_benches.rs) ────────────────

const INT_LIMBS: usize = U64::LIMBS;
type F = MontyField<{ INT_LIMBS * 4 }>;
/// 512-bit field for ECDSA PCS with Int<4> evaluations.
/// Wider than F because CombR = Int<8> (512-bit) requires Fmod = Uint<8>.
type FScalar = MontyField<{ INT_LIMBS * 8 }>;

struct Sha256ZipTypes<CwCoeff, const D_PLUS_ONE: usize>(PhantomData<CwCoeff>);

impl<CwCoeff, const D_PLUS_ONE: usize> ZipTypes for Sha256ZipTypes<CwCoeff, D_PLUS_ONE>
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
    type CombR = Int<{ INT_LIMBS * 6 }>;
    type Comb = DensePolynomial<Self::CombR, D_PLUS_ONE>;
    type EvalDotChal = BinaryPolyInnerProduct<Self::Chal, D_PLUS_ONE>;
    type CombDotChal =
        DensePolyInnerProduct<Self::CombR, Self::Chal, Self::CombR, MBSInnerProduct, D_PLUS_ONE>;
    type ArrCombRDotChal = MBSInnerProduct;
}

/// ZipTypes for ECDSA columns using scalar (Int<4>) evaluations.
///
/// ECDSA values are field elements (scalars), not polynomials.
/// Using Int<4> (256-bit) instead of BinaryPoly<32> saves ~32× per-cell
/// encoding cost in the IPRS NTT, since each cell is 1 coefficient
/// rather than 32 binary coefficients.
struct EcdsaScalarZipTypes;

impl ZipTypes for EcdsaScalarZipTypes {
    const NUM_COLUMN_OPENINGS: usize = 147;
    type Eval = Int<{ INT_LIMBS * 4 }>;      // 256-bit integer
    type Cw = Int<{ INT_LIMBS * 5 }>;        // 320-bit codeword
    type Fmod = Uint<{ INT_LIMBS * 8 }>;     // 512-bit modulus search
    type PrimeTest = MillerRabin;
    type Chal = i128;
    type Pt = i128;
    type CombR = Int<{ INT_LIMBS * 8 }>;     // 512-bit combination ring
    type Comb = Self::CombR;
    type EvalDotChal = ScalarProduct;
    type CombDotChal = ScalarProduct;
    type ArrCombRDotChal = MBSInnerProduct;
}

/// ZipTypes for the SHA-256 integer columns (indices 5–7, 10–14, 19) committed
/// as `Int<1>` (64-bit integer) instead of `BinaryPoly<32>` (256 bytes).
///
/// These columns only participate in Q[X] carry/BitPoly constraints (C7–C11)
/// and their values are 32-bit unsigned integers that fit in a single 64-bit limb.
/// Committing them with `Int<1>` reduces the codeword size from 256 B to ~16 B,
/// saving ~328 KB at 147 column openings for a single SHA-256 proof.
struct Sha256IntZipTypes;

impl ZipTypes for Sha256IntZipTypes {
    const NUM_COLUMN_OPENINGS: usize = 147;
    type Eval = Int<{ INT_LIMBS }>;           // 64-bit integer (Int<1>)
    type Cw = Int<{ INT_LIMBS * 2 }>;         // 128-bit codeword (Int<2>)
    type Fmod = Uint<{ INT_LIMBS * 4 }>;      // 256-bit modulus search
    type PrimeTest = MillerRabin;
    type Chal = i128;
    type Pt = i128;
    type CombR = Int<{ INT_LIMBS * 4 }>;      // 256-bit combination ring
    type Comb = Self::CombR;
    type EvalDotChal = ScalarProduct;
    type CombDotChal = ScalarProduct;
    type ArrCombRDotChal = MBSInnerProduct;
}

// IPRS code types for BinaryPoly<32>
type IprsBPoly32R4B64<const DEPTH: usize, const CHECK: bool> = IprsCode<
    Sha256ZipTypes<i64, 32>,
    PnttConfigF2_16R4B64<DEPTH>,
    BinaryPolyWideningMulByScalar<i64>,
    CHECK,
>;
type IprsBPoly32R4B16<const DEPTH: usize, const CHECK: bool> = IprsCode<
    Sha256ZipTypes<i64, 32>,
    PnttConfigF2_16R4B16<DEPTH>,
    BinaryPolyWideningMulByScalar<i64>,
    CHECK,
>;
type IprsBPoly32R4B32<const DEPTH: usize, const CHECK: bool> = IprsCode<
    Sha256ZipTypes<i64, 32>,
    PnttConfigF2_16R4B32<DEPTH>,
    BinaryPolyWideningMulByScalar<i64>,
    CHECK,
>;

// IPRS code types for Int<4> (scalar ECDSA evaluations)
type IprsInt4R4B64<const DEPTH: usize, const CHECK: bool> = IprsCode<
    EcdsaScalarZipTypes,
    PnttConfigF2_16R4B64<DEPTH>,
    ScalarWideningMulByScalar<Int<{ INT_LIMBS * 5 }>>,
    CHECK,
>;
type IprsInt4R4B16<const DEPTH: usize, const CHECK: bool> = IprsCode<
    EcdsaScalarZipTypes,
    PnttConfigF2_16R4B16<DEPTH>,
    ScalarWideningMulByScalar<Int<{ INT_LIMBS * 5 }>>,
    CHECK,
>;

// IPRS code types for Int<1> (SHA-256 integer columns)
type IprsInt1R4B64<const DEPTH: usize, const CHECK: bool> = IprsCode<
    Sha256IntZipTypes,
    PnttConfigF2_16R4B64<DEPTH>,
    ScalarWideningMulByScalar<Int<{ INT_LIMBS * 2 }>>,
    CHECK,
>;
type IprsInt1R4B16<const DEPTH: usize, const CHECK: bool> = IprsCode<
    Sha256IntZipTypes,
    PnttConfigF2_16R4B16<DEPTH>,
    ScalarWideningMulByScalar<Int<{ INT_LIMBS * 2 }>>,
    CHECK,
>;
type IprsInt1R4B32<const DEPTH: usize, const CHECK: bool> = IprsCode<
    Sha256IntZipTypes,
    PnttConfigF2_16R4B32<DEPTH>,
    ScalarWideningMulByScalar<Int<{ INT_LIMBS * 2 }>>,
    CHECK,
>;

// ─── SHA-256 Trace Parameters ───────────────────────────────────────────────
//
// SHA-256 has 64 rows × 19 columns.
// ECDSA has 258 rows × 9 columns.
//
// IPRS codes need DEPTH ≥ 1 (radix-8 NTT), and row_len = BASE_LEN × 8^DEPTH.
//
// Benchmark configurations:
//   - Single SHA-256: poly_size = 2^7 = 128, 19 polys, R4B16 DEPTH=1 (row_len=128)
//     → 64 real rows + 64 padding = 128 total
//   - 8× SHA-256 + ECDSA: poly_size = 2^10 = 1024, 33 polys
//     → 8×64=512 SHA rows + 258 ECDSA rows = 770 → pad to 1024

const SHA256_NUM_VARS: usize = 7;        // 2^7 = 128 rows (64 real + 64 padding)
const SHA256_BATCH_SIZE: usize = 20;      // 20 SHA-256 columns

// Split SHA-256 batch sizes for two-PCS configuration
const SHA256_POLY_BATCH_SIZE: usize = 11; // 11 BinaryPoly columns (C1–C6)
const SHA256_INT_BATCH_SIZE: usize = 9;   // 9 Int columns (C7–C11)

// For 8× SHA-256 + ECDSA: two separate PCS batches
const SHA256_8X_NUM_VARS: usize = 9;           // 2^9 = 512 rows (8 × 64 SHA rounds)
const ECDSA_NUM_VARS: usize = 9;               // 2^9 = 512 rows (258 ECDSA rows + padding)
const ECDSA_BATCH_SIZE: usize = 9;             // 9 ECDSA columns

// ─── Benchmark helpers ──────────────────────────────────────────────────────

fn generate_sha256_trace(num_vars: usize) -> Vec<DenseMultilinearExtension<BinaryPoly<32>>> {
    let mut rng = rand::rng();
    <Sha256Uair as GenerateWitness<BinaryPoly<32>>>::generate_witness(num_vars, &mut rng)
}

fn generate_random_trace(
    num_vars: usize,
    num_cols: usize,
) -> Vec<DenseMultilinearExtension<BinaryPoly<32>>> {
    let mut rng = rand::rng();
    (0..num_cols)
        .map(|_| DenseMultilinearExtension::rand(num_vars, &mut rng))
        .collect()
}

/// Generate random ECDSA-shaped trace with Int<4> (256-bit scalar) evaluations.
///
/// Using scalars instead of BinaryPoly<32> makes PCS encoding ~32× cheaper
/// per cell since each evaluation is 1 coefficient rather than 32.
fn generate_random_scalar_trace(
    num_vars: usize,
    num_cols: usize,
) -> Vec<DenseMultilinearExtension<Int<{ INT_LIMBS * 4 }>>> {
    let mut rng = rand::rng();
    (0..num_cols)
        .map(|_| DenseMultilinearExtension::rand(num_vars, &mut rng))
        .collect()
}

/// Benchmark the full PCS pipeline (commit + test + evaluate + verify)
/// for a SHA-256-like trace.
fn bench_pcs_pipeline<Zt, Lc, const P: usize, const CHECK: bool>(
    group: &mut BenchmarkGroup<WallTime>,
    label: &str,
    num_rows: usize,
    _batch_size: usize,
    trace: &[DenseMultilinearExtension<BinaryPoly<32>>],
) where
    Zt: ZipTypes<Eval = BinaryPoly<32>>,
    Lc: LinearCode<Zt>,
    F: for<'a> FromWithConfig<&'a Zt::Chal> + for<'a> FromWithConfig<&'a Zt::Pt>,
    <F as Field>::Inner: FromRef<Zt::Fmod>,
    Zt::Eval: ProjectableToField<F>,
    Zt::Cw: ProjectableToField<F>,
{
    let poly_size = 1 << P;
    let row_len = poly_size / num_rows;
    let linear_code = Lc::new(row_len);
    let params = ZipPlusParams::new(P, num_rows, linear_code);

    // Warm up: commit once
    let (hint, commitment) = BatchedZipPlus::<Zt, Lc>::commit(&params, trace)
        .expect("commit failed");

    // ── Prover benchmark (commit + test + evaluate) ─────────────────
    group.bench_function(format!("{label}/Prover"), |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let t = Instant::now();

                // Commit
                let (hint, _) = BatchedZipPlus::<Zt, Lc>::commit(&params, trace)
                    .expect("commit");

                // Test
                let test_tx = BatchedZipPlus::<Zt, Lc>::test::<CHECK>(&params, trace, &hint)
                    .expect("test");

                // Evaluate
                let point: Vec<Zt::Pt> = vec![Zt::Pt::one(); P];
                let (_evals, _proof) =
                    BatchedZipPlus::<Zt, Lc>::evaluate::<F, CHECK>(
                        &params, trace, &point, test_tx,
                    )
                    .expect("evaluate");

                total += t.elapsed();
            }
            total
        });
    });

    // Prepare proof for verifier benchmark
    let test_tx = BatchedZipPlus::<Zt, Lc>::test::<CHECK>(&params, trace, &hint)
        .expect("test");
    let point: Vec<Zt::Pt> = vec![Zt::Pt::one(); P];
    let (_evals_f, proof) =
        BatchedZipPlus::<Zt, Lc>::evaluate::<F, CHECK>(&params, trace, &point, test_tx)
            .expect("evaluate");

    // ── Verifier benchmark ──────────────────────────────────────────
    group.bench_function(format!("{label}/Verifier"), |b| {
        b.iter(|| {
            let result = BatchedZipPlus::<Zt, Lc>::verify::<F, CHECK>(
                &params,
                &commitment,
                &point,
                &proof,
            );
            let _ = black_box(result);
        });
    });

    // ── Proof size ──────────────────────────────────────────────────
    let proof_bytes: Vec<u8> = {
        let tx: zip_plus::pcs_transcript::PcsTranscript = proof.into();
        tx.stream.into_inner()
    };
    // Measure compressed size (deflate level 6)
    let mut encoder = flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::default());
    encoder.write_all(&proof_bytes).unwrap();
    let compressed = encoder.finish().unwrap();
    eprintln!(
        "  {label}: proof_size = {} bytes ({:.1} KB), compressed = {} bytes ({:.1} KB, {:.1}× ratio)",
        proof_bytes.len(),
        proof_bytes.len() as f64 / 1024.0,
        compressed.len(),
        compressed.len() as f64 / 1024.0,
        proof_bytes.len() as f64 / compressed.len() as f64,
    );
}

// ─── Criterion suites ───────────────────────────────────────────────────────

/// Single SHA-256 compression (64 rows padded to 128 × 19 columns = poly_size 2^7).
///
/// Uses R4B16 IPRS codes at rate 1/4 with DEPTH=1 (row_len=128).
///
/// Reports both PCS-only and full pipeline (PCS + PIOP) timings so the
/// paper can cite an honest "total prover time" that includes ideal check
/// and combined-poly-resolver overhead.
///
/// **Proof size claims are valid for DEPTH=1 configurations only.**
/// DEPTH≥2 dramatically reduces deflate compressibility (see §4 of review).
fn sha256_single(c: &mut Criterion) {
    use zinc_sha256_uair::CyclotomicIdeal;
    use zinc_uair::ideal_collector::IdealOrZero;

    let mut group = c.benchmark_group("SHA-256 Single Compression");
    group.sample_size(20);

    let trace = generate_sha256_trace(SHA256_NUM_VARS);

    // ── PCS-only (component benchmark) ──────────────────────────────
    // poly_size = 2^7 = 128, num_rows = 1, row_len = 128
    // R4B16 DEPTH=1 gives row_len = 16 × 8^1 = 128  ✓
    bench_pcs_pipeline::<Sha256ZipTypes<i64, 32>, IprsBPoly32R4B16<1, UNCHECKED>, 7, UNCHECKED>(
        &mut group,
        "1xSHA256/PCS-only",
        1,
        SHA256_BATCH_SIZE,
        &trace,
    );

    // ── Split PCS: BinaryPoly batch + Int batch ─────────────────────
    // Split the 20 SHA-256 columns into two PCS batches:
    //   - BinaryPoly batch (11 cols): columns used in F₂[X] rotation constraints
    //   - Int batch (9 cols): columns only used in Q[X] carry/BitPoly constraints
    // The PIOP still runs on the full 20-column BinaryPoly<32> trace.
    {
        let mut rng = rand::rng();
        let poly_trace = generate_poly_witness(SHA256_NUM_VARS, &mut rng);
        let int_trace = generate_int_witness(SHA256_NUM_VARS, &mut rng);
        assert_eq!(poly_trace.len(), SHA256_POLY_BATCH_SIZE);
        assert_eq!(int_trace.len(), SHA256_INT_BATCH_SIZE);

        type PolyZt = Sha256ZipTypes<i64, 32>;
        type PolyLc = IprsBPoly32R4B16<1, UNCHECKED>;
        type IntZt = Sha256IntZipTypes;
        type IntLc = IprsInt1R4B16<1, UNCHECKED>;

        let poly_lc = PolyLc::new(128);
        let poly_params = ZipPlusParams::<PolyZt, PolyLc>::new(SHA256_NUM_VARS, 1, poly_lc);
        let int_lc = IntLc::new(128);
        let int_params = ZipPlusParams::<IntZt, IntLc>::new(SHA256_NUM_VARS, 1, int_lc);

        // ── Prover (both batches) ───────────────────────────────────
        group.bench_function("1xSHA256/SplitPCS/Prover", |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let t = Instant::now();

                    // BinaryPoly batch
                    let (poly_hint, _) = BatchedZipPlus::<PolyZt, PolyLc>::commit(
                        &poly_params, &poly_trace,
                    ).expect("poly commit");
                    let poly_tx = BatchedZipPlus::<PolyZt, PolyLc>::test::<UNCHECKED>(
                        &poly_params, &poly_trace, &poly_hint,
                    ).expect("poly test");
                    let poly_pt: Vec<i128> = vec![1i128; SHA256_NUM_VARS];
                    let _ = BatchedZipPlus::<PolyZt, PolyLc>::evaluate::<F, UNCHECKED>(
                        &poly_params, &poly_trace, &poly_pt, poly_tx,
                    ).expect("poly evaluate");

                    // Int batch
                    let (int_hint, _) = BatchedZipPlus::<IntZt, IntLc>::commit(
                        &int_params, &int_trace,
                    ).expect("int commit");
                    let int_tx = BatchedZipPlus::<IntZt, IntLc>::test::<UNCHECKED>(
                        &int_params, &int_trace, &int_hint,
                    ).expect("int test");
                    let int_pt: Vec<i128> = vec![1i128; SHA256_NUM_VARS];
                    let _ = BatchedZipPlus::<IntZt, IntLc>::evaluate::<F, UNCHECKED>(
                        &int_params, &int_trace, &int_pt, int_tx,
                    ).expect("int evaluate");

                    total += t.elapsed();
                }
                total
            });
        });

        // Prepare proofs for verifier and size measurement
        let (poly_hint, poly_comm) = BatchedZipPlus::<PolyZt, PolyLc>::commit(
            &poly_params, &poly_trace,
        ).expect("commit");
        let poly_tx = BatchedZipPlus::<PolyZt, PolyLc>::test::<UNCHECKED>(
            &poly_params, &poly_trace, &poly_hint,
        ).expect("test");
        let poly_pt: Vec<i128> = vec![1i128; SHA256_NUM_VARS];
        let (_poly_evals, poly_proof) = BatchedZipPlus::<PolyZt, PolyLc>::evaluate::<F, UNCHECKED>(
            &poly_params, &poly_trace, &poly_pt, poly_tx,
        ).expect("evaluate");

        let (int_hint, int_comm) = BatchedZipPlus::<IntZt, IntLc>::commit(
            &int_params, &int_trace,
        ).expect("commit");
        let int_tx = BatchedZipPlus::<IntZt, IntLc>::test::<UNCHECKED>(
            &int_params, &int_trace, &int_hint,
        ).expect("test");
        let int_pt: Vec<i128> = vec![1i128; SHA256_NUM_VARS];
        let (_int_evals, int_proof) = BatchedZipPlus::<IntZt, IntLc>::evaluate::<F, UNCHECKED>(
            &int_params, &int_trace, &int_pt, int_tx,
        ).expect("evaluate");

        // ── Verifier (both batches) ─────────────────────────────────
        group.bench_function("1xSHA256/SplitPCS/Verifier", |b| {
            b.iter(|| {
                let r1 = BatchedZipPlus::<PolyZt, PolyLc>::verify::<F, UNCHECKED>(
                    &poly_params, &poly_comm, &poly_pt, &poly_proof,
                );
                let _ = black_box(r1);
                let r2 = BatchedZipPlus::<IntZt, IntLc>::verify::<F, UNCHECKED>(
                    &int_params, &int_comm, &int_pt, &int_proof,
                );
                let _ = black_box(r2);
            });
        });

        // ── Proof size comparison ───────────────────────────────────
        let poly_bytes: Vec<u8> = {
            let tx: zip_plus::pcs_transcript::PcsTranscript = poly_proof.into();
            tx.stream.into_inner()
        };
        let int_bytes: Vec<u8> = {
            let tx: zip_plus::pcs_transcript::PcsTranscript = int_proof.into();
            tx.stream.into_inner()
        };
        let combined_size = poly_bytes.len() + int_bytes.len();

        let mut all_bytes = Vec::new();
        all_bytes.extend(&poly_bytes);
        all_bytes.extend(&int_bytes);
        let mut encoder = flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::default());
        encoder.write_all(&all_bytes).unwrap();
        let compressed = encoder.finish().unwrap();

        eprintln!("\n=== Split PCS Proof Size (1xSHA256) ===");
        eprintln!("  BinaryPoly batch ({} cols): {} bytes ({:.1} KB)",
            SHA256_POLY_BATCH_SIZE, poly_bytes.len(), poly_bytes.len() as f64 / 1024.0);
        eprintln!("  Int batch ({} cols):        {} bytes ({:.1} KB)",
            SHA256_INT_BATCH_SIZE, int_bytes.len(), int_bytes.len() as f64 / 1024.0);
        eprintln!("  Combined raw:  {} bytes ({:.1} KB)", combined_size, combined_size as f64 / 1024.0);
        eprintln!("  Compressed:    {} bytes ({:.1} KB, {:.1}× ratio)",
            compressed.len(), compressed.len() as f64 / 1024.0,
            all_bytes.len() as f64 / compressed.len() as f64);
    }

    // ── Full pipeline: PCS + PIOP (headline benchmark) ──────────────
    // This is the honest prover time: IC + CPR + PCS commit + test + evaluate.
    {
        type Lc = IprsBPoly32R4B16<1, UNCHECKED>;
        type Zt = Sha256ZipTypes<i64, 32>;
        let linear_code = Lc::new(128);
        let params = ZipPlusParams::new(SHA256_NUM_VARS, 1, linear_code);

        group.bench_function("1xSHA256/FullPipeline/Prover", |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let t = Instant::now();
                    let _proof = zinc_snark::pipeline::prove::<Sha256Uair, Zt, Lc, 32, UNCHECKED>(
                        &params,
                        &trace,
                        SHA256_NUM_VARS,
                    );
                    total += t.elapsed();
                }
                total
            });
        });

        // Prepare proof for verifier benchmark
        let zinc_proof = zinc_snark::pipeline::prove::<Sha256Uair, Zt, Lc, 32, UNCHECKED>(
            &params,
            &trace,
            SHA256_NUM_VARS,
        );

        group.bench_function("1xSHA256/FullPipeline/Verifier", |b| {
            b.iter(|| {
                let result = zinc_snark::pipeline::verify::<Sha256Uair, Zt, Lc, 32, UNCHECKED, _, _>(
                    &params,
                    &zinc_proof,
                    SHA256_NUM_VARS,
                    |_: &IdealOrZero<CyclotomicIdeal>| zinc_snark::pipeline::TrivialIdeal,
                );
                black_box(result);
            });
        });

        eprintln!(
            "\n  Full pipeline prover breakdown: IC={:?}, CPR={:?}, PCS commit={:?}, test={:?}, evaluate={:?}, total={:?}",
            zinc_proof.timing.ideal_check,
            zinc_proof.timing.combined_poly_resolver,
            zinc_proof.timing.pcs_commit,
            zinc_proof.timing.pcs_test,
            zinc_proof.timing.pcs_evaluate,
            zinc_proof.timing.total,
        );
    }

    group.finish();
}

/// 8× SHA-256 + ECDSA — the paper's target benchmark configuration.
///
/// Uses **two separate PCS batches** with evaluation types matched to the
/// arithmetic of each UAIR:
/// - SHA-256:  20 columns × BinaryPoly<32> (polynomial evaluations, 32 binary coefficients)
/// - ECDSA:    9 columns × Int<4>           (scalar evaluations, 1 × 256-bit integer)
///
/// This is ~32× cheaper per cell for the ECDSA columns compared to
/// BinaryPoly<32>, because the IPRS NTT processes 1 coefficient rather than 32.
///
/// Both batches use DEPTH=1 (row_len=512, poly_size=512), which gives
/// valid proof-size claims under deflate compression.
///
/// Target metrics (MacBook Air M4):
///   - Combined prover:  < 30 ms (or just above)
///   - Combined verifier: < 5 ms
fn sha256_8x_ecdsa(c: &mut Criterion) {
    let mut group = c.benchmark_group("8xSHA256+ECDSA");
    group.sample_size(10);

    // ── Build traces ────────────────────────────────────────────────
    // SHA-256: 20 columns × 512 rows (8 instances × 64 rounds)
    let sha_trace = generate_sha256_trace(SHA256_8X_NUM_VARS);
    assert_eq!(sha_trace.len(), SHA256_BATCH_SIZE);

    // ECDSA: 9 columns × 512 rows (258 real + 254 padding)
    // Int<4> = 256-bit scalar — matches secp256k1 field element width
    let ecdsa_trace = generate_random_scalar_trace(ECDSA_NUM_VARS, ECDSA_BATCH_SIZE);
    assert_eq!(ecdsa_trace.len(), ECDSA_BATCH_SIZE);

    // ── PCS params ──────────────────────────────────────────────────
    // SHA: BinaryPoly<32>, R4B64 DEPTH=1 → row_len = 64 × 8 = 512 ✓
    type ShaZt = Sha256ZipTypes<i64, 32>;
    type ShaLc = IprsBPoly32R4B64<1, UNCHECKED>;
    let sha_params = ZipPlusParams::<ShaZt, ShaLc>::new(
        SHA256_8X_NUM_VARS, 1, ShaLc::new(512),
    );

    // ECDSA: Int<4>, R4B64 DEPTH=1 → row_len = 64 × 8 = 512 ✓
    type EcZt = EcdsaScalarZipTypes;
    type EcLc = IprsInt4R4B64<1, UNCHECKED>;
    let ec_params = ZipPlusParams::<EcZt, EcLc>::new(
        ECDSA_NUM_VARS, 1, EcLc::new(512),
    );

    // ── Combined Prover (SHA PCS + ECDSA PCS) ───────────────────────
    group.bench_function("Combined/PCS/Prover", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let t = Instant::now();

                // SHA PCS: commit + test + evaluate
                let (sha_hint, _) = BatchedZipPlus::<ShaZt, ShaLc>::commit(
                    &sha_params, &sha_trace,
                ).expect("sha commit");
                let sha_tx = BatchedZipPlus::<ShaZt, ShaLc>::test::<UNCHECKED>(
                    &sha_params, &sha_trace, &sha_hint,
                ).expect("sha test");
                let sha_pt: Vec<i128> = vec![1i128; SHA256_8X_NUM_VARS];
                let _ = BatchedZipPlus::<ShaZt, ShaLc>::evaluate::<F, UNCHECKED>(
                    &sha_params, &sha_trace, &sha_pt, sha_tx,
                ).expect("sha evaluate");

                // ECDSA PCS: commit + test + evaluate (Int<4> — ~32× cheaper per cell)
                let (ec_hint, _) = BatchedZipPlus::<EcZt, EcLc>::commit(
                    &ec_params, &ecdsa_trace,
                ).expect("ecdsa commit");
                let ec_tx = BatchedZipPlus::<EcZt, EcLc>::test::<UNCHECKED>(
                    &ec_params, &ecdsa_trace, &ec_hint,
                ).expect("ecdsa test");
                let ec_pt: Vec<i128> = vec![1i128; ECDSA_NUM_VARS];
                let _ = BatchedZipPlus::<EcZt, EcLc>::evaluate::<FScalar, UNCHECKED>(
                    &ec_params, &ecdsa_trace, &ec_pt, ec_tx,
                ).expect("ecdsa evaluate");

                total += t.elapsed();
            }
            total
        });
    });

    // ── Combined Verifier ───────────────────────────────────────────
    // Prepare SHA proof
    let (sha_hint, sha_comm) = BatchedZipPlus::<ShaZt, ShaLc>::commit(
        &sha_params, &sha_trace,
    ).expect("commit");
    let sha_tx = BatchedZipPlus::<ShaZt, ShaLc>::test::<UNCHECKED>(
        &sha_params, &sha_trace, &sha_hint,
    ).expect("test");
    let sha_pt: Vec<i128> = vec![1i128; SHA256_8X_NUM_VARS];
    let (_sha_evals_f, sha_proof) = BatchedZipPlus::<ShaZt, ShaLc>::evaluate::<F, UNCHECKED>(
        &sha_params, &sha_trace, &sha_pt, sha_tx,
    ).expect("evaluate");

    // Prepare ECDSA proof
    let (ec_hint, ec_comm) = BatchedZipPlus::<EcZt, EcLc>::commit(
        &ec_params, &ecdsa_trace,
    ).expect("commit");
    let ec_tx = BatchedZipPlus::<EcZt, EcLc>::test::<UNCHECKED>(
        &ec_params, &ecdsa_trace, &ec_hint,
    ).expect("test");
    let ec_pt: Vec<i128> = vec![1i128; ECDSA_NUM_VARS];
    let (_ec_evals_f, ec_proof) = BatchedZipPlus::<EcZt, EcLc>::evaluate::<FScalar, UNCHECKED>(
        &ec_params, &ecdsa_trace, &ec_pt, ec_tx,
    ).expect("evaluate");

    group.bench_function("Combined/PCS/Verifier", |b| {
        b.iter(|| {
            let (sha_r, ec_r) = rayon::join(
                || {
                    BatchedZipPlus::<ShaZt, ShaLc>::verify::<F, UNCHECKED>(
                        &sha_params, &sha_comm, &sha_pt, &sha_proof,
                    )
                },
                || {
                    BatchedZipPlus::<EcZt, EcLc>::verify::<FScalar, UNCHECKED>(
                        &ec_params, &ec_comm, &ec_pt, &ec_proof,
                    )
                },
            );
            let _ = black_box(sha_r);
            let _ = black_box(ec_r);
        });
    });

    // ── Proof sizes (full pipeline: PCS + PIOP) ─────────────────────
    // Generate full-pipeline proofs for accurate size reporting.
    // SHA: pipeline::prove (BinaryPoly<32>, single-ring PIOP)
    let sha_zinc_proof = zinc_snark::pipeline::prove::<Sha256Uair, ShaZt, ShaLc, 32, UNCHECKED>(
        &sha_params,
        &sha_trace,
        SHA256_8X_NUM_VARS,
    );
    // ECDSA: pipeline::prove_generic (Int<4>, single-ring PIOP)
    let ec_zinc_proof = zinc_snark::pipeline::prove_generic::< 
        EcdsaUair,
        Int<{ INT_LIMBS * 4 }>,
        EcZt,
        EcLc,
        FScalar,
        UNCHECKED,
    >(
        &ec_params,
        &ecdsa_trace,
        ECDSA_NUM_VARS,
    );

    // Helper to compute PIOP proof size from a ZincProof.
    fn piop_size(p: &zinc_snark::pipeline::ZincProof) -> usize {
        p.ic_proof_values.iter().map(|v| v.len()).sum::<usize>()
            + p.cpr_sumcheck_messages.iter().map(|v| v.len()).sum::<usize>()
            + p.cpr_sumcheck_claimed_sum.len()
            + p.cpr_up_evals.iter().map(|v| v.len()).sum::<usize>()
            + p.cpr_down_evals.iter().map(|v| v.len()).sum::<usize>()
            + p.evaluation_point_bytes.iter().map(|v| v.len()).sum::<usize>()
            + p.pcs_evals_bytes.iter().map(|v| v.len()).sum::<usize>()
    }

    let sha_pcs = sha_zinc_proof.pcs_proof_bytes.len();
    let sha_piop = piop_size(&sha_zinc_proof);
    let ec_pcs = ec_zinc_proof.pcs_proof_bytes.len();
    let ec_piop = piop_size(&ec_zinc_proof);
    let total_raw = sha_pcs + sha_piop + ec_pcs + ec_piop;

    // Collect all bytes for compression measurement.
    let mut all_bytes = Vec::new();
    for p in [&sha_zinc_proof, &ec_zinc_proof] {
        all_bytes.extend(&p.pcs_proof_bytes);
        for v in &p.ic_proof_values { all_bytes.extend(v); }
        for v in &p.cpr_sumcheck_messages { all_bytes.extend(v); }
        all_bytes.extend(&p.cpr_sumcheck_claimed_sum);
        for v in &p.cpr_up_evals { all_bytes.extend(v); }
        for v in &p.cpr_down_evals { all_bytes.extend(v); }
        for v in &p.evaluation_point_bytes { all_bytes.extend(v); }
        for v in &p.pcs_evals_bytes { all_bytes.extend(v); }
    }
    let mut encoder = flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::default());
    encoder.write_all(&all_bytes).unwrap();
    let compressed = encoder.finish().unwrap();

    // Compute analytical breakdown of PCS proof components.
    // PCS proof = proximity_phase + column_opening_phase + eval_phase
    //   proximity = row_len × CombR_bytes
    //   column_openings = NUM_OPENINGS × (batch_size × num_rows × Cw_bytes + merkle_proof)
    //   eval = row_len × PcsF_bytes + batch_size × PcsF_bytes  (+length prefixes)
    let sha_col_per_opening = SHA256_BATCH_SIZE * 1 * <ShaZt as ZipTypes>::Cw::NUM_BYTES;
    let ec_col_per_opening = ECDSA_BATCH_SIZE * 1 * <EcZt as ZipTypes>::Cw::NUM_BYTES;

    eprintln!("\n=== 8xSHA256+ECDSA Combined Proof Size (NUM_COLUMN_OPENINGS={}) ===",
        <ShaZt as ZipTypes>::NUM_COLUMN_OPENINGS);
    eprintln!("  SHA:   PCS = {} bytes ({:.1} KB), PIOP = {} bytes ({:.1} KB)",
        sha_pcs, sha_pcs as f64 / 1024.0,
        sha_piop, sha_piop as f64 / 1024.0,
    );
    eprintln!("    column values/opening: {} cols × {} B/cw = {} B",
        SHA256_BATCH_SIZE, <ShaZt as ZipTypes>::Cw::NUM_BYTES, sha_col_per_opening);
    eprintln!("    CombR = Int<6> = {} B, PcsF = MontyField<4> ≈ 16 B",
        <ShaZt as ZipTypes>::CombR::NUM_BYTES);
    eprintln!("  ECDSA: PCS = {} bytes ({:.1} KB), PIOP = {} bytes ({:.1} KB)",
        ec_pcs, ec_pcs as f64 / 1024.0,
        ec_piop, ec_piop as f64 / 1024.0,
    );
    eprintln!("    column values/opening: {} cols × {} B/cw = {} B",
        ECDSA_BATCH_SIZE, <EcZt as ZipTypes>::Cw::NUM_BYTES, ec_col_per_opening);
    eprintln!("    CombR = Int<8> = {} B, PcsF = MontyField<8> ≈ 64 B",
        <EcZt as ZipTypes>::CombR::NUM_BYTES);
    eprintln!("  SHA/ECDSA PCS ratio:  {:.2}×", sha_pcs as f64 / ec_pcs as f64);
    eprintln!("  Total raw:  {} bytes ({:.1} KB)",
        total_raw, total_raw as f64 / 1024.0,
    );
    eprintln!("  Compressed: {} bytes ({:.1} KB, {:.1}× ratio)",
        compressed.len(), compressed.len() as f64 / 1024.0,
        all_bytes.len() as f64 / compressed.len() as f64,
    );

    // ── Split SHA PCS: BinaryPoly (11 cols) + Int (9 cols) + ECDSA ──
    // Splits the 20 SHA-256 columns into two PCS batches by column type,
    // keeping the ECDSA batch unchanged.  Three PCS batches total:
    //   1. SHA BinaryPoly (11 cols) — rotation/shift constraints
    //   2. SHA Int<1>     (9 cols)  — carry/BitPoly constraints
    //   3. ECDSA Int<4>   (9 cols) — scalar arithmetic
    {
        let mut rng = rand::rng();
        let sha_poly_trace = generate_poly_witness(SHA256_8X_NUM_VARS, &mut rng);
        let sha_int_trace  = generate_int_witness(SHA256_8X_NUM_VARS, &mut rng);
        assert_eq!(sha_poly_trace.len(), SHA256_POLY_BATCH_SIZE);
        assert_eq!(sha_int_trace.len(), SHA256_INT_BATCH_SIZE);

        // SHA BinaryPoly batch: same ZipTypes as mono, R4B64 DEPTH=1
        type ShaPolyZt = Sha256ZipTypes<i64, 32>;
        type ShaPolyLc = IprsBPoly32R4B64<1, UNCHECKED>;
        let sha_poly_params = ZipPlusParams::<ShaPolyZt, ShaPolyLc>::new(
            SHA256_8X_NUM_VARS, 1, ShaPolyLc::new(512),
        );

        // SHA Int batch: Int<1> ZipTypes, R4B64 DEPTH=1
        type ShaIntZt = Sha256IntZipTypes;
        type ShaIntLc = IprsInt1R4B64<1, UNCHECKED>;
        let sha_int_params = ZipPlusParams::<ShaIntZt, ShaIntLc>::new(
            SHA256_8X_NUM_VARS, 1, ShaIntLc::new(512),
        );

        // ── Split Prover (3 batches) ────────────────────────────────
        group.bench_function("SplitSHA/PCS/Prover", |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let t = Instant::now();

                    // SHA BinaryPoly batch
                    let (h, _) = BatchedZipPlus::<ShaPolyZt, ShaPolyLc>::commit(
                        &sha_poly_params, &sha_poly_trace,
                    ).expect("sha poly commit");
                    let tx = BatchedZipPlus::<ShaPolyZt, ShaPolyLc>::test::<UNCHECKED>(
                        &sha_poly_params, &sha_poly_trace, &h,
                    ).expect("sha poly test");
                    let pt: Vec<i128> = vec![1i128; SHA256_8X_NUM_VARS];
                    let _ = BatchedZipPlus::<ShaPolyZt, ShaPolyLc>::evaluate::<F, UNCHECKED>(
                        &sha_poly_params, &sha_poly_trace, &pt, tx,
                    ).expect("sha poly evaluate");

                    // SHA Int batch
                    let (h, _) = BatchedZipPlus::<ShaIntZt, ShaIntLc>::commit(
                        &sha_int_params, &sha_int_trace,
                    ).expect("sha int commit");
                    let tx = BatchedZipPlus::<ShaIntZt, ShaIntLc>::test::<UNCHECKED>(
                        &sha_int_params, &sha_int_trace, &h,
                    ).expect("sha int test");
                    let pt: Vec<i128> = vec![1i128; SHA256_8X_NUM_VARS];
                    let _ = BatchedZipPlus::<ShaIntZt, ShaIntLc>::evaluate::<F, UNCHECKED>(
                        &sha_int_params, &sha_int_trace, &pt, tx,
                    ).expect("sha int evaluate");

                    // ECDSA batch (unchanged)
                    let (h, _) = BatchedZipPlus::<EcZt, EcLc>::commit(
                        &ec_params, &ecdsa_trace,
                    ).expect("ecdsa commit");
                    let tx = BatchedZipPlus::<EcZt, EcLc>::test::<UNCHECKED>(
                        &ec_params, &ecdsa_trace, &h,
                    ).expect("ecdsa test");
                    let pt: Vec<i128> = vec![1i128; ECDSA_NUM_VARS];
                    let _ = BatchedZipPlus::<EcZt, EcLc>::evaluate::<FScalar, UNCHECKED>(
                        &ec_params, &ecdsa_trace, &pt, tx,
                    ).expect("ecdsa evaluate");

                    total += t.elapsed();
                }
                total
            });
        });

        // Prepare proofs for verifier and size measurement
        let (sp_h, sp_comm) = BatchedZipPlus::<ShaPolyZt, ShaPolyLc>::commit(
            &sha_poly_params, &sha_poly_trace,
        ).expect("commit");
        let sp_tx = BatchedZipPlus::<ShaPolyZt, ShaPolyLc>::test::<UNCHECKED>(
            &sha_poly_params, &sha_poly_trace, &sp_h,
        ).expect("test");
        let sp_pt: Vec<i128> = vec![1i128; SHA256_8X_NUM_VARS];
        let (_, sp_proof) = BatchedZipPlus::<ShaPolyZt, ShaPolyLc>::evaluate::<F, UNCHECKED>(
            &sha_poly_params, &sha_poly_trace, &sp_pt, sp_tx,
        ).expect("evaluate");

        let (si_h, si_comm) = BatchedZipPlus::<ShaIntZt, ShaIntLc>::commit(
            &sha_int_params, &sha_int_trace,
        ).expect("commit");
        let si_tx = BatchedZipPlus::<ShaIntZt, ShaIntLc>::test::<UNCHECKED>(
            &sha_int_params, &sha_int_trace, &si_h,
        ).expect("test");
        let si_pt: Vec<i128> = vec![1i128; SHA256_8X_NUM_VARS];
        let (_, si_proof) = BatchedZipPlus::<ShaIntZt, ShaIntLc>::evaluate::<F, UNCHECKED>(
            &sha_int_params, &sha_int_trace, &si_pt, si_tx,
        ).expect("evaluate");

        // Reuse ECDSA proof from above (ec_comm, ec_pt, ec_proof)

        // ── Split Verifier (3 batches) ──────────────────────────────
        group.bench_function("SplitSHA/PCS/Verifier", |b| {
            b.iter(|| {
                // Run all 3 verify calls concurrently using rayon::join
                let ((r1, r2), r3) = rayon::join(
                    || {
                        rayon::join(
                            || {
                                BatchedZipPlus::<ShaPolyZt, ShaPolyLc>::verify::<F, UNCHECKED>(
                                    &sha_poly_params, &sp_comm, &sp_pt, &sp_proof,
                                )
                            },
                            || {
                                BatchedZipPlus::<ShaIntZt, ShaIntLc>::verify::<F, UNCHECKED>(
                                    &sha_int_params, &si_comm, &si_pt, &si_proof,
                                )
                            },
                        )
                    },
                    || {
                        BatchedZipPlus::<EcZt, EcLc>::verify::<FScalar, UNCHECKED>(
                            &ec_params, &ec_comm, &ec_pt, &ec_proof,
                        )
                    },
                );
                let _ = black_box(r1);
                let _ = black_box(r2);
                let _ = black_box(r3);
            });
        });

        // ── Proof size comparison ───────────────────────────────────
        let sp_bytes: Vec<u8> = {
            let tx: zip_plus::pcs_transcript::PcsTranscript = sp_proof.into();
            tx.stream.into_inner()
        };
        let si_bytes: Vec<u8> = {
            let tx: zip_plus::pcs_transcript::PcsTranscript = si_proof.into();
            tx.stream.into_inner()
        };

        let split_sha_pcs = sp_bytes.len() + si_bytes.len();
        let split_total_raw = split_sha_pcs + ec_pcs + sha_piop + ec_piop;

        let mut split_all = Vec::new();
        split_all.extend(&sp_bytes);
        split_all.extend(&si_bytes);
        // ECDSA PCS bytes from the full-pipeline proof
        split_all.extend(&ec_zinc_proof.pcs_proof_bytes);
        // PIOP data for both proofs
        for p in [&sha_zinc_proof, &ec_zinc_proof] {
            for v in &p.ic_proof_values { split_all.extend(v); }
            for v in &p.cpr_sumcheck_messages { split_all.extend(v); }
            split_all.extend(&p.cpr_sumcheck_claimed_sum);
            for v in &p.cpr_up_evals { split_all.extend(v); }
            for v in &p.cpr_down_evals { split_all.extend(v); }
            for v in &p.evaluation_point_bytes { split_all.extend(v); }
            for v in &p.pcs_evals_bytes { split_all.extend(v); }
        }
        let mut enc = flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::default());
        enc.write_all(&split_all).unwrap();
        let split_compressed = enc.finish().unwrap();

        let sp_col_opening = SHA256_POLY_BATCH_SIZE * 1 * <ShaPolyZt as ZipTypes>::Cw::NUM_BYTES;
        let si_col_opening = SHA256_INT_BATCH_SIZE * 1 * <ShaIntZt as ZipTypes>::Cw::NUM_BYTES;

        eprintln!("\n=== 8xSHA256+ECDSA Split-SHA Proof Size ===");
        eprintln!("  SHA BinaryPoly ({} cols): PCS = {} bytes ({:.1} KB)",
            SHA256_POLY_BATCH_SIZE, sp_bytes.len(), sp_bytes.len() as f64 / 1024.0);
        eprintln!("    column values/opening: {} cols × {} B/cw = {} B",
            SHA256_POLY_BATCH_SIZE, <ShaPolyZt as ZipTypes>::Cw::NUM_BYTES, sp_col_opening);
        eprintln!("  SHA Int ({} cols):        PCS = {} bytes ({:.1} KB)",
            SHA256_INT_BATCH_SIZE, si_bytes.len(), si_bytes.len() as f64 / 1024.0);
        eprintln!("    column values/opening: {} cols × {} B/cw = {} B",
            SHA256_INT_BATCH_SIZE, <ShaIntZt as ZipTypes>::Cw::NUM_BYTES, si_col_opening);
        eprintln!("  ECDSA ({} cols):          PCS = {} bytes ({:.1} KB)",
            ECDSA_BATCH_SIZE, ec_pcs, ec_pcs as f64 / 1024.0);
        eprintln!("  Split SHA PCS: {} bytes ({:.1} KB) vs mono {} bytes ({:.1} KB) → {:.2}× smaller",
            split_sha_pcs, split_sha_pcs as f64 / 1024.0,
            sha_pcs, sha_pcs as f64 / 1024.0,
            sha_pcs as f64 / split_sha_pcs as f64);
        eprintln!("  Total raw:  {} bytes ({:.1} KB) vs mono {} bytes ({:.1} KB)",
            split_total_raw, split_total_raw as f64 / 1024.0,
            total_raw, total_raw as f64 / 1024.0);
        eprintln!("  Compressed: {} bytes ({:.1} KB, {:.1}× ratio)",
            split_compressed.len(), split_compressed.len() as f64 / 1024.0,
            split_all.len() as f64 / split_compressed.len() as f64);
    }

    group.finish();
}

/// PIOP-only benchmark (ideal check + combined poly resolver)
/// for the SHA-256 UAIR trace.
fn sha256_piop_only(c: &mut Criterion) {
    let mut group = c.benchmark_group("SHA-256 PIOP");
    group.sample_size(20);

    let trace = generate_sha256_trace(SHA256_NUM_VARS);
    let num_vars = SHA256_NUM_VARS;
    let num_constraints = zinc_uair::constraint_counter::count_constraints::<BinaryPoly<32>, Sha256Uair>();
    let max_degree = zinc_uair::degree_counter::count_max_degree::<BinaryPoly<32>, Sha256Uair>();

    // ── Ideal Check prover ──────────────────────────────────────────
    group.bench_function("IdealCheck/Prover", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let mut transcript = zinc_transcript::KeccakTranscript::new();
                let field_cfg = transcript.get_random_field_cfg::<
                    F, <F as Field>::Inner, MillerRabin
                >();
                let t = Instant::now();
                let _ = zinc_piop::ideal_check::IdealCheckProtocol::<
                    F
                >::prove_as_subprotocol::<BinaryPoly<32>, Sha256Uair>(
                    &mut transcript,
                    &trace,
                    num_constraints,
                    num_vars,
                    &field_cfg,
                )
                .expect("IC prover failed");
                total += t.elapsed();
            }
            total
        });
    });

    // ── Combined Poly Resolver prover ───────────────────────────────
    let mut transcript = zinc_transcript::KeccakTranscript::new();
    let field_cfg = transcript.get_random_field_cfg::<
        F, <F as Field>::Inner, MillerRabin
    >();
    let (_ic_proof, ic_state) =
        zinc_piop::ideal_check::IdealCheckProtocol::<
            F
        >::prove_as_subprotocol::<BinaryPoly<32>, Sha256Uair>(
            &mut transcript,
            &trace,
            num_constraints,
            num_vars,
            &field_cfg,
        )
        .expect("IC prover failed");

    group.bench_function("CombinedPolyResolver/Prover", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let mut tr = transcript.clone();
                let t = Instant::now();
                let _ = zinc_piop::combined_poly_resolver::CombinedPolyResolver::<
                    F
                >::prove_as_subprotocol::<BinaryPoly<32>, Sha256Uair>(
                    &mut tr,
                    &ic_state.trace_matrix,
                    &ic_state.evaluation_point,
                    ic_state.projected_scalars.clone(),
                    num_constraints,
                    num_vars,
                    max_degree,
                    &field_cfg,
                )
                .expect("CPR prover failed");
                total += t.elapsed();
            }
            total
        });
    });

    group.finish();
}

/// **True end-to-end benchmark** combining PIOP + PCS for honest total prover time.
///
/// Reports the total wall-clock time for the complete prover:
///   1. PIOP: IdealCheck + CombinedPolyResolver
///   2. PCS: Commit + Test + Evaluate
/// And the total verifier time:
///   1. PCS: Verify
///
/// This is the "honest" prover time — not just PCS in isolation.
fn sha256_end_to_end(c: &mut Criterion) {
    let mut group = c.benchmark_group("SHA-256 E2E (PIOP+PCS)");
    group.sample_size(20);

    let trace = generate_sha256_trace(SHA256_NUM_VARS);
    let num_vars = SHA256_NUM_VARS;
    let num_constraints = zinc_uair::constraint_counter::count_constraints::<BinaryPoly<32>, Sha256Uair>();
    let max_degree = zinc_uair::degree_counter::count_max_degree::<BinaryPoly<32>, Sha256Uair>();

    // ── Total Prover (PIOP + PCS) ────────────────────────────────────
    group.bench_function("TotalProver/1xSHA256", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let t = Instant::now();

                // 1. PIOP: IdealCheck
                let mut transcript = zinc_transcript::KeccakTranscript::new();
                let field_cfg = transcript.get_random_field_cfg::<
                    F, <F as Field>::Inner, MillerRabin
                >();
                let (_ic_proof, ic_state) =
                    zinc_piop::ideal_check::IdealCheckProtocol::<
                        F
                    >::prove_as_subprotocol::<BinaryPoly<32>, Sha256Uair>(
                        &mut transcript,
                        &trace,
                        num_constraints,
                        num_vars,
                        &field_cfg,
                    )
                    .expect("IC prover failed");

                // 2. PIOP: CombinedPolyResolver
                let _ = zinc_piop::combined_poly_resolver::CombinedPolyResolver::<
                    F
                >::prove_as_subprotocol::<BinaryPoly<32>, Sha256Uair>(
                    &mut transcript,
                    &ic_state.trace_matrix,
                    &ic_state.evaluation_point,
                    ic_state.projected_scalars.clone(),
                    num_constraints,
                    num_vars,
                    max_degree,
                    &field_cfg,
                )
                .expect("CPR prover failed");

                // 3. PCS: Commit + Test + Evaluate
                type Lc = IprsBPoly32R4B16<1, UNCHECKED>;
                type Zt = Sha256ZipTypes<i64, 32>;
                let row_len = 128;
                let linear_code = Lc::new(row_len);
                let params = ZipPlusParams::new(SHA256_NUM_VARS, 1, linear_code);

                let (hint, _commitment) = BatchedZipPlus::<Zt, Lc>::commit(&params, &trace)
                    .expect("commit");
                let test_tx = BatchedZipPlus::<Zt, Lc>::test::<UNCHECKED>(&params, &trace, &hint)
                    .expect("test");
                let point: Vec<<Zt as ZipTypes>::Pt> = vec![<Zt as ZipTypes>::Pt::one(); SHA256_NUM_VARS];
                let _ = BatchedZipPlus::<Zt, Lc>::evaluate::<F, UNCHECKED>(
                    &params, &trace, &point, test_tx,
                )
                .expect("evaluate");

                total += t.elapsed();
            }
            total
        });
    });

    // ── Total Verifier (PCS only — PIOP verifier is algebraic/negligible) ─────
    // Prepare a proof for the verifier benchmark
    {
        type Lc = IprsBPoly32R4B16<1, UNCHECKED>;
        type Zt = Sha256ZipTypes<i64, 32>;
        let linear_code = Lc::new(128);
        let params = ZipPlusParams::new(SHA256_NUM_VARS, 1, linear_code);
        let (hint, commitment) = BatchedZipPlus::<Zt, Lc>::commit(&params, &trace)
            .expect("commit");
        let test_tx = BatchedZipPlus::<Zt, Lc>::test::<UNCHECKED>(&params, &trace, &hint)
            .expect("test");
        let point: Vec<<Zt as ZipTypes>::Pt> = vec![<Zt as ZipTypes>::Pt::one(); SHA256_NUM_VARS];
        let (_evals_f, proof) = BatchedZipPlus::<Zt, Lc>::evaluate::<F, UNCHECKED>(
            &params, &trace, &point, test_tx,
        )
        .expect("evaluate");

        group.bench_function("TotalVerifier/1xSHA256", |b| {
            b.iter(|| {
                let result = BatchedZipPlus::<Zt, Lc>::verify::<F, UNCHECKED>(
                    &params,
                    &commitment,
                    &point,
                    &proof,
                );
                let _ = black_box(result);
            });
        });
    }

    group.finish();
}

/// **Full pipeline benchmark** using the `pipeline::prove` and `pipeline::verify` functions.
///
/// This measures the actual end-to-end pipeline including PIOP proof serialization/deserialization,
/// and reports total proof sizes (PCS + PIOP data).
fn sha256_full_pipeline(c: &mut Criterion) {
    use zinc_sha256_uair::CyclotomicIdeal;
    use zinc_uair::ideal_collector::IdealOrZero;
    type Lc = IprsBPoly32R4B16<1, UNCHECKED>;
    type Zt = Sha256ZipTypes<i64, 32>;

    let mut group = c.benchmark_group("SHA-256 Full Pipeline");
    group.sample_size(20);

    let trace = generate_sha256_trace(SHA256_NUM_VARS);
    let num_vars = SHA256_NUM_VARS;
    let linear_code = Lc::new(128);
    let params = ZipPlusParams::new(num_vars, 1, linear_code);

    // ── Full Prover ─────────────────────────────────────────────────
    group.bench_function("FullProver/1xSHA256", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let t = Instant::now();
                let _proof = zinc_snark::pipeline::prove::<Sha256Uair, Zt, Lc, 32, UNCHECKED>(
                    &params,
                    &trace,
                    num_vars,
                );
                total += t.elapsed();
            }
            total
        });
    });

    // Prepare proof for verifier and size measurement
    let zinc_proof = zinc_snark::pipeline::prove::<Sha256Uair, Zt, Lc, 32, UNCHECKED>(
        &params,
        &trace,
        num_vars,
    );

    // ── Full Verifier ───────────────────────────────────────────────
    group.bench_function("FullVerifier/1xSHA256", |b| {
        b.iter(|| {
            let result = zinc_snark::pipeline::verify::<Sha256Uair, Zt, Lc, 32, UNCHECKED, _, _>(
                &params,
                &zinc_proof,
                num_vars,
                |_: &IdealOrZero<CyclotomicIdeal>| zinc_snark::pipeline::TrivialIdeal,
            );
            black_box(result);
        });
    });

    // ── Proof size report ───────────────────────────────────────────
    let pcs_size = zinc_proof.pcs_proof_bytes.len();
    let piop_size: usize = zinc_proof.ic_proof_values.iter().map(|v| v.len()).sum::<usize>()
        + zinc_proof.cpr_sumcheck_messages.iter().map(|v| v.len()).sum::<usize>()
        + zinc_proof.cpr_sumcheck_claimed_sum.len()
        + zinc_proof.cpr_up_evals.iter().map(|v| v.len()).sum::<usize>()
        + zinc_proof.cpr_down_evals.iter().map(|v| v.len()).sum::<usize>()
        + zinc_proof.evaluation_point_bytes.iter().map(|v| v.len()).sum::<usize>()
        + zinc_proof.pcs_evals_bytes.iter().map(|v| v.len()).sum::<usize>();
    let total_size = pcs_size + piop_size;

    // Compressed
    let mut all_bytes = zinc_proof.pcs_proof_bytes.clone();
    for v in &zinc_proof.ic_proof_values { all_bytes.extend(v); }
    for v in &zinc_proof.cpr_sumcheck_messages { all_bytes.extend(v); }
    all_bytes.extend(&zinc_proof.cpr_sumcheck_claimed_sum);
    for v in &zinc_proof.cpr_up_evals { all_bytes.extend(v); }
    for v in &zinc_proof.cpr_down_evals { all_bytes.extend(v); }
    for v in &zinc_proof.evaluation_point_bytes { all_bytes.extend(v); }
    for v in &zinc_proof.pcs_evals_bytes { all_bytes.extend(v); }

    let mut encoder = flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::default());
    encoder.write_all(&all_bytes).unwrap();
    let compressed = encoder.finish().unwrap();

    eprintln!("\n=== Full Pipeline Proof Size ===");
    eprintln!("  PCS proof:     {} bytes ({:.1} KB)", pcs_size, pcs_size as f64 / 1024.0);
    eprintln!("  PIOP proof:    {} bytes ({:.1} KB)", piop_size, piop_size as f64 / 1024.0);
    eprintln!("  Total raw:     {} bytes ({:.1} KB)", total_size, total_size as f64 / 1024.0);
    eprintln!("  Compressed:    {} bytes ({:.1} KB)", compressed.len(), compressed.len() as f64 / 1024.0);
    eprintln!("  Target limit:  800 KB → {}", if total_size <= 819200 { "✓ UNDER" } else { "✗ OVER" });

    // ── Split PCS proof size comparison ─────────────────────────────
    // Run separate PCS for the BinaryPoly (11 cols) and Int (9 cols) batches
    // to measure the proof size savings from the column type split.
    // The PIOP proof size is unchanged — only PCS is split.
    {
        let mut rng = rand::rng();
        let poly_trace = generate_poly_witness(num_vars, &mut rng);
        let int_trace = generate_int_witness(num_vars, &mut rng);

        type IntZt = Sha256IntZipTypes;
        type IntLc = IprsInt1R4B16<1, UNCHECKED>;
        let int_lc = IntLc::new(128);
        let int_params = ZipPlusParams::<IntZt, IntLc>::new(num_vars, 1, int_lc);

        // BinaryPoly batch PCS
        let (poly_hint, _) = BatchedZipPlus::<Zt, Lc>::commit(&params, &poly_trace)
            .expect("poly commit");
        let poly_tx = BatchedZipPlus::<Zt, Lc>::test::<UNCHECKED>(&params, &poly_trace, &poly_hint)
            .expect("poly test");
        let poly_pt: Vec<i128> = vec![1i128; num_vars];
        let (_, poly_proof) = BatchedZipPlus::<Zt, Lc>::evaluate::<F, UNCHECKED>(
            &params, &poly_trace, &poly_pt, poly_tx,
        ).expect("poly evaluate");

        // Int batch PCS
        let (int_hint, _) = BatchedZipPlus::<IntZt, IntLc>::commit(&int_params, &int_trace)
            .expect("int commit");
        let int_tx = BatchedZipPlus::<IntZt, IntLc>::test::<UNCHECKED>(
            &int_params, &int_trace, &int_hint,
        ).expect("int test");
        let int_pt: Vec<i128> = vec![1i128; num_vars];
        let (_, int_proof) = BatchedZipPlus::<IntZt, IntLc>::evaluate::<F, UNCHECKED>(
            &int_params, &int_trace, &int_pt, int_tx,
        ).expect("int evaluate");

        let poly_pcs_bytes: Vec<u8> = {
            let tx: zip_plus::pcs_transcript::PcsTranscript = poly_proof.into();
            tx.stream.into_inner()
        };
        let int_pcs_bytes: Vec<u8> = {
            let tx: zip_plus::pcs_transcript::PcsTranscript = int_proof.into();
            tx.stream.into_inner()
        };
        let split_pcs_total = poly_pcs_bytes.len() + int_pcs_bytes.len();
        let split_total = split_pcs_total + piop_size;

        let mut split_bytes = Vec::new();
        split_bytes.extend(&poly_pcs_bytes);
        split_bytes.extend(&int_pcs_bytes);
        // Include PIOP data for compression
        for v in &zinc_proof.ic_proof_values { split_bytes.extend(v); }
        for v in &zinc_proof.cpr_sumcheck_messages { split_bytes.extend(v); }
        split_bytes.extend(&zinc_proof.cpr_sumcheck_claimed_sum);
        for v in &zinc_proof.cpr_up_evals { split_bytes.extend(v); }
        for v in &zinc_proof.cpr_down_evals { split_bytes.extend(v); }
        for v in &zinc_proof.evaluation_point_bytes { split_bytes.extend(v); }
        for v in &zinc_proof.pcs_evals_bytes { split_bytes.extend(v); }

        let mut enc2 = flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::default());
        enc2.write_all(&split_bytes).unwrap();
        let split_compressed = enc2.finish().unwrap();

        eprintln!("\n=== Split PCS Full Pipeline Proof Size ===");
        eprintln!("  BinaryPoly PCS ({} cols): {} bytes ({:.1} KB)",
            SHA256_POLY_BATCH_SIZE, poly_pcs_bytes.len(), poly_pcs_bytes.len() as f64 / 1024.0);
        eprintln!("  Int PCS ({} cols):        {} bytes ({:.1} KB)",
            SHA256_INT_BATCH_SIZE, int_pcs_bytes.len(), int_pcs_bytes.len() as f64 / 1024.0);
        eprintln!("  Split PCS total: {} bytes ({:.1} KB) vs mono {} bytes ({:.1} KB) → {:.2}× smaller",
            split_pcs_total, split_pcs_total as f64 / 1024.0,
            pcs_size, pcs_size as f64 / 1024.0,
            pcs_size as f64 / split_pcs_total as f64);
        eprintln!("  Split total (PCS+PIOP): {} bytes ({:.1} KB) vs mono {} bytes ({:.1} KB)",
            split_total, split_total as f64 / 1024.0,
            total_size, total_size as f64 / 1024.0);
        eprintln!("  Split compressed: {} bytes ({:.1} KB, {:.1}× ratio)",
            split_compressed.len(), split_compressed.len() as f64 / 1024.0,
            split_bytes.len() as f64 / split_compressed.len() as f64);
    }

    group.finish();
}

criterion_group!(benches, sha256_single, sha256_8x_ecdsa, sha256_piop_only, sha256_end_to_end, sha256_full_pipeline);
criterion_main!(benches);
