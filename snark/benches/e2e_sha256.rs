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

use zinc_utils::peak_mem::MemoryTracker;

use criterion::{
    criterion_group, criterion_main, Criterion,
};
use crypto_bigint::U64;
use crypto_primitives::{
    boolean::Boolean,
    crypto_bigint_int::Int,
    crypto_bigint_monty::MontyField,
    crypto_bigint_uint::Uint,
    Field, FixedSemiring, IntoWithConfig,
};
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
};
use zip_plus::{
    code::{
        LinearCode,
        iprs::{IprsCode, PnttConfigF2_16R4B16, PnttConfigF2_16R4B64},
    },
    pcs::structs::{ZipPlus, ZipPlusParams, ZipTypes},
};

use zinc_ecdsa_uair::EcdsaUairInt;
use zinc_sha256_uair::{Sha256Uair, witness::GenerateWitness};
use zinc_sha256_uair::witness::{generate_poly_witness, generate_int_witness};
use zinc_uair::Uair;
use zinc_piop::projections::{
    project_trace_coeffs, project_trace_to_field,
    project_scalars, project_scalars_to_field,
};
use zinc_piop::lookup::{
    LookupColumnSpec, LookupTableType,
};
use zinc_poly::univariate::dynamic::over_field::DynamicPolynomialF;
use crypto_primitives::PrimeField;

// ─── Type definitions (matching batched_zip_plus_benches.rs) ────────────────

const INT_LIMBS: usize = U64::LIMBS;
type F = MontyField<{ INT_LIMBS * 3 }>;
/// 192-bit field (same as F) for ECDSA PCS with Int<4> evaluations.
type FScalar = MontyField<{ INT_LIMBS * 3 }>;

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
    type Fmod = Uint<{ INT_LIMBS * 3 }>;
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
    type Fmod = Uint<{ INT_LIMBS * 3 }>;     // 192-bit modulus search
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
    type Fmod = Uint<{ INT_LIMBS * 3 }>;      // 192-bit modulus search
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
// IPRS code types for Int<4> (scalar ECDSA evaluations)
type IprsInt4R4B64<const DEPTH: usize, const CHECK: bool> = IprsCode<
    EcdsaScalarZipTypes,
    PnttConfigF2_16R4B64<DEPTH>,
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
// ─── SHA-256 Trace Parameters ───────────────────────────────────────────────
//
// SHA-256 has 64 rows × 26 columns (23 bitpoly + 3 int, 8 public).
// ECDSA has 258 rows × 9 columns (2 public).
//
// IPRS codes need DEPTH ≥ 1 (radix-8 NTT), and row_len = BASE_LEN × 8^DEPTH.
//
// Benchmark configurations:
//   - Single SHA-256: poly_size = 2^7 = 128, 26 polys, R4B16 DEPTH=1 (row_len=128)
//     → 64 real rows + 64 padding = 128 total
//   - 8× SHA-256 + ECDSA: poly_size = 2^10 = 1024, 35 polys
//     → 8×64=512 SHA rows + 258 ECDSA rows = 770 → pad to 1024

const SHA256_NUM_VARS: usize = 7;        // 2^7 = 128 rows (64 real + 64 padding)
const SHA256_BATCH_SIZE: usize = 26;      // 26 SHA-256 columns (23 bitpoly + 3 int)

// Split SHA-256 batch sizes for two-PCS configuration
const SHA256_POLY_BATCH_SIZE: usize = 23; // 23 BinaryPoly columns
const SHA256_INT_BATCH_SIZE: usize = 3;   // 3 Int columns (μ_a, μ_e, μ_W)

// For 8× SHA-256 + ECDSA: two separate PCS batches
const SHA256_8X_NUM_VARS: usize = 9;           // 2^9 = 512 rows (8 × 64 SHA rounds)
const ECDSA_NUM_VARS: usize = 9;               // 2^9 = 512 rows (258 ECDSA rows + padding)
const ECDSA_BATCH_SIZE: usize = 11;             // 11 ECDSA columns (9 data + 2 selector)

// ─── Lookup column specs ────────────────────────────────────────────────────

/// Number of Q[X] bit-polynomial columns that need lookup enforcement.
///
/// Only the 10 Q[X] columns (indices 0–9) require `BitPoly { width: 32 }`
/// lookups.  The 4 F₂[X] columns (S₀, S₁, R₀, R₁, indices 10–13) are
/// shift quotient/remainder values whose membership is enforced by the
/// UAIR constraints — no lookup needed.  The 3 integer carry columns
/// (μ_a, μ_e, μ_W, indices 14–16) are likewise constrained by UAIR.
const SHA256_LOOKUP_COL_COUNT: usize = 10;

fn sha256_lookup_specs() -> Vec<LookupColumnSpec> {
    (0..SHA256_LOOKUP_COL_COUNT)
        .map(|i| LookupColumnSpec {
            column_index: i,
            table_type: LookupTableType::BitPoly { width: 32 },
        })
        .collect()
}

// ECDSA columns are field elements (opened mod q) — no lookup needed.

// ─── Lookup proof size helpers ───────────────────────────────────────────────

/// Count the field elements in a `GkrFractionProof`.
fn gkr_fraction_proof_fe_count(proof: &zinc_piop::lookup::GkrFractionProof<zinc_snark::pipeline::PiopField>) -> usize {
    let mut count = 2usize; // root_p, root_q
    for lp in &proof.layer_proofs {
        // 4 child MLE evaluations per layer
        count += 4;
        // sumcheck messages (if present)
        if let Some(ref sc) = lp.sumcheck_proof {
            count += 1; // claimed_sum
            for msg in &sc.messages {
                count += msg.0.tail_evaluations.len();
            }
        }
    }
    count
}

/// Compute the serialised byte count of a `GkrPipelineLookupProof`.
fn gkr_lookup_proof_byte_count(proof: &zinc_piop::lookup::GkrPipelineLookupProof<zinc_snark::pipeline::PiopField>) -> usize {
    use zinc_snark::pipeline::FIELD_LIMBS;
    let fe_bytes = <crypto_primitives::crypto_bigint_uint::Uint<FIELD_LIMBS> as ConstTranscribable>::NUM_BYTES;
    let mut count = 0usize;
    for gp in &proof.group_proofs {
        for v in &gp.aggregated_multiplicities {
            count += v.len();
        }
        count += gkr_fraction_proof_fe_count(&gp.witness_gkr);
        count += gkr_fraction_proof_fe_count(&gp.table_gkr);
    }
    count * fe_bytes
}

/// Compute the serialised byte count of a `PipelineLookupProof` (classic).
fn classic_lookup_proof_byte_count(proof: &zinc_piop::lookup::PipelineLookupProof<zinc_snark::pipeline::PiopField>) -> usize {
    use zinc_snark::pipeline::FIELD_LIMBS;
    let fe_bytes = <crypto_primitives::crypto_bigint_uint::Uint<FIELD_LIMBS> as ConstTranscribable>::NUM_BYTES;
    let mut count = 0usize;
    for gp in &proof.group_proofs {
        count += 1; // claimed_sum
        for msg in &gp.sumcheck_proof.messages {
            count += msg.0.tail_evaluations.len();
        }
        for v in &gp.aggregated_multiplicities {
            count += v.len();
        }
        for outer in &gp.chunk_inverse_witnesses {
            for inner in outer { count += inner.len(); }
        }
        count += gp.inverse_table.len();
    }
    count * fe_bytes
}

/// Dispatch byte count on the lookup proof variant.
fn lookup_proof_byte_count(data: &zinc_snark::pipeline::LookupProofData) -> usize {
    match data {
        zinc_snark::pipeline::LookupProofData::Gkr(p) => gkr_lookup_proof_byte_count(p),
        zinc_snark::pipeline::LookupProofData::Classic(p) => classic_lookup_proof_byte_count(p),
        zinc_snark::pipeline::LookupProofData::BatchedClassic(p) => {
            batched_classic_lookup_proof_to_bytes(p).len()
        }
    }
}

/// Serialize all field elements in a `GkrPipelineLookupProof` to a byte vector.
fn gkr_lookup_proof_to_bytes(proof: &zinc_piop::lookup::GkrPipelineLookupProof<zinc_snark::pipeline::PiopField>) -> Vec<u8> {
    fn write_fe(buf: &mut Vec<u8>, f: &zinc_snark::pipeline::PiopField) {
        use zinc_snark::pipeline::FIELD_LIMBS;
        let fe_size = <crypto_primitives::crypto_bigint_uint::Uint<FIELD_LIMBS> as ConstTranscribable>::NUM_BYTES;
        let start = buf.len();
        buf.resize(start + fe_size, 0);
        f.inner().write_transcription_bytes(&mut buf[start..]);
    }

    fn write_gkr_fraction_proof(buf: &mut Vec<u8>, proof: &zinc_piop::lookup::GkrFractionProof<zinc_snark::pipeline::PiopField>) {
        write_fe(buf, &proof.root_p);
        write_fe(buf, &proof.root_q);
        for lp in &proof.layer_proofs {
            if let Some(ref sc) = lp.sumcheck_proof {
                write_fe(buf, &sc.claimed_sum);
                for msg in &sc.messages {
                    for f in &msg.0.tail_evaluations { write_fe(buf, f); }
                }
            }
            write_fe(buf, &lp.p_left);
            write_fe(buf, &lp.p_right);
            write_fe(buf, &lp.q_left);
            write_fe(buf, &lp.q_right);
        }
    }

    let mut out = Vec::new();
    for gp in &proof.group_proofs {
        for v in &gp.aggregated_multiplicities {
            for f in v { write_fe(&mut out, f); }
        }
        write_gkr_fraction_proof(&mut out, &gp.witness_gkr);
        write_gkr_fraction_proof(&mut out, &gp.table_gkr);
    }
    out
}

/// Serialize all field elements in a classic `PipelineLookupProof` to a byte vector.
fn classic_lookup_proof_to_bytes(proof: &zinc_piop::lookup::PipelineLookupProof<zinc_snark::pipeline::PiopField>) -> Vec<u8> {
    fn write_fe(buf: &mut Vec<u8>, f: &zinc_snark::pipeline::PiopField) {
        use zinc_snark::pipeline::FIELD_LIMBS;
        let fe_size = <crypto_primitives::crypto_bigint_uint::Uint<FIELD_LIMBS> as ConstTranscribable>::NUM_BYTES;
        let start = buf.len();
        buf.resize(start + fe_size, 0);
        f.inner().write_transcription_bytes(&mut buf[start..]);
    }

    let mut out = Vec::new();
    for gp in &proof.group_proofs {
        write_fe(&mut out, &gp.sumcheck_proof.claimed_sum);
        for msg in &gp.sumcheck_proof.messages {
            for f in &msg.0.tail_evaluations { write_fe(&mut out, f); }
        }
        for v in &gp.aggregated_multiplicities {
            for f in v { write_fe(&mut out, f); }
        }
        for outer in &gp.chunk_inverse_witnesses {
            for inner in outer {
                for f in inner { write_fe(&mut out, f); }
            }
        }
        for f in &gp.inverse_table { write_fe(&mut out, f); }
    }
    out
}

/// Serialize a `BatchedCprLookupProof` to bytes.
fn batched_classic_lookup_proof_to_bytes(proof: &zinc_snark::pipeline::BatchedCprLookupProof) -> Vec<u8> {
    fn write_fe(buf: &mut Vec<u8>, f: &zinc_snark::pipeline::PiopField) {
        use zinc_snark::pipeline::FIELD_LIMBS;
        let fe_size = <crypto_primitives::crypto_bigint_uint::Uint<FIELD_LIMBS> as ConstTranscribable>::NUM_BYTES;
        let start = buf.len();
        buf.resize(start + fe_size, 0);
        f.inner().write_transcription_bytes(&mut buf[start..]);
    }

    let mut out = Vec::new();
    // Multi-degree sumcheck proof.
    for group_msgs in &proof.md_proof.group_messages {
        for msg in group_msgs {
            for f in &msg.0.tail_evaluations { write_fe(&mut out, f); }
        }
    }
    for cs in &proof.md_proof.claimed_sums { write_fe(&mut out, cs); }
    // CPR up/down evals.
    for f in &proof.cpr_up_evals { write_fe(&mut out, f); }
    for f in &proof.cpr_down_evals { write_fe(&mut out, f); }
    // Per-group lookup data.
    for gp in &proof.lookup_group_proofs {
        for v in &gp.aggregated_multiplicities {
            for f in v { write_fe(&mut out, f); }
        }
        for outer in &gp.chunk_inverse_witnesses {
            for inner in outer {
                for f in inner { write_fe(&mut out, f); }
            }
        }
        for f in &gp.inverse_table { write_fe(&mut out, f); }
    }
    out
}

/// Dispatch proof serialization on the lookup proof variant.
fn lookup_proof_to_bytes(data: &zinc_snark::pipeline::LookupProofData) -> Vec<u8> {
    match data {
        zinc_snark::pipeline::LookupProofData::Gkr(p) => gkr_lookup_proof_to_bytes(p),
        zinc_snark::pipeline::LookupProofData::Classic(p) => classic_lookup_proof_to_bytes(p),
        zinc_snark::pipeline::LookupProofData::BatchedClassic(p) => batched_classic_lookup_proof_to_bytes(p),
    }
}

// ─── Benchmark helpers ──────────────────────────────────────────────────────

fn generate_sha256_trace(num_vars: usize) -> Vec<DenseMultilinearExtension<BinaryPoly<32>>> {
    let mut rng = rand::rng();
    <Sha256Uair as GenerateWitness<BinaryPoly<32>>>::generate_witness(num_vars, &mut rng)
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

/// Generate random ECDSA-shaped trace with values in `[0, 2^32)`.
///
/// Generate an all-zero ECDSA trace that trivially satisfies both the
/// ECDSA UAIR constraints *and* `Word { width: 32 }` lookup constraints.
///
/// All seven Jacobian doubling-and-add constraints evaluate to zero when
/// every column is identically zero (same approach as `ecdsa_pipeline` test).
/// Zero is also a valid 32-bit word, so lookup decomposition succeeds.
fn generate_zero_scalar_trace(
    num_vars: usize,
    num_cols: usize,
) -> Vec<DenseMultilinearExtension<Int<{ INT_LIMBS * 4 }>>> {
    let zero = Int::<{ INT_LIMBS * 4 }>::default();
    let rows = 1usize << num_vars;
    (0..num_cols)
        .map(|_| DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            vec![zero; rows],
            zero,
        ))
        .collect()
}

// ─── Criterion suites ───────────────────────────────────────────────────────

/// Single SHA-256 compression (64 rows padded to 128 × 26 columns = poly_size 2^7).
///
/// Uses R4B16 IPRS codes at rate 1/4 with DEPTH=1 (row_len=128).
///
/// **Proof size claims are valid for DEPTH=1 configurations only.**
/// DEPTH≥2 dramatically reduces deflate compressibility (see §4 of review).
fn sha256_single(c: &mut Criterion) {
    let mut group = c.benchmark_group("SHA-256 Single Compression");
    group.sample_size(20);

    // ── Split PCS: BinaryPoly batch + Int batch ─────────────────────
    // Split the 26 SHA-256 columns into two PCS batches:
    //   - BinaryPoly batch (23 cols): columns used in F₂[X] rotation constraints
    //   - Int batch (3 cols): carry columns (μ_a, μ_e, μ_W)
    // The PIOP still runs on the full 26-column trace.
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

        // Derive the real PCS evaluation point from a full pipeline proof.
        // Generate a full 26-column SHA-256 trace, run pipeline, extract the
        // CPR evaluation point used by the PIOP.  Both split batches share
        // this point since they originate from the same PIOP.
        let split_pcs_pt: Vec<F> = {
            let full_trace = generate_sha256_trace(SHA256_NUM_VARS);
            let full_params = ZipPlusParams::<PolyZt, PolyLc>::new(SHA256_NUM_VARS, 1, PolyLc::new(128));
            let full_proof = zinc_snark::pipeline::prove::< Sha256Uair, PolyZt, PolyLc, 32, UNCHECKED>(
                &full_params, &full_trace, SHA256_NUM_VARS, &sha256_lookup_specs(),
            );
            zinc_snark::pipeline::pcs_point_from_proof(&full_proof)
        };

        // ── Prover (both batches) ───────────────────────────────────
        group.bench_function("1xSHA256/SplitPCS/Prover", |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let t = Instant::now();

                    // BinaryPoly batch
                    let (poly_hint, _) = ZipPlus::<PolyZt, PolyLc>::commit(
                        &poly_params, &poly_trace,
                    ).expect("poly commit");
                    let _ = ZipPlus::<PolyZt, PolyLc>::prove::<F, UNCHECKED>(
                        &poly_params, &poly_trace, &split_pcs_pt, &poly_hint,
                    ).expect("poly prove");

                    // Int batch
                    let (int_hint, _) = ZipPlus::<IntZt, IntLc>::commit(
                        &int_params, &int_trace,
                    ).expect("int commit");
                    let _ = ZipPlus::<IntZt, IntLc>::prove::<F, UNCHECKED>(
                        &int_params, &int_trace, &split_pcs_pt, &int_hint,
                    ).expect("int prove");

                    total += t.elapsed();
                }
                total
            });
        });

        // Prepare proofs for verifier and size measurement
        let (poly_hint, poly_comm) = ZipPlus::<PolyZt, PolyLc>::commit(
            &poly_params, &poly_trace,
        ).expect("commit");
        let poly_pt = split_pcs_pt.clone();
        let (poly_eval_f, poly_proof) = ZipPlus::<PolyZt, PolyLc>::prove::<F, UNCHECKED>(
            &poly_params, &poly_trace, &poly_pt, &poly_hint,
        ).expect("prove");

        let (int_hint, int_comm) = ZipPlus::<IntZt, IntLc>::commit(
            &int_params, &int_trace,
        ).expect("commit");
        let int_pt = split_pcs_pt.clone();
        let (int_eval_f, int_proof) = ZipPlus::<IntZt, IntLc>::prove::<F, UNCHECKED>(
            &int_params, &int_trace, &int_pt, &int_hint,
        ).expect("prove");

        // Points are already Vec<F>, no conversion needed
        let poly_pt_f: Vec<F> = poly_pt.clone();
        let int_pt_f: Vec<F> = int_pt.clone();

        // ── Verifier (both batches) ─────────────────────────────────
        group.bench_function("1xSHA256/SplitPCS/Verifier", |b| {
            b.iter(|| {
                let r1 = ZipPlus::<PolyZt, PolyLc>::verify::<F, UNCHECKED>(
                    &poly_params, &poly_comm, &poly_pt_f, &poly_eval_f, &poly_proof,
                );
                let _ = black_box(r1);
                let r2 = ZipPlus::<IntZt, IntLc>::verify::<F, UNCHECKED>(
                    &int_params, &int_comm, &int_pt_f, &int_eval_f, &int_proof,
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

    group.finish();
}

/// 8× SHA-256 + ECDSA — the paper's target benchmark configuration.
///
/// Uses **two separate PCS batches** with evaluation types matched to the
/// arithmetic of each UAIR:
/// - SHA-256:  23 columns × BinaryPoly<32> + 3 columns × Int<1>
/// - ECDSA:    9 columns × Int<4>  (scalar evaluations, 1 × 256-bit integer)
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
    let mem_tracker = MemoryTracker::start();

    let mut group = c.benchmark_group("8xSHA256+ECDSA");
    group.sample_size(10);

    // ── Build traces ────────────────────────────────────────────────
    // SHA-256: 26 columns × 512 rows (8 instances × 64 rounds)
    let sha_trace = generate_sha256_trace(SHA256_8X_NUM_VARS);
    assert_eq!(sha_trace.len(), SHA256_BATCH_SIZE);

    // ECDSA: 9 columns × 512 rows (258 real + 254 padding)
    // Values are 32-bit words (matching Word{width:32} lookup constraints).
    let ecdsa_trace = generate_zero_scalar_trace(ECDSA_NUM_VARS, ECDSA_BATCH_SIZE);
    assert_eq!(ecdsa_trace.len(), ECDSA_BATCH_SIZE);

    // Public columns for ECDSA verifier (b_1 = col 0, b_2 = col 1).
    let ec_public_cols = vec![ecdsa_trace[0].clone(), ecdsa_trace[1].clone()];
    let sha_sig_pub = Sha256Uair::signature();
    let sha_public_cols: Vec<_> = sha_sig_pub.public_columns.iter()
        .map(|&i| sha_trace[i].clone()).collect();

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

    // ── Lookup column specs ──────────────────────────────────────────
    let sha_lookup_specs = sha256_lookup_specs();

    // ── Proof sizes (full pipeline: PCS + PIOP + Lookup) ────────────
    // Generate full-pipeline proofs for accurate size reporting and to
    // derive real PCS evaluation points for standalone PCS benchmarks.
    let sha_zinc_proof = zinc_snark::pipeline::prove::<Sha256Uair, ShaZt, ShaLc, 32, UNCHECKED>(
        &sha_params, &sha_trace, SHA256_8X_NUM_VARS, &sha_lookup_specs,
    );
    let ec_zinc_proof = zinc_snark::pipeline::prove_generic::< EcdsaUairInt, Int<{ INT_LIMBS * 4 }>, EcZt, EcLc, FScalar, UNCHECKED,
    >(
        &ec_params, &ecdsa_trace, ECDSA_NUM_VARS, &[],
    );

    // Derive real PCS evaluation points from the PIOP-generated CPR points.
    let sha_pcs_pt: Vec<F> = zinc_snark::pipeline::pcs_point_from_proof(&sha_zinc_proof);
    let ec_pcs_pt: Vec<F> = zinc_snark::pipeline::pcs_point_from_proof(&ec_zinc_proof);

    // ── Prepare ECDSA proof (reused by split PCS verifier) ────────
    let (ec_hint, ec_comm) = ZipPlus::<EcZt, EcLc>::commit(
        &ec_params, &ecdsa_trace,
    ).expect("commit");
    let ec_pt = ec_pcs_pt.clone();
    let (ec_eval_f, ec_proof) = ZipPlus::<EcZt, EcLc>::prove::<FScalar, UNCHECKED>(
        &ec_params, &ecdsa_trace, &ec_pt, &ec_hint,
    ).expect("prove");

    // ec_pt is already Vec<F> = Vec<FScalar>, no conversion needed
    let ec_pt_f: Vec<FScalar> = ec_pt.clone();

    // ── Full Pipeline Prover (IC + CPR + Lookup + PCS) ──────────────
    group.bench_function("FullPipeline/Prover", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let t = Instant::now();
                let _sha_proof = zinc_snark::pipeline::prove::<Sha256Uair, ShaZt, ShaLc, 32, UNCHECKED>(
                    &sha_params, &sha_trace, SHA256_8X_NUM_VARS, &sha_lookup_specs,
                );
                let _ec_proof = zinc_snark::pipeline::prove_generic::<
                    EcdsaUairInt, Int<{ INT_LIMBS * 4 }>, EcZt, EcLc, FScalar, UNCHECKED,
                >(
                    &ec_params, &ecdsa_trace, ECDSA_NUM_VARS, &[],
                );
                total += t.elapsed();
            }
            total
        });
    });

    // ── Full Pipeline Verifier ───────────────────────────────────────
    {
        use zinc_sha256_uair::CyclotomicIdeal;
        use zinc_ecdsa_uair::EcdsaIdealOverF;
        use zinc_uair::ideal::ImpossibleIdeal;
        use zinc_uair::ideal_collector::IdealOrZero;

        group.bench_function("FullPipeline/Verifier", |b| {
            b.iter(|| {
                let sha_r = zinc_snark::pipeline::verify::<Sha256Uair, ShaZt, ShaLc, 32, UNCHECKED, _, _>(
                    &sha_params, &sha_zinc_proof, SHA256_8X_NUM_VARS,
                    |_: &IdealOrZero<CyclotomicIdeal>| zinc_snark::pipeline::TrivialIdeal,
                    &sha_public_cols,
                );
                let _ = black_box(sha_r);
                let ec_r = zinc_snark::pipeline::verify_generic::<
                    EcdsaUairInt, Int<{ INT_LIMBS * 4 }>, EcZt, EcLc, FScalar, UNCHECKED, EcdsaIdealOverF, _,
                >(
                    &ec_params, &ec_zinc_proof, ECDSA_NUM_VARS,
                    |ideal: &IdealOrZero<ImpossibleIdeal>| match ideal {
                        IdealOrZero::Zero => EcdsaIdealOverF,
                        IdealOrZero::Ideal(_) => panic!("ECDSA has no non-zero ideal constraints"),
                    },
                    &ec_public_cols,
                );
                let _ = black_box(ec_r);
            });
        });
    }

    // ── Prover timing breakdown ─────────────────────────────────────
    eprintln!("\n=== 8xSHA256+ECDSA Full Pipeline Timing ===");
    eprintln!("  SHA prover: IC={:?}, CPR={:?}, Lookup={:?}, PCS(commit={:?}, prove={:?}), total={:?}",
        sha_zinc_proof.timing.ideal_check,
        sha_zinc_proof.timing.combined_poly_resolver,
        sha_zinc_proof.timing.lookup,
        sha_zinc_proof.timing.pcs_commit,
        sha_zinc_proof.timing.pcs_prove,
        sha_zinc_proof.timing.total,
    );
    eprintln!("  ECDSA prover: IC={:?}, CPR={:?}, Lookup={:?}, PCS(commit={:?}, prove={:?}), total={:?}",
        ec_zinc_proof.timing.ideal_check,
        ec_zinc_proof.timing.combined_poly_resolver,
        ec_zinc_proof.timing.lookup,
        ec_zinc_proof.timing.pcs_commit,
        ec_zinc_proof.timing.pcs_prove,
        ec_zinc_proof.timing.total,
    );

    // Helper to compute PIOP + Lookup proof size from a ZincProof.
    fn piop_size(p: &zinc_snark::pipeline::ZincProof) -> usize {
        let base = p.ic_proof_values.iter().map(|v| v.len()).sum::<usize>()
            + p.cpr_sumcheck_messages.iter().map(|v| v.len()).sum::<usize>()
            + p.cpr_sumcheck_claimed_sum.len()
            + p.cpr_up_evals.iter().map(|v| v.len()).sum::<usize>()
            + p.cpr_down_evals.iter().map(|v| v.len()).sum::<usize>()
            + p.evaluation_point_bytes.iter().map(|v| v.len()).sum::<usize>()
            + p.pcs_evals_bytes.iter().map(|v| v.len()).sum::<usize>();
        let lookup = p.lookup_proof.as_ref().map_or(0, lookup_proof_byte_count);
        base + lookup
    }

    let sha_pcs = sha_zinc_proof.pcs_proof_bytes.len();
    let sha_piop = piop_size(&sha_zinc_proof);
    let sha_lookup = sha_zinc_proof.lookup_proof.as_ref().map_or(0, lookup_proof_byte_count);
    let ec_pcs = ec_zinc_proof.pcs_proof_bytes.len();
    let ec_piop = piop_size(&ec_zinc_proof);
    let ec_lookup = ec_zinc_proof.lookup_proof.as_ref().map_or(0, lookup_proof_byte_count);
    let total_raw = sha_pcs + sha_piop + ec_pcs + ec_piop;

    eprintln!("\n=== 8xSHA256+ECDSA Full Pipeline Proof Size ===");
    eprintln!("  SHA: PCS={} B ({:.1} KB), PIOP+Lookup={} B ({:.1} KB) [lookup={} B]",
        sha_pcs, sha_pcs as f64 / 1024.0, sha_piop, sha_piop as f64 / 1024.0, sha_lookup);
    eprintln!("  ECDSA: PCS={} B ({:.1} KB), PIOP+Lookup={} B ({:.1} KB) [lookup={} B]",
        ec_pcs, ec_pcs as f64 / 1024.0, ec_piop, ec_piop as f64 / 1024.0, ec_lookup);
    eprintln!("  Total raw: {} bytes ({:.1} KB)", total_raw, total_raw as f64 / 1024.0);

    // ── Split SHA PCS: BinaryPoly (23 cols) + Int (3 cols) + ECDSA ──
    // Splits the 26 SHA-256 columns into two PCS batches by column type,
    // keeping the ECDSA batch unchanged.  Three PCS batches total:
    //   1. SHA BinaryPoly (23 cols) — rotation/shift constraints
    //   2. SHA Int<1>     (3 cols)  — carry columns (μ_a, μ_e, μ_W)
    //   3. ECDSA Int<4>   (9 cols)  — scalar arithmetic
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
                    let (h, _) = ZipPlus::<ShaPolyZt, ShaPolyLc>::commit(
                        &sha_poly_params, &sha_poly_trace,
                    ).expect("sha poly commit");
                    let _ = ZipPlus::<ShaPolyZt, ShaPolyLc>::prove::<F, UNCHECKED>(
                        &sha_poly_params, &sha_poly_trace, &sha_pcs_pt, &h,
                    ).expect("sha poly prove");

                    // SHA Int batch
                    let (h, _) = ZipPlus::<ShaIntZt, ShaIntLc>::commit(
                        &sha_int_params, &sha_int_trace,
                    ).expect("sha int commit");
                    let _ = ZipPlus::<ShaIntZt, ShaIntLc>::prove::<F, UNCHECKED>(
                        &sha_int_params, &sha_int_trace, &sha_pcs_pt, &h,
                    ).expect("sha int prove");

                    // ECDSA batch
                    let (h, _) = ZipPlus::<EcZt, EcLc>::commit(
                        &ec_params, &ecdsa_trace,
                    ).expect("ecdsa commit");
                    let _ = ZipPlus::<EcZt, EcLc>::prove::<FScalar, UNCHECKED>(
                        &ec_params, &ecdsa_trace, &ec_pcs_pt, &h,
                    ).expect("ecdsa prove");

                    total += t.elapsed();
                }
                total
            });
        });

        // Prepare proofs for verifier and size measurement
        let (sp_h, sp_comm) = ZipPlus::<ShaPolyZt, ShaPolyLc>::commit(
            &sha_poly_params, &sha_poly_trace,
        ).expect("commit");
        let sp_pt = sha_pcs_pt.clone();
        let (sp_eval_f, sp_proof) = ZipPlus::<ShaPolyZt, ShaPolyLc>::prove::<F, UNCHECKED>(
            &sha_poly_params, &sha_poly_trace, &sp_pt, &sp_h,
        ).expect("prove");

        let (si_h, si_comm) = ZipPlus::<ShaIntZt, ShaIntLc>::commit(
            &sha_int_params, &sha_int_trace,
        ).expect("commit");
        let si_pt = sha_pcs_pt.clone();
        let (si_eval_f, si_proof) = ZipPlus::<ShaIntZt, ShaIntLc>::prove::<F, UNCHECKED>(
            &sha_int_params, &sha_int_trace, &si_pt, &si_h,
        ).expect("prove");

        // Reuse ECDSA proof from above (ec_comm, ec_pt, ec_proof)

        // ── Split Verifier (3 batches) ──────────────────────────────
        // Points are already Vec<F>, no conversion needed
        let sp_pt_f: Vec<F> = sp_pt.clone();
        let si_pt_f: Vec<F> = si_pt.clone();

        group.bench_function("SplitSHA/PCS/Verifier", |b| {
            b.iter(|| {
                // Run all 3 verify calls concurrently using rayon::join
                let ((r1, r2), r3) = rayon::join(
                    || {
                        rayon::join(
                            || {
                                ZipPlus::<ShaPolyZt, ShaPolyLc>::verify::<F, UNCHECKED>(
                                    &sha_poly_params, &sp_comm, &sp_pt_f, &sp_eval_f, &sp_proof,
                                )
                            },
                            || {
                                ZipPlus::<ShaIntZt, ShaIntLc>::verify::<F, UNCHECKED>(
                                    &sha_int_params, &si_comm, &si_pt_f, &si_eval_f, &si_proof,
                                )
                            },
                        )
                    },
                    || {
                        ZipPlus::<EcZt, EcLc>::verify::<FScalar, UNCHECKED>(
                            &ec_params, &ec_comm, &ec_pt_f, &ec_eval_f, &ec_proof,
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
        // PIOP + Lookup data for both proofs
        for p in [&sha_zinc_proof, &ec_zinc_proof] {
            for v in &p.ic_proof_values { split_all.extend(v); }
            for v in &p.cpr_sumcheck_messages { split_all.extend(v); }
            split_all.extend(&p.cpr_sumcheck_claimed_sum);
            for v in &p.cpr_up_evals { split_all.extend(v); }
            for v in &p.cpr_down_evals { split_all.extend(v); }
            for v in &p.evaluation_point_bytes { split_all.extend(v); }
            for v in &p.pcs_evals_bytes { split_all.extend(v); }
            if let Some(ref lp) = p.lookup_proof {
                split_all.extend(lookup_proof_to_bytes(lp));
            }
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

    let mem_snapshot = mem_tracker.stop();
    eprintln!("  {mem_snapshot}");

    group.finish();
}

/// PIOP-only benchmark (ideal check + combined poly resolver)
/// for the SHA-256 UAIR trace.
fn sha256_piop_only(c: &mut Criterion) {
    let mut group = c.benchmark_group("SHA-256 PIOP");
    group.sample_size(20);

    let trace = generate_sha256_trace(SHA256_NUM_VARS);
    let num_vars = SHA256_NUM_VARS;
    let num_constraints = zinc_uair::constraint_counter::count_constraints::<Sha256Uair>();
    let max_degree = zinc_uair::degree_counter::count_max_degree::<Sha256Uair>();

    // ── Ideal Check prover ──────────────────────────────────────────
    group.bench_function("IdealCheck/Prover", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let mut transcript = zinc_transcript::KeccakTranscript::new();
                let field_cfg = transcript.get_random_field_cfg::<
                    F, <F as Field>::Inner, MillerRabin
                >();
                let projected_trace = project_trace_coeffs::<F, bool, bool, 32>(
                    &trace, &[], &[], &field_cfg,
                );
                let projected_scalars = project_scalars::<F, Sha256Uair>(|scalar| {
                    let one = F::one_with_cfg(&field_cfg);
                    let zero = F::zero_with_cfg(&field_cfg);
                    DynamicPolynomialF::new(
                        scalar.iter().map(|coeff| {
                            if coeff.into_inner() { one.clone() } else { zero.clone() }
                        }).collect::<Vec<_>>()
                    )
                });
                let t = Instant::now();
                let _ = zinc_piop::ideal_check::IdealCheckProtocol::<
                    F
                >::prove_as_subprotocol::<Sha256Uair>(
                    &mut transcript,
                    &projected_trace,
                    &projected_scalars,
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
    let projected_trace = project_trace_coeffs::<F, bool, bool, 32>(
        &trace, &[], &[], &field_cfg,
    );
    let projected_scalars = project_scalars::<F, Sha256Uair>(|scalar| {
        let one = F::one_with_cfg(&field_cfg);
        let zero = F::zero_with_cfg(&field_cfg);
        DynamicPolynomialF::new(
            scalar.iter().map(|coeff| {
                if coeff.into_inner() { one.clone() } else { zero.clone() }
            }).collect::<Vec<_>>()
        )
    });
    let (_ic_proof, ic_state) =
        zinc_piop::ideal_check::IdealCheckProtocol::<
            F
        >::prove_as_subprotocol::<Sha256Uair>(
            &mut transcript,
            &projected_trace,
            &projected_scalars,
            num_constraints,
            num_vars,
            &field_cfg,
        )
        .expect("IC prover failed");

    let projecting_element: F = transcript.get_field_challenge(&field_cfg);
    let field_trace = project_trace_to_field::<F, 32>(
        &trace, &[], &[], &projecting_element,
    );
    let field_projected_scalars =
        project_scalars_to_field(projected_scalars.clone(), &projecting_element)
            .expect("scalar projection failed");

    group.bench_function("CombinedPolyResolver/Prover", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let mut tr = transcript.clone();
                let t = Instant::now();
                let _ = zinc_piop::combined_poly_resolver::CombinedPolyResolver::<
                    F
                >::prove_as_subprotocol::<Sha256Uair>(
                    &mut tr,
                    field_trace.clone(),
                    &ic_state.evaluation_point,
                    &field_projected_scalars,
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

/// **E2E benchmark for 8×SHA-256 + ECDSA** combining PIOP + Lookup + PCS.
///
/// Uses the full `pipeline::prove` / `pipeline::verify` API which includes:
///   1. IdealCheck → CombinedPolyResolver → Batched Lookup → PCS (prover)
///   2. PCS verify → Lookup verify → CPR verify → IC verify (verifier)
///
/// Both SHA-256 and ECDSA sub-protocols carry lookup column-typing constraints.
///
/// This is the "honest" combined prover/verifier time for the paper's target configuration.
fn sha256_8x_ecdsa_end_to_end(c: &mut Criterion) {
    use zinc_sha256_uair::CyclotomicIdeal;
    use zinc_ecdsa_uair::EcdsaIdealOverF;
    use zinc_uair::ideal::ImpossibleIdeal;
    use zinc_uair::ideal_collector::IdealOrZero;

    let mut group = c.benchmark_group("8xSHA256+ECDSA E2E (PIOP+Lookup+PCS)");
    group.sample_size(10);

    // ── Build traces ────────────────────────────────────────────────
    let sha_trace = generate_sha256_trace(SHA256_8X_NUM_VARS);
    assert_eq!(sha_trace.len(), SHA256_BATCH_SIZE);

    let ecdsa_trace = generate_zero_scalar_trace(ECDSA_NUM_VARS, ECDSA_BATCH_SIZE);
    assert_eq!(ecdsa_trace.len(), ECDSA_BATCH_SIZE);
    let ec_public_cols = vec![ecdsa_trace[0].clone(), ecdsa_trace[1].clone()];
    let sha_sig_pub = Sha256Uair::signature();
    let sha_public_cols: Vec<_> = sha_sig_pub.public_columns.iter()
        .map(|&i| sha_trace[i].clone()).collect();

    // ── PCS params ──────────────────────────────────────────────────
    type ShaZt = Sha256ZipTypes<i64, 32>;
    type ShaLc = IprsBPoly32R4B64<1, UNCHECKED>;
    let sha_params = ZipPlusParams::<ShaZt, ShaLc>::new(
        SHA256_8X_NUM_VARS, 1, ShaLc::new(512),
    );

    type EcZt = EcdsaScalarZipTypes;
    type EcLc = IprsInt4R4B64<1, UNCHECKED>;
    let ec_params = ZipPlusParams::<EcZt, EcLc>::new(
        ECDSA_NUM_VARS, 1, EcLc::new(512),
    );

    // ── Lookup specs ────────────────────────────────────────────────
    let sha_lookup_specs = sha256_lookup_specs();

    // ── Total Prover (IC + CPR + Lookup + PCS for both) ─────────────
    group.bench_function("TotalProver", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let t = Instant::now();
                let _sha = zinc_snark::pipeline::prove::<Sha256Uair, ShaZt, ShaLc, 32, UNCHECKED>(
                    &sha_params, &sha_trace, SHA256_8X_NUM_VARS, &sha_lookup_specs,
                );
                let _ec = zinc_snark::pipeline::prove_generic::<
                    EcdsaUairInt, Int<{ INT_LIMBS * 4 }>, EcZt, EcLc, FScalar, UNCHECKED,
                >(
                    &ec_params, &ecdsa_trace, ECDSA_NUM_VARS, &[],
                );
                total += t.elapsed();
            }
            total
        });
    });

    // ── Prepare proofs for verifier benchmark ───────────────────────
    let sha_proof = zinc_snark::pipeline::prove::<Sha256Uair, ShaZt, ShaLc, 32, UNCHECKED>(
        &sha_params, &sha_trace, SHA256_8X_NUM_VARS, &sha_lookup_specs,
    );
    let ec_proof = zinc_snark::pipeline::prove_generic::<
        EcdsaUairInt, Int<{ INT_LIMBS * 4 }>, EcZt, EcLc, FScalar, UNCHECKED,
    >(
        &ec_params, &ecdsa_trace, ECDSA_NUM_VARS, &[],
    );

    // ── Total Verifier (IC + CPR + Lookup verify + PCS verify) ──────
    group.bench_function("TotalVerifier", |b| {
        b.iter(|| {
            let sha_r = zinc_snark::pipeline::verify::<Sha256Uair, ShaZt, ShaLc, 32, UNCHECKED, _, _>(
                &sha_params, &sha_proof, SHA256_8X_NUM_VARS,
                |_: &IdealOrZero<CyclotomicIdeal>| zinc_snark::pipeline::TrivialIdeal,
                &sha_public_cols,
            );
            let _ = black_box(sha_r);
            let ec_r = zinc_snark::pipeline::verify_generic::<
                EcdsaUairInt, Int<{ INT_LIMBS * 4 }>, EcZt, EcLc, FScalar, UNCHECKED, EcdsaIdealOverF, _,
            >(
                &ec_params, &ec_proof, ECDSA_NUM_VARS,
                |ideal: &IdealOrZero<ImpossibleIdeal>| match ideal {
                    IdealOrZero::Zero => EcdsaIdealOverF,
                    IdealOrZero::Ideal(_) => panic!("ECDSA has no non-zero ideal constraints"),
                },
                &ec_public_cols,
            );
            let _ = black_box(ec_r);
        });
    });

    // ── Timing breakdown ────────────────────────────────────────────
    eprintln!("\n=== 8xSHA256+ECDSA E2E Timing ===");
    eprintln!("  SHA prover: IC={:?}, CPR={:?}, Lookup={:?}, PCS(commit={:?}, prove={:?}), total={:?}",
        sha_proof.timing.ideal_check,
        sha_proof.timing.combined_poly_resolver,
        sha_proof.timing.lookup,
        sha_proof.timing.pcs_commit,
        sha_proof.timing.pcs_prove,
        sha_proof.timing.total,
    );
    eprintln!("  ECDSA prover: IC={:?}, CPR={:?}, Lookup={:?}, PCS(commit={:?}, prove={:?}), total={:?}",
        ec_proof.timing.ideal_check,
        ec_proof.timing.combined_poly_resolver,
        ec_proof.timing.lookup,
        ec_proof.timing.pcs_commit,
        ec_proof.timing.pcs_prove,
        ec_proof.timing.total,
    );

    group.finish();
}

/// **Full pipeline benchmark** using the `pipeline::prove` and `pipeline::verify` functions.
///
/// This measures the actual end-to-end pipeline including PIOP proof serialization/deserialization,
/// and reports total proof sizes (PCS + PIOP data).
fn sha256_full_pipeline(c: &mut Criterion) {
    let mem_tracker = MemoryTracker::start();

    use zinc_sha256_uair::CyclotomicIdeal;
    use zinc_uair::ideal_collector::IdealOrZero;
    type Lc = IprsBPoly32R4B16<1, UNCHECKED>;
    type Zt = Sha256ZipTypes<i64, 32>;

    let mut group = c.benchmark_group("SHA-256 Full Pipeline");
    group.sample_size(20);

    let trace = generate_sha256_trace(SHA256_NUM_VARS);
    let sha_sig = Sha256Uair::signature();
    let sha_public_cols: Vec<_> = sha_sig.public_columns.iter()
        .map(|&i| trace[i].clone()).collect();
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
                    &[],
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
        &[],
    );

    // ── Full Verifier ───────────────────────────────────────────────
    group.bench_function("FullVerifier/1xSHA256", |b| {
        b.iter(|| {
            let result = zinc_snark::pipeline::verify::<Sha256Uair, Zt, Lc, 32, UNCHECKED, _, _>(
                &params,
                &zinc_proof,
                num_vars,
                |_: &IdealOrZero<CyclotomicIdeal>| zinc_snark::pipeline::TrivialIdeal,
                &sha_public_cols,
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
    // Run separate PCS for the BinaryPoly (23 cols) and Int (3 cols) batches
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

        // Derive the real PCS evaluation point from the full pipeline proof.
        let split_pt: Vec<F> = zinc_snark::pipeline::pcs_point_from_proof(&zinc_proof);

        // BinaryPoly batch PCS
        let (poly_hint, _) = ZipPlus::<Zt, Lc>::commit(&params, &poly_trace)
            .expect("poly commit");
        let (_, poly_proof) = ZipPlus::<Zt, Lc>::prove::<F, UNCHECKED>(
            &params, &poly_trace, &split_pt, &poly_hint,
        ).expect("poly prove");

        // Int batch PCS
        let (int_hint, _) = ZipPlus::<IntZt, IntLc>::commit(&int_params, &int_trace)
            .expect("int commit");
        let (_, int_proof) = ZipPlus::<IntZt, IntLc>::prove::<F, UNCHECKED>(
            &int_params, &int_trace, &split_pt, &int_hint,
        ).expect("int prove");

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

    let mem_snapshot = mem_tracker.stop();
    eprintln!("  {mem_snapshot}");

    group.finish();
}

// ─── LogUp Comparison Benchmark ─────────────────────────────────────────────

/// Side-by-side comparison of LogUp proving strategies for 8×SHA-256 + ECDSA.
///
/// Three variants are compared:
/// 1. **NoLookup** — baseline with no lookup constraints.
/// 2. **ClassicLogUp** — classic batched decomposition LogUp where CPR and
///    lookup sumchecks are batched into a single multi-degree sumcheck
///    (via `prove_classic_logup`).
/// 3. **SeparateLogUp** — classic batched decomposition LogUp where CPR
///    and lookup run independent sumchecks sequentially
///    (via `prove`, separate CPR then lookup).
///
/// ECDSA columns have no lookup constraints — only SHA-256's
/// BinaryPoly columns go through LogUp.
fn sha256_8x_ecdsa_logup_comparison(c: &mut Criterion) {
    use zinc_sha256_uair::CyclotomicIdeal;
    use zinc_ecdsa_uair::EcdsaIdealOverF;
    use zinc_uair::ideal::ImpossibleIdeal;
    use zinc_uair::ideal_collector::IdealOrZero;

    let mut group = c.benchmark_group("8xSHA256+ECDSA LogUp Comparison");
    group.sample_size(10);

    // ── Build traces ────────────────────────────────────────────────
    let sha_trace = generate_sha256_trace(SHA256_8X_NUM_VARS);
    assert_eq!(sha_trace.len(), SHA256_BATCH_SIZE);

    let ecdsa_trace = generate_zero_scalar_trace(ECDSA_NUM_VARS, ECDSA_BATCH_SIZE);
    assert_eq!(ecdsa_trace.len(), ECDSA_BATCH_SIZE);
    let ec_public_cols = vec![ecdsa_trace[0].clone(), ecdsa_trace[1].clone()];
    let sha_sig_pub = Sha256Uair::signature();
    let sha_public_cols: Vec<_> = sha_sig_pub.public_columns.iter()
        .map(|&i| sha_trace[i].clone()).collect();

    // ── PCS params ──────────────────────────────────────────────────
    type ShaZt = Sha256ZipTypes<i64, 32>;
    type ShaLc = IprsBPoly32R4B64<1, UNCHECKED>;
    let sha_params = ZipPlusParams::<ShaZt, ShaLc>::new(
        SHA256_8X_NUM_VARS, 1, ShaLc::new(512),
    );

    type EcZt = EcdsaScalarZipTypes;
    type EcLc = IprsInt4R4B64<1, UNCHECKED>;
    let ec_params = ZipPlusParams::<EcZt, EcLc>::new(
        ECDSA_NUM_VARS, 1, EcLc::new(512),
    );

    let sha_lookup_specs = sha256_lookup_specs();

    // ── No-Lookup Baseline (SHA only, no ECDSA) ────────────────────
    group.bench_function("NoLookup/SHAProver", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let t = Instant::now();
                let _sha = zinc_snark::pipeline::prove_classic_logup::<Sha256Uair, ShaZt, ShaLc, 32, UNCHECKED>(
                    &sha_params, &sha_trace, SHA256_8X_NUM_VARS, &[],
                );
                total += t.elapsed();
            }
            total
        });
    });

    // ── ClassicLogUp: batched CPR+Lookup multi-degree sumcheck ──────
    group.bench_function("ClassicLogUp/SHAProver", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let t = Instant::now();
                let _sha = zinc_snark::pipeline::prove_classic_logup::<Sha256Uair, ShaZt, ShaLc, 32, UNCHECKED>(
                    &sha_params, &sha_trace, SHA256_8X_NUM_VARS, &sha_lookup_specs,
                );
                total += t.elapsed();
            }
            total
        });
    });

    group.bench_function("ClassicLogUp/TotalProver", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let t = Instant::now();
                let _sha = zinc_snark::pipeline::prove_classic_logup::<Sha256Uair, ShaZt, ShaLc, 32, UNCHECKED>(
                    &sha_params, &sha_trace, SHA256_8X_NUM_VARS, &sha_lookup_specs,
                );
                let _ec = zinc_snark::pipeline::prove_generic::<
                    EcdsaUairInt, Int<{ INT_LIMBS * 4 }>, EcZt, EcLc, FScalar, UNCHECKED,
                >(
                    &ec_params, &ecdsa_trace, ECDSA_NUM_VARS, &[],
                );
                total += t.elapsed();
            }
            total
        });
    });

    // ── SeparateLogUp: CPR and lookup run independent sumchecks ─────
    group.bench_function("SeparateLogUp/SHAProver", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let t = Instant::now();
                let _sha = zinc_snark::pipeline::prove::<Sha256Uair, ShaZt, ShaLc, 32, UNCHECKED>(
                    &sha_params, &sha_trace, SHA256_8X_NUM_VARS, &sha_lookup_specs,
                );
                total += t.elapsed();
            }
            total
        });
    });

    group.bench_function("SeparateLogUp/TotalProver", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let t = Instant::now();
                let _sha = zinc_snark::pipeline::prove::<Sha256Uair, ShaZt, ShaLc, 32, UNCHECKED>(
                    &sha_params, &sha_trace, SHA256_8X_NUM_VARS, &sha_lookup_specs,
                );
                let _ec = zinc_snark::pipeline::prove_generic::<
                    EcdsaUairInt, Int<{ INT_LIMBS * 4 }>, EcZt, EcLc, FScalar, UNCHECKED,
                >(
                    &ec_params, &ecdsa_trace, ECDSA_NUM_VARS, &[],
                );
                total += t.elapsed();
            }
            total
        });
    });

    // ── Generate proofs for verifier bench & proof size ──────────────
    let batched_sha_proof = zinc_snark::pipeline::prove_classic_logup::<Sha256Uair, ShaZt, ShaLc, 32, UNCHECKED>(
        &sha_params, &sha_trace, SHA256_8X_NUM_VARS, &sha_lookup_specs,
    );
    let separate_sha_proof = zinc_snark::pipeline::prove::<Sha256Uair, ShaZt, ShaLc, 32, UNCHECKED>(
        &sha_params, &sha_trace, SHA256_8X_NUM_VARS, &sha_lookup_specs,
    );
    let ec_proof = zinc_snark::pipeline::prove_generic::<
        EcdsaUairInt, Int<{ INT_LIMBS * 4 }>, EcZt, EcLc, FScalar, UNCHECKED,
    >(
        &ec_params, &ecdsa_trace, ECDSA_NUM_VARS, &[],
    );

    // ── ClassicLogUp (batched) Verifier ──────────────────────────────
    group.bench_function("ClassicLogUp/TotalVerifier", |b| {
        b.iter(|| {
            let sha_r = zinc_snark::pipeline::verify::<Sha256Uair, ShaZt, ShaLc, 32, UNCHECKED, _, _>(
                &sha_params, &batched_sha_proof, SHA256_8X_NUM_VARS,
                |_: &IdealOrZero<CyclotomicIdeal>| zinc_snark::pipeline::TrivialIdeal,
                &sha_public_cols,
            );
            let _ = black_box(sha_r);
            let ec_r = zinc_snark::pipeline::verify_generic::<
                EcdsaUairInt, Int<{ INT_LIMBS * 4 }>, EcZt, EcLc, FScalar, UNCHECKED, EcdsaIdealOverF, _,
            >(
                &ec_params, &ec_proof, ECDSA_NUM_VARS,
                |ideal: &IdealOrZero<ImpossibleIdeal>| match ideal {
                    IdealOrZero::Zero => EcdsaIdealOverF,
                    IdealOrZero::Ideal(_) => panic!("ECDSA has no non-zero ideal constraints"),
                },
                &ec_public_cols,
            );
            let _ = black_box(ec_r);
        });
    });

    // ── SeparateLogUp Verifier ───────────────────────────────────────
    group.bench_function("SeparateLogUp/TotalVerifier", |b| {
        b.iter(|| {
            let sha_r = zinc_snark::pipeline::verify::<Sha256Uair, ShaZt, ShaLc, 32, UNCHECKED, _, _>(
                &sha_params, &separate_sha_proof, SHA256_8X_NUM_VARS,
                |_: &IdealOrZero<CyclotomicIdeal>| zinc_snark::pipeline::TrivialIdeal,
                &sha_public_cols,
            );
            let _ = black_box(sha_r);
            let ec_r = zinc_snark::pipeline::verify_generic::<
                EcdsaUairInt, Int<{ INT_LIMBS * 4 }>, EcZt, EcLc, FScalar, UNCHECKED, EcdsaIdealOverF, _,
            >(
                &ec_params, &ec_proof, ECDSA_NUM_VARS,
                |ideal: &IdealOrZero<ImpossibleIdeal>| match ideal {
                    IdealOrZero::Zero => EcdsaIdealOverF,
                    IdealOrZero::Ideal(_) => panic!("ECDSA has no non-zero ideal constraints"),
                },
                &ec_public_cols,
            );
            let _ = black_box(ec_r);
        });
    });

    // ── Proof size & timing comparison ──────────────────────────────
    let batched_lookup_size = batched_sha_proof.lookup_proof.as_ref().map_or(0, lookup_proof_byte_count);
    let separate_lookup_size = separate_sha_proof.lookup_proof.as_ref().map_or(0, lookup_proof_byte_count);

    eprintln!("\n=== LogUp Comparison: 8xSHA256+ECDSA ===");
    eprintln!("  ClassicLogUp (batched CPR+Lookup multi-degree sumcheck):");
    eprintln!("    SHA prover: IC={:?}, CPR+Lookup={:?}, PCS(commit={:?}, prove={:?}), total={:?}",
        batched_sha_proof.timing.ideal_check,
        batched_sha_proof.timing.combined_poly_resolver,
        batched_sha_proof.timing.pcs_commit,
        batched_sha_proof.timing.pcs_prove,
        batched_sha_proof.timing.total,
    );
    eprintln!("    Lookup proof size: {} bytes ({:.1} KB)", batched_lookup_size, batched_lookup_size as f64 / 1024.0);
    eprintln!("  SeparateLogUp (CPR and lookup run independent sumchecks):");
    eprintln!("    SHA prover: IC={:?}, CPR={:?}, Lookup={:?}, PCS(commit={:?}, prove={:?}), total={:?}",
        separate_sha_proof.timing.ideal_check,
        separate_sha_proof.timing.combined_poly_resolver,
        separate_sha_proof.timing.lookup,
        separate_sha_proof.timing.pcs_commit,
        separate_sha_proof.timing.pcs_prove,
        separate_sha_proof.timing.total,
    );
    eprintln!("    Lookup proof size: {} bytes ({:.1} KB)", separate_lookup_size, separate_lookup_size as f64 / 1024.0);
    if separate_lookup_size > 0 && batched_lookup_size > 0 {
        eprintln!("  Batched vs Separate lookup proof: {:.1}× ratio",
            separate_lookup_size as f64 / batched_lookup_size as f64);
    }
    eprintln!("  ECDSA prover (shared): IC={:?}, CPR={:?}, Lookup={:?}, PCS(commit={:?}, prove={:?}), total={:?}",
        ec_proof.timing.ideal_check,
        ec_proof.timing.combined_poly_resolver,
        ec_proof.timing.lookup,
        ec_proof.timing.pcs_commit,
        ec_proof.timing.pcs_prove,
        ec_proof.timing.total,
    );

    group.finish();
}

criterion_group!(benches, sha256_single, sha256_8x_ecdsa, sha256_piop_only, sha256_8x_ecdsa_end_to_end, sha256_full_pipeline, sha256_8x_ecdsa_logup_comparison);
criterion_main!(benches);
