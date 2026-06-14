//! Criterion benchmark for the Blake3 `F_2[X]` UAIR
//! ([`zinc_test_uair::Blake3F2Uair`]) routed through
//! [`ZincPlusPiopF2`].
//!
//! Sibling to `benches/f2_sha256.rs`. Same three bench groups:
//!
//! 1. `Zinc+ F_2 Blake3` — end-to-end (WitnessGen / Prove / Verify).
//! 2. `Zinc+ F_2 Blake3 Steps` — top-level prover and verifier step
//!    breakdown (Commit / UAIR / Open).
//! 3. `Zinc+ F_2 Blake3 Micro` — fine-grained timings *inside* the
//!    UAIR and Open phases.
//!
//! Blake3-specific knobs:
//! - **No bit-op virtual columns.** The Blake3 F_2 UAIR uses
//!   selector-gated LSB checks instead of SHA-256's packed `κ`
//!   compensator + `SHR^j(PA_C)` bit-op virtuals, so the
//!   `F2BitOpVirtualSpec` list is empty.
//! - **Effective max degree = 2.** `Blake3F2Uair`'s rotation pins and
//!   selector-gated LSB checks are both `selector × residual`, so
//!   `prove_f2_uair_with_groups` dispatches to the row-major
//!   `prove_combined` IC path rather than SHA's MLE-first
//!   `prove_linear` lane. The `UAIR-a-F2NativeIC` micro therefore
//!   benches `prove_combined` directly.
//! - **Trace shape.** 60 rows per compression + 4-row `cv_N` output
//!   prefix. `MIN_NUM_VARS = 6` (64 rows ⊇ 64 active for one
//!   compression). The `NVARS_SWEEP` mirrors `f2_sha256.rs` for an
//!   apples-to-apples comparison.
//!
//! **Required features**: `parallel`, `simd`, `unchecked`. The bench
//! is gated by `required-features` in `Cargo.toml`; running without
//! those features is rejected by cargo.

#![allow(clippy::arithmetic_side_effects)]

use std::hint::black_box;
use std::marker::PhantomData;

use criterion::{
    BenchmarkGroup, BenchmarkId, Criterion, criterion_group, criterion_main,
    measurement::WallTime,
};
use crypto_bigint::U64;
use crypto_primitives::{Field, FromWithConfig, crypto_bigint_int::Int, crypto_bigint_uint::Uint};
use rand::rng;
use zinc_poly::{
    mle::DenseMultilinearExtension,
    univariate::{
        binary::{BinaryPoly, BinaryPolyInnerProduct},
        binary_gf128::BinaryFieldGF128,
        dense::{DensePolyInnerProduct, DensePolynomial},
    },
};
use zinc_primality::MillerRabin;
use zinc_protocol::f2_prove::{
    F2BitOpVirtualSpec, F2FullProof, F2VerifierSubclaim, F2VirtualBpSpec, F2ZincTypes,
    ZincPlusPiopF2,
};
use zinc_test_uair::{
    Blake3F2Ideal, Blake3F2Uair, GenerateRandomTrace, blake3_f2_project_ideal,
    blake3_f2_project_scalar,
};
use zinc_transcript::{Blake3Transcript, traits::Transcript};
use zinc_uair::{constraint_counter::count_constraints, ideal_collector::IdealOrZero};
use zinc_utils::inner_product::MBSInnerProduct;
use zip_plus::{
    code::{
        F2LinearOpener, LinearCode,
        raa::RaaConfig,
        raa_f2::{RaaF2Code, recommended_num_column_openings},
    },
    pcs::structs::{ZipPlusParams, ZipTypes},
};

use zinc_piop::{
    ideal_check::IdealCheckProtocol,
    multipoint_eval::MultipointEval,
    sumcheck::multi_degree::MultiDegreeSumcheck,
};

// ---------------------------------------------------------------------------
// F_2 ZipTypes / F2ZincTypes infrastructure (mirrors f2_sha256.rs).
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct BenchF2ZipTypes<const D: usize> {}

impl<const D: usize> ZipTypes for BenchF2ZipTypes<D> {
    const NUM_COLUMN_OPENINGS: usize = recommended_num_column_openings(REP);
    type Eval = BinaryPoly<D>;
    type Cw = BinaryPoly<D>;
    type Fmod = Uint<{ U64::LIMBS * 4 }>;
    type PrimeTest = MillerRabin;
    type Chal = i128;
    type Pt = i128;
    type CombR = Int<{ U64::LIMBS * 8 }>;
    type Comb = DensePolynomial<Self::CombR, D>;
    type EvalDotChal = BinaryPolyInnerProduct<Self::Chal, D>;
    type CombDotChal =
        DensePolyInnerProduct<Self::CombR, Self::Chal, Self::CombR, MBSInnerProduct, D>;
    type ArrCombRDotChal = MBSInnerProduct;
}

#[derive(Copy, Clone)]
struct BenchRaaConfig;
impl RaaConfig for BenchRaaConfig {
    const PERMUTE_IN_PLACE: bool = false;
    const CHECK_FOR_OVERFLOWS: bool = false;
}

#[derive(Clone, Debug)]
struct BenchF2Types<const D: usize>(PhantomData<()>);

const REP: usize = 4;

impl<const D: usize> F2ZincTypes<D> for BenchF2Types<D> {
    type BinaryZt = BenchF2ZipTypes<64>;
    type BinaryLc = RaaF2Code<Self::BinaryZt, BenchRaaConfig, REP>;
}

// ---------------------------------------------------------------------------
// Bench setup.
// ---------------------------------------------------------------------------

type R = Int<4>;
type U = Blake3F2Uair<R>;
const D: usize = 32;

/// Bit-op virtual columns declared by the Blake3 F_2 UAIR bench.
///
/// This list governs the *open-path* virtualisation hint, which tells
/// the protocol "this committed-looking column can be reconstructed
/// from another column via the named bit op, so don't re-Merkle-commit
/// it". Blake3 currently has no such virtualisable committed columns
/// — every committed cell is independent data. The UAIR's `SHR^j`
/// virtuals on `W_KAPPA_LSB_{A,B}` (added by the κ-lift to extract
/// per-step LSB κ bits, see `Blake3F2Uair::signature`) are declared
/// inside the signature itself and handled automatically by the
/// prover/verifier — they do not need an entry here.
fn blake3_f2_bit_op_virtuals() -> Vec<F2BitOpVirtualSpec> {
    Vec::new()
}

// ---------------------------------------------------------------------------
// Proof-size measurement.
//
// Identical layout to `f2_sha256.rs`'s `f2_full_proof_parts` /
// `eprint_f2_proof_size`. Kept inline (rather than factored into a
// shared module) so the bench file is self-contained — matches the
// architectural convention of the SHA bench.
// ---------------------------------------------------------------------------

#[allow(clippy::arithmetic_side_effects)]
fn f2_full_proof_parts(proof: &F2FullProof<D>) -> Vec<(&'static str, Vec<u8>)> {
    use crypto_primitives::Field;
    use zinc_poly::univariate::F2PackU64;
    use zinc_transcript::traits::{ConstTranscribable, GenTranscribable, Transcribable};

    let commitment: Vec<u8> = (*proof.commitment.root).to_vec();

    const ALPHA_BYTES: usize =
        <<BinaryFieldGF128 as Field>::Inner as ConstTranscribable>::NUM_BYTES;
    let mut alpha = vec![0u8; ALPHA_BYTES];
    proof.uair.alpha.inner().write_transcription_bytes_exact(&mut alpha);

    let mut gamma = vec![0u8; ALPHA_BYTES];
    proof.uair.gamma.inner().write_transcription_bytes_exact(&mut gamma);

    let mut ic_proof = vec![0u8; proof.uair.ic_proof.get_num_bytes()];
    proof.uair.ic_proof.write_transcription_bytes_exact(&mut ic_proof);

    let mut sumcheck = vec![0u8; proof.uair.sumcheck_proof.get_num_bytes()];
    proof.uair.sumcheck_proof.write_transcription_bytes_exact(&mut sumcheck);

    let mut col_evals_at_rstar = vec![0u8; proof.uair.column_evals_at_rstar.len() * ALPHA_BYTES];
    for (i, v) in proof.uair.column_evals_at_rstar.iter().enumerate() {
        v.inner().write_transcription_bytes_exact(
            &mut col_evals_at_rstar[i * ALPHA_BYTES..(i + 1) * ALPHA_BYTES],
        );
    }

    let mut mp_sumcheck =
        vec![0u8; proof.multipoint_eval.sumcheck_proof.get_num_bytes()];
    proof
        .multipoint_eval
        .sumcheck_proof
        .write_transcription_bytes_exact(&mut mp_sumcheck);

    let mut open_evals_at_r_0 = vec![0u8; proof.open_evals_at_r_0.len() * ALPHA_BYTES];
    for (i, v) in proof.open_evals_at_r_0.iter().enumerate() {
        v.inner().write_transcription_bytes_exact(
            &mut open_evals_at_r_0[i * ALPHA_BYTES..(i + 1) * ALPHA_BYTES],
        );
    }

    // Un-lifted open: claim / b-vector / combined-row are `GF128Poly<D>` (D GF(2^128)
    // bit-slice-eval coefficients), each serialising as `D · ALPHA_BYTES`.
    let mut coeff_buf = vec![0u8; ALPHA_BYTES];
    let mut serialize_gf128poly = |out: &mut Vec<u8>, p: &zinc_poly::univariate::binary_gf128::GF128Poly<D>| {
        for c in p.coeffs.iter() {
            c.inner().write_transcription_bytes_exact(&mut coeff_buf);
            out.extend_from_slice(&coeff_buf);
        }
    };

    let mut lifted_claim = Vec::with_capacity(D * ALPHA_BYTES);
    serialize_gf128poly(&mut lifted_claim, &proof.open.lifted_claim);

    let mut b_vector = Vec::with_capacity(proof.open.b_vector.len() * D * ALPHA_BYTES);
    for v in &proof.open.b_vector {
        serialize_gf128poly(&mut b_vector, v);
    }

    let mut combined_row = Vec::with_capacity(proof.open.combined_row.len() * D * ALPHA_BYTES);
    for v in &proof.open.combined_row {
        serialize_gf128poly(&mut combined_row, v);
    }

    let bytes_per_cell = 2 * D.div_ceil(8);
    let mut opened_values: Vec<u8> = Vec::new();
    let mut opened_merkle: Vec<u8> = Vec::new();
    let mut opened_headers: Vec<u8> = Vec::new();
    for col in &proof.open.opened_columns {
        opened_headers.extend_from_slice(&(col.column_idx as u64).to_le_bytes());
        opened_headers
            .extend_from_slice(&(col.merkle_proof.leaf_index as u64).to_le_bytes());
        opened_headers
            .extend_from_slice(&(col.merkle_proof.leaf_count as u64).to_le_bytes());
        for c in &col.column_values {
            let packed = c.pack_u64();
            opened_values.extend_from_slice(&packed.to_le_bytes()[..bytes_per_cell]);
        }
        for sib in &col.merkle_proof.siblings {
            opened_merkle.extend_from_slice(&**sib);
        }
    }

    vec![
        ("commitment.root", commitment),
        ("uair.alpha", alpha),
        ("uair.gamma", gamma),
        ("uair.ic_proof", ic_proof),
        ("uair.sumcheck", sumcheck),
        ("uair.col_evals@r*", col_evals_at_rstar),
        ("mp_eval.sumcheck", mp_sumcheck),
        ("mp_eval.col_evals@r_0", open_evals_at_r_0),
        ("open.lifted_claim", lifted_claim),
        ("open.b_vector", b_vector),
        ("open.combined_row", combined_row),
        ("open.opened/values", opened_values),
        ("open.opened/merkle", opened_merkle),
        ("open.opened/headers", opened_headers),
    ]
}

fn eprint_f2_proof_size(label: &str, proof: &F2FullProof<D>) {
    let parts = f2_full_proof_parts(proof);
    let refs: Vec<(&str, &[u8])> =
        parts.iter().map(|(name, b)| (*name, b.as_slice())).collect();
    zip_plus::utils::eprint_bytes_size_breakdown(label, &refs);
}

struct ProverFixture {
    trace: zinc_uair::UairTrace<'static, BinaryPoly<D>, BinaryPoly<D>, D>,
    pp: ZipPlusParams<
        <BenchF2Types<D> as F2ZincTypes<D>>::BinaryZt,
        <BenchF2Types<D> as F2ZincTypes<D>>::BinaryLc,
    >,
    num_vars: usize,
    num_primary: usize,
    paired_primary_witness: Vec<
        zinc_poly::mle::DenseMultilinearExtension<
            BinaryPoly<{ zinc_protocol::f2_prove::PACKED_STORAGE_WIDTH }>,
        >,
    >,
}

fn setup_prover(num_vars: usize) -> ProverFixture {
    let mut rng_local = rng();
    let num_rows: usize = 8;
    let poly_size = 1usize << num_vars;
    let row_len = poly_size / num_rows;
    assert_eq!(num_rows * row_len, poly_size);

    let trace = U::generate_random_trace(num_vars, &mut rng_local);
    let lc = <BenchF2Types<D> as F2ZincTypes<D>>::BinaryLc::new(row_len);
    let pp = ZipPlusParams::new(num_vars, num_rows, lc);

    let paired_primary_witness = zinc_protocol::f2_prove::pair_primary_witness_polys_pub::<D>(
        &trace.binary_poly,
        zinc_test_uair::blake3_f2::cols::NUM_BIN_PUB,
        &blake3_f2_bit_op_virtuals(),
    );

    ProverFixture {
        trace,
        pp,
        num_vars,
        num_primary: zinc_test_uair::blake3_f2::cols::NUM_BIN,
        paired_primary_witness,
    }
}

// ---------------------------------------------------------------------------
// Per-region: top-level prover & verifier steps.
// ---------------------------------------------------------------------------

fn bench_prover_steps(group: &mut BenchmarkGroup<WallTime>, id: &str, fx: &ProverFixture) {
    group.bench_function(BenchmarkId::new("1-Commit", id), |bench| {
        bench.iter(|| {
            let mut transcript = Blake3Transcript::new();
            let (hint, comm) =
                ZincPlusPiopF2::<BenchF2Types<D>, U, D>::commit_and_absorb_f2_trace_with_virtuals(
                    &mut transcript,
                    &fx.pp,
                    &fx.trace.binary_poly,
                    &blake3_f2_bit_op_virtuals(),
                )
                .expect("commit should succeed");
            black_box((hint, comm));
        });
    });

    group.bench_function(BenchmarkId::new("2-UAIR", id), |bench| {
        bench.iter_batched(
            || {
                let mut transcript = Blake3Transcript::new();
                let _ =
                    ZincPlusPiopF2::<BenchF2Types<D>, U, D>::commit_and_absorb_f2_trace_with_virtuals(
                        &mut transcript,
                        &fx.pp,
                        &fx.trace.binary_poly,
                        &blake3_f2_bit_op_virtuals(),
                    )
                    .expect("commit should succeed");
                transcript
            },
            |mut transcript| {
                let (proof, subclaim, _projected_trace) = ZincPlusPiopF2::<BenchF2Types<D>, U, D>
                    ::prove_f2_uair_with_groups(
                        &mut transcript,
                        &fx.trace,
                        &[] as &[F2VirtualBpSpec],
                        &[],
                        &[],
                        &[],
                        &[],
                        fx.num_vars,
                        blake3_f2_project_scalar::<R>,
                    )
                    .expect("UAIR prove should succeed");
                black_box((proof, subclaim));
            },
            criterion::BatchSize::PerIteration,
        );
    });

    group.bench_function(BenchmarkId::new("3-Open", id), |bench| {
        bench.iter_batched(
            || {
                let mut transcript = Blake3Transcript::new();
                let (hint, _comm) =
                    ZincPlusPiopF2::<BenchF2Types<D>, U, D>::commit_and_absorb_f2_trace_with_virtuals(
                        &mut transcript,
                        &fx.pp,
                        &fx.trace.binary_poly,
                        &blake3_f2_bit_op_virtuals(),
                    )
                    .expect("commit should succeed");
                let (_uair_proof, subclaim, _projected_trace) = ZincPlusPiopF2::<BenchF2Types<D>, U, D>
                    ::prove_f2_uair_with_groups(
                        &mut transcript,
                        &fx.trace,
                        &[],
                        &[],
                        &[],
                        &[],
                        &[],
                        fx.num_vars,
                        blake3_f2_project_scalar::<R>,
                    )
                    .expect("UAIR prove should succeed");
                (hint, subclaim, transcript)
            },
            |(hint, subclaim, mut transcript)| {
                let open_proof = ZincPlusPiopF2::<BenchF2Types<D>, U, D>::prove_f2_open(
                    &mut transcript,
                    &fx.pp,
                    &hint,
                    &fx.trace.binary_poly,
                    &subclaim.sumcheck_point,
                    &subclaim.alpha,
                    recommended_num_column_openings(REP),
                );
                black_box(open_proof);
            },
            criterion::BatchSize::PerIteration,
        );
    });
}

fn bench_verifier_steps(group: &mut BenchmarkGroup<WallTime>, id: &str, fx: &ProverFixture) {
    let proof = {
        let mut transcript = Blake3Transcript::new();
        ZincPlusPiopF2::<BenchF2Types<D>, U, D>::prove_f2_full_with_bit_ops(
            &mut transcript,
            &fx.pp,
            &fx.trace,
            &[],
            &blake3_f2_bit_op_virtuals(),
            fx.num_vars,
            blake3_f2_project_scalar::<R>,
            recommended_num_column_openings(REP),
        )
        .expect("prove for verifier bench should succeed")
    };

    let public_binary_cols = &fx.trace.binary_poly[..zinc_test_uair::blake3_f2::cols::NUM_BIN_PUB];
    group.bench_function(BenchmarkId::new("1-VerifyUAIR", id), |bench| {
        bench.iter_batched(
            || {
                let mut transcript = Blake3Transcript::new();
                ZincPlusPiopF2::<BenchF2Types<D>, U, D>::absorb_commitment(
                    &mut transcript,
                    &proof.commitment,
                );
                ZincPlusPiopF2::<BenchF2Types<D>, U, D>::absorb_public_binary_cols(
                    &mut transcript,
                    public_binary_cols,
                );
                transcript
            },
            |mut transcript| {
                let subclaim = ZincPlusPiopF2::<BenchF2Types<D>, U, D>::verify_f2_uair(
                    &mut transcript,
                    &proof.uair,
                    &[],
                    fx.num_vars,
                    fx.num_primary,
                    |ideal: &IdealOrZero<Blake3F2Ideal>| blake3_f2_project_ideal(ideal),
                )
                .expect("UAIR verify should succeed");
                black_box(subclaim);
            },
            criterion::BatchSize::PerIteration,
        );
    });

    group.bench_function(BenchmarkId::new("2-VerifyOpen", id), |bench| {
        bench.iter_batched(
            || {
                let mut transcript = Blake3Transcript::new();
                ZincPlusPiopF2::<BenchF2Types<D>, U, D>::absorb_commitment(
                    &mut transcript,
                    &proof.commitment,
                );
                ZincPlusPiopF2::<BenchF2Types<D>, U, D>::absorb_public_binary_cols(
                    &mut transcript,
                    public_binary_cols,
                );
                let subclaim =
                    ZincPlusPiopF2::<BenchF2Types<D>, U, D>::verify_f2_uair(
                        &mut transcript,
                        &proof.uair,
                        &[],
                        fx.num_vars,
                        fx.num_primary,
                        |ideal: &IdealOrZero<Blake3F2Ideal>| blake3_f2_project_ideal(ideal),
                    )
                    .expect("UAIR verify should succeed");
                // The full proof opens at the multipoint-eval output point
                // `r_0`, not the sumcheck point `r*`. Advance the transcript
                // through the mp phase (mirror `verify_f2_full_with_bit_ops`)
                // so the open verifies at `r_0`. The prover got `&[]` XOR-
                // virtual-bp specs, so `open_evals_at_r_0` is exactly the
                // primary col evals.
                let mp_subclaim = MultipointEval::<BinaryFieldGF128>::verify_as_subprotocol(
                    &mut transcript,
                    proof.multipoint_eval.clone(),
                    &subclaim.sumcheck_point,
                    &proof.uair.column_evals_at_rstar,
                    &[],
                    &[],
                    fx.num_vars,
                    &(),
                )
                .expect("mp verify should succeed");
                let r_0 = mp_subclaim.sumcheck_subclaim.point.clone();
                let mut buf = vec![
                    0u8;
                    <<BinaryFieldGF128 as Field>::Inner
                        as zinc_transcript::traits::ConstTranscribable>::NUM_BYTES
                ];
                for v in &proof.open_evals_at_r_0 {
                    transcript.absorb_random_field(v, &mut buf);
                }
                let subclaim = F2VerifierSubclaim {
                    ic_evaluation_point: subclaim.ic_evaluation_point,
                    alpha: subclaim.alpha,
                    sumcheck_point: r_0,
                    primary_column_evals: proof.open_evals_at_r_0.clone(),
                    virtual_column_evals: Vec::new(),
                    hadamard_rstar: Vec::new(),
                    hadamard_pairs: Vec::new(),
                };
                (transcript, subclaim)
            },
            |(mut transcript, subclaim)| {
                ZincPlusPiopF2::<BenchF2Types<D>, U, D>::verify_f2_open_with_virtuals(
                    &mut transcript,
                    &fx.pp,
                    &proof.commitment,
                    &proof.open,
                    &subclaim,
                    &blake3_f2_bit_op_virtuals(),
                )
                .expect("open verify should succeed");
                black_box(());
            },
            criterion::BatchSize::PerIteration,
        );
    });
}

// ---------------------------------------------------------------------------
// MICRO breakdown: timing each sub-step inside prove_f2_uair_with_groups.
// ---------------------------------------------------------------------------

/// Project a Blake3-F_2 UAIR scalar to its 64-bit `F_2[X]` bit-pack —
/// the closure the F_2-native IC adapter builds inside
/// `prove_f2_uair_with_groups`. Replicated here so the micro bench
/// can drive `F2NativeIc::prove_combined` directly.
fn blake3_f2_scalar_to_bits(s: &DensePolynomial<R, D>) -> u64 {
    use crypto_primitives::PrimeField;
    let projected = blake3_f2_project_scalar::<R>(s);
    let mut bits: u64 = 0;
    for (i, c) in projected.coeffs.iter().enumerate() {
        if i >= 64 {
            break;
        }
        if !<BinaryFieldGF128 as PrimeField>::is_zero(c) {
            bits |= 1u64 << i;
        }
    }
    bits
}

fn bench_micro_prover_uair(
    group: &mut BenchmarkGroup<WallTime>,
    id: &str,
    fx: &ProverFixture,
) {
    use zinc_poly::univariate::binary_gf128::{
        alpha_powers, eval_f2_poly_d_at_with_powers,
    };
    use zinc_protocol::f2_native_ic::F2NativeIc;

    let num_constraints = count_constraints::<U>();
    let field_cfg = ();

    // ---- 0a) Commit-Pair: just the trace-pairing pre-step ----
    group.bench_function(BenchmarkId::new("Commit-Pair", id), |bench| {
        bench.iter(|| {
            let paired = zinc_protocol::f2_prove::pair_primary_witness_polys_pub::<D>(
                &fx.trace.binary_poly,
                zinc_test_uair::blake3_f2::cols::NUM_BIN_PUB,
                &blake3_f2_bit_op_virtuals(),
            );
            black_box(paired);
        });
    });

    // ---- 0b) Commit-Encode: pair + encode_rows for every paired MLE (parallel) ----
    group.bench_function(BenchmarkId::new("Commit-Encode", id), |bench| {
        use rayon::prelude::*;
        use zip_plus::pcs::structs::ZipPlus;
        type BinZt = <BenchF2Types<D> as F2ZincTypes<D>>::BinaryZt;
        type BinLc = <BenchF2Types<D> as F2ZincTypes<D>>::BinaryLc;
        let paired = zinc_protocol::f2_prove::pair_primary_witness_polys_pub::<D>(
            &fx.trace.binary_poly,
            zinc_test_uair::blake3_f2::cols::NUM_BIN_PUB,
            &blake3_f2_bit_op_virtuals(),
        );
        bench.iter(|| {
            let cw_matrices: Vec<_> = paired
                .par_iter()
                .map(|poly| ZipPlus::<BinZt, BinLc>::encode_rows(&fx.pp, poly))
                .collect();
            black_box(cw_matrices);
        });
    });

    // ---- 0c) Commit-Fused: encode + fused leaf-hash from cw_matrices ----
    group.bench_function(BenchmarkId::new("Commit-Fused", id), |bench| {
        use rayon::prelude::*;
        use zinc_protocol::f2_prove::LEAF_GROUP_SIZE;
        use zip_plus::merkle::MerkleTree;
        use zip_plus::pcs::structs::ZipPlus;
        type BinZt = <BenchF2Types<D> as F2ZincTypes<D>>::BinaryZt;
        type BinLc = <BenchF2Types<D> as F2ZincTypes<D>>::BinaryLc;
        let paired = zinc_protocol::f2_prove::pair_primary_witness_polys_pub::<D>(
            &fx.trace.binary_poly,
            zinc_test_uair::blake3_f2::cols::NUM_BIN_PUB,
            &blake3_f2_bit_op_virtuals(),
        );
        let codeword_len = fx.pp.linear_code.codeword_len();
        let num_rows = fx.pp.num_rows;
        bench.iter(|| {
            let cw_matrices: Vec<_> = paired
                .par_iter()
                .map(|poly| ZipPlus::<BinZt, BinLc>::encode_rows(&fx.pp, poly))
                .collect();
            let mt = MerkleTree::new_from_row_major_grouped(
                &cw_matrices,
                num_rows,
                codeword_len,
                LEAF_GROUP_SIZE,
            );
            black_box((cw_matrices, mt));
        });
    });

    // ---- 0d) Commit-Fused-GPU (macOS + Metal only) ----
    #[cfg(all(feature = "metal_gpu", target_os = "macos"))]
    group.bench_function(BenchmarkId::new("Commit-Fused-GPU", id), |bench| {
        use rayon::prelude::*;
        use zinc_protocol::f2_prove::LEAF_GROUP_SIZE;
        use zip_plus::merkle::MerkleTree;
        use zip_plus::pcs::structs::ZipPlus;
        type BinZt = <BenchF2Types<D> as F2ZincTypes<D>>::BinaryZt;
        type BinLc = <BenchF2Types<D> as F2ZincTypes<D>>::BinaryLc;
        let paired = zinc_protocol::f2_prove::pair_primary_witness_polys_pub::<D>(
            &fx.trace.binary_poly,
            zinc_test_uair::blake3_f2::cols::NUM_BIN_PUB,
            &blake3_f2_bit_op_virtuals(),
        );
        let codeword_len = fx.pp.linear_code.codeword_len();
        let num_rows = fx.pp.num_rows;
        bench.iter(|| {
            let cw_matrices: Vec<_> = paired
                .par_iter()
                .map(|poly| ZipPlus::<BinZt, BinLc>::encode_rows(&fx.pp, poly))
                .collect();
            let mt = MerkleTree::new_from_row_major_grouped_gpu(
                &cw_matrices,
                num_rows,
                codeword_len,
                LEAF_GROUP_SIZE,
            );
            black_box((cw_matrices, mt));
        });
    });

    // ---- 0) Commit (pre-paired entry point) ----
    let num_pub_bin = zinc_test_uair::blake3_f2::cols::NUM_BIN_PUB;
    group.bench_function(BenchmarkId::new("Commit", id), |bench| {
        bench.iter(|| {
            let mut transcript = Blake3Transcript::new();
            let (hint, comm) = ZincPlusPiopF2::<BenchF2Types<D>, U, D>
                ::commit_and_absorb_pre_paired_witness(
                    &mut transcript,
                    &fx.pp,
                    &fx.paired_primary_witness,
                    &fx.trace.binary_poly[..num_pub_bin],
                )
                .expect("commit");
            black_box((hint, comm));
        });
    });

    // ---- a) F_2-native IC ----
    //
    // Blake3 F_2 is degree-2 in trace MLEs (selector × residual on
    // both rotation pins and LSB checks), so the real prover routes
    // through `prove_combined` (row-major). We bench the same lane
    // here so micro and end-to-end measurements stay aligned.
    group.bench_function(BenchmarkId::new("UAIR-a-F2NativeIC", id), |bench| {
        bench.iter_batched(
            || {
                let mut transcript = Blake3Transcript::new();
                let _ = ZincPlusPiopF2::<BenchF2Types<D>, U, D>
                    ::commit_and_absorb_f2_trace_with_virtuals(
                        &mut transcript, &fx.pp, &fx.trace.binary_poly,
                        &blake3_f2_bit_op_virtuals(),
                    )
                    .expect("commit");
                transcript
            },
            |mut transcript| {
                let (proof, state) = F2NativeIc::<U>::prove_combined::<BinaryFieldGF128, _, D>(
                    &mut transcript,
                    &fx.trace.binary_poly,
                    num_constraints,
                    fx.num_vars,
                    &field_cfg,
                    blake3_f2_scalar_to_bits,
                );
                black_box((proof, state));
            },
            criterion::BatchSize::PerIteration,
        );
    });

    // ---- b) AlphaProject (precomputed α-powers) ----
    group.bench_function(BenchmarkId::new("UAIR-b-AlphaProject", id), |bench| {
        bench.iter_batched(
            || {
                let mut transcript = Blake3Transcript::new();
                let alpha: BinaryFieldGF128 =
                    transcript.get_field_challenge(&field_cfg);
                alpha
            },
            |alpha| {
                let pows = alpha_powers(&alpha, D);
                let projected: Vec<DenseMultilinearExtension<BinaryFieldGF128>> = fx
                    .trace
                    .binary_poly
                    .iter()
                    .map(|col| {
                        let evals_at_alpha: Vec<BinaryFieldGF128> = col
                            .evaluations
                            .iter()
                            .map(|cell| {
                                eval_f2_poly_d_at_with_powers::<D>(cell, &pows)
                            })
                            .collect();
                        DenseMultilinearExtension::from_evaluations_vec(
                            col.num_vars,
                            evals_at_alpha,
                            BinaryFieldGF128::zero(),
                        )
                    })
                    .collect();
                black_box(projected);
            },
            criterion::BatchSize::PerIteration,
        );
    });

    // ---- c) Sumcheck (γ-batched + round-1 fast path) ----
    group.bench_function(BenchmarkId::new("UAIR-c-Sumcheck", id), |bench| {
        use zinc_piop::sumcheck::multi_degree::MultiDegreeSumcheckGroup;
        use zinc_protocol::f2_prove::F2EqColRound1FastPath;

        let alpha: BinaryFieldGF128 = {
            let mut t = Blake3Transcript::new();
            t.get_field_challenge(&field_cfg)
        };
        let pows = alpha_powers(&alpha, D);
        let projected: Vec<DenseMultilinearExtension<BinaryFieldGF128>> = fx
            .trace
            .binary_poly
            .iter()
            .map(|col| {
                let evals_at_alpha: Vec<BinaryFieldGF128> = col
                    .evaluations
                    .iter()
                    .map(|cell| eval_f2_poly_d_at_with_powers::<D>(cell, &pows))
                    .collect();
                DenseMultilinearExtension::from_evaluations_vec(
                    col.num_vars,
                    evals_at_alpha,
                    BinaryFieldGF128::zero(),
                )
            })
            .collect();
        let ic_eval_point: Vec<BinaryFieldGF128> = (0..fx.num_vars)
            .map(|i| BinaryFieldGF128::from_with_cfg(i as u64 + 1, &field_cfg))
            .collect();
        let gamma: BinaryFieldGF128 = {
            let mut t = Blake3Transcript::new();
            t.absorb_random_field(
                &alpha,
                &mut vec![
                    0u8;
                    <<BinaryFieldGF128 as Field>::Inner
                        as zinc_transcript::traits::ConstTranscribable>::NUM_BYTES
                ],
            );
            t.get_field_challenge(&field_cfg)
        };

        let zero_inner = *BinaryFieldGF128::zero().inner();
        let num_rows = projected[0].evaluations.len();
        let num_vars = projected[0].num_vars;
        let mut gamma_pows: Vec<BinaryFieldGF128> = Vec::with_capacity(projected.len());
        {
            let mut acc = BinaryFieldGF128::one();
            for _ in 0..projected.len() {
                gamma_pows.push(acc);
                acc *= &gamma;
            }
        }

        bench.iter_batched(
            || {
                let eq_r = zinc_poly::utils::build_eq_x_r_inner(
                    &ic_eval_point,
                    &field_cfg,
                )
                .expect("eq table");
                #[cfg(feature = "parallel")]
                use rayon::prelude::*;
                #[cfg(feature = "parallel")]
                let evals_iter = (0..num_rows).into_par_iter();
                #[cfg(not(feature = "parallel"))]
                let evals_iter = 0..num_rows;
                let weighted_evals: Vec<_> = evals_iter
                    .map(|i| {
                        let mut sum = BinaryFieldGF128::zero();
                        for (g, col) in projected.iter().enumerate() {
                            sum += gamma_pows[g] * col.evaluations[i];
                        }
                        *sum.inner()
                    })
                    .collect();
                let weighted = DenseMultilinearExtension::from_evaluations_vec(
                    num_vars,
                    weighted_evals,
                    zero_inner,
                );
                let fast_path = Box::new(F2EqColRound1FastPath {
                    eq_table: eq_r.evaluations.clone(),
                    weighted_col: weighted.evaluations.clone(),
                    r_0: ic_eval_point[0],
                    num_vars,
                });
                let group = MultiDegreeSumcheckGroup::with_round_1_fast(
                    2,
                    Vec::new(),
                    Box::new(|v: &[BinaryFieldGF128]| v[0] * v[1]),
                    fast_path,
                );
                let transcript = Blake3Transcript::new();
                (vec![group], transcript)
            },
            |(groups, mut transcript)| {
                let (proof, states) =
                    MultiDegreeSumcheck::<BinaryFieldGF128>::prove_as_subprotocol(
                        &mut transcript,
                        groups,
                        fx.num_vars,
                        &field_cfg,
                    );
                black_box((proof, states));
            },
            criterion::BatchSize::PerIteration,
        );
    });

    // ---- d) ColEvalsAtRstar (per-column MLE evals at r*) ----
    group.bench_function(BenchmarkId::new("UAIR-d-ColEvalsAtRstar", id), |bench| {
        use zinc_poly::mle::MultilinearExtensionWithConfig;

        let alpha: BinaryFieldGF128 = {
            let mut t = Blake3Transcript::new();
            t.get_field_challenge(&field_cfg)
        };
        let pows = alpha_powers(&alpha, D);
        let projected: Vec<DenseMultilinearExtension<BinaryFieldGF128>> = fx
            .trace
            .binary_poly
            .iter()
            .map(|col| {
                let evals_at_alpha: Vec<BinaryFieldGF128> = col
                    .evaluations
                    .iter()
                    .map(|cell| eval_f2_poly_d_at_with_powers::<D>(cell, &pows))
                    .collect();
                DenseMultilinearExtension::from_evaluations_vec(
                    col.num_vars,
                    evals_at_alpha,
                    BinaryFieldGF128::zero(),
                )
            })
            .collect();
        let sumcheck_point: Vec<BinaryFieldGF128> = (0..fx.num_vars)
            .map(|i| BinaryFieldGF128::from_with_cfg(i as u64 + 13, &field_cfg))
            .collect();
        let zero_inner = *BinaryFieldGF128::zero().inner();

        bench.iter(|| {
            #[cfg(feature = "parallel")]
            use rayon::prelude::*;
            #[cfg(feature = "parallel")]
            let it = projected.par_iter();
            #[cfg(not(feature = "parallel"))]
            let it = projected.iter();
            let col_evals: Vec<BinaryFieldGF128> = it
                .map(|col| {
                    let inner_mle = DenseMultilinearExtension::from_evaluations_vec(
                        col.num_vars,
                        col.evaluations.iter().map(|x| *x.inner()).collect(),
                        zero_inner,
                    );
                    <DenseMultilinearExtension<_> as MultilinearExtensionWithConfig<
                        BinaryFieldGF128,
                    >>::evaluate_with_config(
                        inner_mle,
                        &sumcheck_point,
                        &field_cfg,
                    )
                    .expect("MLE eval")
                })
                .collect();
            black_box(col_evals);
        });
    });

    // ---- e) UAIR-FULL: the whole prove_f2_uair_with_groups ----
    group.bench_function(BenchmarkId::new("UAIR-FULL", id), |bench| {
        bench.iter_batched(
            || {
                let mut transcript = Blake3Transcript::new();
                let _ = ZincPlusPiopF2::<BenchF2Types<D>, U, D>
                    ::commit_and_absorb_pre_paired_witness(
                        &mut transcript,
                        &fx.pp,
                        &fx.paired_primary_witness,
                        &fx.trace.binary_poly
                            [..zinc_test_uair::blake3_f2::cols::NUM_BIN_PUB],
                    )
                    .expect("commit");
                transcript
            },
            |mut transcript| {
                let (proof, subclaim, _projected_trace) = ZincPlusPiopF2::<BenchF2Types<D>, U, D>
                    ::prove_f2_uair_with_groups(
                        &mut transcript,
                        &fx.trace,
                        &[] as &[F2VirtualBpSpec],
                        &[],
                        &[],
                        &[],
                        &[],
                        fx.num_vars,
                        blake3_f2_project_scalar::<R>,
                    )
                    .expect("UAIR prove should succeed");
                black_box((proof, subclaim));
            },
            criterion::BatchSize::PerIteration,
        );
    });
}

// ---------------------------------------------------------------------------
// MICRO breakdown: timing each sub-step inside `prove_f2_open`.
// ---------------------------------------------------------------------------

fn bench_micro_prover_open(
    group: &mut BenchmarkGroup<WallTime>,
    id: &str,
    fx: &ProverFixture,
) {
    use zinc_poly::univariate::binary_f2_wide::{f2_inner_product, f2_poly_mul};
    use zinc_poly::univariate::binary_gf128::{AlphaPolyBasis, lift_bp_to_f2_poly_1};
    use zinc_protocol::f2_prove::build_lifted_eq_tensor;
    use zinc_poly::univariate::binary_f2_wide::BinaryF2Poly;

    let field_cfg = ();

    let (hint, subclaim) = {
        let mut t = Blake3Transcript::new();
        let (h, _) = ZincPlusPiopF2::<BenchF2Types<D>, U, D>::commit_and_absorb_f2_trace_with_virtuals(
            &mut t,
            &fx.pp,
            &fx.trace.binary_poly,
            &blake3_f2_bit_op_virtuals(),
        )
        .expect("commit");
        let (_proof, sub, _projected_trace) = ZincPlusPiopF2::<BenchF2Types<D>, U, D>
            ::prove_f2_uair_with_groups(
                &mut t,
                &fx.trace,
                &[],
                &[],
                &[],
                &[],
                &[],
                fx.num_vars,
                blake3_f2_project_scalar::<R>,
            )
            .expect("uair");
        (h, sub)
    };
    let alpha = subclaim.alpha;
    let sumcheck_point = subclaim.sumcheck_point.clone();
    let num_rows = fx.pp.num_rows;
    let row_len = fx.pp.linear_code.row_len();
    let num_cols = fx.trace.binary_poly.len();
    let num_open = recommended_num_column_openings(REP);
    let codeword_len = fx.pp.linear_code.codeword_len();

    // ---- a) AlphaBasis ----
    group.bench_function(BenchmarkId::new("Open-a-AlphaBasis", id), |bench| {
        bench.iter(|| {
            let basis = AlphaPolyBasis::new(&alpha);
            black_box(basis);
        });
    });

    // ---- b) LiftedEqTensor ----
    group.bench_function(BenchmarkId::new("Open-b-LiftedEqTensor", id), |bench| {
        let basis = AlphaPolyBasis::new(&alpha);
        bench.iter(|| {
            let (q0, q1) = build_lifted_eq_tensor(num_rows, &sumcheck_point, &basis);
            black_box((q0, q1));
        });
    });

    let basis = AlphaPolyBasis::new(&alpha);
    let (q0, q1) = build_lifted_eq_tensor(num_rows, &sumcheck_point, &basis);

    let gamma: Vec<BinaryF2Poly<2>> = {
        let mut t = Blake3Transcript::new();
        let g: Vec<BinaryFieldGF128> = t.get_field_challenges(num_cols, &field_cfg);
        g.iter().map(|x| basis.lift(x)).collect()
    };
    let coeffs: Vec<BinaryF2Poly<2>> = {
        let mut t = Blake3Transcript::new();
        let c: Vec<BinaryFieldGF128> = t.get_field_challenges(num_rows, &field_cfg);
        c.iter().map(|x| basis.lift(x)).collect()
    };

    // ---- c) Folds — per-column γ-folds (parallel) ----
    group.bench_function(BenchmarkId::new("Open-c-Folds", id), |bench| {
        bench.iter(|| {
            #[cfg(feature = "parallel")]
            use rayon::prelude::*;
            #[cfg(feature = "parallel")]
            let it = fx.trace.binary_poly.par_iter().enumerate();
            #[cfg(not(feature = "parallel"))]
            let it = fx.trace.binary_poly.iter().enumerate();
            let per_col_results: Vec<(Vec<BinaryF2Poly<5>>, BinaryF2Poly<7>)> = it
                .map(|(g, col)| {
                    let mut b_g_scaled: Vec<BinaryF2Poly<5>> = Vec::with_capacity(num_rows);
                    let mut b_g: Vec<BinaryF2Poly<3>> = Vec::with_capacity(num_rows);
                    for i in 0..num_rows {
                        let row_slice = &col.evaluations[i * row_len..(i + 1) * row_len];
                        let row_lifted: Vec<BinaryF2Poly<1>> =
                            row_slice.iter().map(lift_bp_to_f2_poly_1::<D>).collect();
                        let entry: BinaryF2Poly<3> =
                            f2_inner_product::<1, 2, 3>(&row_lifted, &q1);
                        let scaled: BinaryF2Poly<5> =
                            f2_poly_mul::<2, 3, 5>(&gamma[g], &entry);
                        b_g_scaled.push(scaled);
                        b_g.push(entry);
                    }
                    let a_g_prime: BinaryF2Poly<5> =
                        f2_inner_product::<2, 3, 5>(&q0, &b_g);
                    let a_scaled: BinaryF2Poly<7> =
                        f2_poly_mul::<2, 5, 7>(&gamma[g], &a_g_prime);
                    (b_g_scaled, a_scaled)
                })
                .collect();
            black_box(per_col_results);
        });
    });

    // ---- d) CombinedRow ----
    group.bench_function(BenchmarkId::new("Open-d-CombinedRow", id), |bench| {
        bench.iter(|| {
            #[cfg(feature = "parallel")]
            use rayon::prelude::*;
            #[cfg(feature = "parallel")]
            let it = fx.trace.binary_poly.par_iter().enumerate();
            #[cfg(not(feature = "parallel"))]
            let it = fx.trace.binary_poly.iter().enumerate();
            let per_col: Vec<Vec<BinaryF2Poly<5>>> = it
                .map(|(g, col)| {
                    let mut col_partial: Vec<BinaryF2Poly<3>> =
                        vec![BinaryF2Poly::<3>::zero(); row_len];
                    for i in 0..num_rows {
                        let row_slice =
                            &col.evaluations[i * row_len..(i + 1) * row_len];
                        let coeff_i = &coeffs[i];
                        for j in 0..row_len {
                            let cell = lift_bp_to_f2_poly_1::<D>(&row_slice[j]);
                            let prod: BinaryF2Poly<3> =
                                f2_poly_mul::<1, 2, 3>(&cell, coeff_i);
                            col_partial[j] += prod;
                        }
                    }
                    col_partial
                        .into_iter()
                        .map(|entry| f2_poly_mul::<2, 3, 5>(&gamma[g], &entry))
                        .collect::<Vec<BinaryF2Poly<5>>>()
                })
                .collect();
            black_box(per_col);
        });
    });

    // ---- e) MerkleOpens — sample + path generations ----
    group.bench_function(BenchmarkId::new("Open-e-MerkleOpens", id), |bench| {
        use zinc_protocol::f2_prove::LEAF_GROUP_SIZE;
        bench.iter_batched(
            || Blake3Transcript::new(),
            |mut transcript| {
                let mut paths = Vec::with_capacity(num_open);
                for _ in 0..num_open {
                    let column_idx = zinc_protocol::f2_prove::sample_column_idx(
                        &mut transcript,
                        codeword_len,
                    );
                    let group_idx = column_idx / LEAF_GROUP_SIZE;
                    let merkle_proof = hint
                        .merkle_tree
                        .prove(group_idx)
                        .expect("Merkle prove");
                    paths.push(merkle_proof);
                }
                black_box(paths);
            },
            criterion::BatchSize::PerIteration,
        );
    });

    // ---- f) GammaCoeffsLift ----
    group.bench_function(BenchmarkId::new("Open-f-GammaCoeffsLift", id), |bench| {
        bench.iter_batched(
            || Blake3Transcript::new(),
            |mut transcript| {
                let gamma_gf: Vec<BinaryFieldGF128> =
                    transcript.get_field_challenges(num_cols, &field_cfg);
                let gamma_lifted: Vec<BinaryF2Poly<2>> =
                    gamma_gf.iter().map(|g| basis.lift(g)).collect();
                let coeffs_gf: Vec<BinaryFieldGF128> =
                    transcript.get_field_challenges(num_rows, &field_cfg);
                let coeffs_lifted: Vec<BinaryF2Poly<2>> =
                    coeffs_gf.iter().map(|g| basis.lift(g)).collect();
                black_box((gamma_lifted, coeffs_lifted));
            },
            criterion::BatchSize::PerIteration,
        );
    });

    // ---- g) AssembleOpened ----
    group.bench_function(BenchmarkId::new("Open-g-AssembleOpened", id), |bench| {
        let indices: Vec<usize> = {
            let mut t = Blake3Transcript::new();
            (0..num_open)
                .map(|_| zinc_protocol::f2_prove::sample_column_idx(&mut t, codeword_len))
                .collect()
        };
        use zinc_protocol::f2_prove::LEAF_GROUP_SIZE;
        let batch = hint.cw_matrices.len();
        let single_col_len = batch * fx.pp.num_rows;
        let num_rows = fx.pp.num_rows;
        let cw_codeword_len = fx.pp.linear_code.codeword_len();
        bench.iter(|| {
            #[cfg(feature = "parallel")]
            use rayon::prelude::*;
            #[cfg(feature = "parallel")]
            let it = indices.par_iter();
            #[cfg(not(feature = "parallel"))]
            let it = indices.iter();
            let all: Vec<(Vec<BinaryPoly<64>>, _)> = it
                .map(|&column_idx| {
                    let group_idx = column_idx / LEAF_GROUP_SIZE;
                    let group_start = group_idx * LEAF_GROUP_SIZE;
                    let total = LEAF_GROUP_SIZE * single_col_len;
                    let mut group_values: Vec<std::mem::MaybeUninit<BinaryPoly<64>>> =
                        Vec::with_capacity(total);
                    unsafe { group_values.set_len(total); }
                    for m in 0..batch {
                        let m_offset = m * num_rows;
                        let mat_data = &hint.cw_matrices[m].data;
                        for r in 0..num_rows {
                            let row_off = r * cw_codeword_len + group_start;
                            let cells = &mat_data[row_off..row_off + LEAF_GROUP_SIZE];
                            for (l, cell) in cells.iter().enumerate() {
                                let position = l * single_col_len + m_offset + r;
                                unsafe {
                                    group_values
                                        .get_unchecked_mut(position)
                                        .write(cell.clone());
                                }
                            }
                        }
                    }
                    let group_values: Vec<BinaryPoly<64>> = {
                        let mut v = core::mem::ManuallyDrop::new(group_values);
                        let (ptr, len, cap) = (v.as_mut_ptr(), v.len(), v.capacity());
                        unsafe {
                            Vec::from_raw_parts(ptr.cast::<BinaryPoly<64>>(), len, cap)
                        }
                    };
                    let merkle_proof = hint
                        .merkle_tree
                        .prove(group_idx)
                        .expect("Merkle prove");
                    (group_values, merkle_proof)
                })
                .collect();
            black_box(all);
        });
    });

    // ---- h) Open-FULL: the whole prove_f2_open ----
    group.bench_function(BenchmarkId::new("Open-FULL", id), |bench| {
        bench.iter_batched(
            || {
                let mut t = Blake3Transcript::new();
                let (h, _) = ZincPlusPiopF2::<BenchF2Types<D>, U, D>
                    ::commit_and_absorb_pre_paired_witness(
                        &mut t,
                        &fx.pp,
                        &fx.paired_primary_witness,
                        &fx.trace.binary_poly
                            [..zinc_test_uair::blake3_f2::cols::NUM_BIN_PUB],
                    )
                    .expect("commit");
                let (_proof, sub, _projected_trace) = ZincPlusPiopF2::<BenchF2Types<D>, U, D>
                    ::prove_f2_uair_with_groups(
                        &mut t,
                        &fx.trace,
                        &[],
                        &[],
                        &[],
                        &[],
                        &[],
                        fx.num_vars,
                        blake3_f2_project_scalar::<R>,
                    )
                    .expect("UAIR");
                (h, sub, t)
            },
            |(hint, subclaim, mut transcript)| {
                let open_proof = ZincPlusPiopF2::<BenchF2Types<D>, U, D>::prove_f2_open(
                    &mut transcript,
                    &fx.pp,
                    &hint,
                    &fx.trace.binary_poly[zinc_test_uair::blake3_f2::cols::NUM_BIN_PUB..],
                    &subclaim.sumcheck_point,
                    &subclaim.alpha,
                    recommended_num_column_openings(REP),
                );
                black_box(open_proof);
            },
            criterion::BatchSize::PerIteration,
        );
    });
}

// ---------------------------------------------------------------------------
// MICRO breakdown: timing each sub-step inside `verify_f2_uair`.
// ---------------------------------------------------------------------------

fn bench_micro_verifier_uair(
    group: &mut BenchmarkGroup<WallTime>,
    id: &str,
    fx: &ProverFixture,
) {
    let proof = {
        let mut transcript = Blake3Transcript::new();
        ZincPlusPiopF2::<BenchF2Types<D>, U, D>::prove_f2_full_with_bit_ops(
            &mut transcript,
            &fx.pp,
            &fx.trace,
            &[],
            &blake3_f2_bit_op_virtuals(),
            fx.num_vars,
            blake3_f2_project_scalar::<R>,
            recommended_num_column_openings(REP),
        )
        .expect("prove")
    };

    let num_constraints = count_constraints::<U>();
    let field_cfg = ();

    group.bench_function(BenchmarkId::new("VerifyUAIR-a-IC", id), |bench| {
        bench.iter_batched(
            || {
                let mut t = Blake3Transcript::new();
                ZincPlusPiopF2::<BenchF2Types<D>, U, D>::absorb_commitment(&mut t, &proof.commitment);
                ZincPlusPiopF2::<BenchF2Types<D>, U, D>::absorb_public_binary_cols(
                    &mut t,
                    &fx.trace.binary_poly[..zinc_test_uair::blake3_f2::cols::NUM_BIN_PUB],
                );
                t
            },
            |mut transcript| {
                let ic_subclaim = <U as IdealCheckProtocol>
                    ::verify_as_subprotocol::<_, Blake3F2Ideal, _>(
                        &mut transcript,
                        proof.uair.ic_proof.clone(),
                        num_constraints,
                        fx.num_vars,
                        |ideal: &IdealOrZero<Blake3F2Ideal>| blake3_f2_project_ideal(ideal),
                        &field_cfg,
                    )
                    .expect("IC verify");
                black_box(ic_subclaim);
            },
            criterion::BatchSize::PerIteration,
        );
    });

    group.bench_function(BenchmarkId::new("VerifyUAIR-b-Sumcheck", id), |bench| {
        bench.iter_batched(
            || {
                let mut t = Blake3Transcript::new();
                ZincPlusPiopF2::<BenchF2Types<D>, U, D>::absorb_commitment(&mut t, &proof.commitment);
                ZincPlusPiopF2::<BenchF2Types<D>, U, D>::absorb_public_binary_cols(
                    &mut t,
                    &fx.trace.binary_poly[..zinc_test_uair::blake3_f2::cols::NUM_BIN_PUB],
                );
                let _ = <U as IdealCheckProtocol>
                    ::verify_as_subprotocol::<_, Blake3F2Ideal, _>(
                        &mut t,
                        proof.uair.ic_proof.clone(),
                        num_constraints,
                        fx.num_vars,
                        |ideal: &IdealOrZero<Blake3F2Ideal>| blake3_f2_project_ideal(ideal),
                        &field_cfg,
                    )
                    .expect("IC verify");
                let _: BinaryFieldGF128 = t.get_field_challenge(&field_cfg);
                let _: BinaryFieldGF128 = t.get_field_challenge(&field_cfg);
                t
            },
            |mut transcript| {
                let sc = MultiDegreeSumcheck::<BinaryFieldGF128>::verify_as_subprotocol(
                    &mut transcript,
                    fx.num_vars,
                    &proof.uair.sumcheck_proof,
                    &field_cfg,
                )
                .expect("sumcheck verify");
                black_box(sc);
            },
            criterion::BatchSize::PerIteration,
        );
    });

    group.bench_function(BenchmarkId::new("VerifyUAIR-c-BatchedCheck", id), |bench| {
        let (ic_eval_point, sumcheck_point, expected) = {
            let mut t = Blake3Transcript::new();
            ZincPlusPiopF2::<BenchF2Types<D>, U, D>::absorb_commitment(&mut t, &proof.commitment);
            ZincPlusPiopF2::<BenchF2Types<D>, U, D>::absorb_public_binary_cols(
                &mut t,
                &fx.trace.binary_poly[..zinc_test_uair::blake3_f2::cols::NUM_BIN_PUB],
            );
            let ic_subclaim = <U as IdealCheckProtocol>
                ::verify_as_subprotocol::<_, Blake3F2Ideal, _>(
                    &mut t,
                    proof.uair.ic_proof.clone(),
                    num_constraints,
                    fx.num_vars,
                    |ideal: &IdealOrZero<Blake3F2Ideal>| blake3_f2_project_ideal(ideal),
                    &field_cfg,
                )
                .expect("IC");
            let _: BinaryFieldGF128 = t.get_field_challenge(&field_cfg);
            let _: BinaryFieldGF128 = t.get_field_challenge(&field_cfg);
            let md = MultiDegreeSumcheck::<BinaryFieldGF128>::verify_as_subprotocol(
                &mut t,
                fx.num_vars,
                &proof.uair.sumcheck_proof,
                &field_cfg,
            )
            .expect("sumcheck");
            (
                ic_subclaim.evaluation_point,
                md.point().to_vec(),
                md.expected_evaluations()[0],
            )
        };
        let gamma = proof.uair.gamma;
        let evals = proof.uair.column_evals_at_rstar.clone();

        bench.iter(|| {
            let one = BinaryFieldGF128::one();
            let eq_at_rstar_r =
                zinc_poly::utils::eq_eval(&sumcheck_point, &ic_eval_point, one).unwrap();
            let mut batched = BinaryFieldGF128::zero();
            for v in evals.iter().rev() {
                batched = batched * gamma + *v;
            }
            let got = eq_at_rstar_r * batched;
            black_box((got, expected));
        });
    });
}

// ---------------------------------------------------------------------------
// Criterion entry points.
// ---------------------------------------------------------------------------

/// `num_vars` values the e2e bench sweeps — mirrors `f2_sha256.rs`'s
/// sweep for apples-to-apples comparison. Blake3's `MIN_NUM_VARS` is
/// 6 (one compression fits in 2^6 = 64 rows), so 9 is comfortably
/// above the minimum (2^9 = 512 ⊇ floor((512-4)/60) = 8 chained
/// compressions).
const NVARS_SWEEP: &[usize] = &[9, 16, 20, 21, 22];

fn e2e_benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zinc+ F_2 Blake3");
    group.sample_size(10);

    for &num_vars in NVARS_SWEEP {
        let fx = setup_prover(num_vars);
        let id = format!("nvars={num_vars}");

        group.bench_function(BenchmarkId::new("WitnessGen", &id), |bench| {
            let mut rng_local = rng();
            bench.iter(|| {
                black_box(U::generate_random_trace(num_vars, &mut rng_local));
            });
        });

        group.bench_function(BenchmarkId::new("Prove", &id), |bench| {
            bench.iter(|| {
                let mut transcript = Blake3Transcript::new();
                let proof = ZincPlusPiopF2::<BenchF2Types<D>, U, D>
                    ::prove_f2_full_pre_paired_with_bit_ops(
                        &mut transcript,
                        &fx.pp,
                        &fx.trace,
                        &fx.paired_primary_witness,
                        &[],
                        &blake3_f2_bit_op_virtuals(),
                        num_vars,
                        blake3_f2_project_scalar::<R>,
                        recommended_num_column_openings(REP),
                    )
                    .expect("prove_f2_full should succeed");
                black_box(proof);
            });
        });

        let proof = {
            let mut transcript = Blake3Transcript::new();
            ZincPlusPiopF2::<BenchF2Types<D>, U, D>::prove_f2_full_pre_paired_with_bit_ops(
                &mut transcript,
                &fx.pp,
                &fx.trace,
                &fx.paired_primary_witness,
                &[],
                &blake3_f2_bit_op_virtuals(),
                num_vars,
                blake3_f2_project_scalar::<R>,
                recommended_num_column_openings(REP),
            )
            .expect("prove for verifier bench should succeed")
        };

        eprint_f2_proof_size(&id, &proof);

        group.bench_function(BenchmarkId::new("Verify", &id), |bench| {
            bench.iter(|| {
                let mut transcript = Blake3Transcript::new();
                let subclaim = ZincPlusPiopF2::<BenchF2Types<D>, U, D>::verify_f2_full_with_bit_ops(
                    &mut transcript,
                    &fx.pp,
                    &proof,
                    &[],
                    &blake3_f2_bit_op_virtuals(),
                    &fx.trace.binary_poly[..zinc_test_uair::blake3_f2::cols::NUM_BIN_PUB],
                    num_vars,
                    fx.num_primary,
                    |ideal: &IdealOrZero<Blake3F2Ideal>| blake3_f2_project_ideal(ideal),
                )
                .expect("verify_f2_full should succeed");
                black_box(subclaim);
            });
        });
    }

    group.finish();
}

fn step_benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zinc+ F_2 Blake3 Steps");
    let fx = setup_prover(9);
    let id = format!("nvars={}", fx.num_vars);
    bench_prover_steps(&mut group, &id, &fx);
    bench_verifier_steps(&mut group, &id, &fx);
    group.finish();
}

fn micro_benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zinc+ F_2 Blake3 Micro");
    let fx = setup_prover(22);
    let id = format!("nvars={}", fx.num_vars);
    bench_micro_prover_uair(&mut group, &id, &fx);
    bench_micro_prover_open(&mut group, &id, &fx);
    bench_micro_verifier_uair(&mut group, &id, &fx);
    group.finish();
}

criterion_group! {
    name = e2e;
    config = Criterion::default().sample_size(10);
    targets = e2e_benches
}
criterion_group! {
    name = steps;
    config = Criterion::default().sample_size(10);
    targets = step_benches
}
criterion_group! {
    name = micro;
    config = Criterion::default().sample_size(10);
    targets = micro_benches
}
criterion_main!(e2e, steps, micro);
