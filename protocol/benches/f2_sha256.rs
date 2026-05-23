//! Criterion benchmark for the SHA-256 `F_2[X]` UAIR
//! ([`zinc_test_uair::Sha256F2Uair`]) routed through
//! [`ZincPlusPiopF2`].
//!
//! Mirrors the integer-pipeline `bench_real_sha256_e2e` in
//! `benches/e2e.rs` but commits all witness columns as bit-polynomials
//! in `F_2[X]` (no overflow witnesses, no booleanity sumcheck).
//!
//! Three bench groups are reported:
//!
//! 1. `Zinc+ F_2 SHA-256` — end-to-end (WitnessGen / Prove / Verify).
//! 2. `Zinc+ F_2 SHA-256 Steps` — top-level prover and verifier step
//!    breakdown (Commit / UAIR / Open).
//! 3. `Zinc+ F_2 SHA-256 Micro` — fine-grained timings *inside* the
//!    UAIR and Open phases. Identifies the hot paths inside
//!    `prove_f2_uair_with_groups`, `verify_f2_uair`, and
//!    `verify_f2_open` by replicating their inner sub-steps and
//!    timing each.
//!
//! Trace shape: 7 chained compressions, `num_vars = 9` → 2^9 = 512
//! rows (480 active + 32 slack).
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
use crypto_primitives::{FromWithConfig, crypto_bigint_int::Int, crypto_bigint_uint::Uint};
use rand::rng;
use zinc_poly::{
    mle::DenseMultilinearExtension,
    univariate::{
        binary::{BinaryPoly, BinaryPolyInnerProduct},
        binary_gf192::BinaryFieldGF192,
        dense::{DensePolyInnerProduct, DensePolynomial},
    },
};
use zinc_primality::MillerRabin;
use zinc_protocol::f2_prove::{
    F2FullProof, F2VirtualBpSpec, F2ZincTypes, ZincPlusPiopF2, eq_dot_column_groups,
    extract_column_evals_eq_dot_col,
};
use zinc_test_uair::{
    GenerateRandomTrace, Sha256F2Ideal, Sha256F2Uair, sha256_f2_project_ideal,
    sha256_f2_project_scalar,
};
use zinc_transcript::{Blake3Transcript, traits::Transcript};
use zinc_uair::{ideal_collector::IdealOrZero, Uair, constraint_counter::count_constraints};
use zinc_utils::inner_product::MBSInnerProduct;
use zip_plus::{
    code::{
        LinearCode,
        raa::RaaConfig,
        raa_f2::{RaaF2Code, recommended_num_column_openings},
    },
    pcs::structs::{ZipPlusParams, ZipTypes},
};

use zinc_piop::{
    ideal_check::IdealCheckProtocol,
    sumcheck::multi_degree::MultiDegreeSumcheck,
};

// ---------------------------------------------------------------------------
// F_2 ZipTypes / F2ZincTypes infrastructure.
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
    type BinaryZt = BenchF2ZipTypes<D>;
    type BinaryLc = RaaF2Code<Self::BinaryZt, BenchRaaConfig, REP>;
}

// ---------------------------------------------------------------------------
// Bench setup.
// ---------------------------------------------------------------------------

type R = Int<4>;
type U = Sha256F2Uair<R>;
const D: usize = 32;

// ---------------------------------------------------------------------------
// Proof-size measurement.
//
// Serialises an `F2FullProof<D>` into per-region byte buffers
// (commitment + uair sub-parts + open sub-parts) and hands them to
// `zip_plus::utils::eprint_bytes_size_breakdown`, which prints a
// per-region table with raw + zstd-3 columns + percentages.
// ---------------------------------------------------------------------------

/// Build the per-region byte serialisation of an `F2FullProof<D>`,
/// returning one entry per component. The serialisation uses each
/// component's natural wire format (no length prefixes or versioning)
/// since the bench only needs apples-to-apples size accounting.
///
/// The `opened_columns` block — by far the dominant cost — is split
/// into three sub-regions so the breakdown reveals where the bytes
/// actually go: codeword cells, Merkle siblings, and per-opening
/// headers (column index + leaf index/count).
#[allow(clippy::arithmetic_side_effects)]
fn f2_full_proof_parts(proof: &F2FullProof<D>) -> Vec<(&'static str, Vec<u8>)> {
    use crypto_primitives::Field;
    use zinc_poly::univariate::F2PackU64;
    use zinc_transcript::traits::{ConstTranscribable, GenTranscribable, Transcribable};

    // -- commitment.root (32-byte Blake3 Merkle root) --
    let commitment: Vec<u8> = (*proof.commitment.root).to_vec();

    // -- uair.alpha (BinaryFieldGF192::Inner = Uint<3>, 24 bytes) --
    const ALPHA_BYTES: usize =
        <<BinaryFieldGF192 as Field>::Inner as ConstTranscribable>::NUM_BYTES;
    let mut alpha = vec![0u8; ALPHA_BYTES];
    proof.uair.alpha.inner().write_transcription_bytes_exact(&mut alpha);

    // -- uair.ic_proof (Transcribable: combined_mle_values) --
    let mut ic_proof = vec![0u8; proof.uair.ic_proof.get_num_bytes()];
    proof.uair.ic_proof.write_transcription_bytes_exact(&mut ic_proof);

    // -- uair.sumcheck_proof (Transcribable) --
    let mut sumcheck = vec![0u8; proof.uair.sumcheck_proof.get_num_bytes()];
    proof.uair.sumcheck_proof.write_transcription_bytes_exact(&mut sumcheck);

    // -- open.lifted_claim (BinaryF2Poly<10> = 80 bytes) --
    let mut lifted_claim = Vec::with_capacity(80);
    for w in proof.open.lifted_claim.words() {
        lifted_claim.extend_from_slice(&w.to_le_bytes());
    }

    // -- open.b_vector (Vec<BinaryF2Poly<7>>, 7 × u64 = 56 bytes each) --
    let mut b_vector = Vec::with_capacity(proof.open.b_vector.len() * 56);
    for v in &proof.open.b_vector {
        for w in v.words() {
            b_vector.extend_from_slice(&w.to_le_bytes());
        }
    }

    // -- open.combined_row (Vec<BinaryF2Poly<7>>) --
    let mut combined_row = Vec::with_capacity(proof.open.combined_row.len() * 56);
    for v in &proof.open.combined_row {
        for w in v.words() {
            combined_row.extend_from_slice(&w.to_le_bytes());
        }
    }

    // -- open.opened_columns, split into three sub-regions --
    //
    // For the deployed RAA-1/4 + 7-compression SHA F_2 shape this
    // block is ~99% of the raw proof, so splitting it reveals which
    // sub-region zstd compresses how much. `values` (raw cell bits)
    // is sparse for random inputs and compresses heavily; `merkle`
    // is essentially random Blake3 hashes (incompressible);
    // `headers` is small.
    let bytes_per_cell = D.div_ceil(8);
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
        ("uair.ic_proof", ic_proof),
        ("uair.sumcheck", sumcheck),
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
}

fn setup_prover(num_vars: usize) -> ProverFixture {
    let mut rng_local = rng();
    let row_len: usize = 32;
    let poly_size = 1usize << num_vars;
    let num_rows = poly_size / row_len;
    assert_eq!(num_rows * row_len, poly_size);

    let trace = U::generate_random_trace(num_vars, &mut rng_local);
    let lc = <BenchF2Types<D> as F2ZincTypes<D>>::BinaryLc::new(row_len);
    let pp = ZipPlusParams::new(num_vars, num_rows, lc);

    ProverFixture {
        trace,
        pp,
        num_vars,
        num_primary: zinc_test_uair::sha256_f2::cols::NUM_BIN,
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
                ZincPlusPiopF2::<BenchF2Types<D>, U, D>::commit_and_absorb_f2_trace(
                    &mut transcript,
                    &fx.pp,
                    &fx.trace.binary_poly,
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
                    ZincPlusPiopF2::<BenchF2Types<D>, U, D>::commit_and_absorb_f2_trace(
                        &mut transcript,
                        &fx.pp,
                        &fx.trace.binary_poly,
                    )
                    .expect("commit should succeed");
                transcript
            },
            |mut transcript| {
                let (proof, subclaim) = ZincPlusPiopF2::<BenchF2Types<D>, U, D>
                    ::prove_f2_uair_with_groups(
                        &mut transcript,
                        &fx.trace,
                        &[] as &[F2VirtualBpSpec],
                        fx.num_vars,
                        sha256_f2_project_scalar::<R>,
                        eq_dot_column_groups,
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
                    ZincPlusPiopF2::<BenchF2Types<D>, U, D>::commit_and_absorb_f2_trace(
                        &mut transcript,
                        &fx.pp,
                        &fx.trace.binary_poly,
                    )
                    .expect("commit should succeed");
                let (_uair_proof, subclaim) = ZincPlusPiopF2::<BenchF2Types<D>, U, D>
                    ::prove_f2_uair_with_groups(
                        &mut transcript,
                        &fx.trace,
                        &[],
                        fx.num_vars,
                        sha256_f2_project_scalar::<R>,
                        eq_dot_column_groups,
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
        ZincPlusPiopF2::<BenchF2Types<D>, U, D>::prove_f2_full(
            &mut transcript,
            &fx.pp,
            &fx.trace,
            &[],
            fx.num_vars,
            sha256_f2_project_scalar::<R>,
            recommended_num_column_openings(REP),
        )
        .expect("prove for verifier bench should succeed")
    };

    group.bench_function(BenchmarkId::new("1-VerifyUAIR", id), |bench| {
        bench.iter_batched(
            || {
                let mut transcript = Blake3Transcript::new();
                ZincPlusPiopF2::<BenchF2Types<D>, U, D>::absorb_commitment(
                    &mut transcript,
                    &proof.commitment,
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
                    |ideal: &IdealOrZero<Sha256F2Ideal>| sha256_f2_project_ideal(ideal),
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
                let subclaim =
                    ZincPlusPiopF2::<BenchF2Types<D>, U, D>::verify_f2_uair(
                        &mut transcript,
                        &proof.uair,
                        &[],
                        fx.num_vars,
                        fx.num_primary,
                        |ideal: &IdealOrZero<Sha256F2Ideal>| sha256_f2_project_ideal(ideal),
                    )
                    .expect("UAIR verify should succeed");
                (transcript, subclaim)
            },
            |(mut transcript, subclaim)| {
                ZincPlusPiopF2::<BenchF2Types<D>, U, D>::verify_f2_open(
                    &mut transcript,
                    &fx.pp,
                    &proof.commitment,
                    &proof.open,
                    &subclaim,
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
//
// The five sub-steps replicate the body of `prove_f2_uair_with_groups`
// in `protocol/src/f2_prove.rs`:
//   a. RowMajorProject  — `project_f2_trace_row_major`
//   b. ProjectScalars   — `zinc_piop::projections::project_scalars`
//   c. ICProveCombined  — `<U as IdealCheckProtocol>::prove_combined`
//   d. AlphaProject     — α draw + per-cell eval at α to GF(2^192)
//   e. Sumcheck         — `MultiDegreeSumcheck::prove_as_subprotocol`
// ---------------------------------------------------------------------------

/// Project a SHA-F_2 UAIR scalar to its 64-bit `F_2[X]` bit-pack —
/// the closure the F_2-native IC adapter builds inside
/// `prove_f2_uair_with_groups`. Replicated here so the micro bench
/// can drive `F2NativeIc::prove_combined` directly.
fn sha_f2_scalar_to_bits(
    s: &DensePolynomial<R, D>,
) -> u64 {
    use crypto_primitives::PrimeField;
    let projected = sha256_f2_project_scalar::<R>(s);
    let mut bits: u64 = 0;
    for (i, c) in projected.coeffs.iter().enumerate() {
        if i >= 64 {
            break;
        }
        if !<BinaryFieldGF192 as PrimeField>::is_zero(c) {
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
    use zinc_poly::univariate::binary_gf192::{
        alpha_powers, eval_f2_poly_d_at_with_powers,
    };
    use zinc_protocol::f2_native_ic::F2NativeIc;

    let num_constraints = count_constraints::<U>();
    let field_cfg = ();

    // ---- 0) Commit (PCS commit + Merkle root + transcript absorb) ----
    //
    // Without this entry the Micro group's prover sum would miss the
    // pre-IC commit cost — `commit_and_absorb_f2_trace` runs the
    // RAA-F_2 codeword encoder over every committed column and folds
    // the Merkle tree. Belongs to the prover side of the protocol
    // even though it lives outside the IC + sumcheck loop.
    group.bench_function(BenchmarkId::new("Commit", id), |bench| {
        bench.iter(|| {
            let mut transcript = Blake3Transcript::new();
            let (hint, comm) = ZincPlusPiopF2::<BenchF2Types<D>, U, D>
                ::commit_and_absorb_f2_trace(
                    &mut transcript, &fx.pp, &fx.trace.binary_poly,
                )
                .expect("commit");
            black_box((hint, comm));
        });
    });

    // ---- a) F_2-native IC ----
    //
    // The current code path: per-row bit-poly arithmetic + per-bit
    // MLE eval at the IC point. Replaces the old
    // RowMajorProject + ProjectScalars + ICProveCombined trio that
    // earlier versions of this bench measured.
    group.bench_function(BenchmarkId::new("UAIR-a-F2NativeIC", id), |bench| {
        bench.iter_batched(
            || {
                let mut transcript = Blake3Transcript::new();
                let _ = ZincPlusPiopF2::<BenchF2Types<D>, U, D>
                    ::commit_and_absorb_f2_trace(
                        &mut transcript, &fx.pp, &fx.trace.binary_poly,
                    )
                    .expect("commit");
                transcript
            },
            |mut transcript| {
                let (proof, state) = F2NativeIc::<U>::prove_combined::<BinaryFieldGF192, _, D>(
                    &mut transcript,
                    &fx.trace.binary_poly,
                    num_constraints,
                    fx.num_vars,
                    &field_cfg,
                    sha_f2_scalar_to_bits,
                );
                black_box((proof, state));
            },
            criterion::BatchSize::PerIteration,
        );
    });

    // ---- b) AlphaProject (precomputed α-powers) ----
    //
    // Matches the current `prove_f2_uair_with_groups` body:
    // compute `α^0..α^{D-1}` once, then per-cell bit-selected XOR-add.
    group.bench_function(BenchmarkId::new("UAIR-b-AlphaProject", id), |bench| {
        bench.iter_batched(
            || {
                let mut transcript = Blake3Transcript::new();
                let alpha: BinaryFieldGF192 =
                    transcript.get_field_challenge(&field_cfg);
                alpha
            },
            |alpha| {
                let pows = alpha_powers(&alpha, D);
                let projected: Vec<DenseMultilinearExtension<BinaryFieldGF192>> = fx
                    .trace
                    .binary_poly
                    .iter()
                    .map(|col| {
                        let evals_at_alpha: Vec<BinaryFieldGF192> = col
                            .evaluations
                            .iter()
                            .map(|cell| {
                                eval_f2_poly_d_at_with_powers::<D>(cell, &pows)
                            })
                            .collect();
                        DenseMultilinearExtension::from_evaluations_vec(
                            col.num_vars,
                            evals_at_alpha,
                            BinaryFieldGF192::zero(),
                        )
                    })
                    .collect();
                black_box(projected);
            },
            criterion::BatchSize::PerIteration,
        );
    });

    // ---- c) Sumcheck ----
    //
    // `MultiDegreeSumcheck::prove_as_subprotocol` on the
    // α-projected trace under the standard `eq · col` groups.
    group.bench_function(BenchmarkId::new("UAIR-c-Sumcheck", id), |bench| {
        let alpha: BinaryFieldGF192 = {
            let mut t = Blake3Transcript::new();
            t.get_field_challenge(&field_cfg)
        };
        let pows = alpha_powers(&alpha, D);
        let projected: Vec<DenseMultilinearExtension<BinaryFieldGF192>> = fx
            .trace
            .binary_poly
            .iter()
            .map(|col| {
                let evals_at_alpha: Vec<BinaryFieldGF192> = col
                    .evaluations
                    .iter()
                    .map(|cell| eval_f2_poly_d_at_with_powers::<D>(cell, &pows))
                    .collect();
                DenseMultilinearExtension::from_evaluations_vec(
                    col.num_vars,
                    evals_at_alpha,
                    BinaryFieldGF192::zero(),
                )
            })
            .collect();
        let ic_eval_point: Vec<BinaryFieldGF192> = (0..fx.num_vars)
            .map(|i| BinaryFieldGF192::from_with_cfg(i as u64 + 1, &field_cfg))
            .collect();

        bench.iter_batched(
            || {
                let groups = eq_dot_column_groups(&ic_eval_point, &projected, &field_cfg);
                let transcript = Blake3Transcript::new();
                (groups, transcript)
            },
            |(groups, mut transcript)| {
                let (proof, states) =
                    MultiDegreeSumcheck::<BinaryFieldGF192>::prove_as_subprotocol(
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
}

// ---------------------------------------------------------------------------
// MICRO breakdown: timing each sub-step inside `prove_f2_open`.
//
// Five sub-steps, in body order:
//   a. AlphaBasis     — `AlphaPolyBasis::new(α)` (192×192 F_2 Gauss–
//                       Jordan).
//   b. LiftedEqTensor — `build_lifted_eq_tensor` for q_0, q_1.
//   c. Folds          — parallel per-column γ-folds producing the
//                       `b_g_scaled` / `a_scaled` partials that merge
//                       into `b'` / `a'`.
//   d. CombinedRow    — parallel per-column `combined_row` build.
//   e. MerkleOpens    — `t` column-index samples + Merkle-path proofs.
// ---------------------------------------------------------------------------

fn bench_micro_prover_open(
    group: &mut BenchmarkGroup<WallTime>,
    id: &str,
    fx: &ProverFixture,
) {
    use zinc_poly::univariate::binary_f2_wide::{f2_inner_product, f2_poly_mul};
    use zinc_poly::univariate::binary_gf192::{AlphaPolyBasis, lift_bp_to_f2_poly_1};
    use zinc_protocol::f2_prove::build_lifted_eq_tensor;
    use zinc_poly::univariate::binary_f2_wide::BinaryF2Poly;

    let field_cfg = ();

    // Set up a (hint, subclaim, alpha) once — these are inputs the
    // open phase takes from the earlier prove steps.
    let (hint, subclaim) = {
        let mut t = Blake3Transcript::new();
        let (h, _) = ZincPlusPiopF2::<BenchF2Types<D>, U, D>::commit_and_absorb_f2_trace(
            &mut t,
            &fx.pp,
            &fx.trace.binary_poly,
        )
        .expect("commit");
        let (_proof, sub) = ZincPlusPiopF2::<BenchF2Types<D>, U, D>
            ::prove_f2_uair_with_groups(
                &mut t,
                &fx.trace,
                &[],
                fx.num_vars,
                sha256_f2_project_scalar::<R>,
                eq_dot_column_groups,
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

    // Precompute basis + (q0, q1) once for the heavier benches below.
    let basis = AlphaPolyBasis::new(&alpha);
    let (q0, q1) = build_lifted_eq_tensor(num_rows, &sumcheck_point, &basis);

    // Build a deterministic γ vector (matching what the real open
    // draws after committing b'/a' — we just need representative
    // dense GF(2^192) values lifted to BinaryF2Poly<3>).
    let gamma: Vec<BinaryF2Poly<3>> = {
        let mut t = Blake3Transcript::new();
        let g: Vec<BinaryFieldGF192> = t.get_field_challenges(num_cols, &field_cfg);
        g.iter().map(|x| basis.lift(x)).collect()
    };
    let coeffs: Vec<BinaryF2Poly<3>> = {
        let mut t = Blake3Transcript::new();
        let c: Vec<BinaryFieldGF192> = t.get_field_challenges(num_rows, &field_cfg);
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
            let per_col_results: Vec<(Vec<BinaryF2Poly<7>>, BinaryF2Poly<10>)> = it
                .map(|(g, col)| {
                    let mut b_g_scaled: Vec<BinaryF2Poly<7>> = Vec::with_capacity(num_rows);
                    let mut b_g: Vec<BinaryF2Poly<4>> = Vec::with_capacity(num_rows);
                    for i in 0..num_rows {
                        let row_slice = &col.evaluations[i * row_len..(i + 1) * row_len];
                        let row_lifted: Vec<BinaryF2Poly<1>> =
                            row_slice.iter().map(lift_bp_to_f2_poly_1::<D>).collect();
                        let entry: BinaryF2Poly<4> =
                            f2_inner_product::<1, 3, 4>(&row_lifted, &q1);
                        let scaled: BinaryF2Poly<7> =
                            f2_poly_mul::<3, 4, 7>(&gamma[g], &entry);
                        b_g_scaled.push(scaled);
                        b_g.push(entry);
                    }
                    let a_g_prime: BinaryF2Poly<7> =
                        f2_inner_product::<3, 4, 7>(&q0, &b_g);
                    let a_scaled: BinaryF2Poly<10> =
                        f2_poly_mul::<3, 7, 10>(&gamma[g], &a_g_prime);
                    (b_g_scaled, a_scaled)
                })
                .collect();
            black_box(per_col_results);
        });
    });

    // ---- d) CombinedRow — per-column combined_row contributions (parallel) ----
    group.bench_function(BenchmarkId::new("Open-d-CombinedRow", id), |bench| {
        bench.iter(|| {
            #[cfg(feature = "parallel")]
            use rayon::prelude::*;
            #[cfg(feature = "parallel")]
            let it = fx.trace.binary_poly.par_iter().enumerate();
            #[cfg(not(feature = "parallel"))]
            let it = fx.trace.binary_poly.iter().enumerate();
            let per_col: Vec<Vec<BinaryF2Poly<7>>> = it
                .map(|(g, col)| {
                    let mut col_contrib: Vec<BinaryF2Poly<7>> = Vec::with_capacity(row_len);
                    for j in 0..row_len {
                        let column_j_lifted: Vec<BinaryF2Poly<1>> = (0..num_rows)
                            .map(|i| lift_bp_to_f2_poly_1::<D>(&col.evaluations[i * row_len + j]))
                            .collect();
                        let per_col_entry: BinaryF2Poly<4> =
                            f2_inner_product::<1, 3, 4>(&column_j_lifted, &coeffs);
                        let scaled: BinaryF2Poly<7> =
                            f2_poly_mul::<3, 4, 7>(&gamma[g], &per_col_entry);
                        col_contrib.push(scaled);
                    }
                    col_contrib
                })
                .collect();
            black_box(per_col);
        });
    });

    // ---- e) MerkleOpens — t = 987 sample + path generations ----
    group.bench_function(BenchmarkId::new("Open-e-MerkleOpens", id), |bench| {
        bench.iter_batched(
            || {
                // Each iteration drives the loop on a fresh transcript
                // so the sampled column indices look like real ones.
                Blake3Transcript::new()
            },
            |mut transcript| {
                let mut paths = Vec::with_capacity(num_open);
                for _ in 0..num_open {
                    let column_idx = zinc_protocol::f2_prove::sample_column_idx(
                        &mut transcript,
                        codeword_len,
                    );
                    let merkle_proof = hint
                        .merkle_tree
                        .prove(column_idx)
                        .expect("Merkle prove");
                    paths.push(merkle_proof);
                }
                black_box(paths);
            },
            criterion::BatchSize::PerIteration,
        );
    });
}

// ---------------------------------------------------------------------------
// MICRO breakdown: timing each sub-step inside `verify_f2_uair`.
//   a. ICVerify      — `<U as IdealCheckProtocol>::verify_as_subprotocol`
//   b. SumcheckVerify — `MultiDegreeSumcheck::verify_as_subprotocol`
//   c. ExtractEval   — `extract_column_evals_eq_dot_col` (eq inv + 41×mul)
// ---------------------------------------------------------------------------

fn bench_micro_verifier_uair(
    group: &mut BenchmarkGroup<WallTime>,
    id: &str,
    fx: &ProverFixture,
) {
    let proof = {
        let mut transcript = Blake3Transcript::new();
        ZincPlusPiopF2::<BenchF2Types<D>, U, D>::prove_f2_full(
            &mut transcript,
            &fx.pp,
            &fx.trace,
            &[],
            fx.num_vars,
            sha256_f2_project_scalar::<R>,
            recommended_num_column_openings(REP),
        )
        .expect("prove")
    };

    let num_constraints = count_constraints::<U>();
    let field_cfg = ();

    // a) IC verify
    group.bench_function(BenchmarkId::new("VerifyUAIR-a-IC", id), |bench| {
        bench.iter_batched(
            || {
                let mut t = Blake3Transcript::new();
                ZincPlusPiopF2::<BenchF2Types<D>, U, D>::absorb_commitment(&mut t, &proof.commitment);
                t
            },
            |mut transcript| {
                let ic_subclaim = <U as IdealCheckProtocol>
                    ::verify_as_subprotocol::<_, Sha256F2Ideal, _>(
                        &mut transcript,
                        proof.uair.ic_proof.clone(),
                        num_constraints,
                        fx.num_vars,
                        |ideal: &IdealOrZero<Sha256F2Ideal>| sha256_f2_project_ideal(ideal),
                        &field_cfg,
                    )
                    .expect("IC verify");
                black_box(ic_subclaim);
            },
            criterion::BatchSize::PerIteration,
        );
    });

    // b) Sumcheck verify
    group.bench_function(BenchmarkId::new("VerifyUAIR-b-Sumcheck", id), |bench| {
        bench.iter_batched(
            || {
                let mut t = Blake3Transcript::new();
                ZincPlusPiopF2::<BenchF2Types<D>, U, D>::absorb_commitment(&mut t, &proof.commitment);
                let _ = <U as IdealCheckProtocol>
                    ::verify_as_subprotocol::<_, Sha256F2Ideal, _>(
                        &mut t,
                        proof.uair.ic_proof.clone(),
                        num_constraints,
                        fx.num_vars,
                        |ideal: &IdealOrZero<Sha256F2Ideal>| sha256_f2_project_ideal(ideal),
                        &field_cfg,
                    )
                    .expect("IC verify");
                let _: BinaryFieldGF192 = t.get_field_challenge(&field_cfg);
                t
            },
            |mut transcript| {
                let sc = MultiDegreeSumcheck::<BinaryFieldGF192>::verify_as_subprotocol(
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

    // c) Extract eval claims (eq invert + per-column division).
    group.bench_function(BenchmarkId::new("VerifyUAIR-c-Extract", id), |bench| {
        // Precompute (ic_eval_point, md_subclaims) outside the loop.
        let (ic_eval_point, md_subclaims) = {
            let mut t = Blake3Transcript::new();
            ZincPlusPiopF2::<BenchF2Types<D>, U, D>::absorb_commitment(&mut t, &proof.commitment);
            let ic_subclaim = <U as IdealCheckProtocol>
                ::verify_as_subprotocol::<_, Sha256F2Ideal, _>(
                    &mut t,
                    proof.uair.ic_proof.clone(),
                    num_constraints,
                    fx.num_vars,
                    |ideal: &IdealOrZero<Sha256F2Ideal>| sha256_f2_project_ideal(ideal),
                    &field_cfg,
                )
                .expect("IC");
            let _: BinaryFieldGF192 = t.get_field_challenge(&field_cfg);
            let md = MultiDegreeSumcheck::<BinaryFieldGF192>::verify_as_subprotocol(
                &mut t,
                fx.num_vars,
                &proof.uair.sumcheck_proof,
                &field_cfg,
            )
            .expect("sumcheck");
            (ic_subclaim.evaluation_point, md)
        };

        bench.iter(|| {
            let evals =
                extract_column_evals_eq_dot_col(&ic_eval_point, &md_subclaims).expect("eq_inv");
            black_box(evals);
        });
    });
}

// ---------------------------------------------------------------------------
// Criterion entry points.
// ---------------------------------------------------------------------------

/// `num_vars` values the e2e bench sweeps: odd from 9 to 21
/// inclusive. 9 is the SHA-256 F_2 UAIR's minimum (480 active rows
/// fit in 2^9 = 512); larger values zero-pad and measure how the
/// prover/verifier scale with the hypercube size.
const NVARS_SWEEP: &[usize] = &[9,16];

fn e2e_benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zinc+ F_2 SHA-256");
    // Large hypercubes (2^21 ≈ 2.1M rows × 41 cols of bit-poly cells +
    // O(num_rows) GF(2^192) eq-table during prove + per-row IC
    // workspace) push the per-iteration time into the multi-second
    // range. Criterion auto-reduces sample count when iter time is
    // long, but cap it explicitly to keep the run from blowing past
    // the wall-clock budget.
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
                let proof = ZincPlusPiopF2::<BenchF2Types<D>, U, D>::prove_f2_full(
                    &mut transcript,
                    &fx.pp,
                    &fx.trace,
                    &[],
                    num_vars,
                    sha256_f2_project_scalar::<R>,
                    recommended_num_column_openings(REP),
                )
                .expect("prove_f2_full should succeed");
                black_box(proof);
            });
        });

        let proof = {
            let mut transcript = Blake3Transcript::new();
            ZincPlusPiopF2::<BenchF2Types<D>, U, D>::prove_f2_full(
                &mut transcript,
                &fx.pp,
                &fx.trace,
                &[],
                num_vars,
                sha256_f2_project_scalar::<R>,
                recommended_num_column_openings(REP),
            )
            .expect("prove for verifier bench should succeed")
        };

        // Report proof size: raw + zstd-3 compressed. Printed once
        // per `nvars`. Criterion captures stdout but lets stderr
        // through, so `eprintln!` shows up next to the timings.
        eprint_f2_proof_size(&id, &proof);

        group.bench_function(BenchmarkId::new("Verify", &id), |bench| {
            bench.iter(|| {
                let mut transcript = Blake3Transcript::new();
                let subclaim = ZincPlusPiopF2::<BenchF2Types<D>, U, D>::verify_f2_full(
                    &mut transcript,
                    &fx.pp,
                    &proof,
                    &[],
                    num_vars,
                    fx.num_primary,
                    |ideal: &IdealOrZero<Sha256F2Ideal>| sha256_f2_project_ideal(ideal),
                )
                .expect("verify_f2_full should succeed");
                black_box(subclaim);
            });
        });
    }

    group.finish();
}

fn step_benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zinc+ F_2 SHA-256 Steps");
    let fx = setup_prover(9);
    let id = format!("nvars={}", fx.num_vars);
    bench_prover_steps(&mut group, &id, &fx);
    bench_verifier_steps(&mut group, &id, &fx);
    group.finish();
}

fn micro_benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zinc+ F_2 SHA-256 Micro");
    let fx = setup_prover(16);
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
