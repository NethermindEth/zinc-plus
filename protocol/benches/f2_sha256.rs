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
use zinc_uair::BitOp;
use zinc_test_uair::{
    GenerateRandomTrace, Sha256F2Ideal, Sha256F2Uair, sha256_f2_project_ideal,
    sha256_f2_project_scalar,
};
use zinc_transcript::{Blake3Transcript, traits::Transcript};
use zinc_uair::{ideal_collector::IdealOrZero, constraint_counter::count_constraints};
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

/// Number of column openings the prover sends to the verifier.
/// **`987`** is the value returned by
/// `recommended_num_column_openings(REP)` for RAA at REP=4 — the
/// soundness-calibrated count under the RAA proximity gap.
///
/// The sibling bench `f2_sha256_rs.rs` keeps the same code path but
/// drops this to `200` to estimate prover/verifier wall-time and
/// proof size under a Reed-Solomon / MDS proximity bound. See that
/// file's module docs for context.
const BENCH_NUM_OPENINGS: usize = 987;

impl<const D: usize> F2ZincTypes<D> for BenchF2Types<D> {
    // The commit-storage cell is `BinaryPoly<64>`: the protocol pairs
    // two width-D trace cells into one packed storage cell to halve
    // the cw_matrices count, transpose memory traffic, and Merkle
    // leaf-hash input bytes.
    type BinaryZt = BenchF2ZipTypes<64>;
    type BinaryLc = RaaF2Code<Self::BinaryZt, BenchRaaConfig, REP>;
}

// ---------------------------------------------------------------------------
// Bench setup.
// ---------------------------------------------------------------------------

type R = Int<4>;
type U = Sha256F2Uair<R>;
const D: usize = 32;

/// Bit-op virtual columns declared by the SHA-256 F_2 UAIR bench. Each
/// entry tells the protocol that the trace column at `col_idx` is the
/// cell-wise `op` of `source_col_idx`'s cells, so the prover skips
/// encoding/Merkle-committing those columns — the verifier
/// reconstructs their codeword cells from the source's opened cells.
///
/// At present only the pure SHR columns are virtualised:
///   `W_SHR3_W  = SHR^3(W_W)`
///   `W_SHR10_W = SHR^10(W_W)`
/// Both shrink the commit-side workload (fewer paired storage
/// matrices, smaller Merkle leaf bytes) without growing the proof.
/// `Σ_0`/`Σ_1`/`σ_0`/`σ_1` are XOR-sums of three rotations each and
/// could be virtualised the same way, but each then needs three
/// `BitOp::Rot` virtuals XOR'd together — left for a follow-up.
fn sha_f2_bit_op_virtuals() -> Vec<F2BitOpVirtualSpec> {
    use zinc_test_uair::sha256_f2::cols;
    vec![
        F2BitOpVirtualSpec {
            col_idx: cols::W_SHR3_W,
            source_col_idx: cols::W_W,
            op: BitOp::ShiftR(3),
        },
        F2BitOpVirtualSpec {
            col_idx: cols::W_SHR10_W,
            source_col_idx: cols::W_W,
            op: BitOp::ShiftR(10),
        },
    ]
}

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

    // -- uair.alpha (BinaryFieldGF128::Inner = Uint<2>, 16 bytes) --
    const ALPHA_BYTES: usize =
        <<BinaryFieldGF128 as Field>::Inner as ConstTranscribable>::NUM_BYTES;
    let mut alpha = vec![0u8; ALPHA_BYTES];
    proof.uair.alpha.inner().write_transcription_bytes_exact(&mut alpha);

    // -- uair.gamma (γ-batching challenge, 16 bytes) --
    let mut gamma = vec![0u8; ALPHA_BYTES];
    proof.uair.gamma.inner().write_transcription_bytes_exact(&mut gamma);

    // -- uair.ic_proof (Transcribable: combined_mle_values) --
    let mut ic_proof = vec![0u8; proof.uair.ic_proof.get_num_bytes()];
    proof.uair.ic_proof.write_transcription_bytes_exact(&mut ic_proof);

    // -- uair.sumcheck_proof (Transcribable) --
    let mut sumcheck = vec![0u8; proof.uair.sumcheck_proof.get_num_bytes()];
    proof.uair.sumcheck_proof.write_transcription_bytes_exact(&mut sumcheck);

    // -- uair.column_evals_at_rstar (per-column GF(2^128) values) --
    let mut col_evals_at_rstar = vec![0u8; proof.uair.column_evals_at_rstar.len() * ALPHA_BYTES];
    for (i, v) in proof.uair.column_evals_at_rstar.iter().enumerate() {
        v.inner().write_transcription_bytes_exact(
            &mut col_evals_at_rstar[i * ALPHA_BYTES..(i + 1) * ALPHA_BYTES],
        );
    }

    // -- multipoint_eval.sumcheck_proof (Transcribable) --
    let mut mp_sumcheck =
        vec![0u8; proof.multipoint_eval.sumcheck_proof.get_num_bytes()];
    proof
        .multipoint_eval
        .sumcheck_proof
        .write_transcription_bytes_exact(&mut mp_sumcheck);

    // -- open_evals_at_r_0 (per-column GF(2^128) values) --
    let mut open_evals_at_r_0 = vec![0u8; proof.open_evals_at_r_0.len() * ALPHA_BYTES];
    for (i, v) in proof.open_evals_at_r_0.iter().enumerate() {
        v.inner().write_transcription_bytes_exact(
            &mut open_evals_at_r_0[i * ALPHA_BYTES..(i + 1) * ALPHA_BYTES],
        );
    }

    // -- open.lifted_claim (BinaryF2Poly<7> = 80 bytes) --
    let mut lifted_claim = Vec::with_capacity(80);
    for w in proof.open.lifted_claim.words() {
        lifted_claim.extend_from_slice(&w.to_le_bytes());
    }

    // -- open.b_vector (Vec<BinaryF2Poly<5>>, 7 × u64 = 56 bytes each) --
    let mut b_vector = Vec::with_capacity(proof.open.b_vector.len() * 56);
    for v in &proof.open.b_vector {
        for w in v.words() {
            b_vector.extend_from_slice(&w.to_le_bytes());
        }
    }

    // -- open.combined_row (Vec<BinaryF2Poly<5>>) --
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
    // Each opened cell holds two paired trace columns (low / high u32
    // halves), so its wire size is `2 · D.div_ceil(8)` bytes — same
    // total bytes as the un-paired layout (`2 · 41/2 · 2048 · 4` =
    // `41 · 2048 · 4`) but with half the cell count per opening.
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

    // -- Hadamard discharge components (all empty unless hadamard_specs
    // were registered). Before the Approach-B discharge these weren't in
    // the breakdown, so a with- vs without-Hadamard size comparison came
    // out byte-identical; count them so the delta is real. The zerocheck
    // `sumcheck` is the bulk; `bit_slice_evals` + the `r*_H` `pair_evals`
    // and trusted `adder_parents` are 16-byte GF(2^128) values each.
    let (had_sumcheck, had_bit_slice_evals) = match &proof.uair.hadamard_proof {
        Some(h) => {
            let mut sc = vec![0u8; h.sumcheck_proof.get_num_bytes()];
            h.sumcheck_proof.write_transcription_bytes_exact(&mut sc);
            let mut bse = vec![0u8; h.bit_slice_evals.len() * ALPHA_BYTES];
            for (chunk, v) in bse.chunks_mut(ALPHA_BYTES).zip(&h.bit_slice_evals) {
                v.inner().write_transcription_bytes_exact(chunk);
            }
            (sc, bse)
        }
        None => (Vec::new(), Vec::new()),
    };
    let mut had_pair_evals = vec![0u8; proof.uair.hadamard_pair_evals.len() * ALPHA_BYTES];
    for (chunk, v) in had_pair_evals
        .chunks_mut(ALPHA_BYTES)
        .zip(&proof.uair.hadamard_pair_evals)
    {
        v.inner().write_transcription_bytes_exact(chunk);
    }
    let mut had_adder_parents =
        vec![0u8; proof.uair.hadamard_adder_parents.len() * ALPHA_BYTES];
    for (chunk, v) in had_adder_parents
        .chunks_mut(ALPHA_BYTES)
        .zip(&proof.uair.hadamard_adder_parents)
    {
        v.inner().write_transcription_bytes_exact(chunk);
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
        ("uair.hadamard/sumcheck", had_sumcheck),
        ("uair.hadamard/bit_slice_evals", had_bit_slice_evals),
        ("uair.hadamard/pair_evals", had_pair_evals),
        ("uair.hadamard/adder_parents", had_adder_parents),
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
    /// The pre-paired primary witness column slice, ready to be fed
    /// to `commit_pre_paired_witness`. Computed once in
    /// `setup_prover` (so pairing is amortised into "witness gen"
    /// rather than the per-iteration commit step) and used by both
    /// the e2e Prove bench and the `Commit` micro bench via the
    /// `prove_f2_full_pre_paired_with_bit_ops` /
    /// `commit_and_absorb_pre_paired_witness` entry points.
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

    // Pair the primary witness cols once, as part of fixture setup —
    // logically this belongs to "witness generation", and amortising
    // it here removes ~50–150 ms (at SHA-256 F_2 nvars=22) of pairing
    // + MLE-clone work from the per-iteration commit step.
    let paired_primary_witness = zinc_protocol::f2_prove::pair_primary_witness_polys_pub::<D>(
        &trace.binary_poly,
        zinc_test_uair::sha256_f2::cols::NUM_BIN_PUB,
        &sha_f2_bit_op_virtuals(),

    );

    ProverFixture {
        trace,
        pp,
        num_vars,
        num_primary: zinc_test_uair::sha256_f2::cols::NUM_BIN,
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
                    &sha_f2_bit_op_virtuals(),

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
                        &sha_f2_bit_op_virtuals(),

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
                        fx.num_vars,
                        sha256_f2_project_scalar::<R>,
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
                        &sha_f2_bit_op_virtuals(),

                    )
                    .expect("commit should succeed");
                let (_uair_proof, subclaim, _projected_trace) = ZincPlusPiopF2::<BenchF2Types<D>, U, D>
                    ::prove_f2_uair_with_groups(
                        &mut transcript,
                        &fx.trace,
                        &[],
                        &[],
                        &[],
                        fx.num_vars,
                        sha256_f2_project_scalar::<R>,
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
                    BENCH_NUM_OPENINGS,
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
            &sha_f2_bit_op_virtuals(),

            fx.num_vars,
            sha256_f2_project_scalar::<R>,
            BENCH_NUM_OPENINGS,
        )
        .expect("prove for verifier bench should succeed")
    };

    let public_binary_cols = &fx.trace.binary_poly[..zinc_test_uair::sha256_f2::cols::NUM_BIN_PUB];
    group.bench_function(BenchmarkId::new("1-VerifyUAIR", id), |bench| {
        bench.iter_batched(
            || {
                let mut transcript = Blake3Transcript::new();
                ZincPlusPiopF2::<BenchF2Types<D>, U, D>::absorb_commitment(
                    &mut transcript,
                    &proof.commitment,
                );
                // Mirror the prover's `commit_and_absorb_f2_trace_with_virtuals`
                // absorbtion order: root, then public cols. Without this
                // the verifier's transcript diverges from the prover's at
                // the first sampled field challenge (AlphaMismatch). Same
                // pattern as the larger-grained benches at line ~1614.
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
                        |ideal: &IdealOrZero<Sha256F2Ideal>| sha256_f2_project_ideal(ideal),
                    )
                    .expect("UAIR verify should succeed");
                // The full proof opens at the multipoint-eval output point
                // `r_0`, not the sumcheck point `r*`. Advance the transcript
                // through the mp phase (mirror `verify_f2_full_with_bit_ops`)
                // so the open verifies at `r_0`. SHA has no XOR-virtual cols,
                // so `open_evals_at_r_0` is exactly the primary col evals.
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
                // Bench's proof is built via `prove_f2_full_with_bit_ops`
                // with `sha_f2_bit_op_virtuals()` (W_SHR3_W, W_SHR10_W
                // skipped from commit, derived from W_W). The verifier
                // mirror needs the same overlay to reconstruct their
                // MLE evals at the open point.
                ZincPlusPiopF2::<BenchF2Types<D>, U, D>::verify_f2_open_with_virtuals(
                    &mut transcript,
                    &fx.pp,
                    &proof.commitment,
                    &proof.open,
                    &subclaim,
                    &sha_f2_bit_op_virtuals(),
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
//   d. AlphaProject     — α draw + per-cell eval at α to GF(2^128)
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
        alpha_powers, eval_f2_poly_d_at_with_powers, project_column_with_powers,
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
    // ---- 0a) Commit-Pair: just the trace-pairing pre-step ----
    group.bench_function(BenchmarkId::new("Commit-Pair", id), |bench| {
        bench.iter(|| {
            let paired = zinc_protocol::f2_prove::pair_primary_witness_polys_pub::<D>(
            &fx.trace.binary_poly,
            zinc_test_uair::sha256_f2::cols::NUM_BIN_PUB,
            &sha_f2_bit_op_virtuals(),

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
            zinc_test_uair::sha256_f2::cols::NUM_BIN_PUB,
            &sha_f2_bit_op_virtuals(),

        );
        bench.iter(|| {
            let cw_matrices: Vec<_> = paired
                .par_iter()
                .map(|poly| ZipPlus::<BinZt, BinLc>::encode_rows(&fx.pp, poly))
                .collect();
            black_box(cw_matrices);
        });
    });

    // ---- 0b') Commit-Encode-RS: coefficient-wise FFT-based Reed-Solomon
    //          encoding, the binary-field analog of `IprsCode`.
    //
    // Goal: measure raw RS encode throughput at the F_2 commit shape
    // using the structurally-correct encoding: `D` independent
    // GF(2^32) RS encodings, one per coefficient slot of each input
    // cell. This mirrors what `IprsCode` does for integer cells (32
    // independent i64 polynomial slots) but with the underlying ring
    // swapped from `i64` to `BinaryElem32` (= `GF(2^32)` under
    // bolt-rs's GHASH-style pentanomial).
    //
    // **Why coefficient-wise?** Each trace cell is a `BinaryPoly<D>`
    // (D = 32 F_2 coefficients packed in a `u32`). The protocol's
    // codeword cells need to preserve all D slots so the open phase
    // can fold each coefficient slot independently. Packing all D
    // bits into a single GF(2^32) field element (the obvious "one
    // encode per cell" formulation, an earlier version of this entry)
    // *mixes* coefficient slots via field multiplication — equivalent
    // to bolt-rs's standalone usage but algebraically incompatible
    // with the F_2 prover. The IPRS-equivalent encoding does D
    // separate RS encodings instead; each per-slot output is then
    // restacked into the wide codeword cell.
    //
    // Concretely, per (matrix, row):
    //   for k in 0..D:
    //       lane[k][j] = bit k of input cell at position j (embedded
    //                    in GF(2^32) as the value 0 or 1)
    //       rs.encode(lane[k])   →   codeword_len GF(2^32) values
    //   Then codeword cell j = (out[0][j], out[1][j], ..., out[D-1][j]),
    //   = D × 32 bits = 128 bytes per cell (vs 8 bytes for `RaaF2Code`).
    //
    // NOT a drop-in for the protocol — see
    // `documentation/f2-prove-optimizations.md` § "RAA → RS swap" for
    // the open-phase / soundness barriers. We only measure encode time.
    //
    // Rate: 1/2 (matches bolt-rs / ligerito's natural rate); RAA's
    // sibling `Commit-Encode` uses rate 1/4.
    //
    // Encoder setup (twiddles + π polynomials) is built once outside
    // `bench.iter`; per-encode work is measured.
    #[cfg(feature = "bolt_rs_compare")]
    group.bench_function(BenchmarkId::new("Commit-Encode-RS", id), |bench| {
        use bolt_rs::{BinaryElem32, ReedSolomonEncoding};
        use rayon::prelude::*;
        use zinc_poly::univariate::F2PackU64;

        let paired = zinc_protocol::f2_prove::pair_primary_witness_polys_pub::<D>(
            &fx.trace.binary_poly,
            zinc_test_uair::sha256_f2::cols::NUM_BIN_PUB,
            &sha_f2_bit_op_virtuals(),
        );
        let row_len = fx.pp.linear_code.row_len();
        let num_rows = fx.pp.num_rows;
        // RS at rate 1/2 — block_length = 2 × row_len.
        let rs_block_length = row_len * 2;
        // D = number of F_2 coefficient slots per cell. The bench uses
        // `BinaryPoly<32>` halves of paired `BinaryPoly<64>` cells, so
        // the meaningful slot count per *unpaired* matrix is 32. With
        // pairing, each paired matrix carries 64 slots (two unpaired
        // matrices stacked). We treat each paired matrix as 2 ×
        // 32-slot RS-encoding tasks below.
        const D_SLOTS: usize = 32;

        let rs = ReedSolomonEncoding::<BinaryElem32>::new(row_len, rs_block_length);

        // Pre-extract per-slot input vectors. For each paired matrix,
        // for each of the 2 u32 halves (low/high of BinaryPoly<64>),
        // for each row r ∈ 0..num_rows, for each coefficient slot
        // k ∈ 0..D_SLOTS: build a Vec<BinaryElem32> of length row_len
        // where entry j = bit k of cell at position (r, j) of that
        // half. Total slot-vectors: |paired| × 2 × num_rows × D_SLOTS.
        let lanes: Vec<Vec<BinaryElem32>> = paired
            .iter()
            .flat_map(|poly| {
                let mut out: Vec<Vec<BinaryElem32>> =
                    Vec::with_capacity(2 * num_rows * D_SLOTS);
                for half in 0..2usize {
                    for r in 0..num_rows {
                        // Pre-pack the row's u32 halves once, then peel
                        // bit k from each cell in the inner loop.
                        let row_u32s: Vec<u32> = (0..row_len)
                            .map(|j| {
                                let bits = poly.evaluations[r * row_len + j].pack_u64();
                                if half == 0 { bits as u32 } else { (bits >> 32) as u32 }
                            })
                            .collect();
                        for k in 0..D_SLOTS {
                            let lane: Vec<BinaryElem32> = row_u32s
                                .iter()
                                .map(|&v| BinaryElem32::from((v >> k) & 1))
                                .collect();
                            out.push(lane);
                        }
                    }
                }
                out
            })
            .collect();

        bench.iter(|| {
            // One RS encode per coefficient slot. par_iter lets rayon
            // fan the ~5K lane-encodings across worker threads.
            let encoded: Vec<Vec<BinaryElem32>> =
                lanes.par_iter().map(|lane| rs.encode(lane)).collect();
            black_box(encoded);
        });
    });

    // ---- 0c) Commit-EncodeTranspose: + transpose to column-major ----
    group.bench_function(BenchmarkId::new("Commit-EncodeTranspose", id), |bench| {
        use rayon::prelude::*;
        use zip_plus::pcs::phase_commit::transpose_to_columns;
        use zip_plus::pcs::structs::ZipPlus;
        type BinZt = <BenchF2Types<D> as F2ZincTypes<D>>::BinaryZt;
        type BinLc = <BenchF2Types<D> as F2ZincTypes<D>>::BinaryLc;
        let paired = zinc_protocol::f2_prove::pair_primary_witness_polys_pub::<D>(
            &fx.trace.binary_poly,
            zinc_test_uair::sha256_f2::cols::NUM_BIN_PUB,
            &sha_f2_bit_op_virtuals(),

        );
        let codeword_len = fx.pp.linear_code.codeword_len();
        bench.iter(|| {
            let cw_matrices: Vec<_> = paired
                .par_iter()
                .map(|poly| ZipPlus::<BinZt, BinLc>::encode_rows(&fx.pp, poly))
                .collect();
            let cw_columns = transpose_to_columns(&cw_matrices, fx.pp.num_rows, codeword_len);
            black_box((cw_matrices, cw_columns));
        });
    });

    // ---- 0d) Commit-Merkle-only: build Merkle from pre-built cw_columns ----
    group.bench_function(BenchmarkId::new("Commit-Merkle-only", id), |bench| {
        use zinc_protocol::f2_prove::LEAF_GROUP_SIZE;
        use zip_plus::merkle::MerkleTree;
        use zip_plus::pcs::phase_commit::transpose_to_columns;
        use zip_plus::pcs::structs::ZipPlus;
        use rayon::prelude::*;
        type BinZt = <BenchF2Types<D> as F2ZincTypes<D>>::BinaryZt;
        type BinLc = <BenchF2Types<D> as F2ZincTypes<D>>::BinaryLc;
        let paired = zinc_protocol::f2_prove::pair_primary_witness_polys_pub::<D>(
            &fx.trace.binary_poly,
            zinc_test_uair::sha256_f2::cols::NUM_BIN_PUB,
            &sha_f2_bit_op_virtuals(),

        );
        let codeword_len = fx.pp.linear_code.codeword_len();
        let cw_matrices: Vec<_> = paired
            .par_iter()
            .map(|poly| ZipPlus::<BinZt, BinLc>::encode_rows(&fx.pp, poly))
            .collect();
        let cw_columns = transpose_to_columns(&cw_matrices, fx.pp.num_rows, codeword_len);
        bench.iter(|| {
            let column_refs: Vec<&[_]> = cw_columns.iter().map(|c| c.as_slice()).collect();
            let mt = MerkleTree::new_from_column_groups(&column_refs, LEAF_GROUP_SIZE);
            black_box(mt);
        });
    });

    // ---- 0e) Commit-Fused: encode + fused leaf-hash from cw_matrices
    //          (skips transpose entirely, builds Merkle root directly) ----
    group.bench_function(BenchmarkId::new("Commit-Fused", id), |bench| {
        use rayon::prelude::*;
        use zinc_protocol::f2_prove::LEAF_GROUP_SIZE;
        use zip_plus::merkle::MerkleTree;
        use zip_plus::pcs::structs::ZipPlus;
        type BinZt = <BenchF2Types<D> as F2ZincTypes<D>>::BinaryZt;
        type BinLc = <BenchF2Types<D> as F2ZincTypes<D>>::BinaryLc;
        let paired = zinc_protocol::f2_prove::pair_primary_witness_polys_pub::<D>(
            &fx.trace.binary_poly,
            zinc_test_uair::sha256_f2::cols::NUM_BIN_PUB,
            &sha_f2_bit_op_virtuals(),

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

    // ---- 0f) Commit-Fused-GPU: encode + pack leaves into a contiguous
    //          slab + dispatch Metal Blake3 leaf hash + build the tree.
    //          Produces the same Merkle root as Commit-Fused (covered by
    //          `merkle::tests::gpu_row_major_grouped_root_matches_cpu`).
    //          Only built with `--features metal_gpu` on macOS.
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
            zinc_test_uair::sha256_f2::cols::NUM_BIN_PUB,
            &sha_f2_bit_op_virtuals(),

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

    // ---- 0g) Commit-Fused-GPU-Inline: encode + per-matrix scatter into
    //          a shared slab in the SAME par_iter, then GPU dispatch +
    //          tree build from the prebuilt slab. Avoids the dedicated
    //          pack pass that `Commit-Fused-GPU` does internally
    //          (which walks every cell of `cw_matrices` a second time
    //          after the encode loop).
    //
    //          Each worker scatters its matrix's contribution while
    //          that matrix's data is still hot in L1/L2 from the encode
    //          that just produced it. Different `m_idx` values write to
    //          disjoint byte sub-ranges within every leaf → race-free.
    //
    //          Same Merkle root as Commit-Fused (covered by
    //          `merkle::tests::gpu_inline_packed_root_matches_cpu`).
    #[cfg(all(feature = "metal_gpu", target_os = "macos"))]
    group.bench_function(BenchmarkId::new("Commit-Fused-GPU-Inline", id), |bench| {
        use rayon::prelude::*;
        use zinc_protocol::f2_prove::LEAF_GROUP_SIZE;
        use zinc_transcript::traits::ConstTranscribable;
        use zip_plus::merkle::{GpuSlabPtr, MerkleTree};
        use zip_plus::pcs::structs::ZipPlus;
        type BinZt = <BenchF2Types<D> as F2ZincTypes<D>>::BinaryZt;
        type BinLc = <BenchF2Types<D> as F2ZincTypes<D>>::BinaryLc;
        type Cw = <BinZt as zip_plus::pcs::structs::ZipTypes>::Cw;
        let paired = zinc_protocol::f2_prove::pair_primary_witness_polys_pub::<D>(
            &fx.trace.binary_poly,
            zinc_test_uair::sha256_f2::cols::NUM_BIN_PUB,
            &sha_f2_bit_op_virtuals(),

        );
        let codeword_len = fx.pp.linear_code.codeword_len();
        let num_rows = fx.pp.num_rows;
        let batch_size = paired.len();
        let elem_bytes = <Cw as ConstTranscribable>::NUM_BYTES;
        let leaf_bytes = LEAF_GROUP_SIZE * batch_size * num_rows * elem_bytes;
        let num_leaves = codeword_len / LEAF_GROUP_SIZE;
        bench.iter(|| {
            let mut slab = vec![0u8; num_leaves * leaf_bytes];
            let slab_ptr = GpuSlabPtr {
                ptr: slab.as_mut_ptr(),
                len: slab.len(),
            };
            let cw_matrices: Vec<_> = paired
                .par_iter()
                .enumerate()
                .map(|(m_idx, poly)| {
                    let mat = ZipPlus::<BinZt, BinLc>::encode_rows(&fx.pp, poly);
                    // SAFETY: distinct m_idx values per worker write to
                    // disjoint byte sub-ranges within each leaf.
                    unsafe {
                        MerkleTree::scatter_matrix_into_gpu_slab(
                            slab_ptr,
                            m_idx,
                            &mat,
                            num_rows,
                            codeword_len,
                            LEAF_GROUP_SIZE,
                            batch_size,
                            leaf_bytes,
                        );
                    }
                    mat
                })
                .collect();
            let mt = MerkleTree::new_from_packed_slab_gpu(&slab, num_leaves, leaf_bytes);
            black_box((cw_matrices, mt));
        });
    });

    // ---- 0) Commit (PCS commit + Merkle root + transcript absorb) ----
    //
    // Without this entry the Micro group's prover sum would miss the
    // pre-IC commit cost. Uses the pre-paired entry point so the
    // trace-pairing step (`pair_primary_witness_polys_pub`) is
    // amortised into `setup_prover` rather than counted per
    // measurement iteration — matches the architectural choice that
    // pairing belongs to "witness generation".
    let num_pub_bin = zinc_test_uair::sha256_f2::cols::NUM_BIN_PUB;
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

    // ---- 0.5) Commit-AfterPrevProve: same Commit code path but the
    //          MACHINE STATE entering it matches what e2e sees — i.e.
    //          a fresh full Prove (UAIR + Open) has just finished,
    //          dropping its ~32 MB F2OpenProof.combined_row +
    //          opened_columns, and the next iteration's Commit fires
    //          on a cache/allocator state similar to e2e's per-iter
    //          start. If this bench reports the same number as the
    //          plain `Commit` micro, the e2e-vs-micro gap is real
    //          wrap-up; if it reports the slower in-situ number
    //          (~900 ms at nvars=22), the gap is context-dependent
    //          state (cache / allocator / rayon-pool / GPU-warmth).
    group.bench_function(BenchmarkId::new("Commit-AfterPrevProve", id), |bench| {
        bench.iter_batched(
            || {
                // Setup: run a full Prove and drop it, leaving the
                // process state where e2e leaves it between iters.
                let mut t = Blake3Transcript::new();
                let proof = ZincPlusPiopF2::<BenchF2Types<D>, U, D>
                    ::prove_f2_full_pre_paired_with_bit_ops(
                        &mut t,
                        &fx.pp,
                        &fx.trace,
                        &fx.paired_primary_witness,
                        &[],
                        &sha_f2_bit_op_virtuals(),

                        fx.num_vars,
                        sha256_f2_project_scalar::<R>,
                        BENCH_NUM_OPENINGS,
                    )
                    .expect("setup prove");
                drop(proof);
                ()
            },
            |()| {
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
            },
            criterion::BatchSize::PerIteration,
        );
    });


    // ---- a) F_2-native IC ----
    //
    // The IC step. SHA-256 F_2 is degree-1 in trace MLEs (see
    // `sha_f2_uair_is_mle_first_eligible`) so the real prover routes
    // through `prove_linear` (MLE-first). We bench the same lane here
    // so micro and end-to-end measurements stay aligned.
    //
    // `_OldCombined` is left in as a separate sub-bench for the
    // row-major path so the impact of the MLE-first dispatch is
    // visible side-by-side without changing how the headline IC
    // number is read.
    group.bench_function(BenchmarkId::new("UAIR-a-F2NativeIC", id), |bench| {
        bench.iter_batched(
            || {
                let mut transcript = Blake3Transcript::new();
                let _ = ZincPlusPiopF2::<BenchF2Types<D>, U, D>
                    ::commit_and_absorb_f2_trace_with_virtuals(
                        &mut transcript, &fx.pp, &fx.trace.binary_poly,
                        &sha_f2_bit_op_virtuals(),
                    )
                    .expect("commit");
                transcript
            },
            |mut transcript| {
                let (proof, state) = F2NativeIc::<U>::prove_linear::<BinaryFieldGF128, _, D>(
                    &mut transcript,
                    &fx.trace.binary_poly,
                    num_constraints,
                    fx.num_vars,
                    &field_cfg,
                    |s: &<U as zinc_uair::Uair>::Scalar, _cfg: &()| sha256_f2_project_scalar::<R>(s),
                );
                black_box((proof, state));
            },
            criterion::BatchSize::PerIteration,
        );
    });

    group.bench_function(BenchmarkId::new("UAIR-a-F2NativeIC-OldCombined", id), |bench| {
        bench.iter_batched(
            || {
                let mut transcript = Blake3Transcript::new();
                let _ = ZincPlusPiopF2::<BenchF2Types<D>, U, D>
                    ::commit_and_absorb_f2_trace_with_virtuals(
                        &mut transcript, &fx.pp, &fx.trace.binary_poly,
                        &sha_f2_bit_op_virtuals(),
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
    // compute `α^0..α^{D-1}` once, then per-column projection via
    // `project_column_with_powers` — the same function the prove
    // path calls at `f2_prove.rs:899`. Previously this micro reproduced
    // a serial scalar inner loop inline (`eval_f2_poly_d_at_with_powers`
    // mapped over cells), which measured a slower variant than the prove
    // path actually runs (SIMD-x4 chunked path) and gave no A/B signal
    // for the GPU swap. Aligning the micro with the prove-path call
    // makes `UAIR-b-AlphaProject` (CPU) vs `UAIR-b-AlphaProject-GPU`
    // a clean direct comparison.
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
                        let evals_at_alpha =
                            project_column_with_powers::<D>(&col.evaluations, &pows);
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

    // GPU sibling of UAIR-b-AlphaProject. Calls the Metal kernel via
    // `project_column_with_powers_gpu` per column (mirrors the current
    // prove-path's per-column dispatch). Used to measure the kernel
    // in isolation, without the surrounding prove-path overhead.
    //
    // Gated on `metal_gpu` because the GPU dispatcher only exists when
    // that feature is on.
    #[cfg(all(feature = "metal_gpu", target_os = "macos"))]
    group.bench_function(BenchmarkId::new("UAIR-b-AlphaProject-GPU", id), |bench| {
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
                        let evals_at_alpha =
                            zip_plus::metal_gpu::project_column_with_powers_gpu::<D>(
                                &col.evaluations,
                                &pows,
                            );
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
    //
    // γ-batched: a SINGLE degree-2 group `[eq_r, weighted_col]`
    // where `weighted_col(y) = Σ_g γ^g · col_g(y)` consolidates the
    // num_cols per-column sumchecks the old protocol ran. The
    // round-1 message + post-round-1 MLE fold are computed by the
    // F_2 protocol's [`F2EqColRound1FastPath`], skipping ~60 % of
    // the standard round-1 per-slot multiplications via the char-2
    // eq identity `eq_table[2b'] + eq_table[2b'+1] = E[b']`. Bench
    // includes the upfront γ-weighted MLE materialisation since
    // that's the work added by the consolidation.
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
                // Materialize γ-weighted col MLE (row-major).
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
    //
    // After the γ-batched sumcheck, the prover evaluates each
    // projected column's MLE at the sumcheck point r* — these are
    // the `column_evals_at_rstar` values that travel in the proof
    // and seed the PCS opening's per-column claims. Cost: `num_cols`
    // independent O(2^num_vars) MLE-at-point evaluations.
    //
    // This step happens inside `prove_f2_uair_with_groups` body but
    // sits outside the IC + sumcheck loops, so it's missing from
    // both the previous micro breakdown and the published
    // sumcheck/IC cost numbers — yet it scales linearly with
    // `num_cols × 2^num_vars` and dominates the UAIR phase at large
    // `num_vars`.
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
        // A deterministic sumcheck-point stand-in — any field-typed
        // vector of length num_vars works; the per-column MLE-eval
        // cost is point-content-independent.
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
    //
    // Counterpart of summing UAIR-a + UAIR-b + UAIR-c + UAIR-d. The
    // difference between this and the sum of the four kernel sub-
    // benches is the per-iteration "wrap-up" work: trace-MLE clone
    // into `extended_trace` (no virtuals here so it's just the copy),
    // eq-table build, γ-weighted-col build, F2EqColRound1FastPath
    // construction (with eq_table.clone + weighted_col.clone), and
    // the per-col-eval transcript absorb after sumcheck. Without
    // this entry the Micro sum understates the actual UAIR cost.
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
                            [..zinc_test_uair::sha256_f2::cols::NUM_BIN_PUB],
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
                        fx.num_vars,
                        sha256_f2_project_scalar::<R>,
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
//
// Five sub-steps, in body order:
//   a. AlphaBasis     — `AlphaPolyBasis::new(α)` (128×128 F_2 Gauss–
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
    use zinc_poly::univariate::binary_gf128::{AlphaPolyBasis, lift_bp_to_f2_poly_1};
    use zinc_protocol::f2_prove::build_lifted_eq_tensor;
    use zinc_poly::univariate::binary_f2_wide::BinaryF2Poly;

    let field_cfg = ();

    // Set up a (hint, subclaim, alpha) once — these are inputs the
    // open phase takes from the earlier prove steps.
    let (hint, subclaim) = {
        let mut t = Blake3Transcript::new();
        let (h, _) = ZincPlusPiopF2::<BenchF2Types<D>, U, D>::commit_and_absorb_f2_trace_with_virtuals(
            &mut t,
            &fx.pp,
            &fx.trace.binary_poly,
            &sha_f2_bit_op_virtuals(),

        )
        .expect("commit");
        let (_proof, sub, _projected_trace) = ZincPlusPiopF2::<BenchF2Types<D>, U, D>
            ::prove_f2_uair_with_groups(
                &mut t,
                &fx.trace,
                &[],
                &[],
                &[],
                fx.num_vars,
                sha256_f2_project_scalar::<R>,
            )
            .expect("uair");
        (h, sub)
    };
    let alpha = subclaim.alpha;
    let sumcheck_point = subclaim.sumcheck_point.clone();
    let num_rows = fx.pp.num_rows;
    let row_len = fx.pp.linear_code.row_len();
    let num_cols = fx.trace.binary_poly.len();
    let num_open = BENCH_NUM_OPENINGS;
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
    // dense GF(2^128) values lifted to BinaryF2Poly<2>).
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

    // ---- d) CombinedRow — per-column combined_row contributions (parallel) ----
    //
    // Mirrors the loop-reordered structure from the real
    // `prove_f2_open` body (see `f2_prove.rs:1818-1858`): outer over
    // `i` (rows) so reads on `col.evaluations[i * row_len + j]` are
    // sequential and `coeffs[i]` stays in registers across the inner
    // `j` loop. A separate final pass applies `γ_g` to each per-j
    // partial. Same total work, much better cache locality at large
    // `nvars` (at nvars=20 the row stride is 512 KB).
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

    // ---- e) MerkleOpens — t = 987 sample + path generations ----
    //
    // The real `prove_f2_open` maps each sampled `column_idx` to its
    // Merkle leaf via `group_idx = column_idx / LEAF_GROUP_SIZE` before
    // calling `merkle_tree.prove(...)` (see `f2_prove.rs:1924-1926`).
    // We mirror that here so the bench works for any `LEAF_GROUP_SIZE >
    // 1` shape — without the divide we'd hand `prove()` an index past
    // `num_leaves = codeword_len / LEAF_GROUP_SIZE` and trip
    // `InvalidLeafIndex`.
    group.bench_function(BenchmarkId::new("Open-e-MerkleOpens", id), |bench| {
        use zinc_protocol::f2_prove::LEAF_GROUP_SIZE;
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

    // ---- f) GammaCoeffsLift — transcript draws + AlphaPolyBasis lifts ----
    //
    // In `prove_f2_open` body order: after the (q0, q1) tensor and
    // before the per-column folds the prover draws `num_cols`
    // gamma_gf challenges and lifts each via `basis.lift(...)` to
    // `BinaryF2Poly<2>`. After the b'/a' absorb it draws `num_rows`
    // coeffs and does the same lift. Each `lift` is a 128-bit
    // F_2[X] basis-change (Gauss-Jordan-derived linear map) so the
    // aggregate cost is `(num_cols + num_rows) × lift_cost`.
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

    // ---- g) AssembleOpened — per-opening column slice + Merkle prove ----
    //
    // Mirrors `prove_f2_open`'s opened_columns assembly: pre-sampled
    // indices feed a `cfg_iter!` over independent
    // (column-slice + Merkle prove) units. The column data is
    // already laid out column-major in `hint.cw_columns[column_idx]`
    // (built once at commit time), so each opening is one sequential
    // `Vec<BinaryPoly<D>>::clone` rather than `batch_size × num_rows`
    // cache-strided reads through `cw_matrices`.
    group.bench_function(BenchmarkId::new("Open-g-AssembleOpened", id), |bench| {
        // Pre-sample indices deterministically so the per-iter cost
        // is purely the column-slice clone + Merkle prove.
        let indices: Vec<usize> = {
            let mut t = Blake3Transcript::new();
            (0..num_open)
                .map(|_| zinc_protocol::f2_prove::sample_column_idx(&mut t, codeword_len))
                .collect()
        };
        use zinc_protocol::f2_prove::LEAF_GROUP_SIZE;
        // Commit no longer materialises cw_columns — gather from
        // row-major cw_matrices on the fly, mirroring prove_f2_open.
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
    //
    // Counterpart of summing Open-a through Open-g. The difference
    // accounts for `prove_f2_open`'s per-iteration wrap-up:
    // γ_gf challenge draw, absorb (b', a') into the transcript,
    // coeffs challenge draw, absorb combined_row, sample-and-Merkle-
    // open the K column openings, and assemble the F2OpenProof
    // struct. Without this entry the Micro sum understates the
    // actual Open cost.
    group.bench_function(BenchmarkId::new("Open-FULL", id), |bench| {
        bench.iter_batched(
            || {
                // Rebuild a fresh (hint, subclaim, transcript) each
                // iteration so prove_f2_open runs against
                // protocol-consistent inputs and a fresh transcript
                // state, mirroring how the e2e bench flows.
                let mut t = Blake3Transcript::new();
                let (h, _) = ZincPlusPiopF2::<BenchF2Types<D>, U, D>
                    ::commit_and_absorb_pre_paired_witness(
                        &mut t,
                        &fx.pp,
                        &fx.paired_primary_witness,
                        &fx.trace.binary_poly
                            [..zinc_test_uair::sha256_f2::cols::NUM_BIN_PUB],
                    )
                    .expect("commit");
                let (_proof, sub, _projected_trace) = ZincPlusPiopF2::<BenchF2Types<D>, U, D>
                    ::prove_f2_uair_with_groups(
                        &mut t,
                        &fx.trace,
                        &[],
                        &[],
                        &[],
                        fx.num_vars,
                        sha256_f2_project_scalar::<R>,
                    )
                    .expect("UAIR");
                (h, sub, t)
            },
            |(hint, subclaim, mut transcript)| {
                let open_proof = ZincPlusPiopF2::<BenchF2Types<D>, U, D>::prove_f2_open(
                    &mut transcript,
                    &fx.pp,
                    &hint,
                    &fx.trace.binary_poly[zinc_test_uair::sha256_f2::cols::NUM_BIN_PUB..],
                    &subclaim.sumcheck_point,
                    &subclaim.alpha,
                    BENCH_NUM_OPENINGS,
                );
                black_box(open_proof);
            },
            criterion::BatchSize::PerIteration,
        );
    });
}

// ---------------------------------------------------------------------------
// MICRO breakdown: timing each sub-step inside `verify_f2_uair`.
//   a. ICVerify      — `<U as IdealCheckProtocol>::verify_as_subprotocol`
//   b. SumcheckVerify — `MultiDegreeSumcheck::verify_as_subprotocol`
//   c. BatchedCheck  — γ-batched eq/Σ check: 1 eq_eval + (G+1) field
//                      multiplications to fold `Σ_g γ^g · col_g(r*)`
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
            &sha_f2_bit_op_virtuals(),

            fx.num_vars,
            sha256_f2_project_scalar::<R>,
            BENCH_NUM_OPENINGS,
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
        ZincPlusPiopF2::<BenchF2Types<D>, U, D>::absorb_public_binary_cols(
            &mut t,
            &fx.trace.binary_poly[..zinc_test_uair::sha256_f2::cols::NUM_BIN_PUB],
        );
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
        ZincPlusPiopF2::<BenchF2Types<D>, U, D>::absorb_public_binary_cols(
            &mut t,
            &fx.trace.binary_poly[..zinc_test_uair::sha256_f2::cols::NUM_BIN_PUB],
        );
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
                let _: BinaryFieldGF128 = t.get_field_challenge(&field_cfg); // alpha
                let _: BinaryFieldGF128 = t.get_field_challenge(&field_cfg); // gamma
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

    // c) γ-batched check: compute eq(r*, r_IC) · Σ_g γ^g · col_g(r*)
    //    and compare to the sumcheck's expected eval.
    group.bench_function(BenchmarkId::new("VerifyUAIR-c-BatchedCheck", id), |bench| {
        // Precompute (ic_eval_point, sumcheck_point, expected, gamma).
        let (ic_eval_point, sumcheck_point, expected) = {
            let mut t = Blake3Transcript::new();
            ZincPlusPiopF2::<BenchF2Types<D>, U, D>::absorb_commitment(&mut t, &proof.commitment);
        ZincPlusPiopF2::<BenchF2Types<D>, U, D>::absorb_public_binary_cols(
            &mut t,
            &fx.trace.binary_poly[..zinc_test_uair::sha256_f2::cols::NUM_BIN_PUB],
        );
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
            let _: BinaryFieldGF128 = t.get_field_challenge(&field_cfg); // alpha
            let _: BinaryFieldGF128 = t.get_field_challenge(&field_cfg); // gamma
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
// MICRO breakdown: timing each sub-step inside `verify_f2_open`.
//
// Body-order:
//   a. LiftedEqTensor   — `AlphaPolyBasis::new(α)` + `build_lifted_eq_tensor`
//                         (q0, q1) — the verifier rebuilds the same tensors
//                         the prover used.
//   b. EvalConsistency  — Check 1: `Σ_i q0[i] · b'[i] == a'`.
//                         `num_rows × f2_poly_mul<3,7,10>` over the
//                         prover-sent b' vector.
//   c. LiftDischarge    — Check 2: `ψ_α(a') == Σ_g γ_g · col_g(r*)`.
//                         γ-batched lift discharge in GF(2^128).
//   d. Coherence        — Check 3: `<combined_row', q1> == <coeffs, b'>`.
//                         `row_len + num_rows` `f2_poly_mul<3,7,10>`.
//   e. PerOpening       — Check 4: per-opening Merkle verify + γ-weighted
//                         column rebuild + encoding consistency, ×987
//                         opened columns. Dominates `verify_f2_open` total.
// ---------------------------------------------------------------------------

fn bench_micro_verifier_open(
    group: &mut BenchmarkGroup<WallTime>,
    id: &str,
    fx: &ProverFixture,
) {
    use zinc_poly::univariate::binary_f2_wide::{BinaryF2Poly, f2_inner_product, f2_poly_mul};
    use zinc_poly::univariate::binary_gf128::{
        AlphaPolyBasis, eval_f2_wide_poly_at, lift_bp_to_f2_poly_1,
    };
    use zinc_protocol::f2_prove::build_lifted_eq_tensor;

    let proof = {
        let mut transcript = Blake3Transcript::new();
        ZincPlusPiopF2::<BenchF2Types<D>, U, D>::prove_f2_full_with_bit_ops(
            &mut transcript,
            &fx.pp,
            &fx.trace,
            &[],
            &sha_f2_bit_op_virtuals(),

            fx.num_vars,
            sha256_f2_project_scalar::<R>,
            BENCH_NUM_OPENINGS,
        )
        .expect("prove")
    };

    let field_cfg = ();
    let num_rows = fx.pp.num_rows;
    let row_len = fx.pp.linear_code.row_len();
    let codeword_len = fx.pp.linear_code.codeword_len();
    let num_cols = fx.num_primary;

    // The verifier's open path starts with the verifier-side transcript
    // state immediately after the `verify_f2_uair` consumes the IC +
    // sumcheck. We rebuild the same primary_column_evals + sumcheck_point
    // + alpha that downstream checks consume.
    let subclaim = {
        let mut t = Blake3Transcript::new();
        ZincPlusPiopF2::<BenchF2Types<D>, U, D>::absorb_commitment(&mut t, &proof.commitment);
        ZincPlusPiopF2::<BenchF2Types<D>, U, D>::absorb_public_binary_cols(
            &mut t,
            &fx.trace.binary_poly[..zinc_test_uair::sha256_f2::cols::NUM_BIN_PUB],
        );
        ZincPlusPiopF2::<BenchF2Types<D>, U, D>::verify_f2_uair(
            &mut t,
            &proof.uair,
            &[],
            fx.num_vars,
            fx.num_primary,
            |ideal: &IdealOrZero<Sha256F2Ideal>| sha256_f2_project_ideal(ideal),
        )
        .expect("UAIR verify for open-micro setup")
    };

    // Helper: rebuild the transcript up to the start of `verify_f2_open`
    // (immediately after `verify_f2_uair` returned). Used as the
    // setup for benches that consume the post-UAIR transcript.
    let post_uair_transcript = || {
        let mut t = Blake3Transcript::new();
        ZincPlusPiopF2::<BenchF2Types<D>, U, D>::absorb_commitment(&mut t, &proof.commitment);
        ZincPlusPiopF2::<BenchF2Types<D>, U, D>::absorb_public_binary_cols(
            &mut t,
            &fx.trace.binary_poly[..zinc_test_uair::sha256_f2::cols::NUM_BIN_PUB],
        );
        let _ = ZincPlusPiopF2::<BenchF2Types<D>, U, D>::verify_f2_uair(
            &mut t,
            &proof.uair,
            &[],
            fx.num_vars,
            fx.num_primary,
            |ideal: &IdealOrZero<Sha256F2Ideal>| sha256_f2_project_ideal(ideal),
        )
        .expect("UAIR verify");
        t
    };

    // ---- a) LiftedEqTensor (basis + q0/q1) ----
    group.bench_function(BenchmarkId::new("VerifyOpen-a-LiftedEqTensor", id), |bench| {
        bench.iter(|| {
            let basis = AlphaPolyBasis::new(&subclaim.alpha);
            let (q0, q1) = build_lifted_eq_tensor(num_rows, &subclaim.sumcheck_point, &basis);
            black_box((basis, q0, q1));
        });
    });

    // Precompute basis + (q0, q1) + γ + coeffs once for the remaining benches.
    let basis = AlphaPolyBasis::new(&subclaim.alpha);
    let (q0, q1) = build_lifted_eq_tensor(num_rows, &subclaim.sumcheck_point, &basis);
    let (gamma_gf, gamma_lifted, coeffs_lifted): (
        Vec<BinaryFieldGF128>,
        Vec<BinaryF2Poly<2>>,
        Vec<BinaryF2Poly<2>>,
    ) = {
        let mut t = post_uair_transcript();
        let gamma_gf: Vec<BinaryFieldGF128> =
            t.get_field_challenges(num_cols, &field_cfg);
        let gamma_lifted: Vec<BinaryF2Poly<2>> =
            gamma_gf.iter().map(|g| basis.lift(g)).collect();
        zinc_protocol::f2_prove::absorb_f2_poly_slice::<5, _>(&mut t, proof.open.b_vector.iter());
        zinc_protocol::f2_prove::absorb_f2_poly_slice::<7, _>(
            &mut t,
            core::iter::once(&proof.open.lifted_claim),
        );
        let coeffs_gf: Vec<BinaryFieldGF128> =
            t.get_field_challenges(num_rows, &field_cfg);
        let coeffs_lifted: Vec<BinaryF2Poly<2>> =
            coeffs_gf.iter().map(|g| basis.lift(g)).collect();
        (gamma_gf, gamma_lifted, coeffs_lifted)
    };

    // ---- b) EvalConsistency — Σ_i q0[i] · b'[i] == a' ----
    group.bench_function(BenchmarkId::new("VerifyOpen-b-EvalConsistency", id), |bench| {
        bench.iter(|| {
            let mut acc = BinaryF2Poly::<7>::zero();
            for i in 0..num_rows {
                let prod: BinaryF2Poly<7> =
                    f2_poly_mul::<2, 5, 7>(&q0[i], &proof.open.b_vector[i]);
                acc += prod;
            }
            let ok = acc == proof.open.lifted_claim;
            black_box(ok);
        });
    });

    // ---- c) LiftDischarge — ψ_α(a') == Σ_g γ_g · primary_col_evals[g] ----
    group.bench_function(BenchmarkId::new("VerifyOpen-c-LiftDischarge", id), |bench| {
        bench.iter(|| {
            let psi = eval_f2_wide_poly_at::<7>(&proof.open.lifted_claim, &subclaim.alpha);
            let mut expected = BinaryFieldGF128::zero();
            for g in 0..num_cols {
                let mut term = gamma_gf[g];
                term *= &subclaim.primary_column_evals[g];
                expected += &term;
            }
            black_box((psi, expected));
        });
    });

    // ---- d) Coherence — <combined_row, q1> == <coeffs, b'> ----
    group.bench_function(BenchmarkId::new("VerifyOpen-d-Coherence", id), |bench| {
        bench.iter(|| {
            let lhs: BinaryF2Poly<7> = {
                let mut acc = BinaryF2Poly::<7>::zero();
                for j in 0..row_len {
                    let prod: BinaryF2Poly<7> =
                        f2_poly_mul::<2, 5, 7>(&q1[j], &proof.open.combined_row[j]);
                    acc += prod;
                }
                acc
            };
            let rhs: BinaryF2Poly<7> = {
                let mut acc = BinaryF2Poly::<7>::zero();
                for i in 0..num_rows {
                    let prod: BinaryF2Poly<7> =
                        f2_poly_mul::<2, 5, 7>(&coeffs_lifted[i], &proof.open.b_vector[i]);
                    acc += prod;
                }
                acc
            };
            let ok = lhs == rhs;
            black_box(ok);
        });
    });

    // ---- e) PerOpening — Check 4: per-opening Merkle + γ-weighted col + encoding ----
    //
    // Dominant cost of `verify_f2_open` — runs `num_column_openings`
    // (987 for the deployed REP=4 RAA F_2) iterations, each doing:
    //   - Merkle path verify against root
    //   - per-column F_2 cell lift to `BinaryF2Poly<1>` + γ-weighted
    //     accumulate into `BinaryF2Poly<3>` (num_cols × num_rows mults)
    //   - row-major inner product with `coeffs` (num_rows mults)
    //   - one encoded[column_idx] equality check
    //
    // Parallelised across openings via `rayon` to mirror the real
    // `verify_f2_open` (`cfg_iter!` over `opened_columns`). The
    // sequential version was ~8× slower at nvars=16 and didn't
    // reconcile with the e2e Verify total — using the same parallel
    // shape closes that gap.
    group.bench_function(BenchmarkId::new("VerifyOpen-e-PerOpening", id), |bench| {
        // Pre-compute encoded combined_row once (the protocol caches
        // this across openings).
        use zinc_protocol::f2_prove::LEAF_GROUP_SIZE;
        let encoded: Vec<BinaryF2Poly<5>> = fx
            .pp
            .linear_code
            .encode_f2_lin_open::<5>(&proof.open.combined_row);
        let paired_batch = num_cols.div_ceil(2);
        let single_col_len = paired_batch * num_rows;
        bench.iter(|| {
            #[cfg(feature = "parallel")]
            use rayon::prelude::*;
            #[cfg(feature = "parallel")]
            let it = proof.open.opened_columns.par_iter();
            #[cfg(not(feature = "parallel"))]
            let it = proof.open.opened_columns.iter();
            it.for_each(|opened| {
                // (a) Merkle path — recompute leaf hash from the
                //     `LEAF_GROUP_SIZE` columns of the group.
                let group_idx = opened.column_idx / LEAF_GROUP_SIZE;
                let group_slices: Vec<&[BinaryPoly<64>]> = (0..LEAF_GROUP_SIZE)
                    .map(|l| {
                        &opened.column_values[l * single_col_len..(l + 1) * single_col_len]
                    })
                    .collect();
                opened
                    .merkle_proof
                    .verify_grouped(&proof.commitment.root, &group_slices, group_idx)
                    .expect("Merkle verify");

                // (b) γ-weighted encoding consistency on the queried
                //     column within the group. Lift each packed cell
                //     straight from its raw u64 into two
                //     `BinaryF2Poly<1>` halves without an intermediate
                //     `BinaryPoly<D>` allocation. Mirrors the fast
                //     path in `verify_f2_open`.
                use zinc_poly::univariate::F2PackU64;
                let local_idx = opened.column_idx % LEAF_GROUP_SIZE;
                let local_col = &opened.column_values
                    [local_idx * single_col_len..(local_idx + 1) * single_col_len];
                let lo_mask: u64 = 0xFFFF_FFFFu64;
                let mut weighted_col: Vec<BinaryF2Poly<3>> =
                    vec![BinaryF2Poly::<3>::zero(); num_rows];
                for p in 0..paired_batch {
                    let g_lo = 2 * p;
                    let g_hi = 2 * p + 1;
                    let has_hi = g_hi < num_cols;
                    for i in 0..num_rows {
                        let packed_bits = local_col[p * num_rows + i].pack_u64();
                        let lo_lifted = BinaryF2Poly::<1>::from_words([packed_bits & lo_mask]);
                        let prod_lo: BinaryF2Poly<3> =
                            f2_poly_mul::<1, 2, 3>(&lo_lifted, &gamma_lifted[g_lo]);
                        weighted_col[i] += prod_lo;
                        if has_hi {
                            let hi_lifted = BinaryF2Poly::<1>::from_words([packed_bits >> 32]);
                            let prod_hi: BinaryF2Poly<3> =
                                f2_poly_mul::<1, 2, 3>(&hi_lifted, &gamma_lifted[g_hi]);
                            weighted_col[i] += prod_hi;
                        }
                    }
                }
                let actual_at_j: BinaryF2Poly<5> =
                    f2_inner_product::<2, 3, 5>(&coeffs_lifted, &weighted_col);
                let ok = actual_at_j == encoded[opened.column_idx];
                black_box(ok);
            });
        });
    });

    // Silence "unused" warnings if the per-bench iter doesn't pull
    // these names — they exist to mirror the protocol's variables.
    let _ = codeword_len;
}

// ---------------------------------------------------------------------------
// Criterion entry points.
// ---------------------------------------------------------------------------

/// `num_vars` values the e2e bench sweeps: odd from 9 to 21
/// inclusive. 9 is the SHA-256 F_2 UAIR's minimum (480 active rows
/// fit in 2^9 = 512); larger values zero-pad and measure how the
/// prover/verifier scale with the hypercube size.
const NVARS_SWEEP: &[usize] = &[9,14,16,19,20,21,22];

/// `F2_ONLY=<group>` runs only the named criterion group (`e2e`, `steps`,
/// `micro`, `hadamard`) and early-returns the rest. Criterion executes every
/// group's body even under a `-- <filter>`, so without this the heavy `e2e`
/// sweep (up to 2^22 rows) runs its `setup_prover` regardless of the filter —
/// fatal on a memory-constrained box. Unset ⇒ all groups run as before.
fn f2_skip_group(name: &str) -> bool {
    std::env::var("F2_ONLY").is_ok_and(|only| only != name)
}

fn e2e_benches(c: &mut Criterion) {
    if f2_skip_group("e2e") {
        return;
    }
    let mut group = c.benchmark_group("Zinc+ F_2 SHA-256");
    // Large hypercubes (2^21 ≈ 2.1M rows × 41 cols of bit-poly cells +
    // O(num_rows) GF(2^128) eq-table during prove + per-row IC
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
                let proof = ZincPlusPiopF2::<BenchF2Types<D>, U, D>
                    ::prove_f2_full_pre_paired_with_bit_ops(
                        &mut transcript,
                        &fx.pp,
                        &fx.trace,
                        &fx.paired_primary_witness,
                        &[],
                        &sha_f2_bit_op_virtuals(),

                        num_vars,
                        sha256_f2_project_scalar::<R>,
                        BENCH_NUM_OPENINGS,
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
                &sha_f2_bit_op_virtuals(),

                num_vars,
                sha256_f2_project_scalar::<R>,
                BENCH_NUM_OPENINGS,
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
                let subclaim = ZincPlusPiopF2::<BenchF2Types<D>, U, D>::verify_f2_full_with_bit_ops(
                    &mut transcript,
                    &fx.pp,
                    &proof,
                    &[],
                    &sha_f2_bit_op_virtuals(),

                    &fx.trace.binary_poly[..zinc_test_uair::sha256_f2::cols::NUM_BIN_PUB],
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
    if f2_skip_group("steps") {
        return;
    }
    let mut group = c.benchmark_group("Zinc+ F_2 SHA-256 Steps");
    let fx = setup_prover(9);
    let id = format!("nvars={}", fx.num_vars);
    bench_prover_steps(&mut group, &id, &fx);
    bench_verifier_steps(&mut group, &id, &fx);
    group.finish();
}

fn micro_benches(c: &mut Criterion) {
    if f2_skip_group("micro") {
        return;
    }
    let mut group = c.benchmark_group("Zinc+ F_2 SHA-256 Micro");
    let fx = setup_prover(16);
    let id = format!("nvars={}", fx.num_vars);
    bench_micro_prover_uair(&mut group, &id, &fx);
    bench_micro_prover_open(&mut group, &id, &fx);
    bench_micro_verifier_uair(&mut group, &id, &fx);
    bench_micro_verifier_open(&mut group, &id, &fx);
    group.finish();
}

// ---------------------------------------------------------------------------
// Hadamard discharge A/B: cost of registering the 16 SHA-256 Hadamard
// relations (the Approach-B discharge) vs the `&[]`-spec path the rest of
// the bench measures.
// ---------------------------------------------------------------------------

/// Build the SHA-256 Hadamard relation layout for a `2^num_vars`-row
/// trace. Mirror of the `sha256_f2_hadamard_layout` test helper in
/// `f2_prove.rs` — just column-index plumbing from `sha256_f2::cols`
/// (the relation topology lives in `Sha256F2HadamardLayout`). protocol
/// can't read `cols` (test-uair is only a dev-dep), but the bench can.
fn sha256_f2_hadamard_layout(num_vars: usize) -> zinc_protocol::f2_hadamard::Sha256F2HadamardLayout {
    use zinc_test_uair::sha256_f2::cols;
    zinc_protocol::f2_hadamard::Sha256F2HadamardLayout {
        num_vars,
        rows_per_comp: cols::ROWS_PER_COMP,
        rounds_per_comp: cols::ROUNDS_PER_COMP,
        num_compressions: cols::num_compressions(num_vars),
        pa_a: cols::PA_A,
        pa_e: cols::PA_E,
        pa_k: cols::PA_K,
        w_a: cols::W_A,
        w_e: cols::W_E,
        w_w: cols::W_W,
        w_sigma0: cols::W_SIGMA0,
        w_sigma1: cols::W_SIGMA1,
        w_sig0: cols::W_SIG0,
        w_sig1: cols::W_SIG1,
        w_uef: cols::W_UEF,
        w_uneg_e_g: cols::W_UNEG_E_G,
        w_maj: cols::W_MAJ,
        w_t1: cols::W_T1,
        w_t2: cols::W_T2,
        w_w_s1: cols::W_W_S1,
        w_w_s2: cols::W_W_S2,
        w_t1_s1: cols::W_T1_S1,
        w_t1_s2: cols::W_T1_S2,
        w_t1_s3: cols::W_T1_S3,
        w_t1_s4: cols::W_T1_S4,
    }
}

/// A/B the Approach-B Hadamard discharge cost: the same `prove_f2_full*`
/// / `verify_f2_full*` family run **with** the 16 SHA-256 Hadamard
/// relations (3 ANDs + 13 adders, from `Sha256F2HadamardLayout`) vs
/// **without** (the `&[]` specs the rest of the bench uses). The delta
/// between the `*-Hadamard` and `*-NoHadamard` cases is the discharge
/// overhead — on the prover: the per-relation Hadamard zerocheck plus the
/// pointed-shift terms folded into the main multipoint-eval; on the
/// verifier: the recombination + the pointed-shift subclaim check; in the
/// proof: the Hadamard sumcheck messages + pair-evals.
///
/// Both sides use the NON-pre-paired entry (`prove_f2_full_with_*`, not
/// the pre-paired e2e entry), so each includes the witness pairing and the
/// A/B delta isolates the discharge. Absolute numbers therefore run a
/// touch higher than the e2e `Prove` headline (which amortises pairing).
fn bench_hadamard_compare(group: &mut BenchmarkGroup<WallTime>, id: &str, fx: &ProverFixture) {
    let (and_specs, adder_specs) = sha256_f2_hadamard_layout(fx.num_vars).relations();
    assert_eq!(and_specs.len(), 3, "expected 3 AND relations (C12–C14)");
    assert_eq!(adder_specs.len(), 13, "expected 13 adder relations (C5–C11)");

    let bit_ops = sha_f2_bit_op_virtuals();
    let public_cols = &fx.trace.binary_poly[..zinc_test_uair::sha256_f2::cols::NUM_BIN_PUB];

    // -- Prove A/B --
    group.bench_function(BenchmarkId::new("Prove-NoHadamard", id), |bench| {
        bench.iter(|| {
            let mut transcript = Blake3Transcript::new();
            let proof = ZincPlusPiopF2::<BenchF2Types<D>, U, D>::prove_f2_full_with_bit_ops(
                &mut transcript,
                &fx.pp,
                &fx.trace,
                &[],
                &bit_ops,
                fx.num_vars,
                sha256_f2_project_scalar::<R>,
                BENCH_NUM_OPENINGS,
            )
            .expect("prove (no hadamard) should succeed");
            black_box(proof);
        });
    });

    group.bench_function(BenchmarkId::new("Prove-Hadamard", id), |bench| {
        bench.iter(|| {
            let mut transcript = Blake3Transcript::new();
            let proof = ZincPlusPiopF2::<BenchF2Types<D>, U, D>::prove_f2_full_with_hadamard(
                &mut transcript,
                &fx.pp,
                &fx.trace,
                &[],
                &bit_ops,
                &and_specs,
                &adder_specs,
                fx.num_vars,
                sha256_f2_project_scalar::<R>,
                BENCH_NUM_OPENINGS,
            )
            .expect("prove (hadamard) should succeed");
            black_box(proof);
        });
    });

    // Build both proofs once for the verify benches + size report.
    let proof_no_had = {
        let mut t = Blake3Transcript::new();
        ZincPlusPiopF2::<BenchF2Types<D>, U, D>::prove_f2_full_with_bit_ops(
            &mut t,
            &fx.pp,
            &fx.trace,
            &[],
            &bit_ops,
            fx.num_vars,
            sha256_f2_project_scalar::<R>,
            BENCH_NUM_OPENINGS,
        )
        .expect("prove (no hadamard) for verify bench")
    };
    let proof_had = {
        let mut t = Blake3Transcript::new();
        ZincPlusPiopF2::<BenchF2Types<D>, U, D>::prove_f2_full_with_hadamard(
            &mut t,
            &fx.pp,
            &fx.trace,
            &[],
            &bit_ops,
            &and_specs,
            &adder_specs,
            fx.num_vars,
            sha256_f2_project_scalar::<R>,
            BENCH_NUM_OPENINGS,
        )
        .expect("prove (hadamard) for verify bench")
    };
    // Proof-size impact of the discharge (raw + zstd-3), printed once.
    eprint_f2_proof_size(&format!("{id} NoHadamard"), &proof_no_had);
    eprint_f2_proof_size(&format!("{id} Hadamard"), &proof_had);

    // -- Verify A/B --
    group.bench_function(BenchmarkId::new("Verify-NoHadamard", id), |bench| {
        bench.iter(|| {
            let mut transcript = Blake3Transcript::new();
            let subclaim = ZincPlusPiopF2::<BenchF2Types<D>, U, D>::verify_f2_full_with_bit_ops(
                &mut transcript,
                &fx.pp,
                &proof_no_had,
                &[],
                &bit_ops,
                public_cols,
                fx.num_vars,
                fx.num_primary,
                |ideal: &IdealOrZero<Sha256F2Ideal>| sha256_f2_project_ideal(ideal),
            )
            .expect("verify (no hadamard) should succeed");
            black_box(subclaim);
        });
    });

    group.bench_function(BenchmarkId::new("Verify-Hadamard", id), |bench| {
        bench.iter(|| {
            let mut transcript = Blake3Transcript::new();
            let subclaim =
                ZincPlusPiopF2::<BenchF2Types<D>, U, D>::verify_f2_full_with_hadamard::<Sha256F2Ideal>(
                    &mut transcript,
                    &fx.pp,
                    &proof_had,
                    &[],
                    &bit_ops,
                    &and_specs,
                    &adder_specs,
                    public_cols,
                    fx.num_vars,
                    fx.num_primary,
                    |ideal: &IdealOrZero<Sha256F2Ideal>| sha256_f2_project_ideal(ideal),
                )
                .expect("verify (hadamard) should succeed");
            black_box(subclaim);
        });
    });
}

fn hadamard_benches(c: &mut Criterion) {
    if f2_skip_group("hadamard") {
        return;
    }
    let mut group = c.benchmark_group("Zinc+ F_2 SHA-256 Hadamard");
    group.sample_size(10);
    // Small sweep: nvars=9 matches the Steps baseline; 16/20 show how the
    // per-relation zerocheck scales with 2^num_vars. Extend with the e2e
    // NVARS_SWEEP if you want the full curve (slower). `HAD_NVARS=16` (comma-
    // separated) restricts the sweep — each point pays a full `setup_prover`,
    // and nvars=20 needs ~13 GB (swaps on a 16 GB box), so scope it.
    let only: Option<Vec<usize>> = std::env::var("HAD_NVARS").ok().map(|s| {
        s.split(',')
            .filter_map(|x| x.trim().parse::<usize>().ok())
            .collect()
    });
    for &num_vars in &[9usize, 16, 20] {
        if let Some(only) = &only {
            if !only.contains(&num_vars) {
                continue;
            }
        }
        let fx = setup_prover(num_vars);
        let id = format!("nvars={num_vars}");
        bench_hadamard_compare(&mut group, &id, &fx);
    }
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
criterion_group! {
    name = hadamard;
    config = Criterion::default().sample_size(10);
    targets = hadamard_benches
}
criterion_main!(e2e, steps, micro, hadamard);
