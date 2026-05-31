//! All-F_2 prove path.
//!
//! Skeleton implementation of the protocol described in
//! [`f2_prove_plan.md`](../f2_prove_plan.md), stopping at the
//! sumcheck's MLE evaluation claims (which the user has said will be
//! proved in a future iteration).
//!
//! ## Pipeline
//!
//! 1. **Step 0 (commit)** — PCS commit via `RaaF2Code`. (Not run by
//!    [`prove_f2_uair`] yet — the focus of this slice is the
//!    IC/sumcheck wiring. Caller is expected to do the commit
//!    independently and absorb the commitment into the same
//!    transcript before invoking `prove_f2_uair`.)
//!
//! 2. **Step 1 (prime projection)** — mute. The trace is already in
//!    `F_2[X]`; there is no `Z[X] → F_q[X]` reduction step.
//!
//! 3. **Step 2 (ideal check over GF(2^128)[X])** — run
//!    `IdealCheckProtocol::prove_combined::<BinaryFieldGF128>` on
//!    the trace lifted via the trivial `F_2 ⊂ GF(2^128)`
//!    coefficient embedding.
//!
//! 4. **Step 3 (evaluation projection `ψ_α`)** — sample
//!    `α ∈ GF(2^128)` and substitute `X = α` in the trace, producing
//!    `Vec<DenseMultilinearExtension<BinaryFieldGF128>>`. The IC's
//!    `DynamicPolynomialF<BinaryFieldGF128>`-valued combined-MLE
//!    values are likewise evaluated at α to land in `GF(2^128)`.
//!
//! 5. **Step 4 (sumcheck over GF(2^128))** — run
//!    [`MultiDegreeSumcheck`] on the projected trace. The IC's
//!    evaluation point becomes the `eq`-style randomness for a
//!    zerocheck-shaped group; the sumcheck reduces a single
//!    degree-2 group to a final evaluation point + per-MLE
//!    expected evaluations.
//!
//! 6. **Stop.** The output is the bundle of (IC proof, sumcheck
//!    proof, MLE evaluation claims at the sumcheck's final point).
//!    Proving the MLE evaluation claims themselves is a follow-up.

use core::marker::PhantomData;
use crypto_primitives::{Field, FromWithConfig, PrimeField};
use std::fmt::Debug;
use zinc_piop::{
    ideal_check::{IdealCheckProtocol, Proof as IcProof, VerifierSubclaim as IcVerifierSubclaim},
    multipoint_eval::{
        MultipointEval, MultipointEvalError, Proof as MultipointEvalProof,
    },
    projections::project_f2_trace_row_major,
    sumcheck::{
        SumCheckError,
        multi_degree::{
            MultiDegreeSumcheck, MultiDegreeSumcheckGroup,
            MultiDegreeSumcheckProof, Round1FastPath, Round1Output,
        },
    },
};
use zinc_uair::ideal_collector::IdealOrZero;
use zinc_poly::{
    mle::DenseMultilinearExtension,
    univariate::{
        binary::BinaryPoly, binary_f2_wide::BinaryF2Poly, binary_gf128::BinaryFieldGF128,
        dynamic::over_field::DynamicPolynomialF,
    },
};
use zinc_transcript::traits::Transcript;
use zip_plus::{
    ZipError,
    code::{F2LinearOpener, LinearCode},
    merkle::MerkleProof,
    pcs::structs::{ZipPlus, ZipPlusCommitment, ZipPlusHint, ZipPlusParams},
};
use zinc_uair::{BitOp, Uair, UairTrace, constraint_counter::count_constraints};
use zinc_utils::cfg_iter;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Output of [`ZincPlusPiopF2::prove_f2_uair`]: everything the
/// verifier sees up to (but not including) the MLE evaluation claims
/// being themselves proved.
///
/// `ic_proof` is the ideal-check proof over `GF(2^128)[X]`.
/// `sumcheck_proof` is the multi-degree sumcheck proof over
/// `GF(2^128)` — γ-batched into a single degree-2 group, so the
/// proof contains exactly one round-polynomial sequence (versus
/// one per primary+virtual column in the pre-batched layout).
/// `alpha` is the evaluation-projection challenge. `gamma` is the
/// per-column γ-batching challenge drawn after α (recorded as a
/// convenience — the verifier could re-derive it).
///
/// `column_evals_at_rstar` holds the prover-supplied per-column
/// MLE evaluations at the sumcheck's final point `r*` (primary
/// columns first, then virtual columns). In the pre-batched layout
/// the verifier extracted these from the per-group sumcheck
/// subclaims; with γ-batching the sumcheck only exposes
/// `Σ_g γ^g · col_g(r*)`, so the per-column values must travel
/// in the proof and are checked by the verifier against the
/// sumcheck's batched expected evaluation.
#[derive(Clone, Debug)]
pub struct F2Proof {
    pub ic_proof: IcProof<BinaryFieldGF128>,
    pub sumcheck_proof: MultiDegreeSumcheckProof<BinaryFieldGF128>,
    /// `α ∈ GF(2^128)` — the evaluation-projection challenge, drawn
    /// from the transcript after the IC. Recorded here as a
    /// convenience; the verifier could equivalently re-derive it.
    pub alpha: BinaryFieldGF128,
    /// `γ ∈ GF(2^128)` — the per-column batching challenge, drawn
    /// from the transcript after α (and after the α-projected
    /// trace is conceptually fixed via the IC's earlier absorbs).
    /// The sumcheck reduces a single degree-2 group whose
    /// `weighted_col(y) = Σ_g γ^g · col_g(y)`.
    pub gamma: BinaryFieldGF128,
    /// Per-column MLE evaluations at `r*` (primary cols first,
    /// then virtual cols). Sent in the proof so the γ-batched
    /// sumcheck output (`eq(r*, r) · Σ_g γ^g · col_g(r*)`) can be
    /// checked against `Σ_g γ^g · column_evals_at_rstar[g]` after
    /// dividing by `eq(r*, r)`.
    pub column_evals_at_rstar: Vec<BinaryFieldGF128>,
    /// Wiring-R Hadamard zerocheck proof — `Some` iff the prover was
    /// given non-empty `hadamard_specs` (coefficient-wise `W = U ⊙ V`
    /// relations). `None` for UAIRs without such relations.
    pub hadamard_proof: Option<crate::f2_hadamard::F2HadamardProof>,
    /// α-projected MLE evals at the Hadamard sumcheck point `r*_H` of each
    /// distinct `(col, row-shift Δ)` pair the operands reference
    /// (sorted-pair order; see [`crate::f2_hadamard::distinct_pairs`]).
    /// `ψ_α(col ^↓Δ)(r*_H)`. Empty when `hadamard_proof` is `None`. The
    /// verifier derives each operand's parent eval from these (XOR-additive
    /// in ψ_α) and recombines against the bit-slice evals. The `Δ = 0` pair
    /// evals are bound to the commitment by the second-mp discharge; the
    /// `Δ ≠ 0` pair evals are trusted until a row-shift discharge lands.
    pub hadamard_pair_evals: Vec<BinaryFieldGF128>,
}

/// Errors emitted by [`ZincPlusPiopF2::prove_f2_uair`].
#[derive(Debug, thiserror::Error)]
pub enum F2ProveError<U: Uair> {
    #[error("ideal-check failed: {0}")]
    IdealCheck(zinc_piop::ideal_check::IdealCheckError<BinaryFieldGF128, U::Ideal>),
    #[error("evaluation projection failed: {0}")]
    EvalProjection(zinc_poly::EvaluationError),
    #[error("multipoint-eval prove failed: {0}")]
    MultipointEval(MultipointEvalError<BinaryFieldGF128>),
}

/// Verifier subclaim emitted by [`ZincPlusPiopF2::verify_f2_uair`]: the
/// data downstream layers need to discharge the MLE evaluation claims.
///
/// `ic_evaluation_point` is the IC's randomly-sampled point `r` (the
/// `eq`-randomness for the zerocheck-shaped sumcheck). `alpha` is the
/// evaluation-projection challenge drawn between the IC and the
/// sumcheck. `sumcheck_point` is the sumcheck's shared final point
/// `r*`.
///
/// `primary_column_evals[g]` is the verifier-derived expected
/// evaluation of *primary* (committed) column `g`'s projected MLE at
/// `r*`. These feed into the F_2[X] PCS open
/// ([`ZincPlusPiopF2::verify_f2_open`]).
///
/// `virtual_column_evals[k]` is the analogous evaluation for a
/// *virtual* binary_poly column — an F_2-linear combination of
/// primary columns declared at protocol time. Virtual columns are
/// not committed; the verifier derives their evals at `r*` from
/// `primary_column_evals` and the
/// [`F2VirtualBpSpec`] linear-combo definition.
#[derive(Clone, Debug)]
pub struct F2VerifierSubclaim {
    pub ic_evaluation_point: Vec<BinaryFieldGF128>,
    pub alpha: BinaryFieldGF128,
    pub sumcheck_point: Vec<BinaryFieldGF128>,
    pub primary_column_evals: Vec<BinaryFieldGF128>,
    pub virtual_column_evals: Vec<BinaryFieldGF128>,
    /// Hadamard sumcheck point `r*_H` (empty when there are no Hadamard
    /// specs). The sound discharge opens the pair-evals at this point.
    pub hadamard_rstar: Vec<BinaryFieldGF128>,
    /// Distinct Hadamard `(col, row-shift Δ)` pairs (sorted; empty when
    /// none). The `Δ = 0` pairs are bound to the commitment by the
    /// second-mp discharge.
    pub hadamard_pairs: Vec<(usize, usize)>,
}

/// One virtual binary_poly column for an F_2 UAIR: an
/// `F_2[X]`-linear (= XOR) combination of primary witness
/// binary_poly columns.
///
/// Virtual columns are never committed. The UAIR's
/// `constrain_general` can index them as ordinary binary_poly
/// columns *after* the primary witness columns (so primary cols
/// `0..num_primary` and virtual cols
/// `num_primary..num_primary + num_virtual`). The prover
/// materialises each virtual column from the primary trace before
/// running the IC + sumcheck; the verifier derives its MLE
/// evaluation at `r*` from `primary_column_evals` and the spec.
///
/// Coefficients are implicitly `1 ∈ F_2` — listing a primary col
/// index in `primary_col_indices` XORs that column into the virtual
/// column. Omitting an index means coefficient `0`. The order of
/// indices is irrelevant (XOR is commutative); duplicates cancel
/// pairwise.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct F2VirtualBpSpec {
    pub primary_col_indices: Vec<usize>,
}

/// Spec for a **bit-op virtual** binary_poly column. The virtual
/// column at protocol index `col_idx` in the full trace is the
/// cell-wise [`BitOp`] applied to the cells of `source_col_idx`
/// (which must be a *primary* — i.e. committed — column).
///
/// **Why this can skip commit / encoding / Merkle entirely**: the
/// RAA-F_2 encoder is F_2[X]-linear at the cell level
/// (`encode(a + b) = encode(a) + encode(b)`, plus the cells flow
/// through repeats / perms / XOR-accumulates unchanged in their
/// bit pattern). [`BitOp`] (`Rot(c)` / `ShiftR(c)`) is also
/// F_2[X]-linear at the cell level — it permutes / zeroes
/// bit positions within each cell. The two operations commute, so
///
/// ```text
///   encode( op(source_row) )  =  op( encode(source_row) )
/// ```
///
/// cell-wise. Equivalently: the codeword cell of the virtual
/// column at `(row, codeword_col)` is just `op` applied to the
/// source column's codeword cell at the same position. The
/// prover never encodes or Merkle-commits the virtual column; per
/// opening, the verifier derives the virtual column's cell from
/// the source column's opened cell.
///
/// **Soundness**: both the source and the virtual column have
/// their own MLE evaluation claims at `r*` (the verifier doesn't
/// derive one from the other — they're proved separately via
/// sumcheck + γ-batched discharge as for any committed column).
/// The encoding-consistency check at sampled column positions
/// closes the loop: the prover's claimed MLE evals are bound to
/// the actual codeword cells, and for virtual columns the
/// verifier reconstructs those cells from the source via `op`.
/// A prover lying about the virtual column's MLE eval would fail
/// the encoding-consistency check at some sampled position.
///
/// **What `BitOp` does NOT cover**: AND-style nonlinear ops
/// (`W_UEF`, `W_UNEG_E_G`, `W_MAJ`) do not commute with the F_2-
/// linear encoder; virtualising them needs a Hadamard-style
/// product check on top, which is out of scope here.
#[derive(Clone, Debug, Copy, PartialEq, Eq)]
pub struct F2BitOpVirtualSpec {
    pub col_idx: usize,
    pub source_col_idx: usize,
    pub op: BitOp,
}


/// Apply a [`BitOp`] to the low 32 bits of `cell_bits` (treating it
/// as a packed `BinaryPoly<32>`), returning the result in the low
/// 32 bits of a `u64`. High 32 bits of the output are zero.
///
/// `BitOp::Rot(c)`: cyclic LEFT rotation by `c` of the low 32 bits
/// (matches `BitOp`'s coefficient convention
/// `coeffs_out[(i + c) mod 32] = coeffs_in[i]`).
/// `BitOp::ShiftR(c)`: logical right shift by `c` of the low 32
/// bits (zero-fill at top).
#[inline]
pub fn apply_bit_op_u32(cell_bits: u64, op: BitOp) -> u64 {
    let v = cell_bits as u32;
    let out = match op {
        BitOp::Rot(c) => v.rotate_left(c),
        BitOp::ShiftR(c) => v >> c,
    };
    out as u64
}

/// Compute the list of *primary* protocol column indices given the
/// total column count and a slice of [`F2BitOpVirtualSpec`]. The
/// primary indices are sorted ascending and contain every index in
/// `0..num_total_cols` that is **not** mentioned as a `col_idx` in
/// any virtual spec. Asserts that each virtual `col_idx` is in
/// range, no `col_idx` repeats, and that each bit-op source is
/// itself primary (depth-1 derivation).
pub fn primary_col_indices_from_virtuals(
    num_total_cols: usize,
    bit_op_specs: &[F2BitOpVirtualSpec],
) -> Vec<usize> {
    let mut is_virtual = vec![false; num_total_cols];
    for spec in bit_op_specs {
        assert!(
            spec.col_idx < num_total_cols,
            "F2BitOpVirtualSpec: col_idx {} out of range (num_total_cols = {})",
            spec.col_idx,
            num_total_cols,
        );
        assert!(
            !is_virtual[spec.col_idx],
            "F2BitOpVirtualSpec: col_idx {} listed twice",
            spec.col_idx,
        );
        is_virtual[spec.col_idx] = true;
    }
    for spec in bit_op_specs {
        assert!(
            spec.source_col_idx < num_total_cols,
            "F2BitOpVirtualSpec: source_col_idx {} out of range",
            spec.source_col_idx,
        );
        assert!(
            !is_virtual[spec.source_col_idx],
            "F2BitOpVirtualSpec: source_col_idx {} is itself virtual — \
             cascaded bit-op virtuals are not supported",
            spec.source_col_idx,
        );
    }
    (0..num_total_cols).filter(|i| !is_virtual[*i]).collect()
}

/// Errors emitted by [`ZincPlusPiopF2::verify_f2_uair`].
#[derive(Debug, thiserror::Error)]
pub enum F2VerifyError<U: Uair, IdealOverF>
where
    IdealOverF: zinc_uair::ideal::Ideal,
{
    #[error("ideal-check verification failed: {0}")]
    IdealCheck(zinc_piop::ideal_check::IdealCheckError<BinaryFieldGF128, IdealOverF>),
    #[error("sumcheck verification failed: {0}")]
    Sumcheck(SumCheckError<BinaryFieldGF128>),
    #[error("α drawn from transcript ({transcript}) disagrees with proof.alpha ({proof})")]
    AlphaMismatch {
        transcript: BinaryFieldGF128,
        proof: BinaryFieldGF128,
    },
    #[error("eq(r*, r) = 0 — sumcheck point coincides with IC point, cannot derive MLE claims")]
    DegenerateEq,
    #[error("expected {expected} sumcheck groups, got {actual}")]
    GroupCountMismatch { expected: usize, actual: usize },
    #[error(
        "γ-batched sumcheck mismatch: expected {expected:?}, got eq(r*, r)·Σ γ^g·col_g(r*) = {got:?}"
    )]
    BatchedSumcheckMismatch {
        expected: BinaryFieldGF128,
        got: BinaryFieldGF128,
    },
    #[error(
        "virtual column eval mismatch at index {virtual_idx}: sumcheck-extracted ({sumcheck:?}) ≠ derived from primary evals ({derived:?})"
    )]
    VirtualEvalMismatch {
        virtual_idx: usize,
        sumcheck: BinaryFieldGF128,
        derived: BinaryFieldGF128,
    },
    #[error("hadamard proof missing for declared specs")]
    MissingHadamardProof,
    #[error("hadamard phase verification failed: {0}")]
    Hadamard(crate::f2_hadamard::F2HadamardVerifyError),
    #[error("hadamard parent-eval count {got} ≠ distinct columns {expected}")]
    HadamardParentEvalCountMismatch { got: usize, expected: usize },
    #[error("hadamard recombination failed: {0}")]
    HadamardRecombination(
        zinc_piop::lookup::booleanity::BooleanityError<BinaryFieldGF128>,
    ),
    #[error("internal: U::Uair phantom")]
    _Uair(std::marker::PhantomData<U>),
}

/// Storage cell width used inside the commitment layer.
///
/// Two consecutive trace columns of width `DEGREE_PLUS_ONE` are
/// packed into one storage cell of width `PACKED_STORAGE_WIDTH`
/// (low half = even column, high half = odd column). This halves
/// the number of `cw_matrices`, the `transpose_to_columns` memory
/// traffic, and the Merkle leaf-hash input bytes — the dominant
/// costs of commit on F_2 traces. The pairing is F_2[X]-linear and
/// transparent to the encoder / Merkle tree.
///
/// Fixed at `64` for the SHA-256 F_2 protocol (DEGREE_PLUS_ONE = 32).
/// Other UAIRs would generalise via a separate `const STORAGE_WIDTH`
/// generic; left fixed here while only this UAIR uses the F_2 path.
pub const PACKED_STORAGE_WIDTH: usize = 64;

/// Number of consecutive codeword columns hashed into one Merkle leaf.
/// With a 2-row commit (one storage cell per column = 16 B), each
/// un-grouped leaf hash pays a Blake3 setup cost that dwarfs its
/// per-byte work; grouping `LEAF_GROUP_SIZE` columns per leaf
/// amortises that setup over `LEAF_GROUP_SIZE × paired_batch ×
/// num_rows × 8` bytes of payload and reduces the leaf count (and
/// thus tree-internal hash count) by the same factor. Per opening the
/// prover must send all `LEAF_GROUP_SIZE` columns of the queried
/// group so the verifier can recompute the leaf hash — a wire bloat
/// of `LEAF_GROUP_SIZE×` on the column-values portion of the proof,
/// partly offset by a shorter Merkle path (`log2(group_size)` fewer
/// sibling hashes per opening).
pub const LEAF_GROUP_SIZE: usize = 1;

/// All-`F_2` ZincTypes-like trait. Mirrors [`ZincTypes`](crate::ZincTypes)
/// but drops the `ArbitraryZt`/`IntZt` lanes (an all-`F_2` UAIR has
/// neither) and the prime-modulus / challenge / projecting-element
/// machinery (`F_2[X]` doesn't get reduced via a random prime; the
/// projecting element is sampled directly in `GF(2^128)`).
pub trait F2ZincTypes<const DEGREE_PLUS_ONE: usize>: Clone + Debug {
    /// Zip+ types for the (paired) binary polynomial trace columns.
    /// Cells are `BinaryPoly<PACKED_STORAGE_WIDTH>`: the protocol
    /// pairs two consecutive trace columns of width `DEGREE_PLUS_ONE`
    /// into one storage cell before committing.
    type BinaryZt: zip_plus::pcs::structs::ZipTypes<
            Eval = BinaryPoly<PACKED_STORAGE_WIDTH>,
            Cw = BinaryPoly<PACKED_STORAGE_WIDTH>,
        >;

    /// Linear code used in Zip+ for the (paired) binary polynomial
    /// trace columns. Expected to be a flavour of
    /// [`RaaF2Code`](zip_plus::code::raa_f2::RaaF2Code) so that
    /// codewords stay in `F_2[X]/<X^PACKED_STORAGE_WIDTH>` (no
    /// integer widening).
    ///
    /// `F2LinearOpener` is additionally required so the F_2[X]
    /// MLE-opening protocol's proximity check can encode a width-`W`
    /// combined row through the same linear map as the commitment.
    type BinaryLc: zip_plus::code::LinearCode<Self::BinaryZt>
        + zip_plus::code::F2LinearOpener;
}

/// Phantom marker that ties the prove function to its type parameters
/// without storing any data. Mirrors `ZincPlusPiop` from the integer
/// prove path; kept as its own struct so future expansions of the
/// F_2 protocol can hang state off it.
pub struct ZincPlusPiopF2<Zt, U, const DEGREE_PLUS_ONE: usize>(PhantomData<(Zt, U)>)
where
    Zt: F2ZincTypes<DEGREE_PLUS_ONE>,
    U: Uair;

/// Materialize the virtual binary_poly columns from a primary
/// witness trace and a list of [`F2VirtualBpSpec`]s.
///
/// Each virtual column is the XOR (= F_2 sum) of the listed primary
/// witness columns, evaluated row-by-row. The output ordering
/// matches `virtual_specs`. Asserts the primary trace's columns are
/// well-shaped (same length); panics if a spec references an
/// out-of-range primary column.
#[allow(clippy::arithmetic_side_effects)]
pub fn materialize_f2_virtual_bp_cols<const D: usize>(
    primary_trace: &[DenseMultilinearExtension<BinaryPoly<D>>],
    virtual_specs: &[F2VirtualBpSpec],
) -> Vec<DenseMultilinearExtension<BinaryPoly<D>>> {
    if virtual_specs.is_empty() {
        return Vec::new();
    }
    let num_rows = primary_trace
        .first()
        .map(|m| m.evaluations.len())
        .unwrap_or(0);
    let num_vars = primary_trace
        .first()
        .map(|m| m.num_vars)
        .unwrap_or(0);
    for col in primary_trace {
        assert_eq!(
            col.evaluations.len(),
            num_rows,
            "materialize_f2_virtual_bp_cols: primary columns must share length",
        );
        assert_eq!(
            col.num_vars,
            num_vars,
            "materialize_f2_virtual_bp_cols: primary columns must share num_vars",
        );
    }
    let num_primary = primary_trace.len();

    virtual_specs
        .iter()
        .map(|spec| {
            let mut evals: Vec<BinaryPoly<D>> = vec![BinaryPoly::default(); num_rows];
            for &col_idx in &spec.primary_col_indices {
                assert!(
                    col_idx < num_primary,
                    "F2VirtualBpSpec: primary col idx {col_idx} out of range (num_primary = {num_primary})",
                );
                let primary_col = &primary_trace[col_idx];
                for (i, slot) in evals.iter_mut().enumerate() {
                    use zinc_poly::univariate::F2AddAssign;
                    slot.f2_add_assign(&primary_col.evaluations[i]);
                }
            }
            DenseMultilinearExtension::from_evaluations_vec(
                num_vars,
                evals,
                BinaryPoly::default(),
            )
        })
        .collect()
}

/// Derive virtual binary_poly column MLE evaluations at `r*` from
/// the primary column evals at `r*` and the F_2-linear combo specs.
/// Mirrors [`materialize_f2_virtual_bp_cols`]'s arithmetic at the
/// MLE-evaluation level: F_2 addition (= GF(2^128) addition) of the
/// referenced primary evals.
pub fn derive_f2_virtual_evals_at(
    primary_evals: &[BinaryFieldGF128],
    virtual_specs: &[F2VirtualBpSpec],
) -> Vec<BinaryFieldGF128> {
    virtual_specs
        .iter()
        .map(|spec| {
            let mut acc = BinaryFieldGF128::zero();
            for &col_idx in &spec.primary_col_indices {
                acc += &primary_evals[col_idx];
            }
            acc
        })
        .collect()
}

/// Recompute each public binary_poly column's α-projected MLE
/// evaluation at `point`. Public columns are never committed, so the
/// verifier binds the prover's claimed public-col evals to the actual
/// public input by recomputing them locally. Used at every evaluation
/// point the verifier checks (`r*`, `r_0`, and the Hadamard `r_0^H`).
/// Independent across columns — the outer map fans out across rayon
/// workers.
fn recompute_public_col_evals_at<const D: usize>(
    public_binary_trace: &[DenseMultilinearExtension<BinaryPoly<D>>],
    alpha: &BinaryFieldGF128,
    point: &[BinaryFieldGF128],
) -> Vec<BinaryFieldGF128> {
    let alpha_pows: Vec<BinaryFieldGF128> =
        zinc_poly::univariate::binary_gf128::alpha_powers(alpha, D);
    let zero_inner = *BinaryFieldGF128::zero().inner();
    cfg_iter!(public_binary_trace)
        .map(|col| {
            let evals_at_alpha: Vec<<BinaryFieldGF128 as crypto_primitives::Field>::Inner> = col
                .evaluations
                .iter()
                .map(|cell| {
                    *zinc_poly::univariate::binary_gf128::eval_f2_poly_d_at_with_powers::<D>(
                        cell, &alpha_pows,
                    )
                    .inner()
                })
                .collect();
            let inner_mle = DenseMultilinearExtension::from_evaluations_vec(
                col.num_vars,
                evals_at_alpha,
                zero_inner,
            );
            <DenseMultilinearExtension<_> as zinc_poly::mle::MultilinearExtensionWithConfig<
                BinaryFieldGF128,
            >>::evaluate_with_config(inner_mle, point, &())
            .expect("public-col MLE evaluation should succeed")
        })
        .collect()
}

/// Build the γ-batched `weighted_col` MLE from the α-projected
/// trace: `weighted_col(y) = Σ_g γ^g · col_g(y)`.
///
/// Materialised once before the sumcheck so the single-group
/// degree-2 sumcheck's `comb_fn` reduces to `eq · weighted_col`
/// (one multiplication per slot). Computed in parallel by row.
fn build_gamma_weighted_col(
    projected_trace: &[DenseMultilinearExtension<BinaryFieldGF128>],
    gamma: &BinaryFieldGF128,
) -> DenseMultilinearExtension<<BinaryFieldGF128 as crypto_primitives::Field>::Inner> {
    debug_assert!(!projected_trace.is_empty(), "expected ≥1 projected column");
    let num_vars = projected_trace[0].num_vars;
    let num_rows = projected_trace[0].evaluations.len();
    let zero_inner = *BinaryFieldGF128::zero().inner();

    // Precompute γ^0, γ^1, …, γ^{G-1}.
    let mut gamma_pows: Vec<BinaryFieldGF128> = Vec::with_capacity(projected_trace.len());
    let mut acc = BinaryFieldGF128::one();
    for _ in 0..projected_trace.len() {
        gamma_pows.push(acc);
        acc *= gamma;
    }

    // Row-major fold: each row sums γ^g · col_g[i] across g. Parallel
    // over rows so the inner per-row dot product stays in cache.
    let row_indices: Vec<usize> = (0..num_rows).collect();
    let evals: Vec<_> = cfg_iter!(row_indices)
        .map(|&i| {
            let mut sum = BinaryFieldGF128::zero();
            for (g, col) in projected_trace.iter().enumerate() {
                let cell = col.evaluations[i];
                sum += gamma_pows[g] * cell;
            }
            *sum.inner()
        })
        .collect();

    DenseMultilinearExtension::from_evaluations_vec(num_vars, evals, zero_inner)
}

/// Round-1 fast path for the γ-batched `eq · weighted_col` group.
///
/// Round-1 standard cost: ~5 GF(2^128) multiplications per slot
/// (three `comb_fn` calls plus boundary extrapolation for two
/// MLEs). This fast path reduces that to ~2 multiplications per
/// slot by exploiting two structural facts:
///
/// 1. **eq factoring.** `eq_table[2b'] = (1 + r₀) · E[b']` and
///    `eq_table[2b'+1] = r₀ · E[b']` where `E[b'] = eq(b', r_high)`.
///    In characteristic 2 the difference identity collapses to
///    `E[b'] = eq_table[2b'] + eq_table[2b'+1]` — a single XOR per
///    slot, no field multiplication.
/// 2. **Closed-form round message.** Once `S_A = Σ E[b'] · A[b']`
///    and `S_B = Σ E[b'] · B[b']` are accumulated (one multiply
///    per slot per sum), the three round-1 evaluation points are
///    computed from a fixed scalar template:
///
///    ```text
///    p₁(0) = (1 + r₀) · S_A
///    p₁(1) =  r₀     · S_B
///    p₁(2) = eq(2, r₀) · ((1 + 2) · S_A + 2 · S_B)
///    ```
///
///    where `2 = F::from(2)` in GF(2^128) (NOT zero — the
///    bit-pattern convention makes `F::from(2)` the field element
///    `X`, not the integer 2, which is `0` in characteristic 2).
///
/// **Total fast-path cost** (per round-1 invocation): `2 · 2^{n-1}`
/// GF(2^128) multiplications plus O(1) scalar arithmetic, versus
/// `5 · 2^{n-1}` for the standard path — a ~60 % per-slot saving
/// at round 1.
///
/// `fold_with_r1` produces the round-2-ready MLEs in the standard
/// fold convention `(1 − r₁) · low + r₁ · high`, leaving the sumcheck
/// framework's rounds 2..n untouched. The framework's
/// `skip_next_fold` flag then prevents a double-fold in round 2.
pub struct F2EqColRound1FastPath {
    pub eq_table: Vec<<BinaryFieldGF128 as Field>::Inner>,
    pub weighted_col: Vec<<BinaryFieldGF128 as Field>::Inner>,
    pub r_0: BinaryFieldGF128,
    pub num_vars: usize,
}

impl Round1FastPath<BinaryFieldGF128> for F2EqColRound1FastPath {
    #[allow(clippy::arithmetic_side_effects)]
    fn round_1_message(
        &self,
        config: &<BinaryFieldGF128 as PrimeField>::Config,
    ) -> Round1Output<BinaryFieldGF128> {
        let half = 1usize << (self.num_vars - 1);
        debug_assert_eq!(self.eq_table.len(), 2 * half);
        debug_assert_eq!(self.weighted_col.len(), 2 * half);

        // E[b'] = eq_table[2b'] + eq_table[2b'+1] (char-2 identity).
        // S_A = Σ E[b'] · weighted_col[2b'], S_B = Σ E[b'] · weighted_col[2b'+1].
        //
        // Pair-iterate over (b' → 2b', 2b'+1) so the cache line for
        // each pair is touched once. Parallel over b' with a
        // reduce-add — the inner loop is two `*` + two `+=` per pair.
        let row_indices: Vec<usize> = (0..half).collect();
        let (s_a, s_b) = cfg_iter!(row_indices)
            .map(|&b_prime| {
                // Wrap into BinaryFieldGF128 so `+` is GF(2^128) XOR,
                // not raw Uint integer-add (which would overflow).
                let eq_lo = BinaryFieldGF128::new_unchecked_with_cfg(
                    self.eq_table[2 * b_prime].clone(),
                    config,
                );
                let eq_hi = BinaryFieldGF128::new_unchecked_with_cfg(
                    self.eq_table[2 * b_prime + 1].clone(),
                    config,
                );
                let e = eq_lo + eq_hi;
                let a = BinaryFieldGF128::new_unchecked_with_cfg(
                    self.weighted_col[2 * b_prime].clone(),
                    config,
                );
                let b = BinaryFieldGF128::new_unchecked_with_cfg(
                    self.weighted_col[2 * b_prime + 1].clone(),
                    config,
                );
                (e * a, e * b)
            })
            .reduce(
                || (BinaryFieldGF128::zero(), BinaryFieldGF128::zero()),
                |acc, x| (acc.0 + x.0, acc.1 + x.1),
            );

        let one = BinaryFieldGF128::one();
        // F::from(2) under the bit-pattern convention is the field
        // element "X" (the second power-of-X basis element) — NOT
        // zero. Critical: integer-2 in characteristic 2 is 0, but
        // F::from(2) is non-zero by convention so Lagrange-style
        // boundary points work.
        let two = BinaryFieldGF128::from_with_cfg(2u64, config);
        let r_0 = self.r_0.clone();

        let p1_at_0 = (one.clone() + r_0.clone()) * s_a.clone();
        let p1_at_1 = r_0.clone() * s_b.clone();

        // eq(2, r₀) = 1 + 2 + r₀  (char-2 multilinear eq)
        let eq_at_2 = one.clone() + two.clone() + r_0;
        // weighted_col evaluated at X₀ = 2:
        //   col(2, b') = A[b'] + 2 · (B[b'] - A[b']) = (1 + 2) · A + 2 · B  (char-2: − = +)
        // Aggregated: S_col_at_2 = (1 + 2) · S_A + 2 · S_B.
        let s_col_at_2 = (one + two.clone()) * s_a + two * s_b;
        let p1_at_2 = eq_at_2 * s_col_at_2;

        Round1Output {
            asserted_sum: p1_at_0 + p1_at_1.clone(),
            tail_evaluations: vec![p1_at_1, p1_at_2],
        }
    }

    #[allow(clippy::arithmetic_side_effects)]
    fn fold_with_r1(
        self: Box<Self>,
        r_1: &BinaryFieldGF128,
        config: &<BinaryFieldGF128 as PrimeField>::Config,
    ) -> Vec<DenseMultilinearExtension<<BinaryFieldGF128 as Field>::Inner>> {
        let half = 1usize << (self.num_vars - 1);
        let zero_inner = *BinaryFieldGF128::zero().inner();

        // MLE fold at variable 0: folded[b'] = (1 - r_1) · low[b'] + r_1 · high[b'].
        // In char-2, equivalently: folded[b'] = low + r_1 · (high + low).
        let fold = |table: &[<BinaryFieldGF128 as Field>::Inner]|
            -> Vec<<BinaryFieldGF128 as Field>::Inner>
        {
            let row_indices: Vec<usize> = (0..half).collect();
            cfg_iter!(row_indices)
                .map(|&i| {
                    let lo = BinaryFieldGF128::new_unchecked_with_cfg(
                        table[2 * i].clone(),
                        config,
                    );
                    let hi = BinaryFieldGF128::new_unchecked_with_cfg(
                        table[2 * i + 1].clone(),
                        config,
                    );
                    let diff = hi + lo.clone();
                    let folded = lo + r_1.clone() * diff;
                    folded.into_inner()
                })
                .collect()
        };

        let eq_folded = fold(&self.eq_table);
        let weighted_folded = fold(&self.weighted_col);

        vec![
            DenseMultilinearExtension::from_evaluations_vec(
                self.num_vars - 1,
                eq_folded,
                zero_inner,
            ),
            DenseMultilinearExtension::from_evaluations_vec(
                self.num_vars - 1,
                weighted_folded,
                zero_inner,
            ),
        ]
    }
}

impl<Zt, U, const D: usize> ZincPlusPiopF2<Zt, U, D>
where
    Zt: F2ZincTypes<D>,
    U: Uair + 'static,
{
    /// Run the F_2 prove pipeline up to (but not including) the MLE
    /// evaluation claims.
    ///
    /// `transcript` is mutated by the IC, the α draw, and the
    /// sumcheck — caller should absorb any pre-IC commitments (PCS
    /// commitment etc.) into it before invoking.
    ///
    /// `trace` is the all-F_2 trace; its `arbitrary_poly` and `int`
    /// lanes must be empty (asserted by `project_f2_trace_row_major`).
    ///
    /// `project_scalar` lifts each UAIR scalar from
    /// `U::Scalar` to `DynamicPolynomialF<BinaryFieldGF128>` (the
    /// `GF(2^128)[X]` form the IC's combined-poly machinery
    /// expects). For an F_2-typed UAIR with `U::Scalar = BinaryPoly<D>`
    /// the natural choice is per-coefficient `F_2 ⊂ GF(2^128)`
    /// embedding; for UAIRs with no scalars (e.g. `assert_zero`-only)
    /// the closure is never invoked.
    pub fn prove_f2_uair(
        transcript: &mut impl Transcript,
        trace: &UairTrace<'static, BinaryPoly<D>, BinaryPoly<D>, D>,
        num_vars: usize,
        project_scalar: impl Fn(&U::Scalar) -> DynamicPolynomialF<BinaryFieldGF128> + Sync,
    ) -> Result<(F2Proof, F2VerifierSubclaim), F2ProveError<U>> {
        // No virtual columns. Use [`Self::prove_f2_uair_with_groups`]
        // with explicit `virtual_specs` for UAIRs that declare virtual
        // binary_poly columns. Discards the third return value
        // (`projected_trace`) — only the multipoint-eval phase in
        // `prove_f2_full_with_bit_ops` needs that.
        let (proof, subclaim, _projected_trace) = Self::prove_f2_uair_with_groups(
            transcript,
            trace,
            &[],
            &[],
            num_vars,
            project_scalar,
        )?;
        Ok((proof, subclaim))
    }

    /// Virtual-column variant of [`Self::prove_f2_uair`]. Identical
    /// to the default entry point but accepts user-declared virtual
    /// binary_poly columns via `virtual_specs`.
    ///
    /// Returns the wire `F2Proof`, the prover-side
    /// `F2VerifierSubclaim` (mirrors what the verifier would derive
    /// from the same transcript and lets the caller chain into
    /// [`Self::prove_f2_open`] without re-running the IC + sumcheck
    /// on a verifier shim), and the α-projected trace MLEs in
    /// `BinaryFieldGF128::Inner` form. The projected trace is
    /// surfaced so callers running the downstream multipoint-eval
    /// phase (e.g. [`Self::prove_f2_full_with_bit_ops`]) can reuse
    /// it instead of re-projecting; callers that don't need it can
    /// drop the third element. The per-col `column_evals_at_rstar`
    /// inside the returned `F2Proof` is computed via a per-col
    /// clone of the projected trace (the clone goes through the
    /// existing zero-copy `ManuallyDrop` reinterpret for the
    /// `evaluate_with_config` call, so only one copy per col is
    /// paid).
    pub fn prove_f2_uair_with_groups(
        transcript: &mut impl Transcript,
        trace: &UairTrace<'static, BinaryPoly<D>, BinaryPoly<D>, D>,
        virtual_specs: &[F2VirtualBpSpec],
        hadamard_specs: &[crate::f2_hadamard::F2HadamardSpec],
        num_vars: usize,
        project_scalar: impl Fn(&U::Scalar) -> DynamicPolynomialF<BinaryFieldGF128> + Sync,
    ) -> Result<
        (
            F2Proof,
            F2VerifierSubclaim,
            Vec<DenseMultilinearExtension<<BinaryFieldGF128 as Field>::Inner>>,
        ),
        F2ProveError<U>,
    > {
        let num_constraints = count_constraints::<U>();
        let field_cfg = ();
        let num_primary = trace.binary_poly.len();

        // -- Materialise virtual binary_poly columns ----------
        //
        // XOR-virtual cols are appended after the primary witness
        // cols; the UAIR's constraint code references them by their
        // (extended) absolute index. When `virtual_specs.is_empty()`
        // (the SHA-256 F_2 case, since BitOp and K virtuals don't
        // go through this materialisation path), we skip the
        // ~`num_cols × 2^num_vars × 4 B` trace clone entirely and
        // hand the IC + projection loops a borrowed slice straight
        // off `trace.binary_poly`. At nvars=22 that saves ~110 ms
        // (35 cols × 16 MB ≈ 560 MB of memcpy).
        let extended_binary_poly: std::borrow::Cow<
            '_,
            [DenseMultilinearExtension<BinaryPoly<D>>],
        > = if virtual_specs.is_empty() {
            std::borrow::Cow::Borrowed(&trace.binary_poly)
        } else {
            let virtual_cols =
                materialize_f2_virtual_bp_cols::<D>(&trace.binary_poly, virtual_specs);
            let mut all_binary_poly_cols: Vec<
                DenseMultilinearExtension<BinaryPoly<D>>,
            > = trace.binary_poly.iter().cloned().collect();
            all_binary_poly_cols.extend(virtual_cols);
            std::borrow::Cow::Owned(all_binary_poly_cols)
        };

        // -- Step 2: Ideal check over GF(2^128)[X] -----------------
        //
        // F_2-native path: skip the per-cell lift to
        // `DynamicPolynomialF<GF(2^128)>` and evaluate constraint
        // expressions row-by-row in 64-bit bit-poly arithmetic
        // (XORs and carryless shifts of `u64`s). The combined
        // polynomial's per-coefficient MLE-evaluation at the IC
        // point r is a sum-of-eqs over the rows where that bit is
        // set — no DynamicPolynomial allocation per cell. See
        // [`crate::f2_native_ic`] for the algorithm + scope.
        //
        // The caller still supplies `project_scalar` for the
        // sumcheck path below (and for completeness if a fallback
        // is needed). For the IC, we use the bit-pack form of the
        // same scalar: bit `i` set iff the scalar's i-th coefficient
        // is non-zero — for SHA-F_2 where scalars have F_2-coefficient
        // (0/1) values, this matches the `project_scalar` output
        // exactly (coefficient 0 ↔ GF(2^128) zero ↔ bit 0;
        // coefficient 1 ↔ GF(2^128) one ↔ bit 1).
        let project_scalar_to_bits =
            |s: &U::Scalar| -> u64 {
                let projected = project_scalar(s);
                let mut bits: u64 = 0;
                for (i, c) in projected.coeffs.iter().enumerate() {
                    if i >= 64 {
                        break;
                    }
                    if !<BinaryFieldGF128 as crypto_primitives::PrimeField>::is_zero(c) {
                        #[allow(clippy::arithmetic_side_effects)]
                        {
                            bits |= 1u64 << i;
                        }
                    }
                }
                bits
            };

        // Dispatch: MLE-first lane when every non-zero-ideal
        // constraint is degree-1 in the trace MLEs (collapses ~3.3M
        // per-row constraint evaluations down to ~50 column-level
        // ones); otherwise the row-major Combined path. The check
        // is a `const`-fn-ish UAIR property — no runtime cost.
        // SHA-256 F_2 is degree-1 and takes the MLE-first lane; the
        // TinyF2Uair test fixtures may not be, hence the gate.
        let effective_degree = zinc_uair::degree_counter::count_effective_max_degree::<U>();
        let (ic_proof, ic_state) = if effective_degree <= 1 {
            crate::f2_native_ic::F2NativeIc::<U>::prove_linear::<BinaryFieldGF128, _, D>(
                transcript,
                &extended_binary_poly,
                num_constraints,
                num_vars,
                &field_cfg,
                |s: &U::Scalar, cfg: &()| -> DynamicPolynomialF<BinaryFieldGF128> {
                    let _ = cfg;
                    project_scalar(s)
                },
            )
        } else {
            crate::f2_native_ic::F2NativeIc::<U>::prove_combined::<BinaryFieldGF128, _, D>(
                transcript,
                &extended_binary_poly,
                num_constraints,
                num_vars,
                &field_cfg,
                project_scalar_to_bits,
            )
        };

        // -- Step 2.5: Hadamard zerocheck (Wiring R) ---------------
        // Discharge coefficient-wise `W = U ⊙ V` operand relations *before*
        // α is sampled, so α (drawn next) is fresh w.r.t. the bit-slice
        // evals. No-op when `hadamard_specs` is empty. The distinct
        // `(col, Δ)` pair list and `r*_H` (the discharge recombination
        // inputs) land with the discharge phase; here we keep only the
        // zerocheck proof.
        let hadamard_phase = crate::f2_hadamard::prove_f2_hadamard_phase::<D>(
            transcript,
            &extended_binary_poly,
            hadamard_specs,
            &ic_state.evaluation_point,
            num_vars,
        );

        // -- Step 3: Evaluation projection (X = α) -----------------
        let alpha: BinaryFieldGF128 = transcript.get_field_challenge(&field_cfg);

        // Precompute α^0, α^1, ..., α^{D-1} once outside the per-cell
        // loop. With ~21K cells per SHA F_2 trace (41 cols × 512 rows),
        // this drops the per-cell cost from D=32 GF(2^128)
        // multiplications to just bit-selected XOR-adds — saving ~670K
        // field multiplications across the whole projection.
        let alpha_pows: Vec<BinaryFieldGF128> =
            zinc_poly::univariate::binary_gf128::alpha_powers(&alpha, D);

        // -- Step 3.5: Hadamard discharge — per-pair α-evals at r*_H --
        // Compute, for each distinct `(col, Δ)` pair the operands reference,
        // `ψ_α(col ^↓Δ)(r*_H)`, absorb them, and ship them. The verifier
        // derives each operand's parent eval from these (XOR-additive in
        // ψ_α) and recombines `Σ_b α^b·v_b == parent`. α is fresh w.r.t. the
        // bit-slice evals (Wiring R); the `Δ = 0` pair evals are bound to
        // the commitment by the second-mp discharge (the `Δ ≠ 0` ones are
        // trusted until a row-shift discharge lands).
        let (hadamard_proof, hadamard_pair_evals, hadamard_rstar, hadamard_pairs) = match hadamard_phase {
            Some((had_proof, pairs, r_star_h)) => {
                let pair_evals = crate::f2_hadamard::pair_alpha_evals::<D>(
                    &extended_binary_poly,
                    &pairs,
                    &alpha_pows,
                    &r_star_h,
                );
                let mut buf = vec![0u8; <<BinaryFieldGF128 as crypto_primitives::Field>::Inner
                    as zinc_transcript::traits::ConstTranscribable>::NUM_BYTES];
                for v in &pair_evals {
                    transcript.absorb_random_field(v, &mut buf);
                }
                (Some(had_proof), pair_evals, r_star_h, pairs)
            }
            None => (None, Vec::new(), Vec::new(), Vec::new()),
        };
        // SIMD-batched 4-cell projection via `project_column_with_powers`:
        // chunks the column into groups of 4, dispatches each group
        // through `eval_f2_poly_d_at_with_powers_simd_x4` (NEON 4-way
        // XOR on aarch64, scalar branchless fallback elsewhere), and
        // handles the residual `len % 4` cells through the scalar
        // branchy kernel. The 4-cell batch amortises the α-load and
        // (on NEON) replaces 2 scalar XORs per accumulator update
        // with one `veorq_u64`.
        //
        // A/B history on `Micro/UAIR-b-AlphaProject/nvars=22`
        // (Apple M-series, 8 cores, --features parallel,simd,unchecked):
        //   - branchy (per-cell `eval_f2_poly_d_at_with_powers`):
        //     1.0656 s  (baseline)
        //   - dense SIMD-x4 (this):
        //     1.0588 s  (-0.6% vs branchy; p<0.05, statistically
        //                significant but practically marginal — real
        //                SHA-256 cells have low popcount, leaving
        //                little room for dense-D=32 to improve)
        //   - sparse SIMD-x4 (union-bit skip):
        //     1.1048 s  (+3.7% vs branchy; the union of 4 real-trace
        //                cells saturates near D, so the per-iter loop
        //                control overhead outweighs the iter savings)
        // On `binary_gf_compare::project_col_65536/GF128_*` (random
        // 65536-cell column), dense SIMD-x4 is **2.07×** faster than
        // branchy — future UAIRs with denser bit patterns can lean on
        // this kernel directly. See ledger entry "SIMD-batched
        // α-projection" in `documentation/f2x-sha-todo.md`.
        // GPU fast-path for the α-projection. At nvars=22 the CPU
        // implementation was ~1.07 s (the dominant single phase of
        // UAIR, ~97% of UAIR-FULL). The GPU kernel
        // (`zip_plus::metal_gpu::project_columns_with_powers_gpu_batched`)
        // dispatches the inner per-cell masked-XOR loop to Metal as a
        // single command buffer containing one compute encoder per
        // column, with a single `wait_until_completed` at the end.
        // Falls back to the CPU path when the `metal_gpu` feature is
        // off (Linux CI, etc.).
        //
        // History: the first GPU patch issued one command-buffer
        // per column with its own wait, leaving ~400 ms of residual
        // per-launch overhead on top of the ~milliseconds of actual
        // kernel time. The batched-dispatch variant collapses those
        // 50 round-trips into one.
        let projected_trace: Vec<DenseMultilinearExtension<BinaryFieldGF128>> = {
            #[cfg(all(feature = "metal_gpu", target_os = "macos"))]
            {
                let column_inputs: Vec<&[BinaryPoly<D>]> = extended_binary_poly
                    .iter()
                    .map(|col| col.evaluations.as_slice())
                    .collect();
                let outputs =
                    zip_plus::metal_gpu::project_columns_with_powers_gpu_batched::<D>(
                        &column_inputs,
                        &alpha_pows,
                    );
                extended_binary_poly
                    .iter()
                    .zip(outputs)
                    .map(|(col, evals_at_alpha)| {
                        DenseMultilinearExtension::from_evaluations_vec(
                            col.num_vars,
                            evals_at_alpha,
                            BinaryFieldGF128::zero(),
                        )
                    })
                    .collect()
            }
            #[cfg(not(all(feature = "metal_gpu", target_os = "macos")))]
            {
                cfg_iter!(&*extended_binary_poly)
                    .map(|col| {
                        let evals_at_alpha =
                            zinc_poly::univariate::binary_gf128::project_column_with_powers::<D>(
                                &col.evaluations,
                                &alpha_pows,
                            );
                        DenseMultilinearExtension::from_evaluations_vec(
                            col.num_vars,
                            evals_at_alpha,
                            BinaryFieldGF128::zero(),
                        )
                    })
                    .collect()
            }
        };

        // -- Step 4a: γ-batching challenge ------------------------
        //
        // Sample γ ∈ GF(2^128) AFTER the IC + α absorbs so it can
        // depend on the trace and the projection point but not on
        // the sumcheck transcript. Build the single batched MLE
        // `weighted_col(y) = Σ_g γ^g · col_g(y)` over the full
        // primary+virtual projected trace, then run ONE degree-2
        // sumcheck on `[eq_r, weighted_col]` with comb_fn `v[0]*v[1]`
        // instead of 41 separate per-column sumchecks. The savings
        // come from (i) one set of round-message bytes instead of
        // num_total, (ii) one comb_fn / MLE-fold pass per round
        // instead of num_total. Cost added by materialising
        // weighted_col is `num_total · 2^num_vars` field mults,
        // dominated by the per-round savings as soon as num_total
        // exceeds a small constant.
        let gamma: BinaryFieldGF128 = transcript.get_field_challenge(&field_cfg);

        let eq_r =
            zinc_poly::utils::build_eq_x_r_inner(&ic_state.evaluation_point, &field_cfg)
                .expect("eq table construction must succeed for valid IC point");
        let weighted_col = build_gamma_weighted_col(&projected_trace, &gamma);

        // Round-1 fast path: skips ~60% of round-1's per-slot work
        // by exploiting eq's char-2 factoring (`E[b'] = eq[2b'] +
        // eq[2b'+1]`) and closed-form round-1 message construction.
        // See [`F2EqColRound1FastPath`]. The fast path also supplies
        // the post-round-1 folded MLEs so the framework can skip
        // the round-2 entry fold.
        // Move the eq table and γ-weighted col into the fast path
        // — both are large `Vec<<BinaryFieldGF128 as Field>::Inner>`s
        // (~96 MB each at nvars=22) and the local bindings aren't
        // used after this point, so cloning is pure waste. Saves
        // ~75 ms at SHA-256 F_2 nvars=22.
        let fast_path = Box::new(F2EqColRound1FastPath {
            eq_table: eq_r.evaluations,
            weighted_col: weighted_col.evaluations,
            r_0: ic_state.evaluation_point[0],
            num_vars,
        });
        let group = MultiDegreeSumcheckGroup::with_round_1_fast(
            2,
            // `poly` is empty — fold_with_r1 supplies the round-2
            // MLEs and the framework sets `skip_next_fold`.
            Vec::new(),
            Box::new(|v: &[BinaryFieldGF128]| v[0] * v[1]),
            fast_path,
        );

        let (sumcheck_proof, prover_states) =
            MultiDegreeSumcheck::<BinaryFieldGF128>::prove_as_subprotocol(
                transcript,
                vec![group],
                num_vars,
                &field_cfg,
            );

        // -- Derive the prover-side subclaim --------------------
        let sumcheck_point = prover_states[0].randomness.clone();
        // Per-column (primary + virtual) MLE evals at r*. Independent
        // across columns and each eval is O(2^num_vars) GF(2^128)
        // ops, so this is one of the heavier outer loops — fan out
        // across rayon workers.
        //
        // Note: `projected_trace` is **preserved** here (we now return
        // it so the caller can run the multipoint-eval phase without
        // re-α-projecting). For each col we clone its
        // `Vec<BinaryFieldGF128>` once into a workspace (paying one
        // copy ~`24 B × 2^num_vars` per col) and then ManuallyDrop the
        // clone — `BinaryFieldGF128` is `#[repr(transparent)]` around
        // `<BinaryFieldGF128 as Field>::Inner` (`Uint<2>`), so the
        // bytes are layout-identical and we can rebuild the `Vec` at
        // the inner type. The clone is the cost of preserving
        // `projected_trace`; saves ~one α-projection (`~D × 2^num_vars`
        // F-adds per col) in `prove_f2_full_with_bit_ops`'s
        // multipoint-eval phase.
        let zero_inner = *BinaryFieldGF128::zero().inner();
        let all_col_evals: Vec<BinaryFieldGF128> = cfg_iter!(&projected_trace)
            .map(|col| {
                let num_vars = col.num_vars;
                // Per-col clone of the Vec<BinaryFieldGF128> evaluations.
                // The clone is moved into `inner_mle` and then consumed
                // by `evaluate_with_config`; `projected_trace` itself
                // is untouched.
                let cloned_evals: Vec<BinaryFieldGF128> = col.evaluations.clone();
                let inner_evals: Vec<<BinaryFieldGF128 as crypto_primitives::Field>::Inner> = {
                    // SAFETY: BinaryFieldGF128 is #[repr(transparent)]
                    // around <BinaryFieldGF128 as Field>::Inner (Uint<2>),
                    // so the Vec's storage layout is identical. We move
                    // the cloned Vec out (ManuallyDrop) and rebuild
                    // it under the inner element type.
                    let mut me = core::mem::ManuallyDrop::new(cloned_evals);
                    let (ptr, len, cap) = (me.as_mut_ptr(), me.len(), me.capacity());
                    unsafe {
                        Vec::from_raw_parts(
                            ptr.cast::<<BinaryFieldGF128 as crypto_primitives::Field>::Inner>(),
                            len,
                            cap,
                        )
                    }
                };
                let inner_mle = DenseMultilinearExtension::from_evaluations_vec(
                    num_vars,
                    inner_evals,
                    zero_inner,
                );
                <DenseMultilinearExtension<_> as zinc_poly::mle::MultilinearExtensionWithConfig<
                    BinaryFieldGF128,
                >>::evaluate_with_config(
                    inner_mle, &sumcheck_point, &field_cfg
                )
                .expect("MLE evaluation on r* should succeed")
            })
            .collect();

        // Absorb the per-column evals so any downstream PCS-open
        // challenges depend on them — guards against a malicious
        // prover swapping the per-column values after sumcheck.
        let mut buf = vec![0u8; <<BinaryFieldGF128 as crypto_primitives::Field>::Inner
            as zinc_transcript::traits::ConstTranscribable>::NUM_BYTES];
        for v in &all_col_evals {
            transcript.absorb_random_field(v, &mut buf);
        }

        // Split: first `num_primary` are committed; the rest are
        // virtual. The verifier reconstructs the virtual side from
        // the primary side and `virtual_specs`.
        let (primary_evals, virtual_evals) = all_col_evals.split_at(num_primary);
        let subclaim = F2VerifierSubclaim {
            ic_evaluation_point: ic_state.evaluation_point,
            alpha,
            sumcheck_point,
            primary_column_evals: primary_evals.to_vec(),
            virtual_column_evals: virtual_evals.to_vec(),
            hadamard_rstar,
            hadamard_pairs,
        };

        // Convert `projected_trace` from `DenseMLE<BinaryFieldGF128>`
        // to `DenseMLE<<BinaryFieldGF128 as Field>::Inner>` so callers
        // can feed it straight into `MultipointEval::prove_as_subprotocol`
        // (which is generic over `F: InnerTransparentField` and expects
        // `&[DenseMLE<F::Inner>]`). Per col we transmute the `Vec`
        // via the same `ManuallyDrop` trick used for the
        // column_evals_at_rstar pass above — `BinaryFieldGF128` is
        // `#[repr(transparent)]` around `Uint<2>` so the bytes are
        // identical and this is zero-copy.
        let projected_trace_inner: Vec<
            DenseMultilinearExtension<<BinaryFieldGF128 as crypto_primitives::Field>::Inner>,
        > = projected_trace
            .into_iter()
            .map(|col| {
                let num_vars = col.num_vars;
                let inner_evals: Vec<<BinaryFieldGF128 as crypto_primitives::Field>::Inner> = {
                    let mut me = core::mem::ManuallyDrop::new(col.evaluations);
                    let (ptr, len, cap) = (me.as_mut_ptr(), me.len(), me.capacity());
                    // SAFETY: BinaryFieldGF128 is #[repr(transparent)]
                    // around Uint<2>; the storage layout is identical.
                    unsafe {
                        Vec::from_raw_parts(
                            ptr.cast::<<BinaryFieldGF128 as crypto_primitives::Field>::Inner>(),
                            len,
                            cap,
                        )
                    }
                };
                DenseMultilinearExtension::from_evaluations_vec(
                    num_vars,
                    inner_evals,
                    zero_inner,
                )
            })
            .collect();

        Ok((
            F2Proof {
                ic_proof,
                sumcheck_proof,
                alpha,
                gamma,
                column_evals_at_rstar: all_col_evals,
                hadamard_proof,
                hadamard_pair_evals,
            },
            subclaim,
            projected_trace_inner,
        ))
    }

    /// Verify a proof emitted by [`Self::prove_f2_uair`].
    ///
    /// Mirrors the prover's transcript exactly: runs the IC's
    /// `verify_as_subprotocol`, redraws α, then runs the sumcheck's
    /// `verify_as_subprotocol`. Returns the verifier's
    /// [`F2VerifierSubclaim`] — the IC point, α, the sumcheck point
    /// `r*`, and the per-column MLE evaluation claims derived from
    /// the per-group sumcheck subclaims via the `eq · col` group
    /// composition the prover used.
    ///
    /// `project_ideal` lifts `U::Ideal` → `IdealOverF` exactly as the
    /// integer-protocol verifier does in [`crate::verifier`]. For
    /// `assert_zero`-only UAIRs the closure is never invoked (the
    /// IC's per-constraint loop short-circuits on zero ideals).
    pub fn verify_f2_uair<IdealOverF>(
        transcript: &mut impl Transcript,
        proof: &F2Proof,
        virtual_specs: &[F2VirtualBpSpec],
        num_vars: usize,
        num_primary_columns: usize,
        project_ideal: impl Fn(&IdealOrZero<U::Ideal>) -> IdealOverF,
    ) -> Result<F2VerifierSubclaim, F2VerifyError<U, IdealOverF>>
    where
        IdealOverF: zinc_uair::ideal::Ideal
            + zinc_uair::ideal::IdealCheck<DynamicPolynomialF<BinaryFieldGF128>>,
    {
        Self::verify_f2_uair_with_groups(
            transcript,
            proof,
            virtual_specs,
            &[],
            num_vars,
            num_primary_columns,
            project_ideal,
        )
    }

    /// Virtual-column variant of [`Self::verify_f2_uair`]. Verifies
    /// the γ-batched IC + sumcheck pipeline and reconstructs the
    /// per-column MLE evaluation claims at `r*` from the prover-sent
    /// `proof.column_evals_at_rstar`.
    ///
    /// After extraction, the verifier checks each virtual column's
    /// extracted eval against the F_2-linear combo of primary col
    /// evals — soundness for the verifier's "virtual = linear combo"
    /// derivation. Mismatch surfaces as
    /// [`F2VerifyError::VirtualEvalMismatch`].
    pub fn verify_f2_uair_with_groups<IdealOverF>(
        transcript: &mut impl Transcript,
        proof: &F2Proof,
        virtual_specs: &[F2VirtualBpSpec],
        hadamard_specs: &[crate::f2_hadamard::F2HadamardSpec],
        num_vars: usize,
        num_primary_columns: usize,
        project_ideal: impl Fn(&IdealOrZero<U::Ideal>) -> IdealOverF,
    ) -> Result<F2VerifierSubclaim, F2VerifyError<U, IdealOverF>>
    where
        IdealOverF: zinc_uair::ideal::Ideal
            + zinc_uair::ideal::IdealCheck<DynamicPolynomialF<BinaryFieldGF128>>,
    {
        let num_constraints = count_constraints::<U>();
        let field_cfg = ();
        let num_total = num_primary_columns + virtual_specs.len();

        let ic_subclaim: IcVerifierSubclaim<BinaryFieldGF128> =
            <U as IdealCheckProtocol>::verify_as_subprotocol::<_, IdealOverF, _>(
                transcript,
                proof.ic_proof.clone(),
                num_constraints,
                num_vars,
                project_ideal,
                &field_cfg,
            )
            .map_err(F2VerifyError::IdealCheck)?;
        let ic_evaluation_point = ic_subclaim.evaluation_point;

        // -- Step 2.5: Hadamard zerocheck verify (Wiring R) -------
        // Mirrors the prover: runs before α. No-op when empty.
        let (hadamard_pairs, hadamard_rstar): (Vec<(usize, usize)>, Vec<BinaryFieldGF128>) =
            if !hadamard_specs.is_empty() {
                let hp = proof
                    .hadamard_proof
                    .as_ref()
                    .ok_or(F2VerifyError::MissingHadamardProof)?;
                crate::f2_hadamard::verify_f2_hadamard_phase::<D>(
                    transcript,
                    hp,
                    hadamard_specs,
                    &ic_evaluation_point,
                    num_vars,
                )
                .map_err(F2VerifyError::Hadamard)?
            } else {
                (Vec::new(), Vec::new())
            };

        let alpha: BinaryFieldGF128 = transcript.get_field_challenge(&field_cfg);
        if alpha != proof.alpha {
            return Err(F2VerifyError::AlphaMismatch {
                transcript: alpha,
                proof: proof.alpha,
            });
        }

        // -- Step 3.5: Hadamard discharge recombination -----------
        // Mirror the prover: absorb the per-pair α-evals (after α), derive
        // each operand's parent eval from them (XOR-additive in ψ_α), then
        // recombine `Σ_b α^b·v_b == parent` to tie the operand bit-slices to
        // the pair evals. (The `Δ = 0` pair evals are bound to the
        // commitment by the second-mp discharge; `Δ ≠ 0` are trusted.)
        if !hadamard_pairs.is_empty() {
            let hp = proof
                .hadamard_proof
                .as_ref()
                .ok_or(F2VerifyError::MissingHadamardProof)?;
            if proof.hadamard_pair_evals.len() != hadamard_pairs.len() {
                return Err(F2VerifyError::HadamardParentEvalCountMismatch {
                    got: proof.hadamard_pair_evals.len(),
                    expected: hadamard_pairs.len(),
                });
            }
            let mut buf = vec![0u8; <<BinaryFieldGF128 as crypto_primitives::Field>::Inner
                as zinc_transcript::traits::ConstTranscribable>::NUM_BYTES];
            for v in &proof.hadamard_pair_evals {
                transcript.absorb_random_field(v, &mut buf);
            }
            let alpha_pows = zinc_poly::univariate::binary_gf128::alpha_powers(&alpha, D);
            let operand_parents = crate::f2_hadamard::derive_operand_parents(
                hadamard_specs,
                &hadamard_pairs,
                &proof.hadamard_pair_evals,
                &alpha_pows,
            );
            zinc_piop::lookup::booleanity::verify_bit_decomposition_consistency(
                &operand_parents,
                &hp.bit_slice_evals,
                &alpha,
                D,
            )
            .map_err(F2VerifyError::HadamardRecombination)?;
        }

        // γ is sampled BEFORE the sumcheck, mirroring the prover's
        // order. The verifier checks the round-by-round sumcheck
        // first (against the single batched group's claimed sum +
        // round polys), then closes the chain by computing
        // `Σ_g γ^g · column_evals_at_rstar[g]` and checking
        // `eq(r*, r_IC) · that_sum == sumcheck_expected_eval`.
        let gamma: BinaryFieldGF128 = transcript.get_field_challenge(&field_cfg);

        let md_subclaims = MultiDegreeSumcheck::<BinaryFieldGF128>::verify_as_subprotocol(
            transcript,
            num_vars,
            &proof.sumcheck_proof,
            &field_cfg,
        )
        .map_err(F2VerifyError::Sumcheck)?;

        if md_subclaims.expected_evaluations().len() != 1 {
            return Err(F2VerifyError::GroupCountMismatch {
                expected: 1,
                actual: md_subclaims.expected_evaluations().len(),
            });
        }

        let sumcheck_point = md_subclaims.point().to_vec();

        if proof.column_evals_at_rstar.len() != num_total {
            return Err(F2VerifyError::GroupCountMismatch {
                expected: num_total,
                actual: proof.column_evals_at_rstar.len(),
            });
        }

        // Check the γ-batched sumcheck output against the
        // prover-supplied per-column evals.
        // expected_eval == eq(r*, r_IC) · Σ_g γ^g · col_g(r*)
        let one = BinaryFieldGF128::one();
        let eq_at_rstar_r = zinc_poly::utils::eq_eval(
            &sumcheck_point,
            &ic_evaluation_point,
            one,
        )
        .expect("matching length (num_vars) by construction");
        if eq_at_rstar_r.is_zero() {
            return Err(F2VerifyError::DegenerateEq);
        }
        // Horner-fold: Σ_g γ^g · col_g(r*) = col_0(r*) + γ·(col_1(r*) + γ·(...))
        let mut batched = BinaryFieldGF128::zero();
        for v in proof.column_evals_at_rstar.iter().rev() {
            batched = batched * gamma + *v;
        }
        let expected = md_subclaims.expected_evaluations()[0];
        if eq_at_rstar_r * batched != expected {
            return Err(F2VerifyError::BatchedSumcheckMismatch {
                expected,
                got: eq_at_rstar_r * batched,
            });
        }

        // Absorb per-column evals into the transcript (mirror of
        // prover) so subsequent PCS-open challenges depend on them.
        let mut buf = vec![0u8; <<BinaryFieldGF128 as crypto_primitives::Field>::Inner
            as zinc_transcript::traits::ConstTranscribable>::NUM_BYTES];
        for v in &proof.column_evals_at_rstar {
            transcript.absorb_random_field(v, &mut buf);
        }

        let (primary_evals, virtual_evals_from_proof) =
            proof.column_evals_at_rstar.split_at(num_primary_columns);
        let primary_evals = primary_evals.to_vec();

        // Virtual-eval consistency: each virtual col eval claimed
        // by the prover must equal the F_2-linear combo of primary
        // evals defined by the spec.
        let virtual_evals_derived =
            derive_f2_virtual_evals_at(&primary_evals, virtual_specs);
        for (k, (from_proof, derived)) in virtual_evals_from_proof
            .iter()
            .zip(virtual_evals_derived.iter())
            .enumerate()
        {
            if from_proof != derived {
                return Err(F2VerifyError::VirtualEvalMismatch {
                    virtual_idx: k,
                    sumcheck: *from_proof,
                    derived: *derived,
                });
            }
        }

        Ok(F2VerifierSubclaim {
            ic_evaluation_point,
            alpha,
            sumcheck_point,
            primary_column_evals: primary_evals,
            virtual_column_evals: virtual_evals_derived,
            hadamard_rstar,
            hadamard_pairs,
        })
    }

    // -- PCS commit/open plumbing ------------------------------------
    //
    // Step 0 (commit) and Step 7 (open) are exposed as separate
    // functions so callers can compose: commit binary trace columns,
    // absorb the commitment into the transcript, then run the
    // IC + sumcheck via `prove_f2_uair`. The shape mirrors the
    // integer prove path, which keeps commit/open as separate phases
    // around the PIOP.

    /// Step 0: commit to the F_2 trace's binary_poly columns via the
    /// caller-supplied Zip+ params, returning the commitment + a
    /// prover-side hint that's needed at open time.
    ///
    /// Caller is expected to absorb `commitment.root.0` into the
    /// Fiat-Shamir transcript before invoking
    /// [`Self::prove_f2_uair`]; see
    /// [`Self::commit_and_absorb_f2_trace`] for the bundled helper.
    pub fn commit_f2_trace(
        pp: &ZipPlusParams<Zt::BinaryZt, Zt::BinaryLc>,
        trace_binary_cols: &[DenseMultilinearExtension<BinaryPoly<D>>],
    ) -> Result<
        (
            ZipPlusHint<<Zt::BinaryZt as zip_plus::pcs::structs::ZipTypes>::Cw>,
            ZipPlusCommitment,
        ),
        ZipError,
    > {
        Self::commit_f2_trace_with_virtuals(pp, trace_binary_cols, &[])
    }

    /// Like [`Self::commit_f2_trace`] but skips committing the bit-op
    /// virtual columns declared by `bit_op_specs` AND the public
    /// columns (which the verifier already knows and can encode
    /// locally). Only **non-public, non-virtual** witness columns are
    /// paired and Merkle-committed. The full trace (incl materialised
    /// virtuals) is still passed in so the constraint system can
    /// index by the protocol's original column order; the
    /// public/witness split is read off `U::signature()`, mirroring
    /// the integer protocol's [`absorb_public_columns`] pattern.
    ///
    /// See [`F2BitOpVirtualSpec`] for the soundness arguments and
    /// current scope notes.
    pub fn commit_f2_trace_with_virtuals(
        pp: &ZipPlusParams<Zt::BinaryZt, Zt::BinaryLc>,
        trace_binary_cols: &[DenseMultilinearExtension<BinaryPoly<D>>],
        bit_op_specs: &[F2BitOpVirtualSpec],
    ) -> Result<
        (
            ZipPlusHint<<Zt::BinaryZt as zip_plus::pcs::structs::ZipTypes>::Cw>,
            ZipPlusCommitment,
        ),
        ZipError,
    > {
        let sig = U::signature();
        let num_pub_bin = sig.public_cols().num_binary_poly_cols();
        let num_wit_bin = trace_binary_cols.len() - num_pub_bin;

        // Translate bit-op specs from PROTOCOL col indices into
        // WITNESS-local indices (witness slice starts at `num_pub_bin`).
        let witness_bit_op_specs: Vec<F2BitOpVirtualSpec> = bit_op_specs
            .iter()
            .map(|s| {
                assert!(
                    s.col_idx >= num_pub_bin,
                    "F2BitOpVirtualSpec.col_idx {} is in the public range \
                     [0, {}); only witness columns can be virtualised",
                    s.col_idx,
                    num_pub_bin,
                );
                assert!(
                    s.source_col_idx >= num_pub_bin,
                    "F2BitOpVirtualSpec.source_col_idx {} is in the public range; \
                     only witness columns can serve as bit-op sources today",
                    s.source_col_idx,
                );
                F2BitOpVirtualSpec {
                    col_idx: s.col_idx - num_pub_bin,
                    source_col_idx: s.source_col_idx - num_pub_bin,
                    op: s.op,
                }
            })
            .collect();

        // Filter to primary WITNESS cols (those neither public nor
        // declared virtual), then pair + commit.
        let primary_witness_indices =
            primary_col_indices_from_virtuals(num_wit_bin, &witness_bit_op_specs);
        let witness_slice = &trace_binary_cols[num_pub_bin..];
        let primary_cols: Vec<DenseMultilinearExtension<BinaryPoly<D>>> =
            primary_witness_indices
                .iter()
                .map(|&i| witness_slice[i].clone())
                .collect();
        let paired = pair_trace_polys::<D>(&primary_cols);
        ZipPlus::<Zt::BinaryZt, Zt::BinaryLc>::commit_grouped(pp, &paired, LEAF_GROUP_SIZE)
    }

    /// Convenience: commit + absorb the resulting commitment root
    /// into `transcript`. Returns the hint (needed at open time) and
    /// the commitment (so the verifier can be handed the same root).
    pub fn commit_and_absorb_f2_trace(
        transcript: &mut impl Transcript,
        pp: &ZipPlusParams<Zt::BinaryZt, Zt::BinaryLc>,
        trace_binary_cols: &[DenseMultilinearExtension<BinaryPoly<D>>],
    ) -> Result<
        (
            ZipPlusHint<<Zt::BinaryZt as zip_plus::pcs::structs::ZipTypes>::Cw>,
            ZipPlusCommitment,
        ),
        ZipError,
    > {
        Self::commit_and_absorb_f2_trace_with_virtuals(
            transcript,
            pp,
            trace_binary_cols,
            &[],
        )
    }

    /// `commit_and_absorb` variant aware of bit-op virtual columns and
    /// the public/witness split. Public columns are NOT committed —
    /// the verifier knows them — but their bytes are absorbed into the
    /// transcript right after the commitment root so the proof binds
    /// to them (matching the integer protocol's
    /// `absorb_public_columns` pattern in [`crate::verifier::verify`]).
    /// See [`Self::commit_f2_trace_with_virtuals`].
    pub fn commit_and_absorb_f2_trace_with_virtuals(
        transcript: &mut impl Transcript,
        pp: &ZipPlusParams<Zt::BinaryZt, Zt::BinaryLc>,
        trace_binary_cols: &[DenseMultilinearExtension<BinaryPoly<D>>],
        bit_op_specs: &[F2BitOpVirtualSpec],
    ) -> Result<
        (
            ZipPlusHint<<Zt::BinaryZt as zip_plus::pcs::structs::ZipTypes>::Cw>,
            ZipPlusCommitment,
        ),
        ZipError,
    > {
        let (hint, comm) = Self::commit_f2_trace_with_virtuals(
            pp,
            trace_binary_cols,
            bit_op_specs,
        )?;
        // Absorb the Merkle root, then the public column data — same
        // order as the integer prover. The verifier's mirror absorbs
        // both in the same sequence so Fiat-Shamir challenges drawn
        // afterwards depend on every public byte the proof claims to
        // be valid against.
        transcript.absorb_slice(&*comm.root);
        let sig = U::signature();
        let num_pub_bin = sig.public_cols().num_binary_poly_cols();
        crate::absorb_public_columns(transcript, &trace_binary_cols[..num_pub_bin]);
        Ok((hint, comm))
    }

    /// Commit a **pre-paired** witness slice. Use this when the caller
    /// has already applied [`pair_primary_witness_polys_pub`] to the
    /// trace (e.g. in a benchmark fixture that amortises the pairing
    /// across measurement iterations, or in a production driver that
    /// runs witness generation once and many proofs from the same
    /// trace).
    ///
    /// `paired_witness` is the output of `pair_primary_witness_polys_pub`,
    /// i.e. the filtered + paired primary witness cols, ready to be
    /// fed straight to `ZipPlus::commit_grouped` — no spec
    /// translation, no `.clone()` of trace MLEs, no pairing pass.
    /// Compared with [`Self::commit_f2_trace_with_virtuals`], this
    /// shaves the ~50–150 ms at SHA-256 F_2 nvars=22 spent on
    /// spec translation, MLE cloning, and `pair_trace_polys`.
    pub fn commit_pre_paired_witness(
        pp: &ZipPlusParams<Zt::BinaryZt, Zt::BinaryLc>,
        paired_witness: &[DenseMultilinearExtension<BinaryPoly<PACKED_STORAGE_WIDTH>>],
    ) -> Result<
        (
            ZipPlusHint<<Zt::BinaryZt as zip_plus::pcs::structs::ZipTypes>::Cw>,
            ZipPlusCommitment,
        ),
        ZipError,
    > {
        ZipPlus::<Zt::BinaryZt, Zt::BinaryLc>::commit_grouped(
            pp,
            paired_witness,
            LEAF_GROUP_SIZE,
        )
    }

    /// `commit_and_absorb` variant for pre-paired callers. Mirrors
    /// [`Self::commit_and_absorb_f2_trace_with_virtuals`] step-for-step
    /// — root absorb followed by public-column absorb — so the
    /// transcript state after this call is identical to the
    /// trace-pairing variant.
    pub fn commit_and_absorb_pre_paired_witness(
        transcript: &mut impl Transcript,
        pp: &ZipPlusParams<Zt::BinaryZt, Zt::BinaryLc>,
        paired_witness: &[DenseMultilinearExtension<BinaryPoly<PACKED_STORAGE_WIDTH>>],
        public_binary_trace: &[DenseMultilinearExtension<BinaryPoly<D>>],
    ) -> Result<
        (
            ZipPlusHint<<Zt::BinaryZt as zip_plus::pcs::structs::ZipTypes>::Cw>,
            ZipPlusCommitment,
        ),
        ZipError,
    > {
        let (hint, comm) = Self::commit_pre_paired_witness(pp, paired_witness)?;
        transcript.absorb_slice(&*comm.root);
        crate::absorb_public_columns(transcript, public_binary_trace);
        Ok((hint, comm))
    }

    /// Verifier counterpart of [`Self::commit_and_absorb_f2_trace`]:
    /// absorbs a previously-published commitment into the verifier's
    /// transcript at the same point the prover did. Idempotent —
    /// just bytes-into-transcript.
    pub fn absorb_commitment(
        transcript: &mut impl Transcript,
        commitment: &ZipPlusCommitment,
    ) {
        transcript.absorb_slice(&*commitment.root);
    }

    /// Verifier counterpart of the public-column absorption performed by
    /// [`Self::commit_and_absorb_f2_trace_with_virtuals`] right after
    /// the commitment root. Must be called in the same order — i.e.
    /// immediately after [`Self::absorb_commitment`] — so the
    /// Fiat-Shamir randomness drawn afterwards matches the prover's.
    pub fn absorb_public_binary_cols(
        transcript: &mut impl Transcript,
        public_binary_trace: &[DenseMultilinearExtension<BinaryPoly<D>>],
    ) {
        crate::absorb_public_columns(transcript, public_binary_trace);
    }
}

// -- Step 7 (open) via F_2[X] lift-and-project --------------------
//
// See [`f2_open_plan.md`](../f2_open_plan.md) for the full design.
//
// `F2OpenProof` carries: (i) a per-column lifted claim `a_g' ∈
// F_2[X]<≈416>`, (ii) a per-column "b-vector" `b_g ∈ F_2[X]^{num_rows}`
// supporting the evaluation-consistency check. The verifier:
//
//   1. Re-derives `(q_0, q_1)` from the sumcheck's final point `r*`,
//      computing the eq-tensor in `GF(2^128)` and lifting each entry
//      to `BinaryF2Poly<2>` (the canonical degree-<192 representative).
//   2. Checks `Σ_i q_0[i] · b_g[i] = a_g'` in `F_2[X]` (evaluation
//      consistency).
//   3. Checks `ψ_α(a_g') = a_g` in `GF(2^128)` (the lift discharge).
//
// **Proximity not yet bound.** This slice provides the lift +
// evaluation-consistency portion. The proximity check — that `b_g`
// is consistent with the committed codeword matrix via Merkle column
// openings — is the follow-up that ties `b_g` to the PCS. The
// existing `commit_hint` carries everything required (codeword
// matrices + Merkle tree); the missing pieces are the column-sampling
// loop and the `F_2[X]`-lifted re-encoding check at sampled columns.
// See `f2_open_plan.md` § "Piece 4" for the shape.

/// γ-batched data emitted by [`ZincPlusPiopF2::prove_f2_open`].
///
/// **Single-open semantics.** Instead of sending per-column
/// `(a_g', b_g, combined_row_g)` triples, the prover draws a
/// transcript-fresh GF(2^128) challenge `γ_g` for each committed
/// primary column and folds the per-column data into a **single**
/// `(a', b', combined_row')` bundle:
///
/// ```text
/// a'             := Σ_g γ_g · a_g'              ∈ F_2[X]<≈608>
/// b'[i]          := Σ_g γ_g · b_g[i]            ∈ F_2[X]<≈416>
/// combined_row'  := Σ_g γ_g · combined_row_g    ∈ F_2[X]<≈416>
/// ```
///
/// The verifier reconstructs the same `γ_g` from the transcript,
/// checks `Σ_i q_0'[i] · b'[i] = a'` (eval consistency),
/// `ψ_α(a') = Σ_g γ_g · a_g` (lift discharge, where each `a_g` is
/// the per-column MLE claim from the sumcheck subclaim),
/// `<combined_row', q_1'> = <coeffs, b'>` (coherence), and the
/// γ-weighted encoding consistency at each opened column. Soundness
/// against per-column tampering follows from Schwartz-Zippel over
/// the random γ_g.
///
/// Widths: `a' ∈ BinaryF2Poly<7>` (≥ D + 2·192 - 1 + 192 - 1 =
/// 606 bits for D=32); `b'` and `combined_row'` entries in
/// `BinaryF2Poly<5>` (≥ D + 2·192 - 1 = 414 bits).
//
// TODO: when `feature(generic_const_exprs)` stabilises, parameterise
// these widths over `D` and `μ_eq` so the BinaryF2Poly<W> sizes
// shrink to the true bit bounds. Until then the slight
// over-allocation costs ~10% extra transcription bytes per opening;
// not on a hot path for any current workload.
#[derive(Clone, Debug)]
pub struct F2OpenProof<const D: usize> {
    /// `a' = Σ_g γ_g · a_g' ∈ F_2[X]`.
    pub lifted_claim: BinaryF2Poly<7>,
    /// `b'[i] = Σ_g γ_g · b_g[i]`. `num_rows` entries.
    pub b_vector: Vec<BinaryF2Poly<5>>,
    /// `combined_row'[j] = Σ_g γ_g · combined_row_g[j]`. `row_len` entries.
    pub combined_row: Vec<BinaryF2Poly<5>>,
    /// One entry per opened codeword column. Each entry holds the
    /// column's `batch_size · num_rows` codeword cells (concatenated
    /// per-poly in commit order) plus a Merkle proof.
    pub opened_columns: Vec<F2OpenedColumn<D>>,
}

/// A single column opening: the column index, the concatenated
/// codeword cells across all committed (paired) polynomials, and the
/// Merkle proof tying those cells to the commitment root.
///
/// Cells are width-[`PACKED_STORAGE_WIDTH`] packed pairs: each holds
/// two consecutive trace columns' codeword cells (low / high u32).
/// `column_values.len() = paired_batch_size · num_rows`, where
/// `paired_batch_size = ⌈primary_num_cols / 2⌉`. The verifier hashes
/// these bytes directly for the Merkle leaf check and unpacks into
/// `2 · paired_batch_size` per-original-column cells for the
/// encoding-consistency / γ-discharge check.
#[derive(Clone, Debug)]
pub struct F2OpenedColumn<const D: usize> {
    pub column_idx: usize,
    /// `paired_batch_size · num_rows` packed cells.
    pub column_values: Vec<BinaryPoly<PACKED_STORAGE_WIDTH>>,
    pub merkle_proof: MerkleProof,
    /// Marker: the trace-cell-width parameter the protocol is
    /// configured with (used to validate D == PACKED_STORAGE_WIDTH / 2
    /// when pairing).
    #[doc(hidden)]
    pub _trace_d: core::marker::PhantomData<[(); D]>,
}

/// Pack two `BinaryPoly<D>` cells (D ≤ 32) into one
/// `BinaryPoly<PACKED_STORAGE_WIDTH>`: low u32 = `lo`, high u32 = `hi`.
///
/// The pairing is bit-parallel — XOR / permutation on the packed
/// value separates the two halves coordinate-wise, so the F_2-linear
/// encoder produces `(encode(lo), encode(hi))` packed identically.
#[inline]
pub fn pair_cells<const D: usize>(
    lo: &BinaryPoly<D>,
    hi: Option<&BinaryPoly<D>>,
) -> BinaryPoly<PACKED_STORAGE_WIDTH> {
    use zinc_poly::univariate::F2PackU64;
    debug_assert!(D <= 32, "pair_cells requires trace D ≤ 32; got D = {D}");
    debug_assert_eq!(
        2 * D,
        PACKED_STORAGE_WIDTH,
        "pair_cells: trace D ({D}) must equal PACKED_STORAGE_WIDTH/2 ({})",
        PACKED_STORAGE_WIDTH / 2,
    );
    let lo_u = lo.pack_u64() & 0xFFFF_FFFFu64;
    let hi_u = hi.map(|h| h.pack_u64() & 0xFFFF_FFFFu64).unwrap_or(0);
    BinaryPoly::<PACKED_STORAGE_WIDTH>::unpack_u64(lo_u | (hi_u << 32))
}

/// Split a packed `BinaryPoly<PACKED_STORAGE_WIDTH>` back into two
/// `BinaryPoly<D>` cells (lo first, then hi).
#[inline]
pub fn unpair_cell<const D: usize>(
    packed: &BinaryPoly<PACKED_STORAGE_WIDTH>,
) -> (BinaryPoly<D>, BinaryPoly<D>) {
    use zinc_poly::univariate::F2PackU64;
    debug_assert!(D <= 32, "unpair_cell requires trace D ≤ 32; got D = {D}");
    let p = packed.pack_u64();
    let lo = BinaryPoly::<D>::unpack_u64(p & 0xFFFF_FFFFu64);
    let hi = BinaryPoly::<D>::unpack_u64(p >> 32);
    (lo, hi)
}

/// Public wrapper around [`pair_trace_polys`] so per-region benches
/// can stage the pairing pre-step without going through the full
/// commit pipeline. Not part of the verified protocol surface.
pub fn pair_trace_polys_pub<const D: usize>(
    trace_binary_cols: &[DenseMultilinearExtension<BinaryPoly<D>>],
) -> Vec<DenseMultilinearExtension<BinaryPoly<PACKED_STORAGE_WIDTH>>> {
    pair_trace_polys::<D>(trace_binary_cols)
}

/// Filtered variant of [`pair_trace_polys_pub`] that produces the
/// SAME paired-witness-primary slice that
/// [`ZincPlusPiopF2::commit_f2_trace_with_virtuals`] commits — i.e.
/// excludes public cols and [`F2BitOpVirtualSpec`] targets. Per-region
/// benches that want to time the encode/scatter/Merkle path on the
/// *real* commit shape (rather than the inflated full-trace shape)
/// should call this instead of [`pair_trace_polys_pub`].
///
/// `num_pub_bin` is the number of public binary_poly cols, typically
/// `U::signature().public_cols().num_binary_poly_cols()`. Caller
/// supplies it so this helper stays free of the `U: Uair` bound.
pub fn pair_primary_witness_polys_pub<const D: usize>(
    trace_binary_cols: &[DenseMultilinearExtension<BinaryPoly<D>>],
    num_pub_bin: usize,
    bit_op_specs: &[F2BitOpVirtualSpec],
) -> Vec<DenseMultilinearExtension<BinaryPoly<PACKED_STORAGE_WIDTH>>> {
    let num_wit_bin = trace_binary_cols.len() - num_pub_bin;
    let witness_bit_op_specs: Vec<F2BitOpVirtualSpec> = bit_op_specs
        .iter()
        .map(|s| F2BitOpVirtualSpec {
            col_idx: s.col_idx - num_pub_bin,
            source_col_idx: s.source_col_idx - num_pub_bin,
            op: s.op,
        })
        .collect();
    let primary_witness_indices =
        primary_col_indices_from_virtuals(num_wit_bin, &witness_bit_op_specs);
    let witness_slice = &trace_binary_cols[num_pub_bin..];
    let primary_cols: Vec<DenseMultilinearExtension<BinaryPoly<D>>> = primary_witness_indices
        .iter()
        .map(|&i| witness_slice[i].clone())
        .collect();
    pair_trace_polys::<D>(&primary_cols)
}

/// Materialise the bit-op virtual columns from the primary trace.
/// Each [`F2BitOpVirtualSpec`] produces one MLE column at the
/// protocol index `col_idx`; the result is keyed by `col_idx` for
/// easy splicing into the full trace order.
///
/// Used to verify the `trace_binary_cols` passed in by the caller
/// is consistent with the declared specs (sanity test for protocol
/// users that already materialised the full trace).
#[allow(clippy::arithmetic_side_effects)]
pub fn materialize_f2_bit_op_virtual_cols<const D: usize>(
    primary_or_full_trace: &[DenseMultilinearExtension<BinaryPoly<D>>],
    bit_op_specs: &[F2BitOpVirtualSpec],
) -> Vec<(usize, DenseMultilinearExtension<BinaryPoly<D>>)> {
    use zinc_poly::univariate::F2PackU64;
    bit_op_specs
        .iter()
        .map(|spec| {
            let source = &primary_or_full_trace[spec.source_col_idx];
            let evals: Vec<BinaryPoly<D>> = source
                .evaluations
                .iter()
                .map(|c| {
                    let v = apply_bit_op_u32(c.pack_u64(), spec.op);
                    BinaryPoly::<D>::unpack_u64(v)
                })
                .collect();
            (
                spec.col_idx,
                DenseMultilinearExtension::from_evaluations_vec(
                    source.num_vars,
                    evals,
                    BinaryPoly::default(),
                ),
            )
        })
        .collect()
}

/// Pair an even-length-or-shorter slice of trace MLEs into half-as-many
/// packed MLEs, each cell holding `(col_{2k}, col_{2k+1})` (low / high
/// u32). If the input length is odd, the last packed MLE pairs the
/// final column with zero in the high half.
fn pair_trace_polys<const D: usize>(
    trace_binary_cols: &[DenseMultilinearExtension<BinaryPoly<D>>],
) -> Vec<DenseMultilinearExtension<BinaryPoly<PACKED_STORAGE_WIDTH>>> {
    use zinc_poly::univariate::F2PackU64;
    let n = trace_binary_cols.len();
    let pairs = n.div_ceil(2);
    let mut paired: Vec<DenseMultilinearExtension<BinaryPoly<PACKED_STORAGE_WIDTH>>> =
        Vec::with_capacity(pairs);
    for k in 0..pairs {
        let lo_idx = 2 * k;
        let hi_idx = 2 * k + 1;
        let lo_poly = &trace_binary_cols[lo_idx];
        let hi_poly_opt = trace_binary_cols.get(hi_idx);
        let evals: Vec<BinaryPoly<PACKED_STORAGE_WIDTH>> = (0..lo_poly.evaluations.len())
            .map(|i| {
                let lo = &lo_poly.evaluations[i];
                let hi = hi_poly_opt.map(|h| &h.evaluations[i]);
                pair_cells::<D>(lo, hi)
            })
            .collect();
        paired.push(DenseMultilinearExtension::from_evaluations_vec(
            lo_poly.num_vars,
            evals,
            BinaryPoly::<PACKED_STORAGE_WIDTH>::unpack_u64(0),
        ));
    }
    paired
}

/// Errors emitted by [`ZincPlusPiopF2::verify_f2_open`].
#[derive(Debug, thiserror::Error)]
pub enum F2OpenError {
    #[error(
        "evaluation-consistency check failed: Σ_i q_0'[i] · b'[i] ≠ a' in F_2[X]"
    )]
    EvalConsistency,
    #[error(
        "lift discharge failed: ψ_α(a') ({computed:?}) ≠ Σ_g γ_g · a_g ({expected:?})"
    )]
    LiftDischarge {
        computed: BinaryFieldGF128,
        expected: BinaryFieldGF128,
    },
    #[error("F2OpenProof.b_vector has length {got}, expected {expected}")]
    BvecLenMismatch { expected: usize, got: usize },
    #[error(
        "coherence check failed: <combined_row', q_1'> ≠ <coeffs, b'> in F_2[X]"
    )]
    Coherence,
    #[error("F2OpenProof.combined_row has length {got}, expected {expected}")]
    CombinedRowLenMismatch { expected: usize, got: usize },
    #[error("Merkle path verification failed for opened column {column_idx}: {reason}")]
    MerkleVerify { column_idx: usize, reason: String },
    #[error(
        "encoding consistency check failed at opened col idx j={column_idx}: \
         encode(combined_row')[j] ≠ Σ_i coeffs[i] · Σ_g γ_g · cw_M^g[i, j]"
    )]
    EncodingConsistency { column_idx: usize },
    #[error("F2OpenedColumn has {got} entries, expected {expected}")]
    ColumnValuesLenMismatch { expected: usize, got: usize },
}

/// Absorb a slice of `BinaryF2Poly<W>` into the transcript.
/// Pack all entries' `W × u64` words contiguously into a single
/// buffer and absorb in one `absorb_slice` call. Each `absorb_slice`
/// carries framing bytes (0x6 / 0x7) and a Blake3 `update`; batching
/// avoids paying that overhead per 8-byte word (for SHA-256
/// `row_len ≈ 5500`, `combined_row` is ~5500·W=7 = 38.5k words —
/// the old per-word loop made that many `absorb_slice` calls; the
/// batched form makes one). On an empty input we early-return so
/// the transcript state matches the old loop on zero-element inputs.
///
/// Public so per-region benches can exercise the open phase's
/// sub-steps independently; not part of the verified protocol surface.
pub fn absorb_f2_poly_slice<'a, const W: usize, I>(transcript: &mut impl Transcript, iter: I)
where
    I: IntoIterator<Item = &'a BinaryF2Poly<W>>,
{
    // Two-pass: collect a flat byte buffer, then one absorb_slice.
    // The intermediate `Vec` is small (W · 8 ≤ 80 bytes per entry).
    let polys: Vec<&BinaryF2Poly<W>> = iter.into_iter().collect();
    if polys.is_empty() {
        // Match the original per-word loop's transcript state on
        // empty input (no framing bytes absorbed).
        return;
    }
    let mut buf = Vec::with_capacity(polys.len() * W * 8);
    for p in &polys {
        for w in p.words() {
            buf.extend_from_slice(&w.to_le_bytes());
        }
    }
    transcript.absorb_slice(&buf);
}

/// Squeeze a u64 challenge from the transcript and reduce modulo
/// `codeword_len`. `codeword_len` is a power of two in all Zip+
/// instantiations, so the modular reduction is bias-free.
///
/// Public so per-region benches can drive the per-opening loop in
/// isolation; not part of the verified protocol surface.
pub fn sample_column_idx(transcript: &mut impl Transcript, codeword_len: usize) -> usize {
    assert!(
        codeword_len.is_power_of_two(),
        "sample_column_idx requires power-of-two codeword length; got {codeword_len}",
    );
    let raw: u64 = transcript.get_challenge();
    #[allow(clippy::arithmetic_side_effects)]
    let idx = (raw as usize) & (codeword_len - 1);
    idx
}

/// Build `(q_0, q_1)` over `GF(2^128)` then lift each entry to
/// `BinaryF2Poly<2>` *via the α-dependent inverse lift*. Mirrors
/// `zip-plus`'s `point_to_tensor` split convention: `q_0` has length
/// `num_rows` (built from the last `log2(num_rows)` entries of
/// `point`); `q_1` has length `row_len = 2^{point.len() -
/// log2(num_rows)}` (built from the preceding entries).
///
/// **Why the inverse lift** (not the canonical-representative lift):
/// the verifier checks `ψ_α(a') = a`, where `ψ_α` evaluates an
/// F_2[X] polynomial at α via `Σ p_i α^i`. For the lifted claim
///
/// ```text
/// a' = q_1'^T · M_w · q_2'   in F_2[X]
/// ```
///
/// to satisfy `ψ_α(a') = q_1^T · ψ_α(M_w) · q_2 = a`, we need
/// `ψ_α(q_i'[k]) = q_i[k]` per entry. The canonical bit-pattern
/// representative satisfies that *only* when α is the field's
/// quotient generator `X` (mod P); for a transcript-fresh α the
/// inverse lift solves `Σ_j c_j · α^j = q_i[k]` for the unique
/// coefficient vector `c ∈ F_2^{128}` and returns
/// `q_i'[k] = Σ_j c_j X^j`. See `AlphaPolyBasis` in
/// [`binary_gf128`](zinc_poly::univariate::binary_gf128) for the
/// linear-algebra detail.
///
/// `basis` is the precomputed lift table (one per α, shared across
/// all `q_i[k]` entries to amortise the 128×128 F_2 matrix inverse).
///
/// Public so per-region benches can time `q0`/`q1` construction in
/// isolation; not part of the verified protocol surface.
#[allow(clippy::type_complexity)]
pub fn build_lifted_eq_tensor(
    num_rows: usize,
    point: &[BinaryFieldGF128],
    basis: &zinc_poly::univariate::binary_gf128::AlphaPolyBasis,
) -> (Vec<BinaryF2Poly<2>>, Vec<BinaryF2Poly<2>>) {
    assert!(num_rows.is_power_of_two());
    let split = point.len() - (num_rows.ilog2() as usize);
    let (hi, lo) = point.split_at(split);
    let field_cfg = ();
    let q0_gf = if !lo.is_empty() {
        zinc_poly::utils::build_eq_x_r_vec(lo, &field_cfg)
            .expect("build_eq_x_r_vec on lo")
    } else {
        vec![BinaryFieldGF128::one()]
    };
    let q1_gf = if !hi.is_empty() {
        zinc_poly::utils::build_eq_x_r_vec(hi, &field_cfg)
            .expect("build_eq_x_r_vec on hi")
    } else {
        vec![BinaryFieldGF128::one()]
    };
    // q0 is small (≤ num_rows entries; typically 8 for SHA F_2): keep
    // sequential. q1 is large (≤ 2^{num_vars - log2(num_rows)} entries;
    // ~524 K at nvars=22) and `basis.lift` is a 128×128 F_2
    // matrix-vec multiply per call (~600 ops); parallelise it so this
    // step doesn't single-thread the rest of Open.
    let q0: Vec<BinaryF2Poly<2>> = q0_gf.iter().map(|g| basis.lift(g)).collect();
    let q1: Vec<BinaryF2Poly<2>> = cfg_iter!(q1_gf).map(|g| basis.lift(g)).collect();
    (q0, q1)
}

impl<Zt, U, const D: usize> ZincPlusPiopF2<Zt, U, D>
where
    Zt: F2ZincTypes<D>,
    U: Uair + 'static,
{
    /// Step 7 (prove) — γ-batched lift-and-project open.
    ///
    /// Reduces the per-column MLE evaluation claims emitted by the
    /// sumcheck to a single F_2[X] opening: the prover folds each
    /// per-column `(a_g', b_g, combined_row_g)` triple by random
    /// challenges `γ_g ∈ GF(2^128)` into the bundled
    /// `(a', b', combined_row')` carried in [`F2OpenProof`]. The
    /// verifier discharges every column's claim via a single
    /// `ψ_α(a') = Σ_g γ_g · a_g` check; soundness over per-column
    /// tampering follows by Schwartz-Zippel over the γ_g.
    ///
    /// `trace_binary_cols` provides direct witness access (the
    /// `M_{w_g}` matrices); `pp` defines the commit shape (matching
    /// the codeword matrix stored in `commit_hint`); `sumcheck_point`
    /// is `r*` from the sumcheck output. `num_column_openings`
    /// controls the proximity soundness — see
    /// [`zip_plus::code::raa_f2::recommended_num_column_openings`].
    pub fn prove_f2_open(
        transcript: &mut impl Transcript,
        pp: &ZipPlusParams<Zt::BinaryZt, Zt::BinaryLc>,
        commit_hint: &ZipPlusHint<BinaryPoly<PACKED_STORAGE_WIDTH>>,
        trace_binary_cols: &[DenseMultilinearExtension<BinaryPoly<D>>],
        sumcheck_point: &[BinaryFieldGF128],
        alpha: &BinaryFieldGF128,
        num_column_openings: usize,
    ) -> F2OpenProof<D> {
        let num_rows = pp.num_rows;
        let row_len = pp.linear_code.row_len();
        let codeword_len = pp.linear_code.codeword_len();
        let num_cols = trace_binary_cols.len();
        assert!(num_rows.is_power_of_two());

        let basis = zinc_poly::univariate::binary_gf128::AlphaPolyBasis::new(alpha);
        let (q0, q1) = build_lifted_eq_tensor(num_rows, sumcheck_point, &basis);
        debug_assert_eq!(q1.len(), row_len);
        debug_assert_eq!(q0.len(), num_rows);

        // -- Step 7.1: γ_g challenges (one per committed column) ---
        // Drawn early so a', b', combined_row' depend on γ. (γ is
        // the only cross-column entropy in the open; coeffs comes
        // later for proximity.)
        let gamma_gf: Vec<BinaryFieldGF128> =
            transcript.get_field_challenges(num_cols, &());
        let gamma: Vec<BinaryF2Poly<2>> =
            gamma_gf.iter().map(|g| basis.lift(g)).collect();

        // -- Step 7.2: per-column intermediates, folded into b'/a'.
        //
        // For each committed column g:
        //   b_g[i] := Σ_j q_1'[j] · M_w_g[i, j]      (BinaryF2Poly<3>)
        //   a_g'   := Σ_i q_0'[i] · b_g[i]           (BinaryF2Poly<5>)
        //   b'[i] += γ_g · b_g[i]                    (BinaryF2Poly<5>)
        //   a'    += γ_g · a_g'                      (BinaryF2Poly<7>)
        //
        // Parallel structure: each (column, row) pair contributes
        // independently to `b'` and (via `a_g'`) to `a'`. We
        // parallelise across columns, producing per-column
        // partial `(b_g, a_g_scaled)` results, then merge serially.
        // The per-row work within a column stays sequential (row_len
        // is small in the deployed shape).
        let per_col_results: Vec<(Vec<BinaryF2Poly<5>>, BinaryF2Poly<7>)> =
            cfg_iter!(trace_binary_cols)
                .enumerate()
                .map(|(g, col)| {
                    assert_eq!(
                        col.evaluations.len(),
                        num_rows * row_len,
                        "trace column evaluation count must equal num_rows × row_len"
                    );

                    let mut b_g_scaled: Vec<BinaryF2Poly<5>> =
                        Vec::with_capacity(num_rows);
                    let mut b_g: Vec<BinaryF2Poly<3>> = Vec::with_capacity(num_rows);
                    for i in 0..num_rows {
                        let row_slice =
                            &col.evaluations[i * row_len..(i + 1) * row_len];
                        let row_lifted: Vec<BinaryF2Poly<1>> = row_slice
                            .iter()
                            .map(
                                zinc_poly::univariate::binary_gf128::lift_bp_to_f2_poly_1::<D>,
                            )
                            .collect();
                        let entry: BinaryF2Poly<3> =
                            zinc_poly::univariate::binary_f2_wide::f2_inner_product::<1, 2, 3>(
                                &row_lifted, &q1,
                            );
                        // γ_g · b_g[i], to be merged into b'[i].
                        let scaled: BinaryF2Poly<5> =
                            zinc_poly::univariate::binary_f2_wide::f2_poly_mul::<2, 3, 5>(
                                &gamma[g], &entry,
                            );
                        b_g_scaled.push(scaled);
                        b_g.push(entry);
                    }

                    let a_g_prime: BinaryF2Poly<5> =
                        zinc_poly::univariate::binary_f2_wide::f2_inner_product::<2, 3, 5>(
                            &q0, &b_g,
                        );
                    // γ_g · a_g', to be merged into a'.
                    let a_scaled: BinaryF2Poly<7> =
                        zinc_poly::univariate::binary_f2_wide::f2_poly_mul::<2, 5, 7>(
                            &gamma[g], &a_g_prime,
                        );
                    (b_g_scaled, a_scaled)
                })
                .collect();

        // Serial merge: ~`num_cols` u64 XORs per row. Cheap.
        let mut b_prime: Vec<BinaryF2Poly<5>> =
            vec![BinaryF2Poly::<5>::zero(); num_rows];
        let mut a_prime: BinaryF2Poly<7> = BinaryF2Poly::<7>::zero();
        for (b_g_scaled, a_scaled) in per_col_results {
            for i in 0..num_rows {
                b_prime[i] += b_g_scaled[i].clone();
            }
            a_prime += a_scaled;
        }

        // Absorb (b', a') into the transcript so subsequent challenges
        // depend on them.
        absorb_f2_poly_slice::<5, _>(transcript, b_prime.iter());
        absorb_f2_poly_slice::<7, _>(transcript, core::iter::once(&a_prime));

        // -- Step 7.3: proximity coefficients ----------------------
        let coeffs_gf: Vec<BinaryFieldGF128> =
            transcript.get_field_challenges(num_rows, &());
        let coeffs: Vec<BinaryF2Poly<2>> =
            coeffs_gf.iter().map(|g| basis.lift(g)).collect();

        // -- Step 7.4: combined_row' = Σ_g γ_g · (Σ_i coeffs[i] · M_w_g[i, *])
        //
        // Parallelise across the outer (g) loop: each column produces
        // an independent length-`row_len` contribution to
        // `combined_row`. Merge serially.
        let per_col_combined: Vec<Vec<BinaryF2Poly<5>>> = cfg_iter!(trace_binary_cols)
            .enumerate()
            .map(|(g, col)| {
                // Loop-reorder: outer over `i` (rows), inner over `j`
                // (codeword positions). The original (j outer, i inner)
                // order read `col.evaluations[i * row_len + j]` with a
                // `row_len`-stride between successive `i` values. At
                // nvars=20 `row_len = 131072` and `BinaryPoly<32>` is
                // 4 bytes, so the stride is 512 KB — well past what the
                // hardware prefetcher tracks and well past the per-row
                // working set the L1 can hold. Flipping the loops makes
                // the read pattern sequential within each row pass and
                // also keeps `coeffs[i]` in registers across the inner
                // `j` loop instead of reloading it every (i, j) pair.
                //
                // Accumulate per-`j` partials in `col_partial` (W=4),
                // then a final pass multiplies each by `γ_g` (W=7).
                // Same total work, dramatically better cache + register
                // behaviour at large `nvars`.
                let mut col_partial: Vec<BinaryF2Poly<3>> =
                    vec![BinaryF2Poly::<3>::zero(); row_len];
                for i in 0..num_rows {
                    let row_slice =
                        &col.evaluations[i * row_len..(i + 1) * row_len];
                    let coeff_i = &coeffs[i];
                    for j in 0..row_len {
                        let cell =
                            zinc_poly::univariate::binary_gf128::lift_bp_to_f2_poly_1::<D>(
                                &row_slice[j],
                            );
                        let prod: BinaryF2Poly<3> =
                            zinc_poly::univariate::binary_f2_wide::f2_poly_mul::<1, 2, 3>(
                                &cell, coeff_i,
                            );
                        col_partial[j] += prod;
                    }
                }
                // Apply `γ_g` in a separate pass over `col_partial`.
                // Reads stride 1, writes stride 1, kernel is identical
                // per `j` — easy to keep `gamma[g]` in registers.
                col_partial
                    .into_iter()
                    .map(|entry| {
                        zinc_poly::univariate::binary_f2_wide::f2_poly_mul::<2, 3, 5>(
                            &gamma[g], &entry,
                        )
                    })
                    .collect::<Vec<BinaryF2Poly<5>>>()
            })
            .collect();

        let mut combined_row: Vec<BinaryF2Poly<5>> =
            vec![BinaryF2Poly::<5>::zero(); row_len];
        for col_contrib in per_col_combined {
            for j in 0..row_len {
                combined_row[j] += col_contrib[j].clone();
            }
        }

        absorb_f2_poly_slice::<5, _>(transcript, combined_row.iter());

        // -- Step 7.5: sample column indices + Merkle opens --------
        //
        // Two-phase to expose parallelism. `sample_column_idx` is
        // transcript-driven and must run serially so its output
        // matches the verifier's mirror exactly. The per-opening
        // cell extraction + Merkle prove are independent of each
        // other, so once indices are pinned we `cfg_iter!` the
        // heavy work. Mirrors `verify_f2_open`'s structure.
        let column_indices: Vec<usize> = (0..num_column_openings)
            .map(|_| sample_column_idx(transcript, codeword_len))
            .collect();

        // Gather opened-column data directly from `commit_hint.cw_matrices`
        // (the row-major encoded matrices). Commit no longer materialises
        // `cw_columns`; reading on the fly avoids the 44 MB transpose
        // intermediate. The output layout matches the canonical leaf
        // layout used by `hash_grouped_leaf_from_row_major`:
        //   for l in 0..G: for m in 0..batch: for r in 0..num_rows:
        //     cw_matrices[m].data[r * codeword_len + group_start + l]
        let batch = commit_hint.cw_matrices.len();
        let col_len = batch * num_rows;
        let opened_columns: Vec<F2OpenedColumn<D>> = cfg_iter!(column_indices)
            .map(|&column_idx| {
                let group_idx = column_idx / LEAF_GROUP_SIZE;
                let group_start = group_idx * LEAF_GROUP_SIZE;
                let total = LEAF_GROUP_SIZE * col_len;
                let mut group_values: Vec<std::mem::MaybeUninit<BinaryPoly<PACKED_STORAGE_WIDTH>>> =
                    Vec::with_capacity(total);
                // SAFETY: every slot is initialised below before being read.
                unsafe { group_values.set_len(total); }
                // Cache-friendly read pattern: outer (m, r), inner l —
                // one cacheline of G ≤ 8 consecutive cells per (m, r).
                // Writes scatter across the small (≤ G * col_len * 8 ≈
                // 10 KB) destination buffer, which stays in L1.
                for m in 0..batch {
                    let m_offset = m * num_rows;
                    let mat_data = &commit_hint.cw_matrices[m].data;
                    for r in 0..num_rows {
                        let row_off = r * codeword_len + group_start;
                        let cells = &mat_data[row_off..row_off + LEAF_GROUP_SIZE];
                        for (l, cell) in cells.iter().enumerate() {
                            let position = l * col_len + m_offset + r;
                            // SAFETY: position < LEAF_GROUP_SIZE * col_len = total.
                            unsafe {
                                group_values
                                    .get_unchecked_mut(position)
                                    .write(cell.clone());
                            }
                        }
                    }
                }
                // SAFETY: every slot was written above.
                let group_values: Vec<BinaryPoly<PACKED_STORAGE_WIDTH>> = {
                    let mut v = core::mem::ManuallyDrop::new(group_values);
                    let (ptr, len, cap) = (v.as_mut_ptr(), v.len(), v.capacity());
                    unsafe {
                        Vec::from_raw_parts(
                            ptr.cast::<BinaryPoly<PACKED_STORAGE_WIDTH>>(),
                            len,
                            cap,
                        )
                    }
                };
                let merkle_proof = commit_hint
                    .merkle_tree
                    .prove(group_idx)
                    .expect("Merkle prove should succeed for in-range group idx");
                F2OpenedColumn {
                    column_idx,
                    column_values: group_values,
                    merkle_proof,
                    _trace_d: core::marker::PhantomData,
                }
            })
            .collect();

        F2OpenProof {
            lifted_claim: a_prime,
            b_vector: b_prime,
            combined_row,
            opened_columns,
        }
    }

    /// Step 7 (verify) — γ-batched lift-and-project verifier.
    ///
    /// Re-derives the γ_g challenges (one per primary committed
    /// column), runs the four core checks (eval-consistency, ψ_α
    /// discharge, coherence, encoding consistency), and verifies the
    /// Merkle paths for each opened column. Returns `Ok(())` iff all
    /// checks pass; the first failure short-circuits with a
    /// structured error.
    pub fn verify_f2_open(
        transcript: &mut impl Transcript,
        pp: &ZipPlusParams<Zt::BinaryZt, Zt::BinaryLc>,
        commitment: &ZipPlusCommitment,
        proof: &F2OpenProof<D>,
        subclaim: &F2VerifierSubclaim,
    ) -> Result<(), F2OpenError> {
        Self::verify_f2_open_with_virtuals(
            transcript, pp, commitment, proof, subclaim, &[],
        )
    }

    /// `verify_f2_open` variant aware of [`F2BitOpVirtualSpec`]s
    /// **and** the public/witness split. The open is
    /// **witness-only**: public columns aren't part of the Merkle
    /// commitment (verifier evaluates their MLEs locally), and
    /// bit-op virtuals are reconstructed verifier-side from their
    /// source's opened cells via `op`.
    pub fn verify_f2_open_with_virtuals(
        transcript: &mut impl Transcript,
        pp: &ZipPlusParams<Zt::BinaryZt, Zt::BinaryLc>,
        commitment: &ZipPlusCommitment,
        proof: &F2OpenProof<D>,
        subclaim: &F2VerifierSubclaim,
        bit_op_specs: &[F2BitOpVirtualSpec],
    ) -> Result<(), F2OpenError> {
        let num_rows = pp.num_rows;
        let row_len = pp.linear_code.row_len();
        let batch_size = commitment.batch_size;
        let codeword_len = pp.linear_code.codeword_len();
        // The F_2[X] open binds only the WITNESS primary (committed)
        // columns — public cols are never opened; XOR-/BitOp-virtual
        // cols are derived locally.
        let sig = U::signature();
        let num_pub_bin = sig.public_cols().num_binary_poly_cols();
        let num_wit_bin_total = subclaim
            .primary_column_evals
            .len()
            .checked_sub(num_pub_bin)
            .expect(
                "verify_f2_open_with_virtuals: \
                 subclaim.primary_column_evals must include public + witness primary cols",
            );
        let num_cols = num_wit_bin_total;
        let witness_primary_evals =
            &subclaim.primary_column_evals[num_pub_bin..num_pub_bin + num_cols];

        // -- Shape checks -----------------------------------------
        if proof.b_vector.len() != num_rows {
            return Err(F2OpenError::BvecLenMismatch {
                expected: num_rows,
                got: proof.b_vector.len(),
            });
        }
        if proof.combined_row.len() != row_len {
            return Err(F2OpenError::CombinedRowLenMismatch {
                expected: row_len,
                got: proof.combined_row.len(),
            });
        }

        // -- Re-derive γ_g ----------------------------------------
        let basis =
            zinc_poly::univariate::binary_gf128::AlphaPolyBasis::new(&subclaim.alpha);
        let (q0, q1) = build_lifted_eq_tensor(num_rows, &subclaim.sumcheck_point, &basis);
        let gamma_gf: Vec<BinaryFieldGF128> =
            transcript.get_field_challenges(num_cols, &());
        let gamma: Vec<BinaryF2Poly<2>> =
            gamma_gf.iter().map(|g| basis.lift(g)).collect();

        // Absorb (b', a') as the prover did.
        absorb_f2_poly_slice::<5, _>(transcript, proof.b_vector.iter());
        absorb_f2_poly_slice::<7, _>(
            transcript,
            core::iter::once(&proof.lifted_claim),
        );

        // -- Check 1: evaluation consistency in F_2[X] -------------
        //    Σ_i q_0[i] · b'[i]  =  a'.
        let recomputed_a_prime: BinaryF2Poly<7> = {
            let mut acc = BinaryF2Poly::<7>::zero();
            for i in 0..num_rows {
                let prod: BinaryF2Poly<7> =
                    zinc_poly::univariate::binary_f2_wide::f2_poly_mul::<2, 5, 7>(
                        &q0[i],
                        &proof.b_vector[i],
                    );
                acc += prod;
            }
            acc
        };
        if recomputed_a_prime != proof.lifted_claim {
            return Err(F2OpenError::EvalConsistency);
        }

        // -- Check 2: lift discharge in GF(2^128) ------------------
        //    ψ_α(a')  =  Σ_g γ_g_gf · a_g  (g ranges over WITNESS
        // primary cols only — public cols aren't in this batch).
        let psi = zinc_poly::univariate::binary_gf128::eval_f2_wide_poly_at::<7>(
            &proof.lifted_claim,
            &subclaim.alpha,
        );
        let mut expected = BinaryFieldGF128::zero();
        for g in 0..num_cols {
            let mut term = gamma_gf[g];
            term *= &witness_primary_evals[g];
            expected += &term;
        }
        if psi != expected {
            return Err(F2OpenError::LiftDischarge {
                computed: psi,
                expected,
            });
        }

        // -- Re-derive coeffs + absorb combined_row ---------------
        let coeffs_gf: Vec<BinaryFieldGF128> =
            transcript.get_field_challenges(num_rows, &());
        let coeffs: Vec<BinaryF2Poly<2>> =
            coeffs_gf.iter().map(|g| basis.lift(g)).collect();
        absorb_f2_poly_slice::<5, _>(transcript, proof.combined_row.iter());

        // -- Check 3: coherence ------------------------------------
        //    <combined_row', q_1>  =  <coeffs, b'>  in F_2[X]<10>.
        // `q1` (W=3) is ~96 set bits vs `combined_row` (W=7) at ~224
        // — pass the smaller as `a` to minimise schoolbook iterations.
        let lhs: BinaryF2Poly<7> = {
            let mut acc = BinaryF2Poly::<7>::zero();
            for j in 0..row_len {
                let prod: BinaryF2Poly<7> =
                    zinc_poly::univariate::binary_f2_wide::f2_poly_mul::<2, 5, 7>(
                        &q1[j],
                        &proof.combined_row[j],
                    );
                acc += prod;
            }
            acc
        };
        let rhs: BinaryF2Poly<7> = {
            let mut acc = BinaryF2Poly::<7>::zero();
            for i in 0..num_rows {
                let prod: BinaryF2Poly<7> =
                    zinc_poly::univariate::binary_f2_wide::f2_poly_mul::<2, 5, 7>(
                        &coeffs[i],
                        &proof.b_vector[i],
                    );
                acc += prod;
            }
            acc
        };
        if lhs != rhs {
            return Err(F2OpenError::Coherence);
        }

        // -- Check 4: per-column-opening encoding + Merkle ---------
        //
        // Witness-only encoding consistency:
        //   encode(combined_row)[j] = Σ_i coeffs[i] · Σ_{g witness} γ_g · cw_M^g[i, j]
        // where `combined_row` is the prover-sent witness-only batch.
        // Public columns are not part of this equation — the verifier
        // validates their MLE evals at r* upstream from `public_trace`
        // (in `verify_f2_full_with_bit_ops`), not via this open.
        let encoded: Vec<BinaryF2Poly<5>> = pp
            .linear_code
            .encode_f2_lin_open::<5>(&proof.combined_row);
        debug_assert_eq!(encoded.len(), codeword_len);

        // γ is witness-local: gamma[g] corresponds to the g-th
        // witness column (witness-local indexing).
        let num_wit_bin = num_cols;

        // Translate bit-op specs from PROTOCOL indices to
        // WITNESS-local indices for filtering primary witness cols.
        let witness_bit_op_specs: Vec<F2BitOpVirtualSpec> = bit_op_specs
            .iter()
            .map(|s| F2BitOpVirtualSpec {
                col_idx: s.col_idx - num_pub_bin,
                source_col_idx: s.source_col_idx - num_pub_bin,
                op: s.op,
            })
            .collect();

        // Primary witness indices (witness-local). These map 1-1 to
        // the paired storage slots in `opened.column_values`.
        let primary_witness_indices =
            primary_col_indices_from_virtuals(num_wit_bin_total, &witness_bit_op_specs);
        let num_primary_witness = primary_witness_indices.len();
        let paired_primary_count = num_primary_witness.div_ceil(2);
        if batch_size != paired_primary_count {
            return Err(F2OpenError::ColumnValuesLenMismatch {
                expected: paired_primary_count,
                got: batch_size,
            });
        }
        let single_col_len = paired_primary_count * num_rows;
        let expected_column_values_len = LEAF_GROUP_SIZE * single_col_len;

        // Precompute the bit-op virtual specs grouped by source col
        // index (WITNESS-local), so the encoding-consistency loop can
        // fan out the virtual contributions in O(1) per primary cell.
        let mut virtuals_by_source: Vec<Vec<F2BitOpVirtualSpec>> =
            vec![Vec::new(); num_wit_bin];
        for spec in &witness_bit_op_specs {
            virtuals_by_source[spec.source_col_idx].push(*spec);
        }

        // The column-opening loop is the single dominant cost of
        // `verify_f2_open` (~656 F_2[X]<3>·<1> mults per opened
        // column × `num_column_openings = 987` for rate-1/4 RAA).
        // Each iteration is independent of the others *except* for
        // the sequential `sample_column_idx` transcript draws. We
        // therefore pre-sample all expected indices serially first,
        // then parallelise the per-opening verification work.
        let expected_indices: Vec<usize> = (0..proof.opened_columns.len())
            .map(|_| sample_column_idx(transcript, codeword_len))
            .collect();

        cfg_iter!(proof.opened_columns)
            .zip(cfg_iter!(expected_indices))
            .try_for_each(|(opened, expected_idx)| {
                if opened.column_idx != *expected_idx {
                    return Err(F2OpenError::MerkleVerify {
                        column_idx: opened.column_idx,
                        reason: format!(
                            "column index mismatch: prover sent {}, transcript yields {}",
                            opened.column_idx, expected_idx,
                        ),
                    });
                }
                if opened.column_values.len() != expected_column_values_len {
                    return Err(F2OpenError::ColumnValuesLenMismatch {
                        expected: expected_column_values_len,
                        got: opened.column_values.len(),
                    });
                }

                // (a) Merkle path — recompute the leaf hash from the
                //     `LEAF_GROUP_SIZE` columns of the queried group
                //     and verify against the root.
                let group_idx = opened.column_idx / LEAF_GROUP_SIZE;
                let group_slices: Vec<&[BinaryPoly<PACKED_STORAGE_WIDTH>]> = (0
                    ..LEAF_GROUP_SIZE)
                    .map(|l| {
                        &opened.column_values[l * single_col_len..(l + 1) * single_col_len]
                    })
                    .collect();
                opened
                    .merkle_proof
                    .verify_grouped(&commitment.root, &group_slices, group_idx)
                    .map_err(|e| F2OpenError::MerkleVerify {
                        column_idx: opened.column_idx,
                        reason: format!("{e}"),
                    })?;

                // (b) γ-weighted encoding consistency on the queried
                //     column within the group (the verifier only needs
                //     to check the codeword cell at `column_idx`, not
                //     the other group members).
                //
                //  weighted_col[i] = Σ_g γ_g · cw_M^g[i, j]  ∈ F_2[X]<4>
                //  actual_at_j     = Σ_i coeffs[i] · weighted_col[i]  ∈ F_2[X]<7>
                //  expected_at_j   = encoded[j]  ∈ F_2[X]<7>
                //
                // Lift each packed cell straight from its raw u64 into
                // two `BinaryF2Poly<1>` halves (lo = col 2p, hi = col 2p+1)
                // without materialising an intermediate `BinaryPoly<D>` —
                // the canonical bit-pattern lift on `BinaryU64Poly` is
                // just a u64 copy.
                use zinc_poly::univariate::F2PackU64;
                let lo_mask: u64 = 0xFFFF_FFFFu64;
                let local_idx = opened.column_idx % LEAF_GROUP_SIZE;
                let local_col = &opened.column_values
                    [local_idx * single_col_len..(local_idx + 1) * single_col_len];
                let mut weighted_col: Vec<BinaryF2Poly<3>> =
                    vec![BinaryF2Poly::<3>::zero(); num_rows];

                // Witness primary + bit-op virtual contributions:
                // cells from the opened paired storage matrices.
                // γ is witness-local: gamma[wit_local_*] / gamma[spec.col_idx]
                // index witness cols directly (no public offset).
                // Public columns aren't in this batch; their MLE evals
                // at r* are validated upstream from `public_trace`.
                for p in 0..paired_primary_count {
                    let wit_local_lo = primary_witness_indices[2 * p];
                    let wit_local_hi_opt: Option<usize> =
                        primary_witness_indices.get(2 * p + 1).copied();
                    for i in 0..num_rows {
                        let packed_bits = local_col[p * num_rows + i].pack_u64();
                        let lo_cell = packed_bits & lo_mask;

                        // Primary lo contribution.
                        let lo_lifted = BinaryF2Poly::<1>::from_words([lo_cell]);
                        let prod_lo: BinaryF2Poly<3> =
                            zinc_poly::univariate::binary_f2_wide::f2_poly_mul::<1, 2, 3>(
                                &lo_lifted, &gamma[wit_local_lo],
                            );
                        weighted_col[i] += prod_lo;

                        // Virtual contributions sourced from this lo
                        // primary cell (witness-local source idx).
                        for spec in &virtuals_by_source[wit_local_lo] {
                            let v_cell = apply_bit_op_u32(lo_cell, spec.op);
                            let v_lifted = BinaryF2Poly::<1>::from_words([v_cell]);
                            let prod_v: BinaryF2Poly<3> =
                                zinc_poly::univariate::binary_f2_wide::f2_poly_mul::<1, 2, 3>(
                                    &v_lifted, &gamma[spec.col_idx],
                                );
                            weighted_col[i] += prod_v;
                        }

                        if let Some(wit_local_hi) = wit_local_hi_opt {
                            let hi_cell = packed_bits >> 32;
                            // Primary hi contribution.
                            let hi_lifted = BinaryF2Poly::<1>::from_words([hi_cell]);
                            let prod_hi: BinaryF2Poly<3> =
                                zinc_poly::univariate::binary_f2_wide::f2_poly_mul::<1, 2, 3>(
                                    &hi_lifted, &gamma[wit_local_hi],
                                );
                            weighted_col[i] += prod_hi;

                            // Virtual contributions sourced from this hi
                            // primary cell.
                            for spec in &virtuals_by_source[wit_local_hi] {
                                let v_cell = apply_bit_op_u32(hi_cell, spec.op);
                                let v_lifted = BinaryF2Poly::<1>::from_words([v_cell]);
                                let prod_v: BinaryF2Poly<3> =
                                    zinc_poly::univariate::binary_f2_wide::f2_poly_mul::<1, 2, 3>(
                                        &v_lifted, &gamma[spec.col_idx],
                                    );
                                weighted_col[i] += prod_v;
                            }
                        }
                    }
                }
                let actual_at_j: BinaryF2Poly<5> =
                    zinc_poly::univariate::binary_f2_wide::f2_inner_product::<2, 3, 5>(
                        &coeffs,
                        &weighted_col,
                    );
                if actual_at_j != encoded[opened.column_idx] {
                    return Err(F2OpenError::EncodingConsistency {
                        column_idx: opened.column_idx,
                    });
                }
                Ok(())
            })?;

        Ok(())
    }

    // -- Bundled commit + prove + open / verify entry points -----------
    //
    // [`Self::prove_f2_full`] / [`Self::verify_f2_full`] are the
    // single-call public API. They wrap Step 0 (commit + absorb) →
    // Steps 2-4 (IC + α + sumcheck) → Step 7 (γ-batched open) on a
    // single shared transcript, returning / consuming a single
    // [`F2FullProof`].

    /// Run the full F_2 prove pipeline on a single transcript.
    ///
    /// Composes commit → IC → α → sumcheck → open. The same
    /// transcript carries all Fiat-Shamir state across the four
    /// phases; the verifier mirror is [`Self::verify_f2_full`].
    ///
    /// `pp` defines the commit shape (`num_vars`, row layout,
    /// linear code). `num_vars` is the MLE arity for the witness
    /// trace. `project_scalar` lifts UAIR scalars from `U::Scalar`
    /// to `DynamicPolynomialF<GF(2^128)>` (typically the
    /// per-coefficient F_2 ⊂ GF(2^128) embedding).
    /// `num_column_openings` controls the proximity-check
    /// soundness — see
    /// [`zip_plus::code::raa_f2::recommended_num_column_openings`].
    pub fn prove_f2_full(
        transcript: &mut impl Transcript,
        pp: &ZipPlusParams<Zt::BinaryZt, Zt::BinaryLc>,
        trace: &UairTrace<'static, BinaryPoly<D>, BinaryPoly<D>, D>,
        virtual_specs: &[F2VirtualBpSpec],
        num_vars: usize,
        project_scalar: impl Fn(&U::Scalar) -> DynamicPolynomialF<BinaryFieldGF128> + Sync,
        num_column_openings: usize,
    ) -> Result<F2FullProof<D>, F2ProveError<U>> {
        Self::prove_f2_full_with_bit_ops(
            transcript,
            pp,
            trace,
            virtual_specs,
            &[],
            num_vars,
            project_scalar,
            num_column_openings,
        )
    }

    /// `prove_f2_full` variant aware of bit-op virtual columns.
    /// `bit_op_specs` declares which trace columns are bit-op virtual
    /// (derived from a primary source column via [`BitOp`] at the
    /// cell level). Those columns are not committed; their codeword
    /// cells are reconstructed by the verifier from the source's
    /// opened cells. See [`F2BitOpVirtualSpec`] for the soundness
    /// argument.
    pub fn prove_f2_full_with_bit_ops(
        transcript: &mut impl Transcript,
        pp: &ZipPlusParams<Zt::BinaryZt, Zt::BinaryLc>,
        trace: &UairTrace<'static, BinaryPoly<D>, BinaryPoly<D>, D>,
        virtual_specs: &[F2VirtualBpSpec],
        bit_op_specs: &[F2BitOpVirtualSpec],
        num_vars: usize,
        project_scalar: impl Fn(&U::Scalar) -> DynamicPolynomialF<BinaryFieldGF128> + Sync,
        num_column_openings: usize,
    ) -> Result<F2FullProof<D>, F2ProveError<U>> {
        // Step 0: commit primary cols + absorb root. Bit-op virtual
        // cols are skipped at commit time (cells are derivable
        // verifier-side from sources). F_2 linear-combo virtual cols
        // (`virtual_specs`) are materialised inline below for the IC
        // + sumcheck pass and reconstructed by the verifier from
        // primary col MLE evals at `r*`.
        let (hint, commitment) = Self::commit_and_absorb_f2_trace_with_virtuals(
            transcript,
            pp,
            &trace.binary_poly,
            bit_op_specs,
        )
        .expect("F_2 commit should succeed for a well-shaped trace");

        // Steps 2-7 (IC + α + sumcheck + multipoint-eval + open) are
        // shared with the pre-paired and Hadamard entry points; the
        // only thing that differs across them is the commit step
        // above. No Hadamard relations on this entry point.
        Self::prove_f2_full_impl(
            transcript,
            pp,
            trace,
            hint,
            commitment,
            virtual_specs,
            &[],
            num_vars,
            project_scalar,
            num_column_openings,
        )
    }

    /// `prove_f2_full_with_bit_ops` plus coefficient-wise Hadamard
    /// (`W = U ⊙ V`) relations, discharged **soundly**. `hadamard_specs`
    /// declares the relations between (absolute) trace column indices.
    ///
    /// On top of the standard commit + IC + α + sumcheck + multipoint-
    /// eval + open, this runs the Wiring-R Hadamard zerocheck (inside
    /// the IC/sumcheck phase, before α) and a **second** multipoint-eval
    /// + open at the Hadamard sumcheck point `r*_H`, which binds the
    /// per-column α-evals at `r*_H` to the commitment. The verifier's
    /// binding check then makes the trusted in-flow recombination sound.
    /// See [`F2FullProof`]'s `hadamard_*` fields and
    /// `protocol/src/f2_hadamard_plan.md` (§5.7).
    #[allow(clippy::too_many_arguments)]
    pub fn prove_f2_full_with_hadamard(
        transcript: &mut impl Transcript,
        pp: &ZipPlusParams<Zt::BinaryZt, Zt::BinaryLc>,
        trace: &UairTrace<'static, BinaryPoly<D>, BinaryPoly<D>, D>,
        virtual_specs: &[F2VirtualBpSpec],
        bit_op_specs: &[F2BitOpVirtualSpec],
        hadamard_specs: &[crate::f2_hadamard::F2HadamardSpec],
        num_vars: usize,
        project_scalar: impl Fn(&U::Scalar) -> DynamicPolynomialF<BinaryFieldGF128> + Sync,
        num_column_openings: usize,
    ) -> Result<F2FullProof<D>, F2ProveError<U>> {
        let (hint, commitment) = Self::commit_and_absorb_f2_trace_with_virtuals(
            transcript,
            pp,
            &trace.binary_poly,
            bit_op_specs,
        )
        .expect("F_2 commit should succeed for a well-shaped trace");

        Self::prove_f2_full_impl(
            transcript,
            pp,
            trace,
            hint,
            commitment,
            virtual_specs,
            hadamard_specs,
            num_vars,
            project_scalar,
            num_column_openings,
        )
    }

    /// Shared post-commit body for the `prove_f2_full*` entry points:
    /// IC + α + (optional Hadamard zerocheck) + sumcheck, the main
    /// multipoint-eval + open at `r*`/`r_0`, and — when
    /// `hadamard_specs` is non-empty — the second multipoint-eval +
    /// open at `r*_H`/`r_0^H` that makes the Hadamard discharge sound.
    /// The caller supplies the already-absorbed `(hint, commitment)`.
    #[allow(clippy::too_many_arguments)]
    fn prove_f2_full_impl(
        transcript: &mut impl Transcript,
        pp: &ZipPlusParams<Zt::BinaryZt, Zt::BinaryLc>,
        trace: &UairTrace<'static, BinaryPoly<D>, BinaryPoly<D>, D>,
        hint: ZipPlusHint<<Zt::BinaryZt as zip_plus::pcs::structs::ZipTypes>::Cw>,
        commitment: ZipPlusCommitment,
        virtual_specs: &[F2VirtualBpSpec],
        hadamard_specs: &[crate::f2_hadamard::F2HadamardSpec],
        num_vars: usize,
        project_scalar: impl Fn(&U::Scalar) -> DynamicPolynomialF<BinaryFieldGF128> + Sync,
        num_column_openings: usize,
    ) -> Result<F2FullProof<D>, F2ProveError<U>> {
        // Steps 2-4: IC + α + (Hadamard zerocheck, Wiring R) + sumcheck
        // on the primary+virtual extended trace.
        let (uair_proof, subclaim, projected_trace_for_mp) =
            Self::prove_f2_uair_with_groups(
                transcript,
                trace,
                virtual_specs,
                hadamard_specs,
                num_vars,
                project_scalar,
            )?;

        let sig = U::signature();
        let num_pub_bin = sig.public_cols().num_binary_poly_cols();

        // When discharging Hadamard relations we need the α-projected
        // trace again after the main open (for the second multipoint-
        // eval at r*_H). Clone it now — the original is consumed by the
        // main open_evals computation below. The clone is paid only on
        // the Hadamard path; the no-Hadamard fast path is byte-for-byte
        // the pre-Hadamard prover. (Optimisation — collapse only the
        // Hadamard subset, fold both points into one multipoint-eval —
        // is logged in documentation/f2x-sha-todo.md.)
        let projected_trace_for_hadamard: Option<
            Vec<DenseMultilinearExtension<<BinaryFieldGF128 as Field>::Inner>>,
        > = if hadamard_specs.is_empty() {
            None
        } else {
            Some(projected_trace_for_mp.clone())
        };

        // -- Step 4.5: Multipoint-eval reduction (r* -> r_0) ---------
        //
        // Reduces the per-column MLE evaluation claims at the
        // sumcheck point r* (carried in
        // `uair_proof.column_evals_at_rstar`) to a single evaluation
        // claim at a fresh random point r_0, batched across columns
        // via gamma. Row-shifted column evals (when they land) flow
        // through MultipointEval's `down_evals` + `shifts` params here.
        //
        // `projected_trace_for_mp` is the α-projected extended trace
        // surfaced by `prove_f2_uair_with_groups`, so no re-projection
        // is needed here.
        let (mp_proof, mp_prover_state) =
            MultipointEval::<BinaryFieldGF128>::prove_as_subprotocol(
                transcript,
                &projected_trace_for_mp,
                &subclaim.sumcheck_point,
                &uair_proof.column_evals_at_rstar,
                /* down_evals */ &[],
                /* shifts     */ &[],
                &(),
            )
            .map_err(F2ProveError::MultipointEval)?;
        let r_0 = mp_prover_state.eval_point;

        // Per-column MLE evals at r_0. Same pattern (consume each
        // projected col's inner Vec via the existing eval API) as
        // the column_evals_at_rstar computation inside
        // prove_f2_uair_with_groups. These are bound by the
        // multipoint-eval verifier consistency check at r_0; the
        // witness-primary slice is additionally bound by the F_2 PCS
        // open below.
        let open_evals_at_r_0: Vec<BinaryFieldGF128> = zinc_utils::cfg_into_iter!(
            projected_trace_for_mp
        )
        .map(|col| {
            <DenseMultilinearExtension<_> as zinc_poly::mle::MultilinearExtensionWithConfig<
                BinaryFieldGF128,
            >>::evaluate_with_config(col, &r_0, &())
            .expect("MLE evaluation at r_0 should succeed")
        })
        .collect();

        // Absorb open_evals_at_r_0 into the transcript (mirror of
        // the column_evals_at_rstar absorb done inside
        // prove_f2_uair_with_groups) so subsequent PCS-open
        // challenges depend on them — guards against a malicious
        // prover swapping the per-column open values after
        // multipoint-eval.
        let mut buf_r0 = vec![
            0u8;
            <<BinaryFieldGF128 as Field>::Inner
                as zinc_transcript::traits::ConstTranscribable>::NUM_BYTES
        ];
        for v in &open_evals_at_r_0 {
            transcript.absorb_random_field(v, &mut buf_r0);
        }

        // Step 7: γ-batched open over the WITNESS primary trace
        // slice, at the multipoint-eval output point r_0. Public
        // columns aren't committed; the verifier computes their
        // MLE evals at r_0 locally (matches the integer protocol's
        // witness-only PCS open). Bit-op virtual contributions are
        // reconstructed verifier-side via `bit_op_specs`.
        let open_proof = Self::prove_f2_open(
            transcript,
            pp,
            &hint,
            &trace.binary_poly[num_pub_bin..],
            &r_0,
            &subclaim.alpha,
            num_column_openings,
        );

        // -- Step 4.5'/7': Hadamard sound discharge (second mp+open) -
        //
        // Bind the Hadamard parent-evals (the per-column α-evals at the
        // Hadamard sumcheck point r*_H, shipped trusted in
        // `uair.hadamard_pair_evals`) to the commitment by opening at
        // r*_H. Collapse EVERY projected column's eval at r*_H to a
        // fresh point r_0^H via a second multipoint-eval, then open the
        // same contiguous witness slice at r_0^H. Collapsing all columns
        // (not just the Hadamard subset) reuses the witness-slice open
        // verbatim and dodges subset-column mapping; the parent-evals are
        // then a subset of the bound `evals_at_rstar_h`.
        let (
            hadamard_multipoint_eval,
            hadamard_evals_at_rstar_h,
            hadamard_open_evals_at_r0h,
            hadamard_open,
        ) = match projected_trace_for_hadamard {
            None => (None, Vec::new(), Vec::new(), None),
            Some(ptrace) => {
                let r_star_h = &subclaim.hadamard_rstar;

                // up_evals: every projected column's MLE eval at r*_H.
                // Clone per col (evaluate_with_config consumes) so
                // `ptrace` survives for the second multipoint-eval below.
                let evals_at_rstar_h: Vec<BinaryFieldGF128> = cfg_iter!(&ptrace)
                    .map(|col| {
                        <DenseMultilinearExtension<_>
                            as zinc_poly::mle::MultilinearExtensionWithConfig<
                                BinaryFieldGF128,
                            >>::evaluate_with_config(col.clone(), r_star_h, &())
                        .expect("MLE evaluation at r*_H should succeed")
                    })
                    .collect();

                // Absorb the up_evals BEFORE the second multipoint-eval
                // samples its γ challenges — exactly as
                // `prove_f2_uair_with_groups` absorbs `column_evals_at_rstar`
                // before the main mp. Soundness needs this: otherwise a
                // prover seeing γ first could fit fake per-column evals at
                // r*_H to the γ-batched sum while keeping the witness slice
                // honest, defeating the parent-eval binding. With the
                // absorb, Schwartz-Zippel over the now-fresh γ pins every
                // up_eval to the true committed value.
                let mut buf_rstar_h = vec![
                    0u8;
                    <<BinaryFieldGF128 as Field>::Inner
                        as zinc_transcript::traits::ConstTranscribable>::NUM_BYTES
                ];
                for v in &evals_at_rstar_h {
                    transcript.absorb_random_field(v, &mut buf_rstar_h);
                }

                let (mp2_proof, mp2_state) =
                    MultipointEval::<BinaryFieldGF128>::prove_as_subprotocol(
                        transcript,
                        &ptrace,
                        r_star_h,
                        &evals_at_rstar_h,
                        /* down_evals */ &[],
                        /* shifts     */ &[],
                        &(),
                    )
                    .map_err(F2ProveError::MultipointEval)?;
                let r_0_h = mp2_state.eval_point;

                let open_evals_at_r0h: Vec<BinaryFieldGF128> =
                    zinc_utils::cfg_into_iter!(ptrace)
                        .map(|col| {
                            <DenseMultilinearExtension<_>
                                as zinc_poly::mle::MultilinearExtensionWithConfig<
                                    BinaryFieldGF128,
                                >>::evaluate_with_config(col, &r_0_h, &())
                            .expect("MLE evaluation at r_0^H should succeed")
                        })
                        .collect();

                let mut buf_r0h = vec![
                    0u8;
                    <<BinaryFieldGF128 as Field>::Inner
                        as zinc_transcript::traits::ConstTranscribable>::NUM_BYTES
                ];
                for v in &open_evals_at_r0h {
                    transcript.absorb_random_field(v, &mut buf_r0h);
                }

                let open2 = Self::prove_f2_open(
                    transcript,
                    pp,
                    &hint,
                    &trace.binary_poly[num_pub_bin..],
                    &r_0_h,
                    &subclaim.alpha,
                    num_column_openings,
                );

                (
                    Some(mp2_proof),
                    evals_at_rstar_h,
                    open_evals_at_r0h,
                    Some(open2),
                )
            }
        };

        Ok(F2FullProof {
            commitment,
            uair: uair_proof,
            multipoint_eval: mp_proof,
            open_evals_at_r_0,
            open: open_proof,
            hadamard_multipoint_eval,
            hadamard_evals_at_rstar_h,
            hadamard_open_evals_at_r0h,
            hadamard_open,
        })
    }

    /// `prove_f2_full_with_bit_ops` variant that takes a **pre-paired**
    /// witness slice. Use this when pairing has been computed once
    /// during witness generation (e.g. cached inside the prover
    /// fixture) and amortised across multiple proves of the same
    /// trace. The transcript trajectory is identical to
    /// [`Self::prove_f2_full_with_bit_ops`] — same root absorb, same
    /// public-column absorb, same downstream Fiat-Shamir state — so
    /// proofs produced by either entry point verify against the same
    /// `verify_f2_full_with_bit_ops`.
    ///
    /// `paired_primary_witness` must be the output of
    /// [`pair_primary_witness_polys_pub`] applied to `trace.binary_poly`
    /// with the same `bit_op_specs` list passed here; the caller is
    /// responsible for keeping the two in sync (an inconsistency would
    /// produce a commit over the wrong cols and fail PCS spot-checks at
    /// verify time).
    pub fn prove_f2_full_pre_paired_with_bit_ops(
        transcript: &mut impl Transcript,
        pp: &ZipPlusParams<Zt::BinaryZt, Zt::BinaryLc>,
        trace: &UairTrace<'static, BinaryPoly<D>, BinaryPoly<D>, D>,
        paired_primary_witness: &[DenseMultilinearExtension<BinaryPoly<PACKED_STORAGE_WIDTH>>],
        virtual_specs: &[F2VirtualBpSpec],
        bit_op_specs: &[F2BitOpVirtualSpec],
        num_vars: usize,
        project_scalar: impl Fn(&U::Scalar) -> DynamicPolynomialF<BinaryFieldGF128> + Sync,
        num_column_openings: usize,
    ) -> Result<F2FullProof<D>, F2ProveError<U>> {
        let sig = U::signature();
        let num_pub_bin = sig.public_cols().num_binary_poly_cols();

        // Step 0: commit pre-paired witness + absorb root + public cols.
        let (hint, commitment) = Self::commit_and_absorb_pre_paired_witness(
            transcript,
            pp,
            paired_primary_witness,
            &trace.binary_poly[..num_pub_bin],
        )
        .expect("F_2 pre-paired commit should succeed for a well-shaped trace");

        let _ = bit_op_specs; // currently unused after the commit step;
                              // kept in the signature to mirror the
                              // trace-pairing variant.

        // Steps 2-7 are identical to the trace-pairing variant — only
        // the commit above differs. No Hadamard relations on this entry
        // point.
        Self::prove_f2_full_impl(
            transcript,
            pp,
            trace,
            hint,
            commitment,
            virtual_specs,
            &[],
            num_vars,
            project_scalar,
            num_column_openings,
        )
    }

    /// Verifier mirror of [`Self::prove_f2_full`]: absorbs the
    /// commitment, runs the IC + sumcheck verifier, then the
    /// γ-batched open verifier — all on a single shared transcript.
    /// `virtual_specs` must match what the prover used; the
    /// verifier derives virtual column MLE evals at `r*` from
    /// `primary_column_evals` via those specs and checks them
    /// against the sumcheck-extracted virtual evals.
    #[allow(clippy::too_many_arguments)]
    pub fn verify_f2_full<IdealOverF>(
        transcript: &mut impl Transcript,
        pp: &ZipPlusParams<Zt::BinaryZt, Zt::BinaryLc>,
        proof: &F2FullProof<D>,
        virtual_specs: &[F2VirtualBpSpec],
        public_binary_trace: &[DenseMultilinearExtension<BinaryPoly<D>>],
        num_vars: usize,
        num_primary_columns: usize,
        project_ideal: impl Fn(&IdealOrZero<U::Ideal>) -> IdealOverF,
    ) -> Result<F2VerifierSubclaim, F2FullVerifyError<U, IdealOverF>>
    where
        IdealOverF: zinc_uair::ideal::Ideal
            + zinc_uair::ideal::IdealCheck<DynamicPolynomialF<BinaryFieldGF128>>,
    {
        Self::verify_f2_full_with_bit_ops(
            transcript,
            pp,
            proof,
            virtual_specs,
            &[],
            public_binary_trace,
            num_vars,
            num_primary_columns,
            project_ideal,
        )
    }

    /// `verify_f2_full` variant aware of bit-op virtual columns and
    /// the public/witness split. `public_binary_trace` holds the public
    /// binary_poly columns the prover absorbed at commit time; the
    /// verifier absorbs the same bytes here and **computes the public
    /// columns' MLE evaluations at r* directly** (matching the integer
    /// protocol's pattern: public column claims are discharged by the
    /// verifier doing the MLE evaluation itself against `public_trace`,
    /// never via Zip+). The witness columns' MLE evals at r* are proved
    /// by the γ-batched open.
    /// See [`Self::prove_f2_full_with_bit_ops`].
    #[allow(clippy::too_many_arguments)]
    pub fn verify_f2_full_with_bit_ops<IdealOverF>(
        transcript: &mut impl Transcript,
        pp: &ZipPlusParams<Zt::BinaryZt, Zt::BinaryLc>,
        proof: &F2FullProof<D>,
        virtual_specs: &[F2VirtualBpSpec],
        bit_op_specs: &[F2BitOpVirtualSpec],
        public_binary_trace: &[DenseMultilinearExtension<BinaryPoly<D>>],
        num_vars: usize,
        num_primary_columns: usize,
        project_ideal: impl Fn(&IdealOrZero<U::Ideal>) -> IdealOverF,
    ) -> Result<F2VerifierSubclaim, F2FullVerifyError<U, IdealOverF>>
    where
        IdealOverF: zinc_uair::ideal::Ideal
            + zinc_uair::ideal::IdealCheck<DynamicPolynomialF<BinaryFieldGF128>>,
    {
        Self::verify_f2_full_impl(
            transcript,
            pp,
            proof,
            virtual_specs,
            bit_op_specs,
            &[],
            public_binary_trace,
            num_vars,
            num_primary_columns,
            project_ideal,
        )
    }

    /// `verify_f2_full_with_bit_ops` plus the sound Hadamard discharge.
    /// `hadamard_specs` must match the prover's. Runs the in-flow
    /// Hadamard zerocheck + recombination (inside the IC/sumcheck
    /// verifier) and then verifies the second multipoint-eval + open at
    /// `r*_H`/`r_0^H`, binding `uair.hadamard_pair_evals` to the
    /// commitment. See [`Self::prove_f2_full_with_hadamard`].
    #[allow(clippy::too_many_arguments)]
    pub fn verify_f2_full_with_hadamard<IdealOverF>(
        transcript: &mut impl Transcript,
        pp: &ZipPlusParams<Zt::BinaryZt, Zt::BinaryLc>,
        proof: &F2FullProof<D>,
        virtual_specs: &[F2VirtualBpSpec],
        bit_op_specs: &[F2BitOpVirtualSpec],
        hadamard_specs: &[crate::f2_hadamard::F2HadamardSpec],
        public_binary_trace: &[DenseMultilinearExtension<BinaryPoly<D>>],
        num_vars: usize,
        num_primary_columns: usize,
        project_ideal: impl Fn(&IdealOrZero<U::Ideal>) -> IdealOverF,
    ) -> Result<F2VerifierSubclaim, F2FullVerifyError<U, IdealOverF>>
    where
        IdealOverF: zinc_uair::ideal::Ideal
            + zinc_uair::ideal::IdealCheck<DynamicPolynomialF<BinaryFieldGF128>>,
    {
        Self::verify_f2_full_impl(
            transcript,
            pp,
            proof,
            virtual_specs,
            bit_op_specs,
            hadamard_specs,
            public_binary_trace,
            num_vars,
            num_primary_columns,
            project_ideal,
        )
    }

    /// Shared verifier body for the `verify_f2_full*` entry points.
    /// Mirrors `prove_f2_full_impl`: absorb commitment + public cols,
    /// run the IC + (optional Hadamard zerocheck) + sumcheck verifier,
    /// the public/virtual eval checks, the main multipoint-eval + open
    /// at `r*`/`r_0`, and — when `hadamard_specs` is non-empty — the
    /// second multipoint-eval + open at `r*_H`/`r_0^H` plus the
    /// parent-eval binding check that upgrades the trusted recombination
    /// to sound.
    #[allow(clippy::too_many_arguments)]
    fn verify_f2_full_impl<IdealOverF>(
        transcript: &mut impl Transcript,
        pp: &ZipPlusParams<Zt::BinaryZt, Zt::BinaryLc>,
        proof: &F2FullProof<D>,
        virtual_specs: &[F2VirtualBpSpec],
        bit_op_specs: &[F2BitOpVirtualSpec],
        hadamard_specs: &[crate::f2_hadamard::F2HadamardSpec],
        public_binary_trace: &[DenseMultilinearExtension<BinaryPoly<D>>],
        num_vars: usize,
        num_primary_columns: usize,
        project_ideal: impl Fn(&IdealOrZero<U::Ideal>) -> IdealOverF,
    ) -> Result<F2VerifierSubclaim, F2FullVerifyError<U, IdealOverF>>
    where
        IdealOverF: zinc_uair::ideal::Ideal
            + zinc_uair::ideal::IdealCheck<DynamicPolynomialF<BinaryFieldGF128>>,
    {
        // Step 0: absorb the commitment, then the public columns —
        // same order as the prover's
        // `commit_and_absorb_f2_trace_with_virtuals`.
        Self::absorb_commitment(transcript, &proof.commitment);
        crate::absorb_public_columns(transcript, public_binary_trace);

        // Steps 2-4: IC + α + (Hadamard zerocheck + recombination,
        // Wiring R) + sumcheck verifier (with XOR-virtual-col
        // consistency check baked in).
        let subclaim = Self::verify_f2_uair_with_groups(
            transcript,
            &proof.uair,
            virtual_specs,
            hadamard_specs,
            num_vars,
            num_primary_columns,
            project_ideal,
        )
        .map_err(F2FullVerifyError::Uair)?;

        // Public column MLE eval validation. The prover's
        // `column_evals_at_rstar[0..num_pub_bin]` claim values for
        // each public col's α-projected MLE at r*. Recompute them
        // from `public_binary_trace` and check — the only check
        // that binds these claimed evals to the actual public
        // input. (Witness col evals are bound by the Zip+ open.)
        //
        // Per-col cost (nvars=16, D=32): one α-projection pass over
        // 2^16 cells + one length-2^16 MLE eval at r*. Independent
        // across columns — parallelise the outer loop.
        let sig = U::signature();
        let num_pub_bin = sig.public_cols().num_binary_poly_cols();
        if num_pub_bin > 0 {
            assert_eq!(
                public_binary_trace.len(),
                num_pub_bin,
                "verify_f2_full_with_bit_ops: public_binary_trace.len() ({}) must \
                 match U::signature().public_cols().num_binary_poly_cols() ({})",
                public_binary_trace.len(),
                num_pub_bin,
            );
            let computed_evals = recompute_public_col_evals_at::<D>(
                public_binary_trace,
                &subclaim.alpha,
                &subclaim.sumcheck_point,
            );
            for (g, &computed) in computed_evals.iter().enumerate() {
                let claimed = subclaim.primary_column_evals[g];
                if computed != claimed {
                    return Err(F2FullVerifyError::PublicColumnEvalMismatch {
                        public_col_idx: g,
                        computed,
                        claimed,
                    });
                }
            }
        }

        // -- Step 4.5 verify: multipoint-eval reduction (r* -> r_0) --
        //
        // Mirror of the prover's
        // [`Self::prove_f2_full_with_bit_ops`] multipoint-eval phase.
        // Reduces the per-col MLE eval claims at r* (carried in
        // `proof.uair.column_evals_at_rstar`) to `r_0`, then checks
        // the prover-supplied `proof.open_evals_at_r_0` against the
        // reduction. The witness-primary slice of
        // `open_evals_at_r_0` is what the F_2 PCS open at r_0 binds.
        let mp_subclaim = MultipointEval::<BinaryFieldGF128>::verify_as_subprotocol(
            transcript,
            proof.multipoint_eval.clone(),
            &subclaim.sumcheck_point,
            &proof.uair.column_evals_at_rstar,
            /* down_evals */ &[],
            /* shifts     */ &[],
            num_vars,
            &(),
        )
        .map_err(F2FullVerifyError::MultipointEval)?;
        let r_0 = mp_subclaim.sumcheck_subclaim.point.clone();

        // Absorb open_evals_at_r_0 into transcript (mirror of the
        // prover's absorb).
        let mut buf_r0 = vec![
            0u8;
            <<BinaryFieldGF128 as Field>::Inner
                as zinc_transcript::traits::ConstTranscribable>::NUM_BYTES
        ];
        for v in &proof.open_evals_at_r_0 {
            transcript.absorb_random_field(v, &mut buf_r0);
        }

        // Length check: open_evals_at_r_0 must cover every column
        // multipoint-eval batched (primary + virtual), matching the
        // length of `uair.column_evals_at_rstar`.
        if proof.open_evals_at_r_0.len() != proof.uair.column_evals_at_rstar.len() {
            return Err(F2FullVerifyError::OpenEvalsLengthMismatch {
                got: proof.open_evals_at_r_0.len(),
                expected: proof.uair.column_evals_at_rstar.len(),
            });
        }

        // Public col MLE evals at r_0: validate by local
        // recomputation against `public_binary_trace`. Same shape as
        // the at-r* check above, just at the new point.
        if num_pub_bin > 0 {
            let computed_evals_r0 =
                recompute_public_col_evals_at::<D>(public_binary_trace, &subclaim.alpha, &r_0);
            for (g, &computed) in computed_evals_r0.iter().enumerate() {
                let claimed = proof.open_evals_at_r_0[g];
                if computed != claimed {
                    return Err(F2FullVerifyError::PublicColumnEvalMismatchAtR0 {
                        public_col_idx: g,
                        computed,
                        claimed,
                    });
                }
            }
        }

        // XOR-virtual cols: derive their MLE evals at r_0 from
        // primary evals at r_0 and check consistency with the
        // prover's claim (same pattern as the at-r* check inside
        // verify_f2_uair).
        let (primary_evals_at_r_0, virtual_evals_at_r_0_from_proof) =
            proof.open_evals_at_r_0.split_at(num_primary_columns);
        let virtual_evals_at_r_0_derived =
            derive_f2_virtual_evals_at(primary_evals_at_r_0, virtual_specs);
        for (k, (claimed, derived)) in virtual_evals_at_r_0_from_proof
            .iter()
            .zip(virtual_evals_at_r_0_derived.iter())
            .enumerate()
        {
            if claimed != derived {
                return Err(F2FullVerifyError::VirtualColumnEvalMismatchAtR0 {
                    virtual_idx: k,
                    derived: *derived,
                    claimed: *claimed,
                });
            }
        }

        // Finalise multipoint-eval check: verify the sumcheck-
        // reduction is internally consistent with `open_evals_at_r_0`.
        MultipointEval::<BinaryFieldGF128>::verify_subclaim(
            &mp_subclaim,
            &proof.open_evals_at_r_0,
            /* shifts */ &[],
            &(),
        )
        .map_err(F2FullVerifyError::MultipointEval)?;

        // Build a subclaim shifted to r_0 so the PCS open can be
        // verified at the new point. The fields verify_f2_open_with_
        // virtuals reads are: sumcheck_point, alpha, primary_column_
        // evals (sliced for witness), virtual_column_evals. The
        // ic_evaluation_point field is unused on that path.
        let subclaim_at_r_0 = F2VerifierSubclaim {
            ic_evaluation_point: subclaim.ic_evaluation_point.clone(),
            alpha: subclaim.alpha,
            sumcheck_point: r_0,
            primary_column_evals: primary_evals_at_r_0.to_vec(),
            virtual_column_evals: virtual_evals_at_r_0_derived,
            hadamard_rstar: Vec::new(),
            hadamard_pairs: Vec::new(),
        };

        // Step 7: γ-batched open verifier at r_0.
        Self::verify_f2_open_with_virtuals(
            transcript,
            pp,
            &proof.commitment,
            &proof.open,
            &subclaim_at_r_0,
            bit_op_specs,
        )
        .map_err(F2FullVerifyError::Open)?;

        // -- Hadamard sound discharge: second mp+open at r*_H ---------
        //
        // Mirror of the prover's second multipoint-eval + open in
        // `prove_f2_full_impl`. It binds every column's α-eval at the
        // Hadamard sumcheck point r*_H (`hadamard_evals_at_rstar_h`) to
        // the commitment, then checks the trusted in-flow parent-evals
        // (`uair.hadamard_pair_evals`, recombined inside
        // `verify_f2_uair_with_groups`) equal the bound subset — upgrading
        // the recombination from honest-prover-only to sound. Gated on
        // `hadamard_specs` exactly like the prover.
        if !hadamard_specs.is_empty() {
            let num_total = proof.uair.column_evals_at_rstar.len();
            let mp2_proof = proof
                .hadamard_multipoint_eval
                .as_ref()
                .ok_or(F2FullVerifyError::MissingHadamardDischarge)?;
            let open2 = proof
                .hadamard_open
                .as_ref()
                .ok_or(F2FullVerifyError::MissingHadamardDischarge)?;

            if proof.hadamard_evals_at_rstar_h.len() != num_total {
                return Err(F2FullVerifyError::HadamardEvalsAtRstarHLengthMismatch {
                    got: proof.hadamard_evals_at_rstar_h.len(),
                    expected: num_total,
                });
            }
            if proof.hadamard_open_evals_at_r0h.len() != num_total {
                return Err(F2FullVerifyError::HadamardOpenEvalsLengthMismatch {
                    got: proof.hadamard_open_evals_at_r0h.len(),
                    expected: num_total,
                });
            }

            // Absorb the up_evals at r*_H before the second mp samples
            // its γ — mirror of the prover; pins each up_eval via SZ.
            let mut buf_rstar_h = vec![
                0u8;
                <<BinaryFieldGF128 as Field>::Inner
                    as zinc_transcript::traits::ConstTranscribable>::NUM_BYTES
            ];
            for v in &proof.hadamard_evals_at_rstar_h {
                transcript.absorb_random_field(v, &mut buf_rstar_h);
            }

            // Second multipoint-eval reduction (r*_H -> r_0^H), with the
            // claimed per-column evals at r*_H as up_evals.
            let mp2_subclaim = MultipointEval::<BinaryFieldGF128>::verify_as_subprotocol(
                transcript,
                mp2_proof.clone(),
                &subclaim.hadamard_rstar,
                &proof.hadamard_evals_at_rstar_h,
                /* down_evals */ &[],
                /* shifts     */ &[],
                num_vars,
                &(),
            )
            .map_err(F2FullVerifyError::MultipointEval)?;
            let r_0_h = mp2_subclaim.sumcheck_subclaim.point.clone();

            // Absorb open_evals_at_r0h (mirror of the prover).
            let mut buf_r0h = vec![
                0u8;
                <<BinaryFieldGF128 as Field>::Inner
                    as zinc_transcript::traits::ConstTranscribable>::NUM_BYTES
            ];
            for v in &proof.hadamard_open_evals_at_r0h {
                transcript.absorb_random_field(v, &mut buf_r0h);
            }

            // Public col evals at r_0^H: validate by local recomputation.
            if num_pub_bin > 0 {
                let computed =
                    recompute_public_col_evals_at::<D>(public_binary_trace, &subclaim.alpha, &r_0_h);
                for (g, &c) in computed.iter().enumerate() {
                    let claimed = proof.hadamard_open_evals_at_r0h[g];
                    if c != claimed {
                        return Err(
                            F2FullVerifyError::HadamardPublicColumnEvalMismatchAtR0H {
                                public_col_idx: g,
                                computed: c,
                                claimed,
                            },
                        );
                    }
                }
            }

            // XOR-virtual cols at r_0^H: derive from primary evals and
            // check against the prover's claim.
            let (primary_evals_at_r_0_h, virtual_evals_at_r_0_h_from_proof) =
                proof.hadamard_open_evals_at_r0h.split_at(num_primary_columns);
            let virtual_evals_at_r_0_h_derived =
                derive_f2_virtual_evals_at(primary_evals_at_r_0_h, virtual_specs);
            for (k, (claimed, derived)) in virtual_evals_at_r_0_h_from_proof
                .iter()
                .zip(virtual_evals_at_r_0_h_derived.iter())
                .enumerate()
            {
                if claimed != derived {
                    return Err(
                        F2FullVerifyError::HadamardVirtualColumnEvalMismatchAtR0H {
                            virtual_idx: k,
                            derived: *derived,
                            claimed: *claimed,
                        },
                    );
                }
            }

            // Finalise the second multipoint-eval against open_evals_at_r0h.
            MultipointEval::<BinaryFieldGF128>::verify_subclaim(
                &mp2_subclaim,
                &proof.hadamard_open_evals_at_r0h,
                /* shifts */ &[],
                &(),
            )
            .map_err(F2FullVerifyError::MultipointEval)?;

            // Build a subclaim at r_0^H for the second open, then bind the
            // witness slice to the same commitment as the main open.
            let subclaim_at_r_0_h = F2VerifierSubclaim {
                ic_evaluation_point: subclaim.ic_evaluation_point.clone(),
                alpha: subclaim.alpha,
                sumcheck_point: r_0_h,
                primary_column_evals: primary_evals_at_r_0_h.to_vec(),
                virtual_column_evals: virtual_evals_at_r_0_h_derived,
                hadamard_rstar: Vec::new(),
                hadamard_pairs: Vec::new(),
            };
            Self::verify_f2_open_with_virtuals(
                transcript,
                pp,
                &proof.commitment,
                open2,
                &subclaim_at_r_0_h,
                bit_op_specs,
            )
            .map_err(F2FullVerifyError::Open)?;

            // -- Binding check (the soundness step) -------------------
            // The per-column evals at r*_H are now bound to the
            // commitment (witness via the open, public/virtual via the
            // local checks, all pinned individually via the up-eval
            // absorb + SZ). For each `Δ = 0` Hadamard pair, the trusted
            // pair-eval the in-flow recombination used must equal the bound
            // column eval at r*_H. `Δ ≠ 0` pairs are row-shifted and stay
            // trusted (the row-shift discharge gap, plan §6).
            // `subclaim.hadamard_pairs` is the sorted distinct `(col, Δ)`
            // list, aligned 1:1 with `uair.hadamard_pair_evals`.
            if proof.uair.hadamard_pair_evals.len() != subclaim.hadamard_pairs.len() {
                return Err(F2FullVerifyError::HadamardParentEvalCountMismatch {
                    got: proof.uair.hadamard_pair_evals.len(),
                    expected: subclaim.hadamard_pairs.len(),
                });
            }
            for (i, &(col, row_shift)) in subclaim.hadamard_pairs.iter().enumerate() {
                if row_shift != 0 {
                    continue; // row-shifted pair: trusted (row-shift gap).
                }
                let opened = proof.hadamard_evals_at_rstar_h[col];
                let pair_eval = proof.uair.hadamard_pair_evals[i];
                if opened != pair_eval {
                    return Err(F2FullVerifyError::HadamardParentEvalBindingMismatch {
                        distinct_col: col,
                        opened,
                        parent: pair_eval,
                    });
                }
            }
        }

        Ok(subclaim)
    }
}

/// The complete F_2 proof: commitment + IC/sumcheck proof +
/// multipoint-eval reduction + Step 7 γ-batched open. Produced by
/// [`ZincPlusPiopF2::prove_f2_full`] and consumed by
/// [`ZincPlusPiopF2::verify_f2_full`].
///
/// The multipoint-eval phase reduces the per-column MLE evaluation
/// claims at the sumcheck point `r*` (carried in
/// `uair.column_evals_at_rstar`) to a single evaluation claim at a
/// new point `r_0`, batched across columns. Today the F_2 sumcheck
/// surfaces no row-shifted column evals (no Hadamard product
/// support yet — see [`f2_native_ic`](crate::f2_native_ic)), so the
/// reduction is the degenerate one that only batches the "up" col
/// evals and rerandomises the open point from `r*` to `r_0`. When
/// Hadamard / K-discharge wire shifted-col evals into the
/// sumcheck output, multipoint-eval will absorb them via its
/// `shifts` input and fold them into the same `r_0` open.
///
/// `open_evals_at_r_0` carries the prover-supplied per-column MLE
/// evals at `r_0` (primary cols first, then virtual). The
/// multipoint-eval verifier check validates that they are
/// consistent with `uair.column_evals_at_rstar` via the sumcheck
/// reduction; the F_2 PCS open at `r_0` then binds the
/// witness-primary slice against the commitment.
#[derive(Clone, Debug)]
pub struct F2FullProof<const D: usize> {
    pub commitment: ZipPlusCommitment,
    pub uair: F2Proof,
    /// Sumcheck reducing column MLE eval claims at `r*` to a single
    /// open point `r_0`. Today's invocation passes no shifted-col
    /// inputs; the proof gains one degree-2 sumcheck over `num_vars`
    /// rounds.
    pub multipoint_eval: MultipointEvalProof<BinaryFieldGF128>,
    /// Per-column MLE evals at `r_0` (primary cols first, then
    /// virtual), bound by the multipoint-eval consistency check and
    /// — for the witness-primary slice — by [`Self::open`] below.
    pub open_evals_at_r_0: Vec<BinaryFieldGF128>,
    pub open: F2OpenProof<D>,

    // -- Hadamard sound-discharge (second mp+open at `r*_H`) ----------
    //
    // When the UAIR carries coefficient-wise `W = U ⊙ V` operand
    // relations, the Wiring-R Hadamard zerocheck (in `uair`) reduces them
    // to per-slice evals at the Hadamard sumcheck point `r*_H`, recombined
    // against operand parents derived from the **prover-supplied**
    // `uair.hadamard_pair_evals` (one per distinct `(col, Δ)` pair). The
    // `Δ = 0` pair evals are made *sound* by opening them at `r*_H` against
    // the commitment: a SECOND multipoint-eval collapses every projected
    // column's eval at `r*_H` to a fresh point `r_0^H`, then a second
    // γ-batched open binds the witness slice — exactly mirroring the main
    // `multipoint_eval` + `open` at `r*`/`r_0`. The verifier's binding
    // check then ties each `Δ = 0` pair eval to the bound
    // `hadamard_evals_at_rstar_h[col]`. (`Δ ≠ 0` pairs are row-shifted and
    // stay trusted — the row-shift discharge gap.) All four fields are
    // `None`/empty when the UAIR has no Hadamard relations.
    /// Second multipoint-eval reducing the per-column evals at `r*_H`
    /// to a single open point `r_0^H`. `None` when no Hadamard relations.
    pub hadamard_multipoint_eval: Option<MultipointEvalProof<BinaryFieldGF128>>,
    /// Every projected column's MLE eval at `r*_H` (primary cols first,
    /// then virtual) — the `up_evals` the second multipoint-eval reduces.
    /// Bound to the commitment via the second mp+open chain; the `Δ = 0`
    /// Hadamard pairs' `[col]` entries are checked equal to the matching
    /// `uair.hadamard_pair_evals`. Empty when no Hadamard relations.
    pub hadamard_evals_at_rstar_h: Vec<BinaryFieldGF128>,
    /// Every projected column's MLE eval at `r_0^H` (primary cols first,
    /// then virtual), bound by the second multipoint-eval consistency
    /// check and — for the witness-primary slice — by [`Self::hadamard_open`].
    /// Empty when no Hadamard relations.
    pub hadamard_open_evals_at_r0h: Vec<BinaryFieldGF128>,
    /// Second γ-batched open over the witness slice at `r_0^H`, sharing
    /// the same commitment / Merkle tree as [`Self::open`]. `None` when
    /// no Hadamard relations.
    pub hadamard_open: Option<F2OpenProof<D>>,
}

/// Errors emitted by [`ZincPlusPiopF2::verify_f2_full`].
#[derive(Debug, thiserror::Error)]
pub enum F2FullVerifyError<U: Uair, IdealOverF>
where
    IdealOverF: zinc_uair::ideal::Ideal,
{
    #[error("IC + sumcheck verification failed: {0}")]
    Uair(F2VerifyError<U, IdealOverF>),
    #[error("multipoint-eval verification failed: {0}")]
    MultipointEval(MultipointEvalError<BinaryFieldGF128>),
    #[error("F_2[X] open verification failed: {0}")]
    Open(F2OpenError),
    #[error(
        "public column {public_col_idx}: claimed MLE eval at r* ({claimed:?}) doesn't match \
         verifier's local computation from public_trace ({computed:?})"
    )]
    PublicColumnEvalMismatch {
        public_col_idx: usize,
        computed: BinaryFieldGF128,
        claimed: BinaryFieldGF128,
    },
    #[error(
        "public column {public_col_idx}: claimed MLE eval at r_0 ({claimed:?}) doesn't match \
         verifier's local computation from public_trace ({computed:?})"
    )]
    PublicColumnEvalMismatchAtR0 {
        public_col_idx: usize,
        computed: BinaryFieldGF128,
        claimed: BinaryFieldGF128,
    },
    #[error(
        "virtual column {virtual_idx}: claimed MLE eval at r_0 ({claimed:?}) doesn't match \
         verifier's derivation from primary evals at r_0 ({derived:?})"
    )]
    VirtualColumnEvalMismatchAtR0 {
        virtual_idx: usize,
        derived: BinaryFieldGF128,
        claimed: BinaryFieldGF128,
    },
    #[error(
        "open_evals_at_r_0 has length {got}, expected {expected} (= num_primary + num_virtual)"
    )]
    OpenEvalsLengthMismatch { got: usize, expected: usize },
    #[error(
        "Hadamard discharge: proof carries Hadamard specs but the second \
         multipoint-eval / open is missing"
    )]
    MissingHadamardDischarge,
    #[error(
        "Hadamard discharge: hadamard_evals_at_rstar_h has length {got}, expected {expected} \
         (= num_primary + num_virtual)"
    )]
    HadamardEvalsAtRstarHLengthMismatch { got: usize, expected: usize },
    #[error(
        "Hadamard discharge: hadamard_open_evals_at_r0h has length {got}, expected {expected} \
         (= num_primary + num_virtual)"
    )]
    HadamardOpenEvalsLengthMismatch { got: usize, expected: usize },
    #[error(
        "Hadamard discharge: public column {public_col_idx}: claimed MLE eval at r_0^H \
         ({claimed:?}) doesn't match verifier's local computation ({computed:?})"
    )]
    HadamardPublicColumnEvalMismatchAtR0H {
        public_col_idx: usize,
        computed: BinaryFieldGF128,
        claimed: BinaryFieldGF128,
    },
    #[error(
        "Hadamard discharge: virtual column {virtual_idx}: claimed MLE eval at r_0^H \
         ({claimed:?}) doesn't match verifier's derivation from primary evals ({derived:?})"
    )]
    HadamardVirtualColumnEvalMismatchAtR0H {
        virtual_idx: usize,
        derived: BinaryFieldGF128,
        claimed: BinaryFieldGF128,
    },
    #[error(
        "Hadamard discharge: parent-eval count {got} doesn't match the distinct Hadamard \
         column count {expected}"
    )]
    HadamardParentEvalCountMismatch { got: usize, expected: usize },
    #[error(
        "Hadamard discharge: distinct column {distinct_col}'s opened α-eval at r*_H ({opened:?}) \
         doesn't match the trusted parent-eval ({parent:?}) — the parent-evals are not bound to \
         the commitment"
    )]
    HadamardParentEvalBindingMismatch {
        distinct_col: usize,
        opened: BinaryFieldGF128,
        parent: BinaryFieldGF128,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::ConstZero;
    use rand::{Rng, rng};
    use zinc_poly::mle::MultilinearExtensionWithConfig;
    use zinc_transcript::Blake3Transcript;
    use zinc_uair::{
        ConstraintBuilder, PublicColumnLayout, TotalColumnLayout, TraceRow, UairSignature,
        ideal::ImpossibleIdeal,
    };

    /// Smallest viable all-`F_2` UAIR for testing the prove path: two
    /// `binary_poly` columns with the constraint `col_0 == col_1`
    /// (i.e. `col_0 - col_1 ∈ <0>`, an `assert_zero` constraint).
    /// `Scalar = BinaryPoly<32>`; `Ideal = ImpossibleIdeal` (unused).
    #[derive(Clone, Debug, Default)]
    struct TinyF2Uair;

    impl Uair for TinyF2Uair {
        type Ideal = ImpossibleIdeal;
        type Scalar = BinaryPoly<32>;

        fn signature() -> UairSignature {
            UairSignature::new(
                TotalColumnLayout::new(2, 0, 0),
                PublicColumnLayout::default(),
                vec![],
                vec![],
                vec![],
            )
        }

        fn constrain_general<B, FromR, MulByScalar, IFromR>(
            b: &mut B,
            up: TraceRow<B::Expr>,
            _down: TraceRow<B::Expr>,
            _from_ref: FromR,
            _mbs: MulByScalar,
            _ideal_from_ref: IFromR,
        ) where
            B: ConstraintBuilder,
            FromR: Fn(&Self::Scalar) -> B::Expr,
            MulByScalar: Fn(&B::Expr, &Self::Scalar) -> Option<B::Expr>,
            IFromR: Fn(&Self::Ideal) -> B::Ideal,
        {
            b.assert_zero(up.binary_poly[0].clone() - &up.binary_poly[1]);
        }
    }

    /// End-to-end: build a satisfied all-`F_2` trace, run the
    /// F_2 prove path, and assert the resulting proof has the
    /// expected shape (`num_constraints` IC entries, `num_vars`
    /// sumcheck rounds per group, `num_cols` groups).
    ///
    /// The test does NOT verify the proof against a verifier — that
    /// would require a parallel `verify_f2_uair` and is the next
    /// slice. The prover-side test confirms (a) the pipeline runs
    /// without panicking against a real F_2 UAIR + GF(2^128) field,
    /// (b) the wire format of the resulting proof is internally
    /// consistent, and (c) all transcript draws (IC challenge, α,
    /// sumcheck round challenges) are exercised by a real
    /// `Blake3Transcript`.
    #[test]
    fn prove_f2_pipeline_runs_against_tinyf2uair() {
        const D: usize = 32;
        let num_vars: usize = 4;
        let poly_size = 1usize << num_vars;
        let mut r = rng();

        // Build the trace: two identical random binary-poly columns
        // (so the `col_0 == col_1` constraint holds for every row).
        let col0_vals: Vec<BinaryPoly<D>> =
            (0..poly_size).map(|_| BinaryPoly::from(r.random::<u32>())).collect();
        let col1_vals = col0_vals.clone();
        let col0 = DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            col0_vals,
            BinaryPoly::default(),
        );
        let col1 = DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            col1_vals,
            BinaryPoly::default(),
        );
        let trace: UairTrace<'static, BinaryPoly<D>, BinaryPoly<D>, D> = UairTrace {
            binary_poly: vec![col0, col1].into(),
            arbitrary_poly: vec![].into(),
            int: vec![].into(),
        };

        let mut transcript = Blake3Transcript::new();

        // F2ZincTypes-bound entry point. Doesn't need an actual
        // `F2ZincTypes` impl yet — the prove function is generic
        // over U; the F2ZincTypes trait is a future hook for the
        // PCS-commit wiring (not exercised in this test).
        //
        // We invoke prove_f2_uair directly via a phantom struct
        // that satisfies the F2ZincTypes bound trivially. For now
        // there's no concrete F2ZincTypes-implementing type in
        // protocol/, so we use the function statically with U
        // pinned to TinyF2Uair.
        let proof = prove_f2_uair_for_tests::<TinyF2Uair, D>(
            &mut transcript,
            &trace,
            num_vars,
            // `TinyF2Uair` has only an `assert_zero` constraint, so
            // `collect_scalars` returns an empty set and this
            // closure is never called. Provide a sensible default:
            // lift `BinaryPoly<32>` → `DynamicPolynomialF<GF192>` via
            // the F_2 ⊂ GF(2^128) per-coefficient embedding.
            |scalar: &BinaryPoly<32>| -> DynamicPolynomialF<BinaryFieldGF128> {
                let coeffs: Vec<BinaryFieldGF128> = scalar
                    .iter()
                    .map(|b| {
                        if b.into_inner() {
                            BinaryFieldGF128::one()
                        } else {
                            BinaryFieldGF128::zero()
                        }
                    })
                    .collect();
                DynamicPolynomialF { coeffs }
            },
        )
        .expect("prove_f2_uair should succeed on a satisfied F_2 trace");

        // The IC proof has one entry per constraint (1 for
        // TinyF2Uair). For an `assert_zero` constraint the value is
        // `DynamicPolynomialF::ZERO`.
        let num_constraints = count_constraints::<TinyF2Uair>();
        assert_eq!(proof.ic_proof.combined_mle_values.len(), num_constraints);
        for v in &proof.ic_proof.combined_mle_values {
            assert_eq!(v, &DynamicPolynomialF::<BinaryFieldGF128>::ZERO);
        }

        // γ-batched: a SINGLE degree-2 group `[eq_r, weighted_col]`
        // regardless of the number of trace columns. Per-column
        // evals at r* are carried separately in
        // `column_evals_at_rstar`.
        let claimed = proof.sumcheck_proof.claimed_sums();
        assert_eq!(claimed.len(), 1, "γ-batched protocol has one sumcheck group");
        assert_eq!(
            proof.column_evals_at_rstar.len(),
            2,
            "two trace columns → two per-column evals at r*",
        );
        // Each col is identically itself; on the boolean hypercube
        // the sum of `eq(y, r) · col(y)` equals the MLE evaluated
        // at `r` (= IC's evaluation point). That value is finite
        // and not asserted to anything specific in this smoke test.

        // α was drawn from the transcript between the IC and the
        // sumcheck. Any non-default value is fine; assert it
        // isn't the trivial zero (which would indicate the
        // transcript flow is broken — Blake3 of a non-trivial
        // state is overwhelmingly likely to produce a non-zero
        // 128-bit element).
        assert!(
            !proof.alpha.is_zero(),
            "α should be a non-zero GF(2^128) challenge; got {}",
            proof.alpha
        );
    }

    /// Test shim that exposes the prove logic as a free function,
    /// bypassing the `F2ZincTypes`-typed `ZincPlusPiopF2` wrapper.
    /// (The F2ZincTypes trait is reserved for the PCS-commit wiring;
    /// no concrete impl is required for the IC+sumcheck pipeline
    /// itself.)
    fn prove_f2_uair_for_tests<U, const D: usize>(
        transcript: &mut impl Transcript,
        trace: &UairTrace<'static, BinaryPoly<D>, BinaryPoly<D>, D>,
        num_vars: usize,
        project_scalar: impl Fn(&U::Scalar) -> DynamicPolynomialF<BinaryFieldGF128> + Sync,
    ) -> Result<F2Proof, F2ProveError<U>>
    where
        U: Uair + 'static,
    {
        // Re-implementation of `ZincPlusPiopF2::prove_f2_uair`
        // without the `F2ZincTypes` bound — for tests only. The
        // logic is otherwise identical.
        let num_constraints = count_constraints::<U>();
        let field_cfg = ();

        let row_major_trace = project_f2_trace_row_major::<BinaryFieldGF128, _, _, D>(
            trace,
            &field_cfg,
        );

        let scalars =
            zinc_piop::projections::project_scalars::<BinaryFieldGF128, U>(|s| project_scalar(s));

        let (ic_proof, ic_state) = <U as IdealCheckProtocol>::prove_combined::<BinaryFieldGF128>(
            transcript,
            &row_major_trace,
            &scalars,
            num_constraints,
            num_vars,
            &field_cfg,
        )
        .map_err(F2ProveError::IdealCheck)?;

        let alpha: BinaryFieldGF128 = transcript.get_field_challenge(&field_cfg);

        let projected_trace: Vec<DenseMultilinearExtension<BinaryFieldGF128>> = trace
            .binary_poly
            .iter()
            .map(|col| {
                let evals_at_alpha: Vec<BinaryFieldGF128> = col
                    .evaluations
                    .iter()
                    .map(|cell| {
                        zinc_poly::univariate::binary_gf128::eval_f2_poly_d_at::<D>(cell, &alpha)
                    })
                    .collect();
                DenseMultilinearExtension::from_evaluations_vec(
                    col.num_vars,
                    evals_at_alpha,
                    BinaryFieldGF128::zero(),
                )
            })
            .collect();

        let gamma: BinaryFieldGF128 = transcript.get_field_challenge(&field_cfg);

        let eq_r = zinc_poly::utils::build_eq_x_r_inner(
            &ic_state.evaluation_point,
            &field_cfg,
        )
        .expect("eq table construction must succeed for valid IC point");
        let weighted_col = build_gamma_weighted_col(&projected_trace, &gamma);

        let group = MultiDegreeSumcheckGroup::new(
            2,
            vec![eq_r, weighted_col],
            Box::new(|v: &[BinaryFieldGF128]| v[0] * v[1]),
        );

        let (sumcheck_proof, prover_states) =
            MultiDegreeSumcheck::<BinaryFieldGF128>::prove_as_subprotocol(
                transcript,
                vec![group],
                num_vars,
                &field_cfg,
            );

        let sumcheck_point = prover_states[0].randomness.clone();
        let column_evals_at_rstar: Vec<BinaryFieldGF128> = projected_trace
            .iter()
            .map(|col| {
                let zero_inner = *BinaryFieldGF128::zero().inner();
                let inner_mle = DenseMultilinearExtension::from_evaluations_vec(
                    col.num_vars,
                    col.evaluations.iter().map(|x| *x.inner()).collect(),
                    zero_inner,
                );
                <DenseMultilinearExtension<_> as zinc_poly::mle::MultilinearExtensionWithConfig<
                    BinaryFieldGF128,
                >>::evaluate_with_config(
                    inner_mle, &sumcheck_point, &field_cfg
                )
                .expect("MLE evaluation on r* should succeed")
            })
            .collect();

        let mut buf = vec![0u8; <<BinaryFieldGF128 as crypto_primitives::Field>::Inner
            as zinc_transcript::traits::ConstTranscribable>::NUM_BYTES];
        for v in &column_evals_at_rstar {
            transcript.absorb_random_field(v, &mut buf);
        }

        Ok(F2Proof {
            ic_proof,
            sumcheck_proof,
            alpha,
            gamma,
            column_evals_at_rstar,
            hadamard_proof: None,
            hadamard_pair_evals: Vec::new(),
        })
    }

    /// Test shim that mirrors `ZincPlusPiopF2::verify_f2_uair`
    /// without the `F2ZincTypes` bound.
    fn verify_f2_uair_for_tests<U, IdealOverF>(
        transcript: &mut impl Transcript,
        proof: &F2Proof,
        num_vars: usize,
        num_columns: usize,
        project_ideal: impl Fn(&IdealOrZero<U::Ideal>) -> IdealOverF,
    ) -> Result<F2VerifierSubclaim, F2VerifyError<U, IdealOverF>>
    where
        U: Uair + 'static,
        IdealOverF: zinc_uair::ideal::Ideal
            + zinc_uair::ideal::IdealCheck<DynamicPolynomialF<BinaryFieldGF128>>,
    {
        let num_constraints = count_constraints::<U>();
        let field_cfg = ();

        let ic_subclaim = <U as IdealCheckProtocol>::verify_as_subprotocol::<_, IdealOverF, _>(
            transcript,
            proof.ic_proof.clone(),
            num_constraints,
            num_vars,
            project_ideal,
            &field_cfg,
        )
        .map_err(F2VerifyError::IdealCheck)?;
        let ic_evaluation_point = ic_subclaim.evaluation_point;

        let alpha: BinaryFieldGF128 = transcript.get_field_challenge(&field_cfg);
        if alpha != proof.alpha {
            return Err(F2VerifyError::AlphaMismatch {
                transcript: alpha,
                proof: proof.alpha,
            });
        }

        let gamma: BinaryFieldGF128 = transcript.get_field_challenge(&field_cfg);

        let md_subclaims = MultiDegreeSumcheck::<BinaryFieldGF128>::verify_as_subprotocol(
            transcript,
            num_vars,
            &proof.sumcheck_proof,
            &field_cfg,
        )
        .map_err(F2VerifyError::Sumcheck)?;

        let sumcheck_point = md_subclaims.point().to_vec();
        if md_subclaims.expected_evaluations().len() != 1 {
            return Err(F2VerifyError::GroupCountMismatch {
                expected: 1,
                actual: md_subclaims.expected_evaluations().len(),
            });
        }

        if proof.column_evals_at_rstar.len() != num_columns {
            return Err(F2VerifyError::GroupCountMismatch {
                expected: num_columns,
                actual: proof.column_evals_at_rstar.len(),
            });
        }

        let one = BinaryFieldGF128::one();
        let eq_at_rstar_r = zinc_poly::utils::eq_eval(
            &sumcheck_point,
            &ic_evaluation_point,
            one,
        )
        .expect("matching length by construction");
        if eq_at_rstar_r.is_zero() {
            return Err(F2VerifyError::DegenerateEq);
        }
        let mut batched = BinaryFieldGF128::zero();
        for v in proof.column_evals_at_rstar.iter().rev() {
            batched = batched * gamma + *v;
        }
        let expected = md_subclaims.expected_evaluations()[0];
        if eq_at_rstar_r * batched != expected {
            return Err(F2VerifyError::BatchedSumcheckMismatch {
                expected,
                got: eq_at_rstar_r * batched,
            });
        }

        let mut buf = vec![0u8; <<BinaryFieldGF128 as crypto_primitives::Field>::Inner
            as zinc_transcript::traits::ConstTranscribable>::NUM_BYTES];
        for v in &proof.column_evals_at_rstar {
            transcript.absorb_random_field(v, &mut buf);
        }

        // Tests use the shim with virtual-spec-free UAIRs only;
        // mirror the real `verify_f2_uair`'s primary/virtual split
        // by treating all extracted evals as primary.
        Ok(F2VerifierSubclaim {
            ic_evaluation_point,
            alpha,
            sumcheck_point,
            primary_column_evals: proof.column_evals_at_rstar.clone(),
            virtual_column_evals: Vec::new(),
            hadamard_rstar: Vec::new(),
            hadamard_pairs: Vec::new(),
        })
    }

    /// End-to-end roundtrip: prove, then verify with a fresh
    /// transcript, and assert the verifier's subclaim is
    /// internally consistent.
    #[test]
    fn prove_then_verify_f2_pipeline_roundtrips() {
        const D: usize = 32;
        let num_vars: usize = 4;
        let poly_size = 1usize << num_vars;
        let mut r = rng();

        let col0_vals: Vec<BinaryPoly<D>> =
            (0..poly_size).map(|_| BinaryPoly::from(r.random::<u32>())).collect();
        let col1_vals = col0_vals.clone();
        let col0 = DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            col0_vals.clone(),
            BinaryPoly::default(),
        );
        let col1 = DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            col1_vals,
            BinaryPoly::default(),
        );
        let trace: UairTrace<'static, BinaryPoly<D>, BinaryPoly<D>, D> = UairTrace {
            binary_poly: vec![col0.clone(), col1.clone()].into(),
            arbitrary_poly: vec![].into(),
            int: vec![].into(),
        };

        let project_scalar = |scalar: &BinaryPoly<32>| -> DynamicPolynomialF<BinaryFieldGF128> {
            let coeffs: Vec<BinaryFieldGF128> = scalar
                .iter()
                .map(|b| if b.into_inner() { BinaryFieldGF128::one() } else { BinaryFieldGF128::zero() })
                .collect();
            DynamicPolynomialF { coeffs }
        };

        // Prove
        let mut prover_transcript = Blake3Transcript::new();
        let proof = prove_f2_uair_for_tests::<TinyF2Uair, D>(
            &mut prover_transcript,
            &trace,
            num_vars,
            project_scalar,
        )
        .expect("prove should succeed");

        // Verify on a fresh transcript
        let mut verifier_transcript = Blake3Transcript::new();
        let subclaim = verify_f2_uair_for_tests::<TinyF2Uair, _>(
            &mut verifier_transcript,
            &proof,
            num_vars,
            /* num_columns */ 2,
            // `TinyF2Uair`'s single constraint is `assert_zero`, so
            // the IC's ideal-projection closure is never invoked.
            |_ideal| zinc_uair::ideal::ImpossibleIdeal,
        )
        .expect("verify should succeed on an honest proof");

        // Cross-check: column_mle_evals should match each column's
        // projected MLE evaluated at `r*` directly.
        let zero_inner = *BinaryFieldGF128::zero().inner();
        for (g, expected) in subclaim.primary_column_evals.iter().enumerate() {
            let projected_col_inner_evals: Vec<_> = trace.binary_poly[g]
                .evaluations
                .iter()
                .map(|cell| {
                    *zinc_poly::univariate::binary_gf128::eval_f2_poly_d_at::<D>(
                        cell,
                        &subclaim.alpha,
                    )
                    .inner()
                })
                .collect();
            let projected_col_mle = DenseMultilinearExtension::from_evaluations_vec(
                num_vars,
                projected_col_inner_evals,
                zero_inner,
            );
            let direct = projected_col_mle
                .evaluate_with_config(&subclaim.sumcheck_point, &())
                .expect("MLE eval should succeed");
            assert_eq!(
                direct, *expected,
                "column {g}: direct MLE evaluation at r* disagrees with verifier-derived expected",
            );
        }

        assert_eq!(subclaim.alpha, proof.alpha);
        assert_eq!(subclaim.sumcheck_point.len(), num_vars);
        assert_eq!(subclaim.ic_evaluation_point.len(), num_vars);
        assert_eq!(subclaim.primary_column_evals.len(), 2);
    }

    /// Three binary columns, no algebraic constraints — the
    /// `W = U ⊙ V` relation is discharged purely by the Wiring-R
    /// Hadamard phase, not the IC.
    #[derive(Clone, Debug, Default)]
    struct HadF2Uair;

    impl Uair for HadF2Uair {
        type Ideal = ImpossibleIdeal;
        type Scalar = BinaryPoly<32>;

        fn signature() -> UairSignature {
            UairSignature::new(
                TotalColumnLayout::new(3, 0, 0),
                PublicColumnLayout::default(),
                vec![],
                vec![],
                vec![],
            )
        }

        fn constrain_general<B, FromR, MulByScalar, IFromR>(
            _b: &mut B,
            _up: TraceRow<B::Expr>,
            _down: TraceRow<B::Expr>,
            _from_ref: FromR,
            _mbs: MulByScalar,
            _ideal_from_ref: IFromR,
        ) where
            B: ConstraintBuilder,
            FromR: Fn(&Self::Scalar) -> B::Expr,
            MulByScalar: Fn(&B::Expr, &Self::Scalar) -> Option<B::Expr>,
            IFromR: Fn(&Self::Ideal) -> B::Ideal,
        {
            // No algebraic constraints; the Hadamard relation between
            // columns 0,1,2 is enforced by the Wiring-R phase.
        }
    }

    /// Five binary columns, no algebraic constraints — used to exercise
    /// the Phase-B operand features (row-shift / complement / XOR combo)
    /// through the full prove/verify flow.
    #[derive(Clone, Debug, Default)]
    struct HadOpF2Uair;

    impl Uair for HadOpF2Uair {
        type Ideal = ImpossibleIdeal;
        type Scalar = BinaryPoly<32>;

        fn signature() -> UairSignature {
            UairSignature::new(
                TotalColumnLayout::new(5, 0, 0),
                PublicColumnLayout::default(),
                vec![],
                vec![],
                vec![],
            )
        }

        fn constrain_general<B, FromR, MulByScalar, IFromR>(
            _b: &mut B,
            _up: TraceRow<B::Expr>,
            _down: TraceRow<B::Expr>,
            _from_ref: FromR,
            _mbs: MulByScalar,
            _ideal_from_ref: IFromR,
        ) where
            B: ConstraintBuilder,
            FromR: Fn(&Self::Scalar) -> B::Expr,
            MulByScalar: Fn(&B::Expr, &Self::Scalar) -> Option<B::Expr>,
            IFromR: Fn(&Self::Ideal) -> B::Ideal,
        {
            // No algebraic constraints; the operand Hadamard relations are
            // enforced by the Wiring-R phase.
        }
    }

    /// End-to-end through the real `prove_f2_uair_with_groups` /
    /// `verify_f2_uair_with_groups`: an honest `W = U ⊙ V` trace
    /// round-trips, and flipping a bit of `W` makes the in-flow
    /// Hadamard zerocheck reject.
    #[test]
    fn hadamard_phase_roundtrips_in_flow() {
        const D: usize = 32;
        let num_vars: usize = 4;
        let poly_size = 1usize << num_vars;
        let mut r = rng();

        fn project_scalar(scalar: &BinaryPoly<32>) -> DynamicPolynomialF<BinaryFieldGF128> {
            DynamicPolynomialF {
                coeffs: scalar
                    .iter()
                    .map(|b| {
                        if b.into_inner() {
                            BinaryFieldGF128::one()
                        } else {
                            BinaryFieldGF128::zero()
                        }
                    })
                    .collect(),
            }
        }

        let u_words: Vec<u32> = (0..poly_size).map(|_| r.random::<u32>()).collect();
        let v_words: Vec<u32> = (0..poly_size).map(|_| r.random::<u32>()).collect();
        let w_words: Vec<u32> = u_words.iter().zip(&v_words).map(|(a, b)| a & b).collect();
        let mk = |words: &[u32]| {
            DenseMultilinearExtension::from_evaluations_vec(
                num_vars,
                words.iter().map(|&x| BinaryPoly::<D>::from(x)).collect(),
                BinaryPoly::default(),
            )
        };
        let trace: UairTrace<'static, BinaryPoly<D>, BinaryPoly<D>, D> = UairTrace {
            binary_poly: vec![mk(&u_words), mk(&v_words), mk(&w_words)].into(),
            arbitrary_poly: vec![].into(),
            int: vec![].into(),
        };

        let specs = [crate::f2_hadamard::F2HadamardSpec::plain(0, 1, 2)];

        // -- honest prove → verify --
        let mut pt = Blake3Transcript::new();
        let (proof, _subclaim, _proj) =
            ZincPlusPiopF2::<F2Types<D>, HadF2Uair, D>::prove_f2_uair_with_groups(
                &mut pt,
                &trace,
                &[],
                &specs,
                num_vars,
                project_scalar,
            )
            .expect("hadamard prove");
        assert!(proof.hadamard_proof.is_some());

        let mut vt = Blake3Transcript::new();
        ZincPlusPiopF2::<F2Types<D>, HadF2Uair, D>::verify_f2_uair_with_groups::<ImpossibleIdeal>(
            &mut vt,
            &proof,
            &[],
            &specs,
            num_vars,
            /* num_primary_columns */ 3,
            |_ideal| ImpossibleIdeal,
        )
        .expect("hadamard verify");

        // -- corrupt W: in-flow Hadamard zerocheck must reject --
        let mut w_bad = w_words.clone();
        w_bad[0] ^= 1;
        let trace_bad: UairTrace<'static, BinaryPoly<D>, BinaryPoly<D>, D> = UairTrace {
            binary_poly: vec![mk(&u_words), mk(&v_words), mk(&w_bad)].into(),
            arbitrary_poly: vec![].into(),
            int: vec![].into(),
        };
        let mut pt2 = Blake3Transcript::new();
        let (proof_bad, _, _) =
            ZincPlusPiopF2::<F2Types<D>, HadF2Uair, D>::prove_f2_uair_with_groups(
                &mut pt2,
                &trace_bad,
                &[],
                &specs,
                num_vars,
                project_scalar,
            )
            .expect("prove still produces a proof for a corrupt trace");
        let mut vt2 = Blake3Transcript::new();
        let res = ZincPlusPiopF2::<F2Types<D>, HadF2Uair, D>::verify_f2_uair_with_groups::<
            ImpossibleIdeal,
        >(
            &mut vt2,
            &proof_bad,
            &[],
            &specs,
            num_vars,
            3,
            |_ideal| ImpossibleIdeal,
        );
        assert!(matches!(res, Err(F2VerifyError::Hadamard(_))));
    }

    /// Tampering with the proof's α should yield an AlphaMismatch
    /// error rather than a panic or a silent acceptance.
    #[test]
    fn verify_rejects_tampered_alpha() {
        const D: usize = 32;
        let num_vars: usize = 3;
        let poly_size = 1usize << num_vars;
        let mut r = rng();

        let col0_vals: Vec<BinaryPoly<D>> =
            (0..poly_size).map(|_| BinaryPoly::from(r.random::<u32>())).collect();
        let col1_vals = col0_vals.clone();
        let col0 = DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            col0_vals,
            BinaryPoly::default(),
        );
        let col1 = DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            col1_vals,
            BinaryPoly::default(),
        );
        let trace: UairTrace<'static, BinaryPoly<D>, BinaryPoly<D>, D> = UairTrace {
            binary_poly: vec![col0, col1].into(),
            arbitrary_poly: vec![].into(),
            int: vec![].into(),
        };

        let project_scalar = |_: &BinaryPoly<32>| DynamicPolynomialF::<BinaryFieldGF128>::ZERO;

        let mut prover_transcript = Blake3Transcript::new();
        let mut proof = prove_f2_uair_for_tests::<TinyF2Uair, D>(
            &mut prover_transcript,
            &trace,
            num_vars,
            project_scalar,
        )
        .expect("prove should succeed");

        // Mutate α — verifier should reject.
        proof.alpha = proof.alpha + BinaryFieldGF128::one();

        let mut verifier_transcript = Blake3Transcript::new();
        let err = verify_f2_uair_for_tests::<TinyF2Uair, _>(
            &mut verifier_transcript,
            &proof,
            num_vars,
            2,
            |_ideal| zinc_uair::ideal::ImpossibleIdeal,
        )
        .expect_err("tampered α should trigger AlphaMismatch");

        assert!(
            matches!(err, F2VerifyError::AlphaMismatch { .. }),
            "expected AlphaMismatch, got {err:?}",
        );
    }

    // [Removed] `prove_then_verify_with_custom_groups`: the F_2
    // prove path no longer exposes a build_groups/extract_subclaims
    // generic — γ-batching consolidates the per-column sumchecks
    // into a single degree-2 group whose composition is fixed.

    // -- Step 0 (PCS commit) wiring + roundtrip ----------------------
    //
    // Exercises the commit phase plus the IC + sumcheck pipeline
    // against a concrete `F2ZincTypes` impl. The follow-on Step 7
    // (open) is exercised by `prove_then_verify_f2_open_roundtrips`
    // below; the two together cover the full F_2 prove/verify cycle.

    use crypto_primitives::crypto_bigint_int::Int;
    use crypto_primitives::crypto_bigint_uint::Uint;
    use std::marker::PhantomData;
    use zinc_poly::univariate::binary::BinaryPolyInnerProduct;
    use zinc_poly::univariate::dense::{DensePolyInnerProduct, DensePolynomial};
    use zinc_primality::MillerRabin;
    use zinc_utils::inner_product::MBSInnerProduct;
    use zip_plus::code::raa::RaaConfig;
    use zip_plus::code::raa_f2::RaaF2Code;
    use zip_plus::pcs::structs::ZipTypes;

    /// Local F_2 `ZipTypes` impl mirroring zip-plus's test-only
    /// `TestBinPolyF2ZipTypes<D>`. We can't re-use that one because
    /// it lives behind `#[cfg(test)]` inside zip-plus.
    #[derive(Debug, Clone)]
    struct LocalBinPolyF2ZipTypes<const D: usize> {}
    impl<const D: usize> ZipTypes for LocalBinPolyF2ZipTypes<D> {
        const NUM_COLUMN_OPENINGS: usize = 147;
        type Eval = BinaryPoly<D>;
        type Cw = BinaryPoly<D>;
        type Fmod = Uint<{ crypto_bigint::U64::LIMBS * 4 }>;
        type PrimeTest = MillerRabin;
        type Chal = i128;
        type Pt = i128;
        type CombR = Int<{ crypto_bigint::U64::LIMBS * 8 }>;
        type Comb = DensePolynomial<Self::CombR, D>;
        type EvalDotChal = BinaryPolyInnerProduct<Self::Chal, D>;
        type CombDotChal =
            DensePolyInnerProduct<Self::CombR, Self::Chal, Self::CombR, MBSInnerProduct, D>;
        type ArrCombRDotChal = MBSInnerProduct;
    }

    #[derive(Copy, Clone)]
    struct LocalRaaConfig;
    impl RaaConfig for LocalRaaConfig {
        const PERMUTE_IN_PLACE: bool = false;
        const CHECK_FOR_OVERFLOWS: bool = false;
    }

    /// Concrete `F2ZincTypes` impl. Demonstrates the trait can be
    /// satisfied by real Zip+ primitives.
    #[derive(Clone, Debug)]
    struct F2Types<const D: usize>(PhantomData<()>);

    const LOCAL_REP_FACTOR: usize = 4;

    impl<const D: usize> F2ZincTypes<D> for F2Types<D> {
        // BinaryZt stores paired trace cells (width 64).
        type BinaryZt = LocalBinPolyF2ZipTypes<PACKED_STORAGE_WIDTH>;
        type BinaryLc = RaaF2Code<Self::BinaryZt, LocalRaaConfig, LOCAL_REP_FACTOR>;
    }

    /// End-to-end: commit binary trace columns via Zip+, run the
    /// IC + α + sumcheck pipeline with the commitment absorbed into
    /// the transcript at Step 0, then verify on a fresh transcript
    /// with the same commitment absorption. Asserts the verifier's
    /// subclaim matches direct MLE evaluations of the projected
    /// columns at `r*`.
    ///
    /// Open at `r*` is exercised separately by
    /// `prove_then_verify_f2_open_roundtrips`.
    #[test]
    fn commit_prove_verify_f2_roundtrip() {
        const D: usize = 32;
        let num_vars: usize = 6;
        let row_len: usize = 8;
        let poly_size = 1usize << num_vars;
        let num_rows = poly_size.div_ceil(row_len);
        assert_eq!(num_rows * row_len, poly_size);

        let mut rng_local = rng();

        let col0_vals: Vec<BinaryPoly<D>> = (0..poly_size)
            .map(|_| BinaryPoly::from(rng_local.random::<u32>()))
            .collect();
        let col1_vals = col0_vals.clone();
        let col0 = DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            col0_vals,
            BinaryPoly::default(),
        );
        let col1 = DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            col1_vals,
            BinaryPoly::default(),
        );
        let trace: UairTrace<'static, BinaryPoly<D>, BinaryPoly<D>, D> = UairTrace {
            binary_poly: vec![col0.clone(), col1.clone()].into(),
            arbitrary_poly: vec![].into(),
            int: vec![].into(),
        };

        // Step 0: commit -------------------------------------------
        let lc = <F2Types<D> as F2ZincTypes<D>>::BinaryLc::new(row_len);
        let pp: ZipPlusParams<
            <F2Types<D> as F2ZincTypes<D>>::BinaryZt,
            <F2Types<D> as F2ZincTypes<D>>::BinaryLc,
        > = ZipPlusParams::new(num_vars, num_rows, lc);

        let mut prover_transcript = Blake3Transcript::new();
        let (_hint, comm) =
            ZincPlusPiopF2::<F2Types<D>, TinyF2Uair, D>::commit_and_absorb_f2_trace(
                &mut prover_transcript,
                &pp,
                &trace.binary_poly,
            )
            .expect("commit should succeed");
        // 2 primary cols paired into 1 packed storage column.
        assert_eq!(comm.batch_size, 2_usize.div_ceil(2));

        // Steps 2-4: IC + α + sumcheck (via the test shim, since
        // prove_f2_uair is gated on the F2ZincTypes bound which is
        // already satisfied above — we use the shim to also avoid
        // dragging the bound into the call-site verbosity).
        let project_scalar = |_: &BinaryPoly<32>| DynamicPolynomialF::<BinaryFieldGF128>::ZERO;
        let proof = prove_f2_uair_for_tests::<TinyF2Uair, D>(
            &mut prover_transcript,
            &trace,
            num_vars,
            project_scalar,
        )
        .expect("prove should succeed");

        // -- Verifier side ----------------------------------------
        let mut verifier_transcript = Blake3Transcript::new();
        ZincPlusPiopF2::<F2Types<D>, TinyF2Uair, D>::absorb_commitment(
            &mut verifier_transcript,
            &comm,
        );

        let subclaim = verify_f2_uair_for_tests::<TinyF2Uair, _>(
            &mut verifier_transcript,
            &proof,
            num_vars,
            2,
            |_ideal| ImpossibleIdeal,
        )
        .expect("verify should succeed");

        // Sanity: column MLE claims at r* match direct evaluation.
        let zero_inner = *BinaryFieldGF128::zero().inner();
        for (g, expected) in subclaim.primary_column_evals.iter().enumerate() {
            let projected_inner: Vec<_> = trace.binary_poly[g]
                .evaluations
                .iter()
                .map(|cell| {
                    *zinc_poly::univariate::binary_gf128::eval_f2_poly_d_at::<D>(
                        cell,
                        &subclaim.alpha,
                    )
                    .inner()
                })
                .collect();
            let projected_mle = DenseMultilinearExtension::from_evaluations_vec(
                num_vars,
                projected_inner,
                zero_inner,
            );
            let direct = projected_mle
                .evaluate_with_config(&subclaim.sumcheck_point, &())
                .expect("MLE eval should succeed");
            assert_eq!(
                direct, *expected,
                "column {g}: direct MLE evaluation at r* disagrees with verifier-derived expected",
            );
        }
    }

    // -- F_2[X] open (Step 7) roundtrip ------------------------------
    //
    // Exercises the lift-and-project MLE-opening pipeline:
    //   1. Build a satisfied F_2 trace + a sumcheck point r*.
    //   2. Run the prover-side computation of (a_g', b_g) per column.
    //   3. Verifier checks eval-consistency Σ_i q_0' · b_g = a_g' in
    //      F_2[X] and lift discharge ψ_α(a_g') = a_g in GF(2^128).
    //
    // The subclaim that feeds in here is constructed end-to-end
    // (commit → IC → α → sumcheck → verify), so this is the first
    // full pipeline-to-PCS-claim demonstration of the F_2 protocol.

    #[test]
    fn prove_then_verify_f2_open_roundtrips() {
        const D: usize = 32;
        let num_vars: usize = 6;
        let row_len: usize = 8;
        let poly_size = 1usize << num_vars;
        let num_rows = poly_size / row_len;
        assert_eq!(num_rows * row_len, poly_size);

        let mut rng_local = rng();

        // -- Build a satisfied trace: col_0 == col_1 (assert_zero ok)
        let col0_vals: Vec<BinaryPoly<D>> = (0..poly_size)
            .map(|_| BinaryPoly::from(rng_local.random::<u32>()))
            .collect();
        let col1_vals = col0_vals.clone();
        let col0 = DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            col0_vals,
            BinaryPoly::default(),
        );
        let col1 = DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            col1_vals,
            BinaryPoly::default(),
        );
        let trace: UairTrace<'static, BinaryPoly<D>, BinaryPoly<D>, D> = UairTrace {
            binary_poly: vec![col0.clone(), col1.clone()].into(),
            arbitrary_poly: vec![].into(),
            int: vec![].into(),
        };

        // -- Run the full IC + sumcheck pipeline to produce a subclaim.
        let project_scalar =
            |_: &BinaryPoly<32>| DynamicPolynomialF::<BinaryFieldGF128>::ZERO;
        let mut prover_transcript = Blake3Transcript::new();
        let proof = prove_f2_uair_for_tests::<TinyF2Uair, D>(
            &mut prover_transcript,
            &trace,
            num_vars,
            project_scalar,
        )
        .expect("prove should succeed");

        let mut verifier_transcript = Blake3Transcript::new();
        let subclaim = verify_f2_uair_for_tests::<TinyF2Uair, _>(
            &mut verifier_transcript,
            &proof,
            num_vars,
            2,
            |_ideal| ImpossibleIdeal,
        )
        .expect("verify should succeed");

        // -- Step 0: commit + absorb (separate from the IC transcript).
        let lc = <F2Types<D> as F2ZincTypes<D>>::BinaryLc::new(row_len);
        let pp: ZipPlusParams<
            <F2Types<D> as F2ZincTypes<D>>::BinaryZt,
            <F2Types<D> as F2ZincTypes<D>>::BinaryLc,
        > = ZipPlusParams::new(num_vars, num_rows, lc);
        let mut open_prover_transcript = Blake3Transcript::new();
        let (hint, comm) =
            ZincPlusPiopF2::<F2Types<D>, TinyF2Uair, D>::commit_and_absorb_f2_trace(
                &mut open_prover_transcript,
                &pp,
                &trace.binary_poly,
            )
            .expect("commit should succeed");

        // -- Step 7 prover: lift + eval-consistency + proximity.
        let num_column_openings = 4;
        let open_proof = ZincPlusPiopF2::<F2Types<D>, TinyF2Uair, D>::prove_f2_open(
            &mut open_prover_transcript,
            &pp,
            &hint,
            &trace.binary_poly,
            &subclaim.sumcheck_point,
            &subclaim.alpha,
            num_column_openings,
        );
        assert_eq!(open_proof.b_vector.len(), num_rows);
        assert_eq!(open_proof.combined_row.len(), row_len);
        assert_eq!(open_proof.opened_columns.len(), num_column_openings);

        // -- Step 7 verifier: full check (eval + ψ_α + coherence + Merkle).
        let mut open_verifier_transcript = Blake3Transcript::new();
        ZincPlusPiopF2::<F2Types<D>, TinyF2Uair, D>::absorb_commitment(
            &mut open_verifier_transcript,
            &comm,
        );
        ZincPlusPiopF2::<F2Types<D>, TinyF2Uair, D>::verify_f2_open(
            &mut open_verifier_transcript,
            &pp,
            &comm,
            &open_proof,
            &subclaim,
        )
        .expect("F_2[X] open verification should succeed");
    }

    /// Tampering with `a_g'` should produce either an
    /// `EvalConsistency` failure (if the b-vectors are unchanged and
    /// `a_g'` no longer matches the recomputed inner product) or a
    /// `LiftDischarge` failure (if `ψ_α(a_g')` no longer matches
    /// `a_g`). Either is acceptable; the verifier must reject.
    #[test]
    fn verify_f2_open_rejects_tampered_lifted_claim() {
        const D: usize = 32;
        let num_vars: usize = 4;
        let row_len: usize = 4;
        let poly_size = 1usize << num_vars;
        let num_rows = poly_size / row_len;
        let mut rng_local = rng();

        let col0_vals: Vec<BinaryPoly<D>> = (0..poly_size)
            .map(|_| BinaryPoly::from(rng_local.random::<u32>()))
            .collect();
        let col1_vals = col0_vals.clone();
        let col0 = DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            col0_vals,
            BinaryPoly::default(),
        );
        let col1 = DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            col1_vals,
            BinaryPoly::default(),
        );
        let trace: UairTrace<'static, BinaryPoly<D>, BinaryPoly<D>, D> = UairTrace {
            binary_poly: vec![col0, col1].into(),
            arbitrary_poly: vec![].into(),
            int: vec![].into(),
        };

        let project_scalar = |_: &BinaryPoly<32>| DynamicPolynomialF::<BinaryFieldGF128>::ZERO;
        let mut pt = Blake3Transcript::new();
        let proof = prove_f2_uair_for_tests::<TinyF2Uair, D>(&mut pt, &trace, num_vars, project_scalar)
            .expect("prove should succeed");

        let mut vt = Blake3Transcript::new();
        let subclaim = verify_f2_uair_for_tests::<TinyF2Uair, _>(
            &mut vt,
            &proof,
            num_vars,
            2,
            |_| ImpossibleIdeal,
        )
        .expect("verify should succeed");

        let lc = <F2Types<D> as F2ZincTypes<D>>::BinaryLc::new(row_len);
        let pp: ZipPlusParams<
            <F2Types<D> as F2ZincTypes<D>>::BinaryZt,
            <F2Types<D> as F2ZincTypes<D>>::BinaryLc,
        > = ZipPlusParams::new(num_vars, num_rows, lc);
        let mut open_pt = Blake3Transcript::new();
        let (hint, comm) =
            ZincPlusPiopF2::<F2Types<D>, TinyF2Uair, D>::commit_and_absorb_f2_trace(
                &mut open_pt,
                &pp,
                &trace.binary_poly,
            )
            .expect("commit should succeed");

        let mut open_proof = ZincPlusPiopF2::<F2Types<D>, TinyF2Uair, D>::prove_f2_open(
            &mut open_pt,
            &pp,
            &hint,
            &trace.binary_poly,
            &subclaim.sumcheck_point,
            &subclaim.alpha,
            4,
        );

        // Flip the lowest bit of the (γ-batched) lifted claim a'.
        let mut tampered_words = *open_proof.lifted_claim.words();
        tampered_words[0] ^= 1;
        open_proof.lifted_claim = BinaryF2Poly::<7>::from_words(tampered_words);

        let mut open_vt = Blake3Transcript::new();
        ZincPlusPiopF2::<F2Types<D>, TinyF2Uair, D>::absorb_commitment(&mut open_vt, &comm);
        let err = ZincPlusPiopF2::<F2Types<D>, TinyF2Uair, D>::verify_f2_open(
            &mut open_vt,
            &pp,
            &comm,
            &open_proof,
            &subclaim,
        )
        .expect_err("tampered lifted claim must be rejected");

        assert!(
            matches!(
                err,
                F2OpenError::EvalConsistency | F2OpenError::LiftDischarge { .. }
            ),
            "expected EvalConsistency or LiftDischarge, got {err:?}",
        );
    }

    /// Tampering with a b-vector entry while leaving the lifted
    /// claim unchanged should trip the **coherence** check
    /// (`<combined_row, q_1'> = <coeffs, b>` no longer balances) —
    /// the proximity-binding side of the verifier.
    #[test]
    fn verify_f2_open_rejects_tampered_b_vector() {
        const D: usize = 32;
        let num_vars: usize = 4;
        let row_len: usize = 4;
        let poly_size = 1usize << num_vars;
        let num_rows = poly_size / row_len;
        let mut rng_local = rng();

        let col0_vals: Vec<BinaryPoly<D>> = (0..poly_size)
            .map(|_| BinaryPoly::from(rng_local.random::<u32>()))
            .collect();
        let col1_vals = col0_vals.clone();
        let col0 = DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            col0_vals,
            BinaryPoly::default(),
        );
        let col1 = DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            col1_vals,
            BinaryPoly::default(),
        );
        let trace: UairTrace<'static, BinaryPoly<D>, BinaryPoly<D>, D> = UairTrace {
            binary_poly: vec![col0, col1].into(),
            arbitrary_poly: vec![].into(),
            int: vec![].into(),
        };

        let project_scalar = |_: &BinaryPoly<32>| DynamicPolynomialF::<BinaryFieldGF128>::ZERO;
        let mut pt = Blake3Transcript::new();
        let proof =
            prove_f2_uair_for_tests::<TinyF2Uair, D>(&mut pt, &trace, num_vars, project_scalar)
                .expect("prove should succeed");
        let mut vt = Blake3Transcript::new();
        let subclaim = verify_f2_uair_for_tests::<TinyF2Uair, _>(
            &mut vt,
            &proof,
            num_vars,
            2,
            |_| ImpossibleIdeal,
        )
        .expect("verify should succeed");

        let lc = <F2Types<D> as F2ZincTypes<D>>::BinaryLc::new(row_len);
        let pp: ZipPlusParams<
            <F2Types<D> as F2ZincTypes<D>>::BinaryZt,
            <F2Types<D> as F2ZincTypes<D>>::BinaryLc,
        > = ZipPlusParams::new(num_vars, num_rows, lc);
        let mut open_pt = Blake3Transcript::new();
        let (hint, comm) =
            ZincPlusPiopF2::<F2Types<D>, TinyF2Uair, D>::commit_and_absorb_f2_trace(
                &mut open_pt,
                &pp,
                &trace.binary_poly,
            )
            .expect("commit should succeed");
        let mut open_proof = ZincPlusPiopF2::<F2Types<D>, TinyF2Uair, D>::prove_f2_open(
            &mut open_pt,
            &pp,
            &hint,
            &trace.binary_poly,
            &subclaim.sumcheck_point,
            &subclaim.alpha,
            4,
        );

        // Flip one bit in b_vector[0] and re-derive lifted_claim so
        // the eval-consistency check still passes (verifier recomputes
        // Σ q_0 · b' = a' identically). Then either:
        //   - coherence <combined_row', q_1> = <coeffs, b'> fails, because
        //     combined_row' is bound (via Merkle + encoding consistency) to
        //     the genuine M_w while b' now isn't; or
        //   - the lift discharge ψ_α(a') = Σ_g γ_g · a_g fails, because
        //     the rebased a' projects to a different GF(2^128) value.
        let mut tampered_b = *open_proof.b_vector[0].words();
        tampered_b[0] ^= 1;
        open_proof.b_vector[0] = BinaryF2Poly::<5>::from_words(tampered_b);
        // Re-derive a' = Σ_i q_0[i] · b'[i] over F_2[X]<10>.
        let basis = zinc_poly::univariate::binary_gf128::AlphaPolyBasis::new(&subclaim.alpha);
        let (q0, _q1) = {
            let split = subclaim.sumcheck_point.len() - (num_rows.ilog2() as usize);
            let (hi, lo) = subclaim.sumcheck_point.split_at(split);
            let q0_gf = zinc_poly::utils::build_eq_x_r_vec(lo, &()).unwrap();
            let q1_gf = zinc_poly::utils::build_eq_x_r_vec(hi, &()).unwrap();
            let q0: Vec<BinaryF2Poly<2>> = q0_gf.iter().map(|g| basis.lift(g)).collect();
            let q1: Vec<BinaryF2Poly<2>> = q1_gf.iter().map(|g| basis.lift(g)).collect();
            (q0, q1)
        };
        open_proof.lifted_claim = {
            let mut acc = BinaryF2Poly::<7>::zero();
            for i in 0..num_rows {
                let prod: BinaryF2Poly<7> =
                    zinc_poly::univariate::binary_f2_wide::f2_poly_mul::<2, 5, 7>(
                        &q0[i],
                        &open_proof.b_vector[i],
                    );
                acc += prod;
            }
            acc
        };

        let mut open_vt = Blake3Transcript::new();
        ZincPlusPiopF2::<F2Types<D>, TinyF2Uair, D>::absorb_commitment(&mut open_vt, &comm);
        let err = ZincPlusPiopF2::<F2Types<D>, TinyF2Uair, D>::verify_f2_open(
            &mut open_vt,
            &pp,
            &comm,
            &open_proof,
            &subclaim,
        )
        .expect_err("tampered b-vector must trip a downstream check");

        assert!(
            matches!(
                err,
                F2OpenError::LiftDischarge { .. } | F2OpenError::Coherence
            ),
            "expected LiftDischarge or Coherence, got {err:?}",
        );
    }

    /// End-to-end roundtrip through the *bundled* prove/verify entry
    /// points. Exercises commit + IC + sumcheck + γ-batched open on a
    /// single shared transcript per side.
    #[test]
    fn prove_then_verify_f2_full_roundtrips() {
        const D: usize = 32;
        let num_vars: usize = 6;
        let row_len: usize = 8;
        let poly_size = 1usize << num_vars;
        let num_rows = poly_size / row_len;
        assert_eq!(num_rows * row_len, poly_size);

        let mut rng_local = rng();
        let col0_vals: Vec<BinaryPoly<D>> = (0..poly_size)
            .map(|_| BinaryPoly::from(rng_local.random::<u32>()))
            .collect();
        let col1_vals = col0_vals.clone();
        let col0 = DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            col0_vals,
            BinaryPoly::default(),
        );
        let col1 = DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            col1_vals,
            BinaryPoly::default(),
        );
        let trace: UairTrace<'static, BinaryPoly<D>, BinaryPoly<D>, D> = UairTrace {
            binary_poly: vec![col0, col1].into(),
            arbitrary_poly: vec![].into(),
            int: vec![].into(),
        };

        let lc = <F2Types<D> as F2ZincTypes<D>>::BinaryLc::new(row_len);
        let pp: ZipPlusParams<
            <F2Types<D> as F2ZincTypes<D>>::BinaryZt,
            <F2Types<D> as F2ZincTypes<D>>::BinaryLc,
        > = ZipPlusParams::new(num_vars, num_rows, lc);

        // -- Prove (single transcript across commit + uair + open) -
        let mut prover_transcript = Blake3Transcript::new();
        let proof = ZincPlusPiopF2::<F2Types<D>, TinyF2Uair, D>::prove_f2_full(
            &mut prover_transcript,
            &pp,
            &trace,
            /* virtual_specs */ &[],
            num_vars,
            |_| DynamicPolynomialF::<BinaryFieldGF128>::ZERO,
            /* num_column_openings */ 4,
        )
        .expect("prove_f2_full should succeed");

        // Shape sanity.
        // 2 primary cols paired into 1 packed storage column.
        assert_eq!(proof.commitment.batch_size, 2_usize.div_ceil(2));
        assert_eq!(proof.uair.alpha, proof.uair.alpha); // alpha plumbed
        assert_eq!(proof.open.b_vector.len(), num_rows);
        assert_eq!(proof.open.combined_row.len(), row_len);
        assert_eq!(proof.open.opened_columns.len(), 4);

        // -- Verify (single fresh transcript) ---------------------
        let mut verifier_transcript = Blake3Transcript::new();
        let subclaim = ZincPlusPiopF2::<F2Types<D>, TinyF2Uair, D>::verify_f2_full(
            &mut verifier_transcript,
            &pp,
            &proof,
            /* virtual_specs */ &[],
            /* public_binary_trace */ &[],
            num_vars,
            /* num_primary_columns */ 2,
            |_ideal| ImpossibleIdeal,
        )
        .expect("verify_f2_full should succeed");

        assert_eq!(subclaim.alpha, proof.uair.alpha);
        assert_eq!(subclaim.sumcheck_point.len(), num_vars);
        assert_eq!(subclaim.primary_column_evals.len(), 2);
    }

    /// Mutating the bundled proof's open phase (here: flipping a bit
    /// in `lifted_claim`) must surface as a structured `Open(...)`
    /// error from `verify_f2_full`.
    #[test]
    fn verify_f2_full_rejects_tampered_open() {
        const D: usize = 32;
        let num_vars: usize = 4;
        let row_len: usize = 4;
        let poly_size = 1usize << num_vars;
        let num_rows = poly_size / row_len;
        let mut rng_local = rng();
        let col0_vals: Vec<BinaryPoly<D>> = (0..poly_size)
            .map(|_| BinaryPoly::from(rng_local.random::<u32>()))
            .collect();
        let col1_vals = col0_vals.clone();
        let col0 = DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            col0_vals,
            BinaryPoly::default(),
        );
        let col1 = DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            col1_vals,
            BinaryPoly::default(),
        );
        let trace: UairTrace<'static, BinaryPoly<D>, BinaryPoly<D>, D> = UairTrace {
            binary_poly: vec![col0, col1].into(),
            arbitrary_poly: vec![].into(),
            int: vec![].into(),
        };

        let lc = <F2Types<D> as F2ZincTypes<D>>::BinaryLc::new(row_len);
        let pp: ZipPlusParams<
            <F2Types<D> as F2ZincTypes<D>>::BinaryZt,
            <F2Types<D> as F2ZincTypes<D>>::BinaryLc,
        > = ZipPlusParams::new(num_vars, num_rows, lc);

        let mut pt = Blake3Transcript::new();
        let mut proof = ZincPlusPiopF2::<F2Types<D>, TinyF2Uair, D>::prove_f2_full(
            &mut pt,
            &pp,
            &trace,
            &[],
            num_vars,
            |_| DynamicPolynomialF::<BinaryFieldGF128>::ZERO,
            4,
        )
        .expect("prove should succeed");

        // Flip a bit in a'.
        let mut tampered = *proof.open.lifted_claim.words();
        tampered[0] ^= 1;
        proof.open.lifted_claim = BinaryF2Poly::<7>::from_words(tampered);

        let mut vt = Blake3Transcript::new();
        let err = ZincPlusPiopF2::<F2Types<D>, TinyF2Uair, D>::verify_f2_full(
            &mut vt,
            &pp,
            &proof,
            &[],
            /* public_binary_trace */ &[],
            num_vars,
            2,
            |_| ImpossibleIdeal,
        )
        .expect_err("tampered open must be rejected");

        assert!(
            matches!(
                err,
                F2FullVerifyError::Open(
                    F2OpenError::EvalConsistency | F2OpenError::LiftDischarge { .. }
                )
            ),
            "expected Open(EvalConsistency | LiftDischarge), got {err:?}",
        );
    }

    /// End-to-end **sound** Hadamard discharge: drive the full
    /// commit → `prove_f2_full_with_hadamard` →
    /// `verify_f2_full_with_hadamard` pipeline on a 3-column
    /// `W = U ⊙ V` trace.
    ///
    /// Asserts the honest proof round-trips — which exercises the second
    /// multipoint-eval + open at `r*_H` and the parent-eval binding
    /// check (`hadamard_evals_at_rstar_h[distinct] ==
    /// uair.hadamard_pair_evals`, satisfied by construction on every
    /// honest proof) — and that the soundness machinery rejects the
    /// reachable tampers:
    ///   - a flipped `W` bit            → the Hadamard zerocheck,
    ///   - a swapped parent-eval        → the in-flow recombination,
    ///   - a dropped second open        → `MissingHadamardDischarge`,
    ///   - a tampered per-col eval@r*_H → the second multipoint-eval.
    ///
    /// (The `HadamardParentEvalBindingMismatch` variant is defence in
    /// depth: it fires only for a *malicious* prover that ships fake
    /// bit-slice MLEs decoupled from the commitment — the recombination
    /// + zerocheck over-determine the slices, so no single-field tamper
    /// of an honest proof can reach it without tripping an earlier gate.)
    #[test]
    fn prove_then_verify_f2_full_with_hadamard_roundtrips() {
        const D: usize = 32;
        let num_vars: usize = 6;
        let row_len: usize = 8;
        let poly_size = 1usize << num_vars;
        let num_rows = poly_size / row_len;
        assert_eq!(num_rows * row_len, poly_size);

        fn project_scalar(scalar: &BinaryPoly<32>) -> DynamicPolynomialF<BinaryFieldGF128> {
            DynamicPolynomialF {
                coeffs: scalar
                    .iter()
                    .map(|b| {
                        if b.into_inner() {
                            BinaryFieldGF128::one()
                        } else {
                            BinaryFieldGF128::zero()
                        }
                    })
                    .collect(),
            }
        }

        let mut rng_local = rng();
        let u_words: Vec<u32> = (0..poly_size).map(|_| rng_local.random::<u32>()).collect();
        let v_words: Vec<u32> = (0..poly_size).map(|_| rng_local.random::<u32>()).collect();
        let w_words: Vec<u32> = u_words.iter().zip(&v_words).map(|(a, b)| a & b).collect();
        let mk = |words: &[u32]| {
            DenseMultilinearExtension::from_evaluations_vec(
                num_vars,
                words.iter().map(|&x| BinaryPoly::<D>::from(x)).collect(),
                BinaryPoly::default(),
            )
        };
        let make_trace =
            |w: &[u32]| -> UairTrace<'static, BinaryPoly<D>, BinaryPoly<D>, D> {
                UairTrace {
                    binary_poly: vec![mk(&u_words), mk(&v_words), mk(w)].into(),
                    arbitrary_poly: vec![].into(),
                    int: vec![].into(),
                }
            };
        let trace = make_trace(&w_words);

        let specs = [crate::f2_hadamard::F2HadamardSpec::plain(0, 1, 2)];

        let lc = <F2Types<D> as F2ZincTypes<D>>::BinaryLc::new(row_len);
        let pp: ZipPlusParams<
            <F2Types<D> as F2ZincTypes<D>>::BinaryZt,
            <F2Types<D> as F2ZincTypes<D>>::BinaryLc,
        > = ZipPlusParams::new(num_vars, num_rows, lc);

        let prove = |trace: &UairTrace<'static, BinaryPoly<D>, BinaryPoly<D>, D>| {
            let mut pt = Blake3Transcript::new();
            ZincPlusPiopF2::<F2Types<D>, HadF2Uair, D>::prove_f2_full_with_hadamard(
                &mut pt,
                &pp,
                trace,
                /* virtual_specs */ &[],
                /* bit_op_specs  */ &[],
                &specs,
                num_vars,
                project_scalar,
                /* num_column_openings */ 4,
            )
            .expect("prove_f2_full_with_hadamard should succeed")
        };
        let verify = |proof: &F2FullProof<D>| {
            let mut vt = Blake3Transcript::new();
            ZincPlusPiopF2::<F2Types<D>, HadF2Uair, D>::verify_f2_full_with_hadamard::<
                ImpossibleIdeal,
            >(
                &mut vt,
                &pp,
                proof,
                /* virtual_specs */ &[],
                /* bit_op_specs  */ &[],
                &specs,
                /* public_binary_trace */ &[],
                num_vars,
                /* num_primary_columns */ 3,
                |_ideal| ImpossibleIdeal,
            )
        };

        // -- honest round-trip --
        let proof = prove(&trace);
        assert!(proof.hadamard_multipoint_eval.is_some());
        assert!(proof.hadamard_open.is_some());
        assert_eq!(proof.hadamard_evals_at_rstar_h.len(), 3);
        assert_eq!(proof.hadamard_open_evals_at_r0h.len(), 3);
        assert_eq!(proof.uair.hadamard_pair_evals.len(), 3);
        // The binding equality holds by construction on an honest proof:
        // the distinct cols are 0,1,2, so parent_evals[i] == the bound
        // per-column eval at r*_H for column i.
        for (i, &pe) in proof.uair.hadamard_pair_evals.iter().enumerate() {
            assert_eq!(
                pe, proof.hadamard_evals_at_rstar_h[i],
                "honest parent-eval {i} must equal the bound r*_H eval",
            );
        }
        verify(&proof).expect("honest sound-discharge proof must verify");

        // -- corrupt W bit: the Hadamard zerocheck rejects --
        let mut w_bad = w_words.clone();
        w_bad[0] ^= 1;
        let proof_bad_w = prove(&make_trace(&w_bad));
        assert!(
            matches!(
                verify(&proof_bad_w),
                Err(F2FullVerifyError::Uair(F2VerifyError::Hadamard(_)))
            ),
            "flipped W must be rejected by the zerocheck",
        );

        // -- swapped parent-eval: the in-flow recombination rejects --
        let mut proof_bad_parent = proof.clone();
        let one = BinaryFieldGF128::one();
        proof_bad_parent.uair.hadamard_pair_evals[0] += &one;
        assert!(
            matches!(
                verify(&proof_bad_parent),
                Err(F2FullVerifyError::Uair(F2VerifyError::HadamardRecombination(_)))
            ),
            "a swapped parent-eval must be rejected by the recombination",
        );

        // -- dropped second open: explicit MissingHadamardDischarge --
        let mut proof_no_open = proof.clone();
        proof_no_open.hadamard_open = None;
        assert!(
            matches!(
                verify(&proof_no_open),
                Err(F2FullVerifyError::MissingHadamardDischarge)
            ),
            "a missing second open must be rejected",
        );

        // -- tampered per-col eval at r*_H: the second mp rejects --
        let mut proof_bad_up = proof.clone();
        proof_bad_up.hadamard_evals_at_rstar_h[2] += &one;
        assert!(
            matches!(
                verify(&proof_bad_up),
                Err(F2FullVerifyError::MultipointEval(_))
            ),
            "a tampered r*_H eval must be rejected by the second multipoint-eval",
        );
    }

    /// End-to-end sound discharge with the Phase-B **operand** features
    /// (row-shift / complement / XOR combo), driven through the full
    /// commit → `prove_f2_full_with_hadamard` →
    /// `verify_f2_full_with_hadamard` flow on a 5-column trace. Three
    /// relations exercise each operand kind:
    ///   R1 (row-shift): `col0 ⊙ col0^↓1 = col1`,
    ///   R2 (complement): `(1−col0) ⊙ col2 = col3`,
    ///   R3 (XOR combo, C14-shape): `(col0 + col0^↓2) ⊙ (col0^↓1 + col0^↓2)
    ///       = (col4 + col0^↓2)`.
    /// The `Δ = 0` base cols (0,1,2,3,4) are bound by the second-mp
    /// discharge; the `Δ ≠ 0` shifted pairs `(0,1)` and `(0,2)` are trusted
    /// (the row-shift gap). Asserts honest acceptance and a flipped-`col1`
    /// rejection by the zerocheck.
    #[test]
    fn prove_then_verify_f2_full_with_operand_hadamards_roundtrips() {
        const D: usize = 32;
        let num_vars: usize = 6;
        let row_len: usize = 8;
        let poly_size = 1usize << num_vars;
        let num_rows = poly_size / row_len;
        let n = poly_size;
        assert_eq!(num_rows * row_len, poly_size);

        fn project_scalar(scalar: &BinaryPoly<32>) -> DynamicPolynomialF<BinaryFieldGF128> {
            DynamicPolynomialF {
                coeffs: scalar
                    .iter()
                    .map(|b| {
                        if b.into_inner() {
                            BinaryFieldGF128::one()
                        } else {
                            BinaryFieldGF128::zero()
                        }
                    })
                    .collect(),
            }
        }

        let mut rng_local = rng();
        let a: Vec<u32> = (0..n).map(|_| rng_local.random::<u32>()).collect();
        let b: Vec<u32> = (0..n).map(|_| rng_local.random::<u32>()).collect();
        let sh = |arr: &[u32], d: usize, t: usize| if t + d < n { arr[t + d] } else { 0 };
        let mask = u32::MAX;
        // col1 = col0 ⊙ col0^↓1
        let col1: Vec<u32> = (0..n).map(|t| a[t] & sh(&a, 1, t)).collect();
        // col3 = (1−col0) ⊙ col2   (col2 = b)
        let col3: Vec<u32> = (0..n).map(|t| ((!a[t]) & mask) & b[t]).collect();
        // col4 = ((col0+col0^↓2) ⊙ (col0^↓1+col0^↓2)) ^ col0^↓2
        let col4: Vec<u32> = (0..n)
            .map(|t| {
                let u = a[t] ^ sh(&a, 2, t);
                let v = sh(&a, 1, t) ^ sh(&a, 2, t);
                (u & v) ^ sh(&a, 2, t)
            })
            .collect();

        let mk = |words: &[u32]| {
            DenseMultilinearExtension::from_evaluations_vec(
                num_vars,
                words.iter().map(|&x| BinaryPoly::<D>::from(x)).collect(),
                BinaryPoly::default(),
            )
        };
        let make_trace = |c1: &[u32]| -> UairTrace<'static, BinaryPoly<D>, BinaryPoly<D>, D> {
            UairTrace {
                binary_poly: vec![mk(&a), mk(c1), mk(&b), mk(&col3), mk(&col4)].into(),
                arbitrary_poly: vec![].into(),
                int: vec![].into(),
            }
        };
        let trace = make_trace(&col1);

        use crate::f2_hadamard::{F2HadamardSpec, F2Operand, F2OperandTerm};
        let specs = [
            // R1: row-shift.
            F2HadamardSpec {
                u: F2Operand::col(0),
                v: F2Operand::shifted(0, 1),
                w: F2Operand::col(1),
            },
            // R2: complement.
            F2HadamardSpec {
                u: F2Operand::col(0).complemented(),
                v: F2Operand::col(2),
                w: F2Operand::col(3),
            },
            // R3: XOR combo (C14-shape).
            F2HadamardSpec {
                u: F2Operand::xor(vec![
                    F2OperandTerm { col: 0, row_shift: 0 },
                    F2OperandTerm { col: 0, row_shift: 2 },
                ]),
                v: F2Operand::xor(vec![
                    F2OperandTerm { col: 0, row_shift: 1 },
                    F2OperandTerm { col: 0, row_shift: 2 },
                ]),
                w: F2Operand::xor(vec![
                    F2OperandTerm { col: 4, row_shift: 0 },
                    F2OperandTerm { col: 0, row_shift: 2 },
                ]),
            },
        ];

        let lc = <F2Types<D> as F2ZincTypes<D>>::BinaryLc::new(row_len);
        let pp: ZipPlusParams<
            <F2Types<D> as F2ZincTypes<D>>::BinaryZt,
            <F2Types<D> as F2ZincTypes<D>>::BinaryLc,
        > = ZipPlusParams::new(num_vars, num_rows, lc);

        let prove = |trace: &UairTrace<'static, BinaryPoly<D>, BinaryPoly<D>, D>| {
            let mut pt = Blake3Transcript::new();
            ZincPlusPiopF2::<F2Types<D>, HadOpF2Uair, D>::prove_f2_full_with_hadamard(
                &mut pt, &pp, trace, &[], &[], &specs, num_vars, project_scalar, 4,
            )
            .expect("prove_f2_full_with_hadamard should succeed")
        };
        let verify = |proof: &F2FullProof<D>| {
            let mut vt = Blake3Transcript::new();
            ZincPlusPiopF2::<F2Types<D>, HadOpF2Uair, D>::verify_f2_full_with_hadamard::<
                ImpossibleIdeal,
            >(
                &mut vt, &pp, proof, &[], &[], &specs, &[], num_vars, 5, |_ideal| ImpossibleIdeal,
            )
        };

        // -- honest round-trip --
        let proof = prove(&trace);
        // Distinct pairs: (0,0),(0,1),(0,2),(1,0),(2,0),(3,0),(4,0) = 7.
        assert_eq!(proof.uair.hadamard_pair_evals.len(), 7);
        verify(&proof).expect("honest operand sound-discharge proof must verify");

        // -- corrupt col1 (R1's W): zerocheck rejects --
        let mut col1_bad = col1.clone();
        col1_bad[0] ^= 1;
        let proof_bad = prove(&make_trace(&col1_bad));
        assert!(
            matches!(
                verify(&proof_bad),
                Err(F2FullVerifyError::Uair(F2VerifyError::Hadamard(_)))
            ),
            "a flipped W bit must be rejected by the zerocheck",
        );
    }

    // -- Virtual binary_poly column support ---------------------------
    //
    // `VirtualF2Uair` declares 3 binary_poly columns in its signature:
    //   primary col 0 = `a`            (committed)
    //   primary col 1 = `b`            (committed)
    //   virtual col 2 = `a XOR b`     (computed by both prover + verifier)
    //
    // The constraint asserts col_2 == col_0 XOR col_1, which holds by
    // construction since the virtual col IS the XOR. Practically this
    // exercises:
    //   - Prover materialises the virtual column from primary cols.
    //   - IC + sumcheck see 3 columns (2 primary + 1 virtual).
    //   - Verifier derives the virtual eval at r* from primary evals.
    //   - F_2[X] open only opens 2 primary cols.
    //
    // The `assert_zero(col_2 - (col_0 + col_1))` is automatically
    // satisfied by the materialisation; this lets us focus on the
    // wiring rather than the constraint mechanics.

    #[derive(Clone, Debug, Default)]
    struct VirtualF2Uair;

    impl Uair for VirtualF2Uair {
        type Ideal = ImpossibleIdeal;
        type Scalar = BinaryPoly<32>;

        fn signature() -> UairSignature {
            // total binary_poly = 2 primary + 1 virtual = 3.
            // The virtual col index inside `binary_poly` is the next
            // available slot (= 2). The signature counts the virtual
            // col in `total_cols.num_binary_poly_cols` so the IC sees
            // it; the trace itself only holds the 2 primary cols
            // (the prove path materialises col 2 inline).
            UairSignature::new(
                TotalColumnLayout::new(/* binary */ 3, 0, 0),
                PublicColumnLayout::default(),
                vec![],
                vec![],
                vec![],
            )
        }

        fn constrain_general<B, FromR, MulByScalar, IFromR>(
            b: &mut B,
            up: TraceRow<B::Expr>,
            _down: TraceRow<B::Expr>,
            _from_ref: FromR,
            _mbs: MulByScalar,
            _ideal_from_ref: IFromR,
        ) where
            B: ConstraintBuilder,
            FromR: Fn(&Self::Scalar) -> B::Expr,
            MulByScalar: Fn(&B::Expr, &Self::Scalar) -> Option<B::Expr>,
            IFromR: Fn(&Self::Ideal) -> B::Ideal,
        {
            // assert_zero(virtual - (a + b)) ≡ virtual = a XOR b.
            // In F_2 addition IS XOR, so this is a sum.
            let a = up.binary_poly[0].clone();
            let b_col = up.binary_poly[1].clone();
            let v = up.binary_poly[2].clone();
            b.assert_zero(v - (a + b_col));
        }
    }

    /// Build a satisfied trace for `VirtualF2Uair` and run the full
    /// pipeline (commit + IC + sumcheck + γ-batched open) on a
    /// single transcript, verifying both end-to-end and the
    /// virtual-col-derivation consistency.
    #[test]
    fn prove_then_verify_f2_full_with_virtual_column() {
        const D: usize = 32;
        let num_vars: usize = 6;
        let row_len: usize = 8;
        let poly_size = 1usize << num_vars;
        let num_rows = poly_size / row_len;
        assert_eq!(num_rows * row_len, poly_size);

        let mut rng_local = rng();

        // Two arbitrary primary columns `a` and `b`. The virtual col
        // `v = a XOR b` is materialised inside `prove_f2_full`.
        let a_vals: Vec<BinaryPoly<D>> = (0..poly_size)
            .map(|_| BinaryPoly::from(rng_local.random::<u32>()))
            .collect();
        let b_vals: Vec<BinaryPoly<D>> = (0..poly_size)
            .map(|_| BinaryPoly::from(rng_local.random::<u32>()))
            .collect();
        let a = DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            a_vals,
            BinaryPoly::default(),
        );
        let b = DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            b_vals,
            BinaryPoly::default(),
        );
        let trace: UairTrace<'static, BinaryPoly<D>, BinaryPoly<D>, D> = UairTrace {
            binary_poly: vec![a.clone(), b.clone()].into(),
            arbitrary_poly: vec![].into(),
            int: vec![].into(),
        };

        // `v = a XOR b`. Primary col idx 0 and 1.
        let virtual_specs = vec![F2VirtualBpSpec {
            primary_col_indices: vec![0, 1],
        }];

        let lc = <F2Types<D> as F2ZincTypes<D>>::BinaryLc::new(row_len);
        let pp: ZipPlusParams<
            <F2Types<D> as F2ZincTypes<D>>::BinaryZt,
            <F2Types<D> as F2ZincTypes<D>>::BinaryLc,
        > = ZipPlusParams::new(num_vars, num_rows, lc);

        // -- Prove (full pipeline) --------------------------------
        let mut prover_transcript = Blake3Transcript::new();
        let proof = ZincPlusPiopF2::<F2Types<D>, VirtualF2Uair, D>::prove_f2_full(
            &mut prover_transcript,
            &pp,
            &trace,
            &virtual_specs,
            num_vars,
            |_| DynamicPolynomialF::<BinaryFieldGF128>::ZERO,
            4,
        )
        .expect("prove_f2_full with virtual col should succeed");

        // Only the 2 primary cols are committed.
        // 2 primary cols paired into 1 packed storage column.
        assert_eq!(proof.commitment.batch_size, 2_usize.div_ceil(2));

        // -- Verify (full pipeline) -------------------------------
        let mut verifier_transcript = Blake3Transcript::new();
        let subclaim = ZincPlusPiopF2::<F2Types<D>, VirtualF2Uair, D>::verify_f2_full(
            &mut verifier_transcript,
            &pp,
            &proof,
            &virtual_specs,
            /* public_binary_trace */ &[],
            num_vars,
            /* num_primary_columns */ 2,
            |_ideal| ImpossibleIdeal,
        )
        .expect("verify_f2_full with virtual col should succeed");

        assert_eq!(subclaim.primary_column_evals.len(), 2);
        assert_eq!(subclaim.virtual_column_evals.len(), 1);

        // Sanity: the derived virtual eval matches XOR of primary evals.
        let xor =
            subclaim.primary_column_evals[0] + subclaim.primary_column_evals[1];
        assert_eq!(subclaim.virtual_column_evals[0], xor);
    }

    /// A prover supplying mismatched virtual_specs (or none) on a
    /// trace whose IC + sumcheck expected the materialised virtual
    /// column must be rejected. Here we let the prover succeed with
    /// correct specs but feed the verifier *empty* specs — the
    /// verifier should fail (either group-count mismatch from the
    /// sumcheck's extra group, or a downstream check).
    #[test]
    fn verify_f2_full_rejects_missing_virtual_spec() {
        const D: usize = 32;
        let num_vars: usize = 4;
        let row_len: usize = 4;
        let poly_size = 1usize << num_vars;
        let num_rows = poly_size / row_len;
        let mut rng_local = rng();
        let a_vals: Vec<BinaryPoly<D>> = (0..poly_size)
            .map(|_| BinaryPoly::from(rng_local.random::<u32>()))
            .collect();
        let b_vals: Vec<BinaryPoly<D>> = (0..poly_size)
            .map(|_| BinaryPoly::from(rng_local.random::<u32>()))
            .collect();
        let a = DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            a_vals,
            BinaryPoly::default(),
        );
        let b = DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            b_vals,
            BinaryPoly::default(),
        );
        let trace: UairTrace<'static, BinaryPoly<D>, BinaryPoly<D>, D> = UairTrace {
            binary_poly: vec![a, b].into(),
            arbitrary_poly: vec![].into(),
            int: vec![].into(),
        };

        let virtual_specs = vec![F2VirtualBpSpec {
            primary_col_indices: vec![0, 1],
        }];

        let lc = <F2Types<D> as F2ZincTypes<D>>::BinaryLc::new(row_len);
        let pp: ZipPlusParams<
            <F2Types<D> as F2ZincTypes<D>>::BinaryZt,
            <F2Types<D> as F2ZincTypes<D>>::BinaryLc,
        > = ZipPlusParams::new(num_vars, num_rows, lc);

        let mut pt = Blake3Transcript::new();
        let proof = ZincPlusPiopF2::<F2Types<D>, VirtualF2Uair, D>::prove_f2_full(
            &mut pt,
            &pp,
            &trace,
            &virtual_specs,
            num_vars,
            |_| DynamicPolynomialF::<BinaryFieldGF128>::ZERO,
            4,
        )
        .expect("prove should succeed with correct specs");

        // Verifier passes empty specs. The sumcheck proof has 3
        // groups (primary 0, primary 1, virtual 0); we'd claim 2
        // primary cols + 0 virtual = 2 total. Mismatch.
        let mut vt = Blake3Transcript::new();
        let err = ZincPlusPiopF2::<F2Types<D>, VirtualF2Uair, D>::verify_f2_full(
            &mut vt,
            &pp,
            &proof,
            /* virtual_specs */ &[],
            /* public_binary_trace */ &[],
            num_vars,
            2,
            |_| ImpossibleIdeal,
        )
        .expect_err("verifier must reject when virtual_specs don't match the prover's");

        // Any rejection is fine — the structural mismatch can
        // surface as GroupCountMismatch (sumcheck has 3 groups but
        // verifier expected 2) or a downstream check that fails
        // because the sumcheck challenges diverge.
        let _ = err;
    }

    // ---------------------------------------------------------------
    // SHA-256 F_2[X] UAIR — full prove/verify roundtrip.
    //
    // Exercises the real F_2 SHA-256 arithmetisation
    // (`zinc_test_uair::Sha256F2Uair`) through the bundled
    // `prove_f2_full` / `verify_f2_full` entry points. With
    // `NUM_COMPRESSIONS = 7`, the trace fits in `num_vars = 9` =
    // 2^9 = 512 rows (480 active + 32 slack).
    // ---------------------------------------------------------------

    #[test]
    fn prove_then_verify_sha256_f2_roundtrips() {
        use crypto_primitives::crypto_bigint_int::Int;
        use zinc_test_uair::{
            GenerateRandomTrace, Sha256F2Ideal, Sha256F2Uair, sha256_f2_project_ideal,
            sha256_f2_project_scalar,
        };

        const D: usize = 32;
        type R = Int<4>;
        type U = Sha256F2Uair<R>;

        // Trace parameters from the SHA F_2 UAIR spec.
        let num_vars: usize = 9; // 2^9 = 512 ≥ 7·68 + 4 = 480
        let row_len: usize = 32;
        let poly_size = 1usize << num_vars;
        let num_rows = poly_size / row_len;
        assert_eq!(num_rows * row_len, poly_size);

        // Generate honest trace.
        let mut rng_local = rng();
        let trace = U::generate_random_trace(num_vars, &mut rng_local);

        // Set up PCS params with rate-1/4 RAA code.
        let lc = <F2Types<D> as F2ZincTypes<D>>::BinaryLc::new(row_len);
        let pp: ZipPlusParams<
            <F2Types<D> as F2ZincTypes<D>>::BinaryZt,
            <F2Types<D> as F2ZincTypes<D>>::BinaryLc,
        > = ZipPlusParams::new(num_vars, num_rows, lc);

        // Prove.
        let mut prover_transcript = Blake3Transcript::new();
        let proof = ZincPlusPiopF2::<F2Types<D>, U, D>::prove_f2_full(
            &mut prover_transcript,
            &pp,
            &trace,
            /* virtual_specs */ &[],
            num_vars,
            sha256_f2_project_scalar::<R>,
            // Small num_column_openings to keep the test snappy; the
            // soundness-critical bench uses
            // `recommended_num_column_openings(4) = 987`.
            /* num_column_openings */ 4,
        )
        .expect("prove_f2_full on SHA F_2 UAIR should succeed");

        // Sanity-check the proof shape. `batch_size` is the paired
        // batch size of the WITNESS columns — public cols are no
        // longer committed (verifier encodes them locally).
        assert_eq!(
            proof.commitment.batch_size,
            zinc_test_uair::sha256_f2::cols::NUM_BIN_WIT.div_ceil(2)
        );

        // Verify on a fresh transcript.
        let mut verifier_transcript = Blake3Transcript::new();
        let subclaim = ZincPlusPiopF2::<F2Types<D>, U, D>::verify_f2_full(
            &mut verifier_transcript,
            &pp,
            &proof,
            /* virtual_specs */ &[],
            /* public_binary_trace */ &trace.binary_poly[..zinc_test_uair::sha256_f2::cols::NUM_BIN_PUB],
            num_vars,
            zinc_test_uair::sha256_f2::cols::NUM_BIN,
            |ideal: &IdealOrZero<Sha256F2Ideal>| sha256_f2_project_ideal(ideal),
        )
        .expect("verify_f2_full on SHA F_2 UAIR should succeed");

        // Sanity: subclaim is well-shaped.
        assert_eq!(subclaim.sumcheck_point.len(), num_vars);
        assert_eq!(
            subclaim.primary_column_evals.len(),
            zinc_test_uair::sha256_f2::cols::NUM_BIN
        );
        assert_eq!(subclaim.virtual_column_evals.len(), 0);
    }

    /// First real SHA-256 Hadamard relation discharged end-to-end on the
    /// `sha256_f2` trace: **C12** `W_UEF = W_E ⊙ W_E^↓1` (the Ch[t] AND
    /// `u_ef[t] = e[t] & e[t−1]`, `sha256_f2.rs`).
    ///
    /// Under the codebase `↓Δ = row i → col[i+Δ]` convention (`ShiftSpec`,
    /// `build_shifted_bit_slice_mles`), the `i−1` fill is registered by
    /// shifting the result column up too: at zerocheck row `t`,
    /// `W_UEF^↓1[t] = W_E^↓1[t] & W_E[t]`. Pairs: `(W_E,0)` bound by the
    /// second-mp discharge; `(W_E,1)` and `(W_UEF,1)` are row-shifted →
    /// trusted (the honest-prover row-shift gap). Confirms the operand
    /// machinery drives a real SHA AND relation honestly. (Zerocheck
    /// rejection is covered by the synthetic operand tests; corrupting a
    /// SHA `W_UEF` cell here could trip the IC first — `W_UEF` also feeds
    /// the C6b adder — so we keep this to the honest milestone.)
    #[test]
    fn sha256_f2_c12_hadamard_roundtrips() {
        use crypto_primitives::crypto_bigint_int::Int;
        use zinc_test_uair::sha256_f2::cols;
        use zinc_test_uair::{
            GenerateRandomTrace, Sha256F2Ideal, Sha256F2Uair, sha256_f2_project_ideal,
            sha256_f2_project_scalar,
        };

        const D: usize = 32;
        type R = Int<4>;
        type U = Sha256F2Uair<R>;

        let num_vars: usize = 9;
        let row_len: usize = 32;
        let poly_size = 1usize << num_vars;
        let num_rows = poly_size / row_len;

        let mut rng_local = rng();
        let trace = U::generate_random_trace(num_vars, &mut rng_local);

        let lc = <F2Types<D> as F2ZincTypes<D>>::BinaryLc::new(row_len);
        let pp: ZipPlusParams<
            <F2Types<D> as F2ZincTypes<D>>::BinaryZt,
            <F2Types<D> as F2ZincTypes<D>>::BinaryLc,
        > = ZipPlusParams::new(num_vars, num_rows, lc);

        // C12 in the `i+Δ` convention: W_E^↓1 ⊙ W_E = W_UEF^↓1.
        use crate::f2_hadamard::{F2HadamardSpec, F2Operand};
        let specs = [F2HadamardSpec {
            u: F2Operand::shifted(cols::W_E, 1),
            v: F2Operand::col(cols::W_E),
            w: F2Operand::shifted(cols::W_UEF, 1),
        }];

        // Prove + verify the bundled flow with C12 registered.
        let mut pt = Blake3Transcript::new();
        let proof = ZincPlusPiopF2::<F2Types<D>, U, D>::prove_f2_full_with_hadamard(
            &mut pt,
            &pp,
            &trace,
            /* virtual_specs */ &[],
            /* bit_op_specs  */ &[],
            &specs,
            num_vars,
            sha256_f2_project_scalar::<R>,
            /* num_column_openings */ 4,
        )
        .expect("prove_f2_full_with_hadamard on SHA C12 should succeed");
        assert!(proof.uair.hadamard_proof.is_some());

        let mut vt = Blake3Transcript::new();
        ZincPlusPiopF2::<F2Types<D>, U, D>::verify_f2_full_with_hadamard::<Sha256F2Ideal>(
            &mut vt,
            &pp,
            &proof,
            /* virtual_specs */ &[],
            /* bit_op_specs  */ &[],
            &specs,
            /* public_binary_trace */ &trace.binary_poly[..cols::NUM_BIN_PUB],
            num_vars,
            cols::NUM_BIN,
            |ideal: &IdealOrZero<Sha256F2Ideal>| sha256_f2_project_ideal(ideal),
        )
        .expect("honest SHA C12 proof must verify");
    }

}
