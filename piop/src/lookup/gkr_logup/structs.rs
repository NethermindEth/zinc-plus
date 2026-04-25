//! Proof, error, and intermediate types for the GKR-LogUp lookup
//! protocol with polynomial-valued chunk lifts.
//!
//! Distinct from `super::super::structs` (the legacy/stub data types
//! for a different lookup design) — the GKR-LogUp module keeps its own
//! types so the existing scaffolding stays untouched.

use crypto_primitives::PrimeField;
use thiserror::Error;
use zinc_poly::{
    EvaluationError,
    univariate::dynamic::over_field::{DynamicPolyVecF, DynamicPolynomialF},
    utils::ArithErrors,
};
use zinc_transcript::traits::{ConstTranscribable, GenTranscribable, Transcribable};
use zinc_uair::LookupTableType;
use zinc_utils::{add, mul};

use crate::sumcheck::{SumCheckError, SumcheckProof};

// ---------------------------------------------------------------------------
// GKR fraction-tree proof types (ported verbatim from agentic-approach-v1).
// ---------------------------------------------------------------------------

/// Proof for a single GKR fractional sumcheck.
///
/// Proves `Σ_{x ∈ {0,1}^d} p(x)/q(x) = root_p/root_q` using a
/// layer-by-layer GKR protocol with `d` levels.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GkrFractionProof<F: PrimeField> {
    /// Root numerator `P = Σ p_i · Π_{j≠i} q_j`.
    pub root_p: F,
    /// Root denominator `Q = Π q_i`.
    pub root_q: F,
    /// Per-layer proofs, one for each GKR level (0..d).
    pub layer_proofs: Vec<GkrLayerProof<F>>,
}

/// Proof for a single GKR layer.
///
/// At GKR layer `k` (with `2^k` entries), the prover runs a sumcheck
/// over `k` variables (for `k ≥ 1`) or a direct check (for `k = 0`),
/// then reveals the four child-MLE evaluations at the subclaim point.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GkrLayerProof<F: PrimeField> {
    /// Sumcheck proof for this layer (`None` for layer 0 which has 0 variables).
    pub sumcheck_proof: Option<SumcheckProof<F>>,
    /// Evaluation of the left-child numerator MLE at the subclaim point.
    pub p_left: F,
    /// Evaluation of the right-child numerator MLE at the subclaim point.
    pub p_right: F,
    /// Evaluation of the left-child denominator MLE at the subclaim point.
    pub q_left: F,
    /// Evaluation of the right-child denominator MLE at the subclaim point.
    pub q_right: F,
}

/// Proof for L batched GKR fractional sumchecks run layer-by-layer.
///
/// All L trees share the same depth and challenge trajectory; the
/// per-layer sumcheck is batched into a single sumcheck over 1 + 4L
/// MLEs at degree 3.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchedGkrFractionProof<F: PrimeField> {
    /// Per-tree root numerators: `roots_p[ℓ]`.
    pub roots_p: Vec<F>,
    /// Per-tree root denominators: `roots_q[ℓ]`.
    pub roots_q: Vec<F>,
    /// Per-layer proofs, one for each GKR level (0..d).
    pub layer_proofs: Vec<BatchedGkrLayerProof<F>>,
}

/// Proof for a single layer of the batched GKR protocol.
///
/// At layer k, the L trees share one sumcheck proof (for k ≥ 1) or a
/// direct check (k = 0), and each tree contributes 4 evaluations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchedGkrLayerProof<F: PrimeField> {
    /// Shared sumcheck proof for this layer (`None` for k = 0).
    pub sumcheck_proof: Option<SumcheckProof<F>>,
    /// Per-tree left-child numerator evaluations: `p_lefts[ℓ]`.
    pub p_lefts: Vec<F>,
    /// Per-tree right-child numerator evaluations: `p_rights[ℓ]`.
    pub p_rights: Vec<F>,
    /// Per-tree left-child denominator evaluations: `q_lefts[ℓ]`.
    pub q_lefts: Vec<F>,
    /// Per-tree right-child denominator evaluations: `q_rights[ℓ]`.
    pub q_rights: Vec<F>,
}

// ---------------------------------------------------------------------------
// Top-level lookup proof + group payload (chunks-in-clear poly-lift design)
// ---------------------------------------------------------------------------

/// Per-group proof for one [`LookupTableType`] in the new GKR-LogUp
/// protocol.
///
/// **Chunks are neither sent nor committed.** The prover sends only:
///   1. polynomial-valued **chunk lifts** `c_k'^(ell) = MLE[v_k^(ell)](r_inner) ∈ F_q[X]_{<chunk_width}`,
///      one per `(ell, k)` — `chunk_width` field elements each;
///   2. aggregated multiplicities `m_agg^(ell)[j]`;
///   3. the witness-side and table-side GKR fractional sumcheck proofs.
///
/// The protocol layer (caller of `verify_group`) receives a
/// [`GkrLogupGroupSubclaim`] containing the verifier-reconstructed
/// combined polynomial `c^(ell) = Σ_k X^{k·chunk_width} · c_k'^(ell)`
/// at the GKR descent row-half `r_inner`. The caller binds that
/// polynomial to the parent column's PCS commitment via a Zip+ opening
/// at `r_inner`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GkrLogupGroupProof<F: PrimeField> {
    /// `chunk_lifts[ell][k]` is the polynomial-valued chunk MLE
    /// `c_k'^(ell) = MLE[v_k^(ell)](r_inner) ∈ F_q[X]_{<chunk_width}`.
    pub chunk_lifts: Vec<Vec<DynamicPolynomialF<F>>>,
    /// `aggregated_multiplicities[ell]` of length `subtable_size`.
    pub aggregated_multiplicities: Vec<Vec<F>>,
    /// Batched witness-side GKR fractional sumcheck (L trees).
    pub witness_gkr: BatchedGkrFractionProof<F>,
    /// Single-tree table-side GKR fractional sumcheck.
    pub table_gkr: GkrFractionProof<F>,
}

/// Static metadata for one lookup group, ported alongside the proof so
/// the verifier can rebuild the subtable, shifts, and column locations
/// without having the original `LookupColumnSpec`s on hand.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GkrLogupGroupMeta {
    pub table_type: LookupTableType,
    pub num_lookups: usize,
    pub num_chunks: usize,
    pub chunk_width: usize,
    pub witness_len: usize,
    /// Full-trace column indices of the L parent columns in this group.
    pub parent_columns: Vec<usize>,
}

/// Per-lookup verifier sub-claim emitted by `verify_group` for the
/// protocol layer's Zip+ binding.
///
/// For each lookup `ell`, the protocol must verify a Zip+ opening of
/// the parent column `parent_columns[ell]` at point `r_inner`, expecting
/// the polynomial-valued evaluation `combined_polynomial[ell]`.
#[derive(Clone, Debug)]
pub struct GkrLogupGroupSubclaim<F: PrimeField> {
    /// Row-half of the GKR descent point — the point at which the
    /// parent column's MLE must be opened (length = `n_vars`).
    pub r_inner: Vec<F>,
    /// Per-lookup combined polynomial-valued claim at `r_inner`.
    /// `combined_polynomial[ell] ∈ F_q[X]_{<width}` should equal
    /// `MLE[v^(ell)](r_inner)` if all chunk lifts are honest.
    pub combined_polynomial: Vec<DynamicPolynomialF<F>>,
    /// Pass-through of `GkrLogupGroupMeta::parent_columns` for caller
    /// convenience.
    pub parent_columns: Vec<usize>,
}

/// Top-level GKR-LogUp lookup proof: one per-group payload + per-group
/// metadata, in group order. Empty (`groups: vec![]`) when the UAIR
/// declares no lookup specs.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GkrLogupLookupProof<F: PrimeField> {
    pub groups: Vec<GkrLogupGroupProof<F>>,
    pub group_meta: Vec<GkrLogupGroupMeta>,
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors specific to the GKR-LogUp lookup protocol with polynomial-
/// valued chunk lifts.
///
/// Distinct from the legacy `super::super::structs::LookupError` so the
/// existing stub scaffolding is unchanged.
#[derive(Debug, Error)]
pub enum GkrLogupError<F: PrimeField> {
    /// Wraps an underlying sumcheck failure.
    #[error("sumcheck error: {0}")]
    SumCheckError(SumCheckError<F>),
    /// Failed to build the eq polynomial.
    #[error("failed to build eq polynomial: {0}")]
    EqBuildError(ArithErrors),
    /// Failed to evaluate an MLE.
    #[error("MLE evaluation error: {0}")]
    MleEvaluationError(EvaluationError),
    /// A witness entry is not in the lookup table.
    #[error("witness entry not found in lookup table")]
    WitnessNotInTable,
    /// Multiplicity sum does not match the expected witness count.
    #[error("multiplicity sum mismatch: expected {expected}, got {got}")]
    MultiplicitySumMismatch { expected: u64, got: u64 },
    /// GKR root cross-check failed.
    #[error("GKR root cross-check failed")]
    GkrRootMismatch,
    /// GKR leaf-level claim mismatch.
    #[error("GKR leaf-level claim mismatch")]
    GkrLeafMismatch,
    /// GKR layer-0 direct check failed.
    #[error("GKR layer-0 check failed: expected {expected:?}, got {got:?}")]
    GkrLayer0Mismatch {
        /// Expected value.
        expected: F,
        /// Actual value.
        got: F,
    },
    /// Final-evaluation check (within a GKR layer subclaim) failed.
    #[error("final evaluation check failed: expected {expected:?}, got {got:?}")]
    FinalEvaluationMismatch {
        /// Expected evaluation.
        expected: F,
        /// Actual evaluation.
        got: F,
    },
    /// Polynomial-valued chunk lift `c_k'` projects to a value other
    /// than the GKR-bound scalar chunk claim `c_k`.
    #[error("chunk lift consistency check failed: psi(c_k') != c_k")]
    ChunkLiftMismatch,
    /// The Zip+ opening of the parent column at the GKR descent point
    /// disagrees with the verifier's reconstructed combined polynomial.
    #[error("parent PCS opening mismatch at GKR descent point")]
    ParentPcsMismatch,
}

impl<F: PrimeField> From<SumCheckError<F>> for GkrLogupError<F> {
    fn from(e: SumCheckError<F>) -> Self {
        Self::SumCheckError(e)
    }
}

impl<F: PrimeField> From<ArithErrors> for GkrLogupError<F> {
    fn from(e: ArithErrors) -> Self {
        Self::EqBuildError(e)
    }
}

impl<F: PrimeField> From<EvaluationError> for GkrLogupError<F> {
    fn from(e: EvaluationError) -> Self {
        Self::MleEvaluationError(e)
    }
}

// ---------------------------------------------------------------------------
// Transcribable / GenTranscribable
//
// Encoding strategy: each composite proof type writes the field modulus
// once at its top, then encodes inner field elements as raw `F::Inner`
// bytes. Nested `SumcheckProof`s reuse their existing Transcribable impl
// (which independently embeds a modulus); the per-layer redundancy is
// O(num_layers · F::Modulus::NUM_BYTES) bytes — small and tractable.
// ---------------------------------------------------------------------------

#[allow(clippy::arithmetic_side_effects)]
fn write_f_inner_slice<'a, F: PrimeField>(buf: &'a mut [u8], xs: &[F]) -> &'a mut [u8]
where
    F::Inner: ConstTranscribable,
{
    let n = xs.len();
    let total = mul!(n, F::Inner::NUM_BYTES);
    let (head, rest) = buf.split_at_mut(total);
    for (i, x) in xs.iter().enumerate() {
        let off = mul!(i, F::Inner::NUM_BYTES);
        x.inner().write_transcription_bytes_exact(&mut head[off..add!(off, F::Inner::NUM_BYTES)]);
    }
    rest
}

#[allow(clippy::arithmetic_side_effects)]
fn read_f_inner_slice<'a, F: PrimeField>(
    bytes: &'a [u8],
    n: usize,
    cfg: &F::Config,
) -> (Vec<F>, &'a [u8])
where
    F::Inner: ConstTranscribable,
{
    let total = mul!(n, F::Inner::NUM_BYTES);
    let (head, rest) = bytes.split_at(total);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let off = mul!(i, F::Inner::NUM_BYTES);
        let inner = F::Inner::read_transcription_bytes_exact(&head[off..add!(off, F::Inner::NUM_BYTES)]);
        out.push(F::new_unchecked_with_cfg(inner, cfg));
    }
    (out, rest)
}

#[allow(clippy::arithmetic_side_effects)]
fn write_u32_prefix<'a>(buf: &'a mut [u8], val: usize) -> &'a mut [u8] {
    let v = u32::try_from(val).expect("length must fit in u32");
    v.write_transcription_bytes_exact(&mut buf[..u32::NUM_BYTES]);
    &mut buf[u32::NUM_BYTES..]
}

fn read_u32_prefix(bytes: &[u8]) -> (usize, &[u8]) {
    let (v, rest) = u32::read_transcription_bytes_subset(bytes);
    (
        usize::try_from(v).expect("u32 must fit in usize"),
        rest,
    )
}

// ---------- GkrLayerProof ----------

#[allow(clippy::arithmetic_side_effects)]
fn gkr_layer_num_bytes<F: PrimeField>(p: &GkrLayerProof<F>) -> usize
where
    F::Inner: ConstTranscribable,
    F::Modulus: ConstTranscribable,
{
    let sc = match &p.sumcheck_proof {
        Some(sc) => add!(SumcheckProof::<F>::LENGTH_NUM_BYTES, sc.get_num_bytes()),
        None => 0,
    };
    add!(1usize, add!(sc, mul!(4, F::Inner::NUM_BYTES)))
}

#[allow(clippy::arithmetic_side_effects)]
fn write_gkr_layer<'a, F: PrimeField>(
    buf: &'a mut [u8],
    p: &GkrLayerProof<F>,
) -> &'a mut [u8]
where
    F::Inner: ConstTranscribable,
    F::Modulus: ConstTranscribable,
{
    let mut buf = buf;
    if let Some(sc) = &p.sumcheck_proof {
        buf[0] = 1;
        buf = &mut buf[1..];
        buf = sc.write_transcription_bytes_subset(buf);
    } else {
        buf[0] = 0;
        buf = &mut buf[1..];
    }
    write_f_inner_slice(buf, &[p.p_left.clone(), p.p_right.clone(), p.q_left.clone(), p.q_right.clone()])
}

#[allow(clippy::arithmetic_side_effects)]
fn read_gkr_layer<'a, F: PrimeField>(
    bytes: &'a [u8],
    cfg: &F::Config,
) -> (GkrLayerProof<F>, &'a [u8])
where
    F::Inner: ConstTranscribable,
    F::Modulus: ConstTranscribable,
{
    let flag = bytes[0];
    let mut bytes = &bytes[1..];
    let sumcheck_proof = match flag {
        0 => None,
        1 => {
            let (sc, rest) = SumcheckProof::<F>::read_transcription_bytes_subset(bytes);
            bytes = rest;
            Some(sc)
        }
        v => panic!("invalid sumcheck-presence flag: {v}"),
    };
    let (xs, rest) = read_f_inner_slice::<F>(bytes, 4, cfg);
    let mut it = xs.into_iter();
    (
        GkrLayerProof {
            sumcheck_proof,
            p_left: it.next().expect("p_left"),
            p_right: it.next().expect("p_right"),
            q_left: it.next().expect("q_left"),
            q_right: it.next().expect("q_right"),
        },
        rest,
    )
}

// ---------- BatchedGkrLayerProof ----------

#[allow(clippy::arithmetic_side_effects)]
fn batched_layer_num_bytes<F: PrimeField>(p: &BatchedGkrLayerProof<F>) -> usize
where
    F::Inner: ConstTranscribable,
    F::Modulus: ConstTranscribable,
{
    let sc = match &p.sumcheck_proof {
        Some(sc) => add!(SumcheckProof::<F>::LENGTH_NUM_BYTES, sc.get_num_bytes()),
        None => 0,
    };
    let l = p.p_lefts.len();
    add!(
        1usize,
        add!(sc, add!(u32::NUM_BYTES, mul!(mul!(4, l), F::Inner::NUM_BYTES)))
    )
}

#[allow(clippy::arithmetic_side_effects)]
fn write_batched_layer<'a, F: PrimeField>(
    buf: &'a mut [u8],
    p: &BatchedGkrLayerProof<F>,
) -> &'a mut [u8]
where
    F::Inner: ConstTranscribable,
    F::Modulus: ConstTranscribable,
{
    let mut buf = buf;
    if let Some(sc) = &p.sumcheck_proof {
        buf[0] = 1;
        buf = &mut buf[1..];
        buf = sc.write_transcription_bytes_subset(buf);
    } else {
        buf[0] = 0;
        buf = &mut buf[1..];
    }
    let l = p.p_lefts.len();
    buf = write_u32_prefix(buf, l);
    buf = write_f_inner_slice(buf, &p.p_lefts);
    buf = write_f_inner_slice(buf, &p.p_rights);
    buf = write_f_inner_slice(buf, &p.q_lefts);
    write_f_inner_slice(buf, &p.q_rights)
}

#[allow(clippy::arithmetic_side_effects)]
fn read_batched_layer<'a, F: PrimeField>(
    bytes: &'a [u8],
    cfg: &F::Config,
) -> (BatchedGkrLayerProof<F>, &'a [u8])
where
    F::Inner: ConstTranscribable,
    F::Modulus: ConstTranscribable,
{
    let flag = bytes[0];
    let mut bytes = &bytes[1..];
    let sumcheck_proof = match flag {
        0 => None,
        1 => {
            let (sc, rest) = SumcheckProof::<F>::read_transcription_bytes_subset(bytes);
            bytes = rest;
            Some(sc)
        }
        v => panic!("invalid batched sumcheck-presence flag: {v}"),
    };
    let (l, rest) = read_u32_prefix(bytes);
    bytes = rest;
    let (p_lefts, rest) = read_f_inner_slice::<F>(bytes, l, cfg);
    bytes = rest;
    let (p_rights, rest) = read_f_inner_slice::<F>(bytes, l, cfg);
    bytes = rest;
    let (q_lefts, rest) = read_f_inner_slice::<F>(bytes, l, cfg);
    bytes = rest;
    let (q_rights, rest) = read_f_inner_slice::<F>(bytes, l, cfg);
    (
        BatchedGkrLayerProof {
            sumcheck_proof,
            p_lefts,
            p_rights,
            q_lefts,
            q_rights,
        },
        rest,
    )
}

// ---------- GkrFractionProof ----------

#[allow(clippy::arithmetic_side_effects)]
fn fraction_num_bytes<F: PrimeField>(p: &GkrFractionProof<F>) -> usize
where
    F::Inner: ConstTranscribable,
    F::Modulus: ConstTranscribable,
{
    let layers: usize = p.layer_proofs.iter().map(gkr_layer_num_bytes).sum();
    add!(
        mul!(2, F::Inner::NUM_BYTES),
        add!(u32::NUM_BYTES, layers)
    )
}

#[allow(clippy::arithmetic_side_effects)]
fn write_fraction<'a, F: PrimeField>(
    buf: &'a mut [u8],
    p: &GkrFractionProof<F>,
) -> &'a mut [u8]
where
    F::Inner: ConstTranscribable,
    F::Modulus: ConstTranscribable,
{
    let mut buf = write_f_inner_slice(buf, &[p.root_p.clone(), p.root_q.clone()]);
    buf = write_u32_prefix(buf, p.layer_proofs.len());
    for layer in &p.layer_proofs {
        buf = write_gkr_layer(buf, layer);
    }
    buf
}

#[allow(clippy::arithmetic_side_effects)]
fn read_fraction<'a, F: PrimeField>(
    bytes: &'a [u8],
    cfg: &F::Config,
) -> (GkrFractionProof<F>, &'a [u8])
where
    F::Inner: ConstTranscribable,
    F::Modulus: ConstTranscribable,
{
    let (roots, rest) = read_f_inner_slice::<F>(bytes, 2, cfg);
    let mut it = roots.into_iter();
    let root_p = it.next().expect("root_p");
    let root_q = it.next().expect("root_q");
    let (n_layers, mut bytes) = read_u32_prefix(rest);
    let mut layer_proofs = Vec::with_capacity(n_layers);
    for _ in 0..n_layers {
        let (layer, rest) = read_gkr_layer::<F>(bytes, cfg);
        bytes = rest;
        layer_proofs.push(layer);
    }
    (
        GkrFractionProof {
            root_p,
            root_q,
            layer_proofs,
        },
        bytes,
    )
}

// ---------- BatchedGkrFractionProof ----------

#[allow(clippy::arithmetic_side_effects)]
fn batched_fraction_num_bytes<F: PrimeField>(p: &BatchedGkrFractionProof<F>) -> usize
where
    F::Inner: ConstTranscribable,
    F::Modulus: ConstTranscribable,
{
    let l = p.roots_p.len();
    let layers: usize = p.layer_proofs.iter().map(batched_layer_num_bytes).sum();
    add!(
        u32::NUM_BYTES,
        add!(
            mul!(mul!(2, l), F::Inner::NUM_BYTES),
            add!(u32::NUM_BYTES, layers)
        )
    )
}

#[allow(clippy::arithmetic_side_effects)]
fn write_batched_fraction<'a, F: PrimeField>(
    buf: &'a mut [u8],
    p: &BatchedGkrFractionProof<F>,
) -> &'a mut [u8]
where
    F::Inner: ConstTranscribable,
    F::Modulus: ConstTranscribable,
{
    let l = p.roots_p.len();
    let mut buf = write_u32_prefix(buf, l);
    buf = write_f_inner_slice(buf, &p.roots_p);
    buf = write_f_inner_slice(buf, &p.roots_q);
    buf = write_u32_prefix(buf, p.layer_proofs.len());
    for layer in &p.layer_proofs {
        buf = write_batched_layer(buf, layer);
    }
    buf
}

#[allow(clippy::arithmetic_side_effects)]
fn read_batched_fraction<'a, F: PrimeField>(
    bytes: &'a [u8],
    cfg: &F::Config,
) -> (BatchedGkrFractionProof<F>, &'a [u8])
where
    F::Inner: ConstTranscribable,
    F::Modulus: ConstTranscribable,
{
    let (l, bytes) = read_u32_prefix(bytes);
    let (roots_p, bytes) = read_f_inner_slice::<F>(bytes, l, cfg);
    let (roots_q, bytes) = read_f_inner_slice::<F>(bytes, l, cfg);
    let (n_layers, mut bytes) = read_u32_prefix(bytes);
    let mut layer_proofs = Vec::with_capacity(n_layers);
    for _ in 0..n_layers {
        let (layer, rest) = read_batched_layer::<F>(bytes, cfg);
        bytes = rest;
        layer_proofs.push(layer);
    }
    (
        BatchedGkrFractionProof {
            roots_p,
            roots_q,
            layer_proofs,
        },
        bytes,
    )
}

// ---------- GkrLogupGroupMeta ----------

#[allow(clippy::arithmetic_side_effects)]
fn meta_num_bytes(meta: &GkrLogupGroupMeta) -> usize {
    add!(
        LookupTableType::NUM_BYTES,
        add!(
            mul!(4, u32::NUM_BYTES), // num_lookups, num_chunks, chunk_width, witness_len
            add!(u32::NUM_BYTES, mul!(meta.parent_columns.len(), u32::NUM_BYTES))
        )
    )
}

#[allow(clippy::arithmetic_side_effects)]
fn write_meta<'a>(buf: &'a mut [u8], meta: &GkrLogupGroupMeta) -> &'a mut [u8] {
    meta.table_type
        .write_transcription_bytes_exact(&mut buf[..LookupTableType::NUM_BYTES]);
    let mut buf = &mut buf[LookupTableType::NUM_BYTES..];
    buf = write_u32_prefix(buf, meta.num_lookups);
    buf = write_u32_prefix(buf, meta.num_chunks);
    buf = write_u32_prefix(buf, meta.chunk_width);
    buf = write_u32_prefix(buf, meta.witness_len);
    buf = write_u32_prefix(buf, meta.parent_columns.len());
    for &c in &meta.parent_columns {
        buf = write_u32_prefix(buf, c);
    }
    buf
}

#[allow(clippy::arithmetic_side_effects)]
fn read_meta(bytes: &[u8]) -> (GkrLogupGroupMeta, &[u8]) {
    let table_type =
        LookupTableType::read_transcription_bytes_exact(&bytes[..LookupTableType::NUM_BYTES]);
    let bytes = &bytes[LookupTableType::NUM_BYTES..];
    let (num_lookups, bytes) = read_u32_prefix(bytes);
    let (num_chunks, bytes) = read_u32_prefix(bytes);
    let (chunk_width, bytes) = read_u32_prefix(bytes);
    let (witness_len, bytes) = read_u32_prefix(bytes);
    let (n_parents, mut bytes) = read_u32_prefix(bytes);
    let mut parent_columns = Vec::with_capacity(n_parents);
    for _ in 0..n_parents {
        let (c, rest) = read_u32_prefix(bytes);
        bytes = rest;
        parent_columns.push(c);
    }
    (
        GkrLogupGroupMeta {
            table_type,
            num_lookups,
            num_chunks,
            chunk_width,
            witness_len,
            parent_columns,
        },
        bytes,
    )
}

// ---------- GkrLogupGroupProof ----------

#[allow(clippy::arithmetic_side_effects)]
fn group_num_bytes<F: PrimeField>(g: &GkrLogupGroupProof<F>) -> usize
where
    F::Inner: ConstTranscribable,
    F::Modulus: ConstTranscribable,
{
    // chunk_lifts: u32 L, then per-lookup: u32 K, then DynamicPolyVecF subset.
    let mut chunk_lifts_bytes = u32::NUM_BYTES;
    for per_lookup in &g.chunk_lifts {
        let v = DynamicPolyVecF::reinterpret(per_lookup);
        chunk_lifts_bytes = add!(
            chunk_lifts_bytes,
            add!(
                u32::NUM_BYTES,
                add!(DynamicPolyVecF::<F>::LENGTH_NUM_BYTES, v.get_num_bytes())
            )
        );
    }
    // aggregated_multiplicities: u32 L, then per-lookup u32 N, then N × F::Inner.
    let mut mults_bytes = u32::NUM_BYTES;
    for per_lookup in &g.aggregated_multiplicities {
        mults_bytes = add!(
            mults_bytes,
            add!(u32::NUM_BYTES, mul!(per_lookup.len(), F::Inner::NUM_BYTES))
        );
    }
    add!(
        chunk_lifts_bytes,
        add!(
            mults_bytes,
            add!(batched_fraction_num_bytes(&g.witness_gkr), fraction_num_bytes(&g.table_gkr))
        )
    )
}

#[allow(clippy::arithmetic_side_effects)]
fn write_group<'a, F: PrimeField>(
    buf: &'a mut [u8],
    g: &GkrLogupGroupProof<F>,
) -> &'a mut [u8]
where
    F::Inner: ConstTranscribable,
    F::Modulus: ConstTranscribable,
{
    let mut buf = write_u32_prefix(buf, g.chunk_lifts.len());
    for per_lookup in &g.chunk_lifts {
        buf = write_u32_prefix(buf, per_lookup.len());
        let v = DynamicPolyVecF::reinterpret(per_lookup);
        buf = v.write_transcription_bytes_subset(buf);
    }
    buf = write_u32_prefix(buf, g.aggregated_multiplicities.len());
    for mults in &g.aggregated_multiplicities {
        buf = write_u32_prefix(buf, mults.len());
        buf = write_f_inner_slice(buf, mults);
    }
    buf = write_batched_fraction(buf, &g.witness_gkr);
    write_fraction(buf, &g.table_gkr)
}

#[allow(clippy::arithmetic_side_effects)]
fn read_group<'a, F: PrimeField>(
    bytes: &'a [u8],
    cfg: &F::Config,
) -> (GkrLogupGroupProof<F>, &'a [u8])
where
    F::Inner: ConstTranscribable,
    F::Modulus: ConstTranscribable,
{
    let (l1, mut bytes) = read_u32_prefix(bytes);
    let mut chunk_lifts = Vec::with_capacity(l1);
    for _ in 0..l1 {
        let (k, rest) = read_u32_prefix(bytes);
        bytes = rest;
        let (v, rest) = DynamicPolyVecF::<F>::read_transcription_bytes_subset(bytes);
        bytes = rest;
        let polys = v.0;
        assert_eq!(
            polys.len(),
            k,
            "chunk_lifts inner length mismatch: expected {k}, got {}",
            polys.len()
        );
        chunk_lifts.push(polys);
    }
    let (l2, mut bytes) = read_u32_prefix(bytes);
    let mut aggregated_multiplicities = Vec::with_capacity(l2);
    for _ in 0..l2 {
        let (n, rest) = read_u32_prefix(bytes);
        bytes = rest;
        let (mults, rest) = read_f_inner_slice::<F>(bytes, n, cfg);
        bytes = rest;
        aggregated_multiplicities.push(mults);
    }
    let (witness_gkr, bytes) = read_batched_fraction::<F>(bytes, cfg);
    let (table_gkr, bytes) = read_fraction::<F>(bytes, cfg);
    (
        GkrLogupGroupProof {
            chunk_lifts,
            aggregated_multiplicities,
            witness_gkr,
            table_gkr,
        },
        bytes,
    )
}

// ---------- GkrLogupLookupProof ----------
//
// Encoding:
//   [num_groups: u32]
//   if num_groups > 0:
//     [modulus]
//     for g in 0..num_groups:
//         GkrLogupGroupMeta bytes
//         GkrLogupGroupProof bytes (uses the modulus above for inner F)

impl<F> GenTranscribable for GkrLogupLookupProof<F>
where
    F: PrimeField,
    F::Inner: ConstTranscribable,
    F::Modulus: ConstTranscribable,
{
    fn read_transcription_bytes_exact(bytes: &[u8]) -> Self {
        if bytes.is_empty() {
            return Self {
                groups: Vec::new(),
                group_meta: Vec::new(),
            };
        }
        let (n, bytes) = read_u32_prefix(bytes);
        if n == 0 {
            assert!(bytes.is_empty(), "trailing bytes after empty lookup proof");
            return Self {
                groups: Vec::new(),
                group_meta: Vec::new(),
            };
        }
        let mod_size = F::Modulus::NUM_BYTES;
        let cfg = zinc_transcript::read_field_cfg::<F>(&bytes[..mod_size]);
        let mut bytes = &bytes[mod_size..];
        let mut group_meta = Vec::with_capacity(n);
        let mut groups = Vec::with_capacity(n);
        for _ in 0..n {
            let (m, rest) = read_meta(bytes);
            bytes = rest;
            group_meta.push(m);
            let (g, rest) = read_group::<F>(bytes, &cfg);
            bytes = rest;
            groups.push(g);
        }
        assert!(
            bytes.is_empty(),
            "trailing bytes after GkrLogupLookupProof"
        );
        Self { groups, group_meta }
    }

    fn write_transcription_bytes_exact(&self, mut buf: &mut [u8]) {
        let n = self.groups.len();
        assert_eq!(n, self.group_meta.len(), "groups/group_meta length mismatch");
        if buf.is_empty() {
            assert_eq!(n, 0, "empty buffer requires no groups");
            return;
        }
        buf = write_u32_prefix(buf, n);
        if n == 0 {
            assert!(buf.is_empty(), "empty proof should not need extra bytes");
            return;
        }
        // Modulus comes from the first group's first multiplicity element,
        // chunk_lift coefficient, or GKR root — at least one F field is
        // present for any non-empty group. Probe in order.
        let modulus = self.groups[0]
            .aggregated_multiplicities
            .iter()
            .find_map(|v| v.first().map(|f| f.modulus()))
            .or_else(|| Some(self.groups[0].witness_gkr.roots_p[0].modulus()))
            .expect("non-empty group must contain at least one F element");
        buf = zinc_transcript::append_field_cfg::<F>(buf, &modulus);
        for (meta, group) in self.group_meta.iter().zip(self.groups.iter()) {
            buf = write_meta(buf, meta);
            buf = write_group(buf, group);
        }
        assert!(
            buf.is_empty(),
            "GkrLogupLookupProof leftover buffer (encoding underflow)"
        );
    }
}

impl<F> Transcribable for GkrLogupLookupProof<F>
where
    F: PrimeField,
    F::Inner: ConstTranscribable,
    F::Modulus: ConstTranscribable,
{
    #[allow(clippy::arithmetic_side_effects)]
    fn get_num_bytes(&self) -> usize {
        if self.groups.is_empty() {
            return u32::NUM_BYTES;
        }
        let mut total = add!(u32::NUM_BYTES, F::Modulus::NUM_BYTES);
        for (meta, group) in self.group_meta.iter().zip(self.groups.iter()) {
            total = add!(total, meta_num_bytes(meta));
            total = add!(total, group_num_bytes(group));
        }
        total
    }
}
