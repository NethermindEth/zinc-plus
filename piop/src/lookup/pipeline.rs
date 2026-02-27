//! Pipeline integration for the batched decomposed LogUp protocol.
//!
//! This module provides the glue between the UAIR type system and the
//! standalone lookup protocols.  Given a UAIR signature and a projected
//! trace over F_q, it automatically determines which columns need
//! lookup constraints, generates the appropriate tables and chunk
//! decompositions, and runs the [`BatchedDecompLogupProtocol`].
//!
//! ## Supported column types
//!
//! | UAIR column kind       | Lookup table              | Decomposition            |
//! |------------------------|---------------------------|--------------------------|
//! | `binary_poly` (width w)| `{0,1}^{<w}[X]` → 2^w    | K chunks of BitPoly(w/K) |
//! | `int` (width w)        | `[0, 2^w − 1]`  → 2^w    | K chunks of Word(w/K)    |
//! | `arbitrary_poly`       | *(none — unconstrained)*  | —                        |
//!
//! For tables of size 2^{32} the standard configuration decomposes into
//! K = 2 sub-tables of size 2^{16}.
//!
//! ## Usage
//!
//! ```ignore
//! use zinc_piop::lookup::pipeline::{LookupConfig, lookup_prove, lookup_verify};
//!
//! // Build configuration from the UAIR signature.
//! let config = LookupConfig::from_signature::<32>(
//!     &signature,
//!     &projecting_element,
//!     field_cfg,
//! );
//!
//! // Prover: build instance from projected trace columns and run protocol.
//! let (proof, state) = lookup_prove(
//!     &mut transcript,
//!     &config,
//!     &projected_trace_columns,  // Vec<F> per column
//!     field_cfg,
//! )?;
//!
//! // Verifier: replay transcript and verify proof.
//! let subclaim = lookup_verify(
//!     &mut transcript,
//!     &config,
//!     &proof,
//!     witness_len,
//!     field_cfg,
//! )?;
//! ```

use crypto_primitives::{FromPrimitiveWithConfig, PrimeField};
use zinc_transcript::traits::{ConstTranscribable, Transcript};
use zinc_uair::UairSignature;
use zinc_utils::inner_transparent_field::InnerTransparentField;

use super::{
    batched_decomposition::BatchedDecompLogupProtocol,
    structs::{
        BatchedDecompLogupProof, BatchedDecompLogupProverState,
        BatchedDecompLogupVerifierSubClaim, BatchedDecompLookupInstance, LookupError,
    },
    tables::{bitpoly_shift, generate_bitpoly_table, generate_word_table, word_shift},
};

// ─── Configuration ──────────────────────────────────────────────────────────

/// Describes the column-type of a single lookup group.
#[derive(Clone, Debug)]
pub enum LookupTableKind {
    /// Column stores a binary polynomial of degree < `full_width`.
    ///
    /// The projected table is generated at the given projecting element `a`
    /// and has `2^full_width` entries.  Decomposed into `num_chunks`
    /// sub-tables of `BitPoly(chunk_width)`.
    BitPoly {
        /// Total width, e.g. 32.
        full_width: usize,
        /// Width of each sub-table chunk, e.g. 16.
        chunk_width: usize,
        /// Number of chunks (= full_width / chunk_width).
        num_chunks: usize,
    },
    /// Column stores an integer in `[0, 2^full_width − 1]`.
    ///
    /// Decomposed into `num_chunks` sub-tables of `Word(chunk_width)`.
    Word {
        /// Total width, e.g. 32.
        full_width: usize,
        /// Width of each sub-table chunk, e.g. 16.
        chunk_width: usize,
        /// Number of chunks (= full_width / chunk_width).
        num_chunks: usize,
    },
}

/// Full lookup configuration: table kind, generated sub-table, and shift
/// factors for the decomposition.
///
/// A single `LookupConfig` covers **all** columns of the same type
/// (all `binary_poly` columns share one config, all `int` columns
/// share another).
#[derive(Clone, Debug)]
pub struct LookupConfig<F: PrimeField> {
    /// Which column indices (0-based in the full trace) belong to this group.
    pub column_indices: Vec<usize>,
    /// The table kind (BitPoly or Word).
    pub kind: LookupTableKind,
    /// The projected sub-table entries (size `2^chunk_width`).
    pub subtable: Vec<F>,
    /// Shift factors for the decomposition (`num_chunks` entries).
    ///
    /// For BitPoly: `shifts[k] = a^{k * chunk_width}`.
    /// For Word:    `shifts[k] = (2^{chunk_width})^k mod q`.
    pub shifts: Vec<F>,
}

/// Description of all lookup constraints derived from a UAIR signature.
#[derive(Clone, Debug)]
pub struct PipelineLookupConfig<F: PrimeField> {
    /// Per-group configurations.  Typically at most two groups
    /// (one BitPoly, one Word), but either may be absent.
    pub groups: Vec<LookupConfig<F>>,
}

impl<F: PrimeField + FromPrimitiveWithConfig> PipelineLookupConfig<F>
where
    F::Config: Sync,
{
    /// Derive the lookup configuration from a [`UairSignature`].
    ///
    /// # Arguments
    ///
    /// * `signature` — the UAIR column-type counts.
    /// * `poly_width` — the BinaryPoly width (the const-generic `D`,
    ///   e.g. 32 for `BinaryPoly<32>`).
    /// * `int_width` — the integer bit-width (e.g. 32 for `Word(32)`).
    /// * `num_chunks` — decomposition factor (e.g. 2 → two sub-tables
    ///   of size `2^{width/2}`).
    /// * `projecting_element` — the random `a ∈ F_q` used by the
    ///   BitPoly projection ψ_a.
    /// * `field_cfg` — field configuration.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn from_signature(
        signature: &UairSignature,
        poly_width: usize,
        int_width: usize,
        num_chunks: usize,
        projecting_element: &F,
        field_cfg: &F::Config,
    ) -> Self {
        let mut groups = Vec::new();

        // ── BitPoly columns ─────────────────────────────────────────
        if signature.binary_poly_cols > 0 {
            let chunk_width = poly_width / num_chunks;
            debug_assert_eq!(
                chunk_width * num_chunks,
                poly_width,
                "poly_width must be divisible by num_chunks"
            );

            let subtable =
                generate_bitpoly_table(chunk_width, projecting_element, field_cfg);
            let shifts: Vec<F> = (0..num_chunks)
                .map(|k| bitpoly_shift(k * chunk_width, projecting_element))
                .collect();

            let column_indices: Vec<usize> =
                (0..signature.binary_poly_cols).collect();

            groups.push(LookupConfig {
                column_indices,
                kind: LookupTableKind::BitPoly {
                    full_width: poly_width,
                    chunk_width,
                    num_chunks,
                },
                subtable,
                shifts,
            });
        }

        // ── Int columns ─────────────────────────────────────────────
        if signature.int_cols > 0 {
            let chunk_width = int_width / num_chunks;
            debug_assert_eq!(
                chunk_width * num_chunks,
                int_width,
                "int_width must be divisible by num_chunks"
            );

            let subtable = generate_word_table(chunk_width, field_cfg);
            let shifts: Vec<F> = (0..num_chunks)
                .map(|k| word_shift(k * chunk_width, field_cfg))
                .collect();

            // Int columns start after binary_poly + arbitrary_poly.
            let int_start =
                signature.binary_poly_cols + signature.arbitrary_poly_cols;
            let column_indices: Vec<usize> =
                (int_start..int_start + signature.int_cols).collect();

            groups.push(LookupConfig {
                column_indices,
                kind: LookupTableKind::Word {
                    full_width: int_width,
                    chunk_width,
                    num_chunks,
                },
                subtable,
                shifts,
            });
        }

        Self { groups }
    }

    /// Returns `true` if there are no lookup constraints (no binary_poly
    /// or int columns).
    pub fn is_empty(&self) -> bool {
        self.groups.is_empty()
    }

    /// Total number of columns that need lookup.
    pub fn num_lookup_columns(&self) -> usize {
        self.groups.iter().map(|g| g.column_indices.len()).sum()
    }
}

// ─── Instance builder ───────────────────────────────────────────────────────

/// Build a [`BatchedDecompLookupInstance`] for one group of columns.
///
/// # Arguments
///
/// * `config` — the group's lookup configuration.
/// * `projected_columns` — the projected trace columns over F_q.
///   `projected_columns[col_idx]` is the flattened evaluations of column
///   `col_idx`.
/// * `original_entries_per_column` — for each column, the **original**
///   integer indices into the full table (before projection).  These are
///   needed to compute the chunk decomposition.
///
///   For `BitPoly(w)` columns: the index `n` means the binary polynomial
///   whose coefficient bit vector is `n`.
///   For `Word(w)` columns: the index is simply the integer value.
///
/// * `field_cfg` — field configuration.
#[allow(clippy::arithmetic_side_effects)]
pub fn build_batched_instance<F: PrimeField + FromPrimitiveWithConfig + Send + Sync>(
    config: &LookupConfig<F>,
    projected_columns: &[Vec<F>],
    original_indices_per_column: &[Vec<usize>],
    _field_cfg: &F::Config,
) -> BatchedDecompLookupInstance<F>
where
    F::Config: Sync,
{
    let num_chunks = config.shifts.len();
    let chunk_width = match &config.kind {
        LookupTableKind::BitPoly { chunk_width, .. } => *chunk_width,
        LookupTableKind::Word { chunk_width, .. } => *chunk_width,
    };
    let chunk_mask = (1usize << chunk_width) - 1;

    let num_lookups = config.column_indices.len();
    let mut witnesses = Vec::with_capacity(num_lookups);
    let mut all_chunks = Vec::with_capacity(num_lookups);

    for (col_pos, &col_idx) in config.column_indices.iter().enumerate() {
        let witness = projected_columns[col_idx].clone();
        let original_indices = &original_indices_per_column[col_pos];
        let witness_len = witness.len();

        // Decompose each entry into K chunks by extracting bit-fields
        // from the original index.
        let chunks: Vec<Vec<F>> = (0..num_chunks)
            .map(|k| {
                original_indices
                    .iter()
                    .map(|&idx| {
                        let chunk_idx = (idx >> (k * chunk_width)) & chunk_mask;
                        config.subtable[chunk_idx].clone()
                    })
                    .collect()
            })
            .collect();

        debug_assert_eq!(chunks.len(), num_chunks);
        debug_assert!(chunks.iter().all(|c| c.len() == witness_len));

        witnesses.push(witness);
        all_chunks.push(chunks);
    }

    BatchedDecompLookupInstance {
        witnesses,
        subtable: config.subtable.clone(),
        shifts: config.shifts.clone(),
        chunks: all_chunks,
    }
}

/// Build a [`BatchedDecompLookupInstance`] from projected trace columns
/// when the chunk decomposition is **already known**.
///
/// This is the simpler variant when the caller has already computed the
/// chunk vectors (e.g. because the witness generation step naturally
/// produces them).
pub fn build_batched_instance_with_chunks<F: PrimeField>(
    config: &LookupConfig<F>,
    projected_columns: &[Vec<F>],
    precomputed_chunks: Vec<Vec<Vec<F>>>,
) -> BatchedDecompLookupInstance<F> {
    let witnesses: Vec<Vec<F>> = config
        .column_indices
        .iter()
        .map(|&col_idx| projected_columns[col_idx].clone())
        .collect();

    BatchedDecompLookupInstance {
        witnesses,
        subtable: config.subtable.clone(),
        shifts: config.shifts.clone(),
        chunks: precomputed_chunks,
    }
}

// ─── Prover ─────────────────────────────────────────────────────────────────

/// Proof and prover-state for a single lookup group.
#[derive(Clone, Debug)]
pub struct LookupGroupProof<F: PrimeField> {
    /// The batched decomposition + LogUp proof.
    pub proof: BatchedDecompLogupProof<F>,
    /// Prover state (mainly the evaluation point from the sumcheck).
    pub state: BatchedDecompLogupProverState<F>,
}

/// Aggregated proof across all lookup groups.
#[derive(Clone, Debug)]
pub struct PipelineLookupProof<F: PrimeField> {
    /// One proof per group (in the same order as `PipelineLookupConfig::groups`).
    pub group_proofs: Vec<LookupGroupProof<F>>,
}

/// Aggregated prover state.
#[derive(Clone, Debug)]
pub struct PipelineLookupProverState<F: PrimeField> {
    /// Evaluation points from each group's sumcheck.
    pub evaluation_points: Vec<Vec<F>>,
}

/// Run the batched decomposed LogUp prover for all lookup groups.
///
/// For each group of columns (same table type), this function:
/// 1. Builds a `BatchedDecompLookupInstance`.
/// 2. Runs `BatchedDecompLogupProtocol::prove_as_subprotocol`.
///
/// All groups share the same Fiat-Shamir transcript, ensuring
/// soundness across the combined protocol.
///
/// # Arguments
///
/// * `transcript` — the shared Fiat-Shamir transcript.
/// * `config` — the pipeline lookup configuration.
/// * `projected_columns` — projected trace columns (F_q vectors).
/// * `original_indices_per_group` — for each group, for each column
///   in that group, the original table indices.
/// * `field_cfg` — field configuration.
pub fn lookup_prove<F>(
    transcript: &mut impl Transcript,
    config: &PipelineLookupConfig<F>,
    projected_columns: &[Vec<F>],
    original_indices_per_group: &[Vec<Vec<usize>>],
    field_cfg: &F::Config,
) -> Result<(PipelineLookupProof<F>, PipelineLookupProverState<F>), LookupError<F>>
where
    F: InnerTransparentField + FromPrimitiveWithConfig + Send + Sync,
    F::Inner: ConstTranscribable + num_traits::Zero + Default + Send + Sync,
{
    let mut group_proofs = Vec::with_capacity(config.groups.len());
    let mut evaluation_points = Vec::with_capacity(config.groups.len());

    for (group_idx, group_config) in config.groups.iter().enumerate() {
        let instance = build_batched_instance(
            group_config,
            projected_columns,
            &original_indices_per_group[group_idx],
            field_cfg,
        );

        let (proof, state) =
            BatchedDecompLogupProtocol::<F>::prove_as_subprotocol(
                transcript, &instance, field_cfg,
            )?;

        evaluation_points.push(state.evaluation_point.clone());
        group_proofs.push(LookupGroupProof { proof, state });
    }

    Ok((
        PipelineLookupProof { group_proofs },
        PipelineLookupProverState { evaluation_points },
    ))
}

/// Run the batched decomposed LogUp prover when chunk decompositions
/// are already available.
pub fn lookup_prove_with_chunks<F>(
    transcript: &mut impl Transcript,
    config: &PipelineLookupConfig<F>,
    projected_columns: &[Vec<F>],
    precomputed_chunks_per_group: Vec<Vec<Vec<Vec<F>>>>,
    field_cfg: &F::Config,
) -> Result<(PipelineLookupProof<F>, PipelineLookupProverState<F>), LookupError<F>>
where
    F: InnerTransparentField + FromPrimitiveWithConfig + Send + Sync,
    F::Inner: ConstTranscribable + num_traits::Zero + Default + Send + Sync,
{
    let mut group_proofs = Vec::with_capacity(config.groups.len());
    let mut evaluation_points = Vec::with_capacity(config.groups.len());

    for (group_idx, group_config) in config.groups.iter().enumerate() {
        let instance = build_batched_instance_with_chunks(
            group_config,
            projected_columns,
            precomputed_chunks_per_group[group_idx].clone(),
        );

        let (proof, state) =
            BatchedDecompLogupProtocol::<F>::prove_as_subprotocol(
                transcript, &instance, field_cfg,
            )?;

        evaluation_points.push(state.evaluation_point.clone());
        group_proofs.push(LookupGroupProof { proof, state });
    }

    Ok((
        PipelineLookupProof { group_proofs },
        PipelineLookupProverState { evaluation_points },
    ))
}

// ─── Verifier ───────────────────────────────────────────────────────────────

/// Aggregated verifier sub-claims.
#[derive(Clone, Debug)]
pub struct PipelineLookupVerifierSubClaim<F: PrimeField> {
    /// Per-group sub-claims.
    pub sub_claims: Vec<BatchedDecompLogupVerifierSubClaim<F>>,
}

/// Run the batched decomposed LogUp verifier for all lookup groups.
///
/// # Arguments
///
/// * `transcript` — the shared Fiat-Shamir transcript (must be in the
///   same state as after the prover's IC + CPR steps).
/// * `config` — the pipeline lookup configuration.
/// * `proof` — the aggregated lookup proof.
/// * `witness_len` — the number of entries per trace column (power of 2).
/// * `field_cfg` — field configuration.
pub fn lookup_verify<F>(
    transcript: &mut impl Transcript,
    config: &PipelineLookupConfig<F>,
    proof: &PipelineLookupProof<F>,
    witness_len: usize,
    field_cfg: &F::Config,
) -> Result<PipelineLookupVerifierSubClaim<F>, LookupError<F>>
where
    F: InnerTransparentField + FromPrimitiveWithConfig + Send + Sync,
    F::Inner: ConstTranscribable + num_traits::Zero + Default + Send + Sync,
{
    let mut sub_claims = Vec::with_capacity(config.groups.len());

    for (group_idx, group_config) in config.groups.iter().enumerate() {
        let group_proof = &proof.group_proofs[group_idx];
        let num_lookups = group_config.column_indices.len();

        let sub_claim =
            BatchedDecompLogupProtocol::<F>::verify_as_subprotocol(
                transcript,
                &group_proof.proof,
                &group_config.subtable,
                &group_config.shifts,
                num_lookups,
                witness_len,
                field_cfg,
            )?;

        sub_claims.push(sub_claim);
    }

    Ok(PipelineLookupVerifierSubClaim { sub_claims })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crypto_bigint::{U128, const_monty_params};
    use crypto_primitives::crypto_bigint_const_monty::ConstMontyField;
    use zinc_transcript::KeccakTranscript;

    const N: usize = 2;
    const_monty_params!(TestParams, U128, "00000000b933426489189cb5b47d567f");
    type F = ConstMontyField<TestParams, N>;

    /// Helper: project a table index `n` into F_q using the BitPoly
    /// projection at element `a`.
    fn project_bitpoly_index(n: usize, width: usize, a: &F) -> F {
        let mut result = F::from(0u32);
        let mut power = F::from(1u32);
        for bit in 0..width {
            if (n >> bit) & 1 == 1 {
                result += &power;
            }
            power *= a;
        }
        result
    }

    #[test]
    fn pipeline_bitpoly_lookup_roundtrip() {
        let a = F::from(3u32);
        let poly_width = 4;
        let num_chunks = 2;
        // Build configuration.
        let sig = UairSignature {
            binary_poly_cols: 2,
            arbitrary_poly_cols: 0,
            int_cols: 0,
        };
        let config =
            PipelineLookupConfig::from_signature(&sig, poly_width, 0, num_chunks, &a, &());

        assert_eq!(config.groups.len(), 1);
        assert_eq!(config.groups[0].column_indices, vec![0, 1]);

        // Build a small trace: 2 BitPoly(4) columns, 4 rows each.
        // Column 0: indices [0, 3, 5, 15]
        // Column 1: indices [1, 2, 7, 10]
        let indices_col0: Vec<usize> = vec![0, 3, 5, 15];
        let indices_col1: Vec<usize> = vec![1, 2, 7, 10];

        let projected_col0: Vec<F> = indices_col0
            .iter()
            .map(|&n| project_bitpoly_index(n, poly_width, &a))
            .collect();
        let projected_col1: Vec<F> = indices_col1
            .iter()
            .map(|&n| project_bitpoly_index(n, poly_width, &a))
            .collect();

        let projected_columns = vec![projected_col0, projected_col1];
        let original_indices = vec![vec![indices_col0, indices_col1]];

        // Prove.
        let mut pt = KeccakTranscript::new();
        let (proof, _state) = lookup_prove(
            &mut pt,
            &config,
            &projected_columns,
            &original_indices,
            &(),
        )
        .expect("prover should succeed");

        // Verify.
        let mut vt = KeccakTranscript::new();
        let _subclaim = lookup_verify(&mut vt, &config, &proof, 4, &())
            .expect("verifier should accept");
    }

    #[test]
    fn pipeline_word_lookup_roundtrip() {
        let a = F::from(3u32); // Not used for word tables.
        let int_width = 4;
        let num_chunks = 2;

        let sig = UairSignature {
            binary_poly_cols: 0,
            arbitrary_poly_cols: 0,
            int_cols: 2,
        };
        let config =
            PipelineLookupConfig::from_signature(&sig, 0, int_width, num_chunks, &a, &());

        assert_eq!(config.groups.len(), 1);
        assert_eq!(config.groups[0].column_indices, vec![0, 1]);

        // Columns with integer values in [0, 15].
        let indices_col0: Vec<usize> = vec![0, 5, 10, 15];
        let indices_col1: Vec<usize> = vec![1, 3, 7, 12];

        // For Word columns, projection is just value mod q.
        let projected_col0: Vec<F> =
            indices_col0.iter().map(|&n| F::from(n as u32)).collect();
        let projected_col1: Vec<F> =
            indices_col1.iter().map(|&n| F::from(n as u32)).collect();

        let projected_columns = vec![projected_col0, projected_col1];
        let original_indices = vec![vec![indices_col0, indices_col1]];

        let mut pt = KeccakTranscript::new();
        let (proof, _state) = lookup_prove(
            &mut pt,
            &config,
            &projected_columns,
            &original_indices,
            &(),
        )
        .expect("prover should succeed");

        let mut vt = KeccakTranscript::new();
        let _subclaim = lookup_verify(&mut vt, &config, &proof, 4, &())
            .expect("verifier should accept");
    }

    #[test]
    fn pipeline_mixed_bitpoly_and_word() {
        let a = F::from(3u32);
        let poly_width = 4;
        let int_width = 4;
        let num_chunks = 2;

        let sig = UairSignature {
            binary_poly_cols: 2,
            arbitrary_poly_cols: 1, // 1 unconstrained column
            int_cols: 1,
        };
        let config = PipelineLookupConfig::from_signature(
            &sig,
            poly_width,
            int_width,
            num_chunks,
            &a,
            &(),
        );

        assert_eq!(config.groups.len(), 2);
        // BitPoly group: columns 0, 1
        assert_eq!(config.groups[0].column_indices, vec![0, 1]);
        // Word group: column 3 (after 2 binary_poly + 1 arbitrary_poly)
        assert_eq!(config.groups[1].column_indices, vec![3]);

        // BitPoly columns.
        let bp_indices_col0: Vec<usize> = vec![0, 5, 10, 15];
        let bp_indices_col1: Vec<usize> = vec![3, 7, 12, 1];

        let bp_proj_col0: Vec<F> = bp_indices_col0
            .iter()
            .map(|&n| project_bitpoly_index(n, poly_width, &a))
            .collect();
        let bp_proj_col1: Vec<F> = bp_indices_col1
            .iter()
            .map(|&n| project_bitpoly_index(n, poly_width, &a))
            .collect();

        // Arbitrary poly column (unconstrained, not in any group).
        let arb_col: Vec<F> = vec![F::from(99u32); 4];

        // Word column.
        let word_indices_col3: Vec<usize> = vec![2, 8, 14, 0];
        let word_proj_col3: Vec<F> =
            word_indices_col3.iter().map(|&n| F::from(n as u32)).collect();

        // Full projected trace: [bp0, bp1, arb, word]
        let projected_columns = vec![bp_proj_col0, bp_proj_col1, arb_col, word_proj_col3];

        // Original indices per group.
        let bitpoly_group_indices = vec![bp_indices_col0, bp_indices_col1];
        let word_group_indices = vec![word_indices_col3];
        let original_indices = vec![bitpoly_group_indices, word_group_indices];

        let mut pt = KeccakTranscript::new();
        let (proof, _state) = lookup_prove(
            &mut pt,
            &config,
            &projected_columns,
            &original_indices,
            &(),
        )
        .expect("prover should succeed");

        let mut vt = KeccakTranscript::new();
        let _subclaim = lookup_verify(&mut vt, &config, &proof, 4, &())
            .expect("verifier should accept");
    }

    #[test]
    fn pipeline_empty_signature_no_lookup() {
        let a = F::from(3u32);
        let sig = UairSignature {
            binary_poly_cols: 0,
            arbitrary_poly_cols: 5,
            int_cols: 0,
        };
        let config =
            PipelineLookupConfig::from_signature(&sig, 32, 32, 2, &a, &());

        assert!(config.is_empty());
        assert_eq!(config.num_lookup_columns(), 0);
    }
}
