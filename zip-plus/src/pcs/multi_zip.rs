//! Shared-Merkle three-instance Zip+ commit/prove/verify.
//!
//! Wraps three independent [`ZipPlus`] instances (with potentially
//! heterogeneous `Cw` types) under a single Merkle tree whose leaves are
//! the column-wise concatenation of all three encoded matrices. Each
//! instance keeps its own polynomial encoding and proximity argument,
//! but the column-opening loop is run once with shared query indices,
//! emitting a single Merkle path per opening instead of three.
//!
//! Constraints:
//! - All three linear codes must produce codewords of the same length, so
//!   that one query index addresses a leaf in the shared tree.
//! - `NUM_COLUMN_OPENINGS` is taken from `Zt0`; all three instances must
//!   agree (asserted at runtime).
//!
//! For the unfolded ShaEcdsa benches at `protocol/benches/e2e.rs:944` this
//! roughly cuts the column-opening Merkle bytes by 2/3 (one path per
//! opening instead of three).

use crate::{
    ZipError,
    code::LinearCode,
    merkle::MerkleTree,
    pcs::{
        VerifyPreOpen, ZipPlusProveByteBreakdown,
        structs::{ZipPlus, ZipPlusCommitment, ZipPlusParams, ZipTypes},
    },
    pcs_transcript::{PcsProverTranscript, PcsVerifierTranscript},
};
use crypto_primitives::{DenseRowMatrix, FromPrimitiveWithConfig, FromWithConfig, PrimeField};
use itertools::Itertools;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::marker::PhantomData;
use zinc_poly::mle::DenseMultilinearExtension;
use zinc_transcript::traits::{ConstTranscribable, GenTranscribable, Transcribable};
use zinc_utils::{
    cfg_into_iter, cfg_iter, cfg_join, from_ref::FromRef, mul_by_scalar::MulByScalar,
};

/// Full prover-side data for a [`MultiZip3`] commitment: per-instance
/// encoded matrices plus the shared Merkle tree built over them.
#[derive(Debug, Clone)]
pub struct MultiZipHint3<R0, R1, R2> {
    pub cw_matrices_0: Vec<DenseRowMatrix<R0>>,
    pub cw_matrices_1: Vec<DenseRowMatrix<R1>>,
    pub cw_matrices_2: Vec<DenseRowMatrix<R2>>,
    pub merkle_tree: MerkleTree,
}

/// Three-instance Zip+ wrapper sharing a single Merkle tree across three
/// independent Zip+ commitments.
pub struct MultiZip3<Zt0, Zt1, Zt2, Lc0, Lc1, Lc2>(
    PhantomData<(Zt0, Zt1, Zt2, Lc0, Lc1, Lc2)>,
)
where
    Zt0: ZipTypes,
    Zt1: ZipTypes,
    Zt2: ZipTypes,
    Lc0: LinearCode<Zt0>,
    Lc1: LinearCode<Zt1>,
    Lc2: LinearCode<Zt2>;

impl<Zt0, Zt1, Zt2, Lc0, Lc1, Lc2> MultiZip3<Zt0, Zt1, Zt2, Lc0, Lc1, Lc2>
where
    Zt0: ZipTypes,
    Zt1: ZipTypes,
    Zt2: ZipTypes,
    Lc0: LinearCode<Zt0>,
    Lc1: LinearCode<Zt1>,
    Lc2: LinearCode<Zt2>,
{
    /// Commit to three independent batches of polynomials under a single
    /// Merkle tree whose leaves are
    /// `H( bytes(cw0_col_j) || bytes(cw1_col_j) || bytes(cw2_col_j) )`.
    ///
    /// Returns the prover hint and three [`ZipPlusCommitment`]s carrying the
    /// shared root (and per-instance batch sizes), for transcript absorption
    /// compatible with the existing verifier wiring.
    #[allow(clippy::too_many_arguments)]
    pub fn commit(
        pp0: &ZipPlusParams<Zt0, Lc0>,
        pp1: &ZipPlusParams<Zt1, Lc1>,
        pp2: &ZipPlusParams<Zt2, Lc2>,
        polys0: &[DenseMultilinearExtension<Zt0::Eval>],
        polys1: &[DenseMultilinearExtension<Zt1::Eval>],
        polys2: &[DenseMultilinearExtension<Zt2::Eval>],
    ) -> Result<
        (
            MultiZipHint3<Zt0::Cw, Zt1::Cw, Zt2::Cw>,
            ZipPlusCommitment,
            ZipPlusCommitment,
            ZipPlusCommitment,
        ),
        ZipError,
    > {
        let nonempty = (!polys0.is_empty()) as u8 + (!polys1.is_empty()) as u8
            + (!polys2.is_empty()) as u8;
        assert!(
            nonempty >= 2,
            "MultiZip3::commit requires at least two non-empty batches \
             (otherwise there is nothing to share); got {nonempty}"
        );
        // Codeword lengths only need to match across NON-EMPTY instances —
        // an empty instance contributes no rows to the shared Merkle tree,
        // so its `pp` codeword length is irrelevant.
        let mut codeword_len: Option<usize> = None;
        let mut check_or_set = |c: usize| match codeword_len {
            None => codeword_len = Some(c),
            Some(prev) => assert_eq!(
                prev, c,
                "MultiZip3: non-empty instances must agree on codeword length"
            ),
        };
        if !polys0.is_empty() {
            check_or_set(pp0.linear_code.codeword_len());
        }
        if !polys1.is_empty() {
            check_or_set(pp1.linear_code.codeword_len());
        }
        if !polys2.is_empty() {
            check_or_set(pp2.linear_code.codeword_len());
        }
        let _codeword_len = codeword_len.expect("at least one non-empty instance");

        // Encode the three matrices in parallel (matching the
        // per-instance commit_optionally flow's `cfg_join!`). Without
        // this, the cw0 → cw1 → cw2 chain is serialized and we lose
        // the cross-instance parallelism the per-instance path had.
        let (cw0, (cw1, cw2)): (
            Vec<DenseRowMatrix<Zt0::Cw>>,
            (Vec<DenseRowMatrix<Zt1::Cw>>, Vec<DenseRowMatrix<Zt2::Cw>>),
        ) = cfg_join!(
            cfg_iter!(polys0)
                .map(|p| ZipPlus::<Zt0, Lc0>::encode_rows(pp0, p))
                .collect(),
            cfg_join!(
                cfg_iter!(polys1)
                    .map(|p| ZipPlus::<Zt1, Lc1>::encode_rows(pp1, p))
                    .collect(),
                cfg_iter!(polys2)
                    .map(|p| ZipPlus::<Zt2, Lc2>::encode_rows(pp2, p))
                    .collect(),
            ),
        );

        let rows0: Vec<&[Zt0::Cw]> = cw0.iter().flat_map(|m| m.as_rows()).collect();
        let rows1: Vec<&[Zt1::Cw]> = cw1.iter().flat_map(|m| m.as_rows()).collect();
        let rows2: Vec<&[Zt2::Cw]> = cw2.iter().flat_map(|m| m.as_rows()).collect();
        let merkle_tree = MerkleTree::new_combined_3(&rows0, &rows1, &rows2);
        let root = merkle_tree.root();

        let comm0 = ZipPlusCommitment {
            root: root.clone(),
            batch_size: polys0.len(),
        };
        let comm1 = ZipPlusCommitment {
            root: root.clone(),
            batch_size: polys1.len(),
        };
        let comm2 = ZipPlusCommitment {
            root,
            batch_size: polys2.len(),
        };

        Ok((
            MultiZipHint3 {
                cw_matrices_0: cw0,
                cw_matrices_1: cw1,
                cw_matrices_2: cw2,
                merkle_tree,
            },
            comm0,
            comm1,
            comm2,
        ))
    }

    /// Open all three commitments at the (shared) point `point`, sharing one
    /// query index and one Merkle path per opening across the three
    /// instances. Returns the three combined evaluations.
    ///
    /// The transcript layout is:
    /// ```text
    /// [pre-open writes for instance 0: b_0, combined_row_0]
    /// [pre-open writes for instance 1: b_1, combined_row_1]
    /// [pre-open writes for instance 2: b_2, combined_row_2]
    /// [for each of NUM_COLUMN_OPENINGS:
    ///   [column index squeezed once]
    ///   [column values from instance 0 (batch_size_0 * num_rows_0 entries)]
    ///   [column values from instance 1 (batch_size_1 * num_rows_1 entries)]
    ///   [column values from instance 2 (batch_size_2 * num_rows_2 entries)]
    ///   [one shared Merkle path]
    /// ]
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn prove_f<F, const CHECK_FOR_OVERFLOW: bool>(
        transcript: &mut PcsProverTranscript,
        pp0: &ZipPlusParams<Zt0, Lc0>,
        pp1: &ZipPlusParams<Zt1, Lc1>,
        pp2: &ZipPlusParams<Zt2, Lc2>,
        polys0: &[DenseMultilinearExtension<Zt0::Eval>],
        polys1: &[DenseMultilinearExtension<Zt1::Eval>],
        polys2: &[DenseMultilinearExtension<Zt2::Eval>],
        point: &[F],
        hint: &MultiZipHint3<Zt0::Cw, Zt1::Cw, Zt2::Cw>,
        field_cfg: &F::Config,
    ) -> Result<(Option<F>, Option<F>, Option<F>), ZipError>
    where
        F: PrimeField
            + for<'a> FromWithConfig<&'a Zt0::CombR>
            + for<'a> FromWithConfig<&'a Zt1::CombR>
            + for<'a> FromWithConfig<&'a Zt2::CombR>
            + for<'a> MulByScalar<&'a F>
            + FromRef<F>,
        F::Inner: Transcribable,
        F::Modulus: Transcribable,
    {
        assert_eq!(
            Zt0::NUM_COLUMN_OPENINGS,
            Zt1::NUM_COLUMN_OPENINGS,
            "MultiZip3 requires equal NUM_COLUMN_OPENINGS"
        );
        assert_eq!(
            Zt0::NUM_COLUMN_OPENINGS,
            Zt2::NUM_COLUMN_OPENINGS,
            "MultiZip3 requires equal NUM_COLUMN_OPENINGS"
        );
        // Codeword length used for the shared column-index squeeze.
        // Only non-empty instances constrain it (see commit() comment).
        let codeword_len = if !polys0.is_empty() {
            pp0.linear_code.codeword_len()
        } else if !polys1.is_empty() {
            pp1.linear_code.codeword_len()
        } else {
            pp2.linear_code.codeword_len()
        };

        let eval0 = if polys0.is_empty() {
            None
        } else {
            Some(ZipPlus::<Zt0, Lc0>::prove_pre_open_f::<F, CHECK_FOR_OVERFLOW>(
                transcript, pp0, polys0, point, field_cfg,
            )?)
        };
        let eval1 = if polys1.is_empty() {
            None
        } else {
            Some(ZipPlus::<Zt1, Lc1>::prove_pre_open_f::<F, CHECK_FOR_OVERFLOW>(
                transcript, pp1, polys1, point, field_cfg,
            )?)
        };
        let eval2 = if polys2.is_empty() {
            None
        } else {
            Some(ZipPlus::<Zt2, Lc2>::prove_pre_open_f::<F, CHECK_FOR_OVERFLOW>(
                transcript, pp2, polys2, point, field_cfg,
            )?)
        };

        // Layout of the column blob mirrors `new_combined_3`'s leaf
        // input: group 0 rows, then group 1, then group 2, in matrix
        // order. The verifier splits the decompressed buffer by the
        // same per-group byte counts.
        let eb0 = <Zt0::Cw as ConstTranscribable>::NUM_BYTES;
        let eb1 = <Zt1::Cw as ConstTranscribable>::NUM_BYTES;
        let eb2 = <Zt2::Cw as ConstTranscribable>::NUM_BYTES;
        let total_rows_0: usize = hint.cw_matrices_0.iter().map(|m| m.num_rows).sum();
        let total_rows_1: usize = hint.cw_matrices_1.iter().map(|m| m.num_rows).sum();
        let total_rows_2: usize = hint.cw_matrices_2.iter().map(|m| m.num_rows).sum();
        let blob_size = total_rows_0 * eb0 + total_rows_1 * eb1 + total_rows_2 * eb2;

        for _ in 0..Zt0::NUM_COLUMN_OPENINGS {
            let column_idx = transcript.squeeze_challenge_idx(codeword_len);

            let mut col_buf = vec![0_u8; blob_size];
            let mut offset = 0;
            for cw_matrix in &hint.cw_matrices_0 {
                for row in cw_matrix.as_rows() {
                    row[column_idx]
                        .write_transcription_bytes_exact(&mut col_buf[offset..offset + eb0]);
                    offset += eb0;
                }
            }
            for cw_matrix in &hint.cw_matrices_1 {
                for row in cw_matrix.as_rows() {
                    row[column_idx]
                        .write_transcription_bytes_exact(&mut col_buf[offset..offset + eb1]);
                    offset += eb1;
                }
            }
            for cw_matrix in &hint.cw_matrices_2 {
                for row in cw_matrix.as_rows() {
                    row[column_idx]
                        .write_transcription_bytes_exact(&mut col_buf[offset..offset + eb2]);
                    offset += eb2;
                }
            }
            transcript.write_compressed_blob(&col_buf)?;

            let merkle_proof = hint
                .merkle_tree
                .prove(column_idx)
                .map_err(|_| ZipError::InvalidPcsOpen("Failed to open merkle tree".into()))?;
            transcript.write_merkle_proof(&merkle_proof).map_err(|_| {
                ZipError::InvalidPcsOpen("Failed to write a merkle tree proof".into())
            })?;
        }

        Ok((eval0, eval1, eval2))
    }

    /// Like [`Self::prove_f`], but additionally captures per-section byte
    /// counts for each of the three domains. The shared Merkle path bytes
    /// (one path per opening, common to all domains) are attributed in
    /// full to the first non-empty domain (`bd0` if instance 0 is
    /// non-empty, otherwise `bd1`, otherwise `bd2`) — they are not
    /// double-counted across domains.
    #[allow(clippy::too_many_arguments)]
    pub fn prove_f_with_byte_breakdown<F, const CHECK_FOR_OVERFLOW: bool>(
        transcript: &mut PcsProverTranscript,
        pp0: &ZipPlusParams<Zt0, Lc0>,
        pp1: &ZipPlusParams<Zt1, Lc1>,
        pp2: &ZipPlusParams<Zt2, Lc2>,
        polys0: &[DenseMultilinearExtension<Zt0::Eval>],
        polys1: &[DenseMultilinearExtension<Zt1::Eval>],
        polys2: &[DenseMultilinearExtension<Zt2::Eval>],
        point: &[F],
        hint: &MultiZipHint3<Zt0::Cw, Zt1::Cw, Zt2::Cw>,
        field_cfg: &F::Config,
        bd0: &mut ZipPlusProveByteBreakdown,
        bd1: &mut ZipPlusProveByteBreakdown,
        bd2: &mut ZipPlusProveByteBreakdown,
    ) -> Result<(Option<F>, Option<F>, Option<F>), ZipError>
    where
        F: PrimeField
            + for<'a> FromWithConfig<&'a Zt0::CombR>
            + for<'a> FromWithConfig<&'a Zt1::CombR>
            + for<'a> FromWithConfig<&'a Zt2::CombR>
            + for<'a> MulByScalar<&'a F>
            + FromRef<F>,
        F::Inner: Transcribable,
        F::Modulus: Transcribable,
    {
        let pos = |t: &PcsProverTranscript| -> usize { t.stream.position() as usize };
        let snapshot = |t: &PcsProverTranscript, lo: usize, hi: usize| -> Vec<u8> {
            t.stream.get_ref()[lo..hi].to_vec()
        };

        assert_eq!(
            Zt0::NUM_COLUMN_OPENINGS,
            Zt1::NUM_COLUMN_OPENINGS,
            "MultiZip3 requires equal NUM_COLUMN_OPENINGS"
        );
        assert_eq!(
            Zt0::NUM_COLUMN_OPENINGS,
            Zt2::NUM_COLUMN_OPENINGS,
            "MultiZip3 requires equal NUM_COLUMN_OPENINGS"
        );
        let codeword_len = if !polys0.is_empty() {
            pp0.linear_code.codeword_len()
        } else if !polys1.is_empty() {
            pp1.linear_code.codeword_len()
        } else {
            pp2.linear_code.codeword_len()
        };

        // We attribute each domain's `prove_pre_open_f` write span (b +
        // combined_row) to that domain's `combined_row` field — we
        // don't split b vs combined_row here, since the byte boundary
        // depends on transcript-internal length prefixes that aren't
        // exposed. Field `b` keeps its default empty Vec for each
        // domain; the bench helper sums `combined_row + b` for the
        // displayed pre-open total per domain (and reports `b` as 0).

        let p0 = pos(transcript);
        let eval0 = if polys0.is_empty() {
            None
        } else {
            Some(ZipPlus::<Zt0, Lc0>::prove_pre_open_f::<F, CHECK_FOR_OVERFLOW>(
                transcript, pp0, polys0, point, field_cfg,
            )?)
        };
        let p1 = pos(transcript);
        bd0.combined_row.extend(snapshot(transcript, p0, p1));

        let eval1 = if polys1.is_empty() {
            None
        } else {
            Some(ZipPlus::<Zt1, Lc1>::prove_pre_open_f::<F, CHECK_FOR_OVERFLOW>(
                transcript, pp1, polys1, point, field_cfg,
            )?)
        };
        let p2 = pos(transcript);
        bd1.combined_row.extend(snapshot(transcript, p1, p2));

        let eval2 = if polys2.is_empty() {
            None
        } else {
            Some(ZipPlus::<Zt2, Lc2>::prove_pre_open_f::<F, CHECK_FOR_OVERFLOW>(
                transcript, pp2, polys2, point, field_cfg,
            )?)
        };
        let p3 = pos(transcript);
        bd2.combined_row.extend(snapshot(transcript, p2, p3));

        // Choose which domain absorbs the shared Merkle path bytes.
        let shared_merkle_target: usize = if !polys0.is_empty() {
            0
        } else if !polys1.is_empty() {
            1
        } else {
            2
        };

        // Same layout as `prove_f`: one compressed blob per opening,
        // covering group 0, 1, 2 in order. Per-domain byte breakdown
        // can no longer measure each group's contribution exactly (the
        // blob is a single compressed unit), so the full compressed
        // blob is attributed to the shared-merkle target domain
        // alongside the Merkle path bytes.
        let eb0 = <Zt0::Cw as ConstTranscribable>::NUM_BYTES;
        let eb1 = <Zt1::Cw as ConstTranscribable>::NUM_BYTES;
        let eb2 = <Zt2::Cw as ConstTranscribable>::NUM_BYTES;
        let total_rows_0: usize = hint.cw_matrices_0.iter().map(|m| m.num_rows).sum();
        let total_rows_1: usize = hint.cw_matrices_1.iter().map(|m| m.num_rows).sum();
        let total_rows_2: usize = hint.cw_matrices_2.iter().map(|m| m.num_rows).sum();
        let blob_size = total_rows_0 * eb0 + total_rows_1 * eb1 + total_rows_2 * eb2;

        for _ in 0..Zt0::NUM_COLUMN_OPENINGS {
            let column_idx = transcript.squeeze_challenge_idx(codeword_len);

            let mut col_buf = vec![0_u8; blob_size];
            let mut offset = 0;
            for cw_matrix in &hint.cw_matrices_0 {
                for row in cw_matrix.as_rows() {
                    row[column_idx]
                        .write_transcription_bytes_exact(&mut col_buf[offset..offset + eb0]);
                    offset += eb0;
                }
            }
            for cw_matrix in &hint.cw_matrices_1 {
                for row in cw_matrix.as_rows() {
                    row[column_idx]
                        .write_transcription_bytes_exact(&mut col_buf[offset..offset + eb1]);
                    offset += eb1;
                }
            }
            for cw_matrix in &hint.cw_matrices_2 {
                for row in cw_matrix.as_rows() {
                    row[column_idx]
                        .write_transcription_bytes_exact(&mut col_buf[offset..offset + eb2]);
                    offset += eb2;
                }
            }
            let q0 = pos(transcript);
            transcript.write_compressed_blob(&col_buf)?;
            let q1 = pos(transcript);
            let blob_bytes = snapshot(transcript, q0, q1);
            match shared_merkle_target {
                0 => bd0.column_values.extend(blob_bytes),
                1 => bd1.column_values.extend(blob_bytes),
                _ => bd2.column_values.extend(blob_bytes),
            }

            let merkle_proof = hint
                .merkle_tree
                .prove(column_idx)
                .map_err(|_| ZipError::InvalidPcsOpen("Failed to open merkle tree".into()))?;
            let m0 = pos(transcript);
            transcript.write_merkle_proof(&merkle_proof).map_err(|_| {
                ZipError::InvalidPcsOpen("Failed to write a merkle tree proof".into())
            })?;
            let m1 = pos(transcript);
            let bytes = snapshot(transcript, m0, m1);
            match shared_merkle_target {
                0 => bd0.merkle_proofs.extend(bytes),
                1 => bd1.merkle_proofs.extend(bytes),
                _ => bd2.merkle_proofs.extend(bytes),
            }
        }

        Ok((eval0, eval1, eval2))
    }

    /// Verifier counterpart of [`Self::prove_f`]'s column-opening loop.
    ///
    /// The caller must drive the per-instance pre-open phase by hand,
    /// interleaved with alpha sampling, because instance N+1's alphas
    /// depend on the FS state after instance N's pre-open writes:
    ///
    /// ```text
    /// alphas_0 = ZipPlus::<Zt0,_>::sample_alphas(transcript, ...);
    /// eval_f_0 = ...; // from lifted_evals + alphas_0
    /// pre0     = ZipPlus::<Zt0,_>::verify_pre_open(transcript, vp0, comm0, ..., eval_f_0)?;
    /// alphas_1 = ZipPlus::<Zt1,_>::sample_alphas(transcript, ...);
    /// eval_f_1 = ...;
    /// pre1     = ZipPlus::<Zt1,_>::verify_pre_open(transcript, vp1, comm1, ..., eval_f_1)?;
    /// alphas_2 = ZipPlus::<Zt2,_>::sample_alphas(transcript, ...);
    /// eval_f_2 = ...;
    /// pre2     = ZipPlus::<Zt2,_>::verify_pre_open(transcript, vp2, comm2, ..., eval_f_2)?;
    /// MultiZip3::verify_columns_shared(transcript, vp0..vp2, comm0..comm2,
    ///                                  per_poly_alphas_0..2, &pre0..pre2)?;
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn verify_columns_shared<F, const CHECK_FOR_OVERFLOW: bool>(
        transcript: &mut PcsVerifierTranscript,
        vp0: &ZipPlusParams<Zt0, Lc0>,
        vp1: &ZipPlusParams<Zt1, Lc1>,
        vp2: &ZipPlusParams<Zt2, Lc2>,
        comm0: &ZipPlusCommitment,
        comm1: &ZipPlusCommitment,
        comm2: &ZipPlusCommitment,
        per_poly_alphas_0: &[Vec<Zt0::Chal>],
        per_poly_alphas_1: &[Vec<Zt1::Chal>],
        per_poly_alphas_2: &[Vec<Zt2::Chal>],
        pre0: Option<&VerifyPreOpen<Zt0>>,
        pre1: Option<&VerifyPreOpen<Zt1>>,
        pre2: Option<&VerifyPreOpen<Zt2>>,
    ) -> Result<(), ZipError>
    where
        F: FromPrimitiveWithConfig
            + FromRef<F>
            + for<'a> FromWithConfig<&'a Zt0::CombR>
            + for<'a> FromWithConfig<&'a Zt1::CombR>
            + for<'a> FromWithConfig<&'a Zt2::CombR>
            + for<'a> MulByScalar<&'a F>,
        F::Inner: Transcribable,
        F::Modulus: Transcribable,
    {
        assert_eq!(
            comm0.root, comm1.root,
            "MultiZip3 commitments must share root"
        );
        assert_eq!(
            comm0.root, comm2.root,
            "MultiZip3 commitments must share root"
        );
        assert_eq!(
            Zt0::NUM_COLUMN_OPENINGS,
            Zt1::NUM_COLUMN_OPENINGS,
            "MultiZip3 requires equal NUM_COLUMN_OPENINGS"
        );
        assert_eq!(
            Zt0::NUM_COLUMN_OPENINGS,
            Zt2::NUM_COLUMN_OPENINGS,
            "MultiZip3 requires equal NUM_COLUMN_OPENINGS"
        );
        // Same codeword-length convention as `commit` and `prove_f`:
        // empty instances are skipped.
        let codeword_len = if comm0.batch_size > 0 {
            vp0.linear_code.codeword_len()
        } else if comm1.batch_size > 0 {
            vp1.linear_code.codeword_len()
        } else {
            vp2.linear_code.codeword_len()
        };

        let bs0 = comm0.batch_size;
        let bs1 = comm1.batch_size;
        let bs2 = comm2.batch_size;
        let n0 = vp0.num_rows;
        let n1 = vp1.num_rows;
        let n2 = vp2.num_rows;

        // Sanity: pre/comm/alphas presence must be consistent.
        assert_eq!(pre0.is_some(), bs0 > 0);
        assert_eq!(pre1.is_some(), bs1 > 0);
        assert_eq!(pre2.is_some(), bs2 > 0);

        let eb0 = <Zt0::Cw as ConstTranscribable>::NUM_BYTES;
        let eb1 = <Zt1::Cw as ConstTranscribable>::NUM_BYTES;
        let eb2 = <Zt2::Cw as ConstTranscribable>::NUM_BYTES;
        let bytes_0 = bs0 * n0 * eb0;
        let bytes_1 = bs1 * n1 * eb1;
        let bytes_2 = bs2 * n2 * eb2;
        let total_bytes = bytes_0 + bytes_1 + bytes_2;

        let openings: Vec<_> = (0..Zt0::NUM_COLUMN_OPENINGS)
            .map(|_| -> Result<_, ZipError> {
                let column_idx = transcript.squeeze_challenge_idx(codeword_len);
                let (decompressed, leaf) = transcript.read_compressed_blob()?;
                if decompressed.len() != total_bytes {
                    return Err(ZipError::InvalidPcsOpen(format!(
                        "Decompressed combined column blob has unexpected size: got {}, expected {}",
                        decompressed.len(),
                        total_bytes,
                    )));
                }
                let (s0, rest) = decompressed.split_at(bytes_0);
                let (s1, s2) = rest.split_at(bytes_1);
                debug_assert_eq!(s2.len(), bytes_2);
                let cv0: Vec<Zt0::Cw> = s0
                    .chunks_exact(eb0)
                    .map(<Zt0::Cw as GenTranscribable>::read_transcription_bytes_exact)
                    .collect();
                let cv1: Vec<Zt1::Cw> = s1
                    .chunks_exact(eb1)
                    .map(<Zt1::Cw as GenTranscribable>::read_transcription_bytes_exact)
                    .collect();
                let cv2: Vec<Zt2::Cw> = s2
                    .chunks_exact(eb2)
                    .map(<Zt2::Cw as GenTranscribable>::read_transcription_bytes_exact)
                    .collect();
                let proof = transcript.read_merkle_proof().map_err(|e| {
                    ZipError::InvalidPcsOpen(format!("Failed to read Merkle a proof: {e}"))
                })?;
                Ok((column_idx, cv0, cv1, cv2, leaf, proof))
            })
            .try_collect()?;

        cfg_into_iter!(openings).try_for_each(
            |(column_idx, cv0, cv1, cv2, leaf, proof)| -> Result<(), ZipError> {
                if let Some(pre) = pre0 {
                    ZipPlus::<Zt0, Lc0>::verify_column_testing_batched::<CHECK_FOR_OVERFLOW>(
                        per_poly_alphas_0,
                        &pre.coeffs,
                        &pre.encoded_combined_row,
                        &cv0,
                        column_idx,
                        n0,
                        bs0,
                    )?;
                }
                if let Some(pre) = pre1 {
                    ZipPlus::<Zt1, Lc1>::verify_column_testing_batched::<CHECK_FOR_OVERFLOW>(
                        per_poly_alphas_1,
                        &pre.coeffs,
                        &pre.encoded_combined_row,
                        &cv1,
                        column_idx,
                        n1,
                        bs1,
                    )?;
                }
                if let Some(pre) = pre2 {
                    ZipPlus::<Zt2, Lc2>::verify_column_testing_batched::<CHECK_FOR_OVERFLOW>(
                        per_poly_alphas_2,
                        &pre.coeffs,
                        &pre.encoded_combined_row,
                        &cv2,
                        column_idx,
                        n2,
                        bs2,
                    )?;
                }

                proof
                    .verify_with_leaf_hash(&comm0.root, leaf, column_idx)
                    .map_err(|e| {
                        ZipError::InvalidPcsOpen(format!("Column opening verification failed: {e}"))
                    })?;

                Ok(())
            },
        )?;

        Ok(())
    }
}
