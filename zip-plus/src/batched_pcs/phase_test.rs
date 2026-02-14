use crate::{
    ZipError,
    code::LinearCode,
    combine_rows,
    merkle::MerkleProof,
    pcs::{
        structs::ZipTypes,
        utils::validate_input,
    },
    pcs_transcript::PcsTranscript,
};
use num_traits::{ConstOne, ConstZero, Zero};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use zinc_poly::{Polynomial, mle::DenseMultilinearExtension};
use zinc_transcript::traits::{ConstTranscribable, Transcript};
use zinc_utils::{cfg_iter_mut, inner_product::InnerProduct, mul_by_scalar::MulByScalar};

use super::structs::{
    BatchedZipPlus, BatchedZipPlusHint, BatchedZipPlusParams, BatchedZipPlusTestTranscript,
};

impl<Zt: ZipTypes, Lc: LinearCode<Zt>> BatchedZipPlus<Zt, Lc> {
    /// Performs the testing phase for a batch of committed polynomials.
    ///
    /// This mirrors the single-polynomial testing phase but operates on `m`
    /// polynomials simultaneously. Each polynomial gets its own alphas,
    /// coefficients, and combined row (sampled and computed sequentially from
    /// the shared Fiat-Shamir transcript). Column openings are shared: the
    /// same column indices are queried for all polynomials, and a single
    /// Merkle proof is produced per column.
    ///
    /// # Parameters
    /// - `pp`: Public parameters shared across all polynomials.
    /// - `polys`: Slice of multilinear polynomials in the batch.
    /// - `commit_hint`: The batched commitment hint containing all codeword
    ///   matrices and the shared Merkle tree.
    ///
    /// # Returns
    /// A `BatchedZipPlusTestTranscript` to be used in the evaluation phase.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn test<const CHECK_FOR_OVERFLOW: bool>(
        pp: &BatchedZipPlusParams<Zt, Lc>,
        polys: &[DenseMultilinearExtension<Zt::Eval>],
        commit_hint: &BatchedZipPlusHint<Zt::Cw>,
    ) -> Result<BatchedZipPlusTestTranscript, ZipError> {
        let batch_size = polys.len();
        assert_eq!(
            batch_size,
            commit_hint.batch_size(),
            "Number of polynomials must match batch commit hint"
        );

        for poly in polys {
            validate_input::<Zt, Lc, bool>("batched_test", pp.num_vars, &[poly], &[])?;
        }

        let total_rows_per_poly = pp.num_rows;

        let estimated_transcript_size =
            // Combined rows: one per polynomial
            batch_size * pp.linear_code.row_len() * Zt::CombR::NUM_BYTES
            // Column openings
            + Zt::NUM_COLUMN_OPENINGS * (
                // Column values from all m polynomials
                batch_size * total_rows_per_poly * Zt::Cw::NUM_BYTES
                // Single Merkle proof per column
                + MerkleProof::estimate_transcribed_size(commit_hint.merkle_tree.height())
            );
        let mut transcript = PcsTranscript::new_with_capacity(estimated_transcript_size);

        // For each polynomial, compute the combined row  (proximity test)
        if pp.num_rows > 1 {
            for (poly_idx, poly) in polys.iter().enumerate() {
                let alphas = if Zt::Comb::DEGREE_BOUND.is_zero() {
                    vec![Zt::Chal::ONE]
                } else {
                    transcript
                        .fs_transcript
                        .get_challenges::<Zt::Chal>(Zt::Comb::DEGREE_BOUND + 1)
                };

                let coeffs = transcript
                    .fs_transcript
                    .get_challenges::<Zt::Chal>(pp.num_rows);

                let combined_row = combine_rows!(
                    CHECK_FOR_OVERFLOW,
                    &coeffs,
                    poly.evaluations.iter(),
                    |eval| Zt::EvalDotChal::inner_product::<CHECK_FOR_OVERFLOW>(
                        eval,
                        &alphas,
                        Zt::CombR::ZERO
                    ),
                    pp.linear_code.row_len(),
                    Zt::CombR::ZERO
                );

                transcript.write_const_many(&combined_row).map_err(|e| {
                    ZipError::Serialization(format!(
                        "Failed to write combined row for poly {poly_idx}: {e}"
                    ))
                })?;
            }
        }

        // Open Merkle tree for each column drawn — shared across all polynomials
        for _ in 0..Zt::NUM_COLUMN_OPENINGS {
            let column = transcript.squeeze_challenge_idx(pp.linear_code.codeword_len());
            Self::open_merkle_trees_for_column(commit_hint, column, &mut transcript)?;
        }

        assert_eq!(
            transcript.stream.get_ref().len(),
            estimated_transcript_size,
            "Batched PCS transcript capacity was precalculated incorrectly"
        );

        Ok(transcript.into())
    }

    /// Opens a single column of the shared Merkle tree, writing column values
    /// from **all** codeword matrices followed by a single Merkle proof.
    pub(super) fn open_merkle_trees_for_column(
        commit_hint: &BatchedZipPlusHint<Zt::Cw>,
        column_idx: usize,
        transcript: &mut PcsTranscript,
    ) -> Result<(), ZipError> {
        // Write column values from each cw_matrix in order
        for cw_matrix in &commit_hint.cw_matrices {
            let column_values = cw_matrix.as_rows().map(|row| &row[column_idx]);
            transcript.write_const_many_iter(column_values, cw_matrix.num_rows)?;
        }

        // Single Merkle proof from the shared tree
        let merkle_proof = commit_hint
            .merkle_tree
            .prove(column_idx)
            .map_err(|_| ZipError::InvalidPcsOpen("Failed to open batched merkle tree".into()))?;
        transcript
            .write_merkle_proof(&merkle_proof)
            .map_err(|_| {
                ZipError::InvalidPcsOpen(
                    "Failed to write batched merkle tree proof".into(),
                )
            })?;

        Ok(())
    }
}

#[cfg(test)]
#[allow(
    clippy::arithmetic_side_effects,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap
)]
mod tests {
    use super::*;
    use crate::pcs::test_utils::*;
    use crypto_bigint::U64;
    use crypto_primitives::crypto_bigint_int::Int;
    use zinc_utils::CHECKED;

    const INT_LIMBS: usize = U64::LIMBS;
    const N: usize = INT_LIMBS;
    const K: usize = INT_LIMBS * 4;
    const M: usize = INT_LIMBS * 8;

    type Zt = TestZipTypes<N, K, M>;
    type C = crate::code::raa_sign_flip::RaaSignFlippingCode<Zt, TestRaaConfig, 4>;
    type TestBatchedZip = BatchedZipPlus<Zt, C>;

    #[test]
    fn successful_batched_testing() {
        let num_vars = 4;
        let (pp, _) = setup_test_params::<N, K, M>(num_vars);

        let poly1: DenseMultilinearExtension<_> = (1..=16).map(Int::from).collect();
        let poly2: DenseMultilinearExtension<_> = (17..=32).map(Int::from).collect();
        let polys = [poly1, poly2];

        let (hint, _comm) = TestBatchedZip::commit(&pp, &polys).unwrap();
        let result = TestBatchedZip::test::<CHECKED>(&pp, &polys, &hint);
        assert!(result.is_ok(), "Batched test failed: {result:?}");
    }

    #[test]
    fn batched_testing_single_poly() {
        let num_vars = 4;
        let (pp, poly) = setup_test_params(num_vars);

        let (hint, _comm) = TestBatchedZip::commit(&pp, &[poly.clone()]).unwrap();
        let result = TestBatchedZip::test::<CHECKED>(&pp, &[poly], &hint);
        assert!(result.is_ok(), "Single-poly batched test failed: {result:?}");
    }
}
