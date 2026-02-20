use crate::{
    ZipError,
    code::LinearCode,
    merkle::MerkleTree,
    pcs::{
        structs::{ZipPlus, ZipTypes},
        utils::validate_input,
    },
};
use zinc_poly::mle::DenseMultilinearExtension;

use super::structs::{
    BatchedZipPlus, BatchedZipPlusCommitment, BatchedZipPlusHint, BatchedZipPlusParams,
};

impl<Zt: ZipTypes, Lc: LinearCode<Zt>> BatchedZipPlus<Zt, Lc> {
    /// Creates a batched commitment to multiple multilinear polynomials using
    /// the ZIP PCS scheme with a single shared Merkle tree.
    ///
    /// # Algorithm
    /// 1. Validates each polynomial's number of variables against the
    ///    parameters.
    /// 2. Encodes each polynomial's evaluations into a codeword matrix using
    ///    the linear code.
    /// 3. Constructs a **single** Merkle tree whose leaves are formed by
    ///    hashing the concatenated columns across all codeword matrices.
    ///    Specifically, the leaf at column index `j` is the hash of
    ///    `(cw_1[0][j], cw_1[1][j], ..., cw_1[n-1][j], cw_2[0][j], ...,
    ///    cw_m[n-1][j])` where `cw_i` is the codeword matrix for polynomial
    ///    `i`, and `n` is the number of rows.
    /// 4. Returns the full commitment data (for the prover) and a compact
    ///    commitment (for the verifier).
    ///
    /// # Parameters
    /// - `pp`: Public parameters shared across all polynomials in the batch.
    /// - `polys`: Slice of multilinear polynomials to be committed to.
    ///
    /// # Returns
    /// A `Result` containing a tuple of:
    /// - `BatchedZipPlusHint`: Full data including encoded rows per polynomial
    ///   and shared Merkle tree.
    /// - `BatchedZipPlusCommitment`: The compact commitment (single Merkle
    ///   root).
    #[allow(clippy::arithmetic_side_effects)]
    pub fn commit(
        pp: &BatchedZipPlusParams<Zt, Lc>,
        polys: &[DenseMultilinearExtension<Zt::Eval>],
    ) -> Result<(BatchedZipPlusHint<Zt::Cw>, BatchedZipPlusCommitment), ZipError> {
        assert!(!polys.is_empty(), "Batch must contain at least one polynomial");

        let row_len = pp.linear_code.row_len();
        let expected_num_evals = pp.num_rows * row_len;

        // Validate and encode each polynomial
        let cw_matrices: Vec<_> = polys
            .iter()
            .map(|poly| {
                validate_input::<Zt, Lc, bool>(
                    "batched_commit",
                    pp.num_vars,
                    &[poly],
                    &[],
                )?;
                assert_eq!(
                    poly.len(),
                    expected_num_evals,
                    "Polynomial has an incorrect number of evaluations ({}) for the expected matrix size ({})",
                    poly.len(),
                    expected_num_evals
                );
                Ok(ZipPlus::<Zt, Lc>::encode_rows(pp, row_len, poly))
            })
            .collect::<Result<Vec<_>, ZipError>>()?;

        // Build a single Merkle tree from all matrices' rows concatenated.
        // The rows are ordered as: all rows of poly 1, then all rows of poly 2, etc.
        // This means the leaf at column j hashes across all rows of ALL polynomials.
        let all_rows: Vec<&[Zt::Cw]> = cw_matrices
            .iter()
            .flat_map(|m| m.to_rows_slices())
            .collect();

        let merkle_tree = MerkleTree::new(&all_rows);
        let root = merkle_tree.root();

        let batch_size = polys.len();

        Ok((
            BatchedZipPlusHint::new(cw_matrices, merkle_tree),
            BatchedZipPlusCommitment { root, batch_size },
        ))
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

    const INT_LIMBS: usize = U64::LIMBS;
    const N: usize = INT_LIMBS;
    const K: usize = INT_LIMBS * 4;
    const M: usize = INT_LIMBS * 8;

    type Zt = TestZipTypes<N, K, M>;
    type C = crate::code::raa_sign_flip::RaaSignFlippingCode<Zt, TestRaaConfig, 4>;
    type TestBatchedZip = BatchedZipPlus<Zt, C>;

    #[test]
    fn batched_commit_single_poly_matches_single_commit() {
        let num_vars = 4;
        let (pp, poly) = setup_test_params(num_vars);

        // Batched commit with one polynomial
        let (batched_hint, batched_comm) =
            TestBatchedZip::commit(&pp, &[poly.clone()]).unwrap();

        // Single commit
        let (single_hint, single_comm) =
            crate::pcs::structs::ZipPlus::<Zt, C>::commit(&pp, &poly).unwrap();

        // Roots should match when there's only one polynomial
        assert_eq!(batched_comm.root, single_comm.root);
        assert_eq!(batched_hint.cw_matrices.len(), 1);
        assert_eq!(batched_hint.cw_matrices[0], single_hint.cw_matrix);
    }

    #[test]
    fn batched_commit_produces_different_root_than_individual_commits() {
        let num_vars = 4;
        let (pp, _) = setup_test_params::<N, K, M>(num_vars);

        let poly1: DenseMultilinearExtension<_> = (1..=16).map(Int::from).collect();
        let poly2: DenseMultilinearExtension<_> = (17..=32).map(Int::from).collect();

        let (_, batched_comm) =
            TestBatchedZip::commit(&pp, &[poly1.clone(), poly2.clone()]).unwrap();

        let (_, comm1) =
            crate::pcs::structs::ZipPlus::<Zt, C>::commit(&pp, &poly1).unwrap();
        let (_, comm2) =
            crate::pcs::structs::ZipPlus::<Zt, C>::commit(&pp, &poly2).unwrap();

        // The batched root should differ from either individual root
        assert_ne!(batched_comm.root, comm1.root);
        assert_ne!(batched_comm.root, comm2.root);
    }

    #[test]
    fn batched_commit_is_deterministic() {
        let num_vars = 4;
        let (pp, _) = setup_test_params::<N, K, M>(num_vars);

        let poly1: DenseMultilinearExtension<_> = (1..=16).map(Int::from).collect();
        let poly2: DenseMultilinearExtension<_> = (17..=32).map(Int::from).collect();

        let (_, comm_a) = TestBatchedZip::commit(&pp, &[poly1.clone(), poly2.clone()]).unwrap();
        let (_, comm_b) = TestBatchedZip::commit(&pp, &[poly1, poly2]).unwrap();

        assert_eq!(comm_a.root, comm_b.root);
    }

    #[test]
    fn batched_commit_batch_size_is_correct() {
        let num_vars = 4;
        let (pp, _) = setup_test_params::<N, K, M>(num_vars);

        let polys: Vec<DenseMultilinearExtension<_>> = (0..5)
            .map(|offset| {
                let start = offset * 16 + 1;
                (start..start + 16).map(Int::from).collect()
            })
            .collect();

        let (hint, comm) = TestBatchedZip::commit(&pp, &polys).unwrap();
        assert_eq!(comm.batch_size, 5);
        assert_eq!(hint.batch_size(), 5);
    }
}
