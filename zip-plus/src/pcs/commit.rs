use ark_std::{vec, vec::Vec};

use super::{
    structs::{MultilinearZip, MultilinearZipCommitment, MultilinearZipData},
    utils::{MerkleTree, validate_input},
};
use crate::{
    Error,
    code::LinearCode,
    pcs::structs::MultilinearZipParams,
    poly_z::mle::DenseMultilinearExtension,
    traits::{Field, ZipTypes},
    utils::{div_ceil, num_threads, parallelize_iter},
};

impl<ZT: ZipTypes, LC: LinearCode<ZT>> MultilinearZip<ZT, LC> {
    pub fn commit<F: Field>(
        pp: &MultilinearZipParams<ZT, LC>,
        poly: &DenseMultilinearExtension<ZT::N>,
    ) -> Result<(MultilinearZipData<ZT::K>, MultilinearZipCommitment), Error> {
        validate_input("commit", pp.num_vars, [poly], None::<&[F]>)?;

        let row_len = pp.linear_code.row_len();
        let codeword_len = pp.linear_code.codeword_len();
        let merkle_depth: usize = codeword_len.next_power_of_two().ilog2() as usize;

        let rows = Self::encode_rows(pp, codeword_len, row_len, &poly.evaluations);

        let rows_merkle_trees = rows
            .chunks_exact(codeword_len)
            .map(|row| MerkleTree::new(merkle_depth, row))
            .collect::<Vec<_>>();

        assert_eq!(rows_merkle_trees.len(), pp.num_rows);

        let roots = rows_merkle_trees
            .iter()
            .map(|tree| tree.root)
            .collect::<Vec<_>>();

        Ok((
            MultilinearZipData::new(rows, rows_merkle_trees),
            MultilinearZipCommitment { roots },
        ))
    }

    #[allow(clippy::type_complexity)]
    pub fn batch_commit<F: Field>(
        pp: &MultilinearZipParams<ZT, LC>,
        polys: &[DenseMultilinearExtension<ZT::N>],
    ) -> Result<Vec<(MultilinearZipData<ZT::K>, MultilinearZipCommitment)>, Error> {
        polys
            .iter()
            .map(|poly| Self::commit::<F>(pp, poly))
            .collect()
    }

    /// Encodes the rows of the polynomial concatenating each encoded row
    pub fn encode_rows(
        pp: &MultilinearZipParams<ZT, LC>,
        codeword_len: usize,
        row_len: usize,
        evals: &[ZT::N],
    ) -> Vec<ZT::K> {
        let rows_per_thread = div_ceil(pp.num_rows, num_threads());
        let mut encoded_rows = vec![ZT::K::default(); pp.num_rows * codeword_len];

        parallelize_iter(
            encoded_rows
                .chunks_exact_mut(rows_per_thread * codeword_len)
                .zip(evals.chunks_exact(rows_per_thread * row_len)),
            |(encoded_chunk, evals)| {
                for (row, evals) in encoded_chunk
                    .chunks_exact_mut(codeword_len)
                    .zip(evals.chunks_exact(row_len))
                {
                    let encoded: Vec<ZT::K> = pp.linear_code.encode_wide(evals);
                    row.clone_from_slice(encoded.as_slice());
                }
            },
        );

        encoded_rows
    }
}
