use ark_std::{collections::BTreeSet, marker::PhantomData, vec::Vec};
use sha3::{digest::Output, Keccak256};

use super::utils::MerkleTree;
use crate::{
    traits::{Integer, ZipTypes},
    code::LinearCode,
};

pub struct MultilinearZip<ZT: ZipTypes, LC: LinearCode<ZT>>(PhantomData<(ZT, LC)>);

/// Parameters for the Zip PCS.
#[derive(Clone, Debug)]
pub struct MultilinearZipParams<ZT: ZipTypes, LC: LinearCode<ZT>> {
    pub num_vars: usize,
    pub num_rows: usize,
    pub linear_code: LC,
    phantom_data_zt: PhantomData<ZT>,
}

/// Representantation of a zip commitment to a multilinear polynomial
#[derive(Clone, Debug, Default)]
pub struct MultilinearZipData<K: Integer> {
    /// The encoded rows of the polynomial matrix representation, referred to as "u-hat" in the Zinc paper
    pub rows: Vec<K>,
    /// Merkle trees of each row
    pub rows_merkle_trees: Vec<MerkleTree>,
}

/// Representantation of a zip commitment to a multilinear polynomial
#[derive(Clone, Debug, Default)]
pub struct MultilinearZipCommitment {
    /// Roots of the merkle tree of each row
    pub roots: Vec<Output<Keccak256>>,
}

impl<K: Integer> MultilinearZipData<K> {
    pub fn new(rows: Vec<K>, rows_merkle_trees: Vec<MerkleTree>) -> MultilinearZipData<K> {
        MultilinearZipData {
            rows,
            rows_merkle_trees,
        }
    }

    pub fn roots(&self) -> Vec<Output<Keccak256>> {
        self.rows_merkle_trees
            .iter()
            .map(|tree| tree.root)
            .collect::<Vec<_>>()
    }

    pub fn root_at_index(&self, index: usize) -> Output<Keccak256> {
        self.rows_merkle_trees[index].root
    }
}

pub trait ZipTranscript<I: Integer> {
    fn get_encoding_element(&mut self) -> I;
    fn get_u64(&mut self) -> u64;
    fn sample_unique_columns(
        &mut self,
        range: ark_std::ops::Range<usize>,
        columns: &mut BTreeSet<usize>,
        count: usize,
    ) -> usize;
}

impl<ZT: ZipTypes, LC: LinearCode<ZT>> MultilinearZip<ZT, LC> {
    pub fn setup(poly_size: usize, linear_code: LC) -> MultilinearZipParams<ZT, LC> {
        assert!(poly_size.is_power_of_two());
        let num_vars = poly_size.ilog2() as usize;
        let num_rows = ((1 << num_vars) / linear_code.row_len()).next_power_of_two();

        MultilinearZipParams {
            num_vars,
            num_rows,
            linear_code,
            phantom_data_zt: PhantomData,
        }
    }
}
