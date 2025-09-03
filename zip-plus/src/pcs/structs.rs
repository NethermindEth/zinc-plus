use ark_std::{collections::BTreeSet, marker::PhantomData, vec::Vec};

use super::utils::{MerkleTree, MtHash};
use crate::{
    code::LinearCode,
    traits::{Integer, ZipTypes},
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

impl<ZT: ZipTypes, LC: LinearCode<ZT>> MultilinearZipParams<ZT, LC> {
    pub fn new(num_vars: usize, num_rows: usize, linear_code: LC) -> MultilinearZipParams<ZT, LC> {
        Self {
            num_vars,
            num_rows,
            linear_code,
            phantom_data_zt: PhantomData,
        }
    }
}

/// Representantation of a zip commitment to a multilinear polynomial
#[derive(Debug, Default)]
pub struct MultilinearZipData<K: Integer> {
    /// The encoded rows of the polynomial matrix representation, referred to as "u-hat" in the Zinc paper
    pub rows: Vec<K>,
    /// Merkle trees of entire matrix
    pub merkle_tree: MerkleTree<K>,
}

/// Representantation of a zip commitment to a multilinear polynomial
#[derive(Clone, Debug, Default)]
pub struct MultilinearZipCommitment {
    /// Roots of the merkle tree of entire matrix
    pub root: MtHash,
}

impl<K: Integer> MultilinearZipData<K> {
    pub fn new(rows: Vec<K>, merkle_tree: MerkleTree<K>) -> MultilinearZipData<K> {
        MultilinearZipData { rows, merkle_tree }
    }

    pub fn root(&self) -> MtHash {
        self.merkle_tree.root()
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
