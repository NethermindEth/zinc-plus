use std::marker::PhantomData;

use crypto_bigint::Word;
use crypto_primitives::crypto_bigint_int::Int;
use p3_field::Packable;

use crate::{
    code::LinearCode,
    pcs::{
        MerkleTree,
        utils::{AsWords, MtHash},
    },
    utils::ReinterpretVector,
};

pub struct MultilinearZip<
    const N: usize,
    const L: usize,
    const K: usize,
    const M: usize,
    LC: LinearCode<N, L, K, M>,
>(PhantomData<LC>);

/// Parameters for the Zip PCS.
#[derive(Clone, Debug)]
pub struct MultilinearZipParams<
    const N: usize,
    const L: usize,
    const K: usize,
    const M: usize,
    LC: LinearCode<N, L, K, M>,
> {
    pub num_vars: usize,
    pub num_rows: usize,
    pub linear_code: LC,
}

impl<const N: usize, const L: usize, const K: usize, const M: usize, LC: LinearCode<N, L, K, M>>
    MultilinearZipParams<N, L, K, M, LC>
{
    pub fn new(
        num_vars: usize,
        num_rows: usize,
        linear_code: LC,
    ) -> MultilinearZipParams<N, L, K, M, LC> {
        Self {
            num_vars,
            num_rows,
            linear_code,
        }
    }
}

#[derive(Copy, Clone, Default, PartialEq, Eq, Debug)]
#[repr(transparent)]
pub struct PackedInt<const LIMBS: usize>(pub(crate) Int<LIMBS>);

unsafe impl<const N: usize> ReinterpretVector<PackedInt<N>> for Int<N> {}
unsafe impl<const N: usize> ReinterpretVector<Int<N>> for PackedInt<N> {}

impl<const LIMBS: usize> Packable for PackedInt<LIMBS> {}

impl<const LIMBS: usize> AsWords for PackedInt<LIMBS> {
    fn as_words(&self) -> &[Word] {
        self.0.inner().as_words()
    }
}

/// Representantation of a zip commitment to a multilinear polynomial
#[derive(Debug, Default)]
pub struct MultilinearZipData<const K: usize> {
    /// The encoded rows of the polynomial matrix representation, referred to as
    /// "u-hat" in the Zinc paper
    pub rows: Vec<Int<K>>,
    /// Merkle trees of entire matrix
    pub merkle_tree: MerkleTree<PackedInt<K>>,
}

/// Representantation of a zip commitment to a multilinear polynomial
#[derive(Clone, Debug, Default)]
pub struct MultilinearZipCommitment {
    /// Roots of the merkle tree of entire matrix
    pub root: MtHash,
}

impl<const K: usize> MultilinearZipData<K> {
    pub fn new(rows: Vec<Int<K>>, merkle_tree: MerkleTree<PackedInt<K>>) -> MultilinearZipData<K> {
        MultilinearZipData { rows, merkle_tree }
    }

    pub fn root(&self) -> MtHash {
        self.merkle_tree.root()
    }
}

impl<const N: usize, const L: usize, const K: usize, const M: usize, LC: LinearCode<N, L, K, M>>
    MultilinearZip<N, L, K, M, LC>
{
    pub fn setup(poly_size: usize, linear_code: LC) -> MultilinearZipParams<N, L, K, M, LC> {
        assert!(poly_size.is_power_of_two());
        let num_vars = poly_size.ilog2() as usize;
        let num_rows = ((1 << num_vars) / linear_code.row_len()).next_power_of_two();

        MultilinearZipParams {
            num_vars,
            num_rows,
            linear_code,
        }
    }
}
