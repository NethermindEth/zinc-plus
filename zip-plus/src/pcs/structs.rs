use std::marker::PhantomData;

use crate::{
    code::LinearCode,
    pcs::{
        MerkleTree,
        utils::{AsWords, MtHash},
    },
    utils::ReinterpretVector,
};
use crypto_bigint::Word;
use crypto_primitives::{Ring, crypto_bigint_int::Int};
use p3_field::Packable;

/// Zip is a Polynomial Commitment Scheme (PCS) that supports committing to
/// multilinear polynomials.
///
/// Params:
/// - `Eval`: Ring for elements in witness/polynomial evaluations on hypercube.
/// - `Cw`: Ring for codeword elements.
/// - `Comb`: Ring for elements in the linear combination of codewords.
/// - `C`: The linear code used for encoding the polynomial.
pub struct MultilinearZip<Eval, Cw, Comb, C>(PhantomData<(Eval, Cw, Comb, C)>)
where
    Eval: Ring,
    Cw: Ring,
    Comb: Ring,
    C: LinearCode<Eval, Cw, Comb>;

impl<Eval, Cw, Comb, C> MultilinearZip<Eval, Cw, Comb, C>
where
    Eval: Ring,
    Cw: Ring,
    Comb: Ring,
    C: LinearCode<Eval, Cw, Comb>,
{
    #[allow(clippy::arithmetic_side_effects)]
    pub fn setup(poly_size: usize, linear_code: C) -> MultilinearZipParams<Eval, Cw, Comb, C> {
        assert!(poly_size.is_power_of_two());
        let num_vars = poly_size.ilog2() as usize;
        let num_rows = ((1 << num_vars) / linear_code.row_len()).next_power_of_two();
        MultilinearZipParams::new(num_vars, num_rows, linear_code)
    }
}

/// Parameters for the Zip PCS.
#[derive(Clone, Debug)]
pub struct MultilinearZipParams<Eval, Cw, Comb, C>
where
    Eval: Ring,
    Cw: Ring,
    Comb: Ring,
    C: LinearCode<Eval, Cw, Comb>,
{
    pub num_vars: usize,
    pub num_rows: usize,
    pub linear_code: C,
    phantom_data: PhantomData<(Eval, Cw, Comb)>,
}

impl<Eval, Cw, Comb, C> MultilinearZipParams<Eval, Cw, Comb, C>
where
    Eval: Ring,
    Cw: Ring,
    Comb: Ring,
    C: LinearCode<Eval, Cw, Comb>,
{
    pub fn new(num_vars: usize, num_rows: usize, linear_code: C) -> Self {
        Self {
            num_vars,
            num_rows,
            linear_code,
            phantom_data: PhantomData,
        }
    }
}

/// Representantation of a zip commitment to a multilinear polynomial
#[derive(Debug, Default)]
pub struct MultilinearZipData<R: AsPackable> {
    /// The encoded rows of the polynomial matrix representation, referred to as
    /// "u-hat" in the Zinc paper
    pub rows: Vec<R>,
    /// Merkle trees of entire matrix
    pub merkle_tree: MerkleTree<R::Packable>,
}

impl<R: AsPackable> MultilinearZipData<R> {
    pub fn new(rows: Vec<R>, merkle_tree: MerkleTree<R::Packable>) -> MultilinearZipData<R> {
        MultilinearZipData { rows, merkle_tree }
    }

    pub fn root(&self) -> MtHash {
        self.merkle_tree.root()
    }
}

/// Representantation of a zip commitment to a multilinear polynomial
#[derive(Clone, Debug, Default)]
pub struct MultilinearZipCommitment {
    /// Roots of the merkle tree of entire matrix
    pub root: MtHash,
}

pub trait AsPackable: Clone + ReinterpretVector<Self::Packable> {
    type Packable: Packable + AsWords + Clone + Send + Sync;
}

impl<const LIMBS: usize> AsPackable for Int<LIMBS> {
    type Packable = PackedInt<LIMBS>;
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
