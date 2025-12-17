use crate::{
    code::LinearCode,
    merkle::{MerkleTree, MtHash},
};
use crypto_primitives::{ConstIntRing, ConstIntSemiring, DenseRowMatrix, FixedSemiring};
use num_traits::CheckedAdd;
use std::{borrow::Borrow, marker::PhantomData, ops::Neg};
use zinc_poly::{ConstCoeffBitWidth, Polynomial};
use zinc_primality::PrimalityTest;
use zinc_transcript::traits::ConstTranscribable;
use zinc_utils::{
    from_ref::FromRef, inner_product::InnerProduct, mul_by_scalar::MulByScalar, named::Named,
};

pub trait ZipTypes: Send + Sync {
    const NUM_COLUMN_OPENINGS: usize;

    /// Semiring of witness/polynomial evaluations on boolean hypercube
    type Eval: Borrow<Self::EvalDotChal> + Named + ConstCoeffBitWidth + Clone + Send + Sync;

    /// Semiring of codeword elements, at least as wide as the evaluation ring
    type Cw: FixedSemiring
        + ConstCoeffBitWidth
        + ConstTranscribable
        + FromRef<Self::Eval>
        + Named
        + Copy
        + CheckedAdd;

    /// Semiring type used to draft field modulus elements, natural numbers
    type Fmod: ConstIntSemiring + ConstTranscribable + Named;
    type PrimeTest: PrimalityTest<Self::Fmod>;

    /// Ring of challenge elements (coefficients) to perform a random linear
    /// combination of codewords
    type Chal: ConstIntRing + ConstTranscribable + Named;

    /// Ring of point coordinates to evaluate the multilinear polynomial
    type Pt: ConstIntRing;

    /// Coefficient ring of linear combination polynomial [Self::Comb]
    type CombR: ConstIntRing
        + Neg<Output = Self::CombR>
        + ConstTranscribable
        + FromRef<Self::CombR>
        + for<'a> MulByScalar<&'a Self::Chal>;
    /// Ring of elements in the linear combination of codewords, at least as
    /// wide as the evaluation, codeword, and challenge rings.
    type Comb: FixedSemiring
        + Polynomial<Self::CombR>
        + Borrow<Self::CombDotChal>
        + FromRef<Self::Eval>
        + FromRef<Self::Cw>
        + Named;

    type EvalDotChal: InnerProduct<Self::Chal, Self::CombR>;
    type CombDotChal: InnerProduct<Self::Chal, Self::CombR>;
}

/// Zip is a Polynomial Commitment Scheme (PCS) that supports committing to
/// multilinear polynomials.
pub struct ZipPlus<Zt: ZipTypes, Lc: LinearCode<Zt>>(PhantomData<(Zt, Lc)>);

impl<Zt, Lc> ZipPlus<Zt, Lc>
where
    Zt: ZipTypes,
    Lc: LinearCode<Zt>,
{
    #[allow(clippy::arithmetic_side_effects)]
    pub fn setup(poly_size: usize, linear_code: Lc) -> ZipPlusParams<Zt, Lc> {
        assert!(poly_size.is_power_of_two());
        let num_vars = poly_size.ilog2() as usize;
        let num_rows = ((1 << num_vars) / linear_code.row_len()).next_power_of_two();
        ZipPlusParams::new(num_vars, num_rows, linear_code)
    }
}

/// Parameters for the Zip+ PCS.
#[derive(Clone, Debug)]
pub struct ZipPlusParams<Zt: ZipTypes, Lc: LinearCode<Zt>> {
    pub num_vars: usize,
    pub num_rows: usize,
    pub linear_code: Lc,
    phantom_data: PhantomData<Zt>,
}

impl<Zt: ZipTypes, Lc: LinearCode<Zt>> ZipPlusParams<Zt, Lc> {
    pub fn new(num_vars: usize, num_rows: usize, linear_code: Lc) -> Self {
        Self {
            num_vars,
            num_rows,
            linear_code,
            phantom_data: PhantomData,
        }
    }
}

/// Full data of zip commitment to a multilinear polynomial, including encoded
/// rows and Merkle tree, kept by the prover for the testing phase.
#[derive(Debug, Default)]
pub struct ZipPlusHint<R> {
    /// The encoded rows of the polynomial matrix representation, referred to as
    /// "u-hat" in the Zinc paper
    pub cw_matrix: DenseRowMatrix<R>,
    /// Merkle trees of entire matrix
    pub merkle_tree: MerkleTree,
}

impl<R> ZipPlusHint<R> {
    pub fn new(cw_matrix: DenseRowMatrix<R>, merkle_tree: MerkleTree) -> ZipPlusHint<R> {
        ZipPlusHint {
            cw_matrix,
            merkle_tree,
        }
    }
}

/// The compact commitment to a multilinear polynomial, consisting of only the
/// Merkle roots, to be sent to the verifier.
#[derive(Clone, Debug, Default)]
pub struct ZipPlusCommitment {
    /// Roots of the merkle tree of entire matrix
    pub root: MtHash,
}
