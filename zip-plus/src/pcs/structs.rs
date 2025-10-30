use crate::{
    code::LinearCode,
    merkle::{MerkleTree, MtHash},
    poly::{ConstCoeffBitWidth, EvaluatablePolynomial},
    primality::PrimalityTest,
    traits::{ConstTranscribable, FromRef, Named},
};
use crypto_primitives::{
    ConstIntRing, ConstIntSemiring, DenseRowMatrix, FixedSemiring, PrimeField,
    crypto_bigint_int::Int,
};
use num_traits::CheckedMul;
use std::{marker::PhantomData, ops::Neg};

pub trait ZipTypes<const DEGREE: usize>: Send + Sync {
    const NUM_COLUMN_OPENINGS: usize;

    /// Semiring of witness/polynomial evaluations on boolean hypercube
    type Eval: FixedSemiring + Named + ConstCoeffBitWidth;

    /// Semiring of codeword elements, at least as wide as the evaluation ring
    type Cw: FixedSemiring
        + ConstCoeffBitWidth
        + ConstTranscribable
        + FromRef<Self::Eval>
        + Named
        + Copy;

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
        + EvaluatablePolynomial<Self::Chal, Self::CombR>
        + FromRef<Self::Eval>
        + FromRef<Self::Cw>
        + Named;
}

/// Zip is a Polynomial Commitment Scheme (PCS) that supports committing to
/// multilinear polynomials.
pub struct ZipPlus<Zt: ZipTypes<DEGREE>, Lc: LinearCode<Zt, DEGREE>, const DEGREE: usize>(
    PhantomData<(Zt, Lc)>,
);

impl<Zt, Lc, const DEGREE: usize> ZipPlus<Zt, Lc, DEGREE>
where
    Zt: ZipTypes<DEGREE>,
    Lc: LinearCode<Zt, DEGREE>,
{
    #[allow(clippy::arithmetic_side_effects)]
    pub fn setup(poly_size: usize, linear_code: Lc) -> ZipPlusParams<Zt, Lc, DEGREE> {
        assert!(poly_size.is_power_of_two());
        let num_vars = poly_size.ilog2() as usize;
        let num_rows = ((1 << num_vars) / linear_code.row_len()).next_power_of_two();
        ZipPlusParams::new(num_vars, num_rows, linear_code)
    }
}

/// Parameters for the Zip+ PCS.
#[derive(Clone, Debug)]
pub struct ZipPlusParams<Zt: ZipTypes<DEGREE>, Lc: LinearCode<Zt, DEGREE>, const DEGREE: usize> {
    pub num_vars: usize,
    pub num_rows: usize,
    pub linear_code: Lc,
    phantom_data: PhantomData<Zt>,
}

impl<Zt: ZipTypes<DEGREE>, Lc: LinearCode<Zt, DEGREE>, const DEGREE: usize>
    ZipPlusParams<Zt, Lc, DEGREE>
{
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

pub trait MulByScalar<Rhs>: Sized {
    /// Multiplies the current element by a scalar from the right (usually - a
    /// coefficient to obtain a linear combination).
    /// Returns `None` if the multiplication would overflow.
    fn mul_by_scalar(&self, rhs: Rhs) -> Option<Self>;
}

macro_rules! impl_mul_by_scalar_for_primitives {
    ($($t:ty),*) => {
        $(
            impl MulByScalar<&$t> for $t {
                fn mul_by_scalar(&self, rhs: &$t) -> Option<Self> {
                    self.checked_mul(rhs)
                }
            }
        )*
    };
}

impl_mul_by_scalar_for_primitives!(i8, i16, i32, i64, i128);

impl<const LIMBS: usize, const LIMBS2: usize> MulByScalar<&Int<LIMBS2>> for Int<LIMBS> {
    fn mul_by_scalar(&self, rhs: &Int<LIMBS2>) -> Option<Self> {
        if LIMBS < LIMBS2 {
            return None; // Cannot multiply if the left operand has fewer limbs than the right
        }
        self.checked_mul(&rhs.resize())
    }
}

macro_rules! impl_mul_int_by_primitive_scalar {
    ($($t:ty),*) => {
        $(
            impl<const LIMBS: usize> MulByScalar<&$t> for Int<LIMBS> {
                fn mul_by_scalar(&self, rhs: &$t) -> Option<Self> {
                    self.checked_mul(&Self::from_ref(rhs))
                }
            }
        )*
    };
}

impl_mul_int_by_primitive_scalar!(i8, i16, i32, i64, i128);

/// Trait for preparing a projection function to a field element from a current
/// type.
pub trait ProjectableToField<F: PrimeField> {
    /// Prepare a projection function that will project the current type
    /// to a prime field using the given sampled value.
    fn prepare_projection(sampled_value: &F) -> impl Fn(&Self) -> F + 'static;
}
