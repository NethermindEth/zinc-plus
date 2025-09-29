use crate::{
    code::LinearCode,
    pcs::{MerkleTree, utils::MtHash},
    traits::Transcribable,
    utils::ReinterpretVector,
};
use crypto_primitives::{Ring, crypto_bigint_int::Int};
use num_traits::CheckedMul;
use p3_field::Packable;
use std::marker::PhantomData;

/// Zip is a Polynomial Commitment Scheme (PCS) that supports committing to
/// multilinear polynomials.
pub struct MultilinearZip<Eval, Cw, Chal, Comb, C>(PhantomData<(Eval, Cw, Chal, Comb, C)>)
where
    Eval: EvaluationRing,
    Cw: CodewordRing,
    Chal: ChallengeRing,
    Comb: LinearCombinationRing<Eval, Cw, Chal>,
    C: LinearCode<Eval, Cw, Comb>;

impl<Eval, Cw, Chal, Comb, C> MultilinearZip<Eval, Cw, Chal, Comb, C>
where
    Eval: EvaluationRing,
    Cw: CodewordRing,
    Chal: ChallengeRing,
    Comb: LinearCombinationRing<Eval, Cw, Chal>,
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
    type Packable: Packable + Transcribable + Clone + Send + Sync;
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

impl<const LIMBS: usize> Transcribable for PackedInt<LIMBS> {
    const NUM_BYTES: usize = Int::<LIMBS>::NUM_BYTES;

    fn read_transcription_bytes(bytes: &[u8]) -> Self {
        Self(Int::read_transcription_bytes(bytes))
    }

    fn write_transcription_bytes(&self, buf: &mut [u8]) {
        self.0.write_transcription_bytes(buf)
    }
}

//
// Trait aliases for various Rings used in the Zip PCS
//

/// Ring of witness/polynomial evaluations on boolean hypercube
pub trait EvaluationRing: Ring {}
impl<Eval> EvaluationRing for Eval where Eval: Ring {}

/// Ring of codeword elements, at least as wide as the evaluation ring
pub trait CodewordRing: Ring + Transcribable + AsPackable + Copy {}
impl<Cw> CodewordRing for Cw where Cw: Ring + Transcribable + AsPackable + Copy {}

/// Ring of challenge elements (coefficients) to perform a random linear
/// combination of codewords
pub trait ChallengeRing: Ring + Transcribable {}
impl<Chal> ChallengeRing for Chal where Chal: Ring + Transcribable {}

/// Ring of elements in the linear combination of codewords, at least as wide as
/// the evaluation, codeword, and challenge rings.
pub trait LinearCombinationRing<Eval, Cw, Chal>:
    Ring + Transcribable + for<'a> From<&'a Eval> + for<'a> From<&'a Cw> + for<'a> MulByScalar<&'a Chal>
{
}
impl<Eval, Cw, Chal, Comb> LinearCombinationRing<Eval, Cw, Chal> for Comb where
    Comb: Ring
        + Transcribable
        + for<'a> From<&'a Eval>
        + for<'a> From<&'a Cw>
        + for<'a> MulByScalar<&'a Chal>
{
}

pub trait MulByScalar<Rhs>: Sized {
    /// Multiplies the current element by a scalar from the right (usually - a
    /// coefficient to obtain a linear combination).
    /// Returns `None` if the multiplication would overflow.
    fn mul_by_scalar(&self, rhs: Rhs) -> Option<Self>;
}

macro_rules! impl_simple_mul_by_scalar {
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

impl_simple_mul_by_scalar!(i8, i16, i32, i64, i128);

impl<const LIMBS: usize, const LIMBS2: usize> MulByScalar<&Int<LIMBS2>> for Int<LIMBS> {
    fn mul_by_scalar(&self, rhs: &Int<LIMBS2>) -> Option<Self> {
        if LIMBS < LIMBS2 {
            return None; // Cannot multiply if the left operand has fewer limbs than the right
        }
        self.checked_mul(&rhs.resize())
    }
}
