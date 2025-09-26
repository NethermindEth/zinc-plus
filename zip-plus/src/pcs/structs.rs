use crate::{
    code::LinearCode,
    merkle::{MerkleTree, MtHash},
    poly::Polynomial,
    traits::{FromRef, Transcribable},
    utils::ReinterpretVector,
};
use crypto_primitives::{Ring, crypto_bigint_int::Int};
use num_traits::CheckedMul;
use p3_field::Packable;
use std::marker::PhantomData;

pub trait ZipTypes: Send + Sync {
    /// Coefficient ring of evaluation polynomial [Self::Eval]
    type EvalR: Ring;
    /// Ring of witness/polynomial evaluations on boolean hypercube
    type Eval: Ring + Polynomial<Self::EvalR>;

    /// Coefficient ring of codeword polynomial [Self::Cw]
    type CwR: Ring;
    /// Ring of codeword elements, at least as wide as the evaluation ring
    type Cw: Ring + Polynomial<Self::CwR> + Transcribable + AsPackable + Copy;

    /// Ring of challenge elements (coefficients) to perform a random linear
    /// combination of codewords
    type Chal: Ring + Transcribable;

    /// Coefficient ring of linear combination polynomial [Self::Comb]
    type CombR: Ring;
    /// Ring of elements in the linear combination of codewords, at least as
    /// wide as the evaluation, codeword, and challenge rings.
    type Comb: Ring
        + Polynomial<Self::CombR>
        + Transcribable
        + FromRef<Self::Eval>
        + FromRef<Self::Cw>
        + for<'a> MulByScalar<&'a Self::Chal>;

    /// Linear code used to encode the polynomial matrix representation
    type Code: LinearCode<Self::Eval, Self::Cw, Self::Comb>;
}

/// Zip is a Polynomial Commitment Scheme (PCS) that supports committing to
/// multilinear polynomials.
pub struct ZipPlus<Zt: ZipTypes>(PhantomData<Zt>);

impl<Zt> ZipPlus<Zt>
where
    Zt: ZipTypes,
{
    #[allow(clippy::arithmetic_side_effects)]
    pub fn setup(poly_size: usize, linear_code: Zt::Code) -> MultilinearZipParams<Zt> {
        assert!(poly_size.is_power_of_two());
        let num_vars = poly_size.ilog2() as usize;
        let num_rows = ((1 << num_vars) / linear_code.row_len()).next_power_of_two();
        MultilinearZipParams::new(num_vars, num_rows, linear_code)
    }
}

/// Parameters for the Zip PCS.
#[derive(Clone, Debug)]
pub struct MultilinearZipParams<Zt: ZipTypes> {
    pub num_vars: usize,
    pub num_rows: usize,
    pub linear_code: Zt::Code,
    phantom_data: PhantomData<Zt>,
}

impl<Zt: ZipTypes> MultilinearZipParams<Zt> {
    pub fn new(num_vars: usize, num_rows: usize, linear_code: Zt::Code) -> Self {
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

macro_rules! impl_as_packable_for_primitives {
    ($($source:ty as $packable:ty),+) => {
        $(
            unsafe impl ReinterpretVector<$packable> for $source {}

            impl AsPackable for $source {
                type Packable = $packable;
            }
        )+
    };
}

impl_as_packable_for_primitives!(i8 as u8, i16 as u16, i32 as u32, i64 as u64, i128 as u128);

impl<const LIMBS: usize> AsPackable for Int<LIMBS> {
    type Packable = PackedInt<LIMBS>;
}

#[derive(Copy, Clone, Default, PartialEq, Eq, Debug)]
#[repr(transparent)]
pub struct PackedInt<const LIMBS: usize>(pub(crate) Int<LIMBS>);

unsafe impl<const N: usize> ReinterpretVector<PackedInt<N>> for Int<N> {}

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
