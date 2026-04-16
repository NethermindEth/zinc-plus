use crate::{
    code::LinearCode,
    merkle::{MerkleTree, MtHash},
};
use crypto_primitives::{ConstIntRing, ConstIntSemiring, DenseRowMatrix, FixedSemiring};
use num_traits::CheckedAdd;
use std::{fmt::Debug, marker::PhantomData, ops::Neg};
use zinc_poly::{ConstCoeffBitWidth, Polynomial};
use zinc_primality::PrimalityTest;
use zinc_transcript::traits::{ConstTranscribable, GenTranscribable};
use zinc_utils::{
    from_ref::FromRef, inner_product::InnerProduct, mul_by_scalar::MulByScalar, named::Named,
};

pub trait ZipTypes: Clone + Debug + Send + Sync {
    const NUM_COLUMN_OPENINGS: usize;

    /// Semiring of witness/polynomial evaluations on boolean hypercube
    type Eval: ConstCoeffBitWidth + Default + Named + Clone + Debug + Send + Sync;

    /// Semiring of codeword elements, at least as wide as the evaluation ring
    type Cw: FixedSemiring
        + ConstCoeffBitWidth
        + ConstTranscribable
        + FromRef<Self::Eval>
        + CheckedAdd
        + Named
        // TODO(Ilia): Find out if the Copy can be avoided.
        + Copy
        + Debug;

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
        + FromRef<Self::Eval>
        + FromRef<Self::Cw>
        + Named;

    type EvalDotChal: InnerProduct<Self::Eval, Self::Chal, Self::CombR> + Debug;
    type CombDotChal: InnerProduct<Self::Comb, Self::Chal, Self::CombR> + Debug;
    type ArrCombRDotChal: InnerProduct<[Self::CombR], Self::Chal, Self::CombR> + Debug;
}

/// Zip is a Polynomial Commitment Scheme (PCS) that supports committing to
/// multilinear polynomials.
// Note(alex): We cannot define CHECK_FOR_OVERFLOW in ZipTypes because type
// parameters may not be used in const expressions
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
        let row_len = linear_code.row_len();
        assert!(
            row_len > 0 && poly_size.is_multiple_of(row_len),
            "poly_size ({poly_size}) must be divisible by row_len ({row_len})"
        );
        let num_rows = poly_size / row_len;
        assert!(
            num_rows.is_power_of_two(),
            "num_rows ({num_rows}) must be a power of two"
        );
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
#[derive(Debug, Clone, Default)]
pub struct ZipPlusHint<R> {
    /// The encoded rows of the polynomial matrix representation, referred to as
    /// "u-hat" in the Zinc paper
    pub cw_matrices: Vec<DenseRowMatrix<R>>,
    /// Merkle trees of entire matrix
    pub merkle_tree: MerkleTree,
}

impl<R> ZipPlusHint<R> {
    pub fn new(cw_matrices: Vec<DenseRowMatrix<R>>, merkle_tree: MerkleTree) -> ZipPlusHint<R> {
        ZipPlusHint {
            cw_matrices,
            merkle_tree,
        }
    }
}

/// The compact commitment to a multilinear polynomial, consisting of only the
/// Merkle roots, to be sent to the verifier.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ZipPlusCommitment {
    /// Roots of the merkle tree of entire matrix
    pub root: MtHash,
    pub batch_size: usize,
}

impl GenTranscribable for ZipPlusCommitment {
    fn read_transcription_bytes_exact(bytes: &[u8]) -> Self {
        let root = MtHash::read_transcription_bytes_exact(&bytes[..MtHash::NUM_BYTES]);
        let batch_size = u64::read_transcription_bytes_exact(&bytes[MtHash::NUM_BYTES..]);
        let batch_size = usize::try_from(batch_size).expect("num_bytes must fit into usize");
        Self { root, batch_size }
    }

    fn write_transcription_bytes_exact(&self, buf: &mut [u8]) {
        let (root_buf, rest) = buf.split_at_mut(MtHash::NUM_BYTES);
        self.root.write_transcription_bytes_exact(root_buf);
        (self.batch_size as u64).write_transcription_bytes_exact(rest);
    }
}

impl ConstTranscribable for ZipPlusCommitment {
    const NUM_BYTES: usize = MtHash::NUM_BYTES + u64::NUM_BYTES;
}
