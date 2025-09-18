use ark_std::{cfg_iter_mut, iterable::Iterable};
use crypto_bigint::Word;
use crypto_primitives::{PrimeField, crypto_bigint_int::Int};
use itertools::Itertools;
use num_traits::{ToBytes, Zero};
use p3_commit::{BatchOpeningRef, Mmcs};
use p3_field::Packable;
use p3_matrix::{Dimensions, Matrix as P3Matrix, dense::RowMajorMatrix};
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{CryptographicHasher, PseudoCompressionFunction};
use std::{
    fmt,
    fmt::{Display, Formatter},
    io::Write,
};
use uninit::AsMaybeUninit;

use super::{error::MerkleError, structs::MultilinearZipData};
use crate::{
    Error, pcs_transcript::PcsTranscript, poly::dense::DenseMultilinearExtension,
    utils::ReinterpretVector,
};

fn err_too_many_variates(function: &str, upto: usize, got: usize) -> Error {
    Error::InvalidPcsParam(format!(
        "Too many variates of poly to {function} (param supports variates up to {upto} but got {got})"
    ))
}

// Ensures that polynomials and evaluation points are of appropriate size
pub(super) fn validate_input<'a, T: 'a, F: 'a>(
    function: &str,
    param_num_vars: usize,
    polys: impl Iterable<Item = &'a DenseMultilinearExtension<T>>,
    points: impl Iterable<Item = &'a [F]>,
) -> Result<(), Error> {
    // Ensure all the number of variables in the polynomials don't exceed the limit
    for poly in polys.iter() {
        if param_num_vars < poly.num_vars {
            return Err(err_too_many_variates(
                function,
                param_num_vars,
                poly.num_vars,
            ));
        }
    }

    // Ensure all the points are of correct length
    let input_num_vars = polys
        .iter()
        .map(|poly| poly.num_vars)
        .chain(points.iter().map(|point| point.len()))
        .next()
        .expect("To have at least 1 poly or point");

    for point in points.iter() {
        if point.len() != input_num_vars {
            return Err(Error::InvalidPcsParam(format!(
                "Invalid point (expect point to have {input_num_vars} variates but got {})",
                point.len()
            )));
        }
    }
    Ok(())
}

pub trait AsWords {
    /// View the underlying byte array as a slice of `Word`s.
    fn as_words(&self) -> &[Word];
}

/// Cannot reference blake3::OUT_LEN directly in some of the contexts below.
const BLAKE3_OUT_LEN: usize = blake3::OUT_LEN;

#[derive(Clone, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct MtHash(pub(crate) [u8; BLAKE3_OUT_LEN]);

impl Default for MtHash {
    fn default() -> Self {
        MtHash([0; BLAKE3_OUT_LEN])
    }
}

impl Display for MtHash {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let blake3_hash: blake3::Hash = self.0.into();
        <blake3::Hash as Display>::fmt(&blake3_hash, f)
    }
}

#[derive(Debug, Default, Clone)]
pub struct MtHasher;

impl<T: AsWords + Clone> CryptographicHasher<T, [u8; BLAKE3_OUT_LEN]> for MtHasher {
    fn hash_iter<I>(&self, input: I) -> [u8; BLAKE3_OUT_LEN]
    where
        I: IntoIterator<Item = T>,
    {
        let mut hasher = blake3::Hasher::new();
        let mut buf = [0_u8; size_of::<Word>()];
        for item in input {
            for word in item.as_words() {
                // Performance: reuse buffer and help compiler optimize away materializing word
                // bytes
                buf.copy_from_slice(&word.to_be_bytes());
                hasher.write_all(&buf).expect("Failed to write to hasher");
            }
        }
        hasher.finalize().into()
    }
}

#[derive(Debug, Default, Clone)]
pub struct MtPerm;

impl PseudoCompressionFunction<[u8; BLAKE3_OUT_LEN], 2> for MtPerm {
    fn compress(&self, input: [[u8; BLAKE3_OUT_LEN]; 2]) -> [u8; BLAKE3_OUT_LEN] {
        let mut hasher = blake3::Hasher::new();
        for ref item in input {
            hasher.write_all(item).expect("Failed to write to hasher");
        }
        hasher.finalize().into()
    }
}

type Matrix<T> = RowMajorMatrix<T>;
type MtMmcs<T> = MerkleTreeMmcs<T, u8, MtHasher, MtPerm, BLAKE3_OUT_LEN>;
type P3MerkleTree<T> = p3_merkle_tree::MerkleTree<T, u8, Matrix<T>, BLAKE3_OUT_LEN>;

#[derive(Debug, Default)]
pub struct MerkleTree<T>
where
    T: Packable + AsWords + Clone + Send + Sync,
{
    inner: Option<MerkleTreeInner<T>>,
}

#[derive(Debug)]
struct MerkleTreeInner<T> {
    prover_data: P3MerkleTree<T>,
    matrix_dims: Dimensions,
}

impl<T> MerkleTree<T>
where
    T: Packable + AsWords + Clone + Send + Sync,
{
    pub fn new<S>(rows: &[S], row_width: usize) -> Self
    where
        S: ReinterpretVector<T>,
        T: ReinterpretVector<S>,
    {
        assert!(rows.len().is_power_of_two());
        assert!(rows.len().is_multiple_of(row_width));
        assert!(row_width > 0);

        // Each matrix row is hashed together to form a leaf in the Merkle tree.
        // Thus, we need to transpose a matrix to have original columns as leaves.
        let matrix = {
            let mut columns: Vec<T> = Vec::with_capacity(rows.len());
            let column_height = rows.len() / row_width;
            let rows = unsafe { ReinterpretVector::reinterpret_slice(rows) };
            transpose::transpose(
                rows.as_ref_uninit(),
                columns.spare_capacity_mut(),
                row_width,
                column_height,
            );
            // Safe because we just initialized all elements of `columns`, and
            // MaybeUninit<T> is #[repr(transparent)].
            unsafe {
                columns.set_len(rows.len());
            }
            Matrix::new(columns, column_height)
        };

        let matrix_dims = matrix.dimensions();
        let prover_data = P3MerkleTree::new::<T, _, _, _>(&MtHasher, &MtPerm, vec![matrix]);

        Self {
            inner: Some(MerkleTreeInner {
                prover_data,
                matrix_dims,
            }),
        }
    }

    pub fn root(&self) -> MtHash {
        MtHash(
            *self
                .inner
                .as_ref()
                .expect("Merkle tree not initialized")
                .prover_data
                .root()
                .as_ref(),
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MerkleProof {
    pub path: Vec<MtHash>,
    pub matrix_dims: Dimensions,
}

impl MerkleProof {
    pub fn new(path: Vec<MtHash>, matrix_dims: Dimensions) -> Self {
        MerkleProof { path, matrix_dims }
    }

    pub fn create_proof<T>(merkle_tree: &MerkleTree<T>, leaf: usize) -> Result<Self, MerkleError>
    where
        T: Packable + AsWords + Clone,
    {
        let mt = merkle_tree
            .inner
            .as_ref()
            .ok_or(MerkleError::InvalidRootHash)?;
        let prover = MtMmcs::<T>::new(MtHasher, MtPerm);
        let opening = prover.open_batch(leaf, &mt.prover_data);
        let path = opening.opening_proof.into_iter().map(MtHash).collect();
        Ok(Self::new(path, mt.matrix_dims))
    }

    pub fn verify<S, T>(
        &self,
        root: &MtHash,
        leaf_values: Vec<S>,
        leaf_index: usize,
    ) -> Result<(), MerkleError>
    where
        S: ReinterpretVector<T>,
        T: Packable + AsWords + Clone,
    {
        let prover = MtMmcs::<T>::new(MtHasher, MtPerm);

        let leaf_values = unsafe { ReinterpretVector::reinterpret_vector(leaf_values) };

        let values = vec![leaf_values];
        let proof = self.path.iter().map(|h| h.0).collect_vec();
        let proof = BatchOpeningRef::new(&values, &proof);
        prover
            .verify_batch(&root.0.into(), &[self.matrix_dims], leaf_index, proof)
            .map_err(|e| {
                MerkleError::InvalidMerkleProof(format!("Failed to validate Merkle proof: {:?}", e))
            })
    }
}

impl Display for MerkleProof {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f, "Merkle Path: {}", self.path.iter().join(", "))?;
        writeln!(f, "Matrix Dimensions: {}", self.matrix_dims)?;
        Ok(())
    }
}

/// This is a helper struct to open a column in a multilinear polynomial
/// Opening a column `j` in an `n x m` matrix `u_hat` requires opening `m`
/// Merkle trees, one for each row at position j
/// Note that the proof is written to the transcript and the order of the proofs
/// is the same as the order of the columns
#[derive(Clone)]
pub struct ColumnOpening {}

impl ColumnOpening {
    pub fn open_at_column<const K: usize>(
        column: usize,
        commit_data: &MultilinearZipData<K>,
        transcript: &mut PcsTranscript,
    ) -> Result<(), MerkleError> {
        let merkle_path = MerkleProof::create_proof(&commit_data.merkle_tree, column)?;
        transcript
            .write_merkle_proof(&merkle_path)
            .map_err(|_| MerkleError::FailedMerkleProofWriting)?;
        Ok(())
    }

    pub fn verify_column<const N: usize>(
        root: &MtHash,
        column: &[Int<N>],
        column_index: usize,
        transcript: &mut PcsTranscript,
    ) -> Result<(), MerkleError> {
        let proof = transcript
            .read_merkle_proof()
            .map_err(|_| MerkleError::FailedMerkleProofReading)?;
        proof.verify(root, column.to_vec(), column_index)?;
        Ok(())
    }
}

/// For a polynomial arranged in matrix form, this splits the evaluation point
/// into two vectors, `q_0` multiplying on the left and `q_1` multiplying on the
/// right
pub(super) fn point_to_tensor<F>(num_rows: usize, point: &[F]) -> Result<(Vec<F>, Vec<F>), Error>
where
    F: PrimeField,
{
    assert!(num_rows.is_power_of_two());
    let (hi, lo) = point.split_at(point.len() - num_rows.ilog2() as usize);
    // TODO: get rid of these unwraps.
    let q_0 = if !lo.is_empty() {
        build_eq_x_r(lo).unwrap()
    } else {
        DenseMultilinearExtension::zero()
    };

    let q_1 = if !hi.is_empty() {
        build_eq_x_r(hi).unwrap()
    } else {
        DenseMultilinearExtension::zero()
    };

    Ok((q_0.evaluations, q_1.evaluations))
}

/// This function build the eq(x, r) polynomial for any given r.
///
/// Evaluate
///      eq(x,y) = \prod_i=1^num_var (x_i * y_i + (1-x_i)*(1-y_i))
/// over r, which is
///      eq(x,y) = \prod_i=1^num_var (x_i * r_i + (1-x_i)*(1-r_i))
pub fn build_eq_x_r<F>(r: &[F]) -> Result<DenseMultilinearExtension<F>, ArithErrors>
where
    F: PrimeField,
{
    let evals = build_eq_x_r_vec(r)?;
    let mle = DenseMultilinearExtension::from_evaluations_vec(r.len(), evals);

    Ok(mle)
}

/// This function build the eq(x, r) polynomial for any given r, and output the
/// evaluation of eq(x, r) in its vector form.
///
/// Evaluate
///      eq(x,y) = \prod_i=1^num_var (x_i * y_i + (1-x_i)*(1-y_i))
/// over r, which is
///      eq(x,y) = \prod_i=1^num_var (x_i * r_i + (1-x_i)*(1-r_i))
pub fn build_eq_x_r_vec<F>(r: &[F]) -> Result<Vec<F>, ArithErrors>
where
    F: PrimeField,
{
    // we build eq(x,r) from its evaluations
    // we want to evaluate eq(x,r) over x \in {0, 1}^num_vars
    // for example, with num_vars = 4, x is a binary vector of 4, then
    //  0 0 0 0 -> (1-r0)   * (1-r1)    * (1-r2)    * (1-r3)
    //  1 0 0 0 -> r0       * (1-r1)    * (1-r2)    * (1-r3)
    //  0 1 0 0 -> (1-r0)   * r1        * (1-r2)    * (1-r3)
    //  1 1 0 0 -> r0       * r1        * (1-r2)    * (1-r3)
    //  ....
    //  1 1 1 1 -> r0       * r1        * r2        * r3
    // we will need 2^num_var evaluations

    let mut eval = Vec::new();
    build_eq_x_r_helper(r, &mut eval)?;

    Ok(eval)
}

/// A helper function to build eq(x, r) recursively.
/// This function takes `r.len()` steps, and for each step it requires a maximum
/// `r.len()-1` multiplications.
fn build_eq_x_r_helper<F>(r: &[F], buf: &mut Vec<F>) -> Result<(), ArithErrors>
where
    F: PrimeField,
{
    if r.is_empty() {
        return Err(ArithErrors::InvalidParameters("r length is 0".into()));
    } else if r.len() == 1 {
        // initializing the buffer with [1-r_0, r_0]
        buf.push(F::one() - r[0].clone());
        buf.push(r[0].clone());
    } else {
        build_eq_x_r_helper(&r[1..], buf)?;

        // suppose at the previous step we received [b_1, ..., b_k]
        // for the current step we will need
        // if x_0 = 0:   (1-r0) * [b_1, ..., b_k]
        // if x_0 = 1:   r0 * [b_1, ..., b_k]
        // let mut res = vec![];
        // for &b_i in buf.iter() {
        //     let tmp = r[0] * b_i;
        //     res.push(b_i - tmp);
        //     res.push(tmp);
        // }
        // *buf = res;

        let mut res = vec![F::zero(); buf.len() << 1];
        cfg_iter_mut!(res).enumerate().for_each(|(i, val)| {
            let bi = buf[i >> 1].clone();
            let tmp = r[0].clone() * bi.clone();
            if (i & 1) == 0 {
                *val = bi - tmp;
            } else {
                *val = tmp;
            }
        });
        *buf = res;
    }

    Ok(())
}

/// For a polynomial arranged in matrix form, this splits the evaluation point
/// into two vectors, `q_0` multiplying on the left and `q_1` multiplying on the
/// right and returns the left vector only
pub(super) fn left_point_to_tensor<F>(num_rows: usize, point: &[F]) -> Result<Vec<F>, Error>
where
    F: PrimeField,
{
    let (_, lo) = point.split_at(point.len() - num_rows.ilog2() as usize);
    // TODO: get rid of these unwraps.
    let q_0 = if !lo.is_empty() {
        build_eq_x_r(lo).unwrap()
    } else {
        DenseMultilinearExtension::zero()
    };
    Ok(q_0.evaluations)
}

/// A `enum` specifying the possible failure modes of the arithmetics.
#[derive(displaydoc::Display, Debug, Error)]
pub enum ArithErrors {
    /// Invalid parameters: {0}
    InvalidParameters(String),
}

#[cfg(test)]
mod tests {
    use crypto_bigint::Random;
    use crypto_primitives::crypto_bigint_int::Int;
    use rand::rng;

    use crate::{
        pcs::{MerkleTree, utils::MerkleProof},
        utils::combine_rows,
    };

    #[test]
    fn test_basic_combination() {
        let coeffs = vec![1, 2];
        let evaluations = vec![3, 4, 5, 6];
        let row_len = 2;

        let result = combine_rows(coeffs, evaluations, row_len);

        assert_eq!(result, vec![(3 + 2 * 5), (4 + 2 * 6)]);
    }

    #[test]
    fn test_second_combination() {
        let coeffs = vec![3, 4];
        let evaluations = vec![2, 4, 6, 8];
        let row_len = 2;

        let result = combine_rows(coeffs, evaluations, row_len);

        assert_eq!(result, vec![(3 * 2 + 4 * 6), (3 * 4 + 4 * 8)]);
    }
    #[test]
    fn test_large_values() {
        let coeffs = vec![1000, -500];
        let evaluations = vec![2000, -3000, 4000, -5000];
        let row_len = 2;

        let result = combine_rows(coeffs, evaluations, row_len);

        assert_eq!(
            result,
            vec![
                (1000 * 2000 + (-500) * 4000),
                (1000 * -3000 + (-500) * -5000)
            ]
        );
    }

    #[test]
    fn test_merkle_proof() {
        const N: usize = 3;
        let leaves_len = 1024;
        let mut rng = rng();
        let leaves_data = (0..leaves_len)
            .map(|_| Int::random(&mut rng))
            .collect::<Vec<Int<N>>>();

        let merkle_tree = MerkleTree::new(&leaves_data, leaves_data.len());

        // Print tree structure after merklizing
        let root = merkle_tree.root();
        // Create a proof for the first leaf
        for (i, leaf) in leaves_data.iter().enumerate() {
            let proof =
                MerkleProof::create_proof(&merkle_tree, i).expect("Merkle proof creation failed");

            // Verify the proof
            proof
                .verify(&root, vec![*leaf], i)
                .expect("Merkle proof verification failed");
        }
    }
}
