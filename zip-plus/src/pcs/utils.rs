use ark_ff::Zero;
use ark_std::{
    cfg_iter_mut, format,
    iterable::Iterable,
    vec,
    vec::Vec,
};
use displaydoc::Display;
use sha3::{digest::Output, Digest, Keccak256};

use super::{error::MerkleError, structs::MultilinearZipData};
use crate::{
    poly_f::mle::DenseMultilinearExtension as MLE_F,
    poly_z::mle::DenseMultilinearExtension as MLE_Z,
    traits::{Field, Integer},

    pcs_transcript::PcsTranscript,
    utils::{div_ceil, num_threads, parallelize, parallelize_iter},
    Error,
};

fn err_too_many_variates(function: &str, upto: usize, got: usize) -> Error {
    Error::InvalidPcsParam(
        format!(
            "Too many variates of poly to {function} (param supports variates up to {upto} but got {got})"
        )
    )
}

// Ensures that polynomials and evaluation points are of appropriate size
pub(super) fn validate_input<'a, I: Integer + 'a, F: Field + 'a>(
    function: &str,
    param_num_vars: usize,
    polys: impl Iterable<Item = &'a MLE_Z<I>>,
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

// Define a new trait for converting to bytes
pub trait ToBytes {
    fn to_bytes(&self) -> Vec<u8>;
}

/// A merkle tree in which its layers are concatenated together in a single vector
#[derive(Clone, Debug, Default)]
pub struct MerkleTree {
    pub root: Output<Keccak256>,
    pub depth: usize,
    pub layers: Vec<Output<Keccak256>>,
}

impl MerkleTree {
    pub fn new<T: ToBytes + Send + Sync>(depth: usize, leaves: &[T]) -> Self {
        assert!(leaves.len().is_power_of_two());
        assert_eq!(leaves.len(), 1 << depth);
        let mut layers = vec![Output::<Keccak256>::default(); (2 << depth) - 1];
        Self::compute_leaves_hashes(&mut layers[..leaves.len()], leaves);
        Self::merklize_leaves_hashes(depth, &mut layers);
        Self {
            root: layers.pop().unwrap(),
            depth,
            layers,
        }
    }

    fn compute_leaves_hashes<T: ToBytes + Send + Sync>(
        hashes: &mut [Output<Keccak256>],
        leaves: &[T],
    ) {
        parallelize(hashes, |(hashes, start)| {
            let mut hasher = Keccak256::new();
            for (hash, row) in hashes.iter_mut().zip(start..) {
                let bytes = leaves[row].to_bytes();
                <Keccak256 as sha3::digest::Update>::update(&mut hasher, &bytes);
                hasher.finalize_into_reset(hash);
            }
        });
    }

    fn merklize_leaves_hashes(depth: usize, hashes: &mut [Output<Keccak256>]) {
        assert_eq!(hashes.len(), (2 << depth) - 1);
        let mut offset = 0;
        for width in (1..=depth).rev().map(|depth| 1 << depth) {
            let (current_layer, next_layer) = hashes[offset..].split_at_mut(width);

            let chunk_size = div_ceil(next_layer.len(), num_threads());
            parallelize_iter(
                current_layer
                    .chunks(2 * chunk_size)
                    .zip(next_layer.chunks_mut(chunk_size)),
                |(input, output)| {
                    let mut hasher = Keccak256::new();
                    for (input, output) in input.chunks_exact(2).zip(output.iter_mut()) {
                        hasher.update(input[0]);
                        hasher.update(input[1]);
                        hasher.finalize_into_reset(output);
                    }
                },
            );
            offset += width;
        }
    }
}

#[derive(Clone, Debug)]
pub struct MerkleProof {
    pub merkle_path: Vec<Output<Keccak256>>,
}

impl ark_std::fmt::Display for MerkleProof {
    fn fmt(&self, f: &mut ark_std::fmt::Formatter<'_>) -> ark_std::fmt::Result {
        writeln!(f, "Merkle Path:")?;
        for (i, hash) in self.merkle_path.iter().enumerate() {
            writeln!(f, "Level {i}: {hash:?}")?;
        }
        Ok(())
    }
}

impl Default for MerkleProof {
    fn default() -> Self {
        Self::new()
    }
}

impl MerkleProof {
    pub fn new() -> Self {
        Self {
            merkle_path: vec![],
        }
    }

    pub fn from_vec(vec: Vec<Output<Keccak256>>) -> Self {
        Self { merkle_path: vec }
    }

    pub fn create_proof(merkle_tree: &MerkleTree, leaf: usize) -> Result<Self, MerkleError> {
        let mut offset = 0;
        let path: Vec<Output<Keccak256>> = (1..=merkle_tree.depth)
            .rev()
            .map(|depth| {
                let width = 1 << depth;
                let idx = (leaf >> (merkle_tree.depth - depth)) ^ 1;
                let hash = merkle_tree.layers[offset + idx];
                offset += width;
                hash
            })
            .collect();
        Ok(MerkleProof::from_vec(path))
    }

    pub fn verify<T: ToBytes>(
        &self,
        root: Output<Keccak256>,
        leaf_value: &T,
        leaf_index: usize,
    ) -> Result<(), MerkleError> {
        let mut hasher = Keccak256::new();
        let bytes = leaf_value.to_bytes();
        hasher.update(&bytes);
        let mut current = hasher.finalize_reset();

        let mut index = leaf_index;
        for path_hash in &self.merkle_path {
            if (index & 1) == 0 {
                <Keccak256 as sha3::digest::Update>::update(&mut hasher, &current);
                <Keccak256 as sha3::digest::Update>::update(&mut hasher, path_hash);
            } else {
                <Keccak256 as sha3::digest::Update>::update(&mut hasher, path_hash);
                <Keccak256 as sha3::digest::Update>::update(&mut hasher, &current);
            }

            hasher.finalize_into_reset(&mut current);
            index /= 2;
        }
        if current != root {
            return Err(MerkleError::InvalidMerkleProof(
                "Merkle proof verification failed".into(),
            ));
        }
        Ok(())
    }
}

/// This is a helper struct to open a column in a multilinear polynomial
/// Opening a column `j` in an `n x m` matrix `u_hat` requires opening `m` Merkle trees,
/// one for each row at position j
/// Note that the proof is written to the transcript and the order of the proofs is the same as the order of the columns
#[derive(Clone)]
pub struct ColumnOpening {}

impl ColumnOpening {
    pub fn open_at_column<F: Field, M: Integer>(
        column: usize,
        commit_data: &MultilinearZipData<M>,
        transcript: &mut PcsTranscript<F>,
    ) -> Result<(), MerkleError> {
        for row_merkle_tree in commit_data.rows_merkle_trees.iter() {
            let merkle_path = MerkleProof::create_proof(row_merkle_tree, column)?;
            transcript
                .write_merkle_proof(&merkle_path)
                .map_err(|_| MerkleError::FailedMerkleProofWriting)?;
        }
        Ok(())
    }

    pub fn verify_column<F: Field, T: ToBytes>(
        rows_roots: &[Output<Keccak256>],
        column: &[T],
        column_index: usize,
        transcript: &mut PcsTranscript<F>,
    ) -> Result<(), MerkleError> {
        for (root, leaf) in rows_roots.iter().zip(column) {
            let proof = transcript
                .read_merkle_proof()
                .map_err(|_| MerkleError::FailedMerkleProofReading)?;
            proof.verify(*root, leaf, column_index)?;
        }
        Ok(())
    }
}

/// For a polynomial arranged in matrix form, this splits the evaluation point into
/// two vectors, `q_0` multiplying on the left and `q_1` multiplying on the right
pub(super) fn point_to_tensor<F: Field>(
    num_rows: usize,
    point: &[F],
) -> Result<(Vec<F>, Vec<F>), Error> {
    assert!(num_rows.is_power_of_two());
    let (hi, lo) = point.split_at(point.len() - num_rows.ilog2() as usize);
    // TODO: get rid of these unwraps.
    let q_0 = if !lo.is_empty() {
        build_eq_x_r_f(lo).unwrap()
    } else {
        MLE_F::zero()
    };

    let q_1 = if !hi.is_empty() {
        build_eq_x_r_f(hi).unwrap()
    } else {
        MLE_F::zero()
    };

    Ok((q_0.evaluations, q_1.evaluations))
}

/// This function build the eq(x, r) polynomial for any given r.
///
/// Evaluate
///      eq(x,y) = \prod_i=1^num_var (x_i * y_i + (1-x_i)*(1-y_i))
/// over r, which is
///      eq(x,y) = \prod_i=1^num_var (x_i * r_i + (1-x_i)*(1-r_i))
pub fn build_eq_x_r_f<F: Field>(
    r: &[F],
) -> Result<MLE_F<F>, ArithErrors> {
    let evals = build_eq_x_r_vec(r)?;
    let mle = MLE_F::from_evaluations_vec(r.len(), evals);

    Ok(mle)
}

/// This function build the eq(x, r) polynomial for any given r, and output the
/// evaluation of eq(x, r) in its vector form.
///
/// Evaluate
///      eq(x,y) = \prod_i=1^num_var (x_i * y_i + (1-x_i)*(1-y_i))
/// over r, which is
///      eq(x,y) = \prod_i=1^num_var (x_i * r_i + (1-x_i)*(1-r_i))
pub fn build_eq_x_r_vec<F: Field>(r: &[F]) -> Result<Vec<F>, ArithErrors> {
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
fn build_eq_x_r_helper<F: Field>(r: &[F], buf: &mut Vec<F>) -> Result<(), ArithErrors> {
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

/// For a polynomial arranged in matrix form, this splits the evaluation point into
/// two vectors, `q_0` multiplying on the left and `q_1` multiplying on the right
/// and returns the left vector only
pub(super) fn left_point_to_tensor<F: Field>(
    num_rows: usize,
    point: &[F],
) -> Result<Vec<F>, Error> {
    let (_, lo) = point.split_at(point.len() - num_rows.ilog2() as usize);
    // TODO: get rid of these unwraps.
    let q_0 = if !lo.is_empty() {
        build_eq_x_r_f(lo).unwrap()
    } else {
        MLE_F::<F>::zero()
    };
    Ok(q_0.evaluations)
}

/// A `enum` specifying the possible failure modes of the arithmetics.
#[derive(Display, Debug, Error)]
pub enum ArithErrors {
    /// Invalid parameters: {0}
    InvalidParameters(String),
    /// Should not arrive to this point
    ShouldNotArrive,
    /// An error during (de)serialization: {0}
    SerializationErrors(ark_serialize::SerializationError),
}

#[cfg(test)]
mod tests {
    use crypto_bigint::Random;

    use super::*;
    use crate::{field::Int, utils::combine_rows};

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
        let mut rng = ark_std::test_rng();
        let leaves_data = (0..leaves_len)
            .map(|_| Int::random(&mut rng))
            .collect::<Vec<Int<N>>>();

        let merkle_depth = leaves_data.len().next_power_of_two().ilog2() as usize;
        let merkle_tree = MerkleTree::new(merkle_depth, &leaves_data);

        // Print tree structure after merklizing
        let root = merkle_tree.root;
        // Create a proof for the first leaf
        for (i, leaf) in leaves_data.iter().enumerate() {
            let proof =
                MerkleProof::create_proof(&merkle_tree, i).expect("Merkle proof creation failed");

            // Verify the proof
            proof
                .verify(root, leaf, i)
                .expect("Merkle proof verification failed");
        }
    }
}
