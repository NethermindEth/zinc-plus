use ark_std::{cfg_iter_mut, iterable::Iterable};
use crypto_primitives::PrimeField;
use num_traits::Zero;
use thiserror::Error;

use crate::{ZipError, poly::mle::DenseMultilinearExtension, sub};

use crate::{
    merkle::{MerkleError, MerkleProof, MtHash},
    pcs::structs::{AsPackable, ZipPlusHint},
    pcs_transcript::PcsTranscript,
};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

fn err_too_many_variates(function: &str, upto: usize, got: usize) -> ZipError {
    ZipError::InvalidPcsParam(format!(
        "Too many variates of poly to {function} (param supports variates up to {upto} but got {got})"
    ))
}

// Ensures that polynomials and evaluation points are of appropriate size
pub(super) fn validate_input<'a, T: 'a, F: 'a>(
    function: &str,
    param_num_vars: usize,
    polys: impl Iterable<Item = &'a DenseMultilinearExtension<T>>,
    points: impl Iterable<Item = &'a [F]>,
) -> Result<(), ZipError> {
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
            return Err(ZipError::InvalidPcsParam(format!(
                "Invalid point (expect point to have {input_num_vars} variates but got {})",
                point.len()
            )));
        }
    }
    Ok(())
}

/// For a polynomial arranged in matrix form, this splits the evaluation point
/// into two vectors, `q_0` multiplying on the left and `q_1` multiplying on the
/// right
#[allow(clippy::unwrap_used)]
pub(super) fn point_to_tensor<F>(num_rows: usize, point: &[F]) -> Result<(Vec<F>, Vec<F>), ZipError>
where
    F: PrimeField,
{
    assert!(num_rows.is_power_of_two());
    let (hi, lo) = point.split_at(sub!(point.len(), num_rows.ilog2() as usize));
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
        buf.push(F::one() - &r[0]);
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
            let tmp = r[0].clone() * &bi;
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
#[allow(clippy::unwrap_used)]
pub(super) fn left_point_to_tensor<F>(num_rows: usize, point: &[F]) -> Result<Vec<F>, ZipError>
where
    F: PrimeField,
{
    let (_, lo) = point.split_at(sub!(point.len(), num_rows.ilog2() as usize));
    // TODO: get rid of these unwraps.
    let q_0 = if !lo.is_empty() {
        build_eq_x_r(lo).unwrap()
    } else {
        DenseMultilinearExtension::zero()
    };
    Ok(q_0.evaluations)
}

/// This is a helper struct to open a column in a multilinear polynomial
/// Opening a column `j` in an `n x m` matrix `u_hat` requires opening `m`
/// Merkle trees, one for each row at position j
/// Note that the proof is written to the transcript and the order of the proofs
/// is the same as the order of the columns
#[derive(Clone)]
pub struct ColumnOpening {}

impl ColumnOpening {
    pub fn open_at_column<T: AsPackable>(
        column: usize,
        commit_hint: &ZipPlusHint<T>,
        transcript: &mut PcsTranscript,
    ) -> Result<(), MerkleError> {
        let merkle_path = MerkleProof::create_proof(&commit_hint.merkle_tree, column)?;
        transcript
            .write_merkle_proof(&merkle_path)
            .map_err(|_| MerkleError::FailedMerkleProofWriting)?;
        Ok(())
    }

    pub fn verify_column<T: AsPackable>(
        root: &MtHash,
        column: &[T],
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

/// A `enum` specifying the possible failure modes of the arithmetics.
#[derive(displaydoc::Display, Debug, Error)]
pub enum ArithErrors {
    /// Invalid parameters: {0}
    InvalidParameters(String),
}
