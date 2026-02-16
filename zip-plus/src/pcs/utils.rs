use crate::{ZipError, code::LinearCode, pcs::structs::ZipTypes};
use crypto_primitives::PrimeField;
use zinc_poly::{
    ConstCoeffBitWidth, Polynomial, mle::DenseMultilinearExtension, utils::build_eq_x_r,
};
use zinc_transcript::traits::ConstTranscribable;
use zinc_utils::{add, ilog_round_up, sub};

fn err_too_many_variates(function: &str, upto: usize, got: usize) -> ZipError {
    ZipError::InvalidPcsParam(format!(
        "Too many variates of poly to {function} (param supports variates up to {upto} but got {got})"
    ))
}

// Ensures that polynomials and evaluation points are of appropriate size
pub(crate) fn validate_input<Zt: ZipTypes, Lc: LinearCode<Zt>, Pt>(
    function: &str,
    param_num_vars: usize,
    num_rows: usize,
    row_len: usize,
    polys: &[&DenseMultilinearExtension<Zt::Eval>],
    points: &[&[Pt]],
) -> Result<(), ZipError> {
    if num_rows == 0 || !num_rows.is_power_of_two() {
        return Err(ZipError::InvalidPcsParam(format!(
            "Invalid num_rows for {function}: expected a non-zero power of two, got {num_rows}",
        )));
    }

    if row_len == 0 || !row_len.is_power_of_two() {
        return Err(ZipError::InvalidPcsParam(format!(
            "Invalid row_len for {function}: expected a non-zero power of two, got {row_len}",
        )));
    }

    let expected_poly_size = (1usize)
        .checked_shl(param_num_vars as u32)
        .ok_or_else(|| {
            ZipError::InvalidPcsParam(format!(
                "num_vars ({param_num_vars}) is too large to compute polynomial size",
            ))
        })?;

    let matrix_size = num_rows.checked_mul(row_len).ok_or_else(|| {
        ZipError::InvalidPcsParam(format!(
            "num_rows ({num_rows}) * row_len ({row_len}) overflowed usize",
        ))
    })?;

    if matrix_size != expected_poly_size {
        return Err(ZipError::InvalidPcsParam(format!(
            "Invalid matrix dimensions for {function}: num_rows ({num_rows}) * row_len ({row_len}) = {matrix_size}, expected 2^num_vars = {expected_poly_size}",
        )));
    }

    // Check bit-width of the linear combinations
    {
        // Inner ring should be at most 2*log_2(\rho^{-1}*d) + \log_2(d) +
        // challenge_bits + eval_elem_bits - 1, where d is the log-size of a row
        // of the matrix being encoded, i.e. log2(row_len).
        let d = row_len.ilog2() as usize;
        let rep_times_d = Lc::REPETITION_FACTOR.checked_mul(d).ok_or_else(|| {
            ZipError::InvalidPcsParam(format!(
                "Overflow in repetition_factor * d for {function}: {} * {}",
                Lc::REPETITION_FACTOR,
                d
            ))
        })?;
        let codeword_bits = ilog_round_up!(rep_times_d, usize);

        let mut challenge_bits = Zt::Chal::NUM_BITS;
        if Zt::Comb::DEGREE_BOUND > 0 {
            // This means we also draft alphas (multiplied with coeffs), which
            // doubles the number of challenge bits
            challenge_bits = challenge_bits.checked_mul(2).ok_or_else(|| {
                ZipError::InvalidPcsParam(format!(
                    "Overflow in challenge bit-width computation for {function}",
                ))
            })?;
        }

        let max_lc_bits = Zt::CombR::NUM_BITS;
        let two_codeword_bits = codeword_bits.checked_mul(2).ok_or_else(|| {
            ZipError::InvalidPcsParam(format!(
                "Overflow in 2*codeword_bits computation for {function}",
            ))
        })?;
        let d_bits = ilog_round_up!(d, usize);
        let eval_minus_one = Zt::Eval::COEFF_BIT_WIDTH.checked_sub(1).ok_or_else(|| {
            ZipError::InvalidPcsParam(format!(
                "Invalid Eval::COEFF_BIT_WIDTH (< 1) for {function}",
            ))
        })?;

        let actual_lc_bits = two_codeword_bits
            .checked_add(d_bits)
            .and_then(|v| v.checked_add(challenge_bits))
            .and_then(|v| v.checked_add(eval_minus_one))
            .ok_or_else(|| {
                ZipError::InvalidPcsParam(format!(
                    "Overflow while computing linear-combination bit-width for {function}",
                ))
            })?;

        if actual_lc_bits > max_lc_bits {
            return Err(ZipError::InvalidPcsParam(format!(
                "The number of bits used for linear combinations is too large: {actual_lc_bits} bits, max allowed is {max_lc_bits} bits",
            )));
        }
    }

    // Ensure all the number of variables in the polynomials don't exceed the limit
    for poly in polys.iter() {
        if param_num_vars < poly.num_vars {
            return Err(err_too_many_variates(
                function,
                param_num_vars,
                poly.num_vars,
            ));
        }

        if poly.len() != matrix_size {
            return Err(ZipError::InvalidPcsParam(format!(
                "Invalid polynomial size for {function}: expected {matrix_size} evaluations from num_rows ({num_rows}) * row_len ({row_len}), got {}",
                poly.len()
            )));
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
pub(crate) fn point_to_tensor<F>(
    num_rows: usize,
    point: &[F],
    cfg: &F::Config,
) -> Result<(Vec<F>, Vec<F>), ZipError>
where
    F: PrimeField,
{
    assert!(num_rows.is_power_of_two());
    let (hi, lo) = point.split_at(sub!(point.len(), num_rows.ilog2() as usize));
    // TODO: get rid of these unwraps.
    let q_0 = if !lo.is_empty() {
        build_eq_x_r(lo, cfg).unwrap()
    } else {
        DenseMultilinearExtension::zero_vars(F::zero_with_cfg(cfg))
    };

    let q_1 = if !hi.is_empty() {
        build_eq_x_r(hi, cfg).unwrap()
    } else {
        DenseMultilinearExtension::zero_vars(F::zero_with_cfg(cfg))
    };

    Ok((q_0.evaluations, q_1.evaluations))
}
