use crate::{ZipError, code::LinearCode, pcs::structs::ZipTypes};
use crypto_primitives::PrimeField;
use zinc_poly::{
    ConstCoeffBitWidth, Polynomial, mle::DenseMultilinearExtension, utils::build_eq_x_r,
};
use zinc_transcript::traits::ConstTranscribable;
use zinc_utils::{add, ilog_round_up, mul, sub};

fn err_too_many_variates(function: &str, upto: usize, got: usize) -> ZipError {
    ZipError::InvalidPcsParam(format!(
        "Too many variates of poly to {function} (param supports variates up to {upto} but got {got})"
    ))
}

// Ensures that polynomials and evaluation points are of appropriate size
pub(super) fn validate_input<Zt: ZipTypes, Lc: LinearCode<Zt>, Pt>(
    function: &str,
    param_num_vars: usize,
    row_len: usize,
    batch_size: usize,
    polys: &[DenseMultilinearExtension<Zt::Eval>],
    points: &[&[Pt]],
) -> Result<(), ZipError> {
    // Check bit-width of the linear combinations
    {
        // Inner ring should be at most 2*log_2(\rho^{-1}*d) + \log_2(d) +
        // challenge_bits + eval_elem_bits - 1, where d is the log of the size
        // of the messages being encoded (i.e. log2(row_len)).
        let d = row_len.ilog2() as usize;
        let codeword_bits = ilog_round_up!(mul!(Lc::REPETITION_FACTOR, d), usize);
        let mut challenge_bits = Zt::Chal::NUM_BITS;
        if Zt::Comb::DEGREE_BOUND > 0 {
            // This means we also draft alphas (multiplied with coeffs), which
            // doubles the number of challenge bits
            challenge_bits = mul!(challenge_bits, 2);
        }
        let max_lc_bits = Zt::CombR::NUM_BITS;
        let actual_lc_bits: usize = add!(
            add!(
                add!(mul!(codeword_bits, 2), ilog_round_up!(d, usize)),
                challenge_bits
            ),
            add!(
                sub!(Zt::Eval::COEFF_BIT_WIDTH, 1),
                ilog_round_up!(batch_size, usize)
            )
        );
        assert!(
            actual_lc_bits <= max_lc_bits,
            "The number of bits used for linear combinations is too large: {actual_lc_bits} bits, max allowed is {max_lc_bits} bits"
        );
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
pub(super) fn point_to_tensor<F>(
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
        DenseMultilinearExtension::zero_vars(F::one_with_cfg(cfg))
    };

    let q_1 = if !hi.is_empty() {
        build_eq_x_r(hi, cfg).unwrap()
    } else {
        DenseMultilinearExtension::zero_vars(F::one_with_cfg(cfg))
    };

    Ok((q_0.evaluations, q_1.evaluations))
}
