use std::ops::Add;

use num_traits::CheckedAdd;
use zinc_utils::add;

use super::MulByTwiddle;

/// The butterfly table for radix-8 FFT. Each entry is the index of the twiddle
/// factor to be used for the corresponding input element.
pub(crate) const BUTTERFLY_TABLE: [[usize; 7]; 8] = [
    [0, 0, 0, 0, 0, 0, 0],
    [1, 2, 3, 4, 5, 6, 7],
    [2, 4, 6, 0, 2, 4, 6],
    [3, 6, 1, 4, 7, 2, 5],
    [4, 0, 4, 0, 4, 0, 4],
    [5, 2, 7, 4, 1, 6, 3],
    [6, 4, 2, 0, 6, 4, 2],
    [7, 6, 5, 4, 3, 2, 1],
];

/// Apply butterfly given by `twiddles` to a slice
/// of subresults in `x`. `twiddles[j][i]` is the factor to multiply
/// `x[j + 1]` by when the butterfly table requests the `i`-th twiddle.
/// Use `mul_by_twiddle` as a means to multiply `Out` by `Twiddle`.
#[allow(clippy::arithmetic_side_effects)]
pub(crate) fn apply_radix_8_butterflies<R, Twiddle, M, const CHECK_ADDITION: bool>(
    ys: [&mut R; 8],
    xs: &[R],
    twiddles: &[[Twiddle; 8]; 7],
) where
    R: Clone + CheckedAdd + for<'a> Add<&'a R, Output = R>,
    M: MulByTwiddle<R, Twiddle, Output = R>,
{
    ys.into_iter()
        .zip(BUTTERFLY_TABLE.iter())
        .for_each(|(y, butterfly_row)| {
            *y = xs[1..].iter().zip(&butterfly_row[..]).enumerate().fold(
                xs[0].clone(),
                |a, (j_minus_1, (x, &twiddle_idx))| {
                    let term = M::mul_by_twiddle(x, &twiddles[j_minus_1][twiddle_idx]);
                    if CHECK_ADDITION {
                        add!(a, &term)
                    } else {
                        a + &term
                    }
                },
            )
        });
}
