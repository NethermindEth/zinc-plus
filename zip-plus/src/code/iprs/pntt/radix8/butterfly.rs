use num_traits::CheckedAdd;
use zinc_utils::add;

use super::MulByTwiddle;

/// Generate the indices of twiddle factors
/// used for a butterfly.
///
/// In case of radix 8, we have 8 butterflies.
macro_rules! generate_radix_8_butterfly {
    ($i:literal) => {
        [
            0,
            $i % 8,
            (2 * $i) % 8,
            (3 * $i) % 8,
            (4 * $i) % 8,
            (5 * $i) % 8,
            (6 * $i) % 8,
            (7 * $i) % 8,
        ]
    };
    ($i:expr) => {
        match $i {
            0 => generate_radix_8_butterfly!(0),
            1 => generate_radix_8_butterfly!(1),
            2 => generate_radix_8_butterfly!(2),
            3 => generate_radix_8_butterfly!(3),
            4 => generate_radix_8_butterfly!(4),
            5 => generate_radix_8_butterfly!(5),
            6 => generate_radix_8_butterfly!(6),
            7 => generate_radix_8_butterfly!(7),
            _ => panic!("Incorrect butterfly number"),
        }
    };
}

/// Apply butterfly given by `twiddles` to a slice
/// of subresults in `x`. Use `mul_by_twiddle` as a means
/// to multiply `Out` by `Twiddle`.
pub(crate) fn radix_8_butterfly<Out, Twiddle, M, const I: usize>(
    x: &[Out],
    twiddles: &[Twiddle],
    mul_by_twiddle: M,
) -> Out
where
    Out: Clone + CheckedAdd,
    M: MulByTwiddle<Out, Twiddle, Output = Out>,
{
    let butterfly: [usize; 8] = generate_radix_8_butterfly!(I);

    x[1..]
        .iter()
        .zip(&butterfly[1..])
        .fold(x[0].clone(), |a, (x, &twiddle_idx)| {
            add!(a, &mul_by_twiddle.mul_by_twiddle(x, &twiddles[twiddle_idx]))
        })
}
macro_rules! do_all_butterflies {
    ($x:expr, $twiddles:expr, $mbt: expr) => {
        (
            radix_8_butterfly::<_, _, _, 0>($x, $twiddles, $mbt),
            radix_8_butterfly::<_, _, _, 1>($x, $twiddles, $mbt),
            radix_8_butterfly::<_, _, _, 2>($x, $twiddles, $mbt),
            radix_8_butterfly::<_, _, _, 3>($x, $twiddles, $mbt),
            radix_8_butterfly::<_, _, _, 4>($x, $twiddles, $mbt),
            radix_8_butterfly::<_, _, _, 5>($x, $twiddles, $mbt),
            radix_8_butterfly::<_, _, _, 6>($x, $twiddles, $mbt),
            radix_8_butterfly::<_, _, _, 7>($x, $twiddles, $mbt),
        )
    };
}
