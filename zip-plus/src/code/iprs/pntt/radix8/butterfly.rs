use num_traits::{CheckedAdd, CheckedSub};
use zinc_utils::{add, sub};

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
// Twiddles [1, 4096, -256, 16, -1, -4096, 256, -16]
// 0 [1, 1, 1, 1, 1, 1, 1, 1]
// 1 [1, 4096, -256, 16, -1, -4096, 256, -16]
// 2 [1, -256, -1, 256, 1, -256, -1, 256]
// 3 [1, 16, 256, 4096, -1, -16, -256, -4096]
// 4 [1, -1, 1, -1, 1, -1, 1, -1]
// 5 [1, -4096, -256, -16, -1, 4096, 256, 16]
// 6 [1, 256, -1, -256, 1, 256, -1, -256]
// 7 [1, -16, 256, -4096, -1, 16, -256, 4096]

#[allow(clippy::arithmetic_side_effects)]
pub(crate) fn apply_radix_8_butterflies<Out, Twiddle, M>(
    xs: &[Out],
    mut ys: [&mut Out; 8],
    twiddles: &[Twiddle],
    mul_by_twiddle: M,
) where
    Out: Clone + CheckedAdd + CheckedSub,
    M: MulByTwiddle<Out, Twiddle, Output = Out>,
{
    // Step 0. Each y takes xs[0].
    ys.iter_mut().for_each(|y| **y = xs[0].clone());

    // Step 1.
    let x: [Out; 3] = [
        mul_by_twiddle.mul_by_twiddle(&xs[1], &twiddles[1]),
        mul_by_twiddle.mul_by_twiddle(&xs[1], &twiddles[2]),
        mul_by_twiddle.mul_by_twiddle(&xs[1], &twiddles[3]),
    ];

    *(ys[0]) = add!(ys[0], &xs[1]);
    *(ys[4]) = sub!(ys[4], &xs[1]);

    for i in 1..4 {
        *(ys[i]) = add!(ys[i], &x[i - 1]);
        *(ys[i + 4]) = sub!(ys[i + 4], &x[i - 1]);
    }

    // Step 2.
    let x = mul_by_twiddle.mul_by_twiddle(&xs[2], &twiddles[2]);

    *(ys[0]) = add!(ys[0], &xs[2]);
    *(ys[4]) = add!(ys[4], &xs[2]);
    *(ys[1]) = add!(ys[1], &x);
    *(ys[5]) = add!(ys[5], &x);
    *(ys[2]) = sub!(ys[2], &xs[2]);
    *(ys[6]) = sub!(ys[6], &xs[2]);
    *(ys[3]) = sub!(ys[3], &x);
    *(ys[7]) = sub!(ys[7], &x);

    // Step 3.
    let x: [Out; 3] = [
        mul_by_twiddle.mul_by_twiddle(&xs[3], &twiddles[3]),
        mul_by_twiddle.mul_by_twiddle(&xs[3], &twiddles[6]),
        mul_by_twiddle.mul_by_twiddle(&xs[3], &twiddles[1]),
    ];

    *(ys[0]) = add!(ys[0], &xs[3]);
    *(ys[4]) = sub!(ys[4], &xs[3]);

    for i in 1..4 {
        *(ys[i]) = add!(ys[i], &x[i - 1]);
        *(ys[i + 4]) = sub!(ys[i + 4], &x[i - 1]);
    }

    // Step 4.
    ys.iter_mut().enumerate().for_each(|(i, y)| {
        if i % 2 == 0 {
            **y = add!(*y, &xs[4]);
        } else {
            **y = sub!(*y, &xs[4]);
        }
    });

    // Step 5.
    let x: [Out; 3] = [
        mul_by_twiddle.mul_by_twiddle(&xs[5], &twiddles[5]),
        mul_by_twiddle.mul_by_twiddle(&xs[5], &twiddles[2]),
        mul_by_twiddle.mul_by_twiddle(&xs[5], &twiddles[7]),
    ];

    *(ys[0]) = add!(ys[0], &xs[5]);
    *(ys[4]) = sub!(ys[4], &xs[5]);

    for i in 1..4 {
        *(ys[i]) = add!(ys[i], &x[i - 1]);
        *(ys[i + 4]) = sub!(ys[i + 4], &x[i - 1]);
    }

    // Step 6.
    let x = mul_by_twiddle.mul_by_twiddle(&xs[6], &twiddles[6]);
    *(ys[0]) = add!(ys[0], &xs[6]);
    *(ys[4]) = add!(ys[4], &xs[6]);
    *(ys[1]) = add!(ys[1], &x);
    *(ys[5]) = add!(ys[5], &x);
    *(ys[2]) = sub!(ys[2], &xs[6]);
    *(ys[6]) = sub!(ys[6], &xs[6]);
    *(ys[3]) = sub!(ys[3], &x);
    *(ys[7]) = sub!(ys[7], &x);

    // Step 7.
    let x: [Out; 3] = [
        mul_by_twiddle.mul_by_twiddle(&xs[7], &twiddles[7]),
        mul_by_twiddle.mul_by_twiddle(&xs[7], &twiddles[6]),
        mul_by_twiddle.mul_by_twiddle(&xs[7], &twiddles[5]),
    ];

    *(ys[0]) = add!(ys[0], &xs[7]);
    *(ys[4]) = sub!(ys[4], &xs[7]);

    for i in 1..4 {
        *(ys[i]) = add!(ys[i], &x[i - 1]);
        *(ys[i + 4]) = sub!(ys[i + 4], &x[i - 1]);
    }
}
