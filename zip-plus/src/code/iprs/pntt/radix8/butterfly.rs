use num_traits::{CheckedAdd, CheckedSub};
use zinc_utils::{add, sub};

use super::MulByTwiddle;

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
