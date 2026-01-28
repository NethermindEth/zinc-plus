use std::mem::MaybeUninit;

use crypto_primitives::Semiring;

#[allow(clippy::arithmetic_side_effects)]
pub(crate) fn mul_schoolbook_unchecked<R: Semiring>(
    lhs: &[R],
    rhs: &[R],
    out: &mut [MaybeUninit<R>],
) {
    let m = lhs.len() - 1;
    let n = rhs.len() - 1;

    out.iter_mut().enumerate().for_each(|(k, out)| {
        let from = k.saturating_sub(n);
        let to = std::cmp::min(k, m);

        *out = MaybeUninit::new((from + 1..=to).fold(
            lhs[from].clone() * &rhs[k - from],
            |mut acc, next| {
                acc += lhs[next].clone() * &rhs[k - next];
                acc
            },
        ));
    });
}

#[allow(clippy::arithmetic_side_effects)]
pub(crate) fn mul_schoolbook_checked<R: Semiring>(
    lhs: &[R],
    rhs: &[R],
    out: &mut [MaybeUninit<R>],
) -> Option<()> {
    let m = lhs.len() - 1;
    let n = rhs.len() - 1;

    out.iter_mut().enumerate().try_for_each(|(k, out)| {
        let from = k.saturating_sub(n);
        let to = std::cmp::min(k, m);

        *out = MaybeUninit::new((from + 1..=to).try_fold(
            lhs[from].clone() * &rhs[k - from],
            |mut acc, next| {
                acc = acc.checked_add(&lhs[next].checked_mul(&rhs[k - next])?)?;
                Some(acc)
            },
        )?);

        Some(())
    })
}

// TODO(Ilia): Implement faster multiplication algorithms using
//             FFT for fields and Karatsuba for rings.
