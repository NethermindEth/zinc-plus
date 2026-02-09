use std::mem::MaybeUninit;

use crypto_primitives::Semiring;

#[allow(clippy::arithmetic_side_effects)]
pub(crate) fn mul_schoolbook<R: Semiring, const CHECK: bool>(
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
            |acc, next| {
                if CHECK {
                    acc.checked_add(&lhs[next].checked_mul(&rhs[k - next])?)
                } else {
                    Some(acc + lhs[next].clone() * &rhs[k - next])
                }
            },
        )?);

        Some(())
    })?;

    Some(())
}
