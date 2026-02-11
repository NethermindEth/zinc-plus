use crypto_primitives::{FromWithConfig, PrimeField};
use std::marker::PhantomData;
use std::ops::AddAssign;

use zinc_utils::mul_by_scalar::{MulByScalar, WideningMulByScalar};

/// A helper trait that allows to provide
/// the pseudo NTT algorithm a means to
/// multiply output by twiddles.
// TODO(alex): Can we get away with using just MulByScalar?
pub(crate) trait MulByTwiddle<Lhs, Twiddle>: Clone + Send + Sync {
    type Output;

    fn mul_by_twiddle(lhs: &Lhs, twiddle: &Twiddle) -> Self::Output;

    /// Fused multiply-and-add: equivalent to `*acc += mul_by_twiddle(lhs, twiddle)`
    /// but can be overridden to avoid creating a temporary.
    fn mul_by_twiddle_and_add(acc: &mut Self::Output, lhs: &Lhs, twiddle: &Twiddle)
    where
        Self::Output: for<'a> AddAssign<&'a Self::Output>,
    {
        let term = Self::mul_by_twiddle(lhs, twiddle);
        *acc += &term;
    }
}

/// The twiddle multiplication that
/// uses the `MulByScalar` implementation.
#[derive(Debug, Clone, Copy)]
pub(crate) struct MBSMulByTwiddle<const CHECK: bool>;

impl<Lhs, Twiddle, const CHECK: bool> MulByTwiddle<Lhs, Twiddle> for MBSMulByTwiddle<CHECK>
where
    Lhs: for<'a> MulByScalar<&'a Twiddle>,
{
    type Output = Lhs;

    #[inline(always)]
    fn mul_by_twiddle(lhs: &Lhs, twiddle: &Twiddle) -> Lhs {
        if CHECK {
            lhs.mul_by_scalar::<CHECK>(twiddle)
                .expect("Twiddle multiplication overflow")
        } else {
            // SAFETY: mul_by_scalar::<false> always returns Some for standard
            // numeric types — the unchecked path never produces None.
            unsafe {
                lhs.mul_by_scalar::<false>(twiddle).unwrap_unchecked()
            }
        }
    }

    /// Fused multiply-twiddle-and-add using `MulByScalar::mul_by_scalar_and_add_to`.
    /// For types like `DensePolynomial`, this avoids creating an intermediate
    /// polynomial (saves a clone + separate add pass).
    #[inline(always)]
    fn mul_by_twiddle_and_add(acc: &mut Lhs, lhs: &Lhs, twiddle: &Twiddle)
    where
        Lhs: for<'a> AddAssign<&'a Lhs>,
    {
        lhs.mul_by_scalar_and_add_to::<CHECK>(twiddle, acc);
    }
}

/// If we deal with fields we do not have to
/// worry about overflows. Multiplication by twiddle
/// is done by conversions and field operations.
#[derive(Debug, Clone)]
pub(crate) struct FieldMulByTwiddle<F: PrimeField, T>(PhantomData<(F, T)>);

impl<F, Twiddle, T> MulByTwiddle<F, Twiddle> for FieldMulByTwiddle<F, T>
where
    F: PrimeField + FromWithConfig<T>,
    Twiddle: Clone + Into<T>,
    T: Clone + Send + Sync,
{
    type Output = F;

    #[inline(always)]
    fn mul_by_twiddle(lhs: &F, twiddle: &Twiddle) -> F {
        F::from_with_cfg(twiddle.clone().into(), lhs.cfg()) * lhs
    }
}

#[derive(Clone, Default, Copy)]
pub struct WideningMulByTwiddle<WM>(PhantomData<WM>);

impl<Lhs, Twiddle, Inner> MulByTwiddle<Lhs, Twiddle> for WideningMulByTwiddle<Inner>
where
    Inner: WideningMulByScalar<Lhs, Twiddle>,
{
    type Output = Inner::Output;

    #[inline(always)]
    fn mul_by_twiddle(lhs: &Lhs, twiddle: &Twiddle) -> Self::Output {
        Inner::mul_by_scalar_widen(lhs, twiddle)
    }

    /// Fused widen-multiply-and-add: delegates to
    /// [`WideningMulByScalar::widen_and_add`] to allow specialized
    /// implementations (e.g. for `DensePolynomial`) to avoid creating
    /// intermediate temporaries.
    #[inline(always)]
    fn mul_by_twiddle_and_add(acc: &mut Self::Output, lhs: &Lhs, twiddle: &Twiddle)
    where
        Self::Output: for<'a> AddAssign<&'a Self::Output>,
    {
        Inner::widen_and_add(acc, lhs, twiddle);
    }
}
