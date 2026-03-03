use crypto_primitives::{FromWithConfig, PrimeField};
use std::marker::PhantomData;

use zinc_utils::mul_by_scalar::MulByScalar;

/// A helper trait that allows to provide
/// the pseudo NTT algorithm a means to
/// multiply output by twiddles.
// TODO(alex): Can we get away with using just MulByScalar?
// TODO(alex): Yes we can! Make a follow-up PR that removes this.
pub(crate) trait MulByTwiddle<Lhs, Twiddle>: Clone + Send + Sync {
    type Output;

    fn mul_by_twiddle(lhs: &Lhs, twiddle: &Twiddle) -> Self::Output;
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
        lhs.mul_by_scalar::<CHECK>(twiddle)
            .expect("Twiddle multiplication overflow")
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
pub struct WideningMulByTwiddle<Out, const CHECK: bool>(PhantomData<Out>);

impl<Lhs, Twiddle, Out, const CHECK: bool> MulByTwiddle<Lhs, Twiddle>
    for WideningMulByTwiddle<Out, CHECK>
where
    Lhs: for<'a> MulByScalar<&'a Twiddle, Out>,
    Out: Clone + Send + Sync,
{
    type Output = Out;

    #[inline(always)]
    fn mul_by_twiddle(lhs: &Lhs, twiddle: &Twiddle) -> Out {
        lhs.mul_by_scalar::<CHECK>(twiddle)
            .expect("Multiplication by twiddle factor should not overflow")
    }
}
