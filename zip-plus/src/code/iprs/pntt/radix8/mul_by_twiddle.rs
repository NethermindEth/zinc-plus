use crypto_primitives::{FromWithConfig, PrimeField};
use std::marker::PhantomData;

use zinc_utils::mul_by_scalar::{MulByScalar, WideningMulByScalar};

/// A helper trait that allows to provide
/// the pseudo NTT algorithm a means to
/// multiply output by twiddles.
// TODO(alex): Can we get away with using just MulByScalar?
pub(crate) trait MulByTwiddle<Lhs, Twiddle>: Clone + Send + Sync {
    type Output;

    fn mul_by_twiddle(&self, lhs: &Lhs, twiddle: &Twiddle) -> Self::Output;
}

/// The twiddle multiplication that
/// uses the `MulByScalar` implementation.
#[derive(Debug, Clone, Copy)]
pub(crate) struct MBSMulByTwiddle;

impl<Lhs, Twiddle> MulByTwiddle<Lhs, Twiddle> for MBSMulByTwiddle
where
    Lhs: for<'a> MulByScalar<&'a Twiddle>,
{
    type Output = Lhs;

    #[inline(always)]
    fn mul_by_twiddle(&self, lhs: &Lhs, twiddle: &Twiddle) -> Lhs {
        lhs.mul_by_scalar(twiddle)
            .expect("Twiddle multiplication overflow")
    }
}

/// If we deal with fields we do not have to
/// worry about overflows. Multiplication by twiddle
/// is done by conversions and field operations.
#[derive(Debug, Clone)]
pub(crate) struct FieldMulByTwiddle<F: PrimeField, T> {
    config: F::Config,
    _phantom: PhantomData<T>,
}

impl<F: PrimeField, T> FieldMulByTwiddle<F, T> {
    pub fn new(config: F::Config) -> Self {
        Self {
            config,
            _phantom: Default::default(),
        }
    }
}

impl<F, Twiddle, T> MulByTwiddle<F, Twiddle> for FieldMulByTwiddle<F, T>
where
    F: PrimeField + FromWithConfig<T>,
    Twiddle: Clone + Into<T>,
    T: Clone + Send + Sync,
{
    type Output = F;

    #[inline(always)]
    fn mul_by_twiddle(&self, lhs: &F, twiddle: &Twiddle) -> F {
        F::from_with_cfg(twiddle.clone().into(), &self.config) * lhs
    }
}

#[derive(Clone, Default, Copy)]
pub struct WideningMulByTwiddle<WM>(PhantomData<WM>);

impl<Lhs, Twiddle, WM> MulByTwiddle<Lhs, Twiddle> for WideningMulByTwiddle<WM>
where
    WM: WideningMulByScalar<Lhs, Twiddle>,
{
    type Output = WM::Output;

    #[inline(always)]
    fn mul_by_twiddle(&self, lhs: &Lhs, twiddle: &Twiddle) -> Self::Output {
        WM::mul_by_scalar_widen(lhs, twiddle)
    }
}
