use crypto_primitives::{FromWithConfig, PrimeField};
use std::marker::PhantomData;

use zinc_utils::mul_by_scalar::{MulByScalar, WideningMulByScalar};

/// A helper trait that allows to provide
/// the pseudo NTT algorithm a means to
/// multiply output by twiddles.
// TODO(alex): Can we get away with using just MulByScalar?
pub(crate) trait MulByTwiddle<Lhs, Twiddle>: Clone + Send + Sync {
    type Output;

    fn mul_by_twiddle(lhs: &Lhs, twiddle: &Twiddle) -> Self::Output;
}

/// A helper trait for fused multiply-add operations.
/// Computes: acc += lhs * twiddle without intermediate allocation.
/// This is more efficient than MulByTwiddle followed by add.
pub(crate) trait FusedMulAddByTwiddle<Lhs, Twiddle, Acc>: Clone + Send + Sync {
    /// Fused multiply-add: acc += lhs * twiddle
    fn fused_mul_add(acc: &mut Acc, lhs: &Lhs, twiddle: &Twiddle);
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
}

/// Marker for fused multiply-add using zinc_poly's FusedMulAdd trait
#[derive(Clone, Default, Copy)]
pub struct FusedWideningMulByTwiddle<WM>(PhantomData<WM>);

/// Implementation of FusedMulAddByTwiddle that uses zinc_poly's FusedMulAdd trait
#[cfg(feature = "simd")]
impl<Lhs, Twiddle, Acc, Inner> FusedMulAddByTwiddle<Lhs, Twiddle, Acc>
    for FusedWideningMulByTwiddle<Inner>
where
    Twiddle: Clone,
    Inner: WideningMulByScalar<Lhs, Twiddle>,
    Acc: zinc_poly::univariate::binary_u64::FusedMulAdd<Lhs, Twiddle>,
{
    #[inline(always)]
    fn fused_mul_add(acc: &mut Acc, lhs: &Lhs, twiddle: &Twiddle) {
        acc.fused_mul_add(lhs, twiddle.clone());
    }
}

/// Fallback implementation that uses MulByTwiddle + add
#[cfg(not(feature = "simd"))]
impl<Lhs, Twiddle, Acc, Inner> FusedMulAddByTwiddle<Lhs, Twiddle, Acc>
    for FusedWideningMulByTwiddle<Inner>
where
    Inner: WideningMulByScalar<Lhs, Twiddle, Output = Acc>,
    Acc: std::ops::AddAssign,
{
    #[inline(always)]
    fn fused_mul_add(acc: &mut Acc, lhs: &Lhs, twiddle: &Twiddle) {
        *acc += Inner::mul_by_scalar_widen(lhs, twiddle);
    }
}
