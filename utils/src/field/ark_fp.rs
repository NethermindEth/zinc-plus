use ark_ff::{BigInt, MontBackend, MontConfig};
use crypto_primitives::{ConstPrimeField, ark_ff_fp::Fp};

use crate::{inner_transparent_field::InnerTransparentField, mul_by_scalar::MulByScalar};

impl<M: MontConfig<N>, const N: usize> MulByScalar<&Self> for Fp<MontBackend<M, N>, N> {
    #[allow(clippy::arithmetic_side_effects)] // Field operations cannot overflow
    fn mul_by_scalar<const CHECK: bool>(&self, rhs: &Self) -> Option<Self> {
        Some(*self * rhs)
    }
}

impl<M: MontConfig<N>, const N: usize> InnerTransparentField for Fp<MontBackend<M, N>, N> {
    #[allow(clippy::arithmetic_side_effects)] // Field operations cannot overflow
    fn add_inner(lhs: &BigInt<N>, rhs: &BigInt<N>, _config: &Self::Config) -> BigInt<N> {
        (Self::new_unchecked(*lhs) + Self::new_unchecked(*rhs))
            .into_inner()
            .0
    }

    #[allow(clippy::arithmetic_side_effects)] // Field operations cannot overflow
    fn sub_inner(lhs: &BigInt<N>, rhs: &BigInt<N>, _config: &Self::Config) -> BigInt<N> {
        (Self::new_unchecked(*lhs) - Self::new_unchecked(*rhs))
            .into_inner()
            .0
    }

    #[allow(clippy::arithmetic_side_effects)] // Field operations cannot overflow
    fn mul_assign_by_inner(&mut self, rhs: &BigInt<N>) {
        *self *= Self::new_unchecked(*rhs);
    }
}
