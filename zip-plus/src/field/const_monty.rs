use crate::{
    pcs::structs::{MulByScalar, ProjectableToField},
    poly::{Polynomial, dense::DensePolynomial},
    traits::{FromRef, Transcribable},
};
use crypto_bigint::{
    Uint,
    modular::{ConstMontyForm, ConstMontyParams},
};
use crypto_primitives::{
    FixedRing, crypto_bigint_const_monty::ConstMontyField, crypto_bigint_int::Int,
};
use num_traits::CheckedMul;

impl<Mod: ConstMontyParams<LIMBS>, const LIMBS: usize> MulByScalar<&Self>
    for ConstMontyField<Mod, LIMBS>
{
    fn mul_by_scalar(&self, rhs: &Self) -> Option<Self> {
        self.checked_mul(rhs)
    }
}

macro_rules! impl_from_primitive_ref {
    ($($t:ty),* $(,)?) => {
        $(
            impl<Mod: ConstMontyParams<LIMBS>, const LIMBS: usize> FromRef<$t> for ConstMontyField<Mod, LIMBS> {
                #![allow(clippy::arithmetic_side_effects)]
                fn from_ref(value: &$t) -> Self {
                    Self::from(*value)
                }
            }
        )*
    };
}
impl_from_primitive_ref!(u8, u16, u32, u64, u128);
impl_from_primitive_ref!(i8, i16, i32, i64, i128);

impl<Mod: ConstMontyParams<LIMBS>, const LIMBS: usize, const LIMBS2: usize> FromRef<Int<LIMBS2>>
    for ConstMontyField<Mod, LIMBS>
{
    fn from_ref(value: &Int<LIMBS2>) -> Self {
        Self::from(value)
    }
}

impl<Mod: ConstMontyParams<LIMBS>, const LIMBS: usize> FromRef<Self>
    for ConstMontyField<Mod, LIMBS>
{
    fn from_ref(value: &Self) -> Self {
        *value
    }
}

impl<Mod: ConstMontyParams<LIMBS>, const LIMBS: usize, const LIMBS2: usize>
    ProjectableToField<ConstMontyField<Mod, LIMBS>> for Int<LIMBS2>
{
    fn prepare_projection(
        _sampled_value: &ConstMontyField<Mod, LIMBS>,
    ) -> impl Fn(&Self) -> ConstMontyField<Mod, LIMBS> + 'static {
        // No need to read anything
        |value: &Int<LIMBS2>| ConstMontyField::<Mod, LIMBS>::from_ref(value)
    }
}

impl<Mod: ConstMontyParams<LIMBS>, R, const LIMBS: usize, const DEGREE: usize>
    ProjectableToField<ConstMontyField<Mod, LIMBS>> for DensePolynomial<R, DEGREE>
where
    R: FixedRing + Transcribable,
    ConstMontyField<Mod, LIMBS>: FromRef<R>,
{
    #![allow(clippy::arithmetic_side_effects)] // False alert, field operations are safe
    fn prepare_projection(
        sampled_value: &ConstMontyField<Mod, LIMBS>,
    ) -> impl Fn(&Self) -> ConstMontyField<Mod, LIMBS> + 'static {
        let mut r_powers = vec![*sampled_value; DEGREE];
        for i in 1..DEGREE {
            r_powers[i] = r_powers[i - 1] * sampled_value;
        }

        move |poly: &Self| {
            poly.map(ConstMontyField::<Mod, LIMBS>::from_ref)
                .evaluate_at_point(&r_powers)
                .expect("Failed to evaluate polynomial at point")
        }
    }
}

impl<Mod: ConstMontyParams<LIMBS>, const LIMBS: usize> Transcribable
    for ConstMontyForm<Mod, LIMBS>
{
    const NUM_BYTES: usize = Uint::<LIMBS>::NUM_BYTES;

    fn read_transcription_bytes(bytes: &[u8]) -> Self {
        ConstMontyForm::from_montgomery(Uint::read_transcription_bytes(bytes))
    }

    fn write_transcription_bytes(&self, buf: &mut [u8]) {
        self.as_montgomery().write_transcription_bytes(buf)
    }
}

#[cfg(test)]
#[allow(
    clippy::arithmetic_side_effects,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap
)]
mod tests {
    use super::*;

    use crate::poly::dense::DensePolynomial;
    use crypto_bigint::{U128, U256, const_monty_params};
    use crypto_primitives::{crypto_bigint_const_monty::F256, crypto_bigint_int::Int};
    use num_traits::{One, Zero};
    use proptest::prelude::*;

    const_monty_params!(
        ModQ,
        U256,
        "00dca94d8a1ecce3b6e8755d8999787d0524d8ca1ea755e7af84fb646fa31f27"
    );
    type F = F256<ModQ>;

    #[test]
    fn prepare_projection_for_int() {
        // Create a sample field element and an Int value
        let sampled = F::from(5_u64);

        let projection_fn =
            <Int<{ U128::LIMBS }> as ProjectableToField<F>>::prepare_projection(&sampled);

        let int_value = Int::<{ U128::LIMBS }>::from(10_i64);
        let result = projection_fn(&int_value);
        assert_eq!(result, F::from_ref(&int_value));
        assert_eq!(result, F::from(10_u64));

        let int_value = Int::<{ U128::LIMBS }>::from(-7_i64);
        let result = projection_fn(&int_value);
        assert_eq!(result, F::from_ref(&int_value));
        assert_eq!(result + F::from(7_u64), F::zero());
    }

    #[test]
    fn prepare_projection_for_polynomial() {
        type Poly = DensePolynomial<Int<{ U128::LIMBS }>, 3>;

        // 1 - 2x + 3x^2
        let poly = Poly::new([Int::from(1_i64), Int::from(-2_i64), Int::from(3_i64)]);

        // Sample at x = 2
        let sampled = F::from(2_u64);

        // Prepare projection function
        let projection_fn = <Poly as ProjectableToField<F>>::prepare_projection(&sampled);

        // Apply projection - should evaluate polynomial at x = 2
        // P(2) = 1 - 2*2 - 3*2^2 = 1 - 4 + 12 = 9
        let result = projection_fn(&poly);
        assert_eq!(result, F::from(9_u64));
    }

    fn any_u128() -> impl Strategy<Value = u128> {
        any::<u128>()
    }
    fn any_i128() -> impl Strategy<Value = i128> {
        any::<i128>()
    }
    fn any_bool() -> impl Strategy<Value = bool> {
        any::<bool>()
    }

    proptest! {
        #[test]
        fn prop_from_unsigned_matches_sum_of_bits(x in any_u128()) {
            let f = F::from(x);
            let mut acc = F::zero();
            for i in 0..128 {
                if (x >> i) & 1 == 1 { acc += F::from(1u64) * F::from(1u64 << i.min(63)); }
            }
            let u = crypto_bigint::Uint::<{ U256::LIMBS }>::from(x);
            let g2: F = F::from(u);
            prop_assert_eq!(f, g2);
        }

        #[test]
        fn prop_from_signed_is_neg_of_abs_when_negative(x in any_i128()) {
            let f = F::from(x);
            let abs = x.unsigned_abs();
            let g_abs = F::from(abs);
            if x < 0 {
                prop_assert_eq!(f + g_abs, F::zero());
            } else {
                prop_assert_eq!(f, g_abs);
            }
        }

        #[test]
        fn prop_from_bool_is_identity(b in any_bool()) {
            let f = F::from(b);
            prop_assert_eq!(f, if b { F::one() } else { F::zero() });
        }

        #[test]
        fn prop_from_uint_roundtrip_through_uint(x in any_u128()) {
            let u: crypto_bigint::Uint<{ U256::LIMBS }> = crypto_bigint::Uint::from(x);
            let g_from_uint: F = u.into();
            let g_direct: F = F::from(x);
            prop_assert_eq!(g_from_uint, g_direct);
        }

        #[test]
        fn prop_from_ref_generic_matches_owned(x in any::<u64>()) {
            let a: F = F::from(x);
            let b: F = F::from(&x);
            prop_assert_eq!(a, b);
        }
    }
}
