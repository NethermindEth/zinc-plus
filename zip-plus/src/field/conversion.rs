use crypto_bigint::{Int, Uint, Word, Zero};
use crypto_bigint::modular::{ConstMontyForm, ConstMontyParams};
use itertools::Itertools;
use num_traits::{ConstOne, ConstZero};

use crate::{
    field::{ConstMontyField},
    traits::{FromBits, MapIterable},
};
use crate::traits::{ConstNumBytes};

impl<Mod: ConstMontyParams<LIMBS>, const LIMBS: usize> ConstNumBytes for ConstMontyField<Mod, LIMBS> {
    const NUM_BYTES: usize = ConstMontyForm::<Mod, LIMBS>::NUM_BYTES;
}

// Macro to implement From for unsigned integer primitives
macro_rules! impl_from_unsigned {
    ($($t:ty),* $(,)?) => {
        $(
            impl<Mod: ConstMontyParams<LIMBS>, const LIMBS: usize> From<$t> for ConstMontyField<Mod, LIMBS> {
                fn from(value: $t) -> Self {
                    let value = Uint::from(value);
                    Self(ConstMontyForm::new(&value))
                }
            }
        )*
    };
}

// Macro to implement From for signed integer primitives
macro_rules! impl_from_signed {
    ($($t:ty),* $(,)?) => {
        $(
            impl<Mod: ConstMontyParams<LIMBS>, const LIMBS: usize> From<$t> for ConstMontyField<Mod, LIMBS> {
                fn from(value: $t) -> Self {
                    let magnitude = Uint::from(value.abs_diff(0));
                    let form = ConstMontyForm::new(&magnitude);
                    Self(if value.is_negative() { -form } else { form })
                }
            }
        )*
    };
}

impl_from_unsigned!(u8, u16, u32, u64, u128);
impl_from_signed!(i8, i16, i32, i64, i128);

// Manual impls for pointer-sized integers, as requested
impl<Mod: ConstMontyParams<LIMBS>, const LIMBS: usize> From<usize> for ConstMontyField<Mod, LIMBS> {
    fn from(value: usize) -> Self {
        // Cast through u64 as specified
        let value = Uint::from(value as u64);
        Self(ConstMontyForm::new(&value))
    }
}

impl<Mod: ConstMontyParams<LIMBS>, const LIMBS: usize> From<isize> for ConstMontyField<Mod, LIMBS> {
    fn from(value: isize) -> Self {
        // Use i64 for sign handling as specified
        let magnitude = Uint::from((value as i64).abs_diff(0));
        let form = ConstMontyForm::new(&magnitude);
        Self(if value.is_negative() { -form } else { form })
    }
}

impl<Mod: ConstMontyParams<LIMBS>, const LIMBS: usize> From<bool> for ConstMontyField<Mod, LIMBS> {
    fn from(value: bool) -> Self {
        if value { Self::ONE } else { Self::ZERO }
    }
}

impl<Mod: ConstMontyParams<LIMBS>, const LIMBS: usize> From<Uint<LIMBS>> for ConstMontyField<Mod, LIMBS> {
    fn from(value: Uint<LIMBS>) -> Self {
        Self(ConstMontyForm::new(&value))
    }
}

impl<Mod: ConstMontyParams<LIMBS>, const LIMBS: usize> From<ConstMontyForm<Mod, LIMBS>> for ConstMontyField<Mod, LIMBS> {
    fn from(value: ConstMontyForm<Mod, LIMBS>) -> Self {
        Self(value)
    }
}

impl<Mod: ConstMontyParams<LIMBS>, const LIMBS: usize, const LIMBS2: usize> From<Int<LIMBS2>>
    for ConstMontyField<Mod, LIMBS>
{
    fn from(value: Int<LIMBS2>) -> Self {
        assert!(LIMBS >= LIMBS2, "Cannot convert Int with more limbs than ConstMontyField");
        let value = value.resize();
        let result = Self(ConstMontyForm::new(&value.abs()));

        if value.is_negative().into() {
            -result
        } else {
            result
        }
    }
}

impl<Mod: ConstMontyParams<LIMBS>, const LIMBS: usize, T: Clone> From<&T> for ConstMontyField<Mod, LIMBS>
where
    Self: From<T>,
{
    fn from(value: &T) -> Self {
        Self::from(value.clone())
    }
}

impl<const N: usize> FromBits for Uint<N> {
    #[inline]
    fn from_be_bits(bits: &[bool]) -> Self {
        let mut bits = bits.to_vec();
        bits.reverse();
        Self::from_le_bits(&bits)
    }

    fn from_le_bits(bits: &[bool]) -> Self {
        let mut res = Self::zero();
        let limb_bits = Word::BITS as usize;
        for (bit_chunk, res_i) in bits.chunks(limb_bits).zip(res.as_mut_words()) {
            for (i, bit) in bit_chunk.iter().enumerate() {
                // i is always < limb_bits here, so shifting is safe on both 32- and 64-bit
                *res_i |= (*bit as Word) << i;
            }
        }
        res
    }
}

impl<Mod: ConstMontyParams<LIMBS>, const LIMBS: usize> MapIterable for ConstMontyField<Mod, LIMBS> {
    fn map_iterable<'a, const M: usize, I: IntoIterator<Item = &'a Int<M>>>(
        iterable: I,
    ) -> Vec<Self> {
        iterable
            .into_iter()
            .map(Self::from)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use crypto_bigint::{U128, Uint, const_monty_params};
    use num_traits::{One, Zero};

    use super::*;

    const_monty_params!(ModP, U128, "7fffffffffffffffffffffffffffffff");
    type F = ConstMontyField<ModP, { U128::LIMBS }>;

    #[test]
    fn from_unsigned_and_signed() {
        assert_eq!(F::from(0u64), F::zero());
        assert_eq!(F::from(1u32), F::one());
        assert_eq!(F::from(-1i32) + F::one(), F::zero());
        assert_eq!(F::from(-5i64) + F::from(5u64), F::zero());
        assert_eq!(F::from(42usize), F::from(42u64));
        assert_eq!(F::from(-42isize) + F::from(42usize), F::zero());
    }

    #[test]
    fn from_bool() {
        assert_eq!(F::from(true), F::one());
        assert_eq!(F::from(false), F::zero());
    }

    #[test]
    fn from_uint() {
        let u: Uint<{ U128::LIMBS }> = Uint::from(123u64);
        let f: F = u.into();
        assert_eq!(f, F::from(123u64));
    }

    #[test]
    fn from_ref_generic() {
        let x: u64 = 77;
        let f1: F = ConstMontyField::from(x);
        let f2: F = ConstMontyField::from(&x);
        assert_eq!(f1, f2);
    }
}

#[cfg(test)]
mod prop_tests {
    use crypto_bigint::{U256, const_monty_params};
    use num_traits::{One, Zero};
    use proptest::prelude::*;

    use crate::field::F256;

    const_monty_params!(
        ModQ,
        U256,
        "00dca94d8a1ecce3b6e8755d8999787d0524d8ca1ea755e7af84fb646fa31f27"
    );
    type G = F256<ModQ>;

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
            let f = G::from(x);
            let mut acc = G::zero();
            for i in 0..128 {
                if (x >> i) & 1 == 1 { acc += G::from(1u64) * G::from(1u64 << i.min(63)); }
            }
            let u = crypto_bigint::Uint::<{ U256::LIMBS }>::from(x);
            let g2: G = G::from(u);
            prop_assert_eq!(f, g2);
        }

        #[test]
        fn prop_from_signed_is_neg_of_abs_when_negative(x in any_i128()) {
            let f = G::from(x);
            let abs = x.unsigned_abs();
            let g_abs = G::from(abs);
            if x < 0 {
                prop_assert_eq!(f + g_abs, G::zero());
            } else {
                prop_assert_eq!(f, g_abs);
            }
        }

        #[test]
        fn prop_from_bool_is_identity(b in any_bool()) {
            let f = G::from(b);
            prop_assert_eq!(f, if b { G::one() } else { G::zero() });
        }

        #[test]
        fn prop_from_usize_and_u64_agree(x in 0u64..=(usize::MAX as u64)) {
            // On 32-bit platforms, only compare when x fits into usize
            if usize::BITS < 64 && x > (usize::MAX as u64) {
                prop_assume!(false);
            }
            let a: G = G::from(x as usize);
            let b: G = G::from(x);
            prop_assert_eq!(a, b);
        }

        #[test]
        fn prop_from_isize_neg_behaves(x in (isize::MIN as i64)..=(isize::MAX as i64)) {
            // On 32-bit platforms, only compare when x fits into isize
            if isize::BITS < 64 && (x < isize::MIN as i64 || x > isize::MAX as i64) {
                prop_assume!(false);
            }
            let a: G = G::from(x as isize);
            let b_abs: G = G::from(x.unsigned_abs());
            if x < 0 { prop_assert_eq!(a + b_abs, G::zero()); } else { prop_assert_eq!(a, b_abs); }
        }

        #[test]
        fn prop_from_uint_roundtrip_through_uint(x in any_u128()) {
            let u: crypto_bigint::Uint<{ U256::LIMBS }> = crypto_bigint::Uint::from(x);
            let g_from_uint: G = u.into();
            let g_direct: G = G::from(x);
            prop_assert_eq!(g_from_uint, g_direct);
        }

        #[test]
        fn prop_from_ref_generic_matches_owned(x in any::<u64>()) {
            let a: G = G::from(x);
            let b: G = G::from(&x);
            prop_assert_eq!(a, b);
        }
    }
}
