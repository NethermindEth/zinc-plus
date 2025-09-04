use crate::{
    traits::{BigInteger, Integer, PrimitiveConversion, Uinteger},
};
pub use crate::field::biginteger::{Words};
use ark_std::{Zero};
use crypto_primitives::PrimeField;
use crate::field::{BigInt, FieldConfig, RandomField};
use crate::field::Uint;
use crate::field::Int;
use crate::pcs::utils::AsWords;

macro_rules! impl_from_int {
    ($t:ty) => {
        impl<const N: usize, FC: FieldConfig<BigInt<N>>> From<$t> for RandomField<N, FC> {
            fn from(value: $t) -> Self {
                let abs_value = value.abs_diff(0);
                let mut words = Words::<N>::default();

                words[0] = PrimitiveConversion::from_primitive(abs_value);

                if ark_std::mem::size_of::<$t>().div_ceil(8) > 1 && N > 1 {
                    words[1] =
                        PrimitiveConversion::from_primitive(u128::from_primitive(abs_value) >> 64);
                }
                let mut int_value = Uint::<N>::from_words(words).as_int();
                let modulus = Int::<N>::from_words(FC::modulus().to_words());

                int_value %= modulus;
                let mut bigint_value = BigInt::<N>::from(int_value);
                FC::mul_assign(&mut bigint_value, &FC::r_squared());

                let mut r = Self::new_unchecked(bigint_value);

                if value < <$t>::zero() {
                    r = -r;
                }

                r
            }
        }
    };
}

impl_from_int!(i8);
impl_from_int!(u8);
impl_from_int!(i16);
impl_from_int!(u16);
impl_from_int!(i32);
impl_from_int!(u32);
impl_from_int!(i64);
impl_from_int!(u64);
impl_from_int!(i128);
impl_from_int!(u128);
impl_from_int!(isize);
impl_from_int!(usize);

// Implementation for bool
impl<const N: usize, FC: FieldConfig<BigInt<N>>> From<bool> for RandomField<N, FC> {
    fn from(value: bool) -> Self {
        let mut r = BigInt::from(value as u64);
        FC::mul_assign(&mut r, &FC::r_squared());
        Self::new_unchecked(r)
    }
}

impl<const N: usize, FC: FieldConfig<BigInt<N>>> From<&bool> for RandomField<N, FC> {
    fn from(value: &bool) -> Self {
        Self::from(*value)
    }
}

// Implementation of From for BigInt<N>
impl<const N: usize, FC: FieldConfig<BigInt<N>>, const M: usize> From<BigInt<M>> for RandomField<N, FC> {
    fn from(value: BigInt<M>) -> Self {
        Self::from(Int::<M>::from(value))
    }
}

// Implementation for Int<N>
impl<const N: usize, FC: FieldConfig<BigInt<N>>, const M: usize> From<Int<M>> for RandomField<N, FC> {
    fn from(mut value: Int<M>) -> Self {
        let modulus = FC::modulus();
        let modulus: Int<M> = (&modulus).into();

        let mut bigint_value = if M > N {
            value %= modulus;
            BigInt::<N>::from(value)
        } else {
            let mut int_value: Int<N> = (&value).into();
            let modulus: Int<N> = modulus.as_words().try_into().expect("modulus is too large");
            int_value %= modulus;
            int_value.into()
        };

        FC::mul_assign(&mut bigint_value, &FC::r_squared());

        Self::new_unchecked(bigint_value)
    }
}
impl<const N: usize, FC: FieldConfig<BigInt<N>>, const M: usize> From<&Int<M>> for RandomField<N, FC> {
    fn from(value: &Int<M>) -> Self {
        Self::from(*value)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        define_field_config,
        field::{BigInt, FieldConfig, RandomField},
    };
    use ark_std::{fmt::Debug, format, str::FromStr};
    use num_traits::ConstZero;
    use crate::field::FieldConfigBase;

    define_field_config!(Fc23, "23");
    define_field_config!(FcSmall, "243043087159742188419721163456177567");
    define_field_config!(
        FcBig,
        "3618502788666131213697322783095070105623107215331596699973092056135872020481"
    );

    fn test_from<const N: usize, FC: FieldConfig<BigInt<N>>, T: Clone>(value: T, value_str: &str)
    where
        RandomField<N, FC>: From<T>,
        RandomField<N, FC>: From<BigInt<N>>,
    {
        let raw_element = RandomField::<N, FC>::from(value);
        assert_eq!(
            raw_element,
            RandomField::<N, FC>::from(BigInt::<N>::from_str(value_str).unwrap())
        )
    }

    #[test]
    fn converts_u128_to_random_field() {
        test_from::<2, FcSmall<2>, u128>(
            243043087159742188419721163456177516,
            "243043087159742188419721163456177516",
        );
    }

    #[test]
    fn converts_u64_to_random_field() {
        test_from::<2, FcSmall<2>, u64>(23, "23");
    }

    #[test]
    fn converts_u32_to_random_field() {
        test_from::<2, FcSmall<2>, u32>(23, "23");
    }

    #[test]
    fn converts_u16_to_random_field() {
        test_from::<2, FcSmall<2>, u16>(23, "23");
    }

    #[test]
    fn converts_u8_to_random_field() {
        test_from::<2, FcSmall<2>, u8>(23, "23");
    }

    #[test]
    fn converts_false_to_zero() {
        test_from::<2, FcSmall<2>, bool>(false, "0");
    }

    #[test]
    fn converts_true_to_one() {
        test_from::<2, FcSmall<2>, bool>(true, "1");
    }

    macro_rules! test_signed_type_full_range {
        ($type:ty, $N:expr, $FC:ty) => {{
            let modulus = <$FC as FieldConfigBase<_>>::modulus();

            // Test full range for primitive types
            for x in <$type>::MIN..=<$type>::MAX {
                let result: RandomField<$N, $FC> = x.into();
                let expected = if x < 0 {
                    BigInt::<$N>::from((modulus as i64 + x as i64) as u64)
                } else {
                    BigInt::<$N>::from(x as u64)
                };
                assert_eq!(
                    result.into_bigint(),
                    expected,
                    "conversion failed for value: {}",
                    x
                );
            }
        }};
    }

    macro_rules! test_signed_type_edge_cases {
        ($type:ty, $N:expr, $FC:ty) => {{
            let modulus = <$FC as FieldConfigBase>::modulus();

            // Test zero
            let zero = <$type>::from_str("0").unwrap();
            let zero_result: RandomField<$N, $FC> = zero.into();
            assert_eq!(
                zero_result.into_bigint(),
                BigInt::<$N>::ZERO,
                "Zero value should map to field zero"
            );

            // Test maximum value
            let max = <$type>::from_str(&format!("{}", <$type>::MAX)).unwrap();
            let max_result: RandomField<$N, $FC> = max.into();
            assert!(
                max_result.into_bigint() < modulus,
                "Maximum value should be less than field modulus"
            );

            // Test minimum value
            let min = -<$type>::from_str(&format!("{}", <$type>::MAX)).unwrap();
            let min_f = RandomField::<$N, $FC>::from(&min);
            assert!(
                min_f.into_bigint() < modulus,
                "Minimum value should wrap to valid field element"
            );

            // Test positive boundary
            let pos = <$type>::from_str("5").unwrap();
            let pos_result: RandomField<$N, $FC> = pos.into();
            assert_eq!(
                pos_result.into_bigint(),
                BigInt::<$N>::from(5u64),
                "Positive value should map directly to field"
            );

            // Test negative boundary
            let neg = <$type>::from_str("-5").unwrap();
            let neg_result = RandomField::<$N, $FC>::from(neg);
            assert_eq!(
                neg_result.into_bigint(),
                BigInt::<$N>::from((i64::parse(modulus.to_string()) - 5) as u64),
                "Negative value should wrap around field modulus"
            );

            // Test reference conversions
            let ref_zero = RandomField::<$N, $FC>::from(&zero);
            assert_eq!(
                ref_zero.into_bigint(),
                BigInt::<$N>::ZERO,
                "Reference to zero should map to field zero"
            );

            let ref_max = RandomField::<$N, $FC>::from(&max);
            assert!(
                ref_max.into_bigint() < modulus,
                "Reference to maximum value should be less than field modulus"
            );

            let ref_min = RandomField::<$N, $FC>::from(&min);
            assert!(
                ref_min.into_bigint() < modulus,
                "Reference to minimum value should wrap to valid field element"
            );
        }};
    }

    #[test]
    fn test_signed_integers_field_map() {
        define_field_config!(Fc, "18446744069414584321");

        // Test primitive types with full range
        test_signed_type_full_range!(i8, 1, Fc<1>);
        test_signed_type_full_range!(i16, 1, Fc<1>);

        // Test larger primitive types with edge cases only
        test_signed_type_edge_cases!(i32, 1, Fc<1>);
        test_signed_type_edge_cases!(i64, 1, Fc<1>);
        test_signed_type_edge_cases!(i128, 1, Fc<1>);
    }

    macro_rules! test_unsigned_type_full_range {
        ($type:ty, $N:expr, $FC:ty) => {{
            // Test full range for small unsigned types
            for x in <$type>::MIN..=<$type>::MAX {
                let result: RandomField<$N, $FC> = x.into();
                let ref_result: RandomField<$N, $FC> = (&x).into();
                let expected = BigInt::<$N>::from(x as u64);
                assert_eq!(
                    result.into_bigint(),
                    expected,
                    "conversion failed for value: {}",
                    x
                );
                assert_eq!(
                    ref_result.into_bigint(),
                    expected,
                    "reference conversion failed for value: {}",
                    x
                );
            }
        }};
    }

    macro_rules! test_unsigned_type_edge_cases {
        ($type:ty, $N:expr, $FC:ty) => {{
            let modulus = <$FC as FieldConfigBase>::modulus();

            // Test zero
            let zero = <$type>::MIN;
            let zero_result: RandomField<$N, $FC> = zero.into();
            assert_eq!(
                zero_result.into_bigint(),
                BigInt::<$N>::ZERO,
                "Zero value should map to field zero"
            );

            // Test maximum value
            let max = <$type>::MAX;
            let max_result: RandomField<$N, $FC> = max.into();
            assert!(
                max_result.into_bigint() < modulus,
                "Maximum value should be less than field modulus"
            );

            // Test boundary value - using literal instead of From
            let boundary: $type = 5;
            let boundary_result: RandomField<$N, $FC> = boundary.into();
            assert_eq!(
                boundary_result.into_bigint(),
                BigInt::<$N>::from(5u64),
                "Boundary value should map directly to field"
            );

            // Test reference conversions
            let ref_zero: RandomField<$N, $FC> = (&zero).into();
            assert_eq!(
                ref_zero.into_bigint(),
                BigInt::<$N>::ZERO,
                "Reference to zero should map to field zero"
            );

            let ref_max: RandomField<$N, $FC> = (&max).into();
            assert!(
                ref_max.into_bigint() < modulus,
                "Reference to maximum value should be less than field modulus"
            );
        }};
    }

    #[test]
    fn test_unsigned_integers_field_map() {
        define_field_config!(Fc, "18446744069414584321");

        // Test small types with full range
        test_unsigned_type_full_range!(u8, 1, Fc<1>);
        test_unsigned_type_full_range!(u16, 1, Fc<1>);

        // Test larger types with edge cases only
        test_unsigned_type_edge_cases!(u32, 1, Fc<1>);
        test_unsigned_type_edge_cases!(u64, 1, Fc<1>);
        test_unsigned_type_edge_cases!(u128, 1, Fc<1>);
    }
}

#[cfg(test)]
mod bigint_field_map_tests {
    use super::*;
    use crate::{
        big_int, define_field_config,
        field::{BigInt, RandomField},
    };
    use num_traits::ConstZero;

    define_field_config!(Fc, "18446744069414584321");

    #[test]
    fn test_bigint_smaller_than_field() {
        // Using a 2-limb field config with 1-limb BigInt
        let small_bigint = BigInt::<1>::from(12345u64);
        let result: RandomField<2, Fc<2>> = small_bigint.into();

        assert_eq!(
            result.into_bigint().first(),
            12345u64,
            "Small BigInt should be preserved in larger field"
        );
    }

    #[test]
    fn test_bigint_equal_size() {
        let value = big_int!(12345678901234567890, 2);
        let result: RandomField<2, Fc<2>> = value.into();

        // The result should be the value modulo the field modulus
        let expected = big_int!(12345678901234567890);
        assert_eq!(
            result.into_bigint(),
            expected,
            "Equal size BigInt should be correctly converted"
        );
    }

    #[test]
    fn test_bigint_larger_than_field() {
        // Using a 1-limb field config with 2-limb BigInt
        let large_value = big_int!(123456789012345678901, 2);
        let result: RandomField<1, Fc<1>> = large_value.into();

        let expected = BigInt::<1>::from(12776324595858172975u64);
        assert_eq!(
            result.into_bigint(),
            expected,
            "Larger BigInt should be correctly reduced modulo field modulus"
        );
    }

    #[test]
    fn test_bigint_zero() {
        let zero = BigInt::<2>::ZERO;
        let result: RandomField<2, Fc<2>> = zero.into();

        assert!(
            result.into_bigint().is_zero(),
            "Zero BigInt should map to zero field element"
        );
    }

    #[test]
    fn test_bigint_max_value() {
        // Create a BigInt with all bits set to 1
        let max_value = BigInt::from([u64::MAX, u64::MAX]);

        let result: RandomField<2, Fc<2>> = max_value.into();

        assert!(
            result.into_bigint() < Fc::modulus(),
            "Result should be properly reduced modulo field modulus"
        );
    }
}
