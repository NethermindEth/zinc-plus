use ark_ff::{One, Zero};
use ark_std::{
    iter::Sum,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use crate::{field::RandomField};
use crate::field::BigInt;
use crate::field::config::FieldConfig;

macro_rules! impl_ops {
    (
        impl($($gen:tt)*) for $type:ty,
        $trait:ident, $op:ident,
        $trait_assign:ident, $op_assign:ident
    ) => {
        impl<$($gen)*> $trait<Self> for $type {
            type Output = Self;

            fn $op(mut self, rhs: Self) -> Self::Output {
                self.$op_assign(&rhs);
                self
            }
        }

        impl<$($gen)*> $trait<&Self> for $type {
            type Output = Self;

            fn $op(mut self, rhs: &Self) -> Self::Output {
                self.$op_assign(rhs);
                self
            }
        }

        impl<$($gen)*> $trait<Self> for &$type {
            type Output = $type;

            fn $op(self, rhs: Self) -> Self::Output {
                let mut res = self.clone();
                res.$op_assign(rhs);
                res
            }
        }

        impl<$($gen)*> $trait<$type> for &$type {
            type Output = $type;

            fn $op(self, rhs: $type) -> Self::Output {
                let mut res = self.clone();
                res.$op_assign(&rhs);
                res
            }
        }

        impl<$($gen)*> $trait_assign<Self> for $type {
            fn $op_assign(&mut self, rhs: Self) {
                self.$op_assign(&rhs);
            }
        }
    };
}

impl_ops!(impl('cfg, const N: usize, FC: FieldConfig<BigInt<N>>) for RandomField<N, FC>, Add, add, AddAssign, add_assign);
impl_ops!(impl('cfg, const N: usize, FC: FieldConfig<BigInt<N>>) for RandomField<N, FC>, Sub, sub, SubAssign, sub_assign);
impl_ops!(impl('cfg, const N: usize, FC: FieldConfig<BigInt<N>>) for RandomField<N, FC>, Mul, mul, MulAssign, mul_assign);
impl_ops!(impl('cfg, const N: usize, FC: FieldConfig<BigInt<N>>) for RandomField<N, FC>, Div, div, DivAssign, div_assign);

impl<const N: usize, FC: FieldConfig<BigInt<N>>> Add<u32> for RandomField<N, FC> {
    type Output = Self;

    fn add(mut self, rhs: u32) -> Self::Output {
        self.add_assign(rhs);
        self
    }
}

impl<const N: usize, FC: FieldConfig<BigInt<N>>> AddAssign<&Self> for RandomField<N, FC> {
    fn add_assign(&mut self, rhs: &Self) {
        FC::add_assign(&mut self.value, &rhs.value);
    }
}

impl<const N: usize, FC: FieldConfig<BigInt<N>>> AddAssign<u32> for RandomField<N, FC> {
    fn add_assign(&mut self, rhs: u32) {
        todo!()
        // self.with_aligned_config_mut(
        //     rhs.into(),
        //     |lhs, rhs, config| {
        //         config.add_assign(lhs, rhs);
        //     },
        //     |lhs, rhs| {
        //         panic!();
        //     },
        // );
    }
}

impl<const N: usize, FC: FieldConfig<BigInt<N>>> SubAssign<&Self> for RandomField<N, FC> {
    fn sub_assign(&mut self, rhs: &Self) {
        FC::sub_assign(&mut self.value, &rhs.value);
    }
}

impl<const N: usize, FC: FieldConfig<BigInt<N>>> MulAssign<&Self> for RandomField<N, FC> {
    fn mul_assign(&mut self, rhs: &Self) {
        FC::mul_assign(&mut self.value, &rhs.value);
    }
}

impl<const N: usize, FC: FieldConfig<BigInt<N>>> DivAssign<&Self> for RandomField<N, FC> {
    fn div_assign(&mut self, rhs: &Self) {
        if rhs.is_zero() {
            panic!("Attempt to divide by zero");
        }
        FC::mul_assign(&mut self.value, &FC::inverse(&rhs.value).unwrap());
    }
}

impl<const N: usize, FC: FieldConfig<BigInt<N>>> DivAssign<&mut Self> for RandomField<N, FC> {
    fn div_assign(&mut self, rhs: &mut Self) {
        *self /= rhs.clone();
    }
}

impl<const N: usize, FC: FieldConfig<BigInt<N>>> Neg for RandomField<N, FC> {
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        if self.is_zero() {
            return self;
        }

        let tmp = self.value;
        self.value = FC::modulus();
        self.value.sub_with_borrow(&tmp);

        self
    }
}

impl<'a, const N: usize, FC: FieldConfig<BigInt<N>>> Sum<&'a Self> for RandomField<N, FC> {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |mut acc, x| {
            acc.add_assign(x);
            acc
        })
    }
}

impl<const N: usize, FC: FieldConfig<BigInt<N>>> Sum for RandomField<N, FC> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |mut acc, x| {
            acc.add_assign(&x);
            acc
        })
    }
}

impl<'a, const N: usize, FC: FieldConfig<BigInt<N>>> core::iter::Product<&'a Self> for RandomField<N, FC> {
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::one(), core::ops::Mul::mul)
    }
}

impl<'a, const N: usize, FC: FieldConfig<BigInt<N>>> From<&'a Self> for RandomField<N, FC> {
    fn from(value: &'a Self) -> Self {
        value.clone()
    }
}

#[cfg(test)]
mod test {
    use ark_ff::{One, Zero};

    use crate::{big_int, define_field_config, field::{biginteger::BigInt, RandomField}, random_field};
    use crate::field::config::FieldConfig;

    define_field_config!(Fc23, "23");

    #[test]
    fn test_add_wrapping_around_modulus() {
        let lhs: RandomField<1, Fc23<1>> = random_field!(22u32);
        let rhs: RandomField<1, Fc23<1>> = random_field!(2u32);

        let sum = lhs + rhs;
        assert_eq!(sum.into_bigint(), BigInt::one());
    }

    #[test]
    fn test_add_without_wrapping() {
        let lhs: RandomField<1, Fc23<1>> = random_field!(20u32);
        let rhs: RandomField<1, Fc23<1>> = random_field!(20u32);

        let sum = lhs + rhs;
        assert_eq!(sum.into_bigint(), big_int!(17));
    }

    #[test]
    fn test_add_one() {
        let lhs: RandomField<1, Fc23<1>> = random_field!(22u32);
        let rhs: RandomField<1, Fc23<1>> = RandomField::one();

        let sum = lhs + rhs;
        assert_eq!(sum.into_bigint(), BigInt::zero());
    }

    #[test]
    fn test_add_two_ones() {
        let lhs: RandomField<1, Fc23<1>> = RandomField::one();
        let rhs = RandomField::one();
        let expected = RandomField::<1, Fc23<1>>::from_bigint(BigInt::from(2_u32)).unwrap();

        assert_eq!(lhs + rhs, expected);
    }

    #[test]
    fn test_sub_wrapping_around_modulus() {
        let lhs: RandomField<1, Fc23<1>> = random_field!(2u32);
        let rhs: RandomField<1, Fc23<1>> = random_field!(22u32);

        let difference = lhs - rhs;
        assert_eq!(difference.into_bigint(), big_int!(3));
    }

    #[test]
    fn test_sub_identical_values_results_in_zero() {
        let lhs: RandomField<1, Fc23<1>> = random_field!(20u32);
        let rhs: RandomField<1, Fc23<1>> = random_field!(20u32);

        let difference = lhs - rhs;
        assert_eq!(difference.into_bigint(), BigInt::zero());
    }

    #[test]
    fn test_init_sub_raw() {
        let lhs: RandomField<1, Fc23<1>> = random_field!(2u32);
        let rhs = RandomField::one();
        let res = lhs.clone() - rhs;
        let mut expected = lhs;
        expected.set_one();
        assert_eq!(res, expected)
    }

    #[test]
    fn test_sub_assign_works() {
        let mut lhs: RandomField<1, Fc23<1>> = random_field!(10u32);
        let rhs: RandomField<1, Fc23<1>> = random_field!(7u32);

        lhs -= rhs;

        assert_eq!(lhs.into_bigint(), big_int!(3));
    }

    #[test]
    fn test_sub_assign_wraps_modulus() {
        let mut lhs: RandomField<1, Fc23<1>> = random_field!(3u32);
        let rhs: RandomField<1, Fc23<1>> = random_field!(7u32);

        lhs -= rhs;

        assert_eq!(lhs.into_bigint(), big_int!(19)); // 3 - 7 mod 23 = 19
    }

    #[test]
    fn test_mul_wraps_modulus() {
        let lhs: RandomField<1, Fc23<1>> = random_field!(22u32);
        let rhs: RandomField<1, Fc23<1>> = random_field!(2u32);

        let product = lhs * rhs;
        assert_eq!(product.into_bigint(), big_int!(21));
    }

    #[test]
    fn test_mul_without_wrapping() {
        let lhs: RandomField<1, Fc23<1>> = random_field!(20u32);
        let rhs: RandomField<1, Fc23<1>> = random_field!(20u32);

        let product = lhs * rhs;
        assert_eq!(product.into_bigint(), big_int!(9));
    }

    #[test]
    fn test_left_mul_by_zero() {
        let lhs: RandomField<1, Fc23<1>> = random_field!(22u32);
        let rhs = RandomField::zero();

        let product = lhs * rhs;
        assert!(product.is_zero());
    }

    #[test]
    fn test_right_mul_by_zero() {
        let lhs: RandomField<1, Fc23<1>> = RandomField::zero();
        let rhs: RandomField<1, Fc23<1>> = random_field!(22u32);

        let product = lhs * rhs;
        assert!(product.is_zero());
    }

    #[test]
    fn test_mul_assign_works() {
        let mut lhs: RandomField<1, Fc23<1>> = random_field!(5u32);
        let rhs: RandomField<1, Fc23<1>> = random_field!(4u32);

        lhs *= rhs;

        assert_eq!(lhs.into_bigint(), big_int!(20));
    }

    #[test]
    fn test_mul_assign_wraps_modulus() {
        let mut lhs: RandomField<1, Fc23<1>> = random_field!(6u32);
        let rhs: RandomField<1, Fc23<1>> = random_field!(4u32);

        lhs *= rhs;

        assert_eq!(lhs.into_bigint(), big_int!(1)); // 6 * 4 mod 23 = 1
    }

    #[test]
    fn test_div_wraps_modulus() {
        let lhs: RandomField<1, Fc23<1>> = random_field!(22u32);
        let rhs: RandomField<1, Fc23<1>> = random_field!(2u32);

        let quotient = lhs / rhs;
        assert_eq!(quotient.into_bigint(), big_int!(11));
    }

    #[test]
    fn test_div_identical_values_results_in_one() {
        let lhs: RandomField<1, Fc23<1>> = random_field!(20u32);
        let rhs: RandomField<1, Fc23<1>> = random_field!(20u32);

        let quotient = lhs / rhs;
        assert_eq!(quotient.into_bigint(), big_int!(1));
    }

    #[test]
    fn test_div_without_wrapping() {
        let lhs: RandomField<1, Fc23<1>> = random_field!(17u32);
        let rhs: RandomField<1, Fc23<1>> = random_field!(4u32);

        let quotient = lhs / rhs;
        assert_eq!(quotient.into_bigint(), big_int!(10));
    }

    #[test]
    #[should_panic]
    fn test_div_by_zero_should_panic() {
        let lhs: RandomField<1, Fc23<1>> = random_field!(17u32);
        let rhs: RandomField<1, Fc23<1>> = random_field!(0u32);

        let _sum = lhs / rhs;
    }

    #[test]
    fn test_div_bigint256() {
        define_field_config!(Fc, "695962179703626800597079116051991347");

        let a: RandomField<4, Fc<4>> = random_field!(3u32);
        let mut b = RandomField::one();
        b /= a;
        assert_eq!(
            b.into_bigint(),
            big_int!(231987393234542266865693038683997116)
        );

        let a: RandomField<4, Fc<4>> = random_field!(19382769832175u64);

        let b: RandomField<4, Fc<4>> = random_field!(97133987132135u64);

        assert_eq!(
            big_int!(243043087159742188419721163456177516),
            (b / a).into_bigint()
        );
    }

    #[test]
    fn test_div_by_reference_works() {
        let lhs: RandomField<1, Fc23<1>> = random_field!(15u32);
        let rhs = random_field!(3u32);

        #[allow(clippy::op_ref)] // This implementation could be removed?
        let quotient = lhs / &rhs;

        assert_eq!(quotient.into_bigint(), big_int!(5));
    }

    #[test]
    fn test_div_by_mutable_reference_works() {
        let lhs: RandomField<1, Fc23<1>> = random_field!(9u32);
        let rhs = random_field!(3u32);

        #[allow(clippy::op_ref)] // This implementation could be removed?
        let quotient = lhs / &rhs;

        assert_eq!(quotient.into_bigint(), big_int!(3));
    }

    #[test]
    fn test_div_assign_works() {
        let mut lhs: RandomField<1, Fc23<1>> = random_field!(15u32);
        let rhs: RandomField<1, Fc23<1>> = random_field!(3u32);

        lhs /= rhs;

        assert_eq!(lhs.into_bigint(), big_int!(5));
    }

    #[test]
    #[should_panic(expected = "Attempt to divide by zero")]
    fn test_div_assign_by_zero_should_panic() {
        let mut lhs: RandomField<1, Fc23<1>> = random_field!(15u32);
        let rhs = RandomField::zero();

        lhs /= rhs;
    }

    #[test]
    fn test_div_assign_by_mutable_reference() {
        let mut lhs: RandomField<1, Fc23<1>> = random_field!(18u32);
        let mut rhs = random_field!(3u32);

        lhs /= &mut rhs;

        assert_eq!(lhs.into_bigint(), big_int!(6)); // 18 / 3 mod 23 = 6
    }

    #[test]
    fn test_neg_large_value() {
        let operand: RandomField<1, Fc23<1>> = random_field!(22u32);
        let negated = -operand;

        assert_eq!(negated.into_bigint(), big_int!(1));
    }

    #[test]
    fn test_neg_mid_value() {
        let operand: RandomField<1, Fc23<1>> = random_field!(17u32);
        let negated = -operand;

        assert_eq!(negated.into_bigint(), big_int!(6));
    }

    #[test]
    fn test_neg_zero() {
        let operand: RandomField<1, Fc23<1>> = random_field!(0u32);
        let negated = -operand;

        assert_eq!(negated.into_bigint(), BigInt::zero());
    }

    #[test]
    fn test_sum_of_multiple_values() {
        let values = [
            random_field!(2u32),
            random_field!(4u32),
            random_field!(6u32),
        ];

        let sum: RandomField<1, Fc23<1>> = values.iter().sum();

        assert_eq!(sum.into_bigint(), big_int!(12));
    }

    #[test]
    fn test_sum_with_zero() {
        let values = [
            RandomField::zero(),
            random_field!(5u32),
            random_field!(7u32),
        ];

        let sum: RandomField<1, Fc23<1>> = values.iter().sum();

        assert_eq!(sum.into_bigint(), big_int!(12));
    }

    #[test]
    fn test_sum_wraps_modulus() {
        let values = [
            random_field!(10u32),
            random_field!(15u32),
            random_field!(21u32),
        ];

        let sum: RandomField<1, Fc23<1>> = values.iter().sum();

        assert_eq!(sum.into_bigint(), big_int!(0));
    }

    #[test]
    fn test_sum_empty_iterator() {
        let sum: RandomField<1, Fc23<1>> = ark_std::iter::empty::<&RandomField<1, Fc23<1>>>().sum();
        assert!(sum.is_zero()); // Empty sum should return zero
    }

    #[test]
    fn test_sum_single_element() {
        let values = [random_field!(9u32)];

        let sum: RandomField<1, Fc23<1>> = values.iter().sum();

        assert_eq!(sum.into_bigint(), big_int!(9));
    }

    #[test]
    fn test_sum_with_modulus_wrapping() {
        let values = [random_field!(12u32), random_field!(15u32)];

        let sum: RandomField<1, Fc23<1>> = values.iter().sum();

        assert_eq!(sum.into_bigint(), big_int!(4));
    }

    #[test]
    fn test_product_of_multiple_values() {
        let values = [
            random_field!(2u32),
            random_field!(4u32),
            random_field!(6u32),
        ];

        let product: RandomField<1, Fc23<1>> = values.iter().product();

        assert_eq!(product.into_bigint(), big_int!(2));
    }

    #[test]
    fn test_product_with_one() {
        let values = [
            RandomField::one(),
            random_field!(5u32),
            random_field!(7u32),
        ];

        let product: RandomField<1, Fc23<1>> = values.iter().product();

        assert_eq!(product.into_bigint(), big_int!(12));
    }

    #[test]
    fn test_product_with_zero() {
        let values = [
            random_field!(3u32),
            RandomField::zero(),
            random_field!(9u32),
        ];

        let product: RandomField<1, Fc23<1>> = values.iter().product();

        assert!(product.is_zero());
    }

    #[test]
    fn test_product_negative_modular_complements() {
        let values = [
            random_field!(10u32),
            random_field!(15u32),
            random_field!(21u32),
        ];

        let product: RandomField<1, Fc23<1>> = values.iter().product();

        assert_eq!(product.into_bigint(), big_int!(22));
    }

    #[test]
    fn test_product_empty_iterator() {
        let product: RandomField<1, Fc23<1>> = ark_std::iter::empty::<&RandomField<1, Fc23<1>>>().product();
        assert!(product.is_one()); // Empty product should return one
    }

    #[test]
    fn test_product_single_element() {
        let values = [random_field!(9u32)];

        let product: RandomField<1, Fc23<1>> = values.iter().product();

        assert_eq!(product.into_bigint(), big_int!(9));
    }

    #[test]
    fn test_product_with_modulus_wrapping() {
        let values = [random_field!(12u32), random_field!(15u32)];

        let product: RandomField<1, Fc23<1>> = values.iter().product();

        assert_eq!(product.into_bigint(), big_int!(19));
    }
}
