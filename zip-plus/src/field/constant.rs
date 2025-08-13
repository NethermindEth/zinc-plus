use ark_ff::{One, Zero};
use std::marker::PhantomData;
use zeroize::Zeroize;

use crate::{
    field::{RandomField, biginteger::BigInt, config::FieldConfig},
    traits::{Field, FieldMap},
};

impl<const N: usize, FC: FieldConfig<BigInt<N>>> Zero for RandomField<N, FC> {
    fn zero() -> Self {
        RandomField {
            value: BigInt::zero(),
            phantom_data: PhantomData,
        }
    }

    fn set_zero(&mut self) {
        *self.value_mut() = BigInt::zero()
    }

    fn is_zero(&self) -> bool {
        self.value().is_zero()
    }
}

impl<const N: usize, FC: FieldConfig<BigInt<N>>> One for RandomField<N, FC> {
    fn one() -> Self {
        RandomField::new_unchecked(FC::r())
    }

    fn set_one(&mut self) {
        *self.value_mut() = FC::r();
    }

    fn is_one(&self) -> bool {
        self.value == FC::r()
    }
}

impl<const N: usize, FC: FieldConfig<BigInt<N>>> Zeroize for RandomField<N, FC> {
    fn zeroize(&mut self) {
        unsafe { *self = ark_std::mem::zeroed() }
    }
}

#[cfg(test)]
mod tests {
    use ark_ff::{One, Zero};
    use zeroize::Zeroize;

    use crate::{define_field_config, field::RandomField, random_field};

    define_field_config!(FC, "23");

    #[test]
    fn test_zero_creation() {
        let zero_elem = RandomField::<1, FC<1>>::zero();
        assert!(zero_elem.is_zero());
    }

    #[test]
    fn test_set_zero() {
        let mut elem: RandomField<1, FC<1>> = random_field!(7u32);
        elem.set_zero();
        assert!(elem.is_zero());
    }

    #[test]
    fn test_one_creation() {
        let one_elem = RandomField::<1, FC<1>>::one();
        assert!(one_elem.is_one());
    }

    #[test]
    fn test_set_one() {
        let mut elem: RandomField<1, FC<1>> = random_field!(5u32);
        elem.set_one();
        assert!(elem.is_one());
    }

    #[test]
    fn test_set_one_for_raw() {
        let mut raw_field = RandomField::<1, FC<1>>::zero();
        assert!(raw_field.is_zero());

        raw_field.set_one();

        assert!(raw_field.is_one());
    }

    #[test]
    fn test_is_zero_true() {
        let zero_elem = RandomField::<1, FC<1>>::zero();
        assert!(zero_elem.is_zero());
    }

    #[test]
    fn test_is_zero_false() {
        let non_zero_elem = RandomField::<1, FC<1>>::one();
        assert!(!non_zero_elem.is_zero());
    }

    #[test]
    fn test_is_one_true() {
        let one_elem = RandomField::<1, FC<1>>::one();
        assert!(one_elem.is_one());
    }

    #[test]
    fn test_is_one_false() {
        let non_one_elem = RandomField::<1, FC<1>>::from(3u32);
        assert!(!non_one_elem.is_one());
    }

    #[test]
    fn test_zeroize() {
        let mut elem: RandomField<1, FC<1>> = random_field!(12);
        elem.zeroize();
        assert!(elem.is_zero());
    }

    #[test]
    fn test_zero_not_equal_one() {
        let zero_elem = RandomField::<1, FC<1>>::zero();
        let one_elem = RandomField::<1, FC<1>>::one();
        assert_ne!(zero_elem, one_elem);
    }
}
