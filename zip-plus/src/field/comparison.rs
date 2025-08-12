use ark_ff::{One, Zero};

use crate::{
    field::{
        RandomField,
        BigInt,
    },
};
use crate::field::config::FieldConfig;

impl<const N: usize, FC: FieldConfig<BigInt<N>>> PartialEq for RandomField<N, FC> {
    fn eq(&self, other: &Self) -> bool {
        if self.is_one() & other.is_one() {
            return true;
        }
        if self.is_zero() && other.is_zero() {
            return true;
        }

        self.value == other.value
    }
}

impl<const N: usize, FC: FieldConfig<BigInt<N>>> Eq for RandomField<N, FC> {} // Eq requires PartialEq and ensures reflexivity.
