use ark_ff::{One, Zero};

use crate::{
    field::{
        RandomField,
    },
    traits::Field,
};
use crate::field::config::ConstFieldConfig;

impl<const N: usize, FC: ConstFieldConfig<N>> PartialEq for RandomField<'_, N, FC> {
    fn eq(&self, other: &Self) -> bool {
        if self.is_one() & other.is_one() {
            return true;
        }
        if self.is_zero() && other.is_zero() {
            return true;
        }

        match (self.config, other.config) {
            (Some(_), None) | (None, Some(_)) => false,
            (None, None) => self.value() == other.value(),
            (Some(_), Some(_)) => {
                self.value() == other.value() && self.config_ptr() == other.config_ptr()
            }
        }
    }
}

impl<const N: usize, FC: ConstFieldConfig<N>> Eq for RandomField<'_, N, FC> {} // Eq requires PartialEq and ensures reflexivity.
