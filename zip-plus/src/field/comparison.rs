use ark_ff::{One, Zero};

use crate::{
    field::{
        RandomField,
    },
    traits::Field,
};

impl<const N: usize> PartialEq for RandomField<'_, N> {
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

impl<const N: usize> Eq for RandomField<'_, N> {} // Eq requires PartialEq and ensures reflexivity.
