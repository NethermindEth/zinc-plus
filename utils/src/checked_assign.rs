use crypto_primitives::{crypto_bigint_int::Int, crypto_bigint_uint::Uint};
use num_traits::CheckedAdd;

pub trait CheckedAddAssign {
    fn checked_add_assign(&mut self, rhs: &Self) -> Option<()>;
}

macro_rules! impl_checked_add_assign_of_checked_add {
($($t:ty),*) => {
    $(
        impl CheckedAddAssign for $t {
            #[inline(always)]
            fn checked_add_assign(&mut self, rhs: &$t) -> Option<()> {
                *self = <$t as CheckedAdd>::checked_add(self, rhs)?;

                Some(())
            }
        }
    )*
};
}

impl_checked_add_assign_of_checked_add!(i8, u8, i16, u16, i32, u32, i64, u64, i128, u128);

impl<const LIMBS: usize> CheckedAddAssign for Int<LIMBS> {
    #[inline(always)]
    fn checked_add_assign(&mut self, rhs: &Self) -> Option<()> {
        *self = self.checked_add(rhs)?;

        Some(())
    }
}

impl<const LIMBS: usize> CheckedAddAssign for Uint<LIMBS> {
    fn checked_add_assign(&mut self, rhs: &Self) -> Option<()> {
        *self = self.checked_add(rhs)?;

        Some(())
    }
}
