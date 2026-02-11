use crypto_primitives::{boolean::Boolean, crypto_bigint_int::Int, crypto_bigint_uint::Uint};
use num_traits::{ConstOne, ConstZero};

//
// FromRef
//

/// This trait is essentially equivalent to `From<&T>`, other than it allows us
/// to implement it for external types that don't implement it out of the box,
/// most notably primitive types.
pub trait FromRef<T> {
    fn from_ref(value: &T) -> Self;
}

macro_rules! impl_from_ref_for_primitive {
    ($dst:ty, [$($src:ty),+]) => {
        $(
            impl FromRef<$src> for $dst {
                fn from_ref(value: &$src) -> Self {
                    <$dst>::from(*value)
                }
            }
        )+
    };
}

impl_from_ref_for_primitive!(i128, [i128, i64, i32, i16, i8]);
impl_from_ref_for_primitive!(i64, [i64, i32, i16, i8]);
impl_from_ref_for_primitive!(i32, [i32, i16, i8]);
impl_from_ref_for_primitive!(i16, [i16, i8]);
impl_from_ref_for_primitive!(i8, [i8]);

macro_rules! impl_from_boolean_ref_for_primitive {
    ($($dst:ty),+) => {
        $(
            impl FromRef<Boolean> for $dst {
                fn from_ref(value: &Boolean) -> Self {
                    if **value {
                        ConstOne::ONE
                    } else {
                        ConstZero::ZERO
                    }
                }
            }
        )+
    };
}

impl_from_boolean_ref_for_primitive!(i8, i16, i32, i64, i128);
impl_from_boolean_ref_for_primitive!(u8, u16, u32, u64, u128);

macro_rules! impl_int_from_primitive_ref {
    ($($t:ty),+) => {
        $(
            impl<const LIMBS: usize> FromRef<$t> for Int<LIMBS> {
                #[inline(always)]
                fn from_ref(value: &$t) -> Self {
                    Self::from(*value)
                }
            }
        )+
    };
}

impl_int_from_primitive_ref!(i8, i16, i32, i64, i128);

impl<const LIMBS: usize> FromRef<Boolean> for Int<LIMBS> {
    fn from_ref(value: &Boolean) -> Self {
        if **value {
            ConstOne::ONE
        } else {
            ConstZero::ZERO
        }
    }
}

impl FromRef<Boolean> for Boolean {
    fn from_ref(value: &Boolean) -> Self {
        *value
    }
}


impl<const LIMBS: usize, const LIMBS2: usize> FromRef<Int<LIMBS2>> for Int<LIMBS> {
    #[inline]
    fn from_ref(value: &Int<LIMBS2>) -> Self {
        Self::try_from(value.inner()).expect("Destination Int type is too small")
    }
}

impl<const LIMBS: usize, const LIMBS2: usize> FromRef<Uint<LIMBS2>> for Uint<LIMBS> {
    #[inline]
    fn from_ref(value: &Uint<LIMBS2>) -> Self {
        Self::try_from(value.inner()).expect("Destination Uint type is too small")
    }
}
