/// This trait is essentially equivalent to `From<&T>`, other than it allows us
/// to implement it for external types that don't implement it out of the box,
/// most notably primitive types.
pub trait FromRef<T> {
    fn from_ref(t: &T) -> Self;
}

macro_rules! impl_from_ref_for_primitive {
    ($dst:ty, [$($src:ty),+]) => {
        $(
            impl FromRef<$src> for $dst {
                fn from_ref(t: &$src) -> Self {
                    <$dst>::from(*t)
                }
            }
        )+
    };
}

impl_from_ref_for_primitive!(u128, [u64, u32, u16, u8]);
impl_from_ref_for_primitive!(u64, [u32, u16, u8]);
impl_from_ref_for_primitive!(u32, [u16, u8]);
impl_from_ref_for_primitive!(u16, [u8]);

impl_from_ref_for_primitive!(i128, [i64, i32, i16, i8]);
impl_from_ref_for_primitive!(i64, [i32, i16, i8]);
impl_from_ref_for_primitive!(i32, [i16, i8]);
impl_from_ref_for_primitive!(i16, [i8]);

impl_from_ref_for_primitive!(f64, [f32]);
