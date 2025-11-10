use crypto_primitives::{boolean::Boolean, crypto_bigint_int::Int, crypto_bigint_uint::Uint};

pub trait Named {
    /// Returns the name of the type as a string, used in benchmarks for nicer
    /// output.
    fn type_name() -> String;
}

//
// Named implementations
//

macro_rules! impl_named_for_primitives {
    ($($type:ty),+) => {
        $(
            impl Named for $type {
                fn type_name() -> String {
                    stringify!($type).to_string()
                }
            }
        )+
    };
}

impl_named_for_primitives!(i8, i16, i32, i64, i128);
impl_named_for_primitives!(u8, u16, u32, u64, u128);

impl<const LIMBS: usize> Named for Int<LIMBS> {
    fn type_name() -> String {
        format!("Int<{}>", LIMBS)
    }
}

impl<const LIMBS: usize> Named for Uint<LIMBS> {
    fn type_name() -> String {
        format!("Uint<{}>", LIMBS)
    }
}

impl Named for Boolean {
    fn type_name() -> String {
        "b".to_owned()
    }
}
