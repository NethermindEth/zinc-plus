use crate::primality::PrimalityTest;
use crypto_primitives::{ConstIntSemiring, boolean::Boolean, crypto_bigint_int::Int};
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

impl<const LIMBS: usize, const LIMBS2: usize> FromRef<Int<LIMBS2>> for Int<LIMBS> {
    #[inline]
    fn from_ref(value: &Int<LIMBS2>) -> Self {
        Self::try_from(value.inner()).expect("Destination Int type is too small")
    }
}

//
// Named
//

pub trait Named {
    /// Returns the name of the type as a string, used in benchmarks for nicer
    /// output.
    fn type_name() -> String;
}

//
// Transcribable and Transcript
//

/// Trait for types that can be transcribed to and from a byte representation.
/// Byte order is not specified, but it must be portable across platforms.
pub trait Transcribable {
    /// Number of bytes required to represent **length** of this type, could be
    /// zero if known in advance.
    const LENGTH_NUM_BYTES: usize;

    /// Read number of bytes required to represent this type.
    /// The buffer must be exactly `LENGTH_NUM_BYTES` long.
    fn read_num_bytes(bytes: &[u8]) -> usize;

    /// Creates a new instance from a byte buffer.
    /// The buffer must be exactly the length returned by `read_num_bytes`.
    fn read_transcription_bytes(bytes: &[u8]) -> Self;

    /// Returns the number of bytes required to represent this type.
    fn get_num_bytes(&self) -> usize;

    /// Transcribes the current instance into a byte buffer.
    /// The buffer must be exactly the length returned by `get_num_bytes`.
    fn write_transcription_bytes(&self, buf: &mut [u8]);
}

/// If number of bytes for `Transcribable` is known at compile time,
/// there's no need to read and write them from a buffer.
pub trait ConstTranscribable {
    /// Number of bytes required to represent this type.
    const NUM_BYTES: usize;
    /// Number of bits actually used to store data.
    const NUM_BITS: usize = Self::NUM_BYTES * 8;

    /// Creates a new instance from a byte buffer.
    /// The buffer must be exactly `NUM_BYTES` long.
    fn read_transcription_bytes(bytes: &[u8]) -> Self;

    /// Transcribes the current instance into a byte buffer.
    /// Buffer must be exactly `NUM_BYTES` long.
    fn write_transcription_bytes(&self, buf: &mut [u8]);
}

impl<T: ConstTranscribable> Transcribable for T {
    const LENGTH_NUM_BYTES: usize = 0;

    fn read_num_bytes(bytes: &[u8]) -> usize {
        assert_eq!(bytes.len(), 0);
        Self::NUM_BYTES
    }

    fn read_transcription_bytes(bytes: &[u8]) -> Self {
        <Self as ConstTranscribable>::read_transcription_bytes(bytes)
    }

    fn get_num_bytes(&self) -> usize {
        Self::NUM_BYTES
    }

    fn write_transcription_bytes(&self, buf: &mut [u8]) {
        <Self as ConstTranscribable>::write_transcription_bytes(self, buf);
    }
}

pub trait Transcript {
    /// Generates a pseudorandom transcribable value as a challenge based on the
    /// current transcript state, updating it.
    fn get_challenge<T: ConstTranscribable>(&mut self) -> T;

    /// Generates a pseudorandom transcribable values as challenges based on the
    /// current transcript state, updating it.
    fn get_challenges<T: ConstTranscribable>(&mut self, n: usize) -> Vec<T> {
        (0..n).map(|_| self.get_challenge()).collect()
    }

    fn get_prime<R: ConstIntSemiring + ConstTranscribable, T: PrimalityTest<R>>(&mut self) -> R;
}
