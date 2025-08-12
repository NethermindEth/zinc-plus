use ark_std::{
    fmt::Debug,
    iter::{Product, Sum},
    ops::{
        Add, AddAssign, Div, Index, IndexMut, Mul, MulAssign, Neg, Range, RangeTo, RemAssign, Shr,
        Sub, SubAssign,
    },
    vec::Vec,
};
use crypto_bigint::Random;
use num_traits::{ConstOne, ConstZero, One, Zero};

use crate::{
    traits::{FieldMap, FromBytes},
    transcript::KeccakTranscript,
    pcs::utils::ToBytes,
};
use crate::field::FieldConfig;

/// Trait for field elements, requiring arithmetic, assignment, random generation, and conversion traits.
/// Used as a bound for generic code over finite fields.
pub trait Field:
    Debug
    + Clone
    + PartialEq
    + Eq
    + Default
    + Sync
    + Send
    + Zero
    + One
    + Neg<Output = Self>
    + Add<Self, Output = Self>
    + Add<u32, Output = Self>
    + Sub<Self, Output = Self>
    + for<'a> Mul<&'a Self, Output = Self>
    + AddAssign
    + for<'a> AddAssign<&'a Self>
    + MulAssign
    + for<'a> MulAssign<&'a Self>
    + SubAssign
    + Div<Self, Output = Self>
    + Sum
    + for<'a> Product<&'a Self>
    + for<'a> From<&'a Self>
    + Random
{
    /// Integer representation type for the field element.
    type B: BigInteger<W = Self::W> + From<Self::I> + FieldMap<Self, Output = Self>;
    /// Field configuration type.
    type C: FieldConfig<Self::B>;
    /// Word representation type.
    type W: Words;
    /// Cryptographic integer type.
    type I: Integer<W = Self::W, Uint = Self::U> + for<'a> From<&'a Self::B>;
    /// Cryptographic unsigned integer type.
    type U: Uinteger<W = Self::W, Int = Self::I>;

    /// Creates a new field element from config and value, without checking.
    fn new_unchecked(value: Self::B) -> Self;

    /// Generates a random field element with the given config.
    fn rand<R: ark_std::rand::Rng + ?Sized>(rng: &mut R) -> Self;

    /// Returns a reference to the integer value.
    fn value(&self) -> &Self::B;

    /// Returns a mutable reference to the integer value.
    fn value_mut(&mut self) -> &mut Self::B;

    /// Absorbs the field element into a Keccak transcript.
    fn absorb_into_transcript(&self, transcript: &mut KeccakTranscript);
}

/// Trait for integer types used as field element representations.
pub trait BigInteger: From<u64> + From<u32> + Debug + FromBytes + Clone {
    type W: Words;
    /// Converts the integer to its word representation.
    fn to_words(&self) -> Self::W;
    /// Returns the integer one.
    fn one() -> Self;
    /// Constructs from big-endian bits.
    fn from_bits_be(bits: &[bool]) -> Self;
    /// Constructs from little-endian bits.
    fn from_bits_le(bits: &[bool]) -> Self;
    /// Returns the number of bits.
    fn num_bits(&self) -> u32;
    /// Constructs from words.
    fn new(words: Self::W) -> Self;
    /// Converts to big-endian bytes.
    fn to_bytes_be(self) -> Vec<u8>;
    /// Converts to little-endian bytes.
    fn to_bytes_le(self) -> Vec<u8>;
}

/// Trait for word-based representations of integers.
pub trait Words:
    Default
    + Index<usize, Output = Self::Word>
    + IndexMut<usize>
    + Index<Range<usize>, Output = [Self::Word]>
    + IndexMut<Range<usize>, Output = [Self::Word]>
    + Index<RangeTo<usize>, Output = [Self::Word]>
    + IndexMut<RangeTo<usize>, Output = [Self::Word]>
    + Clone
{
    type Word: PrimitiveConversions + Sized;
    /// Returns the number of words.
    fn num_words() -> usize;
}

/// Trait for cryptographic integer types.
pub trait Integer:
    Debug
    + Clone
    + PartialEq
    + Eq
    + PartialOrd
    + Default
    + Send
    + Sync
    + Zero
    + crypto_bigint::Zero
    + One
    + ConstZero
    + ConstOne
    + Neg<Output = Self>
    + Add<Output = Self>
    + for<'a> Add<&'a Self, Output = Self>
    + for<'a> Sub<&'a Self, Output = Self>
    + Mul<Output = Self>
    + for<'a> Mul<&'a Self, Output = Self>
    + AddAssign<Self>
    + for<'a> AddAssign<&'a Self>
    + RemAssign<Self>
    + Sum
    + for<'a> From<&'a Self>
    + From<i64>
    + From<i32>
    + From<i8>
    + From<u8>
    + Random
    + ToBytes
{
    type W: Words;
    type Uint: Uinteger<W = Self::W>;
    type I: BigInteger<W = Self::W> + for<'a> From<&'a Self>;

    fn num_bits() -> usize {
        Self::W::num_words() * <Self::W as Words>::Word::bits()
    }

    /// Constructs from words.
    fn from_words(words: Self::W) -> Self;
    fn as_words(&self) -> &[u64];
    fn from_i64(value: i64) -> Self;
    fn abs(&self) -> Self::Uint;
}

/// Trait for cryptographic unsigned integer types.
pub trait Uinteger: Clone + FromBytes + One + for<'a> SubAssign<&'a Self> {
    type W: Words;
    type Int: Integer<W = Self::W>;
    type PrimalityTest: PrimalityTest<Self>;
    /// Constructs from words.
    fn from_words(words: Self::W) -> Self;
    /// Converts to the signed integer type.
    fn as_int(&self) -> Self::Int;
    /// Converts to words.
    fn to_words(self) -> Self::W;
    fn is_even(&self) -> bool;
}

pub trait ZipTypes: Send + Sync {
    /// Width of elements in witness/polynomial evaluations on hypercube
    type N: Integer;

    /// Width of elements in the encoding matrices
    type L: Integer + for<'a> From<&'a Self::N>;

    /// Width of elements in the code
    type K: Integer + for<'a> From<&'a Self::N> + for<'a> From<&'a Self::L>;

    /// Width of elements in linear combination of code rows
    type M: Integer
        + for<'a> From<&'a Self::N>
        + for<'a> From<&'a Self::L>
        + for<'a> From<&'a Self::K>;
}

pub trait PrimalityTest<U: Uinteger> {
    type Inner;
    fn new(candidate: U) -> Self;
    fn is_probably_prime(&self) -> bool;
}

pub trait PrimitiveConversion<T> {
    fn from_primitive(value: T) -> Self;
}

macro_rules! impl_single_primitive_conversion {
    ($o:ty, $i:ty) => {
        impl PrimitiveConversion<$i> for $o {
            #[inline(always)]
            fn from_primitive(value: $i) -> $o {
                value as $o
            }
        }
    };
}

macro_rules! impl_primitive_conversion {
    ($o:ty) => {
        impl_single_primitive_conversion!($o, u8);
        impl_single_primitive_conversion!($o, u16);
        impl_single_primitive_conversion!($o, u32);
        impl_single_primitive_conversion!($o, u64);
        impl_single_primitive_conversion!($o, u128);
        impl_single_primitive_conversion!($o, usize);
        impl_single_primitive_conversion!($o, i8);
        impl_single_primitive_conversion!($o, i16);
        impl_single_primitive_conversion!($o, i32);
        impl_single_primitive_conversion!($o, i64);
        impl_single_primitive_conversion!($o, i128);
        impl_single_primitive_conversion!($o, isize);
    };
}

impl_primitive_conversion!(u8);
impl_primitive_conversion!(u16);
impl_primitive_conversion!(u32);
impl_primitive_conversion!(u64);
impl_primitive_conversion!(u128);
impl_primitive_conversion!(usize);
impl_primitive_conversion!(i8);
impl_primitive_conversion!(i16);
impl_primitive_conversion!(i32);
impl_primitive_conversion!(i64);
impl_primitive_conversion!(i128);
impl_primitive_conversion!(isize);

pub trait PrimitiveConversions:
    PrimitiveConversion<u8>
    + PrimitiveConversion<u16>
    + PrimitiveConversion<u32>
    + PrimitiveConversion<u64>
    + PrimitiveConversion<u128>
    + PrimitiveConversion<usize>
    + PrimitiveConversion<i8>
    + PrimitiveConversion<i16>
    + PrimitiveConversion<i32>
    + PrimitiveConversion<i64>
    + PrimitiveConversion<i128>
    + PrimitiveConversion<isize>
    + Shr<usize, Output = Self>
{
    fn bits() -> usize;
}
impl<T> PrimitiveConversions for T
where
    T: PrimitiveConversion<u8>
        + PrimitiveConversion<u16>
        + PrimitiveConversion<u32>
        + PrimitiveConversion<u64>
        + PrimitiveConversion<u128>
        + PrimitiveConversion<usize>
        + PrimitiveConversion<i8>
        + PrimitiveConversion<i16>
        + PrimitiveConversion<i32>
        + PrimitiveConversion<i64>
        + PrimitiveConversion<i128>
        + PrimitiveConversion<isize>
        + Shr<usize, Output = Self>,
{
    #[inline(always)]
    fn bits() -> usize {
        size_of::<Self>() * 8
    }
}
