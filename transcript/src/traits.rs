//
// Transcribable and Transcript
//

use crypto_bigint::{BitOps, BoxedUint, Word};
use crypto_primitives::{
    ConstIntSemiring, PrimeField, WORD_FACTOR, boolean::Boolean, crypto_bigint_int::Int,
    crypto_bigint_uint::Uint,
};
use itertools::Itertools;
use zinc_primality::PrimalityTest;
use zinc_utils::{add, from_ref::FromRef, mul};

/// Common trait for both `Transcribable` and `ConstTranscribable` to avoid code
/// duplication in their implementations.
pub trait GenTranscribable: Sized {
    /// Creates a new instance from a byte buffer.
    /// The buffer must be exactly the expected length.
    fn read_transcription_bytes_exact(bytes: &[u8]) -> Self;

    /// Transcribes the current instance into a byte buffer.
    /// The buffer must be exactly the expected length.
    fn write_transcription_bytes_exact(&self, buf: &mut [u8]);
}

/// Trait for types that can be transcribed to and from a byte representation.
/// Byte order is not specified, but it must be portable across platforms.
pub trait Transcribable: GenTranscribable {
    /// Number of bytes required to represent **length** of this type, could be
    /// zero if known in advance.
    // Defaults to 4 gigabytes - way more than we would ever want to handle in
    // practice
    const LENGTH_NUM_BYTES: usize = u32::NUM_BYTES;

    /// Read number of bytes required to represent this type.
    /// The buffer must be exactly `LENGTH_NUM_BYTES` long.
    /// The buffer passed to `read_transcription_bytes` should be exactly the
    /// length returned by this function.
    fn read_num_bytes(bytes: &[u8]) -> usize {
        usize::try_from(u32::read_transcription_bytes_exact(bytes))
            .expect("num_bytes must fit into usize")
    }

    /// Returns the number of bytes required to represent this type.
    /// The buffer passed to `write_transcription_bytes` should be exactly the
    /// length returned by this function.
    fn get_num_bytes(&self) -> usize;

    /// Reads an instance of this type from the beginning of the byte slice, and
    /// returns the instance along with the remaining byte slice.
    fn read_transcription_bytes_subset(bytes: &[u8]) -> (Self, &[u8]) {
        let (bytes_num_bytes, bytes_rem) = bytes.split_at(Self::LENGTH_NUM_BYTES);
        let num_bytes = Self::read_num_bytes(bytes_num_bytes);
        assert!(
            bytes_rem.len() >= num_bytes,
            "Byte slice length is not sufficient for reading Transcribable"
        );
        let (bytes_data, bytes_rem) = bytes_rem.split_at(num_bytes);
        (Self::read_transcription_bytes_exact(bytes_data), bytes_rem)
    }

    /// Writes this instance, prefixed by length, into the beginning of the byte
    /// buffer, and returns the remaining byte buffer.
    fn write_transcription_bytes_subset<'a>(&self, mut buf: &'a mut [u8]) -> &'a mut [u8] {
        let num_bytes = self.get_num_bytes();
        if Self::LENGTH_NUM_BYTES > 0 {
            buf[0..Self::LENGTH_NUM_BYTES]
                .copy_from_slice(&num_bytes.to_le_bytes()[..Self::LENGTH_NUM_BYTES]);
            buf = &mut buf[Self::LENGTH_NUM_BYTES..];
        };
        let (buf, rest) = buf.split_at_mut(num_bytes);
        self.write_transcription_bytes_exact(buf);
        rest
    }
}

/// If number of bytes for `Transcribable` is known at compile time,
/// there's no need to read and write them from a buffer.
pub trait ConstTranscribable: GenTranscribable {
    /// Number of bytes required to represent this type.
    const NUM_BYTES: usize;
    /// Number of bits actually used to store data.
    const NUM_BITS: usize = Self::NUM_BYTES * 8;
}

impl<T: ConstTranscribable> Transcribable for T {
    const LENGTH_NUM_BYTES: usize = 0;

    fn read_num_bytes(bytes: &[u8]) -> usize {
        assert_eq!(bytes.len(), 0);
        Self::NUM_BYTES
    }

    fn get_num_bytes(&self) -> usize {
        Self::NUM_BYTES
    }

    fn read_transcription_bytes_subset(bytes: &[u8]) -> (Self, &[u8]) {
        assert!(
            bytes.len() >= Self::NUM_BYTES,
            "Byte slice length is not sufficient for reading Transcribable"
        );
        let (bytes_data, bytes_rem) = bytes.split_at(Self::NUM_BYTES);
        (Self::read_transcription_bytes_exact(bytes_data), bytes_rem)
    }
}

/// Should not be used directly — use [`delegate_transcribable!`] or
/// [`delegate_const_transcribable!`] instead.
#[macro_export]
macro_rules! delegate_gen_transcribable {
    ($wrapper:ident { $field:tt : $inner_ty:ty }) => {
        impl $crate::traits::GenTranscribable for $wrapper {
            fn read_transcription_bytes_exact(bytes: &[u8]) -> Self {
                Self {
                    $field: <$inner_ty as $crate::traits::GenTranscribable>::read_transcription_bytes_exact(bytes),
                }
            }

            fn write_transcription_bytes_exact(&self, buf: &mut [u8]) {
                $crate::traits::GenTranscribable::write_transcription_bytes_exact(&self.$field, buf)
            }
        }
    };
    ($wrapper:ident <$($gen:tt),+> { $field:tt : $inner_ty:ty } $(where $($bounds:tt)+)?) => {
        impl<$($gen),+> $crate::traits::GenTranscribable for $wrapper<$($gen),+>
        $(where $($bounds)+)?
        {
            fn read_transcription_bytes_exact(bytes: &[u8]) -> Self {
                Self {
                    $field: <$inner_ty as $crate::traits::GenTranscribable>::read_transcription_bytes_exact(bytes),
                }
            }

            fn write_transcription_bytes_exact(&self, buf: &mut [u8]) {
                $crate::traits::GenTranscribable::write_transcription_bytes_exact(&self.$field, buf)
            }
        }
    };
    ($wrapper:ident <const $cg_name:ident : $cg_ty:ty> { $field:tt : $inner_ty:ty } $(where $($bounds:tt)+)?) => {
        impl<const $cg_name: $cg_ty> $crate::traits::GenTranscribable for $wrapper<$cg_name>
        $(where $($bounds)+)?
        {
            fn read_transcription_bytes_exact(bytes: &[u8]) -> Self {
                Self {
                    $field: <$inner_ty as $crate::traits::GenTranscribable>::read_transcription_bytes_exact(bytes),
                }
            }

            fn write_transcription_bytes_exact(&self, buf: &mut [u8]) {
                $crate::traits::GenTranscribable::write_transcription_bytes_exact(&self.$field, buf)
            }
        }
    };
    ($wrapper:ident <$($gen:tt),+, const $cg_name:ident : $cg_ty:ty> { $field:tt : $inner_ty:ty } $(where $($bounds:tt)+)?) => {
        impl<$($gen),+, const $cg_name: $cg_ty> $crate::traits::GenTranscribable for $wrapper<$($gen),+, $cg_name>
        $(where $($bounds)+)?
        {
            fn read_transcription_bytes_exact(bytes: &[u8]) -> Self {
                Self {
                    $field: <$inner_ty as $crate::traits::GenTranscribable>::read_transcription_bytes_exact(bytes),
                }
            }

            fn write_transcription_bytes_exact(&self, buf: &mut [u8]) {
                $crate::traits::GenTranscribable::write_transcription_bytes_exact(&self.$field, buf)
            }
        }
    };
}

/// Delegates `Transcribable` to the single inner field of a newtype.
/// Use this instead of [`delegate_const_transcribable!`] when the inner type
/// only implements `Transcribable` (e.g. `BoxedUint`).
///
/// Supports non-generic and generic types with optional `where` clauses:
/// ```ignore
/// delegate_transcribable!(MyTuple(InnerType));
/// delegate_transcribable!(MyNamed { field: InnerType });
/// delegate_transcribable!(MyGeneric<T> { field: Vec<T> } where T: SomeBound);
/// delegate_transcribable!(MyConst<const N: usize>(SomeType));
/// delegate_transcribable!(MyMixed<T, const N: usize> { f: [T; N] } where T: SomeBound);
/// ```
#[macro_export]
macro_rules! delegate_transcribable {
    ($wrapper:ident ($inner_ty:ty)) => {
        $crate::delegate_transcribable!($wrapper { 0: $inner_ty });
    };
    ($wrapper:ident { $field:tt : $inner_ty:ty }) => {
        $crate::delegate_gen_transcribable!($wrapper { $field: $inner_ty });
        impl $crate::traits::Transcribable for $wrapper {
            const LENGTH_NUM_BYTES: usize =
                <$inner_ty as $crate::traits::Transcribable>::LENGTH_NUM_BYTES;

            fn read_num_bytes(bytes: &[u8]) -> usize {
                <$inner_ty as $crate::traits::Transcribable>::read_num_bytes(bytes)
            }

            fn get_num_bytes(&self) -> usize {
                <$inner_ty as $crate::traits::Transcribable>::get_num_bytes(&self.$field)
            }
        }
    };
    ($wrapper:ident <$($gen:tt),+> ($inner_ty:ty) $(where $($bounds:tt)+)?) => {
        $crate::delegate_transcribable!($wrapper <$($gen),+> { 0: $inner_ty } $(where $($bounds)+)?);
    };
    ($wrapper:ident <$($gen:tt),+> { $field:tt : $inner_ty:ty } $(where $($bounds:tt)+)?) => {
        $crate::delegate_gen_transcribable!($wrapper <$($gen),+> { $field: $inner_ty } $(where $($bounds)+)?);
        impl<$($gen),+> $crate::traits::Transcribable for $wrapper<$($gen),+>
        $(where $($bounds)+)?
        {
            const LENGTH_NUM_BYTES: usize =
                <$inner_ty as $crate::traits::Transcribable>::LENGTH_NUM_BYTES;

            fn read_num_bytes(bytes: &[u8]) -> usize {
                <$inner_ty as $crate::traits::Transcribable>::read_num_bytes(bytes)
            }

            fn get_num_bytes(&self) -> usize {
                <$inner_ty as $crate::traits::Transcribable>::get_num_bytes(&self.$field)
            }
        }
    };
    ($wrapper:ident <const $cg_name:ident : $cg_ty:ty> ($inner_ty:ty) $(where $($bounds:tt)+)?) => {
        $crate::delegate_transcribable!($wrapper <const $cg_name : $cg_ty> { 0: $inner_ty } $(where $($bounds)+)?);
    };
    ($wrapper:ident <const $cg_name:ident : $cg_ty:ty> { $field:tt : $inner_ty:ty } $(where $($bounds:tt)+)?) => {
        $crate::delegate_gen_transcribable!($wrapper <const $cg_name : $cg_ty> { $field: $inner_ty } $(where $($bounds)+)?);
        impl<const $cg_name: $cg_ty> $crate::traits::Transcribable for $wrapper<$cg_name>
        $(where $($bounds)+)?
        {
            const LENGTH_NUM_BYTES: usize =
                <$inner_ty as $crate::traits::Transcribable>::LENGTH_NUM_BYTES;

            fn read_num_bytes(bytes: &[u8]) -> usize {
                <$inner_ty as $crate::traits::Transcribable>::read_num_bytes(bytes)
            }

            fn get_num_bytes(&self) -> usize {
                <$inner_ty as $crate::traits::Transcribable>::get_num_bytes(&self.$field)
            }
        }
    };
    ($wrapper:ident <$($gen:tt),+, const $cg_name:ident : $cg_ty:ty> ($inner_ty:ty) $(where $($bounds:tt)+)?) => {
        $crate::delegate_transcribable!($wrapper <$($gen),+, const $cg_name : $cg_ty> { 0: $inner_ty } $(where $($bounds)+)?);
    };
    ($wrapper:ident <$($gen:tt),+, const $cg_name:ident : $cg_ty:ty> { $field:tt : $inner_ty:ty } $(where $($bounds:tt)+)?) => {
        $crate::delegate_gen_transcribable!($wrapper <$($gen),+, const $cg_name : $cg_ty> { $field: $inner_ty } $(where $($bounds)+)?);
        impl<$($gen),+, const $cg_name: $cg_ty> $crate::traits::Transcribable for $wrapper<$($gen),+, $cg_name>
        $(where $($bounds)+)?
        {
            const LENGTH_NUM_BYTES: usize =
                <$inner_ty as $crate::traits::Transcribable>::LENGTH_NUM_BYTES;

            fn read_num_bytes(bytes: &[u8]) -> usize {
                <$inner_ty as $crate::traits::Transcribable>::read_num_bytes(bytes)
            }

            fn get_num_bytes(&self) -> usize {
                <$inner_ty as $crate::traits::Transcribable>::get_num_bytes(&self.$field)
            }
        }
    };
}

/// Delegates `ConstTranscribable` to the single inner field of a newtype.
/// `Transcribable` is obtained automatically via the blanket impl.
///
/// Supports non-generic and generic types with optional `where` clauses:
/// ```ignore
/// delegate_const_transcribable!(MyTuple(InnerType));
/// delegate_const_transcribable!(MyNamed { field: InnerType });
/// delegate_const_transcribable!(MyGeneric<T> { field: T } where T: SomeBound);
/// delegate_const_transcribable!(MyConst<const N: usize>(SomeType));
/// delegate_const_transcribable!(MyMixed<T, const N: usize> { f: [T; N] } where T: SomeBound);
/// ```
#[macro_export]
macro_rules! delegate_const_transcribable {
    ($wrapper:ident ($inner_ty:ty)) => {
        $crate::delegate_const_transcribable!($wrapper { 0: $inner_ty });
    };
    ($wrapper:ident { $field:tt : $inner_ty:ty }) => {
        $crate::delegate_gen_transcribable!($wrapper { $field: $inner_ty });
        impl $crate::traits::ConstTranscribable for $wrapper {
            const NUM_BYTES: usize = <$inner_ty as $crate::traits::ConstTranscribable>::NUM_BYTES;
            const NUM_BITS: usize = <$inner_ty as $crate::traits::ConstTranscribable>::NUM_BITS;
        }
    };
    ($wrapper:ident <$($gen:tt),+> ($inner_ty:ty) $(where $($bounds:tt)+)?) => {
        $crate::delegate_const_transcribable!($wrapper <$($gen),+> { 0: $inner_ty } $(where $($bounds)+)?);
    };
    ($wrapper:ident <$($gen:tt),+> { $field:tt : $inner_ty:ty } $(where $($bounds:tt)+)?) => {
        $crate::delegate_gen_transcribable!($wrapper <$($gen),+> { $field: $inner_ty } $(where $($bounds)+)?);
        impl<$($gen),+> $crate::traits::ConstTranscribable for $wrapper<$($gen),+>
        $(where $($bounds)+)?
        {
            const NUM_BYTES: usize = <$inner_ty as $crate::traits::ConstTranscribable>::NUM_BYTES;
            const NUM_BITS: usize = <$inner_ty as $crate::traits::ConstTranscribable>::NUM_BITS;
        }
    };
    ($wrapper:ident <const $cg_name:ident : $cg_ty:ty> ($inner_ty:ty) $(where $($bounds:tt)+)?) => {
        $crate::delegate_const_transcribable!($wrapper <const $cg_name : $cg_ty> { 0: $inner_ty } $(where $($bounds)+)?);
    };
    ($wrapper:ident <const $cg_name:ident : $cg_ty:ty> { $field:tt : $inner_ty:ty } $(where $($bounds:tt)+)?) => {
        $crate::delegate_gen_transcribable!($wrapper <const $cg_name : $cg_ty> { $field: $inner_ty } $(where $($bounds)+)?);
        impl<const $cg_name: $cg_ty> $crate::traits::ConstTranscribable for $wrapper<$cg_name>
        $(where $($bounds)+)?
        {
            const NUM_BYTES: usize = <$inner_ty as $crate::traits::ConstTranscribable>::NUM_BYTES;
            const NUM_BITS: usize = <$inner_ty as $crate::traits::ConstTranscribable>::NUM_BITS;
        }
    };
    ($wrapper:ident <$($gen:tt),+, const $cg_name:ident : $cg_ty:ty> ($inner_ty:ty) $(where $($bounds:tt)+)?) => {
        $crate::delegate_const_transcribable!($wrapper <$($gen),+, const $cg_name : $cg_ty> { 0: $inner_ty } $(where $($bounds)+)?);
    };
    ($wrapper:ident <$($gen:tt),+, const $cg_name:ident : $cg_ty:ty> { $field:tt : $inner_ty:ty } $(where $($bounds:tt)+)?) => {
        $crate::delegate_gen_transcribable!($wrapper <$($gen),+, const $cg_name : $cg_ty> { $field: $inner_ty } $(where $($bounds)+)?);
        impl<$($gen),+, const $cg_name: $cg_ty> $crate::traits::ConstTranscribable for $wrapper<$($gen),+, $cg_name>
        $(where $($bounds)+)?
        {
            const NUM_BYTES: usize = <$inner_ty as $crate::traits::ConstTranscribable>::NUM_BYTES;
            const NUM_BITS: usize = <$inner_ty as $crate::traits::ConstTranscribable>::NUM_BITS;
        }
    };
}

pub trait Transcript {
    /// Generates a pseudorandom transcribable value as a challenge based on the
    /// current transcript state, updating it.
    fn get_challenge<T: ConstTranscribable>(&mut self) -> T;

    fn get_field_challenge<F: PrimeField>(&mut self, cfg: &F::Config) -> F
    where
        F::Inner: ConstTranscribable,
    {
        let random_inner = self.get_challenge();

        F::new_with_cfg(random_inner, cfg)
    }

    /// Generates a pseudorandom transcribable values as challenges based on the
    /// current transcript state, updating it.
    // TODO(Alex): `get_field_challenge` is not efficient
    //             to call in a batch because each call allocates its own buffer.
    //             It might make sense to make a separate `get_challenge_with_buf`
    //             alternative to `get_challenge`.
    fn get_field_challenges<F: PrimeField>(&mut self, n: usize, cfg: &F::Config) -> Vec<F>
    where
        F::Inner: ConstTranscribable,
    {
        (0..n).map(|_| self.get_field_challenge(cfg)).collect()
    }

    /// Generates a pseudorandom transcribable values as challenges based on the
    /// current transcript state, updating it.
    fn get_challenges<T: ConstTranscribable>(&mut self, n: usize) -> Vec<T> {
        (0..n).map(|_| self.get_challenge()).collect()
    }

    fn get_prime<R: ConstIntSemiring + ConstTranscribable, T: PrimalityTest<R>>(&mut self) -> R;

    fn get_random_field_cfg<F, FMod, T>(&mut self) -> F::Config
    where
        F: PrimeField,
        FMod: ConstTranscribable + ConstIntSemiring,
        F::Modulus: FromRef<FMod>,
        T: PrimalityTest<FMod>,
    {
        let prime = self.get_prime::<FMod, T>();

        F::make_cfg(&F::Modulus::from_ref(&prime)).expect("prime is guaranteed to be prime")
    }

    /// Absorbs a byte slice into the hash sponge.
    /// This updates the internal state of the hasher with the provided data.
    /// Should not be used directly.
    fn absorb_inner(&mut self, v: &[u8]);

    /// Absorbs a byte slice into the transcript.
    fn absorb_slice(&mut self, buf: &[u8]) {
        self.absorb_inner(&[0x6]);
        self.absorb_inner(buf);
        self.absorb_inner(&[0x7]);
    }

    /// Absorbs a field element into the transcript.
    /// Delegates to the field element's implementation of
    /// absorb_into_transcript.
    // Note: Currently this only works for fields whose modulus and inner element
    // have the same byte length
    fn absorb_random_field<F>(&mut self, v: &F, buf: &mut [u8])
    where
        F: PrimeField,
        F::Inner: Transcribable,
        F::Modulus: Transcribable,
    {
        debug_assert_eq!(F::Inner::LENGTH_NUM_BYTES, F::Modulus::LENGTH_NUM_BYTES);
        debug_assert_eq!(
            F::Inner::get_num_bytes(v.inner()),
            F::Modulus::get_num_bytes(&v.modulus())
        );
        self.absorb_inner(&[0x3]);
        v.modulus().write_transcription_bytes_exact(buf);
        self.absorb_inner(buf);
        self.absorb_inner(&[0x5]);

        self.absorb_inner(&[0x1]);
        v.inner().write_transcription_bytes_exact(buf);
        self.absorb_inner(buf);
        self.absorb_inner(&[0x3])
    }

    /// Absorbs a slice of field element into the transcript.
    /// Delegates to the field element's implementation of
    /// absorb_into_transcript.
    fn absorb_random_field_slice<F>(&mut self, v: &[F], buf: &mut [u8])
    where
        F: PrimeField,
        F::Inner: Transcribable,
        F::Modulus: Transcribable,
    {
        v.iter().for_each(|x| self.absorb_random_field(x, buf));
    }
}

//
// Transcribable implementations
//

macro_rules! impl_transcribable_for_primitives {
    ($($type:ty),+) => {
        $(
            impl GenTranscribable for $type {
                fn read_transcription_bytes_exact(bytes: &[u8]) -> Self {
                    Self::from_le_bytes(bytes.try_into().expect("Invalid byte slice length"))
                }

                fn write_transcription_bytes_exact(&self, buf: &mut [u8]) {
                    debug_assert_eq!(buf.len(), Self::NUM_BYTES);
                    buf.copy_from_slice(&self.to_le_bytes());
                }
            }

            impl ConstTranscribable for $type {
                const NUM_BYTES: usize = std::mem::size_of::<$type>();
            }
        )+
    };
}

impl_transcribable_for_primitives!(u8, u16, u32, u64, u128);
impl_transcribable_for_primitives!(i8, i16, i32, i64, i128);

impl GenTranscribable for Boolean {
    fn read_transcription_bytes_exact(bytes: &[u8]) -> Self {
        (bytes[0] != 0).into()
    }

    fn write_transcription_bytes_exact(&self, buf: &mut [u8]) {
        buf[0] = self.to_u8();
    }
}

impl ConstTranscribable for Boolean {
    const NUM_BYTES: usize = 1;
    const NUM_BITS: usize = 1;
}

impl<const LIMBS: usize> GenTranscribable for Uint<LIMBS> {
    fn read_transcription_bytes_exact(bytes: &[u8]) -> Self {
        // crypto_bigint::Uint stores limbs in least-to-most significant order.
        // It matches little-endian order ef limbs encoding, so platform pointer width
        // does not matter.
        let (chunked, rem) = bytes.as_chunks::<{ 8 / WORD_FACTOR }>();
        assert!(rem.is_empty(), "Invalid byte slice length for Uint");
        let words = chunked
            .iter()
            .map(|chunk| Word::from_le_bytes(*chunk))
            .collect_array::<LIMBS>()
            .expect("Invalid length for Uint");
        Uint::<LIMBS>::from_words(words)
    }

    #[allow(clippy::arithmetic_side_effects)]
    fn write_transcription_bytes_exact(&self, buf: &mut [u8]) {
        // crypto_bigint::Uint stores limbs in least-to-most significant order.
        // It matches little-endian order ef limbs encoding, so platform pointer width
        // does not matter.
        assert_eq!(buf.len(), Self::NUM_BYTES, "Buffer size mismatch for Uint");
        const W_SIZE: usize = size_of::<Word>();
        for (i, w) in self.as_words().iter().enumerate() {
            // Performance: reuse buffer and help compiler optimize away materializing
            // vector
            buf[(i * W_SIZE)..(i * W_SIZE + W_SIZE)].copy_from_slice(w.to_le_bytes().as_ref());
        }
    }
}

impl<const LIMBS: usize> ConstTranscribable for Uint<LIMBS> {
    const NUM_BYTES: usize = 8 * LIMBS / WORD_FACTOR;
}

impl<const LIMBS: usize> GenTranscribable for Int<LIMBS> {
    fn read_transcription_bytes_exact(bytes: &[u8]) -> Self {
        *Uint::<LIMBS>::read_transcription_bytes_exact(bytes).as_int()
    }

    fn write_transcription_bytes_exact(&self, buf: &mut [u8]) {
        self.as_uint().write_transcription_bytes_exact(buf)
    }
}

impl<const LIMBS: usize> ConstTranscribable for Int<LIMBS> {
    const NUM_BYTES: usize = Uint::<LIMBS>::NUM_BYTES;
}

impl GenTranscribable for BoxedUint {
    fn read_transcription_bytes_exact(bytes: &[u8]) -> Self {
        // crypto_bigint::BoxedUint stores limbs in least-to-most significant order.
        // It matches little-endian order ef limbs encoding, so platform pointer width
        // does not matter.
        let (chunked, rem) = bytes.as_chunks::<{ 8 / WORD_FACTOR }>();
        assert!(rem.is_empty(), "Invalid byte slice length for BoxedUint");
        let words = chunked
            .iter()
            .map(|chunk| Word::from_le_bytes(*chunk))
            .collect_vec();
        BoxedUint::from_words(words)
    }

    #[allow(clippy::arithmetic_side_effects)]
    fn write_transcription_bytes_exact(&self, buf: &mut [u8]) {
        // crypto_bigint::BoxedUint stores limbs in least-to-most significant order.
        // It matches little-endian order ef limbs encoding, so platform pointer width
        // does not matter.
        assert_eq!(
            buf.len(),
            self.bytes_precision(),
            "Buffer size mismatch for BoxedUint"
        );
        const W_SIZE: usize = size_of::<Word>();
        for (i, w) in self.as_words().iter().enumerate() {
            // Performance: reuse buffer and help compiler optimize away materializing
            // vector
            buf[(i * W_SIZE)..(i * W_SIZE + W_SIZE)].copy_from_slice(w.to_le_bytes().as_ref());
        }
    }
}

impl Transcribable for BoxedUint {
    /// Up to 255 bytes - so up to 2040 bits - should be plenty.
    const LENGTH_NUM_BYTES: usize = 1;

    fn read_num_bytes(bytes: &[u8]) -> usize {
        assert_eq!(bytes.len(), Self::LENGTH_NUM_BYTES);
        usize::from(bytes[0])
    }

    fn get_num_bytes(&self) -> usize {
        usize::from(u8::try_from(self.bytes_precision()).expect("BoxedUint size must fit into u8"))
    }
}

impl<F> GenTranscribable for Vec<F>
where
    F: PrimeField,
    F::Inner: ConstTranscribable,
    F::Modulus: ConstTranscribable,
{
    fn read_transcription_bytes_exact(bytes: &[u8]) -> Self {
        if bytes.is_empty() {
            return Vec::new();
        }
        let mod_size = F::Modulus::NUM_BYTES;
        let cfg = super::read_field_cfg::<F>(&bytes[..mod_size]);
        super::read_field_vec_with_cfg(&bytes[mod_size..], &cfg)
    }

    fn write_transcription_bytes_exact(&self, buf: &mut [u8]) {
        if self.is_empty() {
            return;
        }
        let buf = super::append_field_cfg::<F>(buf, &self[0].modulus());
        let buf = super::append_field_vec_inner(buf, self);
        assert!(
            buf.is_empty(),
            "Buffer size mismatch for Vec<F> transcription"
        );
    }
}

impl<F> Transcribable for Vec<F>
where
    F: PrimeField,
    F::Inner: ConstTranscribable,
    F::Modulus: ConstTranscribable,
{
    fn get_num_bytes(&self) -> usize {
        if self.is_empty() {
            0
        } else {
            add!(F::Modulus::NUM_BYTES, mul!(self.len(), F::Inner::NUM_BYTES))
        }
    }
}
