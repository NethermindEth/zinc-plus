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
use zinc_utils::from_ref::FromRef;

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
        F::Inner: FromRef<FMod>,
        T: PrimalityTest<FMod>,
    {
        let prime = self.get_prime::<FMod, T>();

        F::make_cfg(&F::Inner::from_ref(&prime)).expect("prime is guaranteed to be prime")
    }

    /// Absorbs a byte slice into the hash sponge.
    fn absorb(&mut self, v: &[u8]);

    /// Absorbs a field element into the transcript.
    /// Delegates to the field element's implementation of
    /// absorb_into_transcript.
    fn absorb_random_field<F>(&mut self, v: &F, buf: &mut [u8])
    where
        F: PrimeField,
        F::Inner: Transcribable;

    /// Absorbs a slice of field element into the transcript.
    /// Delegates to the field element's implementation of
    /// absorb_into_transcript.
    fn absorb_random_field_slice<F>(&mut self, v: &[F], buf: &mut [u8])
    where
        F: PrimeField,
        F::Inner: Transcribable,
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
            impl ConstTranscribable for $type {
                const NUM_BYTES: usize = std::mem::size_of::<$type>();

                fn read_transcription_bytes(bytes: &[u8]) -> Self {
                    Self::from_le_bytes(bytes.try_into().expect("Invalid byte slice length"))
                }

                fn write_transcription_bytes(&self, buf: &mut [u8]) {
                    debug_assert_eq!(buf.len(), std::mem::size_of::<$type>());
                    buf.copy_from_slice(&self.to_le_bytes());
                }
            }
        )+
    };
}

impl_transcribable_for_primitives!(u8, u16, u32, u64, u128);
impl_transcribable_for_primitives!(i8, i16, i32, i64, i128);

impl ConstTranscribable for Boolean {
    const NUM_BYTES: usize = 1;
    const NUM_BITS: usize = 1;

    fn read_transcription_bytes(bytes: &[u8]) -> Self {
        (bytes[0] != 0).into()
    }

    fn write_transcription_bytes(&self, buf: &mut [u8]) {
        buf[0] = self.to_u8();
    }
}

impl<const LIMBS: usize> ConstTranscribable for Uint<LIMBS> {
    const NUM_BYTES: usize = 8 * LIMBS / WORD_FACTOR;

    fn read_transcription_bytes(bytes: &[u8]) -> Self {
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
    fn write_transcription_bytes(&self, buf: &mut [u8]) {
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

impl<const LIMBS: usize> ConstTranscribable for Int<LIMBS> {
    const NUM_BYTES: usize = Uint::<LIMBS>::NUM_BYTES;

    fn read_transcription_bytes(bytes: &[u8]) -> Self {
        *<Uint<LIMBS> as ConstTranscribable>::read_transcription_bytes(bytes).as_int()
    }

    fn write_transcription_bytes(&self, buf: &mut [u8]) {
        ConstTranscribable::write_transcription_bytes(self.as_uint(), buf)
    }
}

impl Transcribable for BoxedUint {
    /// Up to 255 bytes - so up to 2040 bits - should be plenty.
    const LENGTH_NUM_BYTES: usize = 1;

    fn read_num_bytes(bytes: &[u8]) -> usize {
        assert_eq!(bytes.len(), Self::LENGTH_NUM_BYTES);
        usize::from(bytes[0])
    }

    fn read_transcription_bytes(bytes: &[u8]) -> Self {
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

    fn get_num_bytes(&self) -> usize {
        usize::from(u8::try_from(self.bytes_precision()).expect("BoxedUint size must fit into u8"))
    }

    #[allow(clippy::arithmetic_side_effects)]
    fn write_transcription_bytes(&self, buf: &mut [u8]) {
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
