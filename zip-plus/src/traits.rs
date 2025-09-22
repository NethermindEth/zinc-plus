/// Trait for types that can be transcribed to and from a byte representation.
/// Byte order is not specified, but it must be portable across platforms.
pub trait Transcribable {
    /// Number of bytes required to represent this type.
    const NUM_BYTES: usize;

    /// Creates a new instance from a byte buffer.
    /// The buffer must be exactly `NUM_BYTES` long.
    fn read_transcription_bytes(bytes: &[u8]) -> Self;

    /// Transcribes the current instance into a byte buffer.
    /// Buffer must be exactly `NUM_BYTES` long.
    fn write_transcription_bytes(&self, buf: &mut [u8]);
}

macro_rules! impl_transcribable_for_primitives {
    ($($type:ty),+) => {
        $(
            impl Transcribable for $type {
                const NUM_BYTES: usize = std::mem::size_of::<$type>();

                fn read_transcription_bytes(bytes: &[u8]) -> Self {
                    Self::from_le_bytes(bytes.try_into().expect("Invalid byte slice length"))
                }

                fn write_transcription_bytes(&self, buf: &mut [u8]) {
                    assert_eq!(buf.len(), Self::NUM_BYTES);
                    buf.copy_from_slice(&self.to_le_bytes());
                }
            }
        )+
    };
}

impl_transcribable_for_primitives!(u8, u16, u32, u64, u128);
impl_transcribable_for_primitives!(i8, i16, i32, i64, i128);

pub trait Transcript {
    /// Generates a pseudorandom transcribable value as a challenge based on the
    /// current transcript state, updating it.
    fn get_challenge<T: Transcribable>(&mut self) -> T;

    /// Generates a pseudorandom transcribable values as challenges based on the
    /// current transcript state, updating it.
    fn get_challenges<T: Transcribable>(&mut self, n: usize) -> Vec<T> {
        (0..n).map(|_| self.get_challenge()).collect()
    }
}
