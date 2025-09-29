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

macro_rules! impl_transcribable_for_primitive {
    ($type:ty, $num_bytes:expr) => {
        impl Transcribable for $type {
            const NUM_BYTES: usize = $num_bytes;

            fn read_transcription_bytes(bytes: &[u8]) -> Self {
                assert_eq!(bytes.len(), Self::NUM_BYTES);
                let mut arr = [0u8; Self::NUM_BYTES];
                arr.copy_from_slice(bytes);
                Self::from_le_bytes(arr)
            }

            fn write_transcription_bytes(&self, buf: &mut [u8]) {
                assert_eq!(buf.len(), Self::NUM_BYTES);
                buf.copy_from_slice(&self.to_le_bytes());
            }
        }
    };
}

impl_transcribable_for_primitive!(u32, 4);
impl_transcribable_for_primitive!(u64, 8);

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
