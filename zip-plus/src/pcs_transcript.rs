use crate::{ZipError, merkle::MerkleProof};
use crypto_primitives::PrimeField;
use itertools::Itertools;
use std::io::{Cursor, ErrorKind, Read, Write};
use zinc_transcript::{
    KeccakTranscript,
    traits::{ConstTranscribable, Transcribable, Transcript},
};
use zinc_utils::{add, mul, rem};

macro_rules! safe_cast {
    ($value:expr, $from:ident, $to:ident) => {
        $to::try_from($value).map_err(|_err| {
            ZipError::Transcript(
                ErrorKind::Unsupported,
                format!(
                    "Failed to convert {} to {}",
                    stringify!($from),
                    stringify!($to)
                ),
            )
        })
    };
}

/// A transcript for Polynomial Commitment Scheme (PCS) operations.
/// Manages both Fiat-Shamir transformations and serialization/deserialization
/// of proof data.
#[derive(Debug, Clone, Default)]
pub struct PcsTranscript {
    /// Handles Fiat-Shamir transformations for non-interactive zero-knowledge
    /// proofs. Used to absorb field elements and generate cryptographic
    /// challenges.
    pub fs_transcript: KeccakTranscript,

    /// Manages serialization and deserialization of proof data as a byte
    /// stream.
    pub stream: Cursor<Vec<u8>>,
}

impl PcsTranscript {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn new_with_capacity(capacity: usize) -> Self {
        // We zero-initialize the stream vector in advance to avoid dealing with
        // MaybeUninit
        Self {
            fs_transcript: Default::default(),
            stream: Cursor::new(vec![0; capacity]),
        }
    }

    // TODO if we change this to an iterator we may be able to save some memory
    pub fn write_field_elements<F>(&mut self, elems: &[F]) -> Result<(), ZipError>
    where
        F: PrimeField,
        F::Inner: Transcribable,
        F::Modulus: Transcribable,
    {
        if !elems.is_empty() {
            let num_bytes = F::Inner::get_num_bytes(elems[0].inner());
            let num_bytes_arr = num_bytes
                .to_le_bytes()
                .into_iter()
                .take(F::Inner::LENGTH_NUM_BYTES)
                .collect_vec();
            self.stream.write_all(&num_bytes_arr)?;

            let mut buf = vec![0; num_bytes];
            for elem in elems {
                self.write_field_element_no_length(elem, &mut buf)?;
            }
        }

        Ok(())
    }

    pub fn read_field_elements<F>(
        &mut self,
        n: usize,
        field_cfg: &F::Config,
    ) -> Result<Vec<F>, ZipError>
    where
        F: PrimeField,
        F::Inner: Transcribable,
        F::Modulus: Transcribable,
    {
        if n > 0 {
            let mut buf = vec![0; F::Inner::LENGTH_NUM_BYTES];
            self.stream.read_exact(&mut buf)?;
            let num_bytes = F::Inner::read_num_bytes(&buf);

            let mut buf = vec![0; num_bytes];
            (0..n)
                .map(|_| self.read_field_element_no_length(&mut buf, field_cfg))
                .collect::<Result<Vec<_>, _>>()
        } else {
            Ok(vec![])
        }
    }

    /// Reads a field element from the proof stream and absorbs it into the
    /// transcript. Used during proof verification to retrieve and process
    /// field elements.
    ///
    /// Only the inner value is read; the modulus is derived from the provided
    /// field_cfg (which the verifier obtained from the Fiat-Shamir transcript).
    ///
    /// Provided buffer must be of exact size of the field element.
    fn read_field_element_no_length<F>(
        &mut self,
        buf: &mut [u8],
        field_cfg: &F::Config,
    ) -> Result<F, ZipError>
    where
        F: PrimeField,
        F::Inner: Transcribable,
        F::Modulus: Transcribable,
    {
        self.stream.read_exact(buf)?;
        let inner = F::Inner::read_transcription_bytes(buf);
        let fe = F::new_unchecked_with_cfg(inner, field_cfg);
        self.fs_transcript.absorb_random_field(&fe, buf);
        Ok(fe)
    }

    /// Writes a field element to the proof stream and absorbs it into the
    /// transcript. Used during proof generation to store field elements for
    /// later verification.
    ///
    /// Only the inner value is written; the modulus is omitted since the
    /// verifier derives it from the Fiat-Shamir transcript.
    ///
    /// Field element length must've been written before calling this method.
    fn write_field_element_no_length<F>(&mut self, fe: &F, buf: &mut [u8]) -> Result<(), ZipError>
    where
        F: PrimeField,
        F::Inner: Transcribable,
        F::Modulus: Transcribable,
    {
        self.fs_transcript.absorb_random_field(fe, buf);
        fe.inner().write_transcription_bytes(buf);
        self.stream.write_all(buf)?;
        // Modulus is NOT written — verifier derives it from the FS transcript.
        Ok(())
    }

    pub fn write_const<T: ConstTranscribable>(&mut self, v: &T) -> Result<(), ZipError> {
        let prev_pos = safe_cast!(self.stream.position(), u64, usize)?;
        let data_len = T::NUM_BYTES;
        let next_pos = add!(prev_pos, data_len);

        let inner = self.stream.get_mut();
        // Enlarge the inner buffer if needed
        if inner.len() < next_pos {
            inner.resize(next_pos, 0_u8);
        }

        v.write_transcription_bytes(&mut inner[prev_pos..next_pos]);

        self.stream.set_position(safe_cast!(next_pos, usize, u64)?);
        Ok(())
    }

    // Note(alex):
    // Parallelizing this greatly degrades performance rather than improving it.
    // Maybe we should think of breakpoints for parallelization later.
    pub fn write_const_many<T: ConstTranscribable>(&mut self, vs: &[T]) -> Result<(), ZipError> {
        self.write_const_many_iter(vs.iter(), vs.len())
    }

    // Note(alex):
    // Parallelizing this greatly degrades performance rather than improving it.
    // Maybe we should think of breakpoints for parallelization later.
    pub fn write_const_many_iter<'a, T, I>(&mut self, vs: I, vs_len: usize) -> Result<(), ZipError>
    where
        T: ConstTranscribable + 'a,
        I: IntoIterator<Item = &'a T>,
    {
        let prev_pos = safe_cast!(self.stream.position(), u64, usize)?;
        let data_len = mul!(vs_len, T::NUM_BYTES);
        let next_pos = add!(prev_pos, data_len);

        let inner = self.stream.get_mut();
        // Enlarge the inner buffer if needed
        if inner.len() < next_pos {
            inner.resize(next_pos, 0_u8);
        }

        inner[prev_pos..next_pos]
            .chunks_mut(T::NUM_BYTES)
            .zip(vs)
            .for_each(|(chunk, v)| v.write_transcription_bytes(chunk));

        self.stream.set_position(next_pos as u64);
        Ok(())
    }

    pub fn read_const<T: ConstTranscribable>(&mut self) -> Result<T, ZipError> {
        read_stream_slice(&mut self.stream, T::NUM_BYTES, |slice| {
            Ok(T::read_transcription_bytes(slice))
        })
    }

    pub fn read_const_many<T: ConstTranscribable>(&mut self, n: usize) -> Result<Vec<T>, ZipError> {
        read_stream_slice(&mut self.stream, mul!(n, T::NUM_BYTES), |slice| {
            Ok(slice
                .chunks(T::NUM_BYTES)
                .map(T::read_transcription_bytes)
                .collect_vec())
        })
    }

    fn write_usize(&mut self, value: usize) -> Result<(), ZipError> {
        let value_u64 = safe_cast!(value, usize, u64)?;
        self.write_const(&value_u64)
    }

    fn read_usize(&mut self) -> Result<usize, ZipError> {
        safe_cast!(self.read_const::<u64>()?, u64, usize)
    }

    /// Generates a pseudorandom index based on the current transcript state.
    /// Used to create deterministic challenges for zero-knowledge protocols.
    /// Returns an index between 0 and cap-1.
    #[allow(clippy::unwrap_used)]
    pub fn squeeze_challenge_idx(&mut self, cap: usize) -> usize {
        let num = self.fs_transcript.get_challenge::<u32>() as usize;
        rem!(num, cap, "Challenge cap is zero")
    }

    pub fn read_merkle_proof(&mut self) -> Result<MerkleProof, ZipError> {
        // Read the dimensions of matrix used to construct the Merkle tree
        let leaf_index = self.read_usize()?;
        let leaf_count = self.read_usize()?;

        // Read the length of the merkle path first
        let path_length = self.read_usize()?;

        // Read each element of the merkle path
        let merkle_path = self.read_const_many(path_length)?;

        Ok(MerkleProof::new(leaf_index, leaf_count, merkle_path))
    }

    pub fn write_merkle_proof(&mut self, proof: &MerkleProof) -> Result<(), ZipError> {
        // Write the dimensions of matrix used to construct the Merkle tree
        self.write_usize(proof.leaf_index)?;
        self.write_usize(proof.leaf_count)?;

        // Write the length of the merkle path first
        self.write_usize(proof.siblings.len())?;

        // Write each element of the merkle path
        self.write_const_many(&proof.siblings)?;
        Ok(())
    }

    // ── Grinding (proof-of-work) ──────────────────────────────────────

    /// Size in bytes of the grinding nonce written to the proof stream.
    pub const GRINDING_NONCE_BYTES: usize = 8; // u64

    /// Prover: search for a 64-bit nonce such that
    /// `BLAKE3(transcript_state ‖ nonce)` has at least `grinding_bits`
    /// leading zero bits, then write the nonce to the proof stream and
    /// absorb it into the Fiat-Shamir transcript.
    ///
    /// If `grinding_bits == 0` this is a no-op (no bytes written).
    pub fn grind(&mut self, grinding_bits: usize) -> Result<(), ZipError> {
        if grinding_bits == 0 {
            return Ok(());
        }
        assert!(
            grinding_bits <= 64,
            "grinding_bits must be <= 64, got {grinding_bits}"
        );

        // Snapshot the current FS state: clone the hasher, then squeeze
        // a 32-byte challenge seed that commits to everything absorbed so far.
        let seed: [u8; 32] = {
            let mut buf = [0u8; 32];
            self.fs_transcript.fill_with_random_bytes(&mut buf);
            buf
        };

        // Search for a valid nonce.
        // With `parallel` feature, search in parallel using rayon to divide
        // wall-clock time by the number of available cores.
        let nonce = Self::find_grinding_nonce(&seed, grinding_bits);

        // Write nonce to proof stream.
        self.write_const(&nonce)?;
        // Absorb nonce into FS transcript so subsequent challenges depend on it.
        self.fs_transcript.absorb(&nonce.to_le_bytes());

        Ok(())
    }

    /// Find the smallest nonce such that `BLAKE3(seed ‖ nonce)` has at least
    /// `grinding_bits` leading zero bits.
    ///
    /// When the `parallel` feature is enabled, threads search interleaved
    /// nonce positions (thread *t* checks *t*, *t+T*, *t+2T*, …) so the
    /// wall-clock cost is ≈ sequential / num_threads.  The result is always
    /// the globally-smallest qualifying nonce, so proofs are identical
    /// regardless of parallelism.
    fn find_grinding_nonce(seed: &[u8; 32], grinding_bits: usize) -> u64 {
        // Check whether nonce `n` satisfies the proof-of-work.
        // Uses a caller-provided stack buffer to avoid heap allocation.
        #[inline(always)]
        fn check(buf: &mut [u8; 40], n: u64, grinding_bits: usize) -> bool {
            buf[32..].copy_from_slice(&n.to_le_bytes());
            let hash = blake3::hash(buf);
            // Fast leading-zero test: read the first 8 bytes as a big-endian
            // u64 and use hardware `lzcnt`.  Valid for grinding_bits <= 64.
            let head = u64::from_be_bytes(
                hash.as_bytes()[..8]
                    .try_into()
                    .expect("blake3 output is 32 bytes"),
            );
            head.leading_zeros() as usize >= grinding_bits
        }

        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            use std::sync::atomic::{AtomicU64, Ordering};

            let num_threads = rayon::current_num_threads() as u64;
            // Shared upper bound — once any thread finds a valid nonce the
            // others can stop as soon as they've scanned up to that point.
            let upper = AtomicU64::new(u64::MAX);

            (0..num_threads).into_par_iter().for_each(|t| {
                let mut buf = [0u8; 40];
                buf[..32].copy_from_slice(seed);
                let mut n = t;
                loop {
                    if n >= upper.load(Ordering::Relaxed) {
                        return;
                    }
                    if check(&mut buf, n, grinding_bits) {
                        upper.fetch_min(n, Ordering::Relaxed);
                        return;
                    }
                    n += num_threads;
                }
            });

            let result = upper.load(Ordering::SeqCst);
            assert_ne!(
                result,
                u64::MAX,
                "grinding search exhausted (unreachable in practice)"
            );
            result
        }

        #[cfg(not(feature = "parallel"))]
        {
            let mut buf = [0u8; 40];
            buf[..32].copy_from_slice(seed);
            (0u64..)
                .find(|&n| check(&mut buf, n, grinding_bits))
                .expect("grinding search exhausted u64 range (unreachable in practice)")
        }
    }

    /// Verifier: read the grinding nonce from the proof stream and verify
    /// that `BLAKE3(transcript_state ‖ nonce)` has at least `grinding_bits`
    /// leading zero bits. Absorbs the nonce into the FS transcript.
    ///
    /// If `grinding_bits == 0` this is a no-op (no bytes read).
    pub fn verify_grind(&mut self, grinding_bits: usize) -> Result<(), ZipError> {
        if grinding_bits == 0 {
            return Ok(());
        }
        assert!(
            grinding_bits <= 64,
            "grinding_bits must be <= 64, got {grinding_bits}"
        );

        // Reproduce the same seed the prover used.
        let seed: [u8; 32] = {
            let mut buf = [0u8; 32];
            self.fs_transcript.fill_with_random_bytes(&mut buf);
            buf
        };

        // Read the nonce from the proof stream.
        let nonce: u64 = self.read_const()?;

        // Verify the proof-of-work.
        let mut buf = [0u8; 40];
        buf[..32].copy_from_slice(&seed);
        buf[32..].copy_from_slice(&nonce.to_le_bytes());
        let hash = blake3::hash(&buf);
        if leading_zeros(hash.as_bytes()) < grinding_bits {
            return Err(ZipError::InvalidPcsOpen(
                format!(
                    "Grinding verification failed: expected >= {grinding_bits} leading zero bits"
                ),
            ));
        }

        // Absorb nonce into FS transcript (must match prover).
        self.fs_transcript.absorb(&nonce.to_le_bytes());

        Ok(())
    }
}

/// Count the number of leading zero bits in a byte slice (big-endian).
fn leading_zeros(bytes: &[u8]) -> usize {
    // Fast path: check the first 8 bytes as a single u64 using hardware
    // `lzcnt`.  For blake3 output (always 32 bytes) this covers up to 64
    // leading zero bits, which is the maximum grinding_bits we support.
    if bytes.len() >= 8 {
        let head = u64::from_be_bytes(
            bytes[..8]
                .try_into()
                .expect("slice is >= 8 bytes"),
        );
        if head != 0 {
            return head.leading_zeros() as usize;
        }
        // Extremely rare: first 64 bits all zero.  Fall through to count
        // the remaining bytes.
        let mut count = 64;
        for &b in &bytes[8..] {
            if b == 0 {
                count += 8;
            } else {
                count += b.leading_zeros() as usize;
                break;
            }
        }
        count
    } else {
        let mut count = 0;
        for &b in bytes {
            if b == 0 {
                count += 8;
            } else {
                count += b.leading_zeros() as usize;
                break;
            }
        }
        count
    }
}

/// Perform a bounds-checked read from the stream for a length, and
/// execute an action on the resulting slice. After the action is executed,
/// advance the stream position by the length.
#[inline]
fn read_stream_slice<T>(
    stream: &mut Cursor<Vec<u8>>,
    length: usize,
    action: impl Fn(&[u8]) -> Result<T, ZipError>,
) -> Result<T, ZipError> {
    let prev_pos = safe_cast!(stream.position(), u64, usize)?;
    let next_pos = add!(prev_pos, length);

    let stream_vec = stream.get_ref();
    if next_pos > stream_vec.len() {
        return Err(ZipError::Transcript(
            ErrorKind::UnexpectedEof,
            format!(
                "Attempted to read beyond the end of the stream: {} + {} exceeds stream length {}",
                prev_pos,
                length,
                stream_vec.len()
            ),
        ));
    }
    let res = action(&stream_vec[prev_pos..next_pos])?;
    stream.set_position(safe_cast!(next_pos, usize, u64)?);
    Ok(res)
}

// Do not expose this outside
impl From<std::io::Error> for ZipError {
    fn from(err: std::io::Error) -> Self {
        ZipError::Transcript(err.kind(), err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{merkle::MtHash, pcs::ZipPlusProof};

    #[allow(unused_macros)]
    macro_rules! test_read_write {
        // TODO: N is magic
        ($write_fn:ident, $read_fn:ident, $original_value:expr, $assert_msg:expr) => {{
            let mut transcript = PcsTranscript::new();
            transcript
                .$write_fn(&$original_value)
                .expect(&format!("Failed to write {}", $assert_msg));
            let proof: ZipPlusProof = transcript.into();
            let mut transcript: PcsTranscript = proof.into();
            let read_value = transcript
                .$read_fn()
                .expect(&format!("Failed to read {}", $assert_msg));
            assert_eq!(
                $original_value, read_value,
                "{} read does not match original",
                $assert_msg
            );
        }};
    }

    #[allow(unused_macros)]
    macro_rules! test_read_write_vec {
        // TODO: N is magic
        ($write_fn:ident, $read_fn:ident, $original_values:expr, $assert_msg:expr) => {{
            let mut transcript = PcsTranscript::new();
            transcript
                .$write_fn(&$original_values)
                .expect(&format!("Failed to write {}", $assert_msg));
            let proof: ZipPlusProof = transcript.into();
            let mut transcript: PcsTranscript = proof.into();
            let read_values = transcript
                .$read_fn($original_values.len())
                .expect(&format!("Failed to read {}", $assert_msg));
            assert_eq!(
                $original_values, read_values,
                "{} read does not match original",
                $assert_msg
            );
        }};
    }

    #[test]
    fn test_pcs_transcript_read_write() {
        // Test commitment
        let original_commitment = MtHash::default();
        test_read_write!(write_const, read_const, original_commitment, "commitment");

        // Test vector of commitments
        let original_commitments = vec![MtHash::default(); 1024];
        test_read_write_vec!(
            write_const_many,
            read_const_many,
            original_commitments,
            "commitments vector"
        );
    }
}
