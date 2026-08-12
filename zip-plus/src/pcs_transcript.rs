use crate::{merkle::MerkleProof, pcs::structs::ZipPlusCommitment};
use crypto_primitives::BaseFieldConfig;
use itertools::Itertools;
use std::{
    borrow::Borrow,
    io::{Cursor, ErrorKind, Read, Write},
};
use zinc_transcript::{
    Blake3Transcript, TranscriptError,
    traits::{ConstTranscribable, Transcribable, Transcript},
};
use zinc_utils::{add, mul, rem};

macro_rules! safe_cast {
    ($value:expr, $from:ident, $to:ident) => {
        $to::try_from($value).map_err(|_err| {
            TranscriptError(
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

macro_rules! common_methods {
    () => {
        /// Generates a pseudorandom index based on the current transcript state.
        /// Used to create deterministic challenges for zero-knowledge protocols.
        /// Returns an index between 0 and cap-1.
        #[allow(clippy::unwrap_used)]
        pub fn squeeze_challenge_idx(&mut self, cap: usize) -> usize {
            let num = safe_cast!(self.fs_transcript.get_challenge::<u32>(), u32, usize)
                .expect("Conversion from u32 to usize should never fail");
            rem!(num, cap, "Challenge cap is zero")
        }
    };
}

/// A transcript for Polynomial Commitment Scheme (PCS) operations.
/// Manages both Fiat-Shamir transformations and serialization/deserialization
/// of proof data.
///
/// Every byte written to the proof stream is absorbed into the Fiat-Shamir
/// transcript by the same call
#[derive(Debug, Clone)]
pub struct PcsProverTranscript {
    /// Handles Fiat-Shamir transformations for non-interactive zero-knowledge
    /// proofs. Used to absorb field elements and generate cryptographic
    /// challenges.
    pub fs_transcript: Blake3Transcript,

    /// Manages serialization and deserialization of proof data as a byte
    /// stream.
    pub stream: Cursor<Vec<u8>>,
}

// TODO(alex): Review this vs Transcribable, there is some overlap that needs to
//             be resolved
impl PcsProverTranscript {
    pub fn new_from_commitment(comm: &ZipPlusCommitment) -> Self {
        Self::new_from_commitments(std::slice::from_ref(comm).iter())
    }

    pub fn new_from_commitments<'a>(comms: impl Iterator<Item = &'a ZipPlusCommitment>) -> Self {
        let mut result = Self {
            fs_transcript: Blake3Transcript::default(),
            stream: Cursor::default(),
        };

        for comm in comms {
            result.fs_transcript.absorb_bytes(&comm.root);
        }

        result
    }

    pub fn reserve_capacity(&mut self, additional_capacity: usize) {
        self.stream.get_mut().reserve(additional_capacity)
    }

    /// Transform the prover transcript into a verifier transcript by resetting
    /// the stream. Note that the commitment must be absorbed again into the
    /// verifier transcript. This would normally be done by the verifier, but
    /// this allows us more flexibility in how we use the transcript.
    pub fn into_verification_transcript(self) -> PcsVerifierTranscript {
        let mut result = PcsVerifierTranscript {
            fs_transcript: Blake3Transcript::default(),
            stream: self.stream,
        };
        result.stream.set_position(0);

        result
    }

    common_methods!();

    /// Writes field elements to the proof stream and absorbs them into the
    /// transcript, as raw (inner-representation) bytes.
    ///
    /// The field modulus is NOT written here.
    /// Absorbs the elements into the Fiat-Shamir transcript (raw inner
    /// representation) and writes their canonical lifted integers to the
    /// proof stream. The wire never carries raw Montgomery residues.
    pub fn write_field_elements<C>(
        &mut self,
        cfg: &C,
        elems: &[C::Element],
    ) -> Result<(), TranscriptError>
    where
        C: BaseFieldConfig,
        C::Integer: ConstTranscribable,
    {
        self.write_const_many_iter(elems.iter().map(|e| cfg.lift(e)), elems.len())
    }

    pub fn write<T: Transcribable>(&mut self, v: &T) -> Result<(), TranscriptError> {
        let data_len = v.get_num_bytes();

        // Write the length prefix when it is not known at compile time.
        if T::LENGTH_NUM_BYTES > 0 {
            let len_bytes = data_len
                .to_le_bytes()
                .into_iter()
                .take(T::LENGTH_NUM_BYTES)
                .collect_vec();
            self.stream
                .write_all(&len_bytes)
                .map_err(to_transcript_error)?;
            // A length that selects how much of the stream is read later is
            // itself a prover message, so it has to bind.
            self.fs_transcript.absorb_bytes(&len_bytes);
        }

        let prev_pos = safe_cast!(self.stream.position(), u64, usize)?;
        let next_pos = add!(prev_pos, data_len);

        let inner = self.stream.get_mut();
        if inner.len() < next_pos {
            inner.resize(next_pos, 0_u8);
        }

        let inner_slice = &mut inner[prev_pos..next_pos];
        v.write_transcription_bytes_exact(inner_slice);
        self.fs_transcript.absorb_bytes(inner_slice);

        self.stream.set_position(safe_cast!(next_pos, usize, u64)?);
        Ok(())
    }

    // Note(alex):
    // Parallelizing this greatly degrades performance rather than improving it.
    // Maybe we should think of breakpoints for parallelization later.
    pub fn write_const_many<T: ConstTranscribable>(
        &mut self,
        vs: &[T],
    ) -> Result<(), TranscriptError> {
        self.write_const_many_iter::<T, _>(vs, vs.len())
    }

    // Note(alex):
    // Parallelizing this greatly degrades performance rather than improving it.
    // Maybe we should think of breakpoints for parallelization later.
    pub fn write_const_many_iter<'a, T, I>(
        &mut self,
        vs: I,
        vs_len: usize,
    ) -> Result<(), TranscriptError>
    where
        T: ConstTranscribable + 'a,
        I: IntoIterator,
        I::Item: Borrow<T>,
    {
        let prev_pos = safe_cast!(self.stream.position(), u64, usize)?;
        let data_len = mul!(vs_len, T::NUM_BYTES);
        let next_pos = add!(prev_pos, data_len);

        let inner = self.stream.get_mut();
        // Enlarge the inner buffer if needed
        if inner.len() < next_pos {
            inner.resize(next_pos, 0_u8);
        }

        for (chunk, v) in inner[prev_pos..next_pos].chunks_mut(T::NUM_BYTES).zip(vs) {
            v.borrow().write_transcription_bytes_exact(chunk);
            self.fs_transcript.absorb_bytes(chunk);
        }

        self.stream.set_position(next_pos as u64);
        Ok(())
    }

    fn write_usize(&mut self, value: usize) -> Result<(), TranscriptError> {
        let value_u64 = safe_cast!(value, usize, u64)?;
        self.write(&value_u64)
    }

    pub fn write_merkle_proof(&mut self, proof: &MerkleProof) -> Result<(), TranscriptError> {
        // Write the dimensions of matrix used to construct the Merkle tree
        self.write_usize(proof.leaf_index)?;
        self.write_usize(proof.leaf_count)?;

        // Write the length of the merkle path first
        self.write_usize(proof.siblings.len())?;

        // Write each element of the merkle path
        self.write_const_many(&proof.siblings)?;
        Ok(())
    }
}

/// Version of [[PcsProverTranscript]] used for proof verification.
#[derive(Debug, Clone)]
pub struct PcsVerifierTranscript {
    /// Handles Fiat-Shamir transformations for non-interactive zero-knowledge
    /// proofs. Used to absorb field elements and generate cryptographic
    /// challenges.
    pub fs_transcript: Blake3Transcript,

    /// Manages serialization and deserialization of proof data as a byte
    /// stream.
    pub stream: Cursor<Vec<u8>>,
}

impl PcsVerifierTranscript {
    common_methods!();

    /// Returns an error unless the whole proof stream has been consumed.
    ///
    /// Call this once at the end of verification, after every component
    /// sharing the stream has read its section.
    pub fn check_eof(&self) -> Result<(), TranscriptError> {
        let position = safe_cast!(self.stream.position(), u64, usize)?;
        let len = self.stream.get_ref().len();
        if position == len {
            Ok(())
        } else {
            Err(TranscriptError(
                ErrorKind::InvalidData,
                format!(
                    "proof stream not fully consumed: {} unread byte(s)",
                    len.saturating_sub(position)
                ),
            ))
        }
    }

    /// Reads canonical lifted integers from the proof stream, strictly
    /// validates them against the field modulus, projects them into the
    /// field, and absorbs the resulting elements into the transcript. The
    /// mirror of [`PcsProverTranscript::write_field_elements`].
    ///
    /// Rejects any integer `>= modulus`: every field value has exactly one
    /// accepted encoding on the wire.
    pub fn read_field_elements<C>(
        &mut self,
        cfg: &C,
        n: usize,
    ) -> Result<Vec<C::Element>, TranscriptError>
    where
        C: BaseFieldConfig,
        C::Integer: ConstTranscribable,
    {
        let ints: Vec<C::Integer> = self.read_const_many(n)?;
        let modulus = cfg.modulus();
        if ints.iter().any(|int| *int >= modulus) {
            return Err(TranscriptError(
                ErrorKind::InvalidData,
                "Non-canonical field element".to_owned(),
            ));
        }
        let elems = ints.iter().map(|int| cfg.project(int)).collect_vec();
        Ok(elems)
    }

    pub fn read<T: Transcribable>(&mut self) -> Result<T, TranscriptError> {
        let data_len = if T::LENGTH_NUM_BYTES > 0 {
            let mut len_buf = vec![0u8; T::LENGTH_NUM_BYTES];
            self.stream
                .read_exact(&mut len_buf)
                .map_err(to_transcript_error)?;
            self.fs_transcript.absorb_bytes(&len_buf);
            T::read_num_bytes(&len_buf)
        } else {
            // LENGTH_NUM_BYTES == 0 means size is known at compile time via
            // the ConstTranscribable blanket impl; read_num_bytes accepts an
            // empty slice in that case.
            T::read_num_bytes(&[])
        };

        read_stream_slice(&mut self.stream, data_len, |slice| {
            self.fs_transcript.absorb_bytes(slice);
            Ok(T::read_transcription_bytes_exact(slice))
        })
    }

    pub fn read_const_many<T: ConstTranscribable>(
        &mut self,
        n: usize,
    ) -> Result<Vec<T>, TranscriptError> {
        read_stream_slice(&mut self.stream, mul!(n, T::NUM_BYTES), |slice| {
            Ok(slice
                .chunks(T::NUM_BYTES)
                .map(|bs| {
                    self.fs_transcript.absorb_bytes(bs);
                    T::read_transcription_bytes_exact(bs)
                })
                .collect_vec())
        })
    }

    fn read_usize(&mut self) -> Result<usize, TranscriptError> {
        let value = self.read::<u64>()?;
        safe_cast!(value, u64, usize)
    }

    pub fn read_merkle_proof(&mut self) -> Result<MerkleProof, TranscriptError> {
        // Read the dimensions of matrix used to construct the Merkle tree
        let leaf_index = self.read_usize()?;
        let leaf_count = self.read_usize()?;

        // Read the length of the merkle path first
        let path_length = self.read_usize()?;

        // Read each element of the merkle path
        let merkle_path = self.read_const_many(path_length)?;

        Ok(MerkleProof::new(leaf_index, leaf_count, merkle_path))
    }
}

/// Perform a bounds-checked read from the stream for a length, and
/// execute an action on the resulting slice. After the action is executed,
/// advance the stream position by the length.
#[inline]
fn read_stream_slice<T>(
    stream: &mut Cursor<Vec<u8>>,
    length: usize,
    mut action: impl FnMut(&[u8]) -> Result<T, TranscriptError>,
) -> Result<T, TranscriptError> {
    let prev_pos = safe_cast!(stream.position(), u64, usize)?;
    let next_pos = add!(prev_pos, length);

    let stream_vec = stream.get_ref();
    if next_pos > stream_vec.len() {
        return Err(TranscriptError(
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
fn to_transcript_error(err: std::io::Error) -> TranscriptError {
    TranscriptError(err.kind(), err.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::merkle::MtHash;

    #[allow(unused_macros)]
    macro_rules! test_read_write {
        // TODO: N is magic
        ($write_fn:ident, $read_fn:ident, $original_value:expr, $assert_msg:expr) => {{
            let comm = ZipPlusCommitment::default();
            let mut transcript = PcsProverTranscript::new_from_commitment(&comm);
            transcript
                .$write_fn(&$original_value)
                .expect(&format!("Failed to write {}", $assert_msg));
            let mut transcript: PcsVerifierTranscript = transcript.into_verification_transcript();
            transcript.fs_transcript.absorb_bytes(&comm.root);
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
            let comm = ZipPlusCommitment::default();
            let mut transcript = PcsProverTranscript::new_from_commitment(&comm);
            transcript
                .$write_fn(&$original_values)
                .expect(&format!("Failed to write {}", $assert_msg));
            let mut transcript: PcsVerifierTranscript = transcript.into_verification_transcript();
            transcript.fs_transcript.absorb_bytes(&comm.root);
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
        // Test hash
        let original_hash = MtHash::default();
        test_read_write!(write, read, original_hash, "hash");

        // Test vector of hashed
        let original_hashes = vec![MtHash::default(); 1024];
        test_read_write_vec!(
            write_const_many,
            read_const_many,
            original_hashes,
            "hashes vector"
        );
    }

    const CAP: usize = 1 << 24;

    /// Every byte on the wire must bind the challenges drawn after it.
    ///
    /// This pins the invariant whose absence let a prover choose
    /// `combined_row` *after* learning the column indices it was supposed to
    /// be committed to beforehand. Before wire writes were absorbed, the two
    /// payloads below produced identical challenges.
    #[test]
    fn wire_writes_bind_subsequent_challenges() {
        let comm = ZipPlusCommitment::default();

        let challenge_after = |payload: &[u64]| {
            let mut transcript = PcsProverTranscript::new_from_commitment(&comm);
            transcript
                .write_const_many(payload)
                .expect("write should succeed");
            transcript.squeeze_challenge_idx(CAP)
        };

        assert_ne!(
            challenge_after(&[1, 2, 3, 4]),
            challenge_after(&[1, 2, 3, 5]),
            "changing a written value left the next challenge unchanged: \
             wire bytes are not entering the transcript"
        );
    }

    /// A length prefix selects how much of the stream is read later, so it is
    /// a prover message and must bind just like a payload.
    #[test]
    fn length_prefixed_writes_bind_subsequent_challenges() {
        let comm = ZipPlusCommitment::default();

        let challenge_after = |value: u64| {
            let mut transcript = PcsProverTranscript::new_from_commitment(&comm);
            transcript.write(&value).expect("write should succeed");
            transcript.squeeze_challenge_idx(CAP)
        };

        assert_ne!(
            challenge_after(7),
            challenge_after(8),
            "`write` is not absorbing its payload"
        );
    }

    /// Prover and verifier must reach the same state after the same logical
    /// message, no matter how either side chunks it into calls.
    ///
    /// Absorbs are framed, so call granularity is normally significant. The
    /// `write_const_many_iter` / `read_const_many` pair frames one fixed-size
    /// element at a time rather than one call at a time, which is what makes
    /// the split into calls irrelevant here. That property is load-bearing:
    /// the prover writes column values one codeword matrix at a time while
    /// the verifier reads them in a single batched call.
    #[test]
    fn transcript_state_is_independent_of_call_chunking() {
        let comm = ZipPlusCommitment::default();
        let payload: Vec<u64> = (0..8).collect();

        // Prover emits the run as two calls ...
        let mut prover = PcsProverTranscript::new_from_commitment(&comm);
        prover
            .write_const_many(&payload[..3])
            .expect("write should succeed");
        prover
            .write_const_many(&payload[3..])
            .expect("write should succeed");
        let prover_challenge = prover.squeeze_challenge_idx(CAP);

        // ... the verifier consumes it as one.
        let mut verifier = prover.into_verification_transcript();
        verifier.fs_transcript.absorb_bytes(&comm.root);
        let read: Vec<u64> = verifier
            .read_const_many(payload.len())
            .expect("read should succeed");
        let verifier_challenge = verifier.squeeze_challenge_idx(CAP);

        assert_eq!(read, payload, "round-trip lost data");
        assert_eq!(
            prover_challenge, verifier_challenge,
            "prover and verifier transcripts diverged on identical data"
        );
        verifier
            .check_eof()
            .expect("stream should be fully consumed");
    }

    /// Trailing bytes are bytes that never entered the transcript, so they
    /// must be rejected rather than silently ignored.
    #[test]
    fn check_eof_rejects_unread_trailing_bytes() {
        let comm = ZipPlusCommitment::default();
        let mut prover = PcsProverTranscript::new_from_commitment(&comm);
        prover
            .write_const_many(&[1u64, 2, 3])
            .expect("write should succeed");

        let mut verifier = prover.into_verification_transcript();
        verifier.fs_transcript.absorb_bytes(&comm.root);
        let _: Vec<u64> = verifier.read_const_many(2).expect("read should succeed");

        assert!(
            verifier.check_eof().is_err(),
            "check_eof accepted a stream with unread trailing bytes"
        );
    }
}
