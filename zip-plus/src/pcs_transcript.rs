use crypto_primitives::PrimeField;
use itertools::Itertools;
use std::io::{Cursor, ErrorKind, Read, Write};
use zinc_transcript::{
    KeccakTranscript,
    traits::{ConstTranscribable, Transcribable, Transcript},
};
use zinc_utils::rem;

use crate::{ZipError, merkle::MerkleProof};

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

    // TODO if we change this to an iterator we may be able to save some memory
    pub fn write_field_elements<F>(&mut self, elems: &[F]) -> Result<(), ZipError>
    where
        F: PrimeField,
        F::Inner: Transcribable,
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

    pub fn read_field_elements<F>(&mut self, n: usize) -> Result<Vec<F>, ZipError>
    where
        F: PrimeField,
        F::Inner: Transcribable,
    {
        if n > 0 {
            let mut buf = vec![0; F::Inner::LENGTH_NUM_BYTES];
            self.stream.read_exact(&mut buf)?;
            let num_bytes = F::Inner::read_num_bytes(&buf);

            let mut buf = vec![0; num_bytes];
            (0..n)
                .map(|_| self.read_field_element_no_length(&mut buf))
                .collect::<Result<Vec<_>, _>>()
        } else {
            Ok(vec![])
        }
    }

    /// Reads a field element from the proof stream and absorbs it into the
    /// transcript. Used during proof verification to retrieve and process
    /// field elements.
    ///
    /// Provided buffer must be of exact size of the field element.
    fn read_field_element_no_length<F>(&mut self, buf: &mut [u8]) -> Result<F, ZipError>
    where
        F: PrimeField,
        F::Inner: Transcribable,
    {
        self.stream.read_exact(buf)?;
        let inner = F::Inner::read_transcription_bytes(buf);
        self.stream.read_exact(buf)?;
        let modulus = F::Inner::read_transcription_bytes(buf);
        let field_cfg = F::make_cfg(&modulus)?;
        let fe = F::new_unchecked_with_cfg(inner, &field_cfg);
        self.fs_transcript.absorb_random_field(&fe, buf);
        Ok(fe)
    }

    /// Writes a field element to the proof stream and absorbs it into the
    /// transcript. Used during proof generation to store field elements for
    /// later verification.
    ///
    /// Field element length must've been written before calling this method.
    fn write_field_element_no_length<F>(&mut self, fe: &F, buf: &mut [u8]) -> Result<(), ZipError>
    where
        F: PrimeField,
        F::Inner: Transcribable,
    {
        self.fs_transcript.absorb_random_field(fe, buf);
        fe.inner().write_transcription_bytes(buf);
        self.stream.write_all(buf)?;
        fe.modulus().write_transcription_bytes(buf);
        self.stream.write_all(buf)?;
        Ok(())
    }

    pub fn write_const<T: ConstTranscribable>(
        &mut self,
        v: &T,
        buf: &mut [u8],
    ) -> Result<(), ZipError> {
        v.write_transcription_bytes(buf);
        self.stream.write_all(buf)?;
        Ok(())
    }

    pub fn write_const_many<'a, T: ConstTranscribable + 'a, I>(
        &mut self,
        vs: I,
    ) -> Result<(), ZipError>
    where
        I: IntoIterator<Item = &'a T>,
    {
        let mut buf = vec![0; T::NUM_BYTES];
        for v in vs {
            self.write_const(v, &mut buf)?;
        }

        Ok(())
    }

    pub fn read_const<T: ConstTranscribable>(&mut self) -> Result<T, ZipError> {
        let mut buf = vec![0u8; T::NUM_BYTES];
        self.stream.read_exact(&mut buf)?;
        Ok(T::read_transcription_bytes(&buf))
    }

    pub fn read_const_many<T: ConstTranscribable>(&mut self, n: usize) -> Result<Vec<T>, ZipError> {
        (0..n)
            .map(|_| self.read_const())
            .collect::<Result<Vec<_>, _>>()
    }

    fn write_usize(
        &mut self,
        value: usize,
        buf: &mut [u8; size_of::<u64>()],
    ) -> Result<(), ZipError> {
        let value_u64: u64 = value.try_into().map_err(|_err| {
            ZipError::Transcript(
                ErrorKind::Unsupported,
                "Failed to convert usize to u64".to_string(),
            )
        })?;
        self.write_const(&value_u64, buf)
    }

    fn read_usize(&mut self) -> Result<usize, ZipError> {
        self.read_const::<u64>()?.try_into().map_err(|_| {
            ZipError::Transcript(
                ErrorKind::Unsupported,
                "Failed to convert u64 to usize".to_string(),
            )
        })
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
        let mut buf = [0u8; size_of::<u64>()];

        // Write the dimensions of matrix used to construct the Merkle tree
        self.write_usize(proof.leaf_index, &mut buf)?;
        self.write_usize(proof.leaf_count, &mut buf)?;

        // Write the length of the merkle path first
        self.write_usize(proof.siblings.len(), &mut buf)?;

        // Write each element of the merkle path
        self.write_const_many(&proof.siblings)?;
        Ok(())
    }
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
            use ark_std::format;
            let mut buf = vec![0u8; MtHash::NUM_BYTES];
            let mut transcript = PcsTranscript::new();
            transcript
                .$write_fn(&$original_value, &mut buf)
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
            use ark_std::format;
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
