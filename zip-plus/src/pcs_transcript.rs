use std::io::{Cursor, ErrorKind, Read, Write};

use crypto_primitives::PrimeField;
use p3_matrix::Dimensions;

use crate::{
    ZipError,
    pcs::utils::MerkleProof,
    rem,
    traits::{Transcribable, Transcript},
    transcript::KeccakTranscript,
};

/// A transcript for Polynomial Commitment Scheme (PCS) operations.
/// Manages both Fiat-Shamir transformations and serialization/deserialization
/// of proof data.
#[derive(Default, Clone)]
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

    /// Converts the transcript into a serialized proof as a byte vector.
    pub fn into_proof(self) -> Vec<u8> {
        self.stream.into_inner()
    }

    /// Creates a transcript from an existing serialized proof.
    pub fn from_proof(proof: &[u8]) -> Self {
        Self {
            fs_transcript: KeccakTranscript::default(),
            stream: Cursor::new(proof.to_vec()),
        }
    }

    /// Absorbs a field element into the Fiat-Shamir transcript.
    /// This is used to incorporate public values into the transcript for
    /// challenge generation.
    pub fn common_field_element<F>(&mut self, fe: &F)
    where
        F: PrimeField,
        F::Inner: Transcribable,
    {
        self.fs_transcript.absorb_random_field(fe);
    }

    // TODO if we change this to an iterator we may be able to save some memory
    pub fn write_field_elements<F>(&mut self, elems: &[F]) -> Result<(), ZipError>
    where
        F: PrimeField,
        F::Inner: Transcribable,
    {
        let mut buf = vec![0; F::Inner::NUM_BYTES];
        for elem in elems {
            self.write_field_element(elem, &mut buf)?;
        }

        Ok(())
    }

    pub fn read_field_elements<F>(&mut self, n: usize) -> Result<Vec<F>, ZipError>
    where
        F: PrimeField,
        F::Inner: Transcribable,
    {
        (0..n)
            .map(|_| self.read_field_element())
            .collect::<Result<Vec<_>, _>>()
    }

    /// Reads a field element from the proof stream and absorbs it into the
    /// transcript. Used during proof verification to retrieve and process
    /// field elements.
    pub fn read_field_element<F>(&mut self) -> Result<F, ZipError>
    where
        F: PrimeField,
        F::Inner: Transcribable,
    {
        let inner = self.read()?;
        let fe = F::new_unchecked(inner);
        self.common_field_element(&fe);
        Ok(fe)
    }

    /// Writes a field element to the proof stream and absorbs it into the
    /// transcript. Used during proof generation to store field elements for
    /// later verification.
    pub fn write_field_element<F>(&mut self, fe: &F, buf: &mut [u8]) -> Result<(), ZipError>
    where
        F: PrimeField,
        F::Inner: Transcribable,
    {
        self.common_field_element(fe);
        self.write(fe.inner(), buf)
    }

    pub fn write<T: Transcribable>(&mut self, v: &T, buf: &mut [u8]) -> Result<(), ZipError> {
        v.to_transcription_bytes(buf);
        self.stream
            .write_all(buf)
            .map_err(|err| ZipError::Transcript(err.kind(), err.to_string()))?;
        Ok(())
    }

    pub fn write_many<'a, T: Transcribable + 'a, I>(&mut self, vs: I) -> Result<(), ZipError>
    where
        I: IntoIterator<Item = &'a T>,
    {
        let mut buf = vec![0; T::NUM_BYTES];
        for v in vs {
            self.write(v, &mut buf)?;
        }

        Ok(())
    }

    pub fn read<T: Transcribable>(&mut self) -> Result<T, ZipError> {
        let mut buf = vec![0u8; T::NUM_BYTES];
        self.stream
            .read_exact(&mut buf)
            .map_err(|err| ZipError::Transcript(err.kind(), err.to_string()))?;
        Ok(T::from_transcription_bytes(&buf))
    }

    pub fn read_many<T: Transcribable>(&mut self, n: usize) -> Result<Vec<T>, ZipError> {
        (0..n).map(|_| self.read()).collect::<Result<Vec<_>, _>>()
    }

    fn write_usize(
        &mut self,
        value: usize,
        buf: &mut [u8; size_of::<u64>()],
    ) -> Result<(), ZipError> {
        let value_u64: u64 = value.try_into().map_err(|_| {
            ZipError::Transcript(
                ErrorKind::Unsupported,
                "Failed to convert usize to u64".to_string(),
            )
        })?;
        self.write(&value_u64, buf)
    }

    fn read_usize(&mut self) -> Result<usize, ZipError> {
        self.read::<u64>()?.try_into().map_err(|_| {
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
        let width = self.read_usize()?;
        let height = self.read_usize()?;
        let dimensions = Dimensions { width, height };

        // Read the length of the merkle path first
        let path_length = self.read_usize()?;

        // Read each element of the merkle path
        let merkle_path = self.read_many(path_length)?;

        Ok(MerkleProof::new(merkle_path, dimensions))
    }

    pub fn write_merkle_proof(&mut self, proof: &MerkleProof) -> Result<(), ZipError> {
        let mut buf = [0u8; size_of::<u64>()];

        // Write the dimensions of matrix used to construct the Merkle tree
        self.write_usize(proof.matrix_dims.width, &mut buf)?;
        self.write_usize(proof.matrix_dims.height, &mut buf)?;

        // Write the length of the merkle path first
        self.write_usize(proof.path.len(), &mut buf)?;

        // Write each element of the merkle path
        self.write_many(&proof.path)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pcs::utils::MtHash;
    use crypto_bigint::{U256, const_monty_params};

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
            let proof = transcript.into_proof();
            let mut transcript = PcsTranscript::from_proof(&proof);
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
            let proof = transcript.into_proof();
            let mut transcript = PcsTranscript::from_proof(&proof);
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
        const_monty_params!(
            ModP,
            U256,
            "0000000000000000000000000000000000000000000000000000000000000091"
        );

        // Test commitment
        let original_commitment = MtHash::default();
        test_read_write!(write, read, original_commitment, "commitment");

        // Test vector of commitments
        let original_commitments = vec![MtHash::default(); 1024];
        test_read_write_vec!(
            write_many,
            read_many,
            original_commitments,
            "commitments vector"
        );
    }
}
