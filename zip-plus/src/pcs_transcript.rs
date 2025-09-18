use std::io::{Cursor, ErrorKind, Read, Write};

use crypto_bigint::Word;
use crypto_primitives::{PrimeField, crypto_bigint_int::Int};
use p3_matrix::Dimensions;

use crate::{
    ZipError,
    pcs::utils::{HASH_OUT_LEN, MerkleProof, MtHash},
    traits::{ConstNumBytes, FromBytes, ToBytes, Transcribable},
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
        F::Inner: ToBytes,
    {
        self.fs_transcript.absorb_random_field(fe);
    }

    /// Reads a cryptographic commitment from the proof stream.
    /// Used during proof verification to retrieve previously committed values.
    pub fn read_commitment(&mut self) -> Result<MtHash, ZipError> {
        let mut buf = [0; HASH_OUT_LEN];
        self.stream
            .read_exact(&mut buf)
            .map_err(|err| ZipError::Transcript(err.kind(), err.to_string()))?;
        Ok(MtHash(buf))
    }

    /// Writes a cryptographic commitment to the proof stream.
    /// Used during proof generation to store commitments for later
    /// verification.
    pub fn write_commitment(&mut self, comm: &MtHash) -> Result<(), ZipError> {
        self.stream
            .write_all(&comm.0)
            .map_err(|err| ZipError::Transcript(err.kind(), err.to_string()))?;
        Ok(())
    }

    // TODO if we change this to an iterator we may be able to save some memory
    pub fn write_field_elements<F>(&mut self, elems: &[F]) -> Result<(), ZipError>
    where
        F: PrimeField,
        F::Inner: Transcribable,
    {
        for elem in elems {
            self.write_field_element(elem)?;
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
        let mut bytes: Vec<u8> = vec![0; <F::Inner as ConstNumBytes>::NUM_BYTES];

        self.stream
            .read_exact(&mut bytes)
            .map_err(|err| ZipError::Transcript(err.kind(), err.to_string()))?;

        let fe = F::new_unchecked(F::Inner::from_be_bytes(&bytes));

        self.common_field_element(&fe);
        Ok(fe)
    }

    /// Writes a field element to the proof stream and absorbs it into the
    /// transcript. Used during proof generation to store field elements for
    /// later verification.
    pub fn write_field_element<F>(&mut self, fe: &F) -> Result<(), ZipError>
    where
        F: PrimeField,
        F::Inner: Transcribable,
    {
        self.common_field_element(fe);
        let repr = fe.inner().to_be_bytes();
        self.stream
            .write_all(repr.as_slice())
            .map_err(|err| ZipError::Transcript(err.kind(), err.to_string()))
    }

    pub fn write_integer<const M: usize>(&mut self, int: &Int<M>) -> Result<(), ZipError> {
        for &word in int.inner().as_words().iter() {
            let bytes = word.to_be_bytes();
            self.stream
                .write_all(&bytes)
                .map_err(|err| ZipError::Transcript(err.kind(), err.to_string()))?;
        }
        Ok(())
    }

    pub fn write_integers<'a, const M: usize, I>(&mut self, ints: I) -> Result<(), ZipError>
    where
        I: Iterator<Item = &'a Int<M>>,
    {
        for i in ints {
            self.write_integer(i)?;
        }

        Ok(())
    }

    pub fn read_integer<const M: usize>(&mut self) -> Result<Int<M>, ZipError> {
        let mut result = Int::default().into_inner();

        for word in result.as_mut_words() {
            let mut buf = [0u8; size_of::<Word>()];
            self.stream
                .read_exact(&mut buf)
                .map_err(|err| ZipError::Transcript(err.kind(), err.to_string()))?;

            *word = Word::from_be_bytes(buf);
        }
        Ok(result.into())
    }

    pub fn read_integers<const M: usize>(&mut self, n: usize) -> Result<Vec<Int<M>>, ZipError> {
        (0..n)
            .map(|_| self.read_integer())
            .collect::<Result<Vec<_>, _>>()
    }

    pub fn read_commitments(&mut self, n: usize) -> Result<Vec<MtHash>, ZipError> {
        (0..n).map(|_| self.read_commitment()).collect()
    }

    fn write_u64(&mut self, value: u64) -> Result<(), ZipError> {
        self.stream
            .write_all(&value.to_be_bytes())
            .map_err(|err| ZipError::Transcript(err.kind(), err.to_string()))
    }

    fn read_u64(&mut self) -> Result<u64, ZipError> {
        let mut buf = [0u8; 8];
        self.stream
            .read_exact(&mut buf)
            .map_err(|err| ZipError::Transcript(err.kind(), err.to_string()))?;
        Ok(u64::from_be_bytes(buf))
    }

    fn write_usize(&mut self, value: usize) -> Result<(), ZipError> {
        let value_u64: u64 = value.try_into().map_err(|_| {
            ZipError::Transcript(
                ErrorKind::Unsupported,
                "Failed to convert usize to u64".to_string(),
            )
        })?;
        self.write_u64(value_u64)
    }

    fn read_usize(&mut self) -> Result<usize, ZipError> {
        self.read_u64()?.try_into().map_err(|_| {
            ZipError::Transcript(
                ErrorKind::Unsupported,
                "Failed to convert u64 to usize".to_string(),
            )
        })
    }

    pub fn write_commitments<'a>(
        &mut self,
        comms: impl IntoIterator<Item = &'a MtHash>,
    ) -> Result<(), ZipError> {
        for comm in comms.into_iter() {
            self.write_commitment(comm)?;
        }
        Ok(())
    }

    /// Generates a pseudorandom index based on the current transcript state.
    /// Used to create deterministic challenges for zero-knowledge protocols.
    /// Returns an index between 0 and cap-1.
    pub fn squeeze_challenge_idx(&mut self, cap: usize) -> usize {
        let challenge: Int<1> = self.fs_transcript.get_integer_challenge();
        let bytes = challenge.inner().as_words()[0].to_be_bytes();
        let num = u32::from_be_bytes(bytes[..4].try_into().unwrap()) as usize;
        num % cap
    }

    pub fn read_merkle_proof(&mut self) -> Result<MerkleProof, ZipError> {
        // Read the dimensions of matrix used to construct the Merkle tree
        let width = self.read_usize()?;
        let height = self.read_usize()?;
        let dimensions = Dimensions { width, height };

        // Read the length of the merkle path first
        let path_length = self.read_usize()?;

        // Read each element of the merkle path
        let mut merkle_path = Vec::with_capacity(path_length);
        for _ in 0..path_length {
            merkle_path.push(self.read_commitment()?);
        }

        Ok(MerkleProof::new(merkle_path, dimensions))
    }

    pub fn write_merkle_proof(&mut self, proof: &MerkleProof) -> Result<(), ZipError> {
        // Write the dimensions of matrix used to construct the Merkle tree
        self.write_usize(proof.matrix_dims.width)?;
        self.write_usize(proof.matrix_dims.height)?;

        // Write the length of the merkle path first
        self.write_usize(proof.path.len())?;

        // Write each element of the merkle path
        for path_elem in proof.path.iter() {
            self.write_commitment(path_elem)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::pcs_transcript::{MtHash, PcsTranscript};
    use crypto_bigint::{U256, const_monty_params};

    #[allow(unused_macros)]
    macro_rules! test_read_write {
        // TODO: N is magic
        ($write_fn:ident, $read_fn:ident, $original_value:expr, $assert_msg:expr) => {{
            use ark_std::format;
            let mut transcript = PcsTranscript::new();
            transcript
                .$write_fn(&$original_value)
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
        test_read_write!(
            write_commitment,
            read_commitment,
            original_commitment,
            "commitment"
        );
        //TODO put the tests back in for Int<N> types
        // Test vector of commitments
        let original_commitments = vec![MtHash::default(); 1024];
        test_read_write_vec!(
            write_commitments,
            read_commitments,
            original_commitments,
            "commitments vector"
        );
    }
}
