use crate::{combined_poly_resolver::CombinedPolyResolverError, sumcheck::SumcheckProof};
use crypto_primitives::PrimeField;
use zinc_transcript::traits::{ConstTranscribable, GenTranscribable, Transcribable};

/// The proof type of the combined polynomial resolver
/// subprotocol.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Proof<F: PrimeField> {
    /// A proof of the inner sumcheck subprotocol used
    /// for resolving combined polynomial evaluation claims.
    pub sumcheck_proof: SumcheckProof<F>,
    /// The evaluation of the projected trace columns MLEs.
    pub up_evals: Vec<F>,
    /// The evaluations of the shifted projected trace columns MLEs.
    pub down_evals: Vec<F>,
}

impl<F: PrimeField> GenTranscribable for Proof<F>
where
    F::Inner: ConstTranscribable,
    F::Modulus: ConstTranscribable,
{
    fn read_transcription_bytes_exact(bytes: &[u8]) -> Self {
        let (sumcheck_proof, bytes) = SumcheckProof::<F>::read_transcription_bytes_subset(bytes);
        let (up_evals, bytes) = Vec::<F>::read_transcription_bytes_subset(bytes);
        let (down_evals, bytes) = Vec::<F>::read_transcription_bytes_subset(bytes);
        assert!(bytes.is_empty(), "All bytes should be consumed");
        Self {
            sumcheck_proof,
            up_evals,
            down_evals,
        }
    }

    fn write_transcription_bytes_exact(&self, buf: &mut [u8]) {
        let buf = self.sumcheck_proof.write_transcription_bytes_subset(buf);
        let buf = self.up_evals.write_transcription_bytes_subset(buf);
        let buf = self.down_evals.write_transcription_bytes_subset(buf);
        assert!(buf.is_empty(), "All bytes should be consumed");
    }
}

#[allow(clippy::arithmetic_side_effects)]
impl<F: PrimeField> Transcribable for Proof<F>
where
    F::Inner: ConstTranscribable,
    F::Modulus: ConstTranscribable,
{
    fn get_num_bytes(&self) -> usize {
        // Each sub-field is written as: u32 prefix + data (matching
        // read_transcription_bytes_subset)
        3 * u32::NUM_BYTES
            + self.sumcheck_proof.get_num_bytes()
            + self.up_evals.get_num_bytes()
            + self.down_evals.get_num_bytes()
    }
}

impl<F: PrimeField> Proof<F> {
    /// Check if `up_evals` and `down_evals` vectors
    /// has the length `num_cols`.
    pub fn validate_evaluation_sizes(
        &self,
        num_cols: usize,
        num_down_cols: usize,
    ) -> Result<(), CombinedPolyResolverError<F>> {
        if self.up_evals.len() != num_cols {
            return Err(CombinedPolyResolverError::WrongUpEvalsNumber {
                got: self.up_evals.len(),
                expected: num_cols,
            });
        }

        if self.down_evals.len() != num_down_cols {
            return Err(CombinedPolyResolverError::WrongDownEvalsNumber {
                got: self.down_evals.len(),
                expected: num_down_cols,
            });
        }

        Ok(())
    }
}

pub struct ProverState<F: PrimeField> {
    /// The evaluation point yielded by the sumcheck
    /// subprotocol.
    pub evaluation_point: Vec<F>,
}

/// The claim that is left to be proven
/// after the combined polynomial resolver
/// verifier has succeeded verifying.
/// In this case, it is several evaluation claims
/// about the trace columns and the shifted trace columns
/// on the same evaluation point.
#[derive(Clone, Debug)]
pub struct VerifierSubclaim<F: PrimeField> {
    /// Evaluation point for the claims.
    pub evaluation_point: Vec<F>,
    /// Evaluation claims about the trace columns.
    pub up_evals: Vec<F>,
    /// Evaluation claims about the shifted trace columns.
    pub down_evals: Vec<F>,
}
