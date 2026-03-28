use crypto_primitives::PrimeField;
use zinc_poly::univariate::dynamic::over_field::{DynamicPolyVecF, DynamicPolynomialF};
use zinc_transcript::traits::{ConstTranscribable, GenTranscribable, Transcribable};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Proof<F: PrimeField> {
    pub combined_mle_values: Vec<DynamicPolynomialF<F>>,
}

/// Only write modulus once
impl<F> GenTranscribable for Proof<F>
where
    F: PrimeField,
    F::Inner: ConstTranscribable,
    F::Modulus: ConstTranscribable,
{
    fn read_transcription_bytes_exact(bytes: &[u8]) -> Self {
        let combined_mle_values = DynamicPolyVecF::read_transcription_bytes_exact(bytes).0;
        Self {
            combined_mle_values,
        }
    }

    fn write_transcription_bytes_exact(&self, buf: &mut [u8]) {
        DynamicPolyVecF::reinterpret(&self.combined_mle_values)
            .write_transcription_bytes_exact(buf);
    }
}

impl<F> Transcribable for Proof<F>
where
    F: PrimeField,
    F::Inner: ConstTranscribable,
    F::Modulus: ConstTranscribable,
{
    fn get_num_bytes(&self) -> usize {
        DynamicPolyVecF::reinterpret(&self.combined_mle_values).get_num_bytes()
    }
}

#[derive(Clone, Debug)]
pub struct ProverState<F: PrimeField> {
    pub evaluation_point: Vec<F>,
}

#[derive(Clone, Debug)]
pub struct VerifierSubclaim<F: PrimeField> {
    pub evaluation_point: Vec<F>,
    pub values: Vec<DynamicPolynomialF<F>>,
}
