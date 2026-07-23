use itertools::Itertools;
use zinc_poly::univariate::dynamic::{DynamicPolyVec, DynamicPolynomial};
use zinc_transcript::traits::{ConstTranscribable, GenTranscribable, Transcribable};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Proof<F> {
    pub combined_mle_values: Vec<DynamicPolynomial<F>>,
}

impl<F> Proof<F> {
    /// Maps every field element through `f`, preserving structure — used to
    /// lift elements into wire integers and to project wire integers back
    /// into elements at the (de)serialization boundary.
    pub fn try_map<T, E>(&self, f: impl FnMut(&F) -> Result<T, E> + Copy) -> Result<Proof<T>, E> {
        Ok(Proof {
            combined_mle_values: self
                .combined_mle_values
                .iter()
                .map(|p| p.try_map(f))
                .try_collect()?,
        })
    }
}

impl<F> GenTranscribable for Proof<F>
where
    F: ConstTranscribable,
{
    fn read_transcription_bytes_exact(bytes: &[u8]) -> Self {
        let combined_mle_values = DynamicPolyVec::read_transcription_bytes_exact(bytes).0;
        Self {
            combined_mle_values,
        }
    }

    fn write_transcription_bytes_exact(&self, buf: &mut [u8]) {
        DynamicPolyVec::reinterpret(&self.combined_mle_values).write_transcription_bytes_exact(buf);
    }
}

impl<F> Transcribable for Proof<F>
where
    F: ConstTranscribable,
{
    fn get_num_bytes(&self) -> usize {
        DynamicPolyVec::reinterpret(&self.combined_mle_values).get_num_bytes()
    }
}

#[derive(Clone, Debug)]
pub struct VerifierSubclaim<F> {
    pub evaluation_point: Vec<F>,
    pub values: Vec<DynamicPolynomial<F>>,
}
