use crypto_primitives::PrimeField;
use zinc_transcript::traits::{ConstTranscribable, GenTranscribable, Transcribable};
use zinc_utils::add;

use crate::combined_poly_resolver::CombinedPolyResolverError;

/// The proof type of the combined polynomial resolver subprotocol.
///
/// Note: the sumcheck proof now lives at the protocol
/// level as part of `MultiDegreeSumcheckProof`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Proof<F: PrimeField> {
    /// The evaluation of the projected trace columns MLEs at the shared point.
    pub up_evals: Vec<F>,
    /// The evaluations of the shifted projected trace columns MLEs at the
    /// shared point.
    pub down_evals: Vec<F>,
    /// Evaluations of the per-binary_poly-column bit-slice MLEs at the shared
    /// point, flattened column-major-then-bit-major: index `col*D + bit`.
    /// Empty when no binary_poly columns are present.
    pub bit_slice_evals: Vec<F>,
}

impl<F: PrimeField> GenTranscribable for Proof<F>
where
    F::Inner: ConstTranscribable,
    F::Modulus: ConstTranscribable,
{
    fn read_transcription_bytes_exact(bytes: &[u8]) -> Self {
        let (up_evals, bytes) = Vec::<F>::read_transcription_bytes_subset(bytes);
        let (down_evals, bytes) = Vec::<F>::read_transcription_bytes_subset(bytes);
        let (bit_slice_evals, bytes) = Vec::<F>::read_transcription_bytes_subset(bytes);
        assert!(bytes.is_empty(), "All bytes should be consumed");
        Self {
            up_evals,
            down_evals,
            bit_slice_evals,
        }
    }

    fn write_transcription_bytes_exact(&self, buf: &mut [u8]) {
        let buf = self.up_evals.write_transcription_bytes_subset(buf);
        let buf = self.down_evals.write_transcription_bytes_subset(buf);
        let buf = self.bit_slice_evals.write_transcription_bytes_subset(buf);
        assert!(buf.is_empty(), "Entire buffer should be used");
    }
}

impl<F: PrimeField> Transcribable for Proof<F>
where
    F::Inner: ConstTranscribable,
    F::Modulus: ConstTranscribable,
{
    fn get_num_bytes(&self) -> usize {
        add!(
            3 * u32::NUM_BYTES,
            add!(
                self.up_evals.get_num_bytes(),
                add!(
                    self.down_evals.get_num_bytes(),
                    self.bit_slice_evals.get_num_bytes()
                )
            )
        )
    }
}

impl<F: PrimeField> Proof<F> {
    /// Check that `up_evals`, `down_evals`, and `bit_slice_evals` have the
    /// expected lengths.
    pub fn validate_evaluation_sizes(
        &self,
        num_cols: usize,
        num_down_cols: usize,
        num_bit_slices: usize,
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

        if self.bit_slice_evals.len() != num_bit_slices {
            return Err(CombinedPolyResolverError::WrongBitSliceEvalsNumber {
                got: self.bit_slice_evals.len(),
                expected: num_bit_slices,
            });
        }

        Ok(())
    }
}

pub struct ProverState<F: PrimeField> {
    /// The shared evaluation point yielded by the multi-degree sumcheck.
    pub evaluation_point: Vec<F>,
}

/// Ancillary data produced by `prepare_sumcheck_group` and consumed by
/// `finalize_prover`. Holds everything needed to extract `up_evals` /
/// `down_evals` after the shared sumcheck completes.
pub struct CprProverAncillary {
    /// Number of trace (up) columns — used to split the flat evals vec.
    pub num_cols: usize,
    /// Number of shifted (down) columns.
    pub num_down_cols: usize,
    /// Number of bit-slice virtual MLEs (= num_binary_poly_cols * D).
    pub num_bit_slices: usize,
    /// Number of variables — used to index the last challenge.
    pub num_vars: usize,
}

/// Ancillary data produced by `prepare_verifier` and consumed by
/// `finalize_verifier`. Holds state that bridges the pre-sumcheck and
/// post-sumcheck halves of the CPR verifier.
pub struct CprVerifierAncillary<F: PrimeField> {
    /// Powers of the folding challenge α: [1, α, α², ..., α^{k-1}], extended
    /// to cover both UAIR constraints and per-bit-slice booleanity terms.
    pub folding_challenge_powers: Vec<F>,
    /// Number of UAIR constraints (the prefix of `folding_challenge_powers`
    /// that is consumed by `ConstraintFolder`; the remainder is for the
    /// booleanity terms).
    pub num_constraints: usize,
    /// Number of bit-slice virtual MLEs (= num_binary_poly_cols * D).
    pub num_bit_slices: usize,
    /// Evaluation point from the ideal check subclaim (for eq_r computation).
    pub ic_evaluation_point: Vec<F>,
    /// Number of variables (for selector computation).
    pub num_vars: usize,
}

/// The claim that is left to be proven after the combined polynomial resolver
/// verifier has succeeded. It is several evaluation claims about the trace
/// columns and the shifted trace columns at the same evaluation point.
#[derive(Clone, Debug)]
pub struct VerifierSubclaim<F: PrimeField> {
    /// Evaluation point for the claims.
    pub evaluation_point: Vec<F>,
    /// Evaluation claims about the trace columns.
    pub up_evals: Vec<F>,
    /// Evaluation claims about the shifted trace columns.
    pub down_evals: Vec<F>,
    /// Evaluation claims about the per-binary_poly-column bit-slice MLEs.
    /// Flattened column-major-then-bit-major: index `col*D + bit`.
    pub bit_slice_evals: Vec<F>,
}
