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
    /// Evaluations of the per-binary_poly-column bit-slice MLEs at the
    /// shared point, flat layout `col*D + bit`. Populated by the
    /// algebraic booleanity sumcheck (a separate degree-3 group running
    /// alongside the CPR group); empty when no binary_poly columns are
    /// present.
    pub bit_slice_evals: Vec<F>,
    /// Evaluations of the bit-op virtual (down) MLEs at the shared
    /// point — one per `BitOpSpec`. Their consistency is verified in
    /// Step 4.5 against `ψ(op(lifted_eval[source_col]))`.
    pub bit_op_down_evals: Vec<F>,
    /// Evaluations of bit-slice MLEs at *shifted* shared points, per
    /// declared `ShiftedBitSliceSpec`, flat layout `spec*D + bit`.
    /// Bound to the corresponding `down_evals` entry by the projection-
    /// element consistency check (no separate sumcheck participation).
    pub shifted_bit_slice_evals: Vec<F>,
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
        let (bit_op_down_evals, bytes) = Vec::<F>::read_transcription_bytes_subset(bytes);
        let (shifted_bit_slice_evals, bytes) = Vec::<F>::read_transcription_bytes_subset(bytes);
        assert!(bytes.is_empty(), "All bytes should be consumed");
        Self {
            up_evals,
            down_evals,
            bit_slice_evals,
            bit_op_down_evals,
            shifted_bit_slice_evals,
        }
    }

    fn write_transcription_bytes_exact(&self, buf: &mut [u8]) {
        let buf = self.up_evals.write_transcription_bytes_subset(buf);
        let buf = self.down_evals.write_transcription_bytes_subset(buf);
        let buf = self.bit_slice_evals.write_transcription_bytes_subset(buf);
        let buf = self.bit_op_down_evals.write_transcription_bytes_subset(buf);
        let buf = self.shifted_bit_slice_evals.write_transcription_bytes_subset(buf);
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
            5 * u32::NUM_BYTES,
            add!(
                self.up_evals.get_num_bytes(),
                add!(
                    self.down_evals.get_num_bytes(),
                    add!(
                        self.bit_slice_evals.get_num_bytes(),
                        add!(
                            self.bit_op_down_evals.get_num_bytes(),
                            self.shifted_bit_slice_evals.get_num_bytes()
                        )
                    )
                )
            )
        )
    }
}

impl<F: PrimeField> Proof<F> {
    /// Check that all eval vectors have the expected lengths.
    pub fn validate_evaluation_sizes(
        &self,
        num_cols: usize,
        num_down_cols: usize,
        num_bit_slices: usize,
        num_bit_ops: usize,
        num_shifted_bit_slices: usize,
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

        if self.bit_op_down_evals.len() != num_bit_ops {
            return Err(CombinedPolyResolverError::WrongBitOpDownEvalsNumber {
                got: self.bit_op_down_evals.len(),
                expected: num_bit_ops,
            });
        }

        if self.shifted_bit_slice_evals.len() != num_shifted_bit_slices {
            return Err(CombinedPolyResolverError::WrongShiftedBitSliceEvalsNumber {
                got: self.shifted_bit_slice_evals.len(),
                expected: num_shifted_bit_slices,
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
    /// Number of bit-op virtual (down) columns. Their evaluations at r*
    /// land in `CprProof::bit_op_down_evals`.
    pub num_bit_op_cols: usize,
    /// Number of variables — used to index the last challenge.
    pub num_vars: usize,
}

/// Ancillary data produced by `prepare_verifier` and consumed by
/// `finalize_verifier`. Holds state that bridges the pre-sumcheck and
/// post-sumcheck halves of the CPR verifier.
pub struct CprVerifierAncillary<F: PrimeField> {
    /// Powers of the folding challenge α: [1, α, α², ..., α^{k-1}].
    pub folding_challenge_powers: Vec<F>,
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
    /// Bit-slice MLE evaluation claims (flat `col*D + bit`).
    pub bit_slice_evals: Vec<F>,
    /// Bit-op virtual MLE evaluation claims at r* — one per
    /// `BitOpSpec`. The verifier checks each against
    /// `ψ(op(lifted_eval[source_col]))` in Step 4.5.
    pub bit_op_down_evals: Vec<F>,
    /// Shifted bit-slice MLE evaluation claims (flat `spec*D + bit`).
    pub shifted_bit_slice_evals: Vec<F>,
}
