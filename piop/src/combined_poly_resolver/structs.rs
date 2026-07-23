use crate::combined_poly_resolver::CombinedPolyResolverError;
use crypto_primitives::SetElement;
use itertools::Itertools;
use std::fmt::Debug;
use zinc_transcript::traits::{ConstTranscribable, GenTranscribable, Transcribable};
use zinc_utils::add;

/// The proof type of the combined polynomial resolver subprotocol.
///
/// Note: the sumcheck proof now lives at the protocol
/// level as part of `MultiDegreeSumcheckProof`.
///
/// `bit_op_evals` is the prover's claim about the bit-op virtual columns'
/// MLE evaluations at the shared CPR point. These are *not* trusted by
/// themselves: they are bound back to the source columns' lifted openings at
/// the multi-point evaluation endpoint via Lemma 2.3.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Proof<F> {
    /// The evaluation of the projected trace columns MLEs at the shared point.
    pub up_evals: Vec<F>,
    /// The evaluations of the shifted projected trace columns MLEs at the
    /// shared point. Carries *only* the row-shift virtual columns —
    /// bit-op virtuals are sent separately in `bit_op_evals` to avoid
    /// overloading shift semantics on the wire.
    pub down_evals: Vec<F>,
    /// The evaluations of the bit-op virtual columns (ROTR / SHR) at the
    /// shared point, in `UairSignature::bit_op_specs()` order.
    pub bit_op_evals: Vec<F>,
}

impl<F> Proof<F> {
    /// Maps every field element through `f`, preserving structure — used to
    /// lift elements into wire integers and to project wire integers back
    /// into elements at the (de)serialization boundary.
    pub fn try_map<T, E>(&self, f: impl FnMut(&F) -> Result<T, E> + Copy) -> Result<Proof<T>, E> {
        Ok(Proof {
            up_evals: self.up_evals.iter().map(f).try_collect()?,
            down_evals: self.down_evals.iter().map(f).try_collect()?,
            bit_op_evals: self.bit_op_evals.iter().map(f).try_collect()?,
        })
    }
}

impl<F: ConstTranscribable> GenTranscribable for Proof<F> {
    fn read_transcription_bytes_exact(bytes: &[u8]) -> Self {
        let (up_evals, bytes) = Vec::<F>::read_transcription_bytes_subset(bytes);
        let (down_evals, bytes) = Vec::<F>::read_transcription_bytes_subset(bytes);
        let (bit_op_evals, bytes) = Vec::<F>::read_transcription_bytes_subset(bytes);
        assert!(bytes.is_empty(), "All bytes should be consumed");
        Self {
            up_evals,
            down_evals,
            bit_op_evals,
        }
    }

    fn write_transcription_bytes_exact(&self, buf: &mut [u8]) {
        let buf = self.up_evals.write_transcription_bytes_subset(buf);
        let buf = self.down_evals.write_transcription_bytes_subset(buf);
        let buf = self.bit_op_evals.write_transcription_bytes_subset(buf);
        assert!(buf.is_empty(), "Entire buffer should be used");
    }
}

impl<F: ConstTranscribable> Transcribable for Proof<F> {
    fn get_num_bytes(&self) -> usize {
        add!(
            3 * u32::NUM_BYTES,
            add!(
                self.up_evals.get_num_bytes(),
                add!(
                    self.down_evals.get_num_bytes(),
                    self.bit_op_evals.get_num_bytes()
                )
            )
        )
    }
}

impl<F: SetElement> Proof<F> {
    /// Check that the proof's evaluation vectors have the expected lengths.
    pub fn validate_evaluation_sizes(
        &self,
        num_cols: usize,
        num_down_cols: usize,
        num_bit_op_specs: usize,
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

        if self.bit_op_evals.len() != num_bit_op_specs {
            return Err(CombinedPolyResolverError::WrongBitOpEvalsNumber {
                got: self.bit_op_evals.len(),
                expected: num_bit_op_specs,
            });
        }

        Ok(())
    }
}

pub struct ProverState<F> {
    /// The shared evaluation point yielded by the multi-degree sumcheck.
    pub evaluation_point: Vec<F>,
}

/// Ancillary data produced by `prepare_sumcheck_group` and consumed by
/// `finalize_prover`. Holds everything needed to extract `up_evals` /
/// `down_evals` / `bit_op_evals` after the shared sumcheck completes.
pub struct CprProverAncillary {
    /// Number of trace (up) columns — used to split the flat evals vec.
    pub num_cols: usize,
    /// Number of shift-virtual (down) columns.
    pub num_down_cols: usize,
    /// Number of variables — used to index the last challenge.
    pub num_vars: usize,
}

/// Ancillary data produced by `prepare_verifier` and consumed by
/// `finalize_verifier`. Holds state that bridges the pre-sumcheck and
/// post-sumcheck halves of the CPR verifier.
pub struct CprVerifierAncillary<F> {
    /// Powers of the folding challenge α: [1, α, α², ..., α^{k-1}].
    pub folding_challenge_powers: Vec<F>,
    /// Evaluation point from the ideal check subclaim (for eq_r computation).
    pub ic_evaluation_point: Vec<F>,
    /// Number of variables (for selector computation).
    pub num_vars: usize,
}

/// The claim that is left to be proven after the combined polynomial resolver
/// verifier has succeeded. It is a list of evaluation claims at a common
/// evaluation point, covering committed trace columns, row-shift virtual
/// columns, and bit-op virtual columns. Bit-op evals are kept separate so the
/// downstream `MultipointEval` can bind them back to the source columns'
/// lifted openings (per Lemma 2.3 of the Zinc+ paper) rather than treating
/// them as standalone trusted evaluations.
#[derive(Clone, Debug)]
pub struct VerifierSubclaim<F> {
    /// Evaluation point for the claims.
    pub evaluation_point: Vec<F>,
    /// Evaluation claims about the trace columns.
    pub up_evals: Vec<F>,
    /// Evaluation claims about the row-shift virtual columns.
    pub down_evals: Vec<F>,
    /// Evaluation claims about the bit-op virtual columns, in
    /// `UairSignature::bit_op_specs()` order.
    pub bit_op_evals: Vec<F>,
}
