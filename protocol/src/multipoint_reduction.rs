use crypto_primitives::FromPrimitiveWithConfig;
use num_traits::Zero;
use zinc_piop::multipoint_eval::{
    MultipointEval, MultipointEvalError, Proof as MultipointEvalProof,
    Subclaim as MultipointSubclaim,
};
use zinc_poly::mle::DenseMultilinearExtension;
use zinc_transcript::traits::{ConstTranscribable, Transcript};
use zinc_uair::ShiftSpec;
use zinc_utils::{
    delayed_reduction::DelayedFieldProductSum, inner_transparent_field::InnerTransparentField,
};

pub(crate) fn prove_multipoint_reduction<F>(
    transcript: &mut impl Transcript,
    trace_mles: &[DenseMultilinearExtension<F::Inner>],
    eval_point: &[F],
    up_evals: &[F],
    down_evals: &[F],
    shifts: &[ShiftSpec],
    field_cfg: &F::Config,
) -> Result<(MultipointEvalProof<F>, Vec<F>), MultipointEvalError<F>>
where
    F: InnerTransparentField
        + DelayedFieldProductSum
        + FromPrimitiveWithConfig
        + Send
        + Sync
        + 'static,
    F::Inner: ConstTranscribable + Zero + Default + Send + Sync,
    F::Modulus: ConstTranscribable,
{
    let (proof, state) = MultipointEval::prove_as_subprotocol(
        transcript, trace_mles, eval_point, up_evals, down_evals, shifts, field_cfg,
    )?;
    Ok((proof, state.eval_point))
}

pub(crate) fn verify_multipoint_reduction<F>(
    transcript: &mut impl Transcript,
    proof: MultipointEvalProof<F>,
    eval_point: &[F],
    up_evals: &[F],
    down_evals: &[F],
    shifts: &[ShiftSpec],
    num_vars: usize,
    field_cfg: &F::Config,
) -> Result<MultipointSubclaim<F>, MultipointEvalError<F>>
where
    F: InnerTransparentField
        + DelayedFieldProductSum
        + FromPrimitiveWithConfig
        + Send
        + Sync
        + 'static,
    F::Inner: ConstTranscribable + Zero + Default + Send + Sync,
    F::Modulus: ConstTranscribable,
{
    MultipointEval::verify_as_subprotocol(
        transcript, proof, eval_point, up_evals, down_evals, shifts, num_vars, field_cfg,
    )
}
