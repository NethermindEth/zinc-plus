use std::{fmt::Debug, marker::PhantomData};

use crypto_primitives::{FromPrimitiveWithConfig, FromWithConfig, PrimeField};
use zinc_poly::{
    mle::DenseMultilinearExtension, univariate::dynamic::over_field::DynamicPolynomialF,
};
use zinc_transcript::traits::{GenTranscribable, Transcribable, Transcript};
use zinc_utils::{from_ref::FromRef, mul_by_scalar::MulByScalar};

use crate::{
    ZipError,
    code::LinearCode,
    pcs::structs::{ZipPlus, ZipPlusCommitment, ZipPlusHint, ZipPlusParams, ZipTypes},
    pcs_transcript::{PcsProverTranscript, PcsVerifierTranscript},
};

/// Polynomial commitment scheme interface used by the Zinc+ protocol.
///
/// `Eval` is the unprojected witness cell type committed by the backend.
pub trait PCS<F, Eval, const D: usize>: Clone + Debug + Send + Sync
where
    F: PrimeField,
    Eval: Clone + Debug + Send + Sync,
{
    type CommitmentKey: Clone + Debug + Send + Sync;
    type VerifierKey: Clone + Debug + Send + Sync;
    type Commitment: Clone + Debug + Send + Sync;
    type ProverData: Clone + Debug + Send + Sync;

    fn precompute_ck(_ck: &Self::CommitmentKey) {}

    fn commit(
        ck: &Self::CommitmentKey,
        polys: &[DenseMultilinearExtension<Eval>],
    ) -> Result<(Self::ProverData, Self::Commitment), ZipError>;

    fn absorb_commitment<T: Transcript>(transcript: &mut T, commitment: &Self::Commitment);

    fn commitment_num_bytes(commitment: &Self::Commitment) -> usize;

    fn write_commitment_bytes(commitment: &Self::Commitment, buf: &mut Vec<u8>);

    fn batch_size(commitment: &Self::Commitment) -> usize;

    fn prove_open<const CHECK_FOR_OVERFLOW: bool>(
        transcript: &mut PcsProverTranscript,
        ck: &Self::CommitmentKey,
        polys: &[DenseMultilinearExtension<Eval>],
        point: &[F],
        prover_data: &Self::ProverData,
        field_cfg: &F::Config,
    ) -> Result<(), ZipError>
    where
        F::Inner: Transcribable,
        F::Modulus: Transcribable;

    fn verify_open<const CHECK_FOR_OVERFLOW: bool>(
        transcript: &mut PcsVerifierTranscript,
        vk: &Self::VerifierKey,
        commitment: &Self::Commitment,
        point: &[F],
        lifted_evals: &[DynamicPolynomialF<F>],
        field_cfg: &F::Config,
    ) -> Result<(), ZipError>
    where
        F::Inner: Transcribable,
        F::Modulus: Transcribable;
}

#[derive(Clone, Debug)]
pub struct ZipPlusPCS<Zt: ZipTypes, Lc: LinearCode<Zt>>(PhantomData<(Zt, Lc)>);

impl<F, Zt, Lc, const D: usize> PCS<F, Zt::Eval, D> for ZipPlusPCS<Zt, Lc>
where
    F: PrimeField
        + FromPrimitiveWithConfig
        + for<'a> FromWithConfig<&'a Zt::CombR>
        + for<'a> FromWithConfig<&'a Zt::Chal>
        + for<'a> MulByScalar<&'a F>
        + FromRef<F>,
    Zt: ZipTypes,
    Zt::Eval: Clone + Debug + Send + Sync,
    Lc: LinearCode<Zt>,
    F::Modulus: zinc_utils::from_ref::FromRef<Zt::Fmod>,
{
    type CommitmentKey = ZipPlusParams<Zt, Lc>;
    type VerifierKey = ZipPlusParams<Zt, Lc>;
    type Commitment = ZipPlusCommitment;
    type ProverData = Option<ZipPlusHint<Zt::Cw>>;

    fn commit(
        ck: &Self::CommitmentKey,
        polys: &[DenseMultilinearExtension<Zt::Eval>],
    ) -> Result<(Self::ProverData, Self::Commitment), ZipError> {
        if polys.is_empty() {
            return Ok((None, ZipPlusCommitment::default()));
        }
        let (hint, commitment) = ZipPlus::<Zt, Lc>::commit(ck, polys)?;
        Ok((Some(hint), commitment))
    }

    fn absorb_commitment<T: Transcript>(transcript: &mut T, commitment: &Self::Commitment) {
        transcript.absorb_slice(&commitment.root);
        transcript.absorb_slice(&(commitment.batch_size as u64).to_le_bytes());
    }

    fn commitment_num_bytes(commitment: &Self::Commitment) -> usize {
        commitment.get_num_bytes()
    }

    fn write_commitment_bytes(commitment: &Self::Commitment, buf: &mut Vec<u8>) {
        let offset = buf.len();
        buf.resize(offset + commitment.get_num_bytes(), 0);
        commitment.write_transcription_bytes_exact(&mut buf[offset..]);
    }

    fn batch_size(commitment: &Self::Commitment) -> usize {
        commitment.batch_size
    }

    fn prove_open<const CHECK_FOR_OVERFLOW: bool>(
        transcript: &mut PcsProverTranscript,
        ck: &Self::CommitmentKey,
        polys: &[DenseMultilinearExtension<Zt::Eval>],
        point: &[F],
        prover_data: &Self::ProverData,
        field_cfg: &F::Config,
    ) -> Result<(), ZipError>
    where
        F::Inner: Transcribable,
        F::Modulus: Transcribable,
    {
        match (polys.is_empty(), prover_data) {
            (true, None) => {}
            (true, Some(_)) => {
                return Err(ZipError::InvalidPcsParam(
                    "Zip+ prover data must be empty for an empty batch".to_string(),
                ));
            }
            (false, None) => {
                return Err(ZipError::InvalidPcsParam(
                    "Zip+ prover data missing for non-empty batch".to_string(),
                ));
            }
            (false, Some(hint)) => {
                let _ = ZipPlus::<Zt, Lc>::prove_f::<_, CHECK_FOR_OVERFLOW>(
                    transcript, ck, polys, point, hint, field_cfg,
                )?;
            }
        }
        Ok(())
    }

    fn verify_open<const CHECK_FOR_OVERFLOW: bool>(
        transcript: &mut PcsVerifierTranscript,
        vk: &Self::VerifierKey,
        commitment: &Self::Commitment,
        point: &[F],
        lifted_evals: &[DynamicPolynomialF<F>],
        field_cfg: &F::Config,
    ) -> Result<(), ZipError>
    where
        F::Inner: Transcribable,
        F::Modulus: Transcribable,
    {
        if lifted_evals.len() != commitment.batch_size {
            return Err(ZipError::InvalidPcsParam(format!(
                "Zip+ verifier expected {} lifted evals, got {}",
                commitment.batch_size,
                lifted_evals.len()
            )));
        }
        if commitment.batch_size == 0 {
            if commitment.root != Default::default() {
                return Err(ZipError::InvalidPcsParam(
                    "Zip+ empty batch must use the canonical empty commitment".to_string(),
                ));
            }
            return Ok(());
        }

        let per_poly_alphas =
            ZipPlus::<Zt, Lc>::sample_alphas(&mut transcript.fs_transcript, commitment.batch_size);
        let mut eval_f = F::zero_with_cfg(field_cfg);
        for (bar_u, alphas) in lifted_evals.iter().zip(per_poly_alphas.iter()) {
            for (coeff, alpha) in bar_u.coeffs.iter().zip(alphas.iter()) {
                let mut term = F::from_with_cfg(alpha, field_cfg);
                term *= coeff;
                eval_f += &term;
            }
        }

        ZipPlus::<Zt, Lc>::verify_with_alphas::<F, CHECK_FOR_OVERFLOW>(
            transcript,
            vk,
            commitment,
            field_cfg,
            point,
            &eval_f,
            &per_poly_alphas,
        )
    }
}
