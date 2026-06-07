use std::{fmt::Debug, io::Cursor, marker::PhantomData};

use crypto_primitives::{FromPrimitiveWithConfig, FromWithConfig, PrimeField};
use zinc_poly::{
    mle::DenseMultilinearExtension, univariate::dynamic::over_field::DynamicPolynomialF,
};
use zinc_transcript::traits::{GenTranscribable, Transcribable, Transcript};
use zinc_utils::{
    delayed_reduction::DelayedFieldProductSum, from_ref::FromRef, mul_by_scalar::MulByScalar,
};

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
    type OpeningProof: Clone + Debug + Send + Sync + Default;

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
    ) -> Result<Self::OpeningProof, ZipError>
    where
        F::Inner: Transcribable,
        F::Modulus: Transcribable;

    fn verify_open<const CHECK_FOR_OVERFLOW: bool>(
        transcript: &mut PcsVerifierTranscript,
        vk: &Self::VerifierKey,
        commitment: &Self::Commitment,
        point: &[F],
        lifted_evals: &[DynamicPolynomialF<F>],
        opening_proof: &Self::OpeningProof,
        field_cfg: &F::Config,
    ) -> Result<(), ZipError>
    where
        F::Inner: Transcribable,
        F::Modulus: Transcribable;
}

/// Homomorphic extension of [`PCS`] used by instance-axis folding protocols.
///
/// Implementations must satisfy:
///
/// ```text
/// fold_commitments([Com(w_i; eta_i)], theta)
///   = Com(sum_i theta_i w_i; sum_i theta_i eta_i)
/// ```
///
/// Non-homomorphic commitments, such as Merkle roots, must not implement this
/// trait.
pub trait FoldablePCS<F, Eval, const D: usize>: PCS<F, Eval, D>
where
    F: PrimeField,
    Eval: Clone + Debug + Send + Sync,
{
    fn fold_commitments(
        commitments: &[Self::Commitment],
        theta: &[F],
        field_cfg: &F::Config,
    ) -> Result<Self::Commitment, ZipError>;

    fn fold_prover_data(
        prover_data: &[Self::ProverData],
        theta: &[F],
        field_cfg: &F::Config,
    ) -> Result<Self::ProverData, ZipError>;
}

#[derive(Clone, Debug)]
pub struct ZipPlusPCS<Zt: ZipTypes, Lc: LinearCode<Zt>>(PhantomData<(Zt, Lc)>);

impl<F, Zt, Lc, const D: usize> PCS<F, Zt::Eval, D> for ZipPlusPCS<Zt, Lc>
where
    F: PrimeField
        + DelayedFieldProductSum
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
    type OpeningProof = Vec<u8>;

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
    ) -> Result<Self::OpeningProof, ZipError>
    where
        F::Inner: Transcribable,
        F::Modulus: Transcribable,
    {
        let start = transcript.stream.position() as usize;
        if let Some(hint) = prover_data {
            let _ = ZipPlus::<Zt, Lc>::prove_f::<_, CHECK_FOR_OVERFLOW>(
                transcript, ck, polys, point, hint, field_cfg,
            )?;
        }
        let end = transcript.stream.position() as usize;
        Ok(transcript.stream.get_ref()[start..end].to_vec())
    }

    fn verify_open<const CHECK_FOR_OVERFLOW: bool>(
        transcript: &mut PcsVerifierTranscript,
        vk: &Self::VerifierKey,
        commitment: &Self::Commitment,
        point: &[F],
        lifted_evals: &[DynamicPolynomialF<F>],
        opening_proof: &Self::OpeningProof,
        field_cfg: &F::Config,
    ) -> Result<(), ZipError>
    where
        F::Inner: Transcribable,
        F::Modulus: Transcribable,
    {
        if !opening_proof.is_empty() {
            let original_stream =
                std::mem::replace(&mut transcript.stream, Cursor::new(opening_proof.clone()));
            let result = <Self as PCS<F, Zt::Eval, D>>::verify_open::<CHECK_FOR_OVERFLOW>(
                transcript,
                vk,
                commitment,
                point,
                lifted_evals,
                &Vec::new(),
                field_cfg,
            );
            let consumed = transcript.stream.position() == opening_proof.len() as u64;
            transcript.stream = original_stream;
            result?;
            if !consumed {
                return Err(ZipError::InvalidPcsOpen(
                    "PCS opening proof has trailing bytes".to_string(),
                ));
            }
            return Ok(());
        }

        if commitment.batch_size == 0 {
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
