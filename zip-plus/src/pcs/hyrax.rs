#![allow(clippy::arithmetic_side_effects)]

use std::{
    collections::HashSet,
    fmt::Debug,
    io::{Cursor, Read, Write},
    marker::PhantomData,
};

use ark_ec::{AffineRepr, CurveGroup, VariableBaseMSM};
use ark_ff::{AdditiveGroup, BigInteger, PrimeField as ArkPrimeField, UniformRand, Zero};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, Compress};
use crypto_bigint::{BoxedUint, modular::BoxedMontyForm};
use crypto_primitives::{
    FromWithConfig, IntRing, PrimeField, crypto_bigint_boxed_monty::BoxedMontyField,
    crypto_bigint_int::Int, crypto_bigint_monty::MontyField, crypto_bigint_uint::Uint,
};
use num_integer::Integer;
use zinc_poly::{
    mle::DenseMultilinearExtension,
    univariate::{
        binary::BinaryPoly, dense::DensePolynomial, dynamic::over_field::DynamicPolynomialF,
    },
};
use zinc_transcript::traits::{ConstTranscribable, GenTranscribable, Transcribable, Transcript};
use zinc_utils::{cfg_into_iter, cfg_iter, delayed_reduction::DelayedFieldProductSum};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::{
    ZipError,
    pcs::{
        generic::{FoldablePCS, PCS},
        msm_commitment::{
            BoolSubsetMsm, MsmCommitmentEngine, MsmCommitmentKey, MsmError, RowMsmStrategy,
            ScalarPippengerMsm, SignedIntPippengerMsm,
        },
    },
    pcs_transcript::{PcsProverTranscript, PcsVerifierTranscript},
};

#[derive(Clone, Debug)]
pub struct HyraxPCS<C: AffineRepr, Lanes>(PhantomData<(C, Lanes)>);

impl<C, Lanes> HyraxPCS<C, Lanes>
where
    C: AffineRepr,
{
    /// Open a folded Hyrax commitment whose lane values are already scalar
    /// field elements.
    ///
    /// This is needed for instance-axis folds of binary commitments: after
    /// folding by transcript-derived weights, each bit lane is a scalar field
    /// linear combination of bits, not a `bool`.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn prove_open_scalar_lanes<F, const CHECK_FOR_OVERFLOW: bool>(
        transcript: &mut PcsProverTranscript,
        ck: &HyraxCommitmentKey<C>,
        scalar_lanes: &[Vec<Vec<C::ScalarField>>],
        point: &[F],
        prover_data: &HyraxProverData<C>,
        field_cfg: &F::Config,
    ) -> Result<(), ZipError>
    where
        F: HyraxFieldBridge<C>,
        F::Inner: Transcribable,
        F::Modulus: Transcribable,
        Lanes: Clone + Debug + Send + Sync,
    {
        let _ = CHECK_FOR_OVERFLOW;
        if scalar_lanes.is_empty() {
            return Ok(());
        }
        validate_scalar_lanes::<C>(ck, scalar_lanes, point.len(), prover_data)?;

        let n = scalar_lanes[0][0].len();
        let point_scalar = point
            .iter()
            .map(F::field_to_scalar)
            .collect::<Result<Vec<_>, _>>()?;
        let row_vars = prover_data.num_rows.ilog2() as usize;
        let q0_f = eq_tensor_f::<F>(&point[..row_vars], field_cfg);
        let q1_scalar = eq_tensor_scalar::<C>(&point_scalar[row_vars..]);
        let alphas = sample_scalars::<C>(
            &mut transcript.fs_transcript,
            scalar_lanes.len() * prover_data.num_lanes,
        );

        let mut b_scalar = vec![C::ScalarField::zero(); prover_data.num_rows];
        for (poly_idx, lanes) in scalar_lanes.iter().enumerate() {
            for (lane, values) in lanes.iter().enumerate() {
                let alpha = alphas[alpha_index_dynamic(prover_data.num_lanes, poly_idx, lane)];
                for (row_idx, row) in values.chunks(ck.num_cols).enumerate() {
                    let mut row_eval = C::ScalarField::zero();
                    for (col_idx, value) in row.iter().enumerate() {
                        if let Some(weight) = q1_scalar.get(col_idx) {
                            row_eval += *value * weight;
                        }
                    }
                    b_scalar[row_idx] += alpha * row_eval;
                }
            }
        }

        let b_f = b_scalar
            .iter()
            .map(|value| F::scalar_to_field(value, field_cfg))
            .collect::<Result<Vec<_>, _>>()?;
        transcript.write_field_elements(&b_f)?;

        let row_coeffs = if prover_data.num_rows == 1 {
            vec![C::ScalarField::from(1u64)]
        } else {
            sample_scalars::<C>(&mut transcript.fs_transcript, prover_data.num_rows)
        };

        let mut combined_row = vec![C::ScalarField::zero(); ck.num_cols];
        let mut rho_star = C::ScalarField::zero();
        for (poly_idx, lanes) in scalar_lanes.iter().enumerate() {
            for (lane, values) in lanes.iter().enumerate() {
                let alpha = alphas[alpha_index_dynamic(prover_data.num_lanes, poly_idx, lane)];
                for (row_idx, row) in values.chunks(ck.num_cols).enumerate() {
                    let coeff = alpha * row_coeffs[row_idx];
                    if ck.blinding_mode.is_blinded() {
                        let blind_idx = commitment_index_dynamic(
                            prover_data.num_lanes,
                            poly_idx,
                            lane,
                            row_idx,
                            prover_data.num_rows,
                        );
                        rho_star += coeff * prover_data.blinds[blind_idx];
                    }
                    for (col_idx, value) in row.iter().enumerate() {
                        combined_row[col_idx] += coeff * value;
                    }
                }
            }
        }

        write_scalars::<C>(transcript, &combined_row)?;
        if ck.blinding_mode.is_blinded() {
            write_scalar::<C>(transcript, &rho_star)?;
        }

        if q0_f.len() != b_f.len() || n != (1usize << point.len()) {
            return Err(ZipError::InvalidPcsOpen(
                "Hyrax folded scalar-lane opening shape mismatch".to_string(),
            ));
        }

        Ok(())
    }

    /// Open a folded single-row Hyrax commitment from protocol-field lanes.
    ///
    /// ProjectionFold folds binary witnesses by protocol-field challenges, so
    /// folded bit lanes are already field elements rather than booleans.  The
    /// generic scalar-lane path first converts every lane entry into the curve
    /// scalar field and then scans the matrix twice.  For the SHA benchmark the
    /// Hyrax width is the whole row domain (`num_rows == 1`), so we can compute
    /// the transcript's combined row directly in the protocol field, derive the
    /// single `b` value from it, and convert only the final row entries that are
    /// written to the proof stream.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn prove_open_field_lanes_single_row<F, const CHECK_FOR_OVERFLOW: bool>(
        transcript: &mut PcsProverTranscript,
        ck: &HyraxCommitmentKey<C>,
        field_lanes: &[Vec<&[F]>],
        point: &[F],
        prover_data: &HyraxProverData<C>,
        field_cfg: &F::Config,
    ) -> Result<(), ZipError>
    where
        F: HyraxFieldBridge<C> + DelayedFieldProductSum,
        F::Inner: Transcribable,
        F::Modulus: Transcribable,
        Lanes: Clone + Debug + Send + Sync,
    {
        let _ = CHECK_FOR_OVERFLOW;
        if field_lanes.is_empty() {
            return Ok(());
        }
        validate_field_lanes::<C, F>(ck, field_lanes, point.len(), prover_data)?;
        if prover_data.num_rows != 1 {
            return Err(ZipError::InvalidPcsParam(
                "Hyrax field-lane fast opening requires a single row".to_string(),
            ));
        }

        let q1 = eq_tensor_f::<F>(point, field_cfg);
        let alphas = sample_scalars::<C>(
            &mut transcript.fs_transcript,
            field_lanes.len() * prover_data.num_lanes,
        );
        let alpha_fields = alphas
            .iter()
            .map(|alpha| F::scalar_to_field(alpha, field_cfg))
            .collect::<Result<Vec<_>, _>>()?;

        let mut combined_row = vec![F::zero_with_cfg(field_cfg); ck.num_cols];
        let mut rho_star = C::ScalarField::zero();
        for (poly_idx, lanes) in field_lanes.iter().enumerate() {
            for (lane, values) in lanes.iter().enumerate() {
                let alpha_idx = alpha_index_dynamic(prover_data.num_lanes, poly_idx, lane);
                let alpha = &alpha_fields[alpha_idx];
                for (acc, value) in combined_row.iter_mut().zip(values.iter()) {
                    *acc += value.clone() * alpha.clone();
                }
                if ck.blinding_mode.is_blinded() {
                    let blind_idx = commitment_index_dynamic(
                        prover_data.num_lanes,
                        poly_idx,
                        lane,
                        0,
                        prover_data.num_rows,
                    );
                    rho_star += alphas[alpha_idx] * prover_data.blinds[blind_idx];
                }
            }
        }

        let b = F::delayed_sum_of_products(&combined_row, &q1, F::zero_with_cfg(field_cfg));
        transcript.write_field_elements(&[b])?;

        let combined_scalars = combined_row
            .iter()
            .map(F::field_to_scalar)
            .collect::<Result<Vec<_>, _>>()?;
        write_scalars::<C>(transcript, &combined_scalars)?;
        if ck.blinding_mode.is_blinded() {
            write_scalar::<C>(transcript, &rho_star)?;
        }

        Ok(())
    }

    /// Open two folded Hyrax commitments that share the same row bases as one
    /// mixed single-row proof.
    #[allow(clippy::arithmetic_side_effects)]
    #[allow(clippy::too_many_arguments)]
    pub fn prove_open_two_field_lane_groups_single_row<F, const CHECK_FOR_OVERFLOW: bool>(
        transcript: &mut PcsProverTranscript,
        ck_a: &HyraxCommitmentKey<C>,
        field_lanes_a: &[Vec<&[F]>],
        prover_data_a: &HyraxProverData<C>,
        ck_b: &HyraxCommitmentKey<C>,
        field_lanes_b: &[Vec<&[F]>],
        prover_data_b: &HyraxProverData<C>,
        point: &[F],
        field_cfg: &F::Config,
    ) -> Result<(), ZipError>
    where
        F: HyraxFieldBridge<C> + DelayedFieldProductSum,
        F::Inner: Transcribable,
        F::Modulus: Transcribable,
    {
        let _ = CHECK_FOR_OVERFLOW;
        if field_lanes_a.is_empty() || field_lanes_b.is_empty() {
            return Err(ZipError::InvalidPcsParam(
                "Hyrax mixed field-lane opening expects two non-empty groups".to_string(),
            ));
        }
        validate_field_lanes::<C, F>(ck_a, field_lanes_a, point.len(), prover_data_a)?;
        validate_field_lanes::<C, F>(ck_b, field_lanes_b, point.len(), prover_data_b)?;
        validate_shared_commitment_keys(ck_a, ck_b)?;
        if prover_data_a.num_rows != 1 || prover_data_b.num_rows != 1 {
            return Err(ZipError::InvalidPcsParam(
                "Hyrax mixed field-lane opening requires a single row".to_string(),
            ));
        }

        let q1 = eq_tensor_f::<F>(point, field_cfg);
        let alpha_count_a = field_lanes_a.len() * prover_data_a.num_lanes;
        let alpha_count_b = field_lanes_b.len() * prover_data_b.num_lanes;
        let alphas =
            sample_scalars::<C>(&mut transcript.fs_transcript, alpha_count_a + alpha_count_b);
        let alpha_fields = alphas
            .iter()
            .map(|alpha| F::scalar_to_field(alpha, field_cfg))
            .collect::<Result<Vec<_>, _>>()?;

        let mut combined_row = vec![F::zero_with_cfg(field_cfg); ck_a.num_cols];
        let mut rho_star = C::ScalarField::zero();
        for (poly_idx, lanes) in field_lanes_a.iter().enumerate() {
            for (lane, values) in lanes.iter().enumerate() {
                let alpha_idx = alpha_index_dynamic(prover_data_a.num_lanes, poly_idx, lane);
                let alpha = &alpha_fields[alpha_idx];
                for (acc, value) in combined_row.iter_mut().zip(values.iter()) {
                    *acc += value.clone() * alpha.clone();
                }
                if ck_a.blinding_mode.is_blinded() {
                    let blind_idx = commitment_index_dynamic(
                        prover_data_a.num_lanes,
                        poly_idx,
                        lane,
                        0,
                        prover_data_a.num_rows,
                    );
                    rho_star += alphas[alpha_idx] * prover_data_a.blinds[blind_idx];
                }
            }
        }
        for (poly_idx, lanes) in field_lanes_b.iter().enumerate() {
            for (lane, values) in lanes.iter().enumerate() {
                let local_alpha_idx = alpha_index_dynamic(prover_data_b.num_lanes, poly_idx, lane);
                let alpha_idx = alpha_count_a + local_alpha_idx;
                let alpha = &alpha_fields[alpha_idx];
                for (acc, value) in combined_row.iter_mut().zip(values.iter()) {
                    *acc += value.clone() * alpha.clone();
                }
                if ck_b.blinding_mode.is_blinded() {
                    let blind_idx = commitment_index_dynamic(
                        prover_data_b.num_lanes,
                        poly_idx,
                        lane,
                        0,
                        prover_data_b.num_rows,
                    );
                    rho_star += alphas[alpha_idx] * prover_data_b.blinds[blind_idx];
                }
            }
        }

        let b = F::delayed_sum_of_products(&combined_row, &q1, F::zero_with_cfg(field_cfg));
        transcript.write_field_elements(&[b])?;

        let combined_scalars = combined_row
            .iter()
            .map(F::field_to_scalar)
            .collect::<Result<Vec<_>, _>>()?;
        write_scalars::<C>(transcript, &combined_scalars)?;
        if ck_a.blinding_mode.is_blinded() {
            write_scalar::<C>(transcript, &rho_star)?;
        }

        Ok(())
    }

    #[allow(clippy::arithmetic_side_effects)]
    #[allow(clippy::too_many_arguments)]
    pub fn verify_open_two_field_lane_groups_single_row<
        F,
        EvalA,
        LanesA,
        EvalB,
        LanesB,
        const CHECK_FOR_OVERFLOW: bool,
        const D: usize,
    >(
        transcript: &mut PcsVerifierTranscript,
        vk_a: &HyraxVerifierKey<C>,
        commitment_a: &HyraxCommitment<C>,
        lifted_evals_a: &[DynamicPolynomialF<F>],
        vk_b: &HyraxVerifierKey<C>,
        commitment_b: &HyraxCommitment<C>,
        lifted_evals_b: &[DynamicPolynomialF<F>],
        point: &[F],
        opening_proof: &[u8],
        field_cfg: &F::Config,
    ) -> Result<(), ZipError>
    where
        F: HyraxFieldBridge<C>,
        F::Inner: Transcribable,
        F::Modulus: Transcribable,
        EvalA: Clone + Debug + Send + Sync,
        EvalB: Clone + Debug + Send + Sync,
        LanesA: HyraxLanes<C, EvalA, D>,
        LanesB: HyraxLanes<C, EvalB, D>,
    {
        let _ = CHECK_FOR_OVERFLOW;
        let original_stream =
            std::mem::replace(&mut transcript.stream, Cursor::new(opening_proof.to_vec()));
        let result = (|| {
            if commitment_a.blinding_mode != vk_a.blinding_mode
                || commitment_b.blinding_mode != vk_b.blinding_mode
            {
                return Err(ZipError::InvalidPcsParam(
                    "Hyrax commitment blinding mode mismatch".to_string(),
                ));
            }
            validate_commitment_shape::<C, LanesA, EvalA, D>(commitment_a)?;
            validate_commitment_shape::<C, LanesB, EvalB, D>(commitment_b)?;
            validate_shared_verifier_keys(vk_a, vk_b)?;
            if lifted_evals_a.len() != commitment_a.batch_size {
                return Err(ZipError::InvalidPcsParam(format!(
                    "Hyrax verifier expected {} left lifted evals, got {}",
                    commitment_a.batch_size,
                    lifted_evals_a.len()
                )));
            }
            if lifted_evals_b.len() != commitment_b.batch_size {
                return Err(ZipError::InvalidPcsParam(format!(
                    "Hyrax verifier expected {} right lifted evals, got {}",
                    commitment_b.batch_size,
                    lifted_evals_b.len()
                )));
            }
            if commitment_a.batch_size == 0 || commitment_b.batch_size == 0 {
                return Err(ZipError::InvalidPcsParam(
                    "Hyrax mixed opening expects two non-empty commitment groups".to_string(),
                ));
            }

            let n = 1usize << point.len();
            let expected_rows = num_rows(n, vk_a.num_cols)?;
            if expected_rows != 1 || commitment_a.num_rows != 1 || commitment_b.num_rows != 1 {
                return Err(ZipError::InvalidPcsParam(
                    "Hyrax mixed opening verifier requires a single row".to_string(),
                ));
            }

            let point_scalar = point
                .iter()
                .map(F::field_to_scalar)
                .collect::<Result<Vec<_>, _>>()?;
            let q1_scalar = eq_tensor_scalar::<C>(&point_scalar);
            let alpha_count_a = commitment_a.batch_size * commitment_a.num_lanes;
            let alpha_count_b = commitment_b.batch_size * commitment_b.num_lanes;
            let alphas =
                sample_scalars::<C>(&mut transcript.fs_transcript, alpha_count_a + alpha_count_b);

            let b_f = transcript.read_field_elements::<F>(1)?;
            let mut expected_eval = F::zero_with_cfg(field_cfg);
            for (poly_idx, lifted_eval) in lifted_evals_a.iter().enumerate() {
                for lane in 0..commitment_a.num_lanes {
                    let alpha_idx = alpha_index_dynamic(commitment_a.num_lanes, poly_idx, lane);
                    let alpha = F::scalar_to_field(&alphas[alpha_idx], field_cfg)?;
                    let mut term = LanesA::lifted_eval::<F>(lifted_eval, lane, field_cfg)?;
                    term *= &alpha;
                    expected_eval += &term;
                }
            }
            for (poly_idx, lifted_eval) in lifted_evals_b.iter().enumerate() {
                for lane in 0..commitment_b.num_lanes {
                    let local_alpha_idx =
                        alpha_index_dynamic(commitment_b.num_lanes, poly_idx, lane);
                    let alpha_idx = alpha_count_a + local_alpha_idx;
                    let alpha = F::scalar_to_field(&alphas[alpha_idx], field_cfg)?;
                    let mut term = LanesB::lifted_eval::<F>(lifted_eval, lane, field_cfg)?;
                    term *= &alpha;
                    expected_eval += &term;
                }
            }
            if b_f[0] != expected_eval {
                return Err(ZipError::InvalidPcsOpen(
                    "Hyrax mixed evaluation consistency failure".to_string(),
                ));
            }

            let b_scalar = F::field_to_scalar(&b_f[0])?;
            let combined_row = read_scalars::<C>(transcript, vk_a.num_cols)?;
            let rho_star = if vk_a.blinding_mode.is_blinded() {
                Some(read_scalar::<C>(transcript)?)
            } else {
                None
            };

            let mut lhs = C::ScalarField::zero();
            for (value, weight) in combined_row.iter().zip(q1_scalar.iter()) {
                lhs += *value * weight;
            }
            if lhs != b_scalar {
                return Err(ZipError::InvalidPcsOpen(
                    "Hyrax mixed row coherence failure".to_string(),
                ));
            }

            let mut comm_bases =
                Vec::with_capacity(commitment_a.comm_affine.len() + commitment_b.comm_affine.len());
            comm_bases.extend_from_slice(&commitment_a.comm_affine);
            comm_bases.extend_from_slice(&commitment_b.comm_affine);
            let comm_lc = msm_unchecked::<C>(&comm_bases, &alphas)?;

            let mut expected =
                msm_unchecked::<C>(&vk_a.bases[..combined_row.len()], &combined_row)?;
            if let Some(rho_star) = rho_star {
                expected += vk_a.h * rho_star;
            }

            if comm_lc != expected {
                return Err(ZipError::InvalidPcsOpen(
                    "Hyrax mixed commitment opening failure".to_string(),
                ));
            }

            Ok(())
        })();
        let consumed = transcript.stream.position() == opening_proof.len() as u64;
        transcript.stream = original_stream;
        result?;
        if !consumed {
            return Err(ZipError::InvalidPcsOpen(
                "PCS mixed opening proof has trailing bytes".to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HyraxBlindingMode {
    Blinded,
    Unblinded,
}

impl Default for HyraxBlindingMode {
    fn default() -> Self {
        Self::Unblinded
    }
}

impl HyraxBlindingMode {
    fn as_u8(self) -> u8 {
        match self {
            Self::Blinded => 1,
            Self::Unblinded => 0,
        }
    }

    fn is_blinded(self) -> bool {
        matches!(self, Self::Blinded)
    }
}

#[derive(Clone, Debug)]
pub struct HyraxCommitmentKey<C: AffineRepr> {
    pub(crate) num_cols: usize,
    pub(crate) blinding_mode: HyraxBlindingMode,
    pub(crate) msm_ck: MsmCommitmentKey<C>,
}

#[derive(Clone, Debug)]
pub struct HyraxVerifierKey<C: AffineRepr> {
    pub(crate) num_cols: usize,
    pub(crate) bases: Vec<C>,
    pub(crate) h: C::Group,
    pub(crate) blinding_mode: HyraxBlindingMode,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HyraxCommitment<C: AffineRepr> {
    pub(crate) batch_size: usize,
    pub(crate) num_lanes: usize,
    pub(crate) num_rows: usize,
    pub(crate) blinding_mode: HyraxBlindingMode,
    pub(crate) comm: Vec<C::Group>,
    pub(crate) comm_affine: Vec<C>,
    pub(crate) comm_bytes: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HyraxProverData<C: AffineRepr> {
    pub(crate) batch_size: usize,
    pub(crate) num_lanes: usize,
    pub(crate) num_rows: usize,
    pub(crate) blinding_mode: HyraxBlindingMode,
    pub(crate) blinds: Vec<C::ScalarField>,
}

pub trait HyraxFieldBridge<C: AffineRepr>: PrimeField {
    fn field_to_scalar(value: &Self) -> Result<C::ScalarField, ZipError>;
    fn scalar_to_field(value: &C::ScalarField, cfg: &Self::Config) -> Result<Self, ZipError>;
}

impl<C, const LIMBS: usize> HyraxFieldBridge<C> for MontyField<LIMBS>
where
    C: AffineRepr,
{
    fn field_to_scalar(value: &Self) -> Result<C::ScalarField, ZipError> {
        validate_curve_scalar_modulus::<C, LIMBS>(&value.modulus())?;

        let canonical = value.retrieve();
        let mut bytes = vec![0u8; <Uint<LIMBS> as ConstTranscribable>::NUM_BYTES];
        canonical.write_transcription_bytes_exact(&mut bytes);
        Ok(C::ScalarField::from_le_bytes_mod_order(&bytes))
    }

    fn scalar_to_field(value: &C::ScalarField, cfg: &Self::Config) -> Result<Self, ZipError> {
        let actual_modulus = Uint::<LIMBS>::new(cfg.modulus().get());
        validate_curve_scalar_modulus::<C, LIMBS>(&actual_modulus)?;

        let scalar_bigint: <C::ScalarField as ArkPrimeField>::BigInt = value.clone().into();
        let scalar_uint = uint_from_le_bytes::<LIMBS>(&scalar_bigint.to_bytes_le());
        Ok(MontyField::<LIMBS>::from_with_cfg(&scalar_uint, cfg))
    }
}

impl<C> HyraxFieldBridge<C> for BoxedMontyField
where
    C: AffineRepr,
{
    fn field_to_scalar(value: &Self) -> Result<C::ScalarField, ZipError> {
        validate_curve_scalar_modulus_boxed::<C>(&value.modulus())?;

        let canonical = BoxedMontyForm::from(value.clone()).retrieve();
        Ok(C::ScalarField::from_le_bytes_mod_order(
            &canonical.to_le_bytes(),
        ))
    }

    fn scalar_to_field(value: &C::ScalarField, cfg: &Self::Config) -> Result<Self, ZipError> {
        let actual_modulus = cfg.modulus().clone().get();
        validate_curve_scalar_modulus_boxed::<C>(&actual_modulus)?;

        let scalar_bigint: <C::ScalarField as ArkPrimeField>::BigInt = value.clone().into();
        let scalar_uint = BoxedUint::from_le_slice(
            &scalar_bigint.to_bytes_le(),
            actual_modulus.bits_precision(),
        )
        .expect("curve scalar must fit protocol field precision");
        Ok(BoxedMontyField::from_with_cfg(&scalar_uint, cfg))
    }
}

fn validate_fold_inputs<T>(values: &[T], theta_len: usize, label: &str) -> Result<(), ZipError> {
    if values.is_empty() {
        return Err(ZipError::InvalidPcsParam(format!(
            "Hyrax cannot fold empty {label}"
        )));
    }
    if values.len() != theta_len {
        return Err(ZipError::InvalidPcsParam(format!(
            "Hyrax fold {label} count mismatch: got {}, expected {theta_len}",
            values.len()
        )));
    }
    Ok(())
}

pub trait HyraxLanes<C, Eval, const D: usize>: Clone + Debug + Send + Sync
where
    C: AffineRepr,
    Eval: Clone + Debug + Send + Sync,
{
    type LaneValue: Copy + Send + Sync;
    type Strategy: RowMsmStrategy<C, Self::LaneValue>;

    const NUM_LANES: usize;

    fn lane_value(eval: &Eval, lane: usize) -> Result<Self::LaneValue, ZipError>;

    fn lane_to_scalar(value: Self::LaneValue) -> C::ScalarField;

    fn commit_poly(
        _ck: &HyraxCommitmentKey<C>,
        _poly: &DenseMultilinearExtension<Eval>,
        _num_rows: usize,
    ) -> Option<Result<(Vec<C::Group>, Vec<C::ScalarField>), ZipError>> {
        None
    }

    fn accumulate_b(
        row: &[Eval],
        lane: usize,
        q1_scalar: &[C::ScalarField],
    ) -> Result<C::ScalarField, ZipError> {
        let mut row_eval = C::ScalarField::zero();
        for (col_idx, eval) in row.iter().enumerate() {
            let value = Self::lane_to_scalar(Self::lane_value(eval, lane)?);
            if let Some(weight) = q1_scalar.get(col_idx) {
                row_eval += value * weight;
            }
        }
        Ok(row_eval)
    }

    fn accumulate_combined_row(
        row: &[Eval],
        lane: usize,
        coeff: C::ScalarField,
        combined_row: &mut [C::ScalarField],
    ) -> Result<(), ZipError> {
        for (col_idx, eval) in row.iter().enumerate() {
            let value = Self::lane_to_scalar(Self::lane_value(eval, lane)?);
            combined_row[col_idx] += coeff * value;
        }
        Ok(())
    }

    fn accumulate_single_row_opening(
        row: &[Eval],
        lane: usize,
        alpha: C::ScalarField,
        q1_scalar: &[C::ScalarField],
        b_scalar: &mut C::ScalarField,
        combined_row: &mut [C::ScalarField],
    ) -> Result<(), ZipError> {
        for (col_idx, eval) in row.iter().enumerate() {
            let value = Self::lane_to_scalar(Self::lane_value(eval, lane)?);
            let scaled = alpha * value;
            if let Some(weight) = q1_scalar.get(col_idx) {
                *b_scalar += scaled * weight;
            }
            combined_row[col_idx] += scaled;
        }
        Ok(())
    }

    fn lifted_eval<F>(
        lifted_eval: &DynamicPolynomialF<F>,
        lane: usize,
        field_cfg: &F::Config,
    ) -> Result<F, ZipError>
    where
        F: PrimeField;
}

#[derive(Clone, Debug)]
pub struct BinaryLanes;

#[derive(Clone, Debug)]
pub struct IntScalarLane;

#[derive(Clone, Debug)]
pub struct DensePolyScalarLanes;

impl<C: AffineRepr, const D: usize> HyraxLanes<C, BinaryPoly<D>, D> for BinaryLanes {
    type LaneValue = bool;
    type Strategy = BoolSubsetMsm<6>;

    const NUM_LANES: usize = D;

    fn lane_value(eval: &BinaryPoly<D>, lane: usize) -> Result<Self::LaneValue, ZipError> {
        if lane >= D {
            return Err(ZipError::InvalidPcsParam(format!(
                "binary lane {lane} out of range"
            )));
        }
        Ok(eval.coeff(lane))
    }

    fn lane_to_scalar(value: Self::LaneValue) -> C::ScalarField {
        if value {
            C::ScalarField::from(1u64)
        } else {
            C::ScalarField::zero()
        }
    }

    fn commit_poly(
        ck: &HyraxCommitmentKey<C>,
        poly: &DenseMultilinearExtension<BinaryPoly<D>>,
        num_rows: usize,
    ) -> Option<Result<(Vec<C::Group>, Vec<C::ScalarField>), ZipError>> {
        let expected_comm = <Self as HyraxLanes<C, BinaryPoly<D>, D>>::NUM_LANES * num_rows;
        <Self as HyraxLanes<C, BinaryPoly<D>, D>>::Strategy::precompute_ck(&ck.msm_ck);
        let blinds = if ck.blinding_mode.is_blinded() {
            random_scalars::<C>(expected_comm)
        } else {
            Vec::new()
        };

        Some((|| {
            let use_inner_parallelism = use_inner_bool_parallelism(expected_comm);
            let per_row = cfg_into_iter!(0..num_rows)
                .map(|row_idx| {
                    let lower = row_idx * ck.num_cols;
                    let upper = (lower + ck.num_cols).min(poly.evaluations.len());
                    let row_len = upper - lower;
                    let mut row_comms =
                        BoolSubsetMsm::<6>::msm_bool_rows_from_window_masks::<C, D, _>(
                            &ck.msm_ck,
                            row_len,
                            use_inner_parallelism,
                            |offset, len| {
                                let mut masks = [0usize; D];
                                for bit_idx in 0..len {
                                    let eval = &poly.evaluations[lower + offset + bit_idx];
                                    for (lane, mask) in masks.iter_mut().enumerate() {
                                        if eval.coeff(lane) {
                                            *mask |= 1usize << bit_idx;
                                        }
                                    }
                                }
                                masks
                            },
                        )
                        .map_err(msm_err)?;

                    if ck.blinding_mode.is_blinded() {
                        for (lane, row_comm) in row_comms.iter_mut().enumerate() {
                            let blind_idx = lane * num_rows + row_idx;
                            *row_comm += ck.msm_ck.h * blinds[blind_idx];
                        }
                    }
                    Ok::<[C::Group; D], ZipError>(row_comms)
                })
                .collect::<Result<Vec<_>, _>>()?;

            let mut comm = Vec::with_capacity(expected_comm);
            for lane in 0..D {
                for row_comms in &per_row {
                    comm.push(row_comms[lane]);
                }
            }
            Ok((comm, blinds))
        })())
    }

    fn accumulate_b(
        row: &[BinaryPoly<D>],
        lane: usize,
        q1_scalar: &[C::ScalarField],
    ) -> Result<C::ScalarField, ZipError> {
        let mut row_eval = C::ScalarField::zero();
        for (col_idx, eval) in row.iter().enumerate() {
            if <Self as HyraxLanes<C, BinaryPoly<D>, D>>::lane_value(eval, lane)? {
                if let Some(weight) = q1_scalar.get(col_idx) {
                    row_eval += weight;
                }
            }
        }
        Ok(row_eval)
    }

    fn accumulate_combined_row(
        row: &[BinaryPoly<D>],
        lane: usize,
        coeff: C::ScalarField,
        combined_row: &mut [C::ScalarField],
    ) -> Result<(), ZipError> {
        for (col_idx, eval) in row.iter().enumerate() {
            if <Self as HyraxLanes<C, BinaryPoly<D>, D>>::lane_value(eval, lane)? {
                combined_row[col_idx] += coeff;
            }
        }
        Ok(())
    }

    fn accumulate_single_row_opening(
        row: &[BinaryPoly<D>],
        lane: usize,
        alpha: C::ScalarField,
        q1_scalar: &[C::ScalarField],
        b_scalar: &mut C::ScalarField,
        combined_row: &mut [C::ScalarField],
    ) -> Result<(), ZipError> {
        let mut row_eval = C::ScalarField::zero();
        for (col_idx, eval) in row.iter().enumerate() {
            if <Self as HyraxLanes<C, BinaryPoly<D>, D>>::lane_value(eval, lane)? {
                if let Some(weight) = q1_scalar.get(col_idx) {
                    row_eval += weight;
                }
                combined_row[col_idx] += alpha;
            }
        }
        *b_scalar += alpha * row_eval;
        Ok(())
    }

    fn lifted_eval<F>(
        lifted_eval: &DynamicPolynomialF<F>,
        lane: usize,
        field_cfg: &F::Config,
    ) -> Result<F, ZipError>
    where
        F: PrimeField,
    {
        Ok(lifted_eval
            .coeffs
            .get(lane)
            .cloned()
            .unwrap_or_else(|| F::zero_with_cfg(field_cfg)))
    }
}

impl<C: AffineRepr, const LIMBS: usize, const D: usize> HyraxLanes<C, Int<LIMBS>, D>
    for IntScalarLane
{
    type LaneValue = Int<LIMBS>;
    type Strategy = SignedIntPippengerMsm;

    const NUM_LANES: usize = 1;

    fn lane_value(eval: &Int<LIMBS>, lane: usize) -> Result<Self::LaneValue, ZipError> {
        if lane != 0 {
            return Err(ZipError::InvalidPcsParam(format!(
                "int lane {lane} out of range"
            )));
        }
        Ok(*eval)
    }

    fn lane_to_scalar(value: Self::LaneValue) -> C::ScalarField {
        int_to_scalar::<C, LIMBS>(&value).expect("int lane value must convert to scalar")
    }

    fn lifted_eval<F>(
        lifted_eval: &DynamicPolynomialF<F>,
        lane: usize,
        field_cfg: &F::Config,
    ) -> Result<F, ZipError>
    where
        F: PrimeField,
    {
        if lane != 0 {
            return Err(ZipError::InvalidPcsParam(format!(
                "lifted int lane {lane} out of range"
            )));
        }
        Ok(lifted_eval
            .coeffs
            .first()
            .cloned()
            .unwrap_or_else(|| F::zero_with_cfg(field_cfg)))
    }
}

impl<C: AffineRepr, const LIMBS: usize, const D: usize>
    HyraxLanes<C, DensePolynomial<Int<LIMBS>, D>, D> for DensePolyScalarLanes
{
    type LaneValue = C::ScalarField;
    type Strategy = ScalarPippengerMsm;

    const NUM_LANES: usize = D;

    fn lane_value(
        eval: &DensePolynomial<Int<LIMBS>, D>,
        lane: usize,
    ) -> Result<Self::LaneValue, ZipError> {
        eval.coeffs
            .get(lane)
            .ok_or_else(|| ZipError::InvalidPcsParam(format!("dense lane {lane} out of range")))
            .and_then(int_to_scalar::<C, LIMBS>)
    }

    fn lane_to_scalar(value: Self::LaneValue) -> C::ScalarField {
        value
    }

    fn lifted_eval<F>(
        lifted_eval: &DynamicPolynomialF<F>,
        lane: usize,
        field_cfg: &F::Config,
    ) -> Result<F, ZipError>
    where
        F: PrimeField,
    {
        Ok(lifted_eval
            .coeffs
            .get(lane)
            .cloned()
            .unwrap_or_else(|| F::zero_with_cfg(field_cfg)))
    }
}

impl<C: AffineRepr, Lanes> HyraxPCS<C, Lanes> {
    pub fn setup(
        width: usize,
        domain: impl AsRef<[u8]>,
        blinding_mode: HyraxBlindingMode,
    ) -> Result<(HyraxCommitmentKey<C>, HyraxVerifierKey<C>), ZipError> {
        let domain = domain.as_ref();
        let bases = (0..width)
            .map(|idx| hash_to_curve::<C>(domain, b"basis", idx))
            .collect::<Result<Vec<_>, _>>()?;
        let h = hash_to_curve::<C>(domain, b"blinding", 0)?.into_group();
        Self::setup_from_trusted_bases(width, bases, h, blinding_mode)
    }

    pub fn setup_from_bases(
        width: usize,
        bases: Vec<C>,
        h: C::Group,
    ) -> Result<(HyraxCommitmentKey<C>, HyraxVerifierKey<C>), ZipError> {
        Self::setup_from_bases_with_blinding(width, bases, h, HyraxBlindingMode::Blinded)
    }

    pub fn setup_from_bases_with_blinding(
        width: usize,
        bases: Vec<C>,
        h: C::Group,
        blinding_mode: HyraxBlindingMode,
    ) -> Result<(HyraxCommitmentKey<C>, HyraxVerifierKey<C>), ZipError> {
        Self::setup_from_trusted_bases(width, bases, h, blinding_mode)
    }

    pub fn setup_from_trusted_bases(
        width: usize,
        bases: Vec<C>,
        h: C::Group,
        blinding_mode: HyraxBlindingMode,
    ) -> Result<(HyraxCommitmentKey<C>, HyraxVerifierKey<C>), ZipError> {
        validate_trusted_bases(width, &bases, &h)?;
        let msm_ck = msm_key(width, bases.clone(), h)?;
        Ok((
            HyraxCommitmentKey {
                num_cols: width,
                blinding_mode,
                msm_ck,
            },
            HyraxVerifierKey {
                num_cols: width,
                bases,
                h,
                blinding_mode,
            },
        ))
    }
}

impl<F, C, Lanes, Eval, const D: usize> PCS<F, Eval, D> for HyraxPCS<C, Lanes>
where
    F: HyraxFieldBridge<C>,
    C: AffineRepr,
    Eval: Clone + Debug + Send + Sync,
    Lanes: HyraxLanes<C, Eval, D>,
{
    type CommitmentKey = HyraxCommitmentKey<C>;
    type VerifierKey = HyraxVerifierKey<C>;
    type Commitment = HyraxCommitment<C>;
    type ProverData = HyraxProverData<C>;
    type OpeningProof = Vec<u8>;

    fn precompute_ck(ck: &Self::CommitmentKey) {
        Lanes::Strategy::precompute_ck(&ck.msm_ck);
    }

    fn commit(
        ck: &Self::CommitmentKey,
        polys: &[DenseMultilinearExtension<Eval>],
    ) -> Result<(Self::ProverData, Self::Commitment), ZipError> {
        if polys.is_empty() {
            return Ok((
                HyraxProverData {
                    batch_size: 0,
                    num_lanes: Lanes::NUM_LANES,
                    num_rows: 0,
                    blinding_mode: ck.blinding_mode,
                    blinds: Vec::new(),
                },
                HyraxCommitment {
                    batch_size: 0,
                    num_lanes: Lanes::NUM_LANES,
                    num_rows: 0,
                    blinding_mode: ck.blinding_mode,
                    comm: Vec::new(),
                    comm_affine: Vec::new(),
                    comm_bytes: Vec::new(),
                },
            ));
        }

        validate_polys(polys)?;
        let n = polys[0].evaluations.len();
        let num_rows = num_rows(n, ck.num_cols)?;
        Lanes::Strategy::precompute_ck(&ck.msm_ck);

        let per_poly = cfg_iter!(polys)
            .map(|poly| commit_hyrax_poly::<C, Lanes, Eval, D>(ck, poly, num_rows))
            .collect::<Result<Vec<_>, _>>()?;

        let expected_comm = polys.len() * Lanes::NUM_LANES * num_rows;
        let expected_blinds = if ck.blinding_mode.is_blinded() {
            expected_comm
        } else {
            0
        };
        let mut all_comm = Vec::with_capacity(expected_comm);
        let mut all_blinds = Vec::with_capacity(expected_blinds);
        for (comm, blinds) in per_poly {
            all_comm.extend(comm);
            all_blinds.extend(blinds);
        }

        let all_affine = C::Group::normalize_batch(&all_comm);
        let all_bytes = affine_points_bytes::<C>(&all_affine)?;

        Ok((
            HyraxProverData {
                batch_size: polys.len(),
                num_lanes: Lanes::NUM_LANES,
                num_rows,
                blinding_mode: ck.blinding_mode,
                blinds: all_blinds,
            },
            HyraxCommitment {
                batch_size: polys.len(),
                num_lanes: Lanes::NUM_LANES,
                num_rows,
                blinding_mode: ck.blinding_mode,
                comm: all_comm,
                comm_affine: all_affine,
                comm_bytes: all_bytes,
            },
        ))
    }

    fn absorb_commitment<T: Transcript>(transcript: &mut T, commitment: &Self::Commitment) {
        transcript.absorb_slice(b"hyrax_commitment_begin");
        transcript.absorb_slice(&(commitment.batch_size as u64).to_le_bytes());
        transcript.absorb_slice(&(commitment.num_lanes as u64).to_le_bytes());
        transcript.absorb_slice(&(commitment.num_rows as u64).to_le_bytes());
        transcript.absorb_slice(&[commitment.blinding_mode.as_u8()]);
        transcript.absorb_slice(&commitment.comm_bytes);
        transcript.absorb_slice(b"hyrax_commitment_end");
    }

    fn commitment_num_bytes(commitment: &Self::Commitment) -> usize {
        let group_size = C::zero().serialized_size(Compress::Yes);
        3 * core::mem::size_of::<u64>() + 1 + commitment.comm.len() * group_size
    }

    fn write_commitment_bytes(commitment: &Self::Commitment, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&(commitment.batch_size as u64).to_le_bytes());
        buf.extend_from_slice(&(commitment.num_lanes as u64).to_le_bytes());
        buf.extend_from_slice(&(commitment.num_rows as u64).to_le_bytes());
        buf.push(commitment.blinding_mode.as_u8());
        buf.extend_from_slice(&commitment.comm_bytes);
    }

    fn batch_size(commitment: &Self::Commitment) -> usize {
        commitment.batch_size
    }

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
        F::Modulus: Transcribable,
    {
        let _ = CHECK_FOR_OVERFLOW;
        let start = transcript.stream.position() as usize;
        if polys.is_empty() {
            if prover_data.batch_size != 0
                || prover_data.num_lanes != Lanes::NUM_LANES
                || prover_data.num_rows != 0
                || prover_data.blinding_mode != ck.blinding_mode
                || !prover_data.blinds.is_empty()
            {
                return Err(ZipError::InvalidPcsParam(
                    "Hyrax prover data must be canonical for an empty batch".to_string(),
                ));
            }
            let end = transcript.stream.position() as usize;
            return Ok(transcript.stream.get_ref()[start..end].to_vec());
        }
        validate_polys(polys)?;
        validate_hyrax_shape::<C, Lanes, Eval, D>(
            ck.num_cols,
            ck.blinding_mode,
            polys,
            prover_data,
        )?;

        let n = polys[0].evaluations.len();
        if n != (1usize << point.len()) {
            return Err(ZipError::InvalidPcsParam(format!(
                "Hyrax open expected point for {n} evals, got {} variables",
                point.len()
            )));
        }

        let point_scalar = point
            .iter()
            .map(F::field_to_scalar)
            .collect::<Result<Vec<_>, _>>()?;
        let row_vars = prover_data.num_rows.ilog2() as usize;
        let q0_f = eq_tensor_f::<F>(&point[..row_vars], field_cfg);
        let q1_scalar = eq_tensor_scalar::<C>(&point_scalar[row_vars..]);
        let alphas = sample_scalars::<C>(
            &mut transcript.fs_transcript,
            polys.len() * Lanes::NUM_LANES,
        );

        let mut combined_row = vec![C::ScalarField::zero(); ck.num_cols];
        let mut rho_star = C::ScalarField::zero();

        let mut b_scalar = vec![C::ScalarField::zero(); prover_data.num_rows];
        if prover_data.num_rows == 1 {
            for (poly_idx, poly) in polys.iter().enumerate() {
                for lane in 0..Lanes::NUM_LANES {
                    let alpha = alphas[alpha_index_dynamic(Lanes::NUM_LANES, poly_idx, lane)];
                    if ck.blinding_mode.is_blinded() {
                        let blind_idx = commitment_index_dynamic(
                            Lanes::NUM_LANES,
                            poly_idx,
                            lane,
                            0,
                            prover_data.num_rows,
                        );
                        rho_star += alpha * prover_data.blinds[blind_idx];
                    }
                    Lanes::accumulate_single_row_opening(
                        &poly.evaluations,
                        lane,
                        alpha,
                        &q1_scalar,
                        &mut b_scalar[0],
                        &mut combined_row,
                    )?;
                }
            }
        } else {
            b_scalar = cfg_into_iter!(0..prover_data.num_rows)
                .map(|row_idx| {
                    let lower = row_idx * ck.num_cols;
                    let mut acc = C::ScalarField::zero();
                    for (poly_idx, poly) in polys.iter().enumerate() {
                        let upper = (lower + ck.num_cols).min(poly.evaluations.len());
                        let row = &poly.evaluations[lower..upper];
                        for lane in 0..Lanes::NUM_LANES {
                            let alpha =
                                alphas[alpha_index_dynamic(Lanes::NUM_LANES, poly_idx, lane)];
                            let row_eval = Lanes::accumulate_b(row, lane, &q1_scalar)?;
                            acc += alpha * row_eval;
                        }
                    }
                    Ok::<C::ScalarField, ZipError>(acc)
                })
                .collect::<Result<Vec<_>, _>>()?;

            let b_f = b_scalar
                .iter()
                .map(|value| F::scalar_to_field(value, field_cfg))
                .collect::<Result<Vec<_>, _>>()?;
            transcript.write_field_elements(&b_f)?;

            let row_coeffs =
                sample_scalars::<C>(&mut transcript.fs_transcript, prover_data.num_rows);

            combined_row = cfg_into_iter!(0..ck.num_cols)
                .map(|col_idx| {
                    let mut acc = C::ScalarField::zero();
                    for (poly_idx, poly) in polys.iter().enumerate() {
                        for lane in 0..Lanes::NUM_LANES {
                            let alpha =
                                alphas[alpha_index_dynamic(Lanes::NUM_LANES, poly_idx, lane)];
                            for (row_idx, row_coeff) in row_coeffs.iter().copied().enumerate() {
                                let eval_idx = row_idx * ck.num_cols + col_idx;
                                if let Some(eval) = poly.evaluations.get(eval_idx) {
                                    let value =
                                        Lanes::lane_to_scalar(Lanes::lane_value(eval, lane)?);
                                    acc += alpha * row_coeff * value;
                                }
                            }
                        }
                    }
                    Ok::<C::ScalarField, ZipError>(acc)
                })
                .collect::<Result<Vec<_>, _>>()?;

            if ck.blinding_mode.is_blinded() {
                let total_jobs = polys.len() * Lanes::NUM_LANES * prover_data.num_rows;
                let rho_terms = cfg_into_iter!(0..total_jobs)
                    .map(|job_idx| {
                        let poly_stride = Lanes::NUM_LANES * prover_data.num_rows;
                        let poly_idx = job_idx / poly_stride;
                        let lane_row_idx = job_idx % poly_stride;
                        let lane = lane_row_idx / prover_data.num_rows;
                        let row_idx = lane_row_idx % prover_data.num_rows;
                        let alpha = alphas[alpha_index_dynamic(Lanes::NUM_LANES, poly_idx, lane)];
                        alpha * row_coeffs[row_idx] * prover_data.blinds[job_idx]
                    })
                    .collect::<Vec<_>>();
                rho_star = rho_terms
                    .into_iter()
                    .fold(C::ScalarField::zero(), |acc, term| acc + term);
            }
        }

        if prover_data.num_rows == 1 {
            let b_f = b_scalar
                .iter()
                .map(|value| F::scalar_to_field(value, field_cfg))
                .collect::<Result<Vec<_>, _>>()?;
            transcript.write_field_elements(&b_f)?;
        }

        write_scalars::<C>(transcript, &combined_row)?;
        if ck.blinding_mode.is_blinded() {
            write_scalar::<C>(transcript, &rho_star)?;
        }

        if q0_f.len() != b_scalar.len() {
            return Err(ZipError::InvalidPcsOpen(
                "Hyrax b vector shape mismatch".to_string(),
            ));
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
        let _ = CHECK_FOR_OVERFLOW;
        if !opening_proof.is_empty() {
            let original_stream =
                std::mem::replace(&mut transcript.stream, Cursor::new(opening_proof.clone()));
            let result = <Self as PCS<F, Eval, D>>::verify_open::<CHECK_FOR_OVERFLOW>(
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

        if commitment.blinding_mode != vk.blinding_mode {
            return Err(ZipError::InvalidPcsParam(
                "Hyrax commitment blinding mode mismatch".to_string(),
            ));
        }
        validate_commitment_shape::<C, Lanes, Eval, D>(commitment)?;
        if lifted_evals.len() != commitment.batch_size {
            return Err(ZipError::InvalidPcsParam(format!(
                "Hyrax verifier expected {} lifted evals, got {}",
                commitment.batch_size,
                lifted_evals.len()
            )));
        }
        if commitment.batch_size == 0 {
            if commitment.num_rows != 0 {
                return Err(ZipError::InvalidPcsParam(
                    "Hyrax empty batch must use the canonical empty commitment".to_string(),
                ));
            }
            return Ok(());
        }

        let n = 1usize << point.len();
        let expected_rows = num_rows(n, vk.num_cols)?;
        if expected_rows != commitment.num_rows {
            return Err(ZipError::InvalidPcsParam(format!(
                "Hyrax verifier expected {expected_rows} rows from point, commitment has {}",
                commitment.num_rows
            )));
        }

        let row_vars = commitment.num_rows.ilog2() as usize;
        let q0_f = eq_tensor_f::<F>(&point[..row_vars], field_cfg);
        let point_scalar = point
            .iter()
            .map(F::field_to_scalar)
            .collect::<Result<Vec<_>, _>>()?;
        let q1_scalar = eq_tensor_scalar::<C>(&point_scalar[row_vars..]);
        let alphas = sample_scalars::<C>(
            &mut transcript.fs_transcript,
            commitment.batch_size * commitment.num_lanes,
        );

        let b_f = transcript.read_field_elements::<F>(commitment.num_rows)?;
        if b_f.len() != q0_f.len() {
            return Err(ZipError::InvalidPcsOpen(
                "Hyrax b vector shape mismatch".to_string(),
            ));
        }

        let mut expected_eval = F::zero_with_cfg(field_cfg);
        for (poly_idx, lifted_eval) in lifted_evals.iter().enumerate() {
            for lane in 0..commitment.num_lanes {
                let alpha = F::scalar_to_field(
                    &alphas[alpha_index_dynamic(commitment.num_lanes, poly_idx, lane)],
                    field_cfg,
                )?;
                let mut term = Lanes::lifted_eval::<F>(lifted_eval, lane, field_cfg)?;
                term *= &alpha;
                expected_eval += &term;
            }
        }

        let mut b_eval = F::zero_with_cfg(field_cfg);
        for (weight, b) in q0_f.iter().zip(b_f.iter()) {
            let mut term = weight.clone();
            term *= b;
            b_eval += &term;
        }
        if b_eval != expected_eval {
            return Err(ZipError::InvalidPcsOpen(
                "Hyrax evaluation consistency failure".to_string(),
            ));
        }

        let b_scalar = b_f
            .iter()
            .map(F::field_to_scalar)
            .collect::<Result<Vec<_>, _>>()?;
        let row_coeffs = if commitment.num_rows == 1 {
            vec![C::ScalarField::from(1u64)]
        } else {
            sample_scalars::<C>(&mut transcript.fs_transcript, commitment.num_rows)
        };

        let combined_row = read_scalars::<C>(transcript, vk.num_cols)?;
        let rho_star = if vk.blinding_mode.is_blinded() {
            Some(read_scalar::<C>(transcript)?)
        } else {
            None
        };

        let mut lhs = C::ScalarField::zero();
        for (value, weight) in combined_row.iter().zip(q1_scalar.iter()) {
            lhs += *value * weight;
        }
        let mut rhs = C::ScalarField::zero();
        for (coeff, b) in row_coeffs.iter().zip(b_scalar.iter()) {
            rhs += *coeff * b;
        }
        if lhs != rhs {
            return Err(ZipError::InvalidPcsOpen(
                "Hyrax row coherence failure".to_string(),
            ));
        }

        let comm_lc = if commitment.num_rows == 1 {
            msm_unchecked::<C>(&commitment.comm_affine, &alphas)?
        } else {
            let mut comm_lc_scalars = Vec::with_capacity(commitment.comm.len());
            for poly_idx in 0..commitment.batch_size {
                for lane in 0..commitment.num_lanes {
                    let alpha = alphas[alpha_index_dynamic(commitment.num_lanes, poly_idx, lane)];
                    comm_lc_scalars.extend(row_coeffs.iter().map(|row_coeff| alpha * row_coeff));
                }
            }
            msm_unchecked::<C>(&commitment.comm_affine, &comm_lc_scalars)?
        };

        let mut expected = msm_unchecked::<C>(&vk.bases[..combined_row.len()], &combined_row)?;
        if let Some(rho_star) = rho_star {
            expected += vk.h * rho_star;
        }

        if comm_lc != expected {
            return Err(ZipError::InvalidPcsOpen(
                "Hyrax commitment opening failure".to_string(),
            ));
        }

        Ok(())
    }
}

impl<F, C, Lanes, Eval, const D: usize> FoldablePCS<F, Eval, D> for HyraxPCS<C, Lanes>
where
    F: HyraxFieldBridge<C>,
    C: AffineRepr,
    Eval: Clone + Debug + Send + Sync,
    Lanes: HyraxLanes<C, Eval, D>,
{
    fn fold_commitments(
        commitments: &[Self::Commitment],
        theta: &[F],
        field_cfg: &F::Config,
    ) -> Result<Self::Commitment, ZipError> {
        let refs = commitments.iter().collect::<Vec<_>>();
        Self::fold_commitment_refs(&refs, theta, field_cfg)
    }

    fn fold_commitment_refs(
        commitments: &[&Self::Commitment],
        theta: &[F],
        field_cfg: &F::Config,
    ) -> Result<Self::Commitment, ZipError> {
        let _ = field_cfg;
        validate_fold_inputs(commitments, theta.len(), "commitments")?;
        let first = commitments[0];
        validate_commitment_shape::<C, Lanes, Eval, D>(first)?;

        let scalars = theta
            .iter()
            .map(F::field_to_scalar)
            .collect::<Result<Vec<_>, _>>()?;
        for &commitment in commitments {
            validate_commitment_shape::<C, Lanes, Eval, D>(commitment)?;
            if !same_commitment_shape(first, commitment) {
                return Err(ZipError::InvalidPcsParam(
                    "Hyrax commitment fold shape mismatch".to_string(),
                ));
            }
        }
        let folded = msm_shared_weight_commitments_unchecked::<C>(&scalars, commitments)?;
        let folded_affine = C::Group::normalize_batch(&folded);
        let folded_bytes = affine_points_bytes::<C>(&folded_affine)?;

        Ok(HyraxCommitment {
            batch_size: first.batch_size,
            num_lanes: first.num_lanes,
            num_rows: first.num_rows,
            blinding_mode: first.blinding_mode,
            comm: folded,
            comm_affine: folded_affine,
            comm_bytes: folded_bytes,
        })
    }

    fn fold_prover_data(
        prover_data: &[Self::ProverData],
        theta: &[F],
        field_cfg: &F::Config,
    ) -> Result<Self::ProverData, ZipError> {
        let _ = field_cfg;
        validate_fold_inputs(prover_data, theta.len(), "prover data")?;
        let first = &prover_data[0];

        let scalars = theta
            .iter()
            .map(F::field_to_scalar)
            .collect::<Result<Vec<_>, _>>()?;
        for data in prover_data {
            if !same_prover_data_shape(first, data) {
                return Err(ZipError::InvalidPcsParam(
                    "Hyrax prover-data fold shape mismatch".to_string(),
                ));
            }
        }
        let folded_blinds = cfg_into_iter!(0..first.blinds.len())
            .map(|idx| {
                let mut acc = C::ScalarField::zero();
                for (data, scalar) in prover_data.iter().zip(&scalars) {
                    acc += data.blinds[idx] * scalar;
                }
                acc
            })
            .collect();

        Ok(HyraxProverData {
            batch_size: first.batch_size,
            num_lanes: first.num_lanes,
            num_rows: first.num_rows,
            blinding_mode: first.blinding_mode,
            blinds: folded_blinds,
        })
    }
}

fn validate_polys<Eval: Clone>(polys: &[DenseMultilinearExtension<Eval>]) -> Result<(), ZipError> {
    if let Some(first) = polys.first() {
        for poly in polys {
            if poly.num_vars != first.num_vars || poly.evaluations.len() != first.evaluations.len()
            {
                return Err(ZipError::InvalidPcsParam(
                    "Hyrax batch polynomial shape mismatch".to_string(),
                ));
            }
        }
    }
    Ok(())
}

fn validate_scalar_lanes<C>(
    ck: &HyraxCommitmentKey<C>,
    scalar_lanes: &[Vec<Vec<C::ScalarField>>],
    point_len: usize,
    prover_data: &HyraxProverData<C>,
) -> Result<(), ZipError>
where
    C: AffineRepr,
{
    let expected_n = 1usize
        .checked_shl(u32::try_from(point_len).map_err(|_| {
            ZipError::InvalidPcsParam(format!("Hyrax point length {point_len} is too large"))
        })?)
        .ok_or_else(|| {
            ZipError::InvalidPcsParam(format!("Hyrax point length {point_len} is too large"))
        })?;
    let expected_rows = num_rows(expected_n, ck.num_cols)?;
    if prover_data.batch_size != scalar_lanes.len()
        || prover_data.num_rows != expected_rows
        || prover_data.blinding_mode != ck.blinding_mode
    {
        return Err(ZipError::InvalidPcsParam(
            "Hyrax scalar-lane prover data shape mismatch".to_string(),
        ));
    }
    let expected_blinds = if ck.blinding_mode.is_blinded() {
        prover_data.batch_size * prover_data.num_lanes * prover_data.num_rows
    } else {
        0
    };
    if prover_data.blinds.len() != expected_blinds {
        return Err(ZipError::InvalidPcsParam(
            "Hyrax scalar-lane blind count mismatch".to_string(),
        ));
    }
    for lanes in scalar_lanes {
        if lanes.len() != prover_data.num_lanes {
            return Err(ZipError::InvalidPcsParam(
                "Hyrax scalar-lane count mismatch".to_string(),
            ));
        }
        for values in lanes {
            if values.len() != expected_n {
                return Err(ZipError::InvalidPcsParam(format!(
                    "Hyrax scalar-lane length mismatch: got {}, expected {expected_n}",
                    values.len()
                )));
            }
        }
    }
    Ok(())
}

fn validate_field_lanes<'a, C, F>(
    ck: &HyraxCommitmentKey<C>,
    field_lanes: &[Vec<&'a [F]>],
    point_len: usize,
    prover_data: &HyraxProverData<C>,
) -> Result<(), ZipError>
where
    C: AffineRepr,
    F: PrimeField + 'a,
{
    let expected_n = 1usize
        .checked_shl(u32::try_from(point_len).map_err(|_| {
            ZipError::InvalidPcsParam(format!("Hyrax point length {point_len} is too large"))
        })?)
        .ok_or_else(|| {
            ZipError::InvalidPcsParam(format!("Hyrax point length {point_len} is too large"))
        })?;
    let expected_rows = num_rows(expected_n, ck.num_cols)?;
    if prover_data.batch_size != field_lanes.len()
        || prover_data.num_rows != expected_rows
        || prover_data.blinding_mode != ck.blinding_mode
    {
        return Err(ZipError::InvalidPcsParam(
            "Hyrax field-lane prover data shape mismatch".to_string(),
        ));
    }
    let expected_blinds = if ck.blinding_mode.is_blinded() {
        prover_data.batch_size * prover_data.num_lanes * prover_data.num_rows
    } else {
        0
    };
    if prover_data.blinds.len() != expected_blinds {
        return Err(ZipError::InvalidPcsParam(
            "Hyrax field-lane blind count mismatch".to_string(),
        ));
    }
    for lanes in field_lanes {
        if lanes.len() != prover_data.num_lanes {
            return Err(ZipError::InvalidPcsParam(
                "Hyrax field-lane count mismatch".to_string(),
            ));
        }
        for values in lanes {
            if values.len() != expected_n {
                return Err(ZipError::InvalidPcsParam(format!(
                    "Hyrax field-lane length mismatch: got {}, expected {expected_n}",
                    values.len()
                )));
            }
        }
    }
    Ok(())
}

fn same_commitment_shape<C: AffineRepr>(
    lhs: &HyraxCommitment<C>,
    rhs: &HyraxCommitment<C>,
) -> bool {
    lhs.batch_size == rhs.batch_size
        && lhs.num_lanes == rhs.num_lanes
        && lhs.num_rows == rhs.num_rows
        && lhs.blinding_mode == rhs.blinding_mode
        && lhs.comm.len() == rhs.comm.len()
        && lhs.comm_affine.len() == rhs.comm_affine.len()
        && lhs.comm_bytes.len() == rhs.comm_bytes.len()
}

fn same_prover_data_shape<C: AffineRepr>(
    lhs: &HyraxProverData<C>,
    rhs: &HyraxProverData<C>,
) -> bool {
    lhs.batch_size == rhs.batch_size
        && lhs.num_lanes == rhs.num_lanes
        && lhs.num_rows == rhs.num_rows
        && lhs.blinding_mode == rhs.blinding_mode
        && lhs.blinds.len() == rhs.blinds.len()
}

fn validate_shared_commitment_keys<C: AffineRepr>(
    lhs: &HyraxCommitmentKey<C>,
    rhs: &HyraxCommitmentKey<C>,
) -> Result<(), ZipError> {
    if lhs.num_cols != rhs.num_cols
        || lhs.blinding_mode != rhs.blinding_mode
        || lhs.msm_ck.num_cols != rhs.msm_ck.num_cols
        || lhs.msm_ck.bases != rhs.msm_ck.bases
        || lhs.msm_ck.h != rhs.msm_ck.h
    {
        return Err(ZipError::InvalidPcsParam(
            "Hyrax mixed opening requires shared commitment bases".to_string(),
        ));
    }
    Ok(())
}

fn validate_shared_verifier_keys<C: AffineRepr>(
    lhs: &HyraxVerifierKey<C>,
    rhs: &HyraxVerifierKey<C>,
) -> Result<(), ZipError> {
    if lhs.num_cols != rhs.num_cols
        || lhs.blinding_mode != rhs.blinding_mode
        || lhs.bases != rhs.bases
        || lhs.h != rhs.h
    {
        return Err(ZipError::InvalidPcsParam(
            "Hyrax mixed opening requires shared verifier bases".to_string(),
        ));
    }
    Ok(())
}

fn validate_trusted_bases<C: AffineRepr>(
    width: usize,
    bases: &[C],
    h: &C::Group,
) -> Result<(), ZipError> {
    if !width.is_power_of_two() {
        return Err(ZipError::InvalidPcsParam(format!(
            "Hyrax row width must be a power of two, got {width}"
        )));
    }
    if bases.len() != width {
        return Err(ZipError::InvalidPcsParam(format!(
            "Hyrax expected {width} bases, got {}",
            bases.len()
        )));
    }

    let mut seen = HashSet::with_capacity(bases.len());
    for (idx, base) in bases.iter().copied().enumerate() {
        if base.is_zero() {
            return Err(ZipError::InvalidPcsParam(format!(
                "Hyrax base {idx} is the identity"
            )));
        }
        if !seen.insert(base) {
            return Err(ZipError::InvalidPcsParam(format!(
                "Hyrax base {idx} duplicates an earlier base"
            )));
        }
    }

    let h_affine = h.clone().into_affine();
    if h_affine.is_zero() {
        return Err(ZipError::InvalidPcsParam(
            "Hyrax blinding base is the identity".to_string(),
        ));
    }
    if seen.contains(&h_affine) {
        return Err(ZipError::InvalidPcsParam(
            "Hyrax blinding base duplicates a witness base".to_string(),
        ));
    }

    Ok(())
}

fn validate_hyrax_shape<C, Lanes, Eval, const D: usize>(
    width: usize,
    blinding_mode: HyraxBlindingMode,
    polys: &[DenseMultilinearExtension<Eval>],
    prover_data: &HyraxProverData<C>,
) -> Result<(), ZipError>
where
    C: AffineRepr,
    Lanes: HyraxLanes<C, Eval, D>,
    Eval: Clone + Debug + Send + Sync,
{
    let n = polys[0].evaluations.len();
    let num_rows = num_rows(n, width)?;
    let expected_blinds = if blinding_mode.is_blinded() {
        polys.len() * Lanes::NUM_LANES * num_rows
    } else {
        0
    };
    if prover_data.batch_size != polys.len()
        || prover_data.num_lanes != Lanes::NUM_LANES
        || prover_data.num_rows != num_rows
        || prover_data.blinding_mode != blinding_mode
        || prover_data.blinds.len() != expected_blinds
    {
        return Err(ZipError::InvalidPcsParam(
            "Hyrax prover data shape mismatch".to_string(),
        ));
    }
    Ok(())
}

fn validate_commitment_shape<C, Lanes, Eval, const D: usize>(
    commitment: &HyraxCommitment<C>,
) -> Result<(), ZipError>
where
    C: AffineRepr,
    Lanes: HyraxLanes<C, Eval, D>,
    Eval: Clone + Debug + Send + Sync,
{
    if commitment.num_lanes != Lanes::NUM_LANES {
        return Err(ZipError::InvalidPcsParam(format!(
            "Hyrax commitment lane mismatch: expected {}, got {}",
            Lanes::NUM_LANES,
            commitment.num_lanes
        )));
    }
    let expected = commitment.batch_size * commitment.num_lanes * commitment.num_rows;
    if commitment.comm.len() != expected {
        return Err(ZipError::InvalidPcsParam(format!(
            "Hyrax commitment expected {expected} row commitments, got {}",
            commitment.comm.len()
        )));
    }
    if commitment.comm_affine.len() != expected {
        return Err(ZipError::InvalidPcsParam(format!(
            "Hyrax commitment expected {expected} affine row commitments, got {}",
            commitment.comm_affine.len()
        )));
    }
    let expected_bytes = expected * C::zero().serialized_size(Compress::Yes);
    if commitment.comm_bytes.len() != expected_bytes {
        return Err(ZipError::InvalidPcsParam(format!(
            "Hyrax commitment expected {expected_bytes} commitment bytes, got {}",
            commitment.comm_bytes.len()
        )));
    }
    Ok(())
}

fn commit_hyrax_poly<C, Lanes, Eval, const D: usize>(
    ck: &HyraxCommitmentKey<C>,
    poly: &DenseMultilinearExtension<Eval>,
    num_rows: usize,
) -> Result<(Vec<C::Group>, Vec<C::ScalarField>), ZipError>
where
    C: AffineRepr,
    Lanes: HyraxLanes<C, Eval, D>,
    Eval: Clone + Debug + Send + Sync,
{
    if let Some(result) = Lanes::commit_poly(ck, poly, num_rows) {
        return result;
    }

    let per_lane = cfg_into_iter!(0..Lanes::NUM_LANES)
        .map(|lane| {
            let values = lane_values::<C, Lanes, Eval, D>(poly, lane)?;
            if ck.blinding_mode.is_blinded() {
                let blind = MsmCommitmentEngine::<C>::blind(&ck.msm_ck, values.len());
                let commitment = MsmCommitmentEngine::<C>::commit_with::<_, Lanes::Strategy>(
                    &ck.msm_ck, &values, &blind,
                )
                .map_err(msm_err)?;
                Ok::<(Vec<C::Group>, Vec<C::ScalarField>), ZipError>((commitment.comm, blind.blind))
            } else {
                let commitment = MsmCommitmentEngine::<C>::commit_unblinded_with::<
                    _,
                    Lanes::Strategy,
                >(&ck.msm_ck, &values)
                .map_err(msm_err)?;
                Ok::<(Vec<C::Group>, Vec<C::ScalarField>), ZipError>((commitment.comm, Vec::new()))
            }
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut comm = Vec::with_capacity(Lanes::NUM_LANES * num_rows);
    let mut blinds = if ck.blinding_mode.is_blinded() {
        Vec::with_capacity(Lanes::NUM_LANES * num_rows)
    } else {
        Vec::new()
    };
    for (lane_comm, lane_blinds) in per_lane {
        comm.extend(lane_comm);
        blinds.extend(lane_blinds);
    }
    Ok((comm, blinds))
}

fn lane_values<C, Lanes, Eval, const D: usize>(
    poly: &DenseMultilinearExtension<Eval>,
    lane: usize,
) -> Result<Vec<Lanes::LaneValue>, ZipError>
where
    C: AffineRepr,
    Lanes: HyraxLanes<C, Eval, D>,
    Eval: Clone + Debug + Send + Sync,
{
    poly.evaluations
        .iter()
        .map(|eval| Lanes::lane_value(eval, lane))
        .collect()
}

fn random_scalars<C: AffineRepr>(n: usize) -> Vec<C::ScalarField> {
    let mut rng = ark_std::rand::thread_rng();
    (0..n).map(|_| C::ScalarField::rand(&mut rng)).collect()
}

fn use_inner_bool_parallelism(outer_jobs: usize) -> bool {
    #[cfg(feature = "parallel")]
    {
        outer_jobs < rayon::current_num_threads()
    }

    #[cfg(not(feature = "parallel"))]
    {
        let _ = outer_jobs;
        false
    }
}

fn hash_to_curve<C: AffineRepr>(domain: &[u8], label: &[u8], index: usize) -> Result<C, ZipError> {
    let point_bytes = C::zero().serialized_size(Compress::Yes);
    let mut counter = 0u64;
    loop {
        let mut hasher = blake3::Hasher::new();
        absorb_hash_part(&mut hasher, b"zinc-plus-hyrax-setup-v1")?;
        absorb_hash_part(&mut hasher, domain)?;
        absorb_hash_part(&mut hasher, label)?;
        hasher.update(
            &u64::try_from(index)
                .map_err(|_| {
                    ZipError::InvalidPcsParam("Hyrax setup index does not fit u64".to_string())
                })?
                .to_le_bytes(),
        );
        hasher.update(&counter.to_le_bytes());

        let mut bytes = vec![0u8; point_bytes];
        hasher.finalize_xof().fill(&mut bytes);
        if let Some(point) = C::from_random_bytes(&bytes).map(|point| point.clear_cofactor()) {
            if !point.is_zero() {
                return Ok(point);
            }
        }

        counter = counter.checked_add(1).ok_or_else(|| {
            ZipError::InvalidPcsParam("Hyrax hash-to-curve setup exhausted counters".to_string())
        })?;
    }
}

fn absorb_hash_part(hasher: &mut blake3::Hasher, part: &[u8]) -> Result<(), ZipError> {
    hasher.update(
        &u64::try_from(part.len())
            .map_err(|_| {
                ZipError::InvalidPcsParam(
                    "Hyrax setup domain component length does not fit u64".to_string(),
                )
            })?
            .to_le_bytes(),
    );
    hasher.update(part);
    Ok(())
}

fn int_to_scalar<C: AffineRepr, const LIMBS: usize>(
    value: &Int<LIMBS>,
) -> Result<C::ScalarField, ZipError> {
    let (abs, is_negative) = if value.is_negative() {
        (
            value.checked_abs().ok_or_else(|| {
                ZipError::InvalidPcsParam("cannot convert minimum Int to scalar".to_string())
            })?,
            true,
        )
    } else {
        (*value, false)
    };
    let mut scalar = unsigned_int_to_scalar::<C, LIMBS>(&abs);
    if is_negative && !scalar.is_zero() {
        scalar = -scalar;
    }
    Ok(scalar)
}

fn unsigned_int_to_scalar<C: AffineRepr, const LIMBS: usize>(value: &Int<LIMBS>) -> C::ScalarField {
    let mut bytes = Vec::with_capacity(LIMBS * core::mem::size_of::<crypto_bigint::Word>());
    for word in value.as_uint().as_words() {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    C::ScalarField::from_le_bytes_mod_order(&bytes)
}

fn validate_curve_scalar_modulus<C, const LIMBS: usize>(
    actual: &Uint<LIMBS>,
) -> Result<(), ZipError>
where
    C: AffineRepr,
{
    let expected =
        uint_from_le_bytes::<LIMBS>(&<C::ScalarField as ArkPrimeField>::MODULUS.to_bytes_le());
    if actual != &expected {
        return Err(ZipError::InvalidPcsParam(
            "Hyrax field mismatch: protocol field modulus must equal curve scalar modulus"
                .to_string(),
        ));
    }
    Ok(())
}

fn validate_curve_scalar_modulus_boxed<C>(actual: &BoxedUint) -> Result<(), ZipError>
where
    C: AffineRepr,
{
    let expected = BoxedUint::from_le_slice(
        &<C::ScalarField as ArkPrimeField>::MODULUS.to_bytes_le(),
        actual.bits_precision(),
    )
    .expect("curve scalar modulus must fit protocol field precision");
    if actual != &expected {
        return Err(ZipError::InvalidPcsParam(
            "Hyrax field mismatch: protocol field modulus must equal curve scalar modulus"
                .to_string(),
        ));
    }
    Ok(())
}

fn uint_from_le_bytes<const LIMBS: usize>(bytes: &[u8]) -> Uint<LIMBS> {
    let num_bytes = <Uint<LIMBS> as ConstTranscribable>::NUM_BYTES;
    assert!(
        bytes.len() <= num_bytes,
        "integer encoding does not fit in target Uint",
    );
    let mut padded = vec![0u8; num_bytes];
    padded[..bytes.len()].copy_from_slice(bytes);
    Uint::<LIMBS>::read_transcription_bytes_exact(&padded)
}

fn msm_key<C: AffineRepr>(
    width: usize,
    bases: Vec<C>,
    h: C::Group,
) -> Result<MsmCommitmentKey<C>, ZipError> {
    MsmCommitmentEngine::<C>::setup_from_bases(width, bases, h)
        .map(|(ck, _)| ck)
        .map_err(msm_err)
}

fn msm_unchecked<C: AffineRepr>(
    bases: &[C],
    scalars: &[C::ScalarField],
) -> Result<C::Group, ZipError> {
    if bases.len() != scalars.len() {
        return Err(ZipError::InvalidPcsParam(format!(
            "Hyrax MSM expected {} bases, got {}",
            scalars.len(),
            bases.len()
        )));
    }
    if !scalars.iter().any(|scalar| scalar.is_zero()) {
        return Ok(<C::Group as VariableBaseMSM>::msm_unchecked(bases, scalars));
    }

    let non_zero = scalars
        .iter()
        .enumerate()
        .filter(|(_, scalar)| !scalar.is_zero());
    let mut filtered_bases = Vec::new();
    let mut filtered_scalars = Vec::new();
    for (idx, scalar) in non_zero {
        filtered_bases.push(bases[idx]);
        filtered_scalars.push(*scalar);
    }
    if filtered_scalars.is_empty() {
        return Ok(C::Group::zero());
    }

    Ok(<C::Group as VariableBaseMSM>::msm_unchecked(
        &filtered_bases,
        &filtered_scalars,
    ))
}

fn msm_shared_weight_commitments_unchecked<C: AffineRepr>(
    scalars: &[C::ScalarField],
    commitments: &[&HyraxCommitment<C>],
) -> Result<Vec<C::Group>, ZipError> {
    if commitments.is_empty() {
        return Ok(Vec::new());
    }
    let row_count = commitments[0].comm_affine.len();
    debug_assert!(
        commitments
            .iter()
            .all(|commitment| commitment.comm_affine.len() == row_count)
    );

    msm_shared_weights_indexed_unchecked::<C, _>(scalars, row_count, |row_idx, scalar_idx| {
        commitments[scalar_idx].comm_affine[row_idx]
    })
}

fn msm_shared_weights_indexed_unchecked<C, BaseAt>(
    scalars: &[C::ScalarField],
    row_count: usize,
    base_at: BaseAt,
) -> Result<Vec<C::Group>, ZipError>
where
    C: AffineRepr,
    BaseAt: Fn(usize, usize) -> C + Sync,
{
    if row_count == 0 {
        return Ok(Vec::new());
    }

    let one = C::ScalarField::from(1u64);
    let mut unit_indices = Vec::new();
    let mut general_indices = Vec::new();
    let mut general_scalars = Vec::new();
    for (idx, scalar) in scalars.iter().enumerate() {
        if scalar.is_zero() {
            continue;
        }
        if *scalar == one {
            unit_indices.push(idx);
        } else {
            general_indices.push(idx);
            general_scalars.push(scalar.into_bigint());
        }
    }

    if general_indices.is_empty() {
        return Ok(cfg_into_iter!(0..row_count)
            .map(|row_idx| {
                let mut acc = C::Group::zero();
                for &idx in &unit_indices {
                    acc += base_at(row_idx, idx);
                }
                acc
            })
            .collect());
    }

    let window_bits = shared_weight_window_bits(scalars.len());
    let half_window = 1usize << (window_bits - 1);
    let full_window = 1usize << window_bits;
    let bucket_len = half_window;
    let segments =
        <usize as Integer>::div_ceil(&(C::ScalarField::MODULUS_BIT_SIZE as usize), &window_bits)
            + 1;
    let mut carries = vec![0u8; general_scalars.len()];
    let mut signed_windows = Vec::with_capacity(segments);
    for segment in 0..segments {
        let offset = segment * window_bits;
        let mut digits = Vec::with_capacity(general_scalars.len());
        for (idx, scalar) in general_scalars.iter().enumerate() {
            let raw = window_value_from_limbs(scalar.as_ref(), offset, window_bits)
                + usize::from(carries[idx]);
            if raw >= half_window {
                digits.push(-((full_window - raw) as i16));
                carries[idx] = 1;
            } else {
                digits.push(raw as i16);
                carries[idx] = 0;
            }
        }
        signed_windows.push(digits);
    }

    if bucket_len == 4 {
        return Ok(cfg_into_iter!(0..row_count)
            .map(|row_idx| {
                let mut unit_sum = C::Group::zero();
                for &idx in &unit_indices {
                    unit_sum += base_at(row_idx, idx);
                }

                let mut buckets: [C::Group; 4] = std::array::from_fn(|_| C::Group::zero());
                let mut acc = C::Group::zero();
                for digits in signed_windows.iter().rev() {
                    for _ in 0..window_bits {
                        acc.double_in_place();
                    }
                    for bucket in &mut buckets {
                        *bucket = C::Group::zero();
                    }
                    for (general_idx, digit) in digits.iter().enumerate() {
                        if *digit > 0 {
                            buckets[*digit as usize - 1] +=
                                base_at(row_idx, general_indices[general_idx]);
                        } else if *digit < 0 {
                            buckets[(-*digit) as usize - 1] -=
                                base_at(row_idx, general_indices[general_idx]);
                        }
                    }
                    acc += bucket_running_sum(&buckets);
                }

                unit_sum + acc
            })
            .collect());
    }

    Ok(cfg_into_iter!(0..row_count)
        .map(|row_idx| {
            let mut unit_sum = C::Group::zero();
            for &idx in &unit_indices {
                unit_sum += base_at(row_idx, idx);
            }

            let mut buckets = vec![C::Group::zero(); bucket_len];
            let mut acc = C::Group::zero();
            for digits in signed_windows.iter().rev() {
                for _ in 0..window_bits {
                    acc.double_in_place();
                }
                for bucket in &mut buckets {
                    *bucket = C::Group::zero();
                }
                for (general_idx, digit) in digits.iter().enumerate() {
                    if *digit > 0 {
                        buckets[*digit as usize - 1] +=
                            base_at(row_idx, general_indices[general_idx]);
                    } else if *digit < 0 {
                        buckets[(-*digit) as usize - 1] -=
                            base_at(row_idx, general_indices[general_idx]);
                    }
                }
                acc += bucket_running_sum(&buckets);
            }

            unit_sum + acc
        })
        .collect())
}

fn shared_weight_window_bits(n: usize) -> usize {
    if n < 32 {
        3
    } else {
        (usize::BITS - n.leading_zeros()) as usize
    }
}

fn bucket_running_sum<G: CurveGroup>(buckets: &[G]) -> G {
    let mut acc = G::zero();
    let mut running_sum = G::zero();
    for bucket in buckets.iter().rev() {
        running_sum += bucket;
        acc += running_sum;
    }
    acc
}

fn window_value_from_limbs(limbs: &[u64], start: usize, width: usize) -> usize {
    (0..width).fold(0usize, |value, bit_idx| {
        let absolute_bit = start + bit_idx;
        let limb_idx = absolute_bit / u64::BITS as usize;
        let limb_bit = absolute_bit % u64::BITS as usize;
        if limbs
            .get(limb_idx)
            .map(|limb| ((limb >> limb_bit) & 1) == 1)
            .unwrap_or(false)
        {
            value | (1usize << bit_idx)
        } else {
            value
        }
    })
}

fn num_rows(n: usize, width: usize) -> Result<usize, ZipError> {
    if width == 0 {
        return Err(ZipError::InvalidPcsParam(
            "Hyrax row width must be non-zero".to_string(),
        ));
    }
    Ok(<usize as Integer>::div_ceil(&n, &width))
}

fn alpha_index_dynamic(num_lanes: usize, poly_idx: usize, lane: usize) -> usize {
    poly_idx * num_lanes + lane
}

fn commitment_index_dynamic(
    num_lanes: usize,
    poly_idx: usize,
    lane: usize,
    row_idx: usize,
    num_rows: usize,
) -> usize {
    ((poly_idx * num_lanes + lane) * num_rows) + row_idx
}

fn eq_tensor_f<F: PrimeField>(point: &[F], cfg: &F::Config) -> Vec<F> {
    let mut tensor = vec![F::one_with_cfg(cfg)];
    for r in point {
        let one_minus = {
            let mut value = F::one_with_cfg(cfg);
            value -= r;
            value
        };
        let current = tensor.clone();
        tensor.clear();
        for value in &current {
            let mut lo = value.clone();
            lo *= &one_minus;
            tensor.push(lo);
        }
        for value in current {
            let mut hi = value;
            hi *= r;
            tensor.push(hi);
        }
    }
    tensor
}

fn eq_tensor_scalar<C: AffineRepr>(point: &[C::ScalarField]) -> Vec<C::ScalarField> {
    let mut tensor = vec![C::ScalarField::from(1u64)];
    for r in point {
        let one_minus = C::ScalarField::from(1u64) - r;
        let current = tensor.clone();
        tensor.clear();
        for value in &current {
            tensor.push(*value * one_minus);
        }
        for value in current {
            tensor.push(value * r);
        }
    }
    tensor
}

fn sample_scalars<C: AffineRepr>(
    transcript: &mut impl Transcript,
    n: usize,
) -> Vec<C::ScalarField> {
    (0..n)
        .map(|_| {
            let mut bytes = Vec::with_capacity(64);
            for _ in 0..8 {
                let word = transcript.get_challenge::<u64>();
                bytes.extend_from_slice(&word.to_le_bytes());
            }
            C::ScalarField::from_le_bytes_mod_order(&bytes)
        })
        .collect()
}

fn write_scalars<C: AffineRepr>(
    transcript: &mut PcsProverTranscript,
    scalars: &[C::ScalarField],
) -> Result<(), ZipError> {
    for scalar in scalars {
        write_scalar::<C>(transcript, scalar)?;
    }
    Ok(())
}

fn write_scalar<C: AffineRepr>(
    transcript: &mut PcsProverTranscript,
    scalar: &C::ScalarField,
) -> Result<(), ZipError> {
    let bytes = scalar_bytes::<C>(scalar)?;
    transcript.fs_transcript.absorb_slice(&bytes);
    transcript.stream.write_all(&bytes)?;
    Ok(())
}

fn read_scalars<C: AffineRepr>(
    transcript: &mut PcsVerifierTranscript,
    n: usize,
) -> Result<Vec<C::ScalarField>, ZipError> {
    (0..n).map(|_| read_scalar::<C>(transcript)).collect()
}

fn read_scalar<C: AffineRepr>(
    transcript: &mut PcsVerifierTranscript,
) -> Result<C::ScalarField, ZipError> {
    let size = C::ScalarField::zero().serialized_size(Compress::Yes);
    let mut bytes = vec![0u8; size];
    transcript.stream.read_exact(&mut bytes)?;
    transcript.fs_transcript.absorb_slice(&bytes);
    C::ScalarField::deserialize_compressed(bytes.as_slice()).map_err(ark_err)
}

fn scalar_bytes<C: AffineRepr>(scalar: &C::ScalarField) -> Result<Vec<u8>, ZipError> {
    let mut bytes = Vec::with_capacity(scalar.serialized_size(Compress::Yes));
    scalar.serialize_compressed(&mut bytes).map_err(ark_err)?;
    Ok(bytes)
}

fn affine_bytes_into<C: AffineRepr>(affine: &C, bytes: &mut Vec<u8>) -> Result<(), ZipError> {
    affine.serialize_compressed(bytes).map_err(ark_err)
}

fn affine_points_bytes<C: AffineRepr>(points: &[C]) -> Result<Vec<u8>, ZipError> {
    let point_size = C::zero().serialized_size(Compress::Yes);
    let mut bytes = Vec::with_capacity(points.len() * point_size);
    for point in points {
        affine_bytes_into::<C>(point, &mut bytes)?;
    }
    Ok(bytes)
}

fn msm_err(err: MsmError) -> ZipError {
    ZipError::InvalidPcsParam(err.to_string())
}

fn ark_err(err: ark_serialize::SerializationError) -> ZipError {
    ZipError::Serialization(format!("ark serialization error: {err}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    use ark_ec::PrimeGroup;
    use ark_ff::Field as ArkField;
    use crypto_primitives::FromWithConfig;

    fn cfg_from_curve<C: AffineRepr>() -> <MontyField<4> as PrimeField>::Config {
        let modulus =
            uint_from_le_bytes::<4>(&<C::ScalarField as ArkPrimeField>::MODULUS.to_bytes_le());
        <MontyField<4> as PrimeField>::make_cfg(&modulus)
            .expect("curve scalar modulus must be prime")
    }

    fn assert_bridge_round_trip<C: AffineRepr>() -> Result<(), ZipError> {
        let cfg = cfg_from_curve::<C>();
        for value in [0u64, 1, 2, 17, 123, 1 << 20] {
            let field = MontyField::<4>::from_with_cfg(value, &cfg);
            let scalar = <MontyField<4> as HyraxFieldBridge<C>>::field_to_scalar(&field)?;
            assert_eq!(scalar, C::ScalarField::from(value));

            let field_again =
                <MontyField<4> as HyraxFieldBridge<C>>::scalar_to_field(&scalar, &cfg)?;
            assert_eq!(field_again, field);
        }

        let large_values = [
            C::ScalarField::from(2u64).inverse().unwrap(),
            -C::ScalarField::from(1u64),
            C::ScalarField::from_le_bytes_mod_order(&[0xA5; 64]),
        ];
        for scalar in large_values {
            let field = <MontyField<4> as HyraxFieldBridge<C>>::scalar_to_field(&scalar, &cfg)?;
            let scalar_again = <MontyField<4> as HyraxFieldBridge<C>>::field_to_scalar(&field)?;
            assert_eq!(scalar_again, scalar);
        }
        Ok(())
    }

    #[test]
    fn bridge_round_trips_bn254_scalar_field() {
        assert_bridge_round_trip::<ark_bn254::G1Affine>().unwrap();
    }

    #[test]
    fn bridge_round_trips_secp256k1_scalar_field() {
        assert_bridge_round_trip::<ark_secp256k1::Affine>().unwrap();
    }

    #[test]
    fn bridge_rejects_mismatched_field_config() {
        let bn_cfg = cfg_from_curve::<ark_bn254::G1Affine>();
        let bn_field = MontyField::<4>::from_with_cfg(1u64, &bn_cfg);
        let result =
            <MontyField<4> as HyraxFieldBridge<ark_secp256k1::Affine>>::field_to_scalar(&bn_field);
        assert!(matches!(result, Err(ZipError::InvalidPcsParam(_))));
    }

    #[test]
    fn setup_derives_distinct_deterministic_bases() {
        type C = ark_bn254::G1Affine;
        let width = 32;
        let (ck_0, vk_0) = HyraxPCS::<C, BinaryLanes>::setup(
            width,
            b"zinc-plus-hyrax-setup-test",
            HyraxBlindingMode::Unblinded,
        )
        .unwrap();
        let (ck_1, vk_1) = HyraxPCS::<C, BinaryLanes>::setup(
            width,
            b"zinc-plus-hyrax-setup-test",
            HyraxBlindingMode::Unblinded,
        )
        .unwrap();

        assert_eq!(ck_0.msm_ck.bases, ck_1.msm_ck.bases);
        assert_eq!(vk_0.bases, vk_1.bases);
        assert_eq!(ck_0.msm_ck.h, ck_1.msm_ck.h);
        assert_eq!(vk_0.h, vk_1.h);
        assert_eq!(ck_0.blinding_mode, HyraxBlindingMode::Unblinded);
        assert_eq!(vk_0.blinding_mode, HyraxBlindingMode::Unblinded);
        assert!(ck_0.msm_ck.bases.iter().all(|base| !base.is_zero()));
        assert!(!ck_0.msm_ck.h.is_zero());

        let seen = ck_0.msm_ck.bases.iter().copied().collect::<HashSet<_>>();
        assert_eq!(seen.len(), width);
        assert!(!seen.contains(&ck_0.msm_ck.h.into_affine()));
    }

    #[test]
    fn trusted_setup_rejects_bad_bases() {
        type C = ark_bn254::G1Affine;
        let width = 8;
        let generator = <C as AffineRepr>::Group::generator();
        let bases = (1..=width)
            .map(|idx| (generator * <C as AffineRepr>::ScalarField::from(idx as u64)).into_affine())
            .collect::<Vec<_>>();
        let h = generator * <C as AffineRepr>::ScalarField::from((width + 1) as u64);

        assert!(matches!(
            HyraxPCS::<C, BinaryLanes>::setup_from_trusted_bases(
                0,
                Vec::new(),
                <C as AffineRepr>::Group::zero(),
                HyraxBlindingMode::Unblinded,
            ),
            Err(ZipError::InvalidPcsParam(_))
        ));

        assert!(matches!(
            HyraxPCS::<C, BinaryLanes>::setup_from_trusted_bases(
                width,
                bases[..width - 1].to_vec(),
                h,
                HyraxBlindingMode::Unblinded,
            ),
            Err(ZipError::InvalidPcsParam(_))
        ));

        let mut identity_bases = bases.clone();
        identity_bases[0] = C::zero();
        assert!(matches!(
            HyraxPCS::<C, BinaryLanes>::setup_from_trusted_bases(
                width,
                identity_bases,
                h,
                HyraxBlindingMode::Unblinded,
            ),
            Err(ZipError::InvalidPcsParam(_))
        ));

        let mut duplicate_bases = bases.clone();
        duplicate_bases[1] = duplicate_bases[0];
        assert!(matches!(
            HyraxPCS::<C, BinaryLanes>::setup_from_trusted_bases(
                width,
                duplicate_bases,
                h,
                HyraxBlindingMode::Unblinded,
            ),
            Err(ZipError::InvalidPcsParam(_))
        ));

        assert!(matches!(
            HyraxPCS::<C, BinaryLanes>::setup_from_trusted_bases(
                width,
                bases.clone(),
                <C as AffineRepr>::Group::zero(),
                HyraxBlindingMode::Unblinded,
            ),
            Err(ZipError::InvalidPcsParam(_))
        ));

        assert!(matches!(
            HyraxPCS::<C, BinaryLanes>::setup_from_trusted_bases(
                width,
                bases.clone(),
                bases[0].into_group(),
                HyraxBlindingMode::Unblinded,
            ),
            Err(ZipError::InvalidPcsParam(_))
        ));
    }

    fn binary_hyrax_open_verify_round_trip_with_modes(
        commit_mode: HyraxBlindingMode,
        verify_mode: HyraxBlindingMode,
    ) -> Result<(), ZipError> {
        type C = ark_bn254::G1Affine;
        type F = MontyField<4>;
        const D: usize = 32;

        fn bp(bits: u32) -> BinaryPoly<D> {
            BinaryPoly::<D>::from(bits)
        }

        let cfg = cfg_from_curve::<C>();
        let width = 512;
        let (ck, _) = HyraxPCS::<C, BinaryLanes>::setup(
            width,
            b"zinc-plus-hyrax-round-trip-test",
            commit_mode,
        )?;
        let (_, vk) = HyraxPCS::<C, BinaryLanes>::setup(
            width,
            b"zinc-plus-hyrax-round-trip-test",
            verify_mode,
        )?;

        let evals0 = (0..width)
            .map(|idx| bp((idx as u32).wrapping_mul(0x9E37_79B1)))
            .collect::<Vec<_>>();
        let evals1 = (0..width)
            .map(|idx| bp(!((idx as u32).wrapping_mul(0x85EB_CA6B))))
            .collect::<Vec<_>>();
        let polys = vec![
            DenseMultilinearExtension::from_evaluations_vec(9, evals0, bp(0)),
            DenseMultilinearExtension::from_evaluations_vec(9, evals1, bp(0)),
        ];
        let (prover_data, commitment) =
            <HyraxPCS<C, BinaryLanes> as PCS<F, BinaryPoly<D>, D>>::commit(&ck, &polys)?;

        let point = [
            [0x11u8; 64],
            [0x22u8; 64],
            [0x33u8; 64],
            [0x44u8; 64],
            [0x55u8; 64],
            [0x66u8; 64],
            [0x77u8; 64],
            [0x88u8; 64],
            [0xA5u8; 64],
        ]
        .iter()
        .map(|bytes| {
            let scalar = <C as AffineRepr>::ScalarField::from_le_bytes_mod_order(bytes);
            <F as HyraxFieldBridge<C>>::scalar_to_field(&scalar, &cfg)
        })
        .collect::<Result<Vec<_>, _>>()?;
        let eq = eq_tensor_f::<F>(&point, &cfg);
        let lifted_evals = polys
            .iter()
            .map(|poly| {
                let mut coeffs = vec![F::zero_with_cfg(&cfg); D];
                for (weight, eval) in eq.iter().zip(poly.evaluations.iter()) {
                    for (lane, bit) in eval.iter().enumerate() {
                        if bit.inner() {
                            coeffs[lane] += weight;
                        }
                    }
                }
                DynamicPolynomialF::new_trimmed(coeffs)
            })
            .collect::<Vec<_>>();

        let mut prover_transcript = PcsProverTranscript {
            fs_transcript: Default::default(),
            stream: Default::default(),
        };
        <HyraxPCS<C, BinaryLanes> as PCS<F, BinaryPoly<D>, D>>::absorb_commitment(
            &mut prover_transcript.fs_transcript,
            &commitment,
        );
        let mut transcription_buf = vec![0u8; <F as crypto_primitives::Field>::Inner::NUM_BYTES];
        for lifted_eval in &lifted_evals {
            prover_transcript
                .fs_transcript
                .absorb_random_field_slice(&lifted_eval.coeffs, &mut transcription_buf);
        }
        <HyraxPCS<C, BinaryLanes> as PCS<F, BinaryPoly<D>, D>>::prove_open::<true>(
            &mut prover_transcript,
            &ck,
            &polys,
            &point,
            &prover_data,
            &cfg,
        )?;

        let mut verifier_transcript = prover_transcript.into_verification_transcript();
        <HyraxPCS<C, BinaryLanes> as PCS<F, BinaryPoly<D>, D>>::absorb_commitment(
            &mut verifier_transcript.fs_transcript,
            &commitment,
        );
        let mut transcription_buf = vec![0u8; <F as crypto_primitives::Field>::Inner::NUM_BYTES];
        for lifted_eval in &lifted_evals {
            verifier_transcript
                .fs_transcript
                .absorb_random_field_slice(&lifted_eval.coeffs, &mut transcription_buf);
        }
        <HyraxPCS<C, BinaryLanes> as PCS<F, BinaryPoly<D>, D>>::verify_open::<true>(
            &mut verifier_transcript,
            &vk,
            &commitment,
            &point,
            &lifted_evals,
            &Vec::new(),
            &cfg,
        )
    }

    #[test]
    fn binary_hyrax_open_verify_round_trip() {
        binary_hyrax_open_verify_round_trip_with_modes(
            HyraxBlindingMode::Blinded,
            HyraxBlindingMode::Blinded,
        )
        .unwrap();
    }

    #[test]
    fn unblinded_binary_hyrax_open_verify_round_trip() {
        binary_hyrax_open_verify_round_trip_with_modes(
            HyraxBlindingMode::Unblinded,
            HyraxBlindingMode::Unblinded,
        )
        .unwrap();
    }

    #[test]
    fn binary_hyrax_commitment_order_is_poly_lane_row() {
        type C = ark_bn254::G1Affine;
        type F = MontyField<4>;
        const D: usize = 32;

        fn bp(bits: u32) -> BinaryPoly<D> {
            BinaryPoly::<D>::from(bits)
        }

        let width = 8;
        let (ck, _) = HyraxPCS::<C, BinaryLanes>::setup(
            width,
            b"zinc-plus-hyrax-order-test",
            HyraxBlindingMode::Unblinded,
        )
        .unwrap();
        let polys = vec![
            DenseMultilinearExtension::from_evaluations_vec(
                4,
                (0..16).map(|idx| bp((idx * 13 + 7) as u32)).collect(),
                bp(0),
            ),
            DenseMultilinearExtension::from_evaluations_vec(
                4,
                (0..16)
                    .map(|idx| bp(((idx * 29 + 3) as u32).reverse_bits()))
                    .collect(),
                bp(0),
            ),
        ];

        let (prover_data, commitment) =
            <HyraxPCS<C, BinaryLanes> as PCS<F, BinaryPoly<D>, D>>::commit(&ck, &polys).unwrap();

        let mut expected = Vec::new();
        BoolSubsetMsm::<6>::precompute_ck(&ck.msm_ck);
        for poly in &polys {
            for lane in 0..D {
                for row in poly.evaluations.chunks(width) {
                    let values = row.iter().map(|eval| eval.coeff(lane)).collect::<Vec<_>>();
                    let row_comm = if values.iter().copied().any(|bit| bit) {
                        BoolSubsetMsm::<6>::msm_bool_row(&ck.msm_ck, &values, false).unwrap()
                    } else {
                        <C as AffineRepr>::Group::zero()
                    };
                    expected.push(row_comm);
                }
            }
        }

        assert_eq!(prover_data.blinds.len(), 0);
        assert_eq!(commitment.num_rows, 2);
        assert_eq!(commitment.comm, expected);
    }

    #[test]
    fn binary_hyrax_commitment_supports_partial_single_row() {
        type C = ark_bn254::G1Affine;
        type F = MontyField<4>;
        const D: usize = 32;

        fn bp(bits: u32) -> BinaryPoly<D> {
            BinaryPoly::<D>::from(bits)
        }

        let width = 32;
        let (ck, _) = HyraxPCS::<C, BinaryLanes>::setup(
            width,
            b"zinc-plus-hyrax-partial-row-test",
            HyraxBlindingMode::Unblinded,
        )
        .unwrap();
        let polys = vec![DenseMultilinearExtension::from_evaluations_vec(
            4,
            (0..16).map(|idx| bp((idx * 17 + 11) as u32)).collect(),
            bp(0),
        )];

        let (prover_data, commitment) =
            <HyraxPCS<C, BinaryLanes> as PCS<F, BinaryPoly<D>, D>>::commit(&ck, &polys).unwrap();

        assert_eq!(prover_data.num_rows, 1);
        assert_eq!(commitment.num_rows, 1);
        assert_eq!(commitment.comm.len(), D);
        for (lane, comm) in commitment.comm.iter().enumerate() {
            let values = polys[0]
                .evaluations
                .iter()
                .map(|eval| eval.coeff(lane))
                .collect::<Vec<_>>();
            let expected = if values.iter().copied().any(|bit| bit) {
                BoolSubsetMsm::<6>::msm_bool_row(&ck.msm_ck, &values, false).unwrap()
            } else {
                <C as AffineRepr>::Group::zero()
            };
            assert_eq!(*comm, expected);
        }
    }

    #[test]
    fn hyrax_rejects_blinding_mode_mismatch() {
        let result = binary_hyrax_open_verify_round_trip_with_modes(
            HyraxBlindingMode::Unblinded,
            HyraxBlindingMode::Blinded,
        );
        assert!(result.is_err());
    }

    #[test]
    fn folded_binary_hyrax_commitment_opens_from_scalar_lanes() {
        type C = ark_bn254::G1Affine;
        type F = MontyField<4>;
        const D: usize = 32;

        fn bp(bits: u32) -> BinaryPoly<D> {
            BinaryPoly::<D>::from(bits)
        }

        let cfg = cfg_from_curve::<C>();
        let n = 8;
        let width = n;
        let generator = <C as AffineRepr>::Group::generator();
        let bases = (1..=width)
            .map(|idx| (generator * <C as AffineRepr>::ScalarField::from(idx as u64)).into_affine())
            .collect::<Vec<_>>();
        let h = generator * <C as AffineRepr>::ScalarField::from((width + 1) as u64);
        let (ck, vk) = HyraxPCS::<C, BinaryLanes>::setup_from_bases_with_blinding(
            width,
            bases,
            h,
            HyraxBlindingMode::Blinded,
        )
        .unwrap();

        let instance_polys = [
            vec![
                DenseMultilinearExtension::from_evaluations_vec(
                    3,
                    (0..n).map(|idx| bp((idx as u32) * 17 + 3)).collect(),
                    bp(0),
                ),
                DenseMultilinearExtension::from_evaluations_vec(
                    3,
                    (0..n).map(|idx| bp(!((idx as u32) * 11))).collect(),
                    bp(0),
                ),
            ],
            vec![
                DenseMultilinearExtension::from_evaluations_vec(
                    3,
                    (0..n).map(|idx| bp((idx as u32) * 23 + 9)).collect(),
                    bp(0),
                ),
                DenseMultilinearExtension::from_evaluations_vec(
                    3,
                    (0..n).map(|idx| bp(!((idx as u32) * 5 + 7))).collect(),
                    bp(0),
                ),
            ],
        ];

        let mut prover_data = Vec::new();
        let mut commitments = Vec::new();
        for polys in &instance_polys {
            let (data, commitment) =
                <HyraxPCS<C, BinaryLanes> as PCS<F, BinaryPoly<D>, D>>::commit(&ck, polys).unwrap();
            prover_data.push(data);
            commitments.push(commitment);
        }

        let theta = [F::from_with_cfg(3u64, &cfg), F::from_with_cfg(5u64, &cfg)];
        let folded_commitment =
            <HyraxPCS<C, BinaryLanes> as FoldablePCS<F, BinaryPoly<D>, D>>::fold_commitments(
                &commitments,
                &theta,
                &cfg,
            )
            .unwrap();
        let folded_data =
            <HyraxPCS<C, BinaryLanes> as FoldablePCS<F, BinaryPoly<D>, D>>::fold_prover_data(
                &prover_data,
                &theta,
                &cfg,
            )
            .unwrap();

        let theta_scalar = theta
            .iter()
            .map(|theta| <F as HyraxFieldBridge<C>>::field_to_scalar(theta).unwrap())
            .collect::<Vec<_>>();
        let mut scalar_lanes =
            vec![vec![vec![<C as AffineRepr>::ScalarField::zero(); n]; D]; instance_polys[0].len()];
        for (instance_idx, polys) in instance_polys.iter().enumerate() {
            for (poly_idx, poly) in polys.iter().enumerate() {
                for (eval_idx, eval) in poly.evaluations.iter().enumerate() {
                    for (lane, bit) in eval.iter().enumerate() {
                        if bit.inner() {
                            scalar_lanes[poly_idx][lane][eval_idx] += theta_scalar[instance_idx];
                        }
                    }
                }
            }
        }

        let point = [[0x11u8; 64], [0x22u8; 64], [0x33u8; 64]]
            .iter()
            .map(|bytes| {
                let scalar = <C as AffineRepr>::ScalarField::from_le_bytes_mod_order(bytes);
                <F as HyraxFieldBridge<C>>::scalar_to_field(&scalar, &cfg).unwrap()
            })
            .collect::<Vec<_>>();
        let eq = eq_tensor_f::<F>(&point, &cfg);
        let folded_lifted_evals = scalar_lanes
            .iter()
            .map(|lanes| {
                let coeffs = lanes
                    .iter()
                    .map(|values| {
                        values.iter().zip(eq.iter()).fold(
                            F::zero_with_cfg(&cfg),
                            |mut acc, (value, weight)| {
                                acc += <F as HyraxFieldBridge<C>>::scalar_to_field(value, &cfg)
                                    .unwrap()
                                    * weight;
                                acc
                            },
                        )
                    })
                    .collect::<Vec<_>>();
                DynamicPolynomialF::new_trimmed(coeffs)
            })
            .collect::<Vec<_>>();

        let mut prover_transcript = PcsProverTranscript {
            fs_transcript: Default::default(),
            stream: Default::default(),
        };
        <HyraxPCS<C, BinaryLanes> as PCS<F, BinaryPoly<D>, D>>::absorb_commitment(
            &mut prover_transcript.fs_transcript,
            &folded_commitment,
        );
        let mut transcription_buf = vec![0u8; <F as crypto_primitives::Field>::Inner::NUM_BYTES];
        for lifted_eval in &folded_lifted_evals {
            prover_transcript
                .fs_transcript
                .absorb_random_field_slice(&lifted_eval.coeffs, &mut transcription_buf);
        }
        HyraxPCS::<C, BinaryLanes>::prove_open_scalar_lanes::<F, true>(
            &mut prover_transcript,
            &ck,
            &scalar_lanes,
            &point,
            &folded_data,
            &cfg,
        )
        .unwrap();

        let mut verifier_transcript = prover_transcript.into_verification_transcript();
        <HyraxPCS<C, BinaryLanes> as PCS<F, BinaryPoly<D>, D>>::absorb_commitment(
            &mut verifier_transcript.fs_transcript,
            &folded_commitment,
        );
        let mut transcription_buf = vec![0u8; <F as crypto_primitives::Field>::Inner::NUM_BYTES];
        for lifted_eval in &folded_lifted_evals {
            verifier_transcript
                .fs_transcript
                .absorb_random_field_slice(&lifted_eval.coeffs, &mut transcription_buf);
        }
        <HyraxPCS<C, BinaryLanes> as PCS<F, BinaryPoly<D>, D>>::verify_open::<true>(
            &mut verifier_transcript,
            &vk,
            &folded_commitment,
            &point,
            &folded_lifted_evals,
            &Vec::new(),
            &cfg,
        )
        .unwrap();
    }

    #[test]
    fn hyrax_fold_rejects_commitment_shape_mismatch() {
        type C = ark_bn254::G1Affine;
        type F = MontyField<4>;
        const D: usize = 32;

        let cfg = cfg_from_curve::<C>();
        let generator = <C as AffineRepr>::Group::generator();
        let bases = (1..=4)
            .map(|idx| (generator * <C as AffineRepr>::ScalarField::from(idx as u64)).into_affine())
            .collect::<Vec<_>>();
        let h = generator * <C as AffineRepr>::ScalarField::from(5u64);
        let (ck, _) = HyraxPCS::<C, BinaryLanes>::setup_from_bases_with_blinding(
            4,
            bases,
            h,
            HyraxBlindingMode::Unblinded,
        )
        .unwrap();
        let polys_one = vec![DenseMultilinearExtension::from_evaluations_vec(
            2,
            vec![BinaryPoly::<D>::from(1u32); 4],
            BinaryPoly::<D>::from(0u32),
        )];
        let polys_two = vec![DenseMultilinearExtension::from_evaluations_vec(
            3,
            vec![BinaryPoly::<D>::from(2u32); 8],
            BinaryPoly::<D>::from(0u32),
        )];
        let (_, c0) =
            <HyraxPCS<C, BinaryLanes> as PCS<F, BinaryPoly<D>, D>>::commit(&ck, &polys_one)
                .unwrap();
        let (_, c1) =
            <HyraxPCS<C, BinaryLanes> as PCS<F, BinaryPoly<D>, D>>::commit(&ck, &polys_two)
                .unwrap();

        let theta = [F::from_with_cfg(1u64, &cfg), F::from_with_cfg(2u64, &cfg)];
        assert!(matches!(
            <HyraxPCS<C, BinaryLanes> as FoldablePCS<F, BinaryPoly<D>, D>>::fold_commitments(
                &[c0, c1],
                &theta,
                &cfg,
            ),
            Err(ZipError::InvalidPcsParam(_))
        ));
    }

    #[test]
    fn hyrax_rejects_empty_commitment_with_nonempty_lifted_evals() {
        type C = ark_bn254::G1Affine;
        type F = MontyField<4>;
        const D: usize = 32;

        let width = 8;
        let (_, vk) = HyraxPCS::<C, BinaryLanes>::setup(
            width,
            b"zinc-plus-hyrax-empty-reject-test",
            HyraxBlindingMode::Unblinded,
        )
        .unwrap();

        let commitment = HyraxCommitment::<C> {
            batch_size: 0,
            num_lanes: D,
            num_rows: 0,
            blinding_mode: HyraxBlindingMode::Unblinded,
            comm: Vec::new(),
        };
        let cfg = cfg_from_curve::<C>();
        let lifted_evals = vec![DynamicPolynomialF::new_trimmed(vec![F::zero_with_cfg(
            &cfg,
        )])];
        let mut verifier_transcript = PcsVerifierTranscript {
            fs_transcript: Default::default(),
            stream: Default::default(),
        };

        let result = <HyraxPCS<C, BinaryLanes> as PCS<F, BinaryPoly<D>, D>>::verify_open::<true>(
            &mut verifier_transcript,
            &vk,
            &commitment,
            &[],
            &lifted_evals,
            &Vec::new(),
            &cfg,
        );
        assert!(matches!(result, Err(ZipError::InvalidPcsParam(_))));
    }

    #[test]
    fn hyrax_rejects_noncanonical_empty_commitment() {
        type C = ark_bn254::G1Affine;
        type F = MontyField<4>;
        const D: usize = 32;

        let width = 8;
        let (_, vk) = HyraxPCS::<C, BinaryLanes>::setup(
            width,
            b"zinc-plus-hyrax-empty-reject-test-2",
            HyraxBlindingMode::Unblinded,
        )
        .unwrap();

        let commitment = HyraxCommitment::<C> {
            batch_size: 0,
            num_lanes: D,
            num_rows: 1,
            blinding_mode: HyraxBlindingMode::Unblinded,
            comm: Vec::new(),
        };
        let cfg = cfg_from_curve::<C>();
        let mut verifier_transcript = PcsVerifierTranscript {
            fs_transcript: Default::default(),
            stream: Default::default(),
        };

        let result = <HyraxPCS<C, BinaryLanes> as PCS<F, BinaryPoly<D>, D>>::verify_open::<true>(
            &mut verifier_transcript,
            &vk,
            &commitment,
            &[],
            &[],
            &Vec::new(),
            &cfg,
        );
        assert!(matches!(result, Err(ZipError::InvalidPcsParam(_))));
    }
}
