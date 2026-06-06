//! Production SHA ProjectionFold protocol helpers.
//!
//! This module is intentionally separate from the existing single-instance
//! `Proof`: production ProjectionFold has a different transcript order and
//! derives folded commitments only after SumFold fixes the instance-axis point.

use crate::{
    ZincTypes,
    pcs::{PCSCommitments, PCSProverData, ZincPCSTypes},
};
use crypto_primitives::{FromPrimitiveWithConfig, PrimeField};
use thiserror::Error;
use zinc_piop::{
    multipoint_eval::Proof as MultipointEvalProof,
    neutron_nova::SumFoldError,
    neutron_nova::{
        NUM_NONZERO_SHA_FAMILIES, SHA_ROW_VARS, ShaBooleanitySource, ShaProjectionError,
        ShaResidualFamily, ShaSumFoldOutput, ShaWordCol, build_folded_row_sumcheck_group,
        finalize_sha_sumfold, folded_row_integrand_sum, production_sha_nonzero_ideals,
        verify_folded_row_sumcheck_claim,
    },
    sumcheck::{
        SumCheckError,
        multi_degree::{MultiDegreeSumcheck, MultiDegreeSumcheckProof},
    },
};
use zinc_poly::{
    univariate::{
        binary::BinaryPoly, dense::DensePolynomial, dynamic::over_field::DynamicPolynomialF,
    },
    utils::{ArithErrors, build_eq_x_r_vec, eq_eval},
};
use zinc_transcript::traits::{ConstTranscribable, Transcribable, Transcript};
use zinc_uair::ideal::IdealCheck;
use zinc_utils::{
    delayed_reduction::DelayedFieldProductSum, inner_transparent_field::InnerTransparentField,
};
use zip_plus::{ZipError, pcs::generic::FoldablePCS};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProductionShaProof<F: PrimeField, Commitments> {
    pub instance_commitments: Vec<Commitments>,
    pub fresh_ideal_polys: Vec<[DynamicPolynomialF<F>; NUM_NONZERO_SHA_FAMILIES]>,
    pub sumfold_proof: MultiDegreeSumcheckProof<F>,
    pub folded_row_sumcheck: MultiDegreeSumcheckProof<F>,
    pub endpoint_evals: ShaEndpointEvals<F>,
    pub multipoint_eval: MultipointEvalProof<F>,
    pub folded_lifted_evals: Vec<DynamicPolynomialF<F>>,
    pub pcs_opening_bytes: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShaEndpointEvals<F> {
    pub sources: Vec<ShaSourceEndpointEval<F>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShaSourceEndpointEval<F> {
    pub col: ShaWordCol,
    pub shift: usize,
    pub scalarized: F,
    pub bits: [F; 32],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VirtualChMajEndpoint<F> {
    pub ch1: [F; 32],
    pub ch2: [F; 32],
    pub maj: [F; 32],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProductionShaChallenges<F> {
    pub r_ic: [F; SHA_ROW_VARS],
    pub a: F,
    pub lambda: F,
    pub rho: F,
    pub xi: F,
    pub beta: Vec<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FoldedRowSumcheckOutput<F> {
    pub r_star: Vec<F>,
    pub terminal_value: F,
}

#[derive(Debug, Error)]
pub enum ProductionShaError<F: PrimeField> {
    #[error("instance count must be a power of two, got {0}")]
    InstanceCountNotPowerOfTwo(usize),
    #[error("length mismatch for {label}: got {got}, expected {expected}")]
    LengthMismatch {
        label: &'static str,
        got: usize,
        expected: usize,
    },
    #[error("{label} expected exactly one sumcheck group, got {got}")]
    UnexpectedSumcheckGroupCount { label: &'static str, got: usize },
    #[error("SumFold proof has degree {degree}, expected at most 3")]
    SumFoldDegreeTooHigh { degree: usize },
    #[error("SumFold terminal evaluation mismatch")]
    SumFoldTerminalMismatch,
    #[error("row sumcheck proof has degree {degree}, expected at most 3")]
    RowSumcheckDegreeTooHigh { degree: usize },
    #[error("row sumcheck terminal evaluation mismatch")]
    RowSumcheckTerminalMismatch,
    #[error("endpoint scalarization mismatch for {col:?} shift {shift}")]
    EndpointScalarizationMismatch { col: ShaWordCol, shift: usize },
    #[error("missing endpoint eval for {col:?} shift {shift}")]
    MissingEndpointEval { col: ShaWordCol, shift: usize },
    #[error("ideal membership failed")]
    IdealMembership,
    #[error("PCS error: {0}")]
    Pcs(#[from] ZipError),
    #[error("sumcheck error: {0}")]
    Sumcheck(#[from] SumCheckError<F>),
    #[error("SumFold error: {0}")]
    SumFold(#[from] SumFoldError),
    #[error("SHA projection error: {0}")]
    ShaProjection(#[from] ShaProjectionError),
    #[error("equality polynomial error: {0}")]
    Eq(#[from] ArithErrors),
}

pub fn absorb_projected_sha_publics<F>(
    transcript: &mut impl Transcript,
    publics: &[zinc_piop::neutron_nova::ProjectedShaPublic<F>],
) where
    F: PrimeField,
    F::Inner: ConstTranscribable + Transcribable,
    F::Modulus: Transcribable,
{
    let mut buf = vec![0; F::Inner::NUM_BYTES];
    transcript.absorb_slice(b"production_sha_publics_begin");
    transcript.absorb_slice(&(publics.len() as u64).to_le_bytes());
    for (instance_idx, public) in publics.iter().enumerate() {
        transcript.absorb_slice(&(instance_idx as u64).to_le_bytes());
        transcript.absorb_slice(&(public.columns.columns.len() as u64).to_le_bytes());
        for (col_idx, col) in public.columns.columns.iter().enumerate() {
            transcript.absorb_slice(&(col_idx as u64).to_le_bytes());
            transcript.absorb_slice(&(col.len() as u64).to_le_bytes());
            transcript.absorb_random_field_slice(col, &mut buf);
        }
    }
    transcript.absorb_slice(b"production_sha_publics_end");
}

pub fn absorb_fresh_sha_ideal_polys<F>(
    transcript: &mut impl Transcript,
    ideal_polys: &[[DynamicPolynomialF<F>; NUM_NONZERO_SHA_FAMILIES]],
) where
    F: PrimeField,
    F::Inner: ConstTranscribable + Transcribable,
    F::Modulus: Transcribable,
{
    let mut buf = vec![0; F::Inner::NUM_BYTES];
    transcript.absorb_slice(b"production_sha_fresh_ideals_begin");
    transcript.absorb_slice(&(ideal_polys.len() as u64).to_le_bytes());
    for (instance_idx, instance) in ideal_polys.iter().enumerate() {
        transcript.absorb_slice(&(instance_idx as u64).to_le_bytes());
        transcript.absorb_slice(&(instance.len() as u64).to_le_bytes());
        for (family_idx, poly) in instance.iter().enumerate() {
            transcript.absorb_slice(&(family_idx as u64).to_le_bytes());
            transcript.absorb_slice(&(poly.coeffs.len() as u64).to_le_bytes());
            transcript.absorb_random_field_slice(&poly.coeffs, &mut buf);
        }
    }
    transcript.absorb_slice(b"production_sha_fresh_ideals_end");
}

pub fn sample_pre_ideal_challenge<F>(
    transcript: &mut impl Transcript,
    field_cfg: &F::Config,
) -> [F; SHA_ROW_VARS]
where
    F: PrimeField,
    F::Inner: ConstTranscribable,
{
    std::array::from_fn(|_| transcript.get_field_challenge(field_cfg))
}

pub fn sample_post_ideal_challenges<F>(
    transcript: &mut impl Transcript,
    instance_count: usize,
    field_cfg: &F::Config,
) -> Result<(F, F, F, F, Vec<F>), ProductionShaError<F>>
where
    F: PrimeField,
    F::Inner: ConstTranscribable,
{
    if !instance_count.is_power_of_two() {
        return Err(ProductionShaError::InstanceCountNotPowerOfTwo(
            instance_count,
        ));
    }
    let ell = usize::try_from(instance_count.trailing_zeros()).expect("ell fits usize");
    Ok((
        transcript.get_field_challenge(field_cfg),
        transcript.get_field_challenge(field_cfg),
        transcript.get_field_challenge(field_cfg),
        transcript.get_field_challenge(field_cfg),
        transcript.get_field_challenges(ell, field_cfg),
    ))
}

pub fn check_fresh_sha_ideal_membership<F>(
    ideal_polys: &[[DynamicPolynomialF<F>; NUM_NONZERO_SHA_FAMILIES]],
    field_cfg: &F::Config,
) -> Result<(), ProductionShaError<F>>
where
    F: PrimeField,
{
    let ideals = production_sha_nonzero_ideals(field_cfg);
    for values in ideal_polys {
        for (ideal, value) in ideals.iter().zip(values.iter()) {
            if !ideal
                .contains(value)
                .map_err(|_| ProductionShaError::IdealMembership)?
            {
                return Err(ProductionShaError::IdealMembership);
            }
        }
    }
    Ok(())
}

pub fn prove_sha_sumfold_targets<F>(
    transcript: &mut impl Transcript,
    fresh_targets: &[F],
    beta: &[F],
    prefix_vars: usize,
    field_cfg: &F::Config,
) -> Result<(MultiDegreeSumcheckProof<F>, ShaSumFoldOutput<F>), ProductionShaError<F>>
where
    F: InnerTransparentField
        + DelayedFieldProductSum
        + FromPrimitiveWithConfig
        + Send
        + Sync
        + 'static,
    F::Inner: ConstTranscribable + num_traits::Zero,
    F::Modulus: ConstTranscribable,
{
    let claims = zinc_piop::neutron_nova::LinearInstanceClaims::new(fresh_targets.to_vec())?;
    let group = claims.build_hybrid_sumcheck_group(beta, prefix_vars, field_cfg)?;
    let (proof, states) =
        MultiDegreeSumcheck::prove_as_subprotocol(transcript, vec![group], claims.ell(), field_cfg);
    let r_b = states
        .first()
        .ok_or(ProductionShaError::LengthMismatch {
            label: "sumfold states",
            got: 0,
            expected: 1,
        })?
        .randomness
        .clone();
    let c_sf = sumfold_expected_eval(beta, fresh_targets, &r_b, field_cfg)?;
    let output = finalize_sha_sumfold(beta, r_b, c_sf, fresh_targets.len(), field_cfg)?;
    Ok((proof, output))
}

pub fn verify_sha_sumfold_targets<F>(
    transcript: &mut impl Transcript,
    proof: &MultiDegreeSumcheckProof<F>,
    fresh_targets: &[F],
    beta: &[F],
    field_cfg: &F::Config,
) -> Result<ShaSumFoldOutput<F>, ProductionShaError<F>>
where
    F: InnerTransparentField
        + DelayedFieldProductSum
        + FromPrimitiveWithConfig
        + Send
        + Sync
        + 'static,
    F::Inner: ConstTranscribable + num_traits::Zero,
    F::Modulus: ConstTranscribable,
{
    require_single_sumcheck_group(proof, "SHA SumFold")?;
    for &degree in proof.degrees() {
        if degree > 3 {
            return Err(ProductionShaError::SumFoldDegreeTooHigh { degree });
        }
    }
    let claims = zinc_piop::neutron_nova::LinearInstanceClaims::new(fresh_targets.to_vec())?;
    let subclaims =
        MultiDegreeSumcheck::verify_as_subprotocol(transcript, claims.ell(), proof, field_cfg)?;
    let r_b = subclaims.point().to_vec();
    let c_sf = subclaims.expected_evaluations()[0].clone();
    if c_sf != sumfold_expected_eval(beta, fresh_targets, &r_b, field_cfg)? {
        return Err(ProductionShaError::SumFoldTerminalMismatch);
    }
    Ok(finalize_sha_sumfold(
        beta,
        r_b,
        c_sf,
        fresh_targets.len(),
        field_cfg,
    )?)
}

pub fn fold_pcs_commitments<P, Zt, F, const D: usize>(
    commitments: &[PCSCommitments<P, Zt, F, D>],
    theta: &[F],
    field_cfg: &F::Config,
) -> Result<PCSCommitments<P, Zt, F, D>, ProductionShaError<F>>
where
    Zt: ZincTypes<D>,
    F: PrimeField,
    P: ZincPCSTypes<Zt, F, D>,
    P::BinaryPCS: FoldablePCS<F, BinaryPoly<D>, D>,
    P::ArbitraryPCS: FoldablePCS<F, DensePolynomial<Zt::Int, D>, D>,
    P::IntPCS: FoldablePCS<F, Zt::Int, D>,
{
    if commitments.len() != theta.len() {
        return Err(ProductionShaError::LengthMismatch {
            label: "commitments/theta",
            got: commitments.len(),
            expected: theta.len(),
        });
    }
    let binary = commitments
        .iter()
        .map(|commitment| commitment.binary.clone())
        .collect::<Vec<_>>();
    let arbitrary = commitments
        .iter()
        .map(|commitment| commitment.arbitrary.clone())
        .collect::<Vec<_>>();
    let int = commitments
        .iter()
        .map(|commitment| commitment.int.clone())
        .collect::<Vec<_>>();
    Ok(PCSCommitments {
        binary: P::BinaryPCS::fold_commitments(&binary, theta, field_cfg)?,
        arbitrary: P::ArbitraryPCS::fold_commitments(&arbitrary, theta, field_cfg)?,
        int: P::IntPCS::fold_commitments(&int, theta, field_cfg)?,
    })
}

pub fn fold_pcs_prover_data<P, Zt, F, const D: usize>(
    prover_data: &[PCSProverData<P, Zt, F, D>],
    theta: &[F],
    field_cfg: &F::Config,
) -> Result<PCSProverData<P, Zt, F, D>, ProductionShaError<F>>
where
    Zt: ZincTypes<D>,
    F: PrimeField,
    P: ZincPCSTypes<Zt, F, D>,
    P::BinaryPCS: FoldablePCS<F, BinaryPoly<D>, D>,
    P::ArbitraryPCS: FoldablePCS<F, DensePolynomial<Zt::Int, D>, D>,
    P::IntPCS: FoldablePCS<F, Zt::Int, D>,
{
    if prover_data.len() != theta.len() {
        return Err(ProductionShaError::LengthMismatch {
            label: "prover_data/theta",
            got: prover_data.len(),
            expected: theta.len(),
        });
    }
    let binary = prover_data
        .iter()
        .map(|data| data.binary.clone())
        .collect::<Vec<_>>();
    let arbitrary = prover_data
        .iter()
        .map(|data| data.arbitrary.clone())
        .collect::<Vec<_>>();
    let int = prover_data
        .iter()
        .map(|data| data.int.clone())
        .collect::<Vec<_>>();
    Ok(PCSProverData {
        binary: P::BinaryPCS::fold_prover_data(&binary, theta, field_cfg)?,
        arbitrary: P::ArbitraryPCS::fold_prover_data(&arbitrary, theta, field_cfg)?,
        int: P::IntPCS::fold_prover_data(&int, theta, field_cfg)?,
    })
}

pub fn prove_folded_row_sumcheck<F>(
    transcript: &mut impl Transcript,
    row_integrand_values: &[F],
    t_prime: &F,
    field_cfg: &F::Config,
) -> Result<MultiDegreeSumcheckProof<F>, ProductionShaError<F>>
where
    F: InnerTransparentField
        + DelayedFieldProductSum
        + FromPrimitiveWithConfig
        + Send
        + Sync
        + 'static,
    F::Inner: ConstTranscribable + num_traits::Zero,
    F::Modulus: ConstTranscribable,
{
    let claimed = folded_row_integrand_sum(row_integrand_values, field_cfg)?;
    verify_folded_row_sumcheck_claim(&claimed, t_prime)?;
    let group = build_folded_row_sumcheck_group(row_integrand_values, field_cfg)?;
    let (proof, _) =
        MultiDegreeSumcheck::prove_as_subprotocol(transcript, vec![group], SHA_ROW_VARS, field_cfg);
    Ok(proof)
}

pub fn verify_folded_row_sumcheck<F>(
    transcript: &mut impl Transcript,
    proof: &MultiDegreeSumcheckProof<F>,
    t_prime: &F,
    field_cfg: &F::Config,
) -> Result<FoldedRowSumcheckOutput<F>, ProductionShaError<F>>
where
    F: InnerTransparentField
        + DelayedFieldProductSum
        + FromPrimitiveWithConfig
        + Send
        + Sync
        + 'static,
    F::Inner: ConstTranscribable + num_traits::Zero,
    F::Modulus: ConstTranscribable,
{
    require_single_sumcheck_group(proof, "folded row sumcheck")?;
    for &degree in proof.degrees() {
        if degree > 3 {
            return Err(ProductionShaError::RowSumcheckDegreeTooHigh { degree });
        }
    }
    let Some(claimed_sum) = proof.claimed_sums().first() else {
        return Err(ProductionShaError::LengthMismatch {
            label: "folded row claimed sums",
            got: 0,
            expected: 1,
        });
    };
    verify_folded_row_sumcheck_claim(claimed_sum, t_prime)?;
    let subclaims =
        MultiDegreeSumcheck::verify_as_subprotocol(transcript, SHA_ROW_VARS, proof, field_cfg)?;
    Ok(FoldedRowSumcheckOutput {
        r_star: subclaims.point().to_vec(),
        terminal_value: subclaims.expected_evaluations()[0].clone(),
    })
}

pub fn verify_folded_row_terminal_value<F>(
    output: &FoldedRowSumcheckOutput<F>,
    reconstructed_terminal: &F,
) -> Result<(), ProductionShaError<F>>
where
    F: PrimeField,
{
    if &output.terminal_value != reconstructed_terminal {
        return Err(ProductionShaError::RowSumcheckTerminalMismatch);
    }
    Ok(())
}

pub fn verify_endpoint_scalarization<F>(
    endpoint_evals: &ShaEndpointEvals<F>,
    a: &F,
    field_cfg: &F::Config,
) -> Result<(), ProductionShaError<F>>
where
    F: PrimeField,
{
    let powers = zinc_utils::powers(a.clone(), F::one_with_cfg(field_cfg), 32);
    for source in &endpoint_evals.sources {
        let recombined = source
            .bits
            .iter()
            .zip(powers.iter())
            .fold(F::zero_with_cfg(field_cfg), |acc, (bit, power)| {
                acc + bit.clone() * power
            });
        if recombined != source.scalarized {
            return Err(ProductionShaError::EndpointScalarizationMismatch {
                col: source.col,
                shift: source.shift,
            });
        }
    }
    Ok(())
}

pub fn reconstruct_virtual_ch_maj_endpoint<F>(
    endpoint_evals: &ShaEndpointEvals<F>,
    field_cfg: &F::Config,
) -> Result<VirtualChMajEndpoint<F>, ProductionShaError<F>>
where
    F: PrimeField,
{
    let two = F::one_with_cfg(field_cfg) + F::one_with_cfg(field_cfg);
    let bits = |col, shift| source_bits(endpoint_evals, col, shift);
    let e0 = bits(ShaWordCol::E, 0)?;
    let e1 = bits(ShaWordCol::E, 1)?;
    let e2 = bits(ShaWordCol::E, 2)?;
    let a0 = bits(ShaWordCol::A, 0)?;
    let a1 = bits(ShaWordCol::A, 1)?;
    let a2 = bits(ShaWordCol::A, 2)?;
    let uef2 = bits(ShaWordCol::Uef, 2)?;
    let uneg_eg2 = bits(ShaWordCol::UNegEg, 2)?;
    let ch2_comp0 = bits(ShaWordCol::Ch2Comp, 0)?;
    let maj2 = bits(ShaWordCol::Maj, 2)?;
    let maj_comp0 = bits(ShaWordCol::MajComp, 0)?;

    Ok(VirtualChMajEndpoint {
        ch1: std::array::from_fn(|idx| e2[idx].clone() + &e1[idx] - two.clone() * &uef2[idx]),
        ch2: std::array::from_fn(|idx| {
            e2[idx].clone() - &e0[idx]
                + two.clone() * &uneg_eg2[idx]
                + two.clone() * &ch2_comp0[idx]
        }),
        maj: std::array::from_fn(|idx| {
            a0[idx].clone() + &a1[idx] + &a2[idx]
                - two.clone() * &maj2[idx]
                - two.clone() * &maj_comp0[idx]
        }),
    })
}

pub fn booleanity_endpoint_value<F>(
    endpoint_evals: &ShaEndpointEvals<F>,
    source: &ShaBooleanitySource,
    field_cfg: &F::Config,
) -> Result<F, ProductionShaError<F>>
where
    F: PrimeField,
{
    match source {
        ShaBooleanitySource::WordBit { col, bit } => Ok(source_bits(endpoint_evals, *col, 0)?
            .get(*bit)
            .cloned()
            .ok_or(ProductionShaError::LengthMismatch {
                label: "endpoint bit",
                got: *bit,
                expected: 32,
            })?),
        ShaBooleanitySource::VirtualCh1 { bit } => Ok(reconstruct_virtual_ch_maj_endpoint(
            endpoint_evals,
            field_cfg,
        )?
        .ch1
        .get(*bit)
        .cloned()
        .ok_or(ProductionShaError::LengthMismatch {
            label: "virtual Ch1 bit",
            got: *bit,
            expected: 32,
        })?),
        ShaBooleanitySource::VirtualCh2 { bit } => Ok(reconstruct_virtual_ch_maj_endpoint(
            endpoint_evals,
            field_cfg,
        )?
        .ch2
        .get(*bit)
        .cloned()
        .ok_or(ProductionShaError::LengthMismatch {
            label: "virtual Ch2 bit",
            got: *bit,
            expected: 32,
        })?),
        ShaBooleanitySource::VirtualMaj { bit } => Ok(reconstruct_virtual_ch_maj_endpoint(
            endpoint_evals,
            field_cfg,
        )?
        .maj
        .get(*bit)
        .cloned()
        .ok_or(ProductionShaError::LengthMismatch {
            label: "virtual Maj bit",
            got: *bit,
            expected: 32,
        })?),
    }
}

fn source_bits<F>(
    endpoint_evals: &ShaEndpointEvals<F>,
    col: ShaWordCol,
    shift: usize,
) -> Result<&[F; 32], ProductionShaError<F>>
where
    F: PrimeField,
{
    endpoint_evals
        .sources
        .iter()
        .find(|source| source.col == col && source.shift == shift)
        .map(|source| &source.bits)
        .ok_or(ProductionShaError::MissingEndpointEval { col, shift })
}

fn sumfold_expected_eval<F>(
    beta: &[F],
    fresh_targets: &[F],
    r_b: &[F],
    field_cfg: &F::Config,
) -> Result<F, ProductionShaError<F>>
where
    F: PrimeField,
{
    let d = eq_eval(beta, r_b, F::one_with_cfg(field_cfg))?;
    let weights = build_eq_x_r_vec(r_b, field_cfg)?;
    if weights.len() != fresh_targets.len() {
        return Err(ProductionShaError::LengthMismatch {
            label: "sumfold target weights",
            got: weights.len(),
            expected: fresh_targets.len(),
        });
    }
    let claim_at_r = weights
        .iter()
        .zip(fresh_targets)
        .fold(F::zero_with_cfg(field_cfg), |acc, (weight, target)| {
            acc + weight.clone() * target
        });
    Ok(d * claim_at_r)
}

fn require_single_sumcheck_group<F>(
    proof: &MultiDegreeSumcheckProof<F>,
    label: &'static str,
) -> Result<(), ProductionShaError<F>>
where
    F: PrimeField,
{
    let group_count = proof.degrees().len();
    if group_count != 1 {
        return Err(ProductionShaError::UnexpectedSumcheckGroupCount {
            label,
            got: group_count,
        });
    }
    let claimed_count = proof.claimed_sums().len();
    if claimed_count != 1 {
        return Err(ProductionShaError::LengthMismatch {
            label: "sumcheck claimed sums",
            got: claimed_count,
            expected: 1,
        });
    }
    Ok(())
}

#[allow(dead_code)]
fn family_weight_index(family: ShaResidualFamily) -> usize {
    family.index()
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::fixed_prime;
    use crypto_primitives::{
        FromWithConfig, crypto_bigint_monty::MontyField, crypto_bigint_uint::Uint,
    };
    use zinc_piop::neutron_nova::SHA_WORD_BITS;
    use zinc_transcript::{Blake3Transcript, traits::Transcript};

    type F = MontyField<4>;

    fn cfg() -> <F as PrimeField>::Config {
        fixed_prime::secp256k1_field_cfg::<F, Uint<4>>()
    }

    fn f(value: u64) -> F {
        F::from_with_cfg(value, &cfg())
    }

    fn endpoint_source(col: ShaWordCol, shift: usize, seed: u64) -> ShaSourceEndpointEval<F> {
        let field_cfg = cfg();
        let bits = std::array::from_fn(|idx| f(seed + idx as u64 + 1));
        let powers = zinc_utils::powers(f(7), F::one_with_cfg(&field_cfg), 32);
        let scalarized = bits
            .iter()
            .zip(powers.iter())
            .fold(F::zero_with_cfg(&field_cfg), |acc, (bit, power)| {
                acc + bit.clone() * power
            });
        ShaSourceEndpointEval {
            col,
            shift,
            scalarized,
            bits,
        }
    }

    fn endpoint_evals_for_virtuals() -> ShaEndpointEvals<F> {
        ShaEndpointEvals {
            sources: vec![
                endpoint_source(ShaWordCol::E, 0, 10),
                endpoint_source(ShaWordCol::E, 1, 20),
                endpoint_source(ShaWordCol::E, 2, 30),
                endpoint_source(ShaWordCol::A, 0, 40),
                endpoint_source(ShaWordCol::A, 1, 50),
                endpoint_source(ShaWordCol::A, 2, 60),
                endpoint_source(ShaWordCol::Uef, 2, 70),
                endpoint_source(ShaWordCol::UNegEg, 2, 80),
                endpoint_source(ShaWordCol::Ch2Comp, 0, 90),
                endpoint_source(ShaWordCol::Maj, 2, 100),
                endpoint_source(ShaWordCol::MajComp, 0, 110),
            ],
        }
    }

    #[test]
    fn fresh_ideal_coefficients_are_bound_before_a() {
        let field_cfg = cfg();
        let ideals = vec![std::array::from_fn(|idx| {
            DynamicPolynomialF::new_trimmed(vec![f(idx as u64 + 1), f(99)])
        })];
        let mut tampered = ideals.clone();
        tampered[0][0].coeffs[0] += f(1);

        let sample_a = |values: &[[DynamicPolynomialF<F>; NUM_NONZERO_SHA_FAMILIES]]| {
            let mut transcript = Blake3Transcript::new();
            transcript.absorb_slice(b"fresh-commitments-and-public-inputs");
            let _r_ic = sample_pre_ideal_challenge::<F>(&mut transcript, &field_cfg);
            absorb_fresh_sha_ideal_polys(&mut transcript, values);
            let (a, _, _, _, _) =
                sample_post_ideal_challenges::<F>(&mut transcript, 1, &field_cfg).unwrap();
            a
        };

        assert_ne!(sample_a(&ideals), sample_a(&tampered));
    }

    #[test]
    fn fresh_ideal_absorption_binds_polynomial_slot_structure() {
        let field_cfg = cfg();
        let mut packed = vec![std::array::from_fn(|_| DynamicPolynomialF::new(Vec::<F>::new()))];
        packed[0][0] = DynamicPolynomialF::new(vec![f(1), f(2)]);

        let mut split = vec![std::array::from_fn(|_| DynamicPolynomialF::new(Vec::<F>::new()))];
        split[0][0] = DynamicPolynomialF::new(vec![f(1)]);
        split[0][1] = DynamicPolynomialF::new(vec![f(2)]);

        let sample_a = |values: &[[DynamicPolynomialF<F>; NUM_NONZERO_SHA_FAMILIES]]| {
            let mut transcript = Blake3Transcript::new();
            transcript.absorb_slice(b"fresh-commitments-and-public-inputs");
            let _r_ic = sample_pre_ideal_challenge::<F>(&mut transcript, &field_cfg);
            absorb_fresh_sha_ideal_polys(&mut transcript, values);
            let (a, _, _, _, _) =
                sample_post_ideal_challenges::<F>(&mut transcript, 1, &field_cfg).unwrap();
            a
        };

        assert_ne!(sample_a(&packed), sample_a(&split));
    }

    #[test]
    fn sumfold_outputs_instance_fold_point_before_theta() {
        let field_cfg = cfg();
        let fresh_targets = vec![f(2), f(5), f(7), f(11)];
        let beta = vec![f(13), f(17)];

        let mut prover_transcript = Blake3Transcript::new();
        prover_transcript.absorb_slice(b"bound-before-sumfold");
        let (proof, prover_output) =
            prove_sha_sumfold_targets(&mut prover_transcript, &fresh_targets, &beta, 1, &field_cfg)
                .unwrap();

        let mut verifier_transcript = Blake3Transcript::new();
        verifier_transcript.absorb_slice(b"bound-before-sumfold");
        let verifier_output = verify_sha_sumfold_targets(
            &mut verifier_transcript,
            &proof,
            &fresh_targets,
            &beta,
            &field_cfg,
        )
        .unwrap();

        assert_eq!(verifier_output, prover_output);
        assert_eq!(
            prover_output.theta(),
            build_eq_x_r_vec(prover_output.r_b(), &field_cfg).unwrap()
        );
        assert_eq!(prover_output.theta().len(), fresh_targets.len());

        let d = eq_eval(&beta, prover_output.r_b(), F::one_with_cfg(&field_cfg)).unwrap();
        assert_eq!(prover_output.c_sf(), &(d * prover_output.t_prime()));

        let mut bad_targets = fresh_targets;
        bad_targets[0] += f(1);
        let mut bad_transcript = Blake3Transcript::new();
        bad_transcript.absorb_slice(b"bound-before-sumfold");
        assert!(
            verify_sha_sumfold_targets(
                &mut bad_transcript,
                &proof,
                &bad_targets,
                &beta,
                &field_cfg
            )
            .is_err()
        );
    }

    #[test]
    fn sumfold_verifier_rejects_extra_groups() {
        let field_cfg = cfg();
        let fresh_targets = vec![f(2), f(5), f(7), f(11)];
        let beta = vec![f(13), f(17)];
        let claims =
            zinc_piop::neutron_nova::LinearInstanceClaims::new(fresh_targets.clone()).unwrap();
        let group_0 = claims
            .build_hybrid_sumcheck_group(&beta, 1, &field_cfg)
            .unwrap();
        let group_1 = claims
            .build_hybrid_sumcheck_group(&beta, 1, &field_cfg)
            .unwrap();

        let mut prover_transcript = Blake3Transcript::new();
        prover_transcript.absorb_slice(b"bound-before-sumfold");
        let (proof, _) = MultiDegreeSumcheck::prove_as_subprotocol(
            &mut prover_transcript,
            vec![group_0, group_1],
            claims.ell(),
            &field_cfg,
        );

        let mut verifier_transcript = Blake3Transcript::new();
        verifier_transcript.absorb_slice(b"bound-before-sumfold");
        assert!(matches!(
            verify_sha_sumfold_targets(
                &mut verifier_transcript,
                &proof,
                &fresh_targets,
                &beta,
                &field_cfg,
            ),
            Err(ProductionShaError::UnexpectedSumcheckGroupCount {
                label: "SHA SumFold",
                got: 2
            })
        ));
    }

    #[test]
    fn folded_row_sumcheck_claim_is_t_prime() {
        let field_cfg = cfg();
        let row_integrand_values = (0..(1usize << SHA_ROW_VARS))
            .map(|idx| f((idx as u64).wrapping_mul(3) + 1))
            .collect::<Vec<_>>();
        let t_prime = folded_row_integrand_sum(&row_integrand_values, &field_cfg).unwrap();

        let mut prover_transcript = Blake3Transcript::new();
        prover_transcript.absorb_slice(b"folded-row-context");
        let proof = prove_folded_row_sumcheck(
            &mut prover_transcript,
            &row_integrand_values,
            &t_prime,
            &field_cfg,
        )
        .unwrap();
        assert_eq!(proof.claimed_sums(), &[t_prime.clone()]);

        let mut verifier_transcript = Blake3Transcript::new();
        verifier_transcript.absorb_slice(b"folded-row-context");
        let output =
            verify_folded_row_sumcheck(&mut verifier_transcript, &proof, &t_prime, &field_cfg)
                .unwrap();
        assert_eq!(output.r_star.len(), SHA_ROW_VARS);

        let row_weights = build_eq_x_r_vec(&output.r_star, &field_cfg).unwrap();
        let terminal = row_weights
            .iter()
            .zip(row_integrand_values.iter())
            .fold(F::zero_with_cfg(&field_cfg), |acc, (weight, value)| {
                acc + weight.clone() * value
            });
        verify_folded_row_terminal_value(&output, &terminal).unwrap();

        let mut bad_terminal = terminal;
        bad_terminal += f(1);
        assert!(verify_folded_row_terminal_value(&output, &bad_terminal).is_err());

        let mut bad_t = t_prime;
        bad_t += f(1);
        let mut bad_transcript = Blake3Transcript::new();
        bad_transcript.absorb_slice(b"folded-row-context");
        assert!(
            verify_folded_row_sumcheck(&mut bad_transcript, &proof, &bad_t, &field_cfg).is_err()
        );
    }

    #[test]
    fn folded_row_verifier_rejects_extra_groups() {
        let field_cfg = cfg();
        let row_integrand_values = (0..(1usize << SHA_ROW_VARS))
            .map(|idx| f((idx as u64).wrapping_mul(5) + 9))
            .collect::<Vec<_>>();
        let t_prime = folded_row_integrand_sum(&row_integrand_values, &field_cfg).unwrap();
        let group_0 = build_folded_row_sumcheck_group(&row_integrand_values, &field_cfg).unwrap();
        let group_1 = build_folded_row_sumcheck_group(&row_integrand_values, &field_cfg).unwrap();

        let mut prover_transcript = Blake3Transcript::new();
        prover_transcript.absorb_slice(b"folded-row-context");
        let (proof, _) = MultiDegreeSumcheck::prove_as_subprotocol(
            &mut prover_transcript,
            vec![group_0, group_1],
            SHA_ROW_VARS,
            &field_cfg,
        );

        let mut verifier_transcript = Blake3Transcript::new();
        verifier_transcript.absorb_slice(b"folded-row-context");
        assert!(matches!(
            verify_folded_row_sumcheck(&mut verifier_transcript, &proof, &t_prime, &field_cfg),
            Err(ProductionShaError::UnexpectedSumcheckGroupCount {
                label: "folded row sumcheck",
                got: 2
            })
        ));
    }

    #[test]
    fn scalarization_and_virtual_endpoints_use_source_bits_only() {
        let field_cfg = cfg();
        let mut endpoint_evals = endpoint_evals_for_virtuals();
        verify_endpoint_scalarization(&endpoint_evals, &f(7), &field_cfg).unwrap();

        let virtuals = reconstruct_virtual_ch_maj_endpoint(&endpoint_evals, &field_cfg).unwrap();
        let two = f(2);
        for bit in 0..SHA_WORD_BITS {
            let e0 = source_bits(&endpoint_evals, ShaWordCol::E, 0).unwrap()[bit].clone();
            let e1 = source_bits(&endpoint_evals, ShaWordCol::E, 1).unwrap()[bit].clone();
            let e2 = source_bits(&endpoint_evals, ShaWordCol::E, 2).unwrap()[bit].clone();
            let a0 = source_bits(&endpoint_evals, ShaWordCol::A, 0).unwrap()[bit].clone();
            let a1 = source_bits(&endpoint_evals, ShaWordCol::A, 1).unwrap()[bit].clone();
            let a2 = source_bits(&endpoint_evals, ShaWordCol::A, 2).unwrap()[bit].clone();
            let uef2 = source_bits(&endpoint_evals, ShaWordCol::Uef, 2).unwrap()[bit].clone();
            let uneg_eg2 =
                source_bits(&endpoint_evals, ShaWordCol::UNegEg, 2).unwrap()[bit].clone();
            let ch2_comp0 =
                source_bits(&endpoint_evals, ShaWordCol::Ch2Comp, 0).unwrap()[bit].clone();
            let maj2 = source_bits(&endpoint_evals, ShaWordCol::Maj, 2).unwrap()[bit].clone();
            let maj_comp0 =
                source_bits(&endpoint_evals, ShaWordCol::MajComp, 0).unwrap()[bit].clone();

            assert_eq!(virtuals.ch1[bit], e2.clone() + e1 - two.clone() * uef2);
            assert_eq!(
                virtuals.ch2[bit],
                e2 - e0 + two.clone() * uneg_eg2 + two.clone() * ch2_comp0
            );
            assert_eq!(
                virtuals.maj[bit],
                a0 + a1 + a2 - two.clone() * maj2 - two.clone() * maj_comp0
            );
        }

        endpoint_evals.sources[0].scalarized += f(1);
        assert!(verify_endpoint_scalarization(&endpoint_evals, &f(7), &field_cfg).is_err());
    }
}
