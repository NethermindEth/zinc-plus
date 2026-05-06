//! Ideal-check subprotocol.
mod batched_ideal_check;
/// Public to let the protocol crate's dual-prime Z-branch helper
/// reach `compute_combined_polynomials` directly with a custom skip
/// mask, bypassing the unmasked `IdealCheckProtocol` trait wrapper.
pub mod combined_poly_builder;
mod structs;

pub use structs::*;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::projections::{ColumnMajorTrace, RowMajorTrace, ScalarMap};
use batched_ideal_check::*;
use crypto_primitives::PrimeField;
use num_traits::ConstZero;
use thiserror::Error;
use zinc_poly::{
    EvaluationError,
    univariate::dynamic::over_field::DynamicPolynomialF,
    utils::{ArithErrors as PolyArithErrors, build_eq_x_r_vec},
};
use zinc_transcript::traits::{ConstTranscribable, Transcript};
use zinc_uair::{
    Uair,
    degree_counter::linear_constraint_mask,
    ideal::{Ideal, IdealCheck},
    ConstraintRing,
    ideal_collector::{IdealOrZero, collect_ideals},
};
use zinc_utils::{cfg_into_iter, inner_transparent_field::InnerTransparentField};

/// Ideal-check subprotocol.
pub trait IdealCheckProtocol: Uair {
    /// Prover for linear-only UAIRs using MLE-first evaluation.
    ///
    /// Uses column-indexed trace for efficient MLE evaluation:
    /// evaluates trace columns at the challenge point first,
    /// then applies constraints to the evaluated values.
    ///
    /// # Parameters
    /// - `transcript`: the Fiat-Shamir transcript.
    /// - `trace_matrix`: input trace for the UAIR `U` projected to
    ///   `DynamicPolynomialF<F>`, column-indexed: `trace_matrix[col][row]`.
    /// - `projected_scalars`: UAIR scalars projected to
    ///   `DynamicPolynomialF<F>`.
    /// - `num_constraints`: number of constraints this UAIR encodes.
    /// - `num_vars`: number of variables in trace MLEs.
    /// - `field_cfg`: random field configuration sampled on the previous steps
    ///   of the overall protocol.
    #[allow(clippy::type_complexity)]
    fn prove_linear<F>(
        transcript: &mut impl Transcript,
        trace_matrix: &ColumnMajorTrace<F>,
        projected_scalars: &ScalarMap<Self::Scalar, DynamicPolynomialF<F>>,
        num_constraints: usize,
        num_vars: usize,
        field_cfg: &F::Config,
    ) -> Result<(Proof<F>, ProverState<F>), IdealCheckError<F, Self::Ideal>>
    where
        F: InnerTransparentField,
        F::Inner: ConstTranscribable,
        F::Modulus: ConstTranscribable;

    /// Prover for any UAIR using combined polynomial construction.
    ///
    /// Uses row-indexed (transposed) trace for efficient row-by-row
    /// combined polynomial construction.
    ///
    /// # Parameters
    /// - `transcript`: the Fiat-Shamir transcript.
    /// - `trace_matrix`: input trace for the UAIR `U` projected to
    ///   `DynamicPolynomialF<F>`, row-indexed: `trace_matrix[row][col]`.
    /// - `projected_scalars`: UAIR scalars projected to
    ///   `DynamicPolynomialF<F>`.
    /// - `num_constraints`: number of constraints this UAIR encodes.
    /// - `num_vars`: number of variables in trace MLEs.
    /// - `field_cfg`: random field configuration sampled on the previous steps
    ///   of the overall protocol.
    #[allow(clippy::type_complexity)]
    fn prove_combined<F>(
        transcript: &mut impl Transcript,
        trace_matrix: &RowMajorTrace<F>,
        projected_scalars: &ScalarMap<Self::Scalar, DynamicPolynomialF<F>>,
        num_constraints: usize,
        num_vars: usize,
        field_cfg: &F::Config,
    ) -> Result<(Proof<F>, ProverState<F>), IdealCheckError<F, Self::Ideal>>
    where
        F: InnerTransparentField,
        F::Inner: ConstTranscribable,
        F::Modulus: ConstTranscribable;

    /// Prover for mixed-degree UAIRs: routes each constraint through the
    /// MLE-first lane (degree ≤ 1, non-zero-ideal) or the combined-poly
    /// lane (degree > 1, non-zero-ideal). Zero-ideal constraints are
    /// substituted with `ZERO` as in the other paths.
    ///
    /// Both lanes evaluate against the same Fiat-Shamir-sampled IC point,
    /// so the merged per-constraint values are mathematically identical to
    /// what the all-Combined path would produce — the verifier sees the
    /// same `Proof` shape and runs unchanged.
    ///
    /// Takes both layouts because the MLE-first lane requires column-major
    /// while the combined-poly lane requires row-major. Step 1 of the
    /// protocol projects the trace twice for this path.
    ///
    /// # Parameters
    /// - `transcript`: the Fiat-Shamir transcript.
    /// - `row_major_trace` / `column_major_trace`: same projected trace in
    ///   both layouts.
    /// - `projected_scalars`: UAIR scalars projected to
    ///   `DynamicPolynomialF<F>`.
    /// - `num_constraints`: number of constraints this UAIR encodes.
    /// - `num_vars`: number of variables in trace MLEs.
    /// - `field_cfg`: random field configuration sampled on the previous
    ///   steps of the overall protocol.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn prove_hybrid<F>(
        transcript: &mut impl Transcript,
        row_major_trace: &RowMajorTrace<F>,
        column_major_trace: &ColumnMajorTrace<F>,
        projected_scalars: &ScalarMap<Self::Scalar, DynamicPolynomialF<F>>,
        num_constraints: usize,
        num_vars: usize,
        field_cfg: &F::Config,
    ) -> Result<(Proof<F>, ProverState<F>), IdealCheckError<F, Self::Ideal>>
    where
        F: InnerTransparentField,
        F::Inner: ConstTranscribable,
        F::Modulus: ConstTranscribable;

    /// The verifier part of the ideal-check subprotocol.
    ///
    /// The verifier samples a random field element
    /// the same way the prover sampled a random field
    /// element for projecting coefficients but disregards it
    /// as the verifier does not need to project anything.
    /// Then it computes the ideals encoded by the UAIR `U`,
    /// samples a random evaluation point and receives
    /// the evaluations of the combined polynomials sent by the prover
    /// and checks they belong to the corresponding ideals defined
    /// by the UAIR `U`.
    ///
    /// # Parameters
    /// - `transcript`: the Fiat-Shamir transcript.
    /// - `proof`: a purported proof produced by the prover.
    /// - `num_constraints`: the number of constraints the UAIR `U` encodes.
    /// - `num_vars`: the number of variables the trace row MLEs have.
    /// - `ideal_over_f_from_ref`: since the UAIR `U` is not aware of the field
    ///   the ideal check is operating on it defines ideals over the ring
    ///   `IcTypes::Witness`. `ideal_over_f_from_ref` allows to convert the
    ///   ideals over `IcTypes::Witness` into ideals over the field
    ///   `IcTypes::F`. Think of this as a projection for ideals.
    /// - `field_cfg`: random field configuration sampled on the previous steps
    ///   of the overall protocol.
    #[allow(clippy::type_complexity)]
    fn verify_as_subprotocol<F, IdealOverF, IdealOverFFromRef>(
        transcript: &mut impl Transcript,
        proof: Proof<F>,
        num_constraints: usize,
        num_vars: usize,
        ideal_over_f_from_ref: IdealOverFFromRef,
        field_cfg: &F::Config,
    ) -> Result<VerifierSubclaim<F>, IdealCheckError<F, IdealOverF>>
    where
        F: InnerTransparentField,
        F::Inner: ConstTranscribable,
        F::Modulus: ConstTranscribable,
        IdealOverF: Ideal + IdealCheck<DynamicPolynomialF<F>>,
        IdealOverFFromRef: Fn(&IdealOrZero<Self::Ideal>) -> IdealOverF;

    // -----------------------------------------------------------------
    // Dual-prime variants
    // -----------------------------------------------------------------
    //
    // Each `_typed` method accepts a `tag_filter: ConstraintRing` and
    // restricts the ideal check to the matching branch. Slots whose
    // recorded tag (from `IdealCollector::tags`) differs from the filter
    // are forced to `ZERO` in the combined polynomial values BEFORE
    // they are absorbed into the Fiat-Shamir transcript, and the
    // verifier rejects the proof if any off-branch slot carries a
    // non-zero value. Zero-ideal slots (`assert_zero`) are filtered
    // alongside the off-branch slots — they were already discarded by
    // the single-prime variants.

    /// Tag-filtered counterpart of [`Self::prove_linear`].
    #[allow(clippy::type_complexity)]
    fn prove_linear_typed<F>(
        transcript: &mut impl Transcript,
        trace_matrix: &ColumnMajorTrace<F>,
        projected_scalars: &ScalarMap<Self::Scalar, DynamicPolynomialF<F>>,
        num_constraints: usize,
        num_vars: usize,
        field_cfg: &F::Config,
        tag_filter: ConstraintRing,
    ) -> Result<(Proof<F>, ProverState<F>), IdealCheckError<F, Self::Ideal>>
    where
        F: InnerTransparentField,
        F::Inner: ConstTranscribable,
        F::Modulus: ConstTranscribable;

    /// Tag-filtered counterpart of [`Self::prove_combined`].
    #[allow(clippy::type_complexity)]
    fn prove_combined_typed<F>(
        transcript: &mut impl Transcript,
        trace_matrix: &RowMajorTrace<F>,
        projected_scalars: &ScalarMap<Self::Scalar, DynamicPolynomialF<F>>,
        num_constraints: usize,
        num_vars: usize,
        field_cfg: &F::Config,
        tag_filter: ConstraintRing,
    ) -> Result<(Proof<F>, ProverState<F>), IdealCheckError<F, Self::Ideal>>
    where
        F: InnerTransparentField,
        F::Inner: ConstTranscribable,
        F::Modulus: ConstTranscribable;

    /// Tag-filtered counterpart of [`Self::prove_hybrid`].
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn prove_hybrid_typed<F>(
        transcript: &mut impl Transcript,
        row_major_trace: &RowMajorTrace<F>,
        column_major_trace: &ColumnMajorTrace<F>,
        projected_scalars: &ScalarMap<Self::Scalar, DynamicPolynomialF<F>>,
        num_constraints: usize,
        num_vars: usize,
        field_cfg: &F::Config,
        tag_filter: ConstraintRing,
    ) -> Result<(Proof<F>, ProverState<F>), IdealCheckError<F, Self::Ideal>>
    where
        F: InnerTransparentField,
        F::Inner: ConstTranscribable,
        F::Modulus: ConstTranscribable;

    /// Tag-filtered counterpart of [`Self::verify_as_subprotocol`].
    ///
    /// Off-branch slots in the received proof must be `ZERO` — any
    /// other value is a soundness violation and is rejected with
    /// [`IdealCheckError::OffBranchSlotNotZero`].
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn verify_as_subprotocol_typed<F, IdealOverF, IdealOverFFromRef>(
        transcript: &mut impl Transcript,
        proof: Proof<F>,
        num_constraints: usize,
        num_vars: usize,
        ideal_over_f_from_ref: IdealOverFFromRef,
        field_cfg: &F::Config,
        tag_filter: ConstraintRing,
    ) -> Result<VerifierSubclaim<F>, IdealCheckError<F, IdealOverF>>
    where
        F: InnerTransparentField,
        F::Inner: ConstTranscribable,
        F::Modulus: ConstTranscribable,
        IdealOverF: Ideal + IdealCheck<DynamicPolynomialF<F>>,
        IdealOverFFromRef: Fn(&IdealOrZero<Self::Ideal>) -> IdealOverF;
}

impl<U> IdealCheckProtocol for U
where
    U: Uair,
{
    #[allow(clippy::type_complexity)]
    fn prove_linear<F>(
        transcript: &mut impl Transcript,
        trace_matrix: &ColumnMajorTrace<F>,
        projected_scalars: &ScalarMap<U::Scalar, DynamicPolynomialF<F>>,
        num_constraints: usize,
        num_vars: usize,
        field_cfg: &F::Config,
    ) -> Result<(Proof<F>, ProverState<F>), IdealCheckError<F, U::Ideal>>
    where
        F: InnerTransparentField,
        F::Inner: ConstTranscribable,
        F::Modulus: ConstTranscribable,
    {
        // Zero-ideal (`assert_zero`) constraints are identically zero on the
        // hypercube for an honest prover, and the MLE-first evaluation of a
        // nonlinear constraint produces a meaningless value. Both cases are
        // handled inside `CombinedPolyRowBuilder::assert_zero`, which writes
        // `ZERO` into the corresponding slot regardless of the input
        // expression — so no post-hoc overwrite is needed here.
        let evaluation_point = transcript.get_field_challenges(num_vars, field_cfg);

        // Evaluate combined polynomials using MLE-first approach:
        // evaluate trace columns at the point, then apply constraints.
        let combined_mle_values = combined_poly_builder::evaluate_combined_polynomials::<_, U>(
            trace_matrix,
            projected_scalars,
            num_constraints,
            &evaluation_point,
            field_cfg,
        )?;

        let mut transcription_buf: Vec<u8> = vec![0; F::Inner::NUM_BYTES];

        combined_mle_values.iter().for_each(|combined_mle_value| {
            transcript
                .absorb_random_field_slice(&combined_mle_value.coeffs, &mut transcription_buf);
        });

        Ok((
            Proof {
                combined_mle_values,
            },
            ProverState { evaluation_point },
        ))
    }

    #[allow(clippy::type_complexity)]
    fn prove_combined<F>(
        transcript: &mut impl Transcript,
        trace_matrix: &RowMajorTrace<F>,
        projected_scalars: &ScalarMap<U::Scalar, DynamicPolynomialF<F>>,
        num_constraints: usize,
        num_vars: usize,
        field_cfg: &F::Config,
    ) -> Result<(Proof<F>, ProverState<F>), IdealCheckError<F, U::Ideal>>
    where
        F: InnerTransparentField,
        F::Inner: ConstTranscribable,
        F::Modulus: ConstTranscribable,
    {
        // Build per-coefficient MLEs for every constraint, including
        // zero-ideal ones. The earlier short-circuit that replaced the
        // F[X] expression with `ZERO` for zero-ideal constraints was
        // only sound when the per-row F[X] expression was identically
        // the zero polynomial — see `CombinedPolyRowBuilder::assert_zero`
        // for the rationale.
        let is_zero_ideal: Vec<bool> = vec![false; num_constraints];

        let evaluation_point = transcript.get_field_challenges(num_vars, field_cfg);

        // Build combined polynomial MLEs row-by-row and evaluate them.
        let combined_mles = combined_poly_builder::compute_combined_polynomials::<_, U>(
            trace_matrix,
            projected_scalars,
            num_constraints,
            field_cfg,
            &is_zero_ideal,
        );

        let eq_table = build_eq_x_r_vec(&evaluation_point, field_cfg)?;

        // Evaluate coefficient MLEs at the evaluation point.
        let combined_mle_values: Vec<DynamicPolynomialF<F>> = cfg_into_iter!(combined_mles)
            .enumerate()
            .map(|(i, coeff_mles)| {
                // Skip zero-ideal constraints: their combined polynomial
                // is zero for an honest prover.
                if is_zero_ideal[i] {
                    return DynamicPolynomialF::ZERO;
                }
                let coeffs = coeff_mles
                    .into_iter()
                    .map(|coeff_mle| {
                        zinc_poly::utils::mle_eval_with_eq_table(
                            &coeff_mle.evaluations,
                            &eq_table,
                            field_cfg,
                        )
                    })
                    .collect::<Vec<_>>();
                DynamicPolynomialF::new_trimmed(coeffs)
            })
            .collect::<Vec<_>>();

        let mut transcription_buf: Vec<u8> = vec![0; F::Inner::NUM_BYTES];

        combined_mle_values.iter().for_each(|combined_mle_value| {
            transcript
                .absorb_random_field_slice(&combined_mle_value.coeffs, &mut transcription_buf);
        });

        Ok((
            Proof {
                combined_mle_values,
            },
            ProverState { evaluation_point },
        ))
    }

    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn prove_hybrid<F>(
        transcript: &mut impl Transcript,
        row_major_trace: &RowMajorTrace<F>,
        column_major_trace: &ColumnMajorTrace<F>,
        projected_scalars: &ScalarMap<U::Scalar, DynamicPolynomialF<F>>,
        num_constraints: usize,
        num_vars: usize,
        field_cfg: &F::Config,
    ) -> Result<(Proof<F>, ProverState<F>), IdealCheckError<F, U::Ideal>>
    where
        F: InnerTransparentField,
        F::Inner: ConstTranscribable,
        F::Modulus: ConstTranscribable,
    {
        // Per-constraint classification: linear vs non-linear (by degree).
        // Zero-ideal vs non-zero-ideal does NOT factor in here — both go
        // through the combined-poly lane when they're non-linear (see
        // comment below for why).
        let linear_mask = linear_constraint_mask::<U>();
        debug_assert_eq!(linear_mask.len(), num_constraints);

        let evaluation_point = transcript.get_field_challenges(num_vars, field_cfg);

        // MLE-first lane: produces correct values for linear non-zero-ideal
        // slots (and garbage for the rest, which we discard).
        let linear_values = combined_poly_builder::evaluate_combined_polynomials_unchecked::<_, U>(
            column_major_trace,
            projected_scalars,
            num_constraints,
            &evaluation_point,
            field_cfg,
        )?;

        // Combined-poly lane: build coefficient MLEs for every slot
        // that is NOT linear (i.e., every non-linear slot, including
        // zero-ideal ones). We do NOT skip zero-ideal here — although
        // the verifier's `batched_ideal_check` skips zero-ideal slots
        // and the IC's verifier output for them is unused, the values
        // still feed the downstream CPR sumcheck consistency check via
        // `comb_fn`. `CombinedPolyRowBuilder::assert_zero` documents
        // this: an honest prover's per-row F[X] expression for a
        // non-linear `assert_zero` is generally NOT identically the
        // zero polynomial — it can be a non-trivial poly that simply
        // evaluates to zero at the projecting element. Substituting
        // `ZERO` in the merge would diverge from `prove_combined`'s
        // output (and break sumcheck consistency).
        let skip_combined: Vec<bool> = linear_mask.clone();
        let combined_mles = combined_poly_builder::compute_combined_polynomials::<_, U>(
            row_major_trace,
            projected_scalars,
            num_constraints,
            field_cfg,
            &skip_combined,
        );

        let eq_table = build_eq_x_r_vec(&evaluation_point, field_cfg)?;

        // Merge: linear → MLE-first value; everything else (including
        // zero-ideal) → combined-poly value evaluated at the IC point.
        let combined_mle_values: Vec<DynamicPolynomialF<F>> = cfg_into_iter!(0..num_constraints)
            .map(|i| {
                if linear_mask[i] {
                    linear_values[i].clone()
                } else {
                    let coeffs = combined_mles[i]
                        .iter()
                        .map(|coeff_mle| {
                            zinc_poly::utils::mle_eval_with_eq_table(
                                &coeff_mle.evaluations,
                                &eq_table,
                                field_cfg,
                            )
                        })
                        .collect::<Vec<_>>();
                    DynamicPolynomialF::new_trimmed(coeffs)
                }
            })
            .collect::<Vec<_>>();

        let mut transcription_buf: Vec<u8> = vec![0; F::Inner::NUM_BYTES];
        combined_mle_values.iter().for_each(|combined_mle_value| {
            transcript
                .absorb_random_field_slice(&combined_mle_value.coeffs, &mut transcription_buf);
        });

        Ok((
            Proof {
                combined_mle_values,
            },
            ProverState { evaluation_point },
        ))
    }

    fn verify_as_subprotocol<F, IdealOverF, IdealOverFFromRef>(
        transcript: &mut impl Transcript,
        proof: Proof<F>,
        num_constraints: usize,
        num_vars: usize,
        ideal_over_f_from_ref: IdealOverFFromRef,
        field_cfg: &F::Config,
    ) -> Result<VerifierSubclaim<F>, IdealCheckError<F, IdealOverF>>
    where
        F: InnerTransparentField,
        F::Inner: ConstTranscribable,
        F::Modulus: ConstTranscribable,
        IdealOverF: Ideal + IdealCheck<DynamicPolynomialF<F>>,
        IdealOverFFromRef: Fn(&IdealOrZero<U::Ideal>) -> IdealOverF,
    {
        let mut transcription_buf: Vec<u8> = vec![0; F::Inner::NUM_BYTES];

        let combined_mle_values = proof.combined_mle_values;

        let evaluation_point = transcript.get_field_challenges(num_vars, field_cfg);

        for mle_value in &combined_mle_values {
            transcript.absorb_random_field_slice(&mle_value.coeffs, &mut transcription_buf);
        }

        let ideal_collector = collect_ideals::<U>(num_constraints);

        // Only check non-trivial ideals. For assert_zero constraints
        // the ideal is the zero ideal and the combined polynomial
        // value is zero by construction; the sumcheck that follows
        // verifies consistency of the claimed evaluations with the
        // actual trace.
        let (non_trivial_ideals, non_trivial_values): (Vec<_>, Vec<_>) = ideal_collector
            .ideals
            .iter()
            .zip(combined_mle_values.iter())
            .filter(|(ideal, _)| !ideal.is_zero_ideal())
            .map(|(ideal, value)| (ideal_over_f_from_ref(ideal), value.clone()))
            .unzip();

        batched_ideal_check(&non_trivial_ideals, &non_trivial_values)?;

        Ok(VerifierSubclaim {
            evaluation_point,
            values: combined_mle_values,
        })
    }

    // -----------------------------------------------------------------
    // Dual-prime variants (default impls)
    //
    // Implementations zero out off-branch slots BEFORE absorbing into
    // the transcript so the transcript flow exactly matches what the
    // dual-prime verifier reconstructs.
    // -----------------------------------------------------------------

    // Note on dual-prime soundness: the typed variants do NOT zero out
    // off-branch slots in the proof. The off-branch slot values still
    // participate in the downstream CPR sumcheck via comb_fn — folding
    // them out would break the equality between the prover's claimed
    // sum and the verifier's expected sum (computed from
    // ic_subclaim.values). Soundness for off-branch slots is enforced
    // implicitly by that sumcheck consistency, not by an explicit
    // zero-equality check on the proof. Tag filtering only restricts
    // which ideals get verified by `batched_ideal_check`.

    /// Tag-aware MLE-first prover. Validates that **tag-matching**
    /// non-zero-ideal constraints are linear (off-tag constraints may
    /// be non-linear; their values are zeroed out and excluded from
    /// the downstream CPR sumcheck via the tag-filtered
    /// [`ConstraintFolder`](crate::combined_poly_resolver::folder::ConstraintFolder)).
    ///
    /// Skips the global degree check that
    /// [`prove_linear`](Self::prove_linear) performs — that check would
    /// fail when off-tag constraints are non-linear, even though their
    /// values aren't checked by the tag-filtered
    /// `batched_ideal_check` and the tag-filtered CPR.
    fn prove_linear_typed<F>(
        transcript: &mut impl Transcript,
        trace_matrix: &ColumnMajorTrace<F>,
        projected_scalars: &ScalarMap<U::Scalar, DynamicPolynomialF<F>>,
        num_constraints: usize,
        num_vars: usize,
        field_cfg: &F::Config,
        tag_filter: ConstraintRing,
    ) -> Result<(Proof<F>, ProverState<F>), IdealCheckError<F, U::Ideal>>
    where
        F: InnerTransparentField,
        F::Inner: ConstTranscribable,
        F::Modulus: ConstTranscribable,
    {
        // Tag-aware degree check: only validate tag-matching slots.
        let degrees = zinc_uair::degree_counter::count_constraint_degrees::<U>();
        let collector = collect_ideals::<U>(num_constraints);
        let mut bad: Vec<usize> = Vec::new();
        for i in 0..num_constraints {
            if collector.tags[i] == tag_filter
                && !collector.ideals[i].is_zero_ideal()
                && degrees[i] > 1
            {
                bad.push(degrees[i]);
            }
        }
        if !bad.is_empty() {
            return Err(IdealCheckError::MleEvaluationError(
                EvaluationError::UnsupportedConstraintDegrees { degrees: bad },
            ));
        }

        let evaluation_point = transcript.get_field_challenges(num_vars, field_cfg);
        let mut combined_mle_values =
            combined_poly_builder::evaluate_combined_polynomials_unchecked::<_, U>(
                trace_matrix,
                projected_scalars,
                num_constraints,
                &evaluation_point,
                field_cfg,
            )?;

        // Zero out off-tag slots: the tag-filtered CPR sumcheck folder
        // skips them, so the IC absorbed values must match (zero).
        for i in 0..num_constraints {
            if collector.tags[i] != tag_filter {
                combined_mle_values[i] = DynamicPolynomialF::ZERO;
            }
        }

        let mut transcription_buf: Vec<u8> = vec![0; F::Inner::NUM_BYTES];
        combined_mle_values.iter().for_each(|combined_mle_value| {
            transcript
                .absorb_random_field_slice(&combined_mle_value.coeffs, &mut transcription_buf);
        });

        Ok((
            Proof {
                combined_mle_values,
            },
            ProverState { evaluation_point },
        ))
    }

    /// Tag-aware combined-poly prover. Skips per-coefficient MLE
    /// construction for off-tag and zero-ideal slots, and zeros out
    /// their absorbed values to match the tag-filtered CPR.
    fn prove_combined_typed<F>(
        transcript: &mut impl Transcript,
        trace_matrix: &RowMajorTrace<F>,
        projected_scalars: &ScalarMap<U::Scalar, DynamicPolynomialF<F>>,
        num_constraints: usize,
        num_vars: usize,
        field_cfg: &F::Config,
        tag_filter: ConstraintRing,
    ) -> Result<(Proof<F>, ProverState<F>), IdealCheckError<F, U::Ideal>>
    where
        F: InnerTransparentField,
        F::Inner: ConstTranscribable,
        F::Modulus: ConstTranscribable,
    {
        let collector = collect_ideals::<U>(num_constraints);
        let skip: Vec<bool> = (0..num_constraints)
            .map(|i| collector.tags[i] != tag_filter || collector.ideals[i].is_zero_ideal())
            .collect();

        let evaluation_point = transcript.get_field_challenges(num_vars, field_cfg);

        let combined_mles = combined_poly_builder::compute_combined_polynomials::<_, U>(
            trace_matrix,
            projected_scalars,
            num_constraints,
            field_cfg,
            &skip,
        );

        let eq_table = build_eq_x_r_vec(&evaluation_point, field_cfg)?;

        let combined_mle_values: Vec<DynamicPolynomialF<F>> = cfg_into_iter!(combined_mles)
            .enumerate()
            .map(|(i, coeff_mles)| {
                if skip[i] {
                    return DynamicPolynomialF::ZERO;
                }
                let coeffs = coeff_mles
                    .into_iter()
                    .map(|coeff_mle| {
                        zinc_poly::utils::mle_eval_with_eq_table(
                            &coeff_mle.evaluations,
                            &eq_table,
                            field_cfg,
                        )
                    })
                    .collect::<Vec<_>>();
                DynamicPolynomialF::new_trimmed(coeffs)
            })
            .collect();

        let mut transcription_buf: Vec<u8> = vec![0; F::Inner::NUM_BYTES];
        combined_mle_values.iter().for_each(|combined_mle_value| {
            transcript
                .absorb_random_field_slice(&combined_mle_value.coeffs, &mut transcription_buf);
        });

        Ok((
            Proof {
                combined_mle_values,
            },
            ProverState { evaluation_point },
        ))
    }

    /// Tag-aware hybrid prover. Routes on-tag linear slots through the
    /// MLE-first lane, on-tag non-linear slots through the combined-poly
    /// lane, and zeros out off-tag slots entirely. The tag-filtered CPR
    /// sumcheck folder skips off-tag slots, so the matching zero in the
    /// IC absorbed values is what the verifier expects.
    fn prove_hybrid_typed<F>(
        transcript: &mut impl Transcript,
        row_major_trace: &RowMajorTrace<F>,
        column_major_trace: &ColumnMajorTrace<F>,
        projected_scalars: &ScalarMap<U::Scalar, DynamicPolynomialF<F>>,
        num_constraints: usize,
        num_vars: usize,
        field_cfg: &F::Config,
        tag_filter: ConstraintRing,
    ) -> Result<(Proof<F>, ProverState<F>), IdealCheckError<F, U::Ideal>>
    where
        F: InnerTransparentField,
        F::Inner: ConstTranscribable,
        F::Modulus: ConstTranscribable,
    {
        let linear_mask = linear_constraint_mask::<U>();
        debug_assert_eq!(linear_mask.len(), num_constraints);
        let collector = collect_ideals::<U>(num_constraints);

        let evaluation_point = transcript.get_field_challenges(num_vars, field_cfg);

        // MLE-first lane: produces values for every slot. We only keep
        // on-tag linear slots from this output; the rest are overwritten
        // by either the combined-poly lane or zero.
        let linear_values =
            combined_poly_builder::evaluate_combined_polynomials_unchecked::<_, U>(
                column_major_trace,
                projected_scalars,
                num_constraints,
                &evaluation_point,
                field_cfg,
            )?;

        // Combined-poly lane: build coefficient MLEs only for slots that
        // are on-tag AND non-linear. On-tag linear slots use MLE-first
        // (faster); off-tag slots are zeroed.
        let skip_combined: Vec<bool> = (0..num_constraints)
            .map(|i| collector.tags[i] != tag_filter || linear_mask[i])
            .collect();
        let combined_mles = combined_poly_builder::compute_combined_polynomials::<_, U>(
            row_major_trace,
            projected_scalars,
            num_constraints,
            field_cfg,
            &skip_combined,
        );

        let eq_table = build_eq_x_r_vec(&evaluation_point, field_cfg)?;

        let combined_mle_values: Vec<DynamicPolynomialF<F>> = cfg_into_iter!(0..num_constraints)
            .map(|i| {
                if collector.tags[i] != tag_filter {
                    DynamicPolynomialF::ZERO
                } else if linear_mask[i] {
                    linear_values[i].clone()
                } else {
                    let coeffs = combined_mles[i]
                        .iter()
                        .map(|coeff_mle| {
                            zinc_poly::utils::mle_eval_with_eq_table(
                                &coeff_mle.evaluations,
                                &eq_table,
                                field_cfg,
                            )
                        })
                        .collect::<Vec<_>>();
                    DynamicPolynomialF::new_trimmed(coeffs)
                }
            })
            .collect();

        let mut transcription_buf: Vec<u8> = vec![0; F::Inner::NUM_BYTES];
        combined_mle_values.iter().for_each(|combined_mle_value| {
            transcript
                .absorb_random_field_slice(&combined_mle_value.coeffs, &mut transcription_buf);
        });

        Ok((
            Proof {
                combined_mle_values,
            },
            ProverState { evaluation_point },
        ))
    }

    fn verify_as_subprotocol_typed<F, IdealOverF, IdealOverFFromRef>(
        transcript: &mut impl Transcript,
        proof: Proof<F>,
        num_constraints: usize,
        num_vars: usize,
        ideal_over_f_from_ref: IdealOverFFromRef,
        field_cfg: &F::Config,
        tag_filter: ConstraintRing,
    ) -> Result<VerifierSubclaim<F>, IdealCheckError<F, IdealOverF>>
    where
        F: InnerTransparentField,
        F::Inner: ConstTranscribable,
        F::Modulus: ConstTranscribable,
        IdealOverF: Ideal + IdealCheck<DynamicPolynomialF<F>>,
        IdealOverFFromRef: Fn(&IdealOrZero<U::Ideal>) -> IdealOverF,
    {
        let mut transcription_buf: Vec<u8> = vec![0; F::Inner::NUM_BYTES];

        let combined_mle_values = proof.combined_mle_values;

        let evaluation_point = transcript.get_field_challenges(num_vars, field_cfg);

        for mle_value in &combined_mle_values {
            transcript.absorb_random_field_slice(&mle_value.coeffs, &mut transcription_buf);
        }

        let ideal_collector = collect_ideals::<U>(num_constraints);

        // Tag-filtered ideal check: skip zero-ideal slots (assert_zero)
        // and slots whose tag doesn't match the active branch.
        let (non_trivial_ideals, non_trivial_values): (Vec<_>, Vec<_>) = ideal_collector
            .ideals
            .iter()
            .zip(ideal_collector.tags.iter())
            .zip(combined_mle_values.iter())
            .filter(|((ideal, tag), _)| !ideal.is_zero_ideal() && **tag == tag_filter)
            .map(|((ideal, _), value)| (ideal_over_f_from_ref(ideal), value.clone()))
            .unzip();

        batched_ideal_check(&non_trivial_ideals, &non_trivial_values)?;

        Ok(VerifierSubclaim {
            evaluation_point,
            values: combined_mle_values,
        })
    }
}

#[derive(Clone, Debug, Error)]
pub enum IdealCheckError<F: PrimeField, I> {
    #[error("ideal check prover failed to evaluate an mle: {0}")]
    MleEvaluationError(#[from] EvaluationError),
    #[error("mle evaluation ideal check failure: {0}")]
    IdealCollectorError(#[from] BatchedIdealCheckError<DynamicPolynomialF<F>, I>),
    #[error("`eq` polynomial construction failure: {0}")]
    EqPolyConstructionError(#[from] PolyArithErrors),
}

#[cfg(test)]
mod tests {
    use crypto_primitives::{crypto_bigint_int::Int, crypto_bigint_monty::MontyField};

    use rand::rng;
    use zinc_poly::univariate::{dense::DensePolynomial, dynamic::over_field::DynamicPolynomialF};
    use zinc_test_uair::{
        GenerateRandomTrace, TestUairNoMultiplication, TestUairSimpleMultiplication,
    };
    use zinc_transcript::Blake3Transcript;
    use zinc_uair::{
        constraint_counter::count_constraints,
        ideal::{DegreeOneIdeal, Ideal, IdealCheck},
    };

    use crate::test_utils::{
        LIMBS, run_ideal_check_prover_combined, run_ideal_check_prover_linear, test_config,
    };

    use super::*;

    // TODO(Ilia): These tests are absolute joke.
    //             Once we have time we need to create a comprehensive test suite
    //             akin to the one we have for the PCS or the sumcheck.

    fn test_successful_verification_linear<
        U,
        IdealOverF,
        IdealOverFFromRef,
        const DEGREE_PLUS_ONE: usize,
    >(
        num_vars: usize,
        ideal_over_f_from_ref: IdealOverFFromRef,
    ) where
        U: Uair<Scalar = DensePolynomial<Int<5>, DEGREE_PLUS_ONE>>
            + GenerateRandomTrace<DEGREE_PLUS_ONE, PolyCoeff = Int<5>, Int = Int<5>>
            + IdealCheckProtocol,
        IdealOverF: Ideal + IdealCheck<DynamicPolynomialF<MontyField<LIMBS>>>,
        IdealOverFFromRef: Fn(&IdealOrZero<U::Ideal>) -> IdealOverF,
    {
        let mut rng = rng();
        let transcript = Blake3Transcript::new();

        let (proof, prover_state, ..) = run_ideal_check_prover_linear::<U, DEGREE_PLUS_ONE>(
            num_vars,
            &U::generate_random_trace(num_vars, &mut rng),
            &mut transcript.clone(),
        );

        let num_constraints = count_constraints::<U>();

        let verifier_result = U::verify_as_subprotocol(
            &mut transcript.clone(),
            proof,
            num_constraints,
            num_vars,
            ideal_over_f_from_ref,
            &test_config(),
        )
        .expect("Verification failed");

        assert_eq!(
            prover_state.evaluation_point,
            verifier_result.evaluation_point
        );
    }

    fn test_successful_verification_combined<
        U,
        IdealOverF,
        IdealOverFFromRef,
        const DEGREE_PLUS_ONE: usize,
    >(
        num_vars: usize,
        ideal_over_f_from_ref: IdealOverFFromRef,
    ) where
        U: Uair<Scalar = DensePolynomial<Int<5>, DEGREE_PLUS_ONE>>
            + GenerateRandomTrace<DEGREE_PLUS_ONE, PolyCoeff = Int<5>, Int = Int<5>>
            + IdealCheckProtocol,
        IdealOverF: Ideal + IdealCheck<DynamicPolynomialF<MontyField<LIMBS>>>,
        IdealOverFFromRef: Fn(&IdealOrZero<U::Ideal>) -> IdealOverF,
    {
        let mut rng = rng();
        let transcript = Blake3Transcript::new();

        let (proof, prover_state, ..) = run_ideal_check_prover_combined::<U, DEGREE_PLUS_ONE>(
            num_vars,
            &U::generate_random_trace(num_vars, &mut rng),
            &mut transcript.clone(),
        );

        let num_constraints = count_constraints::<U>();

        let verifier_result = U::verify_as_subprotocol(
            &mut transcript.clone(),
            proof,
            num_constraints,
            num_vars,
            ideal_over_f_from_ref,
            &test_config(),
        )
        .expect("Verification failed");

        assert_eq!(
            prover_state.evaluation_point,
            verifier_result.evaluation_point
        );
    }

    #[test]
    fn test_successful_verification() {
        let field_cfg = test_config();

        let num_vars = 2;

        // Linear UAIR - test both approaches
        test_successful_verification_linear::<TestUairNoMultiplication<Int<5>>, _, _, 32>(
            num_vars,
            |ideal_over_ring| ideal_over_ring.map(|i| DegreeOneIdeal::from_with_cfg(i, &field_cfg)),
        );
        test_successful_verification_combined::<TestUairNoMultiplication<Int<5>>, _, _, 32>(
            num_vars,
            |ideal_over_ring| ideal_over_ring.map(|i| DegreeOneIdeal::from_with_cfg(i, &field_cfg)),
        );

        // Non-linear UAIR - only combined approach works
        test_successful_verification_combined::<TestUairSimpleMultiplication<Int<5>>, _, _, 32>(
            num_vars,
            |_ideal_over_ring| IdealOrZero::<DegreeOneIdeal<_>>::zero(),
        );
    }
}
