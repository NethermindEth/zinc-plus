use crate::{
    ZipError,
    code::LinearCode,
    combine_rows,
    pcs::{
        structs::ZipTypes,
        utils::{point_to_tensor, validate_input},
    },
    pcs_transcript::PcsTranscript,
};
use crypto_primitives::{FromWithConfig, IntoWithConfig, PrimeField};
use itertools::Itertools;
use zinc_poly::mle::DenseMultilinearExtension;
use zinc_transcript::traits::{Transcribable, Transcript};
use zinc_utils::{
    UNCHECKED, cfg_iter, cfg_iter_mut,
    from_ref::FromRef,
    inner_product::{InnerProduct, MBSInnerProduct},
    mul_by_scalar::MulByScalar,
    projectable_to_field::ProjectableToField,
};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::structs::{
    BatchedZipPlus, BatchedZipPlusParams, BatchedZipPlusProof, BatchedZipPlusTestTranscript,
};

impl<Zt: ZipTypes, Lc: LinearCode<Zt>> BatchedZipPlus<Zt, Lc> {
    /// Performs the evaluation phase for a batch of polynomials.
    ///
    /// All polynomials are evaluated at the **same** point. Produces one field
    /// evaluation per polynomial, plus a single batched proof.
    ///
    /// # Parameters
    /// - `pp`: Public parameters shared across all polynomials.
    /// - `polys`: Slice of multilinear polynomials in the batch.
    /// - `point`: The shared evaluation point (same for all polynomials).
    /// - `test_transcript`: The transcript from the testing phase.
    ///
    /// # Returns
    /// A tuple `(Vec<F>, BatchedZipPlusProof)` containing one evaluation per
    /// polynomial and the batched proof.
    pub fn evaluate<F, const CHECK_FOR_OVERFLOW: bool>(
        pp: &BatchedZipPlusParams<Zt, Lc>,
        polys: &[DenseMultilinearExtension<Zt::Eval>],
        point: &[Zt::Pt],
        test_transcript: BatchedZipPlusTestTranscript,
    ) -> Result<(Vec<F>, BatchedZipPlusProof), ZipError>
    where
        F: PrimeField
            + for<'a> FromWithConfig<&'a Zt::Chal>
            + for<'a> FromWithConfig<&'a Zt::Pt>
            + for<'a> MulByScalar<&'a F>
            + FromRef<F>,
        F::Inner: FromRef<Zt::Fmod> + Transcribable,
        Zt::Eval: ProjectableToField<F>,
    {
        assert!(!polys.is_empty(), "Batch must contain at least one polynomial");

        for poly in polys {
            validate_input::<Zt, Lc, _>(
                "batched_evaluate",
                pp.num_vars,
                pp.num_rows,
                pp.linear_code.row_len(),
                &[poly],
                &[point],
            )?;
        }

        let mut transcript: PcsTranscript = test_transcript.into();

        let field_cfg = transcript
            .fs_transcript
            .get_random_field_cfg::<F, Zt::Fmod, Zt::PrimeTest>();
        let projecting_element: Zt::Chal = transcript.fs_transcript.get_challenge();
        let projecting_element: F = (&projecting_element).into_with_cfg(&field_cfg);

        let num_rows = pp.num_rows;
        let row_len = pp.linear_code.row_len();

        let point_f: Vec<F> = point
            .iter()
            .map(|v| v.into_with_cfg(&field_cfg))
            .collect_vec();
        let (q_0, q_1) = point_to_tensor(num_rows, &point_f, &field_cfg)?;

        let project = Zt::Eval::prepare_projection(&projecting_element);

        let mut evals_f = Vec::with_capacity(polys.len());

        for poly in polys {
            let evaluations: Vec<F> = cfg_iter!(poly).map(&project).collect();

            let q_0_combined_row = if num_rows > 1 {
                combine_rows!(
                    CHECK_FOR_OVERFLOW,
                    &q_0,
                    evaluations.iter(),
                    Ok::<_, ZipError>,
                    row_len,
                    F::zero_with_cfg(&field_cfg)
                )
            } else {
                evaluations
            };

            transcript.write_field_elements(&q_0_combined_row)?;

            // It is safe to use unchecked inner product since we are in a field.
            let eval_f = MBSInnerProduct::inner_product::<UNCHECKED>(
                &q_0_combined_row,
                &q_1,
                F::zero_with_cfg(&field_cfg),
            )?;

            evals_f.push(eval_f);
        }

        Ok((evals_f, transcript.into()))
    }
}
