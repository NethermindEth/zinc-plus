use crate::{
    ZipError,
    code::LinearCode,
    combine_rows,
    pcs::{
        structs::{ZipPlus, ZipPlusParams, ZipTypes},
        utils::{point_to_tensor, validate_input},
    },
    pcs_transcript::PcsProverTranscript,
};
use crypto_primitives::{FromWithConfig, IntoWithConfig, PrimeField};
use itertools::Itertools;
use zinc_poly::mle::DenseMultilinearExtension;
use zinc_transcript::traits::Transcribable;
use zinc_utils::{
    UNCHECKED, cfg_iter, cfg_iter_mut,
    from_ref::FromRef,
    inner_product::{InnerProduct, MBSInnerProduct},
    mul_by_scalar::MulByScalar,
    projectable_to_field::ProjectableToField,
};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

impl<Zt: ZipTypes, Lc: LinearCode<Zt>> ZipPlus<Zt, Lc> {
    pub fn evaluate<F, const CHECK_FOR_OVERFLOW: bool>(
        transcript: &mut PcsProverTranscript,
        pp: &ZipPlusParams<Zt, Lc>,
        poly: &DenseMultilinearExtension<Zt::Eval>,
        point: &[Zt::Pt],
        field_cfg: &F::Config,
        projecting_element: &Zt::Chal,
    ) -> Result<F, ZipError>
    where
        F: PrimeField
            + for<'a> FromWithConfig<&'a Zt::Chal>
            + for<'a> FromWithConfig<&'a Zt::Pt>
            + for<'a> MulByScalar<&'a F>
            + FromRef<F>,
        F::Inner: Transcribable,
        F::Modulus: FromRef<Zt::Fmod> + Transcribable,
        Zt::Eval: ProjectableToField<F>,
    {
        validate_input::<Zt, Lc, _>("evaluate", pp.num_vars, &[poly], &[point])?;

        let projecting_element: F = projecting_element.into_with_cfg(field_cfg);

        let num_rows = pp.num_rows;
        let row_len = pp.linear_code.row_len();

        // We prove evaluations over the field, so integers need to be mapped to field
        // elements first
        let point = point
            .iter()
            .map(|v| v.into_with_cfg(field_cfg))
            .collect_vec();
        let (q_0, q_1) = point_to_tensor(num_rows, &point, field_cfg)?;

        let project = Zt::Eval::prepare_projection(&projecting_element);
        let evaluations: Vec<F> = cfg_iter!(poly).map(project).collect();

        let q_0_combined_row = if num_rows > 1 {
            // Return the evaluation row combination
            combine_rows!(
                CHECK_FOR_OVERFLOW,
                &q_0,
                evaluations.iter(),
                Ok::<_, ZipError>,
                row_len,
                F::zero_with_cfg(field_cfg)
            )
        } else {
            // If there is only one row, we have no need to take linear combinations
            // We just return the evaluation row combination
            evaluations
        };

        transcript.write_field_elements(&q_0_combined_row)?;
        // It is safe to use unchecked inner product since we are in a field.
        let eval_f = MBSInnerProduct::inner_product::<UNCHECKED>(
            &q_0_combined_row,
            &q_1,
            F::zero_with_cfg(field_cfg),
        )?;
        Ok(eval_f)
    }
}

// There's no point in testing evaluation phase in isolation, so it's covered by
// full (verification) tests
