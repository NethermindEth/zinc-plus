use crate::{
    ZipError,
    code::LinearCode,
    pcs::{
        ZipPlusProof, ZipPlusTestTranscript,
        structs::{ZipPlus, ZipPlusParams, ZipTypes},
        utils::{point_to_tensor, validate_input},
    },
    pcs_transcript::PcsTranscript,
    utils::combine_rows,
};
use crypto_primitives::{FromWithConfig, IntoWithConfig, PrimeField};
use itertools::Itertools;
use zinc_poly::mle::DenseMultilinearExtension;
use zinc_transcript::traits::{Transcribable, Transcript};
use zinc_utils::{
    from_ref::FromRef,
    inner_product::{InnerProduct, MBSInnerProductUnchecked},
    mul_by_scalar::MulByScalar,
    projection_to_field::ProjectionToField,
};

impl<Zt: ZipTypes, Lc: LinearCode<Zt>> ZipPlus<Zt, Lc> {
    pub fn evaluate<F, P>(
        pp: &ZipPlusParams<Zt, Lc>,
        poly: &DenseMultilinearExtension<Zt::Eval>,
        point: &[Zt::Pt],
        test_transcript: ZipPlusTestTranscript,
    ) -> Result<(F, ZipPlusProof), ZipError>
    where
        F: PrimeField
            + for<'a> FromWithConfig<&'a Zt::Chal>
            + for<'a> FromWithConfig<&'a Zt::Pt>
            + for<'a> MulByScalar<&'a F>,
        F::Inner: FromRef<Zt::Fmod> + Transcribable,
        P: ProjectionToField<Zt::Eval, F>,
    {
        validate_input::<Zt, Lc, _>("evaluate", pp.num_vars, [poly], [point])?;

        let mut transcript: PcsTranscript = test_transcript.into();

        let field_modulus = F::Inner::from_ref(
            &transcript
                .fs_transcript
                .get_prime::<Zt::Fmod, Zt::PrimeTest>(),
        );
        let field_cfg = F::make_cfg(&field_modulus)?;
        let projecting_element: Zt::Chal = transcript.fs_transcript.get_challenge();
        let projecting_element: F = (&projecting_element).into_with_cfg(&field_cfg);

        let num_rows = pp.num_rows;
        let row_len = pp.linear_code.row_len();

        // We prove evaluations over the field, so integers need to be mapped to field
        // elements first
        let point = point
            .iter()
            .map(|v| v.into_with_cfg(&field_cfg))
            .collect_vec();
        let (q_0, q_1) = point_to_tensor(num_rows, &point, &field_cfg)?;

        let project = P::prepare_projection(&projecting_element);
        let evaluations: Vec<F> = poly.evaluations.iter().map(project).collect_vec();

        let q_0_combined_row = if num_rows > 1 {
            // Return the evaluation row combination
            combine_rows(&q_0, &evaluations, row_len)
        } else {
            // If there is only one row, we have no need to take linear combinations
            // We just return the evaluation row combination
            evaluations
        };

        transcript.write_field_elements(&q_0_combined_row)?;
        let eval_f = MBSInnerProductUnchecked::inner_product(
            &q_0_combined_row,
            &q_1,
            F::zero_with_cfg(&field_cfg),
        )?;
        Ok((eval_f, transcript.into()))
    }
}

// There's no point in testing evaluation phase in isolation, so it's covered by
// full (verification) tests
