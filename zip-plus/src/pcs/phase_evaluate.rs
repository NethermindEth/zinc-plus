use std::borrow::Cow;

use crate::{
    ZipError,
    code::LinearCode,
    pcs::{
        ZipPlusProof, ZipPlusTestTranscript,
        structs::{MulByScalar, ProjectableToField, ZipPlus, ZipPlusParams, ZipTypes},
        utils::{point_to_tensor, validate_input},
    },
    pcs_transcript::PcsTranscript,
    poly::mle::DenseMultilinearExtension,
    traits::{FromRef, Transcribable, Transcript},
    utils::{combine_rows, inner_product},
};
use crypto_primitives::{FromWithConfig, IntoWithConfig, PrimeField};
use itertools::Itertools;

impl<Zt: ZipTypes<DEGREE>, Lc: LinearCode<Zt, DEGREE>, const DEGREE: usize>
    ZipPlus<Zt, Lc, DEGREE>
{
    pub fn evaluate<F>(
        pp: &ZipPlusParams<Zt, Lc, DEGREE>,
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
        Zt::Eval: ProjectableToField<F>,
    {
        validate_input::<Zt, Lc, _, DEGREE>("evaluate", pp.num_vars, [poly], [point])?;

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

        let project = Zt::Eval::prepare_projection(&projecting_element);
        let evaluations: Vec<F> = poly.evaluations.iter().map(project).collect_vec();

        let q_0_combined_row = if num_rows > 1 {
            // Return the evaluation row combination
            let combined_row = combine_rows(&q_0, &evaluations, row_len);
            Cow::<Vec<F>>::Owned(combined_row)
        } else {
            // If there is only one row, we have no need to take linear combinations
            // We just return the evaluation row combination
            Cow::Borrowed(&evaluations)
        };

        transcript.write_field_elements(&q_0_combined_row)?;
        let eval_f = inner_product(&q_0_combined_row[..], &q_1, F::zero_with_cfg(&field_cfg));
        Ok((eval_f, transcript.into()))
    }
}

// There's no point in testing evaluation phase in isolation, so it's covered by
// full (verification) tests
