#![allow(non_snake_case)]
use ark_std::{borrow::Cow, vec::Vec};
use itertools::izip;

use super::{
    structs::{MultilinearZip, MultilinearZipData},
    utils::{left_point_to_tensor, validate_input, ColumnOpening},
};
use crate::{
    poly_z::mle::DenseMultilinearExtension,
    traits::{Field, FieldMap, ZipTypes},
    code::LinearCode,
    pcs::structs::MultilinearZipParams,
    pcs_transcript::PcsTranscript,
    utils::{combine_rows, expand},
    Error,
};

impl<ZT: ZipTypes, LC: LinearCode<ZT>> MultilinearZip<ZT, LC> {
    pub fn open<F: Field>(
        pp: &MultilinearZipParams<ZT, LC>,
        poly: &DenseMultilinearExtension<ZT::N>,
        commit_data: &MultilinearZipData<ZT::K>,
        point: &[F],
        transcript: &mut PcsTranscript<F>,
    ) -> Result<(), Error>
    where
        ZT::N: FieldMap<F, Output = F>,
    {
        validate_input("open", pp.num_vars, [poly], [point])?;

        Self::prove_testing_phase(pp, poly, commit_data, transcript)?;

        Self::prove_evaluation_phase(pp, transcript, point, poly)?;

        Ok(())
    }

    // TODO Apply 2022/1355 https://eprint.iacr.org/2022/1355.pdf#page=30
    pub fn batch_open<F: Field>(
        pp: &MultilinearZipParams<ZT, LC>,
        polys: &[DenseMultilinearExtension<ZT::N>],
        comms: &[MultilinearZipData<ZT::K>],
        points: &[Vec<F>],
        transcript: &mut PcsTranscript<F>,
    ) -> Result<(), Error>
    where
        ZT::N: FieldMap<F, Output = F>,
    {
        for (poly, comm, point) in izip!(polys.iter(), comms.iter(), points.iter()) {
            Self::open(pp, poly, comm, point, transcript)?;
        }
        Ok(())
    }

    // Subprotocol functions

    fn prove_evaluation_phase<F: Field>(
        pp: &MultilinearZipParams<ZT, LC>,
        transcript: &mut PcsTranscript<F>,
        point: &[F],
        poly: &DenseMultilinearExtension<ZT::N>,
    ) -> Result<(), Error>
    where
        ZT::N: FieldMap<F, Output = F>,
    {
        let num_rows = pp.num_rows;
        let row_len = pp.linear_code.row_len();

        // We prove evaluations over the field, so integers need to be mapped to field elements first
        let q_0 = left_point_to_tensor(num_rows, point)?;

        let evaluations = poly.evaluations.map_to_field();

        let q_0_combined_row = if num_rows > 1 {
            // Return the evaluation row combination
            let combined_row = combine_rows(q_0, evaluations, row_len);
            Cow::<Vec<F>>::Owned(combined_row)
        } else {
            // If there is only one row, we have no need to take linear combinations
            // We just return the evaluation row combination
            Cow::Borrowed(&evaluations)
        };

        transcript.write_field_elements(&q_0_combined_row)
    }

    pub(super) fn prove_testing_phase<F: Field>(
        pp: &MultilinearZipParams<ZT, LC>,
        poly: &DenseMultilinearExtension<ZT::N>,
        commit_data: &MultilinearZipData<ZT::K>,
        transcript: &mut PcsTranscript<F>,
    ) -> Result<(), Error> {
        if pp.num_rows > 1 {
            // If we can take linear combinations
            // perform the proximity test an arbitrary number of times
            for _ in 0..pp.linear_code.num_proximity_testing() {
                let coeffs = transcript.fs_transcript.get_integer_challenges(pp.num_rows);
                let coeffs = coeffs.iter().map(expand::<ZT::N, ZT::M>);

                let evals = poly.evaluations.iter().map(expand::<ZT::N, ZT::M>);

                // u' in the Zinc paper
                let combined_row = combine_rows(coeffs, evals, pp.linear_code.row_len());

                transcript.write_integers(combined_row.iter())?;
            }
        }

        // Open merkle tree for each column drawn
        for _ in 0..pp.linear_code.num_column_opening() {
            let column = transcript.squeeze_challenge_idx(pp.linear_code.codeword_len());
            Self::open_merkle_trees_for_column(pp, commit_data, column, transcript)?;
        }
        Ok(())
    }

    pub(super) fn open_merkle_trees_for_column<F: Field>(
        pp: &MultilinearZipParams<ZT, LC>,
        commit_data: &MultilinearZipData<ZT::K>,
        column: usize,
        transcript: &mut PcsTranscript<F>,
    ) -> Result<(), Error> {
        let column_values = commit_data
            .rows
            .iter()
            .skip(column)
            .step_by(pp.linear_code.codeword_len());

        // Write the elements in the squeezed column to the shared transcript
        transcript.write_integers(column_values)?;

        ColumnOpening::open_at_column(column, commit_data, transcript)
            .map_err(|_| Error::InvalidPcsOpen("Failed to open merkle tree".into()))?;

        Ok(())
    }
}
