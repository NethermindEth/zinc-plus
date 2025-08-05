use ark_std::{iterable::Iterable, vec::Vec};
use sha3::{digest::Output, Keccak256};

use super::{
    structs::{MultilinearZip, MultilinearZipCommitment},
    utils::{point_to_tensor, validate_input, ColumnOpening},
};
use crate::{
    traits::{Field, FieldMap, ZipTypes},
    code::LinearCode,
    pcs::structs::MultilinearZipParams,
    pcs_transcript::PcsTranscript,
    utils::{expand, inner_product},
    Error,
};

impl<ZT: ZipTypes, LC: LinearCode<ZT>> MultilinearZip<ZT, LC> {
    pub fn verify<F: Field>(
        vp: &MultilinearZipParams<ZT, LC>,
        comm: &MultilinearZipCommitment,
        point: &[F],
        eval: F,
        transcript: &mut PcsTranscript<F>,
        field: F::R,
    ) -> Result<(), Error>
    where
        ZT::L: FieldMap<F, Output = F>,
        ZT::K: FieldMap<F, Output = F>,
    {
        validate_input::<ZT::N, F>("verify", vp.num_vars, [], [point])?;

        let columns_opened = Self::verify_testing(vp, &comm.roots, transcript, field)?;

        Self::verify_evaluation_z(vp, point, eval, &columns_opened, transcript, field)?;

        Ok(())
    }

    pub fn batch_verify_z<'a, F: Field>(
        vp: &MultilinearZipParams<ZT, LC>,
        comms: impl Iterable<Item = &'a MultilinearZipCommitment>,
        points: &[Vec<F>],
        evals: &[F],
        transcript: &mut PcsTranscript<F>,
        field: F::R,
    ) -> Result<(), Error>
    where
        ZT::L: FieldMap<F, Output = F>,
        ZT::K: FieldMap<F, Output = F>,
        ZT::N: 'a,
    {
        for (i, (eval, comm)) in evals.iter().zip(comms.iter()).enumerate() {
            Self::verify(vp, comm, &points[i], eval.clone(), transcript, field)?;
        }
        Ok(())
    }

    #[allow(clippy::type_complexity)]
    pub(super) fn verify_testing<F: Field>(
        vp: &MultilinearZipParams<ZT, LC>,
        roots: &[Output<Keccak256>],
        transcript: &mut PcsTranscript<F>,
        field: F::R,
    ) -> Result<Vec<(usize, Vec<ZT::K>)>, Error> {
        // Gather the coeffs and encoded combined rows per proximity test
        let mut encoded_combined_rows: Vec<(Vec<ZT::N>, Vec<ZT::M>)> =
            Vec::with_capacity(vp.linear_code.num_proximity_testing());

        if vp.num_rows > 1 {
            for _ in 0..vp.linear_code.num_proximity_testing() {
                let coeffs = transcript.fs_transcript.get_integer_challenges(vp.num_rows);

                let combined_row: Vec<ZT::M> =
                    transcript.read_integers(vp.linear_code.row_len())?;

                let encoded_combined_row: Vec<ZT::M> = vp.linear_code.encode_wide(&combined_row);
                encoded_combined_rows.push((coeffs, encoded_combined_row));
            }
        }

        let mut columns_opened: Vec<(usize, Vec<ZT::K>)> =
            Vec::with_capacity(vp.linear_code.num_column_opening());

        for _ in 0..vp.linear_code.num_column_opening() {
            let column_idx = transcript.squeeze_challenge_idx(field, vp.linear_code.codeword_len());
            let column_values = transcript.read_integers(vp.num_rows)?;

            for (coeffs, encoded_combined_row) in encoded_combined_rows.iter() {
                Self::verify_column_testing(
                    coeffs,
                    encoded_combined_row,
                    &column_values,
                    column_idx,
                    vp.num_rows,
                )?;
            }

            let _ = ColumnOpening::verify_column(roots, &column_values, column_idx, transcript);
            // TODO: Verify column opening is taking a long time.
            columns_opened.push((column_idx, column_values));
        }

        Ok(columns_opened)
    }

    pub(super) fn verify_column_testing(
        coeffs: &[ZT::N],
        encoded_combined_row: &[ZT::M],
        column_entries: &[ZT::K],
        column: usize,
        num_rows: usize,
    ) -> Result<(), Error> {
        let column_entries_comb: ZT::M = if num_rows > 1 {
            let coeffs: Vec<ZT::M> = coeffs.iter().map(expand::<ZT::N, ZT::M>).collect();
            let column_entries: Vec<ZT::M> =
                column_entries.iter().map(expand::<ZT::K, ZT::M>).collect();
            inner_product(coeffs.iter(), column_entries.iter())
        } else {
            expand(&column_entries[0])
        };

        if column_entries_comb != encoded_combined_row[column] {
            return Err(Error::InvalidPcsOpen("Proximity failure".into()));
        }
        Ok(())
    }

    fn verify_evaluation_z<F: Field>(
        vp: &MultilinearZipParams<ZT, LC>,
        point: &[F],
        eval: F,
        columns_opened: &[(usize, Vec<ZT::K>)],
        transcript: &mut PcsTranscript<F>,
        field: F::R,
    ) -> Result<(), Error>
    where
        ZT::L: FieldMap<F, Output = F>,
        ZT::K: FieldMap<F, Output = F>,
    {
        let q_0_combined_row = transcript.read_field_elements(vp.linear_code.row_len(), field)?;
        let encoded_combined_row = vp.linear_code.encode_f(&q_0_combined_row, field);

        let (q_0, q_1) = point_to_tensor(vp.num_rows, point, field)?;

        if inner_product(&q_0_combined_row, &q_1) != eval {
            return Err(Error::InvalidPcsOpen(
                "Evaluation consistency failure".into(),
            ));
        }
        for (column_idx, column_values) in columns_opened.iter() {
            Self::verify_proximity_q_0(
                &q_0,
                &encoded_combined_row,
                column_values,
                *column_idx,
                vp.num_rows,
                field,
            )?;
        }

        Ok(())
    }

    fn verify_proximity_q_0<F: Field>(
        q_0: &Vec<F>,
        encoded_q_0_combined_row: &[F],
        column_entries: &[ZT::K],
        column: usize,
        num_rows: usize,
        field: F::R,
    ) -> Result<(), Error>
    where
        ZT::K: FieldMap<F, Output = F>,
    {
        let column_entries_comb = if num_rows > 1 {
            let column_entries = column_entries.map_to_field(field);
            inner_product(q_0, &column_entries)
            // TODO: this inner product is taking a long time.
        } else {
            column_entries.first().unwrap().map_to_field(field)
        };
        if column_entries_comb != encoded_q_0_combined_row[column] {
            return Err(Error::InvalidPcsOpen("Proximity failure".into()));
        }

        Ok(())
    }
}
