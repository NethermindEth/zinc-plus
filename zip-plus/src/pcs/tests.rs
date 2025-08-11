use std::marker::PhantomData;
use ark_std::{collections::BTreeSet, ops::Range, vec::Vec};

use crate::{
    define_random_field_zip_types,
    field::Int,
    implement_random_field_zip_types,
    poly_z::mle::DenseMultilinearExtension,
    traits::Integer,
    code::DefaultLinearCodeSpec,
    pcs::structs::{MultilinearZipParams, ZipTranscript},
    utils::div_ceil,
};
use crate::code_raa::RaaCode;

const INT_LIMBS: usize = 1;

define_random_field_zip_types!();
implement_random_field_zip_types!(INT_LIMBS);

type ZT = RandomFieldZipTypes<INT_LIMBS>;

#[derive(Default)]
pub struct MockTranscript {
    pub counter: i64,
}

impl<L: Integer> ZipTranscript<L> for MockTranscript {
    fn get_encoding_element(&mut self) -> L {
        self.counter += 1;
        L::from(self.counter)
    }
    fn get_u64(&mut self) -> u64 {
        self.counter += 1;
        self.counter as u64
    }
    fn sample_unique_columns(
        &mut self,
        range: Range<usize>,
        columns: &mut BTreeSet<usize>,
        count: usize,
    ) -> usize {
        self.counter += 1;
        let mut inserted = 0;
        for i in range.clone() {
            if columns.insert(i) {
                inserted += 1;
                if inserted == count {
                    break;
                }
            }
        }
        inserted
    }
}

pub fn setup_test_params(
    num_vars: usize,
) -> (
    MultilinearZipParams<ZT, RaaCode<ZT>>,
    DenseMultilinearExtension<Int<INT_LIMBS>>,
) {
    let poly_size = 1 << num_vars;
    let num_rows = 1 << div_ceil(num_vars, 2);

    let mut transcript = MockTranscript::default();
    let linear_code = RaaCode::<ZT>::new(&DefaultLinearCodeSpec, poly_size, &mut transcript);
    let pp = MultilinearZipParams { num_vars, num_rows, linear_code, phantom_data_zt: PhantomData };

    let evaluations: Vec<_> = (1..=poly_size).map(|v| Int::from(v as i32)).collect();
    let poly = DenseMultilinearExtension::from_evaluations_vec(num_vars, evaluations);

    (pp, poly)
}
