use ark_std::{collections::BTreeSet, ops::Range};

use crate::{
    define_random_field_zip_types, implement_random_field_zip_types, pcs::structs::ZipTranscript,
    traits::Integer,
};

const INT_LIMBS: usize = 1;

define_random_field_zip_types!();
implement_random_field_zip_types!(INT_LIMBS);

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
