use std::{collections::BTreeSet, ops::Range};
use crypto_bigint::{Int, Word};

use crate::traits::Transcript;

const INT_LIMBS: usize = 1;

const N: usize = INT_LIMBS;
const L: usize = INT_LIMBS * 2;
const K: usize = INT_LIMBS * 4;
const M: usize = INT_LIMBS * 8;

#[derive(Default)]
pub struct MockTranscript {
    pub counter: i64,
}

impl Transcript for MockTranscript {
    fn get_encoding_element<const L: usize>(&mut self) -> Int<L> {
        self.counter += 1;
        Int::from(self.counter)
    }

    fn get_u64(&mut self) -> u64 {
        self.counter += 1;
        self.counter as Word
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
