#![allow(
    clippy::arithmetic_side_effects,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]

use crypto_bigint::Word;
use crypto_primitives::crypto_bigint_int::Int;
use std::{collections::BTreeSet, ops::Range};

use crate::traits::Transcript;

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
