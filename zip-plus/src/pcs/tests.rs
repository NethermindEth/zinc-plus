#![allow(
    clippy::arithmetic_side_effects,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]

use crate::traits::{Transcribable, Transcript};

#[derive(Default)]
pub struct MockTranscript {
    pub counter: i64,
}

impl Transcript for MockTranscript {
    fn get_challenge<T: Transcribable>(&mut self) -> T {
        self.counter += 1;
        let mut bytes = vec![0u8; T::NUM_BYTES];
        let counter_bytes = self.counter.to_le_bytes();
        bytes[..counter_bytes.len()].copy_from_slice(&counter_bytes);
        T::read_transcription_bytes(&bytes)
    }
}
