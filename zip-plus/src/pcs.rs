use zinc_transcript::KeccakTranscript;

use crate::pcs_transcript::PcsTranscript;
use std::io::Cursor;

mod phase_commit;
mod phase_evaluate;
mod phase_test;
mod phase_verify;
pub mod structs;
#[cfg(test)]
pub(crate) mod test_utils;
pub(crate) mod utils;

/// Proof obtained by the verifier after the testing phase, which is essentially
/// a serialized prover transcript.
/// Evaluation phase continues this transcript to produce the final proof.
#[derive(Clone, Debug)]
pub struct ZipPlusTestTranscript(PcsTranscript);

impl From<PcsTranscript> for ZipPlusTestTranscript {
    fn from(transcript: PcsTranscript) -> Self {
        ZipPlusTestTranscript(transcript)
    }
}

impl From<ZipPlusTestTranscript> for PcsTranscript {
    fn from(value: ZipPlusTestTranscript) -> Self {
        value.0
    }
}

/// Proof obtained by the verifier after the evaluation phase, which is
/// essentially a serialized prover transcript.
#[derive(Clone, Debug)]
pub struct ZipPlusProof(Vec<u8>);

impl From<PcsTranscript> for ZipPlusProof {
    fn from(transcript: PcsTranscript) -> Self {
        ZipPlusProof(transcript.stream.into_inner())
    }
}

impl From<ZipPlusProof> for PcsTranscript {
    fn from(proof: ZipPlusProof) -> Self {
        Self {
            fs_transcript: KeccakTranscript::default(),
            stream: Cursor::new(proof.0),
        }
    }
}
