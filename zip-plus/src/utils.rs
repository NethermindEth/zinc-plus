use crate::pcs_transcript::PcsProverTranscript;
use rand::{rngs::StdRng, seq::SliceRandom};
use rand_core::SeedableRng;
use zinc_transcript::traits::Transcribable;

/// Reorder the elements in slice using the given randomness seed
pub(super) fn shuffle_seeded<T>(slice: &mut [T], seed: u64) {
    let mut rng = StdRng::seed_from_u64(seed);
    slice.shuffle(&mut rng);
}

/// Formats a number with spaces as thousands separators, e.g. 1234567 becomes
/// "1 234 567".
#[allow(clippy::unwrap_used)]
fn fmt_thousands(n: usize) -> String {
    let s = n.to_string();
    s.as_bytes()
        .rchunks(3)
        .rev()
        .map(|c| std::str::from_utf8(c).unwrap())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Prints proof size (before and after compression) to stderr.
pub fn eprint_proof_size(label: impl std::fmt::Display, proof: &impl Transcribable) {
    let mut transcript = PcsProverTranscript::new_from_commitments(std::iter::empty());
    transcript
        .write(proof)
        .expect("transcribing proof should not fail");
    let raw = transcript.stream.into_inner();

    eprint_bytes_size(label, &raw);
}

/// Prints byte slice size (before and after compression) to stderr.
pub fn eprint_bytes_size(label: impl std::fmt::Display, raw: &[u8]) {
    macro_rules! print {
        ($details:expr, $size_bytes:expr) => {
            eprintln!(
                "    Proof size ({label}, {}): {} bytes ({} KiB)",
                $details,
                fmt_thousands($size_bytes),
                $size_bytes.div_ceil(1024),
            );
        };
    }
    print!("raw", raw.len());

    let zstd = zstd::encode_all(raw, 22).expect("zstd compression failed");
    print!("zstd-22", zstd.len());
}
