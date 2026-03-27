use crate::pcs_transcript::PcsProverTranscript;
use rand::{rngs::StdRng, seq::SliceRandom};
use rand_core::SeedableRng;
use std::io::Write;
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

/// Prints proof size to stderr in a consistent format across all benchmarks.
pub fn eprint_proof_size(label: impl std::fmt::Display, proof: &impl Transcribable) {
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
    let mut transcript = PcsProverTranscript::new_from_commitments(std::iter::empty());
    transcript
        .write(proof)
        .expect("transcribing proof should not fail");
    let raw = transcript.stream.into_inner();
    print!("raw", raw.len());

    let mut gzip_buf = Vec::new();
    let mut gz = flate2::write::GzEncoder::new(&mut gzip_buf, flate2::Compression::best());
    gz.write_all(&raw).expect("gzip compression failed");
    gz.finish().expect("gzip compression failed");
    print!("gzip-best", gzip_buf.len());

    let zstd = zstd::encode_all(raw.as_slice(), 22).expect("zstd compression failed");
    print!("zstd-22", zstd.len());

    let brotli_params = brotli::enc::BrotliEncoderParams {
        quality: 11,
        ..Default::default()
    };
    let mut brotli_buf = Vec::new();
    brotli::BrotliCompress(&mut raw.as_slice(), &mut brotli_buf, &brotli_params)
        .expect("brotli compression failed");
    print!("brotli-11", brotli_buf.len());
}
