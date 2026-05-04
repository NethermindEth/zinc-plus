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

/// Compression level used by [`eprint_proof_size`] and friends.
/// zstd-3 picked over -22 because at our proof shapes (~100-500 KB)
/// level 19/22 buys essentially no extra ratio over level 3 once the
/// main long-range patterns are caught, while costing 30-100x more
/// CPU. See conversation notes; can be raised again if storage cost
/// ever dominates compression cost.
pub const ZSTD_LEVEL: i32 = 3;

/// Compress `value` with zstd at the configured [`ZSTD_LEVEL`] and
/// return both the compressed bytes and the wall-clock spent on the
/// compression step (excluding serialization). Useful for callers
/// that want to attribute the compression cost to a step in a
/// timings breakdown.
pub fn serialize_and_compress<T: Transcribable>(
    value: &T,
) -> (Vec<u8>, std::time::Duration) {
    let mut buf = vec![0_u8; value.get_num_bytes()];
    value.write_transcription_bytes_exact(&mut buf);
    let t0 = std::time::Instant::now();
    let compressed = zstd::encode_all(&buf[..], ZSTD_LEVEL).expect("zstd compression failed");
    (compressed, t0.elapsed())
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

/// Prints byte slice size (raw + zstd) and the compression / decompression
/// wall-clock to stderr. Compression level is [`ZSTD_LEVEL`].
pub fn eprint_bytes_size(label: impl std::fmt::Display, raw: &[u8]) {
    macro_rules! print_size {
        ($details:expr, $size_bytes:expr) => {
            eprintln!(
                "    Proof size ({label}, {}): {} bytes ({} KiB)",
                $details,
                fmt_thousands($size_bytes),
                $size_bytes.div_ceil(1024),
            );
        };
    }
    print_size!("raw", raw.len());

    let compressed = zstd::encode_all(raw, ZSTD_LEVEL).expect("zstd compression failed");
    print_size!(format_args!("zstd-{ZSTD_LEVEL}"), compressed.len());

    let decompressed = zstd::decode_all(&compressed[..]).expect("zstd decompression failed");
    assert_eq!(decompressed.len(), raw.len(), "zstd round-trip size mismatch");
}

/// Prints a per-part proof size breakdown (raw + zstd-compressed) to stderr.
/// Each part is compressed independently at [`ZSTD_LEVEL`]; the printed
/// totals are the sum of the per-part sizes, which slightly differ from
/// compressing the whole proof as one blob (zstd loses some cross-part
/// redundancy when split).
pub fn eprint_bytes_size_breakdown(label: impl std::fmt::Display, parts: &[(&str, &[u8])]) {
    let mut total_raw: usize = 0;
    let mut total_zstd: usize = 0;
    let mut rows: Vec<(String, usize, usize)> = Vec::with_capacity(parts.len());
    for (name, raw) in parts {
        let zstd = zstd::encode_all(*raw, ZSTD_LEVEL).expect("zstd compression failed");
        total_raw = total_raw.saturating_add(raw.len());
        total_zstd = total_zstd.saturating_add(zstd.len());
        rows.push(((*name).to_string(), raw.len(), zstd.len()));
    }

    let total_raw_f = total_raw.max(1) as f64;
    let total_zstd_f = total_zstd.max(1) as f64;

    let zstd_header = format!("zstd-{ZSTD_LEVEL} B");
    eprintln!("    Proof size breakdown ({label}):");
    eprintln!(
        "      {:<22} {:>14} {:>7}    {:>14} {:>7}",
        "component", "raw bytes", "raw%", zstd_header, "zstd%",
    );
    for (name, raw, zstd) in &rows {
        eprintln!(
            "      {:<22} {:>14} {:>6.1}%    {:>14} {:>6.1}%",
            name,
            fmt_thousands(*raw),
            100.0 * (*raw as f64) / total_raw_f,
            fmt_thousands(*zstd),
            100.0 * (*zstd as f64) / total_zstd_f,
        );
    }
    eprintln!(
        "      {:<22} {:>14} {:>7}    {:>14} {:>7}",
        "TOTAL (sum of parts)",
        fmt_thousands(total_raw),
        "",
        fmt_thousands(total_zstd),
        "",
    );
}
