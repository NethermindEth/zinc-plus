//! Lookup specification types.
//!
//! Pure data types describing which trace columns need lookup verification
//! and against which table type. These live in `zinc-uair` because they
//! belong to the AIR's structural interface — UAIRs declare them as part
//! of [`UairSignature`] via `UairSignature::new(..., lookup_specs)`, the
//! same way shifts are declared.

use zinc_transcript::traits::{ConstTranscribable, GenTranscribable};

/// Describes the type of lookup table a column should be checked against.
/// Full table size = `2^width` for each type; decomposed into chunks of
/// `chunk_width`
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LookupTableType {
    /// Binary polynomials of degree less than `width`, projected into the prime
    /// field.
    BitPoly {
        width: usize,
        chunk_width: Option<usize>,
    },
    /// Unsigned integers fitting in `width` bits.
    Word {
        width: usize,
        chunk_width: Option<usize>,
    },
}

/// Specifies that a trace column should be looked up against a prescribed
/// table.
#[derive(Clone, Debug)]
pub struct LookupColumnSpec {
    /// 0-based index into the projected field-element trace.
    pub column_index: usize,
    /// The lookup table type this column should be checked against.
    pub table_type: LookupTableType,
}

// ---------------------------------------------------------------------------
// Transcribable: fixed 10-byte encoding.
//   [discriminant: u8] [width: u32] [chunk_present: u8] [chunk_width: u32]
// ---------------------------------------------------------------------------

const LOOKUP_TABLE_TYPE_BYTES: usize = 1 + 4 + 1 + 4;

impl GenTranscribable for LookupTableType {
    fn read_transcription_bytes_exact(bytes: &[u8]) -> Self {
        assert_eq!(bytes.len(), LOOKUP_TABLE_TYPE_BYTES);
        let discriminant = bytes[0];
        let width = usize::try_from(u32::read_transcription_bytes_exact(&bytes[1..5]))
            .expect("width must fit in usize");
        let chunk_present = bytes[5];
        let chunk_width = u32::read_transcription_bytes_exact(&bytes[6..10]);
        let chunk_width = match chunk_present {
            0 => None,
            1 => Some(usize::try_from(chunk_width).expect("chunk_width must fit in usize")),
            v => panic!("invalid chunk_width presence flag: {v}"),
        };
        match discriminant {
            0 => Self::BitPoly { width, chunk_width },
            1 => Self::Word { width, chunk_width },
            v => panic!("invalid LookupTableType discriminant: {v}"),
        }
    }

    fn write_transcription_bytes_exact(&self, buf: &mut [u8]) {
        assert_eq!(buf.len(), LOOKUP_TABLE_TYPE_BYTES);
        let (discriminant, width, chunk_width) = match *self {
            Self::BitPoly { width, chunk_width } => (0u8, width, chunk_width),
            Self::Word { width, chunk_width } => (1u8, width, chunk_width),
        };
        buf[0] = discriminant;
        u32::write_transcription_bytes_exact(
            &(u32::try_from(width).expect("width must fit in u32")),
            &mut buf[1..5],
        );
        buf[5] = if chunk_width.is_some() { 1 } else { 0 };
        let cw = u32::try_from(chunk_width.unwrap_or(0)).expect("chunk_width must fit in u32");
        u32::write_transcription_bytes_exact(&cw, &mut buf[6..10]);
    }
}

impl ConstTranscribable for LookupTableType {
    const NUM_BYTES: usize = LOOKUP_TABLE_TYPE_BYTES;
}
