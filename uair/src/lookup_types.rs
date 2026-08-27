//! Lookup specification types.
//!
//! Pure data types describing which trace columns need lookup verification
//! and against which table type. These live in `zinc-uair` because they
//! belong to the AIR's structural interface — UAIRs declare them as part
//! of [`UairSignature`] via `UairSignature::new(..., lookup_specs)`, the
//! same way shifts are declared.

use zinc_transcript::traits::{ConstTranscribable, GenTranscribable, Transcribable};
use zinc_utils::{add, mul};

/// Describes the type of lookup table a column should be checked against.
/// For the width-indexed types the full table size is `2^width`,
/// decomposed into chunks of `chunk_width`.
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
    /// The exact multiset a column must hold: each of `values` once, and
    /// `pad` in every remaining row. Where the other types say what a
    /// cell may be, this says what the whole column is, so a permutation
    /// of a small set is one lookup rather than a degree-|values|
    /// polynomial identity.
    Prescribed { values: Vec<u64>, pad: u64 },
}

impl LookupTableType {
    /// Whether a group of this type reads integer columns. A `BitPoly`
    /// group's parents are binary polynomials; every other table is
    /// indexed by a number, so its parents are integers.
    pub fn reads_int_columns(&self) -> bool {
        !matches!(self, Self::BitPoly { .. })
    }
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
// Transcribable: a width-indexed type is ten bytes.
//   [discriminant: u8] [width: u32] [chunk_present: u8] [chunk_width: u32]
// A prescribed table is as long as the multiset it names.
//   [discriminant: u8] [count: u32] [values: count × u64] [pad: u64]
// ---------------------------------------------------------------------------

const WIDTH_INDEXED_BYTES: usize = 1 + 4 + 1 + 4;
const PRESCRIBED_HEAD_BYTES: usize = 1 + 4;
const PRESCRIBED_DISCRIMINANT: u8 = 2;

impl GenTranscribable for LookupTableType {
    fn read_transcription_bytes_exact(bytes: &[u8]) -> Self {
        if bytes[0] == PRESCRIBED_DISCRIMINANT {
            let count = usize::try_from(u32::read_transcription_bytes_exact(&bytes[1..5]))
                .expect("value count must fit in usize");
            let mut numbers = bytes[PRESCRIBED_HEAD_BYTES..]
                .chunks_exact(u64::NUM_BYTES)
                .map(u64::read_transcription_bytes_exact);
            let values: Vec<u64> = numbers.by_ref().take(count).collect();
            let pad = numbers
                .next()
                .expect("a prescribed table transcribes its pad");
            return Self::Prescribed { values, pad };
        }
        assert_eq!(bytes.len(), WIDTH_INDEXED_BYTES);
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
        assert_eq!(buf.len(), self.get_num_bytes());
        let (discriminant, width, chunk_width) = match self {
            Self::BitPoly { width, chunk_width } => (0u8, *width, *chunk_width),
            Self::Word { width, chunk_width } => (1u8, *width, *chunk_width),
            Self::Prescribed { values, pad } => {
                buf[0] = PRESCRIBED_DISCRIMINANT;
                u32::write_transcription_bytes_exact(
                    &(u32::try_from(values.len()).expect("value count must fit in u32")),
                    &mut buf[1..5],
                );
                for (chunk, number) in buf[PRESCRIBED_HEAD_BYTES..]
                    .chunks_exact_mut(u64::NUM_BYTES)
                    .zip(values.iter().chain(std::iter::once(pad)))
                {
                    u64::write_transcription_bytes_exact(number, chunk);
                }
                return;
            }
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

impl Transcribable for LookupTableType {
    fn get_num_bytes(&self) -> usize {
        match self {
            Self::Prescribed { values, .. } => add!(
                PRESCRIBED_HEAD_BYTES,
                mul!(add!(values.len(), 1), u64::NUM_BYTES)
            ),
            _ => WIDTH_INDEXED_BYTES,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The width-indexed types transcribe as the ten bytes they always
    /// did: a variable-length variant beside them moves none of theirs.
    #[test]
    fn a_width_indexed_table_is_ten_bytes() {
        for (table, expected) in [
            (
                LookupTableType::Word { width: 16, chunk_width: Some(8) },
                [1, 16, 0, 0, 0, 1, 8, 0, 0, 0],
            ),
            (
                LookupTableType::BitPoly { width: 32, chunk_width: None },
                [0, 32, 0, 0, 0, 0, 0, 0, 0, 0],
            ),
        ] {
            let mut buf = vec![0u8; table.get_num_bytes()];
            table.write_transcription_bytes_exact(&mut buf);
            assert_eq!(buf, expected);
            assert_eq!(LookupTableType::read_transcription_bytes_exact(&buf), table);
        }
    }

    /// A prescribed table carries the multiset it names, however long
    /// that is, and reads back as itself.
    #[test]
    fn a_prescribed_table_round_trips() {
        for table in [
            LookupTableType::Prescribed { values: (1..=9).collect(), pad: 0 },
            LookupTableType::Prescribed { values: vec![], pad: u64::MAX },
        ] {
            let mut buf = vec![0u8; add!(LookupTableType::LENGTH_NUM_BYTES, table.get_num_bytes())];
            table.write_transcription_bytes_subset(&mut buf);
            let (read, rest) = LookupTableType::read_transcription_bytes_subset(&buf);
            assert_eq!(read, table);
            assert!(rest.is_empty());
        }
    }
}
