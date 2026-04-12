//! Lookup specification types.
//!
//! Pure data types describing which trace columns need lookup verification
//! and against which table type. These live in `zinc-uair` because they
//! belong to the AIR's structural interface — UAIRs declare them as part
//! of [`UairSignature`] via `UairSignature::new(..., lookup_specs)`, the
//! same way shifts are declared.

use zinc_utils::div;

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

impl LookupTableType {
    pub fn width(&self) -> usize {
        match self {
            Self::BitPoly { width, .. } | Self::Word { width, .. } => *width,
        }
    }

    pub fn chunk_width(&self) -> Option<usize> {
        match self {
            Self::BitPoly { chunk_width, .. } | Self::Word { chunk_width, .. } => *chunk_width,
        }
    }

    pub fn is_decomposed(&self) -> bool {
        self.chunk_width().is_some()
    }

    /// Number of chunks per column. Returns 1 for non-decomposed.
    pub fn num_chunks(&self) -> usize {
        match self.chunk_width() {
            Some(cw) => div!(self.width(), cw),
            None => 1,
        }
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
