//! Lookup specification types.
//!
//! Pure data types describing which trace columns need lookup verification
//! and against which table type. These live in `zinc-uair` because they
//! belong to the AIR's structural interface — UAIRs declare them as part
//! of [`UairSignature`] via `UairSignature::new(..., lookup_specs)`, the
//! same way shifts are declared.

/// Describes the type of lookup table a column should be checked against.
/// Full table size = `2^width` for each type; decomposed into chunks of
/// `chunk_width`
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum LookupTableType {
    /// Binary polynomials of degree less than `width`, projected at element
    /// `a`.
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
