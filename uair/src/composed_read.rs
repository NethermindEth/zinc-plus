//! Composed-read (pointer query) specification types.
//!
//! Pure data types describing value-addressed reads a UAIR declares:
//! `result_col(x)` equals `value_col` at the cube position spelled by
//! `bit_cols` at `x`. These live in `zinc-uair` because they belong to
//! the AIR's structural interface — UAIRs declare them via
//! `UairSignature::with_composed_reads`, the same way shifts and
//! lookups are declared. Protocol design and soundness preconditions
//! (booleanity of the bit columns, padding read-consistency, range):
//! `documentation/pointer-query-design.md`.

/// Specifies one composed read over int witness columns.
///
/// For every row `x` of the trace cube, the entry of `result_col` must
/// equal the entry of `value_col` at the cube position whose binary
/// spelling (low bit first) the `bit_cols` carry at row `x`. All three
/// use flat column indexing: binary_poly || arbitrary_poly || int.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ComposedReadSpec {
    /// The column read through the pointer.
    pub value_col: usize,
    /// The address-bit columns, low bit first; their length must equal
    /// the trace's `num_vars` (checked at proving time, where the
    /// trace size is known). The AIR must constrain these boolean.
    pub bit_cols: Vec<usize>,
    /// The column carrying the dereferenced values.
    pub result_col: usize,
}
