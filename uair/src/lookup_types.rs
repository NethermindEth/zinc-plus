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

/// Affine combination of projected-trace columns, applied row-by-row:
///
/// ```text
///   expr(row) = Σ_{(idx, c) in terms} c · projected_trace[idx][row] + constant
/// ```
///
/// Used by [`LookupColumnSpec`] to express the lookup witness as a linear
/// combination of existing committed columns rather than a single column.
/// This is what lets the verifier enforce identities like
/// `a + b + c − 2·Maj ∈ {0,1}^{<W}[X]` without the prover committing the
/// combination as a separate column.
///
/// Column indices are into the full projected trace (public + witness,
/// concatenated in the order `bin || arb || int`). Coefficients are
/// signed `i64`; the runtime lifts them to the target field via the field
/// config.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AffineExpr {
    pub terms: Vec<(usize, i64)>,
    pub constant: i64,
}

impl AffineExpr {
    /// `1 · col_idx + 0` — shorthand for "the single column at `col_idx`".
    /// Used by every lookup spec that doesn't need a real combination.
    pub fn single(col_idx: usize) -> Self {
        Self {
            terms: vec![(col_idx, 1)],
            constant: 0,
        }
    }

    /// Iterate over `(column_index, coefficient)` pairs in the
    /// combination. Empty iterator for the constant-only case.
    pub fn terms(&self) -> impl Iterator<Item = (usize, i64)> + '_ {
        self.terms.iter().copied()
    }
}

/// Specifies that a linear combination of trace columns should be looked
/// up against a prescribed table.
#[derive(Clone, Debug)]
pub struct LookupColumnSpec {
    /// Affine combination of columns whose per-row value the lookup
    /// argument constrains.
    pub expression: AffineExpr,
    /// The lookup table type this expression's row values should land in.
    pub table_type: LookupTableType,
}

impl LookupColumnSpec {
    /// Shorthand for the common case "the single witness column
    /// `col_idx` must lie in `table_type`".
    pub fn column(col_idx: usize, table_type: LookupTableType) -> Self {
        Self {
            expression: AffineExpr::single(col_idx),
            table_type,
        }
    }
}
