//! UAIR description tools.

pub mod collect_scalars;
pub mod constraint_counter;
pub mod degree_counter;
pub mod do_nothing_builder;
pub mod dummy_semiring;
pub mod ideal;
pub mod ideal_collector;

use crypto_primitives::Semiring;
use zinc_utils::{UNCHECKED, from_ref::FromRef, mul_by_scalar::MulByScalar};

use crate::ideal::{Ideal, IdealCheck};

/// The abstract interface to constraint building logic.
/// In essence it allows to create constraints modulo ideals.
pub trait ConstraintBuilder {
    /// The expressions the constraint builder operates on.
    /// It is opaque from the PoV of an AIR apart from
    /// the fact that arithmetic operations are available on it
    /// and one can check if an expression is in an ideal.
    type Expr: Semiring;
    /// The type of ideals used by the constraint builder.
    type Ideal: Ideal + IdealCheck<Self::Expr>;

    /// Add a constraint saying that `expr` belongs to the ideal `ideal`.
    fn assert_in_ideal(&mut self, expr: Self::Expr, ideal: &Self::Ideal);

    /// Add a constraint saying that `expr` is equal to zero which is
    /// the same as saying that `expr` belongs to the zero ideal.
    fn assert_zero(&mut self, expr: Self::Expr);
}

/// Specifies a shifted column: a view of a normal (committed) column
/// shifted by a constant number of rows.
///
/// For example, `ShiftSpec { source_col: 0, shift_amount: 3 }` means
/// "create a virtual column whose row `i` is the value of column 0 at
/// row `i - 3` (zero for `i < 3`)."
///
/// Multiple `ShiftSpec`s may reference the same `source_col` with
/// different shift amounts.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ShiftSpec {
    /// Index of the normal (committed) column in the flattened trace
    /// (`binary_poly ++ arbitrary_poly ++ int`, i.e. the same indexing
    /// used by `TraceRow::from_slice_with_signature`).
    pub source_col: usize,
    /// Number of rows to shift by (must be > 0).
    pub shift_amount: usize,
}

/// The signature of a UAIR.
/// Contains the number of columns of
/// each of the types: binary polynomials,
/// polynomials with arbitrary coefficients,
/// and integers.
pub struct UairSignature {
    /// Number of columns with binary polynomial elements.
    pub binary_poly_cols: usize,
    /// Number of columns with arbitrary polynomial elements.
    pub arbitrary_poly_cols: usize,
    /// Number of columns with integers.
    pub int_cols: usize,
    /// Shifted columns used by this UAIR's constraints.
    ///
    /// Each entry declares a virtual column that is a shift of a
    /// normal column.  The order here defines the indexing of the
    /// `down` `TraceRow` passed to `constrain_general`:
    /// `down.binary_poly[i]` corresponds to `shifts[i]` (for
    /// `i < shifts.len()`; beyond that the row is empty).
    ///
    /// When this is empty, the legacy behaviour applies: `down` is
    /// a blanket shift-by-1 of all normal columns.
    pub shifts: Vec<ShiftSpec>,
    /// Indices (in the flattened trace layout: `binary_poly ++ arbitrary_poly
    /// ++ int`) of columns designated as **public inputs**.
    ///
    /// Public columns participate in constraints exactly like private ones.
    /// The difference surfaces at evaluation-claim time: the prover does
    /// *not* include their MLE evaluations in the proof — instead, the
    /// verifier computes those evaluations itself from the known public data.
    ///
    /// When empty, all columns are treated as private (the default).
    pub public_columns: Vec<usize>,
}

impl UairSignature {
    /// Maximum number of columns across the three types.
    pub fn max_cols(&self) -> usize {
        [
            self.binary_poly_cols,
            self.arbitrary_poly_cols,
            self.int_cols,
        ]
        .into_iter()
        .max()
        .expect("the iterator is not empty")
    }

    /// The sum of the numbers of columns across
    /// all types.
    #[allow(clippy::arithmetic_side_effects)] // we don't have that many columns
    pub fn total_cols(&self) -> usize {
        self.binary_poly_cols + self.arbitrary_poly_cols + self.int_cols
    }

    /// Number of shifted (virtual) columns.
    pub fn num_shifted_cols(&self) -> usize {
        self.shifts.len()
    }

    /// Whether the UAIR uses the legacy blanket shift-by-1 mode
    /// (no explicit shift specs).
    pub fn uses_legacy_shifts(&self) -> bool {
        self.shifts.is_empty()
    }

    /// Number of columns in the `down` row.
    ///
    /// - Legacy mode (empty shifts): same as `total_cols()`.
    /// - Shift-spec mode: `shifts.len()`.
    pub fn down_total_cols(&self) -> usize {
        if self.shifts.is_empty() {
            self.total_cols()
        } else {
            self.shifts.len()
        }
    }

    /// Number of private (non-public) columns.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn num_private_cols(&self) -> usize {
        self.total_cols() - self.public_columns.len()
    }

    /// Whether the given flattened column index is a public column.
    pub fn is_public_column(&self, col_idx: usize) -> bool {
        self.public_columns.contains(&col_idx)
    }

    /// Whether the shift at the given index sources a public column.
    pub fn is_public_shift(&self, shift_idx: usize) -> bool {
        self.shifts
            .get(shift_idx)
            .is_some_and(|spec| self.is_public_column(spec.source_col))
    }

    /// Signature describing the layout of the `down` (shifted) row.
    ///
    /// In **legacy** mode the layout mirrors the `up` row (all columns
    /// shifted by 1).
    ///
    /// In **shift-spec** mode each `ShiftSpec` contributes one column
    /// to `down`, categorised by the type of the source column:
    ///   - source in the binary-poly range  → `binary_poly`
    ///   - source in the arbitrary-poly range → `arbitrary_poly`
    ///   - source in the int range           → `int`
    ///
    /// The ordering within each bucket follows the order of `shifts`.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn down_signature(&self) -> UairSignature {
        if self.shifts.is_empty() {
            UairSignature {
                binary_poly_cols: self.binary_poly_cols,
                arbitrary_poly_cols: self.arbitrary_poly_cols,
                int_cols: self.int_cols,
                shifts: vec![],
                public_columns: vec![],
            }
        } else {
            let bp_end = self.binary_poly_cols;
            let ap_end = bp_end + self.arbitrary_poly_cols;
            let mut bp = 0usize;
            let mut ap = 0usize;
            let mut ic = 0usize;
            for spec in &self.shifts {
                if spec.source_col < bp_end {
                    bp += 1;
                } else if spec.source_col < ap_end {
                    ap += 1;
                } else {
                    ic += 1;
                }
            }
            UairSignature {
                binary_poly_cols: bp,
                arbitrary_poly_cols: ap,
                int_cols: ic,
                shifts: vec![],
                public_columns: vec![],
            }
        }
    }
}

/// A view on a row of the trace.
/// Contains references to cells of the trace
/// of all types lying in the same trace row.
#[derive(Clone, Copy)]
pub struct TraceRow<'a, Expr> {
    pub binary_poly: &'a [Expr],
    pub arbitrary_poly: &'a [Expr],
    pub int: &'a [Expr],
}

impl<'a, Expr> TraceRow<'a, Expr> {
    /// Given a slice that represents a raw row of the trace,
    /// creates a `TraceRow` from it.
    /// Subdivides the slice according to the given signature `signature`.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn from_slice_with_signature(row: &'a [Expr], signature: &UairSignature) -> Self {
        Self {
            binary_poly: &row[0..signature.binary_poly_cols],
            arbitrary_poly: &row[signature.binary_poly_cols
                ..signature.binary_poly_cols + signature.arbitrary_poly_cols],
            int: &row[signature.binary_poly_cols + signature.arbitrary_poly_cols..],
        }
    }
}

/// The trait that a universal AIR description has to implement.
/// This must include all the constraint description logic of an UAIR.
///
/// One type might implement different UAIR logics for different underlying
/// semirings hence the generic type parameter.
pub trait Uair {
    /// The ideal type the AIR operates with.
    /// Since a `ConstraintBuilder` is "opaque" for a `Uair`
    /// a `Uair` has to have a means to create ideals
    /// so ideals are fixed by this associated types.
    /// At the `constrain*` methods a `Uair` is given
    /// a way to convert its own ideals into builder's ideals
    /// via the `FromRef` trait.
    type Ideal: Ideal;

    /// The type of scalars of the UAIR.
    /// For now, we assume they are of
    /// the type "arbitrary polynomials".
    type Scalar: Semiring;

    /// Signature of the UAIR.
    fn signature() -> UairSignature;

    /// A general method for describing constraints.
    ///
    /// # Arguments
    /// - `b`: a builder encapsulating the constraint storing logic. Its type
    ///   `B` has to have compatible `B::Ideal` with the `Self::Ideal`, i.e. it
    ///   must implement `FromRef<Self::Ideal>` trait.
    /// - `up`: a `TraceRow` of expressions representing the current row of
    ///   UAIR.
    /// - `down`: a `TraceRow` of expressions representing the next row of UAIR.
    ///   It is safe to assume all the members have the same lengths as
    ///   corresponding members of `up`.
    /// - `from_ref`: a closure that turns the underlying ring `R` into
    ///   `B::Expr`. Sometimes (e.g. when dealing with random fields) it is
    ///   convenient to provide a closure instead of a `FromRef` implementation.
    /// - `mbs`: a closure that allows to multiply expressions by `R`. Same
    ///   rationale as for `from_ref`.
    fn constrain_general<B, FromR, MulByScalar, IFromR>(
        b: &mut B,
        up: TraceRow<B::Expr>,
        down: TraceRow<B::Expr>,
        from_ref: FromR,
        mbs: MulByScalar,
        ideal_from_ref: IFromR,
    ) where
        B: ConstraintBuilder,
        FromR: Fn(&Self::Scalar) -> B::Expr,
        MulByScalar: Fn(&B::Expr, &Self::Scalar) -> Option<B::Expr>,
        IFromR: Fn(&Self::Ideal) -> B::Ideal;

    // Same as `constrain_general` but `from_ref` and `mbs`
    // come from the trait implementations.
    fn constrain<B>(b: &mut B, up: TraceRow<B::Expr>, down: TraceRow<B::Expr>)
    where
        B: ConstraintBuilder,
        B::Expr: FromRef<Self::Scalar> + for<'b> MulByScalar<&'b Self::Scalar>,
        B::Ideal: FromRef<Self::Ideal>,
    {
        Self::constrain_general(
            b,
            up,
            down,
            B::Expr::from_ref,
            |x, y| B::Expr::mul_by_scalar::<UNCHECKED>(x, y),
            B::Ideal::from_ref,
        )
    }
}
