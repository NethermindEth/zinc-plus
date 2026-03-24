//! UAIR description tools.

pub mod collect_scalars;
pub mod constraint_counter;
pub mod degree_counter;
pub mod do_nothing_builder;
pub mod dummy_semiring;
pub mod ideal;
pub mod ideal_collector;

use crypto_primitives::Semiring;
use std::borrow::Cow;
use zinc_poly::{
    mle::DenseMultilinearExtension,
    univariate::{binary::BinaryPoly, dense::DensePolynomial},
};
use zinc_utils::{UNCHECKED, add, from_ref::FromRef, mul_by_scalar::MulByScalar};

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

/// Specifies a shifted column
/// `ShiftSpec { source_col: 0, shift_amount: 3 }` means
/// "virtual column whose row i is the value of column 0 at row i+3
/// (zero-padded beyond trace length)."
///
/// Multiple ShiftSpecs may reference the same source_col with
/// different shift amounts.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ShiftSpec {
    /// Index of the committed column in the flattened trace
    /// (binary_poly || arbitrary_poly || int, same indexing as
    /// TraceRow::from_slice_with_layout).
    source_col: usize,
    /// Number of rows to shift by.
    shift_amount: usize,
}

impl ShiftSpec {
    pub fn new(source_col: usize, shift_amount: usize) -> Self {
        assert!(shift_amount > 0, "shift must be non-zero");
        Self {
            source_col,
            shift_amount,
        }
    }

    pub fn source_col(&self) -> usize {
        self.source_col
    }

    pub fn shift_amount(&self) -> usize {
        self.shift_amount
    }
}

// ---------------------------------------------------------------------------
// Column layout types
// ---------------------------------------------------------------------------

/// Column counts per type (binary_poly, arbitrary_poly, int).
/// Shared internals for the semantic newtype wrappers (Total, Public, Virtual,
/// Witness)
#[derive(Clone, Debug, Default)]
pub struct ColumnLayout {
    binary_poly_cols: usize,
    arbitrary_poly_cols: usize,
    int_cols: usize,
}

impl ColumnLayout {
    pub fn new(binary_poly_cols: usize, arbitrary_poly_cols: usize, int_cols: usize) -> Self {
        Self {
            binary_poly_cols,
            arbitrary_poly_cols,
            int_cols,
        }
    }

    pub fn binary_poly_cols(&self) -> usize {
        self.binary_poly_cols
    }

    pub fn arbitrary_poly_cols(&self) -> usize {
        self.arbitrary_poly_cols
    }

    pub fn int_cols(&self) -> usize {
        self.int_cols
    }

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

    /// The sum of the numbers of columns across all types.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn cols(&self) -> usize {
        self.binary_poly_cols + self.arbitrary_poly_cols + self.int_cols
    }
}

macro_rules! column_layout_wrapper {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(Clone, Debug, Default)]
        pub struct $name(ColumnLayout);

        impl $name {
            pub fn new(binary_poly_cols: usize, arbitrary_poly_cols: usize, int_cols: usize) -> Self {
                Self(ColumnLayout::new(binary_poly_cols, arbitrary_poly_cols, int_cols))
            }

            pub fn binary_poly_cols(&self) -> usize { self.0.binary_poly_cols() }
            pub fn arbitrary_poly_cols(&self) -> usize { self.0.arbitrary_poly_cols() }
            pub fn int_cols(&self) -> usize { self.0.int_cols() }
            pub fn max_cols(&self) -> usize { self.0.max_cols() }
            pub fn cols(&self) -> usize { self.0.cols() }
            pub fn as_column_layout(&self) -> &ColumnLayout { &self.0 }
        }
    };
}

column_layout_wrapper!(/// Layout of all trace columns (public + witness) per type.
    TotalColumnLayout);
column_layout_wrapper!(/// Layout of the public column subset.
    PublicColumnLayout);
column_layout_wrapper!(/// Layout of the virtual (shifted/down) columns.
    VirtualColumnLayout);
column_layout_wrapper!(/// Layout of the witness (total minus public) columns.
    WitnessColumnLayout);

// ---------------------------------------------------------------------------
// UairSignature
// ---------------------------------------------------------------------------

/// The signature of a UAIR.
///
/// Public columns precede witness columns within each type group.
/// The flattened trace ordering is:
/// `[pub_bin, wit_bin, pub_arb, wit_arb, pub_int, wit_int]`.
pub struct UairSignature {
    /// Column-type layout of all (public + witness) columns.
    total_cols: TotalColumnLayout,
    /// Public column subset.
    public_cols: PublicColumnLayout,
    /// Shifted columns info sorted by `source_col`.
    shifts: Vec<ShiftSpec>,
    /// Column-type layout of the shifted (down) row.
    down_cols: VirtualColumnLayout,
}

impl UairSignature {
    /// Create a new signature, sorting `shifts` by `source_col`.
    pub fn new(
        total_cols: TotalColumnLayout,
        public_cols: PublicColumnLayout,
        mut shifts: Vec<ShiftSpec>,
    ) -> Self {
        for (name, pub_n, tot_n) in [
            (
                "binary_poly",
                public_cols.binary_poly_cols(),
                total_cols.binary_poly_cols(),
            ),
            (
                "arbitrary_poly",
                public_cols.arbitrary_poly_cols(),
                total_cols.arbitrary_poly_cols(),
            ),
            ("int", public_cols.int_cols(), total_cols.int_cols()),
        ] {
            assert!(
                pub_n <= tot_n,
                "public {name}_cols ({pub_n}) > total ({tot_n})"
            );
        }

        let num_cols = total_cols.cols();
        for spec in &shifts {
            assert!(
                spec.source_col() < num_cols,
                "ShiftSpec source_col {} out of range (total_cols = {}). \
                 source_col uses flat indexing: binary_poly || arbitrary_poly || int.",
                spec.source_col(),
                num_cols,
            );
        }

        shifts.sort_by_key(|spec| spec.source_col());
        let down_cols = Self::compute_down_layout(&total_cols, &shifts);
        Self {
            total_cols,
            public_cols,
            shifts,
            down_cols,
        }
    }

    fn compute_down_layout(
        total_cols: &TotalColumnLayout,
        shifts: &[ShiftSpec],
    ) -> VirtualColumnLayout {
        let bp_end = total_cols.binary_poly_cols();
        let ap_end = add!(bp_end, total_cols.arbitrary_poly_cols());
        let mut bp = 0usize;
        let mut ap = 0usize;
        let mut ic = 0usize;
        for spec in shifts {
            if spec.source_col() < bp_end {
                bp = add!(bp, 1);
            } else if spec.source_col() < ap_end {
                ap = add!(ap, 1);
            } else {
                ic = add!(ic, 1);
            }
        }
        VirtualColumnLayout::new(bp, ap, ic)
    }

    pub fn total_cols(&self) -> &TotalColumnLayout {
        &self.total_cols
    }

    pub fn public_cols(&self) -> &PublicColumnLayout {
        &self.public_cols
    }

    pub fn shifts(&self) -> &[ShiftSpec] {
        &self.shifts
    }

    /// Column-type layout of the shifted (down) row.
    pub fn down_cols(&self) -> &VirtualColumnLayout {
        &self.down_cols
    }

    /// Build correctly-sized dummy up and down `TraceRow`s for static
    /// analysis (constraint counting, degree counting, scalar/ideal
    /// collection).
    pub fn dummy_rows<T: Clone>(&self, val: T) -> (Vec<T>, Vec<T>) {
        let up_size = self.total_cols.cols();
        let down_size = self.down_cols.cols();
        (vec![val.clone(); up_size], vec![val; down_size])
    }

    /// Witness column counts (total minus public) per type.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn witness_cols(&self) -> WitnessColumnLayout {
        WitnessColumnLayout::new(
            self.total_cols.binary_poly_cols() - self.public_cols.binary_poly_cols(),
            self.total_cols.arbitrary_poly_cols() - self.public_cols.arbitrary_poly_cols(),
            self.total_cols.int_cols() - self.public_cols.int_cols(),
        )
    }
}

// ---------------------------------------------------------------------------
// UairTrace
// ---------------------------------------------------------------------------

/// The trace of a UAIR execution (pre-projection).
/// If owned, it contains the full trace, otherwise it contains a view on the
/// full trace (e.g. only public columns).
#[derive(Debug, Clone, Default)]
pub struct UairTrace<'a, PolyCoeff: Clone, Int: Clone, const D: usize> {
    pub binary_poly: Cow<'a, [DenseMultilinearExtension<BinaryPoly<D>>]>,
    pub arbitrary_poly: Cow<'a, [DenseMultilinearExtension<DensePolynomial<PolyCoeff, D>>]>,
    pub int: Cow<'a, [DenseMultilinearExtension<Int>]>,
}

impl<PolyCoeff: Clone, Int: Clone, const D: usize> UairTrace<'static, PolyCoeff, Int, D> {
    /// Returns a layout containing only public columns.
    pub fn public(&self, sig: &UairSignature) -> UairTrace<'_, PolyCoeff, Int, D> {
        let p = sig.public_cols();
        UairTrace {
            binary_poly: Cow::Borrowed(&self.binary_poly[0..p.binary_poly_cols()]),
            arbitrary_poly: Cow::Borrowed(&self.arbitrary_poly[0..p.arbitrary_poly_cols()]),
            int: Cow::Borrowed(&self.int[0..p.int_cols()]),
        }
    }

    /// Returns layout containing only witness columns.
    pub fn witness(&self, sig: &UairSignature) -> UairTrace<'_, PolyCoeff, Int, D> {
        let p = sig.public_cols();
        UairTrace {
            binary_poly: Cow::Borrowed(&self.binary_poly[p.binary_poly_cols()..]),
            arbitrary_poly: Cow::Borrowed(&self.arbitrary_poly[p.arbitrary_poly_cols()..]),
            int: Cow::Borrowed(&self.int[p.int_cols()..]),
        }
    }
}

// ---------------------------------------------------------------------------
// TraceRow
// ---------------------------------------------------------------------------

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
    /// Subdivides the slice according to the given column layout.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn from_slice_with_layout(row: &'a [Expr], layout: &ColumnLayout) -> Self {
        let bp = layout.binary_poly_cols();
        let ap = layout.arbitrary_poly_cols();
        Self {
            binary_poly: &row[0..bp],
            arbitrary_poly: &row[bp..bp + ap],
            int: &row[bp + ap..],
        }
    }
}

// ---------------------------------------------------------------------------
// Uair trait
// ---------------------------------------------------------------------------

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
    // Note: This is usually Z_32[X] (i.e. DensePolynomial<Ring, 32>), but according
    // to @agareta, this in not always the case.
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
    /// - `down`: a `TraceRow` of expressions representing the shifted (down)
    ///   row of the UAIR. Its layout matches `UairSignature::down()`, which may
    ///   have fewer columns than `up` when only a subset of columns are
    ///   shifted.
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
