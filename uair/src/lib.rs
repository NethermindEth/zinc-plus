//! UAIR description tools.

pub mod collect_scalars;
pub mod constraint_counter;
pub mod degree_counter;
pub mod do_nothing_builder;
pub mod dummy_semiring;
pub mod ideal;
pub mod ideal_collector;
pub mod lookup_types;

use crypto_primitives::Semiring;
use std::borrow::Cow;
use zinc_poly::{
    mle::DenseMultilinearExtension,
    univariate::{binary::BinaryPoly, dense::DensePolynomial},
};
use zinc_utils::{UNCHECKED, add, from_ref::FromRef, mul_by_scalar::MulByScalar, sub};

use crate::ideal::{Ideal, IdealCheck};

pub use lookup_types::{LookupColumnSpec, LookupTableType};

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
// BitOpSpec
// ---------------------------------------------------------------------------

/// Bit-wise operation applied to the 32-coefficient F_2[X] interpretation
/// of a binary-poly cell. Both variants require `c < 32`.
///
/// `Rot(c)`: cyclic rotation — `coeffs_out[(i + c) mod 32] = coeffs_in[i]`.
/// `ShiftR(c)`: bitwise right shift — `coeffs_out[i] = coeffs_in[i + c]`
/// for `i < 32 - c`, zero otherwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BitOp {
    Rot(u32),
    ShiftR(u32),
}

impl BitOp {
    /// Construct `Rot(c)` with `c < 32`.
    pub fn rot(c: u32) -> Self {
        assert!(c < 32, "BitOp::rot: c must be < 32, got {c}");
        BitOp::Rot(c)
    }

    /// Construct `ShiftR(c)` with `c < 32`.
    pub fn shift_r(c: u32) -> Self {
        assert!(c < 32, "BitOp::shift_r: c must be < 32, got {c}");
        BitOp::ShiftR(c)
    }

    /// Sort key: discriminate on op kind first (Rot < ShiftR), then on
    /// rotation/shift amount, so that `BitOpSpec` sort is deterministic.
    fn sort_key(&self) -> (u8, u32) {
        match self {
            BitOp::Rot(c) => (0, *c),
            BitOp::ShiftR(c) => (1, *c),
        }
    }
}

/// Specifies a bit-op virtual column.
/// `BitOpSpec { source_col, op }` means
/// "virtual column whose row i is `op` applied bitwise to the cell of
/// `source_col` at row i" (treating each cell as a 32-coefficient F_2[X]
/// polynomial).
///
/// Bit-op virtual columns are MLE-virtual columns referenced by
/// `constrain_general` like row-shift virtual columns, but they reduce
/// differently downstream — they never enter multipoint eval; their
/// consistency is fully discharged in Step 4.5 by checking that the
/// CPR-emitted F-eval matches `ψ(op(lifted_eval[source_col]))`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BitOpSpec {
    /// Index of the source column in the flattened trace.
    source_col: usize,
    /// Bit-op applied to each cell of the source column.
    op: BitOp,
}

impl BitOpSpec {
    pub fn new(source_col: usize, op: BitOp) -> Self {
        Self { source_col, op }
    }

    pub fn source_col(&self) -> usize {
        self.source_col
    }

    pub fn op(&self) -> BitOp {
        self.op
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
    num_binary_poly_cols: usize,
    num_arbitrary_poly_cols: usize,
    num_int_cols: usize,
}

impl ColumnLayout {
    pub fn new(
        num_binary_poly_cols: usize,
        num_arbitrary_poly_cols: usize,
        num_int_cols: usize,
    ) -> Self {
        Self {
            num_binary_poly_cols,
            num_arbitrary_poly_cols,
            num_int_cols,
        }
    }

    pub fn num_binary_poly_cols(&self) -> usize {
        self.num_binary_poly_cols
    }

    pub fn num_arbitrary_poly_cols(&self) -> usize {
        self.num_arbitrary_poly_cols
    }

    pub fn num_int_cols(&self) -> usize {
        self.num_int_cols
    }

    /// Maximum number of columns across the three types.
    pub fn max_cols(&self) -> usize {
        [
            self.num_binary_poly_cols,
            self.num_arbitrary_poly_cols,
            self.num_int_cols,
        ]
        .into_iter()
        .max()
        .expect("the iterator is not empty")
    }

    /// The sum of the numbers of columns across all types.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn cols(&self) -> usize {
        self.num_binary_poly_cols + self.num_arbitrary_poly_cols + self.num_int_cols
    }
}

macro_rules! column_layout_wrapper {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(Clone, Debug, Default)]
        pub struct $name(ColumnLayout);

        impl $name {
            pub fn new(num_binary_poly_cols: usize, num_arbitrary_poly_cols: usize, num_int_cols: usize) -> Self {
                Self(ColumnLayout::new(num_binary_poly_cols, num_arbitrary_poly_cols, num_int_cols))
            }

            pub fn num_binary_poly_cols(&self) -> usize { self.0.num_binary_poly_cols() }
            pub fn num_arbitrary_poly_cols(&self) -> usize { self.0.num_arbitrary_poly_cols() }
            pub fn num_int_cols(&self) -> usize { self.0.num_int_cols() }
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
#[derive(Clone, Debug)]
pub struct UairSignature {
    /// Column-type layout of all (public + witness) columns.
    total_cols: TotalColumnLayout,
    /// Public column subset.
    public_cols: PublicColumnLayout,
    /// Witness column counts (total minus public) per type.
    witness_cols: WitnessColumnLayout,
    /// Shifted columns info sorted by `source_col`.
    shifts: Vec<ShiftSpec>,
    /// Column-type layout of the shifted (down) row.
    down_cols: VirtualColumnLayout,
    /// Lookup specifications: which trace columns are constrained against
    /// which table types.
    lookup_specs: Vec<LookupColumnSpec>,
    /// Witness binary_poly column indices (relative to the witness section
    /// — i.e., 0-based within `binary_poly[num_pub_bin..]`) that the UAIR
    /// asks the protocol to skip from the algebraic booleanity sumcheck.
    /// Use this for columns whose bit-poly nature is already pinned by
    /// other constraints (e.g. shift-decomposition splits where the
    /// equality `W = T + X^k · S` plus booleanity on `W` makes a
    /// separate booleanity check on `T` and `S` redundant). Sorted and
    /// dedup'd by the constructor; entries must be valid witness
    /// binary_poly indices.
    booleanity_skip_indices: Vec<usize>,
    /// Bit-op virtual column specifications. Sorted by
    /// `(source_col, op_kind, c)` for determinism. Bit-op virtual
    /// columns are MLE-virtual columns referenced by
    /// `constrain_general` via `down.bit_op`; they never enter
    /// multipoint eval — their consistency is fully discharged in
    /// Step 4.5.
    bit_op_specs: Vec<BitOpSpec>,
}

impl UairSignature {
    /// Create a new signature, sorting `shifts` by `source_col` and
    /// `bit_op_specs` by `(source_col, op_kind, c)`. No booleanity
    /// skipping by default — every witness binary_poly column is
    /// included in the booleanity sumcheck. Use
    /// [`UairSignature::with_booleanity_skip_indices`] to opt out.
    pub fn new(
        total_cols: TotalColumnLayout,
        public_cols: PublicColumnLayout,
        mut shifts: Vec<ShiftSpec>,
        lookup_specs: Vec<LookupColumnSpec>,
        mut bit_op_specs: Vec<BitOpSpec>,
    ) -> Self {
        for (name, pub_n, tot_n) in [
            (
                "binary_poly",
                public_cols.num_binary_poly_cols(),
                total_cols.num_binary_poly_cols(),
            ),
            (
                "arbitrary_poly",
                public_cols.num_arbitrary_poly_cols(),
                total_cols.num_arbitrary_poly_cols(),
            ),
            ("int", public_cols.num_int_cols(), total_cols.num_int_cols()),
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
        for spec in &bit_op_specs {
            assert!(
                spec.source_col() < num_cols,
                "BitOpSpec source_col {} out of range (total_cols = {}). \
                 source_col uses flat indexing: binary_poly || arbitrary_poly || int.",
                spec.source_col(),
                num_cols,
            );
        }

        shifts.sort_by_key(|spec| spec.source_col());
        bit_op_specs.sort_by_key(|spec| (spec.source_col(), spec.op().sort_key()));
        let down_cols = Self::compute_down_layout(&total_cols, &shifts);
        let witness_cols = WitnessColumnLayout::new(
            sub!(
                total_cols.num_binary_poly_cols(),
                public_cols.num_binary_poly_cols()
            ),
            sub!(
                total_cols.num_arbitrary_poly_cols(),
                public_cols.num_arbitrary_poly_cols()
            ),
            sub!(total_cols.num_int_cols(), public_cols.num_int_cols()),
        );

        Self {
            total_cols,
            public_cols,
            shifts,
            down_cols,
            witness_cols,
            lookup_specs,
            booleanity_skip_indices: Vec::new(),
            bit_op_specs,
        }
    }

    pub fn lookup_specs(&self) -> &[LookupColumnSpec] {
        &self.lookup_specs
    }

    fn compute_down_layout(
        total_cols: &TotalColumnLayout,
        shifts: &[ShiftSpec],
    ) -> VirtualColumnLayout {
        let binary_poly_end = total_cols.num_binary_poly_cols();
        let arbitrary_poly_end = add!(binary_poly_end, total_cols.num_arbitrary_poly_cols());
        let mut num_binary_poly = 0usize;
        let mut num_arbitrary_poly = 0usize;
        let mut num_int = 0usize;
        for spec in shifts {
            if spec.source_col() < binary_poly_end {
                num_binary_poly = add!(num_binary_poly, 1);
            } else if spec.source_col() < arbitrary_poly_end {
                num_arbitrary_poly = add!(num_arbitrary_poly, 1);
            } else {
                num_int = add!(num_int, 1);
            }
        }
        VirtualColumnLayout::new(num_binary_poly, num_arbitrary_poly, num_int)
    }

    pub fn total_cols(&self) -> &TotalColumnLayout {
        &self.total_cols
    }

    pub fn public_cols(&self) -> &PublicColumnLayout {
        &self.public_cols
    }

    /// Witness column counts (total minus public) per type.
    pub fn witness_cols(&self) -> &WitnessColumnLayout {
        &self.witness_cols
    }

    pub fn shifts(&self) -> &[ShiftSpec] {
        &self.shifts
    }

    /// Column-type layout of the shifted (down) row.
    pub fn down_cols(&self) -> &VirtualColumnLayout {
        &self.down_cols
    }

    /// Bit-op virtual column specifications (sorted, deterministic).
    pub fn bit_op_specs(&self) -> &[BitOpSpec] {
        &self.bit_op_specs
    }

    /// Number of bit-op virtual columns (= length of the trailing
    /// `bit_op` slice in the down trace row).
    pub fn bit_op_down_count(&self) -> usize {
        self.bit_op_specs.len()
    }

    /// Build correctly-sized dummy up and down `TraceRow`s for static
    /// analysis (constraint counting, degree counting, scalar/ideal
    /// collection).
    ///
    /// The down dummy length includes both the row-shift virtual columns
    /// and the bit-op virtual columns: layout is
    /// `[binary_poly... | arbitrary_poly... | int... | bit_op...]`.
    pub fn dummy_rows<T: Clone>(&self, val: T) -> (Vec<T>, Vec<T>) {
        let up_size = self.total_cols.cols();
        let down_size = add!(self.down_cols.cols(), self.bit_op_specs.len());
        (vec![val.clone(); up_size], vec![val; down_size])
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
    /// Returns a sub-trace containing only public columns.
    /// Returned trace is borrowed from the full trace.
    pub fn public(&self, sig: &UairSignature) -> UairTrace<'_, PolyCoeff, Int, D> {
        let p = sig.public_cols();
        UairTrace {
            binary_poly: Cow::Borrowed(&self.binary_poly[0..p.num_binary_poly_cols()]),
            arbitrary_poly: Cow::Borrowed(&self.arbitrary_poly[0..p.num_arbitrary_poly_cols()]),
            int: Cow::Borrowed(&self.int[0..p.num_int_cols()]),
        }
    }

    /// Returns a sub-trace containing only witness columns.
    /// Returned trace is borrowed from the full trace.
    pub fn witness(&self, sig: &UairSignature) -> UairTrace<'_, PolyCoeff, Int, D> {
        let p = sig.public_cols();
        UairTrace {
            binary_poly: Cow::Borrowed(&self.binary_poly[p.num_binary_poly_cols()..]),
            arbitrary_poly: Cow::Borrowed(&self.arbitrary_poly[p.num_arbitrary_poly_cols()..]),
            int: Cow::Borrowed(&self.int[p.num_int_cols()..]),
        }
    }
}

// ---------------------------------------------------------------------------
// TraceRow
// ---------------------------------------------------------------------------

/// A view on a row of the trace.
/// Contains references to cells of the trace
/// of all types lying in the same trace row.
///
/// On the up row, `bit_op` is always empty. On the down row it carries
/// the bit-op virtual columns (one per `BitOpSpec` declared by the UAIR);
/// when the UAIR declares no bit-ops, `bit_op` is empty there too.
#[derive(Clone, Copy)]
pub struct TraceRow<'a, Expr> {
    pub binary_poly: &'a [Expr],
    pub arbitrary_poly: &'a [Expr],
    pub int: &'a [Expr],
    pub bit_op: &'a [Expr],
}

impl<'a, Expr> TraceRow<'a, Expr> {
    /// Given a slice that represents a raw row of the trace,
    /// creates a `TraceRow` from it.
    /// Subdivides the slice according to the given column layout.
    /// `bit_op` is set to the empty slice — use
    /// [`Self::from_slice_with_layout_and_bit_op`] to interpret a
    /// trailing `bit_op_count` slots as bit-op virtual columns.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn from_slice_with_layout(row: &'a [Expr], layout: &ColumnLayout) -> Self {
        let num_binary_poly = layout.num_binary_poly_cols();
        let num_arbitrary_poly = layout.num_arbitrary_poly_cols();
        Self {
            binary_poly: &row[0..num_binary_poly],
            arbitrary_poly: &row[num_binary_poly..num_binary_poly + num_arbitrary_poly],
            int: &row[num_binary_poly + num_arbitrary_poly..],
            bit_op: &[],
        }
    }

    /// Like [`Self::from_slice_with_layout`] but interprets the trailing
    /// `bit_op_count` slots as bit-op virtual columns. The slice layout
    /// is `[binary_poly... | arbitrary_poly... | int... | bit_op...]`,
    /// where the first three sections are sized by `layout` and the
    /// trailing section has `bit_op_count` entries.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn from_slice_with_layout_and_bit_op(
        row: &'a [Expr],
        layout: &ColumnLayout,
        bit_op_count: usize,
    ) -> Self {
        let num_binary_poly = layout.num_binary_poly_cols();
        let num_arbitrary_poly = layout.num_arbitrary_poly_cols();
        let num_int = layout.num_int_cols();
        let int_end = num_binary_poly + num_arbitrary_poly + num_int;
        let bit_op_end = int_end + bit_op_count;
        Self {
            binary_poly: &row[0..num_binary_poly],
            arbitrary_poly: &row[num_binary_poly..num_binary_poly + num_arbitrary_poly],
            int: &row[num_binary_poly + num_arbitrary_poly..int_end],
            bit_op: &row[int_end..bit_op_end],
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
pub trait Uair: Clone {
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
    ///
    /// TODO: Consider caching the signature to avoid recomputing it at every
    /// call site. Currently negligible since shifts are small (e.g. ~12 for
    /// SHA/ECDSA), but may matter if signatures grow more expensive to
    /// construct.
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
