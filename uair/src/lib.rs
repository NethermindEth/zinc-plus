//! UAIR description tools.

pub mod collect_scalars;
pub mod constraint_counter;
pub mod degree_counter;
pub mod do_nothing_builder;
pub mod dummy_semiring;
pub mod ideal;
pub mod ideal_collector;
pub mod lookup_types;

use crate::ideal::Ideal;
use crypto_primitives::{Semiring, SemiringConfig, SetElement};
use std::borrow::Cow;
use zinc_poly::{
    mle::DenseMultilinearExtension,
    univariate::{
        binary::BinaryPoly,
        dense::DensePolynomial,
        dynamic::{DynamicPolynomial, HasDynamicPolynomialConfig},
    },
};
use zinc_utils::{add, sub};

pub use lookup_types::{LookupColumnSpec, LookupTableType};

/// The abstract interface to constraint building logic.
/// In essence it allows to create constraints modulo ideals.
pub trait ConstraintBuilder {
    /// The expressions the constraint builder operates on.
    /// It is opaque from the PoV of an AIR: arithmetic operations on it are
    /// provided by the expression config passed to
    /// [`Uair::constrain_general`] alongside the builder.
    type Expr: SetElement;
    /// The type of ideals used by the constraint builder.
    type Ideal: Ideal;
    /// Ideals living over $F_{q_i}[X]$ for the prime tuple declared by
    /// the surrounding [`UairSignature::primes`]. A single
    /// `ConstraintBuilder` shares one runtime type for all primes; the prime
    /// index is passed at the call site via
    /// [`ConstraintBuilder::assert_in_fq_ideal`]. Builders that don't care
    /// about $F_q[X]$-constraints (counters, collectors, etc.) set
    /// this to `ImpossibleIdeal`.
    type FqIdeal: Ideal;

    /// Add a constraint saying that `expr` belongs to the ideal `ideal`.
    fn assert_in_ideal(&mut self, expr: Self::Expr, ideal: &Self::Ideal);

    /// Add a constraint saying that `expr` is equal to zero which is
    /// the same as saying that `expr` belongs to the zero ideal.
    fn assert_zero(&mut self, expr: Self::Expr);

    /// Add a constraint saying that `expr`, after coefficient-wise reduction
    /// mod $q_{\text{prime\_index}}$ (the paper's $\phi_{q_i}$), belongs to
    /// the $F_{q_i}[X]$-ideal `ideal`.
    ///
    /// `prime_idx` indexes into [`UairSignature::primes`] and must be a
    /// valid index for any UAIR that calls this method.
    ///
    /// # Ordering convention
    ///
    /// The order of constraints *within* each family must be stable across
    /// `constrain_general` calls, so `count_constraints` /
    /// `count_constraint_degrees` / `IdealCollector::{ideals, fq_ideals}`
    /// line up per family.
    ///
    /// # Scope: projections of f_0 only
    ///
    /// `expr` is built only from the $Q[X]$-typed `up`/`down` rows passed to
    /// [`Uair::constrain_general`] — i.e. the projection $\phi_{q_i}(\hat f_0)$
    /// of the single integer trace. There is no separate $\hat f_i$ witness
    /// typed natively in $F_{q_i}[X]$; $\phi_{q_i}$ is applied by the PIOP
    /// layer at prove/verify time.
    fn assert_in_fq_ideal(&mut self, prime_idx: usize, expr: Self::Expr, ideal: &Self::FqIdeal);
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
// BitOp virtual columns
// ---------------------------------------------------------------------------

/// An entry-wise `R`-linear endomorphism of the bounded-degree coefficient
/// module `R^{<W}[X]` (cf. Section 2.1.1 of the Zinc+ paper) that defines a
/// virtual column.
///
/// Per Lemma 2.3, any `R`-linear coordinate-wise map on `R^{<W}[X]` commutes
/// with multilinear extension over the row hypercube. Consequently the column
/// `T(v)` need not be committed: the prover materializes it during the
/// constraint-aggregation sumcheck, and the verifier reconstructs its MLE
/// evaluation at the final point `r_0` by applying `T` to the source
/// column's lifted opening, its `W` `F_q`-coefficients, directly.
///
/// `Rot(c)` admits an alternative description as multiplication by `X^{W-c}`
/// modulo `X^W - 1`, i.e. as an endomorphism of `R[X]/(X^W - 1)`. `ShR(c)` is
/// pure zero-padding on coefficient indices and is *not* a quotient-ring
/// operation; both, however, are `R`-linear maps on `R^{<W}[X]` and fall
/// under the same Lemma 2.3 frame.
///
/// Bit-ops are defined only on binary_poly source columns.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BitOp {
    /// Right-rotation by `c` bit positions. The result's coefficient at
    /// position `i` is the source's at `(i + c) mod W`, where `W` is the
    /// cell width.
    Rot(usize),
    /// Right-shift by `c` bit positions. The result's coefficient at
    /// position `i` is the source's at `i + c` if `i + c < W`, else zero.
    ShR(usize),
}

impl BitOp {
    /// The rotation / shift count.
    pub fn count(&self) -> usize {
        match self {
            BitOp::Rot(c) | BitOp::ShR(c) => *c,
        }
    }

    /// Apply the bit operation to a projected bit-polynomial cell.
    pub fn transform<S: SemiringConfig, const D: usize>(
        &self,
        source: &DynamicPolynomial<S::Element>,
        cfg: &S,
    ) -> DynamicPolynomial<S::Element> {
        let poly_cfg = cfg.dyn_poly_cfg();
        match self {
            BitOp::Rot(c) => poly_cfg.rotate_right::<D>(source, *c),
            BitOp::ShR(c) => poly_cfg.shr::<D>(source, *c),
        }
    }
}

/// Specifies a bit-op virtual column.
///
/// `BitOpSpec { source_col: 0, op: BitOp::ShR(3) }` declares a virtual column
/// whose row `i` is `ShR^3` applied entry-wise to the `i`-th cell of column 0.
///
/// `source_col` must reference a binary_poly column; bit-ops are only defined
/// on bit-polynomial cells, i.e. elements of `R^{<W}[X]` with `{0,1}`
/// coefficients.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BitOpSpec {
    /// Flat index of the binary_poly source column. Uses the same
    /// `binary_poly || arbitrary_poly || int` indexing as `ShiftSpec`.
    source_col: usize,
    /// The bit-op applied entry-wise to the source column.
    op: BitOp,
}

impl BitOpSpec {
    pub fn new(source_col: usize, op: BitOp) -> Self {
        assert!(op.count() > 0, "bit-op count must be non-zero");
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
// Affine virtual columns for booleanity targets
// ---------------------------------------------------------------------------

/// One term in an affine virtual binary-polynomial expression.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct AffineVirtualTerm {
    /// Flat total-column index of a binary-polynomial source. Binary-polynomial
    /// columns form the prefix of the trace layout, with public columns before
    /// witness columns.
    source_col: usize,
    /// Integer scalar applied to the source. The Ch/Maj constraints only need
    /// small signed coefficients such as `1`, `-1`, and `-2`.
    coefficient: i64,
    /// Forward row shift. Zero reads the current row; a positive shift reads
    /// `source_col[row + row_shift]`, with zero padding past the trace length.
    row_shift: usize,
}

impl AffineVirtualTerm {
    /// Construct an unshifted term `coefficient * source_col`.
    pub fn new(source_col: usize, coefficient: i64) -> Self {
        assert!(
            coefficient != 0,
            "affine virtual coefficient must be non-zero"
        );
        Self {
            source_col,
            coefficient,
            row_shift: 0,
        }
    }

    /// Construct a shifted term `coefficient * source_col[row + row_shift]`.
    pub fn new_shifted(source_col: usize, coefficient: i64, row_shift: usize) -> Self {
        assert!(row_shift > 0, "row shift must be non-zero");
        Self {
            row_shift,
            ..Self::new(source_col, coefficient)
        }
    }

    pub fn source_col(&self) -> usize {
        self.source_col
    }

    pub fn coefficient(&self) -> i64 {
        self.coefficient
    }

    pub fn row_shift(&self) -> usize {
        self.row_shift
    }
}

/// Declares an affine virtual binary-polynomial expression whose coefficients
/// must be proved boolean.
///
/// The represented expression is
///
/// ```text
/// ones_coefficient * 1_D + sum_i coefficient_i * source_i[row + shift_i]
/// ```
///
/// where `1_D = 1 + X + ... + X^(D-1)` is the all-ones bit polynomial for the
/// binary-poly cell width `D`. This covers the paper's Ch/Maj lookup targets,
/// for example `a + b + c - 2m` and `(1_D - e) + g - 2u`.
///
/// Affine virtual specs are Q-side booleanity targets. They do not add entries
/// to the `down` row, and they are not committed columns.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct AffineVirtualSpec {
    terms: Vec<AffineVirtualTerm>,
    ones_coefficient: i64,
}

impl AffineVirtualSpec {
    /// Construct an affine virtual expression with no `1_D` offset.
    pub fn new(terms: Vec<AffineVirtualTerm>) -> Self {
        Self::with_ones_coefficient(terms, 0)
    }

    /// Construct an affine virtual expression with a scalar multiple of
    /// `1_D` as its constant bit-polynomial offset.
    pub fn with_ones_coefficient(terms: Vec<AffineVirtualTerm>, ones_coefficient: i64) -> Self {
        assert!(
            !terms.is_empty(),
            "affine virtual spec must have at least one non-constant term"
        );
        Self {
            terms,
            ones_coefficient,
        }
    }

    pub fn terms(&self) -> &[AffineVirtualTerm] {
        &self.terms
    }

    pub fn ones_coefficient(&self) -> i64 {
        self.ones_coefficient
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
pub struct UairSignature<Prime: Semiring> {
    /// Column-type layout of all (public + witness) columns.
    total_cols: TotalColumnLayout,
    /// Public column subset.
    public_cols: PublicColumnLayout,
    /// Witness column counts (total minus public) per type.
    witness_cols: WitnessColumnLayout,
    /// Shifted columns info sorted by `source_col`.
    shifts: Vec<ShiftSpec>,
    /// Bit-op virtual column specs, in insertion order. Each spec references a
    /// binary_poly source column and contributes one extra entry to the
    /// binary_poly slice of the down row, appended after the shifted entries.
    bit_op_specs: Vec<BitOpSpec>,
    /// Affine virtual expressions whose cells are Q-side booleanity targets.
    /// These are not committed columns and do not contribute to `down_cols`.
    affine_virtual_specs: Vec<AffineVirtualSpec>,
    /// Column-type layout of the down row (shifted virtuals + bit-op virtuals).
    down_cols: VirtualColumnLayout,
    /// Lookup specifications: which trace columns are constrained against
    /// which table types.
    lookup_specs: Vec<LookupColumnSpec>,
    /// Prime powers `(q_1, ..., q_n)` declared by this UAIR.
    /// $F_{q_i}[X]$-constraints emitted via
    /// [`ConstraintBuilder::assert_in_fq_ideal`] reference these by index.
    /// Empty for $Q[X]$-only UAIRs.
    primes: Vec<Prime>,
}

impl<Prime: Semiring> UairSignature<Prime> {
    /// Create a new signature, sorting `shifts` by `source_col`.
    pub fn new(
        total_cols: TotalColumnLayout,
        public_cols: PublicColumnLayout,
        mut shifts: Vec<ShiftSpec>,
        lookup_specs: Vec<LookupColumnSpec>,
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

        shifts.sort_by_key(|spec| spec.source_col());
        let down_cols = Self::compute_down_layout(&total_cols, &shifts, &[]);
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
            bit_op_specs: Vec::new(),
            affine_virtual_specs: Vec::new(),
            down_cols,
            witness_cols,
            lookup_specs,
            primes: Vec::new(),
        }
    }

    /// Attach the prime-power tuple `(q_1, ..., q_n)` that
    /// $F_{q_i}[X]$-constraints emitted by this UAIR live over.
    pub fn with_primes(mut self, primes: Vec<Prime>) -> Self {
        self.primes = primes;
        self
    }

    /// Prime-power tuple `(q_1, ..., q_n)` declared by this UAIR. Empty for
    /// UAIRs with $Q[X]$-only constraints.
    pub fn primes(&self) -> &[Prime] {
        &self.primes
    }

    /// Attach bit-op virtual column specs to the signature.
    ///
    /// Each spec must reference a binary_poly source column. The bit-op count
    /// must be less than the binary-poly cell width `W`; materialization sites
    /// check that bound with their const `DEGREE_PLUS_ONE` parameter.
    ///
    /// # Down-row ordering invariant
    ///
    /// Bit-op virtuals slot into the `binary_poly` slice of the down
    /// `TraceRow`, *after* the shifted-binary entries and *before* any
    /// non-binary entries. The full ordering of the down row is:
    ///
    /// ```text
    /// [shifted_binary_poly..., bit_op_binary_poly..., shifted_arbitrary_poly..., shifted_int...]
    /// ```
    ///
    /// This keeps `down` consistent with `ColumnLayout`'s
    /// `binary_poly || arbitrary_poly || int` partitioning. Materialization
    /// code in CPR / mp_eval must respect this order; appending bit-op evals
    /// at the tail of `down_evals` would silently misalign constraint indices
    /// on mixed-type shift UAIRs.
    ///
    /// Insertion order of `bit_op_specs` determines the position of each
    /// bit-op virtual within its sub-slice.
    pub fn with_bit_op_specs(mut self, bit_op_specs: Vec<BitOpSpec>) -> Self {
        let binary_poly_end = self.total_cols.num_binary_poly_cols();
        for spec in &bit_op_specs {
            assert!(
                spec.source_col() < binary_poly_end,
                "BitOpSpec source_col {} is not a binary_poly column \
                 (binary_poly_end = {}). Bit-ops are only defined on the \
                 cell ring F_2[X]/(X^W).",
                spec.source_col(),
                binary_poly_end,
            );
        }
        self.bit_op_specs = bit_op_specs;
        self.down_cols =
            Self::compute_down_layout(&self.total_cols, &self.shifts, &self.bit_op_specs);
        self
    }

    /// Attach affine virtual booleanity specs to the signature.
    ///
    /// Each term in each spec must reference a binary_poly column. Affine
    /// virtuals describe Q-side `{0,1}^{<D}[X]` membership targets such as the
    /// Ch/Maj linear combinations in the Zinc+ paper. They are not committed
    /// columns and do not affect the down-row layout.
    pub fn with_affine_virtual_specs(
        mut self,
        affine_virtual_specs: Vec<AffineVirtualSpec>,
    ) -> Self {
        let binary_poly_end = self.total_cols.num_binary_poly_cols();
        for spec in &affine_virtual_specs {
            for term in spec.terms() {
                assert!(
                    term.source_col() < binary_poly_end,
                    "AffineVirtualTerm source_col {} is not a binary_poly column \
                     (binary_poly_end = {}). Affine virtual booleanity targets \
                     are only defined over binary_poly cells.",
                    term.source_col(),
                    binary_poly_end,
                );
            }
        }
        self.affine_virtual_specs = affine_virtual_specs;
        self
    }

    pub fn lookup_specs(&self) -> &[LookupColumnSpec] {
        &self.lookup_specs
    }

    fn compute_down_layout(
        total_cols: &TotalColumnLayout,
        shifts: &[ShiftSpec],
        bit_op_specs: &[BitOpSpec],
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
        num_binary_poly = add!(num_binary_poly, bit_op_specs.len());
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

    /// Bit-op virtual column specs, in insertion order. Each spec contributes
    /// one binary_poly entry to the down row, appended after the shifted
    /// entries.
    pub fn bit_op_specs(&self) -> &[BitOpSpec] {
        &self.bit_op_specs
    }

    /// Affine virtual booleanity specs, in insertion order.
    pub fn affine_virtual_specs(&self) -> &[AffineVirtualSpec] {
        &self.affine_virtual_specs
    }

    /// Number of row-shift virtual columns that precede bit-op virtuals in the
    /// down-row ordering.
    ///
    /// The full down-row order is:
    /// `[shifted_binary_poly..., bit_op_binary_poly...,
    /// shifted_arbitrary_poly..., shifted_int...]`.
    pub fn bit_op_down_offset(&self) -> usize {
        let binary_poly_end = self.total_cols.num_binary_poly_cols();
        self.shifts
            .iter()
            .take_while(|spec| spec.source_col() < binary_poly_end)
            .count()
    }

    /// Column-type layout of the down row (shifted virtuals + bit-op virtuals).
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
}

// ---------------------------------------------------------------------------
// UairTrace
// ---------------------------------------------------------------------------

/// The trace of a UAIR execution (pre-projection).
/// If owned, it contains the full trace, otherwise it contains a view on the
/// full trace (e.g. only public columns).
#[derive(Debug, Clone, Default)]
pub struct UairTrace<
    'a,
    PolyCoeff: Clone,
    Int: Clone,
    const BINARY_POLY_DEGREE_PLUS_ONE: usize,
    const ARBITRARY_POLY_DEGREE_PLUS_ONE: usize,
> {
    pub binary_poly: Cow<'a, [DenseMultilinearExtension<BinaryPoly<BINARY_POLY_DEGREE_PLUS_ONE>>]>,
    pub arbitrary_poly: Cow<
        'a,
        [DenseMultilinearExtension<DensePolynomial<PolyCoeff, ARBITRARY_POLY_DEGREE_PLUS_ONE>>],
    >,
    pub int: Cow<'a, [DenseMultilinearExtension<Int>]>,
}

impl<PolyCoeff: Clone, Int: Clone, const DB: usize, const DA: usize>
    UairTrace<'static, PolyCoeff, Int, DB, DA>
{
    /// Returns a sub-trace containing only public columns.
    /// Returned trace is borrowed from the full trace.
    pub fn public<Prime: Semiring>(
        &self,
        sig: &UairSignature<Prime>,
    ) -> UairTrace<'_, PolyCoeff, Int, DB, DA> {
        let p = sig.public_cols();
        UairTrace {
            binary_poly: Cow::Borrowed(&self.binary_poly[0..p.num_binary_poly_cols()]),
            arbitrary_poly: Cow::Borrowed(&self.arbitrary_poly[0..p.num_arbitrary_poly_cols()]),
            int: Cow::Borrowed(&self.int[0..p.num_int_cols()]),
        }
    }

    /// Returns a sub-trace containing only witness columns.
    /// Returned trace is borrowed from the full trace.
    pub fn witness<Prime: Semiring>(
        &self,
        sig: &UairSignature<Prime>,
    ) -> UairTrace<'_, PolyCoeff, Int, DB, DA> {
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
        let num_binary_poly = layout.num_binary_poly_cols();
        let num_arbitrary_poly = layout.num_arbitrary_poly_cols();
        Self {
            binary_poly: &row[0..num_binary_poly],
            arbitrary_poly: &row[num_binary_poly..num_binary_poly + num_arbitrary_poly],
            int: &row[num_binary_poly + num_arbitrary_poly..],
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

    /// The ideal type for $F_{q_i}[X]$-constraints emitted via
    /// [`ConstraintBuilder::assert_in_fq_ideal`]. UAIRs that do not declare
    /// any primes should set this to [`ideal::ImpossibleIdeal`].
    type FqIdeal: Ideal;

    /// The type of scalars of the UAIR.
    /// For now, we assume they are of the type "arbitrary polynomials".
    // Note: This is usually Z_32[X] (i.e. DensePolynomial<Ring, 32>), but according
    // to @agareta, this in not always the case.
    type Scalar: Semiring;

    /// Type of primes defined in signature. Must be compatible with the field
    /// type we're using.
    type Prime: Semiring;

    /// Signature of the UAIR.
    ///
    /// TODO: Consider caching the signature to avoid recomputing it at every
    /// call site. Currently negligible since shifts are small (e.g. ~12 for
    /// SHA/ECDSA), but may matter if signatures grow more expensive to
    /// construct.
    fn signature() -> UairSignature<Self::Prime>;

    /// A general method for describing constraints.
    ///
    /// # Arguments
    /// - `b`: a builder encapsulating the constraint storing logic. Its type
    ///   `B` has to have compatible `B::Ideal` with the `Self::Ideal`, i.e. it
    ///   must implement `FromRef<Self::Ideal>` trait.
    /// - `expr_cfg`: the [`SemiringConfig`] providing arithmetic operations on
    ///   `B::Expr`. Per-family builder runs pass the family's config (e.g. a
    ///   [`zinc_poly::univariate::dynamic::DynamicPolynomialConfig`] over the
    ///   family's field, or the field config itself); static analyses pass a
    ///   `FixedConfig`.
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
    /// - `ideal_from_ref`: a closure that turns a `Self::Ideal` into `B::Ideal`
    ///   for the $Q[X]$-ideal-membership family.
    /// - `fq_ideal_from_ref`: a closure that turns a `Self::FqIdeal` into
    ///   `B::FqIdeal` for the new $F_{q_i}[X]$-ideal-membership family emitted
    ///   via [`ConstraintBuilder::assert_in_fq_ideal`]. UAIRs without
    ///   $F_q[X]$-constraints can ignore this closure.
    #[allow(clippy::too_many_arguments)]
    fn constrain_general<C, B, FromR, MulByScalar, IFromR, IFqFromR>(
        b: &mut B,
        expr_cfg: &C,
        up: TraceRow<C::Element>,
        down: TraceRow<C::Element>,
        from_ref: FromR,
        mbs: MulByScalar,
        ideal_from_ref: IFromR,
        fq_ideal_from_ref: IFqFromR,
    ) where
        C: SemiringConfig,
        B: ConstraintBuilder<Expr = C::Element>,
        FromR: Fn(&Self::Scalar) -> C::Element,
        MulByScalar: Fn(&C::Element, &Self::Scalar) -> Option<C::Element>,
        IFromR: Fn(&Self::Ideal) -> B::Ideal,
        IFqFromR: Fn(&Self::FqIdeal) -> B::FqIdeal;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn signature_with_mixed_shifts() -> UairSignature<u64> {
        UairSignature::new(
            TotalColumnLayout::new(2, 1, 1),
            PublicColumnLayout::new(0, 0, 0),
            vec![
                ShiftSpec::new(0, 1),
                ShiftSpec::new(2, 1),
                ShiftSpec::new(3, 1),
            ],
            vec![],
        )
    }

    #[test]
    fn bit_op_specs_extend_binary_down_layout() {
        let specs = vec![
            BitOpSpec::new(1, BitOp::ShR(3)),
            BitOpSpec::new(0, BitOp::Rot(2)),
        ];
        let sig = signature_with_mixed_shifts().with_bit_op_specs(specs.clone());

        assert_eq!(sig.bit_op_specs(), specs);
        assert_eq!(sig.bit_op_specs()[0].source_col(), 1);
        assert_eq!(sig.bit_op_specs()[0].op(), BitOp::ShR(3));
        assert_eq!(sig.bit_op_specs()[0].op().count(), 3);
        assert_eq!(sig.down_cols().num_binary_poly_cols(), 3);
        assert_eq!(sig.down_cols().num_arbitrary_poly_cols(), 1);
        assert_eq!(sig.down_cols().num_int_cols(), 1);
        assert_eq!(sig.bit_op_down_offset(), 1);
    }

    #[test]
    fn empty_bit_op_specs_keep_shift_only_down_layout() {
        let sig = signature_with_mixed_shifts().with_bit_op_specs(vec![]);

        assert!(sig.bit_op_specs().is_empty());
        assert_eq!(sig.down_cols().num_binary_poly_cols(), 1);
        assert_eq!(sig.down_cols().num_arbitrary_poly_cols(), 1);
        assert_eq!(sig.down_cols().num_int_cols(), 1);
        assert_eq!(sig.bit_op_down_offset(), 1);
    }

    #[test]
    #[should_panic(expected = "bit-op count must be non-zero")]
    fn bit_op_spec_rejects_zero_count() {
        let _ = BitOpSpec::new(0, BitOp::Rot(0));
    }

    #[test]
    #[should_panic(expected = "is not a binary_poly column")]
    fn bit_op_specs_reject_non_binary_source() {
        let _ =
            signature_with_mixed_shifts().with_bit_op_specs(vec![BitOpSpec::new(2, BitOp::ShR(1))]);
    }

    #[test]
    fn affine_virtual_specs_are_attached_in_order() {
        let specs = vec![
            AffineVirtualSpec::new(vec![
                AffineVirtualTerm::new(0, 1),
                AffineVirtualTerm::new_shifted(1, 1, 2),
                AffineVirtualTerm::new(0, -2),
            ]),
            AffineVirtualSpec::with_ones_coefficient(
                vec![
                    AffineVirtualTerm::new(0, -1),
                    AffineVirtualTerm::new_shifted(1, 1, 1),
                ],
                1,
            ),
        ];

        let sig = signature_with_mixed_shifts().with_affine_virtual_specs(specs.clone());

        assert_eq!(sig.affine_virtual_specs(), specs);
        assert_eq!(sig.affine_virtual_specs()[0].terms()[1].source_col(), 1);
        assert_eq!(sig.affine_virtual_specs()[0].terms()[1].coefficient(), 1);
        assert_eq!(sig.affine_virtual_specs()[0].terms()[1].row_shift(), 2);
        assert_eq!(sig.affine_virtual_specs()[1].ones_coefficient(), 1);
        assert_eq!(sig.down_cols().num_binary_poly_cols(), 1);
        assert_eq!(sig.down_cols().num_arbitrary_poly_cols(), 1);
        assert_eq!(sig.down_cols().num_int_cols(), 1);
    }

    #[test]
    fn empty_affine_virtual_specs_are_supported() {
        let sig = signature_with_mixed_shifts().with_affine_virtual_specs(vec![]);

        assert!(sig.affine_virtual_specs().is_empty());
        assert_eq!(sig.down_cols().num_binary_poly_cols(), 1);
        assert_eq!(sig.down_cols().num_arbitrary_poly_cols(), 1);
        assert_eq!(sig.down_cols().num_int_cols(), 1);
    }

    #[test]
    #[should_panic(expected = "affine virtual coefficient must be non-zero")]
    fn affine_virtual_term_rejects_zero_coefficient() {
        let _ = AffineVirtualTerm::new(0, 0);
    }

    #[test]
    #[should_panic(expected = "affine virtual spec must have at least one non-constant term")]
    fn affine_virtual_spec_rejects_constant_only_expression() {
        let _ = AffineVirtualSpec::with_ones_coefficient(vec![], 1);
    }

    #[test]
    #[should_panic(expected = "is not a binary_poly column")]
    fn affine_virtual_specs_reject_non_binary_source() {
        let _ =
            signature_with_mixed_shifts().with_affine_virtual_specs(vec![AffineVirtualSpec::new(
                vec![AffineVirtualTerm::new(2, 1)],
            )]);
    }
}
