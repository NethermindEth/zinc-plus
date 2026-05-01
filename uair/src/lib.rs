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

/// Declares that the bit slices of a witness binary_poly column should be
/// opened at a shifted row, matching one of the existing `ShiftSpec`
/// entries on the same flat column. Used by virtual booleanity specs that
/// reference values from rows other than the booleanity batch's anchor.
///
/// `witness_col_idx` is witness-relative (0-based, post-public). The
/// `shift_amount` must match one of `UairSignature::shifts()`'s entries
/// targeting this column — its `down_eval` is what the verifier ties the
/// shifted bit slices to via the projection-element consistency check.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShiftedBitSliceSpec {
    pub witness_col_idx: usize,
    pub shift_amount: usize,
}

impl ShiftedBitSliceSpec {
    pub fn new(witness_col_idx: usize, shift_amount: usize) -> Self {
        assert!(shift_amount > 0, "shift must be non-zero");
        Self {
            witness_col_idx,
            shift_amount,
        }
    }
}

/// Source for one term of a virtual booleanity linear combination.
///
/// Virtual booleanity asserts that `Σ_j coeff_j · v_j ∈ {0, 1}` per row,
/// where each `v_j` is the eval of a `VirtualBoolSource`. Coefficients
/// are signed integers cast to F at materialization time.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VirtualBoolSource {
    /// Bit slice of a witness binary_poly column at the un-shifted point.
    /// `witness_col_idx` is witness-relative (0-based, post-public).
    /// `bit_idx ∈ [0, D)`.
    SelfBitSlice {
        witness_col_idx: usize,
        bit_idx: usize,
    },
    /// Bit slice of a witness binary_poly column at a shifted point.
    /// `shifted_spec_idx` indexes into
    /// `UairSignature::shifted_bit_slice_specs`. `bit_idx ∈ [0, D)`.
    ShiftedBitSlice {
        shifted_spec_idx: usize,
        bit_idx: usize,
    },
    /// Bit slice of a *public* binary_poly column at the un-shifted
    /// point. `public_col_idx ∈ [0, num_pub_bin)`. `bit_idx ∈ [0, D)`.
    /// The verifier evaluates the public bit slice MLE at the shared
    /// point locally (no proof bytes); the prover materializes it from
    /// `public_trace`.
    PublicBitSlice {
        public_col_idx: usize,
        bit_idx: usize,
    },
    /// A witness int column entry. `witness_col_idx` is witness-relative
    /// (post-public).
    IntCol { witness_col_idx: usize },
}

/// One virtual booleanity check: the prover materializes the linear
/// combination as an MLE over rows; the booleanity batch verifies it is
/// `{0, 1}`-valued; the verifier reconstructs its closing eval at `r*`
/// from the relevant `bit_slice_evals` / `up_evals`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VirtualBoolSpec {
    pub terms: Vec<(i64, VirtualBoolSource)>,
}

/// Source for one term of a *packed* virtual binary_poly column.
///
/// A virtual binary_poly column is the per-row, per-bit residual
/// `Σ_j coeff_j · v_j[t][i]`, packed into a `BinaryPoly<D>` with the
/// per-bit contributions sharing the same linear-combo structure
/// across all bits. Compared to `D` separate single-bit
/// `VirtualBoolSpec`s, it collapses them into one binary_poly column —
/// booleanity then runs the standard one-XOR-per-row-pair fast path
/// on it instead of `D` separate per-bit dispatches.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VirtualBinaryPolySource {
    /// Witness binary_poly column at the un-shifted point.
    /// `witness_col_idx` is witness-relative (0-based, post-public).
    SelfWitnessCol { witness_col_idx: usize },
    /// Witness binary_poly column at a shifted point.
    /// `shifted_spec_idx` indexes into `shifted_bit_slice_specs`.
    ShiftedWitnessCol { shifted_spec_idx: usize },
    /// Public binary_poly column at the un-shifted point.
    /// `public_col_idx ∈ [0, num_pub_bin)`.
    PublicCol { public_col_idx: usize },
}

/// One packed virtual binary_poly column. Per row t, per bit i:
///   `output[t].bit(i) = (Σ_j coeff_j · source_j[t].bit(i)) mod 2`
/// where the residual `Σ_j coeff_j · source_j[t].bit(i)` must lie
/// in `{0, 1}` for the lookup constraint to hold. The booleanity
/// check on the resulting bit slices catches deviations: a residual
/// outside `{0, 1}` makes its closing override (= residual eval at
/// `r*`) fail the booleanity polynomial identity `v² − v == 0`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VirtualBinaryPolySpec {
    pub terms: Vec<(i64, VirtualBinaryPolySource)>,
}

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
    /// Absolute indices of int trace columns whose entries are
    /// `{0, 1}`-valued and whose binariness should be enforced by the
    /// booleanity sumcheck (alongside binary_poly bit slices). Each
    /// index must point at a *witness* int column (i.e. be `>=`
    /// `public_cols.num_int_cols()`).
    int_witness_bit_cols: Vec<usize>,
    /// Witness binary_poly cols whose bit slices should be opened at
    /// shifted points (in addition to the un-shifted self bit slices),
    /// to support virtual booleanity references to prior/next rows.
    shifted_bit_slice_specs: Vec<ShiftedBitSliceSpec>,
    /// Virtual booleanity specs: each is a linear combination over
    /// self bit slices, shifted bit slices, and witness int cols whose
    /// every-row value must lie in `{0, 1}`.
    virtual_booleanity_cols: Vec<VirtualBoolSpec>,
    /// Packed virtual binary_poly columns: each spec produces one
    /// `BinaryPoly<D>`-typed virtual column carrying `D` per-bit
    /// residuals at once. Booleanity treats them as ordinary binary
    /// cols (one XOR per row pair); per-bit closing overrides bind
    /// them to the spec residual.
    virtual_binary_poly_cols: Vec<VirtualBinaryPolySpec>,
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
            int_witness_bit_cols: Vec::new(),
            shifted_bit_slice_specs: Vec::new(),
            virtual_booleanity_cols: Vec::new(),
            virtual_binary_poly_cols: Vec::new(),
        }
    }

    /// Replace the set of witness binary_poly columns the protocol
    /// should skip from the booleanity sumcheck. Indices are relative
    /// to the witness section (0-based within
    /// `trace.binary_poly[num_pub_bin..]`) — equivalently,
    /// `absolute_index − num_pub_bin`. Out-of-range entries panic so
    /// the misuse is caught at signature construction.
    pub fn with_booleanity_skip_indices(mut self, mut skip: Vec<usize>) -> Self {
        let num_witness_bin = self.witness_cols.num_binary_poly_cols();
        for &i in &skip {
            assert!(
                i < num_witness_bin,
                "booleanity skip index {i} out of range \
                 (num witness binary_poly cols = {num_witness_bin})"
            );
        }
        skip.sort_unstable();
        skip.dedup();
        self.booleanity_skip_indices = skip;
        self
    }

    /// Sorted, dedup'd witness binary_poly indices (relative to the
    /// witness section) the UAIR has asked to skip from booleanity.
    pub fn booleanity_skip_indices(&self) -> &[usize] {
        &self.booleanity_skip_indices
    }

    /// Builder-style: declare shifted bit-slice openings. Each spec must
    /// match an existing `ShiftSpec` on the same witness binary_poly col
    /// (so the shifted parent's `down_eval` is available to bind the
    /// shifted bit slices via the projection-element consistency check).
    #[must_use]
    pub fn with_shifted_bit_slice_specs(mut self, specs: Vec<ShiftedBitSliceSpec>) -> Self {
        let num_pub_bin = self.public_cols.num_binary_poly_cols();
        let num_wit_bin = self.witness_cols.num_binary_poly_cols();
        for spec in &specs {
            assert!(
                spec.witness_col_idx < num_wit_bin,
                "ShiftedBitSliceSpec witness_col_idx {} out of range (num_wit_bin = {num_wit_bin})",
                spec.witness_col_idx,
            );
            let flat_col = spec.witness_col_idx + num_pub_bin;
            let matched = self.shifts.iter().any(|s| {
                s.source_col() == flat_col && s.shift_amount() == spec.shift_amount
            });
            assert!(
                matched,
                "ShiftedBitSliceSpec(col {}, shift {}) has no matching ShiftSpec",
                spec.witness_col_idx, spec.shift_amount,
            );
        }
        self.shifted_bit_slice_specs = specs;
        self
    }

    pub fn shifted_bit_slice_specs(&self) -> &[ShiftedBitSliceSpec] {
        &self.shifted_bit_slice_specs
    }

    /// For each `ShiftedBitSliceSpec`, return its index into the
    /// `down.binary_poly` slice (i.e., its position among binary-source
    /// shifts in `shifts()` order).
    pub fn shifted_bit_slice_down_indices(&self) -> Vec<usize> {
        let num_total_bin = self.total_cols.num_binary_poly_cols();
        let num_pub_bin = self.public_cols.num_binary_poly_cols();
        self.shifted_bit_slice_specs
            .iter()
            .map(|spec| {
                let flat_col = spec.witness_col_idx + num_pub_bin;
                let mut bin_down_idx = 0usize;
                for s in &self.shifts {
                    if s.source_col() < num_total_bin {
                        if s.source_col() == flat_col
                            && s.shift_amount() == spec.shift_amount
                        {
                            return bin_down_idx;
                        }
                        bin_down_idx += 1;
                    }
                }
                panic!(
                    "ShiftedBitSliceSpec(col {}, shift {}) not found in shifts",
                    spec.witness_col_idx, spec.shift_amount
                )
            })
            .collect()
    }

    /// Builder-style: declare absolute int column indices whose entries
    /// are `{0, 1}`-valued, to be range-checked by the booleanity
    /// sumcheck. Indices must be in the witness portion (after the
    /// public int prefix).
    #[must_use]
    pub fn with_int_witness_bit_cols(mut self, cols: Vec<usize>) -> Self {
        let num_int = self.total_cols.num_int_cols();
        let num_int_pub = self.public_cols.num_int_cols();
        for &c in &cols {
            assert!(
                c >= num_int_pub && c < num_int,
                "int_witness_bit_col {c} out of witness int range \
                 [{num_int_pub}, {num_int})",
            );
        }
        self.int_witness_bit_cols = cols;
        self
    }

    pub fn int_witness_bit_cols(&self) -> &[usize] {
        &self.int_witness_bit_cols
    }

    /// Builder-style: register virtual booleanity checks. Each spec is
    /// validated against the current column / bit layout. `bit_width`
    /// is the binary_poly bit-poly width (D).
    #[must_use]
    pub fn with_virtual_booleanity_cols(
        mut self,
        cols: Vec<VirtualBoolSpec>,
        bit_width: usize,
    ) -> Self {
        let num_wit_bin = self.witness_cols.num_binary_poly_cols();
        let num_wit_int = self.witness_cols.num_int_cols();
        let num_shifted = self.shifted_bit_slice_specs.len();
        for (sidx, spec) in cols.iter().enumerate() {
            for (tidx, (_coeff, source)) in spec.terms.iter().enumerate() {
                match source {
                    VirtualBoolSource::SelfBitSlice {
                        witness_col_idx,
                        bit_idx,
                    } => {
                        assert!(
                            *witness_col_idx < num_wit_bin && *bit_idx < bit_width,
                            "VirtualBoolSpec[{sidx}].terms[{tidx}] SelfBitSlice out of range",
                        );
                    }
                    VirtualBoolSource::ShiftedBitSlice {
                        shifted_spec_idx,
                        bit_idx,
                    } => {
                        assert!(
                            *shifted_spec_idx < num_shifted && *bit_idx < bit_width,
                            "VirtualBoolSpec[{sidx}].terms[{tidx}] ShiftedBitSlice out of range",
                        );
                    }
                    VirtualBoolSource::PublicBitSlice {
                        public_col_idx,
                        bit_idx,
                    } => {
                        let num_pub_bin = self.public_cols.num_binary_poly_cols();
                        assert!(
                            *public_col_idx < num_pub_bin && *bit_idx < bit_width,
                            "VirtualBoolSpec[{sidx}].terms[{tidx}] PublicBitSlice out of range",
                        );
                    }
                    VirtualBoolSource::IntCol { witness_col_idx } => {
                        assert!(
                            *witness_col_idx < num_wit_int,
                            "VirtualBoolSpec[{sidx}].terms[{tidx}] IntCol witness idx out of range",
                        );
                    }
                }
            }
        }
        self.virtual_booleanity_cols = cols;
        self
    }

    pub fn virtual_booleanity_cols(&self) -> &[VirtualBoolSpec] {
        &self.virtual_booleanity_cols
    }

    /// Builder-style: register packed virtual binary_poly columns. Each
    /// spec materializes into one `BinaryPoly<D>`-typed virtual column;
    /// booleanity processes them through the same fast path as genuine
    /// binary_poly cols. Per-bit closing overrides on the verifier side
    /// bind each bit's MLE eval to the spec residual.
    #[must_use]
    pub fn with_virtual_binary_poly_cols(
        mut self,
        cols: Vec<VirtualBinaryPolySpec>,
    ) -> Self {
        let num_wit_bin = self.witness_cols.num_binary_poly_cols();
        let num_pub_bin = self.public_cols.num_binary_poly_cols();
        let num_shifted = self.shifted_bit_slice_specs.len();
        for (sidx, spec) in cols.iter().enumerate() {
            for (tidx, (_coeff, source)) in spec.terms.iter().enumerate() {
                match source {
                    VirtualBinaryPolySource::SelfWitnessCol { witness_col_idx } => {
                        assert!(
                            *witness_col_idx < num_wit_bin,
                            "VirtualBinaryPolySpec[{sidx}].terms[{tidx}] SelfWitnessCol out of range",
                        );
                    }
                    VirtualBinaryPolySource::ShiftedWitnessCol { shifted_spec_idx } => {
                        assert!(
                            *shifted_spec_idx < num_shifted,
                            "VirtualBinaryPolySpec[{sidx}].terms[{tidx}] ShiftedWitnessCol out of range",
                        );
                    }
                    VirtualBinaryPolySource::PublicCol { public_col_idx } => {
                        assert!(
                            *public_col_idx < num_pub_bin,
                            "VirtualBinaryPolySpec[{sidx}].terms[{tidx}] PublicCol out of range",
                        );
                    }
                }
            }
        }
        self.virtual_binary_poly_cols = cols;
        self
    }

    pub fn virtual_binary_poly_cols(&self) -> &[VirtualBinaryPolySpec] {
        &self.virtual_binary_poly_cols
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

    /// Verify additional structural properties of the public columns
    /// (i.e., the publicly-known cells of the trace) that the
    /// in-circuit constraints do not — and need not — capture.
    ///
    /// The verifier holds the full `public_trace` before the proof
    /// runs. Some structural properties of public columns (e.g., a
    /// public compensator column is zero on a designated subset of
    /// rows, or a public corrector column is zero everywhere except
    /// on two boundary rows) are most cheaply enforced by direct
    /// row-wise inspection rather than by an in-circuit selector
    /// column and a quadratic constraint. UAIRs override this method
    /// to perform such checks on `public_trace`.
    ///
    /// `num_vars` is the trace-length log; `n = 1 << num_vars` is the
    /// number of trace rows, including any padding at the tail.
    ///
    /// Default implementation: no extra checks.
    fn verify_public_structure<R, IntT, const D: usize>(
        _public_trace: &UairTrace<'_, R, IntT, D>,
        _num_vars: usize,
    ) -> Result<(), PublicStructureError>
    where
        R: Clone,
        IntT: Clone + num_traits::Zero,
    {
        Ok(())
    }
}

/// Error returned by [`Uair::verify_public_structure`] when an
/// expected structural property of one of the public columns is
/// violated.
#[derive(Debug, Clone, thiserror::Error)]
pub enum PublicStructureError {
    /// A public column expected to be zero on a particular row was
    /// non-zero. Carries enough context to localise the failure.
    #[error("public column '{column}' must be zero at row {row}, but is non-zero")]
    NonZeroOnRequiredZeroRow {
        /// Human-readable column identifier (e.g. "PA_C_C7").
        column: &'static str,
        /// Row index at which the violation was detected.
        row: usize,
    },
    /// A public column's value at a specific row did not match the
    /// expected closed-form value (e.g. the tail-corrector boundary
    /// formula).
    #[error("public column '{column}' at row {row} has wrong value")]
    WrongValue {
        column: &'static str,
        row: usize,
    },
}
