#![allow(clippy::arithmetic_side_effects)] // UAIRs should not care about overflows
pub mod ecdsa;
pub mod ecdsa_addition;
pub mod ecdsa_affine;
pub mod ecdsa_doubling;
mod generate_trace;
pub mod sha256;
pub mod sha_ecdsa;

pub use ecdsa::EcdsaUair;
pub use ecdsa_addition::JacobianAdditionUair;
pub use ecdsa_affine::AffineConversionUair;
pub use ecdsa_doubling::{EC_FP_INT_LIMBS, EcdsaFpRing, JacobianDoublingUair};
pub use generate_trace::*;
pub use sha256::{Sha256CompressionSliceUair, Sha256Ideal};
pub use sha_ecdsa::ShaEcdsaUair;

use crypto_primitives::{ConstSemiring, FixedSemiring, Semiring, boolean::Boolean};
use num_traits::Zero;
use rand::{
    distr::{Distribution, StandardUniform},
    prelude::*,
};
use std::marker::PhantomData;
use zinc_poly::{
    EvaluatablePolynomial,
    mle::{DenseMultilinearExtension, MultilinearExtensionRand},
    univariate::{
        binary::BinaryPoly, dense::DensePolynomial,
        dynamic::over_fixed_semiring::DynamicPolynomialFS,
    },
};
use zinc_uair::{
    BitOp, BitOpSpec, ComposedReadSpec, ConstraintBuilder, LookupColumnSpec, LookupTableType,
    PointTie, PublicColumnLayout, ShiftSpec, TotalColumnLayout, TraceRow, Uair, UairSignature,
    UairTrace,
    ideal::{DegreeOneIdeal, ImpossibleIdeal},
};
use zinc_utils::from_ref::FromRef;

#[derive(Clone, Debug)]
pub struct TestUairSimpleMultiplication<R>(PhantomData<R>);

impl<R> Uair for TestUairSimpleMultiplication<R>
where
    R: Semiring + 'static,
{
    type Ideal = ImpossibleIdeal; // Not used
    type Scalar = DensePolynomial<R, 32>;

    fn signature() -> UairSignature {
        let total = TotalColumnLayout::new(0, 3, 0);
        let shifts = (0..3).map(|i| ShiftSpec::new(i, 1)).collect();
        UairSignature::new(total, PublicColumnLayout::default(), shifts, vec![], vec![])
    }

    fn constrain_general<B, FromR, MulByScalar, IFromR>(
        b: &mut B,
        up: TraceRow<B::Expr>,
        down: TraceRow<B::Expr>,
        _from_ref: FromR,
        _mbs: MulByScalar,
        _ideal_from_ref: IFromR,
    ) where
        B: ConstraintBuilder,
    {
        let up = up.arbitrary_poly;
        let down = down.arbitrary_poly;

        b.assert_zero(up[0].clone() * &up[1] - &down[0]);
        b.assert_zero(up[1].clone() * &up[2] - &down[1]);
        b.assert_zero(up[0].clone() * &up[2] - &down[2]);
    }
}

impl<R> GenerateRandomTrace<32> for TestUairSimpleMultiplication<R>
where
    R: FixedSemiring + From<i8> + 'static,
    StandardUniform: Distribution<R>,
{
    type PolyCoeff = R;
    type Int = R;

    fn generate_random_trace<Rng: RngCore + ?Sized>(
        num_vars: usize,
        rng: &mut Rng,
    ) -> UairTrace<'static, R, R, 32> {
        let mut a: Vec<DynamicPolynomialFS<R>> =
            vec![DynamicPolynomialFS::new(vec![R::from(rng.random::<i8>())])];
        let mut b: Vec<DynamicPolynomialFS<R>> = vec![DynamicPolynomialFS::new(vec![
            R::zero(),
            R::from(rng.random::<i8>()),
        ])];
        let mut c: Vec<DynamicPolynomialFS<R>> = vec![DynamicPolynomialFS::new(vec![
            R::zero(),
            R::from(rng.random::<i8>()),
        ])];

        for i in 1..1 << num_vars {
            let prev_a = a[i - 1].clone();
            let prev_b = b[i - 1].clone();
            let prev_c = c[i - 1].clone();

            a.push(prev_a.clone() * &prev_b);
            b.push(prev_b * &prev_c);
            c.push(prev_a * prev_c);
        }

        let arbitrary_poly = vec![
            a.into_iter()
                .map(|x| {
                    assert!(
                        x.degree() < Some(32),
                        "degree bound exceeded: {}",
                        x.degree().expect("if the degree is large it's not None")
                    );
                    DensePolynomial::new(x.coeffs)
                })
                .collect(),
            b.into_iter()
                .map(|x| {
                    assert!(
                        x.degree() < Some(32),
                        "degree bound exceeded: {}",
                        x.degree().expect("if the degree is large it's not None"),
                    );
                    DensePolynomial::new(x.coeffs)
                })
                .collect(),
            c.into_iter()
                .map(|x| {
                    assert!(
                        x.degree() < Some(32),
                        "degree bound exceeded: {}",
                        x.degree().expect("if the degree is large it's not None"),
                    );
                    DensePolynomial::new(x.coeffs)
                })
                .collect(),
        ]
        .into();
        UairTrace {
            arbitrary_poly,
            ..Default::default()
        }
    }
}

/// Synthetic lookup-exercising UAIR: 16 witness binary_poly columns, all
/// declared as a single `BitPoly { width: 32, chunk_width: 8 }` lookup
/// group (L=16, K=4). n_groups = 1, so step 7 uses the two-open fast
/// path (no bin multipoint reducer). The algebraic constraint is trivial
/// — the GKR-LogUp lookup argument carries the soundness work. Used to
/// validate the wired lookup path end-to-end.
#[derive(Clone, Debug)]
pub struct BinLookup16Uair<R>(PhantomData<R>);

impl<R> Uair for BinLookup16Uair<R>
where
    R: ConstSemiring + 'static,
{
    type Ideal = ImpossibleIdeal;
    type Scalar = DensePolynomial<R, 32>;

    fn signature() -> UairSignature {
        let total = TotalColumnLayout::new(16, 0, 0);
        let lookup_specs: Vec<LookupColumnSpec> = (0..16)
            .map(|i| LookupColumnSpec {
                column_index: i,
                table_type: LookupTableType::BitPoly { width: 32, chunk_width: Some(8) },
            })
            .collect();
        UairSignature::new(total, PublicColumnLayout::default(), vec![], lookup_specs, vec![])
    }

    fn constrain_general<B, FromR, MulByScalar, IFromR>(
        b: &mut B,
        up: TraceRow<B::Expr>,
        _down: TraceRow<B::Expr>,
        _from_ref: FromR,
        _mbs: MulByScalar,
        _ideal_from_ref: IFromR,
    ) where
        B: ConstraintBuilder,
    {
        // Trivially-satisfied constraint (lookups carry the soundness work).
        let v = &up.binary_poly[0];
        b.assert_zero(v.clone() - v);
    }
}

impl<R> GenerateRandomTrace<32> for BinLookup16Uair<R>
where
    R: ConstSemiring + 'static,
{
    type PolyCoeff = R;
    type Int = R;

    fn generate_random_trace<Rng: RngCore + ?Sized>(
        num_vars: usize,
        rng: &mut Rng,
    ) -> UairTrace<'static, R, R, 32> {
        let row_count = 1usize << num_vars;
        let cols: Vec<DenseMultilinearExtension<BinaryPoly<32>>> = (0..16)
            .map(|_| {
                (0..row_count)
                    .map(|_| BinaryPoly::<32>::from(rng.next_u32()))
                    .collect::<DenseMultilinearExtension<BinaryPoly<32>>>()
            })
            .collect();
        UairTrace {
            binary_poly: cols.into(),
            ..Default::default()
        }
    }
}

/// Synthetic Word-lookup UAIR: four witness integer columns, all
/// declared as a single `Word { width: 16, chunk_width: 8 }` lookup
/// group. The algebraic constraint is trivial -- the range check is the
/// whole point -- so a proof of this UAIR is exactly the claim that
/// every cell of every int column lies in `[0, 2^16)`.
///
/// This is the shape a range check takes: a slack cell committed as an
/// integer, declared to the table, and refused when it does not fit.
///
/// `WIDTH` is the declared width, 16 unless a test asks for another one.
#[derive(Clone, Debug)]
pub struct IntWordLookupUair<R, const WIDTH: usize = INT_WORD_WIDTH>(PhantomData<R>);

/// How many integer columns [`IntWordLookupUair`] range-checks.
pub const INT_WORD_COLS: usize = 4;
/// The width [`IntWordLookupUair`] range-checks to.
pub const INT_WORD_WIDTH: usize = 16;

impl<R, const WIDTH: usize> Uair for IntWordLookupUair<R, WIDTH>
where
    R: ConstSemiring + 'static,
{
    type Ideal = ImpossibleIdeal;
    type Scalar = DensePolynomial<R, 32>;

    fn signature() -> UairSignature {
        let total = TotalColumnLayout::new(0, 0, INT_WORD_COLS);
        let lookup_specs: Vec<LookupColumnSpec> = (0..INT_WORD_COLS)
            .map(|i| LookupColumnSpec {
                column_index: i,
                table_type: LookupTableType::Word {
                    width: WIDTH,
                    chunk_width: Some(8),
                },
            })
            .collect();
        UairSignature::new(total, PublicColumnLayout::default(), vec![], lookup_specs, vec![])
    }

    fn constrain_general<B, FromR, MulByScalar, IFromR>(
        b: &mut B,
        up: TraceRow<B::Expr>,
        _down: TraceRow<B::Expr>,
        _from_ref: FromR,
        _mbs: MulByScalar,
        _ideal_from_ref: IFromR,
    ) where
        B: ConstraintBuilder,
    {
        // Trivially satisfied: the lookup carries the whole claim.
        let v = &up.int[0];
        b.assert_zero(v.clone() - v);
    }
}

impl<R, const WIDTH: usize> GenerateRandomTrace<32> for IntWordLookupUair<R, WIDTH>
where
    R: ConstSemiring + From<u32> + 'static,
{
    type PolyCoeff = R;
    type Int = R;

    fn generate_random_trace<Rng: RngCore + ?Sized>(
        num_vars: usize,
        rng: &mut Rng,
    ) -> UairTrace<'static, R, R, 32> {
        let row_count = 1usize << num_vars;
        // Cells fit the narrowest declaration, so one trace serves every WIDTH.
        let bound: u32 = 1 << INT_WORD_WIDTH;
        let cols: Vec<DenseMultilinearExtension<R>> = (0..INT_WORD_COLS)
            .map(|_| {
                let evals: Vec<R> = (0..row_count)
                    .map(|_| R::from(rng.next_u32() % bound))
                    .collect();
                DenseMultilinearExtension::from_evaluations_vec(num_vars, evals, R::ZERO)
            })
            .collect();
        UairTrace {
            int: cols.into(),
            ..Default::default()
        }
    }
}

/// Synthetic prescribed-lookup UAIR: four witness integer columns, all
/// declared against one `Prescribed` table -- the values 1..=9 with pad
/// 0 -- so a single group carries them. The algebraic constraint is
/// trivial: the lookup is the whole claim, and that claim is that every
/// column holds each of 1..=9 exactly once and a zero in every other row.
///
/// This is the shape a sudoku row takes: a permutation checked by one
/// lookup rather than by a polynomial identity of that degree.
///
/// `FILL` says how the trace is laid down, so the same declaration can be
/// handed a column that is not the multiset it names.
#[derive(Clone, Debug)]
pub struct IntPrescribedLookupUair<R, const FILL: u8>(PhantomData<R>);

/// How many integer columns [`IntPrescribedLookupUair`] checks.
pub const INT_PRESCRIBED_COLS: usize = 4;
/// A permutation of 1..=9 with zero padding: the multiset the table names.
pub const PRESCRIBED_PERMUTATION: u8 = 0;
/// The eight becomes a second nine: still all in the table, wrong multiset.
pub const PRESCRIBED_REPEAT: u8 = 1;
/// The eight becomes a ten, which the table never names.
pub const PRESCRIBED_OUTSIDE: u8 = 2;
/// One value short, and so one pad too many.
pub const PRESCRIBED_SHORT: u8 = 3;

/// The table [`IntPrescribedLookupUair`] declares for every column.
fn prescribed_row_table() -> LookupTableType {
    LookupTableType::Prescribed { values: (1..=9).collect(), pad: 0 }
}

impl<R, const FILL: u8> Uair for IntPrescribedLookupUair<R, FILL>
where
    R: ConstSemiring + 'static,
{
    type Ideal = ImpossibleIdeal;
    type Scalar = DensePolynomial<R, 32>;

    fn signature() -> UairSignature {
        let total = TotalColumnLayout::new(0, 0, INT_PRESCRIBED_COLS);
        let lookup_specs: Vec<LookupColumnSpec> = (0..INT_PRESCRIBED_COLS)
            .map(|i| LookupColumnSpec {
                column_index: i,
                table_type: prescribed_row_table(),
            })
            .collect();
        UairSignature::new(total, PublicColumnLayout::default(), vec![], lookup_specs, vec![])
    }

    fn constrain_general<B, FromR, MulByScalar, IFromR>(
        b: &mut B,
        up: TraceRow<B::Expr>,
        _down: TraceRow<B::Expr>,
        _from_ref: FromR,
        _mbs: MulByScalar,
        _ideal_from_ref: IFromR,
    ) where
        B: ConstraintBuilder,
    {
        // Trivially satisfied: the lookup carries the whole claim.
        let v = &up.int[0];
        b.assert_zero(v.clone() - v);
    }
}

impl<R, const FILL: u8> GenerateRandomTrace<32> for IntPrescribedLookupUair<R, FILL>
where
    R: ConstSemiring + From<u32> + 'static,
{
    type PolyCoeff = R;
    type Int = R;

    fn generate_random_trace<Rng: RngCore + ?Sized>(
        num_vars: usize,
        rng: &mut Rng,
    ) -> UairTrace<'static, R, R, 32> {
        let row_count = 1usize << num_vars;
        let cols: Vec<DenseMultilinearExtension<R>> = (0..INT_PRESCRIBED_COLS)
            .map(|_| {
                let mut values: Vec<u32> = (1..=9).collect();
                match FILL {
                    PRESCRIBED_REPEAT => values[7] = 9,
                    PRESCRIBED_OUTSIDE => values[7] = 10,
                    PRESCRIBED_SHORT => {
                        values.pop();
                    }
                    _ => {}
                }
                values.shuffle(rng);
                let mut evals: Vec<R> = values.into_iter().map(R::from).collect();
                evals.resize(row_count, R::ZERO);
                DenseMultilinearExtension::from_evaluations_vec(num_vars, evals, R::ZERO)
            })
            .collect();
        UairTrace {
            int: cols.into(),
            ..Default::default()
        }
    }
}

/// Synthetic selected-lookup UAIR: a solved 9x9 sudoku committed as nine
/// witness integer columns, one per grid row, and one `Selected` group
/// carrying all twenty-seven of the grid's obligations -- nine rows, nine
/// strides down a position, nine three-by-three blocks -- each declared
/// to be a permutation of 1..=9. The algebraic constraint is trivial: the
/// lookups are the whole claim, and one proof carries every one of them.
///
/// Rows past the ninth belong to no selection at all, so they are exactly
/// as unconstrained as the mechanism says they are.
///
/// `FILL` says how the grid is laid down, so the same declaration can be
/// handed a grid that does not solve it.
#[derive(Clone, Debug)]
pub struct SudokuSelectedUair<R, const FILL: u8>(PhantomData<R>);

/// How many integer columns [`SudokuSelectedUair`] commits: one per row.
pub const SUDOKU_COLS: usize = 9;
/// The grid solves: every row, stride and block a permutation of 1..=9.
pub const SUDOKU_SOLVED: u8 = 0;
/// A value repeated inside one row, and so one value missing from it.
pub const SUDOKU_DUPLICATE: u8 = 1;
/// A value slid out of its row into a cell no selection names.
pub const SUDOKU_SLID: u8 = 2;
/// Two cells traded between two rows, breaking the multiset of both.
pub const SUDOKU_SWAPPED: u8 = 3;
/// A cell past the ninth row, which no selection names, holding a number
/// no table does either.
pub const SUDOKU_UNSELECTED: u8 = 4;

/// A solved grid, one row per committed column.
pub const SUDOKU_SOLUTION: [[u32; 9]; 9] = [
    [5, 3, 4, 6, 7, 8, 9, 1, 2],
    [6, 7, 2, 1, 9, 5, 3, 4, 8],
    [1, 9, 8, 3, 4, 2, 5, 6, 7],
    [8, 5, 9, 7, 6, 1, 4, 2, 3],
    [4, 2, 6, 8, 5, 3, 7, 9, 1],
    [7, 1, 3, 9, 2, 4, 8, 5, 6],
    [9, 6, 1, 5, 3, 7, 2, 8, 4],
    [2, 8, 7, 4, 1, 9, 6, 3, 5],
    [3, 4, 5, 2, 8, 6, 1, 7, 9],
];

/// The twenty-seven cell sets a sudoku grid must each hold 1..=9 in, as
/// `(column slot, row)` over the nine columns holding the grid's rows.
pub fn sudoku_selections() -> Vec<Vec<(u32, u32)>> {
    let rows = (0..9u32).map(|r| (0..9u32).map(|p| (r, p)).collect());
    let strides = (0..9u32).map(|p| (0..9u32).map(|r| (r, p)).collect());
    let blocks = (0..9u32).map(|b| {
        (0..9u32)
            .map(|c| (b / 3 * 3 + c / 3, b % 3 * 3 + c % 3))
            .collect()
    });
    rows.chain(strides).chain(blocks).collect()
}

/// The table [`SudokuSelectedUair`] declares for every column.
fn sudoku_table() -> LookupTableType {
    LookupTableType::Selected {
        values: (1..=9).collect(),
        selections: sudoku_selections(),
    }
}

impl<R, const FILL: u8> Uair for SudokuSelectedUair<R, FILL>
where
    R: ConstSemiring + 'static,
{
    type Ideal = ImpossibleIdeal;
    type Scalar = DensePolynomial<R, 32>;

    fn signature() -> UairSignature {
        let total = TotalColumnLayout::new(0, 0, SUDOKU_COLS);
        let lookup_specs: Vec<LookupColumnSpec> = (0..SUDOKU_COLS)
            .map(|i| LookupColumnSpec {
                column_index: i,
                table_type: sudoku_table(),
            })
            .collect();
        UairSignature::new(total, PublicColumnLayout::default(), vec![], lookup_specs, vec![])
    }

    fn constrain_general<B, FromR, MulByScalar, IFromR>(
        b: &mut B,
        up: TraceRow<B::Expr>,
        _down: TraceRow<B::Expr>,
        _from_ref: FromR,
        _mbs: MulByScalar,
        _ideal_from_ref: IFromR,
    ) where
        B: ConstraintBuilder,
    {
        // Trivially satisfied: the lookups carry the whole claim.
        let v = &up.int[0];
        b.assert_zero(v.clone() - v);
    }
}

impl<R, const FILL: u8> GenerateRandomTrace<32> for SudokuSelectedUair<R, FILL>
where
    R: ConstSemiring + From<u32> + 'static,
{
    type PolyCoeff = R;
    type Int = R;

    fn generate_random_trace<Rng: RngCore + ?Sized>(
        num_vars: usize,
        _rng: &mut Rng,
    ) -> UairTrace<'static, R, R, 32> {
        let row_count = 1usize << num_vars;
        let mut grid: Vec<Vec<u32>> = SUDOKU_SOLUTION
            .iter()
            .map(|row| {
                let mut cells = row.to_vec();
                cells.resize(row_count, 0);
                cells
            })
            .collect();
        match FILL {
            SUDOKU_DUPLICATE => grid[0][1] = grid[0][0],
            SUDOKU_SLID => {
                grid[0][9] = grid[0][0];
                grid[0][0] = 0;
            }
            SUDOKU_SWAPPED => {
                let held = grid[0][0];
                grid[0][0] = grid[1][1];
                grid[1][1] = held;
            }
            SUDOKU_UNSELECTED => grid[3][12] = 4242,
            _ => {}
        }
        UairTrace {
            int: grid_columns(grid, num_vars).into(),
            ..Default::default()
        }
    }
}

/// The same grid, with a clue's worth of its cells fixed by point ties
/// rather than by a lookup: seventeen cells at positions the statement
/// names, each pinned to the number it holds. A pin is verifier
/// geometry -- the cell's `eq` is evaluated at the sumcheck point rather
/// than read out of the proof -- so nothing about the seventeen
/// positions is committed and nothing is sent.
///
/// `FILL` lays the grid down, `DECL` says what the signature declares,
/// so the same grid can be handed a declaration that does not fit it and
/// the same declaration a grid that does not answer it.
#[derive(Clone, Debug)]
pub struct SudokuPinnedUair<R, const FILL: u8, const DECL: u8>(PhantomData<R>);

/// The grid solves and holds every clue.
pub const PINNED_SOLVED: u8 = 0;
/// A pinned cell holds a number other than the one pinned to it.
pub const PINNED_TAMPERED: u8 = 1;
/// A cell no tie names holds a number no table names either.
pub const PINNED_FREE: u8 = 2;

/// Seventeen pins, one per clue.
pub const DECL_PINS: u8 = 0;
/// Seventeen pins, one of them naming a value the solution does not hold
/// at the cell it names.
pub const DECL_PINS_WRONG: u8 = 1;
/// The seventeen pins and the twenty-seven selections in one statement.
pub const DECL_PINS_LOOKUPS: u8 = 2;
/// Sixteen pins: a declaration one clue short of [`DECL_PINS`].
pub const DECL_PINS_SHORT: u8 = 3;
/// No ties and no lookups: the control the pins' cost is read against.
pub const DECL_NONE: u8 = 4;

/// The seventeen cells the pins fix, as `(grid row, position)` -- the
/// grid's rows being the committed columns, a cell is column `row` at
/// trace row `position`.
pub const SUDOKU_CLUES: [(usize, usize); 17] = [
    (0, 0),
    (0, 4),
    (1, 2),
    (1, 6),
    (2, 1),
    (2, 8),
    (3, 3),
    (3, 5),
    (4, 0),
    (4, 7),
    (5, 2),
    (5, 4),
    (6, 1),
    (6, 6),
    (7, 3),
    (7, 8),
    (8, 5),
];

/// The pins a `DECL` declares.
fn sudoku_pins(decl: u8) -> Vec<PointTie> {
    let clues: &[(usize, usize)] = match decl {
        DECL_NONE => &[],
        DECL_PINS_SHORT => &SUDOKU_CLUES[..16],
        _ => &SUDOKU_CLUES,
    };
    clues
        .iter()
        .enumerate()
        .map(|(i, &(row, position))| {
            let held = u64::from(SUDOKU_SOLUTION[row][position]);
            let value = match (decl, i) {
                (DECL_PINS_WRONG, 0) => held + 1,
                _ => held,
            };
            PointTie::pin(row, position, value)
        })
        .collect()
}

impl<R, const FILL: u8, const DECL: u8> Uair for SudokuPinnedUair<R, FILL, DECL>
where
    R: ConstSemiring + 'static,
{
    type Ideal = ImpossibleIdeal;
    type Scalar = DensePolynomial<R, 32>;

    fn signature() -> UairSignature {
        let total = TotalColumnLayout::new(0, 0, SUDOKU_COLS);
        let lookup_specs: Vec<LookupColumnSpec> = match DECL {
            DECL_PINS_LOOKUPS => (0..SUDOKU_COLS)
                .map(|i| LookupColumnSpec {
                    column_index: i,
                    table_type: sudoku_table(),
                })
                .collect(),
            _ => vec![],
        };
        UairSignature::new(total, PublicColumnLayout::default(), vec![], lookup_specs, vec![])
            .with_point_ties(sudoku_pins(DECL))
    }

    fn constrain_general<B, FromR, MulByScalar, IFromR>(
        b: &mut B,
        up: TraceRow<B::Expr>,
        _down: TraceRow<B::Expr>,
        _from_ref: FromR,
        _mbs: MulByScalar,
        _ideal_from_ref: IFromR,
    ) where
        B: ConstraintBuilder,
    {
        // Trivially satisfied: the ties carry the whole claim.
        let v = &up.int[0];
        b.assert_zero(v.clone() - v);
    }
}

impl<R, const FILL: u8, const DECL: u8> GenerateRandomTrace<32> for SudokuPinnedUair<R, FILL, DECL>
where
    R: ConstSemiring + From<u32> + 'static,
{
    type PolyCoeff = R;
    type Int = R;

    fn generate_random_trace<Rng: RngCore + ?Sized>(
        num_vars: usize,
        _rng: &mut Rng,
    ) -> UairTrace<'static, R, R, 32> {
        let row_count = 1usize << num_vars;
        let mut grid: Vec<Vec<u32>> = SUDOKU_SOLUTION
            .iter()
            .map(|row| {
                let mut cells = row.to_vec();
                cells.resize(row_count, 0);
                cells
            })
            .collect();
        match FILL {
            PINNED_TAMPERED => {
                let (row, position) = SUDOKU_CLUES[0];
                grid[row][position] += 1;
            }
            PINNED_FREE => grid[0][9] = 4242,
            _ => {}
        }
        UairTrace {
            int: grid_columns(grid, num_vars).into(),
            ..Default::default()
        }
    }
}

/// Two tables in one statement: the solved grid of [`SudokuSelectedUair`]
/// in the first nine integer columns, declared against a `Selected` table,
/// and four range-checked columns after it declared against a `Word` one,
/// with four of the clues pinned besides. A statement that owes both a
/// multiset and a range owes a lookup group for each, and each group is
/// discharged at its own point.
///
/// This is the shape a sudoku with slack takes: the grid's obligations
/// and the slack cells' widths are separate tables over separate columns,
/// and neither is a reason to refuse the other.
///
/// `FILL` says how the columns are laid down, so either side can be
/// broken while the other stands.
#[derive(Clone, Debug)]
pub struct SudokuRangedUair<R, const FILL: u8>(PhantomData<R>);

/// The grid's nine columns and the four range-checked ones after them.
pub const RANGED_COLS: usize = SUDOKU_COLS + INT_WORD_COLS;
/// Both tables hold, and so does every pin.
pub const RANGED_SOLVED: u8 = 0;
/// A slack cell wider than the `Word` table names: the range side breaks
/// and the grid stands.
pub const RANGED_OVER_WIDTH: u8 = 1;
/// A value repeated inside a grid row: the multiset side breaks and the
/// range stands.
pub const RANGED_DUPLICATE: u8 = 2;

/// The four clues [`SudokuRangedUair`] pins.
fn ranged_pins() -> Vec<PointTie> {
    SUDOKU_CLUES[..4]
        .iter()
        .map(|&(row, position)| {
            PointTie::pin(row, position, u64::from(SUDOKU_SOLUTION[row][position]))
        })
        .collect()
}

impl<R, const FILL: u8> Uair for SudokuRangedUair<R, FILL>
where
    R: ConstSemiring + 'static,
{
    type Ideal = ImpossibleIdeal;
    type Scalar = DensePolynomial<R, 32>;

    fn signature() -> UairSignature {
        let total = TotalColumnLayout::new(0, 0, RANGED_COLS);
        let grid = (0..SUDOKU_COLS).map(|i| LookupColumnSpec {
            column_index: i,
            table_type: sudoku_table(),
        });
        let slack = (SUDOKU_COLS..RANGED_COLS).map(|i| LookupColumnSpec {
            column_index: i,
            table_type: LookupTableType::Word {
                width: INT_WORD_WIDTH,
                chunk_width: Some(8),
            },
        });
        UairSignature::new(
            total,
            PublicColumnLayout::default(),
            vec![],
            grid.chain(slack).collect(),
            vec![],
        )
        .with_point_ties(ranged_pins())
    }

    fn constrain_general<B, FromR, MulByScalar, IFromR>(
        b: &mut B,
        up: TraceRow<B::Expr>,
        _down: TraceRow<B::Expr>,
        _from_ref: FromR,
        _mbs: MulByScalar,
        _ideal_from_ref: IFromR,
    ) where
        B: ConstraintBuilder,
    {
        // Trivially satisfied: the tables and the ties carry the claim.
        let v = &up.int[0];
        b.assert_zero(v.clone() - v);
    }
}

impl<R, const FILL: u8> GenerateRandomTrace<32> for SudokuRangedUair<R, FILL>
where
    R: ConstSemiring + From<u32> + 'static,
{
    type PolyCoeff = R;
    type Int = R;

    fn generate_random_trace<Rng: RngCore + ?Sized>(
        num_vars: usize,
        rng: &mut Rng,
    ) -> UairTrace<'static, R, R, 32> {
        let row_count = 1usize << num_vars;
        let mut grid: Vec<Vec<u32>> = SUDOKU_SOLUTION
            .iter()
            .map(|row| {
                let mut cells = row.to_vec();
                cells.resize(row_count, 0);
                cells
            })
            .collect();
        let bound: u32 = 1 << INT_WORD_WIDTH;
        let mut slack: Vec<Vec<u32>> = (0..INT_WORD_COLS)
            .map(|_| (0..row_count).map(|_| rng.next_u32() % bound).collect())
            .collect();
        match FILL {
            RANGED_OVER_WIDTH => slack[0][0] = bound,
            RANGED_DUPLICATE => grid[0][1] = grid[0][0],
            _ => {}
        }
        grid.append(&mut slack);
        UairTrace {
            int: grid_columns(grid, num_vars).into(),
            ..Default::default()
        }
    }
}

/// A broadcast tie at work: one cell's private value, carried at every
/// row of a column of its own, read by an ordinary uniform constraint.
///
/// Column 0 holds a row of the solution, column 1 is the tie's target,
/// and column 2 is a column the constraint `c_2 - c_1 = 0` holds equal
/// to it everywhere. The tied cell's value is neither public nor in the
/// proof: the tie says column 1 is constant and that its constant is the
/// cell, and the constraint reads it from there.
#[derive(Clone, Debug)]
pub struct BroadcastTieUair<R, const FILL: u8>(PhantomData<R>);

/// Which cell of column 0 the broadcast carries.
pub const BROADCAST_ROW: usize = 4;
/// The target column is that cell's value at every row.
pub const BROADCAST_OK: u8 = 0;
/// The target column holds the cell's value at the tied row and another
/// value elsewhere: constant is what it is not.
pub const BROADCAST_VARYING: u8 = 1;
/// The target column is constant, at a value the tied cell does not hold.
pub const BROADCAST_WRONG: u8 = 2;

impl<R, const FILL: u8> Uair for BroadcastTieUair<R, FILL>
where
    R: ConstSemiring + 'static,
{
    type Ideal = ImpossibleIdeal;
    type Scalar = DensePolynomial<R, 32>;

    fn signature() -> UairSignature {
        UairSignature::new(
            TotalColumnLayout::new(0, 0, 3),
            PublicColumnLayout::default(),
            vec![],
            vec![],
            vec![],
        )
        .with_point_ties(vec![PointTie::broadcast(0, BROADCAST_ROW, 1)])
    }

    fn constrain_general<B, FromR, MulByScalar, IFromR>(
        b: &mut B,
        up: TraceRow<B::Expr>,
        _down: TraceRow<B::Expr>,
        _from_ref: FromR,
        _mbs: MulByScalar,
        _ideal_from_ref: IFromR,
    ) where
        B: ConstraintBuilder,
    {
        // The reader column equals the broadcast one at every row, which
        // is the private value of a cell this constraint never names.
        b.assert_zero(up.int[2].clone() - &up.int[1]);
    }
}

impl<R, const FILL: u8> GenerateRandomTrace<32> for BroadcastTieUair<R, FILL>
where
    R: ConstSemiring + From<u32> + 'static,
{
    type PolyCoeff = R;
    type Int = R;

    fn generate_random_trace<Rng: RngCore + ?Sized>(
        num_vars: usize,
        _rng: &mut Rng,
    ) -> UairTrace<'static, R, R, 32> {
        let row_count = 1usize << num_vars;
        let mut source = SUDOKU_SOLUTION[0].to_vec();
        source.resize(row_count, 0);
        let held = SUDOKU_SOLUTION[0][BROADCAST_ROW];
        let mut broadcast = match FILL {
            BROADCAST_WRONG => vec![held + 1; row_count],
            _ => vec![held; row_count],
        };
        if FILL == BROADCAST_VARYING {
            broadcast[0] = held + 1;
        }
        let reader = broadcast.clone();
        UairTrace {
            int: grid_columns(vec![source, broadcast, reader], num_vars).into(),
            ..Default::default()
        }
    }
}

/// The integer columns a grid of numbers commits as.
fn grid_columns<R: ConstSemiring + From<u32>>(
    grid: Vec<Vec<u32>>,
    num_vars: usize,
) -> Vec<DenseMultilinearExtension<R>> {
    grid.into_iter()
        .map(|cells| {
            let evals: Vec<R> = cells.into_iter().map(R::from).collect();
            DenseMultilinearExtension::from_evaluations_vec(num_vars, evals, R::ZERO)
        })
        .collect()
}

/// No-lookup control for [`BinLookup16Uair`]: identical 16-column layout
/// and trivial constraint but **no lookup specs**. With this UAIR step 4b
/// early-returns and the step-7 reducer is skipped, so an A/B against the
/// lookup-bearing variant isolates the lookup machinery's cost.
#[derive(Clone, Debug)]
pub struct BinLookup16NoLookupUair<R>(PhantomData<R>);

impl<R> Uair for BinLookup16NoLookupUair<R>
where
    R: ConstSemiring + 'static,
{
    type Ideal = ImpossibleIdeal;
    type Scalar = DensePolynomial<R, 32>;

    fn signature() -> UairSignature {
        let total = TotalColumnLayout::new(16, 0, 0);
        UairSignature::new(total, PublicColumnLayout::default(), vec![], vec![], vec![])
    }

    fn constrain_general<B, FromR, MulByScalar, IFromR>(
        b: &mut B,
        up: TraceRow<B::Expr>,
        _down: TraceRow<B::Expr>,
        _from_ref: FromR,
        _mbs: MulByScalar,
        _ideal_from_ref: IFromR,
    ) where
        B: ConstraintBuilder,
    {
        let v = &up.binary_poly[0];
        b.assert_zero(v.clone() - v);
    }
}

impl<R> GenerateRandomTrace<32> for BinLookup16NoLookupUair<R>
where
    R: ConstSemiring + 'static,
{
    type PolyCoeff = R;
    type Int = R;

    fn generate_random_trace<Rng: RngCore + ?Sized>(
        num_vars: usize,
        rng: &mut Rng,
    ) -> UairTrace<'static, R, R, 32> {
        let row_count = 1usize << num_vars;
        let cols: Vec<DenseMultilinearExtension<BinaryPoly<32>>> = (0..16)
            .map(|_| {
                (0..row_count)
                    .map(|_| BinaryPoly::<32>::from(rng.next_u32()))
                    .collect::<DenseMultilinearExtension<BinaryPoly<32>>>()
            })
            .collect();
        UairTrace {
            binary_poly: cols.into(),
            ..Default::default()
        }
    }
}

/// Multi-group variant of [`BinLookup16Uair`]: 16 witness binary_poly
/// columns split across TWO lookup groups — 12 cols as `BitPoly{32,8}`
/// (L=12) and 4 cols as `BitPoly{32,16}` (L=4). n_groups = 2, so step 7
/// exercises the bin multipoint reducer (folding the two per-group
/// `r_inner` claims plus the step-7 `r_0` claim into ONE Zip+ open at
/// the reduced point r*). Used to validate the reducer path.
#[derive(Clone, Debug)]
pub struct BinLookup16MultiGroupUair<R>(PhantomData<R>);

impl<R> Uair for BinLookup16MultiGroupUair<R>
where
    R: ConstSemiring + 'static,
{
    type Ideal = ImpossibleIdeal;
    type Scalar = DensePolynomial<R, 32>;

    fn signature() -> UairSignature {
        let total = TotalColumnLayout::new(16, 0, 0);
        let mut lookup_specs: Vec<LookupColumnSpec> = (0..12)
            .map(|i| LookupColumnSpec {
                column_index: i,
                table_type: LookupTableType::BitPoly { width: 32, chunk_width: Some(8) },
            })
            .collect();
        lookup_specs.extend((12..16).map(|i| LookupColumnSpec {
            column_index: i,
            table_type: LookupTableType::BitPoly { width: 32, chunk_width: Some(16) },
        }));
        UairSignature::new(total, PublicColumnLayout::default(), vec![], lookup_specs, vec![])
    }

    fn constrain_general<B, FromR, MulByScalar, IFromR>(
        b: &mut B,
        up: TraceRow<B::Expr>,
        _down: TraceRow<B::Expr>,
        _from_ref: FromR,
        _mbs: MulByScalar,
        _ideal_from_ref: IFromR,
    ) where
        B: ConstraintBuilder,
    {
        // Trivially-satisfied constraint (lookups carry the soundness work).
        let v = &up.binary_poly[0];
        b.assert_zero(v.clone() - v);
    }
}

impl<R> GenerateRandomTrace<32> for BinLookup16MultiGroupUair<R>
where
    R: ConstSemiring + 'static,
{
    type PolyCoeff = R;
    type Int = R;

    fn generate_random_trace<Rng: RngCore + ?Sized>(
        num_vars: usize,
        rng: &mut Rng,
    ) -> UairTrace<'static, R, R, 32> {
        let row_count = 1usize << num_vars;
        let cols: Vec<DenseMultilinearExtension<BinaryPoly<32>>> = (0..16)
            .map(|_| {
                (0..row_count)
                    .map(|_| BinaryPoly::<32>::from(rng.next_u32()))
                    .collect::<DenseMultilinearExtension<BinaryPoly<32>>>()
            })
            .collect();
        UairTrace {
            binary_poly: cols.into(),
            ..Default::default()
        }
    }
}

#[derive(Clone, Debug)]
pub struct TestUairNoMultiplication<R>(PhantomData<R>);

impl<R> Uair for TestUairNoMultiplication<R>
where
    R: ConstSemiring + From<i32> + 'static,
{
    type Ideal = DegreeOneIdeal<R>;
    type Scalar = DensePolynomial<R, 32>;

    fn signature() -> UairSignature {
        let total = TotalColumnLayout::new(0, 3, 0);
        UairSignature::new(total, PublicColumnLayout::default(), vec![], vec![], vec![])
    }

    fn constrain_general<B, FromR, MulByScalar, IFromR>(
        b: &mut B,
        up: TraceRow<B::Expr>,
        _down: TraceRow<B::Expr>,
        _from_ref: FromR,
        _mbs: MulByScalar,
        ideal_from_ref: IFromR,
    ) where
        B: ConstraintBuilder,
        IFromR: Fn(&Self::Ideal) -> B::Ideal,
    {
        let up = up.arbitrary_poly;

        b.assert_in_ideal(
            up[0].clone() + &up[1] - &up[2],
            &ideal_from_ref(&DegreeOneIdeal::new(R::from(2))),
        );
    }
}

impl<R> GenerateRandomTrace<32> for TestUairNoMultiplication<R>
where
    R: ConstSemiring + From<i32> + 'static,
{
    type PolyCoeff = R;
    type Int = R;

    fn generate_random_trace<Rng: rand::RngCore + ?Sized>(
        num_vars: usize,
        rng: &mut Rng,
    ) -> UairTrace<'static, R, R, 32> {
        let a: DenseMultilinearExtension<DensePolynomial<R, 32>> =
            DenseMultilinearExtension::rand(num_vars, rng)
                .into_iter()
                .map(|x: u32| {
                    DensePolynomial::from_ref(&DensePolynomial::<Boolean, _>::from(
                        BinaryPoly::<32>::from(x),
                    ))
                })
                .collect();

        let b: DenseMultilinearExtension<_> = DenseMultilinearExtension::rand(num_vars, rng)
            .into_iter()
            .map(|x: u32| {
                DensePolynomial::from_ref(&DensePolynomial::<Boolean, _>::from(
                    BinaryPoly::<32>::from(x),
                ))
            })
            .collect();

        let c = a.clone() + b.clone();

        UairTrace {
            arbitrary_poly: vec![a, b, c].into(),
            ..Default::default()
        }
    }
}

#[derive(Clone, Debug)]
pub struct TestUairScalarMultiplications<R>(PhantomData<R>);

impl<R> Uair for TestUairScalarMultiplications<R>
where
    R: ConstSemiring + From<i8> + 'static,
{
    type Ideal = DegreeOneIdeal<R>;
    type Scalar = DensePolynomial<R, 32>;

    fn signature() -> UairSignature {
        let total = TotalColumnLayout::new(0, 3, 0);
        UairSignature::new(total, PublicColumnLayout::default(), vec![], vec![], vec![])
    }

    fn constrain_general<B, FromR, MulByScalar, IFromR>(
        b: &mut B,
        up: TraceRow<B::Expr>,
        _down: TraceRow<B::Expr>,
        from_ref: FromR,
        mbs: MulByScalar,
        ideal_from_ref: IFromR,
    ) where
        B: ConstraintBuilder,
        IFromR: Fn(&Self::Ideal) -> B::Ideal,
        FromR: Fn(&DensePolynomial<R, 32>) -> B::Expr,
        MulByScalar: Fn(&B::Expr, &DensePolynomial<R, 32>) -> Option<B::Expr>,
    {
        let up = up.arbitrary_poly;

        b.assert_in_ideal(
            mbs(
                &up[0],
                &DensePolynomial::new([R::from(-1), R::from(0), R::from(1)]),
            )
            .expect("arithmetic overflow")
                + &up[1]
                - &up[2]
                + from_ref(&DensePolynomial::new([
                    R::from(1),
                    R::from(2),
                    R::from(3),
                    R::from(4),
                ])),
            &ideal_from_ref(&DegreeOneIdeal::new(R::from(2))),
        );
    }
}

#[derive(Clone, Debug)]
pub struct BinaryDecompositionUair<R>(PhantomData<R>);

impl<R> Uair for BinaryDecompositionUair<R>
where
    R: ConstSemiring + From<u32> + 'static,
{
    type Ideal = DegreeOneIdeal<R>;
    type Scalar = DensePolynomial<R, 32>;

    fn signature() -> UairSignature {
        let total = TotalColumnLayout::new(1, 0, 1);
        UairSignature::new(total, PublicColumnLayout::default(), vec![], vec![], vec![])
    }

    fn constrain_general<B, FromR, MulByScalar, IFromR>(
        b: &mut B,
        up: TraceRow<B::Expr>,
        _down: TraceRow<B::Expr>,
        _from_ref: FromR,
        _mbs: MulByScalar,
        ideal_from_ref: IFromR,
    ) where
        B: ConstraintBuilder,
        FromR: Fn(&Self::Scalar) -> B::Expr,
        MulByScalar: Fn(&B::Expr, &Self::Scalar) -> Option<B::Expr>,
        IFromR: Fn(&Self::Ideal) -> B::Ideal,
    {
        let int_col = &up.int[0];
        let binary_poly_col = &up.binary_poly[0];

        b.assert_in_ideal(
            binary_poly_col.clone() - int_col,
            &ideal_from_ref(&DegreeOneIdeal::new(R::from(2))),
        );
    }
}

impl<R> GenerateRandomTrace<32> for BinaryDecompositionUair<R>
where
    R: ConstSemiring + From<u32> + 'static,
{
    type PolyCoeff = R;
    type Int = R;

    fn generate_random_trace<Rng: rand::RngCore + ?Sized>(
        num_vars: usize,
        rng: &mut Rng,
    ) -> UairTrace<'static, R, R, 32> {
        let int_col_u32: DenseMultilinearExtension<u32> =
            DenseMultilinearExtension::rand(num_vars, rng);

        let binary_poly_col: DenseMultilinearExtension<BinaryPoly<32>> =
            int_col_u32.iter().map(|i| BinaryPoly::from(*i)).collect();

        let int_col = int_col_u32.into_iter().map(R::from).collect();

        UairTrace {
            binary_poly: vec![binary_poly_col].into(),
            arbitrary_poly: vec![].into(),
            int: vec![int_col].into(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct BigLinearUair<R>(PhantomData<R>);

impl<R> Uair for BigLinearUair<R>
where
    R: ConstSemiring + From<u32> + 'static,
{
    type Ideal = DegreeOneIdeal<R>;
    type Scalar = DensePolynomial<R, 32>;

    fn signature() -> UairSignature {
        let total = TotalColumnLayout::new(16, 0, 1);
        let shifts = (0..16).map(|i| ShiftSpec::new(i, 1)).collect();
        UairSignature::new(total, PublicColumnLayout::default(), shifts, vec![], vec![])
    }

    fn constrain_general<B, FromR, MulByScalar, IFromR>(
        b: &mut B,
        up: TraceRow<B::Expr>,
        down: TraceRow<B::Expr>,
        _from_ref: FromR,
        _mbs: MulByScalar,
        ideal_from_ref: IFromR,
    ) where
        B: ConstraintBuilder,
        FromR: Fn(&Self::Scalar) -> B::Expr,
        MulByScalar: Fn(&B::Expr, &Self::Scalar) -> Option<B::Expr>,
        IFromR: Fn(&Self::Ideal) -> B::Ideal,
    {
        let one_ideal = DegreeOneIdeal::new(R::from(1));
        let two_ideal = DegreeOneIdeal::new(R::from(2));

        let sum_of_binary_polys = up.binary_poly[1..]
            .iter()
            .fold(up.binary_poly[0].clone(), |acc, next| acc + next);

        // up.binary_poly[0] + up.binary_poly[1] + ... up.binary_poly[16]
        //      = up.int[0] mod (X - 1)
        b.assert_in_ideal(
            sum_of_binary_polys - &up.int[0],
            &ideal_from_ref(&one_ideal),
        );

        // down.binary_poly[0] = up.int[0] mod (X - 1)
        b.assert_in_ideal(
            down.binary_poly[0].clone() - &up.int[0],
            &ideal_from_ref(&two_ideal),
        );

        // down.binary_poly[i](1) = up.binary_poly[i](1), for all i=1,...,15
        // (preserves popcount across rows, but allows the bit pattern to change)
        up.binary_poly[1..]
            .iter()
            .zip(&down.binary_poly[1..])
            .for_each(|(up, down)| {
                b.assert_in_ideal(up.clone() - down, &ideal_from_ref(&one_ideal));
            });
    }
}

impl<R> GenerateRandomTrace<32> for BigLinearUair<R>
where
    R: ConstSemiring + From<u32> + 'static,
{
    type PolyCoeff = R;
    type Int = R;

    fn generate_random_trace<Rng: rand::RngCore + ?Sized>(
        num_vars: usize,
        rng: &mut Rng,
    ) -> UairTrace<'static, R, R, 32> {
        /// Generate a random binary polynomial with the given number of 1-bits.
        fn random_binary_poly_with_popcount(
            popcount: u32,
            rng: &mut (impl rand::RngCore + ?Sized),
        ) -> BinaryPoly<32> {
            let mut positions: [u8; 32] =
                core::array::from_fn(|i| u8::try_from(i).expect("can't fail"));
            for i in 0..popcount as usize {
                let j = i + rng.next_u32() as usize % (32 - i);
                positions.swap(i, j);
            }
            let mut value: u32 = 0;
            for &pos in &positions[..popcount as usize] {
                value |= 1u32 << pos;
            }
            BinaryPoly::from(value)
        }

        let mut binary_poly_cols: Vec<DenseMultilinearExtension<BinaryPoly<32>>> =
            vec![(0..(1 << num_vars)).map(|_| BinaryPoly::zero()).collect(); 16];
        let mut int_col: DenseMultilinearExtension<Self::Int> =
            (0..(1 << num_vars)).map(|_| R::ZERO).collect();

        binary_poly_cols.iter_mut().for_each(|col| {
            col[0] = rng.random();
        });

        for i in 0..(1 << num_vars) - 1 {
            let int: u32 = binary_poly_cols
                .iter()
                .map(|col| col[i].evaluate_at_point(&1_u32).expect("should be fine"))
                .sum();
            int_col[i] = R::from(int);

            binary_poly_cols[0][i + 1] = BinaryPoly::from(int);
            binary_poly_cols[1..].iter_mut().for_each(|col| {
                let popcount = col[i].evaluate_at_point(&1_u32).expect("should be fine");
                col[i + 1] = random_binary_poly_with_popcount(popcount, rng);
            });
        }

        let len = int_col.len();

        int_col[len - 1] = R::from(
            binary_poly_cols
                .iter()
                .map(|col| {
                    col[len - 1]
                        .evaluate_at_point(&1_u32)
                        .expect("should be fine")
                })
                .sum::<u32>(),
        );

        UairTrace {
            binary_poly: binary_poly_cols.into(),
            arbitrary_poly: vec![].into(),
            int: vec![int_col].into(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct BigLinearUairWithPublicInput<R>(PhantomData<R>);

impl<R> Uair for BigLinearUairWithPublicInput<R>
where
    R: ConstSemiring + From<u32> + 'static,
{
    type Ideal = <BigLinearUair<R> as Uair>::Ideal;
    type Scalar = <BigLinearUair<R> as Uair>::Scalar;

    fn signature() -> UairSignature {
        let total = TotalColumnLayout::new(16, 0, 1);
        let public = PublicColumnLayout::new(4, 0, 0);
        let shifts = (0..16).map(|i| ShiftSpec::new(i, 1)).collect();
        UairSignature::new(total, public, shifts, vec![], vec![])
    }

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
        IFromR: Fn(&Self::Ideal) -> B::Ideal,
    {
        BigLinearUair::<R>::constrain_general(b, up, down, from_ref, mbs, ideal_from_ref)
    }
}

impl<R> GenerateRandomTrace<32> for BigLinearUairWithPublicInput<R>
where
    R: ConstSemiring + From<u32> + 'static,
{
    type PolyCoeff = <BigLinearUair<R> as GenerateRandomTrace<32>>::PolyCoeff;
    type Int = <BigLinearUair<R> as GenerateRandomTrace<32>>::Int;

    fn generate_random_trace<Rng: RngCore + ?Sized>(
        num_vars: usize,
        rng: &mut Rng,
    ) -> UairTrace<'static, Self::PolyCoeff, Self::Int, 32> {
        BigLinearUair::<R>::generate_random_trace(num_vars, rng)
    }
}

/// A second "big linear" UAIR with 14 binary-poly columns and 4 int columns,
/// used as a benchmarking shape distinct from `BigLinearUair`.
///
/// Constraints (0-based; `bp = up.binary_poly`, `int = up.int`):
///
/// - `bp[0][t+1] - bp[1] - bp[2] - bp[3] - int[0] - int[1] - int[2] ∈ (X-2)`
/// - `bp[4][t+4] - bp[5] - bp[6] - bp[7] - int[1] - int[2] - int[3] ∈ (X-2)`
/// - `bp[8] - int[0] ∈ (X-2)`
/// - `bp[9] - int[1] ∈ (X-2)`
/// - `bp[10] - X * bp[11] ∈ (X-1)`
/// - `bp[12] - X * bp[13] ∈ (X-1)`
///
/// Note the asymmetric shift amounts: `bp[0]` is shifted by 1 (used by C1)
/// and `bp[4]` is shifted by 4 (used by C2).
#[derive(Clone, Debug)]
pub struct ShaProxy<R>(PhantomData<R>);

impl<R> Uair for ShaProxy<R>
where
    R: ConstSemiring + From<u32> + 'static,
{
    type Ideal = DegreeOneIdeal<R>;
    type Scalar = DensePolynomial<R, 32>;

    fn signature() -> UairSignature {
        // 14 binary_poly cols, 0 arbitrary_poly cols, 4 int cols.
        let total = TotalColumnLayout::new(14, 0, 4);
        // c_1 (bp[0]) is shifted by 1 (used by C1 as bp[0][t+1]); c_5 (bp[4])
        // is shifted by 4 (used by C2 as bp[4][t+4]).
        let shifts = vec![ShiftSpec::new(0, 1), ShiftSpec::new(4, 4)];
        UairSignature::new(total, PublicColumnLayout::default(), shifts, vec![], vec![])
    }

    fn constrain_general<B, FromR, MulByScalar, IFromR>(
        b: &mut B,
        up: TraceRow<B::Expr>,
        down: TraceRow<B::Expr>,
        _from_ref: FromR,
        mbs: MulByScalar,
        ideal_from_ref: IFromR,
    ) where
        B: ConstraintBuilder,
        FromR: Fn(&Self::Scalar) -> B::Expr,
        MulByScalar: Fn(&B::Expr, &Self::Scalar) -> Option<B::Expr>,
        IFromR: Fn(&Self::Ideal) -> B::Ideal,
    {
        let one_ideal = ideal_from_ref(&DegreeOneIdeal::new(R::ONE));
        let two_ideal = ideal_from_ref(&DegreeOneIdeal::new(R::from(2)));
        // The polynomial X = 0 + 1*X, used to express `X * c_k` via `mbs`.
        let x_scalar = DensePolynomial::<R, 32>::new([R::ZERO, R::from(1)]);

        // `down.binary_poly` is indexed by ShiftSpec position, not source col.
        // Our shifts vec is [ShiftSpec::new(0, 1), ShiftSpec::new(4, 1)], so
        // down.binary_poly[0] = bp[0][t+1], down.binary_poly[1] = bp[4][t+1].

        // (C1) dbp[0] - bp[1] - bp[2] - bp[3] - int[0] - int[1] - int[2] ∈ (X-2)
        b.assert_in_ideal(
            down.binary_poly[0].clone()
                - &up.binary_poly[1]
                - &up.binary_poly[2]
                - &up.binary_poly[3]
                - &up.int[0]
                - &up.int[1]
                - &up.int[2],
            &two_ideal,
        );

        // (C2) dbp[4] - bp[5] - bp[6] - bp[7] - int[1] - int[2] - int[3] ∈ (X-2)
        b.assert_in_ideal(
            down.binary_poly[1].clone()
                - &up.binary_poly[5]
                - &up.binary_poly[6]
                - &up.binary_poly[7]
                - &up.int[1]
                - &up.int[2]
                - &up.int[3],
            &two_ideal,
        );

        // (C3) bp[8] - int[0] ∈ (X-2)
        b.assert_in_ideal(up.binary_poly[8].clone() - &up.int[0], &two_ideal);

        // (C4) bp[9] - int[1] ∈ (X-2)
        b.assert_in_ideal(up.binary_poly[9].clone() - &up.int[1], &two_ideal);

        // (C5) bp[10] - X * bp[11] ∈ (X-1)
        b.assert_in_ideal(
            up.binary_poly[10].clone()
                - &mbs(&up.binary_poly[11], &x_scalar).expect("mul-by-X overflow"),
            &one_ideal,
        );

        // (C6) bp[12] - X * bp[13] ∈ (X-1)
        b.assert_in_ideal(
            up.binary_poly[12].clone()
                - &mbs(&up.binary_poly[13], &x_scalar).expect("mul-by-X overflow"),
            &one_ideal,
        );
    }
}

impl<R> GenerateRandomTrace<32> for ShaProxy<R>
where
    R: ConstSemiring + From<u32> + 'static,
{
    type PolyCoeff = R;
    type Int = R;

    #[allow(clippy::needless_range_loop)]
    fn generate_random_trace<Rng: rand::RngCore + ?Sized>(
        num_vars: usize,
        rng: &mut Rng,
    ) -> UairTrace<'static, R, R, 32> {
        /// Generate a random binary polynomial with the given number of 1-bits.
        fn random_binary_poly_with_popcount(
            popcount: u32,
            rng: &mut (impl rand::RngCore + ?Sized),
        ) -> BinaryPoly<32> {
            let mut positions: [u8; 32] =
                core::array::from_fn(|i| u8::try_from(i).expect("can't fail"));
            for i in 0..popcount as usize {
                let j = i + rng.next_u32() as usize % (32 - i);
                positions.swap(i, j);
            }
            let mut value: u32 = 0;
            for &pos in &positions[..popcount as usize] {
                value |= 1_u32 << pos;
            }
            BinaryPoly::from(value)
        }

        // Bits used by the "small" binary polys that feed into C1/C2 sums.
        // Capping at 28 bits keeps each `eval(2)` value below 2^28 - 1, so the
        // sum used to construct `bp[0]` / `bp[4]` at the next row stays in u32:
        //     3 * (2^28 - 1) + 3 * 31  ≈  8.05 * 10^8  <  2^32 - 1.
        const SMALL_MASK: u32 = (1 << 28) - 1;
        // Range of values for the int columns (small non-negative).
        const INT_MAX_EXCL: u32 = 32;

        let len = 1 << num_vars;

        let mut bp_cols: Vec<DenseMultilinearExtension<BinaryPoly<32>>> =
            vec![(0..len).map(|_| BinaryPoly::zero()).collect(); 14];
        let mut int_cols: Vec<DenseMultilinearExtension<R>> =
            vec![(0..len).map(|_| R::ZERO).collect(); 4];

        // Row 0 / "head row" init for the columns whose value at the head of
        // the trace is unconstrained:
        //   - `bp[0]` is shifted by 1, so only `bp[0][0]` is unconstrained.
        //   - `bp[4]` is shifted by 4, so `bp[4][0..4]` are unconstrained (the C2
        //     fix-up at iteration `i` writes `bp[4][i+4]`, so the first 4 indices are
        //     never written by the loop).
        bp_cols[0][0] = BinaryPoly::from(rng.next_u32() & SMALL_MASK);
        for k in 0..4.min(len) {
            bp_cols[4][k] = BinaryPoly::from(rng.next_u32() & SMALL_MASK);
        }

        for i in 0..len {
            // bp[1..=3]: always small random binary polys (28-bit values).
            for k in 1..=3 {
                bp_cols[k][i] = BinaryPoly::from(rng.next_u32() & SMALL_MASK);
            }

            // For rows where the C2-target `bp[4][i+4]` is past the trace
            // boundary, the protocol reads it as zero-padded. To still
            // satisfy C2 at those rows, the C2 RHS sum must vanish at X = 2;
            // we achieve that by zeroing `bp[5..=7]` and `int[1..=3]` here.
            // (Mirrors the boundary trick used by `TestUairMixedShifts`.)
            // Row `len - 1` is exempt from constraint checking by the
            // protocol's last-row selector, so the zeroing only matters for
            // rows `len - 4 ..= len - 2`.
            let c2_target_oob = i + 4 >= len;

            for k in 5..=7 {
                bp_cols[k][i] = if c2_target_oob {
                    BinaryPoly::zero()
                } else {
                    BinaryPoly::from(rng.next_u32() & SMALL_MASK)
                };
            }

            // int[0..=3]: small non-negative values. Zero out int[1..=3] when
            // we're in the C2 boundary region (int[0] can stay random — it is
            // not used by C2).
            let int_vals: [u32; 4] = if c2_target_oob {
                [rng.next_u32() % INT_MAX_EXCL, 0, 0, 0]
            } else {
                [
                    rng.next_u32() % INT_MAX_EXCL,
                    rng.next_u32() % INT_MAX_EXCL,
                    rng.next_u32() % INT_MAX_EXCL,
                    rng.next_u32() % INT_MAX_EXCL,
                ]
            };
            for (k, v) in int_vals.iter().enumerate() {
                int_cols[k][i] = R::from(*v);
            }

            // (C3)/(C4): bp[8] = BinaryPoly::from(int[0]);
            // bp[9] = BinaryPoly::from(int[1]).
            // Since `BinaryPoly::from(n).evaluate_at_point(2) == n`,
            // this makes `bp[k] - int[k]` vanish at X=2, satisfying the (X-2) ideal check.
            bp_cols[8][i] = BinaryPoly::from(int_vals[0]);
            bp_cols[9][i] = BinaryPoly::from(int_vals[1]);

            // (C5): popcount(bp[10]) == popcount(bp[11]).
            let bp11: BinaryPoly<32> = rng.random();
            let popcount11 = bp11
                .evaluate_at_point(&1_u32)
                .expect("popcount eval should fit in u32");
            bp_cols[11][i] = bp11;
            bp_cols[10][i] = random_binary_poly_with_popcount(popcount11, rng);

            // (C6): popcount(bp[12]) == popcount(bp[13]).
            let bp13: BinaryPoly<32> = rng.random();
            let popcount13 = bp13
                .evaluate_at_point(&1_u32)
                .expect("popcount eval should fit in u32");
            bp_cols[13][i] = bp13;
            bp_cols[12][i] = random_binary_poly_with_popcount(popcount13, rng);

            // Set bp[0][i+1] and bp[4][i+4] so C1 and C2 respectively hold at
            // row i. Each summand fits in u32 (bp eval ≤ 2^28 - 1, ints ≤ 31),
            // so each sum (≤ 3 * (2^28 - 1) + 3 * 31 ≈ 8.05e8) stays well
            // below 2^32. C1 and C2 have different shift amounts now (1 vs 4)
            // and so need separate `if` guards.
            let eval_at_2 = |bp: &BinaryPoly<32>| -> u32 {
                bp.evaluate_at_point(&2_u32)
                    .expect("28-bit binary poly eval at 2 fits in u32")
            };

            // C1 (shift = 1)
            if i + 1 < len {
                let s1 = eval_at_2(&bp_cols[1][i])
                    + eval_at_2(&bp_cols[2][i])
                    + eval_at_2(&bp_cols[3][i])
                    + int_vals[0]
                    + int_vals[1]
                    + int_vals[2];
                bp_cols[0][i + 1] = BinaryPoly::from(s1);
            }

            // C2 (shift = 4)
            if i + 4 < len {
                let s2 = eval_at_2(&bp_cols[5][i])
                    + eval_at_2(&bp_cols[6][i])
                    + eval_at_2(&bp_cols[7][i])
                    + int_vals[1]
                    + int_vals[2]
                    + int_vals[3];
                bp_cols[4][i + 4] = BinaryPoly::from(s2);
            }
        }

        UairTrace {
            binary_poly: bp_cols.into(),
            arbitrary_poly: vec![].into(),
            int: int_cols.into(),
        }
    }
}

/// Test UAIR with mixed shift amounts.
/// 3 columns (a, b, c): column a shifts by 1, column b shifts by 2.
/// Constraints are linear (degree 1).
#[derive(Clone, Debug)]
pub struct TestUairMixedShifts<R>(PhantomData<R>);

impl<R> Uair for TestUairMixedShifts<R>
where
    R: Semiring + 'static,
{
    type Ideal = ImpossibleIdeal;
    type Scalar = DensePolynomial<R, 32>;

    fn signature() -> UairSignature {
        let total = TotalColumnLayout::new(0, 3, 0);
        let shifts = vec![
            ShiftSpec::new(0, 1), // a shifted by 1
            ShiftSpec::new(1, 2), // b shifted by 2
        ];
        UairSignature::new(total, PublicColumnLayout::default(), shifts, vec![], vec![])
    }

    // Constraints:
    //   a[i+1] = a[i] + b[i]  →  down[0] - up[0] - up[1] = 0
    //   c[i]   = b[i+2]       →  up[2] - down[1] = 0
    fn constrain_general<B, FromR, MulByScalar, IFromR>(
        builder: &mut B,
        up: TraceRow<B::Expr>,
        down: TraceRow<B::Expr>,
        _from_ref: FromR,
        _mbs: MulByScalar,
        _ideal_from_ref: IFromR,
    ) where
        B: ConstraintBuilder,
    {
        let up = up.arbitrary_poly;
        let down = down.arbitrary_poly;

        builder.assert_zero(down[0].clone() - &up[0] - &up[1]);
        builder.assert_zero(up[2].clone() - &down[1]);
    }
}

impl<R> GenerateRandomTrace<32> for TestUairMixedShifts<R>
where
    R: FixedSemiring + From<i8> + 'static,
    StandardUniform: Distribution<R>,
{
    type PolyCoeff = R;
    type Int = R;

    // Witness: random b, derive a from a[i+1] = a[i] + b[i], set c[i] = b[i+2].
    fn generate_random_trace<Rng: rand::RngCore + ?Sized>(
        num_vars: usize,
        rng: &mut Rng,
    ) -> UairTrace<'static, R, R, 32> {
        let n = 1 << num_vars;

        // Random b column (degree-0 polynomials to stay under degree 32)
        let b_col: Vec<DynamicPolynomialFS<R>> = (0..n)
            .map(|_| DynamicPolynomialFS::new(vec![R::from(rng.random::<i8>())]))
            .collect();

        // a[0] random, a[i+1] = a[i] + b[i]
        let mut a_col: Vec<DynamicPolynomialFS<R>> =
            vec![DynamicPolynomialFS::new(vec![R::from(rng.random::<i8>())])];
        for i in 0..n - 1 {
            a_col.push(a_col[i].clone() + &b_col[i]);
        }

        // c[i] = b[i+2], zero-padded for last 2 entries
        let mut c_col: Vec<DynamicPolynomialFS<R>> = Vec::with_capacity(n);
        for i in 0..n {
            if i + 2 < n {
                c_col.push(b_col[i + 2].clone());
            } else {
                c_col.push(DynamicPolynomialFS::zero());
            }
        }

        let to_mle = |col: Vec<DynamicPolynomialFS<R>>| -> DenseMultilinearExtension<DensePolynomial<R, 32>> {
            col.into_iter()
                .map(|x| DensePolynomial::new(x.coeffs))
                .collect()
        };

        UairTrace {
            arbitrary_poly: vec![to_mle(a_col), to_mle(b_col), to_mle(c_col)].into(),
            ..Default::default()
        }
    }
}

/// UAIR with mixed-degree non-zero-ideal constraints — the trigger for
/// the hybrid ideal-check dispatch.
///
/// Three `arbitrary_poly` columns `a, b, c`; constraints (all
/// `assert_in_ideal` over `(X − 2)`):
///
/// 1. `a + b - c ∈ (X − 2)`     (linear,    degree 1 in trace MLEs)
/// 2. `a · b - c ∈ (X − 2)`     (non-linear, degree 2)
/// 3. `2·a - b ∈ (X − 2)`       (linear,    degree 1)
///
/// Witness: `a, b` random degree-0 polynomials; `c = a + b = a · b` is
/// satisfiable simultaneously when `a + b = a · b` — but checking the
/// constraint *as polynomial = 0* is enough since `0 ∈ (X − 2)` trivially,
/// so we set `c = a + b` and pick `a, b` such that `a·b = a+b`. To keep
/// the witness generation simple we set `a = b = 2`, giving
/// `a + b = 4 = 2 · 2 = a · b` — both constraints pass with `c = 4`. The
/// third constraint becomes `2·2 − 2 = 2 ∈ (X − 2)` (vacuously, since
/// `2(2) − 2 = 2` is divisible by `(X − 2)` only when evaluated at X=2 it
/// is zero — `2 - 2·2 = -2 ≠ 0`). So pick a different scheme: just verify
/// the whole expression evaluates to `0` (which is trivially in any
/// ideal). Easiest: `a = 0, b = 0, c = 0`.
#[derive(Clone, Debug)]
pub struct TestUairMixedDegrees<R>(PhantomData<R>);

impl<R> Uair for TestUairMixedDegrees<R>
where
    R: ConstSemiring + 'static,
{
    type Ideal = DegreeOneIdeal<R>;
    type Scalar = DensePolynomial<R, 32>;

    fn signature() -> UairSignature {
        let total = TotalColumnLayout::new(0, 3, 0); // a, b, c
        UairSignature::new(total, PublicColumnLayout::default(), vec![], vec![], vec![])
    }

    fn constrain_general<B, FromR, MulByScalar, IFromR>(
        b: &mut B,
        up: TraceRow<B::Expr>,
        _down: TraceRow<B::Expr>,
        _from_ref: FromR,
        _mbs: MulByScalar,
        ideal_from_ref: IFromR,
    ) where
        B: ConstraintBuilder,
        IFromR: Fn(&Self::Ideal) -> B::Ideal,
    {
        let up = up.arbitrary_poly;
        // Use `R::ONE + R::ONE` for the constant 2 so we don't require
        // `From<i32>` on R.
        let two = R::ONE + R::ONE;
        let two_ideal = ideal_from_ref(&DegreeOneIdeal::new(two));

        // Linear: a + b - c ∈ (X-2)  (degree 1 in trace MLEs).
        b.assert_in_ideal(up[0].clone() + &up[1] - &up[2], &two_ideal);

        // Non-linear: a * b - c ∈ (X-2)  (degree 2 in trace MLEs).
        b.assert_in_ideal(up[0].clone() * &up[1] - &up[2], &two_ideal);
    }
}

impl<R> GenerateRandomTrace<32> for TestUairMixedDegrees<R>
where
    R: ConstSemiring + 'static,
{
    type PolyCoeff = R;
    type Int = R;

    /// Honest witness: `a = b = 0`, so `c = a + b = 0` and
    /// `a · b - c = 0 ∈ any ideal`. Trivial but exercises both lanes
    /// cleanly with no soundness risk in the test fixture.
    fn generate_random_trace<Rng: rand::RngCore + ?Sized>(
        num_vars: usize,
        _rng: &mut Rng,
    ) -> UairTrace<'static, R, R, 32> {
        let n = 1 << num_vars;
        let zero_poly = || -> DensePolynomial<R, 32> {
            let coeffs: [R; 32] = core::array::from_fn(|_| R::ZERO);
            DensePolynomial::new(coeffs)
        };
        let zero_col = || -> DenseMultilinearExtension<DensePolynomial<R, 32>> {
            (0..n).map(|_| zero_poly()).collect()
        };
        UairTrace {
            arbitrary_poly: vec![zero_col(), zero_col(), zero_col()].into(),
            ..Default::default()
        }
    }
}

// ---------------------------------------------------------------------------
// BitOpRotUair — synthetic UAIR exercising one BitOp::Rot virtual column.
//
// Two binary witness columns W (col 0) and V (col 1), each cell a 32-bit
// `BinaryPoly`. One bit-op virtual column with `BitOp::Rot(7)` and
// `source_col = 0` (= W). Witness generator sets V[i] = Rot(7)(W[i]) so the
// constraint
//
//   up.binary_poly[1] - down.bit_op[0] == 0   mod (X − 2)
//
// holds row-wise: at X = 2 each side equals the u32-rotation of W[i].
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct BitOpRotUair<R>(PhantomData<R>);

impl<R> Uair for BitOpRotUair<R>
where
    R: ConstSemiring + From<u32> + 'static,
{
    type Ideal = DegreeOneIdeal<R>;
    type Scalar = DensePolynomial<R, 32>;

    fn signature() -> UairSignature {
        let total = TotalColumnLayout::new(2, 0, 0);
        let bit_op_specs = vec![BitOpSpec::new(0, BitOp::rot(7))];
        UairSignature::new(
            total,
            PublicColumnLayout::default(),
            vec![],
            vec![],
            bit_op_specs,
        )
    }

    fn constrain_general<B, FromR, MulByScalar, IFromR>(
        b: &mut B,
        up: TraceRow<B::Expr>,
        down: TraceRow<B::Expr>,
        _from_ref: FromR,
        _mbs: MulByScalar,
        ideal_from_ref: IFromR,
    ) where
        B: ConstraintBuilder,
        IFromR: Fn(&Self::Ideal) -> B::Ideal,
    {
        let two_ideal = ideal_from_ref(&DegreeOneIdeal::new(R::from(2)));
        // V[i] − Rot(7)(W[i]) ≡ 0  mod (X − 2)
        b.assert_in_ideal(
            up.binary_poly[1].clone() - &down.bit_op[0],
            &two_ideal,
        );
    }
}

impl<R> GenerateRandomTrace<32> for BitOpRotUair<R>
where
    R: ConstSemiring + From<u32> + 'static,
{
    type PolyCoeff = R;
    type Int = R;

    fn generate_random_trace<Rng: rand::RngCore + ?Sized>(
        num_vars: usize,
        rng: &mut Rng,
    ) -> UairTrace<'static, R, R, 32> {
        let n = 1usize << num_vars;
        let w_u32: Vec<u32> = (0..n).map(|_| rng.next_u32()).collect();
        let v_u32: Vec<u32> = w_u32.iter().map(|w| w.rotate_left(7)).collect();
        let w_col: DenseMultilinearExtension<BinaryPoly<32>> =
            w_u32.iter().map(|x| BinaryPoly::from(*x)).collect();
        let v_col: DenseMultilinearExtension<BinaryPoly<32>> =
            v_u32.iter().map(|x| BinaryPoly::from(*x)).collect();
        UairTrace {
            binary_poly: vec![w_col, v_col].into(),
            arbitrary_poly: vec![].into(),
            int: vec![].into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use crypto_primitives::crypto_bigint_int::Int;
    use zinc_uair::{
        collect_scalars::collect_scalars,
        constraint_counter::count_constraints,
        degree_counter::{count_constraint_degrees, count_max_degree},
    };

    use super::*;

    const LIMBS: usize = 4;

    #[test]
    fn test_constraint_degrees() {
        fn assert_uair_shape<U: Uair>(expected_degrees: &[usize]) {
            assert_eq!(count_constraints::<U>(), expected_degrees.len());
            assert_eq!(count_constraint_degrees::<U>(), expected_degrees);
            assert_eq!(
                count_max_degree::<U>(),
                *expected_degrees.iter().max().unwrap()
            );
        }

        assert_uair_shape::<TestUairSimpleMultiplication<Int<LIMBS>>>(&[2, 2, 2]);
        assert_uair_shape::<TestUairNoMultiplication<Int<LIMBS>>>(&[1]);
        assert_uair_shape::<TestUairScalarMultiplications<Int<LIMBS>>>(&[1]);
        assert_uair_shape::<BinaryDecompositionUair<u32>>(&[1]);
        assert_uair_shape::<BigLinearUair<u32>>(&[1; 17]);
        assert_uair_shape::<TestUairMixedShifts<Int<LIMBS>>>(&[1, 1]);
        assert_uair_shape::<TestUairMixedDegrees<Int<LIMBS>>>(&[1, 2]);
    }

    #[test]
    fn test_air_scalar_multiplications_correct_collect_scalars() {
        assert_eq!(
            collect_scalars::<TestUairScalarMultiplications<Int<LIMBS>>>(),
            (vec![
                DensePolynomial::new([Int::from_i8(-1), Int::from_i8(0), Int::from_i8(1)]),
                DensePolynomial::new([
                    Int::from_i8(1),
                    Int::from_i8(2),
                    Int::from_i8(3),
                    Int::from_i8(4),
                ])
            ]
            .into_iter()
            .collect())
        );
    }
}

// ---------------------------------------------------------------------------
// PointerHopUair — composed reads (pointer query)
// ---------------------------------------------------------------------------

/// The number of address bits (= trace num_vars) the pointer-hop UAIRs
/// are declared for; `signature()` is static, so the bit-column count
/// must be fixed here and tests must prove at this size.
pub const POINTER_HOP_NUM_VARS: usize = 8;

/// A value-addressed read at fixture scale: int columns
/// `[V, b_0..b_7, R]`. Every row's `R` entry must equal `V` at the cube
/// position the bits spell at that row. The bits ride the booleanity
/// sumcheck (`with_int_witness_bit_cols`); the read itself is declared
/// as a `ComposedReadSpec` and carries the soundness work — the
/// general constraint is trivially satisfied.
/// See `documentation/pointer-query-design.md`.
#[derive(Clone, Debug)]
pub struct PointerHopUair<R>(PhantomData<R>);

fn pointer_hop_signature() -> UairSignature {
    let mu = POINTER_HOP_NUM_VARS;
    let total = TotalColumnLayout::new(0, 0, mu + 2);
    let bit_cols: Vec<usize> = (1..=mu).collect();
    UairSignature::new(total, PublicColumnLayout::default(), vec![], vec![], vec![])
        .with_int_witness_bit_cols(bit_cols.clone())
        .with_composed_reads(vec![ComposedReadSpec {
            value_col: 0,
            bit_cols,
            result_col: mu + 1,
        }])
}

/// The honest pointer-hop trace: random small values, random addresses,
/// `R[x] = V[addr(x)]`, bit columns spelling `addr` low bit first.
/// When `broken` is set, one result entry is bumped off its dereference.
fn pointer_hop_trace<R, Rng>(num_vars: usize, rng: &mut Rng, broken: bool) -> UairTrace<'static, R, R, 32>
where
    R: ConstSemiring + From<i32> + 'static,
    Rng: rand::RngCore + ?Sized,
{
    assert_eq!(
        num_vars, POINTER_HOP_NUM_VARS,
        "PointerHopUair is declared for {POINTER_HOP_NUM_VARS} vars"
    );
    let rows = 1usize << num_vars;
    let values: Vec<i32> = (0..rows).map(|_| (rng.next_u32() % 97) as i32).collect();
    let addrs: Vec<usize> = (0..rows).map(|_| (rng.next_u32() as usize) % rows).collect();
    let mut results: Vec<i32> = addrs.iter().map(|&a| values[a]).collect();
    if broken {
        results[3] += 1;
    }
    let mut cols: Vec<Vec<i32>> = Vec::with_capacity(num_vars + 2);
    cols.push(values);
    for nu in 0..num_vars {
        cols.push(addrs.iter().map(|&a| ((a >> nu) & 1) as i32).collect());
    }
    cols.push(results);
    UairTrace {
        int: cols
            .into_iter()
            .map(|col| col.into_iter().map(R::from).collect())
            .collect::<Vec<_>>()
            .into(),
        ..Default::default()
    }
}

impl<R> Uair for PointerHopUair<R>
where
    R: ConstSemiring + From<i32> + 'static,
{
    type Ideal = ImpossibleIdeal;
    type Scalar = DensePolynomial<R, 32>;

    fn signature() -> UairSignature {
        pointer_hop_signature()
    }

    fn constrain_general<B, FromR, MulByScalar, IFromR>(
        b: &mut B,
        up: TraceRow<B::Expr>,
        _down: TraceRow<B::Expr>,
        _from_ref: FromR,
        _mbs: MulByScalar,
        _ideal_from_ref: IFromR,
    ) where
        B: ConstraintBuilder,
    {
        // Trivially-satisfied constraint (the composed read carries the
        // soundness work; the bits are booleanity's).
        let v = &up.int[0];
        b.assert_zero(v.clone() - v);
    }
}

impl<R> GenerateRandomTrace<32> for PointerHopUair<R>
where
    R: ConstSemiring + From<i32> + 'static,
{
    type PolyCoeff = R;
    type Int = R;

    fn generate_random_trace<Rng: rand::RngCore + ?Sized>(
        num_vars: usize,
        rng: &mut Rng,
    ) -> UairTrace<'static, R, R, 32> {
        pointer_hop_trace::<R, Rng>(num_vars, rng, false)
    }
}

/// PointerHopUair with one forged result entry: `R[3]` is bumped off
/// its dereference. The composed-read check must reject this trace;
/// a verifier that accepts it is not checking the pointer query.
#[derive(Clone, Debug)]
pub struct BrokenPointerHopUair<R>(PhantomData<R>);

impl<R> Uair for BrokenPointerHopUair<R>
where
    R: ConstSemiring + From<i32> + 'static,
{
    type Ideal = ImpossibleIdeal;
    type Scalar = DensePolynomial<R, 32>;

    fn signature() -> UairSignature {
        pointer_hop_signature()
    }

    fn constrain_general<B, FromR, MulByScalar, IFromR>(
        b: &mut B,
        up: TraceRow<B::Expr>,
        _down: TraceRow<B::Expr>,
        _from_ref: FromR,
        _mbs: MulByScalar,
        _ideal_from_ref: IFromR,
    ) where
        B: ConstraintBuilder,
    {
        let v = &up.int[0];
        b.assert_zero(v.clone() - v);
    }
}

impl<R> GenerateRandomTrace<32> for BrokenPointerHopUair<R>
where
    R: ConstSemiring + From<i32> + 'static,
{
    type PolyCoeff = R;
    type Int = R;

    fn generate_random_trace<Rng: rand::RngCore + ?Sized>(
        num_vars: usize,
        rng: &mut Rng,
    ) -> UairTrace<'static, R, R, 32> {
        pointer_hop_trace::<R, Rng>(num_vars, rng, true)
    }
}
