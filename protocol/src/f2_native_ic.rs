//! F_2-native IC prover for `F_2[X]` UAIRs.
//!
//! Replaces the lifted-then-projected IC step
//! ([`zinc_piop::ideal_check`]'s `prove_combined`) with a path that
//! keeps trace cells as bit-polynomials throughout the per-row
//! constraint evaluation and only materialises GF(2^192) values at
//! the very end (combined-poly MLE evaluation at the IC point).
//!
//! ## Why this exists
//!
//! `prove_combined`'s row-by-row evaluation works with
//! `DynamicPolynomialF<F>`-valued cells. For an `F_2[X]`-typed trace
//! (cells are `BinaryPoly<D>`, ~16 set bits average), every cell gets
//! lifted to a 32-coefficient polynomial of GF(2^192) elements (~24
//! bytes each) — ~768 bytes per cell, mostly zeros. Each constraint
//! expression's `mbs(cell, &scalar)` then performs a full dense
//! schoolbook over GF(2^192). For the SHA-256 F_2 UAIR (~21K cells,
//! ~33 expression terms per row), this is the largest single line
//! item in the prover IC step.
//!
//! The F_2-native path keeps everything in `F_2[X]` (1 u64 word per
//! cell — `BinaryF2Poly<1>` is sufficient for the SHA F_2 shape since
//! cell × scalar reaches at most degree 31 + 30 = 61 < 64). Per-row
//! constraint evaluation is bit-poly arithmetic — XORs and carryless
//! shifts of u64s.
//!
//! ## Scope
//!
//! Specialised for UAIRs that:
//!   - have all-`F_2` (i.e., bit-poly) trace cells,
//!   - have effective max constraint degree ≤ 1 in trace MLEs,
//!   - use scalars whose every coefficient is `0` or `1` in F_2,
//!   - are width-bounded so that `cell × scalar` fits in
//!     `BinaryF2Poly<1>` (64 bits).
//!
//! The SHA-256 F_2 UAIR satisfies all four. For UAIRs that don't, the
//! caller should fall back to `prove_combined` on the lifted trace.
//!
//! ## Algorithm
//!
//! 1. Sample the IC's evaluation point `r ∈ F^num_vars` from the
//!    transcript and build the eq-table over r.
//!
//! 2. For each row, evaluate every constraint expression in `F_2[X]`
//!    arithmetic. Each constraint's per-row value is a 64-bit
//!    bit-pattern (`BinaryF2Poly<1>` packed as a `u64`).
//!
//! 3. For each constraint and each bit-position j ∈ [0, 64), build
//!    the per-row bit-MLE: `coeff_mle[i] = (poly_value[i] >> j) & 1`.
//!    Then evaluate at `r` via the eq-table:
//!    `combined[j] = Σ_{i: coeff_mle[i] = 1} eq(i, r) ∈ F`.
//!    The "lift" from F_2 to F happens here — each `eq` term is a
//!    plain field element, no DynamicPolynomial allocation.
//!
//! 4. Pack the per-constraint coefficient evaluations into
//!    `DynamicPolynomialF<F>` (the existing IC `Proof` shape) and
//!    absorb into the transcript.

use core::marker::PhantomData;
use std::fmt::{self, Debug, Display};
use std::hash::Hash;
use std::iter::{Product, Sum};
use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

use crypto_primitives::Semiring;
use num_traits::{CheckedAdd, CheckedMul, CheckedSub, One, Zero};
use zinc_piop::ideal_check::{Proof as IcProof, ProverState as IcProverState};
use zinc_poly::univariate::dynamic::over_field::DynamicPolynomialF;
use zinc_transcript::traits::{ConstTranscribable, Transcript};
use zinc_uair::{
    ConstraintBuilder, Uair,
    ideal::{Ideal, IdealCheck, IdealCheckError},
};
use zinc_utils::from_ref::FromRef;
use zinc_utils::inner_transparent_field::InnerTransparentField;

#[cfg(feature = "parallel")]
use rayon::prelude::*;
use zinc_utils::cfg_into_iter;

// ---------------------------------------------------------------------------
// F2RowExpr: 64-bit bit-polynomial wrapper used as the IC builder's
// `Expr` type during F_2-native row evaluation.
//
// We carry 64 bits because the SHA F_2 UAIR's tightest constraint
// (cell × ρ_σ0) has degree 31 + 30 = 61 < 64. UAIRs with wider scalars
// would need a larger storage; left as a TODO when the need arises.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Default, Hash, PartialEq, Eq)]
pub struct F2RowExpr {
    pub bits: u64,
}

impl F2RowExpr {
    #[inline(always)]
    pub const fn from_bits(bits: u64) -> Self {
        Self { bits }
    }

    #[inline(always)]
    pub const fn bits(self) -> u64 {
        self.bits
    }

    /// `F_2[X]`-multiplication truncated to 64 bits (= `mod X^64`).
    /// Caller is responsible for ensuring the product never exceeds
    /// degree 63 — for the SHA F_2 UAIR this is enforced by the
    /// scalar-degree analysis (`cell × ρ_*` ≤ degree 61).
    #[inline]
    #[allow(clippy::arithmetic_side_effects)]
    pub fn mul_truncated(self, rhs: Self) -> Self {
        let mut acc: u64 = 0;
        let mut a = self.bits;
        while a != 0 {
            let lo = a.trailing_zeros();
            // `wrapping_shl` matches the truncation semantics: bits
            // shifted past position 63 are discarded.
            acc ^= rhs.bits.wrapping_shl(lo);
            a &= a - 1;
        }
        Self { bits: acc }
    }
}

impl Display for F2RowExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "F2RowExpr(0x{:016x})", self.bits)
    }
}

impl From<bool> for F2RowExpr {
    fn from(b: bool) -> Self {
        Self { bits: b as u64 }
    }
}

impl Zero for F2RowExpr {
    fn zero() -> Self {
        Self { bits: 0 }
    }
    fn is_zero(&self) -> bool {
        self.bits == 0
    }
}

impl One for F2RowExpr {
    fn one() -> Self {
        Self { bits: 1 }
    }
}

macro_rules! impl_f2_add {
    ($trait:ident, $method:ident) => {
        impl $trait for F2RowExpr {
            type Output = Self;
            #[inline(always)]
            fn $method(self, rhs: Self) -> Self::Output {
                Self {
                    bits: self.bits ^ rhs.bits,
                }
            }
        }
        impl<'a> $trait<&'a Self> for F2RowExpr {
            type Output = Self;
            #[inline(always)]
            fn $method(self, rhs: &'a Self) -> Self::Output {
                Self {
                    bits: self.bits ^ rhs.bits,
                }
            }
        }
    };
}
impl_f2_add!(Add, add);
impl_f2_add!(Sub, sub); // char-2: a − b = a + b

macro_rules! impl_f2_add_assign {
    ($trait:ident, $method:ident) => {
        impl $trait for F2RowExpr {
            #[inline(always)]
            fn $method(&mut self, rhs: Self) {
                self.bits ^= rhs.bits;
            }
        }
        impl<'a> $trait<&'a Self> for F2RowExpr {
            #[inline(always)]
            fn $method(&mut self, rhs: &'a Self) {
                self.bits ^= rhs.bits;
            }
        }
    };
}
impl_f2_add_assign!(AddAssign, add_assign);
impl_f2_add_assign!(SubAssign, sub_assign);

impl Mul for F2RowExpr {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        self.mul_truncated(rhs)
    }
}
impl<'a> Mul<&'a Self> for F2RowExpr {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: &'a Self) -> Self::Output {
        self.mul_truncated(*rhs)
    }
}
impl MulAssign for F2RowExpr {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.mul_truncated(rhs);
    }
}
impl<'a> MulAssign<&'a Self> for F2RowExpr {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: &'a Self) {
        *self = self.mul_truncated(*rhs);
    }
}

impl CheckedAdd for F2RowExpr {
    fn checked_add(&self, other: &Self) -> Option<Self> {
        Some(Self {
            bits: self.bits ^ other.bits,
        })
    }
}
impl CheckedSub for F2RowExpr {
    fn checked_sub(&self, other: &Self) -> Option<Self> {
        Some(Self {
            bits: self.bits ^ other.bits,
        })
    }
}
impl CheckedMul for F2RowExpr {
    fn checked_mul(&self, other: &Self) -> Option<Self> {
        Some(self.mul_truncated(*other))
    }
}

impl Sum for F2RowExpr {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |a, b| Self {
            bits: a.bits ^ b.bits,
        })
    }
}
impl<'a> Sum<&'a F2RowExpr> for F2RowExpr {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |a, b| Self {
            bits: a.bits ^ b.bits,
        })
    }
}
impl Product for F2RowExpr {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |a, b| a.mul_truncated(b))
    }
}
impl<'a> Product<&'a F2RowExpr> for F2RowExpr {
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |a, b| a.mul_truncated(*b))
    }
}

impl Semiring for F2RowExpr {}

impl FromRef<F2RowExpr> for F2RowExpr {
    fn from_ref(v: &F2RowExpr) -> Self {
        *v
    }
}

// ---------------------------------------------------------------------------
// F_2-native dummy ideal: the IC's row-evaluator never invokes
// `IdealCheck::contains` (it only collects per-constraint values; the
// real check happens at verification on the GF(2^192)-projected
// combined polynomial). We supply a trivial `IdealCheck` impl.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct F2RowIdeal;

impl Ideal for F2RowIdeal {}

impl FromRef<F2RowIdeal> for F2RowIdeal {
    fn from_ref(_: &F2RowIdeal) -> Self {
        F2RowIdeal
    }
}

impl IdealCheck<F2RowExpr> for F2RowIdeal {
    fn contains(&self, _value: &F2RowExpr) -> Result<bool, IdealCheckError> {
        // Row-evaluator never invokes the ideal check — the real
        // membership test happens at verification, on the
        // GF(2^192)-projected combined polynomial.
        Ok(true)
    }
}

// ---------------------------------------------------------------------------
// F2NativeRowBuilder: ConstraintBuilder that collects per-constraint
// `F2RowExpr` values for a single row.
// ---------------------------------------------------------------------------

pub struct F2NativeRowBuilder {
    pub values: Vec<F2RowExpr>,
}

impl F2NativeRowBuilder {
    pub fn new(num_constraints: usize) -> Self {
        Self {
            values: Vec::with_capacity(num_constraints),
        }
    }
}

impl ConstraintBuilder for F2NativeRowBuilder {
    type Expr = F2RowExpr;
    type Ideal = F2RowIdeal;

    fn assert_in_ideal(&mut self, expr: Self::Expr, _ideal: &Self::Ideal) {
        self.values.push(expr);
    }

    fn assert_zero(&mut self, expr: Self::Expr) {
        self.values.push(expr);
    }
}

// ---------------------------------------------------------------------------
// F_2-native IC prover.
//
// The signature mirrors `IdealCheckProtocol::prove_combined` so the
// caller can swap it in transparently; the returned `Proof` /
// `ProverState` are the same shapes the existing verifier consumes.
// ---------------------------------------------------------------------------

/// Public marker that the F_2-native path can produce the exact
/// `IcProof<F>` / `IcProverState<F>` shapes the existing IC verifier
/// consumes. The phantom carries the type parameters without
/// allocating storage.
pub struct F2NativeIc<U>(PhantomData<U>);

impl<U: Uair> F2NativeIc<U>
where
    U::Ideal: Clone,
{
    /// Run the IC's prover side with F_2-native row evaluation.
    ///
    /// `binary_trace`: the F_2 trace's `binary_poly` columns (no
    /// projection). Each cell is a `BinaryPoly<D>` (with `D ≤ 64`);
    /// the SHA F_2 UAIR uses `D = 32`.
    ///
    /// `project_scalar_to_bits`: closure mapping each `U::Scalar`
    /// to its 64-bit `F_2[X]` representation (bit `i` set iff the
    /// scalar's coefficient at position `i` is non-zero). For SHA-F_2
    /// where scalars are `DensePolynomial<R, 32>` with R-integer
    /// coefficients only ever 0 or 1, this is a trivial bit-pack.
    ///
    /// The closure is `Sync` so the per-row evaluation can run in
    /// parallel across rows.
    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    pub fn prove_combined<F, ProjectScalar, const D: usize>(
        transcript: &mut impl Transcript,
        binary_trace: &[zinc_poly::mle::DenseMultilinearExtension<
            zinc_poly::univariate::binary::BinaryPoly<D>,
        >],
        num_constraints: usize,
        num_vars: usize,
        field_cfg: &F::Config,
        project_scalar_to_bits: ProjectScalar,
    ) -> (IcProof<F>, IcProverState<F>)
    where
        F: InnerTransparentField + Send + Sync,
        F::Inner: ConstTranscribable,
        F::Modulus: ConstTranscribable,
        ProjectScalar: Fn(&U::Scalar) -> u64 + Sync,
    {
        const W_BITS: usize = 64;

        let num_cols = binary_trace.len();
        let num_rows = if num_cols > 0 {
            binary_trace[0].evaluations.len()
        } else {
            0
        };
        debug_assert!(D <= W_BITS, "F2NativeIc::prove_combined: D ({D}) must fit in 64 bits");

        // Sample the IC's evaluation point.
        let evaluation_point: Vec<F> = transcript.get_field_challenges(num_vars, field_cfg);
        let eq_table: Vec<F> =
            zinc_poly::utils::build_eq_x_r_vec(&evaluation_point, field_cfg)
                .expect("eq-table build succeeds for num_vars > 0");

        // ---- Step 1: per-row, per-constraint F_2[X] evaluation ----
        //
        // For each row, pack each cell into a `u64` bit-pattern,
        // build the row's `F2RowExpr` values, then run `eval_row`
        // (which invokes `U::constrain_general` with our F_2-native
        // builder).
        //
        // Output `per_row_constraints[row][constraint] = F2RowExpr`.
        let sig = U::signature();
        let shifts = sig.shifts();
        let total_cols_layout = sig.total_cols().as_column_layout().clone();
        let down_cols_layout = sig.down_cols().as_column_layout().clone();
        let bit_op_count = sig.bit_op_down_count();

        // Build the `eval_row` closure inline. We need:
        //   - `from_ref` that maps a scalar reference to F2RowExpr
        //     (used by the UAIR when adding constants); since we
        //     never add scalars directly (only via `mbs`), we panic
        //     here if invoked.
        //   - `mbs` closure: F2RowExpr × Scalar → Option<F2RowExpr>.
        //   - `ideal_from_ref` closure: U::Ideal → F2RowIdeal (the
        //     dummy ideal — the row-evaluator never checks).
        let from_ref_scalar = |_s: &U::Scalar| -> F2RowExpr {
            // F_2-native row eval only encounters scalars via `mbs`.
            // If a UAIR uses `from_ref` directly (e.g. constants
            // added without going through mbs), this needs to lift
            // the scalar to its F_2[X] bit-pattern.
            panic!(
                "F2NativeIc: `from_ref` invoked — UAIR uses raw scalars; \
                 widen the projection or feed scalars via `mbs`."
            );
        };
        let mbs_op = |expr: &F2RowExpr, scalar: &U::Scalar| -> Option<F2RowExpr> {
            Some(expr.mul_truncated(F2RowExpr::from_bits(project_scalar_to_bits(
                scalar,
            ))))
        };
        let ideal_from_ref = |_i: &U::Ideal| -> F2RowIdeal { F2RowIdeal };

        let per_row_constraints: Vec<Vec<F2RowExpr>> = cfg_into_iter!(0..num_rows)
            .map(|row_idx| {
                // Last row gets zeroed (matches the existing
                // `compute_combined_polynomials` convention).
                if row_idx + 1 == num_rows {
                    return vec![F2RowExpr::zero(); num_constraints];
                }
                let up_vals: Vec<F2RowExpr> = (0..num_cols)
                    .map(|col_idx| {
                        let cell = &binary_trace[col_idx].evaluations[row_idx];
                        F2RowExpr::from_bits(bp_to_u64::<D>(cell))
                    })
                    .collect();
                let mut down_vals: Vec<F2RowExpr> = shifts
                    .iter()
                    .map(|spec| {
                        let r = row_idx + spec.shift_amount();
                        if r < num_rows {
                            F2RowExpr::from_bits(bp_to_u64::<D>(
                                &binary_trace[spec.source_col()].evaluations[r],
                            ))
                        } else {
                            F2RowExpr::zero() // zero-padded past the trace
                        }
                    })
                    .collect();
                // Bit-op virtual columns: not supported in this F_2-
                // native path yet (the SHA F_2 UAIR has zero bit-ops
                // by design — its committed Σ/σ/SHR-aux columns
                // absorb what would otherwise be bit-ops). Pad with
                // zeros to match the down layout.
                for _ in 0..bit_op_count {
                    down_vals.push(F2RowExpr::zero());
                }

                let up_row = zinc_uair::TraceRow::from_slice_with_layout(
                    &up_vals,
                    &total_cols_layout,
                );
                let down_row = zinc_uair::TraceRow::from_slice_with_layout_and_bit_op(
                    &down_vals,
                    &down_cols_layout,
                    bit_op_count,
                );

                let mut builder = F2NativeRowBuilder::new(num_constraints);
                U::constrain_general(
                    &mut builder,
                    up_row,
                    down_row,
                    from_ref_scalar,
                    mbs_op,
                    ideal_from_ref,
                );
                builder.values
            })
            .collect();

        // ---- Step 2: per-(constraint, bit-position) MLE eval ----
        //
        // For each constraint `c`, the combined polynomial is the
        // MLE-aggregate over rows of the per-row `F_2[X]` values.
        // Bit-position `j`'s contribution to the combined poly is
        // `Σ_{i: per_row_constraints[i][c].bits & (1<<j) != 0} eq[i]`.
        //
        // The "lift" from F_2 to F = GF(2^192) happens here: each
        // contributing row's `eq[i]` is just XOR-summed into the
        // GF(2^192) accumulator (since the bit-poly coefficient is
        // 1, multiplication by `eq[i]` is identity). No
        // `DynamicPolynomialF` arithmetic, no per-cell allocation.
        let combined_mle_values: Vec<DynamicPolynomialF<F>> =
            cfg_into_iter!(0..num_constraints)
                .map(|c| {
                    // For each of the 64 bit-positions, accumulate
                    // the per-row eq-contributions.
                    let mut coeffs: Vec<F> = vec![F::zero_with_cfg(field_cfg); W_BITS];
                    for (row_idx, row_vals) in per_row_constraints.iter().enumerate() {
                        let mut bits = row_vals[c].bits;
                        while bits != 0 {
                            let j = bits.trailing_zeros() as usize;
                            #[allow(clippy::arithmetic_side_effects)]
                            {
                                bits &= bits - 1;
                            }
                            coeffs[j] += &eq_table[row_idx];
                        }
                    }
                    DynamicPolynomialF::new_trimmed(coeffs)
                })
                .collect();

        // Absorb into the transcript (matches `prove_combined`).
        let mut transcription_buf: Vec<u8> = vec![0; F::Inner::NUM_BYTES];
        combined_mle_values.iter().for_each(|v| {
            transcript.absorb_random_field_slice(&v.coeffs, &mut transcription_buf);
        });

        (
            IcProof {
                combined_mle_values,
            },
            IcProverState { evaluation_point },
        )
    }
}

/// Pack a `BinaryPoly<D>` into a u64 with bit `i` set iff coefficient
/// `i` of the bit-poly is `1`. Requires `D ≤ 64`.
#[inline]
fn bp_to_u64<const D: usize>(
    p: &zinc_poly::univariate::binary::BinaryPoly<D>,
) -> u64 {
    debug_assert!(D <= 64);
    let mut v: u64 = 0;
    for (i, c) in p.iter().enumerate() {
        if c.inner() {
            #[allow(clippy::arithmetic_side_effects)]
            {
                v |= 1u64 << i;
            }
        }
    }
    v
}
