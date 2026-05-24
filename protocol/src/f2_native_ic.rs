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

use crypto_primitives::{PrimeField, Semiring};
use num_traits::{CheckedAdd, CheckedMul, CheckedSub, ConstZero, One, Zero};
use zinc_piop::ideal_check::{Proof as IcProof, ProverState as IcProverState};
use zinc_poly::univariate::dynamic::over_field::DynamicPolynomialF;
use zinc_transcript::traits::{ConstTranscribable, Transcript};
use zinc_uair::{
    ConstraintBuilder, Uair,
    ideal::{Ideal, IdealCheck, IdealCheckError, ImpossibleIdeal},
};
use zinc_utils::from_ref::FromRef;
use zinc_utils::inner_transparent_field::InnerTransparentField;

#[cfg(feature = "parallel")]
use rayon::prelude::*;
use zinc_utils::{cfg_into_iter, cfg_iter};

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
    ///
    /// Uses hardware PMULL (aarch64) / PCLMUL (x86_64) for one 64×64
    /// carryless multiply, then keeps the low 64 bits. The previous
    /// scalar bit-by-bit loop (`popcount(self.bits)` iterations of
    /// shift+XOR) dominated the F_2-native IC's per-row constraint
    /// evaluation at SHA scale; the hardware path drops it to ~1
    /// instruction per call.
    #[inline]
    pub fn mul_truncated(self, rhs: Self) -> Self {
        let prod = clmul_64x64_truncated(self.bits, rhs.bits);
        Self { bits: prod }
    }
}

/// 64×64 → 64-bit carryless multiplication (low half only).
///
/// Mirror of the `clmul_64x64` helpers in `binary_gf128` / `binary_gf192`
/// but specialised to the low-64-bit output — the high half is dropped
/// because `mul_truncated` is `mod X^64` by contract.
#[inline]
fn clmul_64x64_truncated(a: u64, b: u64) -> u64 {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        unsafe {
            use core::arch::aarch64::vmull_p64;
            let prod: u128 = core::mem::transmute(vmull_p64(a, b));
            prod as u64
        }
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
    {
        unsafe {
            use core::arch::x86_64::{__m128i, _mm_clmulepi64_si128, _mm_set_epi64x};
            let av = _mm_set_epi64x(0, a as i64);
            let bv = _mm_set_epi64x(0, b as i64);
            let prod: __m128i = _mm_clmulepi64_si128(av, bv, 0x00);
            let bytes: [u64; 2] = core::mem::transmute(prod);
            bytes[0]
        }
    }
    #[cfg(not(any(
        all(target_arch = "aarch64", target_feature = "neon"),
        all(target_arch = "x86_64", target_feature = "pclmulqdq"),
    )))]
    {
        // Scalar fallback (kept identical to the original loop so the
        // soft-CPU path stays bit-for-bit compatible). Iterates set
        // bits of `a` and XORs `b << shift` into the truncated
        // accumulator.
        let mut acc: u64 = 0;
        let mut a_word = a;
        #[allow(clippy::arithmetic_side_effects)]
        while a_word != 0 {
            let lo = a_word.trailing_zeros();
            acc ^= b.wrapping_shl(lo);
            a_word &= a_word - 1;
        }
        acc
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

    /// MLE-first IC variant for UAIRs whose effective max degree is 1
    /// (i.e. every non-zero-ideal constraint is linear in the trace
    /// MLEs). Mirrors the integer protocol's
    /// `IdealCheckProtocol::prove_linear` but keeps the per-cell side
    /// in F_2-bit-packed form — no per-row constraint evaluation, no
    /// `F_2[X]<64>` arithmetic per row.
    ///
    /// ## Why this is faster than `prove_combined`
    ///
    /// `prove_combined` runs `U::constrain_general` once per row over
    /// the whole hypercube — `2^num_vars` invocations × O(num_constraints)
    /// per row. For SHA-256 F_2 at `num_vars = 16` that's ~3.3M
    /// constraint-expression evaluations, dominating the IC time.
    ///
    /// For UAIRs with linear constraints,
    ///   `Σ_{i} eq[i] · expr(row_i_cells) = expr(Σ_i eq[i] · row_i_cells)`
    /// because `expr` is linear in the trace MLEs. So we can build
    /// per-column MLE-evaluated polynomials once (one
    /// `DynamicPolynomialF<F>` per column, D coefficients each) and
    /// call `U::constrain_general` exactly ONCE on those column-level
    /// values. The 3.3M expression evaluations collapse to ~50.
    ///
    /// Per-column MLE-eval cost: for each cell, the d-th coefficient
    /// is `Σ_{i: bit d of cell_i is set} eq[i]`. Bit-pack u64 input →
    /// XOR-only accumulation of GF(2^192) elements into the
    /// coefficient slots. The d-th bit's contributions are summed
    /// into `coeffs[d]` — same shape as the existing Step 2 in
    /// `prove_combined`, but driven by trace bits rather than
    /// per-row constraint output bits.
    ///
    /// ## Pre-conditions
    ///
    /// Caller is responsible for ensuring that
    /// `count_effective_max_degree::<U>() == 1` (i.e. all non-zero-ideal
    /// constraints are degree-1 in trace MLEs); routing a higher-degree
    /// UAIR through this path produces meaningless combined polynomials.
    /// The dispatch in `prove_f2_uair_with_groups` enforces that gate.
    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    pub fn prove_linear<F, ProjectScalar, const D: usize>(
        transcript: &mut impl Transcript,
        binary_trace: &[zinc_poly::mle::DenseMultilinearExtension<
            zinc_poly::univariate::binary::BinaryPoly<D>,
        >],
        num_constraints: usize,
        num_vars: usize,
        field_cfg: &F::Config,
        project_scalar: ProjectScalar,
    ) -> (IcProof<F>, IcProverState<F>)
    where
        F: InnerTransparentField + Send + Sync,
        F::Inner: ConstTranscribable + Send + Sync,
        F::Modulus: ConstTranscribable,
        ProjectScalar: Fn(&U::Scalar, &F::Config) -> DynamicPolynomialF<F> + Sync,
    {
        const W_BITS: usize = 64;
        debug_assert!(
            D <= W_BITS,
            "F2NativeIc::prove_linear: D ({D}) must fit in 64 bits",
        );

        let num_cols = binary_trace.len();
        let num_rows = if num_cols > 0 {
            binary_trace[0].evaluations.len()
        } else {
            0
        };

        // Sample IC eval point.
        let evaluation_point: Vec<F> = transcript.get_field_challenges(num_vars, field_cfg);
        let eq_table: Vec<F> =
            zinc_poly::utils::build_eq_x_r_vec(&evaluation_point, field_cfg)
                .expect("eq-table build succeeds for num_vars > 0");

        let sig = U::signature();
        let total_cols_layout = sig.total_cols().as_column_layout().clone();
        let down_cols_layout = sig.down_cols().as_column_layout().clone();
        let bit_op_count = sig.bit_op_down_count();

        // Number of rows that contribute to the MLE; the last row is
        // explicitly zero-padded to match `prove_combined` (and the
        // integer protocol's `evaluate_combined_polynomials`).
        let usable_rows = num_rows.saturating_sub(1);

        // ---- Per-column MLE-evals (up_evals[g] ∈ F[X]<D>) ----
        //
        // coeffs[d] = Σ_{i in [0, num_rows-1)} (bit d of cell[g, i]) · eq[i].
        // Parallelise across columns; per column we walk usable_rows
        // cells and XOR-add `eq[row]` into `coeffs[trailing_zeros(bits)]`
        // for each set bit.
        let up_evals: Vec<DynamicPolynomialF<F>> = cfg_iter!(binary_trace)
            .map(|col| {
                let mut coeffs: Vec<F> = (0..D)
                    .map(|_| F::zero_with_cfg(field_cfg))
                    .collect();
                for row in 0..usable_rows {
                    let cell = &col.evaluations[row];
                    let mut bits = bp_to_u64::<D>(cell);
                    while bits != 0 {
                        let d = bits.trailing_zeros() as usize;
                        #[allow(clippy::arithmetic_side_effects)]
                        {
                            bits &= bits - 1;
                        }
                        coeffs[d] += &eq_table[row];
                    }
                }
                DynamicPolynomialF::new_trimmed(coeffs)
            })
            .collect();

        // ---- Shifted columns (down_evals) ----
        //
        // `down_col[i] = source_col[i + s]` for `i in [0, num_rows-1)`
        // AND `i + s < num_rows`, else zero (mirrors the integer
        // protocol's `eval_coeff_mle` zero-pad). Reindex by `j = i + s`:
        //   coeffs[d] = Σ_{j in [s, max_j)} (bit d of source[j]) · eq[j - s]
        // where `max_j = min(num_rows, s + num_rows - 1)`.
        let shifts = sig.shifts();
        let mut down_evals: Vec<DynamicPolynomialF<F>> = cfg_iter!(shifts)
            .map(|spec| {
                let source = &binary_trace[spec.source_col()];
                let s = spec.shift_amount();
                let mut coeffs: Vec<F> = (0..D)
                    .map(|_| F::zero_with_cfg(field_cfg))
                    .collect();
                let max_j = std::cmp::min(
                    num_rows,
                    s.saturating_add(usable_rows),
                );
                for j in s..max_j {
                    let cell = &source.evaluations[j];
                    let mut bits = bp_to_u64::<D>(cell);
                    while bits != 0 {
                        let d = bits.trailing_zeros() as usize;
                        #[allow(clippy::arithmetic_side_effects)]
                        {
                            bits &= bits - 1;
                        }
                        #[allow(clippy::arithmetic_side_effects)]
                        let i = j - s;
                        coeffs[d] += &eq_table[i];
                    }
                }
                DynamicPolynomialF::new_trimmed(coeffs)
            })
            .collect();

        // Bit-op virtual columns: not supported in this F_2-native
        // path yet (the SHA F_2 UAIR has zero by design); pad with
        // zero polynomials so the down-row layout indexes correctly.
        for _ in 0..bit_op_count {
            down_evals.push(DynamicPolynomialF::ZERO);
        }

        // ---- Run constraints ONCE on column-evaluated values ----
        //
        // The constraint expressions are linear in the trace MLEs by
        // pre-condition, so this single invocation produces each
        // constraint's combined polynomial directly.
        let project_local =
            |scalar: &U::Scalar| -> DynamicPolynomialF<F> { project_scalar(scalar, field_cfg) };
        let mbs_op = |expr: &DynamicPolynomialF<F>, scalar: &U::Scalar|
         -> Option<DynamicPolynomialF<F>> {
            let s_projected = project_scalar(scalar, field_cfg);
            Some(expr.clone() * s_projected)
        };
        let ideal_from_ref = |_i: &U::Ideal| -> ImpossibleIdeal { ImpossibleIdeal };

        let mut builder = F2NativeMleFirstBuilder::<F>::new(num_constraints);
        let up_row = zinc_uair::TraceRow::from_slice_with_layout(
            &up_evals,
            &total_cols_layout,
        );
        let down_row = zinc_uair::TraceRow::from_slice_with_layout_and_bit_op(
            &down_evals,
            &down_cols_layout,
            bit_op_count,
        );
        U::constrain_general(
            &mut builder,
            up_row,
            down_row,
            project_local,
            mbs_op,
            ideal_from_ref,
        );

        let mut combined_mle_values = builder.combined_evaluations;
        combined_mle_values.iter_mut().for_each(|v| v.trim());

        // Match the prove_combined / integer prove_linear absorb shape
        // exactly: same trimmed coeff vectors → same transcript state.
        let mut transcription_buf: Vec<u8> = vec![0; F::Inner::NUM_BYTES];
        combined_mle_values.iter().for_each(|v| {
            transcript.absorb_random_field_slice(&v.coeffs, &mut transcription_buf);
        });

        (
            IcProof { combined_mle_values },
            IcProverState { evaluation_point },
        )
    }
}

/// Constraint builder used by [`F2NativeIc::prove_linear`].
///
/// Mirrors `zinc_piop::ideal_check::combined_poly_builder::CombinedPolyRowBuilder`
/// (which is module-private). Both `assert_in_ideal` and `assert_zero`
/// just collect the expression — for an honest prover, zero-ideal
/// constraints will already evaluate to the zero polynomial through
/// `U::constrain_general` and the per-constraint absorption matches.
struct F2NativeMleFirstBuilder<F: PrimeField> {
    combined_evaluations: Vec<DynamicPolynomialF<F>>,
}

impl<F: PrimeField> F2NativeMleFirstBuilder<F> {
    fn new(num_constraints: usize) -> Self {
        Self {
            combined_evaluations: Vec::with_capacity(num_constraints),
        }
    }
}

impl<F: PrimeField> ConstraintBuilder for F2NativeMleFirstBuilder<F> {
    type Expr = DynamicPolynomialF<F>;
    type Ideal = ImpossibleIdeal;

    fn assert_in_ideal(&mut self, expr: Self::Expr, _ideal: &Self::Ideal) {
        self.combined_evaluations.push(expr);
    }

    fn assert_zero(&mut self, expr: Self::Expr) {
        self.combined_evaluations.push(expr);
    }
}

/// Pack a `BinaryPoly<D>` into a u64 with bit `i` set iff coefficient
/// `i` of the bit-poly is `1`. Requires `D ≤ 64`.
///
/// Delegates to [`zinc_poly::univariate::F2PackU64::pack_u64`]. Under
/// the `simd` feature `BinaryPoly = BinaryU64Poly`, where storage IS
/// already a `u64` and `pack_u64` is a single field read. The previous
/// implementation iterated `D` times and rebuilt the same u64 bit by
/// bit — at SHA F_2 scale (`num_rows × num_cols × D ≈ 86 M`
/// iterations) it dominated Step 1 of `F2NativeIc::prove_combined`.
#[inline]
fn bp_to_u64<const D: usize>(
    p: &zinc_poly::univariate::binary::BinaryPoly<D>,
) -> u64 {
    debug_assert!(D <= 64);
    use zinc_poly::univariate::F2PackU64;
    p.pack_u64()
}
