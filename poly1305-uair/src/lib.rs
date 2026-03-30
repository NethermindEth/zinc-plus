//! Poly1305 UAIR+ (Universal Algebraic Intermediate Representation).
//!
//! This crate defines the Poly1305 MAC arithmetization as a UAIR+ with two
//! trace components following the no-F2x approach:
//!
//! - **Bp UAIR**: linking constraints (cyclotomic ideal, degree 1).
//! - **Qx UAIR**: placeholder — full modular-arithmetic constraints require
//!   degree-2 expressions (products of column pairs) which the current UAIR
//!   framework does not support.
//!
//! # Trace layout — one row per Poly1305 iteration
//!
//! Each row encodes one Poly1305 accumulator update:
//!   acc_out = (acc + padded_msg) * r  mod (2^130 - 5)
//!
//! ## Bit-polynomial columns — {0,1}^{<32}[X] (indices 0–18)
//!
//! | Index | Name       | Description                              |
//! |-------|------------|------------------------------------------|
//! | 0–4   | acc[0..4]  | Accumulator input (5 × 32-bit limbs)     |
//! | 5–8   | msg[0..3]  | Message block (4 × 32-bit limbs)         |
//! | 9–13  | r[0..4]    | Clamped r key (5 × 32-bit limbs, public) |
//! | 14–18 | acc_out[0..4] | Accumulator output (5 × 32-bit limbs) |
//!
//! # Constraints
//!
//! ## Bp UAIR (5 linking constraints, cyclotomic ideal)
//!
//! 1–5. `acc_out[i][t-1] = acc[i][t]` for i in 0..5
//!
//! ## Qx UAIR (0 constraints — placeholder)
//!
//! Full Poly1305 verification requires degree-2 constraints (product of
//! two column expressions for the limb-level multiplication), which the
//! current framework does not support.

#![allow(clippy::arithmetic_side_effects)]

pub mod constants;
pub mod witness;

use zinc_poly::univariate::binary::BinaryPoly;
use zinc_poly::univariate::dense::DensePolynomial;
use zinc_uair::{
    ConstraintBuilder, Uair,
    ideal::{Ideal, IdealCheck},
};
use zinc_utils::from_ref::FromRef;

// ─── Column indices ──────────────────────────────────────────────────────────

pub const COL_ACC_0: usize = 0;
pub const COL_ACC_1: usize = 1;
pub const COL_ACC_2: usize = 2;
pub const COL_ACC_3: usize = 3;
pub const COL_ACC_4: usize = 4;

pub const COL_MSG_0: usize = 5;
pub const COL_MSG_1: usize = 6;
pub const COL_MSG_2: usize = 7;
pub const COL_MSG_3: usize = 8;

pub const COL_R_0: usize = 9;
pub const COL_R_1: usize = 10;
pub const COL_R_2: usize = 11;
pub const COL_R_3: usize = 12;
pub const COL_R_4: usize = 13;

pub const COL_ACC_OUT_0: usize = 14;
pub const COL_ACC_OUT_1: usize = 15;
pub const COL_ACC_OUT_2: usize = 16;
pub const COL_ACC_OUT_3: usize = 17;
pub const COL_ACC_OUT_4: usize = 18;

// ─── Counts ─────────────────────────────────────────────────────────────────

/// Number of bit-polynomial columns (19).
pub const NUM_BITPOLY_COLS: usize = 19;

/// Number of integer columns (0 — no carry constraints yet).
pub const NUM_INT_COLS: usize = 0;

/// Total number of trace columns.
pub const NUM_COLS: usize = NUM_BITPOLY_COLS + NUM_INT_COLS;

/// Number of Bp constraints (5 linking).
pub const BP_NUM_CONSTRAINTS: usize = 5;

/// Number of Qx constraints (0 — placeholder).
pub const QX_NUM_CONSTRAINTS: usize = 0;

/// Number of columns for which we run bit-poly lookups.
/// All non-public columns: acc[0..4] + msg[0..3] + acc_out[0..4] = 14.
pub const LOOKUP_COL_COUNT: usize = 14;

// ─── Cyclotomic ideal ─────────────────────────────────────────────────────

/// The cyclotomic ideal (X^32 − 1), used by the Bp UAIR.
#[derive(Clone, Copy, Debug)]
pub struct CyclotomicIdeal;

impl Ideal for CyclotomicIdeal {}

impl FromRef<CyclotomicIdeal> for CyclotomicIdeal {
    #[inline(always)]
    fn from_ref(_: &CyclotomicIdeal) -> Self {
        CyclotomicIdeal
    }
}

impl IdealCheck<BinaryPoly<32>> for CyclotomicIdeal {
    fn contains(&self, value: &BinaryPoly<32>) -> bool {
        value.iter().all(|c| !c.into_inner())
    }
}

// ─── Qx ideal ──────────────────────────────────────────────────────────────

/// Placeholder ideal for the Poly1305 Q[X] UAIR (no constraints).
#[derive(Clone, Copy, Debug)]
pub struct Poly1305QxIdeal;

impl Ideal for Poly1305QxIdeal {}

impl FromRef<Poly1305QxIdeal> for Poly1305QxIdeal {
    #[inline(always)]
    fn from_ref(_: &Poly1305QxIdeal) -> Self {
        Poly1305QxIdeal
    }
}

impl IdealCheck<DensePolynomial<i64, 64>> for Poly1305QxIdeal {
    fn contains(&self, _value: &DensePolynomial<i64, 64>) -> bool {
        true // No constraints — everything passes
    }
}

// ─── Bp UAIR ────────────────────────────────────────────────────────────────

/// Poly1305 Bp UAIR: linking constraints only (cyclotomic ideal, degree 1).
///
/// Emits 5 constraints linking each limb of the accumulator output to the
/// next row's accumulator input: acc_out[i][t-1] = acc[i][t].
pub struct Poly1305UairBp;

// Down-row indices for shifted columns.
const DOWN_BP_ACC_OUT_0: usize = 0;
const DOWN_BP_ACC_OUT_1: usize = 1;
const DOWN_BP_ACC_OUT_2: usize = 2;
const DOWN_BP_ACC_OUT_3: usize = 3;
const DOWN_BP_ACC_OUT_4: usize = 4;

impl Uair for Poly1305UairBp {
    type Ideal = CyclotomicIdeal;
    type Scalar = BinaryPoly<32>;

    fn signature() -> zinc_uair::UairSignature {
        zinc_uair::UairSignature {
            binary_poly_cols: NUM_BITPOLY_COLS,
            arbitrary_poly_cols: 0,
            int_cols: NUM_INT_COLS,
            shifts: vec![
                zinc_uair::ShiftSpec { source_col: COL_ACC_OUT_0, shift_amount: 1 },
                zinc_uair::ShiftSpec { source_col: COL_ACC_OUT_1, shift_amount: 1 },
                zinc_uair::ShiftSpec { source_col: COL_ACC_OUT_2, shift_amount: 1 },
                zinc_uair::ShiftSpec { source_col: COL_ACC_OUT_3, shift_amount: 1 },
                zinc_uair::ShiftSpec { source_col: COL_ACC_OUT_4, shift_amount: 1 },
            ],
            public_columns: vec![COL_R_0, COL_R_1, COL_R_2, COL_R_3, COL_R_4],
        }
    }

    fn constrain_general<B, FromR, MulByScalar, IFromR>(
        b: &mut B,
        up: zinc_uair::TraceRow<B::Expr>,
        down: zinc_uair::TraceRow<B::Expr>,
        _from_ref: FromR,
        _mbs: MulByScalar,
        _ideal_from_ref: IFromR,
    ) where
        B: ConstraintBuilder,
        FromR: Fn(&BinaryPoly<32>) -> B::Expr,
        MulByScalar: Fn(&B::Expr, &BinaryPoly<32>) -> Option<B::Expr>,
        IFromR: Fn(&CyclotomicIdeal) -> B::Ideal,
    {
        let up = up.binary_poly;
        let bp_down = down.binary_poly;

        // ── Linking: acc[i][t] = acc_out[i][t-1] ───────────────────────
        b.assert_zero(bp_down[DOWN_BP_ACC_OUT_0].clone() - &up[COL_ACC_0]);
        b.assert_zero(bp_down[DOWN_BP_ACC_OUT_1].clone() - &up[COL_ACC_1]);
        b.assert_zero(bp_down[DOWN_BP_ACC_OUT_2].clone() - &up[COL_ACC_2]);
        b.assert_zero(bp_down[DOWN_BP_ACC_OUT_3].clone() - &up[COL_ACC_3]);
        b.assert_zero(bp_down[DOWN_BP_ACC_OUT_4].clone() - &up[COL_ACC_4]);
    }
}

// ─── Qx UAIR ────────────────────────────────────────────────────────────────

/// Poly1305 Qx UAIR: placeholder (0 constraints).
///
/// Full Poly1305 modular arithmetic constraints require degree-2
/// expressions which the UAIR framework does not currently support.
pub struct Poly1305UairQx;

impl Uair for Poly1305UairQx {
    type Ideal = Poly1305QxIdeal;
    type Scalar = DensePolynomial<i64, 64>;

    fn signature() -> zinc_uair::UairSignature {
        zinc_uair::UairSignature {
            binary_poly_cols: NUM_BITPOLY_COLS,
            arbitrary_poly_cols: 0,
            int_cols: NUM_INT_COLS,
            shifts: vec![],
            public_columns: vec![COL_R_0, COL_R_1, COL_R_2, COL_R_3, COL_R_4],
        }
    }

    fn constrain_general<B, FromR, MulByScalar, IFromR>(
        _b: &mut B,
        _up: zinc_uair::TraceRow<B::Expr>,
        _down: zinc_uair::TraceRow<B::Expr>,
        _from_ref: FromR,
        _mbs: MulByScalar,
        _ideal_from_ref: IFromR,
    ) where
        B: ConstraintBuilder,
        FromR: Fn(&DensePolynomial<i64, 64>) -> B::Expr,
        MulByScalar: Fn(&B::Expr, &DensePolynomial<i64, 64>) -> Option<B::Expr>,
        IFromR: Fn(&Poly1305QxIdeal) -> B::Ideal,
    {
        // No constraints — placeholder.
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use zinc_uair::{
        constraint_counter::count_constraints,
        degree_counter::count_max_degree,
    };

    #[test]
    fn bp_correct_column_count() {
        let sig = Poly1305UairBp::signature();
        assert_eq!(sig.binary_poly_cols, NUM_BITPOLY_COLS);
        assert_eq!(sig.int_cols, NUM_INT_COLS);
        assert_eq!(sig.total_cols(), NUM_COLS);
    }

    #[test]
    fn bp_correct_constraint_count() {
        assert_eq!(count_constraints::<Poly1305UairBp>(), BP_NUM_CONSTRAINTS);
    }

    #[test]
    fn bp_max_degree_is_one() {
        assert_eq!(count_max_degree::<Poly1305UairBp>(), 1);
    }

    #[test]
    fn qx_correct_column_count() {
        let sig = Poly1305UairQx::signature();
        assert_eq!(sig.binary_poly_cols, NUM_BITPOLY_COLS);
        assert_eq!(sig.int_cols, NUM_INT_COLS);
        assert_eq!(sig.total_cols(), NUM_COLS);
    }

    #[test]
    fn qx_correct_constraint_count() {
        assert_eq!(count_constraints::<Poly1305UairQx>(), QX_NUM_CONSTRAINTS);
    }
}
