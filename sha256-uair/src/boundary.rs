//! Boundary constraints for the SHA-256 no-F₂\[X\] UAIR.
//!
//! Defines [`Sha256UairBoundaryNoF2x`], a separate UAIR that enforces
//! boundary conditions (initial state, final output, message input,
//! and zero-padding) via selector-gated constraints. All constraints
//! are degree 2 (`sel · expr`) and use `assert_zero` only.
//!
//! This keeps the main Bp UAIR at degree 1 (enabling MLE-first IC)
//! by isolating boundary constraints into their own IC/CPR group.
//!
//! # Feature gate
//!
//! Requires both `boundary` and `no-f2x` features.
//!
//! # Boundary columns (appended after the no-f2x base columns)
//!
//! | Offset | Name             | Pattern                              |
//! |--------|------------------|--------------------------------------|
//! | +0     | `sel_init`       | 1 at row 0                           |
//! | +1     | `sel_final_any`  | 1 at last 4 active rows              |
//! | +2     | `sel_msg`        | 1 at rows 0–15                       |
//! | +3     | `sel_zero`       | 1 at rows N−3, N−2, N−1              |
//! | +4     | `out_a`          | Expected â at final rows (a,b,c,d)   |
//! | +5     | `out_e`          | Expected ê at final rows (e,f,g,h)   |
//! | +6     | `msg_expected`   | Expected message word at rows 0–15   |
//!
//! # Constraints (13 total)
//!
//! **B1–B8 (initial state):** `sel_init · (col − H[i])` for all 8
//! state registers at row 0.
//!
//! **B9–B10 (final output):** `sel_final_any · (col − out_col)` for
//! â and ê at the last 4 active rows, matching expected output.
//!
//! **B11 (message input):** `sel_msg · (W_hat − msg_expected)` for
//! the first 16 message words.
//!
//! **B12–B13 (zero-padding):** `sel_zero · col` for â and ê at the
//! last 3 tail rows.

use zinc_poly::univariate::binary::BinaryPoly;
use zinc_uair::{
    ConstraintBuilder, Uair,
    ideal::{Ideal, ImpossibleIdeal},
};
use zinc_utils::from_ref::FromRef;

use crate::{
    COL_A_HAT, COL_E_HAT, COL_W_HAT,
    COL_D_HAT, COL_H_HAT,
    COL_A_TM1, COL_A_TM2, COL_E_TM1, COL_E_TM2,
    constants::H,
};
use crate::no_f2x::NO_F2X_NUM_BITPOLY_COLS;

// ─── Boundary column indices ────────────────────────────────────────────────

/// Boundary selector: 1 at row 0 (initial state).
pub const BDRY_COL_SEL_INIT: usize = NO_F2X_NUM_BITPOLY_COLS;
/// Boundary selector: 1 at the last 4 active rows (final output state).
pub const BDRY_COL_SEL_FINAL_ANY: usize = NO_F2X_NUM_BITPOLY_COLS + 1;
/// Boundary selector: 1 at rows 0–15 (message input).
pub const BDRY_COL_SEL_MSG: usize = NO_F2X_NUM_BITPOLY_COLS + 2;
/// Boundary selector: 1 at rows N−3..N−1 (zero-padding).
pub const BDRY_COL_SEL_ZERO: usize = NO_F2X_NUM_BITPOLY_COLS + 3;
/// Expected â value at final rows: a at last, b at last−1, c at last−2, d at last−3.
pub const BDRY_COL_OUT_A: usize = NO_F2X_NUM_BITPOLY_COLS + 4;
/// Expected ê value at final rows: e at last, f at last−1, g at last−2, h at last−3.
pub const BDRY_COL_OUT_E: usize = NO_F2X_NUM_BITPOLY_COLS + 5;
/// Expected message word at rows 0–15.
pub const BDRY_COL_MSG_EXPECTED: usize = NO_F2X_NUM_BITPOLY_COLS + 6;

/// Number of new boundary columns.
pub const BDRY_NUM_NEW_COLS: usize = 7;

/// Total number of bit-polynomial columns with boundary.
pub const BDRY_NUM_BITPOLY_COLS: usize = NO_F2X_NUM_BITPOLY_COLS + BDRY_NUM_NEW_COLS;

/// Number of boundary constraints.
pub const BDRY_NUM_CONSTRAINTS: usize = 13;

// ─── Boundary UAIR ─────────────────────────────────────────────────────────

/// SHA-256 boundary UAIR for the no-F₂\[X\] variant.
///
/// Enforces 13 boundary constraints (all `assert_zero`, degree 2):
///   B1–B8:  initial state (sel_init × (col − H\[i\]))
///   B9–B10: final output (sel_final_any × (col − out_col))
///   B11:    message input (sel_msg × (W_hat − msg_expected))
///   B12–B13: zero-padding (sel_zero × col)
pub struct Sha256UairBoundaryNoF2x;

impl Uair for Sha256UairBoundaryNoF2x {
    type Ideal = ImpossibleIdeal;
    type Scalar = BinaryPoly<32>;

    fn signature() -> zinc_uair::UairSignature {
        use crate::no_f2x::NO_F2X_NUM_INT_COLS;
        zinc_uair::UairSignature {
            binary_poly_cols: BDRY_NUM_BITPOLY_COLS,
            arbitrary_poly_cols: 0,
            int_cols: NO_F2X_NUM_INT_COLS,
            // Use a single sentinel shift to avoid legacy shift-by-1 mode
            // (which would create 40 unnecessary down columns in IC/CPR).
            // The boundary constraints never reference `down`, so this
            // sentinel column is harmless but keeps `uses_legacy_shifts()`
            // false and `down_total_cols()` == 1 instead of total_cols().
            shifts: vec![zinc_uair::ShiftSpec { source_col: 0, shift_amount: 1 }],
            public_columns: vec![
                BDRY_COL_SEL_INIT,
                BDRY_COL_SEL_FINAL_ANY,
                BDRY_COL_SEL_MSG,
                BDRY_COL_SEL_ZERO,
                BDRY_COL_OUT_A,
                BDRY_COL_OUT_E,
                BDRY_COL_MSG_EXPECTED,
            ],
        }
    }

    fn constrain_general<B, FromR, MulByScalar, IFromR>(
        b: &mut B,
        up: zinc_uair::TraceRow<B::Expr>,
        _down: zinc_uair::TraceRow<B::Expr>,
        from_ref: FromR,
        _mbs: MulByScalar,
        _ideal_from_ref: IFromR,
    ) where
        B: ConstraintBuilder,
        FromR: Fn(&BinaryPoly<32>) -> B::Expr,
        MulByScalar: Fn(&B::Expr, &BinaryPoly<32>) -> Option<B::Expr>,
        IFromR: Fn(&ImpossibleIdeal) -> B::Ideal,
    {
        let bp = up.binary_poly;

        let sel_init = &bp[BDRY_COL_SEL_INIT];
        let sel_final = &bp[BDRY_COL_SEL_FINAL_ANY];
        let sel_msg = &bp[BDRY_COL_SEL_MSG];
        let sel_zero = &bp[BDRY_COL_SEL_ZERO];
        let out_a = &bp[BDRY_COL_OUT_A];
        let out_e = &bp[BDRY_COL_OUT_E];
        let msg_exp = &bp[BDRY_COL_MSG_EXPECTED];

        // ── B1: sel_init · (â − H[0]) = 0 ──────────────────────────────
        let h0 = from_ref(&BinaryPoly::<32>::from(H[0]));
        b.assert_zero(sel_init.clone() * &(bp[COL_A_HAT].clone() - &h0));

        // ── B2: sel_init · (ê − H[4]) = 0 ──────────────────────────────
        let h4 = from_ref(&BinaryPoly::<32>::from(H[4]));
        b.assert_zero(sel_init.clone() * &(bp[COL_E_HAT].clone() - &h4));

        // ── B3: sel_init · (d_hat − H[3]) = 0 ──────────────────────────
        let h3 = from_ref(&BinaryPoly::<32>::from(H[3]));
        b.assert_zero(sel_init.clone() * &(bp[COL_D_HAT].clone() - &h3));

        // ── B4: sel_init · (h_hat − H[7]) = 0 ──────────────────────────
        let h7 = from_ref(&BinaryPoly::<32>::from(H[7]));
        b.assert_zero(sel_init.clone() * &(bp[COL_H_HAT].clone() - &h7));

        // ── B5: sel_init · (a_tm1 − H[1]) = 0 ──────────────────────────
        let h1 = from_ref(&BinaryPoly::<32>::from(H[1]));
        b.assert_zero(sel_init.clone() * &(bp[COL_A_TM1].clone() - &h1));

        // ── B6: sel_init · (a_tm2 − H[2]) = 0 ──────────────────────────
        let h2 = from_ref(&BinaryPoly::<32>::from(H[2]));
        b.assert_zero(sel_init.clone() * &(bp[COL_A_TM2].clone() - &h2));

        // ── B7: sel_init · (e_tm1 − H[5]) = 0 ──────────────────────────
        let h5 = from_ref(&BinaryPoly::<32>::from(H[5]));
        b.assert_zero(sel_init.clone() * &(bp[COL_E_TM1].clone() - &h5));

        // ── B8: sel_init · (e_tm2 − H[6]) = 0 ──────────────────────────
        let h6 = from_ref(&BinaryPoly::<32>::from(H[6]));
        b.assert_zero(sel_init.clone() * &(bp[COL_E_TM2].clone() - &h6));

        // ── B9: sel_final_any · (â − out_a) = 0 ────────────────────────
        b.assert_zero(sel_final.clone() * &(bp[COL_A_HAT].clone() - out_a));

        // ── B10: sel_final_any · (ê − out_e) = 0 ───────────────────────
        b.assert_zero(sel_final.clone() * &(bp[COL_E_HAT].clone() - out_e));

        // ── B11: sel_msg · (W_hat − msg_expected) = 0 ──────────────────
        b.assert_zero(sel_msg.clone() * &(bp[COL_W_HAT].clone() - msg_exp));

        // ── B12: sel_zero · â = 0 ──────────────────────────────────────
        b.assert_zero(sel_zero.clone() * &bp[COL_A_HAT]);

        // ── B13: sel_zero · ê = 0 ──────────────────────────────────────
        b.assert_zero(sel_zero.clone() * &bp[COL_E_HAT]);
    }
}

// ─── Witness generation ─────────────────────────────────────────────────────

use zinc_poly::mle::DenseMultilinearExtension;

/// Extend a no-f2x witness trace with the 7 boundary columns.
///
/// Takes the base no-f2x trace (33 or 39 bitpoly + 3 int columns) and
/// appends sel_init, sel_final_any, sel_msg, sel_zero, out_a, out_e,
/// and msg_expected columns.
///
/// The output columns (`out_a`, `out_e`) are populated from the trace
/// values at the final active rows, so the constraints are trivially
/// satisfied. A verifier receiving these as public columns can check
/// the claimed hash output.
pub fn generate_boundary_witness<const D: usize>(
    base: &[DenseMultilinearExtension<BinaryPoly<D>>],
    num_vars: usize,
) -> Vec<DenseMultilinearExtension<BinaryPoly<D>>>
where
    BinaryPoly<D>: From<u32> + Clone + Default,
{
    let num_rows: usize = 1 << num_vars;
    let last_active = num_rows.saturating_sub(5);
    let zero = BinaryPoly::<D>::from(0u32);
    let one = BinaryPoly::<D>::from(1u32);

    let mut sel_init = vec![BinaryPoly::<D>::from(0u32); num_rows];
    let mut sel_final_any = vec![BinaryPoly::<D>::from(0u32); num_rows];
    let mut sel_msg = vec![BinaryPoly::<D>::from(0u32); num_rows];
    let mut sel_zero = vec![BinaryPoly::<D>::from(0u32); num_rows];
    let mut out_a = vec![BinaryPoly::<D>::from(0u32); num_rows];
    let mut out_e = vec![BinaryPoly::<D>::from(0u32); num_rows];
    let mut msg_expected = vec![BinaryPoly::<D>::from(0u32); num_rows];

    // sel_init: 1 at row 0
    sel_init[0] = one.clone();

    // sel_final_any: 1 at last 4 active rows (last, last-1, last-2, last-3)
    for offset in 0..4 {
        let row = last_active.saturating_sub(offset);
        if row < num_rows {
            sel_final_any[row] = one.clone();
        }
    }

    // sel_msg: 1 at rows 0–15
    for t in 0..16.min(num_rows) {
        sel_msg[t] = one.clone();
    }

    // sel_zero: 1 at rows N-3, N-2, N-1
    for t in num_rows.saturating_sub(3)..num_rows {
        sel_zero[t] = one.clone();
    }

    // out_a: expected â at final rows
    //   row last_active   = a (register a)
    //   row last_active-1 = b (register b = a[t-1])
    //   row last_active-2 = c (register c = a[t-2])
    //   row last_active-3 = d (register d = a[t-3])
    for offset in 0..4 {
        let row = last_active.saturating_sub(offset);
        if row < num_rows {
            out_a[row] = base[COL_A_HAT].evaluations[row].clone();
        }
    }

    // out_e: expected ê at final rows
    //   row last_active   = e (register e)
    //   row last_active-1 = f (register f = e[t-1])
    //   row last_active-2 = g (register g = e[t-2])
    //   row last_active-3 = h (register h = e[t-3])
    for offset in 0..4 {
        let row = last_active.saturating_sub(offset);
        if row < num_rows {
            out_e[row] = base[COL_E_HAT].evaluations[row].clone();
        }
    }

    // msg_expected: expected message words at rows 0-15
    // These match W_hat (column 2) at the first 16 rows.
    for t in 0..16.min(num_rows) {
        msg_expected[t] = base[COL_W_HAT].evaluations[t].clone();
    }

    // Build the extended trace: base bitpoly + boundary columns + base int columns.
    let base_bp_count = NO_F2X_NUM_BITPOLY_COLS;
    let int_count = crate::no_f2x::NO_F2X_NUM_INT_COLS;
    let mut result: Vec<DenseMultilinearExtension<BinaryPoly<D>>> =
        Vec::with_capacity(BDRY_NUM_BITPOLY_COLS + int_count);

    // Copy base bitpoly columns (0 .. base_bp_count).
    for i in 0..base_bp_count {
        result.push(base[i].clone());
    }

    // Append boundary columns.
    for col in [sel_init, sel_final_any, sel_msg, sel_zero, out_a, out_e, msg_expected] {
        result.push(DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            col,
            zero.clone(),
        ));
    }

    // Append int columns.
    for i in base_bp_count..(base_bp_count + int_count) {
        result.push(base[i].clone());
    }

    result
}

/// Generate only the 7 boundary columns without cloning the base trace.
///
/// Returns `[sel_init, sel_final_any, sel_msg, sel_zero, out_a, out_e, msg_expected]`
/// as dense MLEs.  The caller can prepend the base trace columns
/// to reconstruct the full boundary trace if needed (e.g. for the CPR).
pub fn generate_boundary_columns_only<const D: usize>(
    base: &[DenseMultilinearExtension<BinaryPoly<D>>],
    num_vars: usize,
) -> Vec<DenseMultilinearExtension<BinaryPoly<D>>>
where
    BinaryPoly<D>: From<u32> + Clone + Default,
{
    let num_rows: usize = 1 << num_vars;
    let last_active = num_rows.saturating_sub(5);
    let zero = BinaryPoly::<D>::from(0u32);
    let one = BinaryPoly::<D>::from(1u32);

    let mut sel_init = vec![BinaryPoly::<D>::from(0u32); num_rows];
    let mut sel_final_any = vec![BinaryPoly::<D>::from(0u32); num_rows];
    let mut sel_msg = vec![BinaryPoly::<D>::from(0u32); num_rows];
    let mut sel_zero = vec![BinaryPoly::<D>::from(0u32); num_rows];
    let mut out_a = vec![BinaryPoly::<D>::from(0u32); num_rows];
    let mut out_e = vec![BinaryPoly::<D>::from(0u32); num_rows];
    let mut msg_expected = vec![BinaryPoly::<D>::from(0u32); num_rows];

    sel_init[0] = one.clone();

    for offset in 0..4 {
        let row = last_active.saturating_sub(offset);
        if row < num_rows {
            sel_final_any[row] = one.clone();
        }
    }

    for t in 0..16.min(num_rows) {
        sel_msg[t] = one.clone();
    }

    for t in num_rows.saturating_sub(3)..num_rows {
        sel_zero[t] = one.clone();
    }

    for offset in 0..4 {
        let row = last_active.saturating_sub(offset);
        if row < num_rows {
            out_a[row] = base[COL_A_HAT].evaluations[row].clone();
        }
    }

    for offset in 0..4 {
        let row = last_active.saturating_sub(offset);
        if row < num_rows {
            out_e[row] = base[COL_E_HAT].evaluations[row].clone();
        }
    }

    for t in 0..16.min(num_rows) {
        msg_expected[t] = base[COL_W_HAT].evaluations[t].clone();
    }

    [sel_init, sel_final_any, sel_msg, sel_zero, out_a, out_e, msg_expected]
        .into_iter()
        .map(|col| DenseMultilinearExtension::from_evaluations_vec(num_vars, col, zero.clone()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use zinc_uair::{
        collect_scalars::collect_scalars,
        constraint_counter::count_constraints,
        degree_counter::count_max_degree,
    };

    #[test]
    fn boundary_correct_constraint_count() {
        assert_eq!(
            count_constraints::<Sha256UairBoundaryNoF2x>(),
            BDRY_NUM_CONSTRAINTS,
        );
    }

    #[test]
    fn boundary_max_degree_is_two() {
        assert_eq!(count_max_degree::<Sha256UairBoundaryNoF2x>(), 2);
    }

    #[test]
    fn boundary_correct_column_count() {
        let sig = Sha256UairBoundaryNoF2x::signature();
        assert_eq!(sig.binary_poly_cols, BDRY_NUM_BITPOLY_COLS);
    }

    #[test]
    fn boundary_public_columns() {
        let sig = Sha256UairBoundaryNoF2x::signature();
        assert_eq!(sig.public_columns.len(), BDRY_NUM_NEW_COLS);
        // All boundary columns should be public.
        for &col in &sig.public_columns {
            assert!(col >= NO_F2X_NUM_BITPOLY_COLS);
        }
    }
}
