//! ChaCha20 UAIR+ (Universal Algebraic Intermediate Representation).
//!
//! This crate defines the ChaCha20 arithmetization as a UAIR+ with two trace
//! components following the no-F2x approach:
//!
//! - **Bp UAIR**: no constraints (placeholder).
//! - **Qx UAIR**: rotation constraints (cyclotomic ideal) +
//!   carry constraints (trivial or degree-one ideal), all degree 1.
//!   Uses shifted output columns to reference the previous row's outputs
//!   instead of storing separate input columns.
//!
//! # Trace layout — one row per quarter-round
//!
//! Each row encodes a full ChaCha20 quarter-round QR(a,b,c,d):
//!
//! ```text
//! Step 0: a  += b;  d  ^= a;  d  <<<= 16
//! Step 1: c  += d;  b  ^= c;  b  <<<= 12
//! Step 2: a  += b;  d  ^= a;  d  <<<= 8
//! Step 3: c  += d;  b  ^= c;  b  <<<= 7
//! ```
//!
//! The inputs (a, b, c, d) are the previous row's outputs
//! (sum2, b2, sum3, d2), accessed via shift specs. No explicit input
//! columns are stored.
//!
//! ## Bit-polynomial columns — {0,1}^{<32}[X] (indices 0–11)
//!
//! | Index | Name     | Description                                      |
//! |-------|----------|--------------------------------------------------|
//! | 0     | `sum0`   | a + b mod 2^32 (= a1)                            |
//! | 1     | `and0`   | d AND sum0 (XOR auxiliary)                        |
//! | 2     | `d1`     | (d XOR sum0) <<< 16                              |
//! | 3     | `sum1`   | c + d1 mod 2^32 (= c1)                           |
//! | 4     | `and1`   | b AND sum1 (XOR auxiliary)                        |
//! | 5     | `b1`     | (b XOR sum1) <<< 12                              |
//! | 6     | `sum2`   | sum0 + b1 mod 2^32 (= a_out)                     |
//! | 7     | `and2`   | d1 AND sum2 (XOR auxiliary)                       |
//! | 8     | `d2`     | (d1 XOR sum2) <<< 8 (= d_out)                    |
//! | 9     | `sum3`   | sum1 + d2 mod 2^32 (= c_out)                     |
//! | 10    | `and3`   | b1 AND sum3 (XOR auxiliary)                       |
//! | 11    | `b2`     | (b1 XOR sum3) <<< 7 (= b_out)                    |
//!
//! ## Correction columns (true-ideal only, indices 12–19, public)
//!
//! | Index | Name           | Description                              |
//! |-------|----------------|------------------------------------------|
//! | 12    | `corr_add_0`   | Additive correction for carry 0          |
//! | 13    | `corr_add_1`   | Additive correction for carry 1          |
//! | 14    | `corr_add_2`   | Additive correction for carry 2          |
//! | 15    | `corr_add_3`   | Additive correction for carry 3          |
//! | 16    | `corr_sub_0`   | Subtractive correction for carry 0       |
//! | 17    | `corr_sub_1`   | Subtractive correction for carry 1       |
//! | 18    | `corr_sub_2`   | Subtractive correction for carry 2       |
//! | 19    | `corr_sub_3`   | Subtractive correction for carry 3       |
//!
//! ## Integer columns (indices 12–15 or 20–23 in flattened trace)
//!
//! | Index | Name   | Description                                      |
//! |-------|--------|--------------------------------------------------|
//! | 0     | `mu0`  | Carry for sum0 = a + b                           |
//! | 1     | `mu1`  | Carry for sum1 = c + d1                          |
//! | 2     | `mu2`  | Carry for sum2 = sum0 + b1                       |
//! | 3     | `mu3`  | Carry for sum3 = sum1 + d2                       |
//!
//! # Constraints
//!
//! ## Bp UAIR (0 constraints)
//!
//! No constraints — input linking is handled implicitly by the Qx UAIR
//! via shifted column references.
//!
//! ## Qx UAIR (8 constraints)
//!
//! Shifted columns (from previous row):
//!   - `a_prev = sum2[t-1]`, `b_prev = b2[t-1]`
//!   - `c_prev = sum3[t-1]`, `d_prev = d2[t-1]`
//!
//! Rotation constraints (cyclotomic ideal):
//! 1. `(d_prev + sum0 − 2·and0) · X^16 − d1 ∈ (X^32 − 1)`
//! 2. `(b_prev + sum1 − 2·and1) · X^12 − b1 ∈ (X^32 − 1)`
//! 3. `(d1 + sum2 − 2·and2) · X^8  − d2 ∈ (X^32 − 1)`
//! 4. `(b1 + sum3 − 2·and3) · X^7  − b2 ∈ (X^32 − 1)`
//!
//! Carry constraints (trivial / degree-one ideal):
//! 5. `sum0 − a_prev − b_prev + mu0·X^32 ∈ Ideal`
//! 6. `sum1 − c_prev − d1     + mu1·X^32 ∈ Ideal`
//! 7. `sum2 − sum0   − b1     + mu2·X^32 ∈ Ideal`
//! 8. `sum3 − sum1   − d2     + mu3·X^32 ∈ Ideal`

#![allow(clippy::arithmetic_side_effects)] // UAIRs should not care about overflows

pub mod constants;
pub mod witness;

use crypto_primitives::PrimeField;
use zinc_poly::univariate::binary::BinaryPoly;
use zinc_poly::univariate::dense::DensePolynomial;
use zinc_poly::univariate::dynamic::over_field::DynamicPolynomialF;
use zinc_uair::{
    ConstraintBuilder, Uair,
    ideal::{Ideal, IdealCheck},
};
use zinc_utils::from_ref::FromRef;

// ─── Column indices ──────────────────────────────────────────────────────────

pub const COL_SUM0: usize = 0;
pub const COL_AND0: usize = 1;
pub const COL_D1: usize = 2;
pub const COL_SUM1: usize = 3;
pub const COL_AND1: usize = 4;
pub const COL_B1: usize = 5;
pub const COL_SUM2: usize = 6;
pub const COL_AND2: usize = 7;
pub const COL_D2: usize = 8;
pub const COL_SUM3: usize = 9;
pub const COL_AND3: usize = 10;
pub const COL_B2: usize = 11;

// Integer column indices (within the int slice)
pub const COL_INT_MU0: usize = 0;
pub const COL_INT_MU1: usize = 1;
pub const COL_INT_MU2: usize = 2;
pub const COL_INT_MU3: usize = 3;

// ─── Correction column indices (true-ideal only) ────────────────────────────

#[cfg(feature = "true-ideal")]
pub const COL_CORR_ADD_0: usize = 12;
#[cfg(feature = "true-ideal")]
pub const COL_CORR_ADD_1: usize = 13;
#[cfg(feature = "true-ideal")]
pub const COL_CORR_ADD_2: usize = 14;
#[cfg(feature = "true-ideal")]
pub const COL_CORR_ADD_3: usize = 15;
#[cfg(feature = "true-ideal")]
pub const COL_CORR_SUB_0: usize = 16;
#[cfg(feature = "true-ideal")]
pub const COL_CORR_SUB_1: usize = 17;
#[cfg(feature = "true-ideal")]
pub const COL_CORR_SUB_2: usize = 18;
#[cfg(feature = "true-ideal")]
pub const COL_CORR_SUB_3: usize = 19;

// ─── Counts ─────────────────────────────────────────────────────────────────

/// Number of base bit-polynomial columns (12).
pub const BASE_NUM_BITPOLY_COLS: usize = 12;

/// Number of bit-polynomial columns without true-ideal.
#[cfg(not(feature = "true-ideal"))]
pub const NUM_BITPOLY_COLS: usize = 12;

/// Number of bit-polynomial columns with true-ideal (12 + 8 correction).
#[cfg(feature = "true-ideal")]
pub const NUM_BITPOLY_COLS: usize = 20;

/// Number of integer columns (4 carries).
pub const NUM_INT_COLS: usize = 4;

/// Total number of trace columns.
pub const NUM_COLS: usize = NUM_BITPOLY_COLS + NUM_INT_COLS;

/// Number of Bp constraints (0 — linking handled via Qx shifts).
pub const BP_NUM_CONSTRAINTS: usize = 0;

/// Number of Qx constraints (4 rotation + 4 carry = 8).
pub const QX_NUM_CONSTRAINTS: usize = 8;

/// Number of columns for which we run bit-poly lookups.
/// Excludes the 4 AND auxiliary columns which are already
/// constrained by the rotation constraints.
pub const LOOKUP_COL_COUNT: usize = 8;

/// Column indices that require bit-poly lookups (all base columns except AND).
pub const LOOKUP_COLUMNS: [usize; 8] = [
    COL_SUM0, COL_D1,
    COL_SUM1, COL_B1,
    COL_SUM2, COL_D2,
    COL_SUM3, COL_B2,
];

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
        // In F₂[X], (X³² − 1) contains p iff Σ_{i mod 32} c_i = 0 for each
        // residue class. For BinaryPoly<32> (degree < 32) this is just "= 0".
        value.iter().all(|c| !c.into_inner())
    }
}

// ─── Qx ideal ──────────────────────────────────────────────────────────────

/// Ideal type for the Q[X] ChaCha20 UAIR.
#[derive(Clone, Copy, Debug)]
pub enum ChaCha20QxIdeal {
    /// The cyclotomic ideal (X^32 − 1).
    Cyclotomic,
    /// The degree-one ideal (X − 2): p(2) = 0.
    #[cfg(feature = "true-ideal")]
    DegreeOne,
    /// The trivial ideal: contains every polynomial.
    Trivial,
}

impl Ideal for ChaCha20QxIdeal {}

impl FromRef<ChaCha20QxIdeal> for ChaCha20QxIdeal {
    #[inline(always)]
    fn from_ref(ideal: &ChaCha20QxIdeal) -> Self {
        *ideal
    }
}

impl IdealCheck<DensePolynomial<i64, 64>> for ChaCha20QxIdeal {
    fn contains(&self, value: &DensePolynomial<i64, 64>) -> bool {
        match self {
            ChaCha20QxIdeal::Cyclotomic => {
                let mut reduced = [0i64; 32];
                for (i, &c) in value.coeffs.iter().enumerate() {
                    reduced[i % 32] = reduced[i % 32].wrapping_add(c);
                }
                reduced.iter().all(|&c| c == 0)
            }
            #[cfg(feature = "true-ideal")]
            ChaCha20QxIdeal::DegreeOne => {
                let mut eval: i64 = 0;
                for (i, &c) in value.coeffs.iter().enumerate() {
                    eval = eval.wrapping_add(c.wrapping_mul(1i64.wrapping_shl(i as u32)));
                }
                eval == 0
            }
            ChaCha20QxIdeal::Trivial => true,
        }
    }
}

// ─── Field-level ideal for verification ──────────────────────────────────

/// The ChaCha20 Q[X] ideal lifted to a prime field.
#[derive(Clone, Copy, Debug)]
pub enum ChaCha20QxIdealOverF {
    Cyclotomic,
    #[cfg(feature = "true-ideal")]
    DegreeOne,
    Trivial,
}

impl Ideal for ChaCha20QxIdealOverF {}

impl FromRef<ChaCha20QxIdealOverF> for ChaCha20QxIdealOverF {
    #[inline(always)]
    fn from_ref(ideal: &ChaCha20QxIdealOverF) -> Self {
        *ideal
    }
}

impl<F: PrimeField> IdealCheck<DynamicPolynomialF<F>> for ChaCha20QxIdealOverF {
    fn contains(&self, value: &DynamicPolynomialF<F>) -> bool {
        match self {
            ChaCha20QxIdealOverF::Cyclotomic => {
                if value.coeffs.is_empty() {
                    return true;
                }
                let cfg = value.coeffs[0].cfg();
                let zero = F::zero_with_cfg(cfg);
                let mut reduced: Vec<F> = vec![zero; 32];
                for (i, coeff) in value.coeffs.iter().enumerate() {
                    let j = i % 32;
                    reduced[j] = reduced[j].clone() + coeff;
                }
                reduced.iter().all(|c| F::is_zero(c))
            }
            #[cfg(feature = "true-ideal")]
            ChaCha20QxIdealOverF::DegreeOne => {
                if value.coeffs.is_empty() {
                    return true;
                }
                let cfg = value.coeffs[0].cfg();
                let mut result = F::zero_with_cfg(cfg);
                let mut power = F::one_with_cfg(cfg);
                for coeff in &value.coeffs {
                    result = result + &(coeff.clone() * &power);
                    power = power.clone() + &power;
                }
                F::is_zero(&result)
            }
            ChaCha20QxIdealOverF::Trivial => true,
        }
    }
}

// ─── Bp UAIR ────────────────────────────────────────────────────────────────

/// ChaCha20 Bp UAIR: no constraints (placeholder).
///
/// Input linking is handled implicitly by the Qx UAIR via shifted columns.
pub struct ChaCha20UairBp;

impl Uair for ChaCha20UairBp {
    type Ideal = CyclotomicIdeal;
    type Scalar = BinaryPoly<32>;

    fn signature() -> zinc_uair::UairSignature {
        zinc_uair::UairSignature {
            binary_poly_cols: NUM_BITPOLY_COLS,
            arbitrary_poly_cols: 0,
            int_cols: NUM_INT_COLS,
            // Declare shifts on the 4 output columns so they are
            // PCS-excluded and verified via shift sumcheck instead.
            // This reduces the PCS batch_size (and thus proof size).
            shifts: vec![
                zinc_uair::ShiftSpec { source_col: COL_SUM2, shift_amount: 1 },
                zinc_uair::ShiftSpec { source_col: COL_B2,   shift_amount: 1 },
                zinc_uair::ShiftSpec { source_col: COL_SUM3, shift_amount: 1 },
                zinc_uair::ShiftSpec { source_col: COL_D2,   shift_amount: 1 },
            ],
            public_columns: {
                let mut cols = Vec::new();
                #[cfg(feature = "true-ideal")]
                {
                    cols.push(COL_CORR_ADD_0);
                    cols.push(COL_CORR_ADD_1);
                    cols.push(COL_CORR_ADD_2);
                    cols.push(COL_CORR_ADD_3);
                    cols.push(COL_CORR_SUB_0);
                    cols.push(COL_CORR_SUB_1);
                    cols.push(COL_CORR_SUB_2);
                    cols.push(COL_CORR_SUB_3);
                }
                cols
            },
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
        FromR: Fn(&BinaryPoly<32>) -> B::Expr,
        MulByScalar: Fn(&B::Expr, &BinaryPoly<32>) -> Option<B::Expr>,
        IFromR: Fn(&CyclotomicIdeal) -> B::Ideal,
    {
        // No Bp constraints — rotation and carry constraints live in
        // the Qx UAIR. The Bp shifts above serve only to exclude the
        // 4 output columns from PCS (verified via shift sumcheck).
    }
}

// ─── Qx UAIR ────────────────────────────────────────────────────────────────

/// ChaCha20 Qx UAIR: rotation + carry constraints (degree 1).
///
/// Uses legacy shifts (all columns shifted by 1) so that:
///   - `up[col]`  = trace[col][t]   (current row = previous QR's outputs)
///   - `down[col]` = trace[col][t+1] (next row = the QR being constrained)
///
/// All constraints are "forward-looking": they express the relationship
/// between the current row's outputs (up) and the next row's computation (down).
///
/// Emits 8 constraints:
///   C1–C4: rotation constraints (cyclotomic ideal).
///   C5–C8: carry propagation (trivial or degree-one ideal).
pub struct ChaCha20UairQx;

impl Uair for ChaCha20UairQx {
    type Ideal = ChaCha20QxIdeal;
    type Scalar = DensePolynomial<i64, 64>;

    fn signature() -> zinc_uair::UairSignature {
        zinc_uair::UairSignature {
            binary_poly_cols: NUM_BITPOLY_COLS,
            arbitrary_poly_cols: 0,
            int_cols: NUM_INT_COLS,
            shifts: vec![],  // legacy mode: all columns shifted by 1
            public_columns: {
                let mut cols = Vec::new();
                #[cfg(feature = "true-ideal")]
                {
                    cols.push(COL_CORR_ADD_0);
                    cols.push(COL_CORR_ADD_1);
                    cols.push(COL_CORR_ADD_2);
                    cols.push(COL_CORR_ADD_3);
                    cols.push(COL_CORR_SUB_0);
                    cols.push(COL_CORR_SUB_1);
                    cols.push(COL_CORR_SUB_2);
                    cols.push(COL_CORR_SUB_3);
                }
                cols
            },
        }
    }

    fn constrain_general<B, FromR, MulByScalar, IFromR>(
        b: &mut B,
        up: zinc_uair::TraceRow<B::Expr>,
        down: zinc_uair::TraceRow<B::Expr>,
        from_ref: FromR,
        _mbs: MulByScalar,
        ideal_from_ref: IFromR,
    ) where
        B: ConstraintBuilder,
        FromR: Fn(&DensePolynomial<i64, 64>) -> B::Expr,
        MulByScalar: Fn(&B::Expr, &DensePolynomial<i64, 64>) -> Option<B::Expr>,
        IFromR: Fn(&ChaCha20QxIdeal) -> B::Ideal,
    {
        // Forward-looking: up = row t (previous outputs), down = row t+1 (next QR).
        // "prev" values come from up, "current QR" values come from down.
        let bp_up = up.binary_poly;     // previous row's outputs
        let bp_dn = down.binary_poly;   // next row's values (the QR being checked)
        let int_dn = down.int;          // next row's integer columns
        let cyclotomic = ideal_from_ref(&ChaCha20QxIdeal::Cyclotomic);
        #[cfg(not(feature = "true-ideal"))]
        let trivial = ideal_from_ref(&ChaCha20QxIdeal::Trivial);

        // ── Helpers ──────────────────────────────────────────────────────

        fn mono_poly(pos: usize) -> DensePolynomial<i64, 64> {
            let mut coeffs = [0i64; 64];
            coeffs[pos] = 1;
            DensePolynomial { coeffs }
        }

        fn const_poly(val: i64) -> DensePolynomial<i64, 64> {
            let mut coeffs = [0i64; 64];
            coeffs[0] = val;
            DensePolynomial { coeffs }
        }

        let two = from_ref(&const_poly(2));
        let x32 = from_ref(&mono_poly(32));

        // Rotation polynomials (single monomials for left rotation)
        let x16 = from_ref(&mono_poly(16));
        let x12 = from_ref(&mono_poly(12));
        let x8 = from_ref(&mono_poly(8));
        let x7 = from_ref(&mono_poly(7));

        // ── C1: rotation by 16 ─────────────────────────────────────────
        //   (d2[t] XOR sum0[t+1]) <<< 16 = d1[t+1]
        //   d2[t] = up[COL_D2], sum0[t+1] = dn[COL_SUM0], and0[t+1] = dn[COL_AND0]
        let xor0 = bp_up[COL_D2].clone()
            + &bp_dn[COL_SUM0]
            - &(bp_dn[COL_AND0].clone() * &two);
        b.assert_in_ideal(
            xor0 * &x16 - &bp_dn[COL_D1],
            &cyclotomic,
        );

        // ── C2: rotation by 12 ─────────────────────────────────────────
        //   (b2[t] XOR sum1[t+1]) <<< 12 = b1[t+1]
        let xor1 = bp_up[COL_B2].clone()
            + &bp_dn[COL_SUM1]
            - &(bp_dn[COL_AND1].clone() * &two);
        b.assert_in_ideal(
            xor1 * &x12 - &bp_dn[COL_B1],
            &cyclotomic,
        );

        // ── C3: rotation by 8 ──────────────────────────────────────────
        //   (d1[t+1] XOR sum2[t+1]) <<< 8 = d2[t+1]   (intra-row, all from dn)
        let xor2 = bp_dn[COL_D1].clone()
            + &bp_dn[COL_SUM2]
            - &(bp_dn[COL_AND2].clone() * &two);
        b.assert_in_ideal(
            xor2 * &x8 - &bp_dn[COL_D2],
            &cyclotomic,
        );

        // ── C4: rotation by 7 ──────────────────────────────────────────
        //   (b1[t+1] XOR sum3[t+1]) <<< 7 = b2[t+1]   (intra-row, all from dn)
        let xor3 = bp_dn[COL_B1].clone()
            + &bp_dn[COL_SUM3]
            - &(bp_dn[COL_AND3].clone() * &two);
        b.assert_in_ideal(
            xor3 * &x7 - &bp_dn[COL_B2],
            &cyclotomic,
        );

        // ── Carry constraints C5–C8 ─────────────────────────────────────
        // C5: sum0[t+1] = sum2[t] + b2[t]  (mod 2^32)
        let c5_inner = bp_dn[COL_SUM0].clone()
            - &bp_up[COL_SUM2]
            - &bp_up[COL_B2]
            + &(int_dn[COL_INT_MU0].clone() * &x32);

        // C6: sum1[t+1] = sum3[t] + d1[t+1]  (mod 2^32)
        let c6_inner = bp_dn[COL_SUM1].clone()
            - &bp_up[COL_SUM3]
            - &bp_dn[COL_D1]
            + &(int_dn[COL_INT_MU1].clone() * &x32);

        // C7: sum2[t+1] = sum0[t+1] + b1[t+1]  (mod 2^32, intra-row)
        let c7_inner = bp_dn[COL_SUM2].clone()
            - &bp_dn[COL_SUM0]
            - &bp_dn[COL_B1]
            + &(int_dn[COL_INT_MU2].clone() * &x32);

        // C8: sum3[t+1] = sum1[t+1] + d2[t+1]  (mod 2^32, intra-row)
        let c8_inner = bp_dn[COL_SUM3].clone()
            - &bp_dn[COL_SUM1]
            - &bp_dn[COL_D2]
            + &(int_dn[COL_INT_MU3].clone() * &x32);

        #[cfg(not(feature = "true-ideal"))]
        {
            b.assert_in_ideal(c5_inner, &trivial);
            b.assert_in_ideal(c6_inner, &trivial);
            b.assert_in_ideal(c7_inner, &trivial);
            b.assert_in_ideal(c8_inner, &trivial);
        }

        #[cfg(feature = "true-ideal")]
        {
            let degree_one = ideal_from_ref(&ChaCha20QxIdeal::DegreeOne);

            b.assert_in_ideal(
                c5_inner + &bp_dn[COL_CORR_ADD_0] - &bp_dn[COL_CORR_SUB_0],
                &degree_one,
            );
            b.assert_in_ideal(
                c6_inner + &bp_dn[COL_CORR_ADD_1] - &bp_dn[COL_CORR_SUB_1],
                &degree_one,
            );
            b.assert_in_ideal(
                c7_inner + &bp_dn[COL_CORR_ADD_2] - &bp_dn[COL_CORR_SUB_2],
                &degree_one,
            );
            b.assert_in_ideal(
                c8_inner + &bp_dn[COL_CORR_ADD_3] - &bp_dn[COL_CORR_SUB_3],
                &degree_one,
            );
        }
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
        let sig = ChaCha20UairBp::signature();
        assert_eq!(sig.binary_poly_cols, NUM_BITPOLY_COLS);
        assert_eq!(sig.int_cols, NUM_INT_COLS);
        assert_eq!(sig.total_cols(), NUM_COLS);
    }

    #[test]
    fn bp_correct_constraint_count() {
        assert_eq!(count_constraints::<ChaCha20UairBp>(), BP_NUM_CONSTRAINTS);
    }

    #[test]
    fn qx_correct_column_count() {
        let sig = ChaCha20UairQx::signature();
        assert_eq!(sig.binary_poly_cols, NUM_BITPOLY_COLS);
        assert_eq!(sig.int_cols, NUM_INT_COLS);
        assert_eq!(sig.total_cols(), NUM_COLS);
    }

    #[test]
    fn qx_correct_constraint_count() {
        assert_eq!(count_constraints::<ChaCha20UairQx>(), QX_NUM_CONSTRAINTS);
    }

    #[test]
    fn qx_max_degree_is_one() {
        assert_eq!(count_max_degree::<ChaCha20UairQx>(), 1);
    }
}
