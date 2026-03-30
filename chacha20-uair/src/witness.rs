//! Witness generation for the ChaCha20 UAIR+.
//!
//! Generates a trace of chained ChaCha20 quarter-round operations.
//! Each row represents one full quarter-round QR(a,b,c,d) with 4 steps.
//!
//! The trace is a chain: each QR's output (sum2, b2, sum3, d2)
//! feeds directly into the next QR's input via shifted column references.
//! No explicit input columns are stored.

use zinc_poly::{
    mle::DenseMultilinearExtension,
    univariate::binary::BinaryPoly,
};

use crate::{
    NUM_COLS, NUM_BITPOLY_COLS,
    COL_SUM0, COL_AND0, COL_D1,
    COL_SUM1, COL_AND1, COL_B1,
    COL_SUM2, COL_AND2, COL_D2,
    COL_SUM3, COL_AND3, COL_B2,
};

// ─── BinaryPoly helpers ─────────────────────────────────────────────────────

fn u32_to_bp(val: u32) -> BinaryPoly<32> {
    BinaryPoly::<32>::from(val)
}

// ─── ChaCha20 quarter-round (plain u32 arithmetic) ──────────────────────────

/// Compute one quarter-round and return all intermediates.
///
/// Returns `(sum0, and0, d1, sum1, and1, b1, sum2, and2, d2, sum3, and3, b2,
///           mu0, mu1, mu2, mu3)` as u32 values.
#[allow(clippy::type_complexity)]
fn quarter_round_with_intermediates(
    a: u32, b: u32, c: u32, d: u32,
) -> (u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32,
      u32, u32, u32, u32) {
    // Step 0: a += b; d ^= a; d <<<= 16
    let (sum0, mu0) = add_with_carry(a, b);
    let and0 = d & sum0;
    let d1 = (d ^ sum0).rotate_left(16);

    // Step 1: c += d; b ^= c; b <<<= 12
    let (sum1, mu1) = add_with_carry(c, d1);
    let and1 = b & sum1;
    let b1 = (b ^ sum1).rotate_left(12);

    // Step 2: a += b; d ^= a; d <<<= 8
    let (sum2, mu2) = add_with_carry(sum0, b1);
    let and2 = d1 & sum2;
    let d2 = (d1 ^ sum2).rotate_left(8);

    // Step 3: c += d; b ^= c; b <<<= 7
    let (sum3, mu3) = add_with_carry(sum1, d2);
    let and3 = b1 & sum3;
    let b2 = (b1 ^ sum3).rotate_left(7);

    (sum0, and0, d1, sum1, and1, b1, sum2, and2, d2, sum3, and3, b2,
     mu0, mu1, mu2, mu3)
}

/// Addition mod 2^32 returning (sum, carry).
fn add_with_carry(x: u32, y: u32) -> (u32, u32) {
    let full = x as u64 + y as u64;
    #[allow(clippy::cast_possible_truncation)]
    let sum = full as u32;
    #[allow(clippy::cast_possible_truncation)]
    let carry = (full >> 32) as u32;
    (sum, carry)
}

// ─── Witness generation ─────────────────────────────────────────────────────

/// Generate the ChaCha20 witness trace.
///
/// The trace has `2^{num_vars}` rows of chained quarter-rounds.
/// The initial state is derived from ChaCha20 constants and a
/// deterministic key/nonce.
///
/// Returns `NUM_COLS` columns as `DenseMultilinearExtension<BinaryPoly<32>>`.
pub fn generate_chacha20_witness(
    num_vars: usize,
) -> Vec<DenseMultilinearExtension<BinaryPoly<32>>> {
    assert!(
        num_vars >= 4,
        "ChaCha20 requires at least 4 variables (16 rows), got {num_vars}"
    );

    let num_rows = 1usize << num_vars;

    // Allocate column storage.
    let mut cols: Vec<Vec<BinaryPoly<32>>> =
        (0..NUM_COLS).map(|_| vec![BinaryPoly::<32>::from(0u32); num_rows]).collect();

    // Initial state: ChaCha20 constants + deterministic key + counter + nonce.
    let mut a = crate::constants::CHACHA_CONSTANTS[0];
    let mut b_val = 0x0302_0100u32; // deterministic key bytes
    let mut c = 0x0706_0504u32;
    let mut d = 0x0B0A_0908u32;

    for t in 0..num_rows {
        let (sum0, and0, d1, sum1, and1, b1, sum2, and2, d2, sum3, and3, b2,
             mu0, mu1, mu2, mu3) =
            quarter_round_with_intermediates(a, b_val, c, d);

        // Write bit-poly columns (no input columns — they are implicit via shifts).
        cols[COL_SUM0][t] = u32_to_bp(sum0);
        cols[COL_AND0][t] = u32_to_bp(and0);
        cols[COL_D1][t] = u32_to_bp(d1);
        cols[COL_SUM1][t] = u32_to_bp(sum1);
        cols[COL_AND1][t] = u32_to_bp(and1);
        cols[COL_B1][t] = u32_to_bp(b1);
        cols[COL_SUM2][t] = u32_to_bp(sum2);
        cols[COL_AND2][t] = u32_to_bp(and2);
        cols[COL_D2][t] = u32_to_bp(d2);
        cols[COL_SUM3][t] = u32_to_bp(sum3);
        cols[COL_AND3][t] = u32_to_bp(and3);
        cols[COL_B2][t] = u32_to_bp(b2);

        // Write integer columns (carries).
        let int_base = NUM_BITPOLY_COLS;
        cols[int_base + 0][t] = u32_to_bp(mu0);
        cols[int_base + 1][t] = u32_to_bp(mu1);
        cols[int_base + 2][t] = u32_to_bp(mu2);
        cols[int_base + 3][t] = u32_to_bp(mu3);

        // Chain: output feeds next input.
        a = sum2;       // a_out
        b_val = b2;     // b_out
        c = sum3;       // c_out
        d = d2;         // d_out
    }

    // Add true-ideal correction columns.
    #[cfg(feature = "true-ideal")]
    compute_corrections(&mut cols, num_rows);

    // Build DenseMultilinearExtension for each column.
    cols.into_iter()
        .map(|col| {
            DenseMultilinearExtension::from_evaluations_vec(
                num_vars,
                col,
                BinaryPoly::<32>::from(0u32),
            )
        })
        .collect()
}

/// Compute correction columns for the true-ideal feature.
///
/// Each carry constraint (C5–C8) needs correction columns so that
/// the constraint expression evaluates to 0 at X = 2.
#[cfg(feature = "true-ideal")]
fn compute_corrections(
    cols: &mut [Vec<BinaryPoly<32>>],
    num_rows: usize,
) {
    fn bp_val(cols: &[Vec<BinaryPoly<32>>], col: usize, t: usize) -> u64 {
        let bp = &cols[col][t];
        let mut val = 0u64;
        for (i, coeff) in bp.iter().enumerate() {
            if coeff.into_inner() {
                val |= 1u64 << i;
            }
        }
        val
    }

    // Process each carry constraint.
    // C5: sum0 - a_prev - b_prev + mu0·2^32
    //     a_prev = sum2[t-1], b_prev = b2[t-1]
    // C6: sum1 - c_prev - d1 + mu1·2^32
    //     c_prev = sum3[t-1]
    // C7: sum2 - sum0 - b1 + mu2·2^32
    // C8: sum3 - sum1 - d2 + mu3·2^32
    //
    // For C5 and C6, we use the previous row's output columns.
    // For C7 and C8, all references are within the current row.

    let corr_base = crate::BASE_NUM_BITPOLY_COLS; // correction cols start at index 12

    // C7 and C8 are intra-row, same as before.
    let intra_row_specs: [(usize, usize, usize, usize); 2] = [
        (COL_SUM2, COL_SUM0, COL_B1, 2), // C7
        (COL_SUM3, COL_SUM1, COL_D2, 3), // C8
    ];

    for (sum_col, x_col, y_col, idx) in &intra_row_specs {
        let corr_add_col = corr_base + idx;
        let corr_sub_col = corr_base + 4 + idx;

        for t in 0..num_rows {
            let sum_v = bp_val(cols, *sum_col, t) as i64;
            let x_v = bp_val(cols, *x_col, t) as i64;
            let y_v = bp_val(cols, *y_col, t) as i64;
            let mu_v = bp_val(cols, crate::NUM_BITPOLY_COLS + idx, t) as i64;

            let residual = sum_v - x_v - y_v + mu_v * (1i64 << 32);

            if residual < 0 {
                #[allow(clippy::cast_sign_loss)]
                {
                    cols[corr_add_col][t] = u32_to_bp((-residual) as u32);
                }
            } else if residual > 0 {
                #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
                {
                    cols[corr_sub_col][t] = u32_to_bp(residual as u32);
                }
            }
        }
    }

    // C5: sum0 - sum2[t-1] - b2[t-1] + mu0·2^32
    // C6: sum1 - sum3[t-1] - d1 + mu1·2^32
    let cross_row_specs: [(usize, usize, usize, Option<usize>, usize); 2] = [
        (COL_SUM0, COL_SUM2, COL_B2, None, 0),       // C5: x=sum2[t-1], y=b2[t-1]
        (COL_SUM1, COL_SUM3, COL_D1, Some(COL_D1), 1), // C6: x=sum3[t-1], y_cur=d1
    ];

    for (sum_col, prev_x_col, prev_y_or_cur_col, cur_y_col, idx) in &cross_row_specs {
        let corr_add_col = corr_base + idx;
        let corr_sub_col = corr_base + 4 + idx;

        for t in 0..num_rows {
            let sum_v = bp_val(cols, *sum_col, t) as i64;
            let mu_v = bp_val(cols, crate::NUM_BITPOLY_COLS + idx, t) as i64;

            let prev_t = if t == 0 { num_rows - 1 } else { t - 1 };

            let (x_v, y_v) = if let Some(cur_y) = cur_y_col {
                // C6: x = sum3[t-1], y = d1[t] (current row)
                let x = bp_val(cols, *prev_x_col, prev_t) as i64;
                let y = bp_val(cols, *cur_y, t) as i64;
                (x, y)
            } else {
                // C5: x = sum2[t-1], y = b2[t-1]
                let x = bp_val(cols, *prev_x_col, prev_t) as i64;
                let y = bp_val(cols, *prev_y_or_cur_col, prev_t) as i64;
                (x, y)
            };

            let residual = sum_v - x_v - y_v + mu_v * (1i64 << 32);

            if residual < 0 {
                #[allow(clippy::cast_sign_loss)]
                {
                    cols[corr_add_col][t] = u32_to_bp((-residual) as u32);
                }
            } else if residual > 0 {
                #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
                {
                    cols[corr_sub_col][t] = u32_to_bp(residual as u32);
                }
            }
        }
    }
}
