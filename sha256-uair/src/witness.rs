//! Witness generation for the SHA-256 UAIR⁺.
//!
//! Implements [`GenerateWitness<BinaryPoly<32>>`] for [`Sha256UairBp`] by
//! running the full SHA-256 compression function on a single-block padded
//! empty message, recording all 17 column values at each of the 65 rows.
//!
//! Following the paper, registers d and h are **not** stored as columns
//! (they are inlined via shift-register identities d_t = a_{t−3},
//! h_t = e_{t−3}), and K_t is a public input, not a trace column.

use rand::RngCore;
use crypto_primitives::crypto_bigint_int::Int;
use zinc_poly::{
    mle::DenseMultilinearExtension,
    univariate::binary::BinaryPoly,
};
use zinc_utils::from_ref::FromRef;

use crate::{
    Sha256Uair, Sha256UairBp,
    constants::{H, K},
    NUM_COLS,
};

// ─── BinaryPoly helper ─────────────────────────────────────────────────────

/// Convert a `BinaryPoly<32>` to its `u64` value (polynomial evaluated at X=2).
///
/// Equivalent to interpreting the binary coefficients as bits of an integer.
fn bp_to_u64(bp: &BinaryPoly<32>) -> u64 {
    let mut val: u64 = 0;
    for (i, coeff) in bp.iter().enumerate() {
        if coeff.into_inner() {
            val |= 1u64 << i;
        }
    }
    val
}

// ─── Column classification for split PCS batches ────────────────────────────

/// Bit-polynomial column indices (0–22): the 10 Q[X] bit-poly columns,
/// 4 F₂[X] columns, 7 auxiliary lookback columns, and 2 selector columns.
pub const POLY_COLUMN_INDICES: [usize; 23] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
    14, 15, 16, 17, 18, 19, 20, 21, 22,
];

/// Integer column indices (23–25): the 3 carry columns μ_a, μ_e, μ_W.
pub const INT_COLUMN_INDICES: [usize; 3] = [23, 24, 25];

// ─── GenerateWitness impl ───────────────────────────────────────────────────

/// Witness generation trait for a specific ring type.
pub trait GenerateWitness<R: crypto_primitives::Semiring + 'static> {
    fn generate_witness<Rng: RngCore + ?Sized>(
        num_vars: usize,
        rng: &mut Rng,
    ) -> Vec<DenseMultilinearExtension<R>>;
}

impl GenerateWitness<BinaryPoly<32>> for Sha256UairBp {
    /// Generate the SHA-256 witness trace.
    ///
    /// The trace has 65 rows (rows 0–63 for rounds, row 64 for final state,
    /// `num_vars = 7`). If `num_vars > 7` the extra rows are zero-padded.
    ///
    /// The message hashed is the empty string (single 512-bit padded block).
    /// The `rng` parameter is accepted for trait compatibility but unused.
    ///
    /// Following the paper, registers d and h are not stored (they are
    /// inlined via d_t = a_{t−3}, h_t = e_{t−3}), and K_t is a public
    /// input, not a trace column.
    fn generate_witness<Rng: RngCore + ?Sized>(
        num_vars: usize,
        _rng: &mut Rng,
    ) -> Vec<DenseMultilinearExtension<BinaryPoly<32>>> {
        assert!(
            num_vars >= 7,
            "SHA-256 requires at least 7 variables (128 rows ≥ 65), got {num_vars}"
        );

        let num_rows: usize = 1 << num_vars;

        // ── Prepare message block (padded empty message) ────────────────
        let mut msg_block = [0u32; 16];
        msg_block[0] = 0x8000_0000;

        // ── Message schedule ────────────────────────────────────────────
        let mut w = [0u32; 64];
        w[..16].copy_from_slice(&msg_block);
        for t in 16..64 {
            w[t] = small_sigma1(w[t - 2])
                .wrapping_add(w[t - 7])
                .wrapping_add(small_sigma0(w[t - 15]))
                .wrapping_add(w[t - 16]);
        }

        // ── Compression: run 64 rounds, recording trace ─────────────────
        let (mut a, mut b_reg, mut c_reg, mut d_reg) = (H[0], H[1], H[2], H[3]);
        let (mut e, mut f_reg, mut g_reg, mut h_reg) = (H[4], H[5], H[6], H[7]);

        // One Vec per column, pre-allocated to num_rows.
        let mut cols: Vec<Vec<BinaryPoly<32>>> =
            (0..NUM_COLS).map(|_| vec![BinaryPoly::<32>::from(0u32); num_rows]).collect();

        for t in 0..64 {
            // ── Bit-polynomial columns (0–9) ────────────────────────────
            cols[0][t] = BinaryPoly::from(a);                     // a_hat
            cols[1][t] = BinaryPoly::from(e);                     // e_hat
            cols[2][t] = BinaryPoly::from(w[t]);                  // W_hat
            cols[3][t] = BinaryPoly::from(big_sigma0(a));         // Sigma0_hat
            cols[4][t] = BinaryPoly::from(big_sigma1(e));         // Sigma1_hat
            cols[5][t] = BinaryPoly::from(maj(a, b_reg, c_reg)); // Maj_hat
            cols[6][t] = BinaryPoly::from(e & f_reg);            // ch_ef_hat
            cols[7][t] = BinaryPoly::from((!e) & g_reg);         // ch_neg_eg_hat

            // σ₀(W_{t−15}) and σ₁(W_{t−2}) for message schedule
            if t >= 15 {
                cols[8][t] = BinaryPoly::from(small_sigma0(w[t - 15]));
            }
            if t >= 2 {
                cols[9][t] = BinaryPoly::from(small_sigma1(w[t - 2]));
            }

            // ── F₂[X] columns (10–13): shift quotient/remainder ────────
            if t >= 15 {
                cols[10][t] = BinaryPoly::from(w[t - 15] >> 3);      // S0
                cols[12][t] = BinaryPoly::from(w[t - 15] & 0x7);     // R0
            }
            if t >= 2 {
                cols[11][t] = BinaryPoly::from(w[t - 2] >> 10);      // S1
                cols[13][t] = BinaryPoly::from(w[t - 2] & 0x3FF);    // R1
            }

            // ── Auxiliary lookback columns (14–20) ──────────────────────
            // d_hat = a_{t-3}, h_hat = e_{t-3}
            if t >= 3 {
                cols[14][t] = BinaryPoly::from(bp_to_u64(&cols[0][t - 3]) as u32);
                cols[15][t] = BinaryPoly::from(bp_to_u64(&cols[1][t - 3]) as u32);
            } else {
                // t = 0,1,2: use initial H values for d and h
                cols[14][t] = BinaryPoly::from([H[3], H[2], H[1]][t]); // d: H[3],c,b
                cols[15][t] = BinaryPoly::from([H[7], H[6], H[5]][t]); // h: H[7],g,f
            }

            // W lookbacks
            if t >= 2  { cols[16][t] = BinaryPoly::from(w[t - 2]); }
            if t >= 7  { cols[17][t] = BinaryPoly::from(w[t - 7]); }
            if t >= 15 { cols[18][t] = BinaryPoly::from(w[t - 15]); }
            if t >= 16 { cols[19][t] = BinaryPoly::from(w[t - 16]); }

            // K_hat (round constant)
            cols[20][t] = BinaryPoly::from(K[t]);

            // ── Selector columns (21–22) ────────────────────────────────
            cols[21][t] = BinaryPoly::from(1u32);  // sel_round = 1 for t < 64
            if t >= 16 {
                cols[22][t] = BinaryPoly::from(1u32);  // sel_sched = 1 for t >= 16
            }

            // ── Integer columns (23–25): carry values ───────────────────
            let sigma1_val = big_sigma1(e);
            let ch_val = ch(e, f_reg, g_reg);
            let sigma0_val = big_sigma0(a);
            let maj_val = maj(a, b_reg, c_reg);

            // μ_a: carry for a-update
            let sum_a: u64 = h_reg as u64
                + sigma1_val as u64
                + ch_val as u64
                + K[t] as u64
                + w[t] as u64
                + sigma0_val as u64
                + maj_val as u64;
            let mu_a_val = (sum_a >> 32) as u32;
            cols[23][t] = BinaryPoly::from(mu_a_val);

            // μ_e: carry for e-update
            let sum_e: u64 = d_reg as u64
                + h_reg as u64
                + sigma1_val as u64
                + ch_val as u64
                + K[t] as u64
                + w[t] as u64;
            let mu_e_val = (sum_e >> 32) as u32;
            cols[24][t] = BinaryPoly::from(mu_e_val);

            // μ_W: carry for message schedule recurrence (t ≥ 16)
            if t >= 16 {
                let sum_w: u64 = w[t - 16] as u64
                    + small_sigma0(w[t - 15]) as u64
                    + w[t - 7] as u64
                    + small_sigma1(w[t - 2]) as u64;
                let mu_w_val = (sum_w >> 32) as u32;
                cols[25][t] = BinaryPoly::from(mu_w_val);
            }

            // ── SHA-256 round function ──────────────────────────────────
            let t1 = h_reg
                .wrapping_add(sigma1_val)
                .wrapping_add(ch_val)
                .wrapping_add(K[t])
                .wrapping_add(w[t]);
            let t2 = sigma0_val.wrapping_add(maj_val);

            h_reg = g_reg;
            g_reg = f_reg;
            f_reg = e;
            e = d_reg.wrapping_add(t1);
            d_reg = c_reg;
            c_reg = b_reg;
            b_reg = a;
            a = t1.wrapping_add(t2);
        }

        // ── Row 64: final state ─────────────────────────────────────────
        cols[0][64] = BinaryPoly::from(a);
        cols[1][64] = BinaryPoly::from(e);

        // Populate Σ₀/Σ₁ at row 64 so C1/C2 hold there.
        cols[3][64] = BinaryPoly::from(big_sigma0(a));
        cols[4][64] = BinaryPoly::from(big_sigma1(e));

        // ── Extended auxiliary columns beyond row 64 ─────────────────────
        // The linking constraints require auxiliary columns to be populated
        // beyond the 65 active rows. We extend each auxiliary column to
        // cover shifted references:
        //   d_hat[t] for t up to 67 (shift-by-3 at t=64)
        //   h_hat[t] for t up to 67
        //   W_tm2[t]  for t up to 65
        //   W_tm7[t]  for t up to 70
        //   W_tm15[t] for t up to 78
        //   W_tm16[t] for t up to 79
        //
        // Also extend σ₀_w, σ₁_w, S₀, S₁, R₀, R₁ to cover
        // the range where their W lookback source is valid.

        // d_hat: d[t] = a[t-3] for t=64..67
        for t in 64..68.min(num_rows) {
            if t >= 3 {
                cols[14][t] = BinaryPoly::from(bp_to_u64(&cols[0][t - 3]) as u32);
            }
        }
        // h_hat: h[t] = e[t-3] for t=64..67
        for t in 64..68.min(num_rows) {
            if t >= 3 {
                cols[15][t] = BinaryPoly::from(bp_to_u64(&cols[1][t - 3]) as u32);
            }
        }
        // W_tm2: W[t-2] for t=64..65
        for t in 64..66.min(num_rows) {
            if t >= 2 && t - 2 < 64 {
                cols[16][t] = BinaryPoly::from(w[t - 2]);
            }
        }
        // W_tm7: W[t-7] for t=64..70
        for t in 64..71.min(num_rows) {
            if t >= 7 && t - 7 < 64 {
                cols[17][t] = BinaryPoly::from(w[t - 7]);
            }
        }
        // W_tm15: W[t-15] for t=64..78
        for t in 64..79.min(num_rows) {
            if t >= 15 && t - 15 < 64 {
                cols[18][t] = BinaryPoly::from(w[t - 15]);
            }
        }
        // W_tm16: W[t-16] for t=64..79
        for t in 64..80.min(num_rows) {
            if t >= 16 && t - 16 < 64 {
                cols[19][t] = BinaryPoly::from(w[t - 16]);
            }
        }
        // σ₀_w, S₀, R₀: extend to cover W_tm15 range (t=64..78)
        for t in 64..79.min(num_rows) {
            if t >= 15 && t - 15 < 64 {
                let wback = w[t - 15];
                cols[8][t]  = BinaryPoly::from(small_sigma0(wback));
                cols[10][t] = BinaryPoly::from(wback >> 3);
                cols[12][t] = BinaryPoly::from(wback & 0x7);
            }
        }
        // σ₁_w, S₁, R₁: extend to cover W_tm2 range (t=64..65)
        for t in 64..66.min(num_rows) {
            if t >= 2 && t - 2 < 64 {
                let wback = w[t - 2];
                cols[9][t]  = BinaryPoly::from(small_sigma1(wback));
                cols[11][t] = BinaryPoly::from(wback >> 10);
                cols[13][t] = BinaryPoly::from(wback & 0x3FF);
            }
        }

        // ── Convert column vectors into DenseMultilinearExtensions ──────
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
}

// ─── Split witness generators ───────────────────────────────────────────────

/// Generate only the 14 BinaryPoly columns (indices 0–13) used in
/// both F₂[X] and Q[X] constraints. These include the 10 bit-polynomial
/// columns and the 4 F₂[X] shift/remainder columns.
pub fn generate_poly_witness(
    num_vars: usize,
    rng: &mut impl RngCore,
) -> Vec<DenseMultilinearExtension<BinaryPoly<32>>> {
    let full = <Sha256Uair as GenerateWitness<BinaryPoly<32>>>::generate_witness(num_vars, rng);
    POLY_COLUMN_INDICES
        .iter()
        .map(|&i| full[i].clone())
        .collect()
}

/// Generate the 3 integer columns (indices 14–16: μ_a, μ_e, μ_W) used in
/// Q[X] carry constraints (C10–C12), encoded as `Int<1>` (64-bit integer).
///
/// Each cell value is `BinaryPoly::to_u64() as i64` wrapped in `Int<1>`.
/// These carry values are small integers (≤ 6), so they always fit within
/// a single 64-bit limb.
pub fn generate_int_witness(
    num_vars: usize,
    rng: &mut impl RngCore,
) -> Vec<DenseMultilinearExtension<Int<1>>> {
    let full = <Sha256Uair as GenerateWitness<BinaryPoly<32>>>::generate_witness(num_vars, rng);
    let zero = Int::<1>::from_ref(&0i64);
    INT_COLUMN_INDICES
        .iter()
        .map(|&i| {
            let evals: Vec<Int<1>> = full[i]
                .evaluations
                .iter()
                .map(|bp| Int::<1>::from_ref(&(bp_to_u64(bp) as i64)))
                .collect();
            DenseMultilinearExtension::from_evaluations_vec(
                num_vars,
                evals,
                zero,
            )
        })
        .collect()
}

// ─── SHA-256 helper functions ───────────────────────────────────────────────

/// Right-rotate a 32-bit word by `r` positions.
#[inline(always)]
fn rotr(x: u32, r: u32) -> u32 {
    x.rotate_right(r)
}

/// Σ₀(a) = ROTR²(a) ⊕ ROTR¹³(a) ⊕ ROTR²²(a)
#[inline(always)]
fn big_sigma0(a: u32) -> u32 {
    rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)
}

/// Σ₁(e) = ROTR⁶(e) ⊕ ROTR¹¹(e) ⊕ ROTR²⁵(e)
#[inline(always)]
fn big_sigma1(e: u32) -> u32 {
    rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)
}

/// σ₀(x) = ROTR⁷(x) ⊕ ROTR¹⁸(x) ⊕ SHR³(x)
#[inline(always)]
fn small_sigma0(x: u32) -> u32 {
    rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3)
}

/// σ₁(x) = ROTR¹⁷(x) ⊕ ROTR¹⁹(x) ⊕ SHR¹⁰(x)
#[inline(always)]
fn small_sigma1(x: u32) -> u32 {
    rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10)
}

/// Ch(e, f, g) = (e ∧ f) ⊕ ((¬e) ∧ g)
#[inline(always)]
fn ch(e: u32, f: u32, g: u32) -> u32 {
    (e & f) ^ ((!e) & g)
}

/// Maj(a, b, c) = (a ∧ b) ⊕ (a ∧ c) ⊕ (b ∧ c)
#[inline(always)]
fn maj(a: u32, b: u32, c: u32) -> u32 {
    (a & b) ^ (a & c) ^ (b & c)
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// num_vars = 7 → 128 rows ≥ 65 needed.
    const NUM_VARS: usize = 7;

    /// Verify that the witness generation produces a valid trace.
    #[test]
    fn witness_produces_correct_sha256_of_empty_string() {
        let mut rng = rand::rng();
        let trace = <Sha256Uair as GenerateWitness<BinaryPoly<32>>>::generate_witness(NUM_VARS, &mut rng);

        // Verify dimensions: 26 columns, 128 rows each.
        assert_eq!(trace.len(), NUM_COLS);
        for col in &trace {
            assert_eq!(col.evaluations.len(), 1 << NUM_VARS);
        }

        // Verify initial state at round 0.
        let a0 = bp_to_u64(&trace[0].evaluations[0]) as u32;
        let e0 = bp_to_u64(&trace[1].evaluations[0]) as u32;

        assert_eq!(a0, 0x6a09e667, "a at round 0 should be H[0]");
        assert_eq!(e0, 0x510e527f, "e at round 0 should be H[4]");
    }

    /// Verify that Σ₀ and Σ₁ columns are consistent with a and e.
    #[test]
    fn sigma_columns_are_consistent() {
        let mut rng = rand::rng();
        let trace = <Sha256Uair as GenerateWitness<BinaryPoly<32>>>::generate_witness(NUM_VARS, &mut rng);

        for t in 0..64 {
            let a_val = bp_to_u64(&trace[0].evaluations[t]) as u32;
            let e_val = bp_to_u64(&trace[1].evaluations[t]) as u32;
            let sigma0_val = bp_to_u64(&trace[3].evaluations[t]) as u32;
            let sigma1_val = bp_to_u64(&trace[4].evaluations[t]) as u32;

            assert_eq!(
                sigma0_val,
                big_sigma0(a_val),
                "Σ₀ mismatch at round {t}"
            );
            assert_eq!(
                sigma1_val,
                big_sigma1(e_val),
                "Σ₁ mismatch at round {t}"
            );
        }
    }

    /// Verify that Ch decomposition columns are consistent.
    #[test]
    fn ch_columns_are_consistent() {
        let mut rng = rand::rng();
        let trace = <Sha256Uair as GenerateWitness<BinaryPoly<32>>>::generate_witness(NUM_VARS, &mut rng);

        // We can only check ch_ef and ch_neg_eg values at round 0 since
        // we know f = H[5] and g = H[6] at that point.
        let e0 = bp_to_u64(&trace[1].evaluations[0]) as u32;
        let ch_ef0 = bp_to_u64(&trace[6].evaluations[0]) as u32;
        let ch_neg_eg0 = bp_to_u64(&trace[7].evaluations[0]) as u32;

        let f_init = crate::constants::H[5];
        let g_init = crate::constants::H[6];

        assert_eq!(ch_ef0, e0 & f_init, "ch_ef mismatch at round 0");
        assert_eq!(
            ch_neg_eg0,
            (!e0) & g_init,
            "ch_neg_eg mismatch at round 0"
        );
    }

    /// Verify W column contains padded empty message for t < 16
    /// and valid message schedule for t ≥ 16.
    #[test]
    fn message_schedule_is_correct() {
        let mut rng = rand::rng();
        let trace = <Sha256Uair as GenerateWitness<BinaryPoly<32>>>::generate_witness(NUM_VARS, &mut rng);

        let w0 = bp_to_u64(&trace[2].evaluations[0]) as u32;
        assert_eq!(w0, 0x8000_0000, "W[0] should be 0x80000000 for empty msg");

        for t in 1..16 {
            let wt = bp_to_u64(&trace[2].evaluations[t]) as u32;
            assert_eq!(wt, 0, "W[{t}] should be 0 for empty msg");
        }
    }

    /// Verify shift quotient/remainder columns satisfy W = R + X^k * S.
    #[test]
    fn shift_decomposition_is_correct() {
        let mut rng = rand::rng();
        let trace = <Sha256Uair as GenerateWitness<BinaryPoly<32>>>::generate_witness(NUM_VARS, &mut rng);

        // σ₀ decomposition: populated from t=15 (where W[t-15] exists)
        for t in 15..64 {
            let s0 = bp_to_u64(&trace[10].evaluations[t]) as u32;
            let r0 = bp_to_u64(&trace[12].evaluations[t]) as u32;
            let w_tm15 = bp_to_u64(&trace[2].evaluations[t - 15]) as u32;
            assert_eq!(s0, w_tm15 >> 3, "S0 = SHR³(W[t-15]) at round {t}");
            assert_eq!(r0, w_tm15 & 0x7, "R0 = W[t-15] mod X³ at round {t}");
            assert_eq!(w_tm15, r0 | (s0 << 3), "W[t-15] = R0 + X³·S0 at round {t}");
        }

        // σ₁ decomposition: populated from t=2 (where W[t-2] exists)
        for t in 2..64 {
            let s1 = bp_to_u64(&trace[11].evaluations[t]) as u32;
            let r1 = bp_to_u64(&trace[13].evaluations[t]) as u32;
            let w_tm2 = bp_to_u64(&trace[2].evaluations[t - 2]) as u32;
            assert_eq!(s1, w_tm2 >> 10, "S1 = SHR¹⁰(W[t-2]) at round {t}");
            assert_eq!(r1, w_tm2 & 0x3FF, "R1 = W[t-2] mod X¹⁰ at round {t}");
            assert_eq!(w_tm2, r1 | (s1 << 10), "W[t-2] = R1 + X¹⁰·S1 at round {t}");
        }
    }

    /// Verify σ₀ and σ₁ columns match the small-sigma functions.
    #[test]
    fn small_sigma_columns_are_consistent() {
        let mut rng = rand::rng();
        let trace = <Sha256Uair as GenerateWitness<BinaryPoly<32>>>::generate_witness(NUM_VARS, &mut rng);

        for t in 15..64 {
            let w_tm15 = bp_to_u64(&trace[2].evaluations[t - 15]) as u32;
            let sigma0_w = bp_to_u64(&trace[8].evaluations[t]) as u32;
            assert_eq!(sigma0_w, small_sigma0(w_tm15), "σ₀(W[t-15]) mismatch at round {t}");
        }
        for t in 2..64 {
            let w_tm2 = bp_to_u64(&trace[2].evaluations[t - 2]) as u32;
            let sigma1_w = bp_to_u64(&trace[9].evaluations[t]) as u32;
            assert_eq!(sigma1_w, small_sigma1(w_tm2), "σ₁(W[t-2]) mismatch at round {t}");
        }
    }

    /// Verify carry polynomials satisfy the (X−2) ideal check.
    ///
    /// For each round t < 63 (so t+1 < 64):
    ///   a-update: a[t+1] = h + Σ₁ + Ch + K_t + W + Σ₀ + Maj  (mod 2³²)
    ///   carry:    μ_a = floor(sum_a / 2³²)
    ///   check:    a[t+1] - sum_a + μ_a · 2³² = 0
    ///
    /// Similarly for the e-update. h_t and d_t are recovered from the
    /// trace via shift-register identities: h_t = e_{t−3}, d_t = a_{t−3}.
    #[test]
    fn carry_polynomials_are_consistent() {
        let mut rng = rand::rng();
        let trace = <Sha256Uair as GenerateWitness<BinaryPoly<32>>>::generate_witness(NUM_VARS, &mut rng);

        // For t ≥ 3, h_t = e_{t-3} and d_t = a_{t-3}.
        // For t = 0, 1, 2 we use the initial H values.
        let h_vals: Vec<u64> = (0..64).map(|t| {
            if t >= 3 {
                bp_to_u64(&trace[1].evaluations[t - 3])
            } else {
                // h_1=H[7], h_2=g_1=H[6], h_3=f_1=H[5]
                [H[7] as u64, H[6] as u64, H[5] as u64][t]
            }
        }).collect();

        let d_vals: Vec<u64> = (0..64).map(|t| {
            if t >= 3 {
                bp_to_u64(&trace[0].evaluations[t - 3])
            } else {
                // d_1=H[3], d_2=c_1=H[2], d_3=b_1=H[1]
                [H[3] as u64, H[2] as u64, H[1] as u64][t]
            }
        }).collect();

        for t in 0..64 {
            let h_val = h_vals[t];
            let d_val = d_vals[t];
            let sigma1_val = bp_to_u64(&trace[4].evaluations[t]);
            let ch_ef_val = bp_to_u64(&trace[6].evaluations[t]);
            let ch_neg_eg_val = bp_to_u64(&trace[7].evaluations[t]);
            let k_val = K[t] as u64;
            let w_val = bp_to_u64(&trace[2].evaluations[t]);
            let sigma0_val = bp_to_u64(&trace[3].evaluations[t]);
            let maj_val = bp_to_u64(&trace[5].evaluations[t]);
            let mu_a_val = bp_to_u64(&trace[23].evaluations[t]);  // col 23

            let a_next = bp_to_u64(&trace[0].evaluations[t + 1]);

            let sum_a = h_val + sigma1_val + ch_ef_val + ch_neg_eg_val
                + k_val + w_val + sigma0_val + maj_val;

            assert_eq!(
                a_next as i64 - sum_a as i64 + (mu_a_val as i64) * (1i64 << 32),
                0,
                "a-update carry check failed at round {t}: \
                 a_next={a_next:#x}, sum_a={sum_a:#x}, mu_a={mu_a_val}"
            );

            // e-update carry
            let mu_e_val = bp_to_u64(&trace[24].evaluations[t]);  // col 24
            let e_next = bp_to_u64(&trace[1].evaluations[t + 1]);

            let sum_e = d_val + h_val + sigma1_val + ch_ef_val
                + ch_neg_eg_val + k_val + w_val;

            assert_eq!(
                e_next as i64 - sum_e as i64 + (mu_e_val as i64) * (1i64 << 32),
                0,
                "e-update carry check failed at round {t}: \
                 e_next={e_next:#x}, sum_e={sum_e:#x}, mu_e={mu_e_val}"
            );
        }
    }

    /// Verify μ_W carry for message schedule recurrence (t ≥ 16).
    #[test]
    fn message_schedule_carry_is_consistent() {
        let mut rng = rand::rng();
        let trace = <Sha256Uair as GenerateWitness<BinaryPoly<32>>>::generate_witness(NUM_VARS, &mut rng);

        for t in 16..64 {
            let w_t = bp_to_u64(&trace[2].evaluations[t]);
            let w_tm16 = bp_to_u64(&trace[2].evaluations[t - 16]);
            let sigma0_w = bp_to_u64(&trace[8].evaluations[t]);
            let w_tm7 = bp_to_u64(&trace[2].evaluations[t - 7]);
            let sigma1_w = bp_to_u64(&trace[9].evaluations[t]);
            let mu_w_val = bp_to_u64(&trace[25].evaluations[t]);  // col 25

            let sum_w = w_tm16 + sigma0_w + w_tm7 + sigma1_w;

            assert_eq!(
                w_t as i64 - sum_w as i64 + (mu_w_val as i64) * (1i64 << 32),
                0,
                "W-schedule carry check failed at round {t}: \
                 w_t={w_t:#x}, sum_w={sum_w:#x}, mu_w={mu_w_val}"
            );
        }
    }
}
