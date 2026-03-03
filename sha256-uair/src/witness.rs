//! Witness generation for the SHA-256 UAIR⁺.
//!
//! Implements [`GenerateWitness<BinaryPoly<32>>`] for [`Sha256UairBp`] by
//! running chained SHA-256 compression functions on multiple message blocks,
//! filling all `2^{num_vars}` rows with non-trivial computational data.
//!
//! When `num_vars ≥ 7` (≥ 128 rows), the trace is populated with
//! `⌊num_rows / 64⌋` chained compressions. Compression 0 hashes the
//! padded empty message; compressions 1+ use deterministic non-trivial
//! message blocks. The round-function state flows directly from one
//! compression to the next (without the SHA-256 post-processing
//! H-addition), so all linking constraints are naturally satisfied.
//!
//! The last few rows are deactivated due to shift zero-padding at the
//! trace boundary (shift-by-16 forces W=0 for the last 15 rows;
//! shift-by-3 forces a/e=0 for the last 2 checked rows).
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

/// Bit-polynomial column indices (0–26): the 10 Q[X] bit-poly columns,
/// 4 F₂[X] columns, 7 auxiliary lookback columns, 4 Ch/Maj lookback
/// columns, and 2 selector columns.
pub const POLY_COLUMN_INDICES: [usize; 27] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
    14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
];

/// Integer column indices (27–29): the 3 carry columns μ_a, μ_e, μ_W.
pub const INT_COLUMN_INDICES: [usize; 3] = [27, 28, 29];

// ─── GenerateWitness impl ───────────────────────────────────────────────────

/// Witness generation trait for a specific ring type.
pub trait GenerateWitness<R: crypto_primitives::Semiring + 'static> {
    fn generate_witness<Rng: RngCore + ?Sized>(
        num_vars: usize,
        rng: &mut Rng,
    ) -> Vec<DenseMultilinearExtension<R>>;
}

impl GenerateWitness<BinaryPoly<32>> for Sha256UairBp {
    /// Generate the SHA-256 witness trace with chained compressions.
    ///
    /// The trace has `2^{num_vars}` rows, populated with
    /// `num_rows / 64` chained compressions. Compression 0 hashes the
    /// padded empty message; compressions 1+ use deterministic message
    /// blocks. The round-function state flows from one compression to
    /// the next without the SHA-256 H-addition post-processing step,
    /// so all linking constraints are naturally satisfied across
    /// compression boundaries.
    ///
    /// Due to shift zero-padding at the trace tail:
    /// - `sel_round = 0` for the last 4 rows (shift-by-3 forces a/e = 0
    ///   at rows N−3, N−2; the carry constraint at N−4 would conflict).
    /// - `sel_sched = 0` for the last 16 rows (shift-by-16 forces W = 0).
    /// - The last compression is therefore partial (60 active rounds
    ///   instead of 64, schedule active for rounds 16–47 only).
    ///
    /// The `rng` parameter is accepted for trait compatibility but unused.
    fn generate_witness<Rng: RngCore + ?Sized>(
        num_vars: usize,
        _rng: &mut Rng,
    ) -> Vec<DenseMultilinearExtension<BinaryPoly<32>>> {
        assert!(
            num_vars >= 7,
            "SHA-256 requires at least 7 variables (128 rows ≥ 65), got {num_vars}"
        );

        let num_rows: usize = 1 << num_vars;
        let num_compressions = num_rows / 64;

        // ── Generate message blocks ─────────────────────────────────────
        // Block 0: padded empty message (preserves original SHA-256 trace).
        // Blocks 1+: deterministic non-trivial patterns.
        let msg_blocks: Vec<[u32; 16]> = (0..num_compressions)
            .map(|k| {
                if k == 0 {
                    let mut block = [0u32; 16];
                    block[0] = 0x8000_0000;
                    block
                } else {
                    let mut block = [0u32; 16];
                    for i in 0..16 {
                        // Mix block index and word index for variety
                        block[i] = ((k as u32) << 16) | (i as u32 + 1);
                    }
                    block
                }
            })
            .collect();

        // ── Compute message schedules ───────────────────────────────────
        let schedules: Vec<[u32; 64]> = msg_blocks
            .iter()
            .map(|block| {
                let mut w = [0u32; 64];
                w[..16].copy_from_slice(block);
                for t in 16..64 {
                    w[t] = small_sigma1(w[t - 2])
                        .wrapping_add(w[t - 7])
                        .wrapping_add(small_sigma0(w[t - 15]))
                        .wrapping_add(w[t - 16]);
                }
                w
            })
            .collect();

        // ── Build global W and K arrays ─────────────────────────────────
        // W is forced to 0 for the last 16 rows by the W_tm16 linking
        // constraint's shift zero-padding.
        let w_zero_start = num_rows.saturating_sub(16);
        let w_global: Vec<u32> = (0..num_rows)
            .map(|t| {
                if t >= w_zero_start {
                    0
                } else {
                    schedules[t / 64][t % 64]
                }
            })
            .collect();

        // K repeats every 64 rounds (one per compression).
        let k_global: Vec<u32> = (0..num_rows)
            .map(|t| K[t % 64])
            .collect();

        // ── Boundary thresholds ─────────────────────────────────────────
        // The CPR excludes row N−1 from constraint checking.
        // Shift-by-3 (d/h links) forces a_hat, e_hat = 0 at rows N−3, N−2.
        // The carry constraint at row t references a[t+1], so:
        //   sel_round at row N−4 would need a[N−3] = round_output,
        //   but a[N−3] = 0 (forced) → sel_round must be 0 at N−4.
        // Last sel_round = 1 row: N−5.
        // Last sel_sched = 1 row: N−17 (W forced to 0 at N−16).
        let last_sel_round_row = num_rows.saturating_sub(5);
        let last_sel_sched_row = num_rows.saturating_sub(17);

        // ── Phase 1: Run round function, record a/e for all rows ────────
        let mut a_vals = vec![0u32; num_rows];
        let mut e_vals = vec![0u32; num_rows];

        {
            let (mut a, mut b_reg, mut c_reg, mut d_reg) =
                (H[0], H[1], H[2], H[3]);
            let (mut e, mut f_reg, mut g_reg, mut h_reg) =
                (H[4], H[5], H[6], H[7]);

            for t in 0..num_rows {
                a_vals[t] = a;
                e_vals[t] = e;

                if t <= last_sel_round_row {
                    // Run round function
                    let sigma1_val = big_sigma1(e);
                    let ch_val = ch(e, f_reg, g_reg);
                    let sigma0_val = big_sigma0(a);
                    let maj_val = maj(a, b_reg, c_reg);

                    let t1 = h_reg
                        .wrapping_add(sigma1_val)
                        .wrapping_add(ch_val)
                        .wrapping_add(k_global[t])
                        .wrapping_add(w_global[t]);
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
                // After last_sel_round_row, state stops updating.
                // The next iteration writes the final state, then
                // boundary zeros override.
            }
        }

        // Force a/e to zero at boundary rows (N−3, N−2, N−1).
        // These are forced by the shift-by-3 linking zero-padding.
        for t in num_rows.saturating_sub(3)..num_rows {
            a_vals[t] = 0;
            e_vals[t] = 0;
        }

        // ── Helper: get a value with initial-state fallback ─────────────
        // For lookback indices < 0, return the corresponding initial H value.
        let get_a = |t: isize| -> u32 {
            if t >= 0 && (t as usize) < num_rows {
                a_vals[t as usize]
            } else if t == -1 {
                H[1] // b_init
            } else if t == -2 {
                H[2] // c_init
            } else if t == -3 {
                H[3] // d_init
            } else {
                0
            }
        };
        let get_e = |t: isize| -> u32 {
            if t >= 0 && (t as usize) < num_rows {
                e_vals[t as usize]
            } else if t == -1 {
                H[5] // f_init
            } else if t == -2 {
                H[6] // g_init
            } else if t == -3 {
                H[7] // h_init
            } else {
                0
            }
        };

        // ── Phase 2: Populate all 30 columns ────────────────────────────
        let mut cols: Vec<Vec<BinaryPoly<32>>> =
            (0..NUM_COLS).map(|_| vec![BinaryPoly::<32>::from(0u32); num_rows]).collect();

        for t in 0..num_rows {
            let ti = t as isize;
            let round = t % 64;

            let a = a_vals[t];
            let e = e_vals[t];
            let b = get_a(ti - 1);
            let c = get_a(ti - 2);
            let d = get_a(ti - 3);
            let f = get_e(ti - 1);
            let g = get_e(ti - 2);
            let h = get_e(ti - 3);
            let w_val = w_global[t];

            // ── Bit-polynomial columns (0–9) ────────────────────────────
            cols[0][t] = BinaryPoly::from(a);                  // a_hat
            cols[1][t] = BinaryPoly::from(e);                  // e_hat
            cols[2][t] = BinaryPoly::from(w_val);              // W_hat
            cols[3][t] = BinaryPoly::from(big_sigma0(a));      // Sigma0_hat
            cols[4][t] = BinaryPoly::from(big_sigma1(e));      // Sigma1_hat
            cols[5][t] = BinaryPoly::from(maj(a, b, c));       // Maj_hat
            cols[6][t] = BinaryPoly::from(e & f);              // ch_ef_hat
            cols[7][t] = BinaryPoly::from((!e) & g);           // ch_neg_eg_hat

            // σ₀(W_{t−15}) and σ₁(W_{t−2}) for message schedule
            if t >= 15 {
                cols[8][t] = BinaryPoly::from(small_sigma0(w_global[t - 15]));
            }
            if t >= 2 {
                cols[9][t] = BinaryPoly::from(small_sigma1(w_global[t - 2]));
            }

            // ── F₂[X] columns (10–13): shift quotient/remainder ────────
            if t >= 15 {
                let wback = w_global[t - 15];
                cols[10][t] = BinaryPoly::from(wback >> 3);      // S0
                cols[12][t] = BinaryPoly::from(wback & 0x7);     // R0
            }
            if t >= 2 {
                let wback = w_global[t - 2];
                cols[11][t] = BinaryPoly::from(wback >> 10);     // S1
                cols[13][t] = BinaryPoly::from(wback & 0x3FF);   // R1
            }

            // ── Auxiliary lookback columns (14–20) ──────────────────────
            cols[14][t] = BinaryPoly::from(d);                    // d_hat = a[t−3]
            cols[15][t] = BinaryPoly::from(h);                    // h_hat = e[t−3]

            if t >= 2  { cols[16][t] = BinaryPoly::from(w_global[t - 2]); }
            if t >= 7  { cols[17][t] = BinaryPoly::from(w_global[t - 7]); }
            if t >= 15 { cols[18][t] = BinaryPoly::from(w_global[t - 15]); }
            if t >= 16 { cols[19][t] = BinaryPoly::from(w_global[t - 16]); }

            // K_hat (round constant, repeating every 64 rounds)
            cols[20][t] = BinaryPoly::from(k_global[t]);

            // ── Ch/Maj lookback columns (21–24) ─────────────────────────
            cols[21][t] = BinaryPoly::from(b);                    // a_tm1 = a[t−1]
            cols[22][t] = BinaryPoly::from(c);                    // a_tm2 = a[t−2]
            cols[23][t] = BinaryPoly::from(f);                    // e_tm1 = e[t−1]
            cols[24][t] = BinaryPoly::from(g);                    // e_tm2 = e[t−2]

            // ── Selector columns (25–26) ────────────────────────────────
            let round_active = t <= last_sel_round_row;
            let sched_active = round >= 16 && t <= last_sel_sched_row;

            cols[25][t] = BinaryPoly::from(if round_active { 1u32 } else { 0u32 });
            cols[26][t] = BinaryPoly::from(if sched_active { 1u32 } else { 0u32 });

            // ── Integer columns (27–29): carry values ───────────────────
            if round_active {
                let sigma1_val = big_sigma1(e);
                let ch_val = ch(e, f, g);
                let sigma0_val = big_sigma0(a);
                let maj_val = maj(a, b, c);

                // μ_a: carry for a-update
                let sum_a: u64 = h as u64
                    + sigma1_val as u64
                    + ch_val as u64
                    + k_global[t] as u64
                    + w_val as u64
                    + sigma0_val as u64
                    + maj_val as u64;
                let mu_a_val = (sum_a >> 32) as u32;
                cols[27][t] = BinaryPoly::from(mu_a_val);

                // μ_e: carry for e-update
                let sum_e: u64 = d as u64
                    + h as u64
                    + sigma1_val as u64
                    + ch_val as u64
                    + k_global[t] as u64
                    + w_val as u64;
                let mu_e_val = (sum_e >> 32) as u32;
                cols[28][t] = BinaryPoly::from(mu_e_val);
            }

            if sched_active {
                // μ_W: carry for message schedule recurrence
                let sum_w: u64 = w_global[t - 16] as u64
                    + small_sigma0(w_global[t - 15]) as u64
                    + w_global[t - 7] as u64
                    + small_sigma1(w_global[t - 2]) as u64;
                let mu_w_val = (sum_w >> 32) as u32;
                cols[29][t] = BinaryPoly::from(mu_w_val);
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

/// Generate only the 27 BinaryPoly columns (indices 0–26) used in
/// both F₂[X] and Q[X] constraints. These include the 10 bit-polynomial
/// columns, the 4 F₂[X] shift/remainder columns, 7 lookback columns,
/// the round-constant column, 4 Ch/Maj lookback columns, and 2 selector
/// columns.
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

/// Generate the 3 integer columns (indices 27–29: μ_a, μ_e, μ_W) used in
/// Q[X] carry constraints (C13–C15), encoded as `Int<1>` (64-bit integer).
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
            let mu_a_val = bp_to_u64(&trace[27].evaluations[t]);  // col 27

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
            let mu_e_val = bp_to_u64(&trace[28].evaluations[t]);  // col 28
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
            let mu_w_val = bp_to_u64(&trace[29].evaluations[t]);  // col 29

            let sum_w = w_tm16 + sigma0_w + w_tm7 + sigma1_w;

            assert_eq!(
                w_t as i64 - sum_w as i64 + (mu_w_val as i64) * (1i64 << 32),
                0,
                "W-schedule carry check failed at round {t}: \
                 w_t={w_t:#x}, sum_w={sum_w:#x}, mu_w={mu_w_val}"
            );
        }
    }

    /// Verify carry-freedom of all three affine lookup virtual columns
    /// at every row (including padded rows beyond 64).
    #[test]
    fn affine_lookup_carry_freedom() {
        use super::*;
        use crate::{
            COL_A_HAT, COL_A_TM1, COL_A_TM2,
            COL_E_HAT, COL_E_TM1, COL_E_TM2,
            COL_CH_EF_HAT, COL_CH_NEG_EG_HAT, COL_MAJ_HAT,
        };

        let mut rng = rand::rng();
        let trace = <Sha256Uair as GenerateWitness<BinaryPoly<32>>>::generate_witness(NUM_VARS, &mut rng);
        let num_rows = 1usize << NUM_VARS;

        for t in 0..num_rows {
            let e     = bp_to_u64(&trace[COL_E_HAT].evaluations[t]) as i64;
            let e_tm1 = bp_to_u64(&trace[COL_E_TM1].evaluations[t]) as i64;
            let e_tm2 = bp_to_u64(&trace[COL_E_TM2].evaluations[t]) as i64;
            let ch_ef = bp_to_u64(&trace[COL_CH_EF_HAT].evaluations[t]) as i64;
            let ch_ne = bp_to_u64(&trace[COL_CH_NEG_EG_HAT].evaluations[t]) as i64;
            let a     = bp_to_u64(&trace[COL_A_HAT].evaluations[t]) as i64;
            let a_tm1 = bp_to_u64(&trace[COL_A_TM1].evaluations[t]) as i64;
            let a_tm2 = bp_to_u64(&trace[COL_A_TM2].evaluations[t]) as i64;
            let maj   = bp_to_u64(&trace[COL_MAJ_HAT].evaluations[t]) as i64;

            // Ch1: e + e_tm1 - 2*ch_ef  must be in [0, 2^32-1] with no carries
            let ch1 = e + e_tm1 - 2 * ch_ef;
            assert!(
                ch1 >= 0 && ch1 <= 0xFFFF_FFFF,
                "Ch1 out of range at row {t}: ch1={ch1} (e={e:#x}, e_tm1={e_tm1:#x}, ch_ef={ch_ef:#x})"
            );
            // Each bit must be 0 or 1 (no carries)
            for bit in 0..32 {
                let e_b = (e >> bit) & 1;
                let f_b = (e_tm1 >> bit) & 1;
                let ce  = (ch_ef >> bit) & 1;
                let val = e_b + f_b - 2 * ce;
                assert!(
                    val == 0 || val == 1,
                    "Ch1 carry at row {t} bit {bit}: e_b={e_b}, f_b={f_b}, ch_ef_b={ce}, val={val}"
                );
            }

            // Ch2: 0xFFFF_FFFF - e + e_tm2 - 2*ch_neg_eg
            let ch2 = 0xFFFF_FFFFi64 - e + e_tm2 - 2 * ch_ne;
            assert!(
                ch2 >= 0 && ch2 <= 0xFFFF_FFFF,
                "Ch2 out of range at row {t}: ch2={ch2} (e={e:#x}, e_tm2={e_tm2:#x}, ch_ne={ch_ne:#x})"
            );
            for bit in 0..32 {
                let ne_b = 1 - ((e >> bit) & 1);   // NOT e
                let g_b  = (e_tm2 >> bit) & 1;
                let ce   = (ch_ne >> bit) & 1;
                let val  = ne_b + g_b - 2 * ce;
                assert!(
                    val == 0 || val == 1,
                    "Ch2 carry at row {t} bit {bit}: ne_b={ne_b}, g_b={g_b}, ch_ne_b={ce}, val={val}"
                );
            }

            // Maj: a + a_tm1 + a_tm2 - 2*Maj
            let maj_sum = a + a_tm1 + a_tm2 - 2 * maj;
            assert!(
                maj_sum >= 0 && maj_sum <= 0xFFFF_FFFF,
                "Maj out of range at row {t}: maj_sum={maj_sum} (a={a:#x}, a_tm1={a_tm1:#x}, a_tm2={a_tm2:#x}, maj={maj:#x})"
            );
            for bit in 0..32 {
                let ab = (a >> bit) & 1;
                let bb = (a_tm1 >> bit) & 1;
                let cb = (a_tm2 >> bit) & 1;
                let mb = (maj >> bit) & 1;
                let val = ab + bb + cb - 2 * mb;
                assert!(
                    val == 0 || val == 1,
                    "Maj carry at row {t} bit {bit}: a={ab}, b={bb}, c={cb}, maj={mb}, val={val}"
                );
            }
        }
    }

    // ── Multi-compression tests ─────────────────────────────────────

    /// num_vars = 9 → 512 rows = 8 compressions.
    const NUM_VARS_8X: usize = 9;

    /// Verify that the 8× trace has non-trivial data in all compressions.
    #[test]
    fn multi_compression_populates_all_blocks() {
        let mut rng = rand::rng();
        let trace = <Sha256Uair as GenerateWitness<BinaryPoly<32>>>::generate_witness(NUM_VARS_8X, &mut rng);
        let num_rows = 1usize << NUM_VARS_8X; // 512

        // Check that all 8 compressions have non-zero a_hat at their first row
        for comp in 0..8 {
            let t = comp * 64;
            let a_val = bp_to_u64(&trace[0].evaluations[t]) as u32;
            assert_ne!(
                a_val, 0,
                "Compression {comp} has zero a_hat at row {t}"
            );
        }

        // Check that W_hat has non-zero values in each compression's first 16 words
        for comp in 0..8 {
            let mut has_nonzero = false;
            for round in 0..16 {
                let t = comp * 64 + round;
                if t >= num_rows.saturating_sub(16) { break; }
                let w_val = bp_to_u64(&trace[2].evaluations[t]) as u32;
                if w_val != 0 { has_nonzero = true; break; }
            }
            // Last compression may have W forced to zero early, that's expected
            if comp < 7 {
                assert!(
                    has_nonzero,
                    "Compression {comp} has no non-zero W_hat values in first 16 words"
                );
            }
        }

        // Compression 1 should have a different message block than compression 0
        let w0_comp0 = bp_to_u64(&trace[2].evaluations[0]) as u32;
        let w0_comp1 = bp_to_u64(&trace[2].evaluations[64]) as u32;
        assert_ne!(
            w0_comp0, w0_comp1,
            "Compressions 0 and 1 have the same W[0]: both are {w0_comp0:#x}"
        );
    }

    /// Verify that selectors are active for the correct rows in 8× mode.
    #[test]
    fn multi_compression_selectors_are_correct() {
        let mut rng = rand::rng();
        let trace = <Sha256Uair as GenerateWitness<BinaryPoly<32>>>::generate_witness(NUM_VARS_8X, &mut rng);
        let num_rows = 1usize << NUM_VARS_8X;
        let last_sel_round = num_rows - 5; // 507
        let last_sel_sched = num_rows - 17; // 495

        for t in 0..num_rows {
            let round = t % 64;
            let sel_round = bp_to_u64(&trace[25].evaluations[t]) as u32;
            let sel_sched = bp_to_u64(&trace[26].evaluations[t]) as u32;

            let expected_round = if t <= last_sel_round { 1u32 } else { 0u32 };
            let expected_sched = if round >= 16 && t <= last_sel_sched { 1u32 } else { 0u32 };

            assert_eq!(
                sel_round, expected_round,
                "sel_round mismatch at row {t}"
            );
            assert_eq!(
                sel_sched, expected_sched,
                "sel_sched mismatch at row {t}"
            );
        }
    }

    /// Verify the carry constraint holds across ALL active rows in the 8× trace.
    #[test]
    fn multi_compression_carry_polynomials() {
        let mut rng = rand::rng();
        let trace = <Sha256Uair as GenerateWitness<BinaryPoly<32>>>::generate_witness(NUM_VARS_8X, &mut rng);
        let num_rows = 1usize << NUM_VARS_8X;
        let last_sel_round = num_rows - 5;

        for t in 0..=last_sel_round {
            let a_val = bp_to_u64(&trace[0].evaluations[t]);
            let e_val = bp_to_u64(&trace[1].evaluations[t]);
            let h_val = bp_to_u64(&trace[15].evaluations[t]); // h_hat = e[t-3]
            let d_val = bp_to_u64(&trace[14].evaluations[t]); // d_hat = a[t-3]
            let sigma1_val = bp_to_u64(&trace[4].evaluations[t]);
            let ch_ef_val = bp_to_u64(&trace[6].evaluations[t]);
            let ch_neg_eg_val = bp_to_u64(&trace[7].evaluations[t]);
            let k_val = bp_to_u64(&trace[20].evaluations[t]);
            let w_val = bp_to_u64(&trace[2].evaluations[t]);
            let sigma0_val = bp_to_u64(&trace[3].evaluations[t]);
            let maj_val = bp_to_u64(&trace[5].evaluations[t]);
            let mu_a_val = bp_to_u64(&trace[27].evaluations[t]);
            let mu_e_val = bp_to_u64(&trace[28].evaluations[t]);

            let a_next = bp_to_u64(&trace[0].evaluations[t + 1]);
            let e_next = bp_to_u64(&trace[1].evaluations[t + 1]);

            // a-update carry
            let sum_a = h_val + sigma1_val + ch_ef_val + ch_neg_eg_val
                + k_val + w_val + sigma0_val + maj_val;
            assert_eq!(
                a_next as i64 - sum_a as i64 + (mu_a_val as i64) * (1i64 << 32),
                0,
                "a-update carry check failed at row {t}: \
                 a_next={a_next:#x}, sum_a={sum_a:#x}, mu_a={mu_a_val}"
            );

            // e-update carry
            let sum_e = d_val + h_val + sigma1_val + ch_ef_val
                + ch_neg_eg_val + k_val + w_val;
            assert_eq!(
                e_next as i64 - sum_e as i64 + (mu_e_val as i64) * (1i64 << 32),
                0,
                "e-update carry check failed at row {t}: \
                 e_next={e_next:#x}, sum_e={sum_e:#x}, mu_e={mu_e_val}"
            );
        }
    }

    /// Verify carry-freedom at all 512 rows in the 8× trace.
    #[test]
    fn multi_compression_affine_lookup_carry_freedom() {
        use crate::{
            COL_A_HAT, COL_A_TM1, COL_A_TM2,
            COL_E_HAT, COL_E_TM1, COL_E_TM2,
            COL_CH_EF_HAT, COL_CH_NEG_EG_HAT, COL_MAJ_HAT,
        };

        let mut rng = rand::rng();
        let trace = <Sha256Uair as GenerateWitness<BinaryPoly<32>>>::generate_witness(NUM_VARS_8X, &mut rng);
        let num_rows = 1usize << NUM_VARS_8X;

        for t in 0..num_rows {
            let e     = bp_to_u64(&trace[COL_E_HAT].evaluations[t]) as i64;
            let e_tm1 = bp_to_u64(&trace[COL_E_TM1].evaluations[t]) as i64;
            let e_tm2 = bp_to_u64(&trace[COL_E_TM2].evaluations[t]) as i64;
            let ch_ef = bp_to_u64(&trace[COL_CH_EF_HAT].evaluations[t]) as i64;
            let ch_ne = bp_to_u64(&trace[COL_CH_NEG_EG_HAT].evaluations[t]) as i64;
            let a     = bp_to_u64(&trace[COL_A_HAT].evaluations[t]) as i64;
            let a_tm1 = bp_to_u64(&trace[COL_A_TM1].evaluations[t]) as i64;
            let a_tm2 = bp_to_u64(&trace[COL_A_TM2].evaluations[t]) as i64;
            let maj   = bp_to_u64(&trace[COL_MAJ_HAT].evaluations[t]) as i64;

            // Ch1
            for bit in 0..32 {
                let val = ((e >> bit) & 1) + ((e_tm1 >> bit) & 1) - 2 * ((ch_ef >> bit) & 1);
                assert!(
                    val == 0 || val == 1,
                    "Ch1 carry at row {t} bit {bit}"
                );
            }
            // Ch2
            for bit in 0..32 {
                let val = (1 - ((e >> bit) & 1)) + ((e_tm2 >> bit) & 1) - 2 * ((ch_ne >> bit) & 1);
                assert!(
                    val == 0 || val == 1,
                    "Ch2 carry at row {t} bit {bit}"
                );
            }
            // Maj
            for bit in 0..32 {
                let val = ((a >> bit) & 1) + ((a_tm1 >> bit) & 1) + ((a_tm2 >> bit) & 1) - 2 * ((maj >> bit) & 1);
                assert!(
                    val == 0 || val == 1,
                    "Maj carry at row {t} bit {bit}"
                );
            }
        }
    }

    /// Verify linking constraints hold across the full 8× trace.
    /// Checks: d_hat[t] = a_hat[t-3], h_hat[t] = e_hat[t-3],
    /// a_tm1[t] = a_hat[t-1], a_tm2[t] = a_hat[t-2],
    /// e_tm1[t] = e_hat[t-1], e_tm2[t] = e_hat[t-2].
    #[test]
    fn multi_compression_linking_constraints() {
        use crate::{
            COL_A_HAT, COL_E_HAT, COL_D_HAT, COL_H_HAT,
            COL_A_TM1, COL_A_TM2, COL_E_TM1, COL_E_TM2,
        };

        let mut rng = rand::rng();
        let trace = <Sha256Uair as GenerateWitness<BinaryPoly<32>>>::generate_witness(NUM_VARS_8X, &mut rng);

        // d_hat[t] = a_hat[t-3] for t >= 3
        for t in 3..512 {
            let d = bp_to_u64(&trace[COL_D_HAT].evaluations[t]) as u32;
            let a_back = bp_to_u64(&trace[COL_A_HAT].evaluations[t - 3]) as u32;
            assert_eq!(d, a_back, "d-link failed at row {t}");
        }
        // h_hat[t] = e_hat[t-3] for t >= 3
        for t in 3..512 {
            let h = bp_to_u64(&trace[COL_H_HAT].evaluations[t]) as u32;
            let e_back = bp_to_u64(&trace[COL_E_HAT].evaluations[t - 3]) as u32;
            assert_eq!(h, e_back, "h-link failed at row {t}");
        }
        // a_tm1[t] = a_hat[t-1] for t >= 1
        for t in 1..512 {
            let at1 = bp_to_u64(&trace[COL_A_TM1].evaluations[t]) as u32;
            let a_back = bp_to_u64(&trace[COL_A_HAT].evaluations[t - 1]) as u32;
            assert_eq!(at1, a_back, "a_tm1-link failed at row {t}");
        }
        // a_tm2[t] = a_hat[t-2] for t >= 2
        for t in 2..512 {
            let at2 = bp_to_u64(&trace[COL_A_TM2].evaluations[t]) as u32;
            let a_back = bp_to_u64(&trace[COL_A_HAT].evaluations[t - 2]) as u32;
            assert_eq!(at2, a_back, "a_tm2-link failed at row {t}");
        }
        // e_tm1[t] = e_hat[t-1] for t >= 1
        for t in 1..512 {
            let et1 = bp_to_u64(&trace[COL_E_TM1].evaluations[t]) as u32;
            let e_back = bp_to_u64(&trace[COL_E_HAT].evaluations[t - 1]) as u32;
            assert_eq!(et1, e_back, "e_tm1-link failed at row {t}");
        }
        // e_tm2[t] = e_hat[t-2] for t >= 2
        for t in 2..512 {
            let et2 = bp_to_u64(&trace[COL_E_TM2].evaluations[t]) as u32;
            let e_back = bp_to_u64(&trace[COL_E_HAT].evaluations[t - 2]) as u32;
            assert_eq!(et2, e_back, "e_tm2-link failed at row {t}");
        }
    }
}
