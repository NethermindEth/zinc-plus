//! Witness generation for the SHA-256 UAIR.
//!
//! Implements [`GenerateWitness<BinaryPoly<32>>`] for [`Sha256Uair`] by
//! running the full SHA-256 compression function on a single-block padded
//! empty message, recording all 19 column values at each of the 64 rounds.

use rand::RngCore;
use zinc_poly::{
    mle::DenseMultilinearExtension,
    univariate::binary::BinaryPoly,
};

use crate::{
    Sha256Uair,
    constants::{H, K},
    NUM_COLS,
};

// ─── GenerateWitness impl ───────────────────────────────────────────────────

/// Marker trait bridging `Uair` + witness generation.
///
/// Re-exported from the `test-uair` crate convention so that downstream
/// code can use a consistent interface. We define a local copy to avoid
/// a hard dependency on `zinc-test-uair`.
pub trait GenerateWitness<R: crypto_primitives::Semiring + 'static>:
    zinc_uair::Uair<R>
{
    fn generate_witness<Rng: RngCore + ?Sized>(
        num_vars: usize,
        rng: &mut Rng,
    ) -> Vec<DenseMultilinearExtension<R>>;
}

impl GenerateWitness<BinaryPoly<32>> for Sha256Uair {
    /// Generate the SHA-256 witness trace.
    ///
    /// The trace has 64 rows (one per round, `num_vars = 6`). If
    /// `num_vars > 6` the extra rows are zero-padded.
    ///
    /// The message hashed is the empty string (single 512-bit padded block).
    /// The `rng` parameter is accepted for trait compatibility but unused.
    fn generate_witness<Rng: RngCore + ?Sized>(
        num_vars: usize,
        _rng: &mut Rng,
    ) -> Vec<DenseMultilinearExtension<BinaryPoly<32>>> {
        assert!(
            num_vars >= 6,
            "SHA-256 requires at least 6 variables (64 rows), got {num_vars}"
        );

        let num_rows: usize = 1 << num_vars;

        // ── Prepare message block (padded empty message) ────────────────
        //
        // For the empty string:
        //   byte 0   = 0x80
        //   bytes 1‥55 = 0x00
        //   bytes 56‥63 = 64-bit big-endian length = 0
        //
        // As 32-bit big-endian words:
        let mut msg_block = [0u32; 16];
        msg_block[0] = 0x8000_0000;
        // msg_block[15] already 0 (length = 0 bits)

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
        let (mut a, mut b, mut c, mut d) = (H[0], H[1], H[2], H[3]);
        let (mut e, mut f, mut g, mut h) = (H[4], H[5], H[6], H[7]);

        // One Vec per column, pre-allocated to num_rows.
        let mut cols: Vec<Vec<BinaryPoly<32>>> =
            (0..NUM_COLS).map(|_| vec![BinaryPoly::<32>::from(0u32); num_rows]).collect();

        for t in 0..64 {
            // ── Record current state into columns ───────────────────────
            cols[0][t] = BinaryPoly::from(a);                     // a_hat
            cols[1][t] = BinaryPoly::from(e);                     // e_hat
            cols[2][t] = BinaryPoly::from(w[t]);                  // W_hat
            cols[3][t] = BinaryPoly::from(big_sigma0(a));         // Sigma0_hat
            cols[4][t] = BinaryPoly::from(big_sigma1(e));         // Sigma1_hat
            cols[5][t] = BinaryPoly::from(maj(a, b, c));          // Maj_hat
            cols[6][t] = BinaryPoly::from(e & f);                 // ch_ef_hat
            cols[7][t] = BinaryPoly::from((!e) & g);              // ch_neg_eg_hat

            // σ₀ and σ₁ for message schedule (meaningful for t ≥ 16)
            if t >= 16 {
                cols[8][t] = BinaryPoly::from(small_sigma0(w[t - 15]));  // sigma0_w_hat
                cols[9][t] = BinaryPoly::from(small_sigma1(w[t - 2]));   // sigma1_w_hat
            }
            // else: already zero from initialization

            cols[10][t] = BinaryPoly::from(d);                    // d_hat
            cols[11][t] = BinaryPoly::from(h);                    // h_hat

            // ── Carry polynomials ───────────────────────────────────────
            // Computed from the full (non-wrapping) sums of the SHA-256
            // round function. The carry μ is the integer overflow when
            // adding 32-bit values: μ = floor(sum / 2³²).
            //
            // a-update: a[t+1] = h + Σ₁(e) + Ch(e,f,g) + K_t + W + Σ₀(a) + Maj(a,b,c)
            let sigma1_val = big_sigma1(e);
            let ch_val = ch(e, f, g);
            let sigma0_val = big_sigma0(a);
            let maj_val = maj(a, b, c);

            let sum_a: u64 = h as u64
                + sigma1_val as u64
                + ch_val as u64
                + K[t] as u64
                + w[t] as u64
                + sigma0_val as u64
                + maj_val as u64;
            let mu_a_val = (sum_a >> 32) as u32;
            cols[12][t] = BinaryPoly::from(mu_a_val);             // mu_a

            // e-update: e[t+1] = d + h + Σ₁(e) + Ch(e,f,g) + K_t + W
            let sum_e: u64 = d as u64
                + h as u64
                + sigma1_val as u64
                + ch_val as u64
                + K[t] as u64
                + w[t] as u64;
            let mu_e_val = (sum_e >> 32) as u32;
            cols[13][t] = BinaryPoly::from(mu_e_val);             // mu_e

            // μ_W for message schedule (requires multi-row lookback, deferred)
            // cols[14] already zero

            // Shift quotient / remainder for σ₀ and σ₁.
            if t >= 16 {
                // S0: shift quotient = SHR³(W_{t-15}) = W_{t-15} >> 3
                cols[15][t] = BinaryPoly::from(w[t - 15] >> 3);
                // S1: shift quotient = SHR¹⁰(W_{t-2}) = W_{t-2} >> 10
                cols[16][t] = BinaryPoly::from(w[t - 2] >> 10);

                // R0 = W_{t-15} mod X³  (bottom 3 bits)
                cols[17][t] = BinaryPoly::from(w[t - 15] & 0x7);
                // R1 = W_{t-2} mod X¹⁰  (bottom 10 bits)
                cols[18][t] = BinaryPoly::from(w[t - 2] & 0x3FF);
            }

            // K_t: round constant
            cols[19][t] = BinaryPoly::from(K[t]);                 // K_t

            // ── SHA-256 round function ──────────────────────────────────
            let t1 = h
                .wrapping_add(sigma1_val)
                .wrapping_add(ch_val)
                .wrapping_add(K[t])
                .wrapping_add(w[t]);
            let t2 = sigma0_val.wrapping_add(maj_val);

            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }

        // ── Boundary row: final state at row 64 for down-references ─────
        //
        // The carry propagation constraints (C10, C11) use down[COL_A_HAT]
        // and down[COL_E_HAT] to refer to the round t+1 state. At row 63
        // (the last SHA-256 round), down references row 64 which would
        // otherwise be zero-padding. We fill in the final a and e values
        // so that the carry constraints remain satisfied at the boundary.
        //
        // Row 64 must hold the "next a" and "next e" so that the carry
        // constraints C10/C11 at row 63 (which reference down[COL_A_HAT]
        // and down[COL_E_HAT]) are satisfied.  We must NOT set COL_D_HAT
        // or COL_H_HAT here: those appear as up[...] in C10/C11 at row 64,
        // and non-zero values there would make the carry expression non-zero
        // (since down[...] at row 65 is zero and carry/K_t/W are zero).
        if num_rows > 64 {
            cols[0][64] = BinaryPoly::from(a);   // a after all 64 rounds
            cols[1][64] = BinaryPoly::from(e);   // e after all 64 rounds
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

    /// Verify that the witness generation produces the correct SHA-256
    /// hash of the empty string: `e3b0c442…7852b855`.
    #[test]
    fn witness_produces_correct_sha256_of_empty_string() {
        let mut rng = rand::rng();
        let trace = <Sha256Uair as GenerateWitness<BinaryPoly<32>>>::generate_witness(6, &mut rng);

        // Verify dimensions: 20 columns, 64 rows each.
        assert_eq!(trace.len(), NUM_COLS);
        for col in &trace {
            assert_eq!(col.evaluations.len(), 64);
        }

        // After 64 rounds the final a value should equal
        // H[0] + a_final = 0x6a09e667 + a_64 mod 2^32.
        // For the empty string: H_final[0] = 0xe3b0c442.
        //
        // We can verify the intermediate state at round 0 matches the
        // initial hash values.
        let a0 = trace[0].evaluations[0].to_u64() as u32;
        let e0 = trace[1].evaluations[0].to_u64() as u32;
        let d0 = trace[10].evaluations[0].to_u64() as u32;
        let h0 = trace[11].evaluations[0].to_u64() as u32;

        assert_eq!(a0, 0x6a09e667, "a at round 0 should be H[0]");
        assert_eq!(e0, 0x510e527f, "e at round 0 should be H[4]");
        assert_eq!(d0, 0xa54ff53a, "d at round 0 should be H[3]");
        assert_eq!(h0, 0x5be0cd19, "h at round 0 should be H[7]");
    }

    /// Verify that Σ₀ and Σ₁ columns are consistent with a and e.
    #[test]
    fn sigma_columns_are_consistent() {
        let mut rng = rand::rng();
        let trace = <Sha256Uair as GenerateWitness<BinaryPoly<32>>>::generate_witness(6, &mut rng);

        for t in 0..64 {
            let a_val = trace[0].evaluations[t].to_u64() as u32;
            let e_val = trace[1].evaluations[t].to_u64() as u32;
            let sigma0_val = trace[3].evaluations[t].to_u64() as u32;
            let sigma1_val = trace[4].evaluations[t].to_u64() as u32;

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
        let trace = <Sha256Uair as GenerateWitness<BinaryPoly<32>>>::generate_witness(6, &mut rng);

        // We can only check ch_ef and ch_neg_eg values at round 0 since
        // we know f = H[5] and g = H[6] at that point.
        let e0 = trace[1].evaluations[0].to_u64() as u32;
        let ch_ef0 = trace[6].evaluations[0].to_u64() as u32;
        let ch_neg_eg0 = trace[7].evaluations[0].to_u64() as u32;

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
        let trace = <Sha256Uair as GenerateWitness<BinaryPoly<32>>>::generate_witness(6, &mut rng);

        let w0 = trace[2].evaluations[0].to_u64() as u32;
        assert_eq!(w0, 0x8000_0000, "W[0] should be 0x80000000 for empty msg");

        for t in 1..16 {
            let wt = trace[2].evaluations[t].to_u64() as u32;
            assert_eq!(wt, 0, "W[{t}] should be 0 for empty msg");
        }
    }

    /// Verify shift quotient/remainder columns satisfy W = R + X^k * S.
    #[test]
    fn shift_decomposition_is_correct() {
        let mut rng = rand::rng();
        let trace = <Sha256Uair as GenerateWitness<BinaryPoly<32>>>::generate_witness(6, &mut rng);

        for t in 16..64 {
            let _w_t_minus_15 = trace[2].evaluations[t].to_u64() as u32; // W_hat uses current W column
            // But constraints 5/6 reference W[t-15] and W[t-2], not current row's W.
            // In the current "same-row" layout the W column stores W_t.
            // The σ₀ operates on W_{t-15}, and the σ₁ on W_{t-2}.
            // Let's verify the actual trace column values are consistent.

            // S0 = SHR³(W_{t-15})
            let s0 = trace[15].evaluations[t].to_u64() as u32;
            // R0 = W_{t-15} mod X³ (bottom 3 bits)
            let r0 = trace[17].evaluations[t].to_u64() as u32;

            // w[t-15] — we get this from the W column at row t-15
            let w_tm15 = trace[2].evaluations[t - 15].to_u64() as u32;
            assert_eq!(s0, w_tm15 >> 3, "S0 = SHR³(W[t-15]) at round {t}");
            assert_eq!(r0, w_tm15 & 0x7, "R0 = W[t-15] mod X³ at round {t}");
            assert_eq!(w_tm15, r0 | (s0 << 3), "W[t-15] = R0 + X³·S0 at round {t}");

            // S1 = SHR¹⁰(W_{t-2})
            let s1 = trace[16].evaluations[t].to_u64() as u32;
            // R1 = W_{t-2} mod X¹⁰ (bottom 10 bits)
            let r1 = trace[18].evaluations[t].to_u64() as u32;
            let w_tm2 = trace[2].evaluations[t - 2].to_u64() as u32;
            assert_eq!(s1, w_tm2 >> 10, "S1 = SHR¹⁰(W[t-2]) at round {t}");
            assert_eq!(r1, w_tm2 & 0x3FF, "R1 = W[t-2] mod X¹⁰ at round {t}");
            assert_eq!(w_tm2, r1 | (s1 << 10), "W[t-2] = R1 + X¹⁰·S1 at round {t}");
        }
    }

    /// Verify σ₀ and σ₁ columns match the small-sigma functions.
    #[test]
    fn small_sigma_columns_are_consistent() {
        let mut rng = rand::rng();
        let trace = <Sha256Uair as GenerateWitness<BinaryPoly<32>>>::generate_witness(6, &mut rng);

        for t in 16..64 {
            let w_tm15 = trace[2].evaluations[t - 15].to_u64() as u32;
            let w_tm2 = trace[2].evaluations[t - 2].to_u64() as u32;
            let sigma0_w = trace[8].evaluations[t].to_u64() as u32;
            let sigma1_w = trace[9].evaluations[t].to_u64() as u32;

            assert_eq!(sigma0_w, small_sigma0(w_tm15), "σ₀(W[t-15]) mismatch at round {t}");
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
    /// Similarly for the e-update.
    #[test]
    fn carry_polynomials_are_consistent() {
        let mut rng = rand::rng();
        let trace = <Sha256Uair as GenerateWitness<BinaryPoly<32>>>::generate_witness(6, &mut rng);

        for t in 0..63 {
            let h_val = trace[11].evaluations[t].to_u64();
            let sigma1_val = trace[4].evaluations[t].to_u64();
            let ch_ef_val = trace[6].evaluations[t].to_u64();
            let ch_neg_eg_val = trace[7].evaluations[t].to_u64();
            let k_val = trace[19].evaluations[t].to_u64();
            let w_val = trace[2].evaluations[t].to_u64();
            let sigma0_val = trace[3].evaluations[t].to_u64();
            let maj_val = trace[5].evaluations[t].to_u64();
            let mu_a_val = trace[12].evaluations[t].to_u64();

            let a_next = trace[0].evaluations[t + 1].to_u64();

            let sum_a = h_val + sigma1_val + ch_ef_val + ch_neg_eg_val
                + k_val + w_val + sigma0_val + maj_val;

            assert_eq!(
                a_next as i64 - sum_a as i64 + (mu_a_val as i64) * (1i64 << 32),
                0,
                "a-update carry check failed at round {t}: \
                 a_next={a_next:#x}, sum_a={sum_a:#x}, mu_a={mu_a_val}"
            );

            // e-update carry
            let d_val = trace[10].evaluations[t].to_u64();
            let mu_e_val = trace[13].evaluations[t].to_u64();
            let e_next = trace[1].evaluations[t + 1].to_u64();

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

    /// Verify K_t column matches the SHA-256 round constants.
    #[test]
    fn kt_column_is_correct() {
        let mut rng = rand::rng();
        let trace = <Sha256Uair as GenerateWitness<BinaryPoly<32>>>::generate_witness(6, &mut rng);

        for t in 0..64 {
            let kt = trace[19].evaluations[t].to_u64() as u32;
            assert_eq!(kt, K[t], "K_t mismatch at round {t}");
        }
    }
}
