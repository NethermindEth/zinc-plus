//! Witness generation for the Poly1305 UAIR.
//!
//! Generates a trace of chained Poly1305 accumulator updates.
//! Each row represents one iteration: acc = (acc + padded_block) * r mod p.
//!
//! The trace is a chain: each row's `acc_out` feeds the next row's `acc`.

use zinc_poly::{
    mle::DenseMultilinearExtension,
    univariate::binary::BinaryPoly,
};

use crate::{
    NUM_COLS,
    COL_ACC_0,
    COL_MSG_0,
    COL_R_0,
    COL_ACC_OUT_0,
};

// ─── 130-bit arithmetic helpers (5 × 32-bit limbs) ─────────────────────────

/// Add two 5-limb numbers, returning 5-limb result (may overflow into 6th limb
/// but for Poly1305 the inputs are ≤ 131 bits so the result fits in 5 limbs
/// plus at most a 1-bit carry).
fn add_limbs(a: &[u32; 5], b: &[u32; 5]) -> ([u32; 5], u32) {
    let mut result = [0u32; 5];
    let mut carry: u64 = 0;
    for i in 0..5 {
        let sum = a[i] as u64 + b[i] as u64 + carry;
        result[i] = sum as u32;
        carry = sum >> 32;
    }
    #[allow(clippy::cast_possible_truncation)]
    (result, carry as u32)
}

/// Schoolbook multiplication of two 5-limb numbers → 10-limb result.
fn mul_limbs(a: &[u32; 5], b: &[u32; 5]) -> [u32; 10] {
    let mut accum = [0u64; 10];
    for i in 0..5 {
        for j in 0..5 {
            accum[i + j] += a[i] as u64 * b[j] as u64;
        }
    }
    // Propagate carries.
    let mut result = [0u32; 10];
    let mut carry: u64 = 0;
    for i in 0..10 {
        let sum = accum[i] + carry;
        result[i] = sum as u32;
        carry = sum >> 32;
    }
    result
}

/// Multiply a 5-limb number by a small scalar (≤ 5), returning 5 limbs + carry.
fn mul_scalar(a: &[u32; 5], s: u64) -> ([u32; 5], u32) {
    let mut result = [0u32; 5];
    let mut carry: u64 = 0;
    for i in 0..5 {
        let prod = a[i] as u64 * s + carry;
        result[i] = prod as u32;
        carry = prod >> 32;
    }
    #[allow(clippy::cast_possible_truncation)]
    (result, carry as u32)
}

/// Reduce a 10-limb product modulo p = 2^130 - 5.
///
/// Uses: 2^130 ≡ 5 (mod p), so product = lo + 5 * hi where
/// lo = product[0..130 bits], hi = product[130..260 bits].
fn reduce_mod_p(product: &[u32; 10]) -> [u32; 5] {
    // Split at bit 130: limbs 0..3 are full, limb 4 has 2 bits.
    let lo = [product[0], product[1], product[2], product[3], product[4] & 3];

    // hi = product >> 130
    let hi = [
        (product[4] >> 2) | ((product[5] & 3) << 30),
        (product[5] >> 2) | ((product[6] & 3) << 30),
        (product[6] >> 2) | ((product[7] & 3) << 30),
        (product[7] >> 2) | ((product[8] & 3) << 30),
        (product[8] >> 2) | ((product[9] & 3) << 30),
    ];

    let (hi5, hi5_carry) = mul_scalar(&hi, 5);
    let (mut result, mut carry) = add_limbs(&lo, &hi5);

    // Handle the extra carry from hi5.
    carry += hi5_carry;

    // If there's still a carry or result ≥ p, do one more reduction.
    // carry bits spill into bit 130+, so: extra = (result[4] >> 2) + carry_bits
    if carry > 0 || result[4] > 3 {
        let extra = ((result[4] as u64) >> 2) + (carry as u64);
        result[4] &= 3;
        // Add 5 * extra back into the low limbs.
        let add_back = extra * 5;
        let mut c: u64 = add_back;
        for i in 0..5 {
            c += result[i] as u64;
            result[i] = c as u32;
            c >>= 32;
        }
        // Final check: if still ≥ p (can happen if result was exactly p).
        if result[4] > 3 || (result[4] == 3 && result[3] == 0xFFFF_FFFF
            && result[2] == 0xFFFF_FFFF && result[1] == 0xFFFF_FFFF
            && result[0] >= 0xFFFF_FFFB)
        {
            // Subtract p.
            let mut borrow: i64 = 0;
            let p = crate::constants::POLY1305_P;
            for i in 0..5 {
                let diff = result[i] as i64 - p[i] as i64 + borrow;
                if diff < 0 {
                    result[i] = (diff + (1i64 << 32)) as u32;
                    borrow = -1;
                } else {
                    result[i] = diff as u32;
                    borrow = 0;
                }
            }
        }
    }

    result
}

/// Clamp the r key per RFC 7539.
fn clamp_r(r_bytes: &[u8; 16]) -> [u32; 5] {
    let mut words = [0u32; 5];
    for i in 0..4 {
        words[i] = u32::from_le_bytes([
            r_bytes[4 * i],
            r_bytes[4 * i + 1],
            r_bytes[4 * i + 2],
            r_bytes[4 * i + 3],
        ]);
    }
    words[0] &= crate::constants::R_CLAMP_MASKS[0];
    words[1] &= crate::constants::R_CLAMP_MASKS[1];
    words[2] &= crate::constants::R_CLAMP_MASKS[2];
    words[3] &= crate::constants::R_CLAMP_MASKS[3];
    // words[4] = 0 (r is only 128 bits)
    words
}

fn u32_to_bp(val: u32) -> BinaryPoly<32> {
    BinaryPoly::<32>::from(val)
}

// ─── Witness generation ─────────────────────────────────────────────────────

/// Generate the Poly1305 witness trace.
///
/// The trace has `2^{num_vars}` rows of chained Poly1305 iterations.
/// Each row computes: acc_out = (acc + padded_msg) * r mod (2^130 - 5).
///
/// Uses a deterministic key and message blocks for benchmarking.
///
/// Returns `NUM_COLS` columns as `DenseMultilinearExtension<BinaryPoly<32>>`.
pub fn generate_poly1305_witness(
    num_vars: usize,
) -> Vec<DenseMultilinearExtension<BinaryPoly<32>>> {
    assert!(
        num_vars >= 1,
        "Poly1305 requires at least 1 variable (2 rows), got {num_vars}"
    );

    let num_rows = 1usize << num_vars;

    // Allocate column storage.
    let mut cols: Vec<Vec<BinaryPoly<32>>> =
        (0..NUM_COLS).map(|_| vec![BinaryPoly::<32>::from(0u32); num_rows]).collect();

    // Deterministic key for benchmarking.
    let r_bytes: [u8; 16] = [
        0x85, 0xd6, 0xbe, 0x78, 0x57, 0x55, 0x6d, 0x33,
        0x7f, 0x44, 0x52, 0xfe, 0x42, 0xd5, 0x06, 0xa8,
    ];
    let r = clamp_r(&r_bytes);

    // Initial accumulator = 0.
    let mut acc = [0u32; 5];

    for t in 0..num_rows {
        // Deterministic message block: fill with (t+1) pattern.
        #[allow(clippy::cast_possible_truncation)]
        let msg_bytes: [u8; 16] = std::array::from_fn(|i| ((t + 1 + i) & 0xFF) as u8);
        let mut msg = [0u32; 4];
        for i in 0..4 {
            msg[i] = u32::from_le_bytes([
                msg_bytes[4 * i],
                msg_bytes[4 * i + 1],
                msg_bytes[4 * i + 2],
                msg_bytes[4 * i + 3],
            ]);
        }

        // Padded message: msg || 0x01 (add high bit at position 128).
        let padded: [u32; 5] = [msg[0], msg[1], msg[2], msg[3], 1];

        // sum = acc + padded
        let (sum, _) = add_limbs(&acc, &padded);

        // product = sum * r
        let product = mul_limbs(&sum, &r);

        // acc_out = product mod p
        let acc_out = reduce_mod_p(&product);

        // Write columns.
        for i in 0..5 {
            cols[COL_ACC_0 + i][t] = u32_to_bp(acc[i]);
        }
        for i in 0..4 {
            cols[COL_MSG_0 + i][t] = u32_to_bp(msg[i]);
        }
        for i in 0..5 {
            cols[COL_R_0 + i][t] = u32_to_bp(r[i]);
        }
        for i in 0..5 {
            cols[COL_ACC_OUT_0 + i][t] = u32_to_bp(acc_out[i]);
        }

        // Chain: output feeds next input.
        acc = acc_out;
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
/// Currently a placeholder — corrections will be needed when full
/// Qx carry constraints are implemented.
#[cfg(feature = "true-ideal")]
fn compute_corrections(
    _cols: &mut [Vec<BinaryPoly<32>>],
    _num_rows: usize,
) {
    // No Qx carry constraints yet, so no corrections needed.
}
