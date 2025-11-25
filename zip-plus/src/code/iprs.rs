use crypto_primitives::ConstIntSemiring;
use zinc_transcript::traits::ConstTranscribable;
use zinc_utils::named::Named;

use crate::pcs::structs::ZipTypes;

/// Number of branches used in each NTT layer (radix). The construction fixes
/// this to 8 in accordance with the pseudo-Reed–Solomon spec.
const RADIX: usize = 8;
/// Base-case input length handled by the Vandermonde block.
const BASE_DIM: usize = 32;
/// Base-case output length (number of evaluation points).
const BASE_LEN: usize = 64;


#[derive(Debug, Clone)]
pub struct IprsConfig<Twiddle: ConstIntSemiring + ConstTranscribable + Named> {
    pub modulus: Twiddle,
    pub omega: Twiddle,
    pub twiddle_tables: Vec<Vec<[Twiddle; RADIX]>>,
    pub base_matrix: [[Twiddle; BASE_DIM]; BASE_LEN],
    // We fix those values in the config as they define the twiddle tables.
    // depth of the recursion
    pub k: usize,
    // input row length
    pub m: usize,
    // codeword length
    pub n: usize,
}

impl IprsConfig<i128> {
    /// Create a code with custom modulus and generator. The pair must satisfy
    /// that `ω` is a primitive `N`-th root of unity in ℤ_p, where `N = 2^{6+3k}`.
    pub fn new(k: usize, modulus: i128, omega: i128) -> Self {
        let m = 1usize << (5 + 3 * k);
        let n = 1usize << (6 + 3 * k);
        let base_matrix = compute_base_matrix(modulus, omega, n);
        let twiddle_tables = build_twiddle_tables(k, n, modulus, omega);
        Self {
            modulus,
            omega,
            twiddle_tables,
            base_matrix,
            k,
            m,
            n,
        }
    }
    
}

/// Pseudo Reed-Solomon encoder over the integers. Internally uses a radix-8
/// NTT-style recursion with a base Vandermonde matrix of dimensions 64x32.
#[derive(Debug, Clone)]
pub struct IprsCode {
    pub cfg: IprsConfig<i128>,
}

impl IprsCode {
    /// Create a code with the default `(p, ω)` pair e.g. `(7681, 7146)`, valid for `k = 1`.
    pub fn new (config: IprsConfig<i128>) -> Self {
        Self { cfg: config }
    }

    /// Encode without modular reduction, purely over the integers.
    pub fn encode(&self, row: &[i128]) -> Vec<i128> {
        assert_eq!(
            row.len(),
            self.cfg.m,
            "Input length {} does not match expected row length {}",
            row.len(),
            self.cfg.m
        );
        self.encode_ntt(row, self.cfg.k)
    }

    fn encode_ntt(&self, data: &[i128], depth: usize) -> Vec<i128> {
        if depth == 0 {
            // Base-case: multiply the 32-term vector by the precomputed
            // Vandermonde matrix to obtain 64 evaluation points.
            return self.base_multiply(data);
        }

        let chunk_len = data.len() / RADIX;
        let mut subresults = Vec::with_capacity(RADIX);
        for chunk_idx in 0..RADIX {
            let mut chunk = Vec::with_capacity(chunk_len);
            for j in 0..chunk_len {
                chunk.push(data[RADIX * j + chunk_idx]);
            }
            // Recursively evaluate the “child” polynomial corresponding to
            // coset `x ↦ x + ω^{chunk_idx}`.
            subresults.push(self.encode_ntt(&chunk, depth - 1));
        }

        self.combine_stage(&subresults, &self.cfg.twiddle_tables[depth - 1])
    }

    fn base_multiply(&self, chunk: &[i128]) -> Vec<i128> {
        assert_eq!(chunk.len(), BASE_DIM);
        let mut output = vec![0i128; BASE_LEN];
        for (row_idx, matrix_row) in self.cfg.base_matrix.iter().enumerate() {
            let mut acc = 0i128;
            // Dot-product between the i-th row of the Vandermonde matrix and
            // the 32 input coordinates.
            for col in 0..BASE_DIM {
                let term = chunk[col]
                    .checked_mul(matrix_row[col])
                    .expect("Base multiplication overflow");
                acc = acc.checked_add(term).expect("Base addition overflow");
            }
            output[row_idx] = acc;
        }
        output
    }

    fn combine_stage(
        &self,
        subresults: &[Vec<i128>],
        twiddle_table: &[[i128; RADIX]],
    ) -> Vec<i128> {
        // Each index `idx` corresponds to a single output position and therefore
        // to a unique set of twiddle multipliers. We rely on the precomputed stage
        // table to avoid recomputing roots on the fly.
        let sub_len = subresults[0].len();
        debug_assert_eq!(twiddle_table.len(), sub_len * RADIX);

        let mut output = vec![0i128; sub_len * RADIX];
        for (idx, slot) in output.iter_mut().enumerate() {
            let column = idx % sub_len;
            let twiddles = &twiddle_table[idx];
            let mut acc = 0i128;
            for branch in 0..RADIX {
                let term = subresults[branch][column]
                    .checked_mul(twiddles[branch])
                    .expect("Multiplication overflow");
                acc = acc.checked_add(term).expect("Addition overflow");
            }
            *slot = acc;
        }
        output
    }
}

/// Precompute stage-specific twiddle tables. Each table contains `stage_len`
/// entries where entry `i` stores the RADIX consecutive powers of
/// `(omega)^(stride * i)`. This matches the order used in the recursive NTT.
fn build_twiddle_tables(k: usize, n: usize, modulus: i128, omega: i128) -> Vec<Vec<[i128; RADIX]>> {
    let mut tables = Vec::with_capacity(k);
    for depth in 1..=k {
        let stage_len = BASE_LEN * RADIX.pow(depth as u32);
        let stride = (n / stage_len) as u128;

        let mut stage = Vec::with_capacity(stage_len);
        for idx in 0..stage_len {
            let twiddle_base = mod_pow_generic(omega, stride * idx as u128, modulus);
            let mut twiddles = [0i128; RADIX];
            let mut power = 1i128;
            for slot in 0..RADIX {
                twiddles[slot] = to_balanced_rep(power, modulus);
                power = mod_mul_generic(power, twiddle_base, modulus);
            }
            stage.push(twiddles);
        }
        tables.push(stage);
    }
    tables
}

/// Build the 64x32 Vandermonde block used at the leaves of the recursion. The
/// evaluation points follow the NTT ordering induced by `(n / 64)` strides.
fn compute_base_matrix(modulus: i128, omega: i128, n: usize) -> [[i128; BASE_DIM]; BASE_LEN] {
    let mut matrix = [[0i128; BASE_DIM]; BASE_LEN];
    // Step between successive evaluation points at the base (size-64) stage.
    let row_step = mod_pow_generic(omega, (n / BASE_LEN) as u128, modulus);
    let mut current = 1i128;
    for row in 0..BASE_LEN {
        let mut accum = 1i128;
        for col in 0..BASE_DIM {
            matrix[row][col] = accum;
            accum = mod_mul_generic(accum, current, modulus);
        }
        current = mod_mul_generic(current, row_step, modulus);
    }
    matrix
}

fn canonical_mod(value: i128, modulus: i128) -> i128 {
    let mut r = value % modulus;
    if r < 0 {
        r += modulus;
    }
    r
}

fn to_balanced_rep(value: i128, modulus: i128) -> i128 {
    let canonical = canonical_mod(value, modulus);
    let half = (modulus - 1) / 2;
    if canonical > half {
        canonical - modulus
    } else {
        canonical
    }
}

fn mod_mul_generic(a: i128, b: i128, modulus: i128) -> i128 {
    let a_red = canonical_mod(a, modulus);
    let b_red = canonical_mod(b, modulus);
    canonical_mod(a_red * b_red, modulus)
}

fn mod_pow_generic(base: i128, exp: u128, modulus: i128) -> i128 {
    let mut result = 1i128;
    let mut b = canonical_mod(base, modulus);
    let mut e = exp;
    while e > 0 {
        if e & 1 == 1 {
            result = mod_mul_generic(result, b, modulus);
        }
        b = mod_mul_generic(b, b, modulus);
        e >>= 1;
    }
    result
}

#[cfg(test)]
mod tests {
    use crate::code::iprs::IprsConfig;

    use super::{
        IprsCode, canonical_mod,mod_mul_generic, mod_pow_generic,
    };

    /// Default prime/root pair used in the original Gemini description (works for k=1).
    const DEFAULT_P: i128 = 7681;
    const DEFAULT_OMEGA: i128 = 7146;

    /// Alternative pair that works for k=2 (lengths 2^{11} -> 2^{12}).
    const ALT_P_K2: i128 = 12289;
    const ALT_OMEGA_K2: i128 = 1331;

    fn mod_add_generic(a: i128, b: i128, modulus: i128) -> i128 {
        let a_red = canonical_mod(a, modulus);
        let b_red = canonical_mod(b, modulus);
        let tmp = a_red + b_red;
        if tmp >= modulus { tmp - modulus } else { tmp }
    }

    fn mod_sub_generic(a: i128, b: i128, modulus: i128) -> i128 {
        let a_red = canonical_mod(a, modulus);
        let b_red = canonical_mod(b, modulus);
        let diff = a_red - b_red;
        if diff < 0 { diff + modulus } else { diff }
    }


    #[test]
    fn encode_has_expected_lengths() {
        let code = IprsCode::new(IprsConfig::new(1, DEFAULT_P, DEFAULT_OMEGA));
        let input = vec![1i128; code.cfg.m];
        let output = code.encode(&input);
        assert_eq!(output.len(), code.cfg.n);
    }

    #[test]
    fn encode_is_deterministic() {
        let code = IprsCode::new(IprsConfig::new(1, DEFAULT_P, DEFAULT_OMEGA));
        let mut input = vec![0i128; code.cfg.m];
        for (idx, value) in input.iter_mut().enumerate() {
            *value = (idx as i128).pow(3);
        }
        let first = code.encode(&input);
        let second = code.encode(&input);
        assert_eq!(first, second);
    }

    #[test]
    fn mod_p_matches_reduction() {
         let configs = [(1, DEFAULT_P, DEFAULT_OMEGA), (2, ALT_P_K2, ALT_OMEGA_K2)];

        for (k, modulus, omega) in configs {
            let code = IprsCode::new(IprsConfig::new(k, modulus, omega));
            let mut input = vec![0i128; code.cfg.m];
            for (idx, value) in input.iter_mut().enumerate() {
                *value = ((idx * 17 + 5) as i128).pow(2);
            }

            let wide = code.encode(&input);

            let mut padded = vec![0i128; code.cfg.n];
            padded[..input.len()].copy_from_slice(&input);
            radix2_ntt_mod(&mut padded, code.cfg.modulus, code.cfg.omega);

            for (w, n) in wide.iter().zip(padded.iter()) {
                assert_eq!(n, &canonical_mod(*w, code.cfg.modulus));
            }
        }
    }

    // this is purely to check the correctness of the NTT implementation used
    // in the test above
    #[test]
    fn polynomial_multiplication_via_ntt_matches_naive() {
        let modulus = DEFAULT_P;
        let n = 32usize; // power of two >= (16 + 16 - 1)
        // DEFAULT_OMEGA is a primitive 512-th root; derive a primitive 32-th root.
        let base_order = 1usize << (6 + 3 * 1); // 512
        assert_eq!(base_order % n, 0);
        let omega = mod_pow_generic(DEFAULT_OMEGA, (base_order / n) as u128, modulus);

        let mut poly_a = vec![0i128; 16];
        let mut poly_b = vec![0i128; 16];
        for (i, coeff) in poly_a.iter_mut().enumerate() {
            *coeff = ((i * 13 + 7) as i128) % modulus;
        }
        for (i, coeff) in poly_b.iter_mut().enumerate() {
            *coeff = ((i * 9 + 5) as i128) % modulus;
        }

        let mut naive = vec![0i128; poly_a.len() + poly_b.len() - 1];
        for (i, &a) in poly_a.iter().enumerate() {
            for (j, &b) in poly_b.iter().enumerate() {
                let idx = i + j;
                let prod = mod_mul_generic(a, b, modulus);
                naive[idx] = mod_add_generic(naive[idx], prod, modulus);
            }
        }

        let mut a_ntt = vec![0i128; n];
        let mut b_ntt = vec![0i128; n];
        a_ntt[..poly_a.len()].copy_from_slice(&poly_a);
        b_ntt[..poly_b.len()].copy_from_slice(&poly_b);

        radix2_ntt_mod(&mut a_ntt, modulus, omega);
        radix2_ntt_mod(&mut b_ntt, modulus, omega);
        for (a_coeff, b_coeff) in a_ntt.iter_mut().zip(b_ntt.iter()) {
            *a_coeff = mod_mul_generic(*a_coeff, *b_coeff, modulus);
        }
        radix2_intt_mod(&mut a_ntt, modulus, omega);

        assert_eq!(&a_ntt[..naive.len()], naive.as_slice());
    }

    #[test]
    fn cyclic_polynomial_multiplication_via_ntt_matches_naive() {
        let modulus = DEFAULT_P;
        let n = 16usize; // cyclic convolution length
        // DEFAULT_OMEGA is a primitive 512-th root; derive a primitive n-th root.
        let base_order = 1usize << (6 + 3 * 1); // 512
        assert_eq!(base_order % n, 0);
        let omega = mod_pow_generic(DEFAULT_OMEGA, (base_order / n) as u128, modulus);

        let mut poly_a = vec![0i128; 16];
        let mut poly_b = vec![0i128; 16];
        for (i, coeff) in poly_a.iter_mut().enumerate() {
            *coeff = ((i * 11 + 3) as i128) % modulus;
        }
        for (i, coeff) in poly_b.iter_mut().enumerate() {
            *coeff = ((i * 7 + 2) as i128) % modulus;
        }

        let mut naive = vec![0i128; n];
        for (i, &a) in poly_a.iter().enumerate() {
            for (j, &b) in poly_b.iter().enumerate() {
                let idx = (i + j) % n;
                let prod = mod_mul_generic(a, b, modulus);
                naive[idx] = mod_add_generic(naive[idx], prod, modulus);
            }
        }

        let mut a_ntt = vec![0i128; n];
        let mut b_ntt = vec![0i128; n];
        a_ntt[..poly_a.len()].copy_from_slice(&poly_a);
        b_ntt[..poly_b.len()].copy_from_slice(&poly_b);

        radix2_ntt_mod(&mut a_ntt, modulus, omega);
        radix2_ntt_mod(&mut b_ntt, modulus, omega);
        for (a_coeff, b_coeff) in a_ntt.iter_mut().zip(b_ntt.iter()) {
            *a_coeff = mod_mul_generic(*a_coeff, *b_coeff, modulus);
        }
        radix2_intt_mod(&mut a_ntt, modulus, omega);

        assert_eq!(a_ntt, naive);
    }

    #[test]
    fn ntt_goes_back_and_forth() {
        let modulus = DEFAULT_P;
        let n = 64usize; // power of two
        let base_order = 1usize << (6 + 3 * 1); // 512
        assert_eq!(base_order % n, 0);
        let omega = mod_pow_generic(DEFAULT_OMEGA, (base_order / n) as u128, modulus);

        let mut data = vec![0i128; n];
        for (i, coeff) in data.iter_mut().enumerate() {
            *coeff = ((i * 19 + 11) as i128) % modulus;
        }

        let original = data.clone();
        radix2_ntt_mod(&mut data, modulus, omega);
        radix2_intt_mod(&mut data, modulus, omega);

        assert_eq!(data, original);
    }

    /// Straightforward radix-2 decimation-in-time NTT used only in tests.
    /// Requires `omega` to be a primitive n-th root of unity.
    fn radix2_ntt_mod(values: &mut [i128], modulus: i128, omega: i128) {
        let n = values.len();
        assert!(n.is_power_of_two());

        let mut j = 0usize;
        for i in 1..n {
            let mut bit = n >> 1;
            while j & bit != 0 {
                j ^= bit;
                bit >>= 1;
            }
            j ^= bit;
            if i < j {
                values.swap(i, j);
            }
        }

        let mut len = 2;
        while len <= n {
            let w_len = mod_pow_generic(omega, (n / len) as u128, modulus);
            for chunk in values.chunks_mut(len) {
                let mut w = 1i128;
                for i in 0..len / 2 {
                    let u = chunk[i];
                    let v = mod_mul_generic(chunk[i + len / 2], w, modulus);
                    chunk[i] = mod_add_generic(u, v, modulus);
                    chunk[i + len / 2] = mod_sub_generic(u, v, modulus);
                    w = mod_mul_generic(w, w_len, modulus);
                }
            }
            len <<= 1;
        }
    }

    fn radix2_intt_mod(values: &mut [i128], modulus: i128, omega: i128) {
        let n = values.len();
        assert!(n.is_power_of_two());
        let omega_inv = mod_pow_generic(omega, (modulus - 2) as u128, modulus);
        radix2_ntt_mod(values, modulus, omega_inv);
        let inv_n = mod_pow_generic(n as i128, (modulus - 2) as u128, modulus);
        for coeff in values.iter_mut() {
            *coeff = mod_mul_generic(*coeff, inv_n, modulus);
        }
    }
}
