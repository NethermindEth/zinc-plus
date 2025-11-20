/// Number of branches used in each FFT layer (radix). The construction fixes
/// this to 8 in accordance with the pseudo-Reed–Solomon spec.
const RADIX: usize = 8;
/// Base-case input length handled by the Vandermonde block.
const BASE_DIM: usize = 32;
/// Base-case output length (number of evaluation points).
const BASE_LEN: usize = 64;

/// Default prime/root pair used in the original Gemini description (works for k=1).
const DEFAULT_P: u128 = 7681;
const DEFAULT_OMEGA: u128 = 7146;

/// Alternative pair that works for k=2 (lengths 2^{11} -> 2^{12}).
const ALT_P_K2: u128 = 12289;
const ALT_OMEGA_K2: u128 = 1331;

/// Pseudo Reed-Solomon encoder over the integers. Internally uses a radix-8
/// FFT-like recursion with a base Vandermonde matrix of dimensions 64x32.
#[derive(Debug, Clone)]
pub struct IprsCode {
    k: usize,
    m: usize,
    n: usize,
    modulus: u128,
    omega: u128,
    twiddle_tables: Vec<Vec<[u128; RADIX]>>,
    base_matrix: [[u128; BASE_DIM]; BASE_LEN],
}

impl IprsCode {
    /// Create a code with the default `(p, ω)` pair `(7681, 7146)`, valid for `k = 1`.
    pub fn new(k: usize) -> Self {
        Self::with_params(k, DEFAULT_P, DEFAULT_OMEGA)
    }

    /// Create a code with custom modulus and generator. The pair must satisfy
    /// that `ω` is a primitive `N`-th root of unity in ℤ_p, where `N = 2^{6+3k}`.
    pub fn with_params(k: usize, modulus: u128, omega: u128) -> Self {
        assert!(k > 0, "k must be positive");
        let m = 1usize << (5 + 3 * k);
        let n = 1usize << (6 + 3 * k);
        let base_matrix = compute_base_matrix(modulus, omega, n);
        let twiddle_tables = build_twiddle_tables(k, n, modulus, omega);
        Self {
            k,
            m,
            n,
            modulus,
            omega,
            twiddle_tables,
            base_matrix,
        }
    }

    pub fn row_len(&self) -> usize {
        self.m
    }

    pub fn codeword_len(&self) -> usize {
        self.n
    }

    pub fn modulus(&self) -> u128 {
        self.modulus
    }

    pub fn omega(&self) -> u128 {
        self.omega
    }

    /// Encode without modular reduction, purely over the integers.
    pub fn encode(&self, row: &[u128]) -> Vec<u128> {
        assert_eq!(
            row.len(),
            self.m,
            "Input length {} does not match expected row length {}",
            row.len(),
            self.m
        );
        self.encode_fft(row, self.k)
    }

    /// Encode while reducing every stage modulo `p`. This behaves like a
    /// standard NTT/RS encoder and is used for cross-checks.
    pub fn encode_mod_p(&self, row: &[u128]) -> Vec<u128> {
        assert_eq!(
            row.len(),
            self.m,
            "Input length {} does not match expected row length {}",
            row.len(),
            self.m
        );
        self.encode_fft_mod(row, self.k)
    }

    fn encode_fft(&self, data: &[u128], depth: usize) -> Vec<u128> {
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
            subresults.push(self.encode_fft(&chunk, depth - 1));
        }

        self.combine_stage(&subresults, &self.twiddle_tables[depth - 1], AddMode::Wide)
    }

    fn encode_fft_mod(&self, data: &[u128], depth: usize) -> Vec<u128> {
        if depth == 0 {
            return self.base_multiply_mod(data);
        }

        let chunk_len = data.len() / RADIX;
        let mut subresults = Vec::with_capacity(RADIX);
        for chunk_idx in 0..RADIX {
            let mut chunk = Vec::with_capacity(chunk_len);
            for j in 0..chunk_len {
                chunk.push(data[RADIX * j + chunk_idx] % self.modulus);
            }
            subresults.push(self.encode_fft_mod(&chunk, depth - 1));
        }

        self.combine_stage(&subresults, &self.twiddle_tables[depth - 1], AddMode::Modulo)
    }

    fn base_multiply(&self, chunk: &[u128]) -> Vec<u128> {
        assert_eq!(chunk.len(), BASE_DIM);
        let mut output = vec![0u128; BASE_LEN];
        for (row_idx, matrix_row) in self.base_matrix.iter().enumerate() {
            let mut acc = 0u128;
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

    fn base_multiply_mod(&self, chunk: &[u128]) -> Vec<u128> {
        assert_eq!(chunk.len(), BASE_DIM);
        let mut output = vec![0u128; BASE_LEN];
        for (row_idx, matrix_row) in self.base_matrix.iter().enumerate() {
            let mut acc = 0u128;
            for col in 0..BASE_DIM {
                let term = self.mod_mul(chunk[col] % self.modulus, matrix_row[col]);
                acc = self.mod_add(acc, term);
            }
            output[row_idx] = acc;
        }
        output
    }

fn combine_stage(
    &self,
    subresults: &[Vec<u128>],
    twiddle_table: &[[u128; RADIX]],
    mode: AddMode,
) -> Vec<u128> {
    // Each index `idx` corresponds to a single output position and therefore
    // to a unique set of twiddle multipliers. We rely on the precomputed stage
    // table to avoid recomputing roots on the fly.
    let sub_len = subresults[0].len();
    debug_assert_eq!(twiddle_table.len(), sub_len * RADIX);

        let mut output = vec![0u128; sub_len * RADIX];
        for (idx, slot) in output.iter_mut().enumerate() {
            let column = idx % sub_len;
            let twiddles = &twiddle_table[idx];
            match mode {
                AddMode::Wide => {
                    let mut acc = 0u128;
                    for branch in 0..RADIX {
                        let term = subresults[branch][column]
                            .checked_mul(twiddles[branch])
                            .expect("Multiplication overflow");
                        acc = acc.checked_add(term).expect("Addition overflow");
                    }
                    *slot = acc;
                }
                AddMode::Modulo => {
                    let mut acc = 0u128;
                    for branch in 0..RADIX {
                        let term = self.mod_mul(subresults[branch][column], twiddles[branch]);
                        acc = self.mod_add(acc, term);
                    }
                    *slot = acc;
                }
            }
        }
        output
    }

    fn mod_add(&self, a: u128, b: u128) -> u128 {
        let tmp = (a % self.modulus) + (b % self.modulus);
        if tmp >= self.modulus {
            tmp - self.modulus
        } else {
            tmp
        }
    }

    fn mod_mul(&self, a: u128, b: u128) -> u128 {
        ((a % self.modulus) * (b % self.modulus)) % self.modulus
    }
}

enum AddMode {
    Wide,
    Modulo,
}

/// Precompute stage-specific twiddle tables. Each table contains `stage_len`
/// entries where entry `i` stores the RADIX consecutive powers of
/// `(omega)^(stride * i)`. This matches the order used in the recursive FFT.
fn build_twiddle_tables(
    k: usize,
    n: usize,
    modulus: u128,
    omega: u128,
) -> Vec<Vec<[u128; RADIX]>> {
    let mut tables = Vec::with_capacity(k);
    for depth in 1..=k {
        let stage_len = BASE_LEN * RADIX.pow(depth as u32);
        let stride = (n / stage_len) as u128;

        let mut stage = Vec::with_capacity(stage_len);
        for idx in 0..stage_len {
            let twiddle_base = mod_pow_generic(omega, stride * idx as u128, modulus);
            let mut twiddles = [0u128; RADIX];
            let mut power = 1u128;
            for slot in 0..RADIX {
                twiddles[slot] = power;
                power = mod_mul_generic(power, twiddle_base, modulus);
            }
            stage.push(twiddles);
        }
        tables.push(stage);
    }
    tables
}

/// Build the 64x32 Vandermonde block used at the leaves of the recursion. The
/// evaluation points follow the FFT ordering induced by `(n / 64)` strides.
fn compute_base_matrix(modulus: u128, omega: u128, n: usize) -> [[u128; BASE_DIM]; BASE_LEN] {
    let mut matrix = [[0u128; BASE_DIM]; BASE_LEN];
    // Step between successive evaluation points at the base (size-64) stage.
    let row_step = mod_pow_generic(omega, (n / BASE_LEN) as u128, modulus);
    let mut current = 1u128;
    for row in 0..BASE_LEN {
        let mut accum = 1u128;
        for col in 0..BASE_DIM {
            matrix[row][col] = accum;
            accum = mod_mul_generic(accum, current, modulus);
        }
        current = mod_mul_generic(current, row_step, modulus);
    }
    matrix
}

fn mod_mul_generic(a: u128, b: u128, modulus: u128) -> u128 {
    ((a % modulus) * (b % modulus)) % modulus
}

fn mod_add_generic(a: u128, b: u128, modulus: u128) -> u128 {
    let tmp = (a % modulus) + (b % modulus);
    if tmp >= modulus {
        tmp - modulus
    } else {
        tmp
    }
}

fn mod_sub_generic(a: u128, b: u128, modulus: u128) -> u128 {
    if a >= b {
        a - b
    } else {
        a + modulus - b
    }
}

fn mod_pow_generic(base: u128, exp: u128, modulus: u128) -> u128 {
    let mut result = 1u128;
    let mut b = base % modulus;
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
    use super::{
        mod_add_generic, mod_mul_generic, mod_pow_generic, mod_sub_generic, IprsCode, ALT_OMEGA_K2,
        ALT_P_K2, DEFAULT_OMEGA, DEFAULT_P,
    };

    #[test]
    fn encode_has_expected_lengths() {
        let code = IprsCode::new(1);
        let input = vec![1u128; code.row_len()];
        let output = code.encode(&input);
        assert_eq!(output.len(), code.codeword_len());
    }

    #[test]
    fn encode_is_deterministic() {
        let code = IprsCode::new(1);
        let mut input = vec![0u128; code.row_len()];
        for (idx, value) in input.iter_mut().enumerate() {
            *value = (idx as u128).pow(3);
        }
        let first = code.encode(&input);
        let second = code.encode(&input);
        assert_eq!(first, second);
    }

    #[test]
    fn mod_p_matches_reduction() {
        let code = IprsCode::new(1);
        let mut input = vec![0u128; code.row_len()];
        for (idx, value) in input.iter_mut().enumerate() {
            *value = ((idx * 17 + 5) as u128).pow(2);
        }

        let wide = code.encode(&input);
        let narrow = code.encode_mod_p(&input);
        for (w, n) in wide.iter().zip(narrow.iter()) {
            assert_eq!(n, &(w % code.modulus()));
        }
    }

    #[test]
    fn matches_radix2_ntt_mod_p_for_k1_and_k2() {
        let configs = [
            (1, DEFAULT_P, DEFAULT_OMEGA),
            (2, ALT_P_K2, ALT_OMEGA_K2),
        ];

        for (k, modulus, omega) in configs {
            let code = IprsCode::with_params(k, modulus, omega);
            let mut input = vec![0u128; code.row_len()];
            for (idx, value) in input.iter_mut().enumerate() {
                *value = ((idx * 31 + 7) as u128) % modulus;
            }

            let mut padded = vec![0u128; code.codeword_len()];
            padded[..input.len()].copy_from_slice(&input);
            radix2_ntt_mod(&mut padded, modulus, omega);

            let ours = code.encode_mod_p(&input);
            assert_eq!(ours, padded);
        }
    }

    fn radix2_ntt_mod(values: &mut [u128], modulus: u128, omega: u128) {
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
                let mut w = 1u128;
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
}
