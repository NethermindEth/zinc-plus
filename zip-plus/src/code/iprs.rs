const RADIX: usize = 8;
const BASE_DIM: usize = 32;
const BASE_LEN: usize = 64;

// Fixed prime / primitive root pair requested by the user (valid for k = 1).
const P: u128 = 7681;
const OMEGA: u128 = 7146;

/// If we want to support k = 2 with a fully valid root of unity, the pair
/// (p = 12289, \omega = 1331) works because 12289 − 1 = 3 \cdot 4096, so \omega has order
/// 4096 over \ZZ_p.
#[derive(Debug, Clone)]
pub struct IprsCode {
    // k parameter controlling the levels of FFT recursion
    k: usize,
    // length of the vector to be encoded
    m: usize,
    // length of the codeword
    n: usize,
    twiddle_powers: Vec<[u128; RADIX]>,
    base_matrix: [[u128; BASE_DIM]; BASE_LEN],
}

impl IprsCode {
    pub fn new(k: usize) -> Self {
        assert!(k > 0, "k must be positive");
        let m = 1usize << (5 + 3 * k);
        let n = 1usize << (6 + 3 * k);
        let twiddle_powers = compute_twiddles(n);
        let base_matrix = compute_base_matrix();
        Self {
            k,
            m,
            n,
            twiddle_powers,
            base_matrix,
        }
    }

    pub fn row_len(&self) -> usize {
        self.m
    }

    pub fn codeword_len(&self) -> usize {
        self.n
    }

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

    fn encode_fft(&self, data: &[u128], level: usize) -> Vec<u128> {
        if level == 0 {
            return self.base_multiply(data);
        }

        let chunk_len = data.len() / RADIX;
        let mut subresults = Vec::with_capacity(RADIX);
        for chunk_idx in 0..RADIX {
            let mut chunk = Vec::with_capacity(chunk_len);
            for j in 0..chunk_len {
                chunk.push(data[RADIX * j + chunk_idx]);
            }
            subresults.push(self.encode_fft(&chunk, level - 1));
        }

        let sub_len = subresults[0].len();
        let mut output = vec![0u128; sub_len * RADIX];
        let stride = self.n / output.len();
        for (idx, slot) in output.iter_mut().enumerate() {
            let column = idx % sub_len;
            let twiddles = &self.twiddle_powers[idx * stride];
            let mut acc = 0u128;
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

    fn base_multiply(&self, chunk: &[u128]) -> Vec<u128> {
        assert_eq!(
            chunk.len(),
            BASE_DIM,
            "Base multiply expects chunks of length {}",
            BASE_DIM
        );
        let mut output = vec![0u128; BASE_LEN];
        for (row_idx, matrix_row) in self.base_matrix.iter().enumerate() {
            let mut acc = 0u128;
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
}

// assume k = 1 for now
fn compute_twiddles(n: usize) -> Vec<[u128; RADIX]> {
    let mut table = vec![[0u128; RADIX]; n];
    let mut omega_power = 1u128;
    for row in 0..n {
        let mut current = 1u128;
        for col in 0..RADIX {
            table[row][col] = current;
            current = mod_mul(current, omega_power);
        }
        omega_power = mod_mul(omega_power, OMEGA);
    }
    table
}

fn compute_base_matrix() -> [[u128; BASE_DIM]; BASE_LEN] {
    let mut matrix = [[0u128; BASE_DIM]; BASE_LEN];
    let omega_64 = mod_pow(OMEGA, RADIX as u128);
    let mut base_power = 1u128;
    for row in 0..BASE_LEN {
        let mut accum = 1u128;
        for col in 0..BASE_DIM {
            matrix[row][col] = accum;
            accum = mod_mul(accum, base_power);
        }
        base_power = mod_mul(base_power, omega_64);
    }
    matrix
}
// only for preprocessing 
fn mod_mul(a: u128, b: u128) -> u128 {
    ((a % P) * (b % P)) % P
}

fn mod_pow(base: u128, exp: u128) -> u128 {
    let mut result = 1u128;
    let mut b = base % P;
    let mut e = exp;
    while e > 0 {
        if e & 1 == 1 {
            result = mod_mul(result, b);
        }
        b = mod_mul(b, b);
        e >>= 1;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::IprsCode;

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
}
