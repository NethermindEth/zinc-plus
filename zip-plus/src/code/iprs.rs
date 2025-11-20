const RADIX: usize = 8;
const BASE_DIM: usize = 32;
const BASE_LEN: usize = 64;

// Fixed prime / primitive root pair requested by the user (valid for k = 1).
const P: u128 = 7681;
const OMEGA: u128 = 7146;

/// If we want to support k = 2 with a fully valid root of unity, the pair
/// (p = 12289, ω = 1331) works because 12289 − 1 = 3 · 4096, so ω has order
/// 4096 over ℤₚ.
#[derive(Debug, Clone)]
pub struct IprsCode {
    k: usize,
    m: usize,
    n: usize,
    twiddle_tables: Vec<Vec<[u128; RADIX]>>,
    base_matrix: [[u128; BASE_DIM]; BASE_LEN],
}

impl IprsCode {
    pub fn new(k: usize) -> Self {
        assert!(k > 0, "k must be positive");
        let m = 1usize << (5 + 3 * k);
        let n = 1usize << (6 + 3 * k);
        let twiddle_tables = build_twiddle_tables(k, n);
        let base_matrix = compute_base_matrix();
        Self {
            k,
            m,
            n,
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
            return self.base_multiply(data);
        }

        let chunk_len = data.len() / RADIX;
        let mut subresults = Vec::with_capacity(RADIX);
        for chunk_idx in 0..RADIX {
            let mut chunk = Vec::with_capacity(chunk_len);
            for j in 0..chunk_len {
                chunk.push(data[RADIX * j + chunk_idx]);
            }
            subresults.push(self.encode_fft(&chunk, depth - 1));
        }

        combine_stage(
            &subresults,
            &self.twiddle_tables[depth - 1],
            AddMode::Wide,
        )
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
                chunk.push(data[RADIX * j + chunk_idx] % P);
            }
            subresults.push(self.encode_fft_mod(&chunk, depth - 1));
        }

        combine_stage(
            &subresults,
            &self.twiddle_tables[depth - 1],
            AddMode::Modulo,
        )
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

    fn base_multiply_mod(&self, chunk: &[u128]) -> Vec<u128> {
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
                let term = mod_mul(chunk[col] % P, matrix_row[col]);
                acc = mod_add(acc, term);
            }
            output[row_idx] = acc;
        }
        output
    }
}

enum AddMode {
    Wide,
    Modulo,
}

fn combine_stage(
    subresults: &[Vec<u128>],
    twiddle_table: &[[u128; RADIX]],
    mode: AddMode,
) -> Vec<u128> {
    let sub_len = subresults[0].len();
    debug_assert_eq!(twiddle_table.len(), sub_len * RADIX);

    let mut output = vec![0u128; sub_len * RADIX];
    for (idx, slot) in output.iter_mut().enumerate() {
        let column = idx % sub_len;
        let twiddles = &twiddle_table[idx];
        let mut acc = 0u128;
        match mode {
            AddMode::Wide => {
                for branch in 0..RADIX {
                    let term = subresults[branch][column]
                        .checked_mul(twiddles[branch])
                        .expect("Multiplication overflow");
                    acc = acc.checked_add(term).expect("Addition overflow");
                }
                *slot = acc;
            }
            AddMode::Modulo => {
                for branch in 0..RADIX {
                    let term = mod_mul(subresults[branch][column], twiddles[branch]);
                    acc = mod_add(acc, term);
                }
                *slot = acc;
            }
        }
    }

    output
}

fn build_twiddle_tables(k: usize, n: usize) -> Vec<Vec<[u128; RADIX]>> {
    let mut tables = Vec::with_capacity(k);
    for depth in 1..=k {
        let stage_len = BASE_LEN * RADIX.pow(depth as u32);
        let stride = (n / stage_len) as u128;

        let mut stage = Vec::with_capacity(stage_len);
        for idx in 0..stage_len {
            let twiddle_base = mod_pow(OMEGA, stride * idx as u128);
            let mut twiddles = [0u128; RADIX];
            let mut power = 1u128;
            for slot in 0..RADIX {
                twiddles[slot] = power;
                power = mod_mul(power, twiddle_base);
            }
            stage.push(twiddles);
        }
        tables.push(stage);
    }
    tables
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

fn mod_mul(a: u128, b: u128) -> u128 {
    ((a % P) * (b % P)) % P
}

fn mod_add(a: u128, b: u128) -> u128 {
    let tmp = (a % P) + (b % P);
    if tmp >= P {
        tmp - P
    } else {
        tmp
    }
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
    use super::{mod_mul, mod_pow, IprsCode, P};

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
            assert_eq!(n, &(w % P));
        }
    }

    #[test]
    fn matches_radix2_ntt_mod_p() {
        let code = IprsCode::new(1);
        let mut input = vec![0u128; code.row_len()];
        for (idx, value) in input.iter_mut().enumerate() {
            *value = ((idx * 31 + 7) as u128) % P;
        }

        let mut padded = vec![0u128; code.codeword_len()];
        padded[..input.len()].copy_from_slice(&input);
        radix2_ntt(&mut padded);

        let ours = code.encode_mod_p(&input);
        assert_eq!(ours, padded);
    }

    fn radix2_ntt(values: &mut [u128]) {
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
            let w_len = mod_pow(super::OMEGA, (n / len) as u128);
            for chunk in values.chunks_mut(len) {
                let mut w = 1u128;
                for i in 0..len / 2 {
                    let u = chunk[i];
                    let v = mod_mul(chunk[i + len / 2], w);
                    chunk[i] = mod_add(u, v);
                    chunk[i + len / 2] = mod_sub(u, v);
                    w = mod_mul(w, w_len);
                }
            }
            len <<= 1;
        }
    }

    fn mod_add(a: u128, b: u128) -> u128 {
        let tmp = (a % P) + (b % P);
        if tmp >= P {
            tmp - P
        } else {
            tmp
        }
    }

    fn mod_sub(a: u128, b: u128) -> u128 {
        if a >= b {
            a - b
        } else {
            a + P - b
        }
    }
}
