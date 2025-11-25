use std::{array::from_fn};

use num_traits::CheckedAdd;
use zinc_utils::{from_ref::FromRef, mul_by_scalar::MulByScalar};

use crate::{code::LinearCode, pcs::structs::ZipTypes};

/// Number of branches used in each NTT layer (radix). The construction fixes
/// this to 8 in accordance with the pseudo-Reed–Solomon spec.
const RADIX: usize = 8;
/// Base-case input length handled by the Vandermonde block.
const BASE_DIM: usize = 32;
/// Base-case output length (number of evaluation points).
const BASE_LEN: usize = 64;

#[derive(Debug, Clone, Copy)]
pub struct IprsConfig {
    pub modulus: i128,
    pub omega: i128,
    pub k: usize,
    pub m: usize,
    pub n: usize,
}

impl IprsConfig {
    /// Create a code with custom modulus and generator. The pair must satisfy
    /// that `ω` is a primitive `N`-th root of unity in ℤ_p, where `N = 2^{6+3k}`.
    pub fn new(k: usize, modulus: i128, omega: i128) -> Self {
        let m = 1usize << (5 + 3 * k);
        let n = 1usize << (6 + 3 * k);
        Self {
            modulus,
            omega,
            k,
            m,
            n,
        }
    }
}

/// Pseudo Reed-Solomon encoder over the integers. Internally uses a radix-8
/// NTT-style recursion with a base Vandermonde matrix of dimensions 64x32.
#[derive(Debug, Clone)]
pub struct IprsCode<Zt: ZipTypes, const REP: usize> {
    pub cfg: IprsConfig,
    twiddle_tables: Vec<Vec<[Zt::Twiddle; RADIX]>>,
    base_matrix: [[Zt::Twiddle; BASE_DIM]; BASE_LEN],
}

impl<Zt: ZipTypes, const REP: usize> IprsCode<Zt, REP> {
    /// Encode without modular reduction, purely over the integers.
    fn encode_inner(&self, row: &[Zt::Eval]) -> Vec<Zt::Cw> {
        assert_eq!(
            row.len(),
            self.cfg.m,
            "Input length {} does not match expected row length {}",
            row.len(),
            self.cfg.m
        );
        let row_cw = row.iter().map(|v| Zt::Cw::from_ref(v)).collect::<Vec<_>>();

        self.encode_ntt(&row_cw, self.cfg.k)
    }

    fn encode_ntt(&self, data: &[Zt::Cw], depth: usize) -> Vec<Zt::Cw> {
        if depth == 0 {
            // Base-case: multiply the 32-term vector by the precomputed
            // Vandermonde matrix to obtain 64 evaluation points.
            return self.base_multiply(&data);
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

        self.combine_stage(&subresults, &self.twiddle_tables[depth - 1])
    }

    fn base_multiply(&self, chunk: &[Zt::Cw]) -> Vec<Zt::Cw> {
        assert_eq!(chunk.len(), BASE_DIM);
        let mut output = vec![Zt::Cw::default(); BASE_LEN];
        for (row_idx, matrix_row) in self.base_matrix.iter().enumerate() {
            let mut acc = Zt::Cw::default();
            // Dot-product between the i-th row of the Vandermonde matrix and
            // the 32 input coordinates.
            for col in 0..BASE_DIM {
                let term = chunk[col]
                    .mul_by_scalar(&matrix_row[col])
                    .expect("Base multiplication overflow");
                acc = acc.checked_add(&term).expect("Base addition overflow");
            }
            output[row_idx] = acc;
        }
        output
    }

    fn combine_stage(
        &self,
        subresults: &[Vec<Zt::Cw>],
        twiddle_table: &[[Zt::Twiddle; RADIX]],
    ) -> Vec<Zt::Cw> {
        // Each index `idx` corresponds to a single output position and therefore
        // to a unique set of twiddle multipliers. We rely on the precomputed stage
        // table to avoid recomputing roots on the fly.
        let sub_len = subresults[0].len();
        debug_assert_eq!(twiddle_table.len(), sub_len * RADIX);

        let mut output = vec![Zt::Cw::default(); sub_len * RADIX];
        for (idx, slot) in output.iter_mut().enumerate() {
            let column = idx % sub_len;
            let twiddles = &twiddle_table[idx];
            let mut acc = Zt::Cw::default();
            for branch in 0..RADIX {
                let term = subresults[branch][column]
                    .mul_by_scalar(&twiddles[branch])
                    .expect("Multiplication overflow");
                acc = acc.checked_add(&term).expect("Addition overflow");
            }
            *slot = acc;
        }
        output
    }
}

impl<Zt: ZipTypes, const REP: usize> LinearCode<Zt> for IprsCode<Zt, REP>
where
    Zt: ZipTypes,
    // For simplicity, we require that the Twiddle type can be created from i128

    Zt::Twiddle: FromRef<i128>
{
    type Config = IprsConfig;

    const REPETITION_FACTOR: usize = REP; 

    /// Create a code with the default `(p, ω)` pair e.g. `(7681, 7146)`, valid for `k = 1`.
    fn new(poly_size: usize, config: IprsConfig) -> Self {
        assert!(
            poly_size == config.m,
            "Polynomial size {} does not match expected row length {}",
            poly_size,
            config.m
        );
        let base_matrix_raw = compute_base_matrix(config.modulus, config.omega, config.n);
        let twiddle_tables_raw =
            build_twiddle_tables(config.k, config.n, config.modulus, config.omega);
        let base_matrix =
            from_fn(|row| from_fn(|col| Zt::Twiddle::from_ref(&base_matrix_raw[row][col])));
        let twiddle_tables = twiddle_tables_raw
            .into_iter()
            .map(|stage| {
                stage
                    .into_iter()
                    .map(|entry| entry.map(|value| Zt::Twiddle::from_ref(&value)))
                    .collect()
            })
            .collect();
        Self {
            cfg: config,
            twiddle_tables,
            base_matrix,
        }
    }

    fn encode(&self, row: &[Zt::Eval]) -> Vec<Zt::Cw> {
        assert!(
            row.len() == self.cfg.m,
            "Input length {} does not match expected row length {}",
            row.len(),
            self.cfg.m
        );
        self.encode_inner(row)
    }

    fn row_len(&self) -> usize {
        self.cfg.m
    }

    fn codeword_len(&self) -> usize {
        self.cfg.n
    }

    fn encode_wide(&self, _row: &[<Zt as ZipTypes>::CombR]) -> Vec<<Zt as ZipTypes>::CombR> {
        todo!()
    }

    fn encode_f<F>(&self, _row: &[F]) -> Vec<F>
    where
        F: crypto_primitives::PrimeField + zinc_utils::from_ref::FromRef<F>,
    {
        todo!()
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
    use super::{IprsCode, IprsConfig};
    use crate::{code::LinearCode, pcs::test_utils::TestZipTypes};
    use crypto_primitives::crypto_bigint_int::Int;
    use num_traits::{CheckedAdd, CheckedMul, CheckedRem, CheckedSub, One, Zero};
    use zinc_utils::from_ref::FromRef;

    const DEFAULT_P: i128 = 7681;
    const DEFAULT_OMEGA: i128 = 7146;
    const ALT_P_K2: i128 = 12289;
    const ALT_OMEGA_K2: i128 = 1331;

    const EVAL_LIMBS: usize = 4;
    const CW_LIMBS: usize = 4;
    const COMB_LIMBS: usize = 4;

    type TestZt = TestZipTypes<EVAL_LIMBS, CW_LIMBS, COMB_LIMBS>;
    type EvalInt = Int<EVAL_LIMBS>;
    type CwInt = Int<CW_LIMBS>;

    const REPETITION_FACTOR: usize = 4;


    fn make_code(k: usize, modulus: i128, omega: i128) -> IprsCode<TestZt, REPETITION_FACTOR> {
        let cfg = IprsConfig::new(k, modulus, omega);
        let row_len = cfg.m;
        LinearCode::new(row_len, cfg)
    }

    #[test]
    fn encode_has_expected_lengths() {
        let code = make_code(1, DEFAULT_P, DEFAULT_OMEGA);
        let input = vec![EvalInt::one(); code.row_len()];
        let output = code.encode(&input);
        assert_eq!(output.len(), code.codeword_len());
    }

    #[test]
    fn encode_is_deterministic() {
        let code = make_code(1, DEFAULT_P, DEFAULT_OMEGA);
        let mut input = vec![EvalInt::zero(); code.row_len()];
        for (idx, value) in input.iter_mut().enumerate() {
            *value = EvalInt::from((idx as i32).pow(3));
        }
        let first = code.encode(&input);
        let second = code.encode(&input);
        assert_eq!(first, second);
    }

    #[test]
    fn mod_p_matches_reduction() {
        let configs = [(1, DEFAULT_P, DEFAULT_OMEGA), (2, ALT_P_K2, ALT_OMEGA_K2)];

        for (k, modulus_val, omega_val) in configs {
            let code = make_code(k, modulus_val, omega_val);
            let mut input = vec![EvalInt::zero(); code.row_len()];
            for (idx, value) in input.iter_mut().enumerate() {
                *value = EvalInt::from(((idx * 17 + 5) as i128).pow(2));
            }

            let wide = code.encode(&input);

            let modulus = CwInt::from(code.cfg.modulus);
            let omega = CwInt::from(code.cfg.omega);

            let mut padded = vec![CwInt::zero(); code.codeword_len()];
            for (dst, src) in padded.iter_mut().zip(input.iter()) {
                *dst = CwInt::from_ref(src);
            }
            radix2_ntt_mod_int(&mut padded, &modulus, &omega);

            for (w, n) in wide.iter().zip(padded.iter()) {
                assert_eq!(canonical_mod_int(w, &modulus), *n);
            }
        }
    }

    #[test]
    fn polynomial_multiplication_via_ntt_matches_naive() {
        let modulus_val = DEFAULT_P;
        let modulus = CwInt::from(modulus_val);
        let n = 32usize;
        let base_order = 1usize << (6 + 3 * 1);
        assert_eq!(base_order % n, 0);
        let omega_val =
            super::mod_pow_generic(DEFAULT_OMEGA, (base_order / n) as u128, modulus_val);
        let omega = CwInt::from(omega_val);

        let mut poly_a = vec![CwInt::zero(); 16];
        let mut poly_b = vec![CwInt::zero(); 16];
        for (i, coeff) in poly_a.iter_mut().enumerate() {
            *coeff = CwInt::from(((i * 13 + 7) as i128) % modulus_val);
        }
        for (i, coeff) in poly_b.iter_mut().enumerate() {
            *coeff = CwInt::from(((i * 9 + 5) as i128) % modulus_val);
        }

        let mut naive = vec![CwInt::zero(); poly_a.len() + poly_b.len() - 1];
        for (i, a) in poly_a.iter().enumerate() {
            for (j, b) in poly_b.iter().enumerate() {
                let idx = i + j;
                let prod = mod_mul_int(a, b, &modulus);
                naive[idx] = mod_add_int(&naive[idx], &prod, &modulus);
            }
        }

        let mut a_ntt = vec![CwInt::zero(); n];
        let mut b_ntt = vec![CwInt::zero(); n];
        a_ntt[..poly_a.len()].copy_from_slice(&poly_a);
        b_ntt[..poly_b.len()].copy_from_slice(&poly_b);

        radix2_ntt_mod_int(&mut a_ntt, &modulus, &omega);
        radix2_ntt_mod_int(&mut b_ntt, &modulus, &omega);
        for (a_coeff, b_coeff) in a_ntt.iter_mut().zip(b_ntt.iter()) {
            *a_coeff = mod_mul_int(a_coeff, b_coeff, &modulus);
        }
        radix2_intt_mod_int(&mut a_ntt, &modulus, &omega, modulus_val);

        assert_eq!(&a_ntt[..naive.len()], naive.as_slice());
    }

    #[test]
    fn cyclic_polynomial_multiplication_via_ntt_matches_naive() {
        let modulus_val = DEFAULT_P;
        let modulus = CwInt::from(modulus_val);
        let n = 16usize;
        let base_order = 1usize << (6 + 3 * 1);
        assert_eq!(base_order % n, 0);
        let omega_val =
            super::mod_pow_generic(DEFAULT_OMEGA, (base_order / n) as u128, modulus_val);
        let omega = CwInt::from(omega_val);

        let mut poly_a = vec![CwInt::zero(); 16];
        let mut poly_b = vec![CwInt::zero(); 16];
        for (i, coeff) in poly_a.iter_mut().enumerate() {
            *coeff = CwInt::from(((i * 11 + 3) as i128) % modulus_val);
        }
        for (i, coeff) in poly_b.iter_mut().enumerate() {
            *coeff = CwInt::from(((i * 7 + 2) as i128) % modulus_val);
        }

        let mut naive = vec![CwInt::zero(); n];
        for (i, a) in poly_a.iter().enumerate() {
            for (j, b) in poly_b.iter().enumerate() {
                let idx = (i + j) % n;
                let prod = mod_mul_int(a, b, &modulus);
                naive[idx] = mod_add_int(&naive[idx], &prod, &modulus);
            }
        }

        let mut a_ntt = vec![CwInt::zero(); n];
        let mut b_ntt = vec![CwInt::zero(); n];
        a_ntt[..poly_a.len()].copy_from_slice(&poly_a);
        b_ntt[..poly_b.len()].copy_from_slice(&poly_b);

        radix2_ntt_mod_int(&mut a_ntt, &modulus, &omega);
        radix2_ntt_mod_int(&mut b_ntt, &modulus, &omega);
        for (a_coeff, b_coeff) in a_ntt.iter_mut().zip(b_ntt.iter()) {
            *a_coeff = mod_mul_int(a_coeff, b_coeff, &modulus);
        }
        radix2_intt_mod_int(&mut a_ntt, &modulus, &omega, modulus_val);

        assert_eq!(a_ntt, naive);
    }

    #[test]
    fn ntt_goes_back_and_forth() {
        let modulus_val = DEFAULT_P;
        let modulus = CwInt::from(modulus_val);
        let n = 64usize;
        let base_order = 1usize << (6 + 3 * 1);
        assert_eq!(base_order % n, 0);
        let omega_val =
            super::mod_pow_generic(DEFAULT_OMEGA, (base_order / n) as u128, modulus_val);
        let omega = CwInt::from(omega_val);

        let mut data = vec![CwInt::zero(); n];
        for (i, coeff) in data.iter_mut().enumerate() {
            *coeff = CwInt::from(((i * 19 + 11) as i128) % modulus_val);
        }

        let original = data.clone();
        radix2_ntt_mod_int(&mut data, &modulus, &omega);
        radix2_intt_mod_int(&mut data, &modulus, &omega, modulus_val);

        assert_eq!(data, original);
    }

    fn canonical_mod_int<const LIMBS: usize>(
        value: &Int<LIMBS>,
        modulus: &Int<LIMBS>,
    ) -> Int<LIMBS> {
        let mut rem = value.checked_rem(modulus).expect("remainder");
        if rem < Int::<LIMBS>::zero() {
            rem = rem.checked_add(modulus).expect("addition overflow");
        }
        rem
    }

    fn mod_add_int<const LIMBS: usize>(
        a: &Int<LIMBS>,
        b: &Int<LIMBS>,
        modulus: &Int<LIMBS>,
    ) -> Int<LIMBS> {
        let sum = a.checked_add(b).expect("addition overflow");
        canonical_mod_int(&sum, modulus)
    }

    fn mod_sub_int<const LIMBS: usize>(
        a: &Int<LIMBS>,
        b: &Int<LIMBS>,
        modulus: &Int<LIMBS>,
    ) -> Int<LIMBS> {
        let diff = if let Some(val) = a.checked_sub(b) {
            val
        } else {
            a.checked_add(modulus)
                .expect("addition overflow")
                .checked_sub(b)
                .expect("subtraction overflow")
        };
        canonical_mod_int(&diff, modulus)
    }

    fn mod_mul_int<const LIMBS: usize>(
        a: &Int<LIMBS>,
        b: &Int<LIMBS>,
        modulus: &Int<LIMBS>,
    ) -> Int<LIMBS> {
        let prod = a.checked_mul(b).expect("multiplication overflow");
        canonical_mod_int(&prod, modulus)
    }

    fn mod_pow_int<const LIMBS: usize>(
        base: &Int<LIMBS>,
        exp: u128,
        modulus: &Int<LIMBS>,
    ) -> Int<LIMBS> {
        let mut result = Int::<LIMBS>::one();
        let mut b = canonical_mod_int(base, modulus);
        let mut e = exp;
        while e > 0 {
            if e & 1 == 1 {
                result = mod_mul_int(&result, &b, modulus);
            }
            b = mod_mul_int(&b, &b, modulus);
            e >>= 1;
        }
        result
    }

    fn radix2_ntt_mod_int<const LIMBS: usize>(
        values: &mut [Int<LIMBS>],
        modulus: &Int<LIMBS>,
        omega: &Int<LIMBS>,
    ) {
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
            let w_len = mod_pow_int(omega, (n / len) as u128, modulus);
            for chunk in values.chunks_mut(len) {
                let mut w = Int::<LIMBS>::one();
                for i in 0..len / 2 {
                    let u = chunk[i];
                    let v = mod_mul_int(&chunk[i + len / 2], &w, modulus);
                    chunk[i] = mod_add_int(&u, &v, modulus);
                    chunk[i + len / 2] = mod_sub_int(&u, &v, modulus);
                    w = mod_mul_int(&w, &w_len, modulus);
                }
            }
            len <<= 1;
        }
    }

    fn radix2_intt_mod_int<const LIMBS: usize>(
        values: &mut [Int<LIMBS>],
        modulus: &Int<LIMBS>,
        omega: &Int<LIMBS>,
        modulus_scalar: i128,
    ) {
        let n = values.len();
        assert!(n.is_power_of_two());
        let modulus_pow = (modulus_scalar - 2) as u128;
        let omega_inv = mod_pow_int(omega, modulus_pow, modulus);
        radix2_ntt_mod_int(values, modulus, &omega_inv);
        let n_int = Int::<LIMBS>::from(n as i128);
        let inv_n = mod_pow_int(&n_int, modulus_pow, modulus);
        for coeff in values.iter_mut() {
            *coeff = mod_mul_int(coeff, &inv_n, modulus);
        }
    }
}
