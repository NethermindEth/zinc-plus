#![allow(clippy::arithmetic_side_effects)] // arithmetic is safe due to bounded configuration parameters

use std::convert::TryFrom;

use crypto_primitives::PrimeField;
use num_traits::CheckedAdd;
use zinc_utils::{from_ref::FromRef, mul_by_scalar::MulByScalar};

use crate::{code::LinearCode, pcs::structs::ZipTypes};

#[derive(Debug, Clone, Copy)]
pub struct IprsConfig {
    pub k: usize,
    // Parameters defining the code dimensions:
    // m = number of columns in the input matrix
    // n = number of columns in the output matrix (code length)
    // radix = branching factor of the recursion
    // base_dim = number of columns in the base matrix
    // base_len = number of rows in the base matrix
    pub m: usize,
    pub n: usize,
    pub radix: usize,
    pub base_dim: usize,
    pub base_len: usize,
}

trait IprsHelpers<Twiddle> {
    fn build_twiddle_tables(&self) -> Vec<Vec<Vec<Twiddle>>>;
    fn compute_base_matrix(&self) -> Vec<Vec<Twiddle>>;
}

impl IprsConfig {
    pub fn new_any_m_default(m: usize) -> Self {
        const MAX_LOG_BASE_LEN: usize = 6;
        const LOG_RADIX: usize = 3;
        Self::new_any_m(m, MAX_LOG_BASE_LEN, LOG_RADIX)
    }

    pub fn new_any_m(m: usize, max_log_base_len: usize, log_radix: usize) -> Self {
        assert!(m.is_power_of_two(), "m must be a power of two");

        let n = m << 1;

        let mut k = 0;
        let mut base_len = 1 << max_log_base_len;
        while (base_len << (log_radix * k)) < n {
            k += 1;
        }

        base_len = n >> (log_radix * k);

        let radix = 1 << log_radix;
        let base_dim = base_len >> 1;
        Self {
            k,
            m,
            n,
            radix,
            base_dim,
            base_len,
        }
    }

    /// Create a code with custom modulus and generator. The pair must satisfy
    /// that `\omega` is a primitive `N`-th root of unity in \mathbb{Z}_p, where
    /// `N = 2^{6+3k}`.
    pub fn new(k: usize) -> Self {
        const LOG_BASE_LEN: usize = 6;
        const LOG_BASE_DIM: usize = LOG_BASE_LEN - 1;
        const LOG_RADIX: usize = 3;
        let m = 1usize << (LOG_BASE_DIM + LOG_RADIX * k);
        let n = 1usize << (LOG_BASE_LEN + LOG_RADIX * k);
        Self {
            k,
            m,
            n,
            radix: 1 << LOG_RADIX,
            base_dim: 1 << LOG_BASE_DIM,
            base_len: 1 << LOG_BASE_LEN,
        }
    }

    pub fn new_custom(
        k: usize,
        m: usize,
        n: usize,
        radix: usize,
        base_dim: usize,
        base_len: usize,
    ) -> Self {
        Self {
            k,
            m,
            n,
            radix,
            base_dim,
            base_len,
        }
    }
}

/// Pseudo Reed-Solomon encoder over the integers. Internally uses a
/// configurable radix NTT-style recursion with a base Vandermonde matrix sized
/// `base_len x base_dim` (defaults to 64x32).
#[derive(Debug, Clone)]
pub struct IprsCode<Zt: ZipTypes, const REP: usize> {
    pub cfg: IprsConfig,
    twiddle_tables: Vec<Vec<Vec<Zt::Twiddle>>>, /* Per-stage twiddle tables of size stage_len x
                                                 * radix */
    base_matrix: Vec<Vec<Zt::Twiddle>>, // base_len x base_dim Vandermonde block
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
        let row_cw = row.iter().map(Zt::Cw::from_ref).collect::<Vec<_>>();

        self.encode_ntt(&row_cw, self.cfg.k)
    }

    fn encode_ntt(&self, data: &[Zt::Cw], depth: usize) -> Vec<Zt::Cw> {
        if depth == 0 {
            // Base-case: multiply the base_dim-term vector by the precomputed
            // Vandermonde matrix to obtain base_len evaluation points.
            return self.base_multiply(data);
        }

        let radix = self.cfg.radix;
        let chunk_len = data.len() / radix;
        let mut subresults = Vec::with_capacity(radix);
        for chunk_idx in 0..radix {
            let mut chunk = Vec::with_capacity(chunk_len);
            for j in 0..chunk_len {
                chunk.push(data[radix * j + chunk_idx]);
            }
            // Recursively evaluate the "child" polynomial corresponding to
            // coset `x -> x + \omega^{chunk_idx}`.
            subresults.push(self.encode_ntt(&chunk, depth - 1));
        }

        self.combine_stage(&subresults, &self.twiddle_tables[depth - 1])
    }

    fn base_multiply(&self, chunk: &[Zt::Cw]) -> Vec<Zt::Cw> {
        let base_dim = self.cfg.base_dim;
        let base_len = self.cfg.base_len;
        assert_eq!(chunk.len(), base_dim);
        let mut output = vec![Zt::Cw::default(); base_len];
        for (row_idx, matrix_row) in self.base_matrix.iter().enumerate() {
            debug_assert_eq!(matrix_row.len(), base_dim);
            let mut acc = Zt::Cw::default();
            // Dot-product between the i-th row of the Vandermonde matrix and
            // the base_dim input coordinates.
            for col in 0..base_dim {
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
        twiddle_table: &[Vec<Zt::Twiddle>],
    ) -> Vec<Zt::Cw> {
        // Each index `idx` corresponds to a single output position and therefore
        // to a unique set of twiddle multipliers. We rely on the precomputed stage
        // table to avoid recomputing roots on the fly.
        let sub_len = subresults[0].len();
        let radix = self.cfg.radix;
        debug_assert_eq!(twiddle_table.len(), sub_len * radix);
        debug_assert_eq!(subresults.len(), radix);

        let mut output = vec![Zt::Cw::default(); sub_len * radix];
        for (idx, slot) in output.iter_mut().enumerate() {
            let column = idx % sub_len;
            let twiddles = &twiddle_table[idx];
            let mut acc = Zt::Cw::default();
            for branch in 0..radix {
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
    Zt::Twiddle: FromRef<i128>,
    IprsConfig: IprsHelpers<Zt::Twiddle>,
{
    type Config = IprsConfig;

    const REPETITION_FACTOR: usize = REP;

    /// Create a code with the default `(p, \omega)` pair e.g. `(7681, 7146)`,
    /// valid for `k = 1`.
    fn new(poly_size: usize, config: IprsConfig) -> Self {
        assert!(
            poly_size == config.m,
            "Polynomial size {} does not match expected row length {}",
            poly_size,
            config.m
        );

        assert!(
            config.n == config.m * REP,
            "Codeword length {} must equal row length {} times repetition factor {}",
            config.n,
            config.m,
            REP
        );

        let base_matrix = config.compute_base_matrix();
        let twiddle_tables = config.build_twiddle_tables();
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
        F: PrimeField + zinc_utils::from_ref::FromRef<F>,
    {
        todo!()
    }
}

// Precomputed table where the modulus satisfies that `\omega` is a primitive
// `n`-th root of unity in \mathbb{Z}_p`.
const ROOTS_OF_UNITY: &[(usize, i128, i128)] = &[
    (2, 3, 2),
    (4, 5, 2),
    (8, 17, 9),
    (16, 17, 3),
    (32, 97, 28),
    (64, 193, 125),
    (128, 257, 9),
    (256, 257, 3),
    (512, 7681, 7146),
    (1024, 12289, 10302),
    (2048, 12289, 1945),
    (4096, 12289, 1331),
    (8192, 40961, 243),
    (16384, 65537, 81),
    (32768, 65537, 9),
    (65536, 65537, 3),
    (131072, 786433, 213567),
    (262144, 786433, 1000),
    (524288, 5767169, 177147),
    (1048576, 7340033, 2187),
    (2097152, 23068673, 177147),
    (4194304, 104857601, 39193363),
];

fn p_and_root_of_unity(n: usize) -> (i128, i128) {
    ROOTS_OF_UNITY
        .iter()
        .find(|entry| entry.0 == n)
        .map(|(_, p, omega)| (*p, *omega))
        .unwrap_or_else(|| panic!("unsupported N: {}", n))
}

impl<Twiddle: FromRef<i128>> IprsHelpers<Twiddle> for IprsConfig {
    fn build_twiddle_tables(&self) -> Vec<Vec<Vec<Twiddle>>> {
        let (modulus, omega) = p_and_root_of_unity(self.n);
        build_twiddle_tables(self.k, self.n, self.radix, self.base_len, modulus, omega)
            .into_iter()
            .map(|stage| {
                stage
                    .into_iter()
                    .map(|twiddles| {
                        twiddles
                            .into_iter()
                            .map(|value| Twiddle::from_ref(&value))
                            .collect()
                    })
                    .collect()
            })
            .collect()
    }

    fn compute_base_matrix(&self) -> Vec<Vec<Twiddle>> {
        let (modulus, omega) = p_and_root_of_unity(self.n);
        compute_base_matrix(modulus, omega, self.n, self.base_len, self.base_dim)
            .into_iter()
            .map(|row| {
                row.into_iter()
                    .map(|value| Twiddle::from_ref(&value))
                    .collect()
            })
            .collect()
    }
}

// All the arithmetic below is not optimized as it's only used during setup.

/// Precompute stage-specific twiddle tables. Each table contains `stage_len`
/// entries where entry `i` stores the radix consecutive powers of
/// `(omega)^(stride * i)`. This matches the order used in the recursive NTT.
fn build_twiddle_tables(
    k: usize,
    n: usize,
    radix: usize,
    base_len: usize,
    modulus: i128,
    omega: i128,
) -> Vec<Vec<Vec<i128>>> {
    let mut tables = Vec::with_capacity(k);
    for depth in 1..=k {
        let depth_as_u32 = u32::try_from(depth).expect("depth fits within u32");
        let stage_len = base_len * radix.pow(depth_as_u32);
        let stride = u128::try_from(n / stage_len).expect("stride fits into u128");

        let mut stage = Vec::with_capacity(stage_len);
        for idx in 0..stage_len {
            let idx_u128 = u128::try_from(idx).expect("stage idx fits into u128");
            let twiddle_base = mod_pow_generic(omega, stride * idx_u128, modulus);
            let mut twiddles = Vec::with_capacity(radix);
            let mut power = 1i128;
            for _ in 0..radix {
                twiddles.push(to_balanced_rep(power, modulus));
                power = mod_mul_generic(power, twiddle_base, modulus);
            }
            stage.push(twiddles);
        }
        tables.push(stage);
    }
    tables
}

/// Build the base_len x base_dim Vandermonde block used at the leaves of the
/// recursion. The evaluation points follow the NTT ordering induced by `(n /
/// base_len)` strides.
fn compute_base_matrix(
    modulus: i128,
    omega: i128,
    n: usize,
    base_len: usize,
    base_dim: usize,
) -> Vec<Vec<i128>> {
    let mut matrix = vec![vec![0i128; base_dim]; base_len];
    // Step between successive evaluation points at the base (size-64) stage.
    let row_stride = u128::try_from(n / base_len).expect("stride fits into u128");
    let row_step = mod_pow_generic(omega, row_stride, modulus);
    let mut current = 1i128;
    for row in matrix.iter_mut() {
        let mut accum = 1i128;
        for slot in row.iter_mut() {
            *slot = accum;
            accum = mod_mul_generic(accum, current, modulus);
        }
        current = mod_mul_generic(current, row_step, modulus);
    }
    matrix
}

/// Compute the canonical representative of `value` modulo `modulus`.
fn canonical_mod(value: i128, modulus: i128) -> i128 {
    let mut r = value % modulus;
    if r < 0 {
        r += modulus;
    }
    r
}

/// Convert `value` to its balanced representation modulo `modulus`, i.e.
fn to_balanced_rep(value: i128, modulus: i128) -> i128 {
    let canonical = canonical_mod(value, modulus);
    let half = (modulus - 1) / 2;
    if canonical > half {
        canonical - modulus
    } else {
        canonical
    }
}

/// Multiply `a` and `b` modulo `modulus`.
fn mod_mul_generic(a: i128, b: i128, modulus: i128) -> i128 {
    let a_red = canonical_mod(a, modulus);
    let b_red = canonical_mod(b, modulus);
    canonical_mod(a_red * b_red, modulus)
}

/// Compute `base^exp mod modulus` using binary exponentiation.
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
    use crate::{code::{LinearCode, iprs::p_and_root_of_unity}, pcs::test_utils::TestZipTypes};
    use crypto_primitives::crypto_bigint_int::Int;
    use num_traits::{CheckedAdd, CheckedMul, CheckedRem, CheckedSub, One, Zero};
    use zinc_utils::from_ref::FromRef;

    const K_1_P: i128 = 7681;
    const K_1_OMEGA: i128 = 7146;
    const K_2_P: i128 = 12289;
    const K_2_OMEGA: i128 = 1331;

    const EVAL_LIMBS: usize = 4;
    const CW_LIMBS: usize = 4;
    const COMB_LIMBS: usize = 4;

    type TestZt = TestZipTypes<EVAL_LIMBS, CW_LIMBS, COMB_LIMBS>;
    type EvalInt = Int<EVAL_LIMBS>;
    type CwInt = Int<CW_LIMBS>;

    const REPETITION_FACTOR: usize = 2;

    fn make_code(k: usize) -> IprsCode<TestZt, REPETITION_FACTOR> {
        let cfg = IprsConfig::new(k);
        let row_len = cfg.m;
        LinearCode::new(row_len, cfg)
    }

    fn make_custom_code() -> IprsCode<TestZt, REPETITION_FACTOR> {
        let cfg = IprsConfig::new_custom(
            1,      // k
            1 << 8, // m
            1 << 9, // n
            4,      // radix
            1 << 6, // base_dim
            1 << 7, // base_len
        );
        let row_len = cfg.m;
        LinearCode::new(row_len, cfg)
    }

    fn make_any_m_code(m: usize) -> IprsCode<TestZt, REPETITION_FACTOR> {
        let cfg = IprsConfig::new_any_m_default(m);
        let row_len = cfg.m;
        LinearCode::new(row_len, cfg)
    }

    #[test]
    fn encode_has_expected_lengths() {
        let code = make_code(1);
        let input = vec![EvalInt::one(); code.row_len()];
        let output = code.encode(&input);
        assert_eq!(output.len(), code.codeword_len());
    }

    #[test]
    fn encode_is_deterministic() {
        let code = make_code(1);
        let mut input = vec![EvalInt::zero(); code.row_len()];
        for (idx, value) in input.iter_mut().enumerate() {
            let idx_i32 = i32::try_from(idx).expect("index fits into i32");
            *value = EvalInt::from(idx_i32.pow(3));
        }
        let first = code.encode(&input);
        let second = code.encode(&input);
        assert_eq!(first, second);
    }

    #[test]
    fn mod_p_matches_reduction() {
        let configs = [(1, K_1_P, K_1_OMEGA), (2, K_2_P, K_2_OMEGA)];

        for (k, modulus_val, omega_val) in configs {
            let code = make_code(k);
            let mut input = vec![EvalInt::zero(); code.row_len()];
            for (idx, value) in input.iter_mut().enumerate() {
                *value = EvalInt::from(((idx * 17 + 5) as i128).pow(2));
            }

            let wide = code.encode(&input);

            let modulus = CwInt::from(modulus_val);
            let omega = CwInt::from(omega_val);

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
    fn mod_p_matches_reduction_custom() {
        let code = make_custom_code();
        let mut input = vec![EvalInt::zero(); code.row_len()];
        for (idx, value) in input.iter_mut().enumerate() {
            *value = EvalInt::from(((idx * 17 + 5) as i128).pow(2));
        }

        let wide = code.encode(&input);

        let modulus = CwInt::from(K_1_P);
        let omega = CwInt::from(K_1_OMEGA);

        let mut padded = vec![CwInt::zero(); code.codeword_len()];
        for (dst, src) in padded.iter_mut().zip(input.iter()) {
            *dst = CwInt::from_ref(src);
        }
        radix2_ntt_mod_int(&mut padded, &modulus, &omega);

        for (w, n) in wide.iter().zip(padded.iter()) {
            assert_eq!(canonical_mod_int(w, &modulus), *n);
        }
    }

    #[test]
    fn mod_p_matches_reduction_any_m() {
        let m = 1 << 11; // any power of two
        let code = make_any_m_code(m);
        print!("Code config: {:?}", code.cfg);
        let mut input = vec![EvalInt::zero(); code.row_len()];
        for (idx, value) in input.iter_mut().enumerate() {
            *value = EvalInt::from(((idx * 17 + 5) as i128).pow(2));
        }   
        let wide = code.encode(&input);
        let (p, root_of_unity) = p_and_root_of_unity(code.cfg.n);
        let modulus = CwInt::from(p);
        let omega = CwInt::from(root_of_unity);
        let mut padded = vec![CwInt::zero(); code.codeword_len()];
        for (dst, src) in padded.iter_mut().zip(input.iter()) {
            *dst = CwInt::from_ref(src);
        }
        radix2_ntt_mod_int(&mut padded, &modulus, &omega);
        for (w, n) in wide.iter().zip(padded.iter()) {
            assert_eq!(canonical_mod_int(w, &modulus), *n);
        }
    }

    

    #[test]
    fn polynomial_multiplication_via_ntt_matches_naive() {
        let modulus_val = K_1_P;
        let modulus = CwInt::from(modulus_val);
        let n = 32usize;
        let base_order = 1usize << 9;
        assert_eq!(base_order % n, 0);
        let omega_val = super::mod_pow_generic(K_1_OMEGA, (base_order / n) as u128, modulus_val);
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
        let modulus_val = K_1_P;
        let modulus = CwInt::from(modulus_val);
        let n = 16usize;
        let base_order = 1usize << 9;
        assert_eq!(base_order % n, 0);
        let omega_val = super::mod_pow_generic(K_1_OMEGA, (base_order / n) as u128, modulus_val);
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
        let modulus_val = K_1_P;
        let modulus = CwInt::from(modulus_val);
        let n = 64usize;
        let base_order = 1usize << 9;
        assert_eq!(base_order % n, 0);
        let omega_val = super::mod_pow_generic(K_1_OMEGA, (base_order / n) as u128, modulus_val);
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

    #[test]
    fn roots_table_contains_valid_pairs() {
        for (n, modulus, omega) in super::ROOTS_OF_UNITY.iter() {
            let n_u128 = u128::try_from(*n).expect("n fits into u128");
            assert_eq!(
                super::mod_pow_generic(*omega, n_u128, *modulus),
                1,
                "omega^n != 1 mod p for n={}",
                n
            );
            if *n > 2 {
                assert_ne!(
                    super::mod_pow_generic(*omega, n_u128 / 2, *modulus),
                    1,
                    "omega not primitive for n={}",
                    n
                );
            }
        }
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
        let modulus_pow = u128::try_from(modulus_scalar - 2).expect("modulus >= 2");
        let omega_inv = mod_pow_int(omega, modulus_pow, modulus);
        radix2_ntt_mod_int(values, modulus, &omega_inv);
        let n_int = Int::<LIMBS>::from(n as i128);
        let inv_n = mod_pow_int(&n_int, modulus_pow, modulus);
        for coeff in values.iter_mut() {
            *coeff = mod_mul_int(coeff, &inv_n, modulus);
        }
    }
}
