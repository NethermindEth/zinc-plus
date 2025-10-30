use crate::{
    code::LinearCode,
    pcs::structs::{MulByScalar, ZipTypes},
    traits::FromRef,
};
use crypto_primitives::PrimeField;
use num_traits::{WrappingAdd, Zero};
use std::{
    fmt::Debug,
    ops::{Add, AddAssign, BitAnd, Mul},
};
use ark_std::{cfg_iter, cfg_iter_mut};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

// Parameters for k=1
const K: usize = 1;

// Prime field parameters used only to derive the integer constants.
const P: i64 = 7681;
const OMEGA: i64 = 7146; // 512th root of unity

pub const RADIX: usize = 8;

// Coefficient growth analysis for k=1:
// Max magnitude ≈ 2^33.8. Required bit depth T=34 for polynomial bit-slicing.
pub const T: usize = 34;

pub const EVAL_LEN: usize = 1 << (5 + 3 * K); // 256
pub const CW_LEN: usize = 1 << (6 + 3 * K); // 512
pub const BASE_DIM: usize = EVAL_LEN / RADIX; // 32
pub const BASE_LEN: usize = CW_LEN / RADIX; // 64

// Type aliases for constants
pub type Matrix64x32 = [[i64; BASE_DIM]; BASE_LEN];
pub type TwiddleTable = [[i64; RADIX]; CW_LEN];

pub struct ReedSolomon {
    // M': 64x32 Vandermonde matrix.
    M_PRIME: Matrix64x32,
    // Twiddle factors for the DIT combination.
    DIT_TWIDDLES: TwiddleTable,

    row_len: usize,
}

impl<Zt: ZipTypes> LinearCode<Zt> for ReedSolomon {
    type Config = ();

    const REPETITION_FACTOR: usize = 2;

    fn new(poly_size: usize, _cfg: Self::Config) -> Self {
        // Taken from original Zip codes
        let num_vars = poly_size.ilog2();
        let two_pow_num_vars = 1_usize
            .checked_shl(num_vars)
            .expect("2 ** num_vars overflows");
        let row_len: usize = two_pow_num_vars
            .isqrt()
            .checked_next_power_of_two()
            .expect("row_len overflow");

        assert_eq!(row_len, EVAL_LEN, "This is a proof-of-concept implementation for 2^16 polynomials only");
        ReedSolomon {
            M_PRIME: compute_m_prime(),
            DIT_TWIDDLES: compute_dit_twiddles(),
            row_len,
        }
    }

    fn row_len(&self) -> usize {
        self.row_len
    }

    fn codeword_len(&self) -> usize {
        self.row_len * 2
    }

    fn encode(&self, row: &[Zt::Eval]) -> Vec<Zt::Cw> {
        self.encode_vector(row, &Zero::zero())
    }

    fn encode_wide(&self, row: &[Zt::CombR]) -> Vec<Zt::CombR> {
        self.encode_vector(row, &Zero::zero())
    }

    fn encode_f<F>(&self, row: &[F]) -> Vec<F>
    where
        F: PrimeField + FromRef<F> + MulByScalar<i64>,
    {
        let zero = F::zero_with_cfg(row[0].cfg());
        self.encode_vector(row, &zero)
    }
}

// Helper function for modular exponentiation (used only for initialization)
fn mod_pow(mut base: i64, mut exp: usize, modulus: i64) -> i64 {
    let mut result = 1;
    base %= modulus;
    while exp > 0 {
        if exp % 2 == 1 {
            result = (result * base) % modulus;
        }
        exp >>= 1;
        base = (base * base) % modulus;
    }
    result
}

fn compute_m_prime() -> Matrix64x32 {
    let omega_64 = mod_pow(OMEGA, RADIX, P);
    let mut m_prime = [[0; BASE_DIM]; BASE_LEN];

    let mut base_power = 1;
    for i in 0..BASE_LEN {
        let mut power = 1;
        for j in 0..BASE_DIM {
            m_prime[i][j] = power;
            power = (power * base_power) % P;
        }
        base_power = (base_power * omega_64) % P;
    }
    m_prime
}

fn compute_dit_twiddles() -> TwiddleTable {
    let mut table = [[0; RADIX]; CW_LEN];
    for k in 0..CW_LEN {
        let omega_k = mod_pow(OMEGA, k, P);
        let mut omega_ki = 1;
        for i in 0..RADIX {
            table[k][i] = omega_ki;
            omega_ki = (omega_ki * omega_k) % P;
        }
    }
    table
}

impl ReedSolomon {
    /// Encode M vectors in parallel. Generic over T and D.
    pub fn encode_all_vectors<InPoly, OutPoly>(
        &self,
        inputs: &[&[InPoly]],
        zero: &OutPoly,
    ) -> Vec<Vec<OutPoly>>
    where
        InPoly: Send + Sync,
        OutPoly:
            Add<Output = OutPoly> + AddAssign + MulByScalar<i64> + FromRef<InPoly> + Debug + Clone + Send + Sync,
    {
        assert_eq!(inputs.len(), EVAL_LEN);
        cfg_iter!(inputs)
            .map(|v| self.encode_vector(v, zero))
            .collect()
    }

    /// The main encoding function for a single vector of polynomials.
    pub fn encode_vector<InPoly, OutPoly>(&self, evals: &[InPoly], zero: &OutPoly) -> Vec<OutPoly>
    where
        OutPoly:
            Add<Output = OutPoly> + AddAssign + MulByScalar<i64> + FromRef<InPoly> + Debug + Clone + Send + Sync,
    {
        assert_eq!(evals.len(), EVAL_LEN);

        // 1. Split input by striding (DIT) and convert to NaivePolynomial
        //    representation.
        let mut intermediates_input: [[OutPoly; BASE_DIM]; RADIX] =
            vec![vec![zero.clone(); BASE_DIM].try_into().unwrap(); RADIX]
                .try_into()
                .unwrap();
        for j in 0..BASE_DIM {
            for i in 0..RADIX {
                // Conversion handles unpacking the u32 into coefficients.
                intermediates_input[i][j] = OutPoly::from_ref(&evals[RADIX * j + i]);
            }
        }

        // 2. Compute U_i = M' * V_i (Base encoding). Parallelized over the 8 chunks.
        let intermediates: Vec<[OutPoly; BASE_LEN]> = cfg_iter!(intermediates_input)
            .map(|v_i| self.matrix_multiply_base(v_i, zero))
            .collect();

        // 3. Combine using DIT structure. Parallelized over the 512 output elements.
        let mut encoded: Vec<OutPoly> = vec![zero.clone(); CW_LEN];
        let dit_twiddles = &self.DIT_TWIDDLES;

        cfg_iter_mut!(encoded).enumerate().for_each(|(k, val)| {
            let k_mod_64 = k % BASE_LEN;
            let mut sum = zero.clone();
            let twiddles_k = &dit_twiddles[k];

            for i in 0..RADIX {
                let twiddle = twiddles_k[i];
                let u_i_k = intermediates[i][k_mod_64].clone();

                let product = u_i_k.mul_by_scalar(twiddle).unwrap();
                sum += product;
            }
            *val = sum;
        });

        encoded
    }

    // Base encoding: M' * v.
    fn matrix_multiply_base<OutPoly>(
        &self,
        v: &[OutPoly; BASE_DIM],
        zero: &OutPoly,
    ) -> [OutPoly; BASE_LEN]
    where
        OutPoly: Add<Output = OutPoly> + MulByScalar<i64> + AddAssign + Debug + Clone + Send + Sync,
    {
        let mut output: [OutPoly; BASE_LEN] = vec![zero.clone(); BASE_LEN].try_into().unwrap();
        let m_prime = &self.M_PRIME;

        // Parallelize over the 64 rows.
        cfg_iter_mut!(output).enumerate().for_each(|(i, out_val)| {
            let row = &m_prime[i];
            let mut sum = zero.clone();

            for j in 0..BASE_DIM {
                // Uses NaivePolynomial Mul<i64> and AddAssign.
                let product = v[j].mul_by_scalar(row[j]).unwrap();
                sum += product;
            }
            *out_val = sum;
        });

        output
    }
}

// // The main encoding function. Generic over T and D.
// pub fn encode_vector<T, OutCoeff, OutPoly, InterCoeff, InterPoly, const D:
// usize>(rs: &ReedSolomon, v: &InputVector<T>) -> EncodedVector<OutPoly> where
//     T: WithBitMask + WrappingAdd + BitAnd<T, Output = T> + Zero + From<u32> +
// Clone + Copy,     InterCoeff: FromRef<T> + WrappingAdd,
//     InterPoly: Polynomial<InterCoeff> + MulByScalar<u32> + Zero + Clone,
//     OutPoly: Polynomial<OutCoeff> + AddAssign + FromRef<InterPoly> + Zero +
// Clone, {
//     assert_eq!(v.len(), M);
//
//     // 1. Split input by striding (DIT). Keep as raw T. (Small stack
// allocation, OK)     let mut intermediates_input = [[T::zero(); BASE_DIM];
// RADIX];     for j in 0..BASE_DIM {
//         for i in 0..RADIX {
//             intermediates_input[i][j] = v[RADIX * j + i].clone();
//         }
//     }
//
//     // 2. Compute U_i = M' * V_i using the optimized base encoding.
//     // FIX: Move this large allocation (e.g., 128KB for D64) to the heap.
//     // Use map().collect() for safe initialization to avoid large stack
// temporaries.     let mut intermediates_aos: Vec<[InterPoly; BASE_LEN]> =
// vec![std::array::from_fn::<_, BASE_LEN, _>(|_| InterPoly::zero()); RADIX];
//
//     cfg_iter_mut!(intermediates_aos).zip(cfg_iter!(intermediates_input)).
// for_each(|(output, input)| {         *output =
// matrix_multiply_base_optimized(rs, input);     });
//
//     // 3. Optimization: Transpose Intermediates for better locality.
//     // FIX: Move this large allocation (e.g., 128KB for D64) to the heap.
//     let mut intermediates_soa: Vec<[InterPoly; RADIX]> =
// vec![std::array::from_fn::<_, RADIX, _>(|_| InterPoly::zero()); BASE_LEN];
//
//     // Efficient sequential transposition.
//     for i in 0..RADIX {
//         for k in 0..BASE_LEN {
//             // Accessing Vec elements works identically to arrays here.
//             intermediates_soa[k][i] = intermediates_aos[i][k].clone();
//         }
//     }
//
//     // 4. Combine using DIT structure.
//     // The output encoded vector is already heap-allocated (Vec).
//     let mut encoded: EncodedVector<OutPoly> = vec![OutPoly::zero(); N];
//     let dit_twiddles = &rs.DIT_TWIDDLES;
//
//     cfg_iter_mut!(encoded).enumerate().for_each(|(k, val)| {
//         let k_mod_64 = k % BASE_LEN;
//         let mut sum = OutPoly::zero();
//         let twiddles_k = &dit_twiddles[k];
//         // Accessing the transposed structure (now a Vec).
//         let intermediates_k = &intermediates_soa[k_mod_64];
//
//         for i in 0..RADIX {
//             let twiddle = twiddles_k[i];
//             let u_i_k_32 = intermediates_k[i].clone();
//
//             // Twiddle HAS to be u32!
//             debug_assert!(twiddle >= 0 && twiddle < (1 << 32));
//             let twiddle_u32 = twiddle as u32;
//
//             // Optimization: Multiply in u32 space first, then extend.
//             let product_32 = u_i_k_32.mul_by_scalar(twiddle_u32).unwrap();
//             let product_64 = OutPoly::from_ref(&product_32);
//
//             sum += product_64;
//         }
//         *val = sum;
//     });
//
//     encoded
// }
//
//
// // Optimized Base Encoding: Fused Unpack-Multiply-Add (FUMA). Generic over T
// and D. // Note: This function returns a stack array (e.g. 16KB for D64),
// which is acceptable here. fn matrix_multiply_base_optimized<CI, CO,
// PolyOutput>(rs: &ReedSolomon, v_raw: &[CI; BASE_DIM]) -> [PolyOutput;
// BASE_LEN] where
//     CI: WithBitMask + BitAnd<CI, Output = CI> + From<u32> + Copy,
//     CO: WrappingAdd + FromRef<CI>,
//     PolyOutput: Polynomial<CO> + Zero,
// {
//     // assert_eq!(PolyInput::DEGREE_BOUND, PolyOutput::DEGREE_BOUND);
//     let D: usize = PolyOutput::DEGREE_BOUND;
//
//     let mut output: [PolyOutput; BASE_LEN] = std::array::from_fn(|_|
// PolyOutput::zero());     let m_prime = &rs.M_PRIME;
//
//     // Parallelize over the 64 rows of M'.
//     cfg_iter_mut!(output).enumerate().for_each(|(i, out_val)| {
//         let row = &m_prime[i];
//
//         // Iterate over the 32 input polynomials (j)
//         for j in 0..BASE_DIM {
//             // Convert scalar (i64) to the specific Coeff32 type (u32).
//             debug_assert!(row[j] >= 0);
//             let scalar_typed = CI::from(row[j] as u32);
//             let input_poly_raw = &v_raw[j];
//
//             let coeffs = out_val.as_coeffs_mut();
//
//             // Optimized, branchless FUMA kernel (k iterates over D
// coefficients/bits):             for k in 0..D {
//                 // 1. Get the branchless mask (0 or MAX) using the trait
// helper.                 let mask = input_poly_raw.get_kth_bit_mask(k);
//
//                 // 2. Calculate value to add: (scalar & mask).
//                 let value_to_add = scalar_typed & mask;
//
//                 // 3. Add using wrapping arithmetic (optimized for SIMD).
//                 coeffs[k] =
// coeffs[k].wrapping_add(&CO::from_ref(&value_to_add));             }
//         }
//     });
//     output
// }
//
// pub trait WithBitMask {
//     fn get_kth_bit_mask(&self, k: usize) -> Self;
// }
