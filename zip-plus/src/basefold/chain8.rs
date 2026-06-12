//! Radix-8 foldable IPRS chain (true radix-8 lift).
//!
//! One level of this chain relates `Enc_r` (length `n_r`) to *eight* copies
//! of `Enc_{r+1}` (length `n_r / 8`) through the lifted 8-point DFT block:
//! with `H = n_r / 8`, `omega_r = omega^{8^r}` and `zeta = omega^{n/8}` (a
//! primitive 8th root of unity),
//!
//! ```text
//!   Enc_r(v)[j + c*H] = sum_{i=0}^{7} M_j[c][i] * Enc_{r+1}(v^{(i)})[j],
//!   M_j[c][i] = lift(omega_r^{i*j} * zeta^{i*c}),
//! ```
//!
//! where `v^{(i)}` collects the coefficients congruent to `i` mod 8. The
//! crucial difference to stacking three radix-2 levels: each `M_j` entry is
//! the lift of a *single* field product, so the per-level norm growth is
//! one twiddle (~log2(q/2) bits) per 8-fold of the message --- roughly a
//! third of the radix-2 chain's growth per folded variable, which directly
//! shrinks the committed leaf width.
//!
//! The price is the verifier-side derivation: `M_j` has no closed-form
//! integer inverse. The verifier instead solves the transposed system
//! `M_j^T z = alpha` per limb with one fraction-free (Bareiss)
//! triangularization of `[M_j^T | alpha_1 | ... ]` per queried position and
//! exact back-substitution (all divisions are exact because `d * z` is
//! integral, `d = +-det(M_j)`), then checks `d * W == sum_t z_t^T c_t` ---
//! still no rationals anywhere.

use super::{BasefoldError, chain::ChainConfig};
use crypto_primitives::crypto_bigint_int::Int;
use num_traits::{CheckedAdd, CheckedSub, ConstOne, ConstZero, Zero};
use std::{fmt::Debug, marker::PhantomData};
use zinc_utils::{add, mul, mul_by_scalar::MulByScalar, sub};

/// Twiddle integer type (centered lifts).
pub type TwiddleInt = i64;

pub const ARITY: usize = 8;
pub const LOG_ARITY: u32 = 3;

/// A radix-8 foldable IPRS chain with `num_levels` butterfly levels.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Radix8Chain<C: ChainConfig> {
    msg_len: usize,
    codeword_len: usize,
    rep: usize,
    num_levels: usize,
    /// `twiddle_mats[r][j][c][i] = lift(omega_r^{i*j} * zeta^{i*c})` for
    /// `j in [n_r / 8]`.
    twiddle_mats: Vec<Vec<[[TwiddleInt; ARITY]; ARITY]>>,
    /// Lifted Vandermonde base case (`[n_D][m_D]`).
    base_matrix: Vec<Vec<TwiddleInt>>,
    _phantom: PhantomData<C>,
}

impl<C: ChainConfig> Radix8Chain<C> {
    #[allow(clippy::arithmetic_side_effects)] // index arithmetic mod n
    pub fn new(msg_len: usize, rep: usize, num_levels: usize) -> Result<Self, BasefoldError> {
        if !msg_len.is_power_of_two() || !rep.is_power_of_two() {
            return Err(BasefoldError::InvalidParams(format!(
                "msg_len ({msg_len}) and rep ({rep}) must be powers of two"
            )));
        }
        if mul!(num_levels, 3usize) > msg_len.trailing_zeros() as usize {
            return Err(BasefoldError::InvalidParams(format!(
                "num_levels ({num_levels}) exceeds log8(msg_len)"
            )));
        }
        let codeword_len = mul!(msg_len, rep);
        if codeword_len >= C::FIELD_MODULUS as usize || codeword_len < ARITY {
            return Err(BasefoldError::InvalidParams(format!(
                "codeword length {codeword_len} out of range for the base field"
            )));
        }
        let domain = ark_poly::Radix2EvaluationDomain::<C::Field>::new(codeword_len)
            .ok_or_else(|| {
                BasefoldError::InvalidParams(format!(
                    "no radix-2 evaluation domain of size {codeword_len}"
                ))
            })?;
        use ark_poly::EvaluationDomain;
        let roots: Vec<TwiddleInt> = domain.elements().map(C::field_to_int_centered).collect();

        let n = codeword_len;
        let twiddle_mats: Vec<Vec<[[TwiddleInt; ARITY]; ARITY]>> = (0..num_levels)
            .map(|r| {
                let n_r = n >> (3 * r);
                let h = n_r / ARITY;
                (0..h)
                    .map(|j| {
                        std::array::from_fn(|c| {
                            std::array::from_fn(|i| {
                                // omega_r^{i*(j + c*H)} = omega^{8^r * i * (j + c*H)}
                                roots[((i * (j + c * h)) << (3 * r)) % n]
                            })
                        })
                    })
                    .collect()
            })
            .collect();

        let base_cw_len = n >> (3 * num_levels);
        let base_msg_len = msg_len >> (3 * num_levels);
        let base_matrix: Vec<Vec<TwiddleInt>> = (0..base_cw_len)
            .map(|j| {
                (0..base_msg_len)
                    .map(|i| roots[((i * j) << (3 * num_levels)) % n])
                    .collect()
            })
            .collect();

        Ok(Self {
            msg_len,
            codeword_len,
            rep,
            num_levels,
            twiddle_mats,
            base_matrix,
            _phantom: PhantomData,
        })
    }

    pub fn num_levels(&self) -> usize {
        self.num_levels
    }

    pub fn rep(&self) -> usize {
        self.rep
    }

    pub fn msg_len_at(&self, level: usize) -> usize {
        debug_assert!(level <= self.num_levels);
        self.msg_len >> mul!(3usize, level)
    }

    pub fn codeword_len_at(&self, level: usize) -> usize {
        debug_assert!(level <= self.num_levels);
        self.codeword_len >> mul!(3usize, level)
    }

    /// The lifted DFT block of level `level` at sub-position
    /// `j in [codeword_len_at(level + 1)]`.
    pub fn twiddle_matrix(&self, level: usize, j: usize) -> &[[TwiddleInt; ARITY]; ARITY] {
        &self.twiddle_mats[level][j]
    }

    /// Magnitude bound, in bits, of level-`level` codeword entries when the
    /// encoded message has entries of magnitude below `2^msg_mag_bits`:
    /// every butterfly level multiplies the bound by its worst absolute
    /// row sum, and the recursion bottoms out in the dense base block.
    /// Used to size the tight proof-stream serialization.
    #[allow(clippy::arithmetic_side_effects)] // u64 row sums of < 2^27 twiddles
    pub fn codeword_mag_bits(&self, level: usize, msg_mag_bits: u32) -> u32 {
        debug_assert!(level <= self.num_levels);
        let bits_of = |sum: u64| u64::BITS - sum.leading_zeros();
        let row_sum = |row: &[TwiddleInt]| row.iter().map(|t| t.unsigned_abs()).sum::<u64>();

        let base = self
            .base_matrix
            .iter()
            .map(|row| row_sum(row))
            .max()
            .unwrap_or(0);
        let mut bits = msg_mag_bits + bits_of(base);
        for lvl in level..self.num_levels {
            let factor = self.twiddle_mats[lvl]
                .iter()
                .flat_map(|mat| mat.iter().map(|row| row_sum(row)))
                .max()
                .unwrap_or(0);
            bits += bits_of(factor);
        }
        bits
    }

    /// Encode a message under the level-`level` code of the chain.
    /// With the `parallel` feature, large sub-encodes (recursion and the
    /// dense base block) fan out across threads.
    #[allow(clippy::arithmetic_side_effects)] // level/index arithmetic bounded
    pub fn encode_at_level<In, Out, const CHECK: bool>(&self, level: usize, msg: &[In]) -> Vec<Out>
    where
        In: for<'a> MulByScalar<&'a TwiddleInt, Out> + Clone + Debug + Send + Sync,
        Out: for<'a> MulByScalar<&'a TwiddleInt>
            + CheckedAdd
            + CheckedSub
            + ConstZero
            + Clone
            + Debug
            + Send
            + Sync,
    {
        assert!(level <= self.num_levels, "level out of range");
        assert_eq!(
            msg.len(),
            self.msg_len_at(level),
            "message length mismatch at level {level}"
        );

        if level == self.num_levels {
            let base_row = |row: &Vec<TwiddleInt>| {
                row.iter().zip(msg).fold(Out::ZERO, |acc, (tw, x)| {
                    let term = x
                        .mul_by_scalar::<CHECK>(tw)
                        .expect("base-case twiddle multiplication overflow");
                    add!(acc, term, "base-case accumulation overflow")
                })
            };
            #[cfg(feature = "parallel")]
            if self.base_matrix.len().saturating_mul(msg.len()) >= (1 << 16) {
                use rayon::prelude::*;
                return self.base_matrix.par_iter().map(base_row).collect();
            }
            return self.base_matrix.iter().map(base_row).collect();
        }

        // Split into the 8 residue classes mod 8 and recurse.
        let classes: Vec<Vec<In>> = (0..ARITY)
            .map(|i| msg.iter().skip(i).step_by(ARITY).cloned().collect())
            .collect();
        let recurse =
            |cls: &Vec<In>| self.encode_at_level::<In, Out, CHECK>(level + 1, cls);
        #[cfg(feature = "parallel")]
        let subs: Vec<Vec<Out>> = if msg.len() >= (1 << 12) {
            use rayon::prelude::*;
            classes.par_iter().map(recurse).collect()
        } else {
            classes.iter().map(recurse).collect()
        };
        #[cfg(not(feature = "parallel"))]
        let subs: Vec<Vec<Out>> = classes.iter().map(recurse).collect();

        let h = self.codeword_len_at(level + 1);
        let mut out = vec![Out::ZERO; self.codeword_len_at(level)];
        for j in 0..h {
            let mat = &self.twiddle_mats[level][j];
            for (c, row) in mat.iter().enumerate() {
                let mut acc = Out::ZERO;
                for (i, tw) in row.iter().enumerate() {
                    let term = subs[i][j]
                        .mul_by_scalar::<CHECK>(tw)
                        .expect("butterfly twiddle multiplication overflow");
                    acc = add!(acc, term, "butterfly accumulation overflow");
                }
                out[j + c * h] = acc;
            }
        }
        out
    }

    /// Solve `M_j^T z = alpha` for each given `alpha`, fraction-free.
    ///
    /// Returns `(d, zs)` with `d = +-det(M_j) != 0` and integer vectors
    /// `zs[t] = d * M_j^{-T} alphas[t]`, so that for any opened coset `c`:
    /// `sum_i alphas[t][i] * (M_j^{-1} c)[i] = zs[t]^T c / d`.
    ///
    /// The elimination runs at 576 bits regardless of `NW`: Bareiss
    /// intermediates for 27-bit twiddles and 128-bit challenge columns stay
    /// under `2^566` (products of an 8x8 minor with a challenge-column
    /// minor), and narrower multiplications make the lazy per-position
    /// solve ~2x cheaper for the verifier.
    pub fn solve_scaled<const NW: usize>(
        &self,
        level: usize,
        j: usize,
        alphas: &[[Int<NW>; ARITY]],
    ) -> Result<(Int<NW>, Vec<[Int<NW>; ARITY]>), BasefoldError> {
        const NS: usize = 9;
        if NW >= NS {
            let alphas_n: Vec<[Int<NS>; ARITY]> = alphas
                .iter()
                .map(|a| std::array::from_fn(|i| a[i].resize::<NS>()))
                .collect();
            let (d, zs) = self.solve_scaled_at::<NS>(level, j, &alphas_n)?;
            return Ok((
                d.resize::<NW>(),
                zs.iter()
                    .map(|z| std::array::from_fn(|i| z[i].resize::<NW>()))
                    .collect(),
            ));
        }
        self.solve_scaled_at::<NW>(level, j, alphas)
    }

    #[allow(clippy::arithmetic_side_effects)] // fixed 8x(8+L) elimination
    fn solve_scaled_at<const NW: usize>(
        &self,
        level: usize,
        j: usize,
        alphas: &[[Int<NW>; ARITY]],
    ) -> Result<(Int<NW>, Vec<[Int<NW>; ARITY]>), BasefoldError> {
        let mat = self.twiddle_matrix(level, j);
        let cols = ARITY + alphas.len();

        // Augmented matrix [M^T | alpha_1 | alpha_2 | ...], row-major.
        let mut a: Vec<Vec<Int<NW>>> = (0..ARITY)
            .map(|r| {
                let mut row: Vec<Int<NW>> = (0..ARITY)
                    .map(|c| int_from_i64::<NW>(mat[c][r])) // transpose
                    .collect();
                for alpha in alphas {
                    row.push(alpha[r]);
                }
                row
            })
            .collect();

        // Fraction-free (Bareiss) triangularization.
        let mut prev = Int::<NW>::ONE;
        for k in 0..ARITY {
            if a[k][k].is_zero() {
                let Some(l) = (k + 1..ARITY).find(|&l| !a[l][k].is_zero()) else {
                    return Err(BasefoldError::InvalidParams(
                        "singular butterfly block".to_owned(),
                    ));
                };
                a.swap(k, l); // d becomes -det; consistent on both check sides
            }
            for i in k + 1..ARITY {
                for col in k + 1..cols {
                    let num = sub!(
                        mul_int(&a[k][k], &a[i][col]),
                        mul_int(&a[i][k], &a[k][col])
                    );
                    a[i][col] = exact_div(&num, &prev);
                }
                a[i][k] = Int::ZERO;
            }
            prev = a[k][k];
        }
        let d = a[ARITY - 1][ARITY - 1];

        // Exact back-substitution: t_i = d * z_i satisfies
        // U[i][i] * t_i = d * rhs_i - sum_{l>i} U[i][l] * t_l, divisible
        // exactly by U[i][i] because t_i is integral (Cramer).
        let mut zs = Vec::with_capacity(alphas.len());
        for (t_idx, _) in alphas.iter().enumerate() {
            let rhs_col = ARITY + t_idx;
            let mut z = [Int::<NW>::ZERO; ARITY];
            for i in (0..ARITY).rev() {
                let mut num = mul_int(&d, &a[i][rhs_col]);
                for l in i + 1..ARITY {
                    num = sub!(num, mul_int(&a[i][l], &z[l]));
                }
                z[i] = exact_div(&num, &a[i][i]);
            }
            zs.push(z);
        }
        Ok((d, zs))
    }
}

fn int_from_i64<const NW: usize>(v: i64) -> Int<NW> {
    Int::<NW>::ONE
        .mul_by_scalar::<true>(&v)
        .expect("twiddle fits any working width")
}

fn mul_int<const NW: usize>(a: &Int<NW>, b: &Int<NW>) -> Int<NW> {
    a.mul_by_scalar::<true>(b)
        .expect("solver working width exceeded")
}

#[allow(clippy::arithmetic_side_effects)]
fn exact_div<const NW: usize>(num: &Int<NW>, den: &Int<NW>) -> Int<NW> {
    let q: Option<crypto_bigint::Int<NW>> = num.inner().checked_div(den.inner()).into();
    let q = Int::<NW>::new(q.expect("division by zero in fraction-free solve"));
    debug_assert_eq!(
        mul_int(&q, den),
        *num,
        "non-exact division in fraction-free solve"
    );
    q
}

#[cfg(test)]
#[allow(
    clippy::arithmetic_side_effects,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::unwrap_used
)]
mod tests {
    use super::*;
    use crate::basefold::chain::{ChainConfigF65537, ChainConfigF167772161};
    use crypto_primitives::IntRing;
    use rand::{Rng, SeedableRng, rngs::StdRng};

    type I10 = Int<10>;

    #[test]
    fn radix8_locality_identity() {
        // Enc_l(v)[j + c*H] = sum_i M_j[c][i] * Enc_{l+1}(v^{(i)})[j].
        let chain = Radix8Chain::<ChainConfigF65537>::new(512, 4, 2).unwrap();
        let msg: Vec<Int<3>> = (0..512).map(|i| Int::from(11 * i - 999)).collect();
        let cw: Vec<I10> = chain.encode_at_level::<_, _, true>(0, &msg);

        let classes: Vec<Vec<Int<3>>> = (0..ARITY)
            .map(|i| msg.iter().skip(i).step_by(ARITY).cloned().collect())
            .collect();
        let subs: Vec<Vec<I10>> = classes
            .iter()
            .map(|cls| chain.encode_at_level::<_, _, true>(1, cls))
            .collect();

        let h = chain.codeword_len_at(1);
        for j in 0..h {
            let mat = chain.twiddle_matrix(0, j);
            for c in 0..ARITY {
                let mut acc = I10::ZERO;
                for i in 0..ARITY {
                    acc = add!(acc, subs[i][j].mul_by_scalar::<true>(&mat[c][i]).unwrap());
                }
                assert_eq!(cw[j + c * h], acc, "mismatch at (j={j}, c={c})");
            }
        }
    }

    #[test]
    fn radix8_reduces_to_rs_mod_q() {
        let q = u64::from(<ChainConfigF65537 as ChainConfig>::FIELD_MODULUS);
        let msg_len = 64usize;
        let rep = 4usize;
        let n = msg_len * rep;
        let chain = Radix8Chain::<ChainConfigF65537>::new(msg_len, rep, 2).unwrap();
        let msg: Vec<Int<3>> = (0..msg_len as i32).map(|i| Int::from(5 * i + 3)).collect();
        let cw: Vec<I10> = chain.encode_at_level::<_, _, true>(0, &msg);

        let domain =
            ark_poly::Radix2EvaluationDomain::<
                <ChainConfigF65537 as ChainConfig>::Field,
            >::new(n)
            .unwrap();
        use ark_ff::PrimeField as ArkPrimeField;
        use ark_poly::EvaluationDomain;
        let omegas: Vec<u64> = domain.elements().map(|x| x.into_bigint().0[0]).collect();
        let lift = |x: &I10| -> u64 {
            let neg = x.is_negative();
            let mag = x.inner().abs();
            let mut acc: u128 = 0;
            let base = (1u128 << 64) % u128::from(q);
            for w in mag.to_words().iter().rev() {
                acc = (acc * base + u128::from(*w) % u128::from(q)) % u128::from(q);
            }
            let r = acc as u64;
            if neg && r != 0 { q - r } else { r }
        };
        for (jj, omega_j) in omegas.iter().enumerate() {
            let mut acc: u64 = 0;
            let mut pw: u64 = 1;
            for x in &msg {
                let xi = lift(&x.resize::<10>());
                acc = (acc + xi * pw) % q;
                pw = (pw * omega_j) % q;
            }
            assert_eq!(lift(&cw[jj]), acc, "mismatch at position {jj}");
        }
    }

    #[test]
    fn solve_scaled_satisfies_transposed_system() {
        let chain = Radix8Chain::<ChainConfigF167772161>::new(512, 4, 2).unwrap();
        let mut rng = StdRng::seed_from_u64(5);
        for &(level, j) in &[(0usize, 0usize), (0, 17), (0, 255), (1, 3), (1, 31)] {
            let alphas: Vec<[Int<12>; ARITY]> = (0..2)
                .map(|_| {
                    std::array::from_fn(|_| {
                        let mut words = [0u64; 12];
                        words[0] = rng.random();
                        words[1] = rng.random();
                        Int::<12>::new(crypto_bigint::Int::from_words(words))
                    })
                })
                .collect();
            let (d, zs) = chain.solve_scaled::<12>(level, j, &alphas).unwrap();
            assert!(!d.is_zero());
            let mat = chain.twiddle_matrix(level, j);
            for (alpha, z) in alphas.iter().zip(&zs) {
                // M^T z = d * alpha  <=>  for all i: sum_c M[c][i] z_c = d * alpha_i
                for i in 0..ARITY {
                    let mut acc = Int::<12>::ZERO;
                    for c in 0..ARITY {
                        acc = add!(acc, z[c].mul_by_scalar::<true>(&mat[c][i]).unwrap());
                    }
                    assert_eq!(acc, mul_int(&d, &alpha[i]), "row {i} mismatch");
                }
            }
        }
    }

    #[test]
    fn radix8_norm_growth_report() {
        let mut rng = StdRng::seed_from_u64(2);
        let input_bits = 131u32; // balanced digits at p = 132
        let msg_len = 1usize << 9;
        for levels in [1usize, 2, 3] {
            let chain =
                Radix8Chain::<ChainConfigF167772161>::new(msg_len, 4, levels).unwrap();
            let msg: Vec<Int<3>> = (0..msg_len)
                .map(|_| {
                    let mut words = [0u64; 3];
                    words[0] = rng.random();
                    words[1] = rng.random();
                    words[2] = rng.random::<u64>() & 7;
                    let v = Int::<3>::new(crypto_bigint::Int::from_words(words));
                    if rng.random::<bool>() { -v } else { v }
                })
                .collect();
            let cw: Vec<Int<10>> = chain.encode_at_level::<_, _, true>(0, &msg);
            let measured = cw
                .iter()
                .map(|v| v.inner().abs().bits())
                .max()
                .unwrap_or(0);
            eprintln!(
                "radix-8 levels {levels}: measured {measured} bits \
                 (growth {} per level over {input_bits}-bit digits)",
                (measured.saturating_sub(input_bits)) / levels as u32
            );
        }
    }
}
