//! Radix-2 foldable IPRS chain.
//!
//! An IPRS code is the lift of an RS FFT *algorithm* to the integers: every
//! twiddle is replaced by its centered representative and all arithmetic is
//! exact (no modular reduction). Over the integers the recursion depth is
//! part of the code's definition, and FRI-style folding consumes exactly one
//! butterfly level per round: the round-`(r+1)` code must be the *tail* of
//! the round-`r` code. This module materializes the full chain:
//!
//! ```text
//!   Enc_r(v)[j]            = Enc_{r+1}(v_even)[j] + tau_r[j] * Enc_{r+1}(v_odd)[j]
//!   Enc_r(v)[j + n_{r+1}]  = Enc_{r+1}(v_even)[j] - tau_r[j] * Enc_{r+1}(v_odd)[j]
//! ```
//!
//! with `tau_r[j] = centered_lift(omega_r^j)`, `omega_r = omega^{2^r}`, and a
//! lifted-Vandermonde base case after `num_levels` levels. The identity above
//! is exactly what lets the verifier derive tail-code symbols from a pair of
//! openings (the locality lemma of the Zip++ draft); the verifier-side checks
//! clear the denominators `2 tau_r[j]` and stay in integer arithmetic.

use super::BasefoldError;
use ark_ff::PrimeField as ArkPrimeField;
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
use num_traits::{CheckedAdd, CheckedSub, ConstZero};
use std::{fmt::Debug, marker::PhantomData};
use zinc_utils::{add, mul, mul_by_scalar::MulByScalar, sub};

/// Twiddle integer type (centered lifts of base-field elements).
pub type TwiddleInt = i64;

/// Configuration of a foldable IPRS chain: the base field supplying the
/// roots of unity. The field modulus must exceed the codeword length
/// (`n | q - 1` is required for the size-`n` FFT domain to exist).
pub trait ChainConfig: Clone + Debug + Send + Sync {
    type Field: ark_ff::FftField;
    const FIELD_MODULUS: u32;

    /// Centered representative in `[-(q-1)/2, (q-1)/2]`.
    fn field_to_int_centered(x: Self::Field) -> TwiddleInt;
}

mod f65537 {
    #![allow(non_local_definitions)]
    use ark_ff::{Fp64, MontBackend, MontConfig};
    #[derive(MontConfig)]
    #[modulus = "65537"]
    #[generator = "3"]
    pub struct Config;

    pub type Backend = MontBackend<Config, 1>;
    pub type Field = Fp64<Backend>;

    #[allow(clippy::cast_possible_truncation)] // modulus is 17 bits
    pub const MODULUS: u32 = Config::MODULUS.0[0] as u32;
}

/// Chain configuration over `F_65537` (`2^16 + 1`); supports codeword
/// lengths up to `2^16`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChainConfigF65537;

impl ChainConfig for ChainConfigF65537 {
    type Field = f65537::Field;
    const FIELD_MODULUS: u32 = f65537::MODULUS;

    fn field_to_int_centered(x: Self::Field) -> TwiddleInt {
        normalize_field_element(x.into_bigint().0[0], Self::FIELD_MODULUS)
    }
}

mod f167772161 {
    #![allow(non_local_definitions)]
    use ark_ff::{Fp64, MontBackend, MontConfig};
    #[derive(MontConfig)]
    #[modulus = "167772161"]
    #[generator = "3"]
    pub struct Config;

    pub type Backend = MontBackend<Config, 1>;
    pub type Field = Fp64<Backend>;

    #[allow(clippy::cast_possible_truncation)] // modulus is 28 bits
    pub const MODULUS: u32 = Config::MODULUS.0[0] as u32;
}

/// Chain configuration over `F_{5 * 2^25 + 1}` (two-adicity 25); supports
/// codeword lengths up to `2^25`. Twiddle lifts are at most 28 bits, so the
/// per-level worst-case norm growth is ~28 bits.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChainConfigF167772161;

impl ChainConfig for ChainConfigF167772161 {
    type Field = f167772161::Field;
    const FIELD_MODULUS: u32 = f167772161::MODULUS;

    fn field_to_int_centered(x: Self::Field) -> TwiddleInt {
        normalize_field_element(x.into_bigint().0[0], Self::FIELD_MODULUS)
    }
}

/// Centered lift for at-most-32-bit field moduli.
#[allow(clippy::arithmetic_side_effects, clippy::cast_possible_wrap)]
fn normalize_field_element(x: u64, q: u32) -> TwiddleInt {
    debug_assert!(x < u64::from(q));
    let x = x as i64;
    let q = i64::from(q);
    if x >= (q - 1) / 2 { x - q } else { x }
}

/// A full radix-2 foldable IPRS chain with `num_levels` butterfly levels.
///
/// Level `r` (for `0 <= r <= num_levels`) is a code of message length
/// `msg_len / 2^r` and codeword length `codeword_len / 2^r`; level
/// `num_levels` is the lifted-Vandermonde base case.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FoldableIprsChain<C: ChainConfig> {
    msg_len: usize,
    codeword_len: usize,
    rep: usize,
    num_levels: usize,
    /// `twiddles[r][j] = lift(omega_r^j)` for `j in [codeword_len / 2^{r+1}]`.
    twiddles: Vec<Vec<TwiddleInt>>,
    /// Lifted Vandermonde base case: `base_matrix[j][i] = lift(omega_D^{i*j})`
    /// for `j in [n_D]`, `i in [m_D]`.
    base_matrix: Vec<Vec<TwiddleInt>>,
    _phantom: PhantomData<C>,
}

impl<C: ChainConfig> FoldableIprsChain<C> {
    pub fn new(msg_len: usize, rep: usize, num_levels: usize) -> Result<Self, BasefoldError> {
        if !msg_len.is_power_of_two() || !rep.is_power_of_two() {
            return Err(BasefoldError::InvalidParams(format!(
                "msg_len ({msg_len}) and rep ({rep}) must be powers of two"
            )));
        }
        if num_levels > msg_len.trailing_zeros() as usize {
            return Err(BasefoldError::InvalidParams(format!(
                "num_levels ({num_levels}) exceeds log2(msg_len) ({})",
                msg_len.trailing_zeros()
            )));
        }
        let codeword_len = mul!(msg_len, rep);
        if codeword_len >= C::FIELD_MODULUS as usize {
            return Err(BasefoldError::InvalidParams(format!(
                "codeword length {codeword_len} is not below the field modulus {}",
                C::FIELD_MODULUS
            )));
        }
        let domain =
            Radix2EvaluationDomain::<C::Field>::new(codeword_len).ok_or_else(|| {
                BasefoldError::InvalidParams(format!(
                    "no radix-2 evaluation domain of size {codeword_len}"
                ))
            })?;
        // Roots in natural order: omega^0, omega^1, ..., omega^{n-1}.
        let roots: Vec<TwiddleInt> = domain.elements().map(C::field_to_int_centered).collect();

        #[allow(clippy::arithmetic_side_effects)] // index arithmetic bounded by n
        let twiddles: Vec<Vec<TwiddleInt>> = (0..num_levels)
            .map(|r| {
                let half = codeword_len >> (r + 1);
                // omega_r^j = omega^{2^r * j}; 2^r * j < n/2, no reduction needed.
                (0..half).map(|j| roots[j << r]).collect()
            })
            .collect();

        let base_cw_len = codeword_len >> num_levels;
        let base_msg_len = msg_len >> num_levels;
        #[allow(clippy::arithmetic_side_effects)] // index arithmetic mod n
        let base_matrix: Vec<Vec<TwiddleInt>> = (0..base_cw_len)
            .map(|j| {
                (0..base_msg_len)
                    .map(|i| roots[((i * j) << num_levels) % codeword_len])
                    .collect()
            })
            .collect();

        Ok(Self {
            msg_len,
            codeword_len,
            rep,
            num_levels,
            twiddles,
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
        self.msg_len >> level
    }

    pub fn codeword_len_at(&self, level: usize) -> usize {
        debug_assert!(level <= self.num_levels);
        self.codeword_len >> level
    }

    /// `tau_r[j] = lift(omega_r^j)`, the butterfly twiddle of level `level`
    /// at position `j in [codeword_len_at(level + 1)]`.
    pub fn twiddle(&self, level: usize, j: usize) -> TwiddleInt {
        self.twiddles[level][j]
    }

    /// Encode a message under the level-`level` code of the chain (the tail
    /// code with `num_levels - level` butterfly levels remaining).
    pub fn encode_at_level<In, Out, const CHECK: bool>(&self, level: usize, msg: &[In]) -> Vec<Out>
    where
        In: for<'a> MulByScalar<&'a TwiddleInt, Out> + Clone + Debug,
        Out: for<'a> MulByScalar<&'a TwiddleInt>
            + CheckedAdd
            + CheckedSub
            + ConstZero
            + Clone
            + Debug,
    {
        assert!(level <= self.num_levels, "level out of range");
        assert_eq!(
            msg.len(),
            self.msg_len_at(level),
            "message length mismatch at level {level}"
        );
        self.encode_rec::<In, Out, CHECK>(level, msg)
    }

    #[allow(clippy::arithmetic_side_effects)] // level < num_levels; lengths are powers of two
    fn encode_rec<In, Out, const CHECK: bool>(&self, level: usize, msg: &[In]) -> Vec<Out>
    where
        In: for<'a> MulByScalar<&'a TwiddleInt, Out> + Clone + Debug,
        Out: for<'a> MulByScalar<&'a TwiddleInt>
            + CheckedAdd
            + CheckedSub
            + ConstZero
            + Clone
            + Debug,
    {
        if level == self.num_levels {
            // Lifted-Vandermonde base case.
            return self
                .base_matrix
                .iter()
                .map(|row| {
                    row.iter().zip(msg).fold(Out::ZERO, |acc, (tw, x)| {
                        let term = x
                            .mul_by_scalar::<CHECK>(tw)
                            .expect("base-case twiddle multiplication overflow");
                        add!(acc, term, "base-case accumulation overflow")
                    })
                })
                .collect();
        }

        let even: Vec<In> = msg.iter().step_by(2).cloned().collect();
        let odd: Vec<In> = msg.iter().skip(1).step_by(2).cloned().collect();
        let enc_even = self.encode_rec::<In, Out, CHECK>(level + 1, &even);
        let enc_odd = self.encode_rec::<In, Out, CHECK>(level + 1, &odd);

        let half = self.codeword_len_at(level + 1);
        let mut out = vec![Out::ZERO; self.codeword_len_at(level)];
        for j in 0..half {
            let t = enc_odd[j]
                .mul_by_scalar::<CHECK>(&self.twiddles[level][j])
                .expect("butterfly twiddle multiplication overflow");
            out[j] = add!(enc_even[j], t, "butterfly addition overflow");
            #[allow(clippy::arithmetic_side_effects)] // j + half < codeword_len_at(level)
            {
                out[j + half] = sub!(enc_even[j], t, "butterfly subtraction overflow");
            }
        }
        out
    }
}

#[cfg(test)]
#[allow(
    clippy::arithmetic_side_effects,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::unwrap_used
)]
mod tests {
    use super::*;
    use crypto_primitives::{IntRing, crypto_bigint_int::Int};

    type I8w = Int<8>;

    fn lift_to_field(x: &I8w, q: u64) -> u64 {
        // Horner over the magnitude words (most significant first).
        let neg = x.is_negative();
        let mag = x.inner().abs();
        let mut acc: u128 = 0;
        let base = (1u128 << 64) % u128::from(q);
        for w in mag.to_words().iter().rev() {
            acc = (acc * base + u128::from(*w) % u128::from(q)) % u128::from(q);
        }
        let r = acc as u64;
        if neg && r != 0 { q - r } else { r }
    }

    #[test]
    fn chain_construction() {
        let chain = FoldableIprsChain::<ChainConfigF65537>::new(64, 4, 4).unwrap();
        assert_eq!(chain.codeword_len_at(0), 256);
        assert_eq!(chain.msg_len_at(4), 4);
        assert_eq!(chain.codeword_len_at(4), 16);
        // Level-r twiddles have length n_{r+1} and start with lift(1) = 1.
        for r in 0..4 {
            assert_eq!(chain.twiddles[r].len(), 256 >> (r + 1));
            assert_eq!(chain.twiddle(r, 0), 1);
        }
    }

    #[test]
    fn butterfly_locality_identity() {
        // Enc_l(v)[j] = Enc_{l+1}(v_e)[j] + tau * Enc_{l+1}(v_o)[j], and the
        // sign-flipped identity at j + half — the verifier's locality lemma.
        let chain = FoldableIprsChain::<ChainConfigF65537>::new(32, 4, 3).unwrap();
        let msg: Vec<Int<3>> = (0..32).map(|i| Int::from(7 * i - 11)).collect();
        let cw: Vec<I8w> = chain.encode_at_level::<_, _, true>(0, &msg);

        let even: Vec<Int<3>> = msg.iter().step_by(2).cloned().collect();
        let odd: Vec<Int<3>> = msg.iter().skip(1).step_by(2).cloned().collect();
        let cw_e: Vec<I8w> = chain.encode_at_level::<_, _, true>(1, &even);
        let cw_o: Vec<I8w> = chain.encode_at_level::<_, _, true>(1, &odd);

        let half = chain.codeword_len_at(1);
        for j in 0..half {
            let tau = chain.twiddle(0, j);
            let t = cw_o[j].mul_by_scalar::<true>(&tau).unwrap();
            assert_eq!(cw[j], add!(cw_e[j], t));
            assert_eq!(cw[j + half], sub!(cw_e[j], t));
        }
    }

    fn random_digit<const N: usize>(rng: &mut impl rand::Rng, bits: u32) -> Int<N> {
        let mut words = [0u64; N];
        let full_words = (bits / 64) as usize;
        for w in words.iter_mut().take(full_words) {
            *w = rng.random();
        }
        let rem = bits % 64;
        if rem > 0 && full_words < N {
            words[full_words] = rng.random::<u64>() & ((1u64 << rem) - 1);
        }
        let mag = Int::<N>::new(crypto_bigint::Int::from_words(words));
        if rng.random::<bool>() { -mag } else { mag }
    }

    fn max_bits<const N: usize>(values: &[Int<N>]) -> u32 {
        values
            .iter()
            .map(|v| v.inner().abs().bits())
            .max()
            .unwrap_or(0)
    }

    /// Measured vs worst-case codeword norm growth of the chain encoders,
    /// per depth. The worst-case bound (companion thm) is
    /// `input_bits + (depth + 1) * log2(q/2) + log2(msg_len)`; centered
    /// representatives cancel heavily in practice, and the measured value
    /// drives the leaf width (and hence proof size) of the compiled IOPP.
    #[test]
    fn norm_growth_report() {
        use super::super::limbs::num_limbs_for;
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        let input_bits = 129u32; // balanced digits at p = 130
        let msg_len = 1usize << 10;
        let q = f167772161::MODULUS;
        let log_q_half = 32 - (q / 2).leading_zeros();
        for depth in [2usize, 4, 6, 8, 10] {
            let chain =
                FoldableIprsChain::<ChainConfigF167772161>::new(msg_len, 4, depth).unwrap();
            let msg: Vec<Int<3>> = (0..msg_len)
                .map(|_| random_digit::<3>(&mut rng, input_bits))
                .collect();
            let cw: Vec<Int<10>> = chain.encode_at_level::<_, _, true>(0, &msg);
            let measured = max_bits(&cw);
            let worst =
                input_bits + (depth as u32 + 1) * log_q_half + msg_len.ilog2();
            let growth = measured.saturating_sub(input_bits);
            eprintln!(
                "depth {depth:2}: measured {measured:4} bits (growth {growth:3},                  {:.1}/level) vs worst-case {worst} bits",
                f64::from(growth) / depth as f64
            );
            assert!(measured <= worst, "worst-case bound violated");
            // Sanity for the IOPP type budget used by the experiments.
            assert!(num_limbs_for(input_bits, 130) == 1);
        }
    }

    #[test]
    fn big_field_chain_works() {
        // The locality identity over the 5*2^25+1 field.
        let chain = FoldableIprsChain::<ChainConfigF167772161>::new(64, 4, 4).unwrap();
        let msg: Vec<Int<3>> = (0..64).map(|i| Int::from(13 * i - 100)).collect();
        let cw: Vec<I8w> = chain.encode_at_level::<_, _, true>(0, &msg);
        let even: Vec<Int<3>> = msg.iter().step_by(2).cloned().collect();
        let odd: Vec<Int<3>> = msg.iter().skip(1).step_by(2).cloned().collect();
        let cw_e: Vec<I8w> = chain.encode_at_level::<_, _, true>(1, &even);
        let cw_o: Vec<I8w> = chain.encode_at_level::<_, _, true>(1, &odd);
        let half = chain.codeword_len_at(1);
        for j in 0..half {
            let tau = chain.twiddle(0, j);
            let t = cw_o[j].mul_by_scalar::<true>(&tau).unwrap();
            assert_eq!(cw[j], add!(cw_e[j], t));
            assert_eq!(cw[j + half], sub!(cw_e[j], t));
        }
    }

    #[test]
    fn reduces_to_rs_mod_q() {
        // Lift-of-RS sanity: the codeword reduced mod q equals the RS
        // evaluation of the message over F_q on the FFT domain.
        let q = u64::from(<ChainConfigF65537 as ChainConfig>::FIELD_MODULUS);
        let msg_len = 16usize;
        let rep = 4usize;
        let n = msg_len * rep;
        let chain = FoldableIprsChain::<ChainConfigF65537>::new(msg_len, rep, 4).unwrap();
        let msg: Vec<Int<3>> = (0..msg_len as i32).map(|i| Int::from(3 * i + 1)).collect();
        let cw: Vec<I8w> = chain.encode_at_level::<_, _, true>(0, &msg);
        assert_eq!(cw.len(), n);

        let domain = Radix2EvaluationDomain::<f65537::Field>::new(n).unwrap();
        let omegas: Vec<u64> = domain
            .elements()
            .map(|x| x.into_bigint().0[0])
            .collect();
        for (j, omega_j) in omegas.iter().enumerate() {
            // sum_i msg[i] * omega^{i*j} mod q
            let mut acc: u64 = 0;
            let mut pw: u64 = 1;
            for x in &msg {
                let xi = lift_to_field(&x.resize::<8>(), q);
                acc = (acc + xi * pw) % q;
                pw = (pw * omega_j) % q;
            }
            assert_eq!(lift_to_field(&cw[j], q), acc, "mismatch at position {j}");
        }
    }
}
