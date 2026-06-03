mod pntt;

use crate::{ZipError, code::LinearCode, pcs::structs::ZipTypes};
use crypto_primitives::{FromPrimitiveWithConfig, FromWithConfig};
use num_traits::{CheckedAdd, CheckedMul};
use pntt::radix8::params::Config as PnttConfig;
pub use pntt::radix8::params::{PnttConfigF65537, PnttInt, Radix8PnttParams};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::{
    fmt::Debug,
    iter::Sum,
    marker::PhantomData,
    ops::{Add, AddAssign},
};
use zinc_utils::{cfg_into_iter, from_ref::FromRef, mul_by_scalar::MulByScalar};

/// Pseudo Reed-Solomon encoder over the integers. Internally uses a
/// radix-8 NTT-style recursion with a base Vandermonde matrix sized
/// `base_len x base_dim` (defaults to 64x32).
#[derive(Clone)]
pub struct IprsCode<Zt: ZipTypes, Config: PnttConfig, const REP: usize, const CHECK: bool> {
    pntt_params: Radix8PnttParams<Config>,
    _phantom: PhantomData<Zt>,
}

impl<Zt, Config, const REP: usize, const CHECK: bool> IprsCode<Zt, Config, REP, CHECK>
where
    Zt: ZipTypes,
    Config: PnttConfig,
{
    pub fn new(row_len: usize, depth: usize) -> Result<Self, ZipError> {
        // TODO(alex): Calculate max expected Zt::Cw::COEFF_BIT_WIDTH to ensure in
        //             advance that the encoding will not overflow
        Ok(Self {
            pntt_params: Radix8PnttParams::new(row_len, depth, REP)?,
            _phantom: Default::default(),
        })
    }

    /// Create a new IPRS code with the optimal depth heuristics trying to keep
    /// number of columns in the base matrix small.
    /// Currently, keeps number of columns <= 2^8 but this might be tweaked in
    /// the future.
    ///
    /// At higher inverse-rates extra recursion layers are added on top of the
    /// base heuristic: the higher redundancy lets the PCS get away with a
    /// smaller base matrix, and the extra layers trade a small amount of
    /// encoding work for that smaller base. One extra layer is added at
    /// inverse-rate 8, and a second one at inverse-rate 16.
    pub fn new_with_optimal_depth(row_len: usize) -> Result<Self, ZipError> {
        const MAX_BASE_COLS_LOG2: usize = 8;

        let target_base_len = 1 << MAX_BASE_COLS_LOG2;
        // We want depth to be at least 1.
        let base_depth =
            1.max(((1.max(row_len / target_base_len)).ilog2() as usize).div_ceil(3));
        let extra = if REP >= 16 {
            2
        } else if REP >= 8 {
            1
        } else {
            0
        };
        let depth = base_depth + extra;

        Self::new(row_len, depth)
    }

    /// **Experimental (depth-0 only).** Encode a `{0,1}` message by summing the
    /// Vandermonde generator columns at the set bit positions — no
    /// multiplications, just `weight(bits)` column-additions per output cell.
    ///
    /// At `depth == 0` the radix-8 PNTT degenerates to the full
    /// `codeword_len × row_len` Vandermonde matvec: `base_matrix[i][c]` is the
    /// generator entry for output cell `i` and input position `c` (the first
    /// column is the implicit all-ones column `root_i^0 = 1`). For a binary
    /// input this computes the *identical* codeword as
    /// [`LinearCode::encode`] (asserted in the experiment test
    /// `columnsum_experiment::experiment_depth0_columnsum_vs_fft`), but skips
    /// every multiply: `output[i] = Σ_{c : bits[c]} base_matrix[i][c]`.
    ///
    /// Only valid for `depth == 0` (higher depth keeps no materialised full
    /// generator to sum over); panics otherwise.
    pub fn encode_binary_columnsum<Out>(&self, bits: &[bool]) -> Vec<Out>
    where
        Out: num_traits::Zero + Clone + Send + Sync + From<PnttInt>,
    {
        assert_eq!(
            self.pntt_params.depth, 0,
            "encode_binary_columnsum requires a depth-0 (pure Vandermonde) code",
        );
        assert_eq!(
            bits.len(),
            self.pntt_params.row_len,
            "Input length {} does not match expected row length {}",
            bits.len(),
            self.pntt_params.row_len,
        );

        let bm = &self.pntt_params.base_matrix;
        let set_cols: Vec<usize> = (0..self.pntt_params.row_len)
            .filter(|&c| bits[c])
            .collect();

        cfg_into_iter!(0..self.pntt_params.codeword_len)
            .map(|i| {
                let row = &bm[i];
                set_cols
                    .iter()
                    .fold(Out::zero(), |acc, &c| acc + Out::from(row[c]))
            })
            .collect()
    }

    /// Encode without modular reduction, purely over the integers.
    fn encode_inner<In, Out>(&self, row: &[In]) -> Vec<Out>
    where
        In: for<'a> MulByScalar<&'a PnttInt, Out> + Clone + Send + Sync,
        Out: CheckedAdd
            + for<'a> AddAssign<&'a Out>
            + for<'a> Add<&'a Out, Output = Out>
            + CheckedMul
            + for<'a> MulByScalar<&'a PnttInt>
            + Sum
            + FromRef<In>
            + Clone
            + Debug
            + Send
            + Sync,
    {
        assert_eq!(
            row.len(),
            self.pntt_params.row_len,
            "Input length {} does not match expected row length {}",
            row.len(),
            self.pntt_params.row_len,
        );

        macro_rules! mul_fn {
            () => {
                |v, tw| {
                    v.mul_by_scalar::<CHECK>(tw)
                        .expect("Multiplication by twiddle should not overflow")
                }
            };
        }

        pntt::radix8::pntt::<_, _, _, CHECK>(row, &self.pntt_params, mul_fn!(), mul_fn!())
    }

    // Do the encoding but make use of the fact
    // that we are dealing with a field.
    fn encode_inner_f<F>(&self, row: &[F]) -> Vec<F>
    where
        F: FromWithConfig<PnttInt> + FromRef<F>,
    {
        assert_eq!(
            row.len(),
            self.pntt_params.row_len,
            "Input length {} does not match expected row length {}",
            row.len(),
            self.pntt_params.row_len,
        );

        let mul_fn = |f: &F, tw: &PnttInt| f.clone() * F::from_with_cfg(*tw, f.cfg());

        pntt::radix8::pntt::<_, _, _, CHECK>(row, &self.pntt_params, mul_fn, mul_fn)
    }
}

impl<Zt: ZipTypes, Config, const REP: usize, const CHECK: bool> LinearCode<Zt>
    for IprsCode<Zt, Config, REP, CHECK>
where
    Zt: ZipTypes,
    Config: PnttConfig,
    Zt::Eval: for<'a> MulByScalar<&'a PnttInt, Zt::Cw>,
    Zt::CombR: for<'a> MulByScalar<&'a PnttInt>,
    Zt::Cw: CheckedAdd + for<'a> MulByScalar<&'a PnttInt>,
{
    const REPETITION_FACTOR: usize = REP;

    fn encode(&self, row: &[Zt::Eval]) -> Vec<Zt::Cw> {
        assert_eq!(
            row.len(),
            self.pntt_params.row_len,
            "Input length {} does not match expected row length {}",
            row.len(),
            self.pntt_params.row_len,
        );

        self.encode_inner(row)
    }

    fn row_len(&self) -> usize {
        self.pntt_params.row_len
    }

    fn codeword_len(&self) -> usize {
        self.pntt_params.codeword_len
    }

    fn params_string(&self) -> String {
        format!(
            "row_len={}, rate=1/{REP}, depth={}",
            self.row_len(),
            self.pntt_params.depth
        )
    }

    fn encode_wide(&self, row: &[Zt::CombR]) -> Vec<Zt::CombR> {
        self.encode_inner(row)
    }

    fn encode_f<F>(&self, row: &[F]) -> Vec<F>
    where
        F: FromPrimitiveWithConfig + FromRef<F>,
    {
        self.encode_inner_f(row)
    }
}

impl<Zt, Config, const REP: usize, const CHECK: bool> Debug for IprsCode<Zt, Config, REP, CHECK>
where
    Zt: ZipTypes,
    Config: PnttConfig,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IprsCode")
            .field("pntt_params", &self.pntt_params)
            .finish()
    }
}

impl<Zt, Config, const REP: usize, const CHECK: bool> PartialEq for IprsCode<Zt, Config, REP, CHECK>
where
    Config: PnttConfig,
    Zt: ZipTypes,
{
    fn eq(&self, other: &Self) -> bool {
        self.pntt_params == other.pntt_params
    }
}

impl<Zt, Config, const REP: usize, const CHECK: bool> Eq for IprsCode<Zt, Config, REP, CHECK>
where
    Zt: ZipTypes,
    Config: PnttConfig,
{
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pcs::{structs::ZipPlus, test_utils::*};
    use crypto_bigint::U64;
    use crypto_primitives::{
        FixedSemiring, boolean::Boolean, crypto_bigint_int::Int, crypto_bigint_uint::Uint,
    };
    use rand::{
        distr::{Distribution, StandardUniform},
        prelude::ThreadRng,
    };
    use zinc_poly::{
        mle::{DenseMultilinearExtension, MultilinearExtensionRand},
        univariate::{
            binary::{BinaryPoly, BinaryPolyInnerProduct},
            dense::{DensePolyInnerProduct, DensePolynomial},
        },
    };
    use zinc_primality::MillerRabin;
    use zinc_transcript::traits::ConstTranscribable;
    use zinc_utils::{
        CHECKED,
        inner_product::{MBSInnerProduct, ScalarProduct},
        named::Named,
    };

    const INT_LIMBS: usize = U64::LIMBS;
    const N: usize = INT_LIMBS;
    const K: usize = INT_LIMBS * 4;
    const M: usize = INT_LIMBS * 8;
    type Zt = TestZipTypes<N, K, M>;

    type Code = IprsCode<Zt, PnttConfigF65537, REP_FACTOR, CHECKED>;

    #[test]
    fn new_with_different_params() {
        assert!(Code::new(1, 0).is_ok());
        assert!(Code::new(8, 0).is_ok());
        assert!(Code::new(1, 1).is_err());
        assert!(Code::new(8, 1).is_ok());

        assert!(Code::new_with_optimal_depth(1).is_err());
        assert!(Code::new_with_optimal_depth(8).is_ok());
        assert!(Code::new_with_optimal_depth(12).is_err());
        assert!(Code::new_with_optimal_depth(16).is_ok());
    }

    fn do_encode<Zt, const REP: usize>(num_vars: usize)
    where
        Zt: ZipTypes,
        Zt::Eval: for<'a> MulByScalar<&'a PnttInt, Zt::Cw>,
        Zt::CombR: for<'a> MulByScalar<&'a PnttInt>,
        Zt::Cw: CheckedAdd + for<'a> MulByScalar<&'a PnttInt>,
        StandardUniform: Distribution<Zt::Eval>,
    {
        let mut rng = ThreadRng::default();
        let poly_size: usize = 1 << num_vars;
        let mle = DenseMultilinearExtension::rand(num_vars, &mut rng);

        let code = IprsCode::<Zt, PnttConfigF65537, 4, CHECKED>::new_with_optimal_depth(poly_size)
            .unwrap();
        let pp = ZipPlus::setup(poly_size, code);
        ZipPlus::<Zt, _>::encode_rows(&pp, &mle.evaluations);
    }

    /// Test the widest integer encoding used in benchmarks
    #[test]
    fn encode_bench_int() {
        #[derive(Clone, Debug)]
        struct BenchZipTypes {}
        impl ZipTypes for BenchZipTypes {
            const NUM_COLUMN_OPENINGS: usize = 147;
            type Eval = i32;
            type Cw = i128;
            type Fmod = Uint<{ INT_LIMBS * 4 }>;
            type PrimeTest = MillerRabin;
            type Chal = i128;
            type Pt = i128;
            type CombR = Int<{ INT_LIMBS * 3 }>;
            type Comb = Self::CombR;
            type EvalDotChal = ScalarProduct;
            type CombDotChal = ScalarProduct;
            type ArrCombRDotChal = MBSInnerProduct;
        }

        do_encode::<BenchZipTypes, 4>(14);
    }

    /// Test the widest binary polynomial encoding used in benchmarks
    #[test]
    fn encode_bench_poly() {
        const D_PLUS_ONE: usize = 32;

        #[derive(Clone, Debug)]
        struct BenchZipPlusTypes<CwCoeff>(PhantomData<CwCoeff>);
        impl<CwCoeff> ZipTypes for BenchZipPlusTypes<CwCoeff>
        where
            CwCoeff: ConstTranscribable
                + Copy
                + Default
                + FromRef<Boolean>
                + Named
                + FixedSemiring
                + Send
                + Sync,
            Int<5>: FromRef<CwCoeff>,
        {
            const NUM_COLUMN_OPENINGS: usize = 147;
            type Eval = BinaryPoly<D_PLUS_ONE>;
            type Cw = DensePolynomial<CwCoeff, D_PLUS_ONE>;
            type Fmod = Uint<{ INT_LIMBS * 4 }>;
            type PrimeTest = MillerRabin;
            type Chal = i128;
            type Pt = i128;
            type CombR = Int<{ INT_LIMBS * 5 }>;
            type Comb = DensePolynomial<Self::CombR, D_PLUS_ONE>;
            type EvalDotChal = BinaryPolyInnerProduct<Self::Chal, D_PLUS_ONE>;
            type CombDotChal = DensePolyInnerProduct<
                Self::CombR,
                Self::Chal,
                Self::CombR,
                MBSInnerProduct,
                D_PLUS_ONE,
            >;
            type ArrCombRDotChal = MBSInnerProduct;
        }

        do_encode::<BenchZipPlusTypes<i64>, 4>(14);
    }
}

/// Experiment (CLAUDE.md / `documentation/f2x-sha-todo.md`): for `{0,1}`
/// inputs, is a depth-0 IPRS code encoded by *summing Vandermonde columns*
/// (no multiplications) faster than the depth-`d` radix-8 FFT? And how do the
/// resulting codeword entry sizes affect the downstream Merkle commit?
///
/// Run with:
/// ```text
/// cargo test -p zip-plus --release --features parallel \
///     experiment_depth0_columnsum_vs_fft -- --ignored --nocapture
/// ```
#[cfg(test)]
mod columnsum_experiment {
    use super::{IprsCode, PnttConfigF65537};
    use crate::code::LinearCode;
    use crate::merkle::MerkleTree;
    use crate::pcs::structs::ZipTypes;
    use crypto_bigint::U64;
    use crypto_primitives::{
        DenseRowMatrix, crypto_bigint_int::Int, crypto_bigint_uint::Uint,
    };
    use rand::{RngCore, SeedableRng, rngs::StdRng};
    use std::hint::black_box;
    use std::time::Instant;
    use zinc_primality::MillerRabin;
    use zinc_transcript::traits::ConstTranscribable;
    use zinc_utils::inner_product::{MBSInnerProduct, ScalarProduct};

    const INT_LIMBS: usize = U64::LIMBS;

    /// `Eval = i32` (binary inputs), `Cw = i128` (never overflows so we can
    /// measure the true entry magnitude before deciding a narrower commit
    /// type). Mirrors the `encode_bench_int` types.
    #[derive(Clone, Debug)]
    struct Exp;
    impl ZipTypes for Exp {
        const NUM_COLUMN_OPENINGS: usize = 147;
        type Eval = i32;
        type Cw = i128;
        type Fmod = Uint<{ INT_LIMBS * 4 }>;
        type PrimeTest = MillerRabin;
        type Chal = i128;
        type Pt = i128;
        type CombR = Int<{ INT_LIMBS * 3 }>;
        type Comb = Self::CombR;
        type EvalDotChal = ScalarProduct;
        type CombDotChal = ScalarProduct;
        type ArrCombRDotChal = MBSInnerProduct;
    }

    /// Min wall-clock (ms) of `f` over `reps` runs.
    fn bench<F: FnMut()>(reps: usize, mut f: F) -> f64 {
        let mut best = f64::INFINITY;
        for _ in 0..reps {
            let t = Instant::now();
            f();
            best = best.min(t.elapsed().as_secs_f64() * 1e3);
        }
        best
    }

    /// Smallest signed-int byte-width that holds every codeword entry.
    fn min_fit_bytes(data: &[i128]) -> usize {
        let max_abs = data.iter().map(|&x| x.unsigned_abs()).max().unwrap_or(0);
        if max_abs < (1u128 << 15) {
            2
        } else if max_abs < (1u128 << 31) {
            4
        } else if max_abs < (1u128 << 63) {
            8
        } else {
            16
        }
    }

    /// Build a `DenseRowMatrix<T>` (downcasting the i128 codeword to `T`) and
    /// time the grouped Merkle commit over it. `None` if any entry overflows
    /// `T`.
    fn commit_ms<T>(data128: &[i128], num_rows: usize, cw_len: usize, reps: usize) -> Option<f64>
    where
        T: TryFrom<i128> + ConstTranscribable + Send + Sync + Clone,
    {
        let data: Vec<T> = data128
            .iter()
            .map(|&x| T::try_from(x).ok())
            .collect::<Option<_>>()?;
        let mats = vec![DenseRowMatrix {
            num_rows,
            num_cols: cw_len,
            data,
        }];
        Some(bench(reps, || {
            let t = MerkleTree::new_from_row_major_grouped(&mats, num_rows, cw_len, 1);
            black_box(t.root());
        }))
    }

    fn run_case<const REP: usize>(row_len: usize, num_rows: usize) {
        let cw_len = row_len * REP;
        // Guard: the depth-0 Vandermonde matrix is fully materialised
        // (codeword_len × row_len). Skip if it would exceed ~512 MB.
        if cw_len.saturating_mul(row_len) > 64_000_000 {
            println!(
                "  row_len={row_len:<5} rate=1/{REP}  SKIPPED (depth-0 matrix \
                 {cw_len}×{row_len} ≈ {} MB too large)",
                cw_len * row_len * 8 / 1_000_000
            );
            return;
        }

        let code_fft = IprsCode::<Exp, PnttConfigF65537, REP, false>::new_with_optimal_depth(row_len)
            .expect("fft code");
        let code_d0 =
            IprsCode::<Exp, PnttConfigF65537, REP, false>::new(row_len, 0).expect("depth-0 code");

        // Random binary input matrix (num_rows × row_len).
        let mut rng = StdRng::seed_from_u64(0xC0FFEE ^ (row_len as u64) ^ ((REP as u64) << 32));
        let rows_i32: Vec<Vec<i32>> = (0..num_rows)
            .map(|_| (0..row_len).map(|_| (rng.next_u32() & 1) as i32).collect())
            .collect();
        let rows_bits: Vec<Vec<bool>> = rows_i32
            .iter()
            .map(|r| r.iter().map(|&x| x == 1).collect())
            .collect();

        // Correctness gate: column-sum == depth-0 FFT-path encode, bit for bit.
        for (r_i32, r_bits) in rows_i32.iter().zip(rows_bits.iter()) {
            let enc: Vec<i128> = code_d0.encode(r_i32);
            let cs: Vec<i128> = code_d0.encode_binary_columnsum(r_bits);
            assert_eq!(enc, cs, "column-sum must equal depth-0 encode");
        }

        let reps = 12;

        // --- Encode timing (whole matrix, internal rayon over codeword_len) ---
        let t_fft = bench(reps, || {
            for r in &rows_i32 {
                black_box(code_fft.encode(r));
            }
        });
        let t_d0_enc = bench(reps, || {
            for r in &rows_i32 {
                black_box(code_d0.encode(r));
            }
        });
        let t_d0_cs = bench(reps, || {
            for r in &rows_bits {
                black_box(code_d0.encode_binary_columnsum::<i128>(r));
            }
        });

        // --- Codeword magnitudes ---
        let cw_fft: Vec<i128> = rows_i32.iter().flat_map(|r| code_fft.encode(r)).collect();
        let cw_d0: Vec<i128> = rows_bits
            .iter()
            .flat_map(|r| code_d0.encode_binary_columnsum::<i128>(r))
            .collect();
        let bits_of = |d: &[i128]| {
            let m = d.iter().map(|&x| x.unsigned_abs()).max().unwrap_or(0);
            if m == 0 { 0 } else { 128 - m.leading_zeros() }
        };
        let (fft_bits, d0_bits) = (bits_of(&cw_fft), bits_of(&cw_d0));
        let (fft_b, d0_b) = (min_fit_bytes(&cw_fft), min_fit_bytes(&cw_d0));

        // --- Merkle commit timing, at each codeword's minimal cell type and
        //     at the shared i128 baseline ---
        let commit_at = |data: &[i128], bytes: usize| match bytes {
            2 => commit_ms::<i16>(data, num_rows, cw_len, reps),
            4 => commit_ms::<i32>(data, num_rows, cw_len, reps),
            8 => commit_ms::<i64>(data, num_rows, cw_len, reps),
            _ => commit_ms::<i128>(data, num_rows, cw_len, reps),
        };
        let c_fft_min = commit_at(&cw_fft, fft_b).unwrap();
        let c_d0_min = commit_at(&cw_d0, d0_b).unwrap();
        let c_fft_128 = commit_ms::<i128>(&cw_fft, num_rows, cw_len, reps).unwrap();
        let c_d0_128 = commit_ms::<i128>(&cw_d0, num_rows, cw_len, reps).unwrap();

        println!(
            "  row_len={row_len:<5} rate=1/{REP}  ({})",
            code_fft.params_string()
        );
        println!(
            "    encode/matrix(×{num_rows}rows): FFT {t_fft:>8.3} ms | d0-mul {t_d0_enc:>8.3} ms | \
             d0-colsum {t_d0_cs:>8.3} ms  (colsum vs FFT: {:.2}×)",
            t_fft / t_d0_cs
        );
        println!(
            "    max entry bits: FFT {fft_bits:>3} ({fft_b}B cell) | d0 {d0_bits:>3} ({d0_b}B cell)"
        );
        println!(
            "    commit@mincell: FFT {c_fft_min:>8.3} ms ({fft_b}B) | d0 {c_d0_min:>8.3} ms ({d0_b}B)  \
             || commit@i128: FFT {c_fft_128:>8.3} ms | d0 {c_d0_128:>8.3} ms"
        );
    }

    #[test]
    #[ignore = "experiment harness; run explicitly with --ignored --nocapture"]
    fn experiment_depth0_columnsum_vs_fft() {
        let num_rows = 64;
        println!("\n=== IPRS depth-0 column-sum vs depth-d FFT (binary input, num_rows={num_rows}) ===");
        println!("    (encode min over 12 reps; commit = grouped Merkle, group_size=1)\n");
        for &row_len in &[64usize, 256, 1024] {
            run_case::<4>(row_len, num_rows);
            run_case::<8>(row_len, num_rows);
        }
    }
}
