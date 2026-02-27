mod pntt;

use crate::{
    code::{
        LinearCode,
        iprs::pntt::radix8::{
            FieldMulByTwiddle, MBSMulByTwiddle, MulByTwiddle, WideningMulByTwiddle,
            params::Config as PnttConfig,
        },
    },
    pcs::structs::ZipTypes,
};
use crypto_primitives::{FromPrimitiveWithConfig, FromWithConfig};
use num_traits::{CheckedAdd, CheckedMul, ConstZero};
pub use pntt::radix8::params::{
    PnttConfigF2_16_1,
    PnttConfigF2_16R4B1, PnttConfigF2_16R4B2, PnttConfigF2_16R4B4,
    PnttConfigF2_16R4B16, PnttConfigF2_16R4B32, PnttConfigF2_16R4B64,
    PnttInt, Radix8PnttParams,
};
#[cfg(feature = "parallel")]
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::{
    fmt::Debug,
    iter::Sum,
    marker::PhantomData,
    ops::{Add, AddAssign},
};
use zinc_utils::{
    CHECKED,
    from_ref::FromRef,
    mul_by_scalar::{MulByScalar, WideningMulByScalar},
};

/// Pseudo Reed-Solomon encoder over the integers. Internally uses a
/// radix-8 NTT-style recursion with a base Vandermonde matrix sized
/// `base_len x base_dim` (defaults to 64x32).
#[derive(Debug, Clone)]
pub struct IprsCode<Zt: ZipTypes, Config: PnttConfig, MT, const CHECK_ADDITION: bool> {
    pntt_params: Radix8PnttParams<Config>,
    _phantom: PhantomData<(Zt, MT)>,
}

impl<Zt, Config, MT, const CHECK_ADDITION: bool> IprsCode<Zt, Config, MT, CHECK_ADDITION>
where
    Zt: ZipTypes,
    Config: PnttConfig,
{
    /// Encode without modular reduction, purely over the integers.
    fn encode_inner<In, Out, M>(&self, row: &[In]) -> Vec<Out>
    where
        In: Clone + Send + Sync,
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
        M: MulByTwiddle<In, PnttInt, Output = Out>,
    {
        assert_eq!(
            row.len(),
            Config::INPUT_LEN,
            "Input length {} does not match expected row length {}",
            row.len(),
            Config::INPUT_LEN
        );

        pntt::radix8::pntt::<_, _, _, M, MBSMulByTwiddle<CHECK_ADDITION>, CHECK_ADDITION>(
            row,
            &self.pntt_params,
        )
    }

    // Do the encoding but make use of the fact
    // that we are dealing with a field.
    fn encode_inner_f<F>(&self, row: &[F]) -> Vec<F>
    where
        F: FromWithConfig<PnttInt> + FromRef<F>,
    {
        assert_eq!(
            row.len(),
            Config::INPUT_LEN,
            "Input length {} does not match expected row length {}",
            row.len(),
            Config::INPUT_LEN
        );

        pntt::radix8::pntt::<
            _,
            _,
            _,
            FieldMulByTwiddle<_, PnttInt>,
            FieldMulByTwiddle<_, PnttInt>,
            CHECK_ADDITION,
        >(row, &self.pntt_params)
    }
}

impl<Zt: ZipTypes, Config, MT, const CHECK_ADDITION: bool> LinearCode<Zt>
    for IprsCode<Zt, Config, MT, CHECK_ADDITION>
where
    Zt: ZipTypes,
    Config: PnttConfig,
    Zt::CombR: for<'a> MulByScalar<&'a PnttInt>,
    Zt::Cw: CheckedAdd + for<'a> MulByScalar<&'a PnttInt>,
    MT: WideningMulByScalar<Zt::Eval, PnttInt, Output = Zt::Cw>,
{
    const REPETITION_FACTOR: usize = Config::OUTPUT_LEN / Config::INPUT_LEN;

    #[allow(clippy::arithmetic_side_effects)]
    fn new(row_len: usize) -> Self {
        assert_eq!(
            row_len,
            Config::INPUT_LEN,
            "Row length {} does not match expected row length {}",
            row_len,
            Config::INPUT_LEN
        );

        assert_eq!(
            Config::OUTPUT_LEN,
            Config::INPUT_LEN * Self::REPETITION_FACTOR,
            "Codeword length {} must equal row length {} times repetition factor {}",
            Config::OUTPUT_LEN,
            Config::INPUT_LEN,
            Self::REPETITION_FACTOR
        );

        Self {
            pntt_params: Radix8PnttParams::new(),
            _phantom: Default::default(),
        }
    }

    fn encode(&self, row: &[Zt::Eval]) -> Vec<Zt::Cw> {
        assert_eq!(
            row.len(),
            Config::INPUT_LEN,
            "Input length {} does not match expected row length {}",
            row.len(),
            Config::INPUT_LEN
        );

        self.encode_inner::<_, _, WideningMulByTwiddle<MT>>(row)
    }

    fn row_len(&self) -> usize {
        Config::INPUT_LEN
    }

    fn codeword_len(&self) -> usize {
        Config::OUTPUT_LEN
    }

    fn encode_wide(&self, row: &[Zt::CombR]) -> Vec<Zt::CombR> {
        self.encode_inner::<_, _, MBSMulByTwiddle<CHECKED>>(row)
    }

    fn encode_f<F>(&self, row: &[F]) -> Vec<F>
    where
        F: FromPrimitiveWithConfig + FromRef<F>,
    {
        self.encode_inner_f(row)
    }

    /// Compute the encoding at only the given output positions using the
    /// precomputed encoding matrix. Each output is an inner product of
    /// one row of the encoding matrix with the input row:
    ///   `result[k] = Σ_j encoding_matrix[positions[k]][j] * row[j]`
    #[allow(clippy::arithmetic_side_effects)]
    fn encode_wide_at_positions(
        &self,
        row: &[Zt::CombR],
        positions: &[usize],
    ) -> Vec<Zt::CombR> {
        let matrix = &self.pntt_params.encoding_matrix;
        let compute = |&pos: &usize| {
            let mat_row = &matrix[pos];
            let mut acc = Zt::CombR::ZERO;
            for (j, coeff) in mat_row.iter().enumerate() {
                let term = row[j]
                    .mul_by_scalar::<false>(coeff)
                    .expect("encode_wide_at_positions: MulByScalar overflow");
                acc += term;
            }
            acc
        };

        #[cfg(feature = "parallel")]
        { positions.par_iter().map(compute).collect() }

        #[cfg(not(feature = "parallel"))]
        { positions.iter().map(compute).collect() }
    }

    /// Compute the field-element encoding at only the given output positions.
    #[allow(clippy::arithmetic_side_effects)]
    fn encode_f_at_positions<F>(&self, row: &[F], positions: &[usize]) -> Vec<F>
    where
        F: FromPrimitiveWithConfig + FromRef<F> + Send + Sync,
    {
        let matrix = &self.pntt_params.encoding_matrix;
        let field_cfg = row[0].cfg();
        let compute = |&pos: &usize| {
            let mat_row = &matrix[pos];
            mat_row.iter().enumerate().fold(
                F::zero_with_cfg(field_cfg),
                |acc, (j, &coeff)| {
                    let term = F::from_with_cfg(coeff, field_cfg) * &row[j];
                    acc + &term
                },
            )
        };

        #[cfg(feature = "parallel")]
        { positions.par_iter().map(compute).collect() }

        #[cfg(not(feature = "parallel"))]
        { positions.iter().map(compute).collect() }
    }
}