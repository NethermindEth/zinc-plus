mod pntt;

use crate::{
    code::{
        LinearCode,
        iprs::pntt::radix8::{FieldMulByTwiddle, MBSMulByTwiddle},
    },
    pcs::structs::ZipTypes,
};
use crypto_primitives::{FromPrimitiveWithConfig, FromWithConfig};
use itertools::Itertools;
use num_traits::{CheckedAdd, CheckedMul};
use std::{iter::Sum, marker::PhantomData, ops::AddAssign};
use zinc_utils::{from_ref::FromRef, mul_by_scalar::MulByScalar};

pub use pntt::radix8::params::{PnttConfigF2_16_1, Radix8PnttParams};

/// Pseudo Reed-Solomon encoder over the integers. Internally uses a
/// radix-8 NTT-style recursion with a base Vandermonde matrix sized
/// `base_len x base_dim` (defaults to 64x32).
#[derive(Debug, Clone)]
pub struct IprsCode<Zt: ZipTypes, Config: pntt::radix8::params::Config> {
    pntt_params: Radix8PnttParams<Config>,
    _phantom: PhantomData<Zt>,
}

impl<Zt, Config> IprsCode<Zt, Config>
where
    Zt: ZipTypes,
    Config: pntt::radix8::params::Config,
{
    /// Encode without modular reduction, purely over the integers.
    fn encode_inner<R>(&self, row: &[R]) -> Vec<R>
    where
        R: CheckedAdd
            + for<'a> AddAssign<&'a R>
            + CheckedMul
            + for<'a> MulByScalar<&'a Config::Int>
            + Sum
            + Clone
            + Send
            + Sync,
    {
        assert_eq!(
            row.len(),
            Config::INPUT_LEN,
            "Input length {} does not match expected row length {}",
            row.len(),
            Config::INPUT_LEN
        );

        pntt::radix8::pntt(row, &self.pntt_params, MBSMulByTwiddle)
    }

    // Do the encoding but make use of the fact
    // that we are dealing with a field.
    fn encode_inner_f<F, T>(&self, row: &[F]) -> Vec<F>
    where
        F: FromWithConfig<T>,
        Config::Int: Into<T>,
        T: Clone + Send + Sync,
    {
        assert_eq!(
            row.len(),
            Config::INPUT_LEN,
            "Input length {} does not match expected row length {}",
            row.len(),
            Config::INPUT_LEN
        );

        pntt::radix8::pntt(
            row,
            &self.pntt_params,
            FieldMulByTwiddle::<_, T>::new(row[0].cfg().clone()),
        )
    }
}

impl<Zt: ZipTypes, Config> LinearCode<Zt> for IprsCode<Zt, Config>
where
    Zt: ZipTypes,
    Config: pntt::radix8::params::Config,
    Zt::CombR: for<'a> MulByScalar<&'a Config::Int>,
    Zt::Cw: CheckedAdd + for<'a> MulByScalar<&'a Config::Int>,
    Config::Int: Into<i64>,
{
    const REPETITION_FACTOR: usize = Config::OUTPUT_LEN / Config::INPUT_LEN;

    #[allow(clippy::arithmetic_side_effects)]
    fn new(poly_size: usize) -> Self {
        assert_eq!(
            poly_size % Config::INPUT_LEN,
            0,
            "Polynomial size {} is not a multiple of row length {}",
            poly_size,
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

        self.encode_inner(
            &row.iter()
                .map(<Zt::Cw as FromRef<Zt::Eval>>::from_ref)
                .collect_vec(),
        )
    }

    fn row_len(&self) -> usize {
        Config::INPUT_LEN
    }

    fn codeword_len(&self) -> usize {
        Config::OUTPUT_LEN
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
