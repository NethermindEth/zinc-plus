mod pntt;

use num_traits::CheckedAdd;
use std::{iter::Sum, marker::PhantomData, ops::AddAssign};

use crypto_primitives::{FromPrimitiveWithConfig, FromWithConfig};
use itertools::Itertools;
use num_traits::{CheckedMul, ConstZero, Zero};
use zinc_utils::{from_ref::FromRef, mul_by_scalar::MulByScalar};

use crate::{
    code::{
        LinearCode,
        iprs::pntt::radix8::{FieldMulByTwiddle, MBSMulByTwiddle, Radix8PNTTParams},
    },
    pcs::structs::ZipTypes,
};

pub use pntt::radix8::{Config, PNTTConfigF2_16_1};

/// Pseudo Reed-Solomon encoder over the integers. Internally uses a
/// radix-8 NTT-style recursion with a base Vandermonde matrix sized
/// `base_len x base_dim` (defaults to 64x32).
#[derive(Debug, Clone)]
pub struct IprsCode<Zt: ZipTypes, Config: pntt::radix8::Config> {
    pntt_params: Radix8PNTTParams<Config>,
    _phantom: PhantomData<Zt>,
}

impl<Zt, Config> IprsCode<Zt, Config>
where
    Zt: ZipTypes,
    Config: pntt::radix8::Config,
{
    /// Encode without modular reduction, purely over the integers.
    fn encode_inner<In, Out>(&self, row: &[In], zero: Out) -> Vec<Out>
    where
        In: Clone + Send + Sync,
        Out: From<In>
            + Clone
            + CheckedAdd
            + for<'a> AddAssign<&'a Out>
            + CheckedMul
            + for<'a> MulByScalar<&'a Config::Int>
            + Send
            + Sync
            + Sum,
        for<'a> &'a Out: From<&'a In>,
    {
        assert_eq!(
            row.len(),
            Config::INPUT_LEN,
            "Input length {} does not match expected row length {}",
            row.len(),
            Config::INPUT_LEN
        );

        pntt::radix8::pntt(row, zero, &self.pntt_params, MBSMulByTwiddle)
    }

    // Do the encoding but make use of the fact the input
    // and the output are the same field.
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

        let field_cfg = row[0].cfg().clone();
        let zero = F::zero_with_cfg(&field_cfg);

        pntt::radix8::pntt(
            row,
            zero,
            &self.pntt_params,
            FieldMulByTwiddle::<_, T>::new(field_cfg),
        )
    }
}

impl<Zt: ZipTypes, Config> LinearCode<Zt> for IprsCode<Zt, Config>
where
    Zt: ZipTypes,
    Config: pntt::radix8::Config,
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

        assert!(
            Config::OUTPUT_LEN == Config::INPUT_LEN * Self::REPETITION_FACTOR,
            "Codeword length {} must equal row length {} times repetition factor {}",
            Config::OUTPUT_LEN,
            Config::INPUT_LEN,
            Self::REPETITION_FACTOR
        );

        Self {
            pntt_params: Radix8PNTTParams::new(),
            _phantom: Default::default(),
        }
    }

    fn encode(&self, row: &[Zt::Eval]) -> Vec<Zt::Cw> {
        assert!(
            row.len() == Config::INPUT_LEN,
            "Input length {} does not match expected row length {}",
            row.len(),
            Config::INPUT_LEN
        );

        self.encode_inner(
            &row.iter()
                .map(<Zt::Cw as FromRef<Zt::Eval>>::from_ref)
                .collect_vec(),
            Zt::Cw::zero(),
        )
    }

    fn row_len(&self) -> usize {
        Config::INPUT_LEN
    }

    fn codeword_len(&self) -> usize {
        Config::OUTPUT_LEN
    }

    fn encode_wide(&self, row: &[Zt::CombR]) -> Vec<Zt::CombR> {
        self.encode_inner(row, Zt::CombR::ZERO)
    }

    fn encode_f<F>(&self, row: &[F]) -> Vec<F>
    where
        F: FromPrimitiveWithConfig + FromRef<F>,
    {
        self.encode_inner_f(row)
    }
}
