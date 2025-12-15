mod pntt;

use num_traits::CheckedAdd;
use std::{iter::Sum, marker::PhantomData, ops::AddAssign};

use crypto_primitives::{FromPrimitiveWithConfig, FromWithConfig};
use num_traits::{CheckedMul, ConstZero, Zero};
use zinc_utils::{
    from_ref::FromRef,
    mul_by_scalar::{MulByScalar, WideningMulByScalar},
};

use crate::{
    code::{
        LinearCode,
        iprs::pntt::radix8::{
            FieldMulByTwiddle, MBSMulByTwiddle, MulByTwiddle, Radix8PNTTParams,
            WideningMulByTwiddle,
        },
    },
    pcs::structs::ZipTypes,
};

pub use pntt::radix8::{Config, PNTTConfigF2_16_1};

/// Pseudo Reed-Solomon encoder over the integers. Internally uses a
/// radix-8 NTT-style recursion with a base Vandermonde matrix sized
/// `base_len x base_dim` (defaults to 64x32).
#[derive(Debug, Clone)]
pub struct IprsCode<Zt: ZipTypes, Config: pntt::radix8::Config, MT> {
    pntt_params: Radix8PNTTParams<Config>,
    _phantom: PhantomData<(Zt, MT)>,
}

impl<Zt, Config, MT> IprsCode<Zt, Config, MT>
where
    Zt: ZipTypes,
    Config: pntt::radix8::Config,
{
    /// Encode without modular reduction, purely over the integers.
    fn encode_inner<In, Out, M>(&self, row: &[In], zero: Out, mul_in_by_twiddle: M) -> Vec<Out>
    where
        In: Clone + Send + Sync,
        Out: FromRef<In>
            + Clone
            + CheckedAdd
            + for<'a> AddAssign<&'a Out>
            + CheckedMul
            + for<'a> MulByScalar<&'a Config::Int>
            + Send
            + Sync
            + Sum,
        M: MulByTwiddle<In, Config::Int, Output = Out>,
    {
        assert_eq!(
            row.len(),
            Config::N,
            "Input length {} does not match expected row length {}",
            row.len(),
            Config::N
        );

        pntt::radix8::pntt(
            row,
            zero,
            &self.pntt_params,
            mul_in_by_twiddle,
            MBSMulByTwiddle,
        )
    }

    // Do the encoding but make use of the fact the input
    // and the output are the same field.
    fn encode_inner_f<F, T>(&self, row: &[F]) -> Vec<F>
    where
        F: FromWithConfig<T> + FromRef<F>,
        Config::Int: Into<T>,
        T: Clone + Send + Sync,
    {
        assert_eq!(
            row.len(),
            Config::N,
            "Input length {} does not match expected row length {}",
            row.len(),
            Config::N
        );

        let field_cfg = row[0].cfg().clone();
        let zero = F::zero_with_cfg(&field_cfg);

        let mul_by_twiddle = FieldMulByTwiddle::<_, T>::new(field_cfg);

        pntt::radix8::pntt(
            row,
            zero,
            &self.pntt_params,
            mul_by_twiddle.clone(),
            mul_by_twiddle,
        )
    }
}

impl<Zt: ZipTypes, Config, MT> LinearCode<Zt> for IprsCode<Zt, Config, MT>
where
    Zt: ZipTypes,
    Config: pntt::radix8::Config,
    Zt::CombR: for<'a> MulByScalar<&'a Config::Int>,
    Zt::Cw: CheckedAdd + for<'a> MulByScalar<&'a Config::Int>,
    Config::Int: Into<i64>,
    MT: WideningMulByScalar<Zt::Eval, Config::Int, Output = Zt::Cw>,
{
    const REPETITION_FACTOR: usize = Config::M / Config::N;

    #[allow(clippy::arithmetic_side_effects)]
    fn new(poly_size: usize) -> Self {
        assert_eq!(
            poly_size % Config::N,
            0,
            "Polynomial size {} is not a multiple of row length {}",
            poly_size,
            Config::N
        );

        assert!(
            Config::M == Config::N * Self::REPETITION_FACTOR,
            "Codeword length {} must equal row length {} times repetition factor {}",
            Config::M,
            Config::N,
            Self::REPETITION_FACTOR
        );

        Self {
            pntt_params: Radix8PNTTParams::new(),
            _phantom: Default::default(),
        }
    }

    fn encode(&self, row: &[Zt::Eval]) -> Vec<Zt::Cw> {
        assert!(
            row.len() == Config::N,
            "Input length {} does not match expected row length {}",
            row.len(),
            Config::N
        );

        self.encode_inner(row, Zt::Cw::zero(), WideningMulByTwiddle::<MT>::default())
    }

    fn row_len(&self) -> usize {
        Config::N
    }

    fn codeword_len(&self) -> usize {
        Config::M
    }

    fn encode_wide(&self, row: &[Zt::CombR]) -> Vec<Zt::CombR> {
        self.encode_inner(row, Zt::CombR::ZERO, MBSMulByTwiddle)
    }

    fn encode_f<F>(&self, row: &[F]) -> Vec<F>
    where
        F: FromPrimitiveWithConfig + FromRef<F>,
    {
        self.encode_inner_f(row)
    }
}
