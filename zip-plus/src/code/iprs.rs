mod pntt;

use crate::{
    code::{LinearCode, iprs::pntt::radix8::params::Config as PnttConfig},
    pcs::structs::ZipTypes,
};
use crypto_primitives::{FromPrimitiveWithConfig, FromWithConfig};
use num_traits::{CheckedAdd, CheckedMul};
pub use pntt::radix8::params::{
    PnttConfigF65537_1_4, PnttConfigF65537_2_8, PnttConfigF65537_4_16, PnttConfigF65537_16_64,
    PnttConfigF65537_32_64, PnttConfigF65537_32_128, PnttConfigF65537_64_256, PnttInt,
    Radix8PnttParams,
};
use std::{
    fmt::Debug,
    iter::Sum,
    marker::PhantomData,
    ops::{Add, AddAssign},
};
use zinc_utils::{from_ref::FromRef, mul_by_scalar::MulByScalar};

/// Pseudo Reed-Solomon encoder over the integers. Internally uses a
/// radix-8 NTT-style recursion with a base Vandermonde matrix sized
/// `base_len x base_dim` (defaults to 64x32).
#[derive(Debug, Clone)]
pub struct IprsCode<Zt: ZipTypes, Config: PnttConfig, const CHECK: bool> {
    pntt_params: Radix8PnttParams<Config>,
    _phantom: PhantomData<Zt>,
}

impl<Zt, Config, const CHECK: bool> IprsCode<Zt, Config, CHECK>
where
    Zt: ZipTypes,
    Config: PnttConfig,
{
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
            Config::INPUT_LEN,
            "Input length {} does not match expected row length {}",
            row.len(),
            Config::INPUT_LEN
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
            Config::INPUT_LEN,
            "Input length {} does not match expected row length {}",
            row.len(),
            Config::INPUT_LEN
        );

        let mul_fn = |f: &F, tw: &PnttInt| f.clone() * F::from_with_cfg(*tw, f.cfg());

        pntt::radix8::pntt::<_, _, _, CHECK>(row, &self.pntt_params, mul_fn, mul_fn)
    }
}

impl<Zt: ZipTypes, Config, const CHECK: bool> LinearCode<Zt> for IprsCode<Zt, Config, CHECK>
where
    Zt: ZipTypes,
    Config: PnttConfig,
    Zt::Eval: for<'a> MulByScalar<&'a PnttInt, Zt::Cw>,
    Zt::CombR: for<'a> MulByScalar<&'a PnttInt>,
    Zt::Cw: CheckedAdd + for<'a> MulByScalar<&'a PnttInt>,
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

        self.encode_inner(row)
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
