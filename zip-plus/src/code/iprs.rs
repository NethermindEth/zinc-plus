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
use num_traits::{CheckedAdd, CheckedMul};
pub use pntt::radix8::params::{
    PnttConfigF2_16_1, PnttConfigF2_16B16, PnttConfigF2_16B64,
    PnttConfigF2_16R4B1, PnttConfigF2_16R4B2, PnttConfigF2_16R4B4,
    PnttConfigF2_16R4B16, PnttConfigF2_16R4B32, PnttConfigF2_16R4B64,
    PnttConfigF167772161, PnttConfigF167772161B16, PnttConfigF167772161B64,
    PnttConfigF167772161R4B16, PnttConfigF167772161R4B32, PnttConfigF167772161R4B64,
    PnttConfigF1179649B16,
    PnttConfigF1179649R4B16, PnttConfigF1179649R4B32, PnttConfigF1179649R4B64,
    PnttConfigF3329B8, PnttConfigF3329B16,
    PnttConfigF3329R4B2, PnttConfigF3329R4B4, PnttConfigF3329R4B8,
    PnttConfigF7340033B16, PnttConfigF7340033B32, PnttConfigF7340033B64,
    PnttConfigF7340033R4B16, PnttConfigF7340033R4B32, PnttConfigF7340033R4B64,
    PnttInt, Radix8PnttParams,
};
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
}
