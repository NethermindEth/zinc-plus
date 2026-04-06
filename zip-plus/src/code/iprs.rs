mod pntt;

use crate::{ZipError, code::LinearCode, pcs::structs::ZipTypes};
use crypto_primitives::{FromPrimitiveWithConfig, FromWithConfig};
use num_traits::{CheckedAdd, CheckedMul};
use pntt::radix8::params::Config as PnttConfig;
pub use pntt::radix8::params::{PnttConfigF65537, PnttInt, Radix8PnttParams};
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
        Ok(Self {
            pntt_params: Radix8PnttParams::new(row_len, depth, REP)?,
            _phantom: Default::default(),
        })
    }

    /// Create a new IPRS code with the optimal depth heuristics trying to keep
    /// number of columns in the base matrix small.
    /// Currently, keeps number of columns <= 2^7 but this might be tweaked in
    /// the future.
    pub fn new_with_optimal_depth(row_len: usize) -> Result<Self, ZipError> {
        const MAX_BASE_COLS_LOG2: usize = 7;

        let target_base_len = 1 << MAX_BASE_COLS_LOG2;
        // We want depth to be at least 1.
        let depth = 1.max(((1.max(row_len / target_base_len)).ilog2() as usize).div_ceil(3));

        Self::new(row_len, depth)
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
    fn eq(&self, _other: &Self) -> bool {
        true // All IPRS codes with the same config are identical.
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
    use crate::pcs::test_utils::*;
    use crypto_bigint::U64;
    use zinc_utils::CHECKED;

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
}
