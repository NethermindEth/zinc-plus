use super::raa::*;
use crate::{code::LinearCode, pcs::structs::ZipTypes, utils::shuffle_seeded};
use crypto_primitives::{
    BaseFieldConfig, FixedConfig, ProjectPrimitiveIntegersWithConfig, Ring, RingConfig,
};
use std::fmt::Debug;
use zinc_utils::from_ref::FromRef;

/// Implementation of a repeat-accumulate-accumulate (RAA) codes.
/// Flips signs of every second entry in the codeword, starting from the second
/// one.
#[derive(Clone)]
pub struct RaaSignFlippingCode<Zt: ZipTypes, Config: RaaConfig, const REP: usize> {
    raa: RaaCode<Zt, Config, REP>,
}

impl<Zt: ZipTypes, Config: RaaConfig, const REP: usize> RaaSignFlippingCode<Zt, Config, REP>
where
    Zt::Cw: Ring,
{
    pub fn new(row_len: usize) -> Self {
        Self {
            raa: RaaCode::new(row_len),
        }
    }

    /// Do the actual encoding, as per RAA spec
    fn encode_inner<In, C, Map>(&self, cfg: &C, row: &[In], map: Map) -> Vec<C::Element>
    where
        C: RingConfig,
        Map: Fn(&In) -> C::Element + Clone,
    {
        debug_assert_eq!(
            row.len(),
            self.raa.row_len,
            "Row length must match the code's row length"
        );

        let mut result: Vec<C::Element> = repeat(row, REP, map);
        flip_even_signs(cfg, &mut result, Config::CHECK_FOR_OVERFLOWS);
        if Config::PERMUTE_IN_PLACE {
            shuffle_seeded(&mut result, self.raa.perm_1_seed);
        } else {
            result = clone_shuffled(&result, &self.raa.perm_1);
        }
        if Config::CHECK_FOR_OVERFLOWS {
            accumulate(cfg, &mut result);
        } else {
            accumulate_unchecked(cfg, &mut result);
        }
        flip_even_signs(cfg, &mut result, Config::CHECK_FOR_OVERFLOWS);
        if Config::PERMUTE_IN_PLACE {
            shuffle_seeded(&mut result, self.raa.perm_2_seed);
        } else {
            result = clone_shuffled(&result, &self.raa.perm_2);
        }
        if Config::CHECK_FOR_OVERFLOWS {
            accumulate(cfg, &mut result);
        } else {
            accumulate_unchecked(cfg, &mut result);
        }
        debug_assert_eq!(result.len(), self.codeword_len());
        result
    }
}

impl<Zt: ZipTypes, Config: RaaConfig, const REP: usize> LinearCode<Zt>
    for RaaSignFlippingCode<Zt, Config, REP>
where
    Zt::Cw: Ring,
{
    const REPETITION_FACTOR: usize = REP;

    fn row_len(&self) -> usize {
        self.raa.row_len()
    }

    #[allow(clippy::arithmetic_side_effects)]
    fn codeword_len(&self) -> usize {
        self.raa.codeword_len()
    }

    fn params_string(&self) -> String {
        self.raa.params_string()
    }

    fn encode(&self, row: &[Zt::Eval]) -> Vec<Zt::Cw> {
        self.encode_inner(&FixedConfig::default(), row, Zt::Cw::from_ref)
    }

    fn encode_wide(&self, row: &[Zt::CombR]) -> Vec<Zt::CombR> {
        self.encode_inner(&FixedConfig::default(), row, |v| v.clone())
    }

    fn encode_f<C>(&self, cfg: &C, row: &[C::Element]) -> Vec<C::Element>
    where
        C: BaseFieldConfig + ProjectPrimitiveIntegersWithConfig,
    {
        self.encode_inner(cfg, row, |v| v.clone())
    }
}

impl<Zt: ZipTypes, Config: RaaConfig, const REP: usize> Debug
    for RaaSignFlippingCode<Zt, Config, REP>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SignFlipping")
            .field("row_len", &self.raa.row_len)
            .field("perm_1_seed", &self.raa.perm_1_seed)
            .field("perm_2_seed", &self.raa.perm_2_seed)
            .finish()
    }
}

impl<Zt: ZipTypes, Config: RaaConfig, const REP: usize> PartialEq
    for RaaSignFlippingCode<Zt, Config, REP>
{
    fn eq(&self, other: &Self) -> bool {
        self.raa == other.raa
    }
}

impl<Zt: ZipTypes, Config: RaaConfig, const REP: usize> Eq
    for RaaSignFlippingCode<Zt, Config, REP>
{
}

fn flip_even_signs<C: RingConfig>(cfg: &C, result: &mut [C::Element], check_for_overflows: bool) {
    // Flip every other entry in the codeword
    for i in (1..result.len()).step_by(2) {
        if check_for_overflows {
            result[i] = cfg.checked_neg(&result[i]).expect("Negation overflow");
        } else {
            cfg.neg_assign(&mut result[i]);
        }
    }
}
