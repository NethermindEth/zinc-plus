use super::raa::*;
use crate::{code::LinearCode, pcs::structs::ZipTypes, utils::shuffle_seeded};
use crypto_primitives::{PrimeField, Ring};
use num_traits::{CheckedAdd, CheckedNeg};
use std::ops::{AddAssign, Neg};
use zinc_utils::{from_ref::FromRef, neg};

/// Implementation of a repeat-accumulate-accumulate (RAA) codes.
/// Flips signs of every second entry in the codeword, starting from the second
/// one.
#[derive(Debug, Clone)]
pub struct RaaSignFlippingCode<Zt: ZipTypes, const REP: usize>
where
    Zt::Cw: Ring,
{
    raa: RaaCode<Zt, REP>,
}

impl<Zt: ZipTypes, const REP: usize> RaaSignFlippingCode<Zt, REP>
where
    Zt::Cw: Ring,
{
    /// Do the actual encoding, as per RAA spec
    fn encode_inner<In, Out>(&self, row: &[In]) -> Vec<Out>
    where
        Out: Neg<Output = Out>
            + CheckedNeg
            + CheckedAdd
            + for<'a> AddAssign<&'a Out>
            + FromRef<In>
            + Clone,
    {
        debug_assert_eq!(
            row.len(),
            self.raa.row_len,
            "Row length must match the code's row length"
        );

        let mut result: Vec<Out> = repeat(row, REP);
        flip_even_signs(&mut result, self.raa.cfg.check_for_overflows);
        if self.raa.cfg.permute_in_place {
            shuffle_seeded(&mut result, self.raa.perm_1_seed);
        } else {
            result = clone_shuffled(&result, &self.raa.perm_1);
        }
        if self.raa.cfg.check_for_overflows {
            accumulate(&mut result);
        } else {
            accumulate_unchecked(&mut result);
        }
        flip_even_signs(&mut result, self.raa.cfg.check_for_overflows);
        if self.raa.cfg.permute_in_place {
            shuffle_seeded(&mut result, self.raa.perm_2_seed);
        } else {
            result = clone_shuffled(&result, &self.raa.perm_2);
        }
        if self.raa.cfg.check_for_overflows {
            accumulate(&mut result);
        } else {
            accumulate_unchecked(&mut result);
        }
        debug_assert_eq!(result.len(), self.codeword_len());
        result
    }
}

impl<Zt: ZipTypes, const REP: usize> LinearCode<Zt> for RaaSignFlippingCode<Zt, REP>
where
    Zt::Cw: Ring,
{
    type Config = <RaaCode<Zt, REP> as LinearCode<Zt>>::Config;

    const REPETITION_FACTOR: usize = REP;

    fn new(poly_size: usize, cfg: Self::Config) -> Self {
        Self {
            raa: RaaCode::new(poly_size, cfg),
        }
    }

    fn row_len(&self) -> usize {
        self.raa.row_len()
    }

    #[allow(clippy::arithmetic_side_effects)]
    fn codeword_len(&self) -> usize {
        self.raa.codeword_len()
    }

    fn encode(&self, row: &[Zt::Eval]) -> Vec<Zt::Cw> {
        self.encode_inner(row)
    }

    fn encode_wide(&self, row: &[Zt::CombR]) -> Vec<Zt::CombR> {
        self.encode_inner(row)
    }

    fn encode_f<F>(&self, row: &[F]) -> Vec<F>
    where
        F: PrimeField + FromRef<F>,
    {
        self.encode_inner(row)
    }
}

/// Flip every other entry in the codeword, starting from the second one.
fn flip_even_signs<Out>(result: &mut [Out], check_for_overflows: bool)
where
    Out: Neg<Output = Out> + CheckedNeg + Clone,
{
    if check_for_overflows {
        flip_even_signs_checked(result);
    } else {
        flip_even_signs_unchecked(result);
    }
}

fn flip_even_signs_checked<Out>(result: &mut [Out])
where
    Out: CheckedNeg + Clone,
{
    // Flip every other entry in the codeword
    for i in (1..result.len()).step_by(2) {
        result[i] = neg!(result[i]);
    }
}

/// Flip every other entry in the codeword, starting from the second one.
fn flip_even_signs_unchecked<Out>(result: &mut [Out])
where
    Out: Neg<Output = Out> + Clone,
{
    // Flip every other entry in the codeword
    for i in (1..result.len()).step_by(2) {
        result[i] = result[i].clone().neg();
    }
}
