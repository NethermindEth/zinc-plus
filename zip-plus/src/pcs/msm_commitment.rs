#![allow(clippy::arithmetic_side_effects)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_sign_loss)]

use std::{
    marker::PhantomData,
    sync::{Arc, OnceLock},
};

use ark_ec::{AffineRepr, CurveGroup};
use ark_ff::{AdditiveGroup, One, PrimeField, UniformRand, Zero};
use crypto_primitives::{IntRing, crypto_bigint_int::Int};
use num_integer::Integer;
use num_traits::Zero as NumZero;
use thiserror::Error;
use zinc_utils::{cfg_chunks, cfg_iter};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

const DEFAULT_BOOL_WINDOW_BITS: usize = 6;

#[derive(Clone, Debug)]
pub struct MsmCommitmentKey<C: AffineRepr> {
    pub(crate) num_cols: usize,
    pub(crate) bases: Vec<C>,
    pub(crate) h: C::Group,
    bool_tables_6: Arc<OnceLock<BoolWindowTable<C>>>,
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct MsmVerifierKey<C: AffineRepr> {
    pub(crate) num_cols: usize,
    pub(crate) bases: Vec<C>,
    pub(crate) h: C::Group,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MsmCommitment<C: AffineRepr> {
    pub(crate) comm: Vec<C::Group>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MsmBlind<C: AffineRepr> {
    pub(crate) blind: Vec<C::ScalarField>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MsmCommitmentEngine<C: AffineRepr>(PhantomData<C>);

#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum MsmError {
    #[error("MSM commitment width must be non-zero")]
    InvalidWidth,
    #[error("MSM commitment expected {expected} bases, got {actual}")]
    BaseCountMismatch { expected: usize, actual: usize },
    #[error("MSM commitment expected {expected} blinds, got {actual}")]
    BlindCountMismatch { expected: usize, actual: usize },
    #[error("MSM row length must be at most {max}, got {actual}")]
    RowLengthMismatch { max: usize, actual: usize },
    #[error("MSM commitment expected {expected} row commitments, got {actual}")]
    CommitmentShapeMismatch { expected: usize, actual: usize },
    #[error("MSM window size must be in 1..usize::BITS, got {0}")]
    InvalidWindowBits(usize),
    #[error("cannot commit minimum signed integer value")]
    SignedIntegerMinimum,
}

pub trait RowMsmStrategy<C, V>
where
    C: AffineRepr,
    V: Copy + Send + Sync,
{
    fn precompute_ck(ck: &MsmCommitmentKey<C>);

    fn msm_row(ck: &MsmCommitmentKey<C>, values: &[V]) -> Result<C::Group, MsmError>;

    fn is_zero(value: V) -> bool;

    fn to_scalar(value: V) -> C::ScalarField;
}

pub struct BoolSubsetMsm<const WINDOW_BITS: usize = 6>;
pub struct U8BucketMsm;
pub struct ScalarPippengerMsm;
pub struct SignedIntPippengerMsm;

#[derive(Clone, Debug)]
struct BoolWindowTable<C: AffineRepr> {
    tables: Vec<Vec<C::Group>>,
    lens: Vec<usize>,
}

impl<C: AffineRepr> BoolWindowTable<C> {
    fn new(bases: &[C], window_bits: usize) -> Self {
        let built = cfg_chunks!(bases, window_bits)
            .map(|window| {
                let len = window.len();
                let table_len = 1usize << len;
                let mut table = vec![C::Group::zero(); table_len];
                for mask in 1..table_len {
                    let bit = mask.trailing_zeros() as usize;
                    let previous = mask & !(1usize << bit);
                    table[mask] = table[previous] + window[bit];
                }
                (table, len)
            })
            .collect::<Vec<_>>();

        let (tables, lens) = built.into_iter().unzip();
        Self { tables, lens }
    }

    fn msm_row(
        &self,
        values: &[bool],
        window_bits: usize,
        _use_parallelism_internally: bool,
    ) -> C::Group {
        #[cfg(feature = "parallel")]
        if _use_parallelism_internally && self.lens.len() > 1 {
            return self
                .lens
                .par_iter()
                .copied()
                .enumerate()
                .map(|(window_idx, len)| {
                    let offset = window_idx * window_bits;
                    if offset >= values.len() {
                        return C::Group::zero();
                    }
                    let end = (offset + len).min(values.len());
                    self.tables[window_idx][bit_mask(&values[offset..end])]
                })
                .reduce(C::Group::zero, |acc, point| acc + point);
        }

        let mut acc = C::Group::zero();
        for (window_idx, len) in self.lens.iter().copied().enumerate() {
            let offset = window_idx * window_bits;
            if offset >= values.len() {
                break;
            }
            let end = (offset + len).min(values.len());
            acc += self.tables[window_idx][bit_mask(&values[offset..end])];
        }
        acc
    }

    fn msm_rows_from_window_masks<const LANES: usize, M>(
        &self,
        value_len: usize,
        window_bits: usize,
        _use_parallelism_internally: bool,
        mask_at: M,
    ) -> [C::Group; LANES]
    where
        M: Fn(usize, usize) -> [usize; LANES] + Sync,
    {
        #[cfg(feature = "parallel")]
        if _use_parallelism_internally && self.lens.len() > 1 {
            return self
                .lens
                .par_iter()
                .copied()
                .enumerate()
                .map(|(window_idx, len)| {
                    let mut partial = std::array::from_fn(|_| C::Group::zero());
                    let offset = window_idx * window_bits;
                    if offset >= value_len {
                        return partial;
                    }
                    let end = (offset + len).min(value_len);
                    let masks = mask_at(offset, end - offset);
                    for lane in 0..LANES {
                        partial[lane] += self.tables[window_idx][masks[lane]];
                    }
                    partial
                })
                .reduce(
                    || std::array::from_fn(|_| C::Group::zero()),
                    |mut acc, partial| {
                        for lane in 0..LANES {
                            acc[lane] += partial[lane];
                        }
                        acc
                    },
                );
        }

        let mut acc = std::array::from_fn(|_| C::Group::zero());
        for (window_idx, len) in self.lens.iter().copied().enumerate() {
            let offset = window_idx * window_bits;
            if offset >= value_len {
                break;
            }
            let end = (offset + len).min(value_len);
            let masks = mask_at(offset, end - offset);
            for lane in 0..LANES {
                acc[lane] += self.tables[window_idx][masks[lane]];
            }
        }
        acc
    }
}

impl<C: AffineRepr> MsmCommitmentEngine<C> {
    pub fn setup_from_bases(
        width: usize,
        bases: Vec<C>,
        h: C::Group,
    ) -> Result<(MsmCommitmentKey<C>, MsmVerifierKey<C>), MsmError> {
        if width == 0 {
            return Err(MsmError::InvalidWidth);
        }
        if bases.len() != width {
            return Err(MsmError::BaseCountMismatch {
                expected: width,
                actual: bases.len(),
            });
        }

        let vk = MsmVerifierKey {
            num_cols: width,
            bases: bases.clone(),
            h,
        };
        let ck = MsmCommitmentKey {
            num_cols: width,
            bases,
            h,
            bool_tables_6: Arc::new(OnceLock::new()),
        };

        Ok((ck, vk))
    }

    pub fn precompute_ck(ck: &MsmCommitmentKey<C>) {
        <BoolSubsetMsm<DEFAULT_BOOL_WINDOW_BITS> as RowMsmStrategy<C, bool>>::precompute_ck(ck);
        ScalarPippengerMsm::precompute_ck(ck);
    }

    pub fn blind(ck: &MsmCommitmentKey<C>, n: usize) -> MsmBlind<C> {
        let num_rows = num_rows(n, ck.num_cols).unwrap_or(0);
        let mut rng = ark_std::rand::thread_rng();
        let blind = (0..num_rows)
            .map(|_| C::ScalarField::rand(&mut rng))
            .collect();
        MsmBlind { blind }
    }

    pub fn commit_with<V, S>(
        ck: &MsmCommitmentKey<C>,
        values: &[V],
        blind: &MsmBlind<C>,
    ) -> Result<MsmCommitment<C>, MsmError>
    where
        V: Copy + Send + Sync,
        S: RowMsmStrategy<C, V>,
    {
        let expected_rows = num_rows(values.len(), ck.num_cols)?;
        if blind.blind.len() != expected_rows {
            return Err(MsmError::BlindCountMismatch {
                expected: expected_rows,
                actual: blind.blind.len(),
            });
        }

        S::precompute_ck(ck);
        let comm = cfg_chunks!(values, ck.num_cols)
            .enumerate()
            .map(|(row_idx, row)| {
                let mut row_comm = commit_row::<C, V, S>(ck, row)?;
                row_comm += ck.h * blind.blind[row_idx];
                Ok(row_comm)
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(MsmCommitment { comm })
    }

    pub fn commit_unblinded_with<V, S>(
        ck: &MsmCommitmentKey<C>,
        values: &[V],
    ) -> Result<MsmCommitment<C>, MsmError>
    where
        V: Copy + Send + Sync,
        S: RowMsmStrategy<C, V>,
    {
        let expected_rows = num_rows(values.len(), ck.num_cols)?;
        S::precompute_ck(ck);
        let comm = cfg_chunks!(values, ck.num_cols)
            .map(|row| commit_row::<C, V, S>(ck, row))
            .collect::<Result<Vec<_>, _>>()?;
        debug_assert_eq!(comm.len(), expected_rows);

        Ok(MsmCommitment { comm })
    }

    pub fn commit_unblinded(
        ck: &MsmCommitmentKey<C>,
        values: &[C::ScalarField],
    ) -> Result<MsmCommitment<C>, MsmError> {
        Self::commit_unblinded_with::<C::ScalarField, ScalarPippengerMsm>(ck, values)
    }

    pub fn commit(
        ck: &MsmCommitmentKey<C>,
        values: &[C::ScalarField],
        blind: &MsmBlind<C>,
    ) -> Result<MsmCommitment<C>, MsmError> {
        Self::commit_with::<C::ScalarField, ScalarPippengerMsm>(ck, values, blind)
    }

    pub fn commit_zeros(
        ck: &MsmCommitmentKey<C>,
        n: usize,
        blind: &MsmBlind<C>,
    ) -> Result<MsmCommitment<C>, MsmError> {
        let expected_rows = num_rows(n, ck.num_cols)?;
        if blind.blind.len() != expected_rows {
            return Err(MsmError::BlindCountMismatch {
                expected: expected_rows,
                actual: blind.blind.len(),
            });
        }

        let comm = cfg_iter!(blind.blind).map(|r| ck.h * r).collect();
        Ok(MsmCommitment { comm })
    }

    pub fn check_commitment(
        comm: &MsmCommitment<C>,
        n: usize,
        width: usize,
    ) -> Result<(), MsmError> {
        let expected_rows = num_rows(n, width)?;
        if comm.comm.len() != expected_rows {
            return Err(MsmError::CommitmentShapeMismatch {
                expected: expected_rows,
                actual: comm.comm.len(),
            });
        }
        Ok(())
    }
}

impl<C: AffineRepr, const WINDOW_BITS: usize> RowMsmStrategy<C, bool>
    for BoolSubsetMsm<WINDOW_BITS>
{
    fn precompute_ck(ck: &MsmCommitmentKey<C>) {
        if WINDOW_BITS == DEFAULT_BOOL_WINDOW_BITS {
            ck.bool_tables_6
                .get_or_init(|| BoolWindowTable::new(&ck.bases, DEFAULT_BOOL_WINDOW_BITS));
        }
    }

    fn msm_row(ck: &MsmCommitmentKey<C>, values: &[bool]) -> Result<C::Group, MsmError> {
        Self::msm_bool_row(ck, values, false)
    }

    fn is_zero(value: bool) -> bool {
        !value
    }

    fn to_scalar(value: bool) -> C::ScalarField {
        if value {
            C::ScalarField::one()
        } else {
            C::ScalarField::zero()
        }
    }
}

impl<const WINDOW_BITS: usize> BoolSubsetMsm<WINDOW_BITS> {
    pub(crate) fn msm_bool_row<C: AffineRepr>(
        ck: &MsmCommitmentKey<C>,
        values: &[bool],
        use_parallelism_internally: bool,
    ) -> Result<C::Group, MsmError> {
        validate_row_len(ck, values.len())?;
        validate_window_bits(WINDOW_BITS)?;

        if WINDOW_BITS == DEFAULT_BOOL_WINDOW_BITS {
            return Ok(ck
                .bool_tables_6
                .get_or_init(|| BoolWindowTable::new(&ck.bases, DEFAULT_BOOL_WINDOW_BITS))
                .msm_row(values, DEFAULT_BOOL_WINDOW_BITS, use_parallelism_internally));
        }

        let mut acc = C::Group::zero();
        for (window_idx, bits) in values.chunks(WINDOW_BITS).enumerate() {
            let start = window_idx * WINDOW_BITS;
            let end = start + bits.len();
            let table = subset_table::<C>(&ck.bases[start..end])?;
            acc += table[bit_mask(bits)];
        }
        Ok(acc)
    }

    pub(crate) fn msm_bool_rows_from_window_masks<C, const LANES: usize, M>(
        ck: &MsmCommitmentKey<C>,
        value_len: usize,
        use_parallelism_internally: bool,
        mask_at: M,
    ) -> Result<[C::Group; LANES], MsmError>
    where
        C: AffineRepr,
        M: Fn(usize, usize) -> [usize; LANES] + Sync,
    {
        validate_row_len(ck, value_len)?;
        validate_window_bits(WINDOW_BITS)?;

        if WINDOW_BITS == DEFAULT_BOOL_WINDOW_BITS {
            return Ok(ck
                .bool_tables_6
                .get_or_init(|| BoolWindowTable::new(&ck.bases, DEFAULT_BOOL_WINDOW_BITS))
                .msm_rows_from_window_masks(
                    value_len,
                    DEFAULT_BOOL_WINDOW_BITS,
                    use_parallelism_internally,
                    mask_at,
                ));
        }

        let mut acc = std::array::from_fn(|_| C::Group::zero());
        for (window_idx, window) in ck.bases[..value_len].chunks(WINDOW_BITS).enumerate() {
            let offset = window_idx * WINDOW_BITS;
            let masks = mask_at(offset, window.len());
            let table = subset_table::<C>(window)?;
            for lane in 0..LANES {
                acc[lane] += table[masks[lane]];
            }
        }
        Ok(acc)
    }
}

impl<C: AffineRepr> RowMsmStrategy<C, u8> for U8BucketMsm {
    fn precompute_ck(_ck: &MsmCommitmentKey<C>) {}

    fn msm_row(ck: &MsmCommitmentKey<C>, values: &[u8]) -> Result<C::Group, MsmError> {
        validate_row_len(ck, values.len())?;

        let max_value = values.iter().copied().max().unwrap_or(0);
        if max_value == 0 {
            return Ok(C::Group::zero());
        }

        let mut buckets = vec![C::Group::zero(); usize::from(max_value)];
        for (&value, base) in values.iter().zip(ck.bases.iter()) {
            if value != 0 {
                buckets[usize::from(value) - 1] += base;
            }
        }

        Ok(bucket_running_sum(&buckets))
    }

    fn is_zero(value: u8) -> bool {
        value == 0
    }

    fn to_scalar(value: u8) -> C::ScalarField {
        C::ScalarField::from(u64::from(value))
    }
}

impl<C: AffineRepr> RowMsmStrategy<C, C::ScalarField> for ScalarPippengerMsm {
    fn precompute_ck(_ck: &MsmCommitmentKey<C>) {}

    fn msm_row(ck: &MsmCommitmentKey<C>, values: &[C::ScalarField]) -> Result<C::Group, MsmError> {
        validate_row_len(ck, values.len())?;
        signed_window_pippenger::<C>(values, &ck.bases[..values.len()])
    }

    fn is_zero(value: C::ScalarField) -> bool {
        value.is_zero()
    }

    fn to_scalar(value: C::ScalarField) -> C::ScalarField {
        value
    }
}

impl<C: AffineRepr, const LIMBS: usize> RowMsmStrategy<C, Int<LIMBS>> for SignedIntPippengerMsm {
    fn precompute_ck(_ck: &MsmCommitmentKey<C>) {}

    fn msm_row(ck: &MsmCommitmentKey<C>, values: &[Int<LIMBS>]) -> Result<C::Group, MsmError> {
        validate_row_len(ck, values.len())?;
        signed_int_window_pippenger::<C, LIMBS>(values, &ck.bases[..values.len()])
    }

    fn is_zero(value: Int<LIMBS>) -> bool {
        NumZero::is_zero(&value)
    }

    fn to_scalar(value: Int<LIMBS>) -> C::ScalarField {
        signed_int_to_scalar::<C, LIMBS>(&value)
            .expect("signed integer lane value must fit scalar conversion")
    }
}

fn num_rows(n: usize, width: usize) -> Result<usize, MsmError> {
    if width == 0 {
        return Err(MsmError::InvalidWidth);
    }
    Ok(<usize as Integer>::div_ceil(&n, &width))
}

fn validate_row_len<C: AffineRepr>(
    ck: &MsmCommitmentKey<C>,
    actual: usize,
) -> Result<(), MsmError> {
    if actual > ck.num_cols {
        return Err(MsmError::RowLengthMismatch {
            max: ck.num_cols,
            actual,
        });
    }
    Ok(())
}

fn validate_window_bits(window_bits: usize) -> Result<(), MsmError> {
    if window_bits == 0 || window_bits >= usize::BITS as usize {
        return Err(MsmError::InvalidWindowBits(window_bits));
    }
    Ok(())
}

fn commit_row<C, V, S>(ck: &MsmCommitmentKey<C>, row: &[V]) -> Result<C::Group, MsmError>
where
    C: AffineRepr,
    V: Copy + Send + Sync,
    S: RowMsmStrategy<C, V>,
{
    let effective_len = row
        .iter()
        .rposition(|value| !S::is_zero(*value))
        .map_or(0, |pos| pos + 1);
    if effective_len == 0 {
        Ok(C::Group::zero())
    } else {
        S::msm_row(ck, &row[..effective_len])
    }
}

fn bit_mask(bits: &[bool]) -> usize {
    bits.iter().enumerate().fold(
        0usize,
        |mask, (idx, bit)| {
            if *bit { mask | (1usize << idx) } else { mask }
        },
    )
}

fn subset_table<C: AffineRepr>(bases: &[C]) -> Result<Vec<C::Group>, MsmError> {
    validate_window_bits(bases.len())?;
    let table_len = 1usize << bases.len();
    let mut table = vec![C::Group::zero(); table_len];

    for mask in 1..table_len {
        let bit = mask.trailing_zeros() as usize;
        let previous = mask & !(1usize << bit);
        table[mask] = table[previous] + bases[bit];
    }

    Ok(table)
}

fn bucket_running_sum<G: CurveGroup>(buckets: &[G]) -> G {
    let mut acc = G::zero();
    let mut running_sum = G::zero();
    for bucket in buckets.iter().rev() {
        running_sum += bucket;
        acc += running_sum;
    }
    acc
}

fn signed_window_pippenger<C: AffineRepr>(
    scalars: &[C::ScalarField],
    bases: &[C],
) -> Result<C::Group, MsmError> {
    if scalars.len() != bases.len() {
        return Err(MsmError::BaseCountMismatch {
            expected: scalars.len(),
            actual: bases.len(),
        });
    }
    if scalars.is_empty() {
        return Ok(C::Group::zero());
    }

    let window_bits = scalar_window_bits(scalars.len());
    validate_window_bits(window_bits)?;

    let num_bits = C::ScalarField::MODULUS_BIT_SIZE as usize;
    let segments = <usize as Integer>::div_ceil(&num_bits, &window_bits);
    let bucket_len = (1usize << window_bits) - 1;
    let bigints = scalars
        .iter()
        .map(|scalar| scalar.into_bigint())
        .collect::<Vec<_>>();
    let mut buckets = vec![C::Group::zero(); bucket_len];

    let mut acc = C::Group::zero();
    for segment in (0..segments).rev() {
        for _ in 0..window_bits {
            acc.double_in_place();
        }

        let offset = segment * window_bits;
        for bucket in &mut buckets {
            *bucket = C::Group::zero();
        }
        for (j, scalar) in bigints.iter().enumerate() {
            let digit = window_value_from_limbs(scalar.as_ref(), offset, window_bits);
            if digit != 0 {
                buckets[digit - 1] += bases[j];
            }
        }

        acc += bucket_running_sum(&buckets);
    }

    Ok(acc)
}

fn signed_int_window_pippenger<C: AffineRepr, const LIMBS: usize>(
    values: &[Int<LIMBS>],
    bases: &[C],
) -> Result<C::Group, MsmError> {
    if values.len() != bases.len() {
        return Err(MsmError::BaseCountMismatch {
            expected: values.len(),
            actual: bases.len(),
        });
    }
    if values.is_empty() {
        return Ok(C::Group::zero());
    }

    let mut max_bits = 0usize;
    for value in values {
        let (abs, _) = signed_int_abs(value)?;
        max_bits = max_bits.max(bit_len_from_words(abs.as_uint().as_words()));
    }
    if max_bits == 0 {
        return Ok(C::Group::zero());
    }

    let window_bits = scalar_window_bits(values.len()).min(max_bits).max(1);
    validate_window_bits(window_bits)?;

    let segments = <usize as Integer>::div_ceil(&max_bits, &window_bits);
    let bucket_len = (1usize << window_bits) - 1;
    let mut positive_buckets = vec![C::Group::zero(); bucket_len];
    let mut negative_buckets = vec![C::Group::zero(); bucket_len];

    let mut acc = C::Group::zero();
    for segment in (0..segments).rev() {
        for _ in 0..window_bits {
            acc.double_in_place();
        }

        for bucket in &mut positive_buckets {
            *bucket = C::Group::zero();
        }
        for bucket in &mut negative_buckets {
            *bucket = C::Group::zero();
        }

        let offset = segment * window_bits;
        for (value, base) in values.iter().zip(bases.iter()) {
            let (abs, is_negative) = signed_int_abs(value)?;
            let digit = window_value_from_words(abs.as_uint().as_words(), offset, window_bits);
            if digit != 0 {
                if is_negative {
                    negative_buckets[digit - 1] += base;
                } else {
                    positive_buckets[digit - 1] += base;
                }
            }
        }

        acc += bucket_running_sum(&positive_buckets);
        acc -= bucket_running_sum(&negative_buckets);
    }

    Ok(acc)
}

fn signed_int_abs<const LIMBS: usize>(value: &Int<LIMBS>) -> Result<(Int<LIMBS>, bool), MsmError> {
    if value.is_negative() {
        let abs = value.checked_abs().ok_or(MsmError::SignedIntegerMinimum)?;
        Ok((abs, true))
    } else {
        Ok((*value, false))
    }
}

fn signed_int_to_scalar<C: AffineRepr, const LIMBS: usize>(
    value: &Int<LIMBS>,
) -> Result<C::ScalarField, MsmError> {
    let (abs, is_negative) = signed_int_abs(value)?;
    let mut bytes = Vec::with_capacity(LIMBS * core::mem::size_of::<crypto_bigint::Word>());
    for word in abs.as_uint().as_words() {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    let mut scalar = C::ScalarField::from_le_bytes_mod_order(&bytes);
    if is_negative && !scalar.is_zero() {
        scalar = -scalar;
    }
    Ok(scalar)
}

fn scalar_window_bits(n: usize) -> usize {
    if n < 4 {
        1
    } else if n < 32 {
        3
    } else {
        (usize::BITS - n.leading_zeros()) as usize
    }
}

fn bit_len_from_words(words: &[crypto_bigint::Word]) -> usize {
    let word_bits = core::mem::size_of::<crypto_bigint::Word>() * 8;
    for (idx, word) in words.iter().copied().enumerate().rev() {
        if word != 0 {
            return idx * word_bits + word_bits - word.leading_zeros() as usize;
        }
    }
    0
}

fn window_value_from_limbs(limbs: &[u64], start: usize, width: usize) -> usize {
    (0..width).fold(0usize, |value, bit_idx| {
        let absolute_bit = start + bit_idx;
        let limb_idx = absolute_bit / u64::BITS as usize;
        let limb_bit = absolute_bit % u64::BITS as usize;
        if limbs
            .get(limb_idx)
            .map(|limb| ((limb >> limb_bit) & 1) == 1)
            .unwrap_or(false)
        {
            value | (1usize << bit_idx)
        } else {
            value
        }
    })
}

fn window_value_from_words(words: &[crypto_bigint::Word], start: usize, width: usize) -> usize {
    let word_bits = core::mem::size_of::<crypto_bigint::Word>() * 8;
    (0..width).fold(0usize, |value, bit_idx| {
        let absolute_bit = start + bit_idx;
        let word_idx = absolute_bit / word_bits;
        let word_bit = absolute_bit % word_bits;
        if words
            .get(word_idx)
            .map(|word| ((word >> word_bit) & 1) == 1)
            .unwrap_or(false)
        {
            value | (1usize << bit_idx)
        } else {
            value
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use ark_bn254::{Fr, G1Affine, G1Projective};
    use ark_ec::PrimeGroup;
    use ark_ff::UniformRand;

    type TestCurve = G1Affine;

    fn fr(value: usize) -> Fr {
        Fr::from(u64::try_from(value).expect("test value must fit into u64"))
    }

    fn setup(width: usize) -> (MsmCommitmentKey<TestCurve>, MsmVerifierKey<TestCurve>) {
        let generator = G1Projective::generator();
        let bases = (1..=width)
            .map(|idx| (generator * fr(idx)).into_affine())
            .collect();
        let h = generator * fr(width + 1);
        MsmCommitmentEngine::<TestCurve>::setup_from_bases(width, bases, h)
            .expect("valid test setup")
    }

    fn blind(width: usize, n: usize) -> MsmBlind<TestCurve> {
        let rows = <usize as Integer>::div_ceil(&n, &width);
        MsmBlind {
            blind: (0..rows).map(|idx| fr(idx + 11)).collect(),
        }
    }

    fn bool_values(n: usize) -> Vec<bool> {
        (0..n).map(|idx| idx % 3 == 0 || idx % 7 == 1).collect()
    }

    fn u8_values(n: usize, modulus: u8) -> Vec<u8> {
        (0..n)
            .map(|idx| {
                let value = (idx * 17 + 5) % usize::from(modulus);
                u8::try_from(value).expect("test u8 value must fit")
            })
            .collect()
    }

    fn scalars_from_bool(values: &[bool]) -> Vec<Fr> {
        values
            .iter()
            .map(|value| if *value { Fr::one() } else { Fr::zero() })
            .collect()
    }

    fn scalars_from_u8(values: &[u8]) -> Vec<Fr> {
        values
            .iter()
            .map(|value| Fr::from(u64::from(*value)))
            .collect()
    }

    fn scalars_from_int<const LIMBS: usize>(values: &[Int<LIMBS>]) -> Vec<Fr> {
        values
            .iter()
            .map(|value| signed_int_to_scalar::<TestCurve, LIMBS>(value).expect("valid test int"))
            .collect()
    }

    fn naive_scalar_commit(
        ck: &MsmCommitmentKey<TestCurve>,
        values: &[Fr],
        blind: &MsmBlind<TestCurve>,
    ) -> MsmCommitment<TestCurve> {
        let comm = values
            .chunks(ck.num_cols)
            .enumerate()
            .map(|(row_idx, row)| {
                let mut acc = G1Projective::zero();
                for (scalar, base) in row.iter().zip(ck.bases.iter()) {
                    acc += *base * scalar;
                }
                acc += ck.h * blind.blind[row_idx];
                acc
            })
            .collect();
        MsmCommitment { comm }
    }

    fn naive_scalar_commit_unblinded(
        ck: &MsmCommitmentKey<TestCurve>,
        values: &[Fr],
    ) -> MsmCommitment<TestCurve> {
        let comm = values
            .chunks(ck.num_cols)
            .map(|row| {
                let mut acc = G1Projective::zero();
                for (scalar, base) in row.iter().zip(ck.bases.iter()) {
                    acc += *base * scalar;
                }
                acc
            })
            .collect();
        MsmCommitment { comm }
    }

    #[test]
    fn bool_commit_matches_scalar_commit_for_configured_widths() {
        for width in [8, 32, 64] {
            let (ck, _) = setup(width);
            let n = width * 3 + 5;
            let values = bool_values(n);
            let scalars = scalars_from_bool(&values);
            let blind = blind(width, n);

            let bool_comm =
                MsmCommitmentEngine::<TestCurve>::commit_with::<bool, BoolSubsetMsm<6>>(
                    &ck, &values, &blind,
                )
                .expect("bool commit must succeed");
            let scalar_comm = MsmCommitmentEngine::<TestCurve>::commit(&ck, &scalars, &blind)
                .expect("scalar commit must succeed");

            assert_eq!(bool_comm, scalar_comm);
        }
    }

    #[test]
    fn precomputed_bool_commit_matches_scalar_commit_for_wide_rows() {
        for width in [8, 32, 64, 512] {
            let (ck, _) = setup(width);
            let n = width + 5;
            let values = bool_values(n);
            let scalars = scalars_from_bool(&values);
            let blind = blind(width, n);

            let before_precompute = MsmCommitmentEngine::<TestCurve>::commit_with::<
                bool,
                BoolSubsetMsm<6>,
            >(&ck, &values, &blind)
            .expect("bool commit before precompute must succeed");
            MsmCommitmentEngine::<TestCurve>::precompute_ck(&ck);
            let after_precompute = MsmCommitmentEngine::<TestCurve>::commit_with::<
                bool,
                BoolSubsetMsm<6>,
            >(&ck, &values, &blind)
            .expect("bool commit after precompute must succeed");
            let cloned_ck = ck.clone();
            let after_clone =
                MsmCommitmentEngine::<TestCurve>::commit_with::<bool, BoolSubsetMsm<6>>(
                    &cloned_ck, &values, &blind,
                )
                .expect("bool commit through cloned ck must succeed");
            let scalar_comm = MsmCommitmentEngine::<TestCurve>::commit(&ck, &scalars, &blind)
                .expect("scalar commit must succeed");

            assert_eq!(before_precompute, scalar_comm);
            assert_eq!(after_precompute, scalar_comm);
            assert_eq!(after_clone, scalar_comm);
        }
    }

    #[test]
    fn unblinded_bool_commit_matches_scalar_commit_for_wide_rows() {
        for width in [8, 32, 64, 512] {
            let (ck, _) = setup(width);
            let n = width + 7;
            let values = bool_values(n);
            let scalars = scalars_from_bool(&values);

            MsmCommitmentEngine::<TestCurve>::precompute_ck(&ck);
            let bool_comm = MsmCommitmentEngine::<TestCurve>::commit_unblinded_with::<
                bool,
                BoolSubsetMsm<6>,
            >(&ck, &values)
            .expect("unblinded bool commit must succeed");
            let scalar_comm = MsmCommitmentEngine::<TestCurve>::commit_unblinded(&ck, &scalars)
                .expect("unblinded scalar commit must succeed");
            let naive_comm = naive_scalar_commit_unblinded(&ck, &scalars);

            assert_eq!(bool_comm, scalar_comm);
            assert_eq!(bool_comm, naive_comm);
        }
    }

    #[test]
    fn u8_commit_matches_scalar_commit_for_configured_widths() {
        for width in [8, 32, 64] {
            let (ck, _) = setup(width);
            let n = width * 2 + width / 2 + 1;
            let cases = [vec![0; n], vec![1; n], u8_values(n, 32), u8_values(n, 255)];

            for values in cases {
                let scalars = scalars_from_u8(&values);
                let blind = blind(width, n);

                let u8_comm = MsmCommitmentEngine::<TestCurve>::commit_with::<u8, U8BucketMsm>(
                    &ck, &values, &blind,
                )
                .expect("u8 commit must succeed");
                let scalar_comm = MsmCommitmentEngine::<TestCurve>::commit(&ck, &scalars, &blind)
                    .expect("scalar commit must succeed");

                assert_eq!(u8_comm, scalar_comm);
            }
        }
    }

    #[test]
    fn scalar_commit_matches_naive_full_field_commit_for_configured_widths() {
        let mut rng = ark_std::test_rng();
        for width in [8, 32, 64] {
            let (ck, _) = setup(width);
            let n = width * 2 + 3;
            let values = (0..n).map(|_| Fr::rand(&mut rng)).collect::<Vec<_>>();
            let blind = blind(width, n);

            let scalar_comm = MsmCommitmentEngine::<TestCurve>::commit(&ck, &values, &blind)
                .expect("scalar commit must succeed");
            let naive_comm = naive_scalar_commit(&ck, &values, &blind);

            assert_eq!(scalar_comm, naive_comm);
        }
    }

    #[test]
    fn signed_int_commit_matches_scalar_commit_for_small_values() {
        for width in [8, 32, 64] {
            let (ck, _) = setup(width);
            let n = width * 2 + 5;
            let values = (0..n)
                .map(|idx| Int::<1>::from((idx as i64 % 31) - 15))
                .collect::<Vec<_>>();
            let scalars = scalars_from_int(&values);
            let blind = blind(width, n);

            let int_comm = MsmCommitmentEngine::<TestCurve>::commit_with::<
                Int<1>,
                SignedIntPippengerMsm,
            >(&ck, &values, &blind)
            .expect("signed int commit must succeed");
            let scalar_comm = MsmCommitmentEngine::<TestCurve>::commit(&ck, &scalars, &blind)
                .expect("scalar commit must succeed");

            assert_eq!(int_comm, scalar_comm);
        }
    }

    #[test]
    fn commit_zeros_matches_strategy_zero_paths() {
        let width = 32;
        let n = width * 2 + 9;
        let (ck, _) = setup(width);
        let blind = blind(width, n);
        let zeros_bool = vec![false; n];
        let zeros_u8 = vec![0u8; n];
        let zeros_scalar = vec![Fr::zero(); n];

        let zero_comm = MsmCommitmentEngine::<TestCurve>::commit_zeros(&ck, n, &blind)
            .expect("zero commit must succeed");
        let bool_comm = MsmCommitmentEngine::<TestCurve>::commit_with::<bool, BoolSubsetMsm<6>>(
            &ck,
            &zeros_bool,
            &blind,
        )
        .expect("bool zero commit must succeed");
        let u8_comm = MsmCommitmentEngine::<TestCurve>::commit_with::<u8, U8BucketMsm>(
            &ck, &zeros_u8, &blind,
        )
        .expect("u8 zero commit must succeed");
        let scalar_comm = MsmCommitmentEngine::<TestCurve>::commit(&ck, &zeros_scalar, &blind)
            .expect("scalar zero commit must succeed");

        assert_eq!(zero_comm, bool_comm);
        assert_eq!(zero_comm, u8_comm);
        assert_eq!(zero_comm, scalar_comm);
    }

    #[test]
    fn changing_one_blind_changes_only_that_row_by_delta_h() {
        let width = 16;
        let n = width * 3;
        let (ck, _) = setup(width);
        let values = u8_values(n, 32);
        let mut blind_a = blind(width, n);
        let mut blind_b = blind_a.clone();
        let delta = fr(5);
        blind_b.blind[1] += delta;

        let comm_a = MsmCommitmentEngine::<TestCurve>::commit_with::<u8, U8BucketMsm>(
            &ck, &values, &blind_a,
        )
        .expect("first commit must succeed");
        let comm_b = MsmCommitmentEngine::<TestCurve>::commit_with::<u8, U8BucketMsm>(
            &ck, &values, &blind_b,
        )
        .expect("second commit must succeed");

        for row_idx in 0..comm_a.comm.len() {
            let actual_delta = comm_b.comm[row_idx] - comm_a.comm[row_idx];
            let expected_delta = if row_idx == 1 {
                ck.h * delta
            } else {
                G1Projective::zero()
            };
            assert_eq!(actual_delta, expected_delta);
        }

        blind_a.blind[1] += delta;
        assert_eq!(blind_a, blind_b);
    }

    #[test]
    fn rejects_invalid_shapes() {
        let width = 8;
        let (ck, _) = setup(width);
        let n = 17;
        let values = vec![Fr::one(); n];
        let blind = blind(width, n);

        assert!(matches!(
            MsmCommitmentEngine::<TestCurve>::setup_from_bases(0, Vec::new(), G1Projective::zero()),
            Err(MsmError::InvalidWidth)
        ));
        assert!(matches!(
            MsmCommitmentEngine::<TestCurve>::setup_from_bases(
                width,
                vec![G1Affine::generator(); width - 1],
                G1Projective::generator(),
            ),
            Err(MsmError::BaseCountMismatch { .. })
        ));

        let short_blind = MsmBlind {
            blind: blind.blind[..1].to_vec(),
        };
        assert!(matches!(
            MsmCommitmentEngine::<TestCurve>::commit(&ck, &values, &short_blind),
            Err(MsmError::BlindCountMismatch { .. })
        ));

        let comm = MsmCommitment { comm: Vec::new() };
        assert!(matches!(
            MsmCommitmentEngine::<TestCurve>::check_commitment(&comm, n, width),
            Err(MsmError::CommitmentShapeMismatch { .. })
        ));
        assert!(matches!(
            MsmCommitmentEngine::<TestCurve>::check_commitment(&comm, n, 0),
            Err(MsmError::InvalidWidth)
        ));
    }
}
