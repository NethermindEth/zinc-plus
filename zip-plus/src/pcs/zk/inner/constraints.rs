//! The `R_lift`-to-bits compiler (paper §5, Definition 5.1 + Lemma 5.3).
//!
//! The witness is **bits only**: each seed entry, lift `X` and quotient `Y`
//! is represented by its binary decomposition, and the lift/aggregation
//! equations are emitted directly over the bits — so the only non-linear
//! condition is booleanity, left to the IOP's quadratic test. Ranges are
//! power-of-two windows by the protocol's window convention (see
//! [`super::super::mask::MaskSeeds::sample`] and the resampling loop in
//! `ZkZip::commit_single`), so no modulus comparison is ever emitted.

use crate::{
    ZipError,
    pcs::zk::{inner::field::InnerCfg, mask},
};
use crypto_primitives::{
    PrimeField, crypto_bigint_int::Int, crypto_bigint_monty::MontyField, crypto_bigint_uint::Uint,
};
use zinc_utils::{add, mul, sub};

/// Shape of the bit-witness for one outer opening.
///
/// `rows` counts committed rows *including* the blinding row (row 0);
/// `num_columns` is the number of opened columns `|Q|` (with multiplicity,
/// in squeeze order); `seed_bits = bits(p) - 1` is the power-of-two window
/// of seed entries and lifts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LiftLayout {
    pub rows: usize,
    pub mask_dim: usize,
    pub num_columns: usize,
    pub seed_bits: u32,
    pub y_bits: u32,
}

impl LiftLayout {
    #[allow(clippy::arithmetic_side_effects)] // bit-width arithmetic, bounded
    pub fn new(
        rows: usize,
        mask_dim: usize,
        num_columns: usize,
        mask_modulus_bits: u32,
    ) -> Self {
        let seed_bits = sub!(mask_modulus_bits, 1u32, "modulus bits underflow");
        // <G_l, s_i> < D * p^2  =>  Y = quotient < D * p.
        let dim_bits = usize::BITS - mask_dim.leading_zeros();
        let y_bits = add!(add!(mask_modulus_bits, dim_bits), 1u32);
        Self {
            rows,
            mask_dim,
            num_columns,
            seed_bits,
            y_bits,
        }
    }

    fn seed_bits_usize(&self) -> usize {
        usize::try_from(self.seed_bits).expect("seed_bits fits usize")
    }

    fn y_bits_usize(&self) -> usize {
        usize::try_from(self.y_bits).expect("y_bits fits usize")
    }

    fn seed_block_len(&self) -> usize {
        mul!(mul!(self.rows, self.mask_dim), self.seed_bits_usize())
    }

    fn x_block_len(&self) -> usize {
        mul!(mul!(self.rows, self.num_columns), self.seed_bits_usize())
    }

    fn y_block_len(&self) -> usize {
        mul!(mul!(self.rows, self.num_columns), self.y_bits_usize())
    }

    /// Total number of witness bits.
    pub fn witness_len(&self) -> usize {
        add!(
            add!(self.seed_block_len(), self.x_block_len()),
            self.y_block_len()
        )
    }

    /// Index of bit `k` of seed entry `d` of row `i`.
    #[allow(clippy::arithmetic_side_effects)] // index arithmetic bounded by layout
    pub fn seed_bit(&self, row: usize, entry: usize, k: usize) -> usize {
        debug_assert!(row < self.rows && entry < self.mask_dim && k < self.seed_bits_usize());
        (row * self.mask_dim + entry) * self.seed_bits_usize() + k
    }

    /// Index of bit `k` of the lift `X_{row, column_slot}`.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn x_bit(&self, row: usize, column_slot: usize, k: usize) -> usize {
        debug_assert!(
            row < self.rows && column_slot < self.num_columns && k < self.seed_bits_usize()
        );
        self.seed_block_len() + (row * self.num_columns + column_slot) * self.seed_bits_usize() + k
    }

    /// Index of bit `k` of the quotient `Y_{row, column_slot}`.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn y_bit(&self, row: usize, column_slot: usize, k: usize) -> usize {
        debug_assert!(
            row < self.rows && column_slot < self.num_columns && k < self.y_bits_usize()
        );
        self.seed_block_len()
            + self.x_block_len()
            + (row * self.num_columns + column_slot) * self.y_bits_usize()
            + k
    }

    /// Minimal bit length of the inner modulus `p'` so that no constraint
    /// wraps (Lemma 5.2): the lift side is bounded by `D * p^2`, the
    /// aggregation side by `(1 + sum |rho|) * p`.
    #[allow(clippy::arithmetic_side_effects)] // bit-width arithmetic, bounded
    pub fn required_inner_bits(&self, chal_bits: u32) -> u32 {
        let p_bits = add!(self.seed_bits, 1u32);
        let dim_bits = usize::BITS - self.mask_dim.leading_zeros();
        let rows_bits = usize::BITS - self.rows.leading_zeros();
        let lift_side = add!(add!(mul!(2u32, p_bits), dim_bits), 2u32);
        let agg_side = add!(add!(add!(p_bits, chal_bits), rows_bits), 2u32);
        add!(lift_side.max(agg_side), 2u32)
    }
}

/// One sparse linear constraint `sum_j coeff_j * bit_j = rhs` over `F_p'`.
#[derive(Clone, Debug)]
pub struct SparseConstraint<const IL: usize> {
    pub terms: Vec<(usize, MontyField<IL>)>,
    pub rhs: MontyField<IL>,
}

impl<const IL: usize> SparseConstraint<IL> {
    /// Evaluates `sum coeff_j * bit_j - rhs` (zero iff satisfied).
    #[allow(clippy::arithmetic_side_effects)] // field ops are modular (infallible)
    pub fn residual(&self, bits: &[bool], cfg: &InnerCfg<IL>) -> MontyField<IL> {
        let mut acc = MontyField::<IL>::zero_with_cfg(cfg);
        for (index, coeff) in &self.terms {
            if bits[*index] {
                acc += coeff.clone();
            }
        }
        acc - self.rhs.clone()
    }

    pub fn is_satisfied(&self, bits: &[bool], cfg: &InnerCfg<IL>) -> bool {
        MontyField::is_zero(&self.residual(bits, cfg))
    }
}

/// Injects a non-negative `Uint<WL>` into `F_p'`.
fn uint_to_field<const WL: usize, const IL: usize>(
    value: &Uint<WL>,
    cfg: &InnerCfg<IL>,
) -> MontyField<IL> {
    MontyField::<IL>::new_with_cfg(value.resize::<IL>(), cfg)
}

/// Injects a signed `Int<WL>` into `F_p'`.
#[allow(clippy::arithmetic_side_effects)] // field ops are modular (infallible)
fn int_to_field<const WL: usize, const IL: usize>(
    value: &Int<WL>,
    cfg: &InnerCfg<IL>,
) -> MontyField<IL> {
    let abs = Uint::<WL>::new(value.inner().abs());
    let abs_f = uint_to_field::<WL, IL>(&abs, cfg);
    if value.inner().is_negative().into() {
        MontyField::<IL>::zero_with_cfg(cfg) - abs_f
    } else {
        abs_f
    }
}

/// Powers of two `2^0 .. 2^(count-1)` in `F_p'`.
#[allow(clippy::arithmetic_side_effects)] // field ops are modular (infallible)
fn powers_of_two<const IL: usize>(count: usize, cfg: &InnerCfg<IL>) -> Vec<MontyField<IL>> {
    let mut powers = Vec::with_capacity(count);
    let mut current = MontyField::<IL>::one_with_cfg(cfg);
    for _ in 0..count {
        powers.push(current.clone());
        current = current.clone() + current;
    }
    powers
}

/// The integer Vandermonde row `G_l = ([(l+1)^d]_p)_{d < D}` as `Uint<WL>`
/// canonical representatives.
#[allow(clippy::arithmetic_side_effects)] // field ops are modular (infallible)
fn vandermonde_row<const WL: usize>(
    column: usize,
    mask_dim: usize,
    mask_cfg: &<MontyField<WL> as PrimeField>::Config,
) -> Vec<Uint<WL>> {
    let column_u64 = u64::try_from(column).expect("column fits u64");
    let alpha =
        MontyField::<WL>::new_with_cfg(Uint::from_u64(add!(column_u64, 1u64)), mask_cfg);
    let mut powers = Vec::with_capacity(mask_dim);
    let mut current = MontyField::<WL>::one_with_cfg(mask_cfg);
    for _ in 0..mask_dim {
        powers.push(current.retrieve());
        current *= alpha.clone();
    }
    powers
}

/// Emits the full constraint system for one opening: `rows * num_columns`
/// lift constraints followed by `num_columns` aggregation constraints.
///
/// `rho` are the witness-row combination weights (length `rows - 1`; the
/// blinding row has weight 1), `rem` the verifier-computed remainders.
#[allow(clippy::arithmetic_side_effects)] // index arithmetic bounded by layout
pub fn build_constraints<const WL: usize, const IL: usize>(
    layout: &LiftLayout,
    columns: &[usize],
    rho: &[Int<WL>],
    rem: &[Int<WL>],
    mask_modulus: &Uint<WL>,
    inner_cfg: &InnerCfg<IL>,
) -> Result<Vec<SparseConstraint<IL>>, ZipError> {
    if columns.len() != layout.num_columns
        || rem.len() != layout.num_columns
        || rho.len() != sub!(layout.rows, 1usize)
    {
        return Err(ZipError::InvalidPcsParam(
            "lift constraint inputs do not match the layout".into(),
        ));
    }
    let mask_cfg = mask::mask_field_cfg(mask_modulus)?;

    let seed_bits = layout.seed_bits_usize();
    let y_bits = layout.y_bits_usize();
    let pow2_f = powers_of_two::<IL>(seed_bits.max(y_bits), inner_cfg);
    let zero = MontyField::<IL>::zero_with_cfg(inner_cfg);
    let p_f = uint_to_field::<WL, IL>(mask_modulus, inner_cfg);

    let mut constraints =
        Vec::with_capacity(add!(mul!(layout.rows, layout.num_columns), layout.num_columns));

    // Lift constraints: <G_l, s_i> - X_{i,l} - p * Y_{i,l} = 0.
    for (column_slot, column) in columns.iter().enumerate() {
        let g_row: Vec<MontyField<IL>> =
            vandermonde_row::<WL>(*column, layout.mask_dim, &mask_cfg)
                .iter()
                .map(|g| uint_to_field::<WL, IL>(g, inner_cfg))
                .collect();
        for row in 0..layout.rows {
            let mut terms =
                Vec::with_capacity(mul!(layout.mask_dim, seed_bits) + seed_bits + y_bits);
            for (entry, g) in g_row.iter().enumerate() {
                for (k, p2) in pow2_f.iter().take(seed_bits).enumerate() {
                    terms.push((layout.seed_bit(row, entry, k), g.clone() * p2.clone()));
                }
            }
            for (k, p2) in pow2_f.iter().take(seed_bits).enumerate() {
                terms.push((layout.x_bit(row, column_slot, k), zero.clone() - p2.clone()));
            }
            for (k, p2) in pow2_f.iter().take(y_bits).enumerate() {
                terms.push((
                    layout.y_bit(row, column_slot, k),
                    zero.clone() - p_f.clone() * p2.clone(),
                ));
            }
            constraints.push(SparseConstraint {
                terms,
                rhs: zero.clone(),
            });
        }
    }

    // Aggregation constraints: X_{0,l} + sum_j rho_j X_{j,l} = rem_l.
    for (column_slot, rem_value) in rem.iter().enumerate() {
        let mut terms = Vec::with_capacity(mul!(layout.rows, seed_bits));
        for row in 0..layout.rows {
            let weight = if row == 0 {
                MontyField::<IL>::one_with_cfg(inner_cfg)
            } else {
                int_to_field::<WL, IL>(&rho[row - 1], inner_cfg)
            };
            for (k, p2) in pow2_f.iter().take(seed_bits).enumerate() {
                terms.push((
                    layout.x_bit(row, column_slot, k),
                    weight.clone() * p2.clone(),
                ));
            }
        }
        constraints.push(SparseConstraint {
            terms,
            rhs: int_to_field::<WL, IL>(rem_value, inner_cfg),
        });
    }

    Ok(constraints)
}

/// Builds the honest bit-witness from the mask seeds: recomputes every lift
/// exactly over `Z` (in `Uint<IL>`, which the caller must size via
/// [`LiftLayout::required_inner_bits`]) and decomposes seeds, lifts and
/// quotients into bits.
///
/// Fails if a lift escapes the power-of-two window — the commit-time
/// resampling loop guarantees this never happens for honestly generated
/// seeds.
#[allow(clippy::arithmetic_side_effects)] // index/bit arithmetic bounded by layout
pub fn build_witness<const WL: usize, const IL: usize>(
    layout: &LiftLayout,
    seeds: &[Vec<Uint<WL>>],
    columns: &[usize],
    mask_modulus: &Uint<WL>,
) -> Result<Vec<bool>, ZipError> {
    if seeds.len() != layout.rows
        || seeds.iter().any(|seed| seed.len() != layout.mask_dim)
        || columns.len() != layout.num_columns
    {
        return Err(ZipError::InvalidPcsParam(
            "lift witness inputs do not match the layout".into(),
        ));
    }
    let mask_cfg = mask::mask_field_cfg(mask_modulus)?;
    let seed_bits = layout.seed_bits_usize();
    let y_bits = layout.y_bits_usize();
    let p_wide = mask_modulus.resize::<IL>();
    let p_nonzero = Option::<crypto_bigint::NonZero<_>>::from(crypto_bigint::NonZero::new(
        p_wide.into_inner(),
    ))
    .ok_or_else(|| ZipError::InvalidPcsParam("mask modulus is zero".into()))?;

    let mut bits = vec![false; layout.witness_len()];

    // Seed bits.
    for (row, seed) in seeds.iter().enumerate() {
        for (entry, value) in seed.iter().enumerate() {
            if value.inner().bits() > layout.seed_bits {
                return Err(ZipError::InvalidPcsParam(
                    "seed entry escapes the power-of-two window".into(),
                ));
            }
            for k in 0..seed_bits {
                bits[layout.seed_bit(row, entry, k)] = uint_bit(&value.resize::<IL>(), k);
            }
        }
    }

    // Lifts and quotients: t = <G_l, s_i> over Z, X = t mod p, Y = t / p.
    for (column_slot, column) in columns.iter().enumerate() {
        let g_row = vandermonde_row::<WL>(*column, layout.mask_dim, &mask_cfg);
        for (row, seed) in seeds.iter().enumerate() {
            let mut t = Uint::<IL>::from_u64(0);
            for (g, s) in g_row.iter().zip(seed) {
                let product = mul!(
                    g.resize::<IL>(),
                    &s.resize::<IL>(),
                    "lift inner product overflow"
                );
                t = add!(t, &product, "lift inner product overflow");
            }
            let (quotient, remainder) = t.into_inner().div_rem(&p_nonzero);
            let x = Uint::<IL>::new(remainder);
            let y = Uint::<IL>::new(quotient);
            if x.inner().bits() > layout.seed_bits {
                return Err(ZipError::InvalidPcsOpen(
                    "mask lift escapes the power-of-two window; resample the seed".into(),
                ));
            }
            if y.inner().bits() > layout.y_bits {
                return Err(ZipError::InvalidPcsOpen(
                    "mask quotient exceeds its range".into(),
                ));
            }
            for k in 0..seed_bits {
                bits[layout.x_bit(row, column_slot, k)] = uint_bit(&x, k);
            }
            for k in 0..y_bits {
                bits[layout.y_bit(row, column_slot, k)] = uint_bit(&y, k);
            }
        }
    }

    Ok(bits)
}

#[allow(clippy::arithmetic_side_effects)] // word/bit index arithmetic, bounded
fn uint_bit<const IL: usize>(value: &Uint<IL>, k: usize) -> bool {
    (value.as_words()[k / 64] >> (k % 64)) & 1 == 1
}

#[cfg(test)]
#[allow(clippy::arithmetic_side_effects)]
mod tests {
    use super::*;
    use crate::pcs::zk::{mask::MaskSeeds, params::pow2};
    use rand::{SeedableRng, rngs::StdRng};
    use zinc_primality::{MillerRabin, PrimalityTest};

    const WL: usize = 8;
    const IL: usize = 4; // 256-bit inner field, ample for the 61-bit test p

    /// Small mask modulus (least prime above 2^61) for fast tests; the
    /// constraint layer is agnostic to the modulus size.
    fn small_modulus() -> Uint<WL> {
        let mut candidate = add!(pow2::<WL>(61), &Uint::from_u64(1));
        while !MillerRabin::is_probably_prime(&candidate) {
            candidate = add!(candidate, &Uint::from_u64(2));
        }
        candidate
    }

    struct Instance {
        layout: LiftLayout,
        columns: Vec<usize>,
        rho: Vec<Int<WL>>,
        rem: Vec<Int<WL>>,
        seeds: Vec<Vec<Uint<WL>>>,
        p: Uint<WL>,
        inner_cfg: InnerCfg<IL>,
    }

    /// Builds a small honest instance: 1 witness row + blinding row, D = 4,
    /// 3 opened columns (with a duplicate), one negative weight.
    fn honest_instance() -> Instance {
        let p = small_modulus();
        let p_bits = p.inner().bits();
        let layout = LiftLayout::new(2, 4, 3, p_bits);

        let inner_field =
            crate::pcs::zk::inner::field::InnerFieldParams::<IL>::find(
                layout.required_inner_bits(64),
                20,
            )
            .expect("inner prime");
        let inner_cfg = inner_field.cfg().expect("inner cfg");

        let mut rng = StdRng::seed_from_u64(42);
        let seeds = MaskSeeds::<WL>::sample(&mut rng, 2, 4, &p).seeds;
        let columns = vec![5usize, 17, 5];
        let rho = vec![Int::<WL>::from(-3i32)];

        // rem_l = X_{0,l} - 3 * X_{1,l}, computed from the masks directly.
        let mask_cfg = mask::mask_field_cfg(&p).expect("mask cfg");
        let rem: Vec<Int<WL>> = columns
            .iter()
            .map(|column| {
                let x0 = mask::mask_symbol_at(&seeds[0], *column, &mask_cfg);
                let x1 = mask::mask_symbol_at(&seeds[1], *column, &mask_cfg);
                let x0 = *x0.as_int();
                let x1 = *x1.as_int();
                let scaled = x1
                    .mul_by_scalar::<{ zinc_utils::CHECKED }>(&Int::<WL>::from(-3i32))
                    .expect("scaling fits");
                add!(x0, &scaled, "rem overflow")
            })
            .collect();

        Instance {
            layout,
            columns,
            rho,
            rem,
            seeds,
            p,
            inner_cfg,
        }
    }

    use zinc_utils::mul_by_scalar::MulByScalar;

    #[test]
    fn honest_witness_satisfies_all_constraints() {
        let inst = honest_instance();
        let constraints = build_constraints::<WL, IL>(
            &inst.layout,
            &inst.columns,
            &inst.rho,
            &inst.rem,
            &inst.p,
            &inst.inner_cfg,
        )
        .expect("constraints");
        let witness =
            build_witness::<WL, IL>(&inst.layout, &inst.seeds, &inst.columns, &inst.p)
                .expect("witness");

        assert_eq!(witness.len(), inst.layout.witness_len());
        // rows * columns lift constraints + columns aggregation constraints.
        assert_eq!(constraints.len(), 2 * 3 + 3);
        for (index, constraint) in constraints.iter().enumerate() {
            assert!(
                constraint.is_satisfied(&witness, &inst.inner_cfg),
                "constraint {index} violated by the honest witness",
            );
        }
    }

    #[test]
    fn tampered_bit_violates_some_constraint() {
        let inst = honest_instance();
        let constraints = build_constraints::<WL, IL>(
            &inst.layout,
            &inst.columns,
            &inst.rho,
            &inst.rem,
            &inst.p,
            &inst.inner_cfg,
        )
        .expect("constraints");
        let mut witness =
            build_witness::<WL, IL>(&inst.layout, &inst.seeds, &inst.columns, &inst.p)
                .expect("witness");

        // Flip the lowest bit of X_{1, 0}: the lift constraint for that cell
        // and the aggregation constraint for that column must both notice.
        let index = inst.layout.x_bit(1, 0, 0);
        witness[index] = !witness[index];
        assert!(
            constraints
                .iter()
                .any(|c| !c.is_satisfied(&witness, &inst.inner_cfg)),
            "tampered witness must violate at least one constraint",
        );
    }

    #[test]
    fn wrong_rem_is_unsatisfiable_by_honest_witness() {
        let mut inst = honest_instance();
        inst.rem[1] = add!(inst.rem[1], &Int::<WL>::from(1i32), "rem bump");
        let constraints = build_constraints::<WL, IL>(
            &inst.layout,
            &inst.columns,
            &inst.rho,
            &inst.rem,
            &inst.p,
            &inst.inner_cfg,
        )
        .expect("constraints");
        let witness =
            build_witness::<WL, IL>(&inst.layout, &inst.seeds, &inst.columns, &inst.p)
                .expect("witness");
        assert!(
            constraints
                .iter()
                .any(|c| !c.is_satisfied(&witness, &inst.inner_cfg)),
        );
    }

    #[test]
    fn required_inner_bits_covers_the_test_field() {
        let inst = honest_instance();
        // The chosen inner field must satisfy the no-wraparound bound.
        assert!(inst.layout.required_inner_bits(64) <= 256);
    }
}
