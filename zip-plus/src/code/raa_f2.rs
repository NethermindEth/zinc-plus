//! Repeat-Accumulate-Accumulate (RAA) linear code over `F_2`.
//!
//! Like [`RaaCode`](super::raa::RaaCode), but the accumulation step is the
//! `F_2[X]` XOR accumulator instead of integer cumulative sum. This makes
//! the codeword type identical to the input type (no bit-width inflation),
//! which is the natural fit for committing to multilinear extensions whose
//! cells are `F_2^{<D}[X]` (e.g. `BinaryPoly<D>`).
//!
//! The linear map applied to each input row is `M · P_2 · M · P_1 · R`,
//! where `R` is the REP-fold repetition, `P_1, P_2` are fixed seeded
//! permutations, and `M` is the 0/1 lower-triangular accumulation matrix
//! — the same matrix as in the integer RAA code. The semiring under
//! which the accumulation runs is whatever `Out`'s `AddAssign<&Self>`
//! implements; for `BinaryPoly<D>` that is XOR (so `1 + 1 = 0`).

use crate::{
    code::{
        F2LinearOpener, LinearCode,
        raa::{RaaConfig, accumulate_unchecked, clone_shuffled, repeat},
    },
    pcs::structs::ZipTypes,
    utils::shuffle_seeded,
};
use crypto_primitives::PrimeField;
use std::{fmt::Debug, marker::PhantomData, ops::AddAssign};
use zinc_poly::univariate::{F2AddAssign, F2PackU64, binary_f2_wide::BinaryF2Poly};
use zinc_utils::{from_ref::FromRef, mul};

/// RAA code over `F_2` (XOR accumulator, no codeword widening).
///
/// `Config::CHECK_FOR_OVERFLOWS` is ignored — there is no integer-style
/// overflow to track in `F_2`. `Config::PERMUTE_IN_PLACE` is honoured.
#[derive(Clone)]
pub struct RaaF2Code<Zt: ZipTypes, Config: RaaConfig, const REP: usize> {
    pub(crate) row_len: usize,
    pub(crate) perm_1_seed: u64,
    pub(crate) perm_2_seed: u64,
    pub(crate) perm_1: Vec<usize>,
    pub(crate) perm_2: Vec<usize>,
    phantom: PhantomData<(Zt, Config)>,
}

/// Recommended number of column openings for the proximity check
/// when using `RaaF2Code` at a given repetition factor (= inverse
/// rate). These are the values that drive the per-column-opening
/// soundness error below cryptographic thresholds; pass the result
/// to the F_2[X] open's `num_column_openings` argument in
/// `protocol::f2_prove`.
///
/// Currently only rate `1/4` (`REP = 4`) is calibrated, since
/// that's the configuration the F_2 protocol targets. Adding new
/// rates requires re-running the proximity-error analysis for the
/// new `(REP, code distance)` pair.
pub const fn recommended_num_column_openings(rep: usize) -> usize {
    match rep {
        // RAA, rate 1/4 (REP = 4): 987 openings give a soundness
        // error well below 2^{-100} for the protocol's distance
        // bound.
        4 => 987,
        _ => panic!(
            "recommended_num_column_openings: unsupported repetition factor; \
             only REP = 4 (rate 1/4) is currently calibrated"
        ),
    }
}

impl<Zt: ZipTypes, Config: RaaConfig, const REP: usize> RaaF2Code<Zt, Config, REP> {
    pub fn new(row_len: usize) -> Self {
        assert!(
            REP.is_power_of_two(),
            "Repetition factor must be a power of two"
        );
        assert!(
            row_len.is_power_of_two(),
            "Row length must be a power of two"
        );

        // Note: no `codeword_width_bits >= initial_bits + 2*log2(...)` check
        // here — accumulation in `F_2` doesn't widen the codeword type, so
        // we only require `Cw` to be defined.

        const PERM_1_SEED: u64 = 1;
        const PERM_2_SEED: u64 = 2;

        let codeword_len = mul!(row_len, REP);

        let mut perm_1: Vec<usize> = (0..codeword_len).collect();
        shuffle_seeded(&mut perm_1, PERM_1_SEED);
        let mut perm_2: Vec<usize> = (0..codeword_len).collect();
        shuffle_seeded(&mut perm_2, PERM_2_SEED);

        Self {
            row_len,
            perm_1_seed: PERM_1_SEED,
            perm_2_seed: PERM_2_SEED,
            perm_1,
            perm_2,
            phantom: PhantomData,
        }
    }

    /// Encode using non-`F_2` arithmetic (e.g. `Int<M>` or a field).
    /// Used for `encode_wide` and `encode_f`, where the accumulator
    /// is integer / field addition — not `F_2` XOR. The linear-map
    /// matrix is the same `M · P_2 · M · P_1 · R`; only the addition
    /// semantics differ from `encode_f2`.
    fn encode_inner<In, Out>(&self, row: &[In]) -> Vec<Out>
    where
        Out: for<'a> AddAssign<&'a Out> + FromRef<In> + Clone,
    {
        debug_assert_eq!(
            row.len(),
            self.row_len,
            "Row length must match the code's row length"
        );

        let mut result: Vec<Out> = repeat(row, REP);
        if Config::PERMUTE_IN_PLACE {
            shuffle_seeded(&mut result, self.perm_1_seed);
        } else {
            result = clone_shuffled(&result, &self.perm_1);
        }
        accumulate_unchecked(&mut result);
        if Config::PERMUTE_IN_PLACE {
            shuffle_seeded(&mut result, self.perm_2_seed);
        } else {
            result = clone_shuffled(&result, &self.perm_2);
        }
        accumulate_unchecked(&mut result);
        debug_assert_eq!(result.len(), mul!(self.row_len, REP));
        result
    }

    /// Encode using `F_2` (XOR) addition: `1 + 1 = 0`. Used for the
    /// `encode` path where `Eval = Cw` are `F_2[X]/<X^D>`-typed cells.
    ///
    /// Generic fallback that operates directly on `Out` values via the
    /// [`F2AddAssign`] trait. For `D ≤ 64` cells, prefer
    /// [`Self::encode_f2_packed`] — it stays in `u64`-space throughout
    /// the repeat / permute / accumulate steps, avoiding per-cell
    /// Boolean-array work and 32-byte cell clones.
    #[allow(dead_code)]
    fn encode_f2<In, Out>(&self, row: &[In]) -> Vec<Out>
    where
        Out: F2AddAssign + FromRef<In> + Clone,
    {
        debug_assert_eq!(
            row.len(),
            self.row_len,
            "Row length must match the code's row length"
        );

        let mut result: Vec<Out> = repeat(row, REP);
        if Config::PERMUTE_IN_PLACE {
            shuffle_seeded(&mut result, self.perm_1_seed);
        } else {
            result = clone_shuffled(&result, &self.perm_1);
        }
        f2_accumulate(&mut result);
        if Config::PERMUTE_IN_PLACE {
            shuffle_seeded(&mut result, self.perm_2_seed);
        } else {
            result = clone_shuffled(&result, &self.perm_2);
        }
        f2_accumulate(&mut result);
        debug_assert_eq!(result.len(), mul!(self.row_len, REP));
        result
    }

    /// Public F_2-linear encoder over any type implementing
    /// `F2AddAssign + FromRef<In> + Clone`. Same kernel as
    /// [`Self::encode_f2`] (Repeat → Permute → XOR-accumulate →
    /// Permute → XOR-accumulate), exposed for downstream callers
    /// (e.g. the F_2[X] MLE-opening protocol) that need to encode a
    /// row whose entries live in a wider F_2[X]-polynomial type than
    /// `Zt::Cw` while reusing the same code permutations.
    ///
    /// This is sound because the encoder is F_2[X]-linear: the
    /// permutations are coordinate moves (commute with F_2[X]-scalar
    /// multiplication), and XOR-accumulate IS F_2 addition (which
    /// distributes over F_2[X]-scalar multiplication coordinate-by-
    /// coordinate within each cell).
    pub fn encode_f2_lin<In, Out>(&self, row: &[In]) -> Vec<Out>
    where
        Out: F2AddAssign + FromRef<In> + Clone,
    {
        self.encode_f2(row)
    }

    /// Packed-u64 fast path for the F_2 encode. Each input/output cell
    /// must fit in a single `u64` (D ≤ 64), in which case repeat /
    /// permute / accumulate become straight `u64` ops: XOR for the
    /// prefix-sum, `Vec<u64>` clone-shuffle for the permutations,
    /// `Vec::cycle`-style fill for the repeat.
    ///
    /// On a 32-coefficient cell this avoids ~32 Boolean-iterator
    /// touches per `f2_add_assign` and replaces 32-byte cell clones
    /// with single 8-byte word copies — the dominant cost in the
    /// generic path.
    fn encode_f2_packed<In, Out>(&self, row: &[In]) -> Vec<Out>
    where
        In: F2PackU64,
        Out: F2PackU64,
    {
        debug_assert_eq!(
            row.len(),
            self.row_len,
            "Row length must match the code's row length"
        );

        // Pack the input row once. Subsequent passes work on `u64`.
        let packed_row: Vec<u64> = row.iter().map(F2PackU64::pack_u64).collect();

        // Repeat.
        let codeword_len = mul!(self.row_len, REP);
        let mut buf: Vec<u64> = Vec::with_capacity(codeword_len);
        for _ in 0..REP {
            buf.extend_from_slice(&packed_row);
        }

        // First permutation + prefix XOR + second permutation + prefix XOR.
        if Config::PERMUTE_IN_PLACE {
            shuffle_seeded(&mut buf, self.perm_1_seed);
        } else {
            buf = self.perm_1.iter().map(|&i| buf[i]).collect();
        }
        f2_accumulate_u64(&mut buf);
        if Config::PERMUTE_IN_PLACE {
            shuffle_seeded(&mut buf, self.perm_2_seed);
        } else {
            buf = self.perm_2.iter().map(|&i| buf[i]).collect();
        }
        f2_accumulate_u64(&mut buf);

        debug_assert_eq!(buf.len(), codeword_len);
        buf.into_iter().map(Out::unpack_u64).collect()
    }
}

/// `F_2` cumulative-XOR. Same prefix-sum structure as
/// [`accumulate`](super::raa::accumulate), but over `F_2` (XOR), so
/// `1 + 1 = 0` and no overflow notion exists.
fn f2_accumulate<I>(input: &mut [I])
where
    I: F2AddAssign + Clone,
{
    if let Some(first) = input.first().cloned() {
        let mut acc = first;
        for i in 1..input.len() {
            // SAFETY: indices `i` and the slice element at `i` are in
            // bounds — same pattern as `accumulate_unchecked`.
            unsafe {
                acc.f2_add_assign(input.get_unchecked(i));
                *input.get_unchecked_mut(i) = acc.clone();
            };
        }
    }
}

/// `F_2` cumulative-XOR specialized to `u64` cells. Single-instruction
/// XOR per step, no virtual dispatch, no Boolean-iterator walks.
#[inline]
fn f2_accumulate_u64(input: &mut [u64]) {
    let n = input.len();
    if n < 2 {
        return;
    }
    let mut acc = input[0];
    // SAFETY: `i` is bounded by `n = input.len()`.
    unsafe {
        for i in 1..n {
            acc ^= *input.get_unchecked(i);
            *input.get_unchecked_mut(i) = acc;
        }
    }
}

impl<Zt: ZipTypes, Config: RaaConfig, const REP: usize> F2LinearOpener
    for RaaF2Code<Zt, Config, REP>
{
    fn encode_f2_lin_open<const W: usize>(
        &self,
        row: &[BinaryF2Poly<W>],
    ) -> Vec<BinaryF2Poly<W>> {
        // BinaryF2Poly<W> implements F2AddAssign + FromRef<Self> + Clone,
        // so the F_2-linear encoder kernel works directly.
        self.encode_f2_lin::<BinaryF2Poly<W>, BinaryF2Poly<W>>(row)
    }
}

impl<Zt: ZipTypes, Config: RaaConfig, const REP: usize> LinearCode<Zt>
    for RaaF2Code<Zt, Config, REP>
where
    Zt::Cw: F2AddAssign,
{
    const REPETITION_FACTOR: usize = REP;

    fn row_len(&self) -> usize {
        self.row_len
    }

    #[allow(clippy::arithmetic_side_effects)]
    fn codeword_len(&self) -> usize {
        self.row_len * REP
    }

    fn params_string(&self) -> String {
        format!("row_len={}, rate=1/{REP} (F_2)", self.row_len())
    }

    fn encode(&self, row: &[Zt::Eval]) -> Vec<Zt::Cw> {
        // The `Eval -> Cw` map is the only path that runs in `F_2`:
        // cells start as `F_2[X]/<X^D>`-typed and the accumulator must
        // collapse `1 + 1` to `0`, not panic on overflow.
        //
        // We use [`Self::encode_f2`] (which dispatches `f2_add_assign`
        // per cell) rather than [`Self::encode_f2_packed`]. Under the
        // `simd` feature `BinaryPoly` is already `BinaryU64Poly`
        // (`#[repr(transparent)] struct(u64)`), so the two paths do
        // the same XOR ops on the same in-memory representation —
        // but `encode_f2_packed` adds an extra pack pass on input and
        // an extra masked-unpack pass on the codeword, which is pure
        // overhead in that case. For non-simd builds (`BinaryRefPoly`)
        // the packed path is the faster choice; callers in that
        // regime should invoke it directly.
        self.encode_f2(row)
    }

    fn encode_wide(&self, row: &[Zt::CombR]) -> Vec<Zt::CombR> {
        // The combined row lives in `CombR` (a wide integer ring), so
        // this uses regular integer addition for accumulation.
        self.encode_inner(row)
    }

    fn encode_f<F>(&self, row: &[F]) -> Vec<F>
    where
        F: PrimeField + FromRef<F>,
    {
        // Field arithmetic; the prime field's `AddAssign<&F>` provides
        // the right accumulator. Not `F_2`-specific.
        self.encode_inner(row)
    }

    /// Fused F_2 encoder that writes its result into a caller-provided
    /// (uninitialised) destination, reusing `scratch` as the
    /// post-first-accumulate intermediate. Zero allocations per call.
    ///
    /// Pipeline (PERMUTE_IN_PLACE = false path; in-place permutation is
    /// not used by the F_2 protocol):
    ///   1. write `repeat(row, REP)` into `dst` (initialising every slot)
    ///   2. gather `perm_1` from `dst` into `scratch`
    ///   3. F_2 prefix-XOR in place on `scratch`
    ///   4. gather `perm_2` from `scratch` into `dst`
    ///   5. F_2 prefix-XOR in place on `dst`
    #[allow(clippy::arithmetic_side_effects)]
    fn encode_into_with_scratch(
        &self,
        row: &[Zt::Eval],
        scratch: &mut [Zt::Cw],
        dst: &mut [std::mem::MaybeUninit<Zt::Cw>],
    ) {
        let codeword_len = self.row_len * REP;
        debug_assert_eq!(row.len(), self.row_len);
        debug_assert!(scratch.len() >= codeword_len);
        debug_assert_eq!(dst.len(), codeword_len);

        // Step 1: write `repeat(row, REP)` into `dst`, initialising every slot.
        for r in 0..REP {
            let offset = r * self.row_len;
            for (i, x) in row.iter().enumerate() {
                // SAFETY: `offset + i < REP * row_len = codeword_len = dst.len()`.
                unsafe {
                    dst.get_unchecked_mut(offset + i)
                        .write(Zt::Cw::from_ref(x));
                }
            }
        }

        // After Step 1, all `codeword_len` slots of `dst` are initialised.
        // SAFETY: every slot was written above via `MaybeUninit::write`; the
        // layouts of `MaybeUninit<T>` and `T` match.
        let dst_init: &mut [Zt::Cw] = unsafe {
            std::slice::from_raw_parts_mut(dst.as_mut_ptr().cast::<Zt::Cw>(), codeword_len)
        };

        // Step 2: gather perm_1 from `dst_init` into `scratch`.
        // SAFETY: `perm_1` is a permutation of `0..codeword_len`, so every
        // index is in bounds for both `dst_init` and `scratch[..codeword_len]`.
        unsafe {
            for i in 0..codeword_len {
                let p = *self.perm_1.get_unchecked(i);
                *scratch.get_unchecked_mut(i) = dst_init.get_unchecked(p).clone();
            }
        }

        // Step 3: prefix-XOR on scratch[..codeword_len].
        f2_accumulate(&mut scratch[..codeword_len]);

        // Step 4: gather perm_2 from `scratch` into `dst_init`.
        // SAFETY: same permutation invariant as Step 2.
        unsafe {
            for i in 0..codeword_len {
                let p = *self.perm_2.get_unchecked(i);
                *dst_init.get_unchecked_mut(i) = scratch.get_unchecked(p).clone();
            }
        }

        // Step 5: prefix-XOR on `dst_init`.
        f2_accumulate(dst_init);
    }
}

impl<Zt: ZipTypes, Config: RaaConfig, const REP: usize> Debug for RaaF2Code<Zt, Config, REP> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RaaF2Code")
            .field("row_len", &self.row_len)
            .field("perm_1_seed", &self.perm_1_seed)
            .field("perm_2_seed", &self.perm_2_seed)
            .finish()
    }
}

impl<Zt: ZipTypes, Config: RaaConfig, const REP: usize> PartialEq for RaaF2Code<Zt, Config, REP> {
    fn eq(&self, other: &Self) -> bool {
        self.row_len == other.row_len
            && self.perm_1_seed == other.perm_1_seed
            && self.perm_2_seed == other.perm_2_seed
    }
}

impl<Zt: ZipTypes, Config: RaaConfig, const REP: usize> Eq for RaaF2Code<Zt, Config, REP> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pcs::test_utils::TestBinPolyF2ZipTypes;
    use crypto_primitives::boolean::Boolean;
    use num_traits::Zero;
    use zinc_poly::univariate::binary::BinaryPoly;

    const REP: usize = 4;
    const D: usize = 32;

    #[derive(Clone, Copy)]
    struct Cfg;
    impl RaaConfig for Cfg {
        const PERMUTE_IN_PLACE: bool = false;
        const CHECK_FOR_OVERFLOWS: bool = false;
    }

    type Zt = TestBinPolyF2ZipTypes<D>;
    type Code = RaaF2Code<Zt, Cfg, REP>;

    fn bp(bits: u32) -> BinaryPoly<D> {
        let coeffs: Vec<Boolean> = (0..D).map(|i| ((bits >> i) & 1 == 1).into()).collect();
        BinaryPoly::new(coeffs)
    }

    #[test]
    fn zero_row_encodes_to_zero_codeword() {
        let row_len = 8;
        let code = Code::new(row_len);
        let zero_row: Vec<BinaryPoly<D>> = vec![BinaryPoly::zero(); row_len];
        let cw: Vec<BinaryPoly<D>> = code.encode(&zero_row);
        assert_eq!(cw.len(), row_len * REP);
        assert!(cw.iter().all(|x| x.is_zero()));
    }

    #[test]
    fn encoding_is_linear_over_f2() {
        let row_len = 8;
        let code = Code::new(row_len);
        let a: Vec<BinaryPoly<D>> = (0..row_len as u32).map(bp).collect();
        let b: Vec<BinaryPoly<D>> = (0..row_len as u32).map(|i| bp(i.wrapping_mul(7) ^ 0x5a)).collect();

        // F_2 sum is XOR: use `F2AddAssign` so the non-SIMD ref-poly
        // path doesn't panic on `1 + 1` (which it does because
        // `Boolean::Add` enforces an integer-{0,1} contract).
        let sum_ab: Vec<BinaryPoly<D>> = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| {
                let mut x = *x;
                x.f2_add_assign(y);
                x
            })
            .collect();

        let cw_a: Vec<BinaryPoly<D>> = code.encode(&a);
        let cw_b: Vec<BinaryPoly<D>> = code.encode(&b);
        let cw_sum: Vec<BinaryPoly<D>> = code.encode(&sum_ab);

        let sum_cws: Vec<BinaryPoly<D>> = cw_a
            .iter()
            .zip(cw_b.iter())
            .map(|(x, y)| {
                let mut x = *x;
                x.f2_add_assign(y);
                x
            })
            .collect();

        assert_eq!(cw_sum, sum_cws, "F_2 linearity must hold for the encoder");
    }

    #[test]
    fn encoding_is_deterministic() {
        let row_len = 16;
        let code1 = Code::new(row_len);
        let code2 = Code::new(row_len);
        let row: Vec<BinaryPoly<D>> = (0..row_len as u32).map(bp).collect();
        let cw1: Vec<BinaryPoly<D>> = code1.encode(&row);
        let cw2: Vec<BinaryPoly<D>> = code2.encode(&row);
        assert_eq!(cw1, cw2);
    }

    #[test]
    fn one_plus_one_is_zero_via_accumulate() {
        // Two repeats of [1] under REP=2 would yield, after the initial
        // repeat step, [1, 1]; XOR-accumulate then gives [1, 0]. This
        // exercises the 1+1=0 behaviour directly.
        let row_len = 1;

        #[derive(Clone, Copy)]
        struct Cfg2;
        impl RaaConfig for Cfg2 {
            const PERMUTE_IN_PLACE: bool = false;
            const CHECK_FOR_OVERFLOWS: bool = false;
        }
        type Code2 = RaaF2Code<Zt, Cfg2, 2>;

        let code = Code2::new(row_len);
        let row = vec![BinaryPoly::<D>::from(1u32)];
        let cw: Vec<BinaryPoly<D>> = code.encode(&row);
        // After repeat([1], 2)=[1,1]; perm; XOR-acc; perm; XOR-acc — XOR
        // of an even number of 1s collapses to 0 somewhere.
        let total_xor = cw.iter().fold(BinaryPoly::<D>::zero(), |mut acc, x| {
            acc.f2_add_assign(x);
            acc
        });
        // Two 1's under XOR accumulation produce a codeword whose total
        // running-XOR (sum mod 2) is 0 — i.e. F_2 cancellation occurred.
        assert_eq!(total_xor, BinaryPoly::<D>::zero());
    }
}
