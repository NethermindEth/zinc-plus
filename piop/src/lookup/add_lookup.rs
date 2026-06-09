//! Add-table lookup data for the F_2[X] mod-2^32 adder.
//!
//! Builds the public add table `T = {(x,y,cin,s,cout) : s = (x+y+cin) mod 2^ℓ,
//! cout = (x+y+cin) >> ℓ}` (`2^{2ℓ+1}` rows), decomposes a 32-bit two-input add
//! `t = a + b + cin0 (mod 2^32)` into `32/ℓ` byte-limb lookup tuples with an
//! honest inter-limb carry chain, forms the tuple fingerprints that feed the
//! grand-product lookup ([`super::gkr_lookup`]), and computes the table
//! multiplicities (binary, LSB-first).
//!
//! This is the field-/arithmetic-level logic (the prover computes it from the
//! plaintext words). Wiring it to the committed `BinaryPoly<32>` trace — the
//! `ψ_z` read-off that binds the witness fingerprints, and committing the
//! carry bits + multiplicities — is the `f2_prove.rs` integration step.
//!
//! Parameterised by `limb_bits` (`ℓ`): production uses `ℓ = 8` (a `2^17` table);
//! tests use a smaller `ℓ` for speed.

use crypto_primitives::FromPrimitiveWithConfig;
use zinc_utils::inner_transparent_field::InnerTransparentField;

/// SHA-256 word width in bits.
pub const WORD_BITS: usize = 32;

/// A byte-limb lookup tuple `(x, y, cin, s, cout)` — the claim that
/// `s = (x + y + cin) mod 2^ℓ` and `cout = (x + y + cin) >> ℓ`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LimbTuple {
    pub x: u32,
    pub y: u32,
    pub cin: u32,
    pub s: u32,
    pub cout: u32,
}

/// Number of `ℓ`-bit limbs in a 32-bit word.
#[inline]
pub fn num_limbs(limb_bits: usize) -> usize {
    WORD_BITS / limb_bits
}

/// Number of rows in the add table for `ℓ`-bit limbs: `2^{2ℓ+1}`.
#[inline]
pub fn table_size(limb_bits: usize) -> usize {
    1usize << (2 * limb_bits + 1)
}

/// Canonical row index of `(x, y, cin)` (`x,y ∈ [0,2^ℓ)`, `cin ∈ {0,1}`):
/// `(x·2^ℓ + y)·2 + cin`. Matches the build order in [`add_table_fingerprints`].
#[inline]
#[allow(clippy::arithmetic_side_effects)]
pub fn row_index(x: u32, y: u32, cin: u32, limb_bits: usize) -> usize {
    ((((x << limb_bits) | y) << 1) | cin) as usize
}

#[inline]
fn emb<F: FromPrimitiveWithConfig>(v: u32, cfg: &F::Config) -> F {
    F::from_with_cfg(u64::from(v), cfg)
}

/// Tuple fingerprint `fp = emb(x) + γ·emb(y) + γ²·emb(cin) + γ³·emb(s) + γ⁴·emb(cout)`.
#[allow(clippy::arithmetic_side_effects)]
pub fn fingerprint<F>(t: &LimbTuple, gamma: &F, cfg: &F::Config) -> F
where
    F: InnerTransparentField + FromPrimitiveWithConfig,
{
    let g2 = gamma.clone() * gamma;
    let g3 = g2.clone() * gamma;
    let g4 = g3.clone() * gamma;
    emb::<F>(t.x, cfg)
        + &(gamma.clone() * &emb::<F>(t.y, cfg))
        + &(g2 * &emb::<F>(t.cin, cfg))
        + &(g3 * &emb::<F>(t.s, cfg))
        + &(g4 * &emb::<F>(t.cout, cfg))
}

/// All public add-table fingerprints, indexed by [`row_index`].
#[allow(clippy::arithmetic_side_effects)]
pub fn add_table_fingerprints<F>(gamma: &F, limb_bits: usize, cfg: &F::Config) -> Vec<F>
where
    F: InnerTransparentField + FromPrimitiveWithConfig,
{
    let limb = 1u32 << limb_bits;
    let mask = limb - 1;
    let mut table = Vec::with_capacity(table_size(limb_bits));
    for x in 0..limb {
        for y in 0..limb {
            for cin in 0..2u32 {
                let sum = x + y + cin;
                let tup = LimbTuple { x, y, cin, s: sum & mask, cout: sum >> limb_bits };
                table.push(fingerprint(&tup, gamma, cfg));
            }
        }
    }
    table
}

/// Decompose `t = a + b + cin0  (mod 2^32)` into `num_limbs(ℓ)` byte-limb tuples
/// `(a_i, b_i, cin_i, t_i, cout_i)`: `s_i = limb i of t` (the committed result),
/// the carry chain `cin_0 = cin0`, `cin_{i+1} = cout_i` derived honestly from
/// `a, b`. Returns the tuples and the carry-out bits (`carries[n−1]` is the word
/// overflow β). For an honest `t` every tuple lies in the add table; a wrong `t`
/// yields an off-table tuple at the corrupted limb.
#[allow(clippy::arithmetic_side_effects)]
pub fn decompose_add(
    a: u32,
    b: u32,
    t: u32,
    cin0: u32,
    limb_bits: usize,
) -> (Vec<LimbTuple>, Vec<u32>) {
    let n = num_limbs(limb_bits);
    let mask = (1u32 << limb_bits) - 1;
    let mut tuples = Vec::with_capacity(n);
    let mut carries = Vec::with_capacity(n);
    let mut cin = cin0;
    for i in 0..n {
        let shift = i * limb_bits;
        let ai = (a >> shift) & mask;
        let bi = (b >> shift) & mask;
        let ti = (t >> shift) & mask;
        let sum = ai + bi + cin;
        let cout = sum >> limb_bits;
        tuples.push(LimbTuple { x: ai, y: bi, cin, s: ti, cout });
        carries.push(cout);
        cin = cout;
    }
    (tuples, carries)
}

/// Fingerprints of the witness limb tuples (witness side of the lookup).
pub fn tuple_fingerprints<F>(tuples: &[LimbTuple], gamma: &F, cfg: &F::Config) -> Vec<F>
where
    F: InnerTransparentField + FromPrimitiveWithConfig,
{
    tuples.iter().map(|t| fingerprint(t, gamma, cfg)).collect()
}

/// Table multiplicities (binary, LSB-first, `k_bits` per row): how many witness
/// tuples land on each table row's `(x, y, cin)` inputs. `k_bits` must be large
/// enough for the maximum multiplicity (else the binary truncation is wrong).
#[allow(clippy::arithmetic_side_effects)]
pub fn multiplicities(tuples: &[LimbTuple], limb_bits: usize, k_bits: usize) -> Vec<Vec<bool>> {
    let mut counts = vec![0u64; table_size(limb_bits)];
    for t in tuples {
        counts[row_index(t.x, t.y, t.cin, limb_bits)] += 1;
    }
    counts
        .iter()
        .map(|&c| (0..k_bits).map(|b| (c >> b) & 1 == 1).collect())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lookup::gkr_lookup::{
        LookupError, prove_lookup, table_leaves, verify_lookup, witness_leaves,
    };
    use zinc_poly::univariate::binary_gf128::BinaryFieldGF128 as Gf;
    use zinc_transcript::Blake3Transcript;

    const LB: usize = 4; // small limbs → 2^9 = 512-row table, fast test
    const KBITS: usize = 6; // max multiplicity 63, ample for the test traces

    /// Run the full add-lookup over GF(2^128) for a set of `(a, b, t)` adds
    /// (`t` is the — possibly corrupted — committed result), returning the
    /// verifier's verdict.
    fn run(adds: &[(u32, u32, u32)]) -> Result<(), LookupError> {
        let cfg = ();
        let gamma = Gf::from_words([0x1234_5678, 0x9abc_def0]);
        let delta = Gf::from_words([0x0fed_cba9, 0x8765_4321]);

        let table = add_table_fingerprints::<Gf>(&gamma, LB, &cfg);
        let mut tuples = Vec::new();
        for &(a, b, t) in adds {
            let (tup, _carries) = decompose_add(a, b, t, 0, LB);
            tuples.extend(tup);
        }
        let wfps = tuple_fingerprints(&tuples, &gamma, &cfg);
        let mults = multiplicities(&tuples, LB, KBITS);

        let wl = witness_leaves(&wfps, &delta);
        let tl = table_leaves(&table, &mults, &delta, &cfg);

        let mut pt = Blake3Transcript::new();
        let (proof, _bind) = prove_lookup(&mut pt, wl, tl, &cfg);
        let mut vt = Blake3Transcript::new();
        verify_lookup(&mut vt, &proof, &cfg).map(|_| ())
    }

    #[test]
    fn add_lookup_accepts_honest_adds() {
        let adds: Vec<(u32, u32, u32)> =
            [(0x00ff_00ff_u32, 0x0001_0001_u32), (0x1234_5678, 0x8765_4321), (0xffff_ffff, 0x0000_0001)]
                .iter()
                .map(|&(a, b)| (a, b, a.wrapping_add(b)))
                .collect();
        run(&adds).expect("honest adds must be accepted");
    }

    #[test]
    fn add_lookup_rejects_wrong_sum() {
        let good = (0x00ff_00ff_u32, 0x0001_0001_u32, 0x00ff_00ff_u32.wrapping_add(0x0001_0001));
        let (a, b) = (0x1111_2222_u32, 0x3333_4444_u32);
        let bad = (a, b, a.wrapping_add(b) ^ 1); // corrupt the low limb of the result
        assert!(run(&[good, bad]).is_err(), "a wrong sum must be rejected (off-table tuple)");
    }

    #[test]
    fn decompose_reconstructs_sum() {
        // The committed limbs of an honest result recompose to a+b, and the
        // honest carries are consistent (carry chain reproduces the wrapping add).
        let (a, b) = (0xdead_beef_u32, 0x1234_5678_u32);
        let t = a.wrapping_add(b);
        let (tuples, carries) = decompose_add(a, b, t, 0, LB);
        assert_eq!(tuples.len(), num_limbs(LB));
        // every honest tuple satisfies the add relation
        let mask = (1u32 << LB) - 1;
        for tup in &tuples {
            assert_eq!(tup.s, (tup.x + tup.y + tup.cin) & mask);
            assert_eq!(tup.cout, (tup.x + tup.y + tup.cin) >> LB);
        }
        assert_eq!(carries.len(), num_limbs(LB));
    }
}
