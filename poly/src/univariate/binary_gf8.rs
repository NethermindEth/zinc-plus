//! `GF(2^8)` realised as the order-255 subfield of our GHASH `GF(2^128)`,
//! with a verified `F_2`-linear field embedding back into `GF(2^128)`.
//!
//! This is the prover-side **speed lever** for the oblong AND zerocheck
//! (port plan `documentation/f2-hadamard-oblong-port-plan.md`, prerequisite
//! P1): the additive-NTT and the per-extension-point products can run in the
//! 8-bit field (cheap log/antilog mults, 1-byte XORs) instead of `GF(2^128)`
//! (CLMUL + 16-byte XORs), and only the final accumulation lifts to the big
//! field via [`embed`]. See [`super::oblong_and`] for the naive `GF(2^128)`
//! path this accelerates.
//!
//! ## Why derive `GF(2^8)` *from* GHASH (not from the AES polynomial)
//!
//! Binius uses the AES field `GF(2^8) = F_2[X]/(X^8+X^4+X^3+X+1)` and must
//! find its embedding into its `B128`. We instead build `GF(2^8)` *inside*
//! our `GF(2^128)`: the **relative norm** `θ = N_{GF(2^128)/GF(2^8)}(g) =
//! ∏_{k=0}^{15} g^{2^{8k}}` of a suitable `g` lands in the order-255 subfield,
//! and we define `GF(2^8) = F_2[X]/m(X)` with `m = minpoly(θ)`. Then the map
//! `α ↦ θ` (i.e. `Σ cᵢ αⁱ ↦ Σ cᵢ θⁱ`) is **automatically a field
//! homomorphism** — no isomorphism search, no AES basis. The generator
//! `α = X` (byte `0x02`) maps to `θ`, primitive of order 255.
//!
//! The construction (`θ`, `m`, the log/antilog and embed tables) is computed
//! once from our `GF(2^128)` arithmetic and memoised; the
//! `embed_is_a_field_homomorphism` test checks `embed(a·b)=embed(a)·embed(b)`
//! over all 65536 pairs, so the derivation is self-validating.

use std::sync::OnceLock;

use super::binary_gf128::BinaryFieldGF128;

type F128 = BinaryFieldGF128;

/// `255 = 2^8 − 1`, the order of `GF(2^8)*`.
const ORDER_MINUS_1: u32 = 255;

/// An element of `GF(2^8)`, the order-255 subfield of GHASH `GF(2^128)`.
/// Stored as a byte: bit `i` is the coefficient of `αⁱ` (`α = X`, the
/// generator). Addition is XOR; multiplication is log/antilog over the
/// reduction polynomial `m(X) = minpoly(θ)`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
pub struct Gf8(pub u8);

impl Gf8 {
    pub const ZERO: Gf8 = Gf8(0);
    pub const ONE: Gf8 = Gf8(1);

    /// The multiplicative generator `α = X` (byte `0x02`), of order 255.
    #[inline]
    pub const fn generator() -> Gf8 {
        Gf8(2)
    }

    #[inline]
    pub fn is_zero(self) -> bool {
        self.0 == 0
    }

    /// `self · rhs` via the log/antilog tables.
    #[inline]
    #[allow(clippy::arithmetic_side_effects)]
    pub fn mul(self, rhs: Gf8) -> Gf8 {
        if self.0 == 0 || rhs.0 == 0 {
            return Gf8::ZERO;
        }
        let t = tables();
        let l = t.log[self.0 as usize] as u32 + t.log[rhs.0 as usize] as u32;
        Gf8(t.antilog[(l % ORDER_MINUS_1) as usize])
    }

    /// `self + rhs` (XOR).
    #[inline]
    pub fn add(self, rhs: Gf8) -> Gf8 {
        Gf8(self.0 ^ rhs.0)
    }

    /// `self^exp` by square-and-multiply.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn pow(self, mut exp: u32) -> Gf8 {
        let mut acc = Gf8::ONE;
        let mut base = self;
        while exp > 0 {
            if exp & 1 == 1 {
                acc = acc.mul(base);
            }
            exp >>= 1;
            if exp > 0 {
                base = base.mul(base);
            }
        }
        acc
    }

    /// `self^{-1}` (`= self^{254}`); panics on zero.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn inverse(self) -> Gf8 {
        assert!(self.0 != 0, "GF(2^8): zero has no inverse");
        let t = tables();
        let li = t.log[self.0 as usize] as u32;
        Gf8(t.antilog[((ORDER_MINUS_1 - li) % ORDER_MINUS_1) as usize])
    }
}

impl std::ops::Add for Gf8 {
    type Output = Gf8;
    #[inline]
    fn add(self, rhs: Gf8) -> Gf8 {
        Gf8::add(self, rhs)
    }
}

impl std::ops::Sub for Gf8 {
    type Output = Gf8;
    #[inline]
    fn sub(self, rhs: Gf8) -> Gf8 {
        // Characteristic 2: subtraction is XOR.
        Gf8::add(self, rhs)
    }
}

impl std::ops::Mul for Gf8 {
    type Output = Gf8;
    #[inline]
    fn mul(self, rhs: Gf8) -> Gf8 {
        Gf8::mul(self, rhs)
    }
}

/// Lift a `GF(2^8)` element into `GF(2^128)` (the unique field embedding
/// fixing `F_2`, mapping `α ↦ θ`). `F_2`-linear; computed by table lookup.
#[inline]
pub fn embed(a: Gf8) -> F128 {
    tables().embed[a.0 as usize]
}

/// The generator's image `θ ∈ GF(2^128)` (`= embed(Gf8::generator())`).
pub fn theta() -> F128 {
    tables().theta
}

// ---------------------------------------------------------------------------
// One-time construction from our GF(2^128) arithmetic.
// ---------------------------------------------------------------------------

struct Gf8Tables {
    /// Low byte of the reduction polynomial `m(X) = X^8 + (m_low as poly)`.
    /// Retained for the `mul_matches_carryless_reference` cross-check (and as
    /// documentation of the derived field); not needed by the runtime path.
    #[allow(dead_code)]
    m_low: u8,
    /// `antilog[i] = αⁱ` as a byte, `i ∈ 0..255` (covers every nonzero byte).
    antilog: [u8; 255],
    /// `log[v] = i` such that `αⁱ = v`; `log[0]` is an unused sentinel.
    log: [u8; 256],
    /// `embed[b]` = image in `GF(2^128)` of the byte `b`.
    embed: [F128; 256],
    /// Full `256×256` product table: `mul_table[(a<<8)|b] = a·b` in `GF(2^8)`.
    /// One L2 lookup per multiply — no log/antilog/modulo, and (when fetched
    /// once per hot loop via [`gf8_mul_embed_tables`]) no per-op `OnceLock` load,
    /// which matters on aarch64 where there is no GFNI native `GF(2^8)` mul.
    mul_table: Box<[u8; 65536]>,
    theta: F128,
}

fn tables() -> &'static Gf8Tables {
    static T: OnceLock<Gf8Tables> = OnceLock::new();
    T.get_or_init(build_tables)
}

/// Borrow the `(mul_table, embed_table)` once for a hot loop, so the inner
/// `GF(2^8)` multiplies and embeds are plain array lookups with no per-op
/// `OnceLock` fetch. `Gf8(mul_table[(a.0<<8)|b.0]) = a·b`; `embed_table[x.0] =
/// embed(Gf8(x))`.
pub(crate) fn gf8_mul_embed_tables() -> (&'static [u8; 65536], &'static [F128; 256]) {
    let t = tables();
    (&t.mul_table, &t.embed)
}

#[inline]
fn to_u128(f: &F128) -> u128 {
    let w = f.words();
    (w[0] as u128) | ((w[1] as u128) << 64)
}

/// `g^{2^8}` = eight squarings (the `GF(2^8)`-relative Frobenius).
fn frob8(mut g: F128) -> F128 {
    for _ in 0..8 {
        g = g.square();
    }
    g
}

/// The relative norm `N_{GF(2^128)/GF(2^8)}(g) = ∏_{k=0}^{15} g^{2^{8k}}`,
/// which lies in the order-255 subfield.
#[allow(clippy::arithmetic_side_effects)]
fn relative_norm(g: F128) -> F128 {
    let mut acc = F128::one();
    let mut t = g;
    for _ in 0..16 {
        acc = acc * t;
        t = frob8(t);
    }
    acc
}

/// Is `θ` a primitive element of `GF(2^8)*` (order exactly 255)?
/// `255 = 3·5·17`, so check `θ^255 = 1` and `θ^{255/p} ≠ 1` for `p ∈ {3,5,17}`.
fn is_primitive_gf8(theta: F128) -> bool {
    if theta.pow_u32(255) != F128::one() {
        return false;
    }
    theta.pow_u32(85) != F128::one()
        && theta.pow_u32(51) != F128::one()
        && theta.pow_u32(15) != F128::one()
}

/// Find a primitive generator `θ` of the `GF(2^8)` subfield by taking the
/// relative norm of small field elements until one is primitive.
fn find_theta() -> F128 {
    for seed in 2u64..10_000 {
        let g = F128::from_words([seed, 0]);
        let theta = relative_norm(g);
        if is_primitive_gf8(theta) {
            return theta;
        }
    }
    panic!("GF(2^8): no primitive subfield generator found among the first seeds");
}

/// Low byte of `m(X) = minpoly(θ)`: the `F_2` dependency
/// `θ^8 = Σ_{i<8} cᵢ θⁱ`, packed as the byte `Σ cᵢ 2ⁱ`. Found via an XOR
/// linear basis over the 128-bit vectors `θ^0 … θ^8` with provenance tags.
#[allow(clippy::arithmetic_side_effects)]
fn minpoly_low(theta: F128) -> u8 {
    // pivots[h] = Some((reduced_vector, provenance)) with leading bit h.
    let mut pivots: Vec<Option<(u128, u16)>> = vec![None; 128];
    let mut power = F128::one();
    for i in 0..=8u16 {
        let mut v = to_u128(&power);
        let mut tag = 1u16 << i;
        while v != 0 {
            let h = (127 - v.leading_zeros()) as usize;
            match pivots[h] {
                Some((pv, pt)) => {
                    v ^= pv;
                    tag ^= pt;
                }
                None => {
                    pivots[h] = Some((v, tag));
                    break;
                }
            }
        }
        if v == 0 {
            // Dependency Σ (tag bit j)·θ^j = 0; bit 8 = the X^8 term.
            // m(X) = X^8 + Σ_{j<8} cⱼ Xʲ ⇒ low byte = tag & 0xFF.
            debug_assert_eq!(tag >> 8, 1, "first dependency must be at θ^8");
            return (tag & 0xFF) as u8;
        }
        power = power * theta;
    }
    panic!("GF(2^8): θ^0..θ^8 unexpectedly independent (θ not degree 8)");
}

/// Multiply a `GF(2^8)` byte by `α = X`: shift left, reduce by `m_low` if the
/// top bit was set (the `X^8 ≡ m_low` reduction).
#[inline]
#[allow(clippy::arithmetic_side_effects)]
fn mul_by_x(v: u8, m_low: u8) -> u8 {
    let hi = v & 0x80;
    let shifted = v << 1;
    if hi != 0 { shifted ^ m_low } else { shifted }
}

#[allow(clippy::arithmetic_side_effects)]
fn build_tables() -> Gf8Tables {
    let theta = find_theta();
    let m_low = minpoly_low(theta);

    // log / antilog over GF(2^8) = F_2[X]/m(X), generator α = X.
    let mut antilog = [0u8; 255];
    let mut log = [0u8; 256];
    let mut x = 1u8;
    for i in 0..255usize {
        antilog[i] = x;
        log[x as usize] = i as u8;
        x = mul_by_x(x, m_low);
    }
    debug_assert_eq!(x, 1, "α must have order 255 (antilog cycle closes)");

    // embed[b] = Σ_{i: bit i of b} θ^i  (F_2-linear; α ↦ θ).
    let basis: [F128; 8] = {
        let mut p = F128::one();
        std::array::from_fn(|_| {
            let cur = p;
            p = p * theta;
            cur
        })
    };
    let mut embed = [F128::zero(); 256];
    for (b, slot) in embed.iter_mut().enumerate() {
        let mut acc = F128::zero();
        for (i, &bi) in basis.iter().enumerate() {
            if (b >> i) & 1 == 1 {
                acc = acc + bi;
            }
        }
        *slot = acc;
    }

    // Full 256×256 product table from log/antilog (no modulo at lookup time).
    let mut mul_table = Box::new([0u8; 65536]);
    for a in 1..256usize {
        for b in 1..256usize {
            let l = log[a] as usize + log[b] as usize;
            mul_table[(a << 8) | b] = antilog[if l >= 255 { l - 255 } else { l }];
        }
    }

    Gf8Tables {
        m_low,
        antilog,
        log,
        embed,
        mul_table,
        theta,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn antilog_covers_all_nonzero_bytes() {
        // α primitive ⇒ {α^0..α^254} = all 255 nonzero bytes, each once.
        let t = tables();
        let mut seen = [false; 256];
        for &v in &t.antilog {
            assert_ne!(v, 0, "antilog must avoid zero");
            assert!(!seen[v as usize], "antilog repeats a value");
            seen[v as usize] = true;
        }
        assert_eq!(seen.iter().filter(|&&s| s).count(), 255);
    }

    #[test]
    fn log_antilog_are_inverse() {
        let t = tables();
        for i in 0..255u32 {
            let v = t.antilog[i as usize];
            assert_eq!(t.log[v as usize] as u32, i);
        }
    }

    #[test]
    fn field_axioms() {
        // identities, commutativity, inverse, distributivity on a sweep.
        for a in 0..256u16 {
            let a = Gf8(a as u8);
            assert_eq!(a + Gf8::ZERO, a);
            assert_eq!(a.mul(Gf8::ONE), a);
            assert_eq!(a.mul(Gf8::ZERO), Gf8::ZERO);
            if !a.is_zero() {
                assert_eq!(a.mul(a.inverse()), Gf8::ONE);
            }
        }
        for a in (0..256u16).step_by(7) {
            for b in (0..256u16).step_by(5) {
                let (a, b) = (Gf8(a as u8), Gf8(b as u8));
                assert_eq!(a.mul(b), b.mul(a)); // commutative
                // distributive: a·(b+c) = a·b + a·c, pick c = α.
                let c = Gf8::generator();
                assert_eq!(a.mul(b + c), a.mul(b) + a.mul(c));
            }
        }
    }

    #[test]
    fn mul_matches_carryless_reference() {
        // log/antilog mult must match schoolbook carryless mult mod m(X).
        let m_low = tables().m_low;
        let reference = |mut a: u8, mut b: u8| -> u8 {
            let mut acc = 0u8;
            for _ in 0..8 {
                if b & 1 == 1 {
                    acc ^= a;
                }
                b >>= 1;
                let hi = a & 0x80;
                a <<= 1;
                if hi != 0 {
                    a ^= m_low;
                }
            }
            acc
        };
        for a in 0..256u16 {
            for b in 0..256u16 {
                assert_eq!(
                    Gf8(a as u8).mul(Gf8(b as u8)).0,
                    reference(a as u8, b as u8),
                    "mul mismatch a={a} b={b}"
                );
            }
        }
    }

    #[test]
    fn embed_is_a_field_homomorphism() {
        // The decisive correctness gate: embed must respect + and ·, so the
        // GF(2^8)-accelerated NTT/products lift faithfully into GF(2^128).
        for a in 0..256u16 {
            let ea = embed(Gf8(a as u8));
            for b in 0..256u16 {
                let eb = embed(Gf8(b as u8));
                assert_eq!(embed(Gf8(a as u8) + Gf8(b as u8)), ea + eb, "add hom a={a} b={b}");
                assert_eq!(embed(Gf8(a as u8).mul(Gf8(b as u8))), ea * eb, "mul hom a={a} b={b}");
            }
        }
    }

    #[test]
    fn embed_fixes_zero_one_and_generator() {
        assert_eq!(embed(Gf8::ZERO), F128::zero());
        assert_eq!(embed(Gf8::ONE), F128::one());
        assert_eq!(embed(Gf8::generator()), theta());
        // The deterministic small-field skip challenges {α, α², α⁴} have an
        // F_2-independent tensor product (their 8 subset-products are the
        // standard basis), the condition the kernel's eq-basis trick needs.
        let chal = [Gf8::generator(), Gf8::generator().pow(2), Gf8::generator().pow(4)];
        // subset products = α^0..α^7, all distinct nonzero ⇒ independent.
        let mut prods = vec![];
        for mask in 0..8u32 {
            let mut p = Gf8::ONE;
            for (i, &c) in chal.iter().enumerate() {
                if (mask >> i) & 1 == 1 {
                    p = p.mul(c);
                }
            }
            prods.push(p.0);
        }
        prods.sort_unstable();
        prods.dedup();
        assert_eq!(prods.len(), 8, "skip-challenge subset products must be distinct");
    }
}
