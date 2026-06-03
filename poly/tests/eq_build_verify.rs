//! Correctness check for the in-place `eq(x, r)` table builder
//! (`build_eq_x_r_vec` / `build_eq_x_r` / `build_eq_x_r_inner`).
//!
//! Lives as an integration test (separate target) so it builds against the
//! `zinc-poly` library and is unaffected by unrelated unit-test breakage in
//! the crate's `--lib` test target.
//!
//! The check is independent of the implementation: it compares the produced
//! table against the brute-force definition
//! `eval[\sum_i x_i 2^i] = \prod_i (x_i ? r_i : 1 - r_i)`, which also pins
//! down the exact index/bit ordering the rest of the codebase relies on.

#![allow(
    clippy::arithmetic_side_effects,
    clippy::cast_possible_truncation,
    clippy::cast_lossless,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]

use crypto_primitives::{
    IntoWithConfig, PrimeField, crypto_bigint_monty::MontyField, crypto_bigint_uint::Uint,
};
use zinc_poly::utils::{build_eq_x_r, build_eq_x_r_inner, build_eq_x_r_vec};

const LIMBS: usize = 4;
type F = MontyField<LIMBS>;
const MODULUS: &str = "0076F668F4274572E39A3EA8285319B5";

fn cfg() -> <F as PrimeField>::Config {
    let modulus = Uint::new(
        crypto_bigint::Uint::from_str_radix_vartime(MODULUS, 16).expect("modulus hex"),
    );
    MontyField::make_cfg(&modulus).expect("field config")
}

/// `eval[idx] = prod_i (bit_i(idx) ? r_i : 1 - r_i)` — the definition.
fn brute_force(r: &[F], cfg: &<F as PrimeField>::Config) -> Vec<F> {
    let one = F::one_with_cfg(cfg);
    let n = 1usize << r.len();
    (0..n)
        .map(|idx| {
            let mut acc = one.clone();
            for (i, ri) in r.iter().enumerate() {
                let factor = if (idx >> i) & 1 == 1 {
                    ri.clone()
                } else {
                    one.clone() - ri
                };
                acc = acc * &factor;
            }
            acc
        })
        .collect()
}

/// Deterministic pseudo-random field elements (SplitMix64), so the test is
/// reproducible without pulling an RNG into the assertions.
fn sample_r(seed: u64, len: usize, cfg: &<F as PrimeField>::Config) -> Vec<F> {
    let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..len)
        .map(|_| {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            let v = (z ^ (z >> 31)) as u128;
            v.into_with_cfg(cfg)
        })
        .collect()
}

#[test]
fn vec_matches_brute_force_over_lengths_and_seeds() {
    let cfg = cfg();
    for len in 1usize..=12 {
        for seed in 0u64..6 {
            let r = sample_r(seed.wrapping_mul(7).wrapping_add(len as u64), len, &cfg);
            let got = build_eq_x_r_vec(&r, &cfg).expect("build_eq_x_r_vec");
            let want = brute_force(&r, &cfg);
            assert_eq!(got.len(), 1usize << len, "len={len} seed={seed}");
            assert_eq!(got, want, "mismatch at len={len} seed={seed}");
        }
    }
}

#[test]
fn mle_form_matches_vec_form() {
    let cfg = cfg();
    for len in 1usize..=8 {
        let r = sample_r(1000 + len as u64, len, &cfg);
        let mle = build_eq_x_r(&r, &cfg).expect("build_eq_x_r");
        let vec = build_eq_x_r_vec(&r, &cfg).expect("build_eq_x_r_vec");
        assert_eq!(mle.num_vars, len);
        assert_eq!(mle.evaluations, vec, "mle vs vec mismatch at len={len}");
    }
}

#[test]
fn inner_form_matches_vec_form() {
    let cfg = cfg();
    for len in 1usize..=8 {
        let r = sample_r(2000 + len as u64, len, &cfg);
        let inner = build_eq_x_r_inner(&r, &cfg).expect("build_eq_x_r_inner");
        let vec = build_eq_x_r_vec(&r, &cfg).expect("build_eq_x_r_vec");
        assert_eq!(inner.num_vars, len);
        let want_inner: Vec<Uint<LIMBS>> = vec.into_iter().map(|f| f.into_inner()).collect();
        assert_eq!(
            inner.evaluations, want_inner,
            "inner vs vec mismatch at len={len}"
        );
    }
}

#[test]
fn empty_r_is_an_error() {
    let cfg = cfg();
    let r: [F; 0] = [];
    let err = build_eq_x_r_vec(&r, &cfg).unwrap_err();
    assert!(format!("{err}").contains("Invalid parameters"));
}
