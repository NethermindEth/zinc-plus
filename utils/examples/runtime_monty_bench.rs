//! Apples-to-apples benchmark: the value-sized runtime-modulus field
//! [`Fp`] vs the current per-element-config element `MontyForm` (exactly what
//! `crypto_primitives::crypto_bigint_monty::MontyField` wraps — same layout,
//! same arithmetic).
//!
//! Measures, on the prover's hottest vector kernels, both:
//!   * wall-clock time, and
//!   * bytes allocated (via a counting global allocator) + element footprint.
//!
//! Run (release, from the worktree):
//!   cargo run --offline --release -p zinc-utils --example runtime_monty_bench [LOG2_N]
//!
//! Default LOG2_N = 22 (≈4.2M elements).
#![allow(
    clippy::arithmetic_side_effects,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::unwrap_used,
    clippy::needless_range_loop
)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use crypto_bigint::modular::{MontyForm, MontyParams};
use crypto_bigint::{Odd, U256, Uint};
use zinc_utils::define_modulus;
use zinc_utils::field::runtime_monty::{Fp, Modulus};

// ---- counting allocator: total bytes requested over the run ----
static ALLOCATED: AtomicUsize = AtomicUsize::new(0);
struct Counting;
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCATED.fetch_add(l.size(), Ordering::Relaxed);
        // SAFETY: forwarding to the system allocator with the same layout.
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        // SAFETY: forwarding to the system allocator with the same layout.
        unsafe { System.dealloc(p, l) }
    }
}
#[global_allocator]
static GLOBAL: Counting = Counting;

fn allocated() -> usize {
    ALLOCATED.load(Ordering::Relaxed)
}

// secp256k1 base field prime p = 2^256 - 2^32 - 977 (the main-beta projecting prime).
const SECP256K1_P_HEX: &str =
    "fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f";

define_modulus!(Secp, 4);

fn params() -> MontyParams<4> {
    MontyParams::new(Odd::new(U256::from_be_hex(SECP256K1_P_HEX)).into_option().unwrap())
}

// Cheap deterministic limb generator (xorshift); no rand dependency.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn uint(&mut self) -> U256 {
        Uint::from_words([self.next(), self.next(), self.next(), self.next()])
    }
}

/// Build the eq(x,r) multilinear table: out[k] = prod_i (r_i if bit_i(k) else 1-r_i).
/// In-place doubling — the dominant table builder in the sumcheck prover.
fn build_eq<T: Copy>(
    r: &[T],
    one: T,
    zero: T,
    mul: &impl Fn(T, T) -> T,
    sub: &impl Fn(T, T) -> T,
) -> Vec<T> {
    let n = r.len();
    let mut buf = vec![zero; 1usize << n];
    buf[0] = one;
    let mut sz = 1usize;
    for i in 0..n {
        let ri = r[i];
        for j in (0..sz).rev() {
            let hi = mul(buf[j], ri);
            buf[2 * j + 1] = hi;
            buf[2 * j] = sub(buf[j], hi); // buf[j]*(1-ri)
        }
        sz <<= 1;
    }
    buf
}

/// One sumcheck fold: out[j] = a[2j] + c*(a[2j+1]-a[2j]); repeat down to length 1.
fn fold_all<T: Copy>(
    mut a: Vec<T>,
    c: T,
    add: &impl Fn(T, T) -> T,
    sub: &impl Fn(T, T) -> T,
    mul: &impl Fn(T, T) -> T,
) -> T {
    while a.len() > 1 {
        let half = a.len() / 2;
        for j in 0..half {
            let lo = a[2 * j];
            let hi = a[2 * j + 1];
            a[j] = add(lo, mul(c, sub(hi, lo)));
        }
        a.truncate(half);
    }
    a[0]
}

fn hadamard<T: Copy>(a: &[T], b: &[T], mul: &impl Fn(T, T) -> T) -> Vec<T> {
    let mut out = Vec::with_capacity(a.len());
    for i in 0..a.len() {
        out.push(mul(a[i], b[i]));
    }
    out
}

fn dot<T: Copy>(a: &[T], b: &[T], zero: T, add: &impl Fn(T, T) -> T, mul: &impl Fn(T, T) -> T) -> T {
    let mut acc = zero;
    for i in 0..a.len() {
        acc = add(acc, mul(a[i], b[i]));
    }
    acc
}

fn ms(d: std::time::Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn mib(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

fn main() {
    let log2n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(22);
    let n = 1usize << log2n;

    let p = params();
    Secp::install_monty(p);

    // closures for MontyForm (params captured) and Fp (ambient)
    let mf_one = MontyForm::one(p);
    let mf_zero = MontyForm::new(&U256::ZERO, p);
    let mf_mul = |a: MontyForm<4>, b: MontyForm<4>| a * b;
    let mf_add = |a: MontyForm<4>, b: MontyForm<4>| a + b;
    let mf_sub = |a: MontyForm<4>, b: MontyForm<4>| a - b;

    type F = Fp<Secp, 4>;
    let fp_one = F::one();
    let fp_zero = F::zero();
    let fp_mul = |a: F, b: F| a * b;
    let fp_add = |a: F, b: F| a + b;
    let fp_sub = |a: F, b: F| a - b;

    println!("== runtime-modulus field: Fp vs MontyForm ==");
    println!("modulus      : secp256k1 base prime (256-bit, LIMBS=4)");
    println!("vector length: 2^{log2n} = {n} elements\n");

    println!(
        "element size : MontyForm = {} B   Fp = {} B   ({:.2}x smaller)",
        std::mem::size_of::<MontyForm<4>>(),
        std::mem::size_of::<F>(),
        std::mem::size_of::<MontyForm<4>>() as f64 / std::mem::size_of::<F>() as f64
    );

    // ---- memory: allocation to hold one length-n vector of each ----
    let a0 = allocated();
    let v_mf: Vec<MontyForm<4>> = vec![mf_zero; n];
    let mf_alloc = allocated() - a0;
    let a1 = allocated();
    let v_fp: Vec<F> = vec![fp_zero; n];
    let fp_alloc = allocated() - a1;
    println!(
        "vec alloc    : MontyForm = {:.1} MiB   Fp = {:.1} MiB   ({:.2}x smaller)\n",
        mib(mf_alloc),
        mib(fp_alloc),
        mf_alloc as f64 / fp_alloc as f64
    );
    drop(v_mf);
    drop(v_fp);

    // ---- inputs (same random limbs for both types) ----
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let raw: Vec<U256> = (0..n).map(|_| rng.uint()).collect();
    let raw_b: Vec<U256> = (0..n).map(|_| rng.uint()).collect();

    let a_mf: Vec<MontyForm<4>> = raw.iter().map(|u| MontyForm::new(u, p)).collect();
    let b_mf: Vec<MontyForm<4>> = raw_b.iter().map(|u| MontyForm::new(u, p)).collect();
    let a_fp: Vec<F> = raw.iter().map(|u| F::new(*u)).collect();
    let b_fp: Vec<F> = raw_b.iter().map(|u| F::new(*u)).collect();

    // small r vector for eq build
    let r_mf: Vec<MontyForm<4>> = (0..log2n).map(|i| a_mf[i]).collect();
    let r_fp: Vec<F> = (0..log2n).map(|i| a_fp[i]).collect();

    // best-of-K timing (min is the least-noisy estimator on a loaded machine)
    const K: usize = 7;
    fn best_ms(mut f: impl FnMut() -> std::time::Duration) -> f64 {
        let mut best = f64::INFINITY;
        for _ in 0..K {
            best = best.min(ms(f()));
        }
        best
    }
    let row = |name: &str, t_mf: f64, t_fp: f64| {
        println!(
            "{name:<22} MontyForm {t_mf:>8.1} ms   Fp {t_fp:>8.1} ms   ({:.2}x)",
            t_mf / t_fp
        );
    };

    println!("kernel timings — best of {K} (lower is better):");

    // eq table build
    let t_mf = best_ms(|| {
        let t = Instant::now();
        let v = build_eq(&r_mf, mf_one, mf_zero, &mf_mul, &mf_sub);
        let e = t.elapsed();
        std::hint::black_box(&v);
        e
    });
    let t_fp = best_ms(|| {
        let t = Instant::now();
        let v = build_eq(&r_fp, fp_one, fp_zero, &fp_mul, &fp_sub);
        let e = t.elapsed();
        std::hint::black_box(&v);
        e
    });
    row("build_eq(x,r)", t_mf, t_fp);

    // clone (memory bandwidth — where element size bites hardest)
    let t_mf = best_ms(|| {
        let t = Instant::now();
        let v = a_mf.clone();
        let e = t.elapsed();
        std::hint::black_box(&v);
        e
    });
    let t_fp = best_ms(|| {
        let t = Instant::now();
        let v = a_fp.clone();
        let e = t.elapsed();
        std::hint::black_box(&v);
        e
    });
    row("clone vec", t_mf, t_fp);

    // hadamard
    let t_mf = best_ms(|| {
        let t = Instant::now();
        let v = hadamard(&a_mf, &b_mf, &mf_mul);
        let e = t.elapsed();
        std::hint::black_box(&v);
        e
    });
    let t_fp = best_ms(|| {
        let t = Instant::now();
        let v = hadamard(&a_fp, &b_fp, &fp_mul);
        let e = t.elapsed();
        std::hint::black_box(&v);
        e
    });
    row("hadamard a*b", t_mf, t_fp);

    // dot product (compute-bound: smallest expected gap)
    let t_mf = best_ms(|| {
        let t = Instant::now();
        let v = dot(&a_mf, &b_mf, mf_zero, &mf_add, &mf_mul);
        let e = t.elapsed();
        std::hint::black_box(v);
        e
    });
    let t_fp = best_ms(|| {
        let t = Instant::now();
        let v = dot(&a_fp, &b_fp, fp_zero, &fp_add, &fp_mul);
        let e = t.elapsed();
        std::hint::black_box(v);
        e
    });
    row("dot product", t_mf, t_fp);
    // sanity: both representations must agree (canonical form)
    assert_eq!(
        dot(&a_mf, &b_mf, mf_zero, &mf_add, &mf_mul).retrieve(),
        dot(&a_fp, &b_fp, fp_zero, &fp_add, &fp_mul).retrieve(),
        "dot mismatch!"
    );

    // sumcheck fold (clone the prebuilt table outside the timer each round)
    let eq_mf = build_eq(&r_mf, mf_one, mf_zero, &mf_mul, &mf_sub);
    let eq_fp = build_eq(&r_fp, fp_one, fp_zero, &fp_mul, &fp_sub);
    let t_mf = best_ms(|| {
        let table = eq_mf.clone();
        let t = Instant::now();
        let v = fold_all(table, mf_one, &mf_add, &mf_sub, &mf_mul);
        let e = t.elapsed();
        std::hint::black_box(v);
        e
    });
    let t_fp = best_ms(|| {
        let table = eq_fp.clone();
        let t = Instant::now();
        let v = fold_all(table, fp_one, &fp_add, &fp_sub, &fp_mul);
        let e = t.elapsed();
        std::hint::black_box(v);
        e
    });
    row("sumcheck fold", t_mf, t_fp);

    println!("\n(ratios > 1.0 mean Fp is faster; memory ratios are exact/deterministic)");
}
