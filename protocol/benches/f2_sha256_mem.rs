//! Peak-RAM measurement harness for the `F_2[X]` SHA-256 prover.
//!
//! This is **not** a criterion benchmark (`harness = false`, own `main`). It
//! runs the real deployed prove once per process and reports memory, not time:
//!
//!   1. A process-level **peak RSS** via `getrusage(RUSAGE_SELF)` (`ru_maxrss`).
//!   2. A **tracking global allocator** that records *live* and *peak* heap
//!      bytes (the prover's own `Vec` working set, independent of OS RSS
//!      lazy-reclaim / fragmentation).
//!   3. A faithful **per-region** memory map of the real prove path: a peak-RSS
//!      probe is registered with `zinc_utils::prof`, so every `prof::scope`
//!      region (`commit`, `uair`, `uair:sumcheck`, `merged_w_block`,
//!      `multipoint_eval`, `open`, …) is annotated with the process high-water
//!      growth it caused. Printed when `OBLONG_PROFILE=1` is set.
//!
//! `ru_maxrss` is monotonic across a process, so run **one `nvars` per process**
//! for a clean peak. Driver:
//!
//! ```sh
//! cargo bench -p zinc-protocol --bench f2_sha256_mem \
//!     --features parallel,simd,unchecked --no-run
//! BIN=$(ls -t target/release/deps/f2_sha256_mem-* | grep -v '\.d$' | head -1)
//! for nv in 9 12 14 16 18; do OBLONG_PROFILE=1 "$BIN" "$nv"; done
//! ```
//!
//! Args: `<nvars> [mode]`, mode ∈ `merged` (default, the current deployed sound
//! discharge), `oblong`, `linear` (no discharge — shows the discharge's
//! marginal RAM).
//!
//! **Required features**: `parallel`, `simd`, `unchecked` (gated in `Cargo.toml`).

#![allow(
    clippy::arithmetic_side_effects,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicI64, Ordering};
use std::time::Instant;

use crypto_bigint::U64;
use crypto_primitives::{crypto_bigint_int::Int, crypto_bigint_uint::Uint};
use rand::rng;
use zinc_poly::univariate::{
    binary::{BinaryPoly, BinaryPolyInnerProduct},
    dense::{DensePolyInnerProduct, DensePolynomial},
};
use zinc_primality::MillerRabin;
use zinc_protocol::f2_prove::{
    F2BitOpVirtualSpec, F2VirtualBpSpec, F2ZincTypes, ZincPlusPiopF2,
};
use zinc_test_uair::{GenerateRandomTrace, Sha256F2Uair, sha256_f2_project_scalar};
use zinc_transcript::Blake3Transcript;
use zinc_uair::BitOp;
use zinc_utils::inner_product::MBSInnerProduct;
use zip_plus::{
    code::raa::RaaConfig,
    code::raa_f2::{RaaF2Code, recommended_num_column_openings},
    pcs::structs::{ZipPlusParams, ZipTypes},
};

// ---------------------------------------------------------------------------
// Tracking global allocator: live + peak heap bytes.
// ---------------------------------------------------------------------------

static HEAP_LIVE: AtomicI64 = AtomicI64::new(0);
static HEAP_PEAK: AtomicI64 = AtomicI64::new(0);

struct TrackingAlloc;

#[inline]
fn bump(delta: i64) {
    let now = HEAP_LIVE.fetch_add(delta, Ordering::Relaxed) + delta;
    HEAP_PEAK.fetch_max(now, Ordering::Relaxed);
}

// SAFETY: every method delegates to the System allocator and only updates two
// relaxed atomic counters around it; no aliasing or layout assumptions change.
unsafe impl GlobalAlloc for TrackingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            bump(layout.size() as i64);
        }
        p
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc_zeroed(layout) };
        if !p.is_null() {
            bump(layout.size() as i64);
        }
        p
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        HEAP_LIVE.fetch_sub(layout.size() as i64, Ordering::Relaxed);
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let p = unsafe { System.realloc(ptr, layout, new_size) };
        if !p.is_null() {
            bump(new_size as i64 - layout.size() as i64);
        }
        p
    }
}

#[global_allocator]
static GLOBAL: TrackingAlloc = TrackingAlloc;

fn heap_live() -> u64 {
    HEAP_LIVE.load(Ordering::Relaxed).max(0) as u64
}
fn heap_peak() -> u64 {
    HEAP_PEAK.load(Ordering::Relaxed).max(0) as u64
}
fn reset_heap_peak() {
    HEAP_PEAK.store(HEAP_LIVE.load(Ordering::Relaxed), Ordering::Relaxed);
}

/// Process peak RSS in bytes (`getrusage(RUSAGE_SELF).ru_maxrss`). macOS reports
/// bytes; Linux/BSD report KiB. Also the probe handed to `zinc_utils::prof`.
fn peak_rss_bytes() -> u64 {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    // SAFETY: getrusage writes a fully-initialized rusage on success (rc == 0).
    let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    if rc != 0 {
        return 0;
    }
    let usage = unsafe { usage.assume_init() };
    #[cfg(target_os = "macos")]
    {
        usage.ru_maxrss as u64
    }
    #[cfg(not(target_os = "macos"))]
    {
        (usage.ru_maxrss as u64).saturating_mul(1024)
    }
}

fn fmt(bytes: u64) -> String {
    let mib = bytes as f64 / (1024.0 * 1024.0);
    format!("{mib:>9.1} MiB ({:>6.3} GiB)", mib / 1024.0)
}

// ---------------------------------------------------------------------------
// F_2 type plumbing — copied verbatim from `benches/f2_sha256.rs` so this
// harness exercises the identical commit code, RAA-1/4 code, and opening count.
// ---------------------------------------------------------------------------

const REP: usize = 4;
const D: usize = 32;
/// Soundness-calibrated RAA column-opening count at REP=4 (= 987).
const BENCH_NUM_OPENINGS: usize = 987;

type R = Int<4>;
type U = Sha256F2Uair<R>;

#[derive(Debug, Clone)]
struct BenchF2ZipTypes<const DD: usize> {}

impl<const DD: usize> ZipTypes for BenchF2ZipTypes<DD> {
    const NUM_COLUMN_OPENINGS: usize = recommended_num_column_openings(REP);
    type Eval = BinaryPoly<DD>;
    type Cw = BinaryPoly<DD>;
    type Fmod = Uint<{ U64::LIMBS * 4 }>;
    type PrimeTest = MillerRabin;
    type Chal = i128;
    type Pt = i128;
    type CombR = Int<{ U64::LIMBS * 8 }>;
    type Comb = DensePolynomial<Self::CombR, DD>;
    type EvalDotChal = BinaryPolyInnerProduct<Self::Chal, DD>;
    type CombDotChal =
        DensePolyInnerProduct<Self::CombR, Self::Chal, Self::CombR, MBSInnerProduct, DD>;
    type ArrCombRDotChal = MBSInnerProduct;
}

#[derive(Copy, Clone)]
struct BenchRaaConfig;
impl RaaConfig for BenchRaaConfig {
    const PERMUTE_IN_PLACE: bool = false;
    const CHECK_FOR_OVERFLOWS: bool = false;
}

#[derive(Clone, Debug)]
struct BenchF2Types<const DD: usize>(PhantomData<()>);

impl<const DD: usize> F2ZincTypes<DD> for BenchF2Types<DD> {
    type BinaryZt = BenchF2ZipTypes<64>;
    type BinaryLc = RaaF2Code<Self::BinaryZt, BenchRaaConfig, REP>;
}

/// The pure-SHR virtual columns (commit-side skip). The merged discharge passes
/// `&[]` here (it commits the 2 SHR cols); oblong/linear virtualise them.
fn sha_f2_bit_op_virtuals() -> Vec<F2BitOpVirtualSpec> {
    use zinc_test_uair::sha256_f2::cols;
    vec![
        F2BitOpVirtualSpec {
            col_idx: cols::W_SHR3_W,
            source_col_idx: cols::W_W,
            op: BitOp::ShiftR(3),
        },
        F2BitOpVirtualSpec {
            col_idx: cols::W_SHR10_W,
            source_col_idx: cols::W_W,
            op: BitOp::ShiftR(10),
        },
    ]
}

/// SHA-256 Hadamard relation layout (2 ANDs + 12 adders). Mirror of the bench /
/// `f2_prove.rs` test helper.
fn sha256_f2_hadamard_layout(num_vars: usize) -> zinc_protocol::f2_hadamard::Sha256F2HadamardLayout {
    use zinc_test_uair::sha256_f2::cols;
    zinc_protocol::f2_hadamard::Sha256F2HadamardLayout {
        num_vars,
        rows_per_comp: cols::ROWS_PER_COMP,
        rounds_per_comp: cols::ROUNDS_PER_COMP,
        num_compressions: cols::num_compressions(num_vars),
        pa_a: cols::PA_A,
        pa_e: cols::PA_E,
        pa_k: cols::PA_K,
        w_a: cols::W_A,
        w_e: cols::W_E,
        w_w: cols::W_W,
        w_sigma0: cols::W_SIGMA0,
        w_sigma1: cols::W_SIGMA1,
        w_sig0: cols::W_SIG0,
        w_sig1: cols::W_SIG1,
        w_uch: cols::W_UCH,
        w_maj: cols::W_MAJ,
        w_t1: cols::W_T1,
        w_t2: cols::W_T2,
        w_w_s1: cols::W_W_S1,
        w_w_s2: cols::W_W_S2,
        w_t1_s1: cols::W_T1_S1,
        w_t1_s2: cols::W_T1_S2,
        w_t1_s3: cols::W_T1_S3,
    }
}

type Trace = zinc_uair::UairTrace<'static, BinaryPoly<D>, BinaryPoly<D>, D>;
type Pp = ZipPlusParams<
    <BenchF2Types<D> as F2ZincTypes<D>>::BinaryZt,
    <BenchF2Types<D> as F2ZincTypes<D>>::BinaryLc,
>;

/// Inputs the prover needs entering: a random SHA trace + the public params.
/// Lighter than the bench's `setup_prover` (no pre-paired witness — the merged
/// prove re-pairs internally), so the "setup" baseline reflects only the inputs.
fn setup_light(num_vars: usize) -> (Trace, Pp) {
    let mut rng_local = rng();
    let num_rows: usize = 8;
    let row_len = (1usize << num_vars) / num_rows;
    let trace = U::generate_random_trace(num_vars, &mut rng_local);
    let lc = <BenchF2Types<D> as F2ZincTypes<D>>::BinaryLc::new(row_len);
    let pp = ZipPlusParams::new(num_vars, num_rows, lc);
    (trace, pp)
}

fn main() {
    // Register the peak-RSS probe so `prof::scope` regions get a memory column
    // (only printed when OBLONG_PROFILE=1; harmless otherwise).
    zinc_utils::prof::set_rss_probe(peak_rss_bytes);

    let args: Vec<String> = std::env::args().collect();
    let num_vars: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(16);
    let mode: &str = args.get(2).map(String::as_str).unwrap_or("merged");

    use zinc_test_uair::sha256_f2::cols;
    let rows = 1usize << num_vars;
    let comps = cols::num_compressions(num_vars);

    eprintln!(
        "\n=== F_2[X] SHA-256 prover RAM | nvars={num_vars} (2^{num_vars} = {rows} rows, \
         {comps} compressions) | mode={mode} | openings={BENCH_NUM_OPENINGS} ==="
    );

    let (trace, pp) = setup_light(num_vars);
    let (and_specs, adder_specs) = sha256_f2_hadamard_layout(num_vars).relations();
    let bit_ops = sha_f2_bit_op_virtuals();

    let rss_setup = peak_rss_bytes();
    let heap_setup = heap_live();
    eprintln!("  setup (trace + pp inputs):");
    eprintln!("    RSS peak so far : {}", fmt(rss_setup));
    eprintln!("    heap live       : {}", fmt(heap_setup));

    reset_heap_peak();
    let t0 = Instant::now();

    // Run the selected prove once. Keep the proof alive across the measurement.
    let proof_bytes_alive: usize;
    match mode {
        "linear" => {
            let mut t = Blake3Transcript::new();
            let proof = ZincPlusPiopF2::<BenchF2Types<D>, U, D>::prove_f2_full_with_bit_ops(
                &mut t,
                &pp,
                &trace,
                &[] as &[F2VirtualBpSpec],
                &bit_ops,
                num_vars,
                sha256_f2_project_scalar::<R>,
                BENCH_NUM_OPENINGS,
            )
            .expect("linear prove");
            proof_bytes_alive = std::mem::size_of_val(&proof);
            black_box(&proof);
            report(num_vars, mode, t0.elapsed(), rss_setup, heap_setup);
            black_box(proof);
        }
        "oblong" => {
            let mut t = Blake3Transcript::new();
            let proof =
                ZincPlusPiopF2::<BenchF2Types<D>, U, D>::prove_f2_full_with_oblong_hadamard(
                    &mut t,
                    &pp,
                    &trace,
                    &[] as &[F2VirtualBpSpec],
                    &bit_ops,
                    &and_specs,
                    &adder_specs,
                    num_vars,
                    sha256_f2_project_scalar::<R>,
                    BENCH_NUM_OPENINGS,
                )
                .expect("oblong prove");
            proof_bytes_alive = std::mem::size_of_val(&proof);
            black_box(&proof);
            report(num_vars, mode, t0.elapsed(), rss_setup, heap_setup);
            black_box(proof);
        }
        _ => {
            // merged: the current deployed sound discharge. `&[]` bit-ops (it
            // commits the 2 SHR cols) — matches `Prove-Hadamard-Merged`.
            let mut t = Blake3Transcript::new();
            let proof =
                ZincPlusPiopF2::<BenchF2Types<D>, U, D>::prove_f2_full_with_merged_hadamard(
                    &mut t,
                    &pp,
                    &trace,
                    &[] as &[F2VirtualBpSpec],
                    &[] as &[F2BitOpVirtualSpec],
                    &and_specs,
                    &adder_specs,
                    num_vars,
                    sha256_f2_project_scalar::<R>,
                    BENCH_NUM_OPENINGS,
                )
                .expect("merged prove");
            proof_bytes_alive = std::mem::size_of_val(&proof);
            black_box(&proof);
            report(num_vars, mode, t0.elapsed(), rss_setup, heap_setup);
            black_box(proof);
        }
    }
    black_box(proof_bytes_alive);

    // Per-region memory tree (only when OBLONG_PROFILE=1).
    zinc_utils::prof::dump_and_reset(&format!("{mode} prove, nvars={num_vars}"));
}

/// Print the peak figures, measured while the proof is still alive.
fn report(num_vars: usize, mode: &str, dt: std::time::Duration, rss_setup: u64, heap_setup: u64) {
    let rss_peak = peak_rss_bytes();
    let heap_live_now = heap_live();
    let heap_peak_prove = heap_peak();
    let _ = (num_vars, mode);
    eprintln!("  full prove (wall {dt:.2?}):");
    eprintln!("    RSS peak        : {}", fmt(rss_peak));
    eprintln!(
        "    heap peak       : {}   (transient high-water during prove)",
        fmt(heap_peak_prove)
    );
    eprintln!(
        "    heap live (proof retained) : {}",
        fmt(heap_live_now)
    );
    eprintln!(
        "  → prove transient above setup : RSS +{}   heap +{}",
        fmt(rss_peak.saturating_sub(rss_setup)),
        fmt(heap_peak_prove.saturating_sub(heap_setup))
    );
}
