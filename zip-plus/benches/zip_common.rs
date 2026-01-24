//! This module contains common benchmarking code for the Zip+ PCS,
//! both for Zip (integer coefficients) and Zip+ (polynomial coefficients).

#![allow(non_local_definitions)]
#![allow(clippy::eq_op, clippy::arithmetic_side_effects, clippy::unwrap_used)]

use criterion::{BenchmarkGroup, measurement::WallTime};
use crypto_bigint::U64;
use crypto_primitives::{
    DenseRowMatrix, Field, FromWithConfig, IntoWithConfig, PrimeField,
    crypto_bigint_monty::MontyField,
};
use itertools::Itertools;
use num_traits::One;
use rand::{distr::StandardUniform, prelude::*};
use std::{
    alloc::{GlobalAlloc, Layout, System},
    collections::BTreeMap,
    hint::black_box,
    panic,
    sync::{
        Mutex, OnceLock,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};
use zinc_poly::mle::{DenseMultilinearExtension, MultilinearExtensionRand};
use zinc_transcript::traits::ConstTranscribable;
use zinc_utils::{from_ref::FromRef, named::Named, projectable_to_field::ProjectableToField};
use zip_plus::{
    code::LinearCode,
    merkle::MerkleTree,
    pcs::structs::{ZipPlus, ZipPlusParams, ZipTypes},
};

const INT_LIMBS: usize = U64::LIMBS;
type F = MontyField<{ INT_LIMBS * 4 }>;
const FIXED_NUM_ROWS: usize = 1 << 3;
const FIXED_ROW_LEN: usize = 1 << 11;

struct TrackingAlloc;

static CURRENT_ALLOCATED: AtomicUsize = AtomicUsize::new(0);
static PEAK_ALLOCATED: AtomicUsize = AtomicUsize::new(0);
static PEAK_BY_LABEL: OnceLock<Mutex<BTreeMap<String, usize>>> = OnceLock::new();

#[global_allocator]
static GLOBAL_ALLOCATOR: TrackingAlloc = TrackingAlloc;

#[inline]
fn update_peak(current: usize) {
    let mut peak = PEAK_ALLOCATED.load(Ordering::Relaxed);
    while current > peak {
        match PEAK_ALLOCATED.compare_exchange_weak(
            peak,
            current,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => break,
            Err(observed) => peak = observed,
        }
    }
}

unsafe impl GlobalAlloc for TrackingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc(layout);
        if !ptr.is_null() {
            let size = layout.size();
            let current = CURRENT_ALLOCATED.fetch_add(size, Ordering::Relaxed) + size;
            update_peak(current);
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc_zeroed(layout);
        if !ptr.is_null() {
            let size = layout.size();
            let current = CURRENT_ALLOCATED.fetch_add(size, Ordering::Relaxed) + size;
            update_peak(current);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
        CURRENT_ALLOCATED.fetch_sub(layout.size(), Ordering::Relaxed);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_ptr = System.realloc(ptr, layout, new_size);
        if !new_ptr.is_null() {
            let old_size = layout.size();
            if new_size >= old_size {
                let delta = new_size - old_size;
                let current = CURRENT_ALLOCATED.fetch_add(delta, Ordering::Relaxed) + delta;
                update_peak(current);
            } else {
                let delta = old_size - new_size;
                CURRENT_ALLOCATED.fetch_sub(delta, Ordering::Relaxed);
            }
        }
        new_ptr
    }
}

#[inline]
fn reset_peak_alloc() {
    let current = CURRENT_ALLOCATED.load(Ordering::Relaxed);
    PEAK_ALLOCATED.store(current, Ordering::Relaxed);
}

#[inline]
fn peak_alloc_bytes() -> usize {
    PEAK_ALLOCATED.load(Ordering::Relaxed)
}

#[inline]
fn record_peak_alloc(label: &str) {
    let peak = peak_alloc_bytes();
    if peak > 0 {
        let map = PEAK_BY_LABEL.get_or_init(|| Mutex::new(BTreeMap::new()));
        if let Ok(mut guard) = map.lock() {
            let entry = guard.entry(label.to_string()).or_insert(0);
            if peak > *entry {
                *entry = peak;
            }
        }
    }
}

pub fn print_peak_alloc_summary(group_label: &str) {
    let map = PEAK_BY_LABEL.get_or_init(|| Mutex::new(BTreeMap::new()));
    let mut guard = match map.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };

    if guard.is_empty() {
        return;
    }

    eprintln!("PeakAlloc Summary: {group_label}");
    for (label, peak) in guard.iter() {
        let mib = *peak as f64 / (1024.0 * 1024.0);
        eprintln!("  PeakAlloc: {label}: {peak} bytes ({mib:.2} MiB)");
    }

    guard.clear();
}

fn params_with_fixed_rows<Zt: ZipTypes, Lc: LinearCode<Zt>>() -> (ZipPlusParams<Zt, Lc>, usize) {
    let poly_size = FIXED_NUM_ROWS * FIXED_ROW_LEN;
    if poly_size.is_power_of_two() {
        if let Ok(linear_code) = panic::catch_unwind(|| Lc::new(poly_size)) {
            let row_len = linear_code.row_len();
            if row_len == FIXED_ROW_LEN {
                let num_vars = poly_size.ilog2() as usize;
                let params = ZipPlusParams::new(num_vars, FIXED_NUM_ROWS, linear_code);
                return (params, num_vars);
            }
        }
    }

    // Fallback: find a consistent poly_size for the fixed number of rows.
    let mut poly_size = FIXED_NUM_ROWS;
    for _ in 0..20 {
        let linear_code = match panic::catch_unwind(|| Lc::new(poly_size)) {
            Ok(code) => code,
            Err(_) => {
                poly_size = poly_size.saturating_mul(2);
                continue;
            }
        };

        let row_len = linear_code.row_len();
        let target_poly_size = FIXED_NUM_ROWS * row_len;
        if target_poly_size == poly_size {
            let num_vars = poly_size.ilog2() as usize;
            let params = ZipPlusParams::new(num_vars, FIXED_NUM_ROWS, linear_code);
            return (params, num_vars);
        }

        poly_size = if target_poly_size.is_power_of_two() {
            target_poly_size
        } else {
            target_poly_size.next_power_of_two()
        };
    }

    panic!(
        "Failed to find a valid poly_size for fixed num_rows={} within bounds",
        FIXED_NUM_ROWS
    );
}

pub fn do_bench<Zt: ZipTypes, Lc: LinearCode<Zt>>(group: &mut BenchmarkGroup<WallTime>)
where
    StandardUniform: Distribution<Zt::Eval> + Distribution<Zt::Cw>,
    F: for<'a> FromWithConfig<&'a Zt::Chal> + for<'a> FromWithConfig<&'a Zt::Pt>,
    <F as Field>::Inner: FromRef<Zt::Fmod>,
    Zt::Eval: ProjectableToField<F>,
    Zt::Cw: ProjectableToField<F>,
{
    //encode_rows::<Zt, Lc>(group);
    encode_rows::<Zt, Lc>(group);

    // encode_single_row::<Zt, Lc, 128>(group);
    // encode_single_row::<Zt, Lc, 256>(group);
    // encode_single_row::<Zt, Lc, 512>(group);
    // encode_single_row::<Zt, Lc, 1024>(group);

    // merkle_root::<Zt, 12>(group);
    // merkle_root::<Zt, 13>(group);
    //merkle_root::<Zt, 14>(group);
    // merkle_root::<Zt, 15>(group);
    // merkle_root::<Zt, 16>(group);
    // commit::<Zt, Lc>(group);
    commit::<Zt, Lc>(group);
    commit_n::<Zt, Lc, 10>(group);

    // est::<Zt, Lc>(group);
    test::<Zt, Lc>(group);

    //verify::<Zt, Lc>(group);
    verify::<Zt, Lc>(group);
}

pub fn encode_rows<Zt: ZipTypes, Lc: LinearCode<Zt>>(
    group: &mut BenchmarkGroup<WallTime>,
) where
    StandardUniform: Distribution<Zt::Eval>,
{
    let label = format!(
        "EncodeRows: {} -> {}, num_rows=2^3, row_len={}",
        Zt::Eval::type_name(),
        Zt::Cw::type_name(),
        params_with_fixed_rows::<Zt, Lc>().0.linear_code.row_len()
    );

    group.bench_function(label.clone(), |b| {
        let mut rng = ThreadRng::default();
        let (params, num_vars) = params_with_fixed_rows::<Zt, Lc>();
        let row_len = params.linear_code.row_len();
        let poly =
            DenseMultilinearExtension::<<Zt as ZipTypes>::Eval>::rand(num_vars, &mut rng);
        reset_peak_alloc();
        b.iter(|| {
            let cw = ZipPlus::encode_rows(&params, row_len, &poly.evaluations);
            black_box(cw)
        });
        record_peak_alloc(&label);
    });
}

pub fn encode_single_row<Zt: ZipTypes, Lc: LinearCode<Zt>, const ROW_LEN: usize>(
    group: &mut BenchmarkGroup<WallTime>,
) where
    StandardUniform: Distribution<Zt::Eval>,
{
    let mut rng = ThreadRng::default();
    let poly_size = ROW_LEN * ROW_LEN;
    let linear_code = Lc::new(poly_size);
    if linear_code.row_len() != ROW_LEN {
        // TODO(Ilia): Since IPRS codes require
        //             the input size to be known at compile time
        //             this detects IPRS benches.
        //             Ofc, it's a lame way to handle this and
        //             one can come up with a more elegant type safe way
        //             but for the sake of a fast solution it's good enough.
        //             Once we have time address this pls.

        return;
    }

    let label = format!(
        "EncodeMessage: {} -> {}, row_len = {ROW_LEN}",
        Zt::Eval::type_name(),
        Zt::Cw::type_name()
    );

    group.bench_function(label.clone(), |b| {
        let message: Vec<<Zt as ZipTypes>::Eval> = (0..ROW_LEN).map(|_i| rng.random()).collect();
        reset_peak_alloc();
        b.iter(|| {
            let encoded_row: Vec<<Zt as ZipTypes>::Cw> = linear_code.encode(&message);
            black_box(encoded_row);
        });
        record_peak_alloc(&label);
    });
}

pub fn merkle_root<Zt: ZipTypes, const P: usize>(group: &mut BenchmarkGroup<WallTime>)
where
    StandardUniform: Distribution<Zt::Cw>,
{
    let mut rng = ThreadRng::default();

    let num_leaves = 1 << P;
    let leaves = (0..num_leaves)
        .map(|_| rng.random::<<Zt as ZipTypes>::Cw>())
        .collect_vec();
    let matrix: DenseRowMatrix<_> = vec![leaves.clone()].into();
    let rows = matrix.to_rows_slices();

    let label = format!("MerkleRoot: {}, leaves=2^{P}", Zt::Cw::type_name());
    group.bench_function(label.clone(), |b| {
        reset_peak_alloc();
        b.iter(|| {
            let tree = MerkleTree::new(&rows);
            black_box(tree.root());
        });
        record_peak_alloc(&label);
    });
}

pub fn commit<Zt: ZipTypes, Lc: LinearCode<Zt>>(
    group: &mut BenchmarkGroup<WallTime>,
) where
    StandardUniform: Distribution<Zt::Eval>,
{
    let mut rng = ThreadRng::default();
    let (params, num_vars) = params_with_fixed_rows::<Zt, Lc>();
    let row_len = params.linear_code.row_len();

    let label = format!(
        "Commit: Eval={}, Cw={}, Comb={}, num_rows=2^3, row_len={row_len}, poly_size=2^{num_vars}",
        Zt::Eval::type_name(),
        Zt::Cw::type_name(),
        Zt::Comb::type_name()
    );

    group.bench_function(label.clone(), |b| {
        b.iter_custom(|iters| {
            let mut total_duration = Duration::ZERO;
            reset_peak_alloc();
            for _ in 0..iters {
                let poly = DenseMultilinearExtension::rand(num_vars, &mut rng);
                let timer = Instant::now();
                let res = ZipPlus::commit(&params, &poly).expect("Failed to commit");
                black_box(res);
                total_duration += timer.elapsed();
            }

            record_peak_alloc(&label);
            total_duration
        })
    });
}

pub fn commit_n<Zt: ZipTypes, Lc: LinearCode<Zt>, const N: usize>(
    group: &mut BenchmarkGroup<WallTime>,
) where
    StandardUniform: Distribution<Zt::Eval>,
{
    let mut rng = ThreadRng::default();
    let (params, num_vars) = params_with_fixed_rows::<Zt, Lc>();
    let row_len = params.linear_code.row_len();

    let label = format!(
        "Commit x{N}: Eval={}, Cw={}, Comb={}, num_rows=2^3, row_len={row_len}, poly_size=2^{num_vars}",
        Zt::Eval::type_name(),
        Zt::Cw::type_name(),
        Zt::Comb::type_name()
    );

    group.bench_function(label.clone(), |b| {
        b.iter_custom(|iters| {
            let mut total_duration = Duration::ZERO;
            reset_peak_alloc();
            for _ in 0..iters {
                let polys: Vec<_> = (0..N)
                    .map(|_| DenseMultilinearExtension::rand(num_vars, &mut rng))
                    .collect();
                let timer = Instant::now();

                #[cfg(feature = "parallel")]
                {
                    use rayon::prelude::*;
                    polys.par_iter().for_each(|poly| {
                        let res = ZipPlus::commit(&params, poly).expect("Failed to commit");
                        black_box(res);
                    });
                }

                #[cfg(not(feature = "parallel"))]
                {
                    for poly in &polys {
                        let res = ZipPlus::commit(&params, poly).expect("Failed to commit");
                        black_box(res);
                    }
                }

                total_duration += timer.elapsed();
            }

            record_peak_alloc(&label);
            total_duration
        })
    });
}

pub fn test<Zt: ZipTypes, Lc: LinearCode<Zt>>(group: &mut BenchmarkGroup<WallTime>)
where
    StandardUniform: Distribution<Zt::Eval>,
{
    let mut rng = ThreadRng::default();
    let (params, num_vars) = params_with_fixed_rows::<Zt, Lc>();
    let row_len = params.linear_code.row_len();

    let poly = DenseMultilinearExtension::rand(num_vars, &mut rng);
    let (data, _) = ZipPlus::commit(&params, &poly).unwrap();

    let label = format!(
        "Test: Eval={}, Cw={}, Comb={}, num_rows=2^3, row_len={row_len}, poly_size=2^{num_vars}",
        Zt::Eval::type_name(),
        Zt::Cw::type_name(),
        Zt::Comb::type_name(),
    );

    group.bench_function(label.clone(), |b| {
        reset_peak_alloc();
        b.iter(|| {
            let test_transcript =
                ZipPlus::test(&params, &poly, &data).expect("Test phase failed");
            black_box(test_transcript);
        });
        record_peak_alloc(&label);
    });
}

pub fn evaluate<Zt: ZipTypes, Lc: LinearCode<Zt>>(
    group: &mut BenchmarkGroup<WallTime>,
) where
    StandardUniform: Distribution<Zt::Eval>,
    F: for<'a> FromWithConfig<&'a Zt::Chal> + for<'a> FromWithConfig<&'a Zt::Pt>,
    <F as Field>::Inner: FromRef<Zt::Fmod>,
    Zt::Eval: ProjectableToField<F>,
{
    let mut rng = ThreadRng::default();
    let (params, num_vars) = params_with_fixed_rows::<Zt, Lc>();
    let row_len = params.linear_code.row_len();

    let poly = DenseMultilinearExtension::rand(num_vars, &mut rng);
    let (data, _) = ZipPlus::commit(&params, &poly).unwrap();
    let point = vec![Zt::Pt::one(); num_vars];

    let test_transcript = ZipPlus::test(&params, &poly, &data).expect("Test phase failed");

    let label = format!(
        "Evaluate: Eval={}, Cw={}, Comb={}, num_rows=2^3, row_len={row_len}, poly_size=2^{num_vars}, modulus=({} bits)",
        Zt::Eval::type_name(),
        Zt::Cw::type_name(),
        Zt::Comb::type_name(),
        Zt::Fmod::NUM_BYTES * 8
    );

    group.bench_function(label.clone(), |b| {
        b.iter_custom(|iters| {
            let mut total_duration = Duration::ZERO;
            reset_peak_alloc();
            for _ in 0..iters {
                let proof = test_transcript.clone();
                let timer = Instant::now();
                let (eval_f, proof) = ZipPlus::evaluate::<F>(&params, &poly, &point, proof)
                    .expect("Evaluation phase failed");
                total_duration += timer.elapsed();
                black_box((eval_f, proof));
            }
            record_peak_alloc(&label);
            total_duration
        })
    });
}

pub fn verify<Zt: ZipTypes, Lc: LinearCode<Zt>>(
    group: &mut BenchmarkGroup<WallTime>,
) where
    StandardUniform: Distribution<Zt::Eval>,
    F: for<'a> FromWithConfig<&'a Zt::Chal> + for<'a> FromWithConfig<&'a Zt::Pt>,
    <F as Field>::Inner: FromRef<Zt::Fmod>,
    Zt::Eval: ProjectableToField<F>,
    Zt::Cw: ProjectableToField<F>,
{
    let mut rng = ThreadRng::default();
    let (params, num_vars) = params_with_fixed_rows::<Zt, Lc>();
    let row_len = params.linear_code.row_len();

    let poly = DenseMultilinearExtension::rand(num_vars, &mut rng);
    let (data, commitment) = ZipPlus::commit(&params, &poly).unwrap();
    let point = vec![Zt::Pt::one(); num_vars];

    let test_transcript = ZipPlus::test(&params, &poly, &data).expect("Test phase failed");
    let (eval_f, proof) = ZipPlus::evaluate::<F>(&params, &poly, &point, test_transcript)
        .expect("Evaluation phase failed");
    let field_cfg = *eval_f.cfg();
    let point_f: Vec<F> = point.iter().map(|v| v.into_with_cfg(&field_cfg)).collect();

    let label = format!(
        "Verify (skip eval): Eval={}, Cw={}, Comb={}, num_rows=2^3, row_len={row_len}, poly_size=2^{num_vars}, modulus=({} bits)",
        Zt::Eval::type_name(),
        Zt::Cw::type_name(),
        Zt::Comb::type_name(),
        Zt::Fmod::NUM_BYTES * 8
    );

    group.bench_function(label.clone(), |b| {
        reset_peak_alloc();
        b.iter(|| {
            ZipPlus::verify_skip_evaluation(&params, &commitment, &point_f, &proof)
                .expect("Verification failed");
        });
        record_peak_alloc(&label);
    });
}
