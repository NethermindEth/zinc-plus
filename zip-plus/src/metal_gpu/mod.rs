//! Metal GPU backend for the Blake3 Merkle leaf-hash kernel.
//!
//! Adapted from `albertgarreta/bolt-rs:src/metal_gpu/mod.rs` (the
//! `hash_columns_gpu` + Blake3 pipeline portion only). Other pipelines —
//! expander/RAA/FFT, SHA-256/Blake2/Keccak — are not needed by
//! zinc-plus's commit phase and are omitted.
//!
//! The kernel produces leaves byte-for-byte identical to
//! `blake3::Hasher::new().update(&buf).finalize()`, so its output can be
//! fed straight into the existing CPU tree builder
//! ([`crate::merkle::MerkleTree::new_from_leaves`] — see
//! [`crate::merkle::MerkleTree::new_from_row_major_grouped_gpu`]).

use metal::*;
use std::sync::{Mutex, OnceLock};

/// Selects the GPU leaf-hash kernel. Only Blake3 is wired in today —
/// adding SHA-256 / Blake2 / Keccak from bolt-rs would only require
/// pulling the matching `hash_leaves_*` kernel into
/// `shaders/hash_kernels.metal` and an extra pipeline slot here.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GpuHashKind {
    Blake3,
}

/// MSL source for the merkle-leaf hash kernels.
const HASH_KERNELS_MSL: &str = include_str!("shaders/hash_kernels.metal");
/// MSL source for the α-projection kernel (F_2 column → GF(2^128)).
const ALPHA_PROJECT_MSL: &str = include_str!("shaders/alpha_project.metal");
/// MSL source for the RAA F_2 encoder kernels (5 stages: repeat,
/// permute, prefix-XOR phase1/2/3). Adapted from `bolt-rs`'s
/// `expander_inner_product.metal` with row-major + u64 substitutions.
const RAA_F2_ENCODE_MSL: &str = include_str!("shaders/raa_f2_encode.metal");

/// Persistent scratch buffers for `hash_columns_gpu`. Reused across
/// calls so the bench loop doesn't pay a `new_buffer` cost per
/// invocation. Grow-on-demand; tracks current capacity in bytes.
struct HashScratch {
    out_buf: Option<Buffer>,
    out_cap: usize,
    params_buf: Option<Buffer>,
}

impl HashScratch {
    const fn new() -> Self {
        Self { out_buf: None, out_cap: 0, params_buf: None }
    }
}

/// Persistent scratch for `project_columns_at_alpha_gpu`. The α-powers
/// buffer is small (512 B at D=32) and reused identically across every
/// column in a prove call; the output buffer grows on demand and is
/// reused across columns.
struct AlphaProjectScratch {
    out_buf: Option<Buffer>,
    out_cap: usize,
    alpha_pows_buf: Option<Buffer>,
    alpha_pows_cap: usize,
    params_buf: Option<Buffer>,
}

impl AlphaProjectScratch {
    const fn new() -> Self {
        Self {
            out_buf: None,
            out_cap: 0,
            alpha_pows_buf: None,
            alpha_pows_cap: 0,
            params_buf: None,
        }
    }
}

/// Persistent scratch for `raa_f2_encode_matrix_gpu`. The two
/// permutation buffers (`perm_1_buf`, `perm_2_buf`) are uploaded once
/// per `RaaF2Code` lifetime via a (cheap) pointer-identity check on
/// the host-side permutation slabs — see `raa_f2_encode_matrix_gpu`
/// for the cache key. The `block_carries` scratch grows on demand and
/// is sized `n_rows * n_blocks * 8` bytes (small: ~16 KB for
/// `n_rows=8`, `n_blocks=2K`).
struct RaaEncodeScratch {
    perm_1_buf: Option<Buffer>,
    perm_1_cap: usize,
    perm_1_ident: usize, // pointer identity of the host slab last uploaded
    perm_2_buf: Option<Buffer>,
    perm_2_cap: usize,
    perm_2_ident: usize,
    block_carries_buf: Option<Buffer>,
    block_carries_cap: usize,
    /// Single shared params buffer; the MSL kernels all read the same
    /// `RaaParams` struct, so one buffer suffices across all 9
    /// dispatches in a single command-buffer build.
    params_buf: Option<Buffer>,
}

impl RaaEncodeScratch {
    const fn new() -> Self {
        Self {
            perm_1_buf: None,
            perm_1_cap: 0,
            perm_1_ident: 0,
            perm_2_buf: None,
            perm_2_cap: 0,
            perm_2_ident: 0,
            block_carries_buf: None,
            block_carries_cap: 0,
            params_buf: None,
        }
    }
}

/// Persistent Metal state: device, queue, and compiled hash pipelines.
pub struct MetalContext {
    device: Device,
    queue: CommandQueue,
    pipeline_hash_blake3: ComputePipelineState,
    pipeline_alpha_project: ComputePipelineState,
    pipeline_raa_repeat: ComputePipelineState,
    pipeline_raa_permute: ComputePipelineState,
    pipeline_raa_phase1: ComputePipelineState,
    pipeline_raa_phase2: ComputePipelineState,
    pipeline_raa_phase3: ComputePipelineState,
    hash_scratch: Mutex<HashScratch>,
    alpha_project_scratch: Mutex<AlphaProjectScratch>,
    raa_scratch: Mutex<RaaEncodeScratch>,
}

// SAFETY: `metal::*` handles are `Send + Sync` in practice on Apple
// platforms — the driver synchronises internally. `OnceLock<MetalContext>`
// below needs this to share the singleton across threads.
unsafe impl Send for MetalContext {}
unsafe impl Sync for MetalContext {}

static METAL_CONTEXT: OnceLock<MetalContext> = OnceLock::new();

impl MetalContext {
    /// Get or initialize the global MetalContext singleton. First call
    /// compiles the shader library, creates the Blake3 pipeline, and
    /// dispatches a warmup so the first real bench call doesn't pay
    /// driver-first-dispatch overhead.
    pub fn get() -> &'static MetalContext {
        let ctx = METAL_CONTEXT.get_or_init(|| {
            let device = Device::system_default().expect("No Metal device found");
            let queue = device.new_command_queue();

            let hash_library = device
                .new_library_with_source(HASH_KERNELS_MSL, &CompileOptions::new())
                .expect("Failed to compile hash kernels Metal shader");
            let kernel = hash_library
                .get_function("hash_leaves_blake3", None)
                .expect("hash_leaves_blake3 kernel not found");
            let pipeline_hash_blake3 = device
                .new_compute_pipeline_state_with_function(&kernel)
                .expect("Failed to create hash_leaves_blake3 pipeline");

            let alpha_project_library = device
                .new_library_with_source(ALPHA_PROJECT_MSL, &CompileOptions::new())
                .expect("Failed to compile alpha_project Metal shader");
            let alpha_project_kernel = alpha_project_library
                .get_function("project_columns_at_alpha", None)
                .expect("project_columns_at_alpha kernel not found");
            let pipeline_alpha_project = device
                .new_compute_pipeline_state_with_function(&alpha_project_kernel)
                .expect("Failed to create project_columns_at_alpha pipeline");

            // Compile the RAA encoder kernels into a single library and
            // pull out one pipeline per kernel function. Kept in one
            // library so the driver compiles them in one pass.
            let raa_library = device
                .new_library_with_source(RAA_F2_ENCODE_MSL, &CompileOptions::new())
                .expect("Failed to compile raa_f2_encode Metal shader");
            let make_raa_pipeline = |name: &str| -> ComputePipelineState {
                let f = raa_library
                    .get_function(name, None)
                    .unwrap_or_else(|_| panic!("RAA kernel `{name}` not found"));
                device
                    .new_compute_pipeline_state_with_function(&f)
                    .unwrap_or_else(|_| panic!("Failed to create pipeline for RAA kernel `{name}`"))
            };
            let pipeline_raa_repeat = make_raa_pipeline("raa_f2_repeat");
            let pipeline_raa_permute = make_raa_pipeline("raa_f2_permute");
            let pipeline_raa_phase1 = make_raa_pipeline("raa_f2_prefix_xor_phase1");
            let pipeline_raa_phase2 = make_raa_pipeline("raa_f2_prefix_xor_phase2");
            let pipeline_raa_phase3 = make_raa_pipeline("raa_f2_prefix_xor_phase3");

            MetalContext {
                device,
                queue,
                pipeline_hash_blake3,
                pipeline_alpha_project,
                pipeline_raa_repeat,
                pipeline_raa_permute,
                pipeline_raa_phase1,
                pipeline_raa_phase2,
                pipeline_raa_phase3,
                hash_scratch: Mutex::new(HashScratch::new()),
                alpha_project_scratch: Mutex::new(AlphaProjectScratch::new()),
                raa_scratch: Mutex::new(RaaEncodeScratch::new()),
            }
        });
        // Warm up the pipeline on first get() so the first real bench
        // call doesn't pay first-dispatch overhead.
        static WARMED: OnceLock<()> = OnceLock::new();
        WARMED.get_or_init(|| {
            let dummy = [0u8; 32];
            // SAFETY: dummy is alive for the duration of the call;
            // `hash_columns_gpu` blocks on completion before returning.
            let _ = unsafe { ctx.hash_columns_gpu(GpuHashKind::Blake3, dummy.as_ptr(), 1, 32) };
        });
        ctx
    }

    /// Dispatch a GPU hash over `num_cols` columns of `col_byte_len`
    /// bytes each, starting at the raw pointer `base`. Returns a flat
    /// `Vec<u8>` of `num_cols * 32` bytes — digest `i` at offset `i*32`.
    ///
    /// # Safety
    /// Caller must ensure `base` points to at least `num_cols *
    /// col_byte_len` readable bytes and that this range outlives the
    /// command-buffer execution (we block on completion before returning,
    /// so this is satisfied as long as `base` is valid at call time).
    ///
    /// Note: the Blake3 kernel handles arbitrary leaf sizes via a
    /// streaming subtree-stack inside each thread (see
    /// `hash_leaves_blake3` in `shaders/hash_kernels.metal`). The
    /// per-thread stack supports leaves up to ~16 GB; for larger
    /// inputs the CPU path
    /// ([`crate::merkle::MerkleTree::new_from_row_major_grouped`])
    /// remains available.
    pub unsafe fn hash_columns_gpu(
        &self,
        kind: GpuHashKind,
        base: *const u8,
        num_cols: usize,
        col_byte_len: usize,
    ) -> Vec<u8> {
        // 16 GB / leaf upper bound matches the kernel's 24-deep CV
        // stack (`2^24 × 1024 B`). Larger leaves would silently lose
        // data in the bubble-up merge; assert loudly instead.
        const MAX_LEAF_BYTES: usize = 16 * 1024 * 1024 * 1024;
        assert!(
            col_byte_len <= MAX_LEAF_BYTES,
            "GPU Blake3 kernel supports leaves up to {} bytes; got {col_byte_len}",
            MAX_LEAF_BYTES,
        );
        let pipeline = match kind {
            GpuHashKind::Blake3 => &self.pipeline_hash_blake3,
        };

        let in_bytes = num_cols
            .checked_mul(col_byte_len)
            .expect("num_cols*col_byte_len overflow");
        let out_bytes = num_cols * 32;

        #[repr(C)]
        #[derive(Clone, Copy)]
        struct HashParams {
            num_cols: u32,
            col_byte_len: u32,
        }
        let params = HashParams {
            num_cols: u32::try_from(num_cols).expect("num_cols > u32::MAX"),
            col_byte_len: u32::try_from(col_byte_len).expect("col_byte_len > u32::MAX"),
        };

        // Reuse cached output+params buffers. The MutexGuard is held
        // across the whole dispatch+wait so the borrows of `out_buf` /
        // `params_buf` stay valid.
        let mut scratch_guard = self.hash_scratch.lock().expect("hash scratch poisoned");
        if scratch_guard.out_buf.is_none() || scratch_guard.out_cap < out_bytes {
            scratch_guard.out_buf = Some(self.device.new_buffer(
                out_bytes as u64,
                MTLResourceOptions::StorageModeShared,
            ));
            scratch_guard.out_cap = out_bytes;
        }
        if scratch_guard.params_buf.is_none() {
            scratch_guard.params_buf = Some(self.device.new_buffer(
                std::mem::size_of::<HashParams>() as u64,
                MTLResourceOptions::StorageModeShared,
            ));
        }
        // Write params in place.
        // SAFETY: `params_buf` has `size_of::<HashParams>()` shared-storage bytes.
        unsafe {
            let pbuf = scratch_guard.params_buf.as_ref().unwrap();
            std::ptr::copy_nonoverlapping(
                &params as *const _ as *const u8,
                pbuf.contents() as *mut u8,
                std::mem::size_of::<HashParams>(),
            );
        }
        let out_buf = scratch_guard.out_buf.as_ref().unwrap();
        let params_buf = scratch_guard.params_buf.as_ref().unwrap();

        // Input: no-copy since the caller supplies the pointer.
        // The caller (per the `unsafe fn` contract) guarantees
        // `base..base+in_bytes` is readable and outlives the
        // wait_until_completed below.
        let in_buf = self.device.new_buffer_with_bytes_no_copy(
            base as *const std::ffi::c_void,
            in_bytes as u64,
            MTLResourceOptions::StorageModeShared,
            None,
        );

        // Dispatch.
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&in_buf), 0);
        encoder.set_buffer(1, Some(out_buf), 0);
        encoder.set_buffer(2, Some(params_buf), 0);

        // Removed the artificial `min(256)` cap. Let the pipeline's
        // own `max_total_threads_per_threadgroup` win (already
        // accounts for kernel resource use). The previous 256 cap
        // had been re-bench-validated on the artificial 480-row
        // workload (regressed +9% at nvars=16); revisiting on the
        // realistic shape (post-`6e40edd`) to see if the verdict
        // changes when the surrounding work mix is heavier and the
        // GPU bench's noise floor is lower relative to total Prove.
        let exec_width = pipeline.thread_execution_width();
        let max_threads = pipeline.max_total_threads_per_threadgroup();
        let threads_per_threadgroup = if exec_width >= 1 {
            (max_threads / exec_width) * exec_width
        } else {
            max_threads
        } as u64;
        let num_threadgroups =
            (num_cols as u64 + threads_per_threadgroup - 1) / threads_per_threadgroup;
        encoder.dispatch_thread_groups(
            MTLSize::new(num_threadgroups, 1, 1),
            MTLSize::new(threads_per_threadgroup, 1, 1),
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Copy digests out of the shared buffer.
        let mut out = vec![0u8; out_bytes];
        // SAFETY: `out_buf` was allocated with at least `out_bytes`, and
        // `out` has length `out_bytes` with no aliasing.
        unsafe {
            std::ptr::copy_nonoverlapping(
                out_buf.contents() as *const u8,
                out.as_mut_ptr(),
                out_bytes,
            );
        }
        out
    }

    /// Dispatch a GPU α-projection over a single F_2 column.
    ///
    /// Mirrors `project_column_with_powers` (CPU) byte-for-byte:
    /// for each `cell` in `cells_base..cells_base + num_cells`, computes
    /// `Σ_{i: cell_bit_i = 1} alpha_pows[i]` and writes 16 contiguous
    /// bytes into `out_base..out_base + num_cells * 16`. Output layout
    /// is `[u64; 2]` per cell, matching `BinaryFieldGF128::words()`.
    ///
    /// # Parameters
    /// - `cells_base`: pointer to the column's `[BinaryPoly<D>]` storage.
    ///   With the `simd` feature, `BinaryPoly<D> = #[repr(transparent)] u64`,
    ///   so this is byte-equivalent to `&[u64]` with one cell per `u64`.
    /// - `num_cells`: number of cells in the column (= `2^num_vars` for
    ///   the prove-path call site).
    /// - `alpha_pows_base`: pointer to `[BinaryFieldGF128]` of length
    ///   `>= d`. `BinaryFieldGF128 = #[repr(transparent)] [u64; 2]`.
    /// - `d`: number of low bits of each cell that carry data
    ///   (the `D` template arg of the CPU kernel). Must be `<= 64`.
    /// - `out_base`: pointer to `[BinaryFieldGF128]` of length
    ///   `>= num_cells`. Caller owns the allocation.
    ///
    /// Blocks until the dispatch completes before returning.
    ///
    /// # Safety
    /// All four pointers must be valid for the byte ranges above for the
    /// duration of the call; `out_base` must be writable. The caller
    /// must also ensure `cells_base..cells_base + num_cells*8` and
    /// `out_base..out_base + num_cells*16` do not alias each other or
    /// the α-powers slab.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn project_columns_at_alpha_gpu(
        &self,
        cells_base: *const u8,
        num_cells: usize,
        alpha_pows_base: *const u8,
        d: usize,
        out_base: *mut u8,
    ) {
        assert!(d <= 64, "alpha_project GPU kernel requires d <= 64; got {d}");
        if num_cells == 0 {
            return;
        }

        let cells_bytes = num_cells
            .checked_mul(std::mem::size_of::<u64>())
            .expect("num_cells * 8 overflow");
        let alpha_pows_bytes = d
            .checked_mul(2 * std::mem::size_of::<u64>())
            .expect("d * 16 overflow");
        let out_bytes = num_cells
            .checked_mul(2 * std::mem::size_of::<u64>())
            .expect("num_cells * 16 overflow");

        #[repr(C)]
        #[derive(Clone, Copy)]
        struct AlphaProjectParams {
            num_cells: u32,
            d: u32,
        }
        let params = AlphaProjectParams {
            num_cells: u32::try_from(num_cells).expect("num_cells > u32::MAX"),
            d: u32::try_from(d).expect("d > u32::MAX"),
        };

        let mut scratch = self
            .alpha_project_scratch
            .lock()
            .expect("alpha_project scratch poisoned");
        if scratch.out_buf.is_none() || scratch.out_cap < out_bytes {
            scratch.out_buf = Some(self.device.new_buffer(
                out_bytes as u64,
                MTLResourceOptions::StorageModeShared,
            ));
            scratch.out_cap = out_bytes;
        }
        if scratch.alpha_pows_buf.is_none() || scratch.alpha_pows_cap < alpha_pows_bytes {
            scratch.alpha_pows_buf = Some(self.device.new_buffer(
                alpha_pows_bytes as u64,
                MTLResourceOptions::StorageModeShared,
            ));
            scratch.alpha_pows_cap = alpha_pows_bytes;
        }
        if scratch.params_buf.is_none() {
            scratch.params_buf = Some(self.device.new_buffer(
                std::mem::size_of::<AlphaProjectParams>() as u64,
                MTLResourceOptions::StorageModeShared,
            ));
        }

        // Upload α-powers + params into the shared scratch buffers.
        // SAFETY: scratch buffers are at least the required sizes
        // (just sized above) and are shared-storage so writing through
        // `contents()` is the documented Metal pattern. The
        // alpha_pows_base..+alpha_pows_bytes range is readable per the
        // unsafe contract.
        unsafe {
            let ap_buf = scratch.alpha_pows_buf.as_ref().unwrap();
            std::ptr::copy_nonoverlapping(
                alpha_pows_base,
                ap_buf.contents() as *mut u8,
                alpha_pows_bytes,
            );
            let pbuf = scratch.params_buf.as_ref().unwrap();
            std::ptr::copy_nonoverlapping(
                &params as *const _ as *const u8,
                pbuf.contents() as *mut u8,
                std::mem::size_of::<AlphaProjectParams>(),
            );
        }

        let out_buf = scratch.out_buf.as_ref().unwrap();
        let alpha_pows_buf = scratch.alpha_pows_buf.as_ref().unwrap();
        let params_buf = scratch.params_buf.as_ref().unwrap();

        // Zero-copy input: the caller's `cells_base` slab is mapped
        // directly. SAFETY: cells_base..+cells_bytes is readable per
        // the unsafe contract and outlives the wait_until_completed
        // below.
        let cells_buf = self.device.new_buffer_with_bytes_no_copy(
            cells_base as *const std::ffi::c_void,
            cells_bytes as u64,
            MTLResourceOptions::StorageModeShared,
            None,
        );

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipeline_alpha_project);
        encoder.set_buffer(0, Some(&cells_buf), 0);
        encoder.set_buffer(1, Some(alpha_pows_buf), 0);
        encoder.set_buffer(2, Some(out_buf), 0);
        encoder.set_buffer(3, Some(params_buf), 0);

        let exec_width = self.pipeline_alpha_project.thread_execution_width();
        let max_threads = self
            .pipeline_alpha_project
            .max_total_threads_per_threadgroup();
        let threads_per_threadgroup = if exec_width >= 1 {
            (max_threads / exec_width) * exec_width
        } else {
            max_threads
        } as u64;
        let num_threadgroups =
            (num_cells as u64 + threads_per_threadgroup - 1) / threads_per_threadgroup;
        encoder.dispatch_thread_groups(
            MTLSize::new(num_threadgroups, 1, 1),
            MTLSize::new(threads_per_threadgroup, 1, 1),
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Copy results out of the shared output buffer. SAFETY:
        // out_buf has at least `out_bytes`, and the caller guarantees
        // out_base..+out_bytes is writable and non-aliasing.
        unsafe {
            std::ptr::copy_nonoverlapping(
                out_buf.contents() as *const u8,
                out_base,
                out_bytes,
            );
        }
    }

    /// Batched variant of [`Self::project_columns_at_alpha_gpu`]:
    /// dispatch one Metal compute encoder per column into a single
    /// command buffer, then `wait_until_completed` once.
    ///
    /// Why this exists: at nvars=22 with ~50 columns the prove path
    /// was issuing ~50 separate command-buffer submissions, each
    /// blocking on its own `wait_until_completed`. Each submission
    /// carries a fixed per-launch overhead (driver round-trip,
    /// GPU pipeline-state load, command-buffer commit/wait) that on
    /// the post-patch UAIR-FULL bench dominated total UAIR-FULL time
    /// (residual ~400 ms after the kernel itself dropped to ~milliseconds).
    /// Pushing all N dispatches into one command buffer amortises that
    /// round-trip across all columns.
    ///
    /// Output strategy: every output slab is mapped no-copy into a
    /// caller-owned allocation via `new_buffer_with_bytes_no_copy`.
    /// On Apple Silicon's shared-memory architecture this means the
    /// GPU writes land directly in the caller's `Vec<BinaryFieldGF128>`
    /// memory — no post-dispatch copy-out step, no scratch sized to
    /// `num_cols * num_cells * 16` bytes (~3.2 GB at nvars=22, 50 cols).
    ///
    /// # Safety
    /// For each `i`:
    ///   * `cells_bases[i]..cells_bases[i] + num_cells * 8` must be
    ///     readable for the duration of the call.
    ///   * `out_bases[i]..out_bases[i] + num_cells * 16` must be
    ///     writable for the duration of the call.
    ///   * No two output ranges may alias each other or the input
    ///     ranges or the α-powers slab.
    /// `alpha_pows_base..alpha_pows_base + d * 16` must be readable.
    /// All slabs must be distinct allocations (the wrapper handles this
    /// implicitly via `Vec`-owned outputs).
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn project_columns_at_alpha_gpu_batched(
        &self,
        cells_bases: &[*const u8],
        out_bases: &[*mut u8],
        num_cells: usize,
        alpha_pows_base: *const u8,
        d: usize,
    ) {
        assert!(d <= 64, "alpha_project GPU kernel requires d <= 64; got {d}");
        assert_eq!(
            cells_bases.len(),
            out_bases.len(),
            "cells_bases / out_bases length mismatch"
        );
        let num_cols = cells_bases.len();
        if num_cols == 0 || num_cells == 0 {
            return;
        }

        let cells_bytes = num_cells
            .checked_mul(std::mem::size_of::<u64>())
            .expect("num_cells * 8 overflow");
        let alpha_pows_bytes = d
            .checked_mul(2 * std::mem::size_of::<u64>())
            .expect("d * 16 overflow");
        let out_bytes = num_cells
            .checked_mul(2 * std::mem::size_of::<u64>())
            .expect("num_cells * 16 overflow");

        #[repr(C)]
        #[derive(Clone, Copy)]
        struct AlphaProjectParams {
            num_cells: u32,
            d: u32,
        }
        let params = AlphaProjectParams {
            num_cells: u32::try_from(num_cells).expect("num_cells > u32::MAX"),
            d: u32::try_from(d).expect("d > u32::MAX"),
        };

        // Upload α-powers + params into the shared scratch buffers
        // once for the whole batch.
        let mut scratch = self
            .alpha_project_scratch
            .lock()
            .expect("alpha_project scratch poisoned");
        if scratch.alpha_pows_buf.is_none() || scratch.alpha_pows_cap < alpha_pows_bytes {
            scratch.alpha_pows_buf = Some(self.device.new_buffer(
                alpha_pows_bytes as u64,
                MTLResourceOptions::StorageModeShared,
            ));
            scratch.alpha_pows_cap = alpha_pows_bytes;
        }
        if scratch.params_buf.is_none() {
            scratch.params_buf = Some(self.device.new_buffer(
                std::mem::size_of::<AlphaProjectParams>() as u64,
                MTLResourceOptions::StorageModeShared,
            ));
        }
        // SAFETY: scratch buffers just sized at least the required
        // sizes; alpha_pows_base..+alpha_pows_bytes readable per the
        // unsafe contract.
        unsafe {
            std::ptr::copy_nonoverlapping(
                alpha_pows_base,
                scratch.alpha_pows_buf.as_ref().unwrap().contents() as *mut u8,
                alpha_pows_bytes,
            );
            std::ptr::copy_nonoverlapping(
                &params as *const _ as *const u8,
                scratch.params_buf.as_ref().unwrap().contents() as *mut u8,
                std::mem::size_of::<AlphaProjectParams>(),
            );
        }
        let alpha_pows_buf = scratch.alpha_pows_buf.as_ref().unwrap();
        let params_buf = scratch.params_buf.as_ref().unwrap();

        let exec_width = self.pipeline_alpha_project.thread_execution_width();
        let max_threads = self
            .pipeline_alpha_project
            .max_total_threads_per_threadgroup();
        let threads_per_threadgroup = if exec_width >= 1 {
            (max_threads / exec_width) * exec_width
        } else {
            max_threads
        } as u64;
        let num_threadgroups =
            (num_cells as u64 + threads_per_threadgroup - 1) / threads_per_threadgroup;
        let tg_size = MTLSize::new(num_threadgroups, 1, 1);
        let local_size = MTLSize::new(threads_per_threadgroup, 1, 1);

        // Wrap each per-column input and output with a no-copy buffer.
        // These wrappers are cheap on Apple Silicon (shared memory) —
        // they're just driver-side handles around existing host memory.
        // We collect them into Vecs so the buffer objects (and their
        // implicit GPU residency) outlive the dispatch.
        let mut in_bufs: Vec<Buffer> = Vec::with_capacity(num_cols);
        let mut out_bufs: Vec<Buffer> = Vec::with_capacity(num_cols);
        for c in 0..num_cols {
            in_bufs.push(self.device.new_buffer_with_bytes_no_copy(
                cells_bases[c] as *const std::ffi::c_void,
                cells_bytes as u64,
                MTLResourceOptions::StorageModeShared,
                None,
            ));
            // Cast *mut u8 -> *const c_void for the Metal API. The
            // buffer is shared-storage and we will write into it from
            // the GPU; this is the standard pattern.
            out_bufs.push(self.device.new_buffer_with_bytes_no_copy(
                out_bases[c] as *const std::ffi::c_void,
                out_bytes as u64,
                MTLResourceOptions::StorageModeShared,
                None,
            ));
        }

        let command_buffer = self.queue.new_command_buffer();
        for c in 0..num_cols {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.pipeline_alpha_project);
            encoder.set_buffer(0, Some(&in_bufs[c]), 0);
            encoder.set_buffer(1, Some(alpha_pows_buf), 0);
            encoder.set_buffer(2, Some(&out_bufs[c]), 0);
            encoder.set_buffer(3, Some(params_buf), 0);
            encoder.dispatch_thread_groups(tg_size, local_size);
            encoder.end_encoding();
        }
        command_buffer.commit();
        command_buffer.wait_until_completed();
        // Outputs already written into caller-owned slabs (no-copy).
    }

    /// GPU RAA F_2 encoder.
    ///
    /// In-place encode of an `n_rows × codeword_len` matrix of u64
    /// cells through the RAA-F_2 pipeline (repeat → permute π₁ →
    /// prefix-XOR → permute π₂ → prefix-XOR), all in one command
    /// buffer with one `wait_until_completed`.
    ///
    /// CPU mirror: `RaaF2Code::encode_into_with_scratch` looped
    /// `n_rows` times. Both produce byte-identical codeword matrices.
    ///
    /// # Caller contract
    /// * `data_base` points to `n_rows * codeword_len * 8` bytes of
    ///   shared-storage-compatible memory. The first `msg_len` columns
    ///   of every row must already contain the input row to encode;
    ///   the remaining columns are written by the kernel.
    /// * `perm_1_base` / `perm_2_base` each point to `codeword_len * 4`
    ///   bytes of `u32` permutations (each element in `0..codeword_len`).
    ///   They are read by the kernel only and are uploaded to the GPU
    ///   exactly once per (RaaF2Code instance) lifetime — see the
    ///   pointer-identity cache below.
    /// * `block_size` is the prefix-XOR phase-1 strip width. Must
    ///   divide cleanly into `codeword_len` for best balance; a
    ///   trailing partial block is handled correctly by the kernel.
    ///
    /// On return `data_base..+n_rows*codeword_len*8` holds the encoded
    /// matrix.
    ///
    /// # Safety
    /// Caller guarantees:
    ///   * `data_base..+n_rows*codeword_len*8` is readable+writable
    ///     for the duration of the call.
    ///   * `perm_*_base..+codeword_len*4` are readable for the
    ///     duration of the call.
    ///   * `codeword_len == msg_len * q`, `msg_len > 0`, `q >= 2`,
    ///     `block_size >= 1`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn raa_f2_encode_matrix_gpu(
        &self,
        data_base: *mut u8,
        n_rows: usize,
        msg_len: usize,
        q: usize,
        perm_1_base: *const u32,
        perm_2_base: *const u32,
        block_size: usize,
    ) {
        assert!(msg_len > 0, "msg_len must be > 0");
        assert!(q >= 2, "q (REP) must be >= 2");
        assert!(block_size >= 1, "block_size must be >= 1");
        let codeword_len = msg_len
            .checked_mul(q)
            .expect("msg_len * q overflow");
        let n_blocks = codeword_len.div_ceil(block_size);

        let data_bytes = n_rows
            .checked_mul(codeword_len)
            .and_then(|x| x.checked_mul(std::mem::size_of::<u64>()))
            .expect("n_rows * codeword_len * 8 overflow");
        let perm_bytes = codeword_len
            .checked_mul(std::mem::size_of::<u32>())
            .expect("codeword_len * 4 overflow");
        let carries_bytes = n_rows
            .checked_mul(n_blocks)
            .and_then(|x| x.checked_mul(std::mem::size_of::<u64>()))
            .expect("n_rows * n_blocks * 8 overflow");

        // Shared params struct, mirrored in the MSL `RaaParams`.
        #[repr(C)]
        #[derive(Clone, Copy)]
        struct RaaParams {
            n_rows: u32,
            ncols: u32,
            msg_len: u32,
            q: u32,
            block_size: u32,
            n_blocks: u32,
        }
        let params = RaaParams {
            n_rows: u32::try_from(n_rows).expect("n_rows > u32::MAX"),
            ncols: u32::try_from(codeword_len).expect("codeword_len > u32::MAX"),
            msg_len: u32::try_from(msg_len).expect("msg_len > u32::MAX"),
            q: u32::try_from(q).expect("q > u32::MAX"),
            block_size: u32::try_from(block_size).expect("block_size > u32::MAX"),
            n_blocks: u32::try_from(n_blocks).expect("n_blocks > u32::MAX"),
        };

        let mut scratch = self
            .raa_scratch
            .lock()
            .expect("raa_scratch poisoned");

        // ---- Permutation buffers (cached via pointer identity) ----
        //
        // `perm_1_base` and `perm_2_base` are stable for the lifetime
        // of a `RaaF2Code` instance (its `perm_1` / `perm_2` Vecs are
        // built once at code construction). We cache the uploaded GPU
        // buffer by the host pointer's identity (`ptr as usize`) — on
        // a cache hit (the same code's permutations as last call) we
        // skip the upload entirely.
        let perm_1_ident = perm_1_base as usize;
        if scratch.perm_1_buf.is_none()
            || scratch.perm_1_cap < perm_bytes
            || scratch.perm_1_ident != perm_1_ident
        {
            if scratch.perm_1_cap < perm_bytes {
                scratch.perm_1_buf = Some(self.device.new_buffer(
                    perm_bytes as u64,
                    MTLResourceOptions::StorageModeShared,
                ));
                scratch.perm_1_cap = perm_bytes;
            }
            // SAFETY: buffer sized to `perm_bytes`; caller guarantees
            // `perm_1_base..+perm_bytes` readable.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    perm_1_base as *const u8,
                    scratch.perm_1_buf.as_ref().unwrap().contents() as *mut u8,
                    perm_bytes,
                );
            }
            scratch.perm_1_ident = perm_1_ident;
        }
        let perm_2_ident = perm_2_base as usize;
        if scratch.perm_2_buf.is_none()
            || scratch.perm_2_cap < perm_bytes
            || scratch.perm_2_ident != perm_2_ident
        {
            if scratch.perm_2_cap < perm_bytes {
                scratch.perm_2_buf = Some(self.device.new_buffer(
                    perm_bytes as u64,
                    MTLResourceOptions::StorageModeShared,
                ));
                scratch.perm_2_cap = perm_bytes;
            }
            // SAFETY: as above.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    perm_2_base as *const u8,
                    scratch.perm_2_buf.as_ref().unwrap().contents() as *mut u8,
                    perm_bytes,
                );
            }
            scratch.perm_2_ident = perm_2_ident;
        }

        // ---- block_carries buffer ----
        if scratch.block_carries_buf.is_none() || scratch.block_carries_cap < carries_bytes {
            scratch.block_carries_buf = Some(self.device.new_buffer(
                carries_bytes as u64,
                MTLResourceOptions::StorageModeShared,
            ));
            scratch.block_carries_cap = carries_bytes;
        }

        // ---- Params buffer (reused, written per-call) ----
        let params_bytes = std::mem::size_of::<RaaParams>();
        if scratch.params_buf.is_none() {
            scratch.params_buf = Some(self.device.new_buffer(
                params_bytes as u64,
                MTLResourceOptions::StorageModeShared,
            ));
        }
        // SAFETY: params buffer is exactly `params_bytes` of shared
        // storage; the params struct has no padding holes via repr(C).
        unsafe {
            std::ptr::copy_nonoverlapping(
                &params as *const _ as *const u8,
                scratch.params_buf.as_ref().unwrap().contents() as *mut u8,
                params_bytes,
            );
        }

        let perm_1_buf = scratch.perm_1_buf.as_ref().unwrap();
        let perm_2_buf = scratch.perm_2_buf.as_ref().unwrap();
        let carries_buf = scratch.block_carries_buf.as_ref().unwrap();
        let params_buf = scratch.params_buf.as_ref().unwrap();

        // ---- Input data buffer (no-copy, caller-owned) ----
        // SAFETY: caller guarantees `data_base..+data_bytes` is
        // readable+writable for the duration of the call.
        let data_buf = self.device.new_buffer_with_bytes_no_copy(
            data_base as *const std::ffi::c_void,
            data_bytes as u64,
            MTLResourceOptions::StorageModeShared,
            None,
        );

        // ---- Temp buffer for the permute outputs ----
        //
        // RAA needs a second matrix-sized buffer to ping-pong through
        // the two permutations. We allocate it as a fresh shared
        // buffer per call; it's `data_bytes` (e.g. 128 MB at SHA F_2
        // nvars=22). A scratch slot in `RaaEncodeScratch` would let
        // us cache it across calls, at the cost of holding 128 MB
        // even when not commiting. Leaving uncached for now — the
        // allocation cost on Apple Silicon shared memory is just
        // a page-table operation, not an actual copy.
        let temp_buf = self.device.new_buffer(
            data_bytes as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // ---- Build the 9-encoder command buffer ----
        let command_buffer = self.queue.new_command_buffer();

        // Helper: dispatch a kernel with the given pipeline and
        // total-thread count. Picks threadgroup width from the
        // pipeline's hardware preference.
        let dispatch = |pipeline: &ComputePipelineState,
                        encoder: &ComputeCommandEncoderRef,
                        total_threads: usize| {
            let exec_width = pipeline.thread_execution_width();
            let max_threads = pipeline.max_total_threads_per_threadgroup();
            let tg = if exec_width >= 1 {
                (max_threads / exec_width) * exec_width
            } else {
                max_threads
            } as u64;
            let n_groups = (total_threads as u64 + tg - 1) / tg;
            encoder.dispatch_thread_groups(
                MTLSize::new(n_groups, 1, 1),
                MTLSize::new(tg, 1, 1),
            );
        };

        // (1) repeat: data_buf → data_buf (in-place fill of cols msg_len..ncols)
        {
            let enc = command_buffer.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.pipeline_raa_repeat);
            enc.set_buffer(0, Some(&data_buf), 0);
            enc.set_buffer(1, Some(params_buf), 0);
            // Threads: n_rows * msg_len * (q - 1)
            let total = n_rows
                .checked_mul(msg_len)
                .and_then(|x| x.checked_mul(q - 1))
                .expect("repeat thread count overflow");
            dispatch(&self.pipeline_raa_repeat, enc, total);
            enc.end_encoding();
        }
        // (2) permute π₁: data_buf → temp_buf
        {
            let enc = command_buffer.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.pipeline_raa_permute);
            enc.set_buffer(0, Some(&data_buf), 0);
            enc.set_buffer(1, Some(&temp_buf), 0);
            enc.set_buffer(2, Some(perm_1_buf), 0);
            enc.set_buffer(3, Some(params_buf), 0);
            let total = n_rows
                .checked_mul(codeword_len)
                .expect("permute thread count overflow");
            dispatch(&self.pipeline_raa_permute, enc, total);
            enc.end_encoding();
        }
        // (3) prefix-XOR phase 1 on temp_buf
        {
            let enc = command_buffer.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.pipeline_raa_phase1);
            enc.set_buffer(0, Some(&temp_buf), 0);
            enc.set_buffer(1, Some(carries_buf), 0);
            enc.set_buffer(2, Some(params_buf), 0);
            let total = n_rows
                .checked_mul(n_blocks)
                .expect("phase1 thread count overflow");
            dispatch(&self.pipeline_raa_phase1, enc, total);
            enc.end_encoding();
        }
        // (4) prefix-XOR phase 2
        {
            let enc = command_buffer.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.pipeline_raa_phase2);
            enc.set_buffer(0, Some(carries_buf), 0);
            enc.set_buffer(1, Some(params_buf), 0);
            dispatch(&self.pipeline_raa_phase2, enc, n_rows);
            enc.end_encoding();
        }
        // (5) prefix-XOR phase 3 on temp_buf
        {
            let enc = command_buffer.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.pipeline_raa_phase3);
            enc.set_buffer(0, Some(&temp_buf), 0);
            enc.set_buffer(1, Some(carries_buf), 0);
            enc.set_buffer(2, Some(params_buf), 0);
            let total = n_rows
                .checked_mul(codeword_len)
                .expect("phase3 thread count overflow");
            dispatch(&self.pipeline_raa_phase3, enc, total);
            enc.end_encoding();
        }
        // (6) permute π₂: temp_buf → data_buf
        {
            let enc = command_buffer.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.pipeline_raa_permute);
            enc.set_buffer(0, Some(&temp_buf), 0);
            enc.set_buffer(1, Some(&data_buf), 0);
            enc.set_buffer(2, Some(perm_2_buf), 0);
            enc.set_buffer(3, Some(params_buf), 0);
            let total = n_rows
                .checked_mul(codeword_len)
                .expect("permute thread count overflow");
            dispatch(&self.pipeline_raa_permute, enc, total);
            enc.end_encoding();
        }
        // (7) prefix-XOR phase 1 on data_buf
        {
            let enc = command_buffer.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.pipeline_raa_phase1);
            enc.set_buffer(0, Some(&data_buf), 0);
            enc.set_buffer(1, Some(carries_buf), 0);
            enc.set_buffer(2, Some(params_buf), 0);
            let total = n_rows
                .checked_mul(n_blocks)
                .expect("phase1 thread count overflow");
            dispatch(&self.pipeline_raa_phase1, enc, total);
            enc.end_encoding();
        }
        // (8) prefix-XOR phase 2
        {
            let enc = command_buffer.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.pipeline_raa_phase2);
            enc.set_buffer(0, Some(carries_buf), 0);
            enc.set_buffer(1, Some(params_buf), 0);
            dispatch(&self.pipeline_raa_phase2, enc, n_rows);
            enc.end_encoding();
        }
        // (9) prefix-XOR phase 3 on data_buf
        {
            let enc = command_buffer.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.pipeline_raa_phase3);
            enc.set_buffer(0, Some(&data_buf), 0);
            enc.set_buffer(1, Some(carries_buf), 0);
            enc.set_buffer(2, Some(params_buf), 0);
            let total = n_rows
                .checked_mul(codeword_len)
                .expect("phase3 thread count overflow");
            dispatch(&self.pipeline_raa_phase3, enc, total);
            enc.end_encoding();
        }

        command_buffer.commit();
        command_buffer.wait_until_completed();
        // `data_buf` (mapped no-copy onto caller's `data_base`) now
        // holds the final encoded matrix. `temp_buf` is dropped here.
    }
}

/// Typed wrapper around [`MetalContext::project_columns_at_alpha_gpu`].
///
/// CPU mirror: `zinc_poly::univariate::binary_gf128::project_column_with_powers`.
/// Allocates and returns a fresh `Vec<BinaryFieldGF128>` of length
/// `cells.len()` containing the per-cell evaluation `Σ_{i: bit_i(cell) = 1} α^i`.
///
/// Lives in `zip-plus` rather than `poly` because the GPU dispatcher
/// is a zip-plus concern — putting the wrapper here avoids a reverse
/// `poly -> zip-plus` dependency.
///
/// Layout assumptions (validated at runtime by the asserts):
///   * `BinaryPoly<D>` is `#[repr(transparent)] u64` (requires the
///     `simd` feature in `zinc-poly`). Without `simd`, `BinaryPoly`
///     resolves to `BinaryRefPoly`, which has a different layout —
///     this function would then be unsafe to call. The runtime
///     `size_of::<BinaryPoly<D>>() == 8` assert guards against this.
///   * `BinaryFieldGF128` is `#[repr(transparent)]` over a 16-byte
///     payload (two `u64` words), validated by `size_of`.
///
/// # Panics
/// - If the runtime layout assumptions above don't hold.
/// - If `alpha_powers.len() < D`.
pub fn project_column_with_powers_gpu<const D: usize>(
    cells: &[zinc_poly::univariate::binary::BinaryPoly<D>],
    alpha_powers: &[zinc_poly::univariate::binary_gf128::BinaryFieldGF128],
) -> Vec<zinc_poly::univariate::binary_gf128::BinaryFieldGF128> {
    use zinc_poly::univariate::binary::BinaryPoly;
    use zinc_poly::univariate::binary_gf128::BinaryFieldGF128;

    // Layout guards. These are static facts under `--features simd`,
    // but we check at runtime so a non-`simd` build that wires the GPU
    // path in fails loudly instead of corrupting memory.
    assert_eq!(
        std::mem::size_of::<BinaryPoly<D>>(),
        std::mem::size_of::<u64>(),
        "GPU alpha-project requires `simd` feature (BinaryPoly<D> must be repr(transparent) u64)"
    );
    assert_eq!(
        std::mem::size_of::<BinaryFieldGF128>(),
        2 * std::mem::size_of::<u64>(),
        "GPU alpha-project requires BinaryFieldGF128 == [u64; 2] layout"
    );
    assert!(
        alpha_powers.len() >= D,
        "project_column_with_powers_gpu: alpha_powers ({}) shorter than D ({D})",
        alpha_powers.len(),
    );
    assert!(D <= 64, "project_column_with_powers_gpu: D ({D}) > 64");

    let num_cells = cells.len();
    if num_cells == 0 {
        return Vec::new();
    }

    let mut out: Vec<BinaryFieldGF128> = Vec::with_capacity(num_cells);
    let ctx = MetalContext::get();
    // SAFETY:
    //   * `cells.as_ptr()` is valid for `num_cells * 8` bytes (size
    //     of BinaryPoly<D> asserted == 8 above).
    //   * `alpha_powers.as_ptr()` is valid for `D * 16` bytes
    //     (size of BinaryFieldGF128 asserted == 16, len >= D asserted).
    //   * `out.as_mut_ptr()` is valid for `num_cells * 16` bytes
    //     (just reserved that capacity).
    //   * All three pointers are distinct allocations.
    //   * Dispatcher blocks until completion before returning.
    unsafe {
        ctx.project_columns_at_alpha_gpu(
            cells.as_ptr() as *const u8,
            num_cells,
            alpha_powers.as_ptr() as *const u8,
            D,
            out.as_mut_ptr() as *mut u8,
        );
        out.set_len(num_cells);
    }
    out
}

/// Batched typed wrapper: project N F_2 columns at α in a single GPU
/// dispatch. Counterpart of [`project_column_with_powers_gpu`] for the
/// prove-path's per-column loop.
///
/// Returns one `Vec<BinaryFieldGF128>` per input column, each of length
/// `columns[i].len()`. All input columns must have the same length;
/// asserts otherwise.
///
/// # Panics
/// - If layout assumptions in `project_column_with_powers_gpu` don't hold.
/// - If `alpha_powers.len() < D`.
/// - If input columns have differing lengths.
pub fn project_columns_with_powers_gpu_batched<const D: usize>(
    columns: &[&[zinc_poly::univariate::binary::BinaryPoly<D>]],
    alpha_powers: &[zinc_poly::univariate::binary_gf128::BinaryFieldGF128],
) -> Vec<Vec<zinc_poly::univariate::binary_gf128::BinaryFieldGF128>> {
    use zinc_poly::univariate::binary::BinaryPoly;
    use zinc_poly::univariate::binary_gf128::BinaryFieldGF128;

    assert_eq!(
        std::mem::size_of::<BinaryPoly<D>>(),
        std::mem::size_of::<u64>(),
        "batched GPU alpha-project requires `simd` feature (BinaryPoly<D> must be repr(transparent) u64)"
    );
    assert_eq!(
        std::mem::size_of::<BinaryFieldGF128>(),
        2 * std::mem::size_of::<u64>(),
        "batched GPU alpha-project requires BinaryFieldGF128 == [u64; 2] layout"
    );
    assert!(
        alpha_powers.len() >= D,
        "project_columns_with_powers_gpu_batched: alpha_powers ({}) shorter than D ({D})",
        alpha_powers.len(),
    );
    assert!(D <= 64, "project_columns_with_powers_gpu_batched: D ({D}) > 64");

    let num_cols = columns.len();
    if num_cols == 0 {
        return Vec::new();
    }
    let num_cells = columns[0].len();
    for (i, col) in columns.iter().enumerate() {
        assert_eq!(
            col.len(),
            num_cells,
            "column {i} has {} cells, expected {num_cells}",
            col.len()
        );
    }
    if num_cells == 0 {
        return vec![Vec::new(); num_cols];
    }

    // Pre-allocate output Vecs. The GPU writes directly into their
    // spare capacity via no-copy buffer mapping; we `set_len` after
    // the dispatch returns.
    let mut outputs: Vec<Vec<BinaryFieldGF128>> = (0..num_cols)
        .map(|_| Vec::with_capacity(num_cells))
        .collect();
    let cells_bases: Vec<*const u8> = columns
        .iter()
        .map(|c| c.as_ptr() as *const u8)
        .collect();
    let out_bases: Vec<*mut u8> = outputs
        .iter_mut()
        .map(|v| v.as_mut_ptr() as *mut u8)
        .collect();

    let ctx = MetalContext::get();
    // SAFETY:
    //   * For each i, columns[i].as_ptr()..+num_cells*8 is readable
    //     (size_of::<BinaryPoly<D>>() asserted == 8 above).
    //   * For each i, outputs[i].as_mut_ptr()..+num_cells*16 is
    //     writable (just reserved that capacity per Vec).
    //   * outputs[i] Vecs are distinct allocations.
    //   * Dispatcher blocks on completion before returning, so the
    //     `Vec` backing memory is live for the whole GPU operation.
    unsafe {
        ctx.project_columns_at_alpha_gpu_batched(
            &cells_bases,
            &out_bases,
            num_cells,
            alpha_powers.as_ptr() as *const u8,
            D,
        );
        for out in outputs.iter_mut() {
            out.set_len(num_cells);
        }
    }
    outputs
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Cross-check: GPU Blake3 leaf hash matches `blake3::hash`
    /// byte-for-byte across a few representative sizes.
    #[test]
    fn gpu_blake3_matches_cpu_for_various_sizes() {
        let ctx = MetalContext::get();
        // Sizes hit both the single-chunk (≤ 1024 B) and multi-chunk
        // (> 1024 B) branches of the kernel.
        for &col_bytes in &[1usize, 63, 64, 65, 1023, 1024, 1025, 4096, 10_240, 16_384] {
            let num_cols = 32;
            let mut data = vec![0u8; num_cols * col_bytes];
            for (i, b) in data.iter_mut().enumerate() {
                *b = (i % 251) as u8; // arbitrary non-trivial pattern
            }
            let gpu = unsafe {
                ctx.hash_columns_gpu(GpuHashKind::Blake3, data.as_ptr(), num_cols, col_bytes)
            };
            assert_eq!(gpu.len(), num_cols * 32);
            for i in 0..num_cols {
                let leaf = &data[i * col_bytes..(i + 1) * col_bytes];
                let expected: [u8; 32] = blake3::hash(leaf).into();
                assert_eq!(
                    &gpu[i * 32..(i + 1) * 32],
                    &expected[..],
                    "mismatch at leaf {i} of {num_cols}, col_bytes={col_bytes}"
                );
            }
        }
    }

    /// Cross-check: GPU α-projection matches CPU `project_column_with_powers`
    /// byte-for-byte across mixed cell sizes and bit patterns.
    ///
    /// Only meaningful under `--features simd` (which makes
    /// `BinaryPoly<D> = BinaryU64Poly<D>`, the layout the kernel
    /// assumes). When `simd` is off, `BinaryPoly` resolves to
    /// `BinaryRefPoly` and the GPU wrapper's `size_of` assert
    /// fires — so we gate the test on `simd`.
    #[cfg(feature = "simd")]
    #[test]
    fn gpu_alpha_project_matches_cpu() {
        use rand::SeedableRng;
        use rand::Rng;
        use rand::rngs::StdRng;
        use zinc_poly::univariate::binary::BinaryPoly;
        use zinc_poly::univariate::binary_gf128::{
            BinaryFieldGF128, alpha_powers, project_column_with_powers,
        };

        const D: usize = 32;
        let mut rng = StdRng::seed_from_u64(0xa11ce_face_b00c_u64);

        // Cell-count sweep: hits length-1, exec-width boundary
        // (small power-of-two), a non-power-of-two (residual handling),
        // and a larger size that crosses several threadgroups.
        for &num_cells in &[1usize, 7, 32, 33, 1024, 1025, 16_384] {
            // Random α and its precomputed powers (same call the CPU
            // prove path makes at f2_prove.rs:866).
            let alpha = {
                let lo: u64 = rng.random();
                let hi: u64 = rng.random();
                BinaryFieldGF128::from_words([lo, hi])
            };
            let pows = alpha_powers(&alpha, D);

            // Random cells. `BinaryPoly<32>` from `u32` populates the
            // low 32 bits; matches the prove-path's storage shape.
            let cells: Vec<BinaryPoly<D>> = (0..num_cells)
                .map(|_| {
                    let bits: u32 = rng.random();
                    BinaryPoly::<D>::from(bits)
                })
                .collect();

            let cpu = project_column_with_powers::<D>(&cells, &pows);
            let gpu = project_column_with_powers_gpu::<D>(&cells, &pows);

            assert_eq!(cpu.len(), gpu.len(), "len mismatch (n={num_cells})");
            for (i, (c, g)) in cpu.iter().zip(gpu.iter()).enumerate() {
                assert_eq!(
                    c.words(),
                    g.words(),
                    "cell {i} of {num_cells}: cpu={:?} gpu={:?}",
                    c.words(),
                    g.words(),
                );
            }
        }
    }

    /// Batched-dispatch parity: `project_columns_with_powers_gpu_batched`
    /// must match per-column `project_column_with_powers` (CPU)
    /// byte-for-byte across each column.
    #[cfg(feature = "simd")]
    #[test]
    fn gpu_alpha_project_batched_matches_cpu() {
        use rand::SeedableRng;
        use rand::Rng;
        use rand::rngs::StdRng;
        use zinc_poly::univariate::binary::BinaryPoly;
        use zinc_poly::univariate::binary_gf128::{
            BinaryFieldGF128, alpha_powers, project_column_with_powers,
        };

        const D: usize = 32;
        let mut rng = StdRng::seed_from_u64(0xdead_beef_cafe_babe_u64);

        // (num_cols, num_cells_per_col): include a 1-col case,
        // power-of-two col counts, a non-power-of-two, and a SHA-256-
        // ish ratio (40 cols × small num_cells) that exercises the
        // many-encoder path without needing prove-scale memory.
        for &(num_cols, num_cells) in &[
            (1usize, 16usize),
            (2, 1024),
            (8, 33),
            (40, 1024),
            (3, 16_384),
        ] {
            let alpha = {
                let lo: u64 = rng.random();
                let hi: u64 = rng.random();
                BinaryFieldGF128::from_words([lo, hi])
            };
            let pows = alpha_powers(&alpha, D);

            // num_cols column slabs.
            let column_storage: Vec<Vec<BinaryPoly<D>>> = (0..num_cols)
                .map(|_| {
                    (0..num_cells)
                        .map(|_| {
                            let bits: u32 = rng.random();
                            BinaryPoly::<D>::from(bits)
                        })
                        .collect()
                })
                .collect();
            let cpu: Vec<Vec<BinaryFieldGF128>> = column_storage
                .iter()
                .map(|col| project_column_with_powers::<D>(col, &pows))
                .collect();

            let columns_view: Vec<&[BinaryPoly<D>]> =
                column_storage.iter().map(|v| v.as_slice()).collect();
            let gpu = project_columns_with_powers_gpu_batched::<D>(&columns_view, &pows);

            assert_eq!(cpu.len(), gpu.len(), "num_cols mismatch");
            for (ci, (c_col, g_col)) in cpu.iter().zip(gpu.iter()).enumerate() {
                assert_eq!(
                    c_col.len(),
                    g_col.len(),
                    "col {ci} len mismatch (n_cols={num_cols}, n_cells={num_cells})"
                );
                for (i, (c, g)) in c_col.iter().zip(g_col.iter()).enumerate() {
                    assert_eq!(
                        c.words(),
                        g.words(),
                        "col {ci} cell {i} (n_cols={num_cols}, n_cells={num_cells}): cpu={:?} gpu={:?}",
                        c.words(),
                        g.words(),
                    );
                }
            }
        }
    }

    /// Cross-check large-leaf paths (> 64 KB), exercising the streaming
    /// subtree-stack kernel. Sizes here are picked to (a) cross the old
    /// 64 KB hard cap, (b) hit boundary cases for chunk counts that
    /// merge into balanced (power-of-two) vs unbalanced subtrees, and
    /// (c) include a multi-MB leaf representative of SHA-256 F_2's
    /// shape (~14 MB / leaf at nvars=20).
    #[test]
    fn gpu_blake3_matches_cpu_for_large_leaves() {
        let ctx = MetalContext::get();
        // Sizes (bytes): 64 KB, 65 KB (over old cap), 256 KB, ~1 MB,
        // 14 MB (~SHA-256 F_2 nvars=20 leaf size, but rounded).
        // num_cols kept small (4) because each leaf is multi-MB; a 32-
        // col 14 MB-each test would allocate ~450 MB.
        for &col_bytes in &[
            65 * 1024usize,
            64 * 1024 + 1,
            256 * 1024,
            1_048_576,
            4 * 1024 * 1024,
        ] {
            let num_cols = 4;
            let mut data = vec![0u8; num_cols * col_bytes];
            for (i, b) in data.iter_mut().enumerate() {
                *b = ((i * 1103515245).wrapping_add(12345) >> 16) as u8;
            }
            let gpu = unsafe {
                ctx.hash_columns_gpu(GpuHashKind::Blake3, data.as_ptr(), num_cols, col_bytes)
            };
            assert_eq!(gpu.len(), num_cols * 32);
            for i in 0..num_cols {
                let leaf = &data[i * col_bytes..(i + 1) * col_bytes];
                let expected: [u8; 32] = blake3::hash(leaf).into();
                assert_eq!(
                    &gpu[i * 32..(i + 1) * 32],
                    &expected[..],
                    "large-leaf mismatch at leaf {i} of {num_cols}, col_bytes={col_bytes}"
                );
            }
        }
    }
}
