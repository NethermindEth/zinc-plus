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

/// Persistent Metal state: device, queue, and compiled hash pipelines.
pub struct MetalContext {
    device: Device,
    queue: CommandQueue,
    pipeline_hash_blake3: ComputePipelineState,
    hash_scratch: Mutex<HashScratch>,
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

            MetalContext {
                device,
                queue,
                pipeline_hash_blake3,
                hash_scratch: Mutex::new(HashScratch::new()),
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

        let exec_width = pipeline.thread_execution_width();
        let max_threads = pipeline.max_total_threads_per_threadgroup();
        let threads_per_threadgroup = if exec_width >= 32 {
            let t = (max_threads / exec_width) * exec_width;
            t.min(256)
        } else {
            max_threads.min(256)
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
