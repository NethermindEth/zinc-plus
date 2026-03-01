//! Peak and current memory measurement utilities for benchmarks.
//!
//! Provides cross-platform helpers to query resident set size (RSS)
//! so benchmarks can report peak memory consumption alongside timing.
//!
//! # Supported platforms
//!
//! | Platform | Current RSS | Peak RSS |
//! |----------|-------------|----------|
//! | macOS    | `mach_task_info` | `getrusage` (`ru_maxrss`, bytes) |
//! | Linux    | `/proc/self/status` (`VmRSS`) | `/proc/self/status` (`VmPeak`) |
//! | Other    | returns `None` | returns `None` |
//!
//! # Example
//!
//! ```rust,ignore
//! use zinc_utils::peak_mem::MemoryTracker;
//!
//! let tracker = MemoryTracker::start();
//! // ... expensive computation ...
//! let snapshot = tracker.stop();
//! eprintln!("{snapshot}");
//! // prints: Memory: before=120.5 MB, after=245.3 MB, delta=+124.8 MB, process peak=245.3 MB
//! ```

use std::fmt;

/// Returns the current resident set size (RSS) of this process, in bytes.
///
/// Returns `None` on unsupported platforms.
pub fn current_rss() -> Option<usize> {
    platform::current_rss()
}

/// Returns the peak resident set size of this process, in bytes.
///
/// On macOS this is the all-time peak (`ru_maxrss`).
/// On Linux this comes from `VmPeak` in `/proc/self/status`.
///
/// Returns `None` on unsupported platforms.
pub fn peak_rss() -> Option<usize> {
    platform::peak_rss()
}

// ─── MemoryTracker ──────────────────────────────────────────────────────────

/// Captures RSS before and after a section of code.
pub struct MemoryTracker {
    rss_before: Option<usize>,
}

impl MemoryTracker {
    /// Record the current RSS and return a tracker.
    pub fn start() -> Self {
        Self {
            rss_before: current_rss(),
        }
    }

    /// Record the current and peak RSS, returning a [`MemorySnapshot`].
    pub fn stop(self) -> MemorySnapshot {
        MemorySnapshot {
            rss_before: self.rss_before,
            rss_after: current_rss(),
            peak: peak_rss(),
        }
    }
}

/// Result of a memory measurement, capturing before/after RSS and the
/// process-lifetime peak.
#[derive(Debug, Clone, Copy)]
pub struct MemorySnapshot {
    /// RSS when the tracker was started (bytes), or `None` if unavailable.
    pub rss_before: Option<usize>,
    /// RSS when the tracker was stopped (bytes), or `None` if unavailable.
    pub rss_after: Option<usize>,
    /// Process-lifetime peak RSS (bytes), or `None` if unavailable.
    pub peak: Option<usize>,
}

impl MemorySnapshot {
    /// The delta in RSS (`after − before`), in bytes.  Negative means memory
    /// was freed.  Returns `None` when either endpoint is unavailable.
    pub fn delta(&self) -> Option<isize> {
        match (self.rss_before, self.rss_after) {
            (Some(b), Some(a)) => {
                // Careful: usize → isize can wrap on very large values, but
                // in practice RSS fits in isize on 64-bit.
                Some(a as isize - b as isize)
            }
            _ => None,
        }
    }
}

fn fmt_bytes(bytes: usize) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = KB * 1024.0;
    const GB: f64 = MB * 1024.0;
    let b = bytes as f64;
    if b >= GB {
        format!("{:.2} GB", b / GB)
    } else if b >= MB {
        format!("{:.1} MB", b / MB)
    } else if b >= KB {
        format!("{:.1} KB", b / KB)
    } else {
        format!("{bytes} B")
    }
}

fn fmt_signed_bytes(bytes: isize) -> String {
    let sign = if bytes >= 0 { "+" } else { "-" };
    let abs = bytes.unsigned_abs();
    format!("{sign}{}", fmt_bytes(abs))
}

impl fmt::Display for MemorySnapshot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Memory:")?;
        if let Some(b) = self.rss_before {
            write!(f, " before={}", fmt_bytes(b))?;
        }
        if let Some(a) = self.rss_after {
            write!(f, " after={}", fmt_bytes(a))?;
        }
        if let Some(d) = self.delta() {
            write!(f, " delta={}", fmt_signed_bytes(d))?;
        }
        if let Some(p) = self.peak {
            write!(f, " process_peak={}", fmt_bytes(p))?;
        }
        Ok(())
    }
}

// ─── Platform-specific implementations ──────────────────────────────────────

#[cfg(target_os = "macos")]
mod platform {
    use std::mem;

    // Use our own extern to avoid the libc deprecation warning.
    unsafe extern "C" {
        fn mach_task_self() -> u32;
    }

    /// Current RSS via Mach `task_info`.
    pub(super) fn current_rss() -> Option<usize> {
        // SAFETY: FFI call to mach kernel; the struct is zero-initialised and
        // the kernel fills it in.  This is the standard idiom used by tools
        // like `top`, `ps`, and the Rust `jemalloc` crate.
        unsafe {
            let mut info: libc::mach_task_basic_info_data_t = mem::zeroed();
            let mut count = (mem::size_of::<libc::mach_task_basic_info_data_t>()
                / mem::size_of::<libc::mach_msg_type_number_t>())
                as libc::mach_msg_type_number_t;
            let kr = libc::task_info(
                mach_task_self(),
                libc::MACH_TASK_BASIC_INFO,
                (&raw mut info).cast(),
                &raw mut count,
            );
            if kr == libc::KERN_SUCCESS as libc::kern_return_t {
                Some(info.resident_size as usize)
            } else {
                None
            }
        }
    }

    /// Peak RSS via `getrusage(RUSAGE_SELF)`.  On macOS `ru_maxrss` is in
    /// **bytes** (unlike Linux where it is in KB).
    pub(super) fn peak_rss() -> Option<usize> {
        unsafe {
            let mut usage: libc::rusage = mem::zeroed();
            if libc::getrusage(libc::RUSAGE_SELF, &raw mut usage) == 0 {
                Some(usage.ru_maxrss as usize)
            } else {
                None
            }
        }
    }
}

#[cfg(target_os = "linux")]
mod platform {
    use std::fs;

    /// Parse a value in KB from `/proc/self/status`.
    fn parse_proc_status(key: &str) -> Option<usize> {
        let contents = fs::read_to_string("/proc/self/status").ok()?;
        for line in contents.lines() {
            if line.starts_with(key) {
                // Format: "VmRSS:    12345 kB"
                let value_part = line.split_whitespace().nth(1)?;
                let kb: usize = value_part.parse().ok()?;
                return Some(kb * 1024); // convert to bytes
            }
        }
        None
    }

    pub(super) fn current_rss() -> Option<usize> {
        parse_proc_status("VmRSS:")
    }

    pub(super) fn peak_rss() -> Option<usize> {
        parse_proc_status("VmPeak:")
    }
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
mod platform {
    pub(super) fn current_rss() -> Option<usize> {
        None
    }

    pub(super) fn peak_rss() -> Option<usize> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rss_returns_some() {
        // On supported platforms (macOS/Linux), we should get values.
        if cfg!(any(target_os = "macos", target_os = "linux")) {
            assert!(current_rss().is_some());
            assert!(peak_rss().is_some());
        }
    }

    #[test]
    fn tracker_round_trip() {
        let tracker = MemoryTracker::start();
        // Allocate a bit of memory to make sure delta is measurable.
        let _v: Vec<u8> = vec![0u8; 1_000_000];
        let snap = tracker.stop();
        // Just ensure Display doesn't panic.
        let _s = format!("{snap}");
    }
}
