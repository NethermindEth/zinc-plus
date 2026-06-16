//! Lightweight, env-gated per-region wall-clock timing for diagnosing prover
//! hot paths *in situ* — without writing a separate benchmark per sub-step.
//!
//! Sprinkle [`scope`] calls around the regions of interest in the real prover
//! code; each returns an RAII guard that, on drop, records its elapsed time into
//! a thread-local table keyed by a `&'static str` label. Scopes **nest**: a
//! guard alive while an inner guard opens and closes becomes that inner guard's
//! parent, so the table tracks both *inclusive* (wall-clock, self + descendants)
//! and *self* (exclusive of child scopes) time per region. Call
//! [`dump_and_reset`] once per measured unit (e.g. just after a criterion
//! `bench.iter`) to print the tree — in execution order, indented by depth, with
//! each region's share of the total instrumented time — to stderr, then clear.
//!
//! **Zero-cost when off.** Every [`scope`] checks a process-cached flag and,
//! when `OBLONG_PROFILE` is unset, returns an inert guard whose `Drop` does
//! nothing; [`dump_and_reset`] is then a no-op. Enable with `OBLONG_PROFILE=1`
//! in the environment. Because criterion lets stderr through while capturing
//! stdout, the table lands next to the benchmark output.
//!
//! **Threading.** Timing is thread-local: place scopes on the control-flow
//! thread (which blocks on any rayon join inside the region), *not* inside
//! parallel worker closures — a main-thread scope then measures the region's
//! true wall-clock including its parallel section, and nesting stays correct.
//!
//! Intended for the F_2 SHA-256 prover — the e2e prove path (`f2_prove.rs`) and
//! the oblong Hadamard discharge (`prove_oblong_and_*` in `zinc-poly`); see
//! `documentation/f2x-sha-todo.md`.

#![allow(clippy::arithmetic_side_effects)] // diagnostic-only timing arithmetic;
// overflow here would mean a single thread spent ~580 years in one region.

use std::cell::{Cell, RefCell};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

/// Whether profiling is active this process. Cached on first read; set
/// `OBLONG_PROFILE` (to any value) in the environment to enable.
fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("OBLONG_PROFILE").is_some())
}

/// Optional process-wide probe returning the current **peak** resident-set size
/// in bytes (e.g. `getrusage(RUSAGE_SELF).ru_maxrss`). A measurement binary
/// registers it via [`set_rss_probe`]; `utils` itself stays dependency-free (no
/// `libc`). When no probe is registered the profiler behaves exactly as before
/// — the memory columns are simply omitted from [`dump_and_reset`].
static RSS_PROBE: OnceLock<fn() -> u64> = OnceLock::new();

/// Register the peak-RSS probe used to annotate each timed region with the
/// process high-water-mark growth it caused. Call once at program start, before
/// the first [`scope`]. Idempotent; a second call is ignored.
///
/// Because `ru_maxrss` is monotonic, the per-region `Δrss` reported by
/// [`dump_and_reset`] is *how much that region raised the process peak* (a
/// nested region's figure includes its children, like inclusive time). Regions
/// that allocate then free without exceeding an earlier high-water show `Δrss=0`
/// — that transient is attributed to whichever region first set the peak.
pub fn set_rss_probe(probe: fn() -> u64) {
    let _ = RSS_PROBE.set(probe);
}

/// Read the registered peak-RSS probe, or `None` if memory profiling is off.
#[inline]
fn rss_now() -> Option<u64> {
    RSS_PROBE.get().map(|f| f())
}

/// An open (not-yet-dropped) timing region on this thread's scope stack.
struct Frame {
    label: &'static str,
    start: Instant,
    /// Inclusive time charged to direct child scopes opened under this frame.
    children: Duration,
    /// Stack depth at entry (0 = top level).
    depth: usize,
    /// Monotonic entry rank, for stable execution-order printing.
    order: u64,
    /// Peak RSS (bytes) observed at scope entry, if a probe is registered.
    rss_start: Option<u64>,
}

/// One row of the accumulated report.
struct Record {
    label: &'static str,
    /// Wall-clock time inside this region (self + all descendants).
    inclusive: Duration,
    /// Time inside this region but not any child scope.
    self_: Duration,
    count: u64,
    depth: usize,
    order: u64,
    /// Sum of per-entry peak-RSS growth (bytes) charged to this region.
    rss_delta: u64,
    /// Peak RSS (bytes) at the region's last exit.
    rss_end: u64,
    /// Whether any RSS sample was recorded (probe was registered).
    has_rss: bool,
}

thread_local! {
    /// Currently-open frames, innermost last. Popped LIFO on guard drop.
    static STACK: RefCell<Vec<Frame>> = const { RefCell::new(Vec::new()) };
    /// Completed regions, one [`Record`] per distinct label.
    static RECORDS: RefCell<Vec<Record>> = const { RefCell::new(Vec::new()) };
    /// Monotonic entry counter, for execution-order sorting.
    static ORDER: Cell<u64> = const { Cell::new(0) };
}

/// RAII timing guard returned by [`scope`]. On drop it pops this thread's scope
/// stack and folds its elapsed time into the report (and into its parent's
/// child total). An inert guard (profiling off) does nothing on drop.
#[must_use = "the region is only timed for as long as the guard is alive"]
pub struct Scope {
    active: bool,
}

impl Drop for Scope {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        let Some(frame) = STACK.with(|s| s.borrow_mut().pop()) else {
            return;
        };
        let elapsed = frame.start.elapsed();
        let self_time = elapsed.saturating_sub(frame.children);
        // Peak-RSS growth caused by this region (exit peak − entry peak). Since
        // `ru_maxrss` is monotonic the delta is non-negative; `rss_end` is the
        // process high-water at exit.
        let (rss_delta, rss_end, has_rss) = match frame.rss_start {
            Some(start) => {
                let end = rss_now().unwrap_or(start);
                (end.saturating_sub(start), end, true)
            }
            None => (0, 0, false),
        };
        // Charge the full (inclusive) time to the parent's child total.
        STACK.with(|s| {
            if let Some(parent) = s.borrow_mut().last_mut() {
                parent.children = parent.children.saturating_add(elapsed);
            }
        });
        RECORDS.with(|r| {
            let mut records = r.borrow_mut();
            match records.iter_mut().find(|rec| rec.label == frame.label) {
                Some(rec) => {
                    rec.inclusive = rec.inclusive.saturating_add(elapsed);
                    rec.self_ = rec.self_.saturating_add(self_time);
                    rec.count += 1;
                    rec.rss_delta = rec.rss_delta.saturating_add(rss_delta);
                    rec.rss_end = rss_end;
                    rec.has_rss |= has_rss;
                }
                None => records.push(Record {
                    label: frame.label,
                    inclusive: elapsed,
                    self_: self_time,
                    count: 1,
                    depth: frame.depth,
                    order: frame.order,
                    rss_delta,
                    rss_end,
                    has_rss,
                }),
            }
        });
    }
}

/// Open a timing region labelled `label`. Bind the returned guard (e.g.
/// `let _g = prof::scope("uair:alpha_project");`) so the region is timed until
/// the guard drops at end of the enclosing block; guards opened while another is
/// alive nest under it. Cheap no-op when `OBLONG_PROFILE` is unset.
#[inline]
pub fn scope(label: &'static str) -> Scope {
    if !enabled() {
        return Scope { active: false };
    }
    let start = Instant::now();
    let rss_start = rss_now();
    let order = ORDER.with(|o| {
        let v = o.get();
        o.set(v + 1);
        v
    });
    STACK.with(|s| {
        let mut stack = s.borrow_mut();
        let depth = stack.len();
        stack.push(Frame { label, start, children: Duration::ZERO, depth, order, rss_start });
    });
    Scope { active: true }
}

/// Print the accumulated region tree to stderr under `header`, then clear it.
/// Rows are in execution order, indented by nesting depth; each shows inclusive
/// time and its share of the total top-level (depth-0) time, with a trailing
/// `self=…` when the region has child scopes. No-op when profiling is off or no
/// regions were recorded. Call once per measured unit so per-call figures
/// reflect that unit.
pub fn dump_and_reset(header: &str) {
    if !enabled() {
        return;
    }
    RECORDS.with(|r| {
        let mut records = r.borrow_mut();
        if records.is_empty() {
            return;
        }
        records.sort_by_key(|rec| rec.order);
        // Share denominator: sum of top-level regions (a near-complete, non-
        // overlapping partition of the measured work). Fall back to the largest
        // inclusive time if nothing was recorded at depth 0.
        let root: Duration = records
            .iter()
            .filter(|rec| rec.depth == 0)
            .map(|rec| rec.inclusive)
            .sum();
        let denom = if root.is_zero() {
            records.iter().map(|rec| rec.inclusive).max().unwrap_or_default()
        } else {
            root
        };
        let denom_secs = denom.as_secs_f64();
        // Memory annotation is present iff a peak-RSS probe was registered.
        let any_rss = records.iter().any(|rec| rec.has_rss);
        let mib = |bytes: u64| bytes as f64 / (1024.0 * 1024.0);
        eprintln!("┌─ prove profile: {header}");
        for rec in records.iter() {
            let indent = "  ".repeat(rec.depth);
            let share = if denom_secs == 0.0 {
                0.0
            } else {
                rec.inclusive.as_secs_f64() / denom_secs * 100.0
            };
            let incl_s = format!("{:.3?}", rec.inclusive);
            let label_field = format!("{indent}{}", rec.label);
            let count_note = if rec.count == 1 {
                String::new()
            } else {
                format!(" n={}", rec.count)
            };
            // Only surface self-time when it diverges from inclusive (i.e. the
            // region has children); for leaves the two are equal.
            let self_note = if rec.self_ < rec.inclusive {
                format!("  self={:.3?}", rec.self_)
            } else {
                String::new()
            };
            // `Δrss` = peak-RSS growth this region caused; `@` = process peak at
            // exit. Inclusive of children, matching the time columns.
            let mem_note = if rec.has_rss {
                format!("  Δrss=+{:>8.1}MiB  @{:>9.1}MiB", mib(rec.rss_delta), mib(rec.rss_end))
            } else {
                String::new()
            };
            eprintln!(
                "│  {label_field:<26} {incl_s:>12}  {share:5.1}%{mem_note}{self_note}{count_note}"
            );
        }
        let denom_s = format!("{denom:.3?}");
        if any_rss {
            let peak_rss = records.iter().map(|rec| rec.rss_end).max().unwrap_or(0);
            eprintln!(
                "└─ top-level total {denom_s}   peak RSS {:.1} MiB ({:.3} GiB)",
                mib(peak_rss),
                mib(peak_rss) / 1024.0
            );
        } else {
            eprintln!("└─ top-level total {denom_s}");
        }
        records.clear();
        // Reset entry counter so the next unit starts fresh in execution order.
        ORDER.with(|o| o.set(0));
        STACK.with(|s| s.borrow_mut().clear());
    });
}
