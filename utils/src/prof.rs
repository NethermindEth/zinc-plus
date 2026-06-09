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
                }
                None => records.push(Record {
                    label: frame.label,
                    inclusive: elapsed,
                    self_: self_time,
                    count: 1,
                    depth: frame.depth,
                    order: frame.order,
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
    let order = ORDER.with(|o| {
        let v = o.get();
        o.set(v + 1);
        v
    });
    STACK.with(|s| {
        let mut stack = s.borrow_mut();
        let depth = stack.len();
        stack.push(Frame { label, start, children: Duration::ZERO, depth, order });
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
            eprintln!("│  {label_field:<26} {incl_s:>12}  {share:5.1}%{self_note}{count_note}");
        }
        let denom_s = format!("{denom:.3?}");
        eprintln!("└─ top-level total {denom_s}");
        records.clear();
        // Reset entry counter so the next unit starts fresh in execution order.
        ORDER.with(|o| o.set(0));
        STACK.with(|s| s.borrow_mut().clear());
    });
}
