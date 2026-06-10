#![allow(dead_code)]

#[path = "e2e.rs"]
mod e2e;

fn main() {
    e2e::run_hyrax_width_sweep_report();
}
