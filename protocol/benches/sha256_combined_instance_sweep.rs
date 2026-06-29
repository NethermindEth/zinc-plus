#[allow(dead_code)]
#[path = "e2e.rs"]
mod e2e;

fn main() {
    e2e::run_sha256_combined_instance_sweep_report();
}
