#[allow(dead_code)]
#[path = "e2e.rs"]
mod e2e;

fn main() {
    e2e::run_og_sha256_instance_sweep_report();
}
