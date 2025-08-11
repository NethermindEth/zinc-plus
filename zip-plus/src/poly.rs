pub mod errors;


pub(crate) extern crate alloc;

// ark-std v0.5 should re-export alloc/std::sync.
// While already released on crates.io, related versioning changes were reverted on the GitHub repo.
// Let's wait for this issue to stabilize.


#[cfg(not(target_has_atomic = "ptr"))]
pub use ark_std::rc::Rc as RefCounter;
