//! Core trait definitions for recursive verification.

mod air;
mod challenger;
mod mmcs;
mod pcs;
mod recursive;

pub use air::{LookupMetadata, RecursiveAir};
pub use challenger::RecursiveChallenger;
pub use mmcs::{RecursiveExtensionMmcs, RecursiveMmcs};
pub use pcs::{ComsWithOpeningsTargets, RecursivePcs};
pub use recursive::{Recursive, RecursiveLookupGadget};
