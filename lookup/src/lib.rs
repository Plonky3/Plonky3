//! Interaction primitives for STARK proof systems.
//!
//! This crate provides a generic framework for inter-AIR communication via **interactions**.
//!
//! AIRs **send** and **receive** data bundles, enforced via lookup arguments.

#![no_std]

extern crate alloc;

pub mod builder;
pub mod error;
pub mod gadgets;
pub mod interaction;

pub use builder::InteractionCollector;
pub use error::{LookupError, LookupResult};
pub use gadgets::{InteractionGadget, LogUpGadget};
pub use interaction::{AirBuilderWithInteractions, Interaction, InteractionKind, eval_symbolic};

#[cfg(test)]
mod tests;
