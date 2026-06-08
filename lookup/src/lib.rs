//! Lookup arguments for STARKs.
//!
//! Implements the [LogUp] protocol for intra-AIR (local) and cross-AIR
//! (global) lookup arguments.
//!
//! [LogUp]: logup::LogUpGadget

#![no_std]

extern crate alloc;

mod builder;
mod bus;
mod count;
pub mod debug_util;
pub mod folder;
pub mod logup;
pub mod protocol;
pub mod symbolic;
#[cfg(test)]
mod tests;
pub mod traits;
mod types;

pub use builder::{InteractionBuilder, SymbolicInteraction, SymbolicLocalInteraction};
pub use bus::{LookupBus, PermutationCheckBus};
pub use count::Count;
pub use logup::LogUpGadget;
pub use protocol::LookupProtocol;
pub use symbolic::InteractionSymbolicBuilder;
pub use types::*;
