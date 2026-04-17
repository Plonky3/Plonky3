//! Lookup arguments for STARKs.
//!
//! Implements the [LogUp] protocol for intra-AIR (local) and cross-AIR
//! (global) lookup arguments.
//!
//! [LogUp]: logup::LogUpGadget

#![no_std]

extern crate alloc;

pub mod bus;
pub mod debug_util;
pub mod folder;
pub mod logup;
pub mod protocol;
#[cfg(test)]
mod tests;
pub mod traits;
mod types;

pub use logup::LogUpGadget;
pub use protocol::LookupProtocol;
pub use types::*;
