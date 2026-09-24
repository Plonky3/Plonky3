//! An AIR for the BLAKE2s compression function.
//!
//! [`Blake2sBinaryAir`] works over fields of characteristic 2, where XOR is addition.
//!
//! BLAKE2s is the hash Lean Ethereum's XMSS signatures are built from, so its cost on the
//! binary backend is the cost of verifying those signatures.

#![no_std]

extern crate alloc;

mod binary;
mod constants;

pub use binary::*;
