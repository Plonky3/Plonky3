//! An AIR for the BLAKE2s compression function over fields of characteristic 2.
//!
//! BLAKE2s is the hash behind Lean Ethereum's XMSS signatures.
//!
//! Its proving cost is therefore the cost of verifying those signatures.

#![no_std]

extern crate alloc;

mod binary;
mod constants;

pub use binary::*;
