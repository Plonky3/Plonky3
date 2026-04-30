//! WHIR: Reed-Solomon proximity testing with super-fast verification.
//!
//! An IOP of proximity for constrained Reed-Solomon codes that serves as
//! a multilinear polynomial commitment scheme.
//!
//! Reference: <https://eprint.iacr.org/2024/1586>

#![no_std]

extern crate alloc;

pub mod constraints;
pub mod fiat_shamir;
pub mod parameters;
pub mod pcs;
pub mod sumcheck;
pub(crate) mod utils;
