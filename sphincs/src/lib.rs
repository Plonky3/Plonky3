//! SLH-DSA-SHA2-128s (FIPS 205) verification, and many signatures aggregated in one binary-field proof.
//!
//! Native verification follows the algorithms of FIPS 205 one hash call at a time.
//!
//! The same verification is a Boolean circuit of SHA-256 compressions, which Flock proves on every signature of a batch (ePrint 2026/1329).

#![no_std]

extern crate alloc;

mod address;
mod aggregate;
mod circuit;
mod hash;
mod params;
mod signature;
mod verify;

pub use aggregate::{AggregationError, Aggregator, Claim, Statement};
pub use circuit::MESSAGE_BYTES;
pub use params::{
    A, D, FORS_BYTES, H, H_PRIME, K, LEN, LEN1, LEN2, LG_W, M, N, PUBLIC_KEY_BYTES,
    SIGNATURE_BYTES, W, XMSS_BYTES,
};
pub use signature::{PublicKey, Signature, SignatureError};
