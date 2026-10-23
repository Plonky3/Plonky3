//! XMSS-style hash-based signatures over a tweakable hash, and their aggregation.
//!
//! The scheme is the synchronized signature of "Hash-Based Multi-Signatures for Post-Quantum Ethereum" (ePrint 2025/055).
//!
//! Messages are encoded with the target-sum Winternitz encoding of "At the top of the hypercube" (ePrint 2025/889).
//!
//! Over keyed BLAKE3, many verifications are proved at once in one binary-field Flock proof, which binary recursion aggregates further.

#![no_std]

extern crate alloc;

mod binary;
mod blake3_hash;
mod claim;
mod hash;
mod keys;
mod params;
mod scheme;
mod signature;

pub use binary::{AggregationError, BinaryAggregator};
pub use blake3_hash::Blake3Hash;
pub use claim::{Claim, Statement};
pub use hash::TweakableHash;
pub use keys::{PublicKey, SecretKey};
pub use params::Params;
pub use scheme::{InvalidSignature, SignError, Xmss};
pub use signature::Signature;
