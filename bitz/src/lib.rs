//! BitZ: linear claims over a prime field on committed bits, proved through a grand product in the exponent of `GF(2^192)`.
//!
//! A claim `<u1 (x) u2, bits> = mu` is split into integer folds (Construction 4.4, Round 2), each proved as `g^(mu_c)` by a batched grand product (Phase 3, Construction 6.1), whose leaf claim a sumcheck reduces to one evaluation of the bits (Construction 6.2).
//!
//! Every target field is below `2^64`, so Round 1 is skipped and every fold stays far below `ord(g) = 2^192 - 1`.
//!
//! Reference: BitZ, ePrint 2026/2141.

#![no_std]

extern crate alloc;
#[cfg(test)]
extern crate std;

mod claim;
mod error;
mod fold;
mod inner;
mod leaves;
mod params;
mod pcs;
mod proof;
mod reduce;
mod transcript;

#[cfg(test)]
mod tests;

pub use claim::{ClaimBinding, ColumnWeights, RowWeights, TensorClaim};
pub use error::{ClaimError, ParamsError, ProvingError, RegionError, VerificationError};
pub use p3_binary_pcs::whir::preset::keccak64::{
    Challenge, Challenger, Commitment, OpeningProof, ProverData, Val, WitnessCommitment,
};
pub use params::{BitShape, BitzParams};
pub use pcs::BitzPcs;
pub use proof::{BitzProof, BitzReduction, InnerProof};
pub use reduce::Bitz;
