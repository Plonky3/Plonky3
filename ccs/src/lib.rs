//! Customizable constraint systems over structured matrices.
//!
//! A witness z satisfies the system when every row of `sum_i c_i * prod_{j in S_i} (M_j z)` vanishes.
//!
//! Matrices are rules, not tables: each kind applies itself, its transpose, and its multilinear extension.
//!
//! The SuperSpartan reduction turns satisfiability into one witness evaluation claim.
//!
//! The committed proof opens that claim and pins public cells, such as the constant one of an R1CS.
//!
//! Reference: Setty, Thaler and Wahby, "Customizable constraint systems for succinct arguments", ePrint 2023/552.

#![no_std]

extern crate alloc;
#[cfg(test)]
extern crate std;

mod committed;
mod error;
mod mle;
mod operator;
mod proof;
mod prover;
mod public;
mod relation;
mod transcript;
mod verifier;

#[cfg(test)]
mod tests;

pub use committed::{prove_committed, verify_committed};
pub use error::{CommittedVerificationError, StructureError, VerificationError, WitnessError};
pub use operator::{BitShift, BlockDiagonal, LinearOperator, Operator, SparseMatrix, TraceColumn};
pub use proof::{CcsProof, CommittedProof, WitnessClaim};
pub use prover::prove;
pub use public::PublicRegion;
pub use relation::{Ccs, Constraint, RowSelector, Term};
pub use verifier::verify;
