//! Batched product-tree GKR with identity-padded input prefixes.
//!
//! Two multiplication levels are contracted into one radix-four layer.
//! Multiplying by the equality polynomial gives degree five in each sumcheck variable.
//! One random challenge batches the same relation across every tree.
//!
//! A sumcheck round contributes at most `5 / |F|` soundness error.
//! Batching `T` trees contributes at most `(T - 1) / |F|` per layer.
//! Collapsing four child claims contributes at most `2 / |F|` per layer.
//!
//! A batched shape instead holds `2^b` trees in one table, batch index most significant.
//! Its roots are compared at a random point, costing `b / |F|`, and every layer sums over the batch coordinates.
//! An odd height then opens with a binary layer summed the same way.
//!
//! A batched table may prove its bottom radix-four layers from a cheaper representation, through `ProductLeaves`.
//! Those layers must send the messages the dense layers would, so proofs and transcripts are unchanged.
//!
//! These bounds cover only the product reduction.
//! The caller must also authenticate the returned leaf evaluations.

#![no_std]

extern crate alloc;

#[cfg(test)]
extern crate std;

mod error;
mod math;
mod proof;
mod prover;
mod shape;
mod transcript;

pub use error::{ProductGkrError, ProductGkrShapeError};
pub use proof::{
    DenseLeaves, PreparedProductGkr, ProductGkrLayerProof, ProductGkrOutput, ProductGkrProof,
    ProductLeaves, ROUND_POLY_LEN,
};
pub use prover::{LayerRounds, RadixFourBatch};
pub use shape::{ProductGkrRootShape, ProductGkrShape};

#[cfg(test)]
mod batched_tests;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod transcript_tests;
