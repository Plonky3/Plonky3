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
//! These bounds cover only the product reduction.
//! The caller must also authenticate the returned leaf evaluations.

mod error;
mod math;
mod proof;
mod prover;
mod shape;
mod transcript;

pub use error::{ProductGkrError, ProductGkrShapeError};
pub(crate) use proof::ROUND_POLY_LEN;
pub use proof::{ProductGkrLayerProof, ProductGkrOutput, ProductGkrProof};
pub use shape::{ProductGkrRootShape, ProductGkrShape};

#[cfg(test)]
mod tests;

#[cfg(test)]
mod transcript_tests;
