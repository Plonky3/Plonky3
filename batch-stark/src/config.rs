//! Configuration utilities for batch-STARK proofs.

use p3_commit::Pcs;
pub use p3_uni_stark::StarkGenericConfig as SGC;
// Re-export the canonical config and common aliases from uni-stark to avoid duplication.
pub use p3_uni_stark::{Domain, PackedChallenge, PackedVal, PcsError, StarkGenericConfig, Val};

/// The challenge (extension field) type.
pub type Challenge<SC> = <SC as StarkGenericConfig>::Challenge;

/// The [`Pcs`] commitment type for a STARK configuration.
pub type Commitment<SC> = <<SC as StarkGenericConfig>::Pcs as Pcs<
    <SC as StarkGenericConfig>::Challenge,
    <SC as StarkGenericConfig>::Challenger,
>>::Commitment;

/// The [`Pcs`] proof type for a STARK configuration.
pub type PcsProof<SC> = <<SC as StarkGenericConfig>::Pcs as Pcs<
    <SC as StarkGenericConfig>::Challenge,
    <SC as StarkGenericConfig>::Challenger,
>>::Proof;
