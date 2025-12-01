//! Configuration utilities for batch-STARK proofs.

use p3_challenger::FieldChallenger;
use p3_commit::Pcs;
use p3_field::{ExtensionField, PrimeCharacteristicRing};
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

#[inline]
pub fn observe_instance_binding<SC: SGC>(
    ch: &mut SC::Challenger,
    log_ext_degree: usize,
    log_degree: usize,
    width: usize,
    n_quotient_chunks: usize,
) where
    Challenge<SC>: ExtensionField<Val<SC>>,
{
    ch.observe_base_as_algebra_element::<Challenge<SC>>(Val::<SC>::from_usize(log_ext_degree));
    ch.observe_base_as_algebra_element::<Challenge<SC>>(Val::<SC>::from_usize(log_degree));
    ch.observe_base_as_algebra_element::<Challenge<SC>>(Val::<SC>::from_usize(width));
    ch.observe_base_as_algebra_element::<Challenge<SC>>(Val::<SC>::from_usize(n_quotient_chunks));
}
