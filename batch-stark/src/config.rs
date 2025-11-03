//! Configuration utilities for batch-STARK proofs.

use p3_challenger::FieldChallenger;
use p3_commit::Pcs;
use p3_field::{ExtensionField, PrimeCharacteristicRing};
pub use p3_uni_stark::StarkGenericConfig as SGC;
// Re-export the canonical config and common aliases from uni-stark to avoid duplication.
pub use p3_uni_stark::{Domain, PackedChallenge, PackedVal, PcsError, StarkGenericConfig, Val};

/// The challenge (extension field) type.
pub type Challenge<SC> = <SC as StarkGenericConfig>::Challenge;

/// The PCS commitment type for a STARK configuration.
pub type Commitment<SC> = <<SC as StarkGenericConfig>::Pcs as Pcs<
    <SC as StarkGenericConfig>::Challenge,
    <SC as StarkGenericConfig>::Challenger,
>>::Commitment;

/// The PCS proof type for a STARK configuration.
pub type PcsProof<SC> = <<SC as StarkGenericConfig>::Pcs as Pcs<
    <SC as StarkGenericConfig>::Challenge,
    <SC as StarkGenericConfig>::Challenger,
>>::Proof;

/// Helper to observe base field elements as extension field elements for recursion-friendly transcripts.
///
/// This simplifies recursive verifier circuits by using a uniform extension field challenger.
/// Instead of observing a mix of base and extension field elements, we convert all base field
/// observations (metadata, public values) to extension field elements before passing to the challenger.
///
/// # Recursion Benefits
///
/// In recursive proof systems, the verifier circuit needs to verify the inner proof. Since STARK
/// verification operates entirely in the extension field (challenges, opened values, constraint
/// evaluation), having a challenger that only observes extension field elements significantly
/// simplifies the recursive circuit implementation.
#[inline]
pub fn observe_base_as_ext<SC: StarkGenericConfig>(challenger: &mut SC::Challenger, val: Val<SC>)
where
    Challenge<SC>: ExtensionField<Val<SC>>,
{
    challenger.observe_algebra_element(Challenge::<SC>::from(val));
}

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
    observe_base_as_ext::<SC>(ch, Val::<SC>::from_usize(log_ext_degree));
    observe_base_as_ext::<SC>(ch, Val::<SC>::from_usize(log_degree));
    observe_base_as_ext::<SC>(ch, Val::<SC>::from_usize(width));
    observe_base_as_ext::<SC>(ch, Val::<SC>::from_usize(n_quotient_chunks));
}
