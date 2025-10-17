//! Configuration types for multi-STARK proofs.
//!
//! This module re-uses `p3_uni_stark::StarkGenericConfig` as the underlying config trait
//! and provides convenient type aliases for common associated types.

use p3_commit::{Pcs, PolynomialSpace};
use p3_field::{ExtensionField, Field};
pub use p3_uni_stark::StarkGenericConfig;

/// Marker trait for multi-STARK configurations.
/// This is semantically equivalent to `StarkGenericConfig` but provides clarity
/// when used as a bound in multi-STARK functions.
pub trait MultiStarkGenericConfig: StarkGenericConfig {}

/// Blanket implementation: any `StarkGenericConfig` is a `MultiStarkGenericConfig`.
impl<T: StarkGenericConfig> MultiStarkGenericConfig for T {}

/// The PCS error type for a STARK configuration.
pub type PcsError<SC> = <<SC as StarkGenericConfig>::Pcs as Pcs<
    <SC as StarkGenericConfig>::Challenge,
    <SC as StarkGenericConfig>::Challenger,
>>::Error;

/// The domain type for a STARK configuration.
pub type Domain<SC> = <<SC as StarkGenericConfig>::Pcs as Pcs<
    <SC as StarkGenericConfig>::Challenge,
    <SC as StarkGenericConfig>::Challenger,
>>::Domain;

/// The base field value type.
pub type Val<SC> = <Domain<SC> as PolynomialSpace>::Val;

/// The packed base field value type.
pub type PackedVal<SC> = <Val<SC> as Field>::Packing;

/// The packed challenge (extension field) type.
pub type PackedChallenge<SC> =
    <<SC as StarkGenericConfig>::Challenge as ExtensionField<Val<SC>>>::ExtensionPacking;

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
