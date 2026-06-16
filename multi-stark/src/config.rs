//! Configuration tying together the commitment scheme, challenge field, and transcript.

use alloc::vec::Vec;

use p3_commit::MultilinearPcs;
use p3_field::{ExtensionField, Field};
use p3_multilinear_util::poly::Poly;

/// The wiring a multilinear AIR proof depends on.
///
/// One implementation fixes a commitment scheme, a challenge field, and a transcript type.
///
/// The prover and verifier are generic over any implementation.
pub trait MultiStarkConfig {
    /// Base field the trace and committed columns live in.
    type Val: Field;

    /// Extension field that challenges are drawn from.
    ///
    /// Large enough that a random challenge collides with negligible probability.
    type Challenge: ExtensionField<Self::Val>;

    /// Fiat-Shamir transcript used to derive challenges and absorb commitments.
    type Challenger;

    /// Multilinear commitment scheme over the base field.
    type Pcs: MultilinearPcs<Self::Challenge, Self::Challenger, Val = Self::Val>;

    /// Borrow the commitment scheme.
    fn pcs(&self) -> &Self::Pcs;

    /// Pack committed column polynomials into the commitment scheme's witness form.
    ///
    /// The witness representation is private to the commitment scheme:
    /// - it folds in the slot layout and preprocessing depth that scheme expects,
    /// - the generic commitment interface exposes neither,
    /// - so the columns-to-witness bridge lives here, where those details are known.
    ///
    /// # Arguments
    ///
    /// - `columns`: one multilinear polynomial per committed trace column.
    fn build_witness(
        &self,
        columns: Vec<Poly<Self::Val>>,
    ) -> <Self::Pcs as MultilinearPcs<Self::Challenge, Self::Challenger>>::Witness;
}

/// Commitment scheme selected by a configuration.
pub type Pcs<C> = <C as MultiStarkConfig>::Pcs;

/// Succinct commitment produced by a configuration's commitment scheme.
pub type Commitment<C> = <Pcs<C> as MultilinearPcs<
    <C as MultiStarkConfig>::Challenge,
    <C as MultiStarkConfig>::Challenger,
>>::Commitment;

/// Prover-only data retained between commit and open.
pub type ProverData<C> = <Pcs<C> as MultilinearPcs<
    <C as MultiStarkConfig>::Challenge,
    <C as MultiStarkConfig>::Challenger,
>>::ProverData;
