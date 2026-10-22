use alloc::vec::Vec;

use p3_air::Air;
use p3_challenger::{CanObserve, CanSample, FieldChallenger, GrindingChallenger};
use p3_commit::{
    CommitmentOpening, LagrangeSelectors, Pcs, PeriodicColumns, PolynomialSpace, UnivariateStarkPcs,
};
use p3_field::Field;
use p3_verifier_arith::{Arith, Lift, Native};

use crate::{
    Com, Domain, InvalidProofShapeError, PcsError, StarkGenericConfig, Val,
    VerifierConstraintFolder, recompose_quotient_from_chunks,
};

/// Opening proof of a configuration's PCS.
pub type PcsProof<SC> = <<SC as StarkGenericConfig>::Pcs as Pcs<
    <SC as StarkGenericConfig>::Challenge,
    <SC as StarkGenericConfig>::Challenger,
>>::Proof;

/// What a STARK verifier delegates to its arithmetic: domains, commitments and openings.
///
/// [`Native`] implements it for every configuration through the configuration's own types.
pub trait StarkArith<SC: StarkGenericConfig>:
    Lift<Val = Val<SC>, Challenge = SC::Challenge, F: Clone, EF: Clone>
{
    /// A commitment of the configured PCS.
    type Commitment: Clone;

    /// The challenger the native transcript runs on.
    type Challenger: FieldChallenger<Val<SC>>
        + GrindingChallenger<Witness = Val<SC>>
        + CanObserve<Com<SC>>
        + CanSample<SC::Challenge>;

    /// The challenger a proof's transcript starts from.
    fn challenger(&mut self, config: &SC) -> Self::Challenger;

    /// Lift a commitment the prover sent.
    fn load_commitment(&mut self, pcs: &SC::Pcs, commitment: &Com<SC>) -> Self::Commitment;

    /// Lift a commitment the verifier holds, such as its preprocessed columns.
    fn load_key_commitment(&mut self, pcs: &SC::Pcs, commitment: &Com<SC>) -> Self::Commitment;

    /// Announce a commitment the native transcript absorbs next.
    fn will_observe_commitment(&mut self, pcs: &SC::Pcs, commitment: &Self::Commitment);

    /// The Lagrange selectors of a domain at a point, or `None` when the point lies in it.
    fn selectors(
        &mut self,
        domain: &Domain<SC>,
        point: &Self::EF,
    ) -> Option<LagrangeSelectors<Self::EF>>;

    /// The point after `point` in the order of the domain.
    fn next_point(&mut self, domain: &Domain<SC>, point: &Self::EF) -> Option<Self::EF>;

    /// Every periodic column of a domain at a point.
    fn periodic_values(
        &mut self,
        domain: &Domain<SC>,
        columns: PeriodicColumns<'_, Val<SC>>,
        point: &Self::EF,
    ) -> Vec<Self::EF>;

    /// The quotient at a point, recomposed from its chunks.
    ///
    /// # Errors
    ///
    /// When the chunks do not pair one to one with their domains.
    fn recompose_quotient(
        &mut self,
        domains: &[Domain<SC>],
        chunks: &[Vec<Self::EF>],
        point: &Self::EF,
    ) -> Result<Self::EF, InvalidProofShapeError>;

    /// Verify an opening argument.
    ///
    /// `preprocessed` is the position of the claim on the preprocessed commitment, if any.
    ///
    /// # Errors
    ///
    /// When the opening argument does not verify.
    fn verify_opening(
        &mut self,
        pcs: &SC::Pcs,
        claims: Vec<CommitmentOpening<Self::EF, Self::Commitment, Domain<SC>>>,
        proof: &PcsProof<SC>,
        challenger: &mut Self::Challenger,
        preprocessed: Option<usize>,
    ) -> Result<(), PcsError<SC>>;
}

impl<SC> StarkArith<SC> for Native<Val<SC>, SC::Challenge>
where
    SC: StarkGenericConfig<Challenger: GrindingChallenger<Witness = Val<SC>>>,
{
    type Commitment = Com<SC>;
    type Challenger = SC::Challenger;

    #[inline]
    fn challenger(&mut self, config: &SC) -> SC::Challenger {
        config.initialise_challenger()
    }

    #[inline]
    fn load_commitment(&mut self, _pcs: &SC::Pcs, commitment: &Com<SC>) -> Com<SC> {
        commitment.clone()
    }

    #[inline]
    fn load_key_commitment(&mut self, _pcs: &SC::Pcs, commitment: &Com<SC>) -> Com<SC> {
        commitment.clone()
    }

    #[inline]
    fn will_observe_commitment(&mut self, _pcs: &SC::Pcs, _commitment: &Com<SC>) {}

    fn selectors(
        &mut self,
        domain: &Domain<SC>,
        point: &SC::Challenge,
    ) -> Option<LagrangeSelectors<SC::Challenge>> {
        (!domain.vanishing_poly_at_point(*point).is_zero())
            .then(|| domain.selectors_at_point(*point))
    }

    #[inline]
    fn next_point(&mut self, domain: &Domain<SC>, point: &SC::Challenge) -> Option<SC::Challenge> {
        domain.next_point(*point)
    }

    fn periodic_values(
        &mut self,
        domain: &Domain<SC>,
        columns: PeriodicColumns<'_, Val<SC>>,
        point: &SC::Challenge,
    ) -> Vec<SC::Challenge> {
        domain.evaluate_periodic_columns_at(columns, *point)
    }

    fn recompose_quotient(
        &mut self,
        domains: &[Domain<SC>],
        chunks: &[Vec<SC::Challenge>],
        point: &SC::Challenge,
    ) -> Result<SC::Challenge, InvalidProofShapeError> {
        recompose_quotient_from_chunks::<SC>(domains, chunks, *point)
    }

    fn verify_opening(
        &mut self,
        pcs: &SC::Pcs,
        claims: Vec<CommitmentOpening<SC::Challenge, Com<SC>, Domain<SC>>>,
        proof: &PcsProof<SC>,
        challenger: &mut SC::Challenger,
        preprocessed: Option<usize>,
    ) -> Result<(), PcsError<SC>> {
        pcs.verify_with_preprocessing(claims, proof, challenger, preprocessed)
    }
}

/// One trace's openings at `zeta`, as its constraints read them.
pub struct ConstraintInputs<'a, AR: Arith + ?Sized> {
    /// Main trace at `zeta`.
    pub trace_local: &'a [AR::EF],
    /// Main trace at the point after `zeta`.
    pub trace_next: &'a [AR::EF],
    /// Preprocessed trace at `zeta`, empty when there is none.
    pub preprocessed_local: &'a [AR::EF],
    /// Preprocessed trace at the point after `zeta`.
    pub preprocessed_next: &'a [AR::EF],
    /// Periodic columns at `zeta`.
    pub periodic_values: &'a [AR::EF],
    /// Public values of the trace.
    pub public_values: &'a [AR::F],
    /// Lagrange selectors of the trace domain at `zeta`.
    pub selectors: &'a LagrangeSelectors<AR::EF>,
    /// Challenge folding every constraint.
    pub alpha: AR::EF,
}

/// Evaluates an AIR's constraints at `zeta` within an arithmetic.
pub trait ConstraintArith<SC: StarkGenericConfig, A>: StarkArith<SC> {
    /// The AIR's constraints at `zeta`, folded by `alpha`.
    fn fold_constraints(&mut self, air: &A, inputs: &ConstraintInputs<'_, Self>) -> Self::EF;
}

impl<SC, A> ConstraintArith<SC, A> for Native<Val<SC>, SC::Challenge>
where
    SC: StarkGenericConfig<Challenger: GrindingChallenger<Witness = Val<SC>>>,
    A: for<'b> Air<VerifierConstraintFolder<'b, SC>>,
{
    fn fold_constraints(&mut self, air: &A, inputs: &ConstraintInputs<'_, Self>) -> SC::Challenge {
        let mut folder = VerifierConstraintFolder::new(inputs);
        air.eval(&mut folder);
        folder.accumulator
    }
}
