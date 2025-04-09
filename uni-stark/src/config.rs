use core::marker::PhantomData;

use p3_challenger::{CanObserve, CanSample, FieldChallenger};
use p3_commit::{Pcs, PolynomialSpace};
use p3_field::{ExtensionField, Field};

pub type PcsError<SC> = <<SC as StarkGenericConfig>::Pcs as Pcs<
    <SC as StarkGenericConfig>::Challenge,
    <SC as StarkGenericConfig>::Challenger,
>>::Error;

pub type Domain<SC> = <<SC as StarkGenericConfig>::Pcs as Pcs<
    <SC as StarkGenericConfig>::Challenge,
    <SC as StarkGenericConfig>::Challenger,
>>::Domain;

pub type Val<SC> = <Domain<SC> as PolynomialSpace>::Val;

pub type PackedVal<SC> = <Val<SC> as Field>::Packing;

pub type PackedChallenge<SC> =
    <<SC as StarkGenericConfig>::Challenge as ExtensionField<Val<SC>>>::ExtensionPacking;

pub trait StarkGenericConfig {
    /// The PCS used to commit to trace polynomials.
    type Pcs: Pcs<Self::Challenge, Self::Challenger>;

    /// The field from which most random challenges are drawn.
    type Challenge: ExtensionField<Val<Self>>;

    /// The challenger (Fiat-Shamir) implementation used.
    type Challenger: FieldChallenger<Val<Self>>
        + CanObserve<<Self::Pcs as Pcs<Self::Challenge, Self::Challenger>>::Commitment>
        + CanSample<Self::Challenge>;

    /// Get a reference to the PCS used by this proof configuration.
    fn pcs(&self) -> &Self::Pcs;

    /// Get an initialisation of the challenger used by this proof configuration.
    fn initialise_challenger(&self) -> Self::Challenger;
}

#[derive(Debug)]
pub struct StarkConfig<Pcs, Challenge, ChallengerConstants, Challenger> {
    pcs: Pcs,
    challenger_constants: ChallengerConstants,
    init_challenger: fn(ChallengerConstants) -> Challenger,
    _phantom: PhantomData<Challenge>,
}

impl<Pcs, Challenge, ChallengerConstants, Challenger>
    StarkConfig<Pcs, Challenge, ChallengerConstants, Challenger>
{
    pub const fn new(
        pcs: Pcs,
        challenger_constants: ChallengerConstants,
        init_challenger: fn(ChallengerConstants) -> Challenger,
    ) -> Self {
        Self {
            pcs,
            challenger_constants,
            init_challenger,
            _phantom: PhantomData,
        }
    }
}

impl<Pcs, Challenge, ChallengerConstants, Challenger> StarkGenericConfig
    for StarkConfig<Pcs, Challenge, ChallengerConstants, Challenger>
where
    Challenge: ExtensionField<<Pcs::Domain as PolynomialSpace>::Val>,
    Pcs: p3_commit::Pcs<Challenge, Challenger>,
    Challenger: FieldChallenger<<Pcs::Domain as PolynomialSpace>::Val>
        + CanObserve<<Pcs as p3_commit::Pcs<Challenge, Challenger>>::Commitment>
        + CanSample<Challenge>,
    ChallengerConstants: Clone,
{
    type Pcs = Pcs;
    type Challenge = Challenge;
    type Challenger = Challenger;

    fn pcs(&self) -> &Self::Pcs {
        &self.pcs
    }

    fn initialise_challenger(&self) -> Self::Challenger {
        (self.init_challenger)(self.challenger_constants.clone())
    }
}
