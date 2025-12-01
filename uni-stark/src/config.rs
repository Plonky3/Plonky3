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
    /// The [`Pcs`] implementation used to commit to trace polynomials.
    type Pcs: Pcs<Self::Challenge, Self::Challenger>;

    /// The [`ExtensionField`] from which most random challenges are drawn.
    type Challenge: ExtensionField<Val<Self>>;

    /// The [`FieldChallenger`] (Fiat-Shamir) implementation used.
    type Challenger: FieldChallenger<Val<Self>>
        + CanObserve<<Self::Pcs as Pcs<Self::Challenge, Self::Challenger>>::Commitment>
        + CanSample<Self::Challenge>;

    /// Get a reference to the [`Pcs`] used by this proof configuration.
    fn pcs(&self) -> &Self::Pcs;

    /// Get an initialisation of the [`FieldChallenger`] used by this proof configuration.
    fn initialise_challenger(&self) -> Self::Challenger;

    /// Returns 1 if the [`Pcs`] is zero-knowledge, 0 otherwise.
    fn is_zk(&self) -> usize {
        Self::Pcs::ZK as usize
    }
}

#[derive(Debug)]
pub struct StarkConfig<Pcs, Challenge, Challenger> {
    /// The [`Pcs`] used to commit polynomials and produce opening proofs.
    pcs: Pcs,
    /// An initialised instance of the [`FieldChallenger`].
    challenger: Challenger,
    _phantom: PhantomData<Challenge>,
}

impl<Pcs, Challenge, Challenger> StarkConfig<Pcs, Challenge, Challenger> {
    pub const fn new(pcs: Pcs, challenger: Challenger) -> Self {
        Self {
            pcs,
            challenger,
            _phantom: PhantomData,
        }
    }
}

impl<Pcs, Challenge, Challenger> StarkGenericConfig for StarkConfig<Pcs, Challenge, Challenger>
where
    Challenge: ExtensionField<<Pcs::Domain as PolynomialSpace>::Val>,
    Pcs: p3_commit::Pcs<Challenge, Challenger>,
    Challenger: FieldChallenger<<Pcs::Domain as PolynomialSpace>::Val>
        + CanObserve<Pcs::Commitment>
        + CanSample<Challenge>
        + Clone,
{
    type Pcs = Pcs;
    type Challenge = Challenge;
    type Challenger = Challenger;

    fn pcs(&self) -> &Self::Pcs {
        &self.pcs
    }

    fn initialise_challenger(&self) -> Self::Challenger {
        self.challenger.clone()
    }
}
