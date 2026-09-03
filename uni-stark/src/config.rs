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

pub type Com<SC> = <<SC as StarkGenericConfig>::Pcs as Pcs<
    <SC as StarkGenericConfig>::Challenge,
    <SC as StarkGenericConfig>::Challenger,
>>::Commitment;

/// Absorb a commitment into the transcript.
///
/// The prover and verifier pin `GrindingChallenger<Witness = Val<SC>>` so they
/// can grind before the out-of-domain point. That puts a second source of
/// `CanObserve<Val<SC>>` in scope, and `challenger.observe(commitment)` then
/// fails to resolve — method-call syntax has no way to name which `CanObserve`
/// impl is meant. This helper carries no grinding bound, so inside it the
/// choice is unambiguous.
pub fn observe_commitment<SC: StarkGenericConfig>(
    challenger: &mut SC::Challenger,
    commitment: Com<SC>,
) {
    challenger.observe(commitment);
}

pub type PackedVal<SC> = <Val<SC> as Field>::Packing;

pub type PackedChallenge<SC> =
    <<SC as StarkGenericConfig>::Challenge as ExtensionField<Val<SC>>>::ExtensionPacking;

pub trait StarkGenericConfig: Clone {
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

    /// Bits of proof of work the prover must grind before the lookup
    /// argument's challenges are sampled.
    ///
    /// The LogUp fingerprint round's error grows linearly in the trace length,
    /// so it is the one round whose bits degrade as the instance gets bigger
    /// and the one no query count can buy back. Grinding is its only lever.
    /// Credited to [`p3_security::GrindingSites::lookup_challenge`].
    ///
    /// A uni-STARK has no lookups and ignores this. `p3-batch-stark` reads it
    /// when it samples the shared `(alpha, beta)` pair.
    ///
    /// Defaults to `0`, which costs the prover nothing and leaves the
    /// transcript byte-for-byte as it was before this site existed.
    fn lookup_proof_of_work_bits(&self) -> usize {
        0
    }

    /// Bits of proof of work the prover must grind before the DEEP
    /// out-of-domain point `zeta` is sampled.
    ///
    /// The DEEP-ALI round tests the constraint identity at a single point, so
    /// its error is the identity's degree over the field size — a quantity that
    /// grows with the trace height and is fixed by the AIR, not by any
    /// parameter the prover can turn up. Grinding here is therefore the only
    /// way to buy that round bits, and on a small extension field it is the
    /// round that stops a proven-soundness target from being reached at any
    /// query count. Credited to
    /// [`p3_security::GrindingSites::out_of_domain`].
    ///
    /// Defaults to `0`, which costs the prover nothing and leaves the
    /// transcript byte-for-byte as it was before this site existed.
    fn deep_proof_of_work_bits(&self) -> usize {
        0
    }
}

#[derive(Clone, Debug)]
pub struct StarkConfig<Pcs, Challenge, Challenger> {
    /// The [`Pcs`] used to commit polynomials and produce opening proofs.
    pcs: Pcs,
    /// An initialised instance of the [`FieldChallenger`].
    challenger: Challenger,
    /// See [`StarkGenericConfig::lookup_proof_of_work_bits`].
    lookup_proof_of_work_bits: usize,
    /// See [`StarkGenericConfig::deep_proof_of_work_bits`].
    deep_proof_of_work_bits: usize,
    _phantom: PhantomData<Challenge>,
}

impl<Pcs: Clone, Challenge: Clone, Challenger: Clone> StarkConfig<Pcs, Challenge, Challenger> {
    /// A configuration that grinds at neither of the sites outside the
    /// low-degree test.
    ///
    /// Use [`Self::with_deep_proof_of_work_bits`] and
    /// [`Self::with_lookup_proof_of_work_bits`] to add them.
    pub const fn new(pcs: Pcs, challenger: Challenger) -> Self {
        Self {
            pcs,
            challenger,
            lookup_proof_of_work_bits: 0,
            deep_proof_of_work_bits: 0,
            _phantom: PhantomData,
        }
    }

    /// Grind `bits` before sampling the DEEP out-of-domain point.
    ///
    /// Both the prover and the verifier read this from the config, so the two
    /// must be configured identically — a mismatch is a rejected proof, not a
    /// silently weaker one.
    ///
    /// See [`StarkGenericConfig::deep_proof_of_work_bits`] for what the bits
    /// buy.
    #[must_use]
    pub const fn with_deep_proof_of_work_bits(mut self, bits: usize) -> Self {
        self.deep_proof_of_work_bits = bits;
        self
    }

    /// Grind `bits` before sampling the lookup argument's challenges.
    ///
    /// Both the prover and the verifier read this from the config, so the two
    /// must be configured identically — a mismatch is a rejected proof, not a
    /// silently weaker one.
    ///
    /// See [`StarkGenericConfig::lookup_proof_of_work_bits`] for what the bits
    /// buy.
    #[must_use]
    pub const fn with_lookup_proof_of_work_bits(mut self, bits: usize) -> Self {
        self.lookup_proof_of_work_bits = bits;
        self
    }
}

impl<Pcs, Challenge, Challenger> StarkGenericConfig for StarkConfig<Pcs, Challenge, Challenger>
where
    Challenge: ExtensionField<<Pcs::Domain as PolynomialSpace>::Val> + Clone,
    Pcs: p3_commit::Pcs<Challenge, Challenger> + Clone,
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

    fn lookup_proof_of_work_bits(&self) -> usize {
        self.lookup_proof_of_work_bits
    }

    fn deep_proof_of_work_bits(&self) -> usize {
        self.deep_proof_of_work_bits
    }
}
