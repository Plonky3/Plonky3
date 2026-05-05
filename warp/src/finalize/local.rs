//! Local-only finalizer that wraps the existing [`WarpDecider`].
//!
//! `finalize` runs the four decider checks on `(acc.x, acc.w)` and returns
//! `Ok(())` on success. There is no transmissible proof — `verify` always
//! errors with [`FinalizerError::NoTransmissibleProof`].
//!
//! Use this when the party holding the accumulator IS the verifier of last
//! resort (single-prover IVC, in-process verification, etc.).

use core::marker::PhantomData;

use p3_commit::Mmcs;
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, TwoAdicField};

use crate::accumulator::{AccumulatorInstance, AccumulatorWitness};
use crate::code::ReedSolomonCode;
use crate::error::FinalizerError;
use crate::protocol::WarpDecider;
use crate::protocol::prover::ExtProverData;
use crate::relation::BundledPesat;

use super::Finalizer;

/// Local-only finalizer wrapping [`WarpDecider`].
pub struct LocalDeciderFinalizer<'a, F, EF, MT, Dft, Pesat>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    MT: Mmcs<F>,
    Dft: TwoAdicSubgroupDft<F>,
    Pesat: BundledPesat<F, EF>,
{
    decider: WarpDecider<'a, F, EF, MT, Dft, Pesat>,
    _ph: PhantomData<(F, EF)>,
}

impl<'a, F, EF, MT, Dft, Pesat> LocalDeciderFinalizer<'a, F, EF, MT, Dft, Pesat>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    MT: Mmcs<F>,
    Dft: TwoAdicSubgroupDft<F>,
    Pesat: BundledPesat<F, EF>,
    MT::Commitment: PartialEq,
{
    /// Build a finalizer over the given decider components.
    pub fn new(mmcs: &'a MT, code: &'a ReedSolomonCode<F, Dft>, pesat: &'a Pesat) -> Self {
        Self {
            decider: WarpDecider::new(mmcs, code, pesat),
            _ph: PhantomData,
        }
    }
}

impl<'a, F, EF, MT, Dft, Pesat> Finalizer<F, EF, MT, ExtProverData<F, EF, MT>>
    for LocalDeciderFinalizer<'a, F, EF, MT, Dft, Pesat>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    MT: Mmcs<F>,
    Dft: TwoAdicSubgroupDft<F>,
    Pesat: BundledPesat<F, EF>,
    MT::Commitment: PartialEq,
{
    type Proof = ();

    fn finalize(
        &self,
        instance: &AccumulatorInstance<EF, MT::Commitment>,
        witness: &AccumulatorWitness<EF, ExtProverData<F, EF, MT>>,
    ) -> Result<Self::Proof, FinalizerError> {
        self.decider.decide(instance, witness)?;
        Ok(())
    }

    fn verify(
        &self,
        _instance: &AccumulatorInstance<EF, MT::Commitment>,
        _proof: &Self::Proof,
    ) -> Result<(), FinalizerError> {
        // No transmissible proof — the verifier must hold the witness side
        // and call `finalize` directly. This is the documented behaviour.
        Err(FinalizerError::NoTransmissibleProof)
    }
}
