//! Multi-STARK verification inside a circuit, over WHIR openings.

use alloc::vec::Vec;

use p3_air::{Air, BaseAir};
use p3_challenger::fs::TranscriptField;
use p3_challenger::{CanObserve, CanSampleUniformBits, FieldChallenger, GrindingChallenger};
use p3_commit::{ExtensionMmcs, Mmcs, MmcsArith};
use p3_field::{ExtensionField, PrimeField64};
use p3_lookup::InteractionSymbolicBuilder;
use p3_multi_stark::arith::{
    AirArith, AirInputs, MultiStarkArith, eval_air_symbolic, periodic_values_symbolic,
};
use p3_multi_stark::config::{Commitment, MultiStarkConfig};
use p3_multi_stark::folder::FolderEvaluations;
use p3_multi_stark::selectors::PeriodicError;
use p3_multilinear_util::point::Point;
use p3_sumcheck::layout::{Layout, observe_commitment_in};
use p3_sumcheck::{OpeningEvals, OpeningProtocol};
use p3_whir::{PcsProof, VerifierError, WhirDomain, WhirProver};

use super::stark::MirrorChallenger;
use crate::{Base, CircuitArith, Ext, PermutationChip};

impl<C, F, EF, P, Ch, Dft, MT, L> MultiStarkArith<C> for CircuitArith<F, EF, P>
where
    F: PrimeField64 + TranscriptField + Ord,
    EF: ExtensionField<F>,
    P: PermutationChip<F>,
    C: MultiStarkConfig<
            Val = F,
            Challenge = EF,
            Challenger = Ch,
            Pcs = WhirProver<EF, F, Dft, MT, Ch, L>,
        >,
    Ch: FieldChallenger<F>
        + GrindingChallenger<Witness = F>
        + CanSampleUniformBits<F>
        + CanObserve<<MT as Mmcs<F>>::Commitment>
        + MirrorChallenger<
            Self,
            Mirror: FieldChallenger<F>
                        + GrindingChallenger<Witness = F>
                        + CanSampleUniformBits<F>
                        + CanObserve<<MT as Mmcs<F>>::Commitment>,
        >,
    Dft: WhirDomain<F, EF>,
    MT: Mmcs<F>
        + MmcsArith<
            Self,
            Base<F>,
            Commitment = <MT as Mmcs<F>>::Commitment,
            MultiProof = <MT as Mmcs<F>>::MultiProof,
        >,
    for<'s> ExtensionMmcs<F, EF, &'s MT>: MmcsArith<
            Self,
            Ext<F, EF>,
            MultiProof = <MT as Mmcs<F>>::MultiProof,
            Target = <MT as MmcsArith<Self, Base<F>>>::Target,
        >,
    L: Layout<F, EF>,
{
    type Commitment = <MT as MmcsArith<Self, Base<F>>>::Target;
    type Challenger = <Ch as MirrorChallenger<Self>>::Mirror;

    fn load_commitment(&mut self, pcs: &C::Pcs, commitment: &Commitment<C>) -> Self::Commitment {
        pcs.mmcs().load_commitment(self, commitment)
    }

    fn load_key_commitment(
        &mut self,
        pcs: &C::Pcs,
        commitment: &Commitment<C>,
    ) -> Self::Commitment {
        pcs.mmcs().load_key_commitment(self, commitment)
    }

    fn will_observe_commitment(&mut self, pcs: &C::Pcs, commitment: &Self::Commitment) {
        pcs.mmcs().will_observe(self, commitment);
    }

    fn observe_commitment(
        &mut self,
        pcs: &C::Pcs,
        native: &Commitment<C>,
        commitment: &Self::Commitment,
        challenger: &mut Self::Challenger,
    ) {
        observe_commitment_in::<F, _, _, _>(self, challenger, native.clone(), |arith| {
            pcs.mmcs().will_observe(arith, commitment);
        });
    }

    fn verify_at(
        &mut self,
        pcs: &C::Pcs,
        commitment: &Self::Commitment,
        proof: &PcsProof<F, EF, MT>,
        protocol: &OpeningProtocol,
        points: &[Point<Ext<F, EF>>],
        challenger: &mut Self::Challenger,
    ) -> Result<Vec<OpeningEvals<Ext<F, EF>>>, VerifierError> {
        pcs.verify_at_in(self, commitment, proof, protocol, points, challenger)
    }
}

impl<F, EF, P, A> AirArith<A> for CircuitArith<F, EF, P>
where
    F: PrimeField64,
    EF: ExtensionField<F>,
    P: PermutationChip<F>,
    A: BaseAir<F> + Air<InteractionSymbolicBuilder<F, EF>>,
{
    fn eval_air(&mut self, air: &A, inputs: &AirInputs<'_, Self>) -> FolderEvaluations<Ext<F, EF>> {
        eval_air_symbolic(air, inputs)
    }

    fn periodic_values(
        &mut self,
        air: &A,
        point: &[Ext<F, EF>],
    ) -> Result<Vec<Ext<F, EF>>, PeriodicError> {
        periodic_values_symbolic(self, air, point)
    }
}
