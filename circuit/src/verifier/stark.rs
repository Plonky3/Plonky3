//! STARK verification inside a circuit: domains, openings and constraints.

use alloc::vec::Vec;

use p3_air::symbolic::{
    AirLayout, SymbolicAirBuilder, SymbolicExpressionExt, capture_symbolic_constraints,
};
use p3_air::{Air, BaseAir};
use p3_batch_stark::symbolic::{get_constraint_layout, get_symbolic_constraints};
use p3_batch_stark::{BatchArith, VerifierData};
use p3_challenger::{CanObserve, CanSample, DuplexChallenger, FieldChallenger, GrindingChallenger};
use p3_commit::{CommitmentOpening, LagrangeSelectors, Mmcs, MmcsArith, PeriodicColumns};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::coset::TwoAdicMultiplicativeCoset;
use p3_field::{
    Algebra, BasedVectorSpace, ExtensionField, Field, PrimeCharacteristicRing, PrimeField64,
    TwoAdicField,
};
use p3_fri::{BatchMultiOpening, FriProof, TwoAdicFriPcs};
use p3_lookup::InteractionSymbolicBuilder;
use p3_lookup::logup::LogUpGadget;
use p3_symmetric::CryptographicPermutation;
use p3_uni_stark::{
    Com, ConstraintArith, ConstraintInputs, Domain, InvalidProofShapeError, PcsError, PcsProof,
    StarkArith, StarkGenericConfig,
};
use p3_verifier_arith::two_adic::{periodic_values_at, selectors_at, vanishing_at};
use p3_verifier_arith::{Arith, recompose_ext};

use super::constraints::{Leaves, fold_constraints};
use crate::{Base, CircuitArith, DuplexMirror, Ext, PermutationChip};

/// A native challenger a circuit mirrors onto wires.
pub trait MirrorChallenger<A>: Sized {
    /// The challenger that runs natively and on wires in lockstep.
    type Mirror;

    /// The mirror of this challenger, starting from its current state.
    fn mirror(self, arith: &A) -> Self::Mirror;
}

impl<F, EF, P, Perm, const WIDTH: usize, const RATE: usize> MirrorChallenger<CircuitArith<F, EF, P>>
    for DuplexChallenger<F, Perm, WIDTH, RATE>
where
    F: PrimeField64,
    EF: ExtensionField<F>,
    P: PermutationChip<F>,
    Perm: CryptographicPermutation<[F; WIDTH]>,
{
    type Mirror = DuplexMirror<F, EF, P, Perm, WIDTH, RATE>;

    fn mirror(self, arith: &CircuitArith<F, EF, P>) -> Self::Mirror {
        DuplexMirror::new(self, arith.shared.clone())
    }
}

/// A PCS whose opening arguments a circuit verifies.
pub trait CircuitPcs<SC: StarkGenericConfig, A: Arith> {
    /// A commitment inside the circuit.
    type Target: Clone;

    /// The challenger the opening argument runs on.
    type Challenger;

    /// Lift a commitment the prover sent.
    fn load_commitment(&self, arith: &mut A, commitment: &Com<SC>) -> Self::Target;

    /// Lift a commitment the verifier holds.
    fn load_key_commitment(&self, arith: &mut A, commitment: &Com<SC>) -> Self::Target;

    /// Announce a commitment the native transcript absorbs next.
    fn will_observe(&self, arith: &mut A, commitment: &Self::Target);

    /// Verify an opening argument inside the circuit.
    ///
    /// # Errors
    ///
    /// When the opening argument does not verify.
    fn verify_opening(
        &self,
        arith: &mut A,
        claims: Vec<CommitmentOpening<A::EF, Self::Target, Domain<SC>>>,
        proof: &PcsProof<SC>,
        challenger: &mut Self::Challenger,
    ) -> Result<(), PcsError<SC>>;
}

impl<SC, F, EF, P, Dft, InputMmcs, FriMmcs> CircuitPcs<SC, CircuitArith<F, EF, P>>
    for TwoAdicFriPcs<F, Dft, InputMmcs, FriMmcs>
where
    F: PrimeField64 + TwoAdicField,
    EF: ExtensionField<F>,
    P: PermutationChip<F>,
    Dft: TwoAdicSubgroupDft<F> + Sync,
    SC: StarkGenericConfig<Pcs = Self, Challenge = EF>,
    SC::Challenger: MirrorChallenger<CircuitArith<F, EF, P>>,
    <SC::Challenger as MirrorChallenger<CircuitArith<F, EF, P>>>::Mirror: FieldChallenger<F>
        + GrindingChallenger<Witness = F>
        + CanObserve<<FriMmcs as Mmcs<EF>>::Commitment>,
    InputMmcs: Mmcs<F, MultiProof: Sync, Error: Sync>
        + MmcsArith<
            CircuitArith<F, EF, P>,
            Base<F>,
            Commitment = <InputMmcs as Mmcs<F>>::Commitment,
            MultiProof = <InputMmcs as Mmcs<F>>::MultiProof,
            Error = <InputMmcs as Mmcs<F>>::Error,
        >,
    FriMmcs: Mmcs<EF>
        + MmcsArith<
            CircuitArith<F, EF, P>,
            Ext<F, EF>,
            Commitment = <FriMmcs as Mmcs<EF>>::Commitment,
            MultiProof = <FriMmcs as Mmcs<EF>>::MultiProof,
            Error = <FriMmcs as Mmcs<EF>>::Error,
        >,
    Self: p3_commit::Pcs<
            EF,
            SC::Challenger,
            Domain = TwoAdicMultiplicativeCoset<F>,
            Commitment = <InputMmcs as Mmcs<F>>::Commitment,
            Proof = FriProof<EF, FriMmcs, F, Vec<BatchMultiOpening<F, InputMmcs>>>,
            Error = p3_fri::verifier::FriError<
                <FriMmcs as Mmcs<EF>>::Error,
                <InputMmcs as Mmcs<F>>::Error,
            >,
        >,
{
    type Target = <InputMmcs as MmcsArith<CircuitArith<F, EF, P>, Base<F>>>::Target;
    type Challenger = <SC::Challenger as MirrorChallenger<CircuitArith<F, EF, P>>>::Mirror;

    fn load_commitment(
        &self,
        arith: &mut CircuitArith<F, EF, P>,
        commitment: &Com<SC>,
    ) -> Self::Target {
        self.mmcs().load_commitment(arith, commitment)
    }

    fn load_key_commitment(
        &self,
        arith: &mut CircuitArith<F, EF, P>,
        commitment: &Com<SC>,
    ) -> Self::Target {
        self.mmcs().load_key_commitment(arith, commitment)
    }

    fn will_observe(&self, arith: &mut CircuitArith<F, EF, P>, commitment: &Self::Target) {
        self.mmcs().will_observe(arith, commitment);
    }

    fn verify_opening(
        &self,
        arith: &mut CircuitArith<F, EF, P>,
        claims: Vec<CommitmentOpening<Ext<F, EF>, Self::Target, Domain<SC>>>,
        proof: &PcsProof<SC>,
        challenger: &mut Self::Challenger,
    ) -> Result<(), PcsError<SC>> {
        self.verify_in(arith, &claims, proof, challenger)
    }
}

/// A STARK configuration whose proofs a circuit verifies.
///
/// Its PCS opens over two-adic cosets inside the circuit, and its challenger has a mirror.
pub trait CircuitConfig<
    F: PrimeField64 + TwoAdicField,
    EF: ExtensionField<F>,
    P: PermutationChip<F>,
>:
    StarkGenericConfig<
        Challenge = EF,
        Pcs: p3_commit::Pcs<EF, Self::Challenger, Domain = TwoAdicMultiplicativeCoset<F>>
                 + CircuitPcs<
            Self,
            CircuitArith<F, EF, P>,
            Challenger = <Self::Challenger as MirrorChallenger<CircuitArith<F, EF, P>>>::Mirror,
        >,
        Challenger: MirrorChallenger<
            CircuitArith<F, EF, P>,
            Mirror: FieldChallenger<F>
                        + GrindingChallenger<Witness = F>
                        + CanObserve<Com<Self>>
                        + CanSample<EF>,
        >,
    >
{
}

impl<SC, F, EF, P> CircuitConfig<F, EF, P> for SC
where
    F: PrimeField64 + TwoAdicField,
    EF: ExtensionField<F>,
    P: PermutationChip<F>,
    SC: StarkGenericConfig<
            Challenge = EF,
            Pcs: p3_commit::Pcs<EF, SC::Challenger, Domain = TwoAdicMultiplicativeCoset<F>>
                     + CircuitPcs<
                SC,
                CircuitArith<F, EF, P>,
                Challenger = <SC::Challenger as MirrorChallenger<CircuitArith<F, EF, P>>>::Mirror,
            >,
            Challenger: MirrorChallenger<
                CircuitArith<F, EF, P>,
                Mirror: FieldChallenger<F>
                            + GrindingChallenger<Witness = F>
                            + CanObserve<Com<SC>>
                            + CanSample<EF>,
            >,
        >,
{
}

/// The quotient at `zeta`, recomposed from chunks over two-adic cosets.
///
/// Each chunk is scaled by the other domains' vanishing polynomials, normalised at its first point.
///
/// The denominators are constants of the domains, so no inverse is hinted.
fn recompose_quotient<A>(
    domains: &[TwoAdicMultiplicativeCoset<A::Val>],
    chunks: &[Vec<A::EF>],
    zeta: &A::EF,
) -> Result<A::EF, InvalidProofShapeError>
where
    A: Arith<Val: TwoAdicField>,
{
    if chunks.len() != domains.len()
        || chunks
            .iter()
            .any(|chunk| chunk.len() != A::Challenge::DIMENSION)
    {
        return Err(InvalidProofShapeError::OpenedValuesDimensionMismatch);
    }
    let vanishing: Vec<A::EF> = domains.iter().map(|d| vanishing_at::<A>(d, zeta)).collect();
    Ok(domains
        .iter()
        .zip(chunks)
        .enumerate()
        .map(|(i, (domain, chunk))| {
            let first = domain.shift();
            let (numerator, denominator) = domains
                .iter()
                .zip(&vanishing)
                .enumerate()
                .filter(|(j, _)| *j != i)
                .fold((A::EF::ONE, A::Val::ONE), |(num, den), (_, (other, z))| {
                    let at_first = (first * other.shift_inverse()).exp_power_of_2(other.log_size())
                        - A::Val::ONE;
                    (num * z.clone(), den * at_first)
                });
            numerator * A::F::from(denominator.inverse()) * recompose_ext::<A>(chunk)
        })
        .sum())
}

impl<SC, F, EF, P> StarkArith<SC> for CircuitArith<F, EF, P>
where
    F: PrimeField64 + TwoAdicField,
    EF: ExtensionField<F>,
    P: PermutationChip<F>,
    SC: CircuitConfig<F, EF, P>,
{
    type Commitment = <SC::Pcs as CircuitPcs<SC, Self>>::Target;
    type Challenger = <SC::Challenger as MirrorChallenger<Self>>::Mirror;

    fn challenger(&mut self, config: &SC) -> Self::Challenger {
        config.initialise_challenger().mirror(self)
    }

    fn load_commitment(&mut self, pcs: &SC::Pcs, commitment: &Com<SC>) -> Self::Commitment {
        pcs.load_commitment(self, commitment)
    }

    fn load_key_commitment(&mut self, pcs: &SC::Pcs, commitment: &Com<SC>) -> Self::Commitment {
        pcs.load_key_commitment(self, commitment)
    }

    fn will_observe_commitment(&mut self, pcs: &SC::Pcs, commitment: &Self::Commitment) {
        pcs.will_observe(self, commitment);
    }

    fn selectors(
        &mut self,
        domain: &Domain<SC>,
        point: &Ext<F, EF>,
    ) -> Option<LagrangeSelectors<Ext<F, EF>>> {
        selectors_at(self, domain, point, || ()).ok()
    }

    fn next_point(&mut self, domain: &Domain<SC>, point: &Ext<F, EF>) -> Option<Ext<F, EF>> {
        Some(point.clone() * Base::from(domain.subgroup_generator()))
    }

    fn periodic_values(
        &mut self,
        domain: &Domain<SC>,
        columns: PeriodicColumns<'_, F>,
        point: &Ext<F, EF>,
    ) -> Vec<Ext<F, EF>> {
        periodic_values_at(self, domain, columns.as_slice(), point, || ()).unwrap_or_else(|()| {
            panic!("the out-of-domain point lies outside every periodic coset")
        })
    }

    fn recompose_quotient(
        &mut self,
        domains: &[Domain<SC>],
        chunks: &[Vec<Ext<F, EF>>],
        point: &Ext<F, EF>,
    ) -> Result<Ext<F, EF>, InvalidProofShapeError> {
        recompose_quotient::<Self>(domains, chunks, point)
    }

    fn verify_opening(
        &mut self,
        pcs: &SC::Pcs,
        claims: Vec<CommitmentOpening<Ext<F, EF>, Self::Commitment, Domain<SC>>>,
        proof: &PcsProof<SC>,
        challenger: &mut Self::Challenger,
        _preprocessed: Option<usize>,
    ) -> Result<(), PcsError<SC>> {
        pcs.verify_opening(self, claims, proof, challenger)
    }
}

impl<SC, A, F, EF, P> ConstraintArith<SC, A> for CircuitArith<F, EF, P>
where
    F: PrimeField64 + TwoAdicField,
    EF: ExtensionField<F>,
    P: PermutationChip<F>,
    SC: CircuitConfig<F, EF, P>,
    A: BaseAir<F> + Air<SymbolicAirBuilder<F, EF>>,
{
    fn fold_constraints(&mut self, air: &A, inputs: &ConstraintInputs<'_, Self>) -> Ext<F, EF> {
        let layout = AirLayout {
            preprocessed_width: inputs.preprocessed_local.len(),
            main_width: air.width(),
            num_public_values: air.num_public_values(),
            num_periodic_columns: air.num_periodic_columns(),
            ..Default::default()
        };
        let constraints =
            capture_symbolic_constraints::<F, EF>(layout, |builder| air.eval(builder));
        let leaves = Leaves {
            trace_local: inputs.trace_local,
            trace_next: inputs.trace_next,
            preprocessed_local: inputs.preprocessed_local,
            preprocessed_next: inputs.preprocessed_next,
            periodic_values: inputs.periodic_values,
            public_values: inputs.public_values,
            selectors: inputs.selectors,
            permutation_local: &[],
            permutation_next: &[],
            permutation_challenges: &[],
            permutation_values: &[],
        };
        fold_constraints(
            constraints.base_constraints(),
            constraints.extension_constraints(),
            constraints.constraint_layout(),
            &leaves,
            &inputs.alpha,
        )
    }
}

impl<SC, A, F, EF, P> BatchArith<SC, A> for CircuitArith<F, EF, P>
where
    F: PrimeField64 + TwoAdicField,
    EF: ExtensionField<F>,
    P: PermutationChip<F>,
    SC: CircuitConfig<F, EF, P>,
    A: BaseAir<F> + Air<InteractionSymbolicBuilder<F, EF>>,
    SymbolicExpressionExt<F, EF>: Algebra<EF>,
{
    fn fold_constraints(
        &mut self,
        air: &A,
        data: &VerifierData<'_, Self>,
        lookup_gadget: &LogUpGadget,
    ) -> Ext<F, EF> {
        let inputs = &data.inputs;
        let layout = AirLayout {
            preprocessed_width: inputs.preprocessed_local.len(),
            main_width: air.width(),
            num_public_values: air.num_public_values(),
            num_periodic_columns: air.num_periodic_columns(),
            ..Default::default()
        };
        let (base, ext) = get_symbolic_constraints(air, layout, data.lookups, lookup_gadget);
        let constraint_layout = get_constraint_layout(air, layout, data.lookups, lookup_gadget);
        let leaves = Leaves {
            trace_local: inputs.trace_local,
            trace_next: inputs.trace_next,
            preprocessed_local: inputs.preprocessed_local,
            preprocessed_next: inputs.preprocessed_next,
            periodic_values: inputs.periodic_values,
            public_values: inputs.public_values,
            selectors: inputs.selectors,
            permutation_local: data.permutation_local,
            permutation_next: data.permutation_next,
            permutation_challenges: data.permutation_challenges,
            permutation_values: data.permutation_values,
        };
        fold_constraints(&base, &ext, &constraint_layout, &leaves, &inputs.alpha)
    }
}
