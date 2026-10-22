//! Aggregation: one proof that several proofs verify.
//!
//! A node circuit verifies its children and publishes one digest of their public values.
//!
//! Nodes compose into a tree, and the root proof stands for every leaf.

use alloc::vec::Vec;

use p3_air::symbolic::SymbolicExpressionExt;
use p3_air::{Air, BaseAir};
use p3_batch_stark::{
    BatchProof, BatchVerificationError, CommonData, PcsError, StarkGenericConfig, Val,
    verify_batch_in,
};
use p3_challenger::GrindingChallenger;
use p3_field::{Algebra, ExtensionField, PrimeField64, TwoAdicField};
use p3_lookup::InteractionSymbolicBuilder;
use p3_verifier_arith::Lift;

use crate::chip::sponge;
use crate::{Base, Circuit, CircuitArith, CircuitConfig, KeyBinding, PermutationChip, Unsatisfied};

/// A proof a node verifies, with what its verifier holds.
pub struct Child<'a, SC: StarkGenericConfig, A> {
    /// The AIRs the proof is about.
    pub airs: &'a [A],
    /// The verifier's data about those AIRs, including their preprocessed commitment.
    pub common: &'a CommonData<SC>,
    /// The proof.
    pub proof: &'a BatchProof<SC>,
    /// The public values of every AIR.
    pub public_values: &'a [Vec<Val<SC>>],
}

/// The circuit that verifies every child.
///
/// Its public values are [`public_digest`] of the children's public values.
///
/// With [`KeyBinding::Public`], each child's verifying key comes first.
///
/// # Errors
///
/// When a child does not verify.
pub fn aggregate<SC, A, F, EF, P>(
    chip: P,
    config: &SC,
    keys: KeyBinding,
    children: &[Child<'_, SC, A>],
) -> Result<Circuit<F, EF, P>, BatchVerificationError<PcsError<SC>>>
where
    F: PrimeField64 + TwoAdicField,
    EF: ExtensionField<F>,
    P: PermutationChip<F>,
    SC: CircuitConfig<F, EF, P, Challenger: GrindingChallenger<Witness = F>>,
    A: BaseAir<F> + Air<InteractionSymbolicBuilder<F, EF>>,
    SymbolicExpressionExt<F, EF>: Algebra<EF>,
{
    let mut arith = CircuitArith::new(chip).with_key_binding(keys);
    let mut values: Vec<Base<F>> = Vec::new();
    for child in children {
        // The children's public values stay private and reach the public values only as a digest.
        let lifted: Vec<Vec<Base<F>>> = child
            .public_values
            .iter()
            .map(|row| row.iter().map(|&value| arith.load(value)).collect())
            .collect();
        verify_batch_in(
            &mut arith,
            config,
            child.airs,
            child.proof,
            &lifted,
            child.common,
        )?;
        values.extend(lifted.into_iter().flatten());
    }
    arith
        .with_builder(|builder| -> Result<(), Unsatisfied> {
            let digest = sponge(&values, P::WIDTH, P::WIDTH / 2, P::WIDTH / 2, |state| {
                builder.permute(state)
            });
            for value in digest {
                let public = builder.public(builder.value(&value));
                builder.assert_zero(&(public - value))?;
            }
            Ok(())
        })
        .expect("a public copy of a wire equals the wire");
    Ok(arith.build())
}

/// The digest a node publishes for its children's public values.
///
/// The values are hashed child by child and AIR by AIR.
///
/// The sponge has rate and output `WIDTH / 2`.
pub fn public_digest<F: PrimeField64, P: PermutationChip<F>>(
    chip: &P,
    public_values: &[&[Vec<F>]],
) -> Vec<F> {
    let values: Vec<F> = public_values
        .iter()
        .flat_map(|child| child.iter().flatten().copied())
        .collect();
    sponge(&values, P::WIDTH, P::WIDTH / 2, P::WIDTH / 2, |state| {
        chip.permute(state)
    })
}
