use core::marker::PhantomData;

use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::{Pcs, UnivariatePcsWithLde};
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;

pub type PackedVal<SC> = <<SC as StarkGenericConfig>::Val as Field>::Packing;
pub type PackedChallenge<SC> = <<SC as StarkGenericConfig>::Challenge as ExtensionField<
    <SC as StarkGenericConfig>::Val,
>>::ExtensionPacking;

pub trait StarkGenericConfig {
    /// The field over which trace data is encoded.
    type Val: TwoAdicField;

    /// The field from which most random challenges are drawn.
    type Challenge: ExtensionField<Self::Val> + TwoAdicField;

    /// The PCS used to commit to trace polynomials.
    type Pcs: UnivariatePcsWithLde<
        Self::Val,
        Self::Challenge,
        RowMajorMatrix<Self::Val>,
        Self::Challenger,
    >;

    /// The challenger (Fiat-Shamir) implementation used.
    type Challenger: FieldChallenger<Self::Val>
        + CanObserve<<Self::Pcs as Pcs<Self::Val, RowMajorMatrix<Self::Val>>>::Commitment>;

    fn pcs(&self) -> &Self::Pcs;
}

pub struct StarkConfig<Val, Challenge, Pcs, Challenger> {
    pcs: Pcs,
    _phantom: PhantomData<(Val, Challenge, Challenger)>,
}

impl<Val, Challenge, Pcs, Challenger> StarkConfig<Val, Challenge, Pcs, Challenger> {
    pub fn new(pcs: Pcs) -> Self {
        Self {
            pcs,
            _phantom: PhantomData,
        }
    }
}

impl<Val, Challenge, Pcs, Challenger> StarkGenericConfig
    for StarkConfig<Val, Challenge, Pcs, Challenger>
where
    Val: TwoAdicField,
    Challenge: ExtensionField<Val> + TwoAdicField,
    Pcs: UnivariatePcsWithLde<Val, Challenge, RowMajorMatrix<Val>, Challenger>,
    Challenger: FieldChallenger<Val>
        + CanObserve<<Pcs as p3_commit::Pcs<Val, RowMajorMatrix<Val>>>::Commitment>,
{
    type Val = Val;
    type Challenge = Challenge;
    type Pcs = Pcs;
    type Challenger = Challenger;

    fn pcs(&self) -> &Self::Pcs {
        &self.pcs
    }
}
