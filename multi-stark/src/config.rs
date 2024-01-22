use core::marker::PhantomData;

use p3_challenger::FieldChallenger;
use p3_commit::MultivariatePcs;
use p3_field::{AbstractExtensionField, ExtensionField, Field, PackedField};
use p3_matrix::dense::RowMajorMatrixView;

pub trait StarkConfig {
    /// A value of the trace.
    type Val: Field;

    /// The field from which most random challenges are drawn.
    type Challenge: ExtensionField<Self::Val>;

    type PackedChallenge: PackedField<Scalar = Self::Challenge>
        + AbstractExtensionField<<Self::Val as Field>::Packing>;

    /// The PCS used to commit to trace polynomials.
    type Pcs: for<'a> MultivariatePcs<
        Self::Val,
        Self::Challenge,
        RowMajorMatrixView<'a, Self::Val>,
        Self::Challenger,
    >;

    type Challenger: FieldChallenger<Self::Val>;

    fn pcs(&self) -> &Self::Pcs;
}

pub struct StarkConfigImpl<Val, Challenge, PackedChallenge, Pcs, Challenger> {
    pcs: Pcs,
    _phantom: PhantomData<(Val, Challenge, PackedChallenge, Challenger)>,
}

impl<Val, Challenge, PackedChallenge, Pcs, Challenger>
    StarkConfigImpl<Val, Challenge, PackedChallenge, Pcs, Challenger>
{
    pub fn new(pcs: Pcs) -> Self {
        Self {
            pcs,
            _phantom: PhantomData,
        }
    }
}

impl<Val, Challenge, PackedChallenge, Pcs, Challenger> StarkConfig
    for StarkConfigImpl<Val, Challenge, PackedChallenge, Pcs, Challenger>
where
    Val: Field,
    Challenge: ExtensionField<Val>,
    PackedChallenge: PackedField<Scalar = Challenge> + AbstractExtensionField<Val::Packing>,
    Pcs: for<'a> MultivariatePcs<Val, Challenge, RowMajorMatrixView<'a, Val>, Challenger>,
    Challenger: FieldChallenger<Val>,
{
    type Val = Val;
    type Challenge = Challenge;
    type PackedChallenge = PackedChallenge;
    type Pcs = Pcs;
    type Challenger = Challenger;

    fn pcs(&self) -> &Self::Pcs {
        &self.pcs
    }
}
