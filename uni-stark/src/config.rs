use core::marker::PhantomData;

use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::{Pcs, UnivariatePcsWithLde};
use p3_field::{AbstractExtensionField, ExtensionField, Field, PackedField, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;

pub trait StarkConfig {
    /// The field over which trace data is encoded.
    type Val: Field;

    /// The domain over which trace polynomials are defined.
    type Domain: ExtensionField<Self::Val> + TwoAdicField;
    type PackedDomain: PackedField<Scalar = Self::Domain>;

    /// The field from which most random challenges are drawn.
    type Challenge: ExtensionField<Self::Val> + ExtensionField<Self::Domain> + TwoAdicField;
    type PackedChallenge: AbstractExtensionField<Self::PackedDomain, F = Self::Challenge>;

    /// The PCS used to commit to trace polynomials.
    type Pcs: UnivariatePcsWithLde<
        Self::Val,
        Self::Domain,
        Self::Challenge,
        RowMajorMatrix<Self::Val>,
        Self::Challenger,
    >;

    type Challenger: FieldChallenger<Self::Val>
        + CanObserve<<Self::Pcs as Pcs<Self::Val, RowMajorMatrix<Self::Val>>>::Commitment>;

    fn pcs(&self) -> &Self::Pcs;
}

pub struct StarkConfigImpl<Val, Domain, Challenge, PackedChallenge, Pcs, Challenger> {
    pcs: Pcs,
    _phantom: PhantomData<(Val, Domain, Challenge, PackedChallenge, Challenger)>,
}

impl<Val, Domain, Challenge, PackedChallenge, Pcs, Challenger>
    StarkConfigImpl<Val, Domain, Challenge, PackedChallenge, Pcs, Challenger>
{
    pub fn new(pcs: Pcs) -> Self {
        Self {
            pcs,
            _phantom: PhantomData,
        }
    }
}

impl<Val, Domain, Challenge, PackedChallenge, Pcs, Challenger> StarkConfig
    for StarkConfigImpl<Val, Domain, Challenge, PackedChallenge, Pcs, Challenger>
where
    Val: Field,
    Domain: ExtensionField<Val> + TwoAdicField,
    Challenge: ExtensionField<Val> + ExtensionField<Domain> + TwoAdicField,
    PackedChallenge: AbstractExtensionField<Domain::Packing, F = Challenge>,
    Pcs: UnivariatePcsWithLde<Val, Domain, Challenge, RowMajorMatrix<Val>, Challenger>,
    Challenger: FieldChallenger<Val>
        + CanObserve<<Pcs as p3_commit::Pcs<Val, RowMajorMatrix<Val>>>::Commitment>,
{
    type Val = Val;
    type Domain = Domain;
    type PackedDomain = Domain::Packing;
    type Challenge = Challenge;
    type PackedChallenge = PackedChallenge;
    type Pcs = Pcs;
    type Challenger = Challenger;

    fn pcs(&self) -> &Self::Pcs {
        &self.pcs
    }
}
