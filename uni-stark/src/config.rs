use core::marker::PhantomData;

use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::{Pcs, UnivariatePcs};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{AbstractExtensionField, ExtensionField, Field, PackedField, TwoAdicField};
use p3_matrix::dense::RowMajorMatrixView;

pub trait StarkConfig {
    /// A value of the trace.
    type Val: Field;
    // type PackedVal: PackedField<Scalar = Self::Val>;

    /// The domain over which trace polynomials are defined.
    type Domain: ExtensionField<Self::Val> + TwoAdicField;
    type PackedDomain: PackedField<Scalar = Self::Domain>;

    /// The field from which most random challenges are drawn.
    type Challenge: ExtensionField<Self::Domain>;
    type PackedChallenge: PackedField<Scalar = Self::Challenge>
        + AbstractExtensionField<Self::PackedDomain>;

    /// The PCS used to commit to trace polynomials.
    type Pcs: for<'a> UnivariatePcs<Self::Val, RowMajorMatrixView<'a, Self::Val>, Self::Challenger>;

    type Dft: TwoAdicSubgroupDft<Self::Domain>;

    type Challenger: FieldChallenger<Self::Val> + FieldChallenger<Self::Domain>
        + for<'a> CanObserve<<Self::Pcs as Pcs<Self::Val, RowMajorMatrixView<'a, Self::Val>, Self::Challenger>>::Commitment>;

    fn pcs(&self) -> &Self::Pcs;

    fn dft(&self) -> &Self::Dft;
}

pub struct StarkConfigImpl<Val, Domain, Challenge, Pcs, Dft, Challenger> {
    pcs: Pcs,
    dft: Dft,
    _phantom_val: PhantomData<Val>,
    _phantom_domain: PhantomData<Domain>,
    _phantom_challenge: PhantomData<Challenge>,
    _phantom_chal: PhantomData<Challenger>,
}

impl<Val, Domain, Challenge, Pcs, Dft, Challenger>
    StarkConfigImpl<Val, Domain, Challenge, Pcs, Dft, Challenger>
{
    pub fn new(pcs: Pcs, dft: Dft) -> Self {
        Self {
            pcs,
            dft,
            _phantom_val: PhantomData,
            _phantom_domain: PhantomData,
            _phantom_challenge: PhantomData,
            _phantom_chal: PhantomData,
        }
    }
}

impl<Val, Domain, Challenge, Pcs, Dft, Challenger> StarkConfig
    for StarkConfigImpl<Val, Domain, Challenge, Pcs, Dft, Challenger>
where
    Val: Field,
    Domain: ExtensionField<Val> + TwoAdicField,
    Challenge: ExtensionField<Domain>,
    Challenge::Packing: AbstractExtensionField<Domain::Packing>,
    Pcs: for<'a> UnivariatePcs<Val, RowMajorMatrixView<'a, Val>, Challenger>,
    Dft: TwoAdicSubgroupDft<Domain>,
    Challenger: FieldChallenger<Val>
        + FieldChallenger<Domain>
        + for<'a> CanObserve<
            <Pcs as p3_commit::Pcs<Val, RowMajorMatrixView<'a, Val>, Challenger>>::Commitment,
        >,
{
    type Val = Val;
    // type PackedVal = Val::Packing;
    type Domain = Domain;
    type PackedDomain = Domain::Packing;
    type Challenge = Challenge;
    type PackedChallenge = Challenge::Packing;
    type Pcs = Pcs;
    type Dft = Dft;
    type Challenger = Challenger;

    fn pcs(&self) -> &Self::Pcs {
        &self.pcs
    }

    fn dft(&self) -> &Self::Dft {
        &self.dft
    }
}
