use core::marker::PhantomData;

use p3_challenger::FieldChallenger;
use p3_commit::UnivariatePcs;
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{AbstractExtensionField, ExtensionField, Field, PackedField, TwoAdicField};
use p3_matrix::dense::RowMajorMatrixView;

pub trait StarkConfig {
    /// A value of the trace.
    type Val: Field;

    /// The domain over which trace polynomials are defined.
    type Domain: ExtensionField<Self::Val> + TwoAdicField;

    /// The field from which most random challenges are drawn.
    type Challenge: ExtensionField<Self::Domain>;

    type PackedChallenge: PackedField<Scalar = Self::Challenge>
        + AbstractExtensionField<<Self::Domain as Field>::Packing>;

    /// The PCS used to commit to trace polynomials.
    type Pcs: for<'a> UnivariatePcs<Self::Val, RowMajorMatrixView<'a, Self::Val>, Self::Challenger>;

    type Dft: TwoAdicSubgroupDft<Self::Domain>;

    type Challenger: FieldChallenger<Self::Val>;

    fn pcs(&self) -> &Self::Pcs;

    fn dft(&self) -> &Self::Dft;
}

pub struct StarkConfigImpl<Val, Domain, Challenge, PackedChallenge, Pcs, Dft, Challenger> {
    pcs: Pcs,
    dft: Dft,
    _phantom_val: PhantomData<Val>,
    _phantom_domain: PhantomData<Domain>,
    _phantom_challenge: PhantomData<Challenge>,
    _phantom_packed_challenge: PhantomData<PackedChallenge>,
    _phantom_chal: PhantomData<Challenger>,
}

impl<Val, Domain, Challenge, PackedChallenge, Pcs, Dft, Challenger>
    StarkConfigImpl<Val, Domain, Challenge, PackedChallenge, Pcs, Dft, Challenger>
{
    pub fn new(pcs: Pcs, dft: Dft) -> Self {
        Self {
            pcs,
            dft,
            _phantom_val: PhantomData,
            _phantom_domain: PhantomData,
            _phantom_challenge: PhantomData,
            _phantom_packed_challenge: PhantomData,
            _phantom_chal: PhantomData,
        }
    }
}

impl<Val, Domain, Challenge, PackedChallenge, Pcs, Dft, Challenger> StarkConfig
    for StarkConfigImpl<Val, Domain, Challenge, PackedChallenge, Pcs, Dft, Challenger>
where
    Val: Field,
    Domain: ExtensionField<Val> + TwoAdicField,
    Challenge: ExtensionField<Domain>,
    PackedChallenge: PackedField<Scalar = Challenge> + AbstractExtensionField<Domain::Packing>,
    Pcs: for<'a> UnivariatePcs<Val, RowMajorMatrixView<'a, Val>, Challenger>,
    Dft: TwoAdicSubgroupDft<Domain>,
    Challenger: FieldChallenger<Val>,
{
    type Val = Val;
    type Domain = Domain;
    type Challenge = Challenge;
    type PackedChallenge = PackedChallenge;
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
