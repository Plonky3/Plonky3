use core::marker::PhantomData;

use p3_challenger::FieldChallenger;
use p3_commit::UnivariatePCS;
use p3_field::{AbstractExtensionField, ExtensionField, Field, PackedField, TwoAdicField};
use p3_lde::TwoAdicCosetLDE;
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
    type PCS: for<'a> UnivariatePCS<Self::Val, RowMajorMatrixView<'a, Self::Val>, Self::Chal>;

    type LDE: TwoAdicCosetLDE<Self::Val, Self::Domain>;

    type Chal: FieldChallenger<Self::Val>;

    fn pcs(&self) -> &Self::PCS;

    fn lde(&self) -> &Self::LDE;
}

pub struct StarkConfigImpl<Val, Domain, Challenge, PackedChallenge, PCS, LDE, Chal> {
    pcs: PCS,
    lde: LDE,
    _phantom_val: PhantomData<Val>,
    _phantom_domain: PhantomData<Domain>,
    _phantom_challenge: PhantomData<Challenge>,
    _phantom_packed_challenge: PhantomData<PackedChallenge>,
    _phantom_chal: PhantomData<Chal>,
}

impl<Val, Domain, Challenge, PackedChallenge, PCS, LDE, Chal>
    StarkConfigImpl<Val, Domain, Challenge, PackedChallenge, PCS, LDE, Chal>
{
    pub fn new(pcs: PCS, lde: LDE) -> Self {
        Self {
            pcs,
            lde,
            _phantom_val: PhantomData,
            _phantom_domain: PhantomData,
            _phantom_challenge: PhantomData,
            _phantom_packed_challenge: PhantomData,
            _phantom_chal: PhantomData,
        }
    }
}

impl<Val, Domain, Challenge, PackedChallenge, PCS, LDE, Chal> StarkConfig
    for StarkConfigImpl<Val, Domain, Challenge, PackedChallenge, PCS, LDE, Chal>
where
    Val: Field,
    Domain: ExtensionField<Val> + TwoAdicField,
    Challenge: ExtensionField<Domain>,
    PackedChallenge: PackedField<Scalar = Challenge> + AbstractExtensionField<Domain::Packing>,
    PCS: for<'a> UnivariatePCS<Val, RowMajorMatrixView<'a, Val>, Chal>,
    LDE: TwoAdicCosetLDE<Val, Domain>,
    Chal: FieldChallenger<Val>,
{
    type Val = Val;
    type Domain = Domain;
    type Challenge = Challenge;
    type PackedChallenge = PackedChallenge;
    type PCS = PCS;
    type LDE = LDE;
    type Chal = Chal;

    fn pcs(&self) -> &Self::PCS {
        &self.pcs
    }

    fn lde(&self) -> &Self::LDE {
        &self.lde
    }
}
