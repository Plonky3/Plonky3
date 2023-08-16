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
    type PCS: for<'a> UnivariatePcs<Self::Val, RowMajorMatrixView<'a, Self::Val>, Self::Chal>;

    type DFT: TwoAdicSubgroupDft<Self::Domain>;

    type Chal: FieldChallenger<Self::Val>;

    fn pcs(&self) -> &Self::PCS;

    fn dft(&self) -> &Self::DFT;
}

pub struct StarkConfigImpl<Val, Domain, Challenge, PackedChallenge, PCS, DFT, Chal> {
    pcs: PCS,
    dft: DFT,
    _phantom_val: PhantomData<Val>,
    _phantom_domain: PhantomData<Domain>,
    _phantom_challenge: PhantomData<Challenge>,
    _phantom_packed_challenge: PhantomData<PackedChallenge>,
    _phantom_chal: PhantomData<Chal>,
}

impl<Val, Domain, Challenge, PackedChallenge, PCS, DFT, Chal>
    StarkConfigImpl<Val, Domain, Challenge, PackedChallenge, PCS, DFT, Chal>
{
    pub fn new(pcs: PCS, dft: DFT) -> Self {
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

impl<Val, Domain, Challenge, PackedChallenge, PCS, DFT, Chal> StarkConfig
    for StarkConfigImpl<Val, Domain, Challenge, PackedChallenge, PCS, DFT, Chal>
where
    Val: Field,
    Domain: ExtensionField<Val> + TwoAdicField,
    Challenge: ExtensionField<Domain>,
    PackedChallenge: PackedField<Scalar = Challenge> + AbstractExtensionField<Domain::Packing>,
    PCS: for<'a> UnivariatePcs<Val, RowMajorMatrixView<'a, Val>, Chal>,
    DFT: TwoAdicSubgroupDft<Domain>,
    Chal: FieldChallenger<Val>,
{
    type Val = Val;
    type Domain = Domain;
    type Challenge = Challenge;
    type PackedChallenge = PackedChallenge;
    type PCS = PCS;
    type DFT = DFT;
    type Chal = Chal;

    fn pcs(&self) -> &Self::PCS {
        &self.pcs
    }

    fn dft(&self) -> &Self::DFT {
        &self.dft
    }
}
