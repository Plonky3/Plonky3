use core::marker::PhantomData;

use p3_challenger::FieldChallenger;
use p3_commit::MultivariatePCS;
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
    type PCS: for<'a> MultivariatePCS<Self::Val, RowMajorMatrixView<'a, Self::Val>, Self::Chal>;

    type Chal: FieldChallenger<Self::Val>;

    fn pcs(&self) -> &Self::PCS;
}

pub struct StarkConfigImpl<Val, Challenge, PackedChallenge, PCS, Chal> {
    pcs: PCS,
    _phantom_val: PhantomData<Val>,
    _phantom_challenge: PhantomData<Challenge>,
    _phantom_packed_challenge: PhantomData<PackedChallenge>,
    _phantom_chal: PhantomData<Chal>,
}

impl<Val, Challenge, PackedChallenge, PCS, Chal>
    StarkConfigImpl<Val, Challenge, PackedChallenge, PCS, Chal>
{
    pub fn new(pcs: PCS) -> Self {
        Self {
            pcs,
            _phantom_val: PhantomData,
            _phantom_challenge: PhantomData,
            _phantom_packed_challenge: PhantomData,
            _phantom_chal: PhantomData,
        }
    }
}

impl<Val, Challenge, PackedChallenge, PCS, Chal> StarkConfig
    for StarkConfigImpl<Val, Challenge, PackedChallenge, PCS, Chal>
where
    Val: Field,
    Challenge: ExtensionField<Val>,
    PackedChallenge: PackedField<Scalar = Challenge> + AbstractExtensionField<Val::Packing>,
    PCS: for<'a> MultivariatePCS<Val, RowMajorMatrixView<'a, Val>, Chal>,
    Chal: FieldChallenger<Val>,
{
    type Val = Val;
    type Challenge = Challenge;
    type PackedChallenge = PackedChallenge;
    type PCS = PCS;
    type Chal = Chal;

    fn pcs(&self) -> &Self::PCS {
        &self.pcs
    }
}
