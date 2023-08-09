use core::marker::PhantomData;

use p3_commit::MultilinearPCS;
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
    type PCS: for<'a> MultilinearPCS<Self::Val, RowMajorMatrixView<'a, Self::Val>>;

    fn pcs(&self) -> &Self::PCS;
}

pub struct StarkConfigImpl<Val, Challenge, PackedChallenge, PCS> {
    pcs: PCS,
    _phantom_val: PhantomData<Val>,
    _phantom_challenge: PhantomData<Challenge>,
    _phantom_packed_challenge: PhantomData<PackedChallenge>,
}

impl<Val, Challenge, PackedChallenge, PCS> StarkConfigImpl<Val, Challenge, PackedChallenge, PCS> {
    pub fn new(pcs: PCS) -> Self {
        Self {
            pcs,
            _phantom_val: PhantomData,
            _phantom_challenge: PhantomData,
            _phantom_packed_challenge: PhantomData,
        }
    }
}

impl<Val, Challenge, PackedChallenge, PCS> StarkConfig
    for StarkConfigImpl<Val, Challenge, PackedChallenge, PCS>
where
    Val: Field,
    Challenge: ExtensionField<Val>,
    PackedChallenge: PackedField<Scalar = Challenge> + AbstractExtensionField<Val::Packing>,
    PCS: for<'a> MultilinearPCS<Val, RowMajorMatrixView<'a, Val>>,
{
    type Val = Val;
    type Challenge = Challenge;
    type PackedChallenge = PackedChallenge;
    type PCS = PCS;

    fn pcs(&self) -> &Self::PCS {
        &self.pcs
    }
}
