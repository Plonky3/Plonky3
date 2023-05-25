use core::marker::PhantomData;
use p3_commit::PCS;
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_lde::TwoAdicCosetLDE;

pub trait StarkConfig {
    type Val: Field;
    type Domain: ExtensionField<Self::Val> + TwoAdicField;
    type Challenge: ExtensionField<Self::Domain>;
    type PCS: PCS<Self::Val>;
    type LDE: TwoAdicCosetLDE<Self::Val, Self::Domain>;

    fn pcs(&self) -> &Self::PCS;
    fn lde(&self) -> &Self::LDE;
}

pub struct StarkConfigImpl<Val, Domain, Challenge, PCS, LDE> {
    _phantom_val: PhantomData<Val>,
    _phantom_domain: PhantomData<Domain>,
    _phantom_challenge: PhantomData<Challenge>,
    pcs: PCS,
    lde: LDE,
}

impl<Val, Domain, Challenge, PCS, LDE> StarkConfigImpl<Val, Domain, Challenge, PCS, LDE> {
    pub fn new(pcs: PCS, lde: LDE) -> Self {
        Self {
            _phantom_val: PhantomData,
            _phantom_domain: PhantomData,
            _phantom_challenge: PhantomData,
            pcs,
            lde,
        }
    }
}

impl<Val, Domain, Challenge, PCS, LDE> StarkConfig
    for StarkConfigImpl<Val, Domain, Challenge, PCS, LDE>
where
    Val: Field,
    Domain: ExtensionField<Val> + TwoAdicField,
    Challenge: ExtensionField<Domain>,
    PCS: p3_commit::PCS<Val>,
    LDE: TwoAdicCosetLDE<Val, Domain>,
{
    type Val = Val;
    type Domain = Domain;
    type Challenge = Challenge;
    type PCS = PCS;
    type LDE = LDE;

    fn pcs(&self) -> &Self::PCS {
        &self.pcs
    }

    fn lde(&self) -> &Self::LDE {
        &self.lde
    }
}
