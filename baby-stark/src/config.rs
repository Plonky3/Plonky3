use p3_commit::PCS;
use p3_field::{AbstractExtensionField, PrimeField, TwoAdicField};
use p3_lde::TwoAdicCosetLDE;

pub trait StarkConfig {
    type F: PrimeField;
    type Domain: AbstractExtensionField<Self::F> + TwoAdicField;
    type Challenge: AbstractExtensionField<Self::F>;
    type PCS: PCS<Self::F>;
    type LDE: TwoAdicCosetLDE<Self::F, Self::Domain>;
}
