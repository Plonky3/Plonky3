use p3_commit::PCS;
use p3_field::{Field, PrimeField, TwoAdicField};
use p3_lde::TwoAdicLDE;

pub trait StarkConfig {
    type F: PrimeField;
    type Domain: Field<Base = Self::F> + TwoAdicField;
    type Challenge: Field<Base = Self::F>;
    type PCS: PCS<Self::F>;
    type LDE: TwoAdicLDE<Self::F, Self::Domain>;
}
