use p3_commit::PCS;
use p3_field::{FieldExtension, PrimeField, TwoAdicField};
use p3_lde::TwoAdicSubgroupLDE;

pub trait StarkConfig {
    type F: PrimeField;
    type Domain: FieldExtension<Base = Self::F, Extension: TwoAdicField>;
    type Challenge: FieldExtension<Base = Self::F>;
    type PCS: PCS<Self::F>;
    type LDE: TwoAdicSubgroupLDE<Self::Domain>;
}
