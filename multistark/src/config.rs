use hyperfield::field::{FieldExtension, PrimeField};
use hyperpcs::PCS;

pub trait Config {
    type F: PrimeField;
    type FE: FieldExtension<Self::F>;
    type PCS: PCS<Self::F>;
}
