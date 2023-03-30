use p3_field::field::{FieldExtension, PrimeField};
use p3_pcs::UnivariatePCS;

pub trait Config {
    /// The field over which trace data is encoded.
    type F: PrimeField;
    /// The field from which the verifier draws random challenges.
    type FE: FieldExtension<Base = Self::F>;
    /// The polynomial commitment scheme used.
    type PCS: UnivariatePCS<Self::F>;
}
