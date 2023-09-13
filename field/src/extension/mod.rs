use crate::field::Field;
use crate::ExtensionField;

pub mod cubic;
pub mod quadratic;

/// Binomial extension field trait.
/// A extension field with a irreducible polynomial X^d-W
/// such that the extension is `F[X]/(X^d-W)`.
pub trait OptimallyExtendable<const D: usize>: Field + Sized {
    const W: Self;
    const DTH_ROOT: Self;

    fn ext_multiplicative_group_generator() -> [Self; D];
}

impl<F: Field + ExtensionField<F>> OptimallyExtendable<1> for F {
    const W: Self = F::ONE;
    const DTH_ROOT: Self = F::ONE;

    fn ext_multiplicative_group_generator() -> [Self; 1] {
        [F::multiplicative_group_generator()]
    }
}
