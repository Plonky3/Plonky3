use crate::field::Field;
use crate::ExtensionField;

pub mod cubic;
pub mod quadratic;

/// Optimal extension field trait.
/// A degree `d` field extension is optimal if there exists a base field element `W`,
/// such that the extension is `F[X]/(X^d-W)`.

pub trait OptimallyExtendable<const D: usize>: Field + Sized {
    type Extension: Field + From<Self>;

    const W: Self;

    const DTH_ROOT: Self;

    fn ext_multiplicative_group_generator() -> [Self; D];
}

impl<F: Field + ExtensionField<F>> OptimallyExtendable<1> for F {
    type Extension = F;
    const W: Self = F::ONE;
    const DTH_ROOT: Self = F::ONE;

    fn ext_multiplicative_group_generator() -> [Self; 1] {
        [F::multiplicative_group_generator()]
    }
}
