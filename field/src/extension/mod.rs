use crate::field::Field;
use crate::ExtensionField;

pub mod cubic;
pub mod quadratic;

/// Optimal extension field trait.
/// A degree `d` field extension is optimal if there exists a base field element `W`,
/// such that the extension is `F[X]/(X^d-W)`.
#[allow(clippy::upper_case_acronyms)]

pub trait OEF<Base: Field>: ExtensionField<Base> {
    // Element W of BaseField, such that `X^d - W` is irreducible over BaseField.
    const W: Base;

    // Element of BaseField such that DTH_ROOT^D == 1. Implementors
    // should set this to W^((p - 1)/D), where W is as above and p is
    // the order of the BaseField.
    const DTH_ROOT: Base;
}

impl<F: Field> OEF<F> for F {
    const W: F = F::ONE;
    const DTH_ROOT: F = F::ONE;
}

pub trait OptimallyExtendable<const D: usize>: Field + Sized {
    type Extension: Field + OEF<Self> + From<Self>;

    const W: Self;

    const DTH_ROOT: Self;

    fn ext_multiplicate_group_generator() -> [Self; D];
}

impl<F: Field + ExtensionField<F>> OptimallyExtendable<1> for F {
    type Extension = F;
    const W: Self = F::ONE;
    const DTH_ROOT: Self = F::ONE;

    fn ext_multiplicate_group_generator() -> [Self; 1] {
        [F::multiplicative_group_generator()]
    }
}
