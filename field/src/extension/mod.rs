use crate::field::Field;

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
