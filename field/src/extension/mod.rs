use crate::field::Field;
use crate::ExtensionField;

mod binomial_extension;

pub use binomial_extension::*;

/// Binomial extension field trait.
/// A extension field with a irreducible polynomial X^d-W
/// such that the extension is `F[X]/(X^d-W)`.
pub trait BinomiallyExtendable<const D: usize>: Field {
    fn w() -> Self;

    // DTH_ROOT = W^((n - 1)/D).
    // n is the order of base field.
    // Only works when exists k such that n = kD + 1.
    fn dth_root() -> Self;

    fn ext_generator() -> [Self; D];
}

pub trait HasFrobenuis<F: Field>: ExtensionField<F> {
    fn frobenius(&self) -> Self;
    fn repeated_frobenius(&self, count: usize) -> Self;
    fn frobenius_inv(&self) -> Self;
}

/// Optional trait for implementing Two Adic Binomial Extension Field.
pub trait HasTwoAdicBionmialExtension<const D: usize>: BinomiallyExtendable<D> {
    const EXT_TWO_ADICITY: usize;

    fn ext_two_adic_generator(bits: usize) -> [Self; D];
}
