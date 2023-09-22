use crate::field::Field;
use crate::AbstractExtensionField;

pub mod cubic;
pub mod quadratic;
pub mod quintic;
pub mod tesseractic;

/// Binomial extension field trait.
/// A extension field with a irreducible polynomial X^d-W
/// such that the extension is `F[X]/(X^d-W)`.
pub trait BinomiallyExtendable<const D: usize>: Field + Sized {
    const W: Self;
    fn ext_multiplicative_group_generator() -> [Self; D];
}

/// Trait for defining frobenuis endomorphism of extension field.
/// An bionomial extension field with a prime base field.
pub trait HasFrobenuis<const D: usize>: BinomiallyExtendable<D> {
    // DTH_ROOT = W^((n - 1)/D)
    // n is the order of base field.
    const DTH_ROOT: Self;
}

pub trait Frobenius<F: HasFrobenuis<D>, const D: usize>:
    Field + Sized + AbstractExtensionField<F>
{
    /// FrobeniusField automorphisms: x -> x^p, where p is the order of BaseField.
    fn frobenius(&self) -> Self {
        self.repeated_frobenius(1)
    }

    /// Repeated Frobenius automorphisms: x -> x^(p^count).
    ///
    /// Follows precomputation suggestion in Section 11.3.3 of the
    /// Handbook of Elliptic and Hyperelliptic Curve Cryptography.
    fn repeated_frobenius(&self, count: usize) -> Self {
        if count == 0 {
            return *self;
        } else if count >= D {
            // x |-> x^(p^D) is the identity, so x^(p^count) ==
            // x^(p^(count % D))
            return self.repeated_frobenius(count % D);
        }
        let arr: &[F] = self.as_base_slice();

        // z0 = DTH_ROOT^count = W^(k * count) where k = floor((p-1)/D)
        let mut z0 = F::DTH_ROOT;
        for _ in 1..count {
            z0 *= F::DTH_ROOT;
        }

        let mut res = [F::ZERO; D];
        for (i, z) in z0.powers().take(D).enumerate() {
            res[i] = arr[i] * z;
        }

        Self::from_base_slice(&res)
    }
}
