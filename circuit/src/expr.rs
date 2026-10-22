//! Circuit values: constants, or expressions over wires.
//!
//! Ring operations build a shared graph and never touch the circuit.
//!
//! The builder turns a value into table rows only when the value is constrained.

use alloc::sync::Arc;
use core::fmt::Debug;
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use p3_field::{Algebra, Dup, ExtensionField, Field, PrimeCharacteristicRing};

use crate::Wire;

/// A base-field value of a circuit.
#[derive(Clone, Debug)]
pub struct Base<F>(pub(crate) BaseRepr<F>);

/// The two forms a base-field value takes.
#[derive(Clone, Debug)]
pub(crate) enum BaseRepr<F> {
    /// A value fixed by the circuit.
    Const(F),
    /// A value computed from wires.
    Node(Arc<Node<Base<F>>>),
}

/// A challenge-field value of a circuit.
#[derive(Clone, Debug)]
pub struct Ext<F, EF>(pub(crate) ExtRepr<F, EF>);

/// The two forms a challenge-field value takes.
#[derive(Clone, Debug)]
pub(crate) enum ExtRepr<F, EF> {
    /// A value fixed by the circuit.
    Const(EF),
    /// A value computed from wires.
    Node(Arc<Node<Ext<F, EF>>>),
    /// A base-field value used in the challenge field.
    Base(Base<F>),
}

/// One operation of a value graph.
#[derive(Debug)]
pub(crate) enum Node<V> {
    /// The value a wire carries.
    Wire(Wire),
    /// `lhs + rhs`.
    Add(V, V),
    /// `lhs - rhs`.
    Sub(V, V),
    /// `lhs * rhs`.
    Mul(V, V),
}

impl<F> Base<F> {
    /// The value a wire carries.
    pub(crate) fn wire(wire: Wire) -> Self {
        Self(BaseRepr::Node(Arc::new(Node::Wire(wire))))
    }

    /// The constant behind this value, if it is one.
    pub(crate) const fn as_const(&self) -> Option<&F> {
        match &self.0 {
            BaseRepr::Const(value) => Some(value),
            BaseRepr::Node(_) => None,
        }
    }
}

impl<F, EF> Ext<F, EF> {
    /// The value a wire carries.
    pub(crate) fn wire(wire: Wire) -> Self {
        Self(ExtRepr::Node(Arc::new(Node::Wire(wire))))
    }
}

impl<F: Field> Base<F> {
    fn combine(self, rhs: Self, op: fn(F, F) -> F, node: fn(Self, Self) -> Node<Self>) -> Self {
        match (&self.0, &rhs.0) {
            (BaseRepr::Const(a), BaseRepr::Const(b)) => Self(BaseRepr::Const(op(*a, *b))),
            _ => Self(BaseRepr::Node(Arc::new(node(self, rhs)))),
        }
    }
}

impl<F: Field, EF: ExtensionField<F>> Ext<F, EF> {
    /// The constant behind this value, if it is one.
    pub(crate) fn as_const(&self) -> Option<EF> {
        match &self.0 {
            ExtRepr::Const(value) => Some(*value),
            ExtRepr::Base(base) => base.as_const().map(|&value| EF::from(value)),
            ExtRepr::Node(_) => None,
        }
    }

    fn combine(self, rhs: Self, op: fn(EF, EF) -> EF, node: fn(Self, Self) -> Node<Self>) -> Self {
        match (self.as_const(), rhs.as_const()) {
            (Some(a), Some(b)) => Self(ExtRepr::Const(op(a, b))),
            _ => Self(ExtRepr::Node(Arc::new(node(self, rhs)))),
        }
    }
}

impl<F: Field> Add for Base<F> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        match (self.as_const(), rhs.as_const()) {
            (Some(zero), _) if zero.is_zero() => rhs,
            (_, Some(zero)) if zero.is_zero() => self,
            _ => self.combine(rhs, |a, b| a + b, Node::Add),
        }
    }
}

impl<F: Field> Sub for Base<F> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        match rhs.as_const() {
            Some(zero) if zero.is_zero() => self,
            _ => self.combine(rhs, |a, b| a - b, Node::Sub),
        }
    }
}

impl<F: Field> Mul for Base<F> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        match (self.as_const(), rhs.as_const()) {
            (Some(c), _) | (_, Some(c)) if c.is_zero() => Self::ZERO,
            (Some(one), _) if one.is_one() => rhs,
            (_, Some(one)) if one.is_one() => self,
            _ => self.combine(rhs, |a, b| a * b, Node::Mul),
        }
    }
}

impl<F: Field> Neg for Base<F> {
    type Output = Self;

    fn neg(self) -> Self {
        Self::ZERO - self
    }
}

impl<F: Field, EF: ExtensionField<F>> Add for Ext<F, EF> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        match (self.as_const(), rhs.as_const()) {
            (Some(zero), _) if zero.is_zero() => rhs,
            (_, Some(zero)) if zero.is_zero() => self,
            _ => self.combine(rhs, |a, b| a + b, Node::Add),
        }
    }
}

impl<F: Field, EF: ExtensionField<F>> Sub for Ext<F, EF> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        match rhs.as_const() {
            Some(zero) if zero.is_zero() => self,
            _ => self.combine(rhs, |a, b| a - b, Node::Sub),
        }
    }
}

impl<F: Field, EF: ExtensionField<F>> Mul for Ext<F, EF> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        match (self.as_const(), rhs.as_const()) {
            (Some(c), _) | (_, Some(c)) if c.is_zero() => Self::ZERO,
            (Some(one), _) if one.is_one() => rhs,
            (_, Some(one)) if one.is_one() => self,
            _ => self.combine(rhs, |a, b| a * b, Node::Mul),
        }
    }
}

impl<F: Field, EF: ExtensionField<F>> Neg for Ext<F, EF> {
    type Output = Self;

    fn neg(self) -> Self {
        Self::ZERO - self
    }
}

/// Assignment, sum and product forms of the ring operations.
macro_rules! ring_impls {
    ($ty:ty, [$($generics:tt)*]) => {
        impl<$($generics)*> AddAssign for $ty {
            fn add_assign(&mut self, rhs: Self) {
                *self = self.dup() + rhs;
            }
        }

        impl<$($generics)*> SubAssign for $ty {
            fn sub_assign(&mut self, rhs: Self) {
                *self = self.dup() - rhs;
            }
        }

        impl<$($generics)*> MulAssign for $ty {
            fn mul_assign(&mut self, rhs: Self) {
                *self = self.dup() * rhs;
            }
        }

        impl<$($generics)*> Sum for $ty {
            fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
                iter.fold(Self::ZERO, |acc, x| acc + x)
            }
        }

        impl<$($generics)*> Product for $ty {
            fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
                iter.fold(Self::ONE, |acc, x| acc * x)
            }
        }

        impl<$($generics)*> Default for $ty {
            fn default() -> Self {
                Self::ZERO
            }
        }

        impl<$($generics)*> Dup for $ty {
            #[inline]
            fn dup(&self) -> Self {
                self.clone()
            }
        }
    };
}

/// Ring operations with a right-hand side lifted into the value type.
macro_rules! mixed_impls {
    ($ty:ty, $rhs:ty, [$($generics:tt)*]) => {
        impl<$($generics)*> Add<$rhs> for $ty {
            type Output = Self;

            fn add(self, rhs: $rhs) -> Self {
                self + Self::from(rhs)
            }
        }

        impl<$($generics)*> Sub<$rhs> for $ty {
            type Output = Self;

            fn sub(self, rhs: $rhs) -> Self {
                self - Self::from(rhs)
            }
        }

        impl<$($generics)*> Mul<$rhs> for $ty {
            type Output = Self;

            fn mul(self, rhs: $rhs) -> Self {
                self * Self::from(rhs)
            }
        }

        impl<$($generics)*> AddAssign<$rhs> for $ty {
            fn add_assign(&mut self, rhs: $rhs) {
                *self += Self::from(rhs);
            }
        }

        impl<$($generics)*> SubAssign<$rhs> for $ty {
            fn sub_assign(&mut self, rhs: $rhs) {
                *self -= Self::from(rhs);
            }
        }

        impl<$($generics)*> MulAssign<$rhs> for $ty {
            fn mul_assign(&mut self, rhs: $rhs) {
                *self *= Self::from(rhs);
            }
        }
    };
}

ring_impls!(Base<F>, [F: Field]);
ring_impls!(Ext<F, EF>, [F: Field, EF: ExtensionField<F>]);
mixed_impls!(Base<F>, F, [F: Field]);
mixed_impls!(Ext<F, EF>, EF, [F: Field, EF: ExtensionField<F>]);
mixed_impls!(Ext<F, EF>, Base<F>, [F: Field, EF: ExtensionField<F>]);

impl<F> From<F> for Base<F> {
    fn from(value: F) -> Self {
        Self(BaseRepr::Const(value))
    }
}

impl<F: Field, EF: ExtensionField<F>> From<EF> for Ext<F, EF> {
    fn from(value: EF) -> Self {
        Self(ExtRepr::Const(value))
    }
}

impl<F: Field, EF: ExtensionField<F>> From<Base<F>> for Ext<F, EF> {
    fn from(value: Base<F>) -> Self {
        Self(ExtRepr::Base(value))
    }
}

impl<F: Field> PrimeCharacteristicRing for Base<F> {
    type PrimeSubfield = F::PrimeSubfield;

    const ZERO: Self = Self(BaseRepr::Const(F::ZERO));
    const ONE: Self = Self(BaseRepr::Const(F::ONE));
    const TWO: Self = Self(BaseRepr::Const(F::TWO));
    const NEG_ONE: Self = Self(BaseRepr::Const(F::NEG_ONE));

    fn from_prime_subfield(f: Self::PrimeSubfield) -> Self {
        Self(BaseRepr::Const(F::from_prime_subfield(f)))
    }
}

impl<F: Field, EF: ExtensionField<F>> PrimeCharacteristicRing for Ext<F, EF> {
    type PrimeSubfield = F::PrimeSubfield;

    const ZERO: Self = Self(ExtRepr::Const(EF::ZERO));
    const ONE: Self = Self(ExtRepr::Const(EF::ONE));
    const TWO: Self = Self(ExtRepr::Const(EF::TWO));
    const NEG_ONE: Self = Self(ExtRepr::Const(EF::NEG_ONE));

    fn from_prime_subfield(f: Self::PrimeSubfield) -> Self {
        Self(ExtRepr::Const(EF::from(F::from_prime_subfield(f))))
    }
}

// A linear combination in one chunk is a chain of sums, so each product folds into its sum's row.
impl<F: Field> Algebra<F> for Base<F> {
    const BATCHED_LC_CHUNK: usize = 1;
}

impl<F: Field, EF: ExtensionField<F>> Algebra<EF> for Ext<F, EF> {
    const BATCHED_LC_CHUNK: usize = 1;
}

impl<F: Field, EF: ExtensionField<F>> Algebra<Base<F>> for Ext<F, EF> {
    const BATCHED_LC_CHUNK: usize = 1;
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::extension::BinomialExtensionField;

    use super::*;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    #[test]
    fn constants_fold_and_identities_vanish() {
        // Constants combine at construction, so no node is ever created.
        let sum = Base::from(F::new(3)) * Base::from(F::new(5)) + Base::from(F::ONE);
        assert_eq!(sum.as_const(), Some(&F::new(16)));

        // Multiplying a wire by one or adding zero returns the wire itself.
        let x = Base::<F>::wire(Wire(7));
        let same = x.clone() * F::ONE + F::ZERO;
        assert!(matches!(&same.0, BaseRepr::Node(node) if matches!(**node, Node::Wire(Wire(7)))));

        // Multiplying by zero folds to the constant zero.
        assert_eq!((x * F::ZERO).as_const(), Some(&F::ZERO));

        // A base constant lifts into the challenge field as a constant.
        let lifted = Ext::<F, EF>::from(Base::from(F::new(9)));
        assert_eq!(lifted.as_const(), Some(EF::from(F::new(9))));
    }
}
