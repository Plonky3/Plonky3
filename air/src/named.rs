//! Labeled constraint infrastructure.

use core::fmt;
use core::fmt::Display;

use p3_field::{Dup, PrimeCharacteristicRing};

use crate::{AirBuilder, ExtensionBuilder, FilteredAirBuilder};

/// A lazily evaluated constraint label.
///
/// Two kinds of labels are supported out of the box:
///
/// - **Static strings** — cheapest option, also usable as namespaces.
/// - **Closures** — evaluated only when the constraint actually fails.
///
/// Production builders never evaluate the label.
/// Only the debug builder does, so naming constraints is free at proving time.
pub trait Name {
    /// The concrete type produced after evaluation.
    type Output: Display;

    /// Produce the displayable label.
    fn evaluate(self) -> Self::Output;
}

impl Name for &'static str {
    type Output = &'static str;

    #[inline]
    fn evaluate(self) -> Self::Output {
        self
    }
}

impl<F, T> Name for F
where
    F: FnOnce() -> T,
    T: Display,
{
    type Output = T;

    #[inline]
    fn evaluate(self) -> Self::Output {
        self()
    }
}

/// A name that can be cheaply duplicated.
///
/// Required for hierarchical composition.
/// Static strings satisfy this automatically.
/// Closures generally do not.
pub trait Namespace: Name + Copy {}

impl<T> Namespace for T where T: Name + Copy {}

/// Two names composed into a hierarchical label.
///
/// Produces a display string like `"outer::inner"` when evaluated.
/// Also implements [`Namespace`] when both halves are copyable.
#[derive(Copy, Clone)]
pub struct Joined<A, B> {
    left: A,
    right: B,
}

/// The evaluated form of a [`Joined`] label, ready for display.
pub struct EvaluatedJoined<A, B> {
    left: A,
    right: B,
}

impl<A: Display, B: Display> Display for EvaluatedJoined<A, B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}::{}", self.left, self.right)
    }
}

impl<A: Name, B: Name> Name for Joined<A, B> {
    type Output = EvaluatedJoined<A::Output, B::Output>;

    #[inline]
    fn evaluate(self) -> Self::Output {
        EvaluatedJoined {
            left: self.left.evaluate(),
            right: self.right.evaluate(),
        }
    }
}

/// Composition methods for building hierarchical labels.
pub trait NamespaceExt: Namespace + Sized {
    /// Nest a sub-namespace under this one.
    ///
    /// Both sides must be copyable.
    /// Produces `"outer::inner"` when displayed.
    fn join<Ns: Namespace>(self, sub_ns: Ns) -> Joined<Self, Ns> {
        Joined {
            left: self,
            right: sub_ns,
        }
    }

    /// Attach a terminal label under this namespace.
    ///
    /// The label does not need to be copyable, so closures work here.
    /// Produces `"namespace::label"` when displayed.
    fn name<N: Name>(self, name: N) -> Joined<Self, N> {
        Joined {
            left: self,
            right: name,
        }
    }
}

impl<T> NamespaceExt for T where T: Namespace {}

/// Labeled variants of every assertion method.
///
/// Builders that discard labels should implement [`PassthroughNamedAirBuilder`] instead.
/// Its blanket impl overrides **every** method to delegate directly to the unlabeled counterpart.
///
/// Debug builders should implement this trait directly and make the named method the primary implementation.
/// Their unlabeled methods can then delegate here with an empty label.
pub trait NamedAirBuilder: AirBuilder {
    /// Assert zero with a label.
    fn assert_zero_named<I, N>(&mut self, x: I, name: N)
    where
        I: Into<Self::Expr>,
        N: Name;

    /// Assert all elements are zero, with a label.
    fn assert_zeros_named<const M: usize, I, Ns>(&mut self, array: [I; M], name: Ns)
    where
        I: Into<Self::Expr>,
        Ns: Namespace;

    /// Assert one with a label.
    fn assert_one_named<I, N>(&mut self, x: I, name: N)
    where
        I: Into<Self::Expr>,
        N: Name;

    /// Assert equality with a label.
    fn assert_eq_named<I1, I2, N>(&mut self, x: I1, y: I2, name: N)
    where
        I1: Into<Self::Expr>,
        I2: Into<Self::Expr>,
        N: Name;

    /// Assert boolean with a label.
    fn assert_bool_named<I, N>(&mut self, x: I, name: N)
    where
        I: Into<Self::Expr>,
        N: Name;

    /// Assert all elements are boolean, with a label.
    fn assert_bools_named<const M: usize, I, Ns>(&mut self, array: [I; M], name: Ns)
    where
        I: Into<Self::Expr>,
        Ns: Namespace;
}

/// Marker for builders that discard constraint labels.
///
/// Provides a blanket implementation of [`NamedAirBuilder`] that
/// delegates every named method to its unlabeled counterpart.
///
/// Production builders (prover, verifier, symbolic) should implement
/// this trait with an empty impl block.
pub trait PassthroughNamedAirBuilder: AirBuilder {}

impl<T: PassthroughNamedAirBuilder> NamedAirBuilder for T {
    fn assert_zero_named<I, N>(&mut self, x: I, _name: N)
    where
        I: Into<Self::Expr>,
        N: Name,
    {
        self.assert_zero(x);
    }

    fn assert_zeros_named<const M: usize, I, Ns>(&mut self, array: [I; M], _name: Ns)
    where
        I: Into<Self::Expr>,
        Ns: Namespace,
    {
        self.assert_zeros(array);
    }

    fn assert_one_named<I, N>(&mut self, x: I, _name: N)
    where
        I: Into<Self::Expr>,
        N: Name,
    {
        self.assert_one(x);
    }

    fn assert_eq_named<I1, I2, N>(&mut self, x: I1, y: I2, _name: N)
    where
        I1: Into<Self::Expr>,
        I2: Into<Self::Expr>,
        N: Name,
    {
        self.assert_eq(x, y);
    }

    fn assert_bool_named<I, N>(&mut self, x: I, _name: N)
    where
        I: Into<Self::Expr>,
        N: Name,
    {
        self.assert_bool(x);
    }

    fn assert_bools_named<const M: usize, I, Ns>(&mut self, array: [I; M], _name: Ns)
    where
        I: Into<Self::Expr>,
        Ns: Namespace,
    {
        self.assert_bools(array);
    }
}

impl<AB: NamedAirBuilder> NamedAirBuilder for FilteredAirBuilder<'_, AB> {
    fn assert_zero_named<I, N>(&mut self, x: I, name: N)
    where
        I: Into<Self::Expr>,
        N: Name,
    {
        self.inner
            .assert_zero_named(self.condition() * x.into(), name);
    }

    fn assert_zeros_named<const M: usize, I, Ns>(&mut self, array: [I; M], name: Ns)
    where
        I: Into<Self::Expr>,
        Ns: Namespace,
    {
        let condition = self.condition();
        self.inner
            .assert_zeros_named(array.map(|elem| condition.dup() * elem.into()), name);
    }

    fn assert_one_named<I, N>(&mut self, x: I, name: N)
    where
        I: Into<Self::Expr>,
        N: Name,
    {
        self.inner
            .assert_zero_named(self.condition() * (x.into() - Self::Expr::ONE), name);
    }

    fn assert_eq_named<I1, I2, N>(&mut self, x: I1, y: I2, name: N)
    where
        I1: Into<Self::Expr>,
        I2: Into<Self::Expr>,
        N: Name,
    {
        self.inner
            .assert_zero_named(self.condition() * (x.into() - y.into()), name);
    }

    fn assert_bool_named<I, N>(&mut self, x: I, name: N)
    where
        I: Into<Self::Expr>,
        N: Name,
    {
        self.inner
            .assert_zero_named(self.condition() * x.into().bool_check(), name);
    }

    fn assert_bools_named<const M: usize, I, Ns>(&mut self, array: [I; M], name: Ns)
    where
        I: Into<Self::Expr>,
        Ns: Namespace,
    {
        self.assert_zeros_named(array.map(|elem| elem.into().bool_check()), name);
    }
}

/// Labeled variants of extension-field assertions.
///
/// Same two-tier design as [`NamedAirBuilder`] / [`PassthroughNamedAirBuilder`].
///
/// Debug builders implement this directly.
/// Production builders implement [`PassthroughNamedAirBuilder`].
pub trait NamedExtensionBuilder: ExtensionBuilder + NamedAirBuilder {
    /// Assert zero over the extension field, with a label.
    fn assert_zero_ext_named<I, N>(&mut self, x: I, name: N)
    where
        I: Into<Self::ExprEF>,
        N: Name;

    /// Assert equality over the extension field, with a label.
    fn assert_eq_ext_named<I1, I2, N>(&mut self, x: I1, y: I2, name: N)
    where
        I1: Into<Self::ExprEF>,
        I2: Into<Self::ExprEF>,
        N: Name;

    /// Assert one over the extension field, with a label.
    fn assert_one_ext_named<I, N>(&mut self, x: I, name: N)
    where
        I: Into<Self::ExprEF>,
        N: Name;
}

impl<T: PassthroughNamedAirBuilder + ExtensionBuilder> NamedExtensionBuilder for T {
    fn assert_zero_ext_named<I, N>(&mut self, x: I, _name: N)
    where
        I: Into<Self::ExprEF>,
        N: Name,
    {
        self.assert_zero_ext(x);
    }

    fn assert_eq_ext_named<I1, I2, N>(&mut self, x: I1, y: I2, _name: N)
    where
        I1: Into<Self::ExprEF>,
        I2: Into<Self::ExprEF>,
        N: Name,
    {
        self.assert_eq_ext(x, y);
    }

    fn assert_one_ext_named<I, N>(&mut self, x: I, _name: N)
    where
        I: Into<Self::ExprEF>,
        N: Name,
    {
        self.assert_one_ext(x);
    }
}

impl<AB: NamedExtensionBuilder> NamedExtensionBuilder for FilteredAirBuilder<'_, AB> {
    fn assert_zero_ext_named<I, N>(&mut self, x: I, name: N)
    where
        I: Into<Self::ExprEF>,
        N: Name,
    {
        let ext_x: Self::ExprEF = x.into();
        let condition: AB::Expr = self.condition();
        self.inner.assert_zero_ext_named(ext_x * condition, name);
    }

    fn assert_eq_ext_named<I1, I2, N>(&mut self, x: I1, y: I2, name: N)
    where
        I1: Into<Self::ExprEF>,
        I2: Into<Self::ExprEF>,
        N: Name,
    {
        let diff: Self::ExprEF = x.into() - y.into();
        let condition: AB::Expr = self.condition();
        self.inner.assert_zero_ext_named(diff * condition, name);
    }

    fn assert_one_ext_named<I, N>(&mut self, x: I, name: N)
    where
        I: Into<Self::ExprEF>,
        N: Name,
    {
        let diff: Self::ExprEF = x.into() - Self::ExprEF::ONE;
        let condition: AB::Expr = self.condition();
        self.inner.assert_zero_ext_named(diff * condition, name);
    }
}
