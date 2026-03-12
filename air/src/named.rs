//! Labeled constraint infrastructure.

use core::fmt;
use core::fmt::Display;

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
/// No default implementations are provided. This forces each builder to
/// make a deliberate choice about how labels are handled, preventing
/// labels from being silently lost through inherited defaults.
///
/// Builders that discard labels should implement [`PassthroughNamedAirBuilder`] instead.
/// The blanket impl will delegate every named method to the corresponding unlabeled one.
///
/// Debug builders should implement this trait directly and make the
/// named method the primary implementation. Their unlabeled methods
/// can then delegate here with an empty label.
pub trait NamedAirBuilder: AirBuilder {
    /// Assert zero with a label.
    fn assert_zero_named<I, N>(&mut self, x: I, name: N)
    where
        I: Into<Self::Expr>,
        N: Name;
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
}

/// Labeled variants of extension-field assertions.
///
/// Same two-tier design as [`NamedAirBuilder`] / [`PassthroughNamedAirBuilder`].
///
/// Debug builders implement this directly.
/// Production builders implement [`PassthroughNamedExtensionBuilder`].
pub trait NamedExtensionBuilder: ExtensionBuilder + NamedAirBuilder {
    /// Assert zero over the extension field, with a label.
    fn assert_zero_ext_named<I, N>(&mut self, x: I, name: N)
    where
        I: Into<Self::ExprEF>,
        N: Name;
}

/// Marker for builders that discard extension-field constraint labels.
///
/// Provides a blanket implementation of [`NamedExtensionBuilder`] that delegates to the unlabeled counterpart.
pub trait PassthroughNamedExtensionBuilder: ExtensionBuilder + PassthroughNamedAirBuilder {}

impl<T: PassthroughNamedExtensionBuilder> NamedExtensionBuilder for T {
    fn assert_zero_ext_named<I, N>(&mut self, x: I, _name: N)
    where
        I: Into<Self::ExprEF>,
        N: Name,
    {
        self.assert_zero_ext(x);
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
}
