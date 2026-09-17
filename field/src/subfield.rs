//! Small subfields that a kernel can compute inside before lifting its result.

use crate::{Algebra, Field};

/// A field containing a copy of the small field `S`.
///
/// A kernel whose inputs all lie in `S` can do its arithmetic there and lift only the result.
/// The [`Algebra`] supertrait embeds `S` and lets its elements act on this field.
/// This trait answers whether an element lies in that copy of `S`.
///
/// # Contract
///
/// For every `x: Self`, `s: S` and `values: &[Self]`:
///
/// - `Self::from` is an injective ring homomorphism: it sends `S::ONE` to `Self::ONE`, it
///   commutes with addition and multiplication, and distinct elements of `S` have distinct images.
/// - `x + s`, `x - s` and `x * s` equal `x + Self::from(s)`, `x - Self::from(s)` and
///   `x * Self::from(s)`, and so do the assigning forms.
/// - `x.as_subfield() == Some(s)` holds exactly when `x == Self::from(s)`.
///   So it returns `None` exactly when `x` is the image of no element of `S`.
/// - `Self::all_in_subfield(values)` holds exactly when every `v.as_subfield()` is `Some`.
///   In particular it holds on an empty slice.
///
/// Callers narrow values into `S` on the strength of these equalities, so they must hold exactly.
/// A `Some` for an element outside the subfield would silently compute with a different value.
pub trait HasSubfield<S: Field>: Field + Algebra<S> {
    /// The element of `S` whose image this is, or `None` if it lies outside the subfield.
    #[must_use]
    fn as_subfield(&self) -> Option<S>;

    /// Whether every element of `values` lies in the subfield.
    ///
    /// Equal to `values.iter().all(|v| v.as_subfield().is_some())`, which is what the default
    /// computes. Override it where a whole slice can be tested without narrowing each element in
    /// turn.
    #[must_use]
    #[inline]
    fn all_in_subfield(values: &[Self]) -> bool {
        values.iter().all(|v| v.as_subfield().is_some())
    }
}
