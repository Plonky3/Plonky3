//! Compile-time witness that a value has been threaded through the transcript.

/// Marker proving that `T` has been mixed into the transcript.
///
/// Fiat-Shamir is sound only when every value the verifier sees was
/// absorbed by the sponge first.
///
/// Forgetting to absorb a value before using it is a soundness bug.
///
/// This wrapper turns the rule into a compile-time check:
///
/// - The wrapper is opaque to outside callers.
/// - Absorbing or squeezing is the only way to obtain one.
/// - Functions that need a bound input declare it in their signature.
/// - The compiler refuses any caller that forgot to bind.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TranscriptBound<T>(T);

impl<T> TranscriptBound<T> {
    /// Build a bound witness.
    ///
    /// Crate-internal: callers must go through a transcript method.
    pub(in crate::fs) const fn wrap(value: T) -> Self {
        Self(value)
    }

    /// Borrow the inner value without consuming the binding.
    #[must_use]
    pub const fn as_inner(&self) -> &T {
        &self.0
    }

    /// Consume the wrapper and return the bare value, dropping the binding witness.
    #[must_use]
    pub fn into_inner(self) -> T {
        self.0
    }

    /// Lift a deterministic derivation to bound outputs.
    ///
    /// `f` must be a pure function of `T`;
    ///
    /// Non-determinism inside `f` silently launders the binding.
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> TranscriptBound<U> {
        TranscriptBound(f(self.0))
    }

    /// Combine two bound values through a pure function. Same purity caveat as `map`.
    pub fn combine_with<U, V>(
        self,
        other: TranscriptBound<U>,
        f: impl FnOnce(T, U) -> V,
    ) -> TranscriptBound<V> {
        TranscriptBound(f(self.0, other.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn as_inner_and_into_inner_round_trip() {
        // Create, read, consume, recover the original value.
        let b = TranscriptBound::wrap(42u32);
        assert_eq!(*b.as_inner(), 42);
        assert_eq!(b.into_inner(), 42);
    }

    #[test]
    fn map_propagates_binding_through_pure_function() {
        // `map` lifts a pure function on bare values to bound values.
        let a = TranscriptBound::wrap(7u32);
        let b: TranscriptBound<u64> = a.map(|x| (x as u64) * 2);
        assert_eq!(b.into_inner(), 14);
    }

    #[test]
    fn combine_with_lifts_two_argument_derivations() {
        // Bound A + bound B -> bound C through a pure function.
        let a = TranscriptBound::wrap(3u32);
        let b = TranscriptBound::wrap(4u32);
        let c = a.combine_with(b, |x, y| x + y);
        assert_eq!(c.into_inner(), 7);
    }

    #[test]
    fn equality_is_by_inner_value() {
        // PartialEq derive forwards to the inner type.
        let a = TranscriptBound::wrap(99u32);
        let b = TranscriptBound::wrap(99u32);
        assert_eq!(a, b);
    }
}
