//! Compile-time witness that a value has been threaded through the transcript.

/// Marker proving that `T` has been mixed into the transcript.
///
/// Fiat-Shamir is sound only when every value the verifier acts on was
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
///
/// The guarantee is deliberately narrow.
/// It says a `TranscriptBound<T>` was minted by a transcript method, nothing more.
///
/// There is no combinator that carries a binding across a closure.
/// A closure that ignores its argument would launder the witness in silence.
///
/// A derived value has to be rebound, or unwrapped with [`Self::into_inner`].
/// Unwrapping makes the loss of the guarantee visible at the call site.
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
    fn equality_is_by_inner_value() {
        // PartialEq derive forwards to the inner type.
        let a = TranscriptBound::wrap(99u32);
        let b = TranscriptBound::wrap(99u32);
        assert_eq!(a, b);
    }
}
