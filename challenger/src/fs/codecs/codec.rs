//! Stateless adapter between values and a challenger.

/// Stateless absorb/sample adapter for values of type `T` against challenger `C`.
///
/// Codecs are zero-sized types picked at the call site.
///
/// One transcript may invoke several codecs in different roles.
pub trait Codec<C, T> {
    /// Bits of statistical security: `-log2` distance from uniform on `T`.
    const SECURITY_BITS: u32;

    /// Absorb `value` into the challenger.
    fn observe(challenger: &mut C, value: &T);

    /// Sample a fresh value from the challenger.
    fn sample(challenger: &mut C) -> T;
}
