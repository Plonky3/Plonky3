//! A weight the initial sumcheck of a layout proves against, beside the claims it recorded.

use p3_multilinear_util::poly::Poly;

/// One more weight the preprocessing rounds of a layout prove against.
///
/// The layout weighs it by the next power of its batching challenge, after every recorded claim.
pub trait InitialWeight<EF> {
    /// The weight's inner product with the committed polynomial.
    fn claimed_sum(&self) -> EF;

    /// The next round's `h(0)` and `h(inf)`, the variables before it bound to `challenges`.
    ///
    /// Round `i` binds the layout's `i`-th variable in its binding order.
    fn round(&mut self, challenges: &[EF]) -> (EF, EF);

    /// The weight over the variables left once every challenge is bound.
    fn into_folded(self, challenges: &[EF]) -> Poly<EF>
    where
        Self: Sized;
}
