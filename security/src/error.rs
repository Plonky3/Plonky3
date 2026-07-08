//! Uniform currency for soundness error contributions.

use alloc::vec::Vec;

use libm::log2;
use serde::Serialize;

/// `-log2(error_probability)`. Higher = tighter bound.
#[derive(Copy, Clone, Debug, PartialEq, Serialize)]
pub struct ErrorBits(pub f64);

impl ErrorBits {
    pub const fn from_log2(bits: f64) -> Self {
        Self(bits)
    }

    pub fn from_prob(p: f64) -> Self {
        Self(-log2(p))
    }

    /// `-log2(sum of error probabilities)`. Tight composition; use this
    /// when terms are independent or the union bound is the operative
    /// bound.
    pub fn sum(errors: &[Self]) -> Self {
        let total: f64 = errors.iter().map(|e| pow2_neg(e.0)).sum();
        Self::from_prob(total)
    }

    /// Minimum bits across terms. Loose but cheap — equivalent to taking
    /// the largest single error probability.
    #[allow(clippy::missing_const_for_fn)] // iterator / f64::min are not const
    pub fn min(errors: &[Self]) -> Self {
        let m = errors.iter().map(|e| e.0).fold(f64::INFINITY, f64::min);
        Self(m)
    }

    pub const fn bits(self) -> f64 {
        self.0
    }

    pub fn floor(self) -> usize {
        libm::floor(self.0) as usize
    }
}

fn pow2_neg(bits: f64) -> f64 {
    libm::pow(2.0, -bits)
}

/// Convenience wrapper for `ErrorBits::sum` to keep call sites readable.
pub fn sum_errors<I: IntoIterator<Item = ErrorBits>>(iter: I) -> ErrorBits {
    let v: Vec<_> = iter.into_iter().collect();
    ErrorBits::sum(&v)
}
