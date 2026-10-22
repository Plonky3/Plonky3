//! Points of a batch's quirky extension: one univariate coordinate for the six low bits, then multilinear ones.

use alloc::vec::Vec;

use p3_field::PrimeCharacteristicRing;
use p3_multilinear_util::poly::Poly;
use p3_util::log2_strict_usize;

use crate::Challenge;
use crate::lincheck::partial_fold;
use crate::skip::{K_SKIP, SKIP_SIZE, lagrange_weights};

/// A point: `skip` on the skip domain for the six low bits, `rest` multilinear, most significant first.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QuirkyPoint {
    /// The univariate coordinate.
    pub skip: Challenge,
    /// The multilinear coordinates, most significant first: instance bits, then inner bits.
    pub rest: Vec<Challenge>,
}

impl QuirkyPoint {
    /// The weight `L_v(skip) eq(r, i)` of row `64 i + v`, `r` the last `vars` multilinear coordinates.
    ///
    /// # Panics
    ///
    /// - The point has fewer than `vars` multilinear coordinates.
    #[must_use]
    pub fn eq_weights(&self, vars: usize) -> Vec<Challenge> {
        let lagrange = lagrange_weights(self.skip, SKIP_SIZE);
        Poly::new_from_point(&self.rest[self.rest.len() - vars..], Challenge::ONE)
            .iter()
            .flat_map(|&e| lagrange.iter().map(move |&l| e * l))
            .collect()
    }

    /// The extension of packed instance-major bits at this point, by one fold over the instances.
    ///
    /// # Panics
    ///
    /// - The point does not name the bits' variables, or an instance is not `2^k_log` whole words.
    pub fn evaluate_packed(&self, bits: &[u64], k_log: usize) -> Challenge {
        let m = log2_strict_usize(64 * bits.len());
        assert!(m >= k_log, "a hypercube of whole instances");
        assert_eq!(self.rest.len() + K_SKIP, m, "one coordinate per variable");
        let outer = Poly::new_from_point(&self.rest[..m - k_log], Challenge::ONE);
        let folded = partial_fold(bits, k_log, 1 << k_log, outer.as_slice());
        self.eq_weights(k_log - K_SKIP)
            .iter()
            .zip(&folded)
            .map(|(&w, &v)| w * v)
            .sum()
    }

    /// The coordinates in transcript order: the univariate one, then the multilinear ones.
    pub fn coordinates(&self) -> impl Iterator<Item = Challenge> + '_ {
        core::iter::once(self.skip).chain(self.rest.iter().copied())
    }
}

/// A claim on the witness extension at a quirky point.
#[must_use = "the claim is sound only once opened against the witness commitment"]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QuirkyClaim {
    /// The point.
    pub point: QuirkyPoint,
    /// The claimed value.
    pub value: Challenge,
}
