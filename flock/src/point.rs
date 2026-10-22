//! Points of a batch's quirky extension: one univariate coordinate for the six low bits, then multilinear ones.

use alloc::vec::Vec;

use p3_binary_field::fold_rows;
use p3_binary_pcs::whir::{WeightedClaim, WeightedOpening};
use p3_field::{Algebra, Dup, PrimeCharacteristicRing};
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_util::log2_strict_usize;
use p3_verifier_arith::{Arith, Native};

use crate::skip::{K_SKIP, SKIP_SIZE, lagrange_weights};
use crate::{Challenge, Val};

/// A point: `skip` on the skip domain for the six low bits, `rest` multilinear, most significant first.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QuirkyPoint<E = Challenge> {
    /// The univariate coordinate.
    pub skip: E,
    /// The multilinear coordinates, most significant first: instance bits, then inner bits.
    pub rest: Vec<E>,
}

impl<E: Algebra<Challenge>> QuirkyPoint<E> {
    /// As [`QuirkyPoint::eq_weights`], within an arithmetic.
    ///
    /// # Panics
    ///
    /// - The point has fewer than `vars` multilinear coordinates.
    pub fn eq_weights_in<A: Arith<EF = E>>(&self, arith: &mut A, vars: usize) -> Vec<E> {
        let lagrange = lagrange_weights(&self.skip, SKIP_SIZE);
        arith
            .eq_table(&self.rest[self.rest.len() - vars..])
            .into_iter()
            .flat_map(|e| lagrange.iter().map(move |l| e.dup() * l.dup()))
            .collect()
    }

    /// The coordinates in transcript order: the univariate one, then the multilinear ones.
    pub fn coordinates(&self) -> impl Iterator<Item = E> + '_ {
        core::iter::once(self.skip.dup()).chain(self.rest.iter().map(Dup::dup))
    }
}

impl QuirkyPoint {
    /// The weight `L_v(skip) eq(r, i)` of row `64 i + v`, `r` the last `vars` multilinear coordinates.
    ///
    /// # Panics
    ///
    /// - The point has fewer than `vars` multilinear coordinates.
    #[must_use]
    pub fn eq_weights(&self, vars: usize) -> Vec<Challenge> {
        self.eq_weights_in(&mut Native::<Val, Challenge>::new(), vars)
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
        let folded = fold_rows(bits, k_log, 1 << k_log, outer.as_slice());
        self.eq_weights(k_log - K_SKIP)
            .iter()
            .zip(&folded)
            .map(|(&w, &v)| w * v)
            .sum()
    }
}

/// A claim on the witness extension at a quirky point.
#[must_use = "the claim is sound only once opened against the witness commitment"]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QuirkyClaim<E = Challenge> {
    /// The point.
    pub point: QuirkyPoint<E>,
    /// The claimed value.
    pub value: E,
}

impl<E: Algebra<Challenge>> QuirkyClaim<E> {
    /// The claim as a weighted one: Lagrange weights at the skip coordinate on the absorbed bits.
    #[must_use]
    pub fn into_weighted(self) -> WeightedClaim<E> {
        WeightedClaim {
            opening: WeightedOpening {
                high: Point::new(self.point.rest),
                low_weights: lagrange_weights(&self.point.skip, SKIP_SIZE),
            },
            value: self.value,
        }
    }
}
