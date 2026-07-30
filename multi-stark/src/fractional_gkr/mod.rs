use alloc::vec::Vec;

use p3_field::PrimeCharacteristicRing;
use p3_multilinear_util::point::Point;
use serde::{Deserialize, Serialize};

pub(crate) mod materialize;
mod prover;
mod verifier;

#[cfg(test)]
mod test;

pub use prover::prove_fractional_gkr;
pub use verifier::{FractionGkrError, verify_fractional_gkr};

#[derive(Clone, Debug)]
pub struct Fraction<N, D> {
    pub n: N,
    pub d: D,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SplitFraction<T> {
    pub n0: T,
    pub d0: T,
    pub n1: T,
    pub d1: T,
}

impl<A: PrimeCharacteristicRing + Copy> SplitFraction<A> {
    pub(crate) fn gate(&self, lambda: A) -> A {
        self.d1 * self.n0 + self.d0 * self.n1 + lambda * self.d0 * self.d1
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FractionGkrLayerProof<EF> {
    pub round_polys: Vec<[EF; 3]>,
    pub claims: SplitFraction<EF>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FractionGkrProof<EF> {
    /// Numerator of the fully reduced root fraction.
    pub root_numerator: EF,
    /// Denominator of the fully reduced root fraction.
    pub root_denominator: EF,
    pub layers: Vec<FractionGkrLayerProof<EF>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FractionGkrOutput<EF> {
    pub point: Point<EF>,
    pub numerator: EF,
    pub denominator: EF,
}
