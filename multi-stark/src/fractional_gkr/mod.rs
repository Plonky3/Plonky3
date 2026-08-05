//! Fractional GKR for reducing a table of fractions to a single sum.
//!
//! The protocol folds pairs of fractions up a binary tree and proves each
//! reduction with sumcheck. Its output is an opening of the original numerator
//! and denominator tables at a transcript-derived point.

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
/// Verification errors and the verifier for a fractional-GKR proof.
pub use verifier::{FractionGkrError, verify_fractional_gkr};

/// A numerator and denominator pair with independently chosen storage types.
#[derive(Clone, Debug)]
pub struct Fraction<N, D> {
    /// The numerator or numerator evaluation table.
    pub n: N,
    /// The denominator or denominator evaluation table.
    pub d: D,
}

/// The two child fractions opened at the end of a reduction layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SplitFraction<T> {
    /// Numerator of the child at branch zero.
    pub n0: T,
    /// Denominator of the child at branch zero.
    pub d0: T,
    /// Numerator of the child at branch one.
    pub n1: T,
    /// Denominator of the child at branch one.
    pub d1: T,
}

impl<A: PrimeCharacteristicRing + Copy> SplitFraction<A> {
    pub(crate) fn gate(&self, lambda: A) -> A {
        self.d1 * self.n0 + self.d0 * self.n1 + lambda * self.d0 * self.d1
    }
}

/// The proof messages for one layer of the fractional reduction tree.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FractionGkrLayerProof<EF> {
    /// Sumcheck polynomial evaluations at points zero, two, and three.
    pub round_polys: Vec<[EF; 3]>,
    /// The two child fractions to which the layer's sumcheck reduces.
    pub claims: SplitFraction<EF>,
}

/// A proof that a table of fractions reduces to the claimed root fraction.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FractionGkrProof<EF> {
    /// Numerator of the fully reduced root fraction.
    pub root_numerator: EF,
    /// Denominator of the fully reduced root fraction.
    pub root_denominator: EF,
    /// Layer proofs in root-to-input order.
    pub layers: Vec<FractionGkrLayerProof<EF>>,
}

/// The opening claim for the original fraction tables produced by the protocol.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FractionGkrOutput<EF> {
    /// The transcript-derived point at which both tables are opened.
    pub point: Point<EF>,
    /// The multilinear evaluation of the numerator table at `point`.
    pub numerator: EF,
    /// The multilinear evaluation of the denominator table at `point`.
    pub denominator: EF,
}
