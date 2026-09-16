//! Fractional GKR for reducing a table of fractions to a single sum.
//!
//! The protocol folds pairs of fractions up a binary tree and proves each
//! reduction with sumcheck. Its output is an opening of the original numerator
//! and denominator tables at a transcript-derived point.

use alloc::vec::Vec;

use p3_field::{ExtensionField, Field, PrimeCharacteristicRing};
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::{Poly, PolyMaybePacked};
use serde::{Deserialize, Serialize};

pub(crate) mod materialize;
mod prover;
pub mod transcript;
mod verifier;

#[cfg(test)]
mod test;

pub use prover::prove_fractional_gkr;
pub use transcript::FractionGkrShape;
/// Verification errors and the verifier for a fractional-GKR proof.
pub use verifier::{FractionGkrError, verify_fractional_gkr};

/// A numerator and denominator pair with independently chosen storage types.
#[derive(Clone, Copy, Debug)]
pub struct Fraction<N, D> {
    /// The numerator or numerator evaluation table.
    pub n: N,
    /// The denominator or denominator evaluation table.
    pub d: D,
}

/// Numerator storage of the bottom fraction table.
///
/// A counting lookup puts a small integer over each entry, which the base field holds.
///
/// An indexed lookup puts a field-valued weight there, which it does not.
///
/// Only the bottom table branches, since one reduction lifts every layer above it.
#[derive(Clone, Copy, Debug)]
pub enum LeafNumerator<'a, F: Field, EF: ExtensionField<F>> {
    /// Numerators drawn from the base field.
    Base(&'a Poly<F>),
    /// Numerators drawn from the extension field, in the denominator's storage mode.
    Ext(&'a PolyMaybePacked<F, EF>),
}

/// Rejection message for a bottom table whose two halves disagree on storage.
///
/// The round-polynomial kernel reads the two side by side, so a mixed pair has no meaning.
pub(crate) const MIXED_LEAF_STORAGE: &str =
    "leaf numerator and denominator must share a storage mode";

impl<F: Field, EF: ExtensionField<F>> LeafNumerator<'_, F, EF> {
    /// Variable count of the table, counting the variables held in SIMD lanes.
    pub fn num_variables(&self) -> usize {
        match self {
            Self::Base(numer) => numer.num_variables(),
            Self::Ext(numer) => numer.num_variables(),
        }
    }

    /// The two values of a one-variable table, lifted into the extension field.
    pub(crate) fn leaf_pair(&self) -> [EF; 2] {
        match self {
            // A base-field pair only needs embedding.
            Self::Base(numer) => {
                let evals = numer.as_slice();
                [EF::from(evals[0]), EF::from(evals[1])]
            }
            // A packed pair lives in two lanes of one SIMD element, so it is spread out first.
            Self::Ext(numer) => {
                let mut evals = [EF::ZERO; 2];
                numer.unpack_into(&mut evals);
                evals
            }
        }
    }
}

/// The bottom tables one reduction consumes, borrowed from the caller.
pub type LeafFraction<'a, F, EF> = Fraction<LeafNumerator<'a, F, EF>, &'a PolyMaybePacked<F, EF>>;

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
    /// Sumcheck polynomial evaluations at interpolation nodes zero, two and three.
    pub round_polys: Vec<[EF; 3]>,
    /// The two child fractions to which the layer's sumcheck reduces.
    pub claims: SplitFraction<EF>,
}

/// A proof that a table of fractions reduces to a zero root numerator.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FractionGkrProof<EF> {
    /// Denominator of the fully reduced root fraction.
    pub root_denominator: EF,
    /// Layer proofs in root-to-input order.
    pub layers: Vec<FractionGkrLayerProof<EF>>,
}

/// An unauthenticated opening claim for the original fraction tables.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FractionGkrOutput<EF> {
    /// The transcript-derived point at which both tables are claimed to be opened.
    pub point: Point<EF>,
    /// The claimed multilinear evaluation of the numerator table at `point`.
    pub numerator: EF,
    /// The claimed multilinear evaluation of the denominator table at `point`.
    pub denominator: EF,
}
