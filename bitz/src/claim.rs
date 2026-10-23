//! Linear claims over a prime field on committed bits, with tensor-split weights (Construction 4.4, inputs).

use alloc::vec::Vec;

use p3_field::PrimeField64;
use p3_keccak::Keccak256Hash;
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::poly::Poly;
use p3_symmetric::CryptographicHasher;

use crate::error::ClaimError;
use crate::params::BitShape;

/// Weights `u1` of the rows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RowWeights<F> {
    /// One weight per row.
    Dense(Vec<F>),
    /// `u1[i S + s] = outer[i] slots[s]`, with `S` slots per outer index.
    Tensor {
        /// Weights of the outer index.
        outer: Vec<F>,
        /// Weights of the slot, the low row bits.
        slots: Vec<F>,
    },
}

/// Weights `u2` of the columns.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ColumnWeights<F> {
    /// One weight per column.
    Dense(Vec<F>),
    /// `u2[c] = eq(point, c)`, the first coordinate on the most significant bit of `c`.
    Equality(Vec<F>),
}

/// The claim `sum_(b, c) u1[b] u2[c] bit(c, b) = value` in `F`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorClaim<F> {
    /// Weights of the rows.
    pub rows: RowWeights<F>,
    /// Weights of the columns.
    pub columns: ColumnWeights<F>,
    /// The claimed value.
    pub value: F,
}

/// How the claims are bound into the transcript.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ClaimBinding {
    /// A digest of the claims is bound before any challenge.
    Public,
    /// The claims are functions of the transcript so far, which already binds them.
    Derived,
}

impl ClaimBinding {
    /// The tag bound into the transcript.
    pub(crate) const fn tag(self) -> u8 {
        match self {
            Self::Public => 0,
            Self::Derived => 1,
        }
    }
}

impl<F: PrimeField64> TensorClaim<F> {
    /// Check the weights against the shape.
    pub(crate) const fn check(&self, index: usize, shape: BitShape) -> Result<(), ClaimError> {
        let rows = match &self.rows {
            RowWeights::Dense(weights) => weights.len(),
            RowWeights::Tensor { outer, slots } => outer.len().saturating_mul(slots.len()),
        };
        if rows != shape.rows() {
            return Err(ClaimError::RowLength {
                index,
                expected: shape.rows(),
                actual: rows,
            });
        }
        let (expected, actual) = match &self.columns {
            ColumnWeights::Dense(weights) => (shape.columns(), weights.len()),
            ColumnWeights::Equality(point) => (shape.log_columns(), point.len()),
        };
        if expected != actual {
            return Err(ClaimError::ColumnLength {
                index,
                expected,
                actual,
            });
        }
        Ok(())
    }

    /// `gamma = lift(u1)`, the canonical lifts in `[0, p)` (Construction 4.4, Phase 3).
    pub(crate) fn lifted_rows(&self) -> Vec<u64> {
        match &self.rows {
            RowWeights::Dense(weights) => weights.par_iter().map(F::as_canonical_u64).collect(),
            RowWeights::Tensor { outer, slots } => outer
                .par_iter()
                .flat_map_iter(|&o| slots.iter().map(move |&s| (o * s).as_canonical_u64()))
                .collect(),
        }
    }

    /// `u2`, expanded to one weight per column.
    pub(crate) fn column_weights(&self) -> Vec<F> {
        match &self.columns {
            ColumnWeights::Dense(weights) => weights.clone(),
            ColumnWeights::Equality(point) => Poly::new_from_point(point, F::ONE).into_evals(),
        }
    }

    /// Append the canonical encoding: kind tags, lengths and canonical little-endian values.
    fn encode(&self, out: &mut Vec<u8>) {
        let elements = |tag: u8, weights: &[F], out: &mut Vec<u8>| {
            out.push(tag);
            out.extend_from_slice(&(weights.len() as u64).to_le_bytes());
            for w in weights {
                out.extend_from_slice(&w.as_canonical_u64().to_le_bytes());
            }
        };
        match &self.rows {
            RowWeights::Dense(weights) => elements(0, weights, out),
            RowWeights::Tensor { outer, slots } => {
                elements(1, outer, out);
                elements(2, slots, out);
            }
        }
        match &self.columns {
            ColumnWeights::Dense(weights) => elements(0, weights, out),
            ColumnWeights::Equality(point) => elements(1, point, out),
        }
        out.extend_from_slice(&self.value.as_canonical_u64().to_le_bytes());
    }
}

/// Keccak-256 of the claims' canonical encoding, for `ClaimBinding::Public`.
pub(crate) fn digest<F: PrimeField64>(claims: &[TensorClaim<F>]) -> [u8; 32] {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(claims.len() as u64).to_le_bytes());
    for claim in claims {
        claim.encode(&mut bytes);
    }
    Keccak256Hash.hash_iter(bytes)
}
