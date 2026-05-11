//! WHIR backend state for recursive verification.
//!
//! This module mirrors the construction style of [`super::fri`] but keeps WHIR
//! separate from the generic uni-STARK recursion backend. The copied recursion
//! API verifies `p3-uni-stark` proofs through the univariate `p3_commit::Pcs`
//! trait. The WHIR implementation in this repository is a multilinear PCS, so
//! a recursive WHIR baseline needs a circuit that replays native multilinear
//! `p3_whir::pcs::WhirProof` transcripts.

use crate::ops::Poseidon2Config;

/// WHIR-backed recursion backend state.
///
/// The backend carries the non-native operation parameters shared with the FRI
/// backend: Poseidon2 challenger constants and the packing width for field
/// recomposition. A native WHIR recursive verifier can use this type to build
/// the circuit without depending on the FRI-specific proof shapes.
#[derive(Clone)]
pub struct WhirRecursionBackend<const WIDTH: usize = 16, const RATE: usize = 8> {
    /// Poseidon2 configuration used for the Fiat-Shamir challenger circuit.
    pub challenger_perm_config: Poseidon2Config,
    /// Number of recompose operations packed per AIR row.
    pub recompose_lanes: usize,
}

impl<const WIDTH: usize, const RATE: usize> WhirRecursionBackend<WIDTH, RATE> {
    /// Create a new WHIR recursion backend with the given challenger constants.
    pub const fn new(challenger_perm_config: Poseidon2Config) -> Self {
        Self {
            challenger_perm_config,
            recompose_lanes: 1,
        }
    }

    /// Override the number of recompose operations packed per AIR row.
    pub const fn with_recompose_lanes(mut self, lanes: usize) -> Self {
        self.recompose_lanes = if lanes < 1 { 1 } else { lanes };
        self
    }

    /// Tag this backend for a fixed extension degree used by the recursive
    /// proof layer.
    pub const fn for_extension_degree<const D: usize>(
        self,
    ) -> WhirRecursionBackendForExt<D, WIDTH, RATE> {
        WhirRecursionBackendForExt(self)
    }
}

/// WHIR recursion backend tagged with batch/extension field degree `D`.
#[derive(Clone)]
pub struct WhirRecursionBackendForExt<
    const D: usize,
    const WIDTH: usize = 16,
    const RATE: usize = 8,
>(
    /// The inner backend holding the challenger permutation config.
    pub(crate) WhirRecursionBackend<WIDTH, RATE>,
);

impl<const D: usize, const WIDTH: usize, const RATE: usize>
    WhirRecursionBackendForExt<D, WIDTH, RATE>
{
    /// Access the untagged backend state.
    pub const fn inner(&self) -> &WhirRecursionBackend<WIDTH, RATE> {
        &self.0
    }
}
