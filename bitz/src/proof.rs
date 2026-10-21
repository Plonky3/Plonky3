//! What the prover sends, carried in serde structs rather than transcript bytes.

use alloc::vec::Vec;

use p3_binary_pcs::whir::preset::keccak64::OpeningProof;
use p3_gkr::ProductGkrProof;
use p3_sumcheck::SumcheckData;
use serde::{Deserialize, Serialize};

use crate::{Challenge, Val};

/// The inner-product rounds and the bit evaluation they end at.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct InnerProof {
    /// One quadratic round per row variable.
    pub sumcheck: SumcheckData<Val, Challenge>,
    /// `e = B~(rho')`, the committed bits at `(rho_c, rho')`.
    pub eval: Challenge,
}

/// The reduction of the claims to one claim on the committed bits.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BitzReduction {
    /// The integer folds, claim major, packed little-endian at the fold width (Construction 4.4, Round 2).
    pub folds: Vec<u64>,
    /// The grand product proving every fold in the exponent (Construction 6.1).
    pub gkr: ProductGkrProof<Challenge>,
    /// The leaf claim reduced to one evaluation of the bits.
    pub inner: InnerProof,
}

/// A whole proof on a commitment of its own.
#[derive(Clone, Serialize, Deserialize)]
pub struct BitzProof {
    /// The reduction.
    pub reduction: BitzReduction,
    /// The opening of the claim it leaves.
    pub opening: OpeningProof,
}
