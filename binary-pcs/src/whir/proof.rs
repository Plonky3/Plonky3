//! The proof one Boolean opening carries.

use alloc::vec::Vec;

use p3_binary_field::TowerLevel;
use p3_commit::Mmcs;
use p3_field::Field;
use p3_sumcheck::ring_switch::bits::BitRingSwitchProof;
use p3_whir::PcsProof;
use serde::{Deserialize, Serialize};

/// One Boolean opening: the reductions, and the proximity opening that discharges them.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "EF: TowerLevel, MT::Commitment: Serialize, MT::MultiProof: Serialize",
    deserialize = "EF: TowerLevel, MT::Commitment: Deserialize<'de>, MT::MultiProof: Deserialize<'de>"
))]
pub struct BooleanWhirProof<EF: Field + Send + Sync, MT: Mmcs<EF>> {
    /// One bit-alphabet ring switch per claim, in the order the claims came in.
    pub reductions: Vec<BitRingSwitchProof<EF>>,
    /// The single proximity opening that discharges every packed claim.
    pub opening: PcsProof<EF, EF, MT>,
}
