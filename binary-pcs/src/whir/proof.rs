//! The proof one Boolean opening carries.

use p3_binary_field::TowerLevel;
use p3_commit::Mmcs;
use p3_field::Field;
use p3_sumcheck::ring_switch::bits::BitRingSwitchClaimsProof;
use p3_whir::PcsProof;
use serde::{Deserialize, Serialize};

/// One Boolean opening: the reductions, and the proximity opening that discharges them.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "EF: TowerLevel, MT::Commitment: Serialize, MT::MultiProof: Serialize",
    deserialize = "EF: TowerLevel, MT::Commitment: Deserialize<'de>, MT::MultiProof: Deserialize<'de>"
))]
pub struct BooleanWhirProof<EF: Field + Send + Sync, MT: Mmcs<EF>> {
    /// One batched bit-alphabet ring switch, with every claim's elements in the order they came in.
    pub reduction: BitRingSwitchClaimsProof<EF>,
    /// The single proximity opening that discharges the one surviving claim.
    pub opening: PcsProof<EF, EF, MT>,
}
