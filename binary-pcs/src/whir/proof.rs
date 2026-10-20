//! The proof one Boolean opening carries.

use p3_binary_field::BitCoordinates;
use p3_commit::Mmcs;
use p3_field::{ExtensionField, Field};
use p3_sumcheck::ring_switch::bits::{BitRingSwitchClaimsProof, BitRingSwitchDeferredProof};
use p3_whir::{PcsProof, WhirProof};
use serde::{Deserialize, Serialize};

/// One Boolean opening: the reductions, and the proximity opening that discharges them.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: BitCoordinates, EF: BitCoordinates, MT::Commitment: Serialize, MT::MultiProof: Serialize",
    deserialize = "F: BitCoordinates, EF: BitCoordinates, MT::Commitment: Deserialize<'de>, MT::MultiProof: Deserialize<'de>"
))]
pub struct BooleanWhirProof<F: Field, EF: ExtensionField<F>, MT: Mmcs<F>> {
    /// One batched bit-alphabet ring switch, with every claim's elements in the order they came in.
    pub reduction: BitRingSwitchClaimsProof<F, EF>,
    /// The single proximity opening that discharges the one surviving claim.
    pub opening: PcsProof<F, EF, MT>,
}

/// One Boolean opening whose ring switch defers its sum to the proximity argument's first fold.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: BitCoordinates, EF: BitCoordinates, MT::Commitment: Serialize, MT::MultiProof: Serialize",
    deserialize = "F: BitCoordinates, EF: BitCoordinates, MT::Commitment: Deserialize<'de>, MT::MultiProof: Deserialize<'de>"
))]
pub struct BooleanWhirDeferredProof<F: Field, EF: ExtensionField<F>, MT: Mmcs<F>> {
    /// Every claim's elements, in the order they came in.
    pub reduction: BitRingSwitchDeferredProof<F, EF>,
    /// The proximity opening whose first fold proves the batch's sum.
    pub opening: WhirProof<F, EF, MT>,
}
