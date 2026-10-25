//! The proof one Boolean opening carries.

use p3_binary_field::BitCoordinates;
use p3_commit::Mmcs;
use p3_field::{ExtensionField, Field};
use p3_sumcheck::ring_switch::bits::{
    BitRingSwitchClaimsProof, BitRingSwitchDeferredProof, ClaimElements,
};
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

/// One Boolean opening, its ring switch either run as a sumcheck or deferred to the first fold.
///
/// A batch whose openings read no successor row defers.
///
/// Any other batch runs the sumcheck.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: BitCoordinates, EF: BitCoordinates, MT::Commitment: Serialize, MT::MultiProof: Serialize",
    deserialize = "F: BitCoordinates, EF: BitCoordinates, MT::Commitment: Deserialize<'de>, MT::MultiProof: Deserialize<'de>"
))]
pub enum BooleanWhirOpening<F: Field, EF: ExtensionField<F>, MT: Mmcs<F>> {
    /// The ring switch run as its own sumcheck, then an opening at the point it leaves.
    Batched(BooleanWhirProof<F, EF, MT>),
    /// The ring switch's sum proven by the proximity argument's first fold.
    Deferred(BooleanWhirDeferredProof<F, EF, MT>),
}

impl<F: Field, EF: ExtensionField<F>, MT: Mmcs<F>> BooleanWhirOpening<F, EF, MT> {
    /// Every claim's elements, in the order the claims came in.
    #[must_use]
    pub fn elements(&self) -> &[ClaimElements<F, EF>] {
        match self {
            Self::Batched(proof) => &proof.reduction.claims,
            Self::Deferred(proof) => &proof.reduction.claims,
        }
    }
}
