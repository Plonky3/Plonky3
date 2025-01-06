use alloc::vec::Vec;

use p3_commit::Mmcs;
use p3_field::Field;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone)]
#[serde(bound(
    serialize = "Witness: Serialize, RoundProof<F, M>: Serialize",
    deserialize = "Witness: Deserialize<'de>, RoundProof<F, M>: Deserialize<'de>"
))]
pub struct StirProof<F: Field, M: Mmcs<F>, Witness> {
    pub(crate) round_proofs: Vec<RoundProof<F, M>>,
    pub(crate) final_polynomial: Vec<F>,
    pub(crate) pow_witness: Witness,
    pub(crate) pow_nonce: Option<usize>,

    // NP TODO path pruning/batch opening
    // pub(crate) queries_to_final: (Vec<Vec<F>>, MultiPath<MerkleConfig>),
    pub(crate) queries_to_final: (Vec<F>, Vec<M::Proof>),
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(bound(
    serialize = "Vec<F>: Serialize",
    deserialize = "Vec<F>: Deserialize<'de>", 
))]
pub struct RoundProof<F: Field, M: Mmcs<F>> {
    pub(crate) g_root: M::Commitment,
    pub(crate) betas: Vec<F>,
    pub(crate) ans_polynomial: Vec<F>,
    pub(crate) queries_to_prev: (Vec<F>, Vec<M::Proof>),
    pub(crate) shake_polynomial: Vec<F>,
    pub(crate) pow_nonce: Option<usize>,
}





