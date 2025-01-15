use alloc::vec::Vec;

use p3_commit::Mmcs;
use p3_field::Field;
use serde::{Deserialize, Serialize};

use crate::polynomial::Polynomial;

#[derive(Serialize, Deserialize, Clone)]
#[serde(bound(
    serialize = "Witness: Serialize, RoundProof<F, M, Witness>: Serialize, Polynomial<F>: Serialize",
    deserialize = "Witness: Deserialize<'de>, RoundProof<F, M, Witness>: Deserialize<'de>, Polynomial<F>: Deserialize<'de>"
))]
pub struct StirProof<F: Field, M: Mmcs<F>, Witness> {
    pub(crate) round_proofs: Vec<RoundProof<F, M, Witness>>,
    pub(crate) final_polynomial: Polynomial<F>,
    pub(crate) pow_witness: Witness,

    // NP TODO path pruning/batch opening
    // pub(crate) queries_to_final: (Vec<Vec<F>>, MultiPath<MerkleConfig>),
    pub(crate) queries_to_final: Vec<(Vec<Vec<F>>, M::Proof)>,
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(bound(
    serialize = "Witness: Serialize, Polynomial<F>: Serialize",
    deserialize = "Witness: Deserialize<'de>, Polynomial<F>: Deserialize<'de>",
))]
pub struct RoundProof<F: Field, M: Mmcs<F>, Witness> {
    pub(crate) g_root: M::Commitment,
    pub(crate) betas: Vec<F>,
    pub(crate) ans_polynomial: Polynomial<F>,
    pub(crate) query_proofs: Vec<(Vec<Vec<F>>, M::Proof)>,
    pub(crate) shake_polynomial: Polynomial<F>,
    pub(crate) pow_witness: Witness,
}
