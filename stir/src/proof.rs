use alloc::vec::Vec;

use p3_commit::Mmcs;
use p3_field::Field;
use p3_poly::Polynomial;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone)]
#[serde(bound(
    serialize = "Witness: Serialize, RoundProof<F, M, Witness>: Serialize, Polynomial<F>: Serialize",
    deserialize = "Witness: Deserialize<'de>, RoundProof<F, M, Witness>: Deserialize<'de>, Polynomial<F>: Deserialize<'de>"
))]
pub struct StirProof<F: Field, M: Mmcs<F>, Witness> {
    pub(crate) commitment: M::Commitment,
    pub(crate) round_proofs: Vec<RoundProof<F, M, Witness>>,
    pub(crate) final_polynomial: Polynomial<F>,
    pub(crate) pow_witness: Witness,
    pub(crate) final_round_queries: Vec<(Vec<F>, M::Proof)>,
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(bound(
    serialize = "Witness: Serialize, Polynomial<F>: Serialize",
    deserialize = "Witness: Deserialize<'de>, Polynomial<F>: Deserialize<'de>",
))]
pub struct RoundProof<F: Field, M: Mmcs<F>, Witness> {
    // The indices are given in the following frame of reference: Self is
    // produced inside prove_round for round i (in {1, ..., num_rounds}). The
    // final round, with index num_rounds + 1, does not produce a RoundProof.

    // Commitment to the vector of evaluations of g_i over the domain L_i
    pub(crate) g_root: M::Commitment,

    // Replies b_{i, j} to the OOD queries  to g_i
    pub(crate) betas: Vec<F>,

    // Polynomial interpolating the betas at the OOD places, and
    // g_i(r_{i, j}^shift) at the r_{i, j}^shift
    pub(crate) ans_polynomial: Polynomial<F>,

    // Opening proofs of evaluations of g_{i - 1} necessary to compute f_{i - 1}
    // at the poitns which get folded into g_i(r_{i, j}^shift)
    pub(crate) query_proofs: Vec<(Vec<F>, M::Proof)>,

    // Auxiliary polynomial helping the verifier evaluate ans_polynomial at all
    // the required points more efficiently
    pub(crate) shake_polynomial: Polynomial<F>,

    // Solution to the PoW challenge in round i
    pub(crate) pow_witness: Witness,
}
