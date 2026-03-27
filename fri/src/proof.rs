use alloc::vec::Vec;

use p3_commit::Mmcs;
use p3_field::Field;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone)]
#[serde(bound(
    serialize = "Witness: Serialize, InputProof: Serialize",
    deserialize = "Witness: Deserialize<'de>, InputProof: Deserialize<'de>"
))]
pub struct FriProof<F: Field, M: Mmcs<F>, Witness, InputProof> {
    pub commit_phase_commits: Vec<M::Commitment>,
    pub commit_pow_witnesses: Vec<Witness>,
    pub query_proofs: Vec<QueryProof<F, M, InputProof>>,
    pub final_poly: Vec<F>,
    pub query_pow_witness: Witness,
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(bound(
    serialize = "InputProof: Serialize",
    deserialize = "InputProof: Deserialize<'de>",
))]
pub struct QueryProof<F: Field, M: Mmcs<F>, InputProof> {
    pub input_proof: InputProof,
    /// For each commit phase commitment, this contains openings of a commit phase codeword at the
    /// queried location, along with an opening proof.
    pub commit_phase_openings: Vec<CommitPhaseProofStep<F, M>>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(bound = "")]
pub struct CommitPhaseProofStep<F: Field, M: Mmcs<F>> {
    /// The log2 of the folding arity used for this step.
    pub log_arity: u8,
    /// The openings of the commit phase codeword at the sibling locations.
    /// For arity k, this contains k-1 sibling values.
    pub sibling_values: Vec<F>,

    pub opening_proof: M::Proof,
}

impl<F: Field, M: Mmcs<F>> CommitPhaseProofStep<F, M> {
    /// Validate protocol-level arity constraints and return `(log_arity, arity)`
    /// on success.
    ///
    /// This separates "data is deserializable" from "data is protocol-valid".
    #[inline]
    pub(crate) fn checked_arity(
        &self,
        max_log_arity: usize,
        log_current_height: usize,
    ) -> Option<(usize, usize)> {
        let log_arity = self.log_arity as usize;
        if log_arity > max_log_arity || log_arity > log_current_height {
            return None;
        }
        let shift = u32::try_from(log_arity).ok()?;
        let arity = 1usize.checked_shl(shift)?;
        Some((log_arity, arity))
    }
}
