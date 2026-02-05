//! Compressed FRI proof format with deduplicated Merkle proofs.
//!
//! This module provides a compressed representation of FRI proofs that uses
//! batch Merkle proofs to deduplicate shared sibling nodes across multiple queries.
//! This can significantly reduce proof size when the number of queries is large.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use p3_commit::Mmcs;
use p3_field::Field;
use p3_merkle_tree::BatchMerkleProof;
use serde::{Deserialize, Serialize};

use crate::{CommitPhaseProofStep, FriProof, QueryProof};

/// A compressed FRI proof that uses batch Merkle proofs for deduplication.
///
/// In the standard FRI proof, each query contains its own Merkle proofs for each
/// commit phase round. When there are many queries (e.g., 100), there's significant
/// redundancy because queries at related indices share Merkle path siblings.
///
/// This format groups all proofs for each commit phase round together and uses
/// batch Merkle proofs to deduplicate shared nodes.
#[derive(Serialize, Deserialize, Clone)]
#[serde(bound(
    serialize = "Witness: Serialize, InputProof: Serialize, [Digest; DIGEST_ELEMS]: Serialize",
    deserialize = "Witness: for<'a> Deserialize<'a>, InputProof: for<'a> Deserialize<'a>, [Digest; DIGEST_ELEMS]: for<'a> Deserialize<'a>"
))]
pub struct CompressedFriProof<
    F: Field,
    M: Mmcs<F>,
    Witness,
    InputProof,
    Digest,
    const DIGEST_ELEMS: usize,
> {
    /// Commitments to each commit phase polynomial.
    pub commit_phase_commits: Vec<M::Commitment>,
    /// Proof of work witnesses for each commit phase round.
    pub commit_pow_witnesses: Vec<Witness>,
    /// Query indices sampled during the query phase.
    /// These are needed to reconstruct Merkle paths during verification.
    pub query_indices: Vec<usize>,
    /// Input proofs for each query, in query order.
    pub input_proofs: Vec<InputProof>,
    /// For each query and each commit phase round, the sibling value at that round.
    /// Indexed as `sibling_values[query_idx][round_idx]`.
    pub sibling_values: Vec<Vec<F>>,
    /// Batched Merkle proofs for each commit phase round.
    /// For round `i`, `commit_phase_batch_proofs[i]` contains the batch proof
    /// for all queries' openings of that round's commitment.
    pub commit_phase_batch_proofs: Vec<BatchMerkleProof<[Digest; DIGEST_ELEMS]>>,
    /// Coefficients of the final low-degree polynomial.
    pub final_poly: Vec<F>,
    /// Proof of work witness for query phase.
    pub query_pow_witness: Witness,
}

impl<F, M, Witness, InputProof, Digest, const DIGEST_ELEMS: usize>
    CompressedFriProof<F, M, Witness, InputProof, Digest, DIGEST_ELEMS>
where
    F: Field,
    M: Mmcs<F>,
    InputProof: Clone,
    Digest: Clone + Eq,
    M::Proof: AsRef<Vec<[Digest; DIGEST_ELEMS]>>,
{
    /// Compress a standard FRI proof into the compressed format.
    ///
    /// The `query_indices` must be the indices that were used to generate the query proofs,
    /// in the same order as they appear in `proof.query_proofs`.
    pub fn from_standard_proof(
        proof: FriProof<F, M, Witness, InputProof>,
        query_indices: Vec<usize>,
    ) -> Self {
        let num_queries = proof.query_proofs.len();
        let num_rounds = if num_queries > 0 {
            proof.query_proofs[0].commit_phase_openings.len()
        } else {
            0
        };

        // Extract input proofs
        let input_proofs: Vec<InputProof> = proof
            .query_proofs
            .iter()
            .map(|qp| qp.input_proof.clone())
            .collect();

        // Extract sibling values
        let sibling_values: Vec<Vec<F>> = proof
            .query_proofs
            .iter()
            .map(|qp| {
                qp.commit_phase_openings
                    .iter()
                    .map(|step| step.sibling_value)
                    .collect()
            })
            .collect();

        // Create batch proofs for each commit phase round
        let mut commit_phase_batch_proofs = Vec::with_capacity(num_rounds);

        for round in 0..num_rounds {
            // Collect all individual proofs for this round
            let round_proofs: Vec<Vec<[Digest; DIGEST_ELEMS]>> = proof
                .query_proofs
                .iter()
                .map(|qp| {
                    qp.commit_phase_openings[round]
                        .opening_proof
                        .as_ref()
                        .clone()
                })
                .collect();

            // Compute the indices for this round (shifted by round number)
            let round_indices: Vec<usize> = query_indices.iter().map(|&idx| idx >> round).collect();

            // Create batch proof
            let batch_proof = BatchMerkleProof::from_single_proofs(&round_proofs, &round_indices);
            commit_phase_batch_proofs.push(batch_proof);
        }

        CompressedFriProof {
            commit_phase_commits: proof.commit_phase_commits,
            commit_pow_witnesses: proof.commit_pow_witnesses,
            query_indices,
            input_proofs,
            sibling_values,
            commit_phase_batch_proofs,
            final_poly: proof.final_poly,
            query_pow_witness: proof.query_pow_witness,
        }
    }
}

impl<F, M, Witness, InputProof, Digest, const DIGEST_ELEMS: usize>
    CompressedFriProof<F, M, Witness, InputProof, Digest, DIGEST_ELEMS>
where
    F: Field,
    M: Mmcs<F>,
{
    /// Returns statistics about the compression achieved.
    pub fn compression_stats(&self, original_proof_digests_per_query: usize) -> CompressionStats {
        let num_queries = self.sibling_values.len();
        let num_rounds = self.commit_phase_batch_proofs.len();

        let original_digests = num_queries * original_proof_digests_per_query;
        let compressed_digests: usize = self
            .commit_phase_batch_proofs
            .iter()
            .map(|bp| bp.num_digests())
            .sum();

        CompressionStats {
            num_queries,
            num_rounds,
            original_digests,
            compressed_digests,
            savings_ratio: if original_digests > 0 {
                1.0 - (compressed_digests as f64 / original_digests as f64)
            } else {
                0.0
            },
        }
    }

    /// Returns the number of queries in this proof.
    pub fn num_queries(&self) -> usize {
        self.input_proofs.len()
    }

    /// Returns the number of FRI rounds in this proof.
    pub fn num_rounds(&self) -> usize {
        self.commit_phase_batch_proofs.len()
    }
}

impl<F, M, Witness, InputProof, Digest, const DIGEST_ELEMS: usize>
    CompressedFriProof<F, M, Witness, InputProof, Digest, DIGEST_ELEMS>
where
    F: Field,
    M: Mmcs<F>,
    Witness: Clone,
    InputProof: Clone,
    Digest: Copy + Eq,
    M::Proof: From<Vec<[Digest; DIGEST_ELEMS]>>,
{
    /// Expand the compressed proof back to standard FRI proof format.
    ///
    /// This reconstructs individual Merkle proofs from the batch proofs,
    /// allowing verification with the standard FRI verifier.
    ///
    /// Note: This requires being able to reconstruct individual proofs from
    /// batch proofs, which may not always be possible if siblings are fully
    /// deduplicated. In such cases, use `verify_compressed` instead.
    pub fn expand(&self) -> Result<FriProof<F, M, Witness, InputProof>, ExpandError> {
        let num_queries = self.query_indices.len();
        let num_rounds = self.commit_phase_batch_proofs.len();

        if num_queries == 0 {
            return Err(ExpandError::EmptyProof);
        }

        // Reconstruct individual proofs for each round
        let mut round_proofs: Vec<Vec<Vec<[Digest; DIGEST_ELEMS]>>> = Vec::with_capacity(num_rounds);

        for (round, batch_proof) in self.commit_phase_batch_proofs.iter().enumerate() {
            // Compute indices for this round
            let round_indices: Vec<usize> = self
                .query_indices
                .iter()
                .map(|&idx| idx >> round)
                .collect();

            // Try to expand batch proof to individual proofs
            let individual_proofs = batch_proof
                .into_single_proofs(&round_indices)
                .map_err(|e| ExpandError::BatchProofExpansion(format!("{:?}", e)))?;

            round_proofs.push(individual_proofs);
        }

        // Reconstruct QueryProofs
        let query_proofs: Vec<QueryProof<F, M, InputProof>> = (0..num_queries)
            .map(|q| {
                let commit_phase_openings: Vec<CommitPhaseProofStep<F, M>> = (0..num_rounds)
                    .map(|r| CommitPhaseProofStep {
                        sibling_value: self.sibling_values[q][r],
                        opening_proof: round_proofs[r][q].clone().into(),
                    })
                    .collect();

                QueryProof {
                    input_proof: self.input_proofs[q].clone(),
                    commit_phase_openings,
                }
            })
            .collect();

        Ok(FriProof {
            commit_phase_commits: self.commit_phase_commits.clone(),
            commit_pow_witnesses: self.commit_pow_witnesses.clone(),
            query_proofs,
            final_poly: self.final_poly.clone(),
            query_pow_witness: self.query_pow_witness.clone(),
        })
    }
}

/// Errors that can occur when expanding a compressed proof.
#[derive(Debug, Clone)]
pub enum ExpandError {
    /// The proof is empty (no queries).
    EmptyProof,
    /// Failed to expand batch proof to individual proofs.
    BatchProofExpansion(String),
}

/// Statistics about compression achieved by the compressed proof format.
#[derive(Debug, Clone)]
pub struct CompressionStats {
    /// Number of queries in the proof.
    pub num_queries: usize,
    /// Number of FRI rounds.
    pub num_rounds: usize,
    /// Total digest elements in the original (uncompressed) proof.
    pub original_digests: usize,
    /// Total digest elements in the compressed proof.
    pub compressed_digests: usize,
    /// Compression ratio (0.0 = no savings, 1.0 = 100% savings).
    pub savings_ratio: f64,
}
