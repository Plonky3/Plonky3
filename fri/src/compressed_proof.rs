//! Compressed FRI proof format with deduplicated Merkle proofs.
//!
//! This module provides a compressed representation of FRI proofs that uses
//! batch Merkle proofs to deduplicate shared sibling nodes across multiple queries.
//! This can significantly reduce proof size when the number of queries is large.

use alloc::vec;
use alloc::vec::Vec;

use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::{BatchOpening, Mmcs};
use p3_field::coset::TwoAdicMultiplicativeCoset;
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_merkle_tree::BatchMerkleProof;
use p3_util::reverse_bits_len;
use serde::{Deserialize, Serialize};

use crate::verifier::{FriError, open_input};
use crate::{CommitmentWithOpeningPoints, FriFoldingStrategy, FriParameters, FriProof};

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

        let input_proofs: Vec<InputProof> = proof
            .query_proofs
            .iter()
            .map(|qp| qp.input_proof.clone())
            .collect();

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

        let mut commit_phase_batch_proofs = Vec::with_capacity(num_rounds);

        for round in 0..num_rounds {
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

            let round_indices: Vec<usize> = query_indices.iter().map(|&idx| idx >> round).collect();
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
    Digest: Copy + Eq,
{
    /// Verify the batch Merkle proofs for a single FRI round.
    ///
    /// # Arguments
    /// * `round` - The FRI round index
    /// * `commitment` - The commitment for this round (as raw digest)
    /// * `leaf_hashes` - Pre-computed leaf hashes for each query at this round
    /// * `compress` - The compression function
    ///
    /// Returns Ok if the batch proof verifies, Err otherwise.
    pub fn verify_round<C>(
        &self,
        round: usize,
        commitment: &[Digest; DIGEST_ELEMS],
        leaf_hashes: &[[Digest; DIGEST_ELEMS]],
        compress: C,
    ) -> Result<(), CompressedVerifyError>
    where
        C: Fn([[Digest; DIGEST_ELEMS]; 2]) -> [Digest; DIGEST_ELEMS],
    {
        if round >= self.commit_phase_batch_proofs.len() {
            return Err(CompressedVerifyError::InvalidRound);
        }

        let round_indices: Vec<usize> =
            self.query_indices.iter().map(|&idx| idx >> round).collect();

        self.commit_phase_batch_proofs[round]
            .verify(commitment, &round_indices, leaf_hashes, compress)
            .map_err(|_| CompressedVerifyError::BatchProofMismatch)
    }
}

/// Errors during compressed proof verification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompressedVerifyError {
    /// Invalid round index.
    InvalidRound,
    /// Batch Merkle proof verification failed.
    BatchProofMismatch,
    /// Query indices don't match challenger output.
    QueryIndexMismatch,
    /// Input proof verification failed.
    InputProofError,
    /// Final polynomial evaluation mismatch.
    FinalPolyMismatch,
    /// Invalid proof shape.
    InvalidProofShape,
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

/// Verifies a compressed FRI proof natively without expanding to the full format.
///
/// This function performs batch verification of Merkle proofs for each FRI round,
/// rather than verifying each query independently. This matches how the compressed
/// proofs are structured and avoids the need to reconstruct individual proofs.
///
/// # Type Parameters
/// - `Folding`: The FRI folding strategy
/// - `Val`: The base field
/// - `Challenge`: The extension field
/// - `InputMmcs`: The MMCS for input commitments
/// - `FriMmcs`: The MMCS for FRI commitments
/// - `Challenger`: The Fiat-Shamir challenger
/// - `Digest`: The digest type used in Merkle trees
/// - `H`: Hash function type for leaf hashing
/// - `C`: Compress function type for internal nodes
/// - `DIGEST_ELEMS`: Number of field elements in each digest
pub fn verify_compressed_fri<
    Folding,
    Val,
    Challenge,
    InputMmcs,
    FriMmcs,
    Challenger,
    Digest,
    H,
    C,
    const DIGEST_ELEMS: usize,
>(
    folding: &Folding,
    params: &FriParameters<FriMmcs>,
    proof: &CompressedFriProof<
        Challenge,
        FriMmcs,
        Challenger::Witness,
        Folding::InputProof,
        Digest,
        DIGEST_ELEMS,
    >,
    challenger: &mut Challenger,
    commitments_with_opening_points: &[CommitmentWithOpeningPoints<
        Challenge,
        InputMmcs::Commitment,
        TwoAdicMultiplicativeCoset<Val>,
    >],
    input_mmcs: &InputMmcs,
    hash_leaves: H,
    compress: C,
    commitment_to_digest: impl Fn(&FriMmcs::Commitment) -> [Digest; DIGEST_ELEMS],
) -> Result<(), FriError<FriMmcs::Error, InputMmcs::Error>>
where
    Val: TwoAdicField,
    Challenge: ExtensionField<Val>,
    InputMmcs: Mmcs<Val>,
    FriMmcs: Mmcs<Challenge>,
    Challenger: FieldChallenger<Val> + GrindingChallenger + CanObserve<FriMmcs::Commitment>,
    Folding: FriFoldingStrategy<
            Val,
            Challenge,
            InputError = InputMmcs::Error,
            InputProof = Vec<BatchOpening<Val, InputMmcs>>,
        >,
    Digest: Copy + Eq,
    H: Fn(&[[Challenge; 2]]) -> Vec<[Digest; DIGEST_ELEMS]>,
    C: Fn([[Digest; DIGEST_ELEMS]; 2]) -> [Digest; DIGEST_ELEMS],
{
    let alpha: Challenge = challenger.sample_algebra_element();

    let log_global_max_height =
        proof.commit_phase_batch_proofs.len() + params.log_blowup + params.log_final_poly_len;

    if proof.commit_pow_witnesses.len() != proof.commit_phase_commits.len() {
        return Err(FriError::InvalidProofShape);
    }

    let betas: Vec<Challenge> = proof
        .commit_phase_commits
        .iter()
        .zip(&proof.commit_pow_witnesses)
        .map(|(comm, witness)| {
            challenger.observe(comm.clone());
            if !challenger.check_witness(params.commit_proof_of_work_bits, *witness) {
                return Err(FriError::InvalidPowWitness);
            }
            Ok(challenger.sample_algebra_element())
        })
        .collect::<Result<Vec<_>, _>>()?;

    if proof.final_poly.len() != params.final_poly_len() {
        return Err(FriError::InvalidProofShape);
    }

    challenger.observe_algebra_slice(&proof.final_poly);

    if proof.num_queries() != params.num_queries {
        return Err(FriError::InvalidProofShape);
    }

    if !challenger.check_witness(params.query_proof_of_work_bits, proof.query_pow_witness) {
        return Err(FriError::InvalidPowWitness);
    }

    let log_final_height = params.log_blowup + params.log_final_poly_len;
    let num_rounds = proof.num_rounds();

    // Verify proof has the expected number of rounds
    if num_rounds != log_global_max_height - log_final_height {
        return Err(FriError::InvalidProofShape);
    }
    let num_queries = proof.num_queries();

    // Verify query indices match challenger samples
    let expected_indices: Vec<usize> = (0..num_queries)
        .map(|_| challenger.sample_bits(log_global_max_height + folding.extra_query_index_bits()))
        .collect();

    if proof.query_indices != expected_indices {
        return Err(FriError::InvalidProofShape);
    }

    // Process input proofs and get reduced openings for each query
    // query_state[i] = (domain_index, folded_eval, remaining_reduced_openings)
    let mut query_state: Vec<(usize, Challenge, Vec<(usize, Challenge)>)> =
        Vec::with_capacity(num_queries);

    for (query_idx, input_proof) in proof.input_proofs.iter().enumerate() {
        let index = proof.query_indices[query_idx];
        let ro = open_input(
            params,
            log_global_max_height,
            index,
            input_proof,
            alpha,
            input_mmcs,
            commitments_with_opening_points,
        )?;

        let mut ro_iter = ro.into_iter().peekable();
        if ro_iter.peek().is_none() || ro_iter.peek().unwrap().0 != log_global_max_height {
            return Err(FriError::InvalidProofShape);
        }
        let initial_eval = ro_iter.next().unwrap().1;
        let remaining_ro: Vec<_> = ro_iter.collect();

        let domain_index = index >> folding.extra_query_index_bits();
        query_state.push((domain_index, initial_eval, remaining_ro));
    }

    // For each round, batch verify all queries then fold
    for (round, &beta) in betas.iter().enumerate() {
        let log_folded_height = log_global_max_height - 1 - round;

        // Build leaf values for batch verification
        let mut leaf_pairs: Vec<[Challenge; 2]> = Vec::with_capacity(num_queries);

        for (query_idx, (domain_index, folded_eval, _)) in query_state.iter().enumerate() {
            let sibling_value = proof.sibling_values[query_idx][round];
            let index_sibling = domain_index ^ 1;

            let mut evals = [*folded_eval; 2];
            evals[index_sibling % 2] = sibling_value;
            leaf_pairs.push(evals);
        }

        // Hash leaves and verify batch proof
        let leaf_hashes = hash_leaves(&leaf_pairs);
        let commitment_digest = commitment_to_digest(&proof.commit_phase_commits[round]);
        let round_indices: Vec<usize> = query_state.iter().map(|(idx, _, _)| *idx >> 1).collect();

        proof.commit_phase_batch_proofs[round]
            .verify(&commitment_digest, &round_indices, &leaf_hashes, &compress)
            .map_err(|_| FriError::InvalidProofShape)?;

        // Fold all queries and roll in reduced openings
        for (query_idx, (domain_index, folded_eval, remaining_ro)) in
            query_state.iter_mut().enumerate()
        {
            let sibling_value = proof.sibling_values[query_idx][round];
            let index_sibling = *domain_index ^ 1;

            let mut evals = vec![*folded_eval; 2];
            evals[index_sibling % 2] = sibling_value;

            *domain_index >>= 1;
            *folded_eval =
                folding.fold_row(*domain_index, log_folded_height, beta, evals.into_iter());

            // Roll in reduced openings at this height
            if let Some((lh, _)) = remaining_ro.first() {
                if *lh == log_folded_height {
                    let (_, ro) = remaining_ro.remove(0);
                    *folded_eval += beta.square() * ro;
                }
            }
        }
    }

    // Verify all reduced openings were consumed
    for (_, _, remaining_ro) in &query_state {
        if !remaining_ro.is_empty() {
            return Err(FriError::InvalidProofShape);
        }
    }

    // Check final polynomial for each query
    for (domain_index, folded_eval, _) in &query_state {
        let x = Val::two_adic_generator(log_global_max_height)
            .exp_u64(reverse_bits_len(*domain_index, log_global_max_height) as u64);

        let mut eval = Challenge::ZERO;
        for &coeff in proof.final_poly.iter().rev() {
            eval = eval * x + coeff;
        }

        if eval != *folded_eval {
            return Err(FriError::FinalPolyMismatch);
        }
    }

    Ok(())
}
