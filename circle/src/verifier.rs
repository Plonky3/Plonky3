use alloc::vec::Vec;

use itertools::Itertools;
use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::{BatchOpeningRef, Mmcs};
use p3_field::{ExtensionField, Field};
use p3_fri::verifier::FriError;
use p3_fri::{FriFoldingStrategy, FriParameters};
use p3_matrix::Dimensions;

use crate::{CircleCommitPhaseProofStep, CircleFriProof};

pub fn verify<Folding, Val, Challenge, M, Challenger>(
    folding: &Folding,
    params: &FriParameters<M>,
    proof: &CircleFriProof<Challenge, M, Challenger::Witness, Folding::InputProof>,
    challenger: &mut Challenger,
    open_input: impl Fn(
        usize,
        &Folding::InputProof,
    ) -> Result<Vec<(usize, Challenge)>, Folding::InputError>,
) -> Result<(), FriError<M::Error, Folding::InputError>>
where
    Val: Field,
    Challenge: ExtensionField<Val>,
    M: Mmcs<Challenge>,
    Challenger: FieldChallenger<Val> + GrindingChallenger + CanObserve<M::Commitment>,
    Folding: FriFoldingStrategy<Val, Challenge>,
{
    // Phase 1: Derive folding challenges
    //
    // In Circle-FRI, the verifier must produce one random challenge (beta)
    // per commit-phase round. Each commitment is observed into the Fiat-Shamir
    // transcript, then a challenge is sampled.
    // This yields exactly as many betas as there are commit-phase rounds.
    let betas: Vec<Challenge> = proof
        .commit_phase_commits
        .iter()
        .map(|comm| {
            // Absorb this round's commitment into the transcript.
            challenger.observe(comm.clone());
            // Squeeze a field-extension element to use as the folding challenge.
            challenger.sample_algebra_element()
        })
        .collect();

    // Absorb the prover's claimed constant polynomial into the transcript.
    // After all folding rounds, the result should reduce to this constant.
    challenger.observe_algebra_element(proof.final_poly);

    // Phase 2: Structural shape checks
    //
    // Before doing any expensive cryptographic work, validate that the proof
    // has the right shape. A malicious prover could submit too few (or too
    // many) query proofs, mismatched opening counts, or inconsistent arity
    // schedules. Catching these early avoids wasted work and gives precise
    // error variants.

    // The number of query proofs must match the security parameter.
    if proof.query_proofs.len() != params.num_queries {
        return Err(FriError::QueryProofCountMismatch {
            expected: params.num_queries,
            got: proof.query_proofs.len(),
        });
    }

    // Verify proof-of-work: a grinding witness that the prover must compute
    // to raise the cost of brute-forcing query positions.
    if !challenger.check_witness(params.query_proof_of_work_bits, proof.pow_witness) {
        return Err(FriError::InvalidPowWitness);
    }

    // Each query proof must carry exactly one opening per commit-phase round.
    //
    //     commit_phase_commits:       [c_0, c_1, ..., c_{n-1}]   (n rounds)
    //     qp.commit_phase_openings:   [o_0, o_1, ..., o_{n-1}]   (must also be n)
    //
    // A mismatch means the prover omitted or duplicated round data.
    let expected_rounds = proof.commit_phase_commits.len();
    for (query, qp) in proof.query_proofs.iter().enumerate() {
        let got_rounds = qp.commit_phase_openings.len();
        if got_rounds != expected_rounds {
            return Err(FriError::QueryCommitPhaseOpeningsCountMismatch {
                query,
                expected: expected_rounds,
                got: got_rounds,
            });
        }
    }

    // In variable-arity FRI, each round folds by 2^{log_arity_i} points.
    // All query proofs must agree on the per-round arity schedule, otherwise
    // different queries would fold through incompatible domain decompositions.
    //
    // We take the first query proof's schedule as the reference:
    let log_arities: Vec<usize> = proof
        .query_proofs
        .first()
        .map(|qp| {
            qp.commit_phase_openings
                .iter()
                .map(|o| o.log_arity as usize)
                .collect()
        })
        .unwrap_or_default();

    // Compare every subsequent query proof against the reference schedule.
    for (query, qp) in proof.query_proofs.iter().enumerate().skip(1) {
        let got_log_arities: Vec<usize> = qp
            .commit_phase_openings
            .iter()
            .map(|o| o.log_arity as usize)
            .collect();
        if got_log_arities != log_arities {
            return Err(FriError::QueryLogAritiesMismatch {
                query,
                expected: log_arities,
                got: got_log_arities,
            });
        }
    }

    // Phase 3: Query verification
    //
    // The initial evaluation domain has size 2^{log_max_height}, where
    // log_max_height = sum(log_arities) + log_blowup.
    // Each folding round reduces the domain by 2^{log_arity_i}, so after
    // all rounds the domain shrinks to 2^{log_blowup} (the blowup factor).
    let total_log_reduction: usize = log_arities.iter().sum();
    let log_max_height = total_log_reduction + params.log_blowup;

    for qp in &proof.query_proofs {
        // Sample a random query index uniformly from the initial domain.
        let index = challenger.sample_bits(log_max_height + folding.extra_query_index_bits());

        // Open the input polynomials at this query index.
        // Returns (log_height, evaluation) pairs sorted by height descending.
        let ro = open_input(index, &qp.input_proof).map_err(FriError::InputError)?;

        // Sanity check: reduced openings must arrive in strictly descending
        // height order so they are folded in at the correct domain sizes.
        debug_assert!(
            ro.iter().tuple_windows().all(|((l, _), (r, _))| l > r),
            "reduced openings sorted by height descending"
        );

        // Zip the challenges, commitments, and openings together for folding.
        //
        // Invariant: all three iterators have the same length here.
        // - The challenges are derived from the commitments (one per round).
        // - The openings count was validated to match the commitment count.
        // Plain zip is safe; it cannot silently truncate.
        let fold_data_iter = betas
            .iter()
            .zip(proof.commit_phase_commits.iter())
            .zip(qp.commit_phase_openings.iter());

        // Walk the FRI folding chain: at each round, verify the Merkle
        // opening against the commitment, then fold the sibling evaluations
        // using the challenge beta to produce the next-round evaluation.
        let folded_eval = verify_query(
            folding,
            params,
            index >> folding.extra_query_index_bits(),
            fold_data_iter,
            ro,
            log_max_height,
        )?;

        // After all rounds, the polynomial has been folded to a constant.
        // That constant must equal the prover's claimed final polynomial.
        if folded_eval != proof.final_poly {
            return Err(FriError::FinalPolyMismatch);
        }
    }

    Ok(())
}

/// One round's worth of data needed to verify a Circle-FRI fold.
///
/// Groups together:
/// - The random folding challenge for this round.
/// - The Merkle commitment to the evaluations on this round's domain.
/// - The prover-supplied sibling values and Merkle opening proof.
type CommitStep<'a, F, M> = (
    (&'a F, &'a <M as Mmcs<F>>::Commitment),
    &'a CircleCommitPhaseProofStep<F, M>,
);

/// Verify one query chain in the Circle-FRI proof.
///
/// Starting from a leaf in the initial evaluation domain, this walks
/// up the folding tree one round at a time:
///
/// ```text
///     domain size:  2^{log_max_height}  →  ...  →  2^{log_blowup}
///     round:              0                           last
/// ```
///
/// At each round:
/// - Roll in any reduced openings whose height matches the current domain.
/// - Reconstruct the full sibling group from the queried evaluation
///   plus the (arity - 1) sibling values provided by the prover.
/// - Verify the Merkle opening against the round commitment.
/// - Fold the sibling group with the challenge beta to produce the
///   parent evaluation for the next round.
///
/// With variable arity, each round may fold by a different factor
/// (2^{log_arity_i} siblings per group).
///
/// # Returns
///
/// The final folded evaluation, which the caller checks against
/// the prover's claimed constant.
fn verify_query<'a, Folding, F, EF, M>(
    folding: &Folding,
    params: &FriParameters<M>,
    mut index: usize,
    steps: impl ExactSizeIterator<Item = CommitStep<'a, EF, M>>,
    reduced_openings: Vec<(usize, EF)>,
    log_max_height: usize,
) -> Result<EF, FriError<M::Error, Folding::InputError>>
where
    F: Field,
    EF: ExtensionField<F>,
    M: Mmcs<EF> + 'a,
    Folding: FriFoldingStrategy<F, EF>,
{
    // Running accumulator: starts at zero and accumulates reduced openings
    // and folding results as we walk up the tree.
    let mut folded_eval = EF::ZERO;

    // Reduced openings arrive sorted by height descending.
    // We consume them as the current domain height matches.
    let mut ro_iter = reduced_openings.into_iter().peekable();

    // Current domain size is 2^{log_current_height}; decreases each round.
    let mut log_current_height = log_max_height;

    for (round, ((&beta, comm), opening)) in steps.enumerate() {
        // This round folds 2^{log_arity} siblings into one parent.
        let log_arity = opening.log_arity as usize;
        let arity = 1 << log_arity;

        // Shape check: the prover must supply exactly (arity - 1) siblings.
        // The queried evaluation itself is the remaining one, so the full
        // group has arity elements total.
        //
        //     sibling_values: [s_0, s_1, ..., s_{arity-2}]   (arity - 1 elements)
        //     queried value:  folded_eval                     (1 element)
        //     full group:     arity elements
        if opening.sibling_values.len() != arity - 1 {
            return Err(FriError::SiblingValuesLengthMismatch {
                round,
                expected: arity - 1,
                got: opening.sibling_values.len(),
            });
        }

        // If there are input polynomials evaluated at this domain height,
        // add their contribution before folding. This is the "roll-in" step
        // that combines multiple polynomials into the FRI batch.
        if let Some((_, ro)) = ro_iter.next_if(|(lh, _)| *lh == log_current_height) {
            folded_eval += ro;
        }

        // Reconstruct the full evaluation group for this node.
        // The queried index within the group tells us where our value sits;
        // the prover's sibling values fill the remaining positions.
        //
        //     arity = 4, index_in_group = 1:
        //     evals = [sibling_0, folded_eval, sibling_1, sibling_2]
        let index_in_group = index % arity;
        let mut evals = EF::zero_vec(arity);
        evals[index_in_group] = folded_eval;

        // Fill in siblings at every position except the queried one.
        let mut sibling_idx = 0;
        #[allow(clippy::needless_range_loop)]
        for j in 0..arity {
            if j != index_in_group {
                evals[j] = opening.sibling_values[sibling_idx];
                sibling_idx += 1;
            }
        }

        // After folding, the domain halves (or shrinks by 2^{log_arity}).
        let log_folded_height = log_current_height - log_arity;

        // Dimensions for the MMCS verification: one matrix of width = arity
        // at the folded height. This tells the Merkle tree the expected shape.
        let dims = &[Dimensions {
            width: arity,
            height: 1 << log_folded_height,
        }];

        // Move from the leaf index to its parent in the folding tree.
        index >>= log_arity;

        // Verify the Merkle opening: the sibling evaluations the prover
        // gave us must be consistent with the round commitment.
        params
            .mmcs
            .verify_batch(
                comm,
                dims,
                index,
                BatchOpeningRef::new(&[evals.clone()], &opening.opening_proof),
            )
            .map_err(FriError::CommitPhaseMmcsError)?;

        // Fold the full sibling group down to a single evaluation using
        // the random challenge beta. This is the core FRI step.
        folded_eval =
            folding.fold_row(index, log_folded_height, log_arity, beta, evals.into_iter());

        // Advance to the next (smaller) domain.
        log_current_height = log_folded_height;
    }

    // After all rounds, we should have folded down to 2^{log_blowup}.
    // If not, the proof has the wrong number of rounds for the domain size.
    if log_current_height != params.log_blowup {
        return Err(FriError::FinalFoldHeightMismatch {
            expected: params.log_blowup,
            got: log_current_height,
        });
    }

    // All input polynomial evaluations should have been consumed during
    // folding. Leftovers mean the proof contains data for heights that
    // were never reached.
    if let Some((next_log_height, _)) = ro_iter.next() {
        return Err(FriError::UnconsumedReducedOpenings {
            next_log_height,
            remaining: 1 + ro_iter.count(),
        });
    }

    Ok(folded_eval)
}
