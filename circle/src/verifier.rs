use alloc::vec;
use alloc::vec::Vec;
use core::iter;

use itertools::{Itertools, izip};
use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_field::ExtensionField;
use p3_field::extension::ComplexExtendable;
use p3_fri::verifier::FriError;
use p3_fri::{FriFoldingStrategy, FriParameters};
use p3_matrix::Dimensions;

use crate::folding::{fold_row_with_inv_twiddle, query_x_twiddles_inv};
use crate::{CircleCommitPhaseMultiStep, CircleFriProof};

/// Arguments:
/// - `open_inputs`: checks every input commitment's shared multi-opening and returns,
///   for each query, its reduced openings sorted by height descending.
pub fn verify<Folding, Val, Challenge, M, Challenger>(
    folding: &Folding,
    params: &FriParameters<M>,
    proof: &CircleFriProof<Challenge, M, Challenger::Witness, Folding::InputProof>,
    challenger: &mut Challenger,
    open_inputs: impl FnOnce(
        &[usize],
        &Folding::InputProof,
    ) -> Result<Vec<Vec<(usize, Challenge)>>, Folding::InputError>,
) -> Result<(), FriError<M::Error, Folding::InputError>>
where
    Val: ComplexExtendable,
    Challenge: ExtensionField<Val>,
    M: Mmcs<Challenge>,
    Challenger: FieldChallenger<Val> + GrindingChallenger + CanObserve<M::Commitment>,
    Folding: FriFoldingStrategy<Val, Challenge>,
{
    // Reject a vacuous instance before any transcript work.
    // With zero queries the per-query loop never runs.
    // Any final polynomial would then pass.
    if params.num_queries == 0 {
        return Err(FriError::ZeroQueries);
    }

    // There must be exactly one commit-phase proof-of-work witness per round.
    if proof.commit_pow_witnesses.len() != proof.commit_phase_commits.len() {
        return Err(FriError::CommitPowWitnessCountMismatch {
            expected: proof.commit_phase_commits.len(),
            got: proof.commit_pow_witnesses.len(),
        });
    }

    // Phase 1: Derive folding challenges
    //
    // In Circle-FRI, the verifier must produce one random challenge (beta)
    // per commit-phase round. Each commitment is observed into the Fiat-Shamir
    // transcript, the round's PoW witness is checked, then a challenge is sampled.
    // This yields exactly as many betas as there are commit-phase rounds.
    let betas: Vec<Challenge> = proof
        .commit_phase_commits
        .iter()
        .zip(&proof.commit_pow_witnesses)
        .map(|(comm, witness)| {
            // Absorb this round's commitment into the transcript.
            challenger.observe(comm.clone());
            // Check the per-round grinding witness before sampling the challenge.
            if !challenger.check_witness(params.commit_proof_of_work_bits, *witness) {
                return Err(FriError::InvalidPowWitness);
            }
            // Squeeze a field-extension element to use as the folding challenge.
            Ok(challenger.sample_algebra_element())
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Absorb the prover's claimed constant polynomial into the transcript.
    // After all folding rounds, the result should reduce to this constant.
    challenger.observe_algebra_element(proof.final_poly);

    // Phase 2: Structural shape checks
    //
    // Before doing any expensive cryptographic work, validate that the proof
    // has the right shape. A malicious prover could submit too few (or too
    // many) round openings, mismatched query counts, or an invalid arity
    // schedule. Catching these early avoids wasted work and gives precise
    // error variants.

    // One commit-phase opening set per commitment.
    let expected_rounds = proof.commit_phase_commits.len();
    if proof.commit_phase_openings.len() != expected_rounds {
        return Err(FriError::CommitPhaseOpeningsCountMismatch {
            expected: expected_rounds,
            got: proof.commit_phase_openings.len(),
        });
    }

    // In variable-arity FRI, each round folds by 2^{log_arity_i} points. The
    // schedule is a protocol-wide constant, so it lives once per round.
    let log_arities: Vec<usize> = proof
        .commit_phase_openings
        .iter()
        .enumerate()
        .map(|(round, opening)| {
            opening
                .checked_log_arity(params.max_log_arity)
                .ok_or(FriError::InvalidLogArity {
                    round,
                    log_arity: opening.log_arity as usize,
                    max: params.max_log_arity,
                })
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Every round must open every query, and each opening must carry exactly
    // arity - 1 sibling values.
    //
    //     sibling_values[query]: [s_0, ..., s_{arity-2}]   (arity - 1 elements)
    //     queried value:         folded_eval                (1 element)
    //     full group:            arity elements
    for (round, (opening, &log_arity)) in
        izip!(&proof.commit_phase_openings, &log_arities).enumerate()
    {
        if opening.sibling_values.len() != params.num_queries {
            return Err(FriError::CommitPhaseQueryCountMismatch {
                round,
                expected: params.num_queries,
                got: opening.sibling_values.len(),
            });
        }
        let arity = 1 << log_arity;
        for siblings in &opening.sibling_values {
            if siblings.len() != arity - 1 {
                return Err(FriError::SiblingValuesLengthMismatch {
                    round,
                    expected: arity - 1,
                    got: siblings.len(),
                });
            }
        }
    }

    // Verify proof-of-work: a grinding witness that the prover must compute
    // to raise the cost of brute-forcing query positions.
    if !challenger.check_witness(params.query_proof_of_work_bits, proof.pow_witness) {
        return Err(FriError::InvalidPowWitness);
    }

    // Phase 3: Query verification
    //
    // The initial evaluation domain has size 2^{log_max_height}, where
    // log_max_height = sum(log_arities) + log_blowup.
    // Each folding round reduces the domain by 2^{log_arity_i}, so after
    // all rounds the domain shrinks to 2^{log_blowup} (the blowup factor).
    let total_log_reduction: usize = log_arities.iter().sum();
    let log_max_height = total_log_reduction + params.log_blowup;

    // Invariant: the query-index width fits the circle group of order 2^CIRCLE_TWO_ADICITY.
    //
    //     num_index_bits = sum(log_arities) + log_blowup + extra_query_index_bits
    //     field order    = 2^CIRCLE_TWO_ADICITY - 1   (one short of the group order)
    //     => a width of CIRCLE_TWO_ADICITY bits is unsampleable
    //
    // A malformed arity schedule inflates the round count, hence the width.
    let num_index_bits = log_max_height + folding.extra_query_index_bits();
    if num_index_bits >= Val::CIRCLE_TWO_ADICITY {
        return Err(FriError::GlobalMaxHeightTooLarge {
            log_global_max_height: num_index_bits,
            two_adicity: Val::CIRCLE_TWO_ADICITY,
        });
    }

    // Sample every query index. The transcript is identical to sampling one
    // index per query proof: nothing is observed between samples.
    let indices: Vec<usize> = iter::repeat_with(|| challenger.sample_bits(num_index_bits))
        .take(params.num_queries)
        .collect();

    // Check the input commitments' shared multi-openings and reduce each query's
    // opened rows to (log_height, evaluation) pairs sorted by height descending.
    let reduced_openings =
        open_inputs(&indices, &proof.input_openings).map_err(FriError::InputError)?;

    // Walk every query's fold chain (pure arithmetic), reconstructing the full
    // evaluation row the prover committed to at each round. The rows are
    // authenticated afterwards, one shared check per round.
    let mut group_indices_by_round: Vec<Vec<usize>> =
        vec![Vec::with_capacity(params.num_queries); expected_rounds];
    // `rows_by_round[round][query]` holds the opened rows of the round's single
    // committed matrix, in the `opened_values[query][matrix]` shape that the
    // multi-opening verification expects.
    let mut rows_by_round: Vec<Vec<Vec<Vec<Challenge>>>> =
        vec![Vec::with_capacity(params.num_queries); expected_rounds];

    for (query, (&index, ro)) in izip!(&indices, reduced_openings).enumerate() {
        // Sanity check: reduced openings must arrive in strictly descending
        // height order so they are folded in at the correct domain sizes.
        debug_assert!(
            ro.iter().tuple_windows().all(|((l, _), (r, _))| l > r),
            "reduced openings sorted by height descending"
        );

        // The whole x-fold chain for this query is index-derived (no Merkle or proof data
        // needed), so it is precomputed and batch-inverted once up front instead of each
        // round recomputing its own twiddle from scratch.
        let top_level_index = index >> folding.extra_query_index_bits();
        let x_twiddle_inv =
            query_x_twiddles_inv::<Val>(top_level_index, log_max_height, log_arities.len());

        let folded_eval = fold_query(
            params,
            query,
            top_level_index,
            &betas,
            &log_arities,
            &proof.commit_phase_openings,
            ro,
            log_max_height,
            &x_twiddle_inv,
            &mut group_indices_by_round,
            &mut rows_by_round,
        )?;

        // After all rounds, the polynomial has been folded to a constant.
        // That constant must equal the prover's claimed final polynomial.
        if folded_eval != proof.final_poly {
            return Err(FriError::FinalPolyMismatch);
        }
    }

    // Verify the commitment to the evaluations of every queried group, one shared
    // amortized check per round. Paths that share a parent reuse a single
    // compression instead of recomputing it once per query.
    let mut log_current_height = log_max_height;
    for (round, ((comm, opening), &log_arity)) in izip!(
        proof
            .commit_phase_commits
            .iter()
            .zip(&proof.commit_phase_openings),
        &log_arities
    )
    .enumerate()
    {
        let arity = 1 << log_arity;
        let log_folded_height = log_current_height - log_arity;
        let dims = &[Dimensions {
            width: arity,
            height: 1 << log_folded_height,
        }];
        params
            .mmcs
            .verify_multi_batch(
                comm,
                dims,
                &group_indices_by_round[round],
                &rows_by_round[round],
                &opening.opening_proof,
            )
            .map_err(FriError::CommitPhaseMmcsError)?;
        log_current_height = log_folded_height;
    }

    Ok(())
}

/// Fold one query chain in the Circle-FRI proof.
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
/// - Record the group index and reconstructed row for the round's shared
///   authentication, performed once by the caller.
/// - Fold the sibling group with the challenge beta to produce the
///   parent evaluation for the next round.
///
/// This pass is pure arithmetic; nothing here reads a Merkle proof.
///
/// # Returns
///
/// The final folded evaluation, which the caller checks against
/// the prover's claimed constant.
#[expect(clippy::too_many_arguments)]
fn fold_query<F, EF, M, InputError>(
    params: &FriParameters<M>,
    query: usize,
    mut index: usize,
    betas: &[EF],
    log_arities: &[usize],
    commit_phase_openings: &[CircleCommitPhaseMultiStep<EF, M>],
    reduced_openings: Vec<(usize, EF)>,
    log_max_height: usize,
    x_twiddle_inv: &[F],
    group_indices_by_round: &mut [Vec<usize>],
    rows_by_round: &mut [Vec<Vec<Vec<EF>>>],
) -> Result<EF, FriError<M::Error, InputError>>
where
    F: ComplexExtendable,
    EF: ExtensionField<F>,
    M: Mmcs<EF>,
    InputError: core::fmt::Debug,
{
    // Running accumulator: starts at zero and accumulates reduced openings
    // and folding results as we walk up the tree.
    let mut folded_eval = EF::ZERO;

    // Reduced openings arrive sorted by height descending.
    // We consume them as the current domain height matches.
    let mut ro_iter = reduced_openings.into_iter().peekable();

    // Current domain size is 2^{log_current_height}; decreases each round.
    let mut log_current_height = log_max_height;

    for (round, (&beta, &log_arity, opening)) in
        izip!(betas, log_arities, commit_phase_openings).enumerate()
    {
        let arity = 1 << log_arity;

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
        //
        // The row is authenticated by the round's shared opening proof, which
        // binds this query's folded value to the committed codeword.
        let index_in_group = index % arity;
        let mut evals = EF::zero_vec(arity);
        evals[index_in_group] = folded_eval;

        // Fill in siblings at every position except the queried one.
        let siblings = &opening.sibling_values[query];
        let mut sibling_idx = 0;
        for (j, eval) in evals.iter_mut().enumerate() {
            if j != index_in_group {
                *eval = siblings[sibling_idx];
                sibling_idx += 1;
            }
        }

        // After folding, the domain halves (or shrinks by 2^{log_arity}).
        let log_folded_height = log_current_height - log_arity;

        // Move from the leaf index to its parent in the folding tree.
        index >>= log_arity;

        // Record the group index and reconstructed row for the round's shared
        // verification. `evals` is tiny, so the clone is cheap.
        group_indices_by_round[round].push(index);
        rows_by_round[round].push(vec![evals.clone()]);

        // Fold the full sibling group down to a single evaluation using the random
        // challenge beta. Circle PCS only ever folds by arity 2 (`CircleFriFolding::fold_row`
        // asserts this too); the twiddle for this round was already precomputed for the
        // whole query chain, so this is now pure arithmetic with no domain construction,
        // scalar multiplication, or inversion left to do.
        assert_eq!(log_arity, 1, "Circle PCS currently only supports arity 2");
        folded_eval = fold_row_with_inv_twiddle(x_twiddle_inv[round], beta, evals.into_iter());

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
