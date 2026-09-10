use alloc::vec;
use alloc::vec::Vec;

use itertools::{Itertools, izip};
use p3_commit::Mmcs;
use p3_field::extension::ComplexExtendable;
use p3_field::{ExtensionField, Field};
use p3_fri::verifier::{FriError, PowPhase};
use p3_fri::{FriFoldingStrategy, FriParameters};
use p3_matrix::Dimensions;

use crate::folding::{fold_row_with_inv_twiddle, query_x_twiddles_inv};
use crate::{CircleCommitPhaseMultiStep, CircleFriProof};

/// Check every length a Circle-FRI proof declares against the configured run.
///
/// # Overview
///
/// Nothing here reads a challenge, and nothing here is described by the transcript.
///
/// It runs before the transcript is seeded.
/// A proof of the wrong shape is therefore rejected without a driver ever existing.
///
/// # Arguments
///
/// - `params`: the parameters for this FRI instance.
/// - `proof`: the proof whose declared lengths are being checked.
/// - `num_commit_rounds`: the round count the configuration fixes.
///
/// # Returns
///
/// The per-round log-arity schedule, one entry per commit round.
///
/// # Errors
///
/// - The instance is vacuous: no queries, or rate 1.
/// - The folding cap is one this verifier cannot fold with.
/// - The proof declares a round count the configuration does not fix.
/// - A per-round list does not carry one entry per round, or one entry per query.
/// - An opened group is not exactly one value short of its round's arity.
/// - A grinding witness is not the value its zero difficulty admits.
pub(crate) fn validate_proof_shape<Challenge, M, Witness, InputProof, InputErr>(
    params: &FriParameters<M>,
    proof: &CircleFriProof<Challenge, M, Witness, InputProof>,
    num_commit_rounds: usize,
) -> Result<Vec<usize>, FriError<M::Error, InputErr>>
where
    Challenge: Field,
    M: Mmcs<Challenge>,
    Witness: Field,
    InputErr: core::fmt::Debug,
{
    // Reject a vacuous instance before any transcript work.
    // With zero queries the per-query loop never runs.
    // Any final polynomial would then pass.
    if params.num_queries == 0 {
        return Err(FriError::ZeroQueries);
    }
    // Reject rate-1 FRI: every length-N word is a degree-<N codeword, so an arbitrary
    // reduced-opening word would pass the folding chain unchecked.
    if params.log_blowup == 0 {
        return Err(FriError::ZeroBlowup);
    }

    // Circle folding halves the domain and nothing else.
    //
    // Capping the arity at one forces every per-round arity to one.
    // The pinned height sum then determines the schedule uniquely.
    //
    // A larger cap would let a proof declare an arity this fold cannot apply.
    if params.max_log_arity != 1 {
        return Err(FriError::UnsupportedFoldingCap {
            max_log_arity: params.max_log_arity,
        });
    }

    // The round count is fixed by the claimed heights, not by the proof.
    //
    //     H_claim = max claimed log_n + log_blowup
    //     rounds  = H_claim - 1 - log_blowup      (the first layer takes one bit)
    //
    // One commitment per round.
    //
    // So the two counts are the same number twice.
    if proof.commit_phase_commits.len() != num_commit_rounds {
        return Err(FriError::CommitRoundCountMismatch {
            expected: num_commit_rounds,
            got: proof.commit_phase_commits.len(),
        });
    }

    // There must be exactly one commit-phase proof-of-work witness per round.
    if proof.commit_pow_witnesses.len() != num_commit_rounds {
        return Err(FriError::CommitPowWitnessCountMismatch {
            expected: num_commit_rounds,
            got: proof.commit_pow_witnesses.len(),
        });
    }

    // A zero difficulty leaves both witnesses unread, so their values are pinned here
    // rather than by their grinds.
    //
    // Why: `check_witness` returns `true` at zero bits without absorbing, and the
    // transcript elides the step, so nothing downstream compares the field to anything.
    //
    //     bits = 0 -> prover emits zero, verifier reads nothing -> pin the field here
    //     bits > 0 -> prover grinds,     verifier resamples     -> the grind pins it
    //
    // Every commit-round entry is pinned, not only the count checked just above: a count
    // check is not a value check.
    if params.commit_proof_of_work_bits == 0
        && proof
            .commit_pow_witnesses
            .iter()
            .any(|w| *w != Witness::ZERO)
    {
        return Err(FriError::NonCanonicalPowWitness {
            phase: PowPhase::CommitPhase,
        });
    }
    if params.query_proof_of_work_bits == 0 && proof.pow_witness != Witness::ZERO {
        return Err(FriError::NonCanonicalPowWitness {
            phase: PowPhase::Query,
        });
    }

    // One commit-phase opening set per commitment.
    if proof.commit_phase_openings.len() != num_commit_rounds {
        return Err(FriError::CommitPhaseOpeningsCountMismatch {
            expected: num_commit_rounds,
            got: proof.commit_phase_openings.len(),
        });
    }

    // The folding schedule, derived rather than read.
    //
    // The cap check above pinned the arity to two.
    //
    // So every round folds by exactly one bit.
    //
    // The round count then fixes the whole schedule.
    let log_arities = vec![1; num_commit_rounds];

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

    Ok(log_arities)
}

/// Check every query's fold chain against the commit-phase commitments.
///
/// # Overview
///
/// This pass reads no challenge from a sponge.
///
/// The caller replayed the transcript, closed it, and hands the results here.
/// Everything below is arithmetic and Merkle work over already-validated shapes.
///
/// # Arguments
///
/// - `folding`: the Circle folding strategy.
/// - `params`: the parameters for this FRI instance.
/// - `proof`: the proof being checked.
/// - `betas`: the folding challenge of each round, redrawn from the transcript.
/// - `indices`: the query indices, redrawn from the transcript.
/// - `log_arities`: the validated schedule, one entry per round.
/// - `open_inputs`: checks every input commitment's shared multi-opening.
///   It returns, for each query, that query's reduced openings.
///   Those arrive sorted by height, tallest first.
///
/// # Errors
///
/// - An input opening fails its own check.
/// - A fold chain lands on a value the claimed constant does not match.
/// - A reconstructed row fails the round's shared authentication.
pub(crate) fn verify_queries<Folding, Val, Challenge, M, Witness>(
    folding: &Folding,
    params: &FriParameters<M>,
    proof: &CircleFriProof<Challenge, M, Witness, Folding::InputProof>,
    betas: &[Challenge],
    indices: &[usize],
    log_arities: &[usize],
    open_inputs: impl FnOnce(
        &[usize],
        &Folding::InputProof,
    ) -> Result<Vec<Vec<(usize, Challenge)>>, Folding::InputError>,
) -> Result<(), FriError<M::Error, Folding::InputError>>
where
    Val: ComplexExtendable,
    Challenge: ExtensionField<Val>,
    M: Mmcs<Challenge>,
    Folding: FriFoldingStrategy<Val, Challenge>,
{
    // The initial evaluation domain has size 2^{log_max_height}, where
    // log_max_height = sum(log_arities) + log_blowup.
    // Each folding round reduces the domain by 2^{log_arity_i}, so after
    // all rounds the domain shrinks to 2^{log_blowup} (the blowup factor).
    let total_log_reduction: usize = log_arities.iter().sum();
    let log_max_height = total_log_reduction + params.log_blowup;

    // Check the input commitments' shared multi-openings and reduce each query's
    // opened rows to (log_height, evaluation) pairs sorted by height descending.
    let reduced_openings =
        open_inputs(indices, &proof.input_openings).map_err(FriError::InputError)?;

    // Walk every query's fold chain (pure arithmetic), reconstructing the full
    // evaluation row the prover committed to at each round. The rows are
    // authenticated afterwards, one shared check per round.
    let num_rounds = log_arities.len();
    let mut group_indices_by_round: Vec<Vec<usize>> =
        vec![Vec::with_capacity(params.num_queries); num_rounds];
    // `rows_by_round[round][query]` holds the opened rows of the round's single
    // committed matrix, in the `opened_values[query][matrix]` shape that the
    // multi-opening verification expects.
    let mut rows_by_round: Vec<Vec<Vec<Vec<Challenge>>>> =
        vec![Vec::with_capacity(params.num_queries); num_rounds];

    for (query, (&index, ro)) in izip!(indices, reduced_openings).enumerate() {
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
            betas,
            log_arities,
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
        log_arities
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

        // Fold the full sibling group down to a single evaluation using the random
        // challenge beta. Borrowing the row leaves it owned for the collector below.
        //
        // The cap is one on entry, so this is always a halving.
        debug_assert_eq!(log_arity, 1, "circle folding is always a halving");
        folded_eval = fold_row_with_inv_twiddle(x_twiddle_inv[round], beta, evals.iter().copied());

        // Hand this query's group index and reconstructed row to the round's shared
        // verification; the round's single proof authenticates every query at once.
        group_indices_by_round[round].push(index);
        rows_by_round[round].push(vec![evals]);

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
