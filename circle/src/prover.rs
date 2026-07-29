use alloc::vec;
use alloc::vec::Vec;
use core::iter;

use itertools::{Itertools, izip};
use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_field::{ExtensionField, Field};
use p3_fri::{FriFoldingStrategy, FriParameters, compute_log_arity_for_round};
use p3_matrix::dense::RowMajorMatrix;
use p3_util::log2_strict_usize;
use tracing::{info_span, instrument};

use crate::{CircleCommitPhaseMultiStep, CircleFriProof};

/// Arguments:
/// - `open_inputs`: opens every input commitment at all query indices at once,
///   so the shared authentication paths can be deduplicated.
#[instrument(name = "FRI prover", skip_all)]
pub fn prove<Folding, Val, Challenge, M, Challenger>(
    folding: &Folding,
    params: &FriParameters<M>,
    inputs: Vec<Vec<Challenge>>,
    challenger: &mut Challenger,
    open_inputs: impl FnOnce(&[usize]) -> Folding::InputProof,
) -> CircleFriProof<Challenge, M, Challenger::Witness, Folding::InputProof>
where
    Val: Field,
    Challenge: ExtensionField<Val>,
    M: Mmcs<Challenge>,
    Challenger: FieldChallenger<Val> + GrindingChallenger + CanObserve<M::Commitment>,
    Folding: FriFoldingStrategy<Val, Challenge>,
{
    assert!(
        params.max_log_arity > 0,
        "max_log_arity must be at least 1 to guarantee folding progress"
    );

    // A zero-query instance performs no low-degree spot checks.
    // The verifier rejects it, so refuse to emit one locally.
    assert!(
        params.num_queries > 0,
        "num_queries must be at least 1 for Circle-FRI soundness"
    );

    // check sorted descending
    assert!(
        inputs
            .iter()
            .tuple_windows()
            .all(|(l, r)| l.len() >= r.len())
    );

    let log_max_height = log2_strict_usize(inputs[0].len());

    let commit_phase_result = commit_phase(folding, params, inputs, challenger);

    let pow_witness = challenger.grind(params.query_proof_of_work_bits);

    // Sampling every index in one block leaves the transcript identical to sampling
    // them one query at a time: nothing is observed between samples.
    let indices: Vec<usize> = iter::repeat_with(|| {
        challenger.sample_bits(log_max_height + folding.extra_query_index_bits())
    })
    .take(params.num_queries)
    .collect();

    let (input_openings, commit_phase_openings) = info_span!("query phase").in_scope(|| {
        // Openings of the inputs and of every commit-phase codeword at all queried
        // locations. Queries into one tree share a single proof, so overlapping
        // authentication paths are deduplicated instead of being shipped once per query.
        let input_openings = open_inputs(&indices);
        let domain_indices: Vec<usize> = indices
            .iter()
            .map(|&index| index >> folding.extra_query_index_bits())
            .collect();
        let commit_phase_openings = answer_queries(
            params,
            &commit_phase_result.log_arities,
            &commit_phase_result.data,
            &domain_indices,
        );
        (input_openings, commit_phase_openings)
    });

    CircleFriProof {
        commit_phase_commits: commit_phase_result.commits,
        commit_pow_witnesses: commit_phase_result.pow_witnesses,
        input_openings,
        commit_phase_openings,
        final_poly: commit_phase_result.final_poly,
        pow_witness,
    }
}

struct CommitPhaseResult<F: Field, M: Mmcs<F>, Witness> {
    commits: Vec<M::Commitment>,
    data: Vec<M::ProverData<RowMajorMatrix<F>>>,
    log_arities: Vec<usize>,
    pow_witnesses: Vec<Witness>,
    final_poly: F,
}

#[instrument(name = "commit phase", skip_all)]
fn commit_phase<Folding, Val, Challenge, M, Challenger>(
    folding: &Folding,
    params: &FriParameters<M>,
    inputs: Vec<Vec<Challenge>>,
    challenger: &mut Challenger,
) -> CommitPhaseResult<Challenge, M, Challenger::Witness>
where
    Val: Field,
    Challenge: ExtensionField<Val>,
    M: Mmcs<Challenge>,
    Challenger: FieldChallenger<Val> + GrindingChallenger + CanObserve<M::Commitment>,
    Folding: FriFoldingStrategy<Val, Challenge>,
{
    assert!(
        params.max_log_arity > 0,
        "max_log_arity must be at least 1 to guarantee folding progress"
    );

    let mut inputs_iter = inputs.into_iter().peekable();
    let mut folded = inputs_iter.next().unwrap();
    let mut commits = vec![];
    let mut data = vec![];
    let mut log_arities = vec![];
    let mut pow_witnesses = vec![];

    // For Circle, we fold down to blowup elements (no separate final_poly_len)
    let log_final_height = params.log_blowup;

    while folded.len() > params.blowup() {
        let log_current_height = log2_strict_usize(folded.len());
        let next_input_log_height = inputs_iter.peek().map(|v| log2_strict_usize(v.len()));

        // Compute the arity for this round
        let log_arity = compute_log_arity_for_round(
            log_current_height,
            next_input_log_height,
            log_final_height,
            params.max_log_arity,
        );
        let arity = 1 << log_arity;
        log_arities.push(log_arity);

        let leaves = RowMajorMatrix::new(folded, arity);
        let (commit, prover_data) = params.mmcs.commit_matrix(leaves);
        challenger.observe(commit.clone());

        pow_witnesses.push(challenger.grind(params.commit_proof_of_work_bits));

        let beta: Challenge = challenger.sample_algebra_element();
        // We passed ownership of `current` to the MMCS, so get a reference to it
        let leaves = params.mmcs.get_matrices(&prover_data).pop().unwrap();
        folded = folding.fold_matrix(beta, log_arity, leaves.as_view());

        commits.push(commit);
        data.push(prover_data);

        if let Some(v) = inputs_iter.next_if(|v| v.len() == folded.len()) {
            izip!(&mut folded, v).for_each(|(c, x)| *c += x);
        }
    }

    // We should be left with `blowup` evaluations of a constant polynomial.
    assert_eq!(folded.len(), params.blowup());
    let final_poly = folded[0];
    for x in folded {
        assert_eq!(x, final_poly);
    }
    challenger.observe_algebra_element(final_poly);

    CommitPhaseResult {
        commits,
        data,
        log_arities,
        pow_witnesses,
        final_poly,
    }
}

/// For each round, return every query's sibling values together with one shared
/// opening proof covering all of that round's queried groups.
fn answer_queries<F, M>(
    params: &FriParameters<M>,
    log_arities: &[usize],
    commit_phase_commits: &[M::ProverData<RowMajorMatrix<F>>],
    start_indices: &[usize],
) -> Vec<CircleCommitPhaseMultiStep<F, M>>
where
    F: Field,
    M: Mmcs<F>,
{
    let mut current_indices = start_indices.to_vec();

    commit_phase_commits
        .iter()
        .enumerate()
        .map(|(i, commit)| {
            let log_arity = log_arities[i];
            let arity = 1 << log_arity;

            // Index of each query's group (row in the committed matrix).
            let group_indices: Vec<usize> = current_indices
                .iter()
                .map(|&index| index >> log_arity)
                .collect();

            // One shared proof that all queried groups are correct.
            let (opened_rows, opening_proof) = params.mmcs.open_multi_batch(&group_indices, commit);

            // For each query, keep all siblings (exclude the queried element).
            let sibling_values: Vec<Vec<F>> = izip!(&current_indices, opened_rows)
                .map(|(&index, mut rows)| {
                    assert_eq!(rows.len(), 1);
                    let opened_row = rows.pop().unwrap();
                    assert_eq!(
                        opened_row.len(),
                        arity,
                        "Committed data should have arity {} elements",
                        arity
                    );
                    let index_in_group = index % arity;
                    opened_row
                        .into_iter()
                        .enumerate()
                        .filter(|(j, _)| *j != index_in_group)
                        .map(|(_, v)| v)
                        .collect()
                })
                .collect();

            // Move to the parent groups for the next round.
            current_indices = group_indices;

            CircleCommitPhaseMultiStep {
                log_arity: log_arity as u8,
                sibling_values,
                opening_proof,
            }
        })
        .collect()
}
