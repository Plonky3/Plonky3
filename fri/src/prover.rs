use alloc::vec;
use alloc::vec::Vec;
use core::iter;

use itertools::{izip, Itertools};
use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_field::{ExtensionField, Field};
use p3_matrix::dense::RowMajorMatrix;
use p3_util::log2_strict_usize;
use tracing::{info_span, instrument};

use crate::{CommitPhaseProofStep, FriConfig, FriGenericConfig, FriProof, QueryProof};

#[instrument(name = "FRI prover", skip_all)]
pub fn prove<G, Val, Challenge, M, Challenger>(
    g: &G,
    config: &FriConfig<M>,
    inputs: Vec<Vec<Challenge>>,
    challenger: &mut Challenger,
    open_input: impl Fn(usize) -> G::InputProof,
) -> FriProof<Challenge, M, Challenger::Witness, G::InputProof>
where
    Val: Field,
    Challenge: ExtensionField<Val>,
    M: Mmcs<Challenge>,
    Challenger: FieldChallenger<Val> + GrindingChallenger + CanObserve<M::Commitment>,
    G: FriGenericConfig<Challenge>,
{
    // check sorted descending
    assert!(inputs
        .iter()
        .tuple_windows()
        .all(|(l, r)| l.len() >= r.len()));

    let log_max_height = log2_strict_usize(inputs[0].len());

    let commit_phase_result = commit_phase(g, config, inputs, challenger);

    let pow_witness = challenger.grind(config.proof_of_work_bits);

    let query_proofs = info_span!("query phase").in_scope(|| {
        iter::repeat_with(|| challenger.sample_bits(log_max_height + g.extra_query_index_bits()))
            .take(config.num_queries)
            .map(|index| QueryProof {
                input_proof: open_input(index),
                commit_phase_openings: answer_query(
                    config,
                    &commit_phase_result.data,
                    index >> g.extra_query_index_bits(),
                ),
            })
            .collect()
    });

    FriProof {
        log_max_height,
        folded_evals: commit_phase_result.folded_evals,
        commit_phase_commits: commit_phase_result.commits,
        query_proofs,
        final_poly: commit_phase_result.final_poly,
        pow_witness,
    }
}

struct CommitPhaseResult<F: Field, M: Mmcs<F>> {
    commits: Vec<M::Commitment>,
    data: Vec<M::ProverData<RowMajorMatrix<F>>>,
    // TODO[osama]: think if this can be done more efficiently
    folded_evals: Vec<Vec<F>>,
    final_poly: F,
}

#[instrument(name = "commit phase", skip_all)]
fn commit_phase<G, Val, Challenge, M, Challenger>(
    g: &G,
    config: &FriConfig<M>,
    inputs: Vec<Vec<Challenge>>,
    challenger: &mut Challenger,
) -> CommitPhaseResult<Challenge, M>
where
    Val: Field,
    Challenge: ExtensionField<Val>,
    M: Mmcs<Challenge>,
    Challenger: FieldChallenger<Val> + CanObserve<M::Commitment>,
    G: FriGenericConfig<Challenge>,
{
    let mut inputs_iter = inputs.into_iter().peekable();
    let mut folded = inputs_iter.next().unwrap();
    let mut commits = vec![];
    let mut data = vec![];
    let mut folded_evals = vec![];

    // TODO: add better comments
    // We only commit every arity_bits layers

    let mut next_commit_layer = folded.len();

    while folded.len() > config.blowup() {
        folded_evals.push(folded.clone());

        let pairs = RowMajorMatrix::new(folded.clone(), 2);
        let mut folding_beta = Challenge::ONE;

        if folded.len() == next_commit_layer {
            // Commit layer
            next_commit_layer = folded.len() / (1 << config.arity_bits);

            let folded_matrix = RowMajorMatrix::new(folded, 1 << config.arity_bits);
            let (commit, prover_data) = config.mmcs.commit_matrix(folded_matrix);
            challenger.observe(commit.clone());

            folding_beta = challenger.sample_ext_element();

            commits.push(commit);
            data.push(prover_data);
        }

        // Fold the current layer by a factor of 2
        folded = g.fold_matrix(folding_beta, pairs);

        if let Some(v) = inputs_iter.next_if(|v| v.len() == folded.len()) {
            izip!(&mut folded, v).for_each(|(c, x)| *c += x);
        }
    }

    // We should be left with at most `blowup` evaluations of a constant polynomial.
    let final_poly = folded[0];
    for x in folded {
        assert_eq!(x, final_poly);
    }
    challenger.observe_ext_element(final_poly);

    CommitPhaseResult {
        commits,
        data,
        folded_evals,
        final_poly,
    }
}

fn answer_query<F, M>(
    config: &FriConfig<M>,
    commit_phase_commits: &[M::ProverData<RowMajorMatrix<F>>],
    index: usize,
) -> Vec<CommitPhaseProofStep<F, M>>
where
    F: Field,
    M: Mmcs<F>,
{
    commit_phase_commits
        .iter()
        .enumerate()
        .map(|(i, commit)| {
            let index_i = index >> (i * config.arity_bits);
            // let index_i_sibling = index_i ^ 1;
            let index_row = index_i >> config.arity_bits;

            let (mut opened_rows, opening_proof) = config.mmcs.open_batch(index_row, commit);
            assert_eq!(opened_rows.len(), 1);
            let opened_row = opened_rows.pop().unwrap();
            assert_eq!(
                opened_row.len(),
                1 << config.arity_bits,
                "Committed data should be of the correct arity"
            );

            CommitPhaseProofStep {
                opened_row,
                opening_proof,
            }
        })
        .collect()
}
