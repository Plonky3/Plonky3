use alloc::vec;
use alloc::vec::Vec;

use itertools::Itertools;
use p3_challenger::{CanObserve, CanSample, CanSampleBits, GrindingChallenger};
use p3_commit::{DirectMmcs, Mmcs};
use p3_matrix::dense::RowMajorMatrix;
use tracing::{info_span, instrument};

use crate::fold_even_odd::fold_even_odd;
use crate::{CommitPhaseProofStep, FriConfig, FriProof, QueryProof};

#[instrument(name = "FRI prover", skip_all)]
pub fn prove<FC: FriConfig>(
    config: &FC,
    input: &[Option<Vec<FC::Challenge>>; 32],
    challenger: &mut FC::Challenger,
) -> (FriProof<FC>, Vec<usize>) {
    let log_max_height = input.iter().rposition(Option::is_some).unwrap();

    let commit_phase_result = commit_phase::<FC>(config, input, log_max_height, challenger);

    let pow_witness = challenger.grind(config.proof_of_work_bits());

    let query_indices: Vec<usize> = (0..config.num_queries())
        .map(|_| challenger.sample_bits(log_max_height))
        .collect();

    let query_proofs = info_span!("query phase").in_scope(|| {
        query_indices
            .iter()
            .map(|&index| answer_query(config, &commit_phase_result.data, index))
            .collect()
    });

    (
        FriProof {
            commit_phase_commits: commit_phase_result.commits,
            query_proofs,
            final_poly: commit_phase_result.final_poly,
            pow_witness,
        },
        query_indices,
    )
}

fn answer_query<FC: FriConfig>(
    config: &FC,
    commit_phase_commits: &[<FC::CommitPhaseMmcs as Mmcs<FC::Challenge>>::ProverData],
    index: usize,
) -> QueryProof<FC> {
    let commit_phase_openings = commit_phase_commits
        .iter()
        .enumerate()
        .map(|(i, commit)| {
            let index_i = index >> i;
            let index_i_sibling = index_i ^ 1;
            let index_pair = index_i >> 1;

            let (mut opened_rows, opening_proof) =
                config.commit_phase_mmcs().open_batch(index_pair, commit);
            assert_eq!(opened_rows.len(), 1);
            let opened_row = opened_rows.pop().unwrap();
            assert_eq!(opened_row.len(), 2, "Committed data should be in pairs");
            let sibling_value = opened_row[index_i_sibling % 2];

            CommitPhaseProofStep {
                sibling_value,
                opening_proof,
            }
        })
        .collect();

    QueryProof {
        commit_phase_openings,
    }
}

#[instrument(name = "commit phase", skip_all)]
fn commit_phase<FC: FriConfig>(
    config: &FC,
    input: &[Option<Vec<FC::Challenge>>; 32],
    log_max_height: usize,
    challenger: &mut FC::Challenger,
) -> CommitPhaseResult<FC> {
    let mut current = input[log_max_height].as_ref().unwrap().clone();

    let mut commits = vec![];
    let mut data = vec![];

    for log_folded_height in (config.log_blowup()..log_max_height).rev() {
        let leaves = RowMajorMatrix::new(current.clone(), 2);
        let (commit, prover_data) = config.commit_phase_mmcs().commit_matrix(leaves);
        challenger.observe(commit.clone());
        commits.push(commit);
        data.push(prover_data);

        let beta: FC::Challenge = challenger.sample();
        current = fold_even_odd(current, beta);

        if let Some(v) = &input[log_folded_height] {
            current.iter_mut().zip_eq(v).for_each(|(c, v)| *c += *v);
        }
    }

    // We should be left with `blowup` evaluations of a constant polynomial.
    assert_eq!(current.len(), config.blowup());
    let final_poly = current[0];
    for x in current {
        assert_eq!(x, final_poly);
    }

    CommitPhaseResult {
        commits,
        data,
        final_poly,
    }
}

struct CommitPhaseResult<FC: FriConfig> {
    commits: Vec<<FC::CommitPhaseMmcs as Mmcs<FC::Challenge>>::Commitment>,
    data: Vec<<FC::CommitPhaseMmcs as Mmcs<FC::Challenge>>::ProverData>,
    final_poly: FC::Challenge,
}
