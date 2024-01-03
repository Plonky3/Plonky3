use alloc::vec;
use alloc::vec::Vec;

use p3_challenger::{CanObserve, CanSampleBits, FieldChallenger, GrindingChallenger};
use p3_commit::{DirectMmcs, Mmcs};
use p3_field::AbstractField;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_util::log2_strict_usize;
use tracing::{info_span, instrument};

use crate::fold_even_odd::fold_even_odd;
use crate::matrix_reducer::MatrixReducer;
use crate::{CommitPhaseProofStep, FriConfig, FriProof, InputOpening, QueryProof};

#[instrument(name = "FRI prover", skip_all)]
pub(crate) fn prove<FC: FriConfig>(
    config: &FC,
    input_mmcs: &[FC::InputMmcs],
    input_data: &[&<FC::InputMmcs as Mmcs<FC::Val>>::ProverData],
    challenger: &mut FC::Challenger,
) -> FriProof<FC> {
    let max_height = input_mmcs
        .iter()
        .zip(input_data)
        .map(|(mmcs, commit)| mmcs.get_max_height(commit))
        .max()
        .unwrap_or_else(|| panic!("No matrices?"));
    let log_max_height = log2_strict_usize(max_height);

    let commit_phase_result =
        commit_phase::<FC>(config, input_mmcs, input_data, log_max_height, challenger);

    let query_indices: Vec<usize> = (0..config.num_queries())
        .map(|_| challenger.sample_bits(log_max_height))
        .collect();

    let pow_witness = generate_pow_witness(config, challenger);

    let query_proofs = info_span!("query phase").in_scope(|| {
        query_indices
            .into_iter() // TODO: into_par_iter?
            .map(|index| {
                answer_query(
                    config,
                    input_mmcs,
                    input_data,
                    &commit_phase_result.data,
                    index,
                )
            })
            .collect()
    });

    FriProof {
        commit_phase_commits: commit_phase_result.commits,
        query_proofs,
        final_poly: commit_phase_result.final_poly,
        pow_witness,
    }
}

fn answer_query<FC: FriConfig>(
    config: &FC,
    input_mmcs: &[FC::InputMmcs],
    input_data: &[&<FC::InputMmcs as Mmcs<FC::Val>>::ProverData],
    commit_phase_commits: &[<FC::CommitPhaseMmcs as Mmcs<FC::Challenge>>::ProverData],
    index: usize,
) -> QueryProof<FC> {
    let input_openings = input_mmcs
        .iter()
        .zip(input_data)
        .map(|(mmcs, commit)| {
            let (opened_values, opening_proof) = mmcs.open_batch(index, commit);
            InputOpening {
                opened_values,
                opening_proof,
            }
        })
        .collect();

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
        input_openings,
        commit_phase_openings,
    }
}

#[instrument(name = "commit phase", skip_all)]
fn commit_phase<FC: FriConfig>(
    config: &FC,
    input_mmcs: &[FC::InputMmcs],
    input_data: &[&<FC::InputMmcs as Mmcs<FC::Val>>::ProverData],
    log_max_height: usize,
    challenger: &mut FC::Challenger,
) -> CommitPhaseResult<FC> {
    let max_height = 1 << log_max_height;

    let mut matrices_by_log_height: Vec<Vec<_>> = vec![];
    matrices_by_log_height.resize_with(log_max_height + 1, Default::default);
    for (mmcs, commit) in input_mmcs.iter().zip(input_data) {
        for mat in mmcs.get_matrices(commit) {
            matrices_by_log_height[log2_strict_usize(mat.height())].push(mat);
        }
    }

    let largest_matrices = &matrices_by_log_height[log_max_height];
    let alpha: FC::Challenge = challenger.sample_ext_element();
    let alpha_reducer = MatrixReducer::new(alpha);
    let mut current = vec![FC::Challenge::zero(); max_height];
    alpha_reducer.reduce_matrices(&mut current, max_height, largest_matrices);

    let mut commits = vec![];
    let mut data = vec![];

    for log_folded_height in (config.log_blowup()..log_max_height).rev() {
        let folded_height = 1 << log_folded_height;
        // TODO: avoid cloning
        let leaves = RowMajorMatrix::new(current.clone(), 2);
        let (commit, prover_data) = config.commit_phase_mmcs().commit_matrix(leaves);
        challenger.observe(commit.clone());
        commits.push(commit);
        data.push(prover_data);

        let beta: FC::Challenge = challenger.sample_ext_element();
        current = fold_even_odd(current, beta);

        let matrices = &matrices_by_log_height[log_folded_height];
        if !matrices.is_empty() {
            alpha_reducer.reduce_matrices(&mut current, folded_height, matrices);
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

pub fn generate_pow_witness<FC: FriConfig>(config: &FC, challenger: &FC::Challenger) -> FC::Val {
    challenger.grind(config.proof_of_work_bits() as usize)
}
