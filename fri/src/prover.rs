use alloc::vec;
use alloc::vec::Vec;

use itertools::Itertools;
use p3_challenger::{CanObserve, CanSampleBits, FieldChallenger};
use p3_commit::{DirectMmcs, Mmcs};
use p3_field::{AbstractField, ExtensionField, Field};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::{Matrix, MatrixRows};
use p3_maybe_rayon::{MaybeIntoParIter, ParallelIterator};
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
    let matrices_with_log_height = |log_height| {
        input_mmcs
            .iter()
            .zip(input_data)
            .flat_map(|(mmcs, commit)| mmcs.get_matrices(commit))
            .filter(|mat| mat.height() == 1usize << log_height)
            .collect_vec()
    };

    let largest_matrices = matrices_with_log_height(log_max_height);
    let alpha: FC::Challenge = challenger.sample_ext_element();
    let mut alpha_reducer = MatrixReducer::new(alpha);
    let mut current = vec![FC::Challenge::zero(); max_height];
    alpha_reducer.reduce_matrices(&mut current, max_height, &largest_matrices);

    // let mut current = reduce_matrices(max_height, &zero_vec, &largest_matrices, alpha);

    let mut commits = vec![];
    let mut data = vec![];

    for log_folded_height in (config.log_blowup()..log_max_height).rev() {
        // TODO: Can we avoid cloning?
        let (commit, prover_data) = config
            .commit_phase_mmcs()
            // TODO: Need to interleave the other way, unless we change things so the input comes bit-reversed.
            .commit_matrix(RowMajorMatrix::new(current.clone(), 2));
        challenger.observe(commit.clone());
        commits.push(commit);
        data.push(prover_data);

        let folded_height = 1 << log_folded_height;
        let beta: FC::Challenge = challenger.sample_ext_element();
        current = fold_even_odd(&current, beta);

        let matrices = matrices_with_log_height(log_folded_height);
        if !matrices.is_empty() {
            alpha_reducer.reduce_matrices(&mut current, folded_height, &matrices);
            // current = reduce_matrices(folded_height, &current, &matrices, alpha);
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

#[instrument(
    name = "fold in matrices",
    level = "debug",
    skip(init, matrices, alpha)
)]
fn reduce_matrices<F, Challenge, Mat>(
    height: usize,
    init: &[Challenge],
    matrices: &[Mat],
    alpha: Challenge,
) -> Vec<Challenge>
where
    F: Field,
    Challenge: ExtensionField<F>,
    Mat: MatrixRows<F> + Sync,
{
    let rows = info_span!("compute quotient rows").in_scope(|| {
        (0..height)
            .map(|r| {
                matrices
                    .iter()
                    .map(move |m| m.row(r).into_iter().collect_vec())
                    .collect_vec()
            })
            .collect_vec()
    });
    info_span!("reduce").in_scope(|| {
        rows.into_iter()
            .enumerate()
            .map(|(r, mat_rows)| {
                let mut reduced = init[r];
                for row in mat_rows {
                    for col in row {
                        reduced *= alpha;
                        reduced += col;
                    }
                }
                reduced
            })
            .collect()
    })
    /*
        (0..height)
            .into_par_iter()
            .map(|r| {
            })
            .collect()
    */
}
