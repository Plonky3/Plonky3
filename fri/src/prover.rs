use alloc::vec;
use alloc::vec::Vec;

use itertools::Itertools;
use p3_challenger::{CanObserve, CanSampleBits, FieldChallenger};
use p3_commit::{DirectMmcs, Mmcs};
use p3_field::{AbstractField, ExtensionField, Field};
use p3_matrix::{Matrix, MatrixRows};
use p3_util::log2_strict_usize;
use tracing::{info_span, instrument};

use crate::fold_even_odd::fold_even_odd;
use crate::{FriConfig, FriProof, QueryProof};

#[instrument(name = "FRI prover", skip_all)]
pub(crate) fn prove<FC: FriConfig>(
    config: &FC,
    input_mmcs: &[FC::InputMmcs],
    input_commits: &[&<FC::InputMmcs as Mmcs<FC::Domain>>::ProverData],
    challenger: &mut FC::Challenger,
) -> FriProof<FC> {
    let max_height = input_mmcs
        .iter()
        .zip(input_commits)
        .map(|(mmcs, commit)| mmcs.get_max_height(commit))
        .max()
        .unwrap_or_else(|| panic!("No matrices?"));
    let log_max_height = log2_strict_usize(max_height);

    let commit_phase_commits = commit_phase::<FC>(
        config,
        input_mmcs,
        input_commits,
        log_max_height,
        challenger,
    );
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
                    input_commits,
                    &commit_phase_commits,
                    index,
                )
            })
            .collect()
    });
    FriProof { query_proofs }
}

fn answer_query<FC: FriConfig>(
    config: &FC,
    input_mmcs: &[FC::InputMmcs],
    input_commits: &[&<FC::InputMmcs as Mmcs<FC::Domain>>::ProverData],
    commit_phase_commits: &[<FC::CommitPhaseMmcs as Mmcs<FC::Challenge>>::ProverData],
    index: usize,
) -> QueryProof<FC> {
    let input_openings = input_mmcs
        .iter()
        .zip(input_commits)
        .map(|(mmcs, commit)| mmcs.open_batch(index, commit))
        .collect();
    let commit_phase_openings = commit_phase_commits
        .iter()
        .enumerate()
        .map(|(i, commit)| config.commit_phase_mmcs().open_batch(index >> i, commit))
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
    input_commits: &[&<FC::InputMmcs as Mmcs<FC::Domain>>::ProverData],
    log_max_height: usize,
    challenger: &mut FC::Challenger,
) -> Vec<<FC::CommitPhaseMmcs as Mmcs<FC::Challenge>>::ProverData> {
    let max_height = 1 << log_max_height;
    let matrices_with_log_height = |log_height| {
        input_mmcs
            .iter()
            .zip(input_commits)
            .flat_map(|(mmcs, commit)| mmcs.get_matrices(commit))
            .filter(|mat| mat.height() == 1usize << log_height)
            .collect_vec()
    };

    let largest_matrices = matrices_with_log_height(log_max_height);
    let zero_vec = vec![FC::Challenge::ZERO; max_height];
    let alpha: FC::Challenge = challenger.sample_ext_element();
    let mut current = reduce_matrices(max_height, &zero_vec, &largest_matrices, alpha);

    // TODO: Can we avoid cloning?
    let (largest_commit, largest_prover_data) =
        config.commit_phase_mmcs().commit_vec(current.clone());
    challenger.observe(largest_commit);
    let mut commits = vec![largest_prover_data];

    for log_height in (1..log_max_height).rev() {
        let height = 1 << log_height;
        let beta: FC::Challenge = challenger.sample_ext_element();
        current = fold_even_odd(&current, beta);

        // TODO: Can we avoid cloning?
        let (commit, prover_data) = config.commit_phase_mmcs().commit_vec(current.clone());
        challenger.observe(commit);
        commits.push(prover_data);

        let matrices = matrices_with_log_height(log_height);
        current = reduce_matrices(height, &current, &matrices, alpha);
    }

    commits
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
    Mat: MatrixRows<F>,
{
    (0..height)
        .map(|r| {
            let mut reduced = init[r];
            for mat in matrices {
                for col in mat.row(r) {
                    reduced = reduced * alpha + col;
                }
            }
            reduced
        })
        .collect()
}
