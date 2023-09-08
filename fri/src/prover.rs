use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Reverse;

use itertools::Itertools;
use p3_challenger::{CanObserve, CanSampleBits, FieldChallenger};
use p3_commit::{DirectMmcs, Mmcs};
use p3_field::{AbstractField, ExtensionField, Field};
use p3_matrix::{Matrix, MatrixRows};
use p3_util::log2_strict_usize;
use tracing::{info_span, instrument};

use crate::fold_even_odd::fold_even_odd;
use crate::{FriConfig, FriProof, QueryProof};

#[instrument(name="FRI prover", skip_all)]
pub(crate) fn prove<FC: FriConfig>(
    config: &FC,
    input_mmcs: &[FC::InputMmcs],
    input_commits: &[&<FC::InputMmcs as Mmcs<FC::Domain>>::ProverData],
    challenger: &mut FC::Challenger,
) -> FriProof<FC> {
    let n = input_mmcs
        .iter()
        .zip(input_commits)
        .map(|(mmcs, commit)| mmcs.get_max_height(commit))
        .max()
        .unwrap_or_else(|| panic!("No matrices?"));
    let log_n = log2_strict_usize(n);

    let commit_phase_commits = commit_phase::<FC>(config, input_mmcs, input_commits, challenger);
    let query_indices: Vec<usize> = (0..config.num_queries())
        .map(|_| challenger.sample_bits(log_n))
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

#[instrument(name="commit phase", skip_all)]
fn commit_phase<FC: FriConfig>(
    config: &FC,
    input_mmcs: &[FC::InputMmcs],
    input_commits: &[&<FC::InputMmcs as Mmcs<FC::Domain>>::ProverData],
    challenger: &mut FC::Challenger,
) -> Vec<<FC::CommitPhaseMmcs as Mmcs<FC::Challenge>>::ProverData> {
    let inputs_by_desc_height = input_mmcs
        .iter()
        .zip(input_commits)
        .flat_map(|(mmcs, commit)| mmcs.get_matrices(commit))
        .sorted_by_key(|mat| Reverse(mat.height()))
        .group_by(|mat| mat.height());
    let mut inputs_by_desc_height = inputs_by_desc_height.into_iter();

    let (max_height, largest_matrices_iter) = inputs_by_desc_height.next().expect("No matrices?");
    let largest_matrices = largest_matrices_iter.collect_vec();
    let zero_vec = vec![FC::Challenge::ZERO; max_height];
    let alpha: FC::Challenge = challenger.sample_ext_element();
    let mut current = reduce_matrices(max_height, &zero_vec, &largest_matrices, alpha);

    // TODO: Can we avoid cloning?
    let (largest_commit, largest_prover_data) =
        config.commit_phase_mmcs().commit_vec(current.clone());
    challenger.observe(largest_commit);
    let mut commits = vec![largest_prover_data];

    for (height, matrices) in inputs_by_desc_height {
        while current.len() < height {
            let beta: FC::Challenge = challenger.sample_ext_element();
            current = fold_even_odd(&current, beta);
        }

        // TODO: Can we avoid cloning?
        let (commit, prover_data) = config.commit_phase_mmcs().commit_vec(current.clone());
        challenger.observe(commit);
        commits.push(prover_data);

        current = reduce_matrices::<FC::Domain, FC::Challenge, <FC::InputMmcs as Mmcs<_>>::Mat<'_>>(
            height,
            &current,
            &matrices.collect_vec(),
            alpha,
        );
    }

    commits
}

#[instrument(name="fold in matrices", skip_all)]
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
