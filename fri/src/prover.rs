use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Reverse;

use itertools::Itertools;
use p3_challenger::Challenger;
use p3_commit::{DirectMMCS, MMCS};
use p3_field::{AbstractField, ExtensionField, Field, PrimeField64};
use p3_matrix::{Matrix, MatrixRows};

use crate::{FriConfig, FriProof, QueryProof};

pub(crate) fn prove<F, Challenge, M, MC, Chal>(
    input_commits: &[M::ProverData],
    config: &FriConfig,
    challenger: &mut Chal,
) -> FriProof<F, Challenge, M, MC>
where
    F: PrimeField64,
    Challenge: ExtensionField<F>,
    M: MMCS<F>,
    MC: DirectMMCS<Challenge>,
    Chal: Challenger<F>,
{
    let n = input_commits
        .iter()
        .map(|cw| M::get_max_height(cw))
        .max()
        .unwrap_or_else(|| panic!("No matrices?"));

    let commit_phase_commits = commit_phase::<F, Challenge, M, MC, Chal>(input_commits, challenger);
    let query_indices: Vec<usize> = (0..config.num_queries)
        .map(|_| challenger.random_usize(n))
        .collect();
    // TODO: into_par_iter?
    let query_proofs = query_indices
        .into_iter()
        .map(|index| answer_query(input_commits, &commit_phase_commits, index))
        .collect();
    FriProof { query_proofs }
}

fn answer_query<F, Challenge, M, MC>(
    input_commits: &[M::ProverData],
    commit_phase_commits: &[MC::ProverData],
    index: usize,
) -> QueryProof<F, Challenge, M, MC>
where
    F: Field,
    Challenge: ExtensionField<F>,
    M: MMCS<F>,
    MC: DirectMMCS<Challenge>,
{
    let input_openings = input_commits
        .iter()
        .map(|commit| M::open_batch(index, commit))
        .collect();
    let commit_phase_openings = commit_phase_commits
        .iter()
        .map(|commit| MC::open_batch(index, commit))
        .collect();
    QueryProof {
        input_openings,
        commit_phase_openings,
    }
}

fn commit_phase<F, Challenge, M, MC, Chal>(
    input_commits: &[M::ProverData],
    challenger: &mut Chal,
) -> Vec<MC::ProverData>
where
    F: Field,
    Challenge: ExtensionField<F>,
    M: MMCS<F>,
    MC: DirectMMCS<Challenge>,
    Chal: Challenger<F>,
{
    let alpha: Challenge = challenger.random_ext_element();
    let inputs_by_desc_height = input_commits
        .iter()
        .flat_map(|data| M::get_matrices(data))
        .sorted_by_key(|mat| Reverse(mat.height()))
        .group_by(|mat| mat.height());
    let mut inputs_by_desc_height = inputs_by_desc_height.into_iter();

    let (max_height, largest_matrices_iter) = inputs_by_desc_height.next().expect("No matrices?");
    let largest_matrices = largest_matrices_iter.collect_vec();
    let zero_vec = vec![Challenge::ZERO; max_height];
    let mut current = reduce_matrices(max_height, zero_vec, largest_matrices, alpha);
    let mut committed = vec![current.clone()];

    for (height, matrices) in inputs_by_desc_height {
        while current.len() < height {
            let beta = <Challenge as AbstractField>::ZERO; // TODO
            current = fold_even_odd(&current, beta);
        }
        committed.push(current.clone());
        current = reduce_matrices::<F, Challenge, M::Mat>(
            height,
            current.clone(),
            matrices.collect(),
            alpha,
        );
    }
    todo!()
}

/// Fold a polynomial `p(x) = p_even(x^2) + x p_odd(x^2)` into `p_even(x) + beta * p_odd(x)`.
fn fold_even_odd<F: Field>(_poly: &[F], _beta: F) -> Vec<F> {
    todo!()
}

#[allow(clippy::needless_pass_by_value)]
fn reduce_matrices<F, Challenge, Mat>(
    height: usize,
    init: Vec<Challenge>,
    matrices: Vec<&Mat>,
    alpha: Challenge,
) -> Vec<Challenge>
where
    F: Field,
    Challenge: ExtensionField<F>,
    Mat: for<'a> MatrixRows<'a, F>,
{
    (0..height)
        .map(|r| {
            let mut reduced = init[r];
            for mat in &matrices {
                for col in mat.row(r) {
                    reduced = reduced * alpha + *col;
                }
            }
            reduced
        })
        .collect()
}
