use crate::FriProof;
use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Reverse;
use itertools::Itertools;
use p3_challenger::Challenger;
use p3_commit::{DirectMMCS, MMCS};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

pub(crate) fn prove<F, M, MC, Chal>(
    codewords: &[M::ProverData],
    challenger: &mut Chal,
) -> FriProof<F, M, MC>
where
    F: Field,
    M: MMCS<F::Base>,
    MC: DirectMMCS<F::Base>,
    Chal: Challenger<F::Base>,
{
    let commit_phase_commits = commit_phase::<F, M, MC, Chal>(codewords, challenger);
    let queries = todo!();
    todo!()
}

pub(crate) fn commit_phase<F, M, MC, Chal>(
    codewords: &[M::ProverData],
    challenger: &mut Chal,
) -> Vec<MC::ProverData>
where
    F: Field,
    M: MMCS<F::Base>,
    MC: DirectMMCS<F::Base>,
    Chal: Challenger<F::Base>,
{
    let alpha: F = <F as AbstractField>::ZERO; // TODO challenger.random_ext_element();
    let matrices_by_desc_height = codewords
        .iter()
        .flat_map(|data| M::get_matrices(data))
        .sorted_by_key(|mat| Reverse(mat.height()))
        .group_by(|mat| mat.height());
    let mut matrices_by_desc_height = matrices_by_desc_height.into_iter();

    let (max_height, largest_matrices_iter) = matrices_by_desc_height.next().expect("No matrices?");
    let largest_matrices = largest_matrices_iter.collect_vec();
    let zero_vec = vec![<F as AbstractField>::ZERO; max_height];
    let mut current = reduce_matrices(max_height, zero_vec, largest_matrices, alpha);
    let mut committed = vec![current.clone()];

    for (height, matrices) in matrices_by_desc_height {
        while current.len() < height {
            let beta = <F as AbstractField>::ZERO; // TODO
            current = fold_even_odd(&current, beta);
        }
        committed.push(current.clone());
        current = reduce_matrices::<F, M::Mat>(height, current.clone(), matrices.collect(), alpha);
    }
    todo!()
}

/// Fold a polynomial `p(x) = p_even(x^2) + x p_odd(x^2)` into `p_even(x) + beta * p_odd(x)`.
fn fold_even_odd<F: Field>(poly: &[F], beta: F) -> Vec<F> {
    todo!()
}

fn reduce_matrices<F, Mat>(height: usize, init: Vec<F>, matrices: Vec<&Mat>, alpha: F) -> Vec<F>
where
    F: Field,
    Mat: Matrix<F::Base>,
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
