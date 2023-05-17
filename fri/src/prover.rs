use crate::FriProof;
use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Reverse;
use itertools::Itertools;
use p3_challenger::Challenger;
use p3_commit::{DirectMMCS, MMCS};
use p3_field::{AbstractField, Field, FieldExtension};
use p3_matrix::Matrix;

pub(crate) fn prove<FE, M, MC, Chal>(
    codewords: &[M::ProverData],
    challenger: &mut Chal,
) -> FriProof<FE, M, MC>
where
    FE: FieldExtension,
    M: MMCS<FE::Base>,
    MC: DirectMMCS<FE::Base>,
    Chal: Challenger<FE::Base>,
{
    let _commit_phase_commits = commit_phase::<FE, M, MC, Chal>(codewords, challenger);
    //let queries = todo!();
    todo!()
}

pub(crate) fn commit_phase<FE, M, MC, Chal>(
    codewords: &[M::ProverData],
    _challenger: &mut Chal,
) -> Vec<MC::ProverData>
where
    FE: FieldExtension,
    M: MMCS<FE::Base>,
    MC: DirectMMCS<FE::Base>,
    Chal: Challenger<FE::Base>,
{
    let alpha = <FE::Extension as AbstractField>::ZERO; // TODO challenger.random_ext_element();
    let matrices_by_desc_height = codewords
        .iter()
        .flat_map(|data| M::get_matrices(data))
        .sorted_by_key(|mat| Reverse(mat.height()))
        .group_by(|mat| mat.height());
    let mut matrices_by_desc_height = matrices_by_desc_height.into_iter();

    let (max_height, largest_matrices_iter) = matrices_by_desc_height.next().expect("No matrices?");
    let largest_matrices = largest_matrices_iter.collect_vec();
    let zero_vec = vec![<FE::Extension as AbstractField>::ZERO; max_height];
    let mut current = reduce_matrices::<FE, _>(max_height, zero_vec, largest_matrices, alpha);
    let mut committed = vec![current.clone()];

    for (height, matrices) in matrices_by_desc_height {
        while current.len() < height {
            let beta = <FE::Extension as AbstractField>::ZERO; // TODO
            current = fold_even_odd(&current, beta);
        }
        committed.push(current.clone());
        current = reduce_matrices::<FE, M::Mat>(height, current.clone(), matrices.collect(), alpha);
    }
    todo!()
}

/// Fold a polynomial `p(x) = p_even(x^2) + x p_odd(x^2)` into `p_even(x) + beta * p_odd(x)`.
fn fold_even_odd<F: Field>(_poly: &[F], _beta: F) -> Vec<F> {
    todo!()
}

fn reduce_matrices<FE, Mat>(
    height: usize,
    init: Vec<FE::Extension>,
    matrices: Vec<&Mat>,
    alpha: FE::Extension,
) -> Vec<FE::Extension>
where
    FE: FieldExtension,
    Mat: Matrix<FE::Base>,
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
