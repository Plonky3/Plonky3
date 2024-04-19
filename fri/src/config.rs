use alloc::vec::Vec;

use p3_field::Field;
use p3_matrix::Matrix;
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;

#[derive(Debug)]
pub struct FriConfig<M> {
    pub log_blowup: usize,
    pub num_queries: usize,
    pub proof_of_work_bits: usize,
    pub mmcs: M,
}

impl<M> FriConfig<M> {
    pub const fn blowup(&self) -> usize {
        1 << self.log_blowup
    }
}

pub trait FriFolder<F: Field>: Send + Sync {
    /// Fold a row, returning a single column.
    /// Right now the input row will always be 2 columns wide,
    /// but we may support higher folding arity in the future.
    fn fold_row(index: usize, log_height: usize, beta: F, evals: impl Iterator<Item = F>) -> F;
    fn fold_matrix<M: Matrix<F>>(beta: F, m: M) -> Vec<F> {
        let log_height = log2_strict_usize(m.height());
        m.par_rows()
            .enumerate()
            .map(|(i, r)| Self::fold_row(i, log_height, beta, r))
            .collect()
    }
}
