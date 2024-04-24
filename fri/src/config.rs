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

/// Whereas `FriConfig` encompasses parameters the end user can set, `FriGenericConfig` is
/// set by the PCS calling fri, and abstracts over implementation details of the PCS.
pub trait FriGenericConfig<F: Field> {
    type InputProof;

    // We can ask FRI to sample extra query bits (LSB) for our own purposes.
    fn extra_query_index_bits(&self) -> usize;

    /// Fold a row, returning a single column.
    /// Right now the input row will always be 2 columns wide,
    /// but we may support higher folding arity in the future.
    fn fold_row(
        &self,
        index: usize,
        log_height: usize,
        beta: F,
        evals: impl Iterator<Item = F>,
    ) -> F;

    fn fold_matrix<M: Matrix<F>>(&self, beta: F, m: M) -> Vec<F> {
        let log_height = log2_strict_usize(m.height());
        m.par_rows()
            .enumerate()
            .map(|(i, r)| self.fold_row(i, log_height, beta, r))
            .collect()
    }
}
