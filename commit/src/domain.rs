use alloc::vec::Vec;

use itertools::Itertools;
use p3_field::{
    batch_multiplicative_inverse, cyclic_subgroup_coset_known_order, ExtensionField, Field,
    TwoAdicField,
};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_util::{log2_ceil_usize, log2_strict_usize};

#[derive(Debug)]
pub struct LagrangeSelectors<T> {
    pub is_first_row: T,
    pub is_last_row: T,
    pub is_transition: T,
    pub inv_zeroifier: T,
}

pub trait PolynomialSpace: Copy {
    type Val: Field;

    fn size(&self) -> usize;

    fn first_point(&self) -> Self::Val;

    // This is only defined for cosets.
    fn next_point<Ext: ExtensionField<Self::Val>>(&self, x: Ext) -> Option<Ext>;

    // There are many choices for this, but we must pick a canonical one
    // for both prover/verifier determinism and LDE caching.
    fn create_disjoint_domain(&self, min_size: usize) -> Self;

    /// Split this domain into `num_chunks` even chunks.
    /// `num_chunks` is assumed to be a power of two.
    fn split_domains(&self, num_chunks: usize) -> Vec<Self>;
    // Split the evals into chunks of evals, corresponding to each domain
    // from `split_domains`.
    fn split_evals(
        &self,
        num_chunks: usize,
        evals: RowMajorMatrix<Self::Val>,
    ) -> Vec<RowMajorMatrix<Self::Val>>;

    fn zp_at_point<Ext: ExtensionField<Self::Val>>(&self, point: Ext) -> Ext;

    // Unnormalized
    fn selectors_at_point<Ext: ExtensionField<Self::Val>>(
        &self,
        point: Ext,
    ) -> LagrangeSelectors<Ext>;

    // Unnormalized
    fn selectors_on_coset(&self, coset: Self) -> LagrangeSelectors<Vec<Self::Val>>;
}
