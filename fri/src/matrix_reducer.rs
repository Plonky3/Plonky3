use core::marker::PhantomData;

use alloc::vec;
use alloc::vec::Vec;
use itertools::Itertools;
use p3_field::{AbstractExtensionField, ExtensionField, Field, PackedField};
use p3_matrix::{MatrixRowSlices, MatrixRows};
use p3_maybe_rayon::{IndexedParallelIterator, MaybeIntoParIter, ParallelIterator};
use tracing::{info_span, instrument};

pub(crate) struct MatrixReducer<F: Field, EF> {
    alpha: EF,
    transposed_alphas: Vec<F::Packing>,
    current_alpha_pow: EF,
}

impl<F: Field, EF: ExtensionField<F>> MatrixReducer<F, EF> {
    pub(crate) fn new(alpha: EF) -> Self {
        let alpha_pows = alpha.powers().take(F::Packing::WIDTH).collect_vec();
        let transposed_alphas = (0..EF::D)
            .map(|i| F::Packing::from_fn(|j| alpha_pows[j].as_base_slice()[i]))
            .collect_vec();
        Self {
            alpha,
            transposed_alphas,
            current_alpha_pow: EF::one(),
        }
    }

    #[instrument(name = "fold in matrices", level = "debug", skip(self, reduced, mats))]
    pub(crate) fn reduce_matrices<M: MatrixRows<F> + Sync>(
        &mut self,
        reduced: &mut [EF],
        height: usize,
        mats: &[M],
    ) {
        let alpha_pow_width = self.alpha.exp_u64(F::Packing::WIDTH as u64);
        reduced
            .into_par_iter()
            .enumerate()
            .for_each(|(r, reduced_row)| {
                let mut current = self.current_alpha_pow;
                for mat in mats {
                    let row_slice = mat.row(r).into_iter().collect_vec();
                    let num_leftover = row_slice.len() % F::Packing::WIDTH;
                    let leftover_start = row_slice.len() - num_leftover;
                    let packed_row = F::Packing::pack_slice(&row_slice[..leftover_start]);
                    let mut horiz_sum = vec![F::zero(); EF::D];
                    for &packed_col in packed_row {
                        for i in 0..EF::D {
                            let x: F::Packing = packed_col * self.transposed_alphas[i];
                            horiz_sum[i] = x.as_slice().iter().copied().sum::<F>();
                        }
                        *reduced_row += current * EF::from_base_slice(&horiz_sum);
                        current *= alpha_pow_width;
                    }
                    for &col in &row_slice[leftover_start..] {
                        *reduced_row += current * col;
                        current *= self.alpha;
                    }
                }
            });
        self.current_alpha_pow *= self
            .alpha
            .exp_u64(mats.iter().map(|m| m.width()).sum::<usize>() as u64);
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use p3_baby_bear::BabyBear;
    use p3_field::{extension::BinomialExtensionField, AbstractField};
    use p3_matrix::{dense::RowMajorMatrix, MatrixRows};
    use rand::Rng;

    use super::MatrixReducer;

    type F = BabyBear;
    // 5 instead of 4 to make sure it works when EF::D != Packing::WIDTH
    type EF = BinomialExtensionField<F, 5>;

    #[test]
    fn test_matrix_reducer() {
        let mut rng = rand::thread_rng();
        let alpha: EF = rng.gen();
        let height = 32;
        let mats0: &[RowMajorMatrix<F>] = &[
            RowMajorMatrix::rand(&mut rng, height, 37),
            RowMajorMatrix::rand(&mut rng, height, 13),
        ];
        let mats1: &[RowMajorMatrix<F>] = &[
            RowMajorMatrix::rand(&mut rng, height, 41),
            RowMajorMatrix::rand(&mut rng, height, 10),
        ];
        let mut reducer = MatrixReducer::new(alpha);
        let mut reduced = vec![EF::zero(); height];
        reducer.reduce_matrices(&mut reduced, height, mats0);
        reducer.reduce_matrices(&mut reduced, height, mats1);

        let mut correct = vec![EF::zero(); height];
        for r in 0..height {
            let mut current = EF::one();
            for mat in mats0.iter().chain(mats1) {
                for col in mat.row(r) {
                    correct[r] += current * col;
                    current *= alpha;
                }
            }
        }

        assert_eq!(reduced, correct);
    }
}
