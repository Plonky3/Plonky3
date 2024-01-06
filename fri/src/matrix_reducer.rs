use alloc::vec;
use alloc::vec::Vec;

use itertools::Itertools;
use p3_field::{ExtensionField, Field, PackedField};
use p3_matrix::MatrixRows;
use p3_maybe_rayon::MaybeIntoParIter;
#[cfg(feature = "parallel")]
use p3_maybe_rayon::{IndexedParallelIterator, ParallelIterator};
use p3_util::indices_arr;
use tracing::instrument;

// seems to be the sweet spot, could be tweaked based on benches
const BATCH_SIZE: usize = 8;

/// Optimized matrix reducer. Only works for binomial extension fields, or extension fields
/// in which multiplication and addition with a base field element are simple elementwise operations.
pub(crate) struct MatrixReducer<F: Field, EF> {
    alpha: EF,
    alpha_pow_width: EF,
    transposed_alphas: Vec<[F::Packing; BATCH_SIZE]>,
}

impl<F: Field, EF: ExtensionField<F>> MatrixReducer<F, EF> {
    pub(crate) fn new(alpha: EF) -> Self {
        let alpha_pows = alpha
            .powers()
            .take(F::Packing::WIDTH * BATCH_SIZE)
            .collect_vec();
        let transposed_alphas = (0..EF::D)
            .map(|i| {
                indices_arr::<BATCH_SIZE>().map(|j| {
                    F::Packing::from_fn(|k| {
                        alpha_pows[j * F::Packing::WIDTH + k].as_base_slice()[i]
                    })
                })
            })
            .collect_vec();
        Self {
            alpha,
            alpha_pow_width: alpha.exp_u64(F::Packing::WIDTH as u64),
            transposed_alphas,
        }
    }

    #[instrument(name = "fold in matrices", level = "debug", skip(self, reduced, mats))]
    pub(crate) fn reduce_matrices<M: MatrixRows<F> + Sync>(
        &self,
        reduced: &mut [EF],
        height: usize,
        mats: &[M],
    ) {
        // precompute alpha_pows, since we are not doing horner
        let mut current_alpha_pow = EF::one();
        let mut alpha_pows = vec![];
        for mat in mats {
            let num_packed = mat.width() / F::Packing::WIDTH;
            let num_leftover = mat.width() % F::Packing::WIDTH;
            for chunk in &(0..num_packed).chunks(BATCH_SIZE) {
                alpha_pows.push(current_alpha_pow);
                current_alpha_pow *= self.alpha_pow_width.exp_u64(chunk.count() as u64);
            }
            for _ in 0..num_leftover {
                alpha_pows.push(current_alpha_pow);
                current_alpha_pow *= self.alpha;
            }
        }

        /*
        We have packed base elements ys and extension field α^n's
        We want α^0 * ys[0] + α^1 * ys[1] + ...

        Level 0:
            alpha.powers().zip(cols).map(|alpha_pow, col| alpha_pow * col).sum()

        Level 1: (assume Packing::WIDTH=4, D=2, for clarity, although D will usually be higher)

            transposed_alphas
            α^0 α^1 α^2 α^3
            [1] [.] [.] [.]
            [1] [.] [.] [.]
         ys:[a   b   c   d] [e f g h] ...

        We multiply ys vertically, then sum horizontally.
        Aka, if β is α^0*a + α^1*b + α^2*c + α^3*d, then each limb of β is
        determined by multiplying the packed ys by the appropriate row of
        transposed_alphas, then summing horizontally.

        This assumes we are in an extension field where multiplication and addition
        by a base element is simply elementwise.

        Then, to fold it into our running reduction, we perform one extension multiplication.
        In this scheme, the extension mul takes about 40% of the time in this function.

        Level 2: Batching. To delay the horizontal sum and extension mul as much as possible,
        we precompute even more columns of transposed_alphas, and sum the packed results per
        batch, and only do one horizontal sum and extension mul per batch.

        */

        reduced
            .into_par_iter()
            .enumerate()
            .for_each(|(r, reduced_row)| {
                let mut alpha_pow_iter = alpha_pows.iter();
                for mat in mats {
                    let row_vec = mat.row_vec(r);
                    let (packed_row, sfx) = F::Packing::pack_slice_with_suffix(&row_vec);
                    for packed_col_chunk in packed_row.chunks(BATCH_SIZE) {
                        let chunk_sum = EF::from_base_fn(|i| {
                            let chunk_limb_sum = packed_col_chunk
                                .iter()
                                .zip(self.transposed_alphas[i])
                                .map(|(&packed_col, packed_alpha)| packed_col * packed_alpha)
                                .sum::<F::Packing>();
                            chunk_limb_sum.as_slice().iter().copied().sum::<F>()
                        });
                        *reduced_row += *alpha_pow_iter.next().unwrap() * chunk_sum;
                    }
                    for &col in sfx {
                        *reduced_row += *alpha_pow_iter.next().unwrap() * col;
                    }
                }
            })
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_baby_bear::BabyBear;
    use p3_field::extension::BinomialExtensionField;
    use p3_field::AbstractField;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_matrix::MatrixRows;
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
        let reducer = MatrixReducer::new(alpha);
        let mut reduced = vec![EF::zero(); height];
        reducer.reduce_matrices(&mut reduced, height, mats0);
        reducer.reduce_matrices(&mut reduced, height, mats1);

        let mut correct = vec![EF::zero(); height];
        for (r, correct_reduced) in correct.iter_mut().enumerate() {
            for batch in [mats0, mats1] {
                let mut current = EF::one();
                for mat in batch {
                    for col in mat.row(r) {
                        *correct_reduced += current * col;
                        current *= alpha;
                    }
                }
            }
        }

        assert_eq!(reduced, correct);
    }
}
