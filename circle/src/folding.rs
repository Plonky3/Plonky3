use alloc::vec::Vec;
use core::fmt::Debug;
use core::marker::PhantomData;

use itertools::Itertools;
use p3_commit::Mmcs;
use p3_field::extension::ComplexExtendable;
use p3_field::{batch_multiplicative_inverse, ExtensionField};
use p3_fri::FriGenericConfig;
use p3_matrix::row_index_mapped::{RowIndexMap, RowIndexMappedView};
use p3_matrix::Matrix;
use p3_util::{log2_strict_usize, reverse_bits_len};

use crate::domain::CircleDomain;
use crate::{InputError, InputProof};

pub(crate) fn fold_bivariate<F: ComplexExtendable, EF: ExtensionField<F>>(
    beta: EF,
    evals: impl Matrix<EF>,
) -> Vec<EF> {
    let log_n = log2_strict_usize(evals.height() * evals.width());
    let domain = CircleDomain::standard(log_n);
    let mut twiddles = batch_multiplicative_inverse(
        &domain
            .points()
            .take(evals.height())
            .map(|p| p.imag())
            .collect_vec(),
    );
    twiddles = circle_bitrev_permute(&twiddles);
    fold(evals, beta, &twiddles)
}

pub(crate) fn fold_bivariate_row<F: ComplexExtendable, EF: ExtensionField<F>>(
    index: usize,
    log_folded_height: usize,
    beta: EF,
    evals: impl Iterator<Item = EF>,
) -> EF {
    let evals = evals.collect_vec();
    assert_eq!(evals.len(), 2);
    let log_arity = log2_strict_usize(evals.len());

    let orig_index = circle_bitrev_idx(index, log_folded_height);
    let t = CircleDomain::<F>::standard(log_folded_height + log_arity)
        .nth_point(orig_index)
        .imag()
        .inverse();

    let sum = evals[0] + evals[1];
    let diff = (evals[0] - evals[1]) * t;
    (sum + beta * diff).halve()
}

pub(crate) struct CircleFriGenericConfig<F, InputProof, InputError>(
    pub(crate) PhantomData<(F, InputProof, InputError)>,
);

pub(crate) type CircleFriConfig<Val, Challenge, InputMmcs, FriMmcs> = CircleFriGenericConfig<
    Val,
    InputProof<Val, Challenge, InputMmcs, FriMmcs>,
    InputError<<InputMmcs as Mmcs<Val>>::Error, <FriMmcs as Mmcs<Challenge>>::Error>,
>;

impl<F: ComplexExtendable, EF: ExtensionField<F>, InputProof, InputError: Debug>
    FriGenericConfig<EF> for CircleFriGenericConfig<F, InputProof, InputError>
{
    type InputProof = InputProof;
    type InputError = InputError;

    fn extra_query_index_bits(&self) -> usize {
        1
    }

    fn fold_row(
        &self,
        index: usize,
        log_folded_height: usize,
        beta: EF,
        evals: impl Iterator<Item = EF>,
    ) -> EF {
        let evals = evals.collect_vec();
        assert_eq!(evals.len(), 2);
        let log_arity = log2_strict_usize(evals.len());

        let orig_index = circle_bitrev_idx(index, log_folded_height);
        // +1 because twiddles after the first layer come from the x coordinates of the larger domain.
        let t = CircleDomain::<F>::standard(log_folded_height + log_arity + 1)
            .nth_point(orig_index)
            .real()
            .inverse();

        let sum = evals[0] + evals[1];
        let diff = (evals[0] - evals[1]) * t;
        (sum + beta * diff).halve()
    }

    fn fold_matrix<M: Matrix<EF>>(&self, beta: EF, m: M) -> Vec<EF> {
        let log_n = log2_strict_usize(m.width() * m.height());
        // +1 because twiddles after the first layer come from the x coordinates of the larger domain.
        let domain = CircleDomain::standard(log_n + 1);
        let mut twiddles = batch_multiplicative_inverse(
            &domain
                .points()
                .take(m.height())
                .map(|p| p.real())
                .collect_vec(),
        );
        twiddles = circle_bitrev_permute(&twiddles);
        fold(m, beta, &twiddles)
    }
}

fn fold<F: ComplexExtendable, EF: ExtensionField<F>>(
    evals: impl Matrix<EF>,
    beta: EF,
    twiddles: &[F],
) -> Vec<EF> {
    evals
        .rows()
        .zip(twiddles)
        .map(|(mut row, &t)| {
            let (lo, hi) = row.next_tuple().unwrap();
            let sum = lo + hi;
            let diff = (lo - hi) * t;
            (sum + beta * diff).halve()
        })
        .collect_vec()
}

// circlebitrev -> natural
// can make faster with:
// https://lemire.me/blog/2018/02/21/iterating-over-set-bits-quickly/
pub fn circle_bitrev_idx(mut idx: usize, bits: usize) -> usize {
    idx = reverse_bits_len(idx, bits);
    for i in 0..bits {
        if idx & (1 << i) != 0 {
            idx ^= (1 << i) - 1;
        }
    }
    idx
}

// can do in place if use cycles? bitrev makes it harder
pub(crate) fn circle_bitrev_permute<T: Clone>(xs: &[T]) -> Vec<T> {
    let bits = log2_strict_usize(xs.len());
    (0..xs.len())
        .map(|i| xs[circle_bitrev_idx(i, bits)].clone())
        .collect()
}

pub struct CircleBitrevPerm {
    log_height: usize,
}

pub type CircleBitrevView<M> = RowIndexMappedView<CircleBitrevPerm, M>;

impl CircleBitrevPerm {
    pub fn new<T: Send + Sync, M: Matrix<T>>(inner: M) -> RowIndexMappedView<CircleBitrevPerm, M> {
        RowIndexMappedView {
            index_map: CircleBitrevPerm {
                log_height: log2_strict_usize(inner.height()),
            },
            inner,
        }
    }
}

impl RowIndexMap for CircleBitrevPerm {
    fn height(&self) -> usize {
        1 << self.log_height
    }
    fn map_row_index(&self, r: usize) -> usize {
        circle_bitrev_idx(r, self.log_height)
    }
}

#[cfg(test)]
mod tests {
    use hashbrown::HashSet;
    use p3_field::extension::BinomialExtensionField;
    use p3_field::AbstractExtensionField;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_matrix::Matrix;
    use p3_mersenne_31::Mersenne31;
    use rand::{thread_rng, Rng};

    use super::*;
    use crate::Cfft;

    #[test]
    fn test_circle_bitrev() {
        assert_eq!(circle_bitrev_permute(&[0]), &[0]);
        assert_eq!(circle_bitrev_permute(&[0, 1]), &[0, 1]);
        assert_eq!(circle_bitrev_permute(&[0, 1, 2, 3]), &[0, 3, 1, 2]);
        assert_eq!(
            circle_bitrev_permute(&[0, 1, 2, 3, 4, 5, 6, 7]),
            &[0, 7, 3, 4, 1, 6, 2, 5]
        );
    }

    type F = Mersenne31;
    type EF = BinomialExtensionField<F, 3>;

    fn do_test_folding(log_n: usize, log_blowup: usize) {
        let mut rng = thread_rng();

        let mut evals: Vec<EF> = {
            let evals = RowMajorMatrix::<F>::rand(
                &mut rng,
                1 << log_n,
                <EF as AbstractExtensionField<F>>::D,
            );
            let lde = Cfft::default().lde(
                evals,
                CircleDomain::standard(log_n),
                CircleDomain::standard(log_n + log_blowup),
            );
            lde.rows()
                .map(|r| EF::from_base_slice(&r.collect_vec()))
                .collect()
        };

        evals = circle_bitrev_permute(&evals);

        let g: CircleFriGenericConfig<F, (), ()> = CircleFriGenericConfig(PhantomData);

        evals = fold_bivariate::<F, _>(rng.gen(), RowMajorMatrix::new(evals, 2));
        for i in log_blowup..(log_n + log_blowup - 1) {
            evals = g.fold_matrix(rng.gen(), RowMajorMatrix::new(evals, 2));
            if i != log_n + log_blowup - 2 {
                // check that we aren't degenerate, we (probabilistically) should be unique before the final layer
                assert_eq!(
                    evals.iter().copied().collect::<HashSet<_>>().len(),
                    evals.len()
                );
            }
        }
        assert_eq!(evals.len(), 1 << log_blowup);
        assert!(evals.iter().all(|&x| x == evals[0]));
    }

    #[test]
    fn test_folding() {
        do_test_folding(4, 1);
        do_test_folding(5, 2);
    }

    #[test]
    fn test_fold_row_matrix_same() {
        let mut rng = thread_rng();
        let evals = RowMajorMatrix::<EF>::rand(&mut rng, 1 << 5, 2);
        let beta: EF = rng.gen();

        let mat_folded = fold_bivariate::<F, EF>(beta, evals.clone());
        let row_folded = evals
            .rows()
            .enumerate()
            .map(|(index, evals)| fold_bivariate_row::<F, EF>(index, 5, beta, evals))
            .collect_vec();
        assert_eq!(
            mat_folded, row_folded,
            "bivariate fold_matrix and fold_row do not match"
        );

        let g: CircleFriGenericConfig<F, (), ()> = CircleFriGenericConfig(PhantomData);
        let mat_folded = g.fold_matrix(beta, evals.clone());
        let row_folded = evals
            .rows()
            .enumerate()
            .map(|(index, evals)| g.fold_row(index, 5, beta, evals))
            .collect_vec();
        assert_eq!(
            mat_folded, row_folded,
            "univariate fold_matrix and fold_row do not match"
        );
    }
}
