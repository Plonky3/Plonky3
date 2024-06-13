use itertools::{izip, Itertools};
use p3_dft::divide_by_height;
use p3_field::{
    batch_multiplicative_inverse,
    extension::{Complex, ComplexExtendable},
    AbstractField, Field,
};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_util::{reverse_bits_len, reverse_slice_index_bits};

use crate::domain::CircleDomain;

pub struct CircleEvaluations<F> {
    domain: CircleDomain<F>,
    values: RowMajorMatrix<F>,
}

fn cfft2nat(index: usize, log_height: usize) -> usize {
    let msb = index & 1;
    let index = reverse_bits_len(index >> 1, log_height - 1) << 1;
    if msb == 0 {
        index
    } else {
        (1 << log_height) - index - 1
    }
}

fn nat2cfft(index: usize, log_height: usize) -> usize {
    let (index, lsb) = (index >> 1, index & 1);
    reverse_bits_len(
        if lsb == 0 {
            index
        } else {
            (1 << log_height) - index - 1
        },
        log_height,
    )
}

impl<F: Copy + Send + Sync> CircleEvaluations<F> {
    pub fn from_natural(domain: CircleDomain<F>, values: impl Matrix<F>) -> Self {
        let dims = values.dimensions();
        assert_eq!(1 << domain.log_n, dims.height);
        Self {
            domain,
            values: RowMajorMatrix::new(
                (0..dims.height)
                    .flat_map(|r| values.row(cfft2nat(r, domain.log_n)))
                    .collect(),
                dims.width,
            ),
        }
    }
    pub fn to_natural(&self) -> RowMajorMatrix<F> {
        let dims = self.values.dimensions();
        RowMajorMatrix::new(
            (0..dims.height)
                .flat_map(|r| self.values.row(nat2cfft(r, self.domain.log_n)))
                .collect(),
            dims.width,
        )
    }
}

fn coset<F: ComplexExtendable>(
    log_n: usize,
    shift: Complex<F>,
) -> impl Iterator<Item = Complex<F>> {
    F::circle_two_adic_generator(log_n - 1)
        .shifted_powers(shift)
        .take(1 << (log_n - 1))
}

fn compute_twiddles<F: ComplexExtendable>(log_n: usize, shift: Complex<F>) -> Vec<Vec<F>> {
    (0..log_n)
        .map(|i| {
            let mut twiddles = if i == 0 {
                coset(log_n, shift).map(|pt| pt.imag()).collect_vec()
            } else {
                coset(log_n - (i - 1), shift.exp_power_of_2(i - 1))
                    .take(1 << (log_n - i - 1))
                    .map(|pt| pt.real())
                    .collect_vec()
            };
            reverse_slice_index_bits(&mut twiddles);
            twiddles
        })
        .collect()
}

fn apply_twiddles<F: Field>(values: &mut [F], twiddles: &[F], f: impl Fn(F, F, F) -> (F, F)) {
    let blk_sz = values.len() / twiddles.len();
    for (&t, blk) in izip!(twiddles, values.chunks_exact_mut(blk_sz)) {
        let (los, his) = blk.split_at_mut(blk_sz / 2);
        for (lo, hi) in izip!(los, his) {
            (*lo, *hi) = f(*lo, *hi, t);
        }
    }
}

pub fn cfft<F: ComplexExtendable>(evals: CircleEvaluations<F>) -> RowMajorMatrix<F> {
    let CircleEvaluations { domain, mut values } = evals;
    let CircleDomain { log_n, shift } = domain;
    for t in compute_twiddles(log_n, shift) {
        apply_twiddles(
            &mut values.values,
            &batch_multiplicative_inverse(&t),
            |lo, hi, t| (lo + hi, (lo - hi) * t),
        );
    }
    divide_by_height(&mut values);
    values
}

pub fn icfft<F: ComplexExtendable>(
    mut coeffs: RowMajorMatrix<F>,
    domain: CircleDomain<F>,
) -> CircleEvaluations<F> {
    let CircleDomain { log_n, shift } = domain;
    assert_eq!(coeffs.height(), 1 << log_n);
    for t in compute_twiddles(log_n, shift).into_iter().rev() {
        apply_twiddles(&mut coeffs.values, &t, |lo, hi, t| {
            (lo + hi * t, lo - hi * t)
        });
    }
    CircleEvaluations {
        domain,
        values: coeffs,
    }
}

pub fn circle_basis<F: ComplexExtendable>(pt: Complex<F>, log_n: usize) -> Vec<F> {
    let mut b = vec![F::one(), pt.imag()];
    let mut x = pt.real();
    for _ in 0..(log_n - 1) {
        for i in 0..b.len() {
            b.push(b[i] * x);
        }
        x = x.square().double() - F::one();
    }
    assert_eq!(b.len(), 1 << log_n);
    b
}

#[cfg(test)]
mod tests {
    use super::*;

    use alloc::vec;
    use itertools::Itertools;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_mersenne_31::Mersenne31;
    use rand::{random, thread_rng};

    type F = Mersenne31;

    #[test]
    fn test_cfft2nat() {
        assert_eq!(
            (0..8).map(|i| cfft2nat(i, 3)).collect_vec(),
            vec![0, 7, 4, 3, 2, 5, 6, 1],
        );
        for log_n in 1..5 {
            for i in 0..(1 << log_n) {
                assert_eq!(i, cfft2nat(nat2cfft(i, log_n), log_n));
            }
        }
    }

    fn do_test_cfft(log_n: usize) {
        let n = 1 << log_n;
        let shift = F::circle_two_adic_generator(F::CIRCLE_TWO_ADICITY).exp_u64(random());
        let domain = CircleDomain::new(log_n, shift);

        let evals = RowMajorMatrix::<F>::rand(&mut thread_rng(), n, 1 << 1);

        let coeffs = cfft(CircleEvaluations::from_natural(domain, evals.clone()));

        for (i, pt) in domain.points().enumerate() {
            assert_eq!(
                &*evals.row_slice(i),
                coeffs.columnwise_dot_product(&circle_basis(pt, log_n))
            );
        }

        let evals2 = icfft(coeffs, domain);

        assert_eq!(evals, evals2.to_natural());
    }

    #[test]
    fn test_cfft() {
        do_test_cfft(4);
        do_test_cfft(8);
    }
}
