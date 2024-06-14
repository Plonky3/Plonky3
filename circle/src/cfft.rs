use itertools::{izip, Itertools};
use p3_commit::PolynomialSpace;
use p3_dft::divide_by_height;
use p3_field::{batch_multiplicative_inverse, extension::ComplexExtendable, ExtensionField, Field};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_util::reverse_slice_index_bits;

use crate::{
    domain::CircleDomain,
    natural_slice_to_cfft,
    ordering::{CfftAsNaturalPerm, NaturalAsCfftPerm},
    point::Point,
    CfftAsNaturalView, NaturalAsCfftView,
};

#[derive(Clone)]
pub struct CircleEvaluations<F, M = RowMajorMatrix<F>> {
    pub(crate) domain: CircleDomain<F>,
    pub(crate) values: M,
}

impl<F: Copy + Send + Sync, M: Matrix<F>> CircleEvaluations<F, M> {
    pub(crate) fn from_cfft_order(domain: CircleDomain<F>, values: M) -> Self {
        assert_eq!(1 << domain.log_n, values.height());
        Self { domain, values }
    }
    pub fn from_natural_order(
        domain: CircleDomain<F>,
        values: M,
    ) -> CircleEvaluations<F, NaturalAsCfftView<M>> {
        CircleEvaluations::from_cfft_order(domain, NaturalAsCfftPerm::view(values))
    }
    pub(crate) fn to_cfft_order(self) -> M {
        self.values
    }
    pub fn to_natural_order(self) -> CfftAsNaturalView<M> {
        CfftAsNaturalPerm::view(self.values)
    }
}

impl<F: ComplexExtendable, M: Matrix<F>> CircleEvaluations<F, M> {
    pub fn interpolate(self) -> RowMajorMatrix<F> {
        let CircleEvaluations { domain, values } = self;
        let mut values = values.to_row_major_matrix();
        for twiddles in compute_twiddles(domain) {
            fft_layer(
                &mut values.values,
                &batch_multiplicative_inverse(&twiddles),
                |lo, hi, t| (lo + hi, (lo - hi) * t),
            );
        }
        divide_by_height(&mut values);
        values
    }
    pub fn extrapolate(
        self,
        target_domain: CircleDomain<F>,
    ) -> CircleEvaluations<F, RowMajorMatrix<F>> {
        assert!(target_domain.size() >= self.domain.size());
        let mut coeffs = self.interpolate();
        coeffs.pad_to_height(target_domain.size(), F::zero());
        CircleEvaluations::<F>::evaluate(target_domain, coeffs)
    }
    pub fn evaluate_at_point<EF: ExtensionField<F>>(&self, point: Point<EF>) -> Vec<EF> {
        let v_n = point.v_n(self.domain.log_n) - self.domain.shift.v_n(self.domain.log_n);
        self.values
            .columnwise_dot_product(&natural_slice_to_cfft(&self.domain.lagrange_basis(point)))
            .into_iter()
            .map(|x| x * v_n)
            .collect_vec()
    }

    #[cfg(test)]
    pub(crate) fn dim(&self) -> usize
    where
        M: Clone,
    {
        let coeffs = self.clone().interpolate();
        for (i, mut row) in coeffs.rows().enumerate() {
            if row.all(|x| x.is_zero()) {
                return i;
            }
        }
        coeffs.height()
    }
}

impl<F: ComplexExtendable> CircleEvaluations<F, RowMajorMatrix<F>> {
    pub fn evaluate(domain: CircleDomain<F>, mut coeffs: RowMajorMatrix<F>) -> Self {
        assert_eq!(coeffs.height(), 1 << domain.log_n);
        for twiddles in compute_twiddles(domain).into_iter().rev() {
            fft_layer(&mut coeffs.values, &twiddles, |lo, hi, t| {
                (lo + hi * t, lo - hi * t)
            });
        }
        Self::from_cfft_order(domain, coeffs)
    }
}

fn fft_layer<F: Field>(values: &mut [F], twiddles: &[F], f: impl Fn(F, F, F) -> (F, F)) {
    let blk_sz = values.len() / twiddles.len();
    for (&t, blk) in izip!(twiddles, values.chunks_exact_mut(blk_sz)) {
        let (los, his) = blk.split_at_mut(blk_sz / 2);
        for (lo, hi) in izip!(los, his) {
            (*lo, *hi) = f(*lo, *hi, t);
        }
    }
}

fn compute_twiddles<F: ComplexExtendable>(domain: CircleDomain<F>) -> Vec<Vec<F>> {
    assert!(domain.log_n >= 2);
    let mut pts = domain.coset0().collect_vec();
    reverse_slice_index_bits(&mut pts);
    let mut twiddles = vec![
        pts.iter().map(|p| p.y).collect_vec(),
        pts.iter().step_by(2).map(|p| p.x).collect_vec(),
    ];
    for i in 0..(domain.log_n - 2) {
        let prev = twiddles.last().unwrap();
        assert_eq!(prev.len(), 1 << (domain.log_n - 2 - i));
        let cur = prev
            .iter()
            .step_by(2)
            .map(|x| x.square().double() - F::one())
            .collect_vec();
        twiddles.push(cur);
    }
    twiddles
}

pub fn circle_basis<F: ComplexExtendable>(p: Point<F>, log_n: usize) -> Vec<F> {
    let mut b = vec![F::one(), p.y];
    let mut x = p.x;
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

    use itertools::iproduct;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_mersenne_31::Mersenne31;
    use rand::{random, thread_rng};

    type F = Mersenne31;
    type EF = Mersenne31;
    // type EF = BinomialExtensionField<F, 3>;

    #[test]
    fn test_cfft_icfft() {
        for (log_n, width) in iproduct!(2..5, [1, 2, 4]) {
            let shift = Point::generator(F::CIRCLE_TWO_ADICITY) * random();
            let domain = CircleDomain::<F>::new(log_n, shift);
            let trace = RowMajorMatrix::<F>::rand(&mut thread_rng(), 1 << log_n, width);
            let coeffs = CircleEvaluations::from_natural_order(domain, trace.clone()).interpolate();
            assert_eq!(
                CircleEvaluations::evaluate(domain, coeffs.clone())
                    .to_natural_order()
                    .to_row_major_matrix(),
                trace,
                "icfft(cfft(evals)) is identity",
            );
            for (i, pt) in domain.points().enumerate() {
                assert_eq!(
                    &*trace.row_slice(i),
                    coeffs.columnwise_dot_product(&circle_basis(pt, log_n)),
                    "coeffs can be evaluated with circle_basis",
                );
            }
        }
    }

    #[test]
    fn test_extrapolation() {
        for (log_n, log_blowup) in iproduct!(2..5, [1, 2]) {
            let evals = CircleEvaluations::<F>::from_natural_order(
                CircleDomain::standard(log_n),
                RowMajorMatrix::rand(&mut thread_rng(), 1 << log_n, 4),
            );
            let lde = evals
                .clone()
                .extrapolate(CircleDomain::standard(log_n + log_blowup));

            let coeffs = evals.interpolate();
            let lde_coeffs = lde.interpolate();

            for r in 0..coeffs.height() {
                assert_eq!(&*coeffs.row_slice(r), &*lde_coeffs.row_slice(r));
            }
            for r in coeffs.height()..lde_coeffs.height() {
                assert!(lde_coeffs.row(r).all(|x| x.is_zero()));
            }
        }
    }

    #[test]
    fn test_barycentric() {
        for (log_n, width) in iproduct!(2..5, [1, 2, 4]) {
            let evals = CircleEvaluations::<F>::from_natural_order(
                CircleDomain::standard(log_n),
                RowMajorMatrix::rand(&mut thread_rng(), 1 << log_n, width),
            );

            let pt = Point::<EF>::from_projective_line(random());

            assert_eq!(
                evals.clone().evaluate_at_point(pt),
                evals
                    .interpolate()
                    .columnwise_dot_product(&circle_basis(pt, log_n))
            );
        }
    }
}
