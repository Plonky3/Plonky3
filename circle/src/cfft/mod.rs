use alloc::vec;
use alloc::vec::Vec;

use itertools::{iterate, Itertools};
use p3_field::extension::ComplexExtendable;
use p3_field::{ExtensionField, Field};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_maybe_rayon::prelude::*;
use p3_util::reverse_slice_index_bits;
use tracing::instrument;

use crate::domain::CircleDomain;
use crate::point::{compute_lagrange_den_batched, Point};
use crate::{cfft_permute_index, cfft_permute_slice, CfftPermutable, CfftView};

mod par_chunked;

pub use par_chunked::ParChunkedCfft;

pub trait CfftAlgorithm<F: ComplexExtendable> {
    fn interpolate<M: Matrix<F>>(&self, evals: CircleEvaluations<F, M>) -> RowMajorMatrix<F>;

    fn evaluate(&self, domain: CircleDomain<F>, coeffs: RowMajorMatrix<F>) -> CircleEvaluations<F>;

    #[instrument(skip_all, fields(
        dims = %evals.values.dimensions(),
        target = target_domain.log_n,
    ))]
    fn extrapolate<M: Matrix<F>>(
        &self,
        target_domain: CircleDomain<F>,
        evals: CircleEvaluations<F, M>,
    ) -> CircleEvaluations<F> {
        assert!(target_domain.log_n >= evals.domain.log_n);
        self.evaluate(target_domain, self.interpolate(evals))
    }
}

#[derive(Clone)]
pub struct CircleEvaluations<F, M = RowMajorMatrix<F>> {
    pub(crate) domain: CircleDomain<F>,
    // Stored in "cfft order."
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
    ) -> CircleEvaluations<F, CfftView<M>> {
        CircleEvaluations::from_cfft_order(domain, values.cfft_perm_rows())
    }
    pub fn to_cfft_order(self) -> M {
        self.values
    }
    pub fn to_natural_order(self) -> CfftView<M> {
        self.values.cfft_perm_rows()
    }
}

impl<F: ComplexExtendable, M: Matrix<F>> CircleEvaluations<F, M> {
    pub fn evaluate_at_point<EF: ExtensionField<F>>(&self, point: Point<EF>) -> Vec<EF> {
        // Compute z_H
        let lagrange_num = self.domain.zeroifier(point);

        // Permute the domain to get it into the right format.
        let permuted_points = cfft_permute_slice(&self.domain.points().collect_vec());

        // Compute the lagrange denominators. This is batched as it lets us make use of batched_multiplicative_inverse.
        let lagrange_den = compute_lagrange_den_batched(&permuted_points, point, self.domain.log_n);

        // The columnwise_dot_product here consumes about 5% of the runtime for example prove_poseidon2_m31_keccak.
        // Definately something worth optimising further.
        self.values
            .columnwise_dot_product(&lagrange_den)
            .into_iter()
            .map(|x| x * lagrange_num)
            .collect_vec()
    }

    #[cfg(test)]
    pub(crate) fn dim(&self) -> usize
    where
        M: Clone,
    {
        use par_chunked::ParChunkedCfft;
        let coeffs = ParChunkedCfft::default().interpolate(self.clone());
        for (i, mut row) in coeffs.rows().enumerate() {
            if row.all(|x| x.is_zero()) {
                return i;
            }
        }
        coeffs.height()
    }
}

impl<F: ComplexExtendable> CircleDomain<F> {
    pub(crate) fn y_twiddles(&self) -> Vec<F> {
        let mut ys = self.coset0().map(|p| p.y).collect_vec();
        reverse_slice_index_bits(&mut ys);
        ys
    }
    pub(crate) fn nth_y_twiddle(&self, index: usize) -> F {
        self.nth_point(cfft_permute_index(index << 1, self.log_n)).y
    }
    pub(crate) fn x_twiddles(&self, layer: usize) -> Vec<F> {
        let gen = self.gen() * (1 << layer);
        let shift = self.shift * (1 << layer);
        let mut xs = iterate(shift, move |&p| p + gen)
            .map(|p| p.x)
            .take(1 << (self.log_n - layer - 2))
            .collect_vec();
        reverse_slice_index_bits(&mut xs);
        xs
    }
    pub(crate) fn nth_x_twiddle(&self, index: usize) -> F {
        (self.shift + self.gen() * index).x
    }
}

fn compute_twiddles<F: ComplexExtendable>(domain: CircleDomain<F>) -> Vec<Vec<F>> {
    assert!(domain.log_n >= 1);
    let mut pts = domain.coset0().collect_vec();
    reverse_slice_index_bits(&mut pts);
    let mut twiddles = vec![pts.iter().map(|p| p.y).collect_vec()];
    if domain.log_n >= 2 {
        twiddles.push(pts.iter().step_by(2).map(|p| p.x).collect_vec());
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
    }
    twiddles
}

pub fn circle_basis<F: Field>(p: Point<F>, log_n: usize) -> Vec<F> {
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
    use itertools::iproduct;
    use p3_field::extension::BinomialExtensionField;
    use p3_mersenne_31::Mersenne31;
    use par_chunked::ParChunkedCfft;
    use rand::{random, thread_rng};

    use super::*;

    type F = Mersenne31;
    type EF = BinomialExtensionField<F, 3>;

    #[test]
    fn test_par_chunked_cfft() {
        let cfft = ParChunkedCfft::default();
        test_cfft_algo(&cfft);
    }

    fn test_cfft_algo<Cfft: CfftAlgorithm<F>>(cfft: &Cfft) {
        test_cfft_icfft(cfft);
        test_extrapolation(cfft);
        eval_at_point_matches_cfft(cfft);
        eval_at_point_matches_lde(cfft);
    }

    fn test_cfft_icfft<Cfft: CfftAlgorithm<F>>(cfft: &Cfft) {
        for (log_n, width) in iproduct!(2..5, [1, 4, 11]) {
            let shift = Point::generator(F::CIRCLE_TWO_ADICITY) * random();
            let domain = CircleDomain::<F>::new(log_n, shift);
            let trace = RowMajorMatrix::<F>::rand(&mut thread_rng(), 1 << log_n, width);
            let coeffs =
                cfft.interpolate(CircleEvaluations::from_natural_order(domain, trace.clone()));
            assert_eq!(
                cfft.evaluate(domain, coeffs.clone())
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

    fn test_extrapolation<Cfft: CfftAlgorithm<F>>(cfft: &Cfft) {
        for (log_n, log_blowup) in iproduct!(2..5, [1, 2, 3]) {
            let evals = CircleEvaluations::<F>::from_natural_order(
                CircleDomain::standard(log_n),
                RowMajorMatrix::rand(&mut thread_rng(), 1 << log_n, 11),
            );
            let lde = cfft.extrapolate(CircleDomain::standard(log_n + log_blowup), evals.clone());

            let coeffs = cfft.interpolate(evals);
            let lde_coeffs = cfft.interpolate(lde);

            for r in 0..coeffs.height() {
                assert_eq!(&*coeffs.row_slice(r), &*lde_coeffs.row_slice(r));
            }
            for r in coeffs.height()..lde_coeffs.height() {
                assert!(lde_coeffs.row(r).all(|x| x.is_zero()));
            }
        }
    }

    fn eval_at_point_matches_cfft<Cfft: CfftAlgorithm<F>>(cfft: &Cfft) {
        for (log_n, width) in iproduct!(2..5, [1, 4, 11]) {
            let evals = CircleEvaluations::<F>::from_natural_order(
                CircleDomain::standard(log_n),
                RowMajorMatrix::rand(&mut thread_rng(), 1 << log_n, width),
            );

            let pt = Point::<EF>::from_projective_line(random());

            assert_eq!(
                evals.clone().evaluate_at_point(pt),
                cfft.interpolate(evals)
                    .columnwise_dot_product(&circle_basis(pt, log_n))
            );
        }
    }

    fn eval_at_point_matches_lde<Cfft: CfftAlgorithm<F>>(cfft: &Cfft) {
        for (log_n, width, log_blowup) in iproduct!(2..8, [1, 4, 11], [1, 2]) {
            let evals = CircleEvaluations::<F>::from_natural_order(
                CircleDomain::standard(log_n),
                RowMajorMatrix::rand(&mut thread_rng(), 1 << log_n, width),
            );
            let lde = cfft.extrapolate(CircleDomain::standard(log_n + log_blowup), evals.clone());
            let zeta = Point::<EF>::from_projective_line(random());
            assert_eq!(evals.evaluate_at_point(zeta), lde.evaluate_at_point(zeta));
            assert_eq!(
                evals.evaluate_at_point(zeta),
                cfft.interpolate(evals)
                    .columnwise_dot_product(&circle_basis(zeta, log_n))
            );
            assert_eq!(
                lde.evaluate_at_point(zeta),
                cfft.interpolate(lde)
                    .columnwise_dot_product(&circle_basis(zeta, log_n + log_blowup))
            );
        }
    }
}
