use alloc::vec;
use alloc::vec::Vec;

use itertools::Itertools;
use p3_commit::{LagrangeSelectors, PolynomialSpace};
use p3_field::extension::{Complex, ComplexExtendable};
use p3_field::{batch_multiplicative_inverse, AbstractField, ExtensionField};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_util::{log2_ceil_usize, log2_strict_usize};
use tracing::instrument;

use crate::util::{point_to_univariate, s_p_at_p, univariate_to_point, v_0, v_n};

/// A twin-coset of the circle group on F. It has a power-of-two size and an arbitrary shift.
///
/// X is generator, O is the first coset, goes counterclockwise
/// ```text
///   O X .
///  .     .
/// .       O <- start = shift
/// .   .   - (1,0)
/// O       .
///  .     .
///   . . O
/// ```
///
/// For ordering reasons, the other half will start at gen / shift:
/// ```text
///   . X O  <- start = gen/shift
///  .     .
/// O       .
/// .   .   - (1,0)
/// .       O
///  .     .
///   O . .
/// ```
///
/// The full domain is the interleaving of these two cosets
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct CircleDomain<F> {
    // log_n corresponds to the log size of the WHOLE domain
    pub(crate) log_n: usize,
    pub(crate) shift: Complex<F>,
}

impl<F: ComplexExtendable> CircleDomain<F> {
    pub(crate) const fn new(log_n: usize, shift: Complex<F>) -> Self {
        Self { log_n, shift }
    }
    pub(crate) fn standard(log_n: usize) -> Self {
        Self {
            log_n,
            shift: F::circle_two_adic_generator(log_n + 1),
        }
    }
    fn is_standard(&self) -> bool {
        self.shift == F::circle_two_adic_generator(self.log_n + 1)
    }
    pub(crate) fn points(&self) -> impl Iterator<Item = Complex<F>> {
        let half_gen = F::circle_two_adic_generator(self.log_n - 1);
        let coset0 = half_gen.shifted_powers(self.shift);
        let coset1 = half_gen.shifted_powers(half_gen / self.shift);
        coset0.interleave(coset1).take(1 << self.log_n)
    }
    pub(crate) fn nth_point(&self, idx: usize) -> Complex<F> {
        // Only implemented for standard position.
        assert!(self.is_standard());
        let gen = F::circle_two_adic_generator(self.log_n);
        self.shift * gen.exp_u64(idx as u64)
    }

    /// Computes the lagrange basis at point, not yet normalized by the value of the domain
    /// vanishing poly, since that is more efficient to compute after the dot product.
    #[instrument(skip_all, fields(log_n = %self.log_n))]
    pub(crate) fn lagrange_basis<EF: ExtensionField<F>>(&self, point: Complex<EF>) -> Vec<EF> {
        let domain = self.points().collect_vec();

        // the denominator so that the lagrange basis is normalized to 1
        // TODO: this depends only on domain, so should be precomputed
        let lagrange_normalizer: Vec<F> = domain
            .iter()
            .map(|p| s_p_at_p(p.real(), p.imag(), self.log_n))
            .collect();

        let basis = domain
            .into_iter()
            .zip(&lagrange_normalizer)
            .map(|(p, &ln)| {
                // ext * base
                // TODO: this can be sped up
                v_0(p.conjugate().rotate(point)) * ln
            })
            .collect_vec();

        batch_multiplicative_inverse(&basis)
    }
}

impl<F: ComplexExtendable> PolynomialSpace for CircleDomain<F> {
    type Val = F;

    fn size(&self) -> usize {
        1 << self.log_n
    }

    fn first_point(&self) -> Self::Val {
        point_to_univariate(self.shift).unwrap()
    }

    fn next_point<Ext: ExtensionField<Self::Val>>(&self, x: Ext) -> Option<Ext> {
        // Only in standard position do we have an algebraic expression to access the next point.
        if self.is_standard() {
            let gen = F::circle_two_adic_generator(self.log_n);
            Some(point_to_univariate(gen.rotate(univariate_to_point(x).unwrap())).unwrap())
        } else {
            None
        }
    }

    fn create_disjoint_domain(&self, min_size: usize) -> Self {
        // Right now we simply guarantee the domain is disjoint by returning a
        // larger standard position coset, which is fine because we always ask for a larger
        // domain. If we wanted good performance for a disjoint domain of the same size,
        // we could change the shift. Also we could support nonstandard twin cosets.
        assert!(
            self.is_standard(),
            "create_disjoint_domain not currently supported for nonstandard twin cosets"
        );
        let log_n = log2_ceil_usize(min_size);
        // Any standard position coset that is not the same size as us will be disjoint.
        Self::standard(if log_n == self.log_n {
            log_n + 1
        } else {
            log_n
        })
    }

    fn zp_at_point<Ext: ExtensionField<Self::Val>>(&self, point: Ext) -> Ext {
        v_n(univariate_to_point(point).unwrap().real(), self.log_n)
            - v_n(self.shift.real(), self.log_n)
    }

    fn selectors_at_point<Ext: ExtensionField<Self::Val>>(
        &self,
        point: Ext,
    ) -> LagrangeSelectors<Ext> {
        let zeroifier = self.zp_at_point(point);
        let p = univariate_to_point(point).unwrap();
        LagrangeSelectors {
            is_first_row: zeroifier / v_0(self.shift.conjugate().rotate(p)),
            is_last_row: zeroifier / v_0(self.shift.rotate(p)),
            // This is the transition selector from the paper, but seems to send
            // the quotient out of FFT space. It has a simple zero at the last point
            // and a simple pole at the negation of the last point.
            // is_transition: v_0(self.shift.rotate(p)),
            // Instead we use this selector which has two zeros at the last point,
            // which seems to work. TODO: More investigation is needed.
            is_transition: self.shift.rotate(p).real() - Ext::one(),
            inv_zeroifier: zeroifier.inverse(),
        }
    }

    // wow, really slow!
    #[instrument(skip_all, fields(log_n = %coset.log_n))]
    fn selectors_on_coset(&self, coset: Self) -> LagrangeSelectors<Vec<Self::Val>> {
        let sels = coset
            .points()
            .map(|p| self.selectors_at_point(point_to_univariate(p).unwrap()))
            .collect_vec();
        LagrangeSelectors {
            is_first_row: sels.iter().map(|s| s.is_first_row).collect(),
            is_last_row: sels.iter().map(|s| s.is_last_row).collect(),
            is_transition: sels.iter().map(|s| s.is_transition).collect(),
            inv_zeroifier: sels.iter().map(|s| s.inv_zeroifier).collect(),
        }
    }

    /// Decompose a domain into disjoint twin-cosets.
    fn split_domains(&self, num_chunks: usize) -> Vec<Self> {
        assert!(self.is_standard());
        let log_chunks = log2_strict_usize(num_chunks);
        self.points()
            .take(num_chunks)
            .map(|shift| CircleDomain {
                log_n: self.log_n - log_chunks,
                shift,
            })
            .collect()
    }

    /*
    chunks=2:

          1 . 1
         .     .
        0       0 <-- start
        .   .   - (1,0)
        0       0
         .     .
          1 . 1


    idx -> which chunk to put it in:
    chunks=2: 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 0
    chunks=4: 0 1 2 3 3 2 1 0 0 1 2 3 3 2 1 0
    */
    fn split_evals(
        &self,
        num_chunks: usize,
        evals: RowMajorMatrix<Self::Val>,
    ) -> Vec<RowMajorMatrix<Self::Val>> {
        let log_chunks = log2_strict_usize(num_chunks);
        assert!(evals.height() >> (log_chunks + 1) >= 1);
        let width = evals.width();
        let mut values: Vec<Vec<Self::Val>> = vec![vec![]; num_chunks];
        evals
            .rows()
            .enumerate()
            .for_each(|(i, row)| values[forward_backward_index(i, num_chunks)].extend(row));
        values
            .into_iter()
            .map(|v| RowMajorMatrix::new(v, width))
            .collect()
    }
}

// 0 1 2 .. len-1 len len len-1 .. 1 0 0 1 ..
fn forward_backward_index(mut i: usize, len: usize) -> usize {
    i %= 2 * len;
    if i < len {
        i
    } else {
        2 * len - 1 - i
    }
}

#[cfg(test)]
mod tests {
    use hashbrown::HashSet;
    use itertools::izip;
    use p3_mersenne_31::Mersenne31;
    use rand::{random, thread_rng};

    use super::*;
    use crate::util::eval_circle_polys;
    use crate::Cfft;

    fn assert_is_twin_coset<F: ComplexExtendable>(d: CircleDomain<F>) {
        let pts = d.points().collect_vec();
        let half_n = pts.len() >> 1;
        for (l, r) in izip!(&pts[..half_n], pts[half_n..].iter().rev()) {
            assert_eq!(*l, r.conjugate());
        }
    }

    fn do_test_circle_domain(log_n: usize, width: usize) {
        let n = 1 << log_n;

        type F = Mersenne31;
        let d = CircleDomain::<F>::standard(log_n);

        // we can move around the circle and end up where we started
        let p0 = d.first_point();
        let mut p1 = p0;
        for _ in 0..(n - 1) {
            p1 = d.next_point(p1).unwrap();
            assert_ne!(p1, p0);
        }
        assert_eq!(d.next_point(p1).unwrap(), p0);

        // .points() is the same as first_point -> next_point
        let mut uni_point = d.first_point();
        for p in d.points() {
            assert_eq!(univariate_to_point(uni_point), Some(p));
            uni_point = d.next_point(uni_point).unwrap();
        }

        // disjoint domain is actually disjoint, and large enough
        let seen: HashSet<Complex<F>> = d.points().collect();
        for disjoint_size in [10, 100, n - 5, n + 15] {
            let dd = d.create_disjoint_domain(disjoint_size);
            assert!(dd.size() >= disjoint_size);
            for pt in dd.points() {
                assert!(!seen.contains(&pt));
            }
        }

        // zp is zero
        for p in d.points() {
            assert_eq!(d.zp_at_point(point_to_univariate(p).unwrap()), F::zero());
        }

        // split domains
        let evals = RowMajorMatrix::rand(&mut thread_rng(), n, width);
        let orig: Vec<(Complex<F>, Vec<F>)> = d
            .points()
            .zip(evals.rows().map(|r| r.collect_vec()))
            .collect();
        for num_chunks in [1, 2, 4, 8] {
            let mut combined = vec![];

            let sds = d.split_domains(num_chunks);
            assert_eq!(sds.len(), num_chunks);
            let ses = d.split_evals(num_chunks, evals.clone());
            assert_eq!(ses.len(), num_chunks);
            for (sd, se) in izip!(sds, ses) {
                // Split domains are twin cosets
                assert_is_twin_coset(sd);
                // Split domains have correct size wrt original domain
                assert_eq!(sd.size() * num_chunks, d.size());
                assert_eq!(se.width(), evals.width());
                assert_eq!(se.height() * num_chunks, d.size());
                combined.extend(sd.points().zip(se.rows().map(|r| r.collect_vec())));
            }
            // Union of split domains and evals is the original domain and evals
            assert_eq!(
                orig.iter().map(|x| x.0).collect::<HashSet<_>>(),
                combined.iter().map(|x| x.0).collect::<HashSet<_>>(),
                "union of split domains is orig domain"
            );
            assert_eq!(
                orig.iter().map(|x| &x.1).collect::<HashSet<_>>(),
                combined.iter().map(|x| &x.1).collect::<HashSet<_>>(),
                "union of split evals is orig evals"
            );
            assert_eq!(
                orig.iter().collect::<HashSet<_>>(),
                combined.iter().collect::<HashSet<_>>(),
                "split domains and evals correspond to orig domains and evals"
            );
        }
    }

    #[test]
    fn test_circle_domain() {
        do_test_circle_domain(4, 32);
        do_test_circle_domain(10, 32);
    }

    #[test]
    fn test_barycentric() {
        let log_n = 10;
        let n = 1 << log_n;

        type F = Mersenne31;

        let evals = RowMajorMatrix::<F>::rand(&mut thread_rng(), n, 16);

        let cfft = Cfft::default();

        let shift: Complex<F> = univariate_to_point(random()).unwrap();
        let d = CircleDomain { shift, log_n };

        let coeffs = cfft.coset_cfft_batch(evals.clone(), shift);

        // simple barycentric
        let zeta: Complex<F> = univariate_to_point(random()).unwrap();

        let basis = d.lagrange_basis(zeta);
        let v_n_at_zeta = v_n(zeta.real(), log_n) - v_n(shift.real(), log_n);

        let ys = evals
            .columnwise_dot_product(&basis)
            .into_iter()
            .map(|x| x * v_n_at_zeta)
            .collect_vec();

        assert_eq!(ys, eval_circle_polys(&coeffs, zeta));
    }
}
