use core::marker::PhantomData;

use alloc::vec;
use alloc::vec::Vec;
use itertools::Itertools;
use p3_commit::{LagrangeSelectors, PolynomialDomain};
use p3_field::{
    extension::{Complex, ComplexExtendable},
    AbstractField, ExtensionField, Field,
};
use p3_matrix::{dense::RowMajorMatrix, Matrix, MatrixRowSlices, MatrixRows};
use p3_util::log2_strict_usize;

use crate::{
    util::{point_to_univariate, rotate_univariate, univariate_to_point, v_0, v_n},
    Cfft,
};

/// Given a generator h for H and an element k, generate points in the twin coset kH u k^{-1}H.
/// The ordering is important here, the points will generated in the following interleaved pattern:
/// {k, k^{-1}h, kh, k^{-1}h^2, kh^2, ..., k^{-1}h^{-1}, kh^{-1}, k^{-1}}.
/// Size controls how many of these we want to compute. It should either be |H| or |H|/2 depending on if
/// we want simply the twiddles or the full domain. If k is has order 2|H| this is equal to cfft_domain.
pub(crate) fn twin_coset_domain<F: ComplexExtendable>(
    generator: Complex<F>,
    coset_elem: Complex<F>,
    size: usize,
) -> impl Iterator<Item = Complex<F>> {
    generator
        .shifted_powers(coset_elem)
        .interleave(generator.shifted_powers(generator * coset_elem.inverse()))
        .take(size)
}

pub(crate) fn cfft_domain<F: ComplexExtendable>(log_n: usize) -> impl Iterator<Item = Complex<F>> {
    let g = F::circle_two_adic_generator(log_n - 1);
    let shift = F::circle_two_adic_generator(log_n + 1);
    twin_coset_domain(g, shift, 1 << log_n)
}

// Only cfft (standard position) domain for now.
#[derive(Copy, Clone)]
pub struct CircleDomain<F> {
    pub(crate) log_n: usize,
    _phantom: PhantomData<F>,
}

impl<F: ComplexExtendable> CircleDomain<F> {
    pub(crate) fn new(log_n: usize) -> Self {
        Self {
            log_n,
            _phantom: PhantomData,
        }
    }
    fn gen(&self) -> Complex<F> {
        F::circle_two_adic_generator(self.log_n)
    }
    fn univariate_gen(&self) -> F {
        point_to_univariate(self.gen()).expect("generator is never (-1, 0)")
    }
}

impl<F: ComplexExtendable> PolynomialDomain for CircleDomain<F> {
    type Val = F;

    fn size(&self) -> usize {
        1 << self.log_n
    }

    fn next_point<Ext: ExtensionField<Self::Val>>(&self, x: Ext) -> Ext {
        rotate_univariate(x, self.univariate_gen()).expect("rotated point to (-1, 0)")
    }

    fn create_disjoint_domain(&self, min_size: usize) -> Self {
        let mut log_n = log2_strict_usize(min_size);
        // As we only work on standard position domain, we can only give a smaller or larger domain
        // With general twin cosets, we could give a disjoint domain of the same size
        while log_n == self.log_n {
            log_n += 1;
        }
        Self::new(log_n)
    }

    // For now we do the naive decomposition: do a full CFFT/ICFFT, taking every nth coefficient
    fn decomposed_width_factor(&self, max_size: usize) -> usize {
        let log_max_size = log2_strict_usize(max_size);
        assert!(self.log_n >= log_max_size);
        let decomposed_bits = self.log_n - log_max_size;
        1 << decomposed_bits
    }
    fn decomposed_domain(&self, max_size: usize) -> Self {
        let log_max_size = log2_strict_usize(max_size);
        assert!(self.log_n >= log_max_size);
        Self::new(log_max_size)
    }
    fn decomposed_point<Ext: ExtensionField<Self::Val>>(
        &self,
        point: Ext,
        _max_size: usize,
    ) -> Ext {
        point
    }
    fn decompose(
        &self,
        evals: RowMajorMatrix<Self::Val>,
        max_size: usize,
    ) -> RowMajorMatrix<Self::Val> {
        let cfft = Cfft::default();
        let coeffs = cfft.cfft_batch(evals);

        let n_chunks = self.decomposed_width_factor(max_size);
        let chunks = (0..n_chunks)
            .map(|i| {
                let view = coeffs.as_view();
                cfft.icfft_batch(view.vertically_strided(n_chunks, i).to_row_major_matrix())
            })
            .collect_vec();

        let mut hcat = vec![];
        for i in 0..coeffs.height() {
            for ch in &chunks {
                hcat.extend_from_slice(ch.row_slice(i));
            }
        }
        RowMajorMatrix::new(hcat, coeffs.width() * n_chunks)
    }
    fn recompose<Ext: ExtensionField<Self::Val>>(
        &self,
        evals: &[Ext],
        max_size: usize,
        point: Ext,
    ) -> Vec<Ext> {
        let n_chunks = self.decomposed_width_factor(max_size);
        todo!()
    }

    fn selectors_at_point<Ext: ExtensionField<Self::Val>>(
        &self,
        point: Ext,
    ) -> LagrangeSelectors<Ext> {
        let p = univariate_to_point(point);
        let zeroifier = v_n(p.real(), self.log_n);
        let first_point = F::circle_two_adic_generator(self.log_n + 1);
        let last_point = first_point.inverse();
        LagrangeSelectors {
            is_first_row: zeroifier / v_0(first_point.conjugate().rotate(p)),
            is_last_row: zeroifier / v_0(last_point.conjugate().rotate(p)),
            is_transition: v_0(last_point.conjugate().rotate(p)),
            inv_zeroifier: zeroifier.inverse(),
        }
    }

    // wow, really slow!
    fn selectors_on_coset(&self, coset: Self) -> LagrangeSelectors<Vec<Self::Val>> {
        let sels = cfft_domain::<F>(coset.log_n)
            .map(|p| self.selectors_at_point(point_to_univariate(p).unwrap()))
            .collect_vec();
        LagrangeSelectors {
            is_first_row: sels.iter().map(|s| s.is_first_row).collect(),
            is_last_row: sels.iter().map(|s| s.is_last_row).collect(),
            is_transition: sels.iter().map(|s| s.is_transition).collect(),
            inv_zeroifier: sels.iter().map(|s| s.inv_zeroifier).collect(),
        }
    }
}
