use alloc::vec::Vec;
use core::mem;
use itertools::Itertools;
use p3_field::{
    batch_multiplicative_inverse,
    extension::{Complex, ComplexExtendable},
};
use p3_util::linear_map::LinearMap;
use tracing::instrument;

use crate::domain::twin_coset_domain;

#[derive(Default)]
pub(crate) struct TwiddleCache<F>(
    // (log_n, shift) -> (twiddles, inverse_twiddles)
    LinearMap<(usize, Complex<F>), (Vec<Vec<F>>, Option<Vec<Vec<F>>>)>,
);

impl<F: ComplexExtendable> TwiddleCache<F> {
    pub(crate) fn get_twiddles<'a>(
        &'a mut self,
        log_n: usize,
        shift: Complex<F>,
        inv: bool,
    ) -> &'a Vec<Vec<F>> {
        let cache = self
            .0
            .get_or_insert_with((log_n, shift), || (compute_twiddles(log_n, shift), None));
        if !inv {
            &cache.0
        } else {
            cache.1.get_or_insert_with(|| {
                cache
                    .0
                    .iter()
                    .map(|xs| batch_multiplicative_inverse(&xs))
                    .collect()
            })
        }
    }
}

#[instrument(skip(shift))]
fn compute_twiddles<F: ComplexExtendable>(log_n: usize, shift: Complex<F>) -> Vec<Vec<F>> {
    let size = 1 << (log_n - 1);
    let g = F::circle_two_adic_generator(log_n - 1);

    let init_domain = twin_coset_domain(g, shift, size).collect_vec();

    let mut working_domain: Vec<_> = init_domain
        .iter()
        .take(size / 2)
        .map(|x| x.real())
        .collect(); // After the first step we only need the real part.

    (0..log_n)
        .map(|i| {
            let size = working_domain.len();
            let output = if i == 0 {
                init_domain.iter().map(|x| x.imag()).collect_vec() // The twiddles in step one are the inverse imaginary parts.
            } else {
                let new_working_domain = working_domain
                    .iter()
                    .take(size / 2)
                    .map(|x| F::two() * x.square() - F::one())
                    .collect(); // When we square a point, the real part changes as x -> 2x^2 - 1.
                mem::replace(&mut working_domain, new_working_domain)
            };
            output
        })
        .collect()
}
