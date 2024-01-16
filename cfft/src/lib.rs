//! This crate contains implementations of the CFT (Circle Fourier Transform) from the "Circle Stark" Paper

#![no_std]

extern crate alloc;
mod radix_2_butterfly;
mod traits;
mod util;

#[cfg(test)]
mod testing;

pub use radix_2_butterfly::*;
pub use traits::*;

use alloc::vec::Vec;

use p3_field::{AbstractExtensionField};
use p3_mersenne_31::{Mersenne31};

// Finally we move on to defining the CFFT, Inverse CFFT and Coset Extrapolation algorithms.
// Handily, the Inverse CFFT and Coset Extrapolation algorithms are actually identical, we just need to change our choice of twiddles.

/// Given a vector v of length N = 2^n, we consider this vector as the evaluations of a polynomial
/// f \in F_P[X, Y]/<X^2 + Y^2 = 1> on the circle coset gH = {g, g^3, ..., g^{2N - 1}}.
/// Here H is the unique subgroup of order N and g is an element of order 2N.
/// We can assume that deg_X(f) < N/2 and deg_Y(f) <= 1.
/// Then cfft computes the coefficients of f (In a slightly unusual basis).
pub fn cfft<F: AbstractExtensionField<Mersenne31>>(coeffs: &mut [F], twiddles: &[Vec<Mersenne31>]) {
    let n = coeffs.len();
    let n_u32: u32 = n.try_into().unwrap();
    let log_n: usize = n.trailing_zeros().try_into().unwrap();
    debug_assert_eq!(1_u32 << log_n, n_u32); // Our input better be a power of 2.

    for (i, twiddle) in twiddles.iter().enumerate() {
        let block_size = 1 << (log_n - i);
        let half_block_size = block_size >> 1;

        for chunck in coeffs.chunks_mut(block_size) {
            // Need this if we didn't change the twiddles.
            // chunck[half_block_size..].reverse();
            for j in 0..half_block_size {
                let s = chunck[j].clone();
                let t = chunck[block_size - j - 1].clone();
                chunck[j] = s.clone() + t.clone();
                chunck[block_size - j - 1] = (s - t) * twiddle[j];
            }
        }
    }
}

/// This is the inverse of the CFFT. Given the coefficients it computes the evaluations.
/// Depending on the twiddles input, this can also compute the coset evaluation as the algorithms are otherwise identical.
pub fn cfft_inv<F: AbstractExtensionField<Mersenne31>>(
    coeffs: &mut [F],
    twiddles: &[Vec<Mersenne31>],
) {
    let n = coeffs.len();
    let n_u32: u32 = n.try_into().unwrap();
    let log_n = n.trailing_zeros().try_into().unwrap();
    debug_assert_eq!(1_u32 << log_n, n_u32); // Our input better be a power of 2.

    for i in 0..log_n {
        let block_size = 1 << (i + 1);
        let half_block_size = block_size >> 1;

        for chunck in coeffs.chunks_mut(block_size) {
            for j in 0..half_block_size {
                let s = chunck[j].clone();
                let t = chunck[block_size - j - 1].clone() * twiddles[log_n - i - 1][j];

                chunck[j] = s.clone() + t.clone();
                chunck[block_size - j - 1] = s - t;
            }
            // Would need this if we don't start with the assumption that our input is shuffled in the right way.
            // chunck[half_block_size..].reverse();
        }
    }
}

#[cfg(test)]
mod tests {
    
    use crate::util::{cfft_domain, twin_coset_domain};
    use super::*;
    use crate::testing::evaluate_cfft_poly;
    use p3_field::{Field, AbstractField, ComplexExtension};
    use p3_mersenne_31::{Mersenne31Complex}; 

    #[test]
    fn fft_size_2() {
        let mut a = [(1 << 31) - 3, 2].map(Mersenne31::from_canonical_u32);
        let twiddles = cfft_twiddles::<Mersenne31, Mersenne31Complex<Mersenne31>>(1);

        cfft(&mut a, &twiddles);

        assert_eq!(a, [0, 4].map(Mersenne31::from_canonical_u32));
    }

    #[test]
    fn fft_size_4() {
        let mut a = [1, 11, 5, 1234].map(Mersenne31::from_canonical_u32);
        let size: usize = a.len();
        let size_u32 = size as u32;
        let log_size = size.trailing_zeros() as usize;

        let twiddles = cfft_twiddles::<Mersenne31, Mersenne31Complex<Mersenne31>>(log_size);

        cfft(&mut a, &twiddles);

        let points = cfft_domain::<Mersenne31, Mersenne31Complex<Mersenne31>>(log_size, size);

        let evals: Vec<_> = points
            .into_iter()
            .map(|point| {
                Mersenne31::from_canonical_u32(size_u32).inverse() * evaluate_cfft_poly(&a, point)
            })
            .collect();
        assert_eq!(evals, [1, 11, 5, 1234].map(Mersenne31::from_canonical_u32))
        // [4, 2, 1, 3]
    }

    #[test]
    fn fft_size_8() {
        let mut a = [1, 2, 1235, 4, 9, 11, 17, 6].map(Mersenne31::from_canonical_u32);
        let size: usize = a.len();
        let size_u32 = size as u32;
        let log_size = size.trailing_zeros() as usize;

        let twiddles = cfft_twiddles::<Mersenne31, Mersenne31Complex<Mersenne31>>(log_size);

        cfft(&mut a, &twiddles);

        let points = cfft_domain::<Mersenne31, Mersenne31Complex<Mersenne31>>(log_size, size);

        let evals: Vec<_> = points
            .into_iter()
            .map(|point| {
                Mersenne31::from_canonical_u32(size_u32).inverse() * evaluate_cfft_poly(&a, point)
            })
            .collect();

        assert_eq!(
            evals,
            [1, 2, 1235, 4, 9, 11, 17, 6].map(Mersenne31::from_canonical_u32)
        )
    }

    #[test]
    fn fft_size_16() {
        let mut a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
            .map(Mersenne31::from_canonical_u32);
        let size: usize = a.len();
        let size_u32 = size as u32;
        let log_size = size.trailing_zeros() as usize;
        let twiddles = cfft_twiddles::<Mersenne31, Mersenne31Complex<Mersenne31>>(log_size);

        let expected = a;

        cfft(&mut a, &twiddles);

        let points = cfft_domain::<Mersenne31, Mersenne31Complex<Mersenne31>>(log_size, size);

        let evals: Vec<_> = points
            .into_iter()
            .map(|point| {
                Mersenne31::from_canonical_u32(size_u32).inverse() * evaluate_cfft_poly(&a, point)
            })
            .collect();

        assert_eq!(evals, expected)
    }

    #[test]
    fn fft_inv_size_16() {
        let mut a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
            .map(Mersenne31::from_canonical_u32)
            .to_vec();
        let size: u32 = a.len() as u32;
        let log_size: usize = size.trailing_zeros() as usize;
        let cfft_twiddles = cfft_twiddles::<Mersenne31, Mersenne31Complex<Mersenne31>>(log_size);

        let expected: Vec<_> = a
            .iter()
            .map(|val| Mersenne31::from_canonical_u32(size) * *val)
            .collect();
        cfft(&mut a, &cfft_twiddles);

        let cfft_inv_twiddles = cfft_inv_twiddles::<Mersenne31, Mersenne31Complex<Mersenne31>>(log_size);
        cfft_inv(&mut a, &cfft_inv_twiddles);

        assert_eq!(a, expected);
    }

    #[test]
    fn coset_extrapolation_16() {
        let mut a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
            .map(Mersenne31::from_canonical_u32)
            .to_vec();

        let size: usize = a.len();
        let size_u32: u32 = size.try_into().unwrap();
        let log_size: usize = size_u32.trailing_zeros() as usize;

        let cfft_twiddles = cfft_twiddles::<Mersenne31, Mersenne31Complex<Mersenne31>>(log_size);

        // shuffle(&mut a);
        cfft(&mut a, &cfft_twiddles);

        let coset_elem = Mersenne31Complex::circle_two_adic_generator(10);
        let group_generator = Mersenne31Complex::circle_two_adic_generator(log_size - 1);

        let coset_points = twin_coset_domain(group_generator, coset_elem, size);

        let coset_evals: Vec<_> = coset_points
            .into_iter()
            .map(|point| evaluate_cfft_poly(&a, point))
            .collect();

        let coset_twiddles = coset_eval_twiddles(log_size, coset_elem);
        cfft_inv(&mut a, &coset_twiddles);

        assert_eq!(coset_evals, a)
    }
}
