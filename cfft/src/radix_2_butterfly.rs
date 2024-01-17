use crate::traits::CircleSubgroupFFT;
use crate::util::{cfft_domain, twin_coset_domain};
use alloc::vec::Vec;
use p3_field::{ComplexExtension, Field};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;

/// An O(N Log(N)) implementation of the CFT roughly analagous to the FFT approach to the DFT.
/// See the paper "Circle Stark".
#[derive(Default, Clone)]
pub struct Radix2CFT;

impl<Base: Field, Ext: ComplexExtension<Base>> CircleSubgroupFFT<Base, Ext> for Radix2CFT {
    type Evaluations = RowMajorMatrix<Base>;

    fn cfft_batch(
        &self,
        mut mat: RowMajorMatrix<Base>,
    ) -> <Self as CircleSubgroupFFT<Base, Ext>>::Evaluations {
        let n = mat.height();
        let n_u32: u32 = n.try_into().unwrap();
        let log_n: usize = n.trailing_zeros().try_into().unwrap();
        debug_assert_eq!(1_u32 << log_n, n_u32); // The Height better be a power of 2.

        let width = mat.width();

        let twiddles = cfft_twiddles::<Base, Ext>(log_n);

        for (i, twiddle) in twiddles.iter().enumerate() {
            let block_size = 1 << (log_n - i);
            let half_block_size = block_size >> 1;

            for chuncks in mat.values.chunks_mut(block_size * width) {
                let (low_chunks, high_chunks) = chuncks.split_at_mut(half_block_size * width);

                low_chunks
                    .chunks_mut(width)
                    .zip(high_chunks.chunks_mut(width).rev())
                    .zip(twiddle)
                    .for_each(|((low_chunk, hi_chunk), twiddle)| {
                        butterfly_cfft(low_chunk, hi_chunk, *twiddle)
                    })
            }
        }
        mat
    }

    fn icfft_batch(&self, mut mat: RowMajorMatrix<Base>) -> RowMajorMatrix<Base> {
        let n = mat.height();
        let n_u32: u32 = n.try_into().unwrap();
        let log_n = n.trailing_zeros().try_into().unwrap();
        debug_assert_eq!(1_u32 << log_n, n_u32); // Our input better be a power of 2.

        let width = mat.width();

        let twiddles = cfft_inv_twiddles::<Base, Ext>(log_n);

        for (i, twiddle) in twiddles.iter().rev().enumerate() {
            let block_size = 1 << (i + 1);
            let half_block_size = block_size >> 1;

            for chuncks in mat.values.chunks_mut(block_size * width) {
                let (low_chunks, high_chunks) = chuncks.split_at_mut(half_block_size * width);

                low_chunks
                    .chunks_mut(width)
                    .zip(high_chunks.chunks_mut(width).rev())
                    .zip(twiddle)
                    .for_each(|((low_chunk, hi_chunk), twiddle)| {
                        butterfly_icfft(low_chunk, hi_chunk, *twiddle)
                    })
            }
        }

        mat
    }

    fn coset_cfft_batch(
        &self,
        _: RowMajorMatrix<Base>,
        _: Ext,
    ) -> <Self as CircleSubgroupFFT<Base, Ext>>::Evaluations {
        todo!()
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

fn butterfly_cfft<Base: Field>(low_chunk: &mut [Base], high_chunk: &mut [Base], twiddle: Base) {
    for (low, high) in low_chunk.iter_mut().zip(high_chunk) {
        let sum = *low + *high;
        let diff = (*low - *high) * twiddle;
        *low = sum;
        *high = diff;
    }
}

fn butterfly_icfft<Base: Field>(low_chunk: &mut [Base], high_chunk: &mut [Base], twiddle: Base) {
    for (low, high) in low_chunk.iter_mut().zip(high_chunk) {
        let high_twiddle = *high * twiddle;
        let sum = *low + high_twiddle;
        let diff = *low - high_twiddle;
        *low = sum;
        *high = diff;
    }
}

// Code for computing the twiddles.
// Currently, this requires us to recompute the twiddles every time we want to use the CFFT.
// This is somewhat expensive so, in the long run, it would be ideal to modify the structure so they can be precomputed.

/// Compute the twiddles for the CFFT.
/// Let N = 2^n be the size of our initial set. Then we start with the domain
/// {g, g^3, ..., g^{-3}, g^{-1}} for g a 2N'th root of unity.
/// The initial twiddle domain is the first half of the full domain.
/// In the first step our twiddles are the inverse imaginary parts and we simply halve the domain size.
/// In all subsequent steps our twiddles are the inverse real parts and we both halve the domain size and square every element.
pub fn cfft_twiddles<Base: Field, Ext: ComplexExtension<Base>>(log_n: usize) -> Vec<Vec<Base>> {
    let size = 1 << (log_n - 1);
    let init_domain = cfft_domain::<Base, Ext>(log_n, size); // Get the starting domain.

    let mut working_domain: Vec<_> = init_domain
        .iter()
        .take(size / 2)
        .map(|x| x.real())
        .collect(); // After the first step we only need the real part.

    (0..log_n)
        .map(|i| {
            let size = working_domain.len();
            if i == 0 {
                init_domain.iter().map(|x| x.imag().inverse()).collect() // The twiddles in step one are the inverse imaginary parts.
            } else {
                let output = working_domain.iter().map(|x| x.inverse()).collect(); // The twiddles in subsequent steps are the inverse real parts.
                working_domain = working_domain
                    .iter()
                    .take(size / 2)
                    .map(|x| Base::two() * *x * *x - Base::one())
                    .collect(); // When we square a point, the real part changes as x -> 2x^2 - 1.
                output
            }
        })
        .collect()
}

/// Compute the twiddles for the inverse CFFT.
/// The twiddles are essentially the same as in the CFFT case except we no longer need to take inverses.
pub fn cfft_inv_twiddles<Base: Field, Ext: ComplexExtension<Base>>(log_n: usize) -> Vec<Vec<Base>> {
    let size = 1 << (log_n - 1);
    let init_domain = cfft_domain::<Base, Ext>(log_n, size);

    let mut working_domain: Vec<_> = init_domain
        .iter()
        .take(size / 2)
        .map(|x| x.real())
        .collect();

    (0..log_n)
        .map(|i| {
            let size = working_domain.len();
            if i == 0 {
                init_domain.iter().map(|x| x.imag()).collect() // The twiddles in the outer step are the imaginary parts.
            } else {
                // Not sure if this is the cleanest was of doing this.

                let output = working_domain.clone(); // The twiddles in all other steps are the real parts.
                working_domain = working_domain
                    .iter()
                    .take(size / 2)
                    .map(|x| Base::two() * *x * *x - Base::one())
                    .collect();
                output
            }
        })
        .collect()
}

/// Compute the twiddles for the coset evaluation.
/// Unlike the previous cases, here we actually need to start with a given group element.
/// TODO: Explain what these twiddles are.
pub fn coset_eval_twiddles<Base: Field, Ext: ComplexExtension<Base>>(
    log_n: usize,
    coset_elem: Ext,
) -> Vec<Vec<Base>> {
    let size = 1 << (log_n - 1);
    let generator = Ext::circle_two_adic_generator(log_n - 1);

    let init_domain = twin_coset_domain::<Base, Ext>(generator, coset_elem, size);

    let mut working_domain: Vec<_> = init_domain
        .iter()
        .take(size / 2)
        .map(|x| x.real())
        .collect();

    (0..log_n)
        .map(|i| {
            let size = working_domain.len();
            if i == 0 {
                init_domain.iter().map(|x| x.imag()).collect()
            } else {
                let output = working_domain.clone();
                working_domain = working_domain
                    .iter()
                    .take(size / 2)
                    .map(|x| Base::two() * *x * *x - Base::one())
                    .collect();
                output
            }
        })
        .collect()
}

//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {

    use super::*;
    use crate::old::{cfft, cfft_inv, evaluate_cfft_poly};
    use crate::testing::{fft_ifft_test, fft_test};
    use crate::util::twin_coset_domain;
    use alloc::vec::Vec;
    use p3_field::{AbstractField, ComplexExtension};
    use p3_mersenne_31::{Mersenne31, Mersenne31Complex};

    #[test]
    fn fft_size_16() {
        fft_test::<Mersenne31, Mersenne31Complex<Mersenne31>, Radix2CFT, 16>();
    }

    #[test]
    fn fft_size_32() {
        fft_test::<Mersenne31, Mersenne31Complex<Mersenne31>, Radix2CFT, 32>();
    }

    #[test]
    fn fft_ifft_size_16() {
        fft_ifft_test::<Mersenne31, Mersenne31Complex<Mersenne31>, Radix2CFT, 16>();
    }

    #[test]
    fn fft_ifft_size_32() {
        fft_ifft_test::<Mersenne31, Mersenne31Complex<Mersenne31>, Radix2CFT, 32>();
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
        cfft::<Mersenne31>(&mut a, &cfft_twiddles);

        let cfft_inv_twiddles =
            cfft_inv_twiddles::<Mersenne31, Mersenne31Complex<Mersenne31>>(log_size);
        cfft_inv::<Mersenne31>(&mut a, &cfft_inv_twiddles);

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
        cfft::<Mersenne31>(&mut a, &cfft_twiddles);

        let coset_elem = Mersenne31Complex::circle_two_adic_generator(10);
        let group_generator = Mersenne31Complex::circle_two_adic_generator(log_size - 1);

        let coset_points = twin_coset_domain(group_generator, coset_elem, size);

        let coset_evals: Vec<_> = coset_points
            .into_iter()
            .map(|point| evaluate_cfft_poly(&a, point))
            .collect();

        let coset_twiddles = coset_eval_twiddles(log_size, coset_elem);
        cfft_inv::<Mersenne31>(&mut a, &coset_twiddles);

        assert_eq!(coset_evals, a)
    }
}
