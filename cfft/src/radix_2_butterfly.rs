use alloc::vec::Vec;

use itertools::izip;
use p3_field::{batch_multiplicative_inverse, ComplexExtension, Field};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_util::log2_strict_usize;

use crate::traits::CircleSubgroupFt;
use crate::util::{cfft_domain, twin_coset_domain};

/// An O(N Log(N)) implementation of the CFT roughly analagous to the FFT approach to the DFT.
/// See the paper "Circle Stark".
#[derive(Default, Clone)]
pub struct Radix2Cft;

impl<Base: Field, Ext: ComplexExtension<Base>> CircleSubgroupFt<Base, Ext> for Radix2Cft {
    type Evaluations = RowMajorMatrix<Base>;

    fn cfft_batch(
        &self,
        mut mat: RowMajorMatrix<Base>,
    ) -> <Self as CircleSubgroupFt<Base, Ext>>::Evaluations {
        let n = mat.height();
        let log_n: usize = log2_strict_usize(n);

        let width = mat.width();

        let twiddles = cfft_twiddles::<Base, Ext>(log_n, true); // These should be precomputed.

        for (i, twiddle) in twiddles.iter().enumerate() {
            let block_size = 1 << (log_n - i);
            let half_block_size = block_size >> 1;

            for chunks in mat.values.chunks_exact_mut(block_size * width) {
                let (low_chunks, high_chunks) = chunks.split_at_mut(half_block_size * width);

                for (twiddle, lo, hi) in izip!(
                    twiddle,
                    low_chunks.chunks_exact_mut(width),
                    high_chunks.chunks_exact_mut(width).rev(),
                ) {
                    butterfly_cfft(lo, hi, *twiddle)
                }
            }
        }
        mat
    }

    fn coset_icfft_batch(
        &self,
        mut mat: RowMajorMatrix<Base>,
        coset_elem: Ext,
    ) -> RowMajorMatrix<Base> {
        let n = mat.height();
        let log_n = log2_strict_usize(n);
        let width = mat.width();

        let twiddles = coset_eval_twiddles::<Base, Ext>(log_n, coset_elem); // Likely fast to precompute these.

        for (i, twiddle) in twiddles.iter().rev().enumerate() {
            let block_size = 1 << (i + 1);
            let half_block_size = block_size >> 1;

            for chunks in mat.values.chunks_exact_mut(block_size * width) {
                let (low_chunks, high_chunks) = chunks.split_at_mut(half_block_size * width);

                for (twiddle, lo, hi) in izip!(
                    twiddle,
                    low_chunks.chunks_exact_mut(width),
                    high_chunks.chunks_exact_mut(width).rev(),
                ) {
                    butterfly_icfft(lo, hi, *twiddle)
                }
            }
        }

        mat
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
/// In the first step our twiddles come from the imaginary parts and we simply halve the domain size.
/// In all subsequent steps our twiddles come from the real parts and we both halve the domain size and square every element.
/// If inv is True, we invert all twiddles to get the cfft. If inv is false this produces the twiddles for the icfft.
pub fn cfft_twiddles<Base: Field, Ext: ComplexExtension<Base>>(
    log_n: usize,
    inv: bool,
) -> Vec<Vec<Base>> {
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
            let output = if i == 0 {
                init_domain.iter().map(|x| x.imag()).collect::<Vec<_>>() // The twiddles in step one are the inverse imaginary parts.
            } else {
                let output = working_domain.to_vec(); // The twiddles in subsequent steps are the inverse real parts.
                working_domain = working_domain
                    .iter()
                    .take(size / 2)
                    .map(|x| Base::two() * *x * *x - Base::one())
                    .collect(); // When we square a point, the real part changes as x -> 2x^2 - 1.
                output
            };
            if inv {
                batch_multiplicative_inverse(&output) // It would be faster to move this outside the loop.
            } else {
                output
            }
        })
        .collect()
}

/// Compute the twiddles for the coset evaluation.
/// Fix an (N/2)'th root of unity h and let k = coset_elem. Then we start with the domain
/// {k, k^{-1}h, kh, k^{-1}h^2, ..., kh^{-1}, k^{-1}}.
/// The initial twiddle domain is the first half of the full domain.
/// In the first step our twiddles are the imaginary parts and we simply halve the domain size.
/// In all subsequent steps our twiddles are the real parts and we both halve the domain size and square every element.
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

    use p3_mersenne_31::{Mersenne31, Mersenne31Complex};

    use super::*;
    use crate::testing::{coset_eval_test, coset_lde_test, fft_ifft_test, fft_test, ifft_test};

    #[test]
    fn fft_size_16() {
        fft_test::<Mersenne31, Mersenne31Complex<Mersenne31>, Radix2Cft, 16>();
    }

    #[test]
    fn fft_size_32() {
        fft_test::<Mersenne31, Mersenne31Complex<Mersenne31>, Radix2Cft, 32>();
    }

    #[test]
    fn ifft_size_16() {
        ifft_test::<Mersenne31, Mersenne31Complex<Mersenne31>, Radix2Cft, 16>();
    }

    #[test]
    fn ifft_size_32() {
        ifft_test::<Mersenne31, Mersenne31Complex<Mersenne31>, Radix2Cft, 32>();
    }

    #[test]
    fn fft_ifft_size_16_5() {
        fft_ifft_test::<Mersenne31, Mersenne31Complex<Mersenne31>, Radix2Cft, 16, 5>();
    }

    #[test]
    fn fft_ifft_size_32_3() {
        fft_ifft_test::<Mersenne31, Mersenne31Complex<Mersenne31>, Radix2Cft, 32, 3>();
    }

    #[test]
    fn coset_eval_size_16() {
        coset_eval_test::<Mersenne31, Mersenne31Complex<Mersenne31>, Radix2Cft, 16>();
    }

    #[test]
    fn coset_eval_size_32() {
        coset_eval_test::<Mersenne31, Mersenne31Complex<Mersenne31>, Radix2Cft, 32>();
    }

    #[test]
    fn coset_lde_size_16() {
        coset_lde_test::<Mersenne31, Mersenne31Complex<Mersenne31>, Radix2Cft, 16, 5>();
    }

    #[test]
    fn coset_lde_size_32() {
        coset_lde_test::<Mersenne31, Mersenne31Complex<Mersenne31>, Radix2Cft, 32, 3>();
    }
}
