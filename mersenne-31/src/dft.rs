//! Implementation of DFT for `Mersenne31`.
//!
//! Strategy follows: https://www.robinscheibler.org/2013/02/13/real-fft.html
//! In short, fold a Mersenne31 DFT of length n into a Mersenne31Complex DFT
//! of length n/2. Some pre/post-processing is necessary so that the result
//! of the transform behaves as expected wrt the convolution theorem etc.
//!
//! Note that we don't return the final n/2 - 1 elements since we know that
//! the "complex conjugate" of the (n-k)th element equals the kth element.
//! The convolution theorem maintains this relationship and so these final
//! n/2 - 1 elements are essentially redundant.

use alloc::vec::Vec;

use itertools::Itertools;
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{AbstractField, Field, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::{Matrix, MatrixRowSlices, MatrixRows};
use p3_util::log2_strict_usize;

use crate::{Mersenne31, Mersenne31Complex};

/// Given an hxw matrix M = (m_{ij}) where h is even, return an
/// (h/2)xw matrix N whose (k,l) entry is
///
///    Mersenne31Complex(m_{2k,l}, m_{2k+1,l})
///
/// i.e. the even rows become the real parts and the odd rows become
/// the imaginary parts.
///
/// This packing is suitable as input to a Fourier Transform over the
/// domain Mersenne31Complex; it is inverse to `idft_postprocess()`
/// below.
fn dft_preprocess(
    input: RowMajorMatrix<Mersenne31>,
) -> RowMajorMatrix<Mersenne31Complex<Mersenne31>> {
    assert!(input.height() % 2 == 0, "input height must be even");
    RowMajorMatrix::new(
        input
            .rows()
            .tuples()
            .flat_map(|(row_0, row_1)| {
                // For each pair of rows in input, convert each
                // two-element column into a Mersenne31Complex
                // treating the first row as the real part and the
                // second row as the imaginary part.
                row_0
                    .iter()
                    .zip(row_1)
                    .map(|(&x, &y)| Mersenne31Complex::new(x, y))
            })
            .collect(),
        input.width(),
    )
}

/// Transform the result of applying the DFT to the packed
/// `Mersenne31` values so that the convolution theorem holds.
///
/// Source: https://www.robinscheibler.org/2013/02/13/real-fft.html
///
/// NB: This function and `idft_preprocess()` are inverses.
fn dft_postprocess(
    input: RowMajorMatrix<Mersenne31Complex<Mersenne31>>,
) -> RowMajorMatrix<Mersenne31Complex<Mersenne31>> {
    let h = input.height();
    let log2_h = log2_strict_usize(h); // checks that h is a power of two

    // NB: The original real matrix had height 2h, hence log2(2h) = log2(h) + 1.
    // omega is a 2h-th root of unity
    let omega = Mersenne31Complex::primitive_root_of_unity(log2_h + 1);
    let mut omega_j = omega;

    let mut output = Vec::with_capacity((h + 1) * input.width());
    output.extend(
        input
            .first_row()
            .map(|x| Mersenne31Complex::new_real(x.real() + x.imag())),
    );

    for j in 1..h {
        let row_x = input.row_slice(j);
        let row_y = input.row_slice(h - j);

        let row = row_x.iter().zip(row_y).map(|(&x, y)| {
            let even = x + y.conjugate();
            // odd = (x - y.conjugate()) * -i
            let odd = Mersenne31Complex::new(x.imag() + y.imag(), y.real() - x.real());
            (even + odd * omega_j).div_2exp_u64(1)
        });
        output.extend(row);
        omega_j *= omega;
    }

    output.extend(
        input
            .first_row()
            .map(|x| Mersenne31Complex::new_real(x.real() - x.imag())),
    );
    debug_assert_eq!(output.len(), (h + 1) * input.width());
    RowMajorMatrix::new(output, input.width())
}

/// Undo the transform of the DFT matrix in `dft_postprocess()` so
/// that the inverse DFT can be applied.
///
/// Source: https://www.robinscheibler.org/2013/02/13/real-fft.html
///
/// NB: This function and `dft_postprocess()` are inverses.
fn idft_preprocess(
    input: RowMajorMatrix<Mersenne31Complex<Mersenne31>>,
) -> RowMajorMatrix<Mersenne31Complex<Mersenne31>> {
    let h = input.height() - 1;
    let log2_h = log2_strict_usize(h); // checks that h is a power of two

    // NB: The original real matrix had length 2h, hence log2(2h) = log2(h) + 1.
    // omega is a 2n-th root of unity
    let omega = Mersenne31Complex::primitive_root_of_unity(log2_h + 1).inverse();
    let mut omega_j = Mersenne31Complex::ONE;

    let mut output = Vec::with_capacity(h * input.width());
    // TODO: Specialise j = 0 and j = n (which we know must be real)?
    for j in 0..h {
        let row_x = input.row_slice(j);
        let row_y = input.row_slice(h - j);

        let row = row_x.iter().zip(row_y).map(|(&x, y)| {
            let even = x + y.conjugate();
            // odd = (x - y.conjugate()) * -i
            let odd = Mersenne31Complex::new(x.imag() + y.imag(), y.real() - x.real());
            (even - odd * omega_j).div_2exp_u64(1)
        });
        output.extend(row);
        omega_j *= omega;
    }
    RowMajorMatrix::new(output, input.width())
}

/// Given an (h/2)xw matrix M = (m_{kl}) = (a_{kl} + I*b_{kl}) (where
/// I is the imaginary unit), return the hxw matrix N whose (i,j)
/// entry is a_{i/2,j} if i is even and b_{(i-1)/2,j} if i is odd.
///
/// This function is inverse to `dft_preprocess()` above.
fn idft_postprocess(
    input: RowMajorMatrix<Mersenne31Complex<Mersenne31>>,
) -> RowMajorMatrix<Mersenne31> {
    // TODO: Re-write this without using `unzip()`, which needlessly
    // allocates two new temporary vectors while processing each row.
    RowMajorMatrix::new(
        input
            .rows()
            .flat_map(|row| {
                // Convert each row of input into two rows, the first row
                // having the real parts of the input, the second row
                // having the imaginary parts.
                let (reals, imags): (Vec<_>, Vec<_>) =
                    row.iter().map(|x| (x.real(), x.imag())).unzip();
                reals.into_iter().chain(imags)
            })
            .collect(),
        input.width(),
    )
}

/// The DFT for Mersenne31
#[derive(Default, Clone)]
pub struct Mersenne31Dft;

impl Mersenne31Dft {
    /// Compute the DFT of each column of `mat`.
    ///
    /// NB: The DFT works by packing pairs of `Mersenne31` values into
    /// a `Mersenne31Complex` and doing a (half-length) DFT on the
    /// result. In particular, the type of the result elements are in
    /// the extension field, not the domain field.
    pub fn dft_batch<Dft: TwoAdicSubgroupDft<Mersenne31Complex<Mersenne31>>>(
        mat: RowMajorMatrix<Mersenne31>,
    ) -> RowMajorMatrix<Mersenne31Complex<Mersenne31>> {
        let dft = Dft::default();
        dft_postprocess(dft.dft_batch(dft_preprocess(mat)))
    }

    /// Compute the inverse DFT of each column of `mat`.
    ///
    /// NB: See comment on `dft_batch()` for information on packing.
    pub fn idft_batch<Dft: TwoAdicSubgroupDft<Mersenne31Complex<Mersenne31>>>(
        mat: RowMajorMatrix<Mersenne31Complex<Mersenne31>>,
    ) -> RowMajorMatrix<Mersenne31> {
        let dft = Dft::default();
        idft_postprocess(dft.idft_batch(idft_preprocess(mat)))
    }
}

#[cfg(test)]
mod tests {
    use p3_dft::Radix2Dit;
    use rand::distributions::{Distribution, Standard};
    use rand::{thread_rng, Rng};

    use super::*;
    use crate::Mersenne31;

    #[test]
    fn consistency()
    where
        Standard: Distribution<Mersenne31>,
    {
        const N: usize = 1 << 12;
        let input = thread_rng()
            .sample_iter(Standard)
            .take(N)
            .collect::<Vec<Mersenne31>>();
        let input = RowMajorMatrix::new_col(input);
        let fft_input = Mersenne31Dft::dft_batch::<Radix2Dit>(input.clone());
        let output = Mersenne31Dft::idft_batch::<Radix2Dit>(fft_input);
        assert_eq!(input, output);
    }

    #[test]
    fn convolution()
    where
        Standard: Distribution<Mersenne31>,
    {
        const N: usize = 1 << 12;
        let a = thread_rng()
            .sample_iter(Standard)
            .take(N)
            .collect::<Vec<Mersenne31>>();
        let a = RowMajorMatrix::new_col(a);
        let b = thread_rng()
            .sample_iter(Standard)
            .take(N)
            .collect::<Vec<Mersenne31>>();
        let b = RowMajorMatrix::new_col(b);

        let fft_a = Mersenne31Dft::dft_batch::<Radix2Dit>(a.clone());
        let fft_b = Mersenne31Dft::dft_batch::<Radix2Dit>(b.clone());

        let fft_c = fft_a
            .values
            .iter()
            .zip(fft_b.values.iter())
            .map(|(&xi, &yi)| xi * yi)
            .collect();
        let fft_c = RowMajorMatrix::new_col(fft_c);

        let c = Mersenne31Dft::idft_batch::<Radix2Dit>(fft_c);

        let mut conv = Vec::with_capacity(N);
        for i in 0..N {
            let mut t = Mersenne31::ZERO;
            for j in 0..N {
                t += a.values[j] * b.values[(N + i - j) % N];
            }
            conv.push(t);
        }

        assert_eq!(c.values, conv);
    }
}
