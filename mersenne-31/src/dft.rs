use crate::{Mersenne31, Mersenne31Complex};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{AbstractField, Field, TwoAdicField};
use p3_matrix::{dense::RowMajorMatrix, Matrix, MatrixRowSlices, MatrixRows};
use p3_util::log2_strict_usize;

use alloc::vec::Vec;
use itertools::Itertools;

/// Given a vector u = (u_0, ..., u_{n-1}), where n is even, return a
/// vector U = (U_0, ..., U_{N/2 - 1}) whose jth entry is
/// Mersenne31Complex(u_{2j}, u_{2j + 1}); i.e. the even elements
/// become the real parts and the odd elements become the imaginary
/// parts.
///
/// This packing is suitable as input to a Fourier Transform over the
/// domain Mersenne31Complex.
fn dft_preprocess(
    input: RowMajorMatrix<Mersenne31>,
) -> RowMajorMatrix<Mersenne31Complex<Mersenne31>> {
    assert!(input.height() % 2 == 0, "input height must be even");
    RowMajorMatrix::new(
        input
            .rows()
            .tuples()
            .map(|(row_0, row_1)| {
                // For each pair of rows in input, convert each
                // two-element column into a Mersenne31Complex
                // treating the first row as the real part and the
                // second row as the imaginary part.
                row_0
                    .iter()
                    .zip(row_1)
                    .map(|(&x, &y)| Mersenne31Complex::new(x, y))
            })
            .flatten()
            .collect(),
        input.width(),
    )
}

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
            let odd = Mersenne31Complex::new(x.imag() + y.imag(), y.real() - x.real());
            (even - odd * omega_j).div_2exp_u64(1)
        });
        output.extend(row);
        omega_j *= omega;
    }
    RowMajorMatrix::new(output, input.width())
}

fn idft_postprocess(
    input: RowMajorMatrix<Mersenne31Complex<Mersenne31>>,
) -> RowMajorMatrix<Mersenne31> {
    RowMajorMatrix::new(
        input
            .rows()
            .map(|row| {
                // Convert each row of input into two rows, the first row
                // having the real parts of the input, the second row
                // having the imaginary parts.
                let (reals, imags): (Vec<_>, Vec<_>) =
                    row.iter().map(|x| (x.real(), x.imag())).unzip();
                reals.into_iter().chain(imags.into_iter())
            })
            .flatten()
            .collect(),
        input.width(),
    )
}

/// The DFT for Mersenne31
#[derive(Default, Clone)]
pub struct Mersenne31Dft;

impl Mersenne31Dft {
    pub fn dft_batch<Dft: TwoAdicSubgroupDft<Mersenne31Complex<Mersenne31>>>(
        mat: RowMajorMatrix<Mersenne31>,
    ) -> RowMajorMatrix<Mersenne31Complex<Mersenne31>> {
        let dft = Dft::default();
        dft_postprocess(dft.dft_batch(dft_preprocess(mat)))
    }

    pub fn idft_batch<Dft: TwoAdicSubgroupDft<Mersenne31Complex<Mersenne31>>>(
        mat: RowMajorMatrix<Mersenne31Complex<Mersenne31>>,
    ) -> RowMajorMatrix<Mersenne31> {
        let dft = Dft::default();
        idft_postprocess(dft.idft_batch(idft_preprocess(mat)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Mersenne31;
    use p3_dft::Radix2Dit;
    use rand::distributions::{Distribution, Standard};
    use rand::{thread_rng, Rng};

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
