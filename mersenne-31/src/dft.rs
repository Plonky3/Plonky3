use crate::{Mersenne31, Mersenne31Complex};
use p3_field::{AbstractField, Field, TwoAdicField};
use p3_util::log2_strict_usize;

use alloc::vec::Vec;

// TODO: Redo this in terms of RowMajorMatrix{,View{,Mut}}

/// Given a vector u = (u_0, ..., u_{n-1}), where n is even, return a
/// vector U = (U_0, ..., U_{N/2 - 1}) whose jth entry is
/// Mersenne31Complex(u_{2j}, u_{2j + 1}); i.e. the even elements
/// become the real parts and the odd elements become the imaginary
/// parts.
///
/// This packing is suitable as input to a Fourier Transform over the
/// domain Mersenne31Complex.
pub fn dft_preprocess(input: Vec<Mersenne31>) -> Vec<Mersenne31Complex<Mersenne31>> {
    assert!(input.len() % 2 == 0, "input vector length must be even");
    input
        .chunks_exact(2)
        .map(|pair| Mersenne31Complex::new(pair[0], pair[1]))
        .collect()
}

pub fn dft_postprocess(
    input: Vec<Mersenne31Complex<Mersenne31>>,
) -> Vec<Mersenne31Complex<Mersenne31>> {
    let n = input.len();
    let log2_n = log2_strict_usize(n); // checks that n is a power of two

    // NB: The original vector was length 2n, hence log2(2n) = log2(n) + 1.
    // omega is a 2n-th root of unity
    let omega = Mersenne31Complex::primitive_root_of_unity(log2_n + 1);
    let mut omega_j = omega;

    let mut output = Vec::with_capacity(n + 1);
    output.push(Mersenne31Complex::new_real(
        input[0].real() + input[0].imag(),
    ));
    for j in 1..n {
        let even = input[j] + input[n - j].conjugate();
        let odd = input[j] - input[n - j].conjugate();
        // TODO: pull apart components and integrate the multiplication-by-i
        let odd = Mersenne31Complex::new(odd.imag(), -odd.real()); // odd *= -i
        output.push((even + odd * omega_j).div_2exp_u64(1));
        omega_j *= omega;
    }
    output.push(Mersenne31Complex::new_real(
        input[0].real() - input[0].imag(),
    ));
    output
}

pub fn idft_preprocess(
    input: Vec<Mersenne31Complex<Mersenne31>>,
) -> Vec<Mersenne31Complex<Mersenne31>> {
    let n = input.len() - 1;
    let log2_n = log2_strict_usize(n); // checks that n is a power of two

    // NB: The original vector was length 2n, hence log2(2n) = log2(n) + 1.
    // omega is a 2n-th root of unity
    let omega = Mersenne31Complex::primitive_root_of_unity(log2_n + 1).inverse();
    let mut omega_j = Mersenne31Complex::ONE;

    let mut output = Vec::with_capacity(n);
    // TODO: Specialise j = 0 and j = n (which we know must be real)?
    for j in 0..n {
        let even = input[j] + input[n - j].conjugate();
        let odd = input[j] - input[n - j].conjugate();
        // TODO: pull apart components and integrate the multiplication-by-i
        let odd = Mersenne31Complex::new(-odd.imag(), odd.real());
        output.push((even + odd * omega_j).div_2exp_u64(1));
        omega_j *= omega;
    }
    output
}

pub fn idft_postprocess(input: Vec<Mersenne31Complex<Mersenne31>>) -> Vec<Mersenne31> {
    // TODO: The memory layout of input and output are identical; find a way
    // to just do a memcpy.
    let mut output = Vec::with_capacity(2 * input.len());
    for x in input {
        output.push(x.real());
        output.push(x.imag());
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Mersenne31;
    use p3_dft::{NaiveDft, Radix2Dit, TwoAdicSubgroupDft};
    use rand::distributions::{Distribution, Standard};
    use rand::{thread_rng, Rng};

    #[test]
    fn consistency()
    where
        Standard: Distribution<Mersenne31>,
    {
        const N: usize = 1 << 12;
        let a = thread_rng()
            .sample_iter(Standard)
            .take(32)
            .collect::<Vec<Mersenne31>>();
        let b = dft_preprocess(a.clone());
        let c = Radix2Dit.dft(b.clone());
        let d = dft_postprocess(c);
        let e = idft_preprocess(d);
        let f = Radix2Dit.idft(e);
        let g = idft_postprocess(f);

        assert!(a == g);
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
        let b = thread_rng()
            .sample_iter(Standard)
            .take(N)
            .collect::<Vec<Mersenne31>>();

        let fft_a = dft_preprocess(a.clone());
        let fft_a = Radix2Dit.dft(fft_a);
        let fft_a = dft_postprocess(fft_a);

        let fft_b = dft_preprocess(b.clone());
        let fft_b = Radix2Dit.dft(fft_b);
        let fft_b = dft_postprocess(fft_b);

        let fft_c = fft_a
            .iter()
            .zip(fft_b.iter())
            .map(|(&xi, &yi)| xi * yi)
            .collect();
        let c = idft_preprocess(fft_c);
        let c = Radix2Dit.idft(c);
        let c = idft_postprocess(c);

        let mut conv = Vec::with_capacity(N);
        for i in 0..N {
            let mut t = Mersenne31::ZERO;
            for j in 0..N {
                t += a[j] * b[(N + i - j) % N];
            }
            conv.push(t);
        }

        assert!(c == conv);
    }
}
