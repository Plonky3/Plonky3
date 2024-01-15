use itertools::Itertools;
use p3_field::{Field, AbstractField, AbstractExtensionField, TwoAdicField, Powers};
use p3_mersenne_31::{Mersenne31, Mersenne31Complex};

use core::fmt::Debug;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// A point in F^2. When F = M31 this is literally the point x + iy in M31[i] but
// we distingush this as we might want to let F be something more general.
#[derive(Debug)]
pub struct Point<F> {
    x: F,
    y: F,
}

// Next we define some helper functions.

pub fn from_complex(val: Mersenne31Complex<Mersenne31>) -> Point<Mersenne31> {
    Point {
        x: val.real(),
        y: val.imag(),
    }
}

/// Get the cfft polynomial basis.
/// The basis consists off all multi-linear products of: y, x, 2x^2 - 1, 2(2x^2 - 1)^2 - 1, ...
/// The ordering of these basis elements is the bit reversal of the sequence: 1, y, x, xy, (2x^2 - 1), (2x^2 - 1)y, ...
/// We also need to throw in a couple of negative signs for technical reasons.
fn cfft_poly_basis<F: AbstractField>(point: &Point<F>, n: u32) -> Vec<F> {
    if n == 0 {
        return vec![F::one()]; // Base case
    }

    let mut output = vec![F::one(), point.y.clone()]; // The n = 1 case is also special as y only appears once.

    let mut current = point.x.clone();

    for _ in 1..n {
        let new = output.clone().into_iter().map(|val| val * current.clone()); // New basis elements to add.

        output = output.into_iter().interleave(new).collect(); // Interleave the two basis together to keep the bit reversal ordering.

        current = (F::two()) * current.clone() * current.clone() - F::one(); // Find the next basis vector.
    }

    // We need to handle the negatives which can appear in our cFFT method.
    // For the i'th basis element, we multiply it by -1 for every occurance of 11 in the binary decomposition of i.
    // There is almost certainly a better way to do this but this code is only here for cross checks and won't be used in production.
    for i in 0..output.len() {
        let mut last = false;
        for j in 0..n {
            let test_bit = 1 << j;
            let non_zero_test = i & test_bit != 0;
            if non_zero_test && last {
                output[i] = -output[i].clone();
            }
            last = non_zero_test;
        }
    }

    output
}

/// Evaluate a polynomial with coefficents given in the CFFT basis at a point (x, y)
/// len(coeffs) needs to be a power of 2.
/// Gives a simple O(n^2) equivalent to check our CFFT against.
pub fn evaluate_cfft_poly<F: AbstractField>(coeffs: &[F], point: Point<F>) -> F {
    let n = coeffs.len();

    debug_assert!(n.is_power_of_two()); // If n is not a power of 2 something has gone badly wrong.

    let log_n = n.trailing_zeros();

    let basis = cfft_poly_basis(&point, log_n); // Get the cfft polynomial basis evaluated at the point x.

    let mut output = F::zero();

    for i in 0..n {
        output += coeffs[i].clone() * basis[i].clone() // Dot product the basis with the coefficients.
    }

    output
}

/// Given an integer bits, generate half of the points in the coset gH.
/// Here H is the unique subgroup of order 2^bits and g is an element of order 2^{bits + 1}.
/// The output will be a list {g, g^3, g^5, ...} of length size.
/// Use size = 2^{bits} for the full domain or 2^{bits - 1} for the half domain.
#[inline]
fn cfft_domain(bits: usize, size: usize) -> Vec<Mersenne31Complex<Mersenne31>> {
    let generator = Mersenne31Complex::two_adic_generator(bits + 1);

    let powers = Powers {
        base: generator * generator,
        current: generator,
    };

    powers.take(size).collect()
}

/// Given a generator h for H and an element k, generate points in the twin coset kH u k^{-1}H.
/// The ordering is important here, the points will generated in the following interleaved pattern:
/// {k, k^{-1}h, kh, k^{-1}h^2, kh^2, ..., k^{-1}h^{-1}, kh^{-1}, k^{-1}}.
/// Size controls how many of these we want to compute. It should either be |H| or |H|/2 depending on if
/// we want simply the twiddles or the full domain. If k is has order 2|H| this is equal to cfft_domain.
#[inline]
fn twin_coset_domain(
    generator: Mersenne31Complex<Mersenne31>,
    coset_elem: Mersenne31Complex<Mersenne31>,
    size: usize,
) -> Vec<Mersenne31Complex<Mersenne31>> {
    let coset_powers = Powers {
        base: generator,
        current: coset_elem,
    };

    let inv_coset_powers = Powers {
        base: generator,
        current: generator * coset_elem.inverse(),
    };

    coset_powers
        .interleave(inv_coset_powers)
        .take(size)
        .collect()
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Finally we move on to definint the CFFT, Inverse CFFT and Coset Extrapolation algorithms.
// Handily, the Inverse CFFT and Coset Extrapolation algorithms are actually identical, we just need to change our choice of twiddles.

/// Compute the twiddles for the CFFT.
/// Let N = 2^n be the size of our initial set. Then we start with the domain
/// {g, g^3, ..., g^{-3}, g^{-1}} for g a 2N'th root of unity.
/// The initial twiddle domain is the first half of the full domain.
/// In the first step our twiddles are the inverse imaginary parts and we simply halve the domain size.
/// In all subsequent steps our twiddles are the inverse real parts and we both halve the domain size and square every element.
pub fn cfft_twiddles(log_n: usize) -> Vec<Vec<Mersenne31>> {
    let size = 1 << (log_n - 1);
    let init_domain = cfft_domain(log_n, size); // Get the starting domain.

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
                    .map(|x| Mersenne31::two() * *x * *x - Mersenne31::one())
                    .collect(); // When we square a point, the real part changes as x -> 2x^2 - 1.
                output
            }
        })
        .collect()
}

/// Given a vector v of length N = 2^n, we consider this vector as the evaluations of a polynomial
/// f \in F_P[X, Y]/<X^2 + Y^2 = 1> on the circle coset gH = {g, g^3, ..., g^{2N - 1}}.
/// Here H is the unique subgroup of order N and g is an element of order 2N.
/// We can assume that deg_X(f) < N/2 and deg_Y(f) <= 1.
/// Then cfft computes the coefficients of f (In a slightly unusual basis).
pub fn cfft<F: AbstractExtensionField<Mersenne31>>(
    coeffs: &mut [F],
    twiddles: &[Vec<Mersenne31>],
) {
    let n = coeffs.len();
    let n_u32: u32 = n.try_into().unwrap();
    let log_n = n.trailing_zeros().try_into().unwrap();
    debug_assert_eq!(1_u32 << log_n, n_u32); // Our input better be a power of 2.

    for i in 0..log_n {
        let block_size = 1 << (log_n - i);
        let half_block_size = block_size >> 1;

        for chunck in coeffs.chunks_mut(block_size) {
            // Need this if we didn't change the twiddles.
            // chunck[half_block_size..].reverse();
            for j in 0..half_block_size {
                let s = chunck[j].clone();
                let t = chunck[block_size - j - 1].clone();
                chunck[j] = s.clone() + t.clone();
                chunck[block_size - j - 1] = (s - t) * twiddles[i][j];
            }
        }
    }
}

/// Compute the twiddles for the inverse CFFT.
/// The twiddles are essentially the same as in the CFFT case except we no longer need to take inverses.
pub fn cfft_inv_twiddles(log_n: usize) -> Vec<Vec<Mersenne31>> {
    let size = 1 << (log_n - 1);
    let init_domain = cfft_domain(log_n, size);

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
                    .map(|x| Mersenne31::two() * *x * *x - Mersenne31::one())
                    .collect();
                output
            }
        })
        .collect()
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

/// Compute the twiddles for the coset evaluation.
/// Unlike the previous cases, here we actually need to start with a given group element.
/// TODO: Explain what these twiddles are.
pub fn coset_eval_twiddles(log_n: usize, coset_elem: Mersenne31Complex<Mersenne31>) -> Vec<Vec<Mersenne31>> {
    let size = 1 << (log_n - 1);
    let generator = Mersenne31Complex::two_adic_generator(log_n - 1);

    let init_domain = twin_coset_domain(generator, coset_elem, size);

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
                    .map(|x| Mersenne31::two() * *x * *x - Mersenne31::one())
                    .collect();
                output
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use p3_field::TwoAdicField;
    

    #[test]
    fn fft_size_2() {
        let mut a = [(1 << 31) - 3, 2].map(Mersenne31::from_canonical_u32);
        let twiddles = cfft_twiddles(1);

        cfft(&mut a, &twiddles);

        assert_eq!(a, [0, 4].map(Mersenne31::from_canonical_u32));
    }

    #[test]
    fn fft_size_4() {
        let mut a = [1, 11, 5, 1234].map(Mersenne31::from_canonical_u32);
        let size: usize = a.len();
        let size_u32 = size as u32;
        let log_size = size.trailing_zeros() as usize;

        let twiddles = cfft_twiddles(log_size);

        cfft(&mut a, &twiddles);

        let points = cfft_domain(log_size, size);

        let evals: Vec<_> = points
            .into_iter()
            .map(|point| {
                Mersenne31::from_canonical_u32(size_u32).inverse() * evaluate_cfft_poly(&a, from_complex(point))
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

        let twiddles = cfft_twiddles(log_size);

        cfft(&mut a, &twiddles);

        let points = cfft_domain(log_size, size);

        let evals: Vec<_> = points
            .into_iter()
            .map(|point| {
                Mersenne31::from_canonical_u32(size_u32).inverse() * evaluate_cfft_poly(&a, from_complex(point))
            })
            .collect();

        assert_eq!(evals, [1, 2, 1235, 4, 9, 11, 17, 6].map(Mersenne31::from_canonical_u32))
    }

    #[test]
    fn fft_size_16() {
        let mut a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16].map(Mersenne31::from_canonical_u32);
        let size: usize = a.len();
        let size_u32 = size as u32;
        let log_size = size.trailing_zeros() as usize;
        let twiddles = cfft_twiddles(log_size);

        let expected = a.clone();

        cfft(&mut a, &twiddles);

        let points = cfft_domain(log_size, size);

        let evals: Vec<_> = points
            .into_iter()
            .map(|point| {
                Mersenne31::from_canonical_u32(size_u32).inverse() * evaluate_cfft_poly(&a, from_complex(point))
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
        let cfft_twiddles = cfft_twiddles(log_size);

        let expected: Vec<_> = a.iter().map(|val| Mersenne31::from_canonical_u32(size) * *val).collect();
        cfft(&mut a, &cfft_twiddles);

        let cfft_inv_twiddles = cfft_inv_twiddles(log_size);
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

        let cfft_twiddles = cfft_twiddles(log_size);

        // shuffle(&mut a);
        cfft(&mut a, &cfft_twiddles);

        let coset_elem = Mersenne31Complex::two_adic_generator(10);
        let group_generator = Mersenne31Complex::two_adic_generator(log_size - 1);

        let coset_points = twin_coset_domain(group_generator, coset_elem, size);

        let coset_evals: Vec<_> = coset_points
            .into_iter()
            .map(|point| evaluate_cfft_poly(&a, from_complex(point)))
            .collect();

        let coset_twiddles = coset_eval_twiddles(log_size, coset_elem.clone());
        cfft_inv(&mut a, &coset_twiddles);

        assert_eq!(coset_evals, a)
    }
}
