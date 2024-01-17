// A place to store some old/naive implementations we keep around for testing purposes.
// Plan is to eventually remove this before merging the branch and move some of these to testing.rs

use alloc::vec;
use alloc::vec::Vec;
use itertools::Itertools;

use p3_field::{AbstractField, ComplexExtension, Field};

/// Get the cfft polynomial basis.
/// The basis consists off all multi-linear products of: y, x, 2x^2 - 1, 2(2x^2 - 1)^2 - 1, ...
/// The ordering of these basis elements is the bit reversal of the sequence: 1, y, x, xy, (2x^2 - 1), (2x^2 - 1)y, ...
/// We also need to throw in a couple of negative signs for technical reasons.
pub fn cfft_poly_basis<Base: AbstractField + Field, Ext: ComplexExtension<Base>>(
    point: &Ext,
    n: u32,
) -> Vec<Base> {
    if n == 0 {
        return vec![Base::one()]; // Base case
    }

    let mut output = vec![Base::one(), point.imag()]; // The n = 1 case is also special as y only appears once.

    let mut current = point.real();

    for _ in 1..n {
        let new = output.clone().into_iter().map(|val| val * current); // New basis elements to add.

        output = output.into_iter().interleave(new).collect(); // Interleave the two basis together to keep the bit reversal ordering.

        current = (Base::two()) * current * current - Base::one();
        // Find the next basis vector.
    }

    // We need to handle the negatives which can appear in our cFFT method.
    // For the i'th basis element, we multiply it by -1 for every occurance of 11 in the binary decomposition of i.
    // There is almost certainly a better way to do this but this code is only here for cross checks and won't be used in production.
    for (i, val) in output.iter_mut().enumerate() {
        let mut last = false;
        for j in 0..n {
            let test_bit = 1 << j;
            let non_zero_test = i & test_bit != 0;
            if non_zero_test && last {
                *val *= -Base::one();
            }
            last = non_zero_test;
        }
    }

    output
}

/// Evaluate a polynomial with coefficents given in the CFFT basis at a point (x, y)
/// len(coeffs) needs to be a power of 2.
/// Gives a simple O(n^2) equivalent to check our CFFT against.
pub fn evaluate_cfft_poly<Base: AbstractField + Field, Ext: ComplexExtension<Base>>(
    coeffs: &[Base],
    point: Ext,
) -> Base {
    let n = coeffs.len();

    debug_assert!(n.is_power_of_two()); // If n is not a power of 2 something has gone badly wrong.

    let log_n = n.trailing_zeros();

    let basis = cfft_poly_basis(&point, log_n); // Get the cfft polynomial basis evaluated at the point x.

    let mut output = Base::zero();

    for i in 0..n {
        output += coeffs[i] * basis[i] // Dot product the basis with the coefficients.
    }

    output
}

// Finally we move on to defining the CFFT, Inverse CFFT and Coset Extrapolation algorithms.
// Handily, the Inverse CFFT and Coset Extrapolation algorithms are actually identical, we just need to change our choice of twiddles.

/// Given a vector v of length N = 2^n, we consider this vector as the evaluations of a polynomial
/// f \in F_P[X, Y]/<X^2 + Y^2 = 1> on the circle coset gH = {g, g^3, ..., g^{2N - 1}}.
/// Here H is the unique subgroup of order N and g is an element of order 2N.
/// We can assume that deg_X(f) < N/2 and deg_Y(f) <= 1.
/// Then cfft computes the coefficients of f (In a slightly unusual basis).
pub fn cfft<Base: Field>(coeffs: &mut [Base], twiddles: &[Vec<Base>]) {
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
                let s = chunck[j];
                let t = chunck[block_size - j - 1];
                chunck[j] = s + t;
                chunck[block_size - j - 1] = (s - t) * twiddle[j];
            }
        }
    }
}

/// This is the inverse of the CFFT. Given the coefficients it computes the evaluations.
/// Depending on the twiddles input, this can also compute the coset evaluation as the algorithms are otherwise identical.
pub fn cfft_inv<Base: Field>(coeffs: &mut [Base], twiddles: &[Vec<Base>]) {
    let n = coeffs.len();
    let n_u32: u32 = n.try_into().unwrap();
    let log_n = n.trailing_zeros().try_into().unwrap();
    debug_assert_eq!(1_u32 << log_n, n_u32); // Our input better be a power of 2.

    for i in 0..log_n {
        let block_size = 1 << (i + 1);
        let half_block_size = block_size >> 1;

        for chunck in coeffs.chunks_mut(block_size) {
            for j in 0..half_block_size {
                let s = chunck[j];
                let t = chunck[block_size - j - 1] * twiddles[log_n - i - 1][j];

                chunck[j] = s + t;
                chunck[block_size - j - 1] = s - t;
            }
            // Would need this if we don't start with the assumption that our input is shuffled in the right way.
            // chunck[half_block_size..].reverse();
        }
    }
}
