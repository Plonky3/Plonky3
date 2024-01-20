use alloc::vec;
use alloc::vec::Vec;

use itertools::Itertools;
use p3_field::extension::{Complex, ComplexExtendable};
use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_util::log2_strict_usize;
use rand::distributions::{Distribution, Standard};
use rand::Rng;

use crate::util::{cfft_domain, twin_coset_domain};
use crate::CircleSubgroupFt;

/// Unlike the standard DFT where both directions can be reinterpreted as polynomial evaluation,
/// In the CFFT only the iCFFT naturally corresponds to such an evaluation.
/// Thus instead of writing a "Naive" CFFT, we just give the polynomial evaluation and some auxillary functions.

// Ensure that if we perform a Cfft to get coefficients and then evaluate those same coefficients we get back what we started with.
pub(crate) fn fft_test<F, Cfft, const N: usize>()
where
    F: ComplexExtendable,
    Standard: Distribution<F>,
    Cfft: CircleSubgroupFt<F>,
{
    let mut rng = rand::thread_rng();
    let values: [F; N] = core::array::from_fn(|_| rng.gen::<F>());
    let expected = values;
    let size_u32 = N as u32;
    let log_size = log2_strict_usize(N);

    let cfft_fn = Cfft::default();
    let cfft_coeffs = cfft_fn.cfft(values.to_vec());

    let points = cfft_domain::<F>(log_size, N);

    let evals: Vec<_> = points
        .into_iter()
        .map(|point| {
            F::from_canonical_u32(size_u32).inverse() * evaluate_cfft_poly(&cfft_coeffs, point)
        })
        .collect();

    assert_eq!(evals, expected)
}

pub(crate) fn ifft_test<F, Cfft, const N: usize>()
where
    F: ComplexExtendable,
    Standard: Distribution<F>,
    Cfft: CircleSubgroupFt<F>,
{
    let mut rng = rand::thread_rng();
    let values: [F; N] = core::array::from_fn(|_| rng.gen::<F>());
    let log_size = log2_strict_usize(N);

    let cfft_fn = Cfft::default();
    let cfft_evals = cfft_fn.icfft(values.to_vec());

    let points = cfft_domain::<F>(log_size, N);
    let evals: Vec<_> = points
        .into_iter()
        .map(|point| evaluate_cfft_poly(&values, point))
        .collect();

    assert_eq!(cfft_evals, evals);
}

pub(crate) fn coset_eval_test<F, Cfft, const N: usize>()
where
    F: ComplexExtendable,
    Standard: Distribution<F>,
    Cfft: CircleSubgroupFt<F>,
{
    let mut rng = rand::thread_rng();
    let values: [F; N] = core::array::from_fn(|_| rng.gen::<F>());

    // The following chooses a uniform random element from S^1.
    let rng_elem = rng.gen::<Complex<F>>();
    let coset_elem = rng_elem * rng_elem * rng_elem.norm().inverse();
    assert!(coset_elem.norm().is_one());

    let log_size = log2_strict_usize(N);

    let cfft_fn = Cfft::default();
    let cfft_evals = cfft_fn.coset_icfft(values.to_vec(), coset_elem);

    let points = twin_coset_domain::<F>(F::circle_two_adic_generator(log_size - 1), coset_elem, N);
    let evals: Vec<_> = points
        .into_iter()
        .map(|point| evaluate_cfft_poly(&values, point))
        .collect();

    assert_eq!(cfft_evals, evals);
}

// Test that the cfft and icfft are inverses.
pub(crate) fn fft_ifft_test<F, Cfft, const N: usize, const BATCH_SIZE: usize>()
where
    F: ComplexExtendable,
    Standard: Distribution<F>,
    Cfft: CircleSubgroupFt<F>,
{
    let mut rng = rand::thread_rng();

    let values = RowMajorMatrix::<F>::rand(&mut rng, N, BATCH_SIZE);

    let cfft_fn = Cfft::default();

    let cfft_coeffs = cfft_fn.cfft_batch(values.clone());
    let mut cfft_evals = cfft_fn.icfft_batch(cfft_coeffs);

    divide_by_height(&mut cfft_evals);

    assert_eq!(values, cfft_evals);
}

// Check that doing the lde extension is the same as the cfft followed by coset_icfft
pub(crate) fn coset_lde_test<F, Cfft, const N: usize, const BATCH_SIZE: usize>()
where
    F: ComplexExtendable,
    Standard: Distribution<F>,
    Cfft: CircleSubgroupFt<F>,
{
    let mut rng = rand::thread_rng();

    // The following chooses a uniform random element from S^1.
    let rng_elem = rng.gen::<Complex<F>>();
    let coset_elem = rng_elem * rng_elem * rng_elem.norm().inverse();
    assert!(coset_elem.norm().is_one());

    let values = RowMajorMatrix::<F>::rand(&mut rng, N, BATCH_SIZE);

    let cfft_fn = Cfft::default();

    let cfft_evals = cfft_fn.coset_icfft_batch(cfft_fn.cfft_batch(values.clone()), coset_elem);
    let cfft_lde_evals = cfft_fn.coset_lde_batch(values, coset_elem);

    assert_eq!(cfft_evals, cfft_lde_evals);
}

///Divide each coefficient of the given matrix by its height.
fn divide_by_height<F: Field>(mat: &mut RowMajorMatrix<F>) {
    let h = mat.height();
    let h_inv = F::from_canonical_usize(h).inverse();
    let (prefix, shorts, suffix) = unsafe { mat.values.align_to_mut::<F::Packing>() };
    prefix.iter_mut().for_each(|x| *x *= h_inv);
    shorts.iter_mut().for_each(|x| *x *= h_inv);
    suffix.iter_mut().for_each(|x| *x *= h_inv);
}

/// Get the cfft polynomial basis.
/// The basis consists off all multi-linear products of: y, x, 2x^2 - 1, 2(2x^2 - 1)^2 - 1, ...
/// The ordering of these basis elements is the bit reversal of the sequence: 1, y, x, xy, (2x^2 - 1), (2x^2 - 1)y, ...
/// We also need to throw in a couple of negative signs for technical reasons.
pub fn cfft_poly_basis<F: ComplexExtendable>(point: &Complex<F>, n: usize) -> Vec<F> {
    if n == 0 {
        return vec![F::one()]; // Base case
    }

    let mut output = vec![F::one(), point.imag()]; // The n = 1 case is also special as y only appears once.

    let mut current = point.real();

    for _ in 1..n {
        let new = output.clone().into_iter().map(|val| val * current); // New basis elements to add.

        output = output.into_iter().interleave(new).collect(); // Interleave the two basis together to keep the bit reversal ordering.

        current = (F::two()) * current * current - F::one();
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
                *val *= -F::one();
            }
            last = non_zero_test;
        }
    }

    output
}

/// Evaluate a polynomial with coefficents given in the CFFT basis at a point (x, y)
/// len(coeffs) needs to be a power of 2.
/// Gives a simple O(n^2) equivalent to check our CFFT against.
fn evaluate_cfft_poly<F: ComplexExtendable>(coeffs: &[F], point: Complex<F>) -> F {
    coeffs
        .iter()
        .zip(cfft_poly_basis(&point, log2_strict_usize(coeffs.len())))
        .map(|(&coeff, basis)| coeff * basis)
        .sum()
}
