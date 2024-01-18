use alloc::vec::Vec;

use p3_field::{ComplexExtension, Field};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;

use rand::distributions::{Distribution, Standard};
use rand::Rng;

use crate::util::{cfft_domain, twin_coset_domain};
use crate::{cfft, cfft_twiddles, evaluate_cfft_poly, CircleSubgroupFFT};

/// Unlike the standard DFT where both directions can be reinterpreted as polynomial evaluation,
/// In the CFFT only the iCFFT naturally corresponds to such an evaluation.
/// Thus instead of writing a "Naive" CFFT, we just give the polynomial evaluation and some auxillary functions.

// Test that on input a single vector, the Cfft algorithm output is the same as the old Cfft algorithm saved here.
// Additionally ensure that if we perform a Cfft to get coefficients and then evaluate those same coefficients we get back what we started with.
pub(crate) fn fft_test<Base, Ext, Cfft, const N: usize>()
where
    Base: Field,
    Standard: Distribution<Base>,
    Ext: ComplexExtension<Base>,
    Cfft: CircleSubgroupFFT<Base, Ext>,
{
    let mut rng = rand::thread_rng();
    let mut values: [Base; N] = core::array::from_fn(|_| rng.gen::<Base>());
    let expected = values;
    let size_u32 = N as u32;
    let log_size = N.trailing_zeros() as usize;
    let twiddles = cfft_twiddles::<Base, Ext>(log_size);

    let cfft_fn = Cfft::default();
    let cfft_coeffs = cfft_fn.cfft(values.to_vec());

    cfft(&mut values, &twiddles);

    assert_eq!(cfft_coeffs, values);

    let points = cfft_domain::<Base, Ext>(log_size, N);

    let evals: Vec<_> = points
        .into_iter()
        .map(|point| {
            Base::from_canonical_u32(size_u32).inverse() * evaluate_cfft_poly(&cfft_coeffs, point)
        })
        .collect();

    assert_eq!(evals, expected)
}

pub(crate) fn ifft_test<Base, Ext, Cfft, const N: usize>()
where
    Base: Field,
    Standard: Distribution<Base>,
    Ext: ComplexExtension<Base>,
    Cfft: CircleSubgroupFFT<Base, Ext>,
{
    let mut rng = rand::thread_rng();
    let values: [Base; N] = core::array::from_fn(|_| rng.gen::<Base>());
    let log_size = N.trailing_zeros() as usize;

    let cfft_fn = Cfft::default();
    let cfft_evals = cfft_fn.icfft(values.to_vec());

    let points = cfft_domain::<Base, Ext>(log_size, N);
    let evals: Vec<_> = points
        .into_iter()
        .map(|point| evaluate_cfft_poly(&values, point))
        .collect();

    assert_eq!(cfft_evals, evals);
}

pub(crate) fn coset_eval_test<Base, Ext, Cfft, const N: usize>()
where
    Base: Field,
    Standard: Distribution<Base>,
    Ext: ComplexExtension<Base>,
    Cfft: CircleSubgroupFFT<Base, Ext>,
{
    let mut rng = rand::thread_rng();
    let values: [Base; N] = core::array::from_fn(|_| rng.gen::<Base>());
    let coset_elem = Ext::circle_two_adic_generator(20);
    let log_size = N.trailing_zeros() as usize;

    let cfft_fn = Cfft::default();
    let cfft_evals = cfft_fn.coset_icfft(values.to_vec(), coset_elem);

    let points =
        twin_coset_domain::<Base, Ext>(Ext::circle_two_adic_generator(log_size - 1), coset_elem, N);
    let evals: Vec<_> = points
        .into_iter()
        .map(|point| evaluate_cfft_poly(&values, point))
        .collect();

    assert_eq!(cfft_evals, evals);
}

// Test that the cfft and icfft are inverses.
pub(crate) fn fft_ifft_test<Base, Ext, Cfft, const N: usize, const BATCH_SIZE: usize>()
where
    Base: Field,
    Standard: Distribution<Base>,
    Ext: ComplexExtension<Base>,
    Cfft: CircleSubgroupFFT<Base, Ext>,
{
    let mut rng = rand::thread_rng();

    let values = RowMajorMatrix::<Base>::rand(&mut rng, N, BATCH_SIZE);

    let cfft_fn = Cfft::default();

    let cfft_coeffs = cfft_fn.cfft_batch(values.clone());
    let mut cfft_evals = cfft_fn.icfft_batch(cfft_coeffs);

    divide_by_height(&mut cfft_evals);

    assert_eq!(values, cfft_evals);
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
