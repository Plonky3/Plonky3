use alloc::vec::Vec;

use p3_field::{ComplexExtension, Field};
use rand::distributions::{Distribution, Standard};
use rand::Rng;

use crate::util::cfft_domain;
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

// Test that the cfft and icfft are inverses.
pub(crate) fn fft_ifft_test<Base, Ext, Cfft, const N: usize>()
where
    Base: Field,
    Standard: Distribution<Base>,
    Ext: ComplexExtension<Base>,
    Cfft: CircleSubgroupFFT<Base, Ext>,
{
    let mut rng = rand::thread_rng();
    let values: [Base; N] = core::array::from_fn(|_| rng.gen::<Base>());
    let initial = values.to_vec();

    let cfft_fn = Cfft::default();
    let cfft_coeffs = cfft_fn.cfft(values.to_vec());
    let cfft_evals_scaled = cfft_fn.icfft(cfft_coeffs);

    let scale_inv = Base::from_canonical_u32(N as u32).inverse();

    let cfft_evals: Vec<_> = cfft_evals_scaled
        .into_iter()
        .map(|x| x * scale_inv)
        .collect();

    assert_eq!(initial, cfft_evals);
}
