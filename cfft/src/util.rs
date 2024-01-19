use alloc::vec::Vec;

use itertools::Itertools;
use p3_field::{
    extension::{Complex, ComplexExtendable},
    AbstractField, ComplexExtension, Field,
};

/// Given an integer bits, generate half of the points in the coset gH.
/// Here H is the unique subgroup of order 2^bits and g is an element of order 2^{bits + 1}.
/// The output will be a list {g, g^3, g^5, ...} of length size.
/// Use size = 2^{bits} for the full domain or 2^{bits - 1} for the half domain.
#[inline]
pub(crate) fn cfft_domain<F: ComplexExtendable>(bits: usize, size: usize) -> Vec<Complex<F>> {
    let generator = F::circle_two_adic_generator(bits + 1);

    (generator * generator)
        .shifted_powers(generator)
        .take(size)
        .collect()
}

/// Given a generator h for H and an element k, generate points in the twin coset kH u k^{-1}H.
/// The ordering is important here, the points will generated in the following interleaved pattern:
/// {k, k^{-1}h, kh, k^{-1}h^2, kh^2, ..., k^{-1}h^{-1}, kh^{-1}, k^{-1}}.
/// Size controls how many of these we want to compute. It should either be |H| or |H|/2 depending on if
/// we want simply the twiddles or the full domain. If k is has order 2|H| this is equal to cfft_domain.
#[inline]
pub(crate) fn twin_coset_domain<F: ComplexExtendable>(
    generator: Complex<F>,
    coset_elem: Complex<F>,
    size: usize,
) -> Vec<Complex<F>> {
    generator
        .shifted_powers(coset_elem)
        .interleave(generator.shifted_powers(generator * coset_elem.inverse()))
        .take(size)
        .collect()
}
