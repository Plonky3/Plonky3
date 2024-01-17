use alloc::vec::Vec;

use itertools::Itertools;
use p3_field::{ComplexExtension, Field, Powers};
// use p3_matrix::dense::RowMajorMatrix;
// use p3_matrix::Matrix;

/// Given an integer bits, generate half of the points in the coset gH.
/// Here H is the unique subgroup of order 2^bits and g is an element of order 2^{bits + 1}.
/// The output will be a list {g, g^3, g^5, ...} of length size.
/// Use size = 2^{bits} for the full domain or 2^{bits - 1} for the half domain.
#[inline]
pub(crate) fn cfft_domain<Base: Field, Ext: ComplexExtension<Base>>(
    bits: usize,
    size: usize,
) -> Vec<Ext> {
    let generator = Ext::circle_two_adic_generator(bits + 1);

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
pub(crate) fn twin_coset_domain<Base: Field, Ext: ComplexExtension<Base>>(
    generator: Ext,
    coset_elem: Ext,
    size: usize,
) -> Vec<Ext> {
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

// Divide each coefficient of the given matrix by its height.
// pub(crate) fn divide_by_height<F: Field>(mat: &mut RowMajorMatrix<F>) {
//     let h = mat.height();
//     let h_inv = F::from_canonical_usize(h).inverse();
//     let (prefix, shorts, suffix) = unsafe { mat.values.align_to_mut::<F::Packing>() };
//     prefix.iter_mut().for_each(|x| *x *= h_inv);
//     shorts.iter_mut().for_each(|x| *x *= h_inv);
//     suffix.iter_mut().for_each(|x| *x *= h_inv);
// }
